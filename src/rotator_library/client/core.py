# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# CRITICAL LOAD-BEARING IMPORT ORDER:
# 1. AIOHTTP_NO_EXTENSIONS=1 must be set before any aiohttp import
# 2. dns_fix.py must run before litellm/aiohttp import
# 3. patch_litellm_finish_reason.py must run before litellm import
# 4. SSL monkey-patch fires when HTTP_SSL_VERIFY=false
# Do NOT reorder or simplify these imports.

import os

# Load .env file BEFORE reading any environment variables
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

# CRITICAL: Apply DNS fix BEFORE importing litellm/aiohttp
# This fixes DNS hijacking by VPN/proxy/antivirus that returns wrong IPs
from ..dns_fix import apply_dns_fix

apply_dns_fix()

# CRITICAL: Apply finish_reason patch BEFORE importing litellm/openai
# LiteLLM caches OpenAI models on import, so patch must run first
from ..utils.patch_litellm_finish_reason import patch_litellm_finish_reason

patch_litellm_finish_reason()

# CRITICAL: Patch aiohttp.TCPConnector to use TLS 1.2 and disable SSL verification
# This fixes ConnectionResetError and SSLCertVerificationError with servers like noobrouterproduction.azurewebsites.net
from ..ssl_patch import _patch_aiohttp_connector
from ..quota_reporter import QuotaReporter
from ..anthropic_adapter import AnthropicAdapter

_patch_aiohttp_connector()

import asyncio
import fnmatch
import logging
import re
import time
import random
import uuid
import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator, Optional, Union, Tuple
from urllib.parse import urlparse

from ..utils.json_utils import json_deep_copy
from ..utils.json_utils import STREAM_DONE  # noqa: E402 – after litellm patching

lib_logger = logging.getLogger("rotator_library")
# Ensure the logger is configured to propagate to the root logger
# which is set up in main.py. This allows the main app to control
# log levels and handlers centrally.
lib_logger.propagate = False

DEFAULT_API_KEY_MAX_CONCURRENT_REQUESTS = 40

# Providers that require stream=true when max_tokens exceeds a threshold.
# {provider_prefix: max_tokens_threshold}
# Fireworks API returns 400 "Requests with max_tokens > 4096 must have stream=true"
_STREAM_REQUIRED_PROVIDERS = {
    "fireworks": 4096,
}

from ..usage_manager import UsageManager
from ..failure_logger import log_failure, configure_failure_logger
from ..error_types import (
    PreRequestCallbackError,
    CredentialNeedsReauthError,
    NoAvailableKeysError,
    RequestErrorAccumulator,
    mask_credential,
    ContextOverflowError,
    ClassifiedError,
)
from ..error_handler import (
    classify_error,
    should_rotate_on_error,
    should_retry_same_key,
    get_retry_backoff,
)
from ..provider_config import ProviderConfig
from ..http_client_pool import HttpClientPool, get_http_pool, close_http_pool
from ..providers import PROVIDER_PLUGINS
from ..providers.openai_compatible_provider import OpenAICompatibleProvider
from ..request_sanitizer import sanitize_request_payload
from ..model_info_service import get_model_info_service
from ..resilience import ResilienceOrchestrator
from ..credential_manager import CredentialManager
from ..background_refresher import BackgroundRefresher
from ..anthropic_compat.models import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
)
from ..model_definitions import ModelDefinitions
from ..transaction_logger import TransactionLogger
from ..utils.paths import get_default_root, get_logs_dir, get_oauth_dir
from ..utils.suppress_litellm_warnings import suppress_litellm_serialization_warnings
from ..utils.model_utils import extract_provider_from_model, get_or_create_provider_instance, normalize_model_string
from ..utils.provider_registry import get_provider_registry
from ..config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_ROTATION_TOLERANCE,
    CIRCUIT_BREAKER_PROVIDER_OVERRIDES,
)

from ..env_cache import _provider_env_cache, _PROVIDER_ENV_PREFIXES  # noqa: F401

# Import mixin classes for method inheritance
from ._helpers import HelpersMixin
from ._streaming import StreamingMixin
from ._retry import RetryMixin, _RetryContext


class RotatingClient(HelpersMixin, StreamingMixin, RetryMixin):
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """

    def __init__(
        self,
        api_keys: Optional[Dict[str, List[str]]] = None,
        oauth_credentials: Optional[Dict[str, List[str]]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        usage_file_path: Optional[Union[str, Path]] = None,
        configure_logging: bool = True,
        global_timeout: int = DEFAULT_GLOBAL_TIMEOUT,
        abort_on_callback_error: bool = True,
        litellm_provider_params: Optional[Dict[str, Any]] = None,
        ignore_models: Optional[Dict[str, List[str]]] = None,
        whitelist_models: Optional[Dict[str, List[str]]] = None,
        enable_request_logging: bool = False,
        max_concurrent_requests_per_key: Optional[Dict[str, int]] = None,
        rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE,
        data_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the RotatingClient with intelligent credential rotation.

        Args:
            api_keys: Dictionary mapping provider names to lists of API keys
            oauth_credentials: Dictionary mapping provider names to OAuth credential paths
            max_retries: Maximum number of retry attempts per credential
            usage_file_path: Path to store usage statistics. If None, uses data_dir/key_usage.json
            configure_logging: Whether to configure library logging
            global_timeout: Global timeout for requests in seconds
            abort_on_callback_error: Whether to abort on pre-request callback errors
            litellm_provider_params: Provider-specific parameters for LiteLLM
            ignore_models: Models to ignore/blacklist per provider
            whitelist_models: Models to explicitly whitelist per provider
            enable_request_logging: Whether to enable detailed request logging
            max_concurrent_requests_per_key: Max concurrent requests per key by provider
            rotation_tolerance: Tolerance for weighted random credential rotation.
                - 0.0: Deterministic, least-used credential always selected
                - 2.0 - 4.0 (default, recommended): Balanced randomness, can pick credentials within 2 uses of max
                - 5.0+: High randomness, more unpredictable selection patterns
            data_dir: Root directory for all data files (logs, cache, oauth_creds, key_usage.json).
                      If None, auto-detects: EXE directory if frozen, else current working directory.
        """
        # Resolve data_dir early - this becomes the root for all file operations
        if data_dir is not None:
            self.data_dir = Path(data_dir).resolve()
        else:
            self.data_dir = get_default_root()

        # Configure failure logger to use correct logs directory
        configure_failure_logger(get_logs_dir(self.data_dir))

        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.set_verbose = False
        litellm.drop_params = True

        # Suppress harmless Pydantic serialization warnings from litellm
        # See: https://github.com/BerriAI/litellm/issues/11759
        # TODO: Remove this workaround once litellm patches the issue
        suppress_litellm_serialization_warnings()

        if configure_logging:
            # When True, this allows logs from this library to be handled
            # by the parent application's logging configuration.
            lib_logger.propagate = True
            # Remove any default handlers to prevent duplicate logging
            if lib_logger.hasHandlers():
                lib_logger.handlers.clear()
                lib_logger.addHandler(logging.NullHandler())
        else:
            lib_logger.propagate = False

        api_keys = api_keys or {}
        oauth_credentials = oauth_credentials or {}

        # Filter out providers with empty lists of credentials to ensure validity
        api_keys = {provider: keys for provider, keys in api_keys.items() if keys}
        oauth_credentials = {
            provider: paths for provider, paths in oauth_credentials.items() if paths
        }

        if not api_keys and not oauth_credentials:
            lib_logger.warning(
                "No provider credentials configured. The client will be unable to make any API requests."
            )

        self.api_keys = api_keys
        # Use provided oauth_credentials directly if available (already discovered by main.py)
        # Only call discover_and_prepare() if no credentials were passed
        if oauth_credentials:
            self.oauth_credentials = oauth_credentials
        else:
            self.credential_manager = CredentialManager(
                os.environ, oauth_dir=get_oauth_dir(self.data_dir)
            )
            self.oauth_credentials = self.credential_manager.discover_and_prepare()
        self.background_refresher = BackgroundRefresher(self)
        self.oauth_providers = set(self.oauth_credentials.keys())

        all_credentials = {}
        for provider, keys in api_keys.items():
            all_credentials.setdefault(provider, []).extend(keys)
        for provider, paths in self.oauth_credentials.items():
            all_credentials.setdefault(provider, []).extend(paths)
        self.all_credentials = all_credentials
        self._cred_offset: Dict[str, int] = {}

        self.max_retries = max_retries
        self.global_timeout = global_timeout
        self.abort_on_callback_error = abort_on_callback_error

        # Initialize provider plugins early so they can be used for rotation mode detection
        self._provider_plugins = PROVIDER_PLUGINS
        self._provider_instances = get_provider_registry()

        # Build all provider-specific configuration via extracted module
        from ..client_config import build_all_provider_configs
        provider_configs = build_all_provider_configs(
            self.all_credentials, self._provider_plugins
        )

        # Resolve usage file path - use provided path or default to data_dir
        if usage_file_path is not None:
            resolved_usage_path = Path(usage_file_path)
        else:
            resolved_usage_path = self.data_dir / "key_usage.json"

        self.usage_manager = UsageManager(
            file_path=resolved_usage_path,
            rotation_tolerance=rotation_tolerance,
            provider_rotation_modes=provider_configs["provider_rotation_modes"],
            provider_plugins=PROVIDER_PLUGINS,
            priority_multipliers=provider_configs["priority_multipliers"],
            priority_multipliers_by_mode=provider_configs["priority_multipliers_by_mode"],
            sequential_fallback_multipliers=provider_configs["sequential_fallback_multipliers"],
            fair_cycle_enabled=provider_configs["fair_cycle_enabled"],
            fair_cycle_tracking_mode=provider_configs["fair_cycle_tracking_mode"],
            fair_cycle_cross_tier=provider_configs["fair_cycle_cross_tier"],
            fair_cycle_duration=provider_configs["fair_cycle_duration"],
            exhaustion_cooldown_threshold=provider_configs["exhaustion_cooldown_threshold"],
            custom_caps=provider_configs["custom_caps"],
            credential_to_provider=self._build_credential_to_provider_map(),
        )
        self._model_list_cache = {}
        # Use HttpClientPool singleton for optimized connection management
        self._http_pool: Optional[HttpClientPool] = None
        self._pool_initialized = False
        # Cache for provider API endpoints (for pre-warming)
        self._provider_endpoints: Dict[str, str] = {}

        # Credential priority cache for fast lookups (TTL=300s, auto-expiry)
        from cachetools import TTLCache
        self._credential_priority_cache: TTLCache = TTLCache(maxsize=64, ttl=300)

        self.provider_config = ProviderConfig()
        self._resilience = ResilienceOrchestrator(
            provider_overrides=CIRCUIT_BREAKER_PROVIDER_OVERRIDES,
        )
        # Expose sub-components for backward compatibility
        self.cooldown_manager = self._resilience.cooldown
        self.ip_throttle_detector = self._resilience.ip_throttle
        self.circuit_breaker = self._resilience.circuit_breaker
        self.litellm_provider_params = litellm_provider_params or {}
        self.ignore_models = ignore_models or {}
        self.whitelist_models = whitelist_models or {}
        self.enable_request_logging = enable_request_logging
        self.model_definitions = ModelDefinitions()

        # Initialize ModelRegistry for context window lookups (used by token calculator)
        self._model_registry = get_model_info_service()

        # Store and validate max concurrent requests per key
        self.max_concurrent_requests_per_key = dict(
            max_concurrent_requests_per_key or {}
        )

        for provider in self.api_keys:
            self.max_concurrent_requests_per_key.setdefault(
                provider, DEFAULT_API_KEY_MAX_CONCURRENT_REQUESTS
            )

        # Validate all values are >= 1
        for provider, max_val in self.max_concurrent_requests_per_key.items():
            if max_val < 1:
                lib_logger.warning(
                    f"Invalid max_concurrent for '{provider}': {max_val}. Setting to 1."
                )
                self.max_concurrent_requests_per_key[provider] = 1

        # Track consecutive quota failures per credential for intelligent rotation
        self._consecutive_quota_failures: Dict[str, int] = {}

    # --- Core methods that remain in this module ---

    async def _prepare_request_kwargs(
        self,
        base_kwargs: Dict[str, Any],
        provider: str,
        credential: str,
        model: str,
        *,
        include_reasoning_effort: bool = False,
    ) -> Dict[str, Any]:
        """Clone and normalize per-attempt request kwargs before provider execution."""
        litellm_kwargs = base_kwargs.copy()

        self._strip_client_headers(litellm_kwargs)
        await self._apply_provider_headers(litellm_kwargs, provider, credential)

        if include_reasoning_effort and "reasoning_effort" in base_kwargs:
            litellm_kwargs["reasoning_effort"] = base_kwargs["reasoning_effort"]

        if provider in self.litellm_provider_params:
            litellm_kwargs["litellm_params"] = {
                **self.litellm_provider_params[provider],
                **litellm_kwargs.get("litellm_params", {}),
            }

        provider_plugin = self._get_provider_instance(provider)
        if provider_plugin and hasattr(provider_plugin, "get_model_options"):
            model_options = provider_plugin.get_model_options(model)
            if model_options:
                for key, value in model_options.items():
                    if key == "reasoning_effort":
                        litellm_kwargs["reasoning_effort"] = value
                    elif key not in litellm_kwargs:
                        litellm_kwargs[key] = value

        return litellm_kwargs

    def increment_quota_failures(self, credential_id: str) -> bool:
        """
        Increment consecutive quota failure counter for a credential.

        Args:
            credential_id: The credential identifier

        Returns:
            True if rotation is needed (>= 3 consecutive failures), False otherwise
        """
        self._consecutive_quota_failures[credential_id] = (
            self._consecutive_quota_failures.get(credential_id, 0) + 1
        )
        count = self._consecutive_quota_failures[credential_id]
        lib_logger.debug(
            f"Quota failure increment for {mask_credential(credential_id)}: {count}/3"
        )
        return count >= 3

    def reset_quota_failures(self, credential_id: str) -> None:
        """
        Reset consecutive quota failure counter for a credential on success.

        Args:
            credential_id: The credential identifier
        """
        if credential_id in self._consecutive_quota_failures:
            del self._consecutive_quota_failures[credential_id]
            lib_logger.debug(
                f"Quota failure counter reset for {mask_credential(credential_id)}"
            )

    def _is_client_usable(self, client: Optional[httpx.AsyncClient]) -> bool:
        """
        Check if an HTTP client is usable for requests.

        This is more thorough than just checking is_closed - it also checks
        the internal transport state which can be closed independently.

        Args:
            client: The client to check

        Returns:
            True if the client is usable, False otherwise
        """
        if client is None:
            return False
        if client.is_closed:
            return False
        return True

    async def _ensure_http_pool(self) -> HttpClientPool:
        """
        Ensure the HTTP client pool is initialized.

        Uses the global singleton pool for optimal connection sharing.
        Pre-warms connections to known provider endpoints.
        """
        if self._http_pool is None or not self._pool_initialized:
            self._http_pool = await get_http_pool()
            if not self._http_pool.is_initialized:
                # Build list of endpoints to pre-warm
                warmup_hosts = self._get_provider_endpoints()
                await self._http_pool.initialize(warmup_hosts=warmup_hosts)
            self._pool_initialized = True
            lib_logger.debug("HTTP client pool initialized with pre-warmed connections")
        return self._http_pool

    def _get_provider_endpoints(self) -> List[str]:
        """
        Get list of API endpoints for all configured providers.

        Returns:
            List of URLs to pre-warm connections for
        """
        endpoints = []

        # Map of provider names to their default API base URLs
        # These are only used as fallbacks if no custom API_BASE is configured
        provider_urls = {
            "openai": "https://api.openai.com",
            "anthropic": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com",
            "antigravity": "https://api.antigravity.ai",
            "iflow": "https://api.iflow.ai",
        }

        # Add endpoints for configured providers
        # Priority: custom API_BASE from env > hardcoded defaults
        for provider in self.all_credentials.keys():
            # First check if provider has a custom API_BASE configured
            if provider in self.provider_config.api_bases:
                api_base = self.provider_config.api_bases[provider]
                if api_base:
                    # Extract just the origin for warmup
                    parsed = urlparse(api_base)
                    if parsed.scheme and parsed.netloc:
                        endpoints.append(f"{parsed.scheme}://{parsed.netloc}")
                        continue
            # Fall back to hardcoded defaults
            if provider in provider_urls:
                endpoints.append(provider_urls[provider])

        # Cache resolved endpoints for later use
        self._provider_endpoints = {}
        for provider in self.all_credentials.keys():
            if provider in self.provider_config.api_bases:
                api_base = self.provider_config.api_bases[provider]
                if api_base:
                    parsed = urlparse(api_base)
                    if parsed.scheme and parsed.netloc:
                        self._provider_endpoints[provider] = (
                            f"{parsed.scheme}://{parsed.netloc}"
                        )
                        continue
            if provider in provider_urls:
                self._provider_endpoints[provider] = provider_urls[provider]

        return list(set(endpoints))[:5]  # Dedupe and limit

    def _get_http_client(self, streaming: bool = False) -> httpx.AsyncClient:
        """
        Get HTTP client from the pool (sync version for compatibility).

        Prefer _get_http_client_async() for production use.

        Args:
            streaming: Whether this client will be used for streaming requests

        Returns:
            httpx.AsyncClient instance
        """
        # If pool reference not set, bind to the singleton pool
        # (this should rarely happen if initialize() is called properly)
        if self._http_pool is None:
            lib_logger.warning("HTTP pool accessed before initialization")
            self._http_pool = HttpClientPool()
        return self._http_pool.get_client(streaming=streaming)

    async def _get_http_client_async(
        self, streaming: bool = False
    ) -> httpx.AsyncClient:
        """
        Get HTTP client from the pool with automatic recovery.

        This is the preferred method for getting an HTTP client.
        It ensures the pool is initialized and returns a healthy client.

        Args:
            streaming: Whether this client will be used for streaming requests

        Returns:
            Usable httpx.AsyncClient instance
        """
        pool = await self._ensure_http_pool()
        return await pool.get_client_async(streaming=streaming)

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Property that returns client from pool (non-streaming by default)."""
        return self._get_http_client(streaming=False)

    def _parse_custom_cap_env_key(
        self, remainder: str
    ) -> Tuple[Optional[Union[int, Tuple[int, ...], str]], Optional[str]]:
        """Delegate to client_config.parse_custom_cap_env_key."""
        from ..client_config import parse_custom_cap_env_key
        return parse_custom_cap_env_key(remainder)

    def _is_model_ignored(self, provider: str, model_id: str) -> bool:
        """
        Checks if a model should be ignored based on the ignore list.
        Supports full glob/fnmatch patterns for both full model IDs and model names.

        Pattern examples:
        - "gpt-4" - exact match
        - "gpt-4*" - prefix wildcard (matches gpt-4, gpt-4-turbo, etc.)
        - "*-preview" - suffix wildcard (matches gpt-4-preview, o1-preview, etc.)
        - "*-preview*" - contains wildcard (matches anything with -preview)
        - "*" - match all
        """
        model_provider = model_id.split("/")[0]
        if model_provider not in self.ignore_models:
            return False

        ignore_list = self.ignore_models[model_provider]
        if ignore_list == ["*"]:
            return True

        try:
            # This is the model name as the provider sees it (e.g., "gpt-4" or "google/gemma-7b")
            provider_model_name = model_id.split("/", 1)[1]
        except IndexError:
            provider_model_name = model_id

        for ignored_pattern in ignore_list:
            # Use fnmatch for full glob pattern support
            if fnmatch.fnmatch(provider_model_name, ignored_pattern) or fnmatch.fnmatch(
                model_id, ignored_pattern
            ):
                return True
        return False

    def _is_model_whitelisted(self, provider: str, model_id: str) -> bool:
        """
        Checks if a model is explicitly whitelisted.
        Supports full glob/fnmatch patterns for both full model IDs and model names.

        Pattern examples:
        - "gpt-4" - exact match
        - "gpt-4*" - prefix wildcard (matches gpt-4, gpt-4-turbo, etc.)
        - "*-preview" - suffix wildcard (matches gpt-4-preview, o1-preview, etc.)
        - "*-preview*" - contains wildcard (matches anything with -preview)
        - "*" - match all
        """
        model_provider = model_id.split("/")[0]
        if model_provider not in self.whitelist_models:
            return False

        whitelist = self.whitelist_models[model_provider]

        try:
            # This is the model name as the provider sees it (e.g., "gpt-4" or "google/gemma-7b")
            provider_model_name = model_id.split("/", 1)[1]
        except IndexError:
            provider_model_name = model_id

        for whitelisted_pattern in whitelist:
            # Use fnmatch for full glob pattern support
            if fnmatch.fnmatch(
                provider_model_name, whitelisted_pattern
            ) or fnmatch.fnmatch(model_id, whitelisted_pattern):
                return True
        return False

    def get_oauth_credentials(self) -> Dict[str, List[str]]:
        return self.oauth_credentials

    def _is_custom_openai_compatible_provider(self, provider_name: str) -> bool:
        """
        Checks if a provider is a custom OpenAI-compatible provider.

        Custom providers are identified by:
        1. Having a _API_BASE environment variable set, AND
        2. NOT being in the list of known LiteLLM providers
        """
        return self.provider_config.is_custom_provider(provider_name)

    def _build_credential_to_provider_map(self) -> Dict[str, str]:
        """Build a reverse mapping from credential identifier to provider name."""
        mapping: Dict[str, str] = {}
        for provider, creds in self.all_credentials.items():
            for cred in creds:
                mapping[cred] = provider
        return mapping

    def _get_provider_instance(self, provider_name: str):
        """
        Lazily initializes and returns a provider instance.
        Only initializes providers that have configured credentials.

        Args:
            provider_name: The name of the provider to get an instance for.
                          For OAuth providers, this may include "_oauth" suffix
                          (e.g., "antigravity_oauth"), but credentials are stored
                          under the base name (e.g., "antigravity").

        Returns:
            Provider instance if credentials exist, None otherwise.
        """
        # For OAuth providers, credentials are stored under base name (without _oauth suffix)
        # e.g., "antigravity_oauth" plugin -> credentials under "antigravity"
        credential_key = provider_name
        if provider_name.endswith("_oauth"):
            base_name = provider_name[:-6]  # Remove "_oauth"
            if base_name in self.oauth_providers:
                credential_key = base_name

        # Only initialize providers for which we have credentials
        if credential_key not in self.all_credentials:
            lib_logger.debug(
                f"Skipping provider '{provider_name}' initialization: no credentials configured"
            )
            return None

        # Try shared lazy-load path first
        result = get_or_create_provider_instance(
            provider_name, self._provider_plugins, self._provider_instances
        )
        if result is not None:
            return result

        # Client-specific fallback: custom OpenAI-compatible providers
        if self._is_custom_openai_compatible_provider(provider_name):
            try:
                instance = OpenAICompatibleProvider(provider_name)
                self._provider_instances.register(provider_name, instance)
                return instance
            except ValueError:
                return None
        else:
            # Check if already registered (e.g. by usage_manager)
            return self._provider_instances.get(provider_name)

    def _resolve_model_id(self, model: str, provider: str) -> str:
        """
        Resolves the actual model ID to send to the provider.

        For custom models with name/ID mappings, returns the ID.
        Otherwise, returns the model name unchanged.

        Args:
            model: Full model string with provider (e.g., "iflow/DS-v3.2")
            provider: Provider name (e.g., "iflow")

        Returns:
            Full model string with ID (e.g., "iflow/deepseek-v3.2")
        """
        # Extract model name from "provider/model_name" format
        model_name = model.split("/")[-1] if "/" in model else model

        # Try to get provider instance to check for model definitions
        provider_plugin = self._get_provider_instance(provider)

        # Check if provider has model definitions
        if provider_plugin and hasattr(provider_plugin, "model_definitions"):
            model_id = provider_plugin.model_definitions.get_model_id(
                provider, model_name
            )
            if model_id and model_id != model_name:
                # Return with provider prefix
                return f"{provider}/{model_id}"

        # Fallback: use client's own model definitions
        model_id = self.model_definitions.get_model_id(provider, model_name)
        if model_id and model_id != model_name:
            return f"{provider}/{model_id}"

        # No conversion needed, return original
        return model

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Close the HTTP client pool to prevent resource leaks."""
        # Note: We don't close the global pool here as it may be shared
        # across multiple RotatingClient instances.
        # The pool will be closed on application shutdown via close_http_pool().
        self._http_pool = None
        self._pool_initialized = False

    def acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Union[Any, AsyncGenerator[Any, None]]:
        """
        Dispatcher for completion requests.

        Args:
            request: Optional request object, used for client disconnect checks and logging.
            pre_request_callback: Optional async callback function to be called before each API request attempt.
                The callback will receive the `request` object and the prepared request `kwargs` as arguments.
                This can be used for custom logic such as request validation, logging, or rate limiting.
                If the callback raises an exception, the completion request will be aborted and the exception will propagate.

        Returns:
            The completion response object, or an async generator for streaming responses, or None if all retries fail.
        """
        # Providers that don't support stream_options parameter
        # These providers return 400/406 errors when stream_options is sent
        STREAM_OPTIONS_UNSUPPORTED_PROVIDERS = {"iflow", "kilocode"}

        model = normalize_model_string(kwargs.get("model", ""))
        kwargs["model"] = model
        provider = extract_provider_from_model(model)

        # Remove stream_options for providers that don't support it
        if (
            provider in STREAM_OPTIONS_UNSUPPORTED_PROVIDERS
            and "stream_options" in kwargs
        ):
            lib_logger.debug(
                f"Removing stream_options for {provider} provider (not supported)"
            )
            kwargs.pop("stream_options", None)

        # Check if provider requires forced streaming for high max_tokens
        # Some providers (e.g., Fireworks) reject non-streaming requests when
        # max_tokens exceeds a threshold with: "Requests with max_tokens > N must have stream=true"
        forced_streaming = False
        if not kwargs.get("stream"):
            max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
            if max_tokens and provider in _STREAM_REQUIRED_PROVIDERS:
                threshold = _STREAM_REQUIRED_PROVIDERS[provider]
                if max_tokens > threshold:
                    lib_logger.info(
                        f"Forcing stream=true for {provider} provider "
                        f"(max_tokens={max_tokens} > threshold={threshold})"
                    )
                    kwargs["stream"] = True
                    forced_streaming = True

        if kwargs.get("stream"):
            # Only add stream_options for providers that support it
            if provider not in STREAM_OPTIONS_UNSUPPORTED_PROVIDERS:
                if "stream_options" not in kwargs:
                    kwargs["stream_options"] = {}
                if "include_usage" not in kwargs["stream_options"]:
                    kwargs["stream_options"]["include_usage"] = True

            if forced_streaming:
                # Internally stream but collect into a non-streaming ModelResponse
                return self._forced_streaming_acompletion(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **kwargs,
                )

            return self._streaming_acompletion_with_retry(
                request=request, pre_request_callback=pre_request_callback, **kwargs
            )
        else:
            return self._execute_with_retry(
                litellm.acompletion,
                request=request,
                pre_request_callback=pre_request_callback,
                **kwargs,
            )

    def aembedding(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """
        Executes an embedding request with retry logic.

        Args:
            request: Optional request object, used for client disconnect checks and logging.
            pre_request_callback: Optional async callback function to be called before each API request attempt.
                The callback will receive the `request` object and the prepared request `kwargs` as arguments.
                This can be used for custom logic such as request validation, logging, or rate limiting.
                If the callback raises an exception, the embedding request will be aborted and the exception will propagate.

        Returns:
            The embedding response object, or None if all retries fail.
        """
        return self._execute_with_retry(
            litellm.aembedding,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    def token_count(self, **kwargs) -> int:
        """Calculates the number of tokens for a given text or list of messages.

        For Antigravity provider models, this also includes the preprompt tokens
        that get injected during actual API calls (agent instruction + identity override).
        This ensures token counts match actual usage.
        """
        model = kwargs.get("model")
        text = kwargs.get("text")
        messages = kwargs.get("messages")

        if not model:
            raise ValueError("'model' is a required parameter.")

        # Calculate base token count
        if messages:
            base_count = token_counter(model=model, messages=messages)
        elif text:
            base_count = token_counter(model=model, text=text)
        else:
            raise ValueError("Either 'text' or 'messages' must be provided.")

        # Add preprompt tokens for Antigravity provider
        # The Antigravity provider injects system instructions during actual API calls,
        # so we need to account for those tokens in the count
        provider = extract_provider_from_model(model)
        if provider == "antigravity":
            try:
                from ..providers.antigravity.constants import (
                    get_antigravity_preprompt_text,
                )

                preprompt_text = get_antigravity_preprompt_text()
                if preprompt_text:
                    preprompt_tokens = token_counter(model=model, text=preprompt_text)
                    base_count += preprompt_tokens
            except ImportError:
                # Provider not available, skip preprompt token counting
                pass

        return base_count

    async def get_available_models(self, provider: str) -> List[str]:
        """Returns a list of available models for a specific provider, with caching."""
        lib_logger.info(f"Getting available models for provider: {provider}")
        if provider in self._model_list_cache:
            lib_logger.debug(f"Returning cached models for provider: {provider}")
            return self._model_list_cache[provider]

        credentials_for_provider = self.all_credentials.get(provider)
        if not credentials_for_provider:
            lib_logger.warning(f"No credentials for provider: {provider}")
            return []

        # Create a copy and rotate it for round-robin credential selection
        shuffled_credentials = list(credentials_for_provider)
        offset = self._cred_offset.get(provider, 0)
        self._cred_offset[provider] = (offset + 1) % len(shuffled_credentials)
        shuffled_credentials = shuffled_credentials[offset:] + shuffled_credentials[:offset]

        provider_instance = self._get_provider_instance(provider)
        if provider_instance:
            # For providers with hardcoded models (like gemini_cli), we only need to call once.
            # For others, we might need to try multiple keys if one is invalid.
            # The current logic of iterating works for both, as the credential is not
            # always used in get_models.
            for credential in shuffled_credentials:
                try:
                    # Display last 6 chars for API keys, or the filename for OAuth paths
                    cred_display = mask_credential(credential)
                    lib_logger.debug(
                        f"Attempting to get models for {provider} with credential {cred_display}"
                    )
                    models = await provider_instance.get_models(
                        credential, await self._get_http_client_async(streaming=False)
                    )
                    lib_logger.info(
                        f"Got {len(models)} models for provider: {provider}"
                    )

                    # Whitelist and blacklist logic
                    final_models = []
                    for m in models:
                        is_whitelisted = self._is_model_whitelisted(provider, m)
                        is_blacklisted = self._is_model_ignored(provider, m)

                        if is_whitelisted:
                            final_models.append(m)
                            continue

                        if not is_blacklisted:
                            final_models.append(m)

                    if len(final_models) != len(models):
                        lib_logger.info(
                            f"Filtered out {len(models) - len(final_models)} models for provider {provider}."
                        )

                    self._model_list_cache[provider] = final_models
                    return final_models
                except Exception as e:
                    classified_error = classify_error(e, provider=provider)
                    cred_display = mask_credential(credential)
                    lib_logger.debug(
                        f"Failed to get models for provider {provider} with credential {cred_display}: {classified_error.error_type}. Trying next credential."
                    )
                    continue  # Try the next credential

        lib_logger.error(
            f"Failed to get models for provider {provider} after trying all credentials."
        )
        return []

    async def get_all_available_models(
        self, grouped: bool = True
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Returns a list of all available models, either grouped by provider or as a flat list."""
        lib_logger.info("Getting all available models...")

        all_providers = list(self.all_credentials.keys())
        tasks = [self.get_available_models(provider) for provider in all_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_provider_models = {}
        for provider, result in zip(all_providers, results):
            if isinstance(result, Exception):
                lib_logger.error(
                    f"Failed to get models for provider {provider}: {result}"
                )
                all_provider_models[provider] = []
            else:
                all_provider_models[provider] = result

        lib_logger.info("Finished getting all available models.")
        if grouped:
            return all_provider_models
        else:
            flat_models = []
            for models in all_provider_models.values():
                flat_models.extend(models)
            return flat_models

    @property
    def quota_reporter(self):
        if not hasattr(self, '_quota_reporter_instance'):
            self._quota_reporter_instance = QuotaReporter(
                self.usage_manager,
                self._provider_plugins,
                self._provider_instances,
                self.all_credentials,
            )
        return self._quota_reporter_instance

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.quota_reporter.get_quota_stats(provider_filter)

    async def reload_usage_from_disk(self) -> None:
        """
        Force reload usage data from disk.

        Useful when wanting fresh stats without making external API calls.
        """
        await self.usage_manager.reload_from_disk()

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.quota_reporter.force_refresh_quota(provider, credential)

    # --- Anthropic API Compatibility Methods ---

    @property
    def anthropic_adapter(self):
        if not hasattr(self, '_anthropic_adapter_instance'):
            self._anthropic_adapter_instance = AnthropicAdapter(
                self.acompletion,
                self.token_count,
                extract_provider_from_model,
                self.all_credentials,
                self.enable_request_logging,
            )
        return self._anthropic_adapter_instance

    async def anthropic_messages(
        self,
        request: "AnthropicMessagesRequest",
        raw_request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        raw_body_data: Optional[dict] = None,
    ) -> Any:
        return await self.anthropic_adapter.anthropic_messages(
            request, raw_request, pre_request_callback, raw_body_data
        )

    async def anthropic_count_tokens(
        self,
        request: "AnthropicCountTokensRequest",
    ) -> dict:
        return await self.anthropic_adapter.anthropic_count_tokens(request)
