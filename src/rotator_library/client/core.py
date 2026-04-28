# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# CRITICAL LOAD-BEARING IMPORT ORDER:
# 1. AIOHTTP_NO_EXTENSIONS=1 must be set before any aiohttp import
# 2. dns_fix.py must run before litellm/aiohttp import
# 3. litellm_patches.patch_litellm_finish_reason must run before litellm import
# 4. SSL monkey-patch fires when HTTP_SSL_VERIFY=false
# Do NOT reorder or simplify these imports.

import os
import sys
import time

# CRITICAL: Apply DNS fix BEFORE importing litellm/aiohttp
# This fixes DNS hijacking by VPN/proxy/antivirus that returns wrong IPs
from ..dns_fix import apply_dns_fix

apply_dns_fix()

# CRITICAL: Apply finish_reason patch BEFORE importing litellm/openai
# LiteLLM caches OpenAI models on import, so patch must run first
from ..utils.litellm_patches import patch_litellm_finish_reason

patch_litellm_finish_reason()

# CRITICAL: Patch aiohttp.TCPConnector to use TLS 1.2 and disable SSL verification
# This fixes ConnectionResetError and SSLCertVerificationError with servers like noobrouterproduction.azurewebsites.net
from ..ssl_patch import _patch_aiohttp_connector

_patch_aiohttp_connector()

import asyncio
import logging
from collections.abc import Mapping
import httpx
import litellm
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..anthropic_compat.models import (
        AnthropicMessagesRequest,
        AnthropicCountTokensRequest,
    )

lib_logger = logging.getLogger("rotator_library")








try:
    DEFAULT_API_KEY_MAX_CONCURRENT_REQUESTS = int(
        os.environ.get("API_KEY_MAX_CONCURRENT_REQUESTS", 40)
    )
except ValueError:
    lib_logger.warning(
        "Invalid integer value for API_KEY_MAX_CONCURRENT_REQUESTS env var, using default"
    )
    DEFAULT_API_KEY_MAX_CONCURRENT_REQUESTS = 40

# Providers that require stream=true when max_tokens exceeds a threshold.
# {provider_prefix: max_tokens_threshold}
# Fireworks API returns 400 "Requests with max_tokens > 4096 must have stream=true"
_STREAM_REQUIRED_PROVIDERS = {
    "fireworks": 4096,
}

from ..usage_manager import UsageManager
from ..failure_logger import configure_failure_logger
from litellm.llms.openai.common_utils import OpenAIError

from ..error_types import (
    mask_credential,
    NoAvailableKeysError,
)
from ..error_handler import (
    classify_error,
)
from ..provider_routing_config import ProviderConfig
from ..http_client_pool import HttpClientPool, close_http_pool
from ..providers import PROVIDER_PLUGINS
from ..providers.openai_compatible_provider import OpenAICompatibleProvider
from ..model_info_service import get_model_info_service
from ..resilience import ResilienceOrchestrator
from ..credential_manager import CredentialManager
from ..background_refresher import BackgroundRefresher
from ..model_definitions import ModelDefinitions
from ..utils.paths import get_default_root, get_logs_dir, get_oauth_dir
from ..utils.litellm_patches import suppress_litellm_serialization_warnings
from ..utils.model_utils import (
    extract_provider_from_model,
    get_or_create_provider_instance,
    normalize_model_string,
)
from ..utils.provider_locks import ProviderLockManager
from ..utils.provider_registry import get_provider_registry
from ..config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_ROTATION_TOLERANCE,
    CIRCUIT_BREAKER_PROVIDER_OVERRIDES,
)


# Import mixin classes for method inheritance
from ._helpers import HelpersMixin
from ._streaming import StreamingMixin
from ._retry import RetryMixin
from ._media import MediaMixin, _IMAGE_PASSTHROUGH_PARAMS, _IMAGE_NATIVE_PROVIDERS
from ._models import ModelsMixin
from ._quota import QuotaMixin
from ._anthropic import AnthropicCompatibilityMixin


class RotatingClient(
    HelpersMixin,
    StreamingMixin,
    RetryMixin,
    MediaMixin,
    ModelsMixin,
    QuotaMixin,
    AnthropicCompatibilityMixin,
):
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """

    _MODEL_LIST_CACHE_TTL = 300.0

    def __init__(
        self,
        api_keys: Optional[dict[str, list[str]]] = None,
        oauth_credentials: Optional[dict[str, list[str]]] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        usage_file_path: Optional[Union[str, Path]] = None,
        configure_logging: bool = True,
        global_timeout: int = DEFAULT_GLOBAL_TIMEOUT,
        abort_on_callback_error: bool = True,
        litellm_provider_params: Optional[dict[str, Any]] = None,
        ignore_models: Optional[dict[str, list[str]]] = None,
        whitelist_models: Optional[dict[str, list[str]]] = None,
        enable_request_logging: bool = False,
        max_concurrent_requests_per_key: Optional[dict[str, int]] = None,
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
        self._cred_offset: dict[str, int] = {}
        self._lock_manager = ProviderLockManager()

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
            priority_multipliers_by_mode=provider_configs[
                "priority_multipliers_by_mode"
            ],
            sequential_fallback_multipliers=provider_configs[
                "sequential_fallback_multipliers"
            ],
            fair_cycle_enabled=provider_configs["fair_cycle_enabled"],
            fair_cycle_tracking_mode=provider_configs["fair_cycle_tracking_mode"],
            fair_cycle_cross_tier=provider_configs["fair_cycle_cross_tier"],
            fair_cycle_duration=provider_configs["fair_cycle_duration"],
            exhaustion_cooldown_threshold=provider_configs[
                "exhaustion_cooldown_threshold"
            ],
            custom_caps=provider_configs["custom_caps"],
            credential_to_provider=self._build_credential_to_provider_map(),
        )
        self._model_list_cache: dict[str, tuple[list[str], float]] = {}
        # Use HttpClientPool singleton for optimized connection management
        self._http_pool: Optional[HttpClientPool] = None
        self._pool_initialized = False
        # Cache for provider API endpoints (for pre-warming)
        self._provider_endpoints: dict[str, str] = {}

        # Credential priority cache for fast lookups (TTL=300s, auto-expiry)
        from cachetools import TTLCache

        self._credential_priority_cache: TTLCache = TTLCache(maxsize=64, ttl=300)

        # Model ID resolution cache — avoids repeated provider lookups per request
        self._resolve_model_id_cache: TTLCache = TTLCache(maxsize=256, ttl=300)

        self.provider_config = ProviderConfig()
        self._resilience = ResilienceOrchestrator(
            provider_overrides=CIRCUIT_BREAKER_PROVIDER_OVERRIDES,
        )
        # Expose sub-components for backward compatibility
        self.cooldown_manager = self._resilience.cooldown
        self.ip_throttle_detector = self._resilience.ip_throttle
        self.circuit_breaker = self._resilience.circuit_breaker
        self.rate_limiter = self._resilience.rate_limiter
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
        self._consecutive_quota_failures: dict[str, int] = {}

        # Global backpressure semaphore — limits total concurrent outbound
        # API requests across all providers/keys. Prevents resource exhaustion.
        _default_max_concurrent = 128 if sys.platform == "win32" else 256
        try:
            _max_concurrent = int(
                os.getenv("MAX_CONCURRENT_REQUESTS", str(_default_max_concurrent))
            )
        except ValueError:
            lib_logger.warning(
                "Invalid integer value for MAX_CONCURRENT_REQUESTS env var, using default"
            )
            _max_concurrent = _default_max_concurrent
        self._global_semaphore = asyncio.Semaphore(_max_concurrent)

    # --- Core methods that remain in this module ---

    async def _prepare_request_kwargs(
        self,
        base_kwargs: dict[str, Any],
        provider: str,
        credential: str,
        model: str,
        *,
        include_reasoning_effort: bool = False,
    ) -> dict[str, Any]:
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

        litellm_kwargs["num_retries"] = 0

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




    def get_oauth_credentials(self) -> dict[str, list[str]]:
        return self.oauth_credentials



    def _is_custom_openai_compatible_provider(self, provider_name: str) -> bool:
        """
        Checks if a provider is a custom OpenAI-compatible provider.

        Custom providers are identified by:
        1. Having a _API_BASE environment variable set, AND
        2. NOT being in the list of known LiteLLM providers
        """
        return self.provider_config.is_custom_provider(provider_name)

    def _build_credential_to_provider_map(self) -> dict[str, str]:
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

        # Fallback: known providers with api_base but no plugin get OpenAICompatibleProvider
        # This fixes providers like openrouter, xai, openai, moonshot that have keys
        # and api_base but no dedicated provider plugin class
        api_base = self.provider_config.api_bases.get(provider_name)
        if api_base:
            try:
                instance = OpenAICompatibleProvider(provider_name)
                self._provider_instances.register(provider_name, instance)
                return instance
            except ValueError:
                return None

        # Check if already registered (e.g. by usage_manager)
        return self._provider_instances.get(provider_name)


    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """Release reference to the shared HTTP client pool.

        Does NOT close the pool — it is a singleton shared across all
        RotatingClient instances.  Pool lifecycle is managed via
        close_http_pool() during application shutdown.
        """
        await close_http_pool()
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

        if self._is_image_only_model(model):
            prompt = self._extract_prompt_from_chat_messages(kwargs.get("messages", []))
            if prompt:
                image_kwargs = {
                    key: value
                    for key, value in kwargs.items()
                    if key in _IMAGE_PASSTHROUGH_PARAMS
                }
                image_kwargs["model"] = model
                image_kwargs["prompt"] = prompt
                if provider in _IMAGE_NATIVE_PROVIDERS:
                    image_kwargs["_native_provider"] = provider
                return self.aimage_generation(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **image_kwargs,
                )

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
                if not isinstance(kwargs.get("stream_options"), dict):
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

            return self._rate_limited_streaming(
                request=request, pre_request_callback=pre_request_callback, **kwargs
            )
        else:
            return self._rate_limited_execute(
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
        return self._rate_limited_execute(
            litellm.aembedding,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )




















    # --- Provider-specific API Methods (non-litellm) ---

    async def call_provider_method(
        self,
        provider_name: str,
        method_name: str,
        **kwargs,
    ) -> Any:
        """Call a provider-specific method with credential rotation.

        Picks an available credential for the provider, gets the provider
        instance, and delegates the call. Raises AttributeError if the
        provider or method doesn't exist.
        """
        provider_instance = self._get_provider_instance(provider_name)
        if provider_instance is None:
            raise ValueError(f"No provider instance for '{provider_name}'")

        method = getattr(provider_instance, method_name, None)
        if method is None:
            raise AttributeError(
                f"Provider '{provider_name}' has no method '{method_name}'"
            )

        credentials = self.all_credentials.get(provider_name)
        if not credentials:
            raise ValueError(f"No credentials for provider '{provider_name}'")

        http_client = await self._get_http_client_async(streaming=False)

        for credential in credentials:
            try:
                return await method(credential, http_client, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    lib_logger.warning(
                        f"Rate limited on credential {mask_credential(credential)} "
                        f"for {provider_name}.{method_name}, trying next"
                    )
                    continue
                raise
            except (httpx.HTTPError, TimeoutError, ConnectionError, ValueError, RuntimeError) as e:
                lib_logger.debug(
                    f"Failed on credential for {provider_name}.{method_name}: {type(e).__name__}: {e}, "
                    f"trying next"
                )
                continue

        raise RuntimeError(
            f"All credentials exhausted for {provider_name}.{method_name}"
        )

    # --- Anthropic API Compatibility Methods ---



