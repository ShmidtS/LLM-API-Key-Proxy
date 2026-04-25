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
from ..quota_reporter import QuotaReporter
from ..anthropic_adapter import AnthropicAdapter

_patch_aiohttp_connector()

import asyncio
import fnmatch
import logging
from collections.abc import Mapping
import httpx
import litellm
from litellm.litellm_core_utils.token_counter import token_counter
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ..anthropic_compat.models import (
        AnthropicMessagesRequest,
        AnthropicCountTokensRequest,
    )


lib_logger = logging.getLogger("rotator_library")


# Image-only model detection — these models are rejected by /chat/completions upstream
# and must be redirected to /v1/images/generations. Covers flux, z-image, dall-e, sd3, etc.
_IMAGE_ONLY_SUBSTRINGS = (
    "z-image",
    "flux-",
    "gpt-image",
    "dall-e",
    "sd3",
    "stable-diffusion",
    "imagen",
    "firefly",
    "cogview-4",
)
_IMAGE_ONLY_SUFFIXES = (
    "-image",
    "-image-pro",
    "-image-turbo",
    "-image-gen",
)
_IMAGE_ONLY_PATH_FRAGMENTS = ("/flux-", "/image-")

# Allow-list of params accepted by image-generation endpoints.
# Chat-specific params (messages, tools, stream, max_tokens, etc.)
# are deliberately excluded.
_IMAGE_PASSTHROUGH_PARAMS = {
    "n",
    "size",
    "quality",
    "style",
    "response_format",
    "user",
    "extra_headers",
    "extra_body",
    "timeout",
}

_IMAGE_NATIVE_PROVIDERS = {"fireworks", "fireworks_ai", "qwen", "dashscope", "zai", "z.ai"}


def _is_image_only_model(model: str) -> bool:
    """Return True if model name matches known image-only patterns.

    Used to redirect chat-completion calls for image models (flux, z-image,
    dall-e, etc.) to the image generation endpoint.
    """
    if not model:
        return False
    m = model.lower()
    if any(s in m for s in _IMAGE_ONLY_SUBSTRINGS):
        return True
    if any(m.endswith(sfx) for sfx in _IMAGE_ONLY_SUFFIXES):
        return True
    if any(frag in m for frag in _IMAGE_ONLY_PATH_FRAGMENTS):
        return True
    return False


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


class RotatingClient(HelpersMixin, StreamingMixin, RetryMixin):
    """
    A client that intelligently rotates and retries API keys using LiteLLM,
    with support for both streaming and non-streaming responses.
    """

    _MODEL_LIST_CACHE_TTL = 300.0

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
        self._provider_endpoints: Dict[str, str] = {}

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
        self._consecutive_quota_failures: Dict[str, int] = {}

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

    def _match_model_pattern(
        self,
        provider: str,
        model_id: str,
        pattern_dict: dict,
        wildcard_return: bool = False,
    ) -> bool:
        """
        Checks if a model matches any pattern in the given dict.

        Args:
            provider: Provider name
            model_id: Full model ID (e.g., "openai/gpt-4")
            pattern_dict: Dict mapping provider -> list of fnmatch patterns
            wildcard_return: Return value when pattern list is ["*"] (True for ignore, False for whitelist)

        Pattern examples:
        - "gpt-4" - exact match
        - "gpt-4*" - prefix wildcard (matches gpt-4, gpt-4-turbo, etc.)
        - "*-preview" - suffix wildcard
        - "*" - match all
        """
        model_provider = model_id.split("/")[0]
        if model_provider not in pattern_dict:
            return False

        pattern_list = pattern_dict[model_provider]
        if pattern_list == ["*"]:
            return wildcard_return

        try:
            provider_model_name = model_id.split("/", 1)[1]
        except IndexError:
            provider_model_name = model_id

        for pattern in pattern_list:
            if fnmatch.fnmatch(provider_model_name, pattern) or fnmatch.fnmatch(
                model_id, pattern
            ):
                return True
        return False

    def _is_model_ignored(self, provider: str, model_id: str) -> bool:
        """Checks if a model should be ignored based on the ignore list."""
        return self._match_model_pattern(
            provider, model_id, self.ignore_models, wildcard_return=True
        )

    def _is_model_whitelisted(self, provider: str, model_id: str) -> bool:
        """Checks if a model is explicitly whitelisted."""
        return self._match_model_pattern(
            provider, model_id, self.whitelist_models, wildcard_return=False
        )

    def get_oauth_credentials(self) -> Dict[str, List[str]]:
        return self.oauth_credentials

    def _is_image_only_model(self, model: str) -> bool:
        """Instance wrapper around module-level _is_image_only_model helper."""
        return _is_image_only_model(model)

    def _extract_prompt_from_chat_messages(self, messages: list) -> Optional[str]:
        """Extract the last user-message text content to use as image prompt.

        Supports both string content and list-of-parts content (OpenAI vision
        format). Returns None when no user text is found.
        """
        if not messages:
            return None
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                text = content.strip()
                return text or None
            if isinstance(content, list):
                parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text_val = part.get("text")
                        if isinstance(text_val, str) and text_val:
                            parts.append(text_val)
                if parts:
                    return "\n".join(parts)
        return None

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

    async def _resolve_model_id(self, model: str, provider: str) -> str:
        """
        Resolves the actual model ID to send to the provider.

        For custom models with name/ID mappings, returns the ID.
        Otherwise, returns the model name unchanged.

        Results are cached with TTL to avoid repeated provider lookups.
        Cache is invalidated when providers are refreshed.

        Args:
            model: Full model string with provider (e.g., "iflow/DS-v3.2")
            provider: Provider name (e.g., "iflow")

        Returns:
            Full model string with ID (e.g., "iflow/deepseek-v3.2")
        """
        cache_key = (model, provider)
        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            cached = self._resolve_model_id_cache.get(cache_key)
            if cached is not None:
                return cached

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
                    result = f"{provider}/{model_id}"
                    self._resolve_model_id_cache[cache_key] = result
                    return result

            # Fallback: use client's own model definitions
            model_id = self.model_definitions.get_model_id(provider, model_name)
            if model_id and model_id != model_name:
                result = f"{provider}/{model_id}"
                self._resolve_model_id_cache[cache_key] = result
                return result

            # No conversion needed, return original
            self._resolve_model_id_cache[cache_key] = model
            return model

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

        if _is_image_only_model(model):
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

    def _media_request(
        self,
        endpoint_fn: callable,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        return self._rate_limited_execute(
            endpoint_fn,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    async def aimage_generation(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """Generate an image via the image generation endpoint.

        Tries /v1/images/generations first. If the provider doesn't support
        that endpoint (400 Invalid path, 404 Not Found, etc.), falls back to
        /chat/completions with prompt conversion. The runtime error decides
        which path to take — no hardcoded provider lists.
        """
        # Auto-resolve unsupported image sizes — let the model pick the best fit
        size = kwargs.get("size")
        if size and size.lower() not in {
            "1024x1024",
            "1024x1536",
            "1536x1024",
            "1792x1024",
            "1024x1792",
            "auto",
        }:
            kwargs = kwargs.copy()
            kwargs["size"] = "auto"
            lib_logger.info(
                "Remapping unsupported image size %s to auto for model %s",
                size,
                kwargs.get("model", ""),
            )

        response_format = kwargs.get("response_format")
        if isinstance(response_format, Mapping):
            response_format_type = response_format.get("type")
            if response_format_type in {"image", "url"}:
                kwargs = kwargs.copy()
                kwargs["response_format"] = "url"
            elif response_format_type in {"b64_json", "base64"}:
                kwargs = kwargs.copy()
                kwargs["response_format"] = "b64_json"

        model = normalize_model_string(str(kwargs.get("model", "")))
        native_provider = kwargs.pop("_native_provider", None)
        provider = native_provider or extract_provider_from_model(model)
        model_name = model.split("/", 1)[1] if "/" in model else model
        api_base = str(kwargs.get("api_base", "")).lower()
        if provider == "openai" and ("dashscope" in api_base or model_name.startswith("z-image")):
            provider = "qwen"
        if provider in _IMAGE_NATIVE_PROVIDERS:
            kwargs = kwargs.copy()
            kwargs["model"] = model
            kwargs["_native_provider"] = provider
            return await self._rate_limited_execute(
                self._native_image_generation,
                request=request,
                pre_request_callback=pre_request_callback,
                **kwargs,
            )

        # Truncate long prompts — some models reject content > 1000 chars
        # at /images/generations (e.g. Qwen/DashScope returns 400).
        prompt = kwargs.get("prompt")
        if prompt and isinstance(prompt, str) and len(prompt) > 1000:
            kwargs = kwargs.copy()
            kwargs["prompt"] = prompt[:997] + "..."
            lib_logger.info(
                "Truncated image prompt from %d to 1000 chars for model %s",
                len(prompt), kwargs.get("model", ""),
            )

        try:
            response = await self._rate_limited_execute(
                litellm.aimage_generation,
                request=request,
                pre_request_callback=pre_request_callback,
                **kwargs,
            )
            if self._is_image_endpoint_mismatch_response(response):
                lib_logger.info(
                    "Provider doesn't support /images/generations, falling back to /chat/completions for model=%s",
                    kwargs.get("model", ""),
                )
                return await self._image_via_chat_completion(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **kwargs,
                )
            return response
        except (
            litellm.BadRequestError,
            litellm.NotFoundError,
            litellm.APIError,
            OpenAIError,
        ) as e:
            err_lower = str(e).lower()
            is_endpoint_mismatch = self._is_image_endpoint_mismatch_text(err_lower)
            is_html_404 = "<!doctype" in err_lower and ("not found" in err_lower or "404" in err_lower)
            if not is_endpoint_mismatch and not is_html_404:
                raise
            if _is_image_only_model(str(kwargs.get("model", ""))):
                lib_logger.error(
                    "Provider doesn't support /images/generations for image-only model=%s. Failing fast.",
                    kwargs.get("model", ""),
                )
                raise NoAvailableKeysError(
                    f"Model {kwargs.get('model', 'unknown')} is not supported by this provider for image generation"
                ) from e

            lib_logger.info(
                "Provider doesn't support /images/generations, falling back to /chat/completions for model=%s",
                kwargs.get("model", ""),
            )
            try:
                return await self._image_via_chat_completion(
                    request=request,
                    pre_request_callback=pre_request_callback,
                    **kwargs,
                )
            except (litellm.NotFoundError, OpenAIError) as chat_e:
                chat_err_lower = str(chat_e).lower()
                if (
                    isinstance(chat_e, OpenAIError)
                    and "not found" not in chat_err_lower
                    and "404" not in chat_err_lower
                ):
                    raise
                lib_logger.error(
                    "Provider doesn't support /images/generations or /chat/completions for image model=%s. Failing fast.",
                    kwargs.get("model", ""),
                )
                raise NoAvailableKeysError(
                    f"Model {kwargs.get('model', 'unknown')} is not supported by this provider for image generation"
                ) from chat_e

    async def _native_image_generation(self, **kwargs) -> Any:
        model = normalize_model_string(str(kwargs.get("model", "")))
        provider = kwargs.pop("_native_provider", None) or extract_provider_from_model(model)
        model_name = model.split("/", 1)[1] if "/" in model else model
        if provider == "openai" and model_name.startswith("z-image"):
            provider = "qwen"
        provider = {"dashscope": "qwen", "fireworks_ai": "fireworks", "z.ai": "zai"}.get(provider, provider)
        api_key = kwargs.pop("api_key")
        provider_plugin = self._get_provider_instance(provider)
        if not provider_plugin or not hasattr(provider_plugin, "native_image_generation"):
            raise ValueError(f"Unsupported native image provider: {provider}")
        http_client = await self._get_http_client_async(streaming=False)
        data = await provider_plugin.native_image_generation(
            http_client, api_key, timeout=kwargs.get("timeout", self.global_timeout), **kwargs
        )
        return self._native_image_response(data)

    def _native_image_response(self, data: Any) -> dict:
        import time as _time

        images = self._extract_native_images(data)
        return {"created": int(_time.time()), "data": images, "object": "list"}

    def _extract_native_images(self, data: Any) -> list[dict]:
        if not isinstance(data, Mapping):
            return []
        images = []
        for key in ("base64", "b64_json"):
            if isinstance(data.get(key), str):
                images.append({"b64_json": data[key]})
        for key in ("url", "image_url", "sample", "image"):
            if isinstance(data.get(key), str):
                value = data[key]
                if key == "image" and not value.startswith("http"):
                    images.append({"b64_json": value})
                else:
                    images.append({"url": value})
        result = data.get("result")
        if isinstance(result, str):
            images.append({"url": result})
        elif isinstance(result, Mapping):
            images.extend(self._extract_native_images(result))
        output = data.get("output")
        if isinstance(output, Mapping):
            for key in ("results", "images"):
                items = output.get(key)
                if isinstance(items, list):
                    for item in items:
                        images.extend(self._extract_native_images(item))
            choices = output.get("choices")
            if isinstance(choices, list):
                for choice in choices:
                    images.extend(self._extract_native_images(choice))
            for key in ("url", "image_url"):
                if isinstance(output.get(key), str):
                    images.append({"url": output[key]})
        if isinstance(data.get("data"), list):
            for item in data["data"]:
                images.extend(self._extract_native_images(item))
        for key in ("results", "images"):
            if isinstance(data.get(key), list):
                for item in data[key]:
                    images.extend(self._extract_native_images(item))
        return images

    def _is_image_endpoint_mismatch_response(self, response: Any) -> bool:
        if not isinstance(response, Mapping):
            return False
        error = response.get("error")
        if not isinstance(error, Mapping):
            return False
        error_type = str(error.get("type", "")).lower()
        message = str(error.get("message", "")).lower()
        return error_type == "invalid_request" and self._is_image_endpoint_mismatch_text(message)

    def _is_image_endpoint_mismatch_text(self, text: str) -> bool:
        return any(
            pattern in text
            for pattern in (
                "only accepts the path",
                "invalid_path",
                "path not found",
                "not found: /v1/images/generations",
                "/v1/images/generations",
                "images/generations endpoint",
                "image generation endpoint",
                "endpoint does not support",
                "endpoint not found",
            )
        )

    async def _image_via_chat_completion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """Route image generation through /chat/completions as fallback.

        Called when /v1/images/generations is rejected by the provider.
        Sends the prompt as a user message and converts the chat response
        into OpenAI image generation format.
        """
        import re
        import time as _time

        model = kwargs.get("model", "")
        prompt = kwargs.get("prompt", "")
        n = kwargs.get("n", 1)
        size = kwargs.get("size", "1024x1024")

        # Truncate prompt to avoid exceeding chat API content limits (e.g. Qwen)
        max_prompt_len = 2000
        if prompt and len(prompt) > max_prompt_len:
            prompt = prompt[:max_prompt_len - 3] + "..."
            lib_logger.info(
                "Truncated image prompt from %d to %d chars for chat fallback",
                len(kwargs.get("prompt", "")), max_prompt_len,
            )

        size_hint = f" (size: {size})" if size else ""
        user_content = f"Generate an image: {prompt}{size_hint}"

        chat_kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
        }
        # Drop response_format — chat APIs reject 'image' (only accept json_object/text)
        for key in ("temperature", "top_p", "seed"):
            if key in kwargs:
                chat_kwargs[key] = kwargs[key]

        chat_resp = await self._rate_limited_execute(
            litellm.acompletion,
            request=request,
            pre_request_callback=pre_request_callback,
            **chat_kwargs,
        )

        if isinstance(chat_resp, Mapping) and "error" in chat_resp:
            return chat_resp

        images = []
        for choice in chat_resp.get("choices") or []:
            msg = choice.get("message", {})
            content = msg.get("content", "") if isinstance(msg, dict) else str(msg)

            url_matches = re.findall(
                r"https?://\S+\.(?:png|jpg|jpeg|webp|gif)\S*", content
            )
            b64_matches = re.findall(
                r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", content
            )

            for url in url_matches:
                images.append({"url": url})
            for b64 in b64_matches:
                images.append({"b64_json": b64.split(",", 1)[1] if "," in b64 else b64})

            if not url_matches and not b64_matches and content.strip():
                images.append({"url": content.strip()})

        while len(images) < n and images:
            images.append(images[-1])

        return {
            "created": int(_time.time()),
            "data": images[:n] if n else images,
            "object": "list",
        }

    def aimage_edit(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """Edit an image via the image edit endpoint."""
        return self._media_request(
            litellm.aimage_edit,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    def aimage_variation(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """Create a variation of an image via the image variation endpoint."""
        return self._media_request(
            litellm.aimage_variation,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    def aspeech(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """Generate speech audio via the speech endpoint."""
        return self._media_request(
            litellm.aspeech,
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

    def atranscription(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """Transcribe audio via the transcription endpoint."""
        return self._media_request(
            litellm.atranscription,
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
                lib_logger.debug(
                    "Provider not available, skip preprompt token counting"
                )

        return base_count

    async def get_available_models(self, provider: str) -> List[str]:
        """Returns a list of available models for a specific provider, with caching."""
        lib_logger.info("Getting available models for provider: %s", provider)
        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            if provider in self._model_list_cache:
                cached_models, cached_at = self._model_list_cache[provider]
                if time.monotonic() - cached_at < self._MODEL_LIST_CACHE_TTL:
                    lib_logger.debug("Returning cached models for provider: %s", provider)
                    return cached_models

            credentials_for_provider = self.all_credentials.get(provider)
            if not credentials_for_provider:
                lib_logger.warning("No credentials for provider: %s", provider)
                return []

            shuffled_credentials = list(credentials_for_provider)
            offset = self._cred_offset.get(provider, 0)
            self._cred_offset[provider] = (offset + 1) % len(shuffled_credentials)
            shuffled_credentials = (
                shuffled_credentials[offset:] + shuffled_credentials[:offset]
            )

        provider_instance = self._get_provider_instance(provider)
        if provider_instance:
            # For providers with hardcoded models (like gemini_cli), we only need to call once.
            # For others, we might need to try multiple keys if one is invalid.
            # The current logic of iterating works for both, as the credential is not
            # always used in get_models.
            consecutive_auth_errors = 0
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

                    consecutive_auth_errors = 0  # Reset on success

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

                    async with lock:
                        self._model_list_cache[provider] = (final_models, time.monotonic())
                    return final_models
                except Exception as e:
                    classified_error = classify_error(e, provider=provider)
                    cred_display = mask_credential(credential)
                    is_auth_error = classified_error.error_type in (
                        "authentication",
                        "forbidden",
                    )
                    if is_auth_error:
                        consecutive_auth_errors += 1
                        lib_logger.warning(
                            f"Auth error for {provider} with {cred_display}: {classified_error.error_type} ({consecutive_auth_errors} consecutive)"
                        )
                        if consecutive_auth_errors >= 2:
                            lib_logger.warning(
                                f"Stopping model discovery for {provider}: {consecutive_auth_errors} consecutive auth errors"
                            )
                            break
                    else:
                        lib_logger.debug(
                            f"Failed to get models for provider {provider} with credential {cred_display}: {classified_error.error_type}. Trying next credential."
                        )
                    continue  # Try the next credential

        # Discovery failure is a degradation (static models still usable),
        # not a hard failure — downgrade to warning and include static count.
        static_fallback = []
        try:
            static_fallback = self.model_definitions.get_all_provider_models(provider)
        except Exception:
            static_fallback = []
        lib_logger.warning(
            "Failed to get models for provider %s after trying all credentials; "
            "provider unreachable, using %d static models",
            provider,
            len(static_fallback),
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
        if not hasattr(self, "_quota_reporter_instance"):
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
            except Exception as e:
                lib_logger.debug(
                    f"Failed on credential for {provider_name}.{method_name}: {e}, "
                    f"trying next"
                )
                continue

        raise RuntimeError(
            f"All credentials exhausted for {provider_name}.{method_name}"
        )

    # --- Anthropic API Compatibility Methods ---

    @property
    def anthropic_adapter(self):
        if not hasattr(self, "_anthropic_adapter_instance"):
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
