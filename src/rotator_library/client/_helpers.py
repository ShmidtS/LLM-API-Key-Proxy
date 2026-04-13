# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Helper mixin for RotatingClient — headers, HTTP pool, priority cache,
litellm cache reset, safety settings, model ignore/whitelist, litellm logger,
provider resolution."""

import asyncio
import fnmatch
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import httpx
import litellm
import orjson
from ..utils.json_utils import json_deep_copy

from ..config import CIRCUIT_BREAKER_PROVIDER_OVERRIDES
from ..env_cache import _provider_env_cache
from ..error_types import mask_credential, ClassifiedError
from ..http_client_pool import HttpClientPool, get_http_pool
from ..providers import PROVIDER_PLUGINS
from ..providers.openai_compatible_provider import OpenAICompatibleProvider
from ..providers.utilities import DEFAULT_GENERIC_SAFETY_SETTINGS, DEFAULT_SAFETY_SETTINGS
from ..utils.model_utils import get_or_create_provider_instance

lib_logger = logging.getLogger("rotator_library")


class HelpersMixin:
    """Mixin with helper methods for RotatingClient."""

    def _build_request_headers(self, request: Optional[Any]) -> Dict[str, Any]:
        """Build a stable request headers dict for failure logging."""
        if request is None:
            return {}
        headers = getattr(request, "headers", None)
        return dict(headers) if headers else {}

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

    async def _sleep_within_budget(
        self, attempt: int, deadline: float, classified_error: "ClassifiedError"
    ) -> bool:
        """Sleep using shared retry policy when remaining budget allows it."""
        base_wait = classified_error.retry_after or (2**attempt * random.uniform(0.5, 1.5))
        wait_time = base_wait

        remaining_budget = deadline - time.time()
        if wait_time > remaining_budget:
            return False

        await asyncio.sleep(wait_time)
        return True

    async def _process_rate_limit(
        self,
        provider: str,
        credential: str,
        error: Exception,
        error_body: Optional[str],
        classified_error: "ClassifiedError",
    ) -> bool:
        """
        Process a rate limit error through the unified 429 handler.

        Returns True if IP-level throttle was detected (circuit breaker opened).
        The caller can use the return value for logging; rotation loops now
        rely on circuit_breaker.can_attempt() instead of short-circuiting.
        """
        return await self._resilience.handle_429(
            provider=provider,
            credential=credential,
            error=error,
            error_body=error_body,
            classified_error=classified_error,
        )

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

    def _build_credential_priority_cache(
        self, provider: str, credentials: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Build or update the credential priority cache for a provider.

        Uses TTLCache (maxsize=64, ttl=300s) for automatic expiry.
        The result (priorities, tier_names) tuple is cached directly
        for O(1) return on cache hits.

        Args:
            provider: Provider name
            credentials: List of credentials to cache priorities for

        Returns:
            Tuple of (credential_priorities, credential_tier_names)
        """
        # Fast path: return cached result tuple directly
        cached_result = self._credential_priority_cache.get(provider + ":result")
        if cached_result is not None:
            return cached_result

        # Need to rebuild cache
        provider_plugin = self._get_provider_instance(provider)
        priorities = {}
        tier_names = {}
        cache_entry = {}

        if provider_plugin:
            has_priority = hasattr(provider_plugin, "get_credential_priority")
            has_tier_name = hasattr(provider_plugin, "get_credential_tier_name")

            for cred in credentials:
                cred_cache = {}

                if has_priority:
                    priority = provider_plugin.get_credential_priority(cred)
                    if priority is not None:
                        priorities[cred] = priority
                        cred_cache["priority"] = priority

                if has_tier_name:
                    tier_name = provider_plugin.get_credential_tier_name(cred)
                    if tier_name:
                        tier_names[cred] = tier_name
                        cred_cache["tier_name"] = tier_name

                if cred_cache:
                    cache_entry[cred] = cred_cache

        result = (priorities, tier_names)
        self._credential_priority_cache[provider] = cache_entry
        self._credential_priority_cache[provider + ":result"] = result

        return result

    def _invalidate_priority_cache(self, provider: str) -> None:
        """
        Invalidate the priority cache for a provider.

        Call this when credentials are added or removed.
        """
        self._credential_priority_cache.pop(provider, None)
        self._credential_priority_cache.pop(provider + ":result", None)

    def _reset_litellm_client_cache(self) -> None:
        """
        Reset LiteLLM's internal HTTP client cache.

        LiteLLM caches async HTTP clients internally. When a connection error
        occurs (e.g., "Cannot send a request, as the client has been closed"),
        we need to clear this cache to force LiteLLM to create a fresh client.

        This addresses the issue where LiteLLM's cached client becomes unusable
        after certain network errors.
        """
        try:
            # LiteLLM caches clients in litellm.llms.openai.openai module
            # We need to clear the async client cache
            if hasattr(litellm, "_async_client_cache"):
                litellm._async_client_cache.clear()
                lib_logger.debug("Cleared LiteLLM async client cache")

            # Also clear any provider-specific client caches
            from litellm.llms import custom_httpx

            if hasattr(custom_httpx, "httpx_handler"):
                handler = custom_httpx.httpx_handler
                if hasattr(handler, "_async_client_cache"):
                    handler._async_client_cache.clear()
                    lib_logger.debug("Cleared custom_httpx async client cache")

        except Exception as e:
            # Non-critical - just log and continue
            lib_logger.debug(f"Could not reset LiteLLM client cache: {e}")

    def _reset_cache_on_auth_error(
        self, classified_error, raw_exception: Optional[Exception] = None,
        provider=None, credential: Optional[str] = None,
    ) -> bool:
        """Reset LiteLLM client cache if error is auth-related.

        Checks both classified error type and raw HTTP status code for
        consistency across all error-handling code paths.

        Returns True if cache was reset.
        """
        if classified_error.error_type == "authentication":
            self._reset_litellm_client_cache()
            if provider and hasattr(provider, 'reset_auth_caches') and credential:
                provider.reset_auth_caches(credential)
            return True
        if raw_exception is not None:
            status = getattr(
                getattr(raw_exception, "response", None), "status_code", None
            )
            if status in (401, 403):
                self._reset_litellm_client_cache()
                return True
        return False

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

    def _sanitize_litellm_log(self, log_data: dict) -> dict:
        """
        Recursively removes large data fields and sensitive information from litellm log
        dictionaries to keep debug logs clean and secure.
        """
        if not isinstance(log_data, dict):
            return log_data

        # Keys to remove at any level of the dictionary
        keys_to_pop = [
            "messages",
            "input",
            "response",
            "data",
            "api_key",
            "api_base",
            "original_response",
            "additional_args",
        ]

        # Keys that might contain nested dictionaries to clean
        nested_keys = ["kwargs", "litellm_params", "model_info", "proxy_server_request"]

        # Create a deep copy to avoid modifying the original log object in memory
        clean_data = json_deep_copy(log_data)

        def clean_recursively(data_dict):
            if not isinstance(data_dict, dict):
                return

            # Remove sensitive/large keys
            for key in keys_to_pop:
                data_dict.pop(key, None)

            # Recursively clean nested dictionaries
            for key in nested_keys:
                if key in data_dict and isinstance(data_dict[key], dict):
                    clean_recursively(data_dict[key])

            # Also iterate through all values to find any other nested dicts
            for key, value in list(data_dict.items()):
                if isinstance(value, dict):
                    clean_recursively(value)

        clean_recursively(clean_data)
        return clean_data

    def _litellm_logger_callback(self, log_data: dict):
        """
        Callback function to redirect litellm's logs to the library's logger.
        This allows us to control the log level and destination of litellm's output.
        It also cleans up error logs for better readability in debug files.
        """
        # Filter out verbose pre_api_call and post_api_call logs
        log_event_type = log_data.get("log_event_type")
        if log_event_type in ["pre_api_call", "post_api_call"]:
            return  # Skip these verbose logs entirely

        # For successful calls or pre-call logs, a simple debug message is enough.
        if not log_data.get("exception"):
            sanitized_log = self._sanitize_litellm_log(log_data)
            # We log it at the DEBUG level to ensure it goes to the debug file
            # and not the console, based on the main.py configuration.
            lib_logger.debug(f"LiteLLM Log: {sanitized_log}")
            return

        # For failures, extract key info to make debug logs more readable.
        model = log_data.get("model", "N/A")
        call_id = log_data.get("litellm_call_id", "N/A")
        error_info = log_data.get("standard_logging_object", {}).get(
            "error_information", {}
        )
        error_class = error_info.get("error_class", "UnknownError")
        error_message = error_info.get(
            "error_message", str(log_data.get("exception", ""))
        )
        error_message = " ".join(error_message.split())  # Sanitize

        lib_logger.debug(
            f"LiteLLM Callback Handled Error: Model={model} | "
            f"Type={error_class} | Message='{error_message}'"
        )

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

    def _apply_default_safety_settings(
        self, litellm_kwargs: Dict[str, Any], provider: str
    ):
        """
        Ensure default Gemini safety settings are present when calling the Gemini provider.
        This will not override any explicit settings provided by the request. It accepts
        either OpenAI-compatible generic `safety_settings` (dict) or direct Gemini-style
        `safetySettings` (list of dicts). Missing categories will be added with safe defaults.
        """
        if provider != "gemini":
            return

        # If generic form is present, ensure missing generic keys are filled in
        if "safety_settings" in litellm_kwargs and isinstance(
            litellm_kwargs["safety_settings"], dict
        ):
            for k, v in DEFAULT_GENERIC_SAFETY_SETTINGS.items():
                if k not in litellm_kwargs["safety_settings"]:
                    litellm_kwargs["safety_settings"][k] = v
            return

        # If Gemini form is present, ensure missing gemini categories are appended
        if "safetySettings" in litellm_kwargs and isinstance(
            litellm_kwargs["safetySettings"], list
        ):
            present = {
                item.get("category")
                for item in litellm_kwargs["safetySettings"]
                if isinstance(item, dict)
            }
            for default_setting in DEFAULT_SAFETY_SETTINGS:
                if default_setting["category"] not in present:
                    litellm_kwargs["safetySettings"].append(dict(default_setting))
            return

        # Neither present: set generic defaults so provider conversion will translate them
        if (
            "safety_settings" not in litellm_kwargs
            and "safetySettings" not in litellm_kwargs
        ):
            litellm_kwargs["safety_settings"] = DEFAULT_GENERIC_SAFETY_SETTINGS.copy()

    # Pre-computed lookup sets for _strip_client_headers — avoids per-call list rebuild + O(n*m) scan
    _STRIP_EXACT: frozenset = frozenset({
        "authorization", "x-api-key", "api-key", "api_key",
        "anthropic-version", "anthropic-dangerous-direct-browser-access",
        "anthropic-beta",
    })
    _STRIP_PREFIXES: tuple = ("x-anthropic-",)

    def _strip_client_headers(self, litellm_kwargs: Dict[str, Any]) -> None:
        """
        Remove client-provided headers/api_key from top-level litellm_kwargs
        that could override provider credentials.

        Args:
            litellm_kwargs: The kwargs dict to clean in-place
        """
        for key in list(litellm_kwargs.keys()):
            if not isinstance(key, str):
                continue
            key_lower = key.lower()
            if key_lower in self._STRIP_EXACT:
                litellm_kwargs.pop(key, None)
            elif key_lower.startswith(self._STRIP_PREFIXES):
                litellm_kwargs.pop(key, None)

    async def _apply_provider_headers(
        self, litellm_kwargs: Dict[str, Any], provider: str, credential: str
    ):
        """
        Apply correct provider headers and remove problematic client headers.
        """

        def _remove_problematic_headers(headers_dict: dict, source: str):
            """Remove headers that interfere with litellm's provider routing."""
            problematic_keys = [
                "authorization", "x-api-key", "api-key", "api_key",
                "anthropic-version", "anthropic-dangerous-direct-browser-access",
                "anthropic-beta",
            ]
            removed = []
            for key in list(headers_dict.keys()):
                if isinstance(key, str):
                    key_lower = key.lower()
                    if any(
                        key_lower == pk.lower() or key_lower.startswith("x-anthropic-")
                        for pk in problematic_keys
                    ):
                        removed.append(key)
                        headers_dict.pop(key)
            if removed:
                lib_logger.debug(
                    f"Removed {source} headers that may interfere with litellm: {removed}"
                )

        # Remove problematic headers from existing headers dict
        if "headers" in litellm_kwargs and isinstance(litellm_kwargs["headers"], dict):
            _remove_problematic_headers(litellm_kwargs["headers"], "headers")

        # Add provider-specific headers from environment variables if configured
        # These headers should be used instead of any client-provided ones
        provider_headers_key = f"{provider.upper()}_API_HEADERS"
        provider_headers = _provider_env_cache.get(provider_headers_key)

        if provider_headers:
            try:
                # Parse headers from JSON format
                headers_dict = orjson.loads(provider_headers)
                if isinstance(headers_dict, dict):
                    # Use headers parameter if available, otherwise create it
                    if "headers" not in litellm_kwargs:
                        litellm_kwargs["headers"] = {}

                    # Clean provider headers before merging
                    _remove_problematic_headers(headers_dict, "provider env")
                    litellm_kwargs["headers"].update(headers_dict)
                    lib_logger.debug(
                        f"Applied provider-specific headers for {provider} from env"
                    )
            except (orjson.JSONDecodeError, ValueError) as e:
                lib_logger.warning(
                    f"Failed to parse {provider_headers_key}: {e}"
                )

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

        # No conversion needed, return original
        return model
