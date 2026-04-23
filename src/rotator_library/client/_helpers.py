# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Helper mixin for RotatingClient — headers, HTTP pool, priority cache,
litellm cache reset, safety settings, model ignore/whitelist, litellm logger,
provider resolution."""

import asyncio
import json
import logging
from ..utils.http_retry import compute_backoff_with_jitter
import time
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
from urllib.parse import urlparse

import httpx
import litellm
from ..utils.json_utils import json_deep_copy, json_loads

if TYPE_CHECKING:
    from ..error_handler import ClassifiedError

from ..env_cache import get_provider_env_cache
from ..error_types import mask_credential
from ..http_client_pool import HttpClientPool, get_http_pool
from ..providers.utilities import DEFAULT_GENERIC_SAFETY_SETTINGS, DEFAULT_SAFETY_SETTINGS

lib_logger = logging.getLogger("rotator_library")


class HelpersMixin:
    """Mixin with helper methods for RotatingClient."""

    _last_cache_reset_time: float = 0.0
    _pool_init_lock: asyncio.Lock | None = None

    def _build_request_headers(self, request: Optional[Any]) -> Dict[str, Any]:
        """Build a stable request headers dict for failure logging."""
        if request is None:
            return {}
        headers = getattr(request, "headers", None)
        return dict(headers) if headers else {}

    async def _sleep_within_budget(
        self, attempt: int, deadline: float, classified_error: "ClassifiedError"
    ) -> bool:
        """Sleep using shared retry policy when remaining budget allows it."""
        base_wait = compute_backoff_with_jitter(attempt, retry_after=classified_error.retry_after)
        wait_time = base_wait

        remaining_budget = deadline - time.monotonic()
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

    async def increment_quota_failures(self, credential_id: str, provider: str) -> bool:
        """
        Increment consecutive quota failure counter for a credential.

        Args:
            credential_id: The credential identifier
            provider: The provider name (for lock acquisition)

        Returns:
            True if rotation is needed (>= 3 consecutive failures), False otherwise
        """
        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            self._consecutive_quota_failures[credential_id] = (
                self._consecutive_quota_failures.get(credential_id, 0) + 1
            )
            count = self._consecutive_quota_failures[credential_id]
        lib_logger.debug(
            "Quota failure increment for %s: %s/3",
            mask_credential(credential_id), count,
        )
        return count >= 3

    async def reset_quota_failures(self, credential_id: str, provider: str) -> None:
        """
        Reset consecutive quota failure counter for a credential on success.

        Args:
            credential_id: The credential identifier
            provider: The provider name (for lock acquisition)
        """
        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            if credential_id in self._consecutive_quota_failures:
                del self._consecutive_quota_failures[credential_id]
        lib_logger.debug(
            "Quota failure counter reset for %s",
            mask_credential(credential_id),
        )

    async def _apply_quota_cooldown(
        self,
        provider: str,
        credential: str,
        classified_error: "ClassifiedError",
    ) -> None:
        """
        Apply cooldown for quota_exceeded errors using retry_after from
        the provider-specific parse_quota_error result.

        For ZAI INSUFFICIENT_BALANCE (reason='INSUFFICIENT_BALANCE'), this is
        an account-level error — all credentials of the provider share the same
        balance, so cooldown is propagated to every credential.
        """
        retry_after = classified_error.retry_after
        if not retry_after:
            return

        is_account_level = getattr(classified_error, "reason", None) == "INSUFFICIENT_BALANCE"

        if is_account_level:
            provider_creds = self.all_credentials.get(provider, [])
            for cred in provider_creds:
                await self._resilience.cooldown.start_cooldown(cred, retry_after)
            lib_logger.warning(
                "Account-level quota (INSUFFICIENT_BALANCE) for '%s': "
                "all %s credentials cooled down for %ss "
                "(until next day).",
                provider, len(provider_creds), retry_after,
            )
        else:
            await self._resilience.cooldown.start_cooldown(credential, retry_after)
            lib_logger.info(
                "Per-credential quota cooldown for %s: "
                "%ss.",
                mask_credential(credential), retry_after,
            )

    async def _build_credential_priority_cache(
        self, provider: str, credentials: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, str]]:
        """
        Build or update the credential priority cache for a provider.

        Uses TTLCache (maxsize=64, ttl=300s) for automatic expiry.
        The result (priorities, tier_names) tuple is cached directly
        for O(1) return on cache hits.

        Protected by per-provider lock to prevent concurrent cache corruption.

        Args:
            provider: Provider name
            credentials: List of credentials to cache priorities for

        Returns:
            Tuple of (credential_priorities, credential_tier_names)
        """
        lock = await self._lock_manager.get_lock(provider)
        async with lock:
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

    def _reset_litellm_client_cache(self) -> None:
        """
        Reset LiteLLM's internal HTTP client cache.

        LiteLLM caches async HTTP clients internally. When a connection error
        occurs (e.g., "Cannot send a request, as the client has been closed"),
        we need to clear this cache to force LiteLLM to create a fresh client.

        This addresses the issue where LiteLLM's cached client becomes unusable
        after certain network errors.

        Throttled: does not reset more than once per 60 seconds to avoid
        killing connection pooling on repeated auth errors.
        """
        now = time.monotonic()
        if now - self._last_cache_reset_time < 60.0:
            return
        self._last_cache_reset_time = now
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
            lib_logger.debug("Could not reset LiteLLM client cache: %s", e)

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
            if provider:
                provider_instance = self._get_provider_instance(provider)
                if provider_instance and hasattr(provider_instance, 'reset_auth_caches') and credential:
                    provider_instance.reset_auth_caches(credential)
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
        Double-check locking prevents concurrent initialization.
        """
        if self._http_pool is not None and self._pool_initialized:
            return self._http_pool
        if self._pool_init_lock is None:
            self._pool_init_lock = asyncio.Lock()
        async with self._pool_init_lock:
            # Double-check after acquiring lock
            if self._http_pool is not None and self._pool_initialized:
                return self._http_pool
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
        # Map of provider names to their default API base URLs
        # These are only used as fallbacks if no custom API_BASE is configured
        provider_urls = {
            "openai": "https://api.openai.com",
            "anthropic": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com",
            "antigravity": "https://api.antigravity.ai",
            "iflow": "https://api.iflow.ai",
        }

        # Build endpoint dict in a single pass, then derive list from it
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

        return list(set(self._provider_endpoints.values()))[:5]  # Dedupe and limit

    async def _get_http_client(self, streaming: bool = False) -> httpx.AsyncClient:
        """
        Get HTTP client from the pool (async version with lock protection).

        Prefer _get_http_client_async() for production use.

        Args:
            streaming: Whether this client will be used for streaming requests

        Returns:
            httpx.AsyncClient instance
        """
        # If pool reference not set, bind to the singleton pool
        # (this should rarely happen if initialize() is called properly)
        if self._http_pool is None or not self._pool_initialized:
            lib_logger.warning("HTTP pool accessed before initialization")
            pool = await self._ensure_http_pool()
        else:
            pool = self._http_pool
        return await pool.get_client_async(streaming=streaming)

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

    async def http_client(self) -> httpx.AsyncClient:
        """Async property that returns client from pool (non-streaming by default)."""
        return await self._get_http_client(streaming=False)

    def _parse_custom_cap_env_key(
        self, remainder: str
    ) -> Tuple[Optional[Union[int, Tuple[int, ...], str]], Optional[str]]:
        """Delegate to client_config.parse_custom_cap_env_key."""
        from ..client_config import parse_custom_cap_env_key
        return parse_custom_cap_env_key(remainder)

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
        if log_event_type in {"pre_api_call", "post_api_call"}:
            return  # Skip these verbose logs entirely

        # For successful calls or pre-call logs, log minimal fields without deep copy.
        if not log_data.get("exception"):
            lib_logger.debug(
                "LiteLLM Log: event=%s model=%s",
                log_data.get("log_event_type"),
                log_data.get("model", "N/A"),
            )
            return

        # For failures, extract key info to make debug logs more readable.
        model = log_data.get("model", "N/A")
        error_info = log_data.get("standard_logging_object", {}).get(
            "error_information", {}
        )
        error_class = error_info.get("error_class", "UnknownError")
        error_message = error_info.get(
            "error_message", str(log_data.get("exception", ""))
        )
        error_message = " ".join(error_message.split())  # Sanitize

        lib_logger.debug(
            "LiteLLM Callback Handled Error: Model=%s | "
            "Type=%s | Message='%s'",
            model, error_class, error_message,
        )

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
        that could override provider credentials, and strip internal kwargs
        (prefixed with ``_``) that must never reach litellm.

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
            elif key.startswith("_"):
                litellm_kwargs.pop(key, None)

    async def _apply_provider_headers(
        self, litellm_kwargs: Dict[str, Any], provider: str, credential: str
    ):
        """
        Apply correct provider headers and remove problematic client headers.
        """
        # Remove problematic headers from existing headers dict
        if "headers" in litellm_kwargs and isinstance(litellm_kwargs["headers"], dict):
            self._strip_client_headers(litellm_kwargs["headers"])

        # Add provider-specific headers from environment variables if configured
        # These headers should be used instead of any client-provided ones
        provider_headers_key = f"{provider.upper()}_API_HEADERS"
        provider_headers = get_provider_env_cache().get(provider_headers_key)

        if provider_headers:
            try:
                # Parse headers from JSON format
                headers_dict = json_loads(provider_headers)
                if isinstance(headers_dict, dict):
                    # Use headers parameter if available, otherwise create it
                    if "headers" not in litellm_kwargs:
                        litellm_kwargs["headers"] = {}

                    # Clean provider headers before merging
                    self._strip_client_headers(headers_dict)
                    litellm_kwargs["headers"].update(headers_dict)
                    lib_logger.debug(
                        "Applied provider-specific headers for %s from env",
                        provider,
                    )
            except (json.JSONDecodeError, ValueError) as e:
                lib_logger.warning(
                    "Failed to parse %s: %s",
                    provider_headers_key, e,
                )


