# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Retry mixin for RotatingClient — prepare context, execute_with_retry,
streaming_acompletion_with_retry, forced_streaming_acompletion."""

import asyncio
import codecs
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Optional

import httpx
import litellm
import orjson
from litellm.exceptions import APIConnectionError, BadRequestError, InvalidRequestError

from ..error_handler import (
    classify_error,
    get_retry_backoff,
    should_retry_same_key,
    should_rotate_on_error,
)
from ..error_types import (
    ContextOverflowError,
    NoAvailableKeysError,
    PreRequestCallbackError,
    RequestErrorAccumulator,
    mask_credential,
)
from ..failure_logger import log_failure
from ..request_sanitizer import sanitize_request_payload
from ..utils.json_utils import STREAM_DONE
from ..utils.model_utils import (
    extract_provider_from_model,
    normalize_model_string,
)

lib_logger = logging.getLogger("rotator_library")


@dataclass
class _RetryContext:
    model: str
    provider: str
    credentials_for_provider: list
    provider_plugin: Any
    deadline: float
    transaction_logger: Any
    tried_creds: set = field(default_factory=set)
    last_exception: Optional[Exception] = None
    parent_log_dir: Optional[str] = None
    credential_priorities: dict = field(default_factory=dict)
    credential_tier_names: dict = field(default_factory=dict)
    error_accumulator: Any = None


class RetryMixin:
    """Mixin with retry logic methods for RotatingClient."""

    async def _prepare_request_context(
        self,
        model: str,
        provider: str,
        credentials_for_provider: list,
        provider_plugin: Any,
    ) -> dict:
        """
        Prepare common request context shared by streaming and non-streaming retry paths.

        Handles:
        - Model ID resolution
        - Credential tier filtering (keep compatible/unknown, filter incompatible)
        - Building credential priority cache
        - Initializing RequestErrorAccumulator
        - Circuit breaker check

        Returns:
            dict with keys:
                - model: resolved model string
                - credentials: filtered credential list
                - credential_priorities: priority map (may be None)
                - credential_tier_names: tier names map
                - error_accumulator: RequestErrorAccumulator instance
        """
        # Resolve model ID early, before any credential operations
        # This ensures consistent model ID usage for acquisition, release, and tracking
        resolved_model = self._resolve_model_id(model, provider)
        if resolved_model != model:
            lib_logger.info(f"Resolved model '{model}' to '{resolved_model}'")
            model = resolved_model

        # [NEW] Filter by model tier requirement and build priority map
        credential_priorities = None
        if provider_plugin and hasattr(provider_plugin, "get_model_tier_requirement"):
            required_tier = provider_plugin.get_model_tier_requirement(model)
            if required_tier is not None:
                # Filter OUT only credentials we KNOW are too low priority
                # Keep credentials with unknown priority (None) - they might be high priority
                incompatible_creds = []
                compatible_creds = []
                unknown_creds = []

                has_priority = hasattr(provider_plugin, "get_credential_priority")
                if has_priority:
                    get_priority = provider_plugin.get_credential_priority
                for cred in credentials_for_provider:
                    if has_priority:
                        priority = get_priority(cred)
                        if priority is None:
                            # Unknown priority - keep it, will be discovered on first use
                            unknown_creds.append(cred)
                        elif priority <= required_tier:
                            # Known compatible priority
                            compatible_creds.append(cred)
                        else:
                            # Known incompatible priority (too low)
                            incompatible_creds.append(cred)
                    else:
                        # Provider doesn't support priorities - keep all
                        unknown_creds.append(cred)

                # If we have any known-compatible or unknown credentials, use them
                tier_compatible_creds = compatible_creds + unknown_creds
                if tier_compatible_creds:
                    credentials_for_provider = tier_compatible_creds
                    if compatible_creds and unknown_creds:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(compatible_creds)} known-compatible + {len(unknown_creds)} unknown-tier credentials."
                        )
                    elif compatible_creds:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(compatible_creds)} known-compatible credentials."
                        )
                    else:
                        lib_logger.info(
                            f"Model {model} requires priority <= {required_tier}. "
                            f"Using {len(unknown_creds)} unknown-tier credentials (will discover on use)."
                        )
                elif incompatible_creds:
                    # Only known-incompatible credentials remain
                    lib_logger.warning(
                        f"Model {model} requires priority <= {required_tier} credentials, "
                        f"but all {len(incompatible_creds)} known credentials have priority > {required_tier}. "
                        f"Request will likely fail."
                    )

        # Build priority map and tier names map for usage_manager (using cache)
        credential_priorities, credential_tier_names = (
            self._build_credential_priority_cache(provider, credentials_for_provider)
        )

        if credential_priorities:
            lib_logger.debug(
                f"Credential priorities for {provider}: {', '.join(f'P{p}={len([c for c in credentials_for_provider if credential_priorities.get(c) == p])}' for p in sorted(set(credential_priorities.values())))}"
            )

        # Initialize error accumulator for tracking errors across credential rotation
        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        # Circuit breaker check moved to retry loop - allows rotation to other keys
        # when circuit is OPEN instead of failing immediately on all provider requests

        return {
            "model": model,
            "credentials": credentials_for_provider,
            "credential_priorities": credential_priorities,
            "credential_tier_names": credential_tier_names,
            "error_accumulator": error_accumulator,
        }

    async def _prepare_retry_context(self, **kwargs) -> _RetryContext:
        model = normalize_model_string(kwargs.get("model"))
        if not model:
            raise ValueError("'model' is a required parameter.")
        kwargs["model"] = model

        provider = extract_provider_from_model(model)
        if not provider:
            raise ValueError("'model' must be in 'provider/model' format.")
        if provider not in self.all_credentials:
            raise ValueError(
                f"No API keys or OAuth credentials configured for provider: {provider}"
            )

        parent_log_dir = kwargs.pop("_parent_log_dir", None)
        deadline = time.time() + self.global_timeout

        transaction_logger = None
        if self.enable_request_logging:
            from ..transaction_logger import TransactionLogger
            transaction_logger = TransactionLogger(
                provider,
                model,
                enabled=True,
                api_format="oai",
                parent_dir=parent_log_dir,
            )
            transaction_logger.log_request(kwargs)

        credentials_for_provider = list(self.all_credentials[provider])
        offset = self._cred_offset.get(provider, 0)
        self._cred_offset[provider] = (offset + 1) % len(credentials_for_provider)
        credentials_for_provider = credentials_for_provider[offset:] + credentials_for_provider[:offset]

        provider_plugin = self._get_provider_instance(provider)

        ctx = await self._prepare_request_context(
            model, provider, credentials_for_provider, provider_plugin
        )

        return _RetryContext(
            model=ctx["model"],
            provider=provider,
            credentials_for_provider=ctx["credentials"],
            provider_plugin=provider_plugin,
            deadline=deadline,
            transaction_logger=transaction_logger,
            tried_creds=set(),
            last_exception=None,
            parent_log_dir=parent_log_dir,
            credential_priorities=ctx["credential_priorities"],
            credential_tier_names=ctx["credential_tier_names"],
            error_accumulator=ctx["error_accumulator"],
        )

    async def _execute_with_retry(
        self,
        api_call: callable,
        request: Optional[Any],
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """A generic retry mechanism for non-streaming API calls."""
        rc = await self._prepare_retry_context(**kwargs)
        kwargs["model"] = rc.model
        model = rc.model
        provider = rc.provider
        credentials_for_provider = rc.credentials_for_provider
        provider_plugin = rc.provider_plugin
        deadline = rc.deadline
        transaction_logger = rc.transaction_logger
        tried_creds = rc.tried_creds
        last_exception = rc.last_exception
        credential_priorities = rc.credential_priorities
        credential_tier_names = rc.credential_tier_names
        error_accumulator = rc.error_accumulator

        while (
            len(tried_creds) < len(credentials_for_provider)
            and time.time() < deadline
        ):
            current_cred = None
            key_acquired = False
            try:
                creds_to_try = [
                    c for c in credentials_for_provider if c not in tried_creds
                ]
                if not creds_to_try:
                    break

                # Check circuit breaker before acquiring key
                # Back off briefly if circuit is OPEN, then retry
                if not await self._resilience.can_attempt(provider):
                    remaining = await self._resilience.get_cooldown_remaining(provider)
                    backoff = min(remaining, 5.0)
                    lib_logger.debug(
                        f"Circuit breaker OPEN for provider '{provider}', "
                        f"backing off {backoff:.1f}s (recovery in {remaining:.0f}s)"
                    )
                    if time.time() + backoff < deadline:
                        await asyncio.sleep(backoff)
                    continue

                # Get count of credentials not on cooldown for this model
                availability_stats = (
                    await self.usage_manager.get_credential_availability_stats(
                        creds_to_try, model, credential_priorities
                    )
                )
                available_count = availability_stats["available"]
                total_count = len(credentials_for_provider)
                on_cooldown = availability_stats["on_cooldown"]
                fc_excluded = availability_stats["fair_cycle_excluded"]

                # Build compact exclusion breakdown
                exclusion_parts = []
                if on_cooldown > 0:
                    exclusion_parts.append(f"cd:{on_cooldown}")
                if fc_excluded > 0:
                    exclusion_parts.append(f"fc:{fc_excluded}")
                exclusion_str = (
                    f",{','.join(exclusion_parts)}" if exclusion_parts else ""
                )

                lib_logger.info(
                    f"Acquiring key for model {model}. Tried keys: {len(tried_creds)}/{available_count}({total_count}{exclusion_str})"
                )
                max_concurrent = self.max_concurrent_requests_per_key.get(provider, 1)

                current_cred = await self.usage_manager.acquire_key(
                    available_keys=creds_to_try,
                    model=model,
                    deadline=deadline,
                    max_concurrent=max_concurrent,
                    credential_priorities=credential_priorities,
                    credential_tier_names=credential_tier_names,
                    all_provider_credentials=credentials_for_provider,
                )
                key_acquired = True
                tried_creds.add(current_cred)

                litellm_kwargs = await self._prepare_request_kwargs(
                    kwargs, provider, current_cred, model
                )

                if provider_plugin and provider_plugin.has_custom_logic():
                    lib_logger.debug(
                        f"Provider '{provider}' has custom logic. Delegating call."
                    )
                    litellm_kwargs["credential_identifier"] = current_cred
                    litellm_kwargs["transaction_context"] = (
                        transaction_logger.get_context() if transaction_logger else None
                    )

                    # Retry loop for custom providers - mirrors streaming path error handling
                    for attempt in range(self.max_retries):
                        try:
                            lib_logger.info(
                                f"Attempting call with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                            )

                            if pre_request_callback:
                                try:
                                    await pre_request_callback(request, litellm_kwargs)
                                except Exception as e:
                                    if self.abort_on_callback_error:
                                        raise PreRequestCallbackError(
                                            f"Pre-request callback failed: {e}"
                                        ) from e
                                    else:
                                        lib_logger.warning(
                                            f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                        )

                            http_client = await self._get_http_client_async(
                                streaming=False
                            )
                            response = await provider_plugin.acompletion(
                                http_client, **litellm_kwargs
                            )

                            # For non-streaming, success is immediate
                            # Parallelize independent bookkeeping operations
                            await asyncio.gather(
                                self.usage_manager.record_success(
                                    current_cred, model, response
                                ),
                                self._resilience.record_success(provider),
                                self.usage_manager.release_key(current_cred, model),
                            )
                            key_acquired = False

                            # Log response to transaction logger
                            if transaction_logger:
                                response_data = (
                                    response.model_dump()
                                    if hasattr(response, "model_dump")
                                    else response
                                )
                                transaction_logger.log_response(response_data)

                            # Reset consecutive quota failures on success
                            self.reset_quota_failures(current_cred)

                            return response

                        except (
                            litellm.RateLimitError,
                            httpx.HTTPStatusError,
                        ) as e:
                            last_exception = e
                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            # Reset LiteLLM client cache on auth errors (401/403)
                            # Bad credentials can poison cached clients; don't clear on 429/500
                            self._reset_cache_on_auth_error(classified_error, e, provider=provider, credential=current_cred)

                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )

                            # Record in accumulator for client reporting
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                await self._process_rate_limit(
                                    provider, current_cred, e, str(e) if e else None, classified_error
                                )
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}) during custom provider call. Failing."
                                )
                                raise last_exception

                            # Only rate_limit errors can indicate IP-level throttling;
                            # quota_exceeded / invalid_request / auth errors are per-credential
                            if classified_error.error_type == "rate_limit":
                                await self._process_rate_limit(
                                    provider, current_cred, e, str(e) if e else None, classified_error
                                )

                            # Track consecutive quota failures and force rotation if needed
                            if classified_error.error_type == "quota_exceeded":
                                if self.increment_quota_failures(current_cred):
                                    lib_logger.error(
                                        f"Cred {mask_credential(current_cred)} quota failure limit reached (3/3), forcing rotation."
                                    )
                                    await self.usage_manager.record_failure(
                                        current_cred, model, classified_error
                                    )
                                    break  # Force rotation

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            lib_logger.warning(
                                f"Cred {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code}). Rotating."
                            )
                            break  # Rotate to next credential

                        except (
                            APIConnectionError,
                            litellm.InternalServerError,
                            litellm.ServiceUnavailableError,
                            RuntimeError,  # "Cannot send a request, as the client has been closed"
                        ) as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred,
                                model,
                                classified_error,
                                increment_consecutive_failures=False,
                            )

                            if attempt >= self.max_retries - 1:
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} failed after max retries. Rotating."
                                )
                                break

                            if not await self._sleep_within_budget(
                                attempt, deadline, classified_error
                            ):
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    "Retry wait exceeds budget. Rotating."
                                )
                                break

                            lib_logger.warning(
                                f"Cred {mask_credential(current_cred)} server error. Retrying within remaining budget."
                            )

                            # Reset LiteLLM internal HTTP client cache on connection errors
                            if isinstance(
                                e, RuntimeError
                            ) and "client has been closed" in str(e):
                                self._reset_litellm_client_cache()

                            # CRITICAL: Ensure HTTP client is usable before retry
                            # Connection errors can leave the client in a closed state
                            await self._get_http_client_async(streaming=False)
                            continue

                    # If the inner loop breaks, it means the key failed and we need to rotate.
                    # Continue to the next iteration of the outer while loop to pick a new key.
                    continue

                else:  # This is the standard API Key / litellm-handled provider logic
                    is_oauth = provider in self.oauth_providers
                    if is_oauth:  # Standard OAuth provider (not custom)
                        # ... (logic to set headers) ...
                        pass
                    else:  # API Key
                        litellm_kwargs["api_key"] = current_cred

                    # [FIX] Remove problematic headers and add correct provider headers
                    # This ensures that authorization/x-api-key from client requests
                    # are replaced with the correct values from configuration
                    await self._apply_provider_headers(litellm_kwargs, provider, current_cred)

                    if provider_plugin:
                        # Ensure default Gemini safety settings are present (without overriding request)
                        try:
                            self._apply_default_safety_settings(
                                litellm_kwargs, provider
                            )
                        except Exception as exc:
                            # If anything goes wrong here, avoid breaking the request flow.
                            lib_logger.warning(
                                "Could not apply default safety settings for %s: %s; continuing.",
                                provider, type(exc).__name__,
                            )

                        if "safety_settings" in litellm_kwargs:
                            converted_settings = (
                                provider_plugin.convert_safety_settings(
                                    litellm_kwargs["safety_settings"]
                                )
                            )
                            if converted_settings is not None:
                                litellm_kwargs["safety_settings"] = converted_settings
                            else:
                                del litellm_kwargs["safety_settings"]

                    if provider == "gemini" and provider_plugin:
                        provider_plugin.handle_thinking_parameter(
                            litellm_kwargs, model
                        )
                    if provider == "nvidia" and provider_plugin:
                        provider_plugin.handle_thinking_parameter(
                            litellm_kwargs, model
                        )

                    if "gemma-3" in model and "messages" in litellm_kwargs:
                        litellm_kwargs["messages"] = [
                            (
                                {"role": "user", "content": m["content"]}
                                if m.get("role") == "system"
                                else m
                            )
                            for m in litellm_kwargs["messages"]
                        ]

                    litellm_kwargs, should_reject = sanitize_request_payload(
                        litellm_kwargs, model, registry=self._model_registry
                    )

                    # Reject request if input exceeds context window
                    if should_reject:
                        raise ContextOverflowError(
                            f"Input tokens exceed context window for model {model}. "
                            f"Request rejected to prevent API error."
                        )

                    # If the provider is 'nvidia', set the custom provider for LiteLLM
                    # (litellm knows nvidia as 'nvidia_nim') and strip the prefix from the model name.
                    if provider == "nvidia":
                        litellm_kwargs["custom_llm_provider"] = "nvidia_nim"
                        litellm_kwargs["model"] = model.split("/", 1)[1]

                    # Inception Labs also requires model name without prefix
                    # This MUST happen before convert_for_litellm and AllProviders routing
                    # to avoid double prefixing (inception/mercury-2 → mercury-2 → openai/mercury-2)
                    # Use rsplit to handle both "inception/mercury-2" and "openai/inception/mercury-2"
                    if provider == "inception":
                        litellm_kwargs["model"] = model.rsplit("/", 1)[1]

                    for attempt in range(self.max_retries):
                        try:
                            lib_logger.info(
                                f"Attempting call with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                            )

                            if pre_request_callback:
                                try:
                                    await pre_request_callback(request, litellm_kwargs)
                                except Exception as e:
                                    if self.abort_on_callback_error:
                                        raise PreRequestCallbackError(
                                            f"Pre-request callback failed: {e}"
                                        ) from e
                                    else:
                                        lib_logger.warning(
                                            f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                        )

                            # Convert model parameters for custom providers right before LiteLLM call
                            final_kwargs = self.provider_config.convert_for_litellm(
                                **litellm_kwargs
                            )

                            response = await api_call(
                                **final_kwargs,
                                logger_fn=self._litellm_logger_callback,
                            )

                            # Parallelize independent bookkeeping operations
                            await asyncio.gather(
                                self.usage_manager.record_success(
                                    current_cred, model, response
                                ),
                                self._resilience.record_success(provider),
                                self.usage_manager.release_key(current_cred, model),
                            )
                            key_acquired = False

                            # Log response to transaction logger
                            if transaction_logger:
                                response_data = (
                                    response.model_dump()
                                    if hasattr(response, "model_dump")
                                    else response
                                )
                                transaction_logger.log_response(response_data)

                            # Reset consecutive quota failures on success
                            self.reset_quota_failures(current_cred)

                            return response

                        except litellm.RateLimitError as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )
                            classified_error = classify_error(e, provider=provider)

                            # Extract a clean error message for the user-facing log
                            error_message = str(e).split("\n")[0]

                            # Record in accumulator for client reporting
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            lib_logger.info(
                                f"Key {mask_credential(current_cred)} hit rate limit for {model}. Rotating key."
                            )

                            # Only rate_limit errors can indicate IP-level throttling;
                            # quota_exceeded / invalid_request / auth errors are per-credential
                            if classified_error.error_type == "rate_limit":
                                await self._process_rate_limit(
                                    provider, current_cred, e, str(e) if e else None, classified_error
                                )

                            # Track consecutive quota failures and force rotation if needed
                            if classified_error.error_type == "quota_exceeded":
                                if self.increment_quota_failures(current_cred):
                                    lib_logger.error(
                                        f"Cred {mask_credential(current_cred)} quota failure limit reached (3/3), forcing rotation."
                                    )
                                    await self.usage_manager.record_failure(
                                        current_cred, model, classified_error
                                    )
                                    break  # Force rotation

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            break  # Move to the next key

                        except (
                            APIConnectionError,
                            litellm.InternalServerError,
                            litellm.ServiceUnavailableError,
                            RuntimeError,  # "Cannot send a request, as the client has been closed"
                        ) as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            # Reset quota failures on connection errors (not quota-related)
                            self.reset_quota_failures(current_cred)

                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred,
                                model,
                                classified_error,
                                increment_consecutive_failures=False,
                            )

                            if attempt >= self.max_retries - 1:
                                # Record in accumulator only on final failure for this key
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    f"Key {mask_credential(current_cred)} failed after max retries due to server error. Rotating."
                                )
                                break  # Move to the next key

                            # For temporary errors, wait before retrying with the same key.
                            base_wait = classified_error.retry_after or (2**attempt * random.uniform(0.5, 1.5))
                            wait_time = base_wait
                            remaining_budget = deadline - time.time()

                            # If the required wait time exceeds the budget, don't wait; rotate to the next key immediately.
                            if wait_time > remaining_budget:
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )
                                lib_logger.warning(
                                    f"Retry wait ({wait_time:.2f}s) exceeds budget ({remaining_budget:.2f}s). Rotating key."
                                )
                                break

                            lib_logger.warning(
                                f"Key {mask_credential(current_cred)} server error. Retrying in {wait_time:.2f}s."
                            )
                            await asyncio.sleep(wait_time)

                            # Reset LiteLLM internal HTTP client cache on connection errors
                            if isinstance(
                                e, RuntimeError
                            ) and "client has been closed" in str(e):
                                self._reset_litellm_client_cache()

                            # CRITICAL: Ensure HTTP client is usable before retry
                            # Connection errors can leave the client in a closed state
                            await self._get_http_client_async(streaming=False)
                            continue  # Retry with the same key

                        except httpx.HTTPStatusError as e:
                            # Handle HTTP errors from httpx (e.g., from custom providers like Antigravity)
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )

                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            # Reset LiteLLM client cache on auth errors (401/403)
                            # Bad credentials can poison cached clients; don't clear on 429/500
                            self._reset_cache_on_auth_error(classified_error, e, provider=provider, credential=current_cred)

                            lib_logger.warning(
                                f"Key {mask_credential(current_cred)} HTTP {e.response.status_code} ({classified_error.error_type})."
                            )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                await self._process_rate_limit(
                                    provider, current_cred, e, str(e) if e else None, classified_error
                                )
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}). Failing request."
                                )
                                raise last_exception

                            # Record in accumulator after confirming it's a rotatable error
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            # Only rate_limit errors can indicate IP-level throttling;
                            # quota_exceeded / invalid_request / auth errors are per-credential
                            if classified_error.error_type == "rate_limit":
                                await self._process_rate_limit(
                                    provider, current_cred, e, str(e) if e else None, classified_error
                                )

                            # Check if we should retry same key (server errors with retries left)
                            if (
                                should_retry_same_key(classified_error)
                                and attempt < self.max_retries - 1
                            ):
                                base_wait = classified_error.retry_after or (2**attempt * random.uniform(0.5, 1.5))
                                wait_time = base_wait
                                remaining_budget = deadline - time.time()
                                if wait_time <= remaining_budget:
                                    lib_logger.warning(
                                        f"Server error, retrying same key in {wait_time:.2f}s."
                                    )
                                    await asyncio.sleep(wait_time)

                                    # CRITICAL: Ensure HTTP client is usable before retry
                                    # Connection errors can leave the client in a closed state
                                    await self._get_http_client_async(streaming=False)
                                    continue

                            # Record failure and rotate to next key
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            lib_logger.info(
                                f"Rotating to next key after {classified_error.error_type} error."
                            )
                            break

                        except Exception as e:
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )

                            if request and await request.is_disconnected():
                                lib_logger.warning(
                                    f"Client disconnected. Aborting retries for {mask_credential(current_cred)}."
                                )
                                raise last_exception

                            classified_error = classify_error(e, provider=provider)
                            error_message = str(e).split("\n")[0]

                            lib_logger.warning(
                                f"Key {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code})."
                            )

                            # Only rate_limit errors can indicate IP-level throttling;
                            # quota_exceeded / invalid_request / auth errors are per-credential
                            if classified_error.error_type == "rate_limit":
                                await self._process_rate_limit(
                                    provider, current_cred, e, str(e) if e else None, classified_error
                                )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}). Failing request."
                                )
                                raise last_exception

                            # Record in accumulator after confirming it's a rotatable error
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message
                            )

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            break  # Try next key for other errors
            finally:
                if key_acquired and current_cred:
                    await self.usage_manager.release_key(current_cred, model)

        # Check if we exhausted all credentials or timed out
        if time.time() >= deadline:
            error_accumulator.timeout_occurred = True

        if error_accumulator.has_errors():
            # Log concise summary for server logs
            lib_logger.error(error_accumulator.build_log_message())

            # Return the structured error response for the client
            return error_accumulator.build_client_error_response()

        # Return None to indicate failure without error details (shouldn't normally happen)
        lib_logger.warning(
            "Unexpected state: request failed with no recorded errors. "
            "This may indicate a logic error in error tracking."
        )
        return None

    async def _streaming_acompletion_with_retry(
        self,
        request: Optional[Any],
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """A dedicated generator for retrying streaming completions with full request preparation and per-key retries."""
        from ._streaming import _StreamedException

        rc = await self._prepare_retry_context(**kwargs)
        kwargs["model"] = rc.model
        model = rc.model
        provider = rc.provider
        credentials_for_provider = rc.credentials_for_provider
        provider_plugin = rc.provider_plugin
        deadline = rc.deadline
        transaction_logger = rc.transaction_logger
        tried_creds = rc.tried_creds
        last_exception = rc.last_exception
        credential_priorities = rc.credential_priorities
        credential_tier_names = rc.credential_tier_names
        error_accumulator = rc.error_accumulator

        consecutive_quota_failures = 0

        try:
            while (
                len(tried_creds) < len(credentials_for_provider)
                and time.time() < deadline
            ):
                current_cred = None
                key_acquired = False
                try:
                    creds_to_try = [
                        c for c in credentials_for_provider if c not in tried_creds
                    ]
                    if not creds_to_try:
                        lib_logger.warning(
                            f"All credentials for provider {provider} have been tried. No more credentials to rotate to."
                        )
                        break

                    # Check circuit breaker before acquiring key
                    # Back off briefly if circuit is OPEN, then retry
                    if not await self._resilience.can_attempt(provider):
                        remaining = await self._resilience.get_cooldown_remaining(provider)
                        backoff = min(remaining, 5.0)
                        lib_logger.debug(
                            f"Circuit breaker OPEN for provider '{provider}', "
                            f"backing off {backoff:.1f}s (recovery in {remaining:.0f}s)"
                        )
                        if time.time() + backoff < deadline:
                            await asyncio.sleep(backoff)
                        continue

                    # Get count of credentials not on cooldown for this model
                    availability_stats = (
                        await self.usage_manager.get_credential_availability_stats(
                            creds_to_try, model, credential_priorities
                        )
                    )
                    available_count = availability_stats["available"]
                    total_count = len(credentials_for_provider)
                    on_cooldown = availability_stats["on_cooldown"]
                    fc_excluded = availability_stats["fair_cycle_excluded"]

                    # Build compact exclusion breakdown
                    exclusion_parts = []
                    if on_cooldown > 0:
                        exclusion_parts.append(f"cd:{on_cooldown}")
                    if fc_excluded > 0:
                        exclusion_parts.append(f"fc:{fc_excluded}")
                    exclusion_str = (
                        f",{','.join(exclusion_parts)}" if exclusion_parts else ""
                    )

                    lib_logger.info(
                        f"Acquiring credential for model {model}. Tried credentials: {len(tried_creds)}/{available_count}({total_count}{exclusion_str})"
                    )
                    max_concurrent = self.max_concurrent_requests_per_key.get(
                        provider, 1
                    )
                    current_cred = await self.usage_manager.acquire_key(
                        available_keys=creds_to_try,
                        model=model,
                        deadline=deadline,
                        max_concurrent=max_concurrent,
                        credential_priorities=credential_priorities,
                        credential_tier_names=credential_tier_names,
                        all_provider_credentials=credentials_for_provider,
                    )
                    key_acquired = True
                    tried_creds.add(current_cred)

                    litellm_kwargs = kwargs.copy()

                    # [FIX] Remove client-provided headers/api_key that could override provider credentials
                    self._strip_client_headers(litellm_kwargs)

                    if "reasoning_effort" in kwargs:
                        litellm_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]

                    # [NEW] Merge provider-specific params
                    if provider in self.litellm_provider_params:
                        litellm_kwargs["litellm_params"] = {
                            **self.litellm_provider_params[provider],
                            **litellm_kwargs.get("litellm_params", {}),
                        }


                    # Model ID is already resolved before the loop, and kwargs['model'] is updated.
                    # No further resolution needed here.

                    # Apply model-specific options for custom providers
                    if provider_plugin and hasattr(
                        provider_plugin, "get_model_options"
                    ):
                        model_options = provider_plugin.get_model_options(model)
                        if model_options:
                            # Merge model options into litellm_kwargs
                            for key, value in model_options.items():
                                if key == "reasoning_effort":
                                    litellm_kwargs["reasoning_effort"] = value
                                elif key not in litellm_kwargs:
                                    litellm_kwargs[key] = value
                    if provider_plugin and provider_plugin.has_custom_logic():
                        lib_logger.debug(
                            f"Provider '{provider}' has custom logic. Delegating call."
                        )
                        litellm_kwargs["credential_identifier"] = current_cred
                        litellm_kwargs["transaction_context"] = (
                            transaction_logger.get_context()
                            if transaction_logger
                            else None
                        )

                        for attempt in range(self.max_retries):
                            try:
                                lib_logger.info(
                                    f"Attempting stream with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                                )

                                if pre_request_callback:
                                    try:
                                        await pre_request_callback(
                                            request, litellm_kwargs
                                        )
                                    except Exception as e:
                                        if self.abort_on_callback_error:
                                            raise PreRequestCallbackError(
                                                f"Pre-request callback failed: {e}"
                                            ) from e
                                        else:
                                            lib_logger.warning(
                                                f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                            )

                                http_client = await self._get_http_client_async(
                                    streaming=True
                                )
                                response = await provider_plugin.acompletion(
                                    http_client, **litellm_kwargs
                                )

                                lib_logger.info(
                                    f"Stream connection established for credential {mask_credential(current_cred)}. Processing response."
                                )

                                key_acquired = False
                                stream_generator = self._safe_streaming_wrapper(
                                    response,
                                    current_cred,
                                    model,
                                    request,
                                    provider_plugin,
                                )

                                # Skip transaction logging wrapper when disabled (avoids extra async gen layer)
                                if transaction_logger:
                                    async for chunk in self._transaction_logging_stream_wrapper(
                                        stream_generator, transaction_logger, kwargs
                                    ):
                                        yield chunk
                                else:
                                    async for chunk in stream_generator:
                                        yield chunk
                                return

                            except (
                                _StreamedException,
                                litellm.RateLimitError,
                                httpx.HTTPStatusError,
                                BadRequestError,
                                InvalidRequestError,
                            ) as e:
                                last_exception = e
                                # If the exception is our custom wrapper, unwrap the original error
                                original_exc = getattr(e, "data", e)
                                classified_error = classify_error(
                                    original_exc, provider=provider
                                )
                                error_message = str(original_exc).split("\n")[0]

                                # Reset LiteLLM client cache on auth errors (401/403)
                                self._reset_cache_on_auth_error(classified_error, original_exc, provider=provider, credential=current_cred)

                                log_failure(
                                    api_key=current_cred,
                                    model=model,
                                    attempt=attempt + 1,
                                    error=e,
                                    request_headers=(
                                        dict(request.headers) if request else {}
                                    ),
                                )

                                # Record in accumulator for client reporting
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )

                                # Check if this error should trigger rotation
                                if not should_rotate_on_error(classified_error):
                                    await self._process_rate_limit(
                                        provider, current_cred, e, str(e) if e else None, classified_error
                                    )
                                    lib_logger.error(
                                        f"Non-recoverable error ({classified_error.error_type}) during custom stream. Failing."
                                    )
                                    raise last_exception

                                # Only rate_limit errors can indicate IP-level throttling;
                                # quota_exceeded / invalid_request / auth errors are per-credential
                                if classified_error.error_type == "rate_limit":
                                    await self._process_rate_limit(
                                        provider, current_cred, e, str(e) if e else None, classified_error
                                    )

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code}). Rotating."
                                )
                                break

                            except (
                                APIConnectionError,
                                litellm.InternalServerError,
                                litellm.ServiceUnavailableError,
                                RuntimeError,  # "Cannot send a request, as the client has been closed"
                            ) as e:
                                last_exception = e
                                log_failure(
                                    api_key=current_cred,
                                    model=model,
                                    attempt=attempt + 1,
                                    error=e,
                                    request_headers=(
                                        dict(request.headers) if request else {}
                                    ),
                                )
                                classified_error = classify_error(e, provider=provider)
                                error_message = str(e).split("\n")[0]

                                # Provider-level error: don't increment consecutive failures
                                await self.usage_manager.record_failure(
                                    current_cred,
                                    model,
                                    classified_error,
                                    increment_consecutive_failures=False,
                                )

                                if attempt >= self.max_retries - 1:
                                    error_accumulator.record_error(
                                        current_cred, classified_error, error_message
                                    )
                                    lib_logger.warning(
                                        f"Cred {mask_credential(current_cred)} failed after max retries. Rotating."
                                    )
                                    break

                                # Reset LiteLLM internal HTTP client cache on connection errors
                                if isinstance(
                                    e, RuntimeError
                                ) and "client has been closed" in str(e):
                                    self._reset_litellm_client_cache()

                                base_wait = classified_error.retry_after or (2**attempt * random.uniform(0.5, 1.5))
                                wait_time = base_wait
                                remaining_budget = deadline - time.time()
                                if wait_time > remaining_budget:
                                    error_accumulator.record_error(
                                        current_cred, classified_error, error_message
                                    )
                                    lib_logger.warning(
                                        f"Retry wait ({wait_time:.2f}s) exceeds budget. Rotating."
                                    )
                                    break

                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} server error. Retrying in {wait_time:.2f}s."
                                )
                                await asyncio.sleep(wait_time)

                                # CRITICAL: Ensure HTTP client is usable before retry
                                # Connection errors can leave the client in a closed state
                                await self._get_http_client_async(streaming=True)
                                continue

                            except Exception as e:
                                last_exception = e
                                log_failure(
                                    api_key=current_cred,
                                    model=model,
                                    attempt=attempt + 1,
                                    error=e,
                                    request_headers=(
                                        dict(request.headers) if request else {}
                                    ),
                                )
                                classified_error = classify_error(e, provider=provider)
                                error_message = str(e).split("\n")[0]

                                # Record in accumulator
                                error_accumulator.record_error(
                                    current_cred, classified_error, error_message
                                )

                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} {classified_error.error_type} (HTTP {classified_error.status_code})."
                                )

                                # Check if this error should trigger rotation
                                if not should_rotate_on_error(classified_error):
                                    await self._process_rate_limit(
                                        provider, current_cred, e, str(e) if e else None, classified_error
                                    )
                                    lib_logger.error(
                                        f"Non-recoverable error ({classified_error.error_type}). Failing."
                                    )
                                    raise last_exception

                                # Only rate_limit errors can indicate IP-level throttling;
                                # quota_exceeded / invalid_request / auth errors are per-credential
                                if classified_error.error_type == "rate_limit":
                                    await self._process_rate_limit(
                                        provider, current_cred, e, str(e) if e else None, classified_error
                                    )

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                break

                        # If the inner loop breaks, it means the key failed and we need to rotate.
                        # Continue to the next iteration of the outer while loop to pick a new key.
                        continue

                    else:  # This is the standard API Key / litellm-handled provider logic
                        is_oauth = provider in self.oauth_providers
                        if is_oauth:  # Standard OAuth provider (not custom)
                            # ... (logic to set headers) ...
                            pass
                        else:  # API Key
                            litellm_kwargs["api_key"] = current_cred

                    # [FIX] Remove problematic headers and add correct provider headers
                    # This ensures that authorization/x-api-key from client requests
                    # are replaced with the correct values from configuration
                    await self._apply_provider_headers(litellm_kwargs, provider, current_cred)

                    if provider_plugin:
                        # Ensure default Gemini safety settings are present (without overriding request)
                        try:
                            self._apply_default_safety_settings(
                                litellm_kwargs, provider
                            )
                        except Exception as exc:
                            lib_logger.warning(
                                "Could not apply default safety settings for streaming path %s: %s; continuing.",
                                provider, type(exc).__name__,
                            )

                        if "safety_settings" in litellm_kwargs:
                            converted_settings = (
                                provider_plugin.convert_safety_settings(
                                    litellm_kwargs["safety_settings"]
                                )
                            )
                            if converted_settings is not None:
                                litellm_kwargs["safety_settings"] = converted_settings
                            else:
                                del litellm_kwargs["safety_settings"]

                    if provider == "gemini" and provider_plugin:
                        provider_plugin.handle_thinking_parameter(
                            litellm_kwargs, model
                        )
                    if provider == "nvidia" and provider_plugin:
                        provider_plugin.handle_thinking_parameter(
                            litellm_kwargs, model
                        )

                    if "gemma-3" in model and "messages" in litellm_kwargs:
                        litellm_kwargs["messages"] = [
                            (
                                {"role": "user", "content": m["content"]}
                                if m.get("role") == "system"
                                else m
                            )
                            for m in litellm_kwargs["messages"]
                        ]

                    litellm_kwargs, should_reject = sanitize_request_payload(
                        litellm_kwargs, model, registry=self._model_registry
                    )

                    # Reject request if input exceeds context window
                    if should_reject:
                        raise ContextOverflowError(
                            f"Input tokens exceed context window for model {model}. "
                            f"Request rejected to prevent API error."
                        )

                    # If the provider is 'qwen_code', set the custom provider to 'qwen'
                    # and strip the prefix from the model name for LiteLLM.
                    if provider == "qwen_code":
                        litellm_kwargs["custom_llm_provider"] = "qwen"
                        litellm_kwargs["model"] = model.split("/", 1)[1]

                    # If the provider is 'nvidia', set the custom provider for LiteLLM
                    # (litellm knows nvidia as 'nvidia_nim') and strip the prefix from the model name.
                    if provider == "nvidia":
                        litellm_kwargs["custom_llm_provider"] = "nvidia_nim"
                        litellm_kwargs["model"] = model.split("/", 1)[1]

                    for attempt in range(self.max_retries):
                        try:
                            lib_logger.info(
                                f"Attempting stream with credential {mask_credential(current_cred)} (Attempt {attempt + 1}/{self.max_retries})"
                            )

                            if pre_request_callback:
                                try:
                                    await pre_request_callback(request, litellm_kwargs)
                                except Exception as e:
                                    if self.abort_on_callback_error:
                                        raise PreRequestCallbackError(
                                            f"Pre-request callback failed: {e}"
                                        ) from e
                                    else:
                                        lib_logger.warning(
                                            f"Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: {e}"
                                        )

                            # Convert model parameters for custom providers right before LiteLLM call
                            final_kwargs = self.provider_config.convert_for_litellm(
                                **litellm_kwargs
                            )

                            response = await litellm.acompletion(
                                **final_kwargs,
                                logger_fn=self._litellm_logger_callback,
                            )

                            lib_logger.info(
                                f"Stream connection established for credential {mask_credential(current_cred)}. Processing response."
                            )

                            key_acquired = False
                            stream_generator = self._safe_streaming_wrapper(
                                response,
                                current_cred,
                                model,
                                request,
                                provider_plugin,
                            )

                            # Skip transaction logging wrapper when disabled (avoids extra async gen layer)
                            if transaction_logger:
                                async for chunk in self._transaction_logging_stream_wrapper(
                                    stream_generator, transaction_logger, kwargs
                                ):
                                    yield chunk
                            else:
                                async for chunk in stream_generator:
                                    yield chunk
                            return

                        except (
                            _StreamedException,
                            litellm.RateLimitError,
                            httpx.HTTPStatusError,
                            BadRequestError,
                            InvalidRequestError,
                        ) as e:
                            last_exception = e

                            # This is the final, robust handler for streamed errors.
                            error_payload = {}
                            cleaned_str = None
                            # The actual exception might be wrapped in our StreamedAPIError.
                            original_exc = getattr(e, "data", e)
                            classified_error = classify_error(
                                original_exc, provider=provider
                            )

                            # Reset LiteLLM client cache on auth errors (401/403)
                            self._reset_cache_on_auth_error(classified_error, original_exc, provider=provider, credential=current_cred)

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                await self._process_rate_limit(
                                    provider, current_cred, original_exc,
                                    str(original_exc) if original_exc else None, classified_error
                                )

                                # Add exponential backoff delay before retry
                                wait_time = 2**attempt * random.uniform(0.5, 1.5)
                                lib_logger.warning(
                                    f"Rate limit ({classified_error.error_type}) during litellm stream. "
                                    f"Waiting {wait_time:.2f}s before retry."
                                )
                                await asyncio.sleep(wait_time)

                                # Record failure and rotate to next credential
                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} {classified_error.error_type} "
                                    f"(HTTP {classified_error.status_code}). Rotating."
                                )
                                break  # Rotate to next credential

                            try:
                                # The full error JSON is in the string representation of the exception.
                                json_str_match = re.search(
                                    r"(\{.*\})", str(original_exc), re.DOTALL
                                )
                                if json_str_match:
                                    cleaned_str = codecs.decode(
                                        json_str_match.group(1), "unicode_escape"
                                    )
                                    error_payload = orjson.loads(cleaned_str)
                            except (orjson.JSONDecodeError, TypeError):
                                error_payload = {}

                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                                raw_response_text=cleaned_str,
                            )

                            error_details = error_payload.get("error", {})
                            error_status = error_details.get("status", "")
                            error_message_text = error_details.get(
                                "message", str(original_exc).split("\n")[0]
                            )

                            # Record in accumulator for client reporting
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message_text
                            )

                            if (
                                "quota" in error_message_text.lower()
                                or "resource_exhausted" in error_status.lower()
                            ):
                                consecutive_quota_failures += 1

                                quota_value = "N/A"
                                quota_id = "N/A"
                                if "details" in error_details and isinstance(
                                    error_details.get("details"), list
                                ):
                                    for detail in error_details["details"]:
                                        if isinstance(detail.get("violations"), list):
                                            for violation in detail["violations"]:
                                                if "quotaValue" in violation:
                                                    quota_value = violation[
                                                        "quotaValue"
                                                    ]
                                                if "quotaId" in violation:
                                                    quota_id = violation["quotaId"]
                                                if (
                                                    quota_value != "N/A"
                                                    and quota_id != "N/A"
                                                ):
                                                    break

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )

                                if consecutive_quota_failures >= 3:
                                    # Fatal: likely input data too large
                                    client_error_message = (
                                        f"Request failed after 3 consecutive quota errors (input may be too large). "
                                        f"Limit: {quota_value} (Quota ID: {quota_id})"
                                    )
                                    lib_logger.error(
                                        f"Fatal quota error for {mask_credential(current_cred)}. ID: {quota_id}, Limit: {quota_value}"
                                    )
                                    yield {'error': {'message': client_error_message, 'type': 'proxy_fatal_quota_error'}}
                                    yield STREAM_DONE
                                    return
                                else:
                                    lib_logger.warning(
                                        f"Cred {mask_credential(current_cred)} quota error ({consecutive_quota_failures}/3). Rotating."
                                    )
                                    break

                            else:
                                consecutive_quota_failures = 0

                                # For transient server errors (mid-stream), retry same key with backoff before rotating
                                if (
                                    should_retry_same_key(classified_error)
                                    and attempt < self.max_retries - 1
                                ):
                                    backoff = get_retry_backoff(
                                        classified_error, attempt, provider
                                    )
                                    remaining_budget = deadline - time.time()
                                    if backoff <= remaining_budget:
                                        lib_logger.warning(
                                            f"Mid-stream {classified_error.error_type} for {mask_credential(current_cred)} "
                                            f"(attempt {attempt + 1}/{self.max_retries}). Retrying same key in {backoff:.2f}s."
                                        )
                                        await asyncio.sleep(backoff)
                                        await self._get_http_client_async(
                                            streaming=True
                                        )
                                        continue  # Retry same key instead of rotating

                                lib_logger.warning(
                                    f"Cred {mask_credential(current_cred)} {classified_error.error_type}. Rotating."
                                )

                                # Only rate_limit errors can indicate IP-level throttling;
                                # quota_exceeded / invalid_request / auth errors are per-credential
                                if classified_error.error_type == "rate_limit":
                                    await self._process_rate_limit(
                                        provider, current_cred, original_exc,
                                        str(original_exc) if original_exc else None, classified_error
                                    )

                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                                break

                        except (
                            APIConnectionError,
                            litellm.InternalServerError,
                            litellm.ServiceUnavailableError,
                            RuntimeError,  # "Cannot send a request, as the client has been closed"
                        ) as e:
                            consecutive_quota_failures = 0
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message_text = str(e).split("\n")[0]

                            # Record error in accumulator (server errors are transient, not abnormal)
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message_text
                            )

                            # Provider-level error: don't increment consecutive failures
                            await self.usage_manager.record_failure(
                                current_cred,
                                model,
                                classified_error,
                                increment_consecutive_failures=False,
                            )

                            if attempt >= self.max_retries - 1:
                                lib_logger.warning(
                                    f"Credential {mask_credential(current_cred)} failed after max retries for model {model} due to a server error. Rotating key silently."
                                )
                                # [MODIFIED] Do not yield to the client here.
                                break

                            # Reset LiteLLM internal HTTP client cache on connection errors
                            # This fixes "Cannot send a request, as the client has been closed"
                            if isinstance(
                                e, RuntimeError
                            ) and "client has been closed" in str(e):
                                self._reset_litellm_client_cache()

                            if not await self._sleep_within_budget(
                                attempt, deadline, classified_error
                            ):
                                lib_logger.warning(
                                    "Required retry wait exceeds remaining budget. Rotating key early."
                                )
                                break

                            lib_logger.warning(
                                f"Credential {mask_credential(current_cred)} encountered a server error for model {model}. Reason: '{error_message_text}'. Retrying within remaining budget."
                            )

                            # CRITICAL: Ensure HTTP client is usable before retry
                            # Connection errors can leave the client in a closed state
                            await self._get_http_client_async(streaming=True)
                            continue

                        except Exception as e:
                            consecutive_quota_failures = 0
                            last_exception = e
                            log_failure(
                                api_key=current_cred,
                                model=model,
                                attempt=attempt + 1,
                                error=e,
                                request_headers=self._build_request_headers(request),
                            )
                            classified_error = classify_error(e, provider=provider)
                            error_message_text = str(e).split("\n")[0]

                            # Record error in accumulator
                            error_accumulator.record_error(
                                current_cred, classified_error, error_message_text
                            )

                            lib_logger.warning(
                                f"Credential {mask_credential(current_cred)} failed with {classified_error.error_type} (Status: {classified_error.status_code}). Error: {error_message_text}."
                            )

                            # Only rate_limit errors can indicate IP-level throttling;
                            # quota_exceeded / invalid_request / auth errors are per-credential
                            if classified_error.error_type == "rate_limit":
                                await self._process_rate_limit(
                                    provider, current_cred, e, str(e) if e else None, classified_error
                                )

                            # Check if this error should trigger rotation
                            if not should_rotate_on_error(classified_error):
                                lib_logger.error(
                                    f"Non-recoverable error ({classified_error.error_type}). Failing request."
                                )
                                raise last_exception

                            # Record failure and rotate to next key
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            lib_logger.info(
                                f"Rotating to next key after {classified_error.error_type} error."
                            )
                            break

                finally:
                    if key_acquired and current_cred:
                        await self.usage_manager.release_key(current_cred, model)

            # Build detailed error response using error accumulator
            error_accumulator.timeout_occurred = time.time() >= deadline

            if error_accumulator.has_errors():
                # Log concise summary for server logs
                lib_logger.error(error_accumulator.build_log_message())

                # Build structured error response for client
                error_response = error_accumulator.build_client_error_response()
                error_data = error_response
            else:
                # Fallback if no errors were recorded (shouldn't happen)
                final_error_message = (
                    "Request failed: No available API keys after rotation or timeout."
                )
                if last_exception:
                    final_error_message = (
                        f"Request failed. Last error: {str(last_exception)}"
                    )
                error_data = {
                    "error": {"message": final_error_message, "type": "proxy_error"}
                }
                lib_logger.error(final_error_message)

            yield error_data
            yield STREAM_DONE

        except NoAvailableKeysError as e:
            lib_logger.error(
                f"A streaming request failed because no keys were available within the time budget: {e}"
            )
            error_data = {"error": {"message": str(e), "type": "proxy_busy"}}
            yield error_data
            yield STREAM_DONE
        except Exception as e:
            # This will now only catch fatal errors that should be raised, like invalid requests.
            lib_logger.error(
                f"An unhandled exception occurred in streaming retry logic: {e}",
                exc_info=True,
            )
            error_data = {
                "error": {
                    "message": f"An unexpected error occurred: {str(e)}",
                    "type": "proxy_internal_error",
                }
            }
            yield error_data
            yield STREAM_DONE

    async def _forced_streaming_acompletion(
        self,
        request: Optional[Any] = None,
        pre_request_callback: Optional[callable] = None,
        **kwargs,
    ) -> Any:
        """
        Execute a streaming request internally but return a non-streaming ModelResponse.

        Used when a provider requires stream=true (e.g., Fireworks with max_tokens > 4096)
        but the client requested a non-streaming response. This method streams from the
        provider, collects all chunks, and assembles them into a single ModelResponse
        identical to what litellm.acompletion(stream=False) would return.
        """
        model = kwargs.get("model", "")
        # Get the streaming generator from the normal streaming path
        stream_generator = self._streaming_acompletion_with_retry(
            request=request,
            pre_request_callback=pre_request_callback,
            **kwargs,
        )

        # Collect dict chunks and assemble into a non-streaming response
        # This mirrors the aggregation logic in main.py's streaming_response_wrapper
        _content_parts: list = []
        aggregated_tool_calls: dict = {}
        final_message: dict = {"role": "assistant"}
        usage_data = None
        finish_reason = None
        model_id = None
        created_ts = None
        response_id = None

        async for chunk in stream_generator:
            # STREAM_DONE sentinel: stream is complete
            if chunk is STREAM_DONE:
                break

            if not isinstance(chunk, dict):
                continue

            # Detect error payloads from the streaming retry layer
            if "error" in chunk and "choices" not in chunk:
                error_info = chunk["error"]
                error_msg = error_info.get("message", str(error_info)) if isinstance(error_info, dict) else str(error_info)
                raise litellm.InternalServerError(error_msg)

            # Capture metadata from first chunk
            if response_id is None and chunk.get("id"):
                response_id = chunk["id"]
            if created_ts is None and chunk.get("created"):
                created_ts = chunk["created"]
            if model_id is None and chunk.get("model"):
                model_id = chunk["model"]

            # Aggregate choices
            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                for key, value in delta.items():
                    if value is None:
                        continue
                    if key == "content":
                        if value:
                            _content_parts.append(value)
                    elif key == "tool_calls":
                        for tc_chunk in value:
                            index = tc_chunk["index"]
                            if index not in aggregated_tool_calls:
                                aggregated_tool_calls[index] = {
                                    "type": "function",
                                    "function": {
                                        "name_parts": [],
                                        "args_parts": [],
                                    },
                                }
                            tc = aggregated_tool_calls[index]
                            if tc_chunk.get("id"):
                                tc["id"] = tc_chunk["id"]
                            if "function" in tc_chunk:
                                fn = tc_chunk["function"]
                                if fn.get("name") is not None:
                                    tc["function"]["name_parts"].append(fn["name"])
                                if fn.get("arguments") is not None:
                                    tc["function"]["args_parts"].append(fn["arguments"])
                    elif key == "function_call":
                        if "function_call" not in final_message:
                            final_message["function_call"] = {
                                "_name_parts": [],
                                "_args_parts": [],
                            }
                        if value.get("name") is not None:
                            final_message["function_call"]["_name_parts"].append(value["name"])
                        if value.get("arguments") is not None:
                            final_message["function_call"]["_args_parts"].append(value["arguments"])

                # Capture finish_reason
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

            # Capture usage data
            if "usage" in chunk and isinstance(chunk["usage"], dict):
                usage_data = chunk["usage"]

        # Assemble the final ModelResponse
        content = "".join(_content_parts) if _content_parts else None

        # Assemble tool calls
        tool_calls_list = None
        if aggregated_tool_calls:
            tool_calls_list = []
            for index in sorted(aggregated_tool_calls.keys()):
                tc = aggregated_tool_calls[index]
                tc_entry = {
                    "index": index,
                    "type": tc.get("type", "function"),
                    "id": tc.get("id", f"call_{index}"),
                    "function": {
                        "name": "".join(tc["function"]["name_parts"]),
                        "arguments": "".join(tc["function"]["args_parts"]),
                    },
                }
                tool_calls_list.append(tc_entry)

        # Assemble function_call (legacy)
        if "function_call" in final_message:
            fc = final_message["function_call"]
            final_message["function_call"] = {
                "name": "".join(fc.get("_name_parts", [])),
                "arguments": "".join(fc.get("_args_parts", [])),
            }

        # Build final message
        if content is not None:
            final_message["content"] = content
        if tool_calls_list:
            final_message["tool_calls"] = tool_calls_list
        if finish_reason:
            final_message["finish_reason"] = finish_reason
        elif tool_calls_list:
            final_message["finish_reason"] = "tool_calls"
        elif content:
            final_message["finish_reason"] = "stop"

        # Build ModelResponse object
        model_response = litellm.ModelResponse(
            id=response_id or f"chatcmpl-{id(chunk)}",
            created=created_ts or int(time.time()),
            model=model_id or model,
            choices=[
                {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": final_message.get("finish_reason", "stop"),
                }
            ],
            usage=usage_data or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

        lib_logger.debug(
            f"Forced streaming completion assembled: model={model_id}, "
            f"finish_reason={finish_reason}, usage={usage_data}"
        )

        return model_response
