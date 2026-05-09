# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Non-streaming retry execution."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

import httpx
from litellm.exceptions import APIConnectionError, InternalServerError, RateLimitError, ServiceUnavailableError  # type: ignore[import-untyped]

from ..retry_base import HalfOpenSlot
from ...error_types import (
    ClassifiedError,
    GarbageResponseError,
    NoAvailableKeysError,
    mask_credential,
)

lib_logger = logging.getLogger("rotator_library")


class NonStreamingRetryMixin:
    """Non-streaming retry execution."""

    async def _execute_with_retry(
        self,
        api_call: Callable,
        request: Optional[Any],
        pre_request_callback: Optional[Callable] = None,
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
        max_retries = self.max_retries
        global_semaphore = self._global_semaphore
        _cached_request_headers = self._build_request_headers(request)
        account_billing_error_count = 0
        total_api_attempts = 0
        remaining_budget = deadline - time.monotonic()

        while len(tried_creds) < len(credentials_for_provider) and remaining_budget > 0:
            now = time.monotonic()
            remaining_budget = deadline - now
            current_cred = None
            key_acquired = False
            _cb_slot_held = False
            try:
                sel = await self._select_next_key(
                    credentials_for_provider, tried_creds,
                    model, provider, deadline,
                    credential_priorities, credential_tier_names,
                )
                if sel.loop_action == "break":
                    break
                if sel.loop_action == "continue":
                    continue
                current_cred = sel.current_cred
                _cb_slot_held = sel.cb_slot_held
                key_acquired = True

                litellm_kwargs = await self._prepare_request_kwargs(
                    kwargs, provider, current_cred, model
                )

                if provider_plugin and provider_plugin.has_custom_logic:
                    lib_logger.debug(
                        "Provider '%s' has custom logic. Delegating call.",
                        provider,
                    )
                    litellm_kwargs["credential_identifier"] = current_cred
                    litellm_kwargs["transaction_context"] = (
                        transaction_logger.get_context() if transaction_logger else None
                    )

                    # Retry loop for custom providers - mirrors streaming path error handling
                    for attempt in range(max_retries):
                        total_api_attempts += 1
                        self._check_max_attempts(total_api_attempts, last_exception)
                        try:
                            await self._execute_single_attempt(
                                attempt, max_retries, current_cred, "call",
                                pre_request_callback, request, litellm_kwargs, provider,
                            )

                            http_client = await self._get_http_client_async(
                                streaming=False
                            )
                            async with global_semaphore:
                                response = await provider_plugin.acompletion(
                                    http_client, **litellm_kwargs
                                )

                            await self._record_attempt_metrics(
                                current_cred, model, provider,
                                response=response, transaction_logger=transaction_logger,
                            )
                            key_acquired = False
                            _cb_slot_held = False  # record_success already released the slot
                            return response

                        except (
                            RateLimitError,
                            httpx.HTTPStatusError,
                        ) as e:
                            last_exception = e
                            dec = await self._handle_rate_limit_error(
                                e, provider, current_cred, model,
                                error_accumulator,
                                _cached_request_headers,
                                attempt=attempt,
                            )
                            if dec.action == "fail":
                                async with HalfOpenSlot(self._resilience, provider):
                                    raise last_exception
                            async with HalfOpenSlot(self._resilience, provider):
                                break

                        except (
                            APIConnectionError,
                            InternalServerError,
                            ServiceUnavailableError,
                            RuntimeError,  # "Cannot send a request, as the client has been closed"
                        ) as e:
                            last_exception = e
                            dec = await self._handle_retry_error(
                                e, provider, current_cred, model,
                                attempt, deadline,
                                _cached_request_headers,
                                error_accumulator,
                            )
                            if dec.action == "retry_same_key":
                                await asyncio.sleep(dec.wait_time)
                                continue
                            async with HalfOpenSlot(self._resilience, provider):
                                break

                        except GarbageResponseError as gre:
                            # Garbage response - rotate to next credential immediately
                            # _cb_slot_held may be False if record_success already released it
                            await self.usage_manager.record_failure(
                                current_cred, model, ClassifiedError(
                                    error_type="garbage_response",
                                    original_exception=last_exception,
                                    status_code=503,
                                    reason=gre.reason,
                                )
                            )
                            if _cb_slot_held:
                                async with HalfOpenSlot(self._resilience, provider):
                                    break
                            else:
                                break

                    # If the inner loop breaks, it means the key failed and we need to rotate.
                    # Continue to the next iteration of the outer while loop to pick a new key.
                    continue

                else:  # This is the standard API Key / litellm-handled provider logic
                    is_oauth = provider in self.oauth_providers
                    if is_oauth:  # Standard OAuth provider (not custom)
                        # ... (logic to set headers) ...
                        lib_logger.debug("OAuth header handling handled by _apply_provider_headers")
                    else:  # API Key
                        litellm_kwargs["api_key"] = current_cred

                    # _prepare_request_kwargs already called _apply_provider_headers above
                    self._apply_common_provider_overrides(
                        litellm_kwargs, model, provider, provider_plugin,
                        api_call=api_call,
                    )

                    for attempt in range(max_retries):
                        total_api_attempts += 1
                        self._check_max_attempts(total_api_attempts, last_exception)
                        # Ensure HTTP client pool is healthy at the start of each attempt
                        # This replaces redundant per-error-handler calls below
                        await self._get_http_client_async(streaming=False)
                        try:
                            await self._execute_single_attempt(
                                attempt, max_retries, current_cred, "call",
                                pre_request_callback, request, litellm_kwargs, provider,
                            )

                            if "_native_provider" in litellm_kwargs:
                                final_kwargs = litellm_kwargs
                            else:
                                final_kwargs = self.provider_config.convert_for_litellm(
                                    **litellm_kwargs
                                )

                            async with global_semaphore:
                                response = await api_call(
                                    **final_kwargs,
                                    logger_fn=self._litellm_logger_callback,
                                )

                            await self._record_attempt_metrics(
                                current_cred, model, provider,
                                response=response, transaction_logger=transaction_logger,
                            )
                            key_acquired = False
                            _cb_slot_held = False  # record_success already released the slot
                            return response

                        except RateLimitError as e:
                            last_exception = e
                            dec = await self._handle_rate_limit_error(
                                e, provider, current_cred, model,
                                error_accumulator,
                                _cached_request_headers,
                                attempt=attempt,
                            )
                            if dec.action == "fail":
                                async with HalfOpenSlot(self._resilience, provider):
                                    raise last_exception
                            async with HalfOpenSlot(self._resilience, provider):
                                break

                        except (
                            APIConnectionError,
                            InternalServerError,
                            ServiceUnavailableError,
                            RuntimeError,  # "Cannot send a request, as the client has been closed"
                        ) as e:
                            last_exception = e
                            dec = await self._handle_retry_error(
                                e, provider, current_cred, model,
                                attempt, deadline,
                                _cached_request_headers,
                                error_accumulator,
                                reset_quota=True,
                            )
                            if dec.action == "retry_same_key":
                                await asyncio.sleep(dec.wait_time)
                                continue
                            async with HalfOpenSlot(self._resilience, provider):
                                break

                        except httpx.HTTPStatusError as e:
                            last_exception = e
                            dec = await self._handle_rate_limit_error(
                                e, provider, current_cred, model,
                                error_accumulator,
                                _cached_request_headers,
                                attempt=attempt,
                                deadline=deadline,
                            )
                            if dec.action == "fail":
                                async with HalfOpenSlot(self._resilience, provider):
                                    raise last_exception
                            if dec.action == "retry_same_key":
                                await asyncio.sleep(dec.wait_time)
                                continue
                            await self.usage_manager.record_failure(
                                current_cred, model, dec.classified_error
                            )
                            async with HalfOpenSlot(self._resilience, provider):
                                break

                        except asyncio.CancelledError:
                            raise
                        except GarbageResponseError as gre:
                            # Garbage response from validate_response_quality - rotate immediately
                            # _cb_slot_held may be False if record_success already released it
                            classified_error = ClassifiedError(
                                error_type="garbage_response",
                                status_code=503,
                                reason=gre.reason,
                            )
                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )
                            error_accumulator.record_error(
                                current_cred, classified_error,
                                "Garbage response detected, rotating"
                            )
                            if _cb_slot_held:
                                async with HalfOpenSlot(self._resilience, provider):
                                    break
                            else:
                                break
                        except (TypeError, AttributeError, KeyError):
                            raise
                        except Exception as e:
                            lib_logger.error(f"Unexpected error in retry loop: {type(e).__name__}: {e}")
                            last_exception = e
                            dec = await self._handle_retry_error(
                                e, provider, current_cred, model,
                                attempt, deadline,
                                _cached_request_headers,
                                error_accumulator,
                                request=request,
                            )
                            # Track account billing errors for logging, but don't fast-fail:
                            # each credential may belong to a different account, so hitting
                            # N bad keys doesn't mean ALL keys are bad. Per-credential
                            # cooldown already prevents re-selecting the same bad key.
                            if getattr(dec.classified_error, "reason", None) == "account_billing_issue":
                                account_billing_error_count += 1
                                lib_logger.warning(
                                    "Account billing error #%d for '%s' on cred %s; rotating to next key.",
                                    account_billing_error_count, provider,
                                    mask_credential(current_cred),
                                )
                            if dec.action == "fail":
                                _cb_slot_held = False
                                async with HalfOpenSlot(self._resilience, provider):
                                    raise last_exception
                            _cb_slot_held = False
                            async with HalfOpenSlot(self._resilience, provider):
                                break
            finally:
                await self._release_cred(current_cred if current_cred is not None else "", model, key_acquired, provider, _cb_slot_held)

        # Check if we exhausted all credentials or timed out
        error_accumulator.timeout_occurred = time.monotonic() >= deadline

        if error_accumulator.has_errors():
            # Log concise summary for server logs
            lib_logger.error(error_accumulator.build_log_message())

            # Return the structured error response for the client
            return error_accumulator.build_client_error_response()

        lib_logger.warning(
            "Unexpected state: request failed with no recorded errors. "
            "This may indicate a logic error in error tracking."
        )
        raise NoAvailableKeysError(
            f"All credentials exhausted for {provider}/{model} with no recorded errors"
        )

    async def _rate_limited_execute(self, api_call, request, pre_request_callback=None, **kwargs):
        """Backpressure-gated entry point for non-streaming API calls.

        The semaphore is acquired per-attempt inside _execute_with_retry,
        not around the entire retry cycle, so backoff sleeps don't hold
        a semaphore slot.
        """
        return await self._execute_with_retry(
            api_call, request, pre_request_callback=pre_request_callback, **kwargs
        )
