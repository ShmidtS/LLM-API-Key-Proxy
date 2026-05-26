# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Streaming retry orchestrator state-machine."""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable, Optional

import httpx
import litellm  # type: ignore[import-untyped]
from litellm.exceptions import (
    APIConnectionError,
    APIError as LiteLLMAPIError,
    BadRequestError,
    InternalServerError,
    InvalidRequestError,
    RateLimitError,
    ServiceUnavailableError,
)  # type: ignore[import-untyped]

from ..retry_base import HalfOpenSlot
from ...error_handler import (
    classify_error,
    get_retry_backoff,
    should_retry_same_key,
    should_rotate_on_error,
)
from ...error_types import NoAvailableKeysError, mask_credential
from ...failure_logger import log_failure
from ...utils.json_utils import STREAM_DONE

lib_logger = logging.getLogger("rotator_library")

async def _streaming_acompletion_with_retry(
    self,
    request: Optional[Any],
    pre_request_callback: Optional[Callable] = None,
    **kwargs,
) -> AsyncGenerator[Any, None]:
    """A dedicated generator for retrying streaming completions with full request preparation and per-key retries."""
    from .._streaming import _StreamedException

    # --- Phase 5: respond tool flag ---
    # Injected by core.py::acompletion() into kwargs. Pop before retry context
    # so it doesn't leak into litellm or provider kwargs.
    _respond_tool_injected = kwargs.pop("_respond_tool_injected", False)

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

    # Cache request headers once to avoid repeated dict() conversion in error handlers
    _cached_request_headers = dict(request.headers) if request else {}
    max_retries = self.max_retries
    global_semaphore = self._global_semaphore
    provider_params = self.litellm_provider_params.get(provider)
    has_provider_params = provider in self.litellm_provider_params

    shared_litellm_kwargs = kwargs.copy()
    shared_litellm_kwargs["num_retries"] = 0

    # [FIX] Remove client-provided headers/api_key that could override provider credentials
    self._strip_client_headers(shared_litellm_kwargs)

    if "reasoning_effort" in kwargs:
        shared_litellm_kwargs["reasoning_effort"] = kwargs["reasoning_effort"]

    # [NEW] Merge provider-specific params
    if has_provider_params:
        shared_litellm_kwargs["litellm_params"] = {
            **provider_params,
            **shared_litellm_kwargs.get("litellm_params", {}),
        }

    consecutive_quota_failures: dict[str, int] = {}
    account_billing_error_count = 0
    total_api_attempts = 0
    remaining_budget = deadline - time.monotonic()

    try:
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
                    suppress_cb_logging=True,
                )
                if sel.loop_action == "break":
                    lib_logger.warning(
                        "All credentials for provider %s have been tried. No more credentials to rotate to.",
                        provider,
                    )
                    break
                if sel.loop_action == "continue":
                    continue
                current_cred = sel.current_cred
                _cb_slot_held = sel.cb_slot_held
                key_acquired = True

                litellm_kwargs = shared_litellm_kwargs.copy()
                if has_provider_params:
                    litellm_kwargs["litellm_params"] = litellm_kwargs["litellm_params"].copy()

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
                if provider_plugin and provider_plugin.has_custom_logic:
                    lib_logger.debug(
                        "Provider '%s' has custom logic. Delegating call.",
                        provider,
                    )
                    litellm_kwargs["credential_identifier"] = current_cred
                    litellm_kwargs["transaction_context"] = (
                        transaction_logger.get_context()
                        if transaction_logger
                        else None
                    )

                    for attempt in range(max_retries):
                        total_api_attempts += 1
                        self._check_max_attempts(total_api_attempts, last_exception)
                        try:
                            await self._execute_single_attempt(
                                attempt, max_retries, current_cred, "stream",
                                pre_request_callback, request, litellm_kwargs, provider,
                            )

                            http_client = await self._get_http_client_async(
                                streaming=True
                            )
                            async with global_semaphore:
                                response = await provider_plugin.acompletion(
                                    http_client, **litellm_kwargs
                                )

                                if lib_logger.isEnabledFor(logging.INFO):
                                    lib_logger.info(
                                        "Stream connection established for credential %s. Processing response.",
                                        mask_credential(current_cred),
                                    )

                                stream_generator = self._safe_streaming_wrapper(
                                    response,
                                    current_cred,
                                    model,
                                    request,
                                    provider_plugin,
                                    respond_tool_active=_respond_tool_injected,
                                )

                                # Release the key when the stream consumer finishes iterating.
                                # The outer finally only fires on generator finalization;
                                # this try/finally ensures the credential is released
                                # promptly whether the stream completes normally, errors
                                # out, or the consumer stops early.
                                try:
                                    if transaction_logger:
                                        async for (
                                            chunk
                                        ) in self._transaction_logging_stream_wrapper(
                                            stream_generator, transaction_logger, kwargs
                                        ):
                                            yield chunk
                                    else:
                                        async for chunk in stream_generator:
                                            yield chunk
                                finally:
                                    await self.usage_manager.release_key(
                                        current_cred, model
                                    )
                                    key_acquired = False  # prevent outer finally from double-releasing
                            # Streaming completed successfully (semaphore released) — mirror non-streaming bookkeeping
                            await self._record_attempt_metrics(
                                current_cred, model, provider, streaming=True,
                            )
                            _cb_slot_held = False  # record_success already released the slot
                            return

                        except (
                            _StreamedException,
                            RateLimitError,
                            httpx.HTTPStatusError,
                            BadRequestError,
                            InvalidRequestError,
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
                            RuntimeError,
                            httpx.ReadTimeout,
                            httpx.PoolTimeout,
                            httpx.RemoteProtocolError,
                            httpx.ConnectError,
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

                        except asyncio.CancelledError:
                            raise
                        except (TypeError, AttributeError, KeyError):
                            raise
                        except Exception as e:
                            last_exception = e
                            dec = await self._handle_retry_error(
                                e, provider, current_cred, model,
                                attempt, deadline,
                                _cached_request_headers,
                                error_accumulator,
                            )
                            if dec.action == "fail":
                                async with HalfOpenSlot(self._resilience, provider):
                                    raise last_exception
                            async with HalfOpenSlot(self._resilience, provider):
                                break

                    # If the inner loop breaks, it means the key failed and we need to rotate.
                    # Continue to the next iteration of the outer while loop to pick a new key.
                    continue

                else:  # This is the standard API Key / litellm-handled provider logic
                    is_oauth = provider in self.oauth_providers
                    if not is_oauth:  # API Key
                        litellm_kwargs["api_key"] = current_cred

                # [FIX] Remove problematic headers and add correct provider headers
                # This ensures that authorization/x-api-key from client requests
                # are replaced with the correct values from configuration
                await self._apply_provider_headers(
                    litellm_kwargs, provider, current_cred
                )

                self._apply_common_provider_overrides(
                    litellm_kwargs, model, provider, provider_plugin,
                    log_label="streaming path",
                    api_call=litellm.acompletion,
                )

                for attempt in range(max_retries):
                    total_api_attempts += 1
                    self._check_max_attempts(total_api_attempts, last_exception)
                    # Ensure HTTP client pool is healthy at the start of each attempt
                    # This replaces redundant per-error-handler calls below
                    await self._get_http_client_async(streaming=True)
                    try:
                        await self._execute_single_attempt(
                            attempt, max_retries, current_cred, "stream",
                            pre_request_callback, request, litellm_kwargs, provider,
                        )

                        if "_native_provider" in litellm_kwargs:
                            final_kwargs = litellm_kwargs
                        else:
                            final_kwargs = self.provider_config.convert_for_litellm(
                                **litellm_kwargs
                            )

                        async with global_semaphore:
                            response = await litellm.acompletion(
                                **final_kwargs,
                                logger_fn=self._litellm_logger_callback,
                            )

                            if response is None:
                                lib_logger.error(
                                    "litellm.acompletion returned None for credential %s.",
                                    mask_credential(current_cred),
                                )
                                last_exception = RuntimeError(
                                    "litellm.acompletion returned None — provider did not return a stream"
                                )
                                continue

                            if lib_logger.isEnabledFor(logging.INFO):
                                lib_logger.info(
                                    "Stream connection established for credential %s. Processing response.",
                                    mask_credential(current_cred),
                                )

                            stream_generator = self._safe_streaming_wrapper(
                                response,
                                current_cred,
                                model,
                                request,
                                provider_plugin,
                                respond_tool_active=_respond_tool_injected,
                            )

                            # Release the key when the stream consumer finishes iterating.
                            try:
                                if transaction_logger:
                                    async for (
                                        chunk
                                    ) in self._transaction_logging_stream_wrapper(
                                        stream_generator, transaction_logger, kwargs
                                    ):
                                        yield chunk
                                else:
                                    async for chunk in stream_generator:
                                        yield chunk
                            finally:
                                await self.usage_manager.release_key(
                                    current_cred, model
                                )
                                key_acquired = (
                                    False  # prevent outer finally from double-releasing
                                )
                        # Streaming completed successfully (semaphore released) — mirror non-streaming bookkeeping
                        await self._record_attempt_metrics(
                            current_cred, model, provider, streaming=True,
                        )
                        _cb_slot_held = False  # record_success already released the slot
                        return

                    except (
                        _StreamedException,
                        RateLimitError,
                        httpx.HTTPStatusError,
                        BadRequestError,
                        InvalidRequestError,
                        LiteLLMAPIError,
                    ) as e:
                        last_exception = e

                        # This is the final, robust handler for streamed errors.
                        original_exc = getattr(e, "data", e)
                        classified_error = classify_error(
                            original_exc, provider=provider
                        )

                        # Reset LiteLLM client cache on auth errors (401/403)
                        self._reset_cache_on_auth_error(
                            classified_error,
                            original_exc,
                            provider=provider,
                            credential=current_cred,
                        )

                        # Track account billing errors for logging, but don't fast-fail:
                        # each credential may belong to a different account, so hitting
                        # N bad keys doesn't mean ALL keys are bad. Per-credential
                        # cooldown already prevents re-selecting the same bad key.
                        if getattr(classified_error, "reason", None) == "account_billing_issue":
                            account_billing_error_count += 1
                            lib_logger.warning(
                                "Account billing error #%d for '%s' on cred %s; rotating to next key.",
                                account_billing_error_count, provider,
                                mask_credential(current_cred),
                            )

                        # Check if this error should trigger rotation
                        if not should_rotate_on_error(classified_error):
                            # Non-rotatable errors (invalid_request, context_window_exceeded,
                            # pre_request_callback_error, ip_rate_limit) — fail immediately.
                            # Rotating keys won't fix a malformed request.
                            if classified_error.error_type in {"rate_limit", "ip_rate_limit"}:
                                await self._process_rate_limit(
                                    provider,
                                    current_cred,
                                    original_exc,
                                    str(original_exc) if original_exc else None,
                                    classified_error,
                                )
                            lib_logger.error(
                                "Fatal %s error for %s (HTTP %s). Failing immediately.",
                                classified_error.error_type,
                                mask_credential(current_cred),
                                classified_error.status_code,
                            )
                            lib_logger.error(
                                "Fatal error payload detail: %s",
                                str(last_exception),
                            )
                            _cb_slot_held = False
                            async with HalfOpenSlot(self._resilience, provider):
                                yield {
                                    "error": {
                                        "message": str(last_exception),
                                        "type": classified_error.error_type,
                                    }
                                }
                                yield STREAM_DONE
                                return

                        # Parse error payload from streaming exception
                        error_payload, cleaned_str = self._parse_streaming_error_payload(original_exc)

                        log_failure(
                            api_key=current_cred,
                            model=model,
                            attempt=attempt + 1,
                            error=e,
                            request_headers=_cached_request_headers,
                            raw_response_text=cleaned_str or "",
                        )

                        error_details = error_payload.get("error", {})
                        if not isinstance(error_details, dict):
                            error_details = {"message": str(error_details)}
                        error_status = error_details.get("status", "")
                        error_message_text = error_details.get(
                            "message", str(original_exc).partition("\n")[0]
                        )

                        # Record in accumulator for client reporting
                        error_accumulator.record_error(
                            current_cred, classified_error, error_message_text,
                            attempt_number=attempt + 1,
                        )

                        if (
                            "quota" in error_message_text.lower()
                            or "resource_exhausted" in error_status.lower()
                        ):
                            consecutive_quota_failures[current_cred] = (
                                consecutive_quota_failures.get(current_cred, 0) + 1
                            )

                            quota_value, quota_id = self._extract_quota_info(error_details)

                            await self.usage_manager.record_failure(
                                current_cred, model, classified_error
                            )

                            if consecutive_quota_failures.get(current_cred, 0) >= 3:
                                # Fatal: likely input data too large
                                client_error_message = (
                                    "Request failed after 3 consecutive quota errors (input may be too large). "
                                    f"Limit: {quota_value} (Quota ID: {quota_id})"
                                )
                                lib_logger.error(
                                    "Fatal quota error for %s. ID: %s, Limit: %s",
                                    mask_credential(current_cred), quota_id, quota_value,
                                )
                                yield {
                                    "error": {
                                        "message": client_error_message,
                                        "type": "proxy_fatal_quota_error",
                                    }
                                }
                                yield STREAM_DONE
                                return
                            else:
                                lib_logger.warning(
                                    "Cred %s quota error (%s/3). Rotating.",
                                    mask_credential(current_cred),
                                    consecutive_quota_failures.get(current_cred, 0),
                                )
                                async with HalfOpenSlot(self._resilience, provider):
                                    break

                        else:
                            consecutive_quota_failures.pop(current_cred, None)

                            # For transient server errors (mid-stream), retry same key with backoff before rotating
                            if (
                                should_retry_same_key(classified_error)
                                and attempt < max_retries - 1
                            ):
                                backoff = get_retry_backoff(
                                    classified_error, attempt, provider
                                )
                                if backoff <= remaining_budget:
                                    lib_logger.warning(
                                        "Mid-stream %s for %s "
                                        "(attempt %s/%s). Retrying same key in %2.2fs.",
                                        classified_error.error_type,
                                        mask_credential(current_cred),
                                        attempt + 1, max_retries, backoff,
                                    )
                                    await asyncio.sleep(backoff)
                                    # HTTP client pool refreshed at start of next attempt
                                    continue  # Retry same key instead of rotating

                            lib_logger.warning(
                                "Cred %s %s. Rotating.",
                                mask_credential(current_cred),
                                classified_error.error_type,
                            )

                            action = await self._apply_error_classifications(
                                provider, current_cred, model, original_exc, classified_error,
                            )
                            if action != "force_rotate":
                                await self.usage_manager.record_failure(
                                    current_cred, model, classified_error
                                )
                            async with HalfOpenSlot(self._resilience, provider):
                                break

                    except (
                        APIConnectionError,
                        InternalServerError,
                        ServiceUnavailableError,
                        RuntimeError,
                        httpx.ReadTimeout,
                        httpx.PoolTimeout,
                        httpx.RemoteProtocolError,
                        httpx.ConnectError,
                    ) as e:
                        consecutive_quota_failures.pop(current_cred, None)
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

                    except asyncio.CancelledError:
                        raise
                    except (TypeError, AttributeError, KeyError):
                        raise
                    except Exception as e:
                        consecutive_quota_failures.pop(current_cred, None)
                        last_exception = e
                        dec = await self._handle_retry_error(
                            e, provider, current_cred, model,
                            attempt, deadline,
                            _cached_request_headers,
                            error_accumulator,
                            request=request,
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

        # Build detailed error response using error accumulator
        error_accumulator.timeout_occurred = time.monotonic() >= deadline

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
                "error": {"message": final_error_message, "type": "proxy_busy"}
            }
            lib_logger.error(final_error_message)

        yield error_data
        yield STREAM_DONE

    except asyncio.CancelledError:
        raise
    except NoAvailableKeysError as e:
        lib_logger.error(
            "A streaming request failed because no keys were available within the time budget: %s",
            e,
        )
        error_data = {"error": {"message": str(e), "type": "proxy_busy"}}
        yield error_data
        yield STREAM_DONE
    except (TypeError, AttributeError, KeyError):
        raise
    except Exception as e:
        # This will now only catch fatal errors that should be raised, like invalid requests.
        lib_logger.error(
            "An unhandled exception occurred in streaming retry logic: %s",
            e,
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
