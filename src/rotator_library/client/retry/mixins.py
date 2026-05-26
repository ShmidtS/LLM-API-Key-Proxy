# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Composed retry mixin."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Optional

import httpx
import litellm  # type: ignore[import-untyped]
from litellm.exceptions import APIConnectionError, InternalServerError, ServiceUnavailableError  # type: ignore[import-untyped]

from ..retry_base import HalfOpenSlot, _ErrorDecision, RetryBaseMixin
from ...config.defaults import MAX_TOTAL_ATTEMPTS, RETRY_SAME_KEY_MAX_WAIT
from ...error_handler import (
    classify_error,
    should_retry_same_key,
    should_rotate_on_error,
    validate_response_quality,
)
from ...error_types import (
    ClassifiedError,
    ContextOverflowError,
    GarbageResponseError,
    NoAvailableKeysError,
    PreRequestCallbackError,
    mask_credential,
)
from ...failure_logger import log_failure
from ...request_sanitizer import sanitize_request_payload
from ...utils.http_retry import compute_backoff_with_jitter
from .context_builder import RetryContextBuilderMixin

lib_logger = logging.getLogger("rotator_library")

_COMPLETION_ONLY_PARAMS = (
    "temperature", "max_tokens", "max_output_tokens",
    "top_p", "frequency_penalty", "presence_penalty",
    "stream", "stream_options", "n",
    "reasoning_effort", "thinking",
)

from .non_streaming import NonStreamingRetryMixin
from .streaming import StreamingRetryMixin


class RetryCommonMixin:
    """Shared retry helpers used by streaming and non-streaming paths."""

    def _apply_common_provider_overrides(
        self,
        litellm_kwargs: dict,
        model: str,
        provider: str,
        provider_plugin: Any,
        log_label: str = "",
        api_call: Any = None,
    ):
        """Apply shared provider-specific overrides to litellm_kwargs.

        Handles safety settings, thinking parameter, gemma-3 system-role
        conversion, request payload sanitization, context-window rejection,
        and provider-specific model name / custom_llm_provider overrides.

        Args:
            litellm_kwargs: Mutable kwargs dict to modify in place.
            model: Resolved model string.
            provider: Provider name.
            provider_plugin: Provider plugin instance (may be None).
            log_label: Optional label for safety-settings warning messages
                       (e.g. "streaming path").
        """
        is_non_completion = api_call not in (
            litellm.acompletion, litellm.completion,
        )
        if provider_plugin:
            # Skip safety settings for non-completion endpoints (embeddings, TTS, etc.)
            if not is_non_completion:
                try:
                    self._apply_default_safety_settings(litellm_kwargs, provider)
                except asyncio.CancelledError:
                    raise
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    label = f" {log_label}" if log_label else ""
                    lib_logger.warning(
                        "Could not apply default safety settings for%s %s: %s; continuing.",
                        label, provider, type(exc).__name__,
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

        # Strip completion-only params from embedding/media requests
        if is_non_completion:
            for key in _COMPLETION_ONLY_PARAMS:
                litellm_kwargs.pop(key, None)
            # NVIDIA rejects encoding_format=None — default to "float"
            if provider == "nvidia":
                if litellm_kwargs.get("encoding_format") is None:
                    litellm_kwargs["encoding_format"] = "float"

        if provider == "gemini" and provider_plugin:
            provider_plugin.handle_thinking_parameter(litellm_kwargs, model)
        if provider == "nvidia" and provider_plugin:
            provider_plugin.handle_thinking_parameter(litellm_kwargs, model)

        if "gemma-3" in model and "messages" in litellm_kwargs:
            litellm_kwargs["messages"] = [
                (
                    {"role": "user", "content": m["content"]}
                    if m.get("role") == "system"
                    else m
                )
                for m in litellm_kwargs["messages"]
            ]

        sanitized_kwargs, should_reject = sanitize_request_payload(
            litellm_kwargs, model, registry=self._model_registry,
            auto_adjust_max_tokens=not is_non_completion,
        )
        # Update in-place so caller sees the changes
        if sanitized_kwargs is not litellm_kwargs:
            litellm_kwargs.clear()
            litellm_kwargs.update(sanitized_kwargs)

        if should_reject:
            raise ContextOverflowError(
                f"Input tokens exceed context window for model {model}. "
                "Request rejected to prevent API error."
            )

        # Re-strip completion-only params after sanitization (sanitize may re-add max_tokens)
        # and add drop_params so litellm strips its own internally-injected params
        if is_non_completion:
            for key in _COMPLETION_ONLY_PARAMS:
                litellm_kwargs.pop(key, None)
            if provider == "nvidia":
                if litellm_kwargs.get("encoding_format") is None:
                    litellm_kwargs["encoding_format"] = "float"
            litellm_kwargs.setdefault("drop_params", True)

        model_suffix = model.split("/", 1)[1] if provider in ("qwen_code", "nvidia", "fireworks") else None

        if provider == "qwen_code":
            litellm_kwargs["custom_llm_provider"] = "qwen"
            litellm_kwargs["model"] = model_suffix

        if provider == "nvidia":
            litellm_kwargs["custom_llm_provider"] = "nvidia_nim"
            litellm_kwargs["model"] = model_suffix

        if provider == "inception":
            litellm_kwargs["model"] = model.rsplit("/", 1)[1]

        if provider == "fireworks":
            litellm_kwargs["custom_llm_provider"] = "fireworks_ai"
            litellm_kwargs["model"] = f"fireworks_ai/{model_suffix}"

    async def _invoke_pre_request_callback(self, pre_request_callback, request, litellm_kwargs, provider: str):
        """Invoke pre_request_callback with abort_on_callback_error handling."""
        try:
            await pre_request_callback(request, litellm_kwargs)
        except asyncio.CancelledError:
            raise
        except (ValueError, TypeError, KeyError, RuntimeError) as e:
            if self.abort_on_callback_error:
                async with HalfOpenSlot(self._resilience, provider):
                    raise PreRequestCallbackError(
                        f"Pre-request callback failed: {e}"
                    ) from e
            else:
                lib_logger.warning(
                    "Pre-request callback failed but abort_on_callback_error is False. Proceeding with request. Error: %s",
                    e,
                )

    async def _classify_log_error(
        self, e, provider: str, current_cred: str, model: str,
        attempt: int, request_headers: dict,
    ):
        """Classify error, extract message, and log failure. Returns (ClassifiedError, str)."""
        classified = classify_error(e, provider=provider)
        error_message = str(e).partition("\n")[0]
        log_failure(
            api_key=current_cred, model=model,
            attempt=attempt, error=e,
            request_headers=request_headers,
        )
        return classified, error_message

    async def _apply_error_classifications(
        self, provider: str, current_cred: str, model: str, e: Exception, classified_error: ClassifiedError,
    ) -> str:
        """Handle rate_limit and quota_exceeded classification logic.
        Returns "fail" for non-retryable requests, "force_rotate" if quota failure limit reached, "rotate" otherwise.
        """
        if classified_error.error_type == "invalid_request":
            lib_logger.warning(
                "Non-retryable invalid_request; failing fast without "
                "rotating/credit-burn. Cred=%s model=%s status=%s",
                mask_credential(current_cred), model,
                classified_error.status_code,
            )
            return "fail"
        if classified_error.error_type == "rate_limit":
            await self._process_rate_limit(
                provider, current_cred, e,
                str(e) if e else None, classified_error,
            )
            await self._resilience.record_rate_429(
                provider, retry_after=classified_error.retry_after
            )
        if classified_error.error_type == "quota_exceeded":
            await self._apply_quota_cooldown(
                provider, current_cred, classified_error
            )
            if await self.increment_quota_failures(current_cred, provider):
                lib_logger.error(
                    "Cred %s quota failure limit reached (3/3), forcing rotation.",
                    mask_credential(current_cred),
                )
                await self.usage_manager.record_failure(
                    current_cred, model, classified_error
                )
                return "force_rotate"
        return "rotate"

    async def _handle_server_error(
        self,
        e: Exception,
        provider: str,
        current_cred: str,
        model: str,
        attempt: int,
        deadline: float,
        request_headers: dict,
        error_accumulator,
        reset_quota: bool = False,
    ) -> _ErrorDecision:
        """Handle APIConnectionError/InternalServerError/ServiceUnavailableError/RuntimeError.

        Returns _ErrorDecision with action="retry_same_key" (with wait_time),
        "rotate" (max retries exceeded or budget exceeded).
        """
        classified_error, error_message = await self._classify_log_error(
            e, provider, current_cred, model, attempt + 1, request_headers,
        )

        if reset_quota:
            await self.reset_quota_failures(current_cred, provider)

        await self.usage_manager.record_failure(
            current_cred, model, classified_error,
            increment_consecutive_failures=False,
        )

        if isinstance(e, RuntimeError) and "client has been closed" in str(e):
            self._reset_litellm_client_cache()

        if attempt >= self.max_retries - 1:
            error_accumulator.record_error(
                current_cred, classified_error, error_message,
                attempt_number=attempt + 1,
            )
            lib_logger.warning(
                "Cred %s failed after max retries. Rotating.",
                mask_credential(current_cred),
            )
            return _ErrorDecision(
                action="rotate", classified_error=classified_error,
                error_message=error_message,
            )

        wait_time = compute_backoff_with_jitter(
            attempt, max_wait=RETRY_SAME_KEY_MAX_WAIT, retry_after=classified_error.retry_after,
        )
        remaining_budget = deadline - time.monotonic()
        if wait_time > remaining_budget:
            error_accumulator.record_error(
                current_cred, classified_error, error_message,
                attempt_number=attempt + 1,
            )
            lib_logger.warning(
                "Retry wait (%2.2fs) exceeds budget (%2.2fs). Rotating.",
                wait_time, remaining_budget,
            )
            return _ErrorDecision(
                action="rotate", classified_error=classified_error,
                error_message=error_message,
            )

        lib_logger.warning(
            "Cred %s server error. Retrying in %2.2fs.",
            mask_credential(current_cred), wait_time,
        )
        return _ErrorDecision(
            action="retry_same_key", wait_time=wait_time,
            classified_error=classified_error, error_message=error_message,
        )

    async def _handle_transport_error(
        self,
        e: Exception,
        provider: str,
        current_cred: str,
        model: str,
        attempt: int,
        deadline: float,
        request_headers: dict,
        error_accumulator,
    ) -> _ErrorDecision:
        """Handle httpx.ReadTimeout/PoolTimeout/RemoteProtocolError/ConnectError.

        Returns _ErrorDecision with action="retry_same_key" (with wait_time),
        "rotate" (max retries or budget exceeded).
        """
        classified_error, error_message = await self._classify_log_error(
            e, provider, current_cred, model, attempt + 1, request_headers,
        )

        error_accumulator.record_error(
            current_cred, classified_error, error_message,
            attempt_number=attempt + 1,
        )

        lib_logger.warning(
            "Cred %s transport error (%s): %s.",
            mask_credential(current_cred), type(e).__name__, error_message,
        )

        await self.usage_manager.record_failure(
            current_cred, model, classified_error,
            increment_consecutive_failures=False,
        )

        if attempt >= self.max_retries - 1:
            error_accumulator.record_error(
                current_cred, classified_error, error_message,
                attempt_number=attempt + 1,
            )
            lib_logger.warning(
                "Cred %s failed after max retries. Rotating.",
                mask_credential(current_cred),
            )
            return _ErrorDecision(
                action="rotate", classified_error=classified_error,
                error_message=error_message,
            )

        if not self._sleep_within_budget(attempt, deadline, classified_error):
            error_accumulator.record_error(
                current_cred, classified_error, error_message,
                attempt_number=attempt + 1,
            )
            lib_logger.warning("Retry wait exceeds budget. Rotating.")
            return _ErrorDecision(
                action="rotate", classified_error=classified_error,
                error_message=error_message,
            )

        lib_logger.warning(
            "Cred %s transport error. Retrying within remaining budget.",
            mask_credential(current_cred),
        )
        wait_time = compute_backoff_with_jitter(
            attempt, max_wait=RETRY_SAME_KEY_MAX_WAIT, retry_after=classified_error.retry_after,
        )
        return _ErrorDecision(
            action="retry_same_key", wait_time=wait_time,
            classified_error=classified_error, error_message=error_message,
        )

    async def _handle_rate_limit_error(
        self,
        e: Exception,
        provider: str,
        current_cred: str,
        model: str,
        error_accumulator,
        request_headers: dict,
        attempt: int = 0,
        deadline: float = 0.0,
    ) -> _ErrorDecision:
        """Handle RateLimitError/HTTPStatusError/BadRequestError/InvalidRequestError.

        Returns _ErrorDecision with action="rotate", "retry_same_key", or "fail".
        """
        original_exc = getattr(e, "data", e)
        classified_error = classify_error(original_exc, provider=provider)
        error_message = str(original_exc).partition("\n")[0]

        self._reset_cache_on_auth_error(
            classified_error, original_exc,
            provider=provider, credential=current_cred,
        )

        log_failure(
            api_key=current_cred, model=model,
            attempt=attempt + 1, error=e,
            request_headers=request_headers,
        )

        error_accumulator.record_error(
            current_cred, classified_error, error_message,
            attempt_number=attempt + 1,
        )

        if not should_rotate_on_error(classified_error):
            await self._process_rate_limit(
                provider, current_cred, e,
                str(e) if e else None, classified_error,
            )
            return _ErrorDecision(
                action="fail", classified_error=classified_error,
                error_message=error_message,
            )

        action = await self._apply_error_classifications(
            provider, current_cred, model, e, classified_error,
        )

        # For transient server errors (e.g. HTTPStatusError 5xx), retry same key with backoff
        if (
            should_retry_same_key(classified_error)
            and attempt < self.max_retries - 1
            and deadline > 0
        ):
            wait_time = compute_backoff_with_jitter(
                attempt, max_wait=RETRY_SAME_KEY_MAX_WAIT,
                retry_after=classified_error.retry_after,
            )
            remaining_budget = deadline - time.monotonic()
            if wait_time <= remaining_budget:
                return _ErrorDecision(
                    action="retry_same_key", wait_time=wait_time,
                    classified_error=classified_error,
                    error_message=error_message,
                )

        if action != "force_rotate":
            await self.usage_manager.record_failure(
                current_cred, model, classified_error
            )

        lib_logger.warning(
            "Cred %s %s (HTTP %s). Rotating.",
            mask_credential(current_cred),
            classified_error.error_type,
            classified_error.status_code,
        )
        return _ErrorDecision(
            action="rotate", classified_error=classified_error,
            error_message=error_message,
        )

    async def _handle_generic_error(
        self,
        e: Exception,
        provider: str,
        current_cred: str,
        model: str,
        error_accumulator,
        request_headers: dict,
        request: Optional[Any] = None,
    ) -> _ErrorDecision:
        """Handle unexpected/unclassified errors in the retry loop.

        Returns _ErrorDecision with action="rotate" or "fail".
        """
        classified_error, error_message = await self._classify_log_error(
            e, provider, current_cred, model, 0, request_headers,
        )

        if request and await request.is_disconnected():
            lib_logger.warning(
                "Client disconnected. Aborting retries for %s.",
                mask_credential(current_cred),
            )
            return _ErrorDecision(
                action="fail", classified_error=classified_error,
                error_message=error_message,
            )

        lib_logger.warning(
            "Key %s %s (HTTP %s).",
            mask_credential(current_cred),
            classified_error.error_type,
            classified_error.status_code,
        )

        await self._apply_error_classifications(
            provider, current_cred, model, e, classified_error,
        )

        if not should_rotate_on_error(classified_error):
            return _ErrorDecision(
                action="fail", classified_error=classified_error,
                error_message=error_message,
            )

        error_accumulator.record_error(
            current_cred, classified_error, error_message
        )

        await self.usage_manager.record_failure(
            current_cred, model, classified_error
        )
        return _ErrorDecision(
            action="rotate", classified_error=classified_error,
            error_message=error_message,
        )

    async def _record_non_streaming_success(
        self, current_cred, model, provider, response, transaction_logger,
    ):
        """Record success bookkeeping for non-streaming response.

        Performs: record_success, release_key, log response, reset quota,
        validate_response_quality.  Raises GarbageResponseError on garbage
        so the caller's except handler can rotate.
        """
        await asyncio.gather(
            self.usage_manager.record_success(current_cred, model, response),
            self._resilience.record_success(provider),
            self._resilience.record_rate_success(provider),
            self.usage_manager.release_key(current_cred, model),
        )
        if transaction_logger:
            response_data = (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response
            )
            await transaction_logger.log_response(response_data)
        await self.reset_quota_failures(current_cred, provider)
        try:
            validate_response_quality(response, provider=provider, model=model)
        except GarbageResponseError as exc:
            lib_logger.warning(
                "Garbage response detected for %s/%s, rotating to next credential: %s",
                provider, model, exc.message if hasattr(exc, 'message') else exc,
            )
            raise

    async def _record_streaming_success(self, current_cred: str, model: str, provider: str) -> None:
        """Record success bookkeeping after streaming completes.

        Performs: resilience record_success/rate_success, reset quota.
        Key release is handled by the stream wrapper's finally block.
        """
        await asyncio.gather(
            self._resilience.record_success(provider),
            self._resilience.record_rate_success(provider),
            self.reset_quota_failures(current_cred, provider),
        )

    def _check_max_attempts(self, total_api_attempts: int, last_exception: Optional[Exception]):
        """Raise if total API attempts exceed MAX_TOTAL_ATTEMPTS.

        Called at the start of each inner retry loop iteration to prevent
        runaway retry cycles across all credentials.

        Args:
            total_api_attempts: Current count of total API attempts across
                all credentials and inner retry iterations.
            last_exception: Last exception to re-raise, or None.

        Raises:
            NoAvailableKeysError: When max attempts exceeded and no last exception.
            last_exception: When max attempts exceeded with a prior exception.
        """
        if total_api_attempts > MAX_TOTAL_ATTEMPTS:
            lib_logger.warning(
                "Total API attempts (%s) exceeded MAX_TOTAL_ATTEMPTS (%s). Aborting.",
                total_api_attempts, MAX_TOTAL_ATTEMPTS,
            )
            raise last_exception or NoAvailableKeysError(
                f"Exceeded max total attempts ({MAX_TOTAL_ATTEMPTS})"
            )

    async def _execute_single_attempt(
        self,
        attempt: int,
        max_retries: int,
        current_cred: str,
        call_label: str,
        pre_request_callback: Optional[Callable],
        request: Optional[Any],
        litellm_kwargs: dict,
        provider: str,
    ):
        """Prepare for a single API attempt with pre-attempt boilerplate.

        Handles debug logging of the attempt number and invocation of the
        optional pre-request callback.  The actual API call is performed by
        the caller after this method returns.

        Args:
            attempt: Current attempt index (0-based).
            max_retries: Maximum retries per key.
            current_cred: Current credential string.
            call_label: Label for debug log (e.g. "call" or "stream").
            pre_request_callback: Optional async callback invoked before
                the API call; may modify litellm_kwargs in-place.
            request: Optional request object passed to the callback.
            litellm_kwargs: Mutable kwargs dict for the upcoming API call.
            provider: Provider name.
        """
        if lib_logger.isEnabledFor(logging.DEBUG):
            lib_logger.debug(
                "Attempting %s with credential %s (Attempt %s/%s)",
                call_label, mask_credential(current_cred),
                attempt + 1, max_retries,
            )

        if pre_request_callback:
            await self._invoke_pre_request_callback(
                pre_request_callback, request, litellm_kwargs, provider,
            )

    async def _handle_retry_error(
        self,
        e: Exception,
        provider: str,
        current_cred: str,
        model: str,
        attempt: int,
        deadline: float,
        request_headers: dict,
        error_accumulator,
        *,
        reset_quota: bool = False,
        request: Optional[Any] = None,
    ) -> _ErrorDecision:
        """Handle server, transport, and generic errors in the retry loop.

        Dispatches to the appropriate specialized handler based on exception
        type.  Reraises CancelledError and TypeError / AttributeError /
        KeyError immediately.  Does NOT handle RateLimitError,
        httpx.HTTPStatusError, GarbageResponseError, _StreamedException,
        LiteLLMAPIError, BadRequestError, or InvalidRequestError -- those
        require path-specific handling by the caller.

        Args:
            e: The caught exception.
            provider: Provider name.
            current_cred: Current credential string.
            model: Model string.
            attempt: Current attempt index (0-based).
            deadline: Monotonic deadline for the overall retry cycle.
            request_headers: Cached request headers for logging.
            error_accumulator: RequestErrorAccumulator instance.
            reset_quota: Whether to reset quota failures (standard path).
            request: Optional request object for generic error handler.

        Returns:
            _ErrorDecision with action "retry_same_key", "rotate", or "fail".

        Raises:
            asyncio.CancelledError: Reraised immediately.
            TypeError, AttributeError, KeyError: Reraised immediately.
        """
        if isinstance(e, asyncio.CancelledError):
            raise
        if isinstance(e, (TypeError, AttributeError, KeyError)):
            raise

        if isinstance(e, (APIConnectionError, InternalServerError,
                          ServiceUnavailableError, RuntimeError)):
            return await self._handle_server_error(
                e, provider, current_cred, model,
                attempt, deadline, request_headers, error_accumulator,
                reset_quota=reset_quota,
            )

        if isinstance(e, (httpx.ReadTimeout, httpx.PoolTimeout,
                          httpx.RemoteProtocolError, httpx.ConnectError)):
            return await self._handle_transport_error(
                e, provider, current_cred, model,
                attempt, deadline, request_headers, error_accumulator,
            )

        return await self._handle_generic_error(
            e, provider, current_cred, model,
            error_accumulator, request_headers, request=request,
        )

    async def _record_attempt_metrics(
        self,
        current_cred: str,
        model: str,
        provider: str,
        *,
        streaming: bool = False,
        response: Any = None,
        transaction_logger: Any = None,
    ):
        """Record success metrics after a completed API attempt.

        Delegates to _record_non_streaming_success or _record_streaming_success
        based on the streaming flag.

        Args:
            current_cred: Current credential string.
            model: Model string.
            provider: Provider name.
            streaming: Whether this was a streaming attempt.
            response: Response object (non-streaming only).
            transaction_logger: Optional transaction logger (non-streaming only).
        """
        if streaming:
            await self._record_streaming_success(current_cred, model, provider)
        else:
            await self._record_non_streaming_success(
                current_cred, model, provider, response, transaction_logger,
            )


class RetryMixin(
    RetryCommonMixin,
    RetryContextBuilderMixin,
    NonStreamingRetryMixin,
    StreamingRetryMixin,
    RetryBaseMixin,
):
    """Mixin with retry logic methods for RotatingClient."""
