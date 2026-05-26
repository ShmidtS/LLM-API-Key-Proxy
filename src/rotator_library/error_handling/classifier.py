# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
from typing import Dict, Optional

import httpx
from litellm.exceptions import (  # type: ignore[import-untyped]
    APIConnectionError,
    APIError as LiteLLMAPIError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.llms.openai.common_utils import OpenAIError  # type: ignore[import-untyped]

from ..error_types import (
    ClassifiedError,
    CredentialNeedsReauthError,
    EmptyResponseError,
    GarbageResponseError,
    PreRequestCallbackError,
    TransientQuotaError,
)
from .quota_parser import detect_ip_throttle, extract_quota_details, try_parse_provider_quota_error
from .retry_after import get_retry_after

lib_logger = logging.getLogger("rotator_library")

_CONTEXT_WINDOW_ERROR_PATTERNS = (
    "context_length",
    "max_tokens",
    "token limit",
    "context window",
    "too many tokens",
    "too long",
)

_POLICY_ERROR_PATTERNS = (
    "policy",
    "safety",
    "content blocked",
    "prompt blocked",
)

_UPSTREAM_ERROR_PATTERNS = (
    "provider returned error",
    "upstream error",
    "upstream temporarily unavailable",
    "upstream service unavailable",
    "service overloaded",
    "overloaded",
    "try again later",
)

_ACCOUNT_BILLING_ERROR_PATTERNS = (
    "account_overdue",
    "overdue account",
    "arrearage",
    "overdue payment",
    "account in good standing",
    "insufficient balance",
    "please recharge",
    "out of credit",
    "payment required",
    "add credits to continue",
    "credits required",
    "usage_limit_exceeded",
)

_LITELLM_API_CREDIT_PATTERNS = (
    "account_overdue",
    "overdue account",
    "add credits to continue",
    "credits required",
    "usage_limit_exceeded",
    "insufficient balance",
    "out of credit",
    "payment required",
    "quota",
)

_AUTHENTICATION_ERROR_PATTERNS = (
    "invalid_iam_token",
    "invalid iam token",
)

_SERVER_ERROR_PATTERNS = (
    "server had an error",
    "the server had an error",
)


def _match_patterns(text: str, patterns: tuple[str, ...]) -> bool:
    """Check if any pattern string is a substring of *text*."""
    return any(p in text for p in patterns)


def classify_rate_limit(
    e: Exception,
    error_text: str,
    status_code: int,
    retry_after: Optional[int],
    provider: Optional[str] = None,
    response_text: Optional[str] = None,
) -> ClassifiedError:
    """
    Shared logic for classifying 429 / rate-limit errors.

    Distinguishes between:
    - quota_exceeded (contains "quota" or "resource_exhausted")
    - ip_rate_limit (detected via _detect_ip_throttle)
    - rate_limit (fallback)

    If *response_text* is provided, attempts to extract quotaValue / quotaId
    from Google/Gemini API error JSON.
    """
    if "quota" in error_text or "resource_exhausted" in error_text:
        quota_value = None
        quota_id = None
        if response_text:
            try:
                quota_value, quota_id = extract_quota_details(response_text)
            except (ValueError, OSError, KeyError, TypeError):
                lib_logger.debug("Could not read error response for quota details", exc_info=True)
        return ClassifiedError(
            error_type="quota_exceeded",
            original_exception=e,
            status_code=status_code,
            retry_after=retry_after,
            quota_value=quota_value,
            quota_id=quota_id,
        )

    ip_throttle_cooldown = detect_ip_throttle(error_text, provider=provider)
    if ip_throttle_cooldown is not None:
        return ClassifiedError(
            error_type="ip_rate_limit",
            original_exception=e,
            status_code=status_code,
            retry_after=retry_after or ip_throttle_cooldown,
        )

    return ClassifiedError(
        error_type="rate_limit",
        original_exception=e,
        status_code=status_code,
        retry_after=retry_after,
    )


def is_provider_abort(raw_response: Optional[Dict]) -> bool:
    """
    Detect if provider aborted the stream.

    Returns True if:
    - finish_reason == 'error'
    - native_finish_reason == 'abort'
    - Empty content with error indication
    """
    if not raw_response:
        return False

    finish_reason = raw_response.get("finish_reason")
    native_reason = raw_response.get("native_finish_reason")

    if finish_reason == "error":
        return True
    if native_reason == "abort":
        return True

    choices = raw_response.get("choices", [])
    if choices:
        for choice in choices:
            if choice.get("finish_reason") == "error":
                return True
            message = choice.get("message", {})
            delta = choice.get("delta", {})
            if not message.get("content") and not delta.get("content"):
                if choice.get("native_finish_reason") == "abort":
                    return True

    return False


def classify_stream_error(raw_response: Dict) -> ClassifiedError:
    """
    Classify streaming errors from provider response.

    Creates ClassifiedError appropriate for retry logic.
    """
    raw_str = str(raw_response).lower()
    if "inception" in raw_str:
        if _match_patterns(raw_str, _SERVER_ERROR_PATTERNS):
            return ClassifiedError(
                error_type="server_error",
                status_code=503,
                original_exception=None,
                retry_after=int(5.0),
            )

    if is_provider_abort(raw_response):
        return ClassifiedError(
            error_type="api_connection",
            status_code=503,
            original_exception=None,
            retry_after=2,
        )

    return ClassifiedError(
        error_type="server_error",
        status_code=500,
        original_exception=None,
        retry_after=5,
    )


def _classify_stream_dict(e: Exception) -> Optional[ClassifiedError]:
    if not isinstance(e, dict):
        return None
    if is_provider_abort(e):
        lib_logger.warning(
            "Provider abort detected in stream: finish_reason=%s, native_finish_reason=%s",
            e.get("finish_reason"), e.get("native_finish_reason"),
        )
        return classify_stream_error(e)
    error_obj = e.get("error")
    if isinstance(error_obj, dict):
        if is_provider_abort(error_obj):
            return classify_stream_error(error_obj)
    return None


def _classify_http_status_error(
    e: httpx.HTTPStatusError, provider: Optional[str]
) -> ClassifiedError:
    status_code = e.response.status_code

    try:
        response_text_raw = e.response.text if hasattr(e.response, "text") else ""
    except (AttributeError, OSError):
        lib_logger.debug("Could not read httpx error response body", exc_info=True)
        response_text_raw = ""
    error_body_lower = response_text_raw.lower()

    if status_code == 401:
        return ClassifiedError(
            error_type="authentication",
            original_exception=e,
            status_code=status_code,
            raw_response_body=response_text_raw,
        )
    if status_code == 403:
        if "edge_ip_restricted" in error_body_lower or "error 1034" in error_body_lower or (
            "cloudflare" in error_body_lower and "owner_action_required" in error_body_lower
        ):
            return ClassifiedError(
                error_type="ip_rate_limit",
                original_exception=e,
                status_code=status_code,
                raw_response_body=response_text_raw,
            )
        return ClassifiedError(
            error_type="forbidden",
            original_exception=e,
            status_code=status_code,
            raw_response_body=response_text_raw,
        )
    if status_code == 429:
        retry_after = get_retry_after(e)
        result = classify_rate_limit(
            e,
            error_text=error_body_lower,
            status_code=status_code,
            retry_after=retry_after,
            provider=provider,
            response_text=response_text_raw,
        )
        # Attach raw response body to rate limit errors
        result.raw_response_body = response_text_raw
        return result
    if status_code == 400:
        result = _classify_http_400(e, status_code, error_body_lower)
        result.raw_response_body = response_text_raw
        return result
    if 400 <= status_code < 500:
        return ClassifiedError(
            error_type="invalid_request",
            original_exception=e,
            status_code=status_code,
            raw_response_body=response_text_raw,
        )
    if 500 <= status_code:
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=status_code,
            raw_response_body=response_text_raw,
        )
    return ClassifiedError(
        error_type="unknown",
        original_exception=e,
        status_code=status_code,
        raw_response_body=response_text_raw,
    )


def _classify_http_400(
    e: Exception, status_code: int, error_body_lower: str
) -> ClassifiedError:
    if _match_patterns(error_body_lower, _CONTEXT_WINDOW_ERROR_PATTERNS):
        return ClassifiedError(
            error_type="context_window_exceeded",
            original_exception=e,
            status_code=status_code,
        )

    if _match_patterns(error_body_lower, _POLICY_ERROR_PATTERNS):
        return ClassifiedError(
            error_type="invalid_request",
            original_exception=e,
            status_code=status_code,
        )

    if _match_patterns(error_body_lower, _UPSTREAM_ERROR_PATTERNS):
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=503,
        )

    return ClassifiedError(
        error_type="invalid_request",
        original_exception=e,
        status_code=status_code,
    )


def _classify_builtin_error(
    e: Exception, error_str: str, status_code: Optional[int]
) -> Optional[ClassifiedError]:
    if isinstance(e, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)):
        return ClassifiedError(
            error_type="api_connection", original_exception=e, status_code=status_code
        )

    if isinstance(e, PreRequestCallbackError):
        return ClassifiedError(
            error_type="pre_request_callback_error",
            original_exception=e,
            status_code=400,
        )

    if isinstance(e, CredentialNeedsReauthError):
        return ClassifiedError(
            error_type="credential_reauth_needed",
            original_exception=e,
            status_code=401,
        )

    if isinstance(e, EmptyResponseError):
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=503,
        )

    if isinstance(e, TransientQuotaError):
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=503,
        )

    if isinstance(e, GarbageResponseError):
        return ClassifiedError(
            error_type="garbage_response",
            original_exception=e,
            status_code=503,
            reason=e.reason if hasattr(e, 'reason') else error_str,
        )

    return None


def _classify_invalid_request_error(
    e: Exception,
    provider: Optional[str],
    error_str_lower: str,
    status_code: Optional[int],
) -> ClassifiedError:
    if _match_patterns(error_str_lower, _AUTHENTICATION_ERROR_PATTERNS):
        return ClassifiedError(
            error_type="authentication",
            original_exception=e,
            status_code=401,
        )

    if _match_patterns(error_str_lower, _UPSTREAM_ERROR_PATTERNS):
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=status_code or 503,
        )

    if _match_patterns(error_str_lower, _ACCOUNT_BILLING_ERROR_PATTERNS):
        return ClassifiedError(
            error_type="quota_exceeded",
            original_exception=e,
            status_code=status_code or 402,
            retry_after=7200,
            reason="account_billing_issue",
        )

    if provider:
        result = try_parse_provider_quota_error(e, provider, status_code=status_code or 400)
        if result is not None:
            return result

    return ClassifiedError(
        error_type="invalid_request",
        original_exception=e,
        status_code=status_code or 400,
    )


def _classify_api_connection_error(
    e: Exception, error_str_lower: str, status_code: Optional[int]
) -> ClassifiedError:
    if _match_patterns(error_str_lower, _SERVER_ERROR_PATTERNS):
        return ClassifiedError(
            error_type='api_connection',
            original_exception=e,
            status_code=503,
            retry_after=5,
        )
    return ClassifiedError(
        error_type="api_connection",
        original_exception=e,
        status_code=status_code or 503,
    )


def _classify_openai_error(
    e: Exception, error_str_lower: str, status_code: Optional[int]
) -> ClassifiedError:
    if "not found" in error_str_lower or "404" in error_str_lower:
        return ClassifiedError(
            error_type="invalid_request",
            original_exception=e,
            status_code=status_code or 404,
            reason="not_found_openai_error",
        )
    return ClassifiedError(
        error_type="server_error",
        original_exception=e,
        status_code=status_code or 500,
        reason="openai_error_unclassified",
    )


def _classify_not_found_error(
    e: Exception, error_str_lower: str, status_code: Optional[int]
) -> ClassifiedError:
    if "invalid path" in error_str_lower or "only accepts" in error_str_lower:
        return ClassifiedError(
            error_type="invalid_request",
            original_exception=e,
            status_code=status_code or 404,
            reason="endpoint_not_found",
        )
    return ClassifiedError(
        error_type="invalid_request",
        original_exception=e,
        status_code=status_code or 404,
        reason="model_not_found",
    )


def _classify_litellm_api_error(
    e: Exception, error_str_lower: str, status_code: Optional[int]
) -> ClassifiedError:
    if _match_patterns(error_str_lower, _ACCOUNT_BILLING_ERROR_PATTERNS):
        return ClassifiedError(
            error_type="quota_exceeded",
            original_exception=e,
            status_code=status_code or 402,
            retry_after=7200,
            reason="account_billing_issue",
        )
    if _match_patterns(error_str_lower, _LITELLM_API_CREDIT_PATTERNS):
        return ClassifiedError(
            error_type="quota_exceeded",
            original_exception=e,
            status_code=status_code or 402,
            retry_after=300,
            reason="litellm_api_credits",
        )
    if status_code == 403:
        return ClassifiedError(
            error_type="forbidden",
            original_exception=e,
            status_code=status_code,
        )
    if (
        "invalid api key" in error_str_lower
        or "invalid_api_key" in error_str_lower
        or _match_patterns(error_str_lower, _AUTHENTICATION_ERROR_PATTERNS)
    ):
        return ClassifiedError(
            error_type="authentication",
            original_exception=e,
            status_code=status_code or 401,
        )
    return ClassifiedError(
        error_type="unknown",
        original_exception=e,
        status_code=status_code,
        reason="litellm_api_error_unclassified",
    )


def classify_error(e: Exception, provider: Optional[str] = None) -> ClassifiedError:
    """
    Classifies an exception into a structured ClassifiedError object.
    Now handles both litellm and httpx exceptions.

    If provider is specified and has a parse_quota_error() method,
    attempts provider-specific error parsing first before falling back
    to generic classification.
    """
    error_str = str(e)
    error_str_lower = error_str.lower()

    if isinstance(e, httpx.HTTPStatusError):
        early_status_code = e.response.status_code
    else:
        early_status_code = getattr(e, "status_code", None)

    if early_status_code in (None, 429):
        result = try_parse_provider_quota_error(e, provider, status_code=429)
        if result is not None:
            return result

    stream_result = _classify_stream_dict(e)
    if stream_result is not None:
        return stream_result

    status_code = getattr(e, "status_code", None)

    if isinstance(e, httpx.HTTPStatusError):
        return _classify_http_status_error(e, provider)

    builtin_result = _classify_builtin_error(e, error_str, status_code)
    if builtin_result is not None:
        return builtin_result

    if isinstance(e, RateLimitError):
        retry_after = get_retry_after(e)
        return classify_rate_limit(
            e,
            error_text=error_str_lower,
            status_code=status_code or 429,
            retry_after=retry_after,
            provider=provider,
        )

    if isinstance(e, (AuthenticationError,)):
        return ClassifiedError(
            error_type="authentication",
            original_exception=e,
            status_code=status_code or 401,
        )

    if isinstance(e, (InvalidRequestError, BadRequestError)):
        return _classify_invalid_request_error(e, provider, error_str_lower, status_code)

    if isinstance(e, ContextWindowExceededError):
        return ClassifiedError(
            error_type="context_window_exceeded",
            original_exception=e,
            status_code=status_code or 400,
        )

    if isinstance(e, (APIConnectionError, Timeout)):
        return _classify_api_connection_error(e, error_str_lower, status_code)

    if isinstance(e, (ServiceUnavailableError, InternalServerError)):
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=status_code or 503,
        )

    if isinstance(e, OpenAIError):
        return _classify_openai_error(e, error_str_lower, status_code)

    if isinstance(e, NotFoundError):
        return _classify_not_found_error(e, error_str_lower, status_code)

    if isinstance(e, LiteLLMAPIError):
        return _classify_litellm_api_error(e, error_str_lower, status_code)

    return ClassifiedError(
        error_type="unknown", original_exception=e, status_code=status_code
    )
