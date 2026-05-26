# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
import functools
from typing import Optional

from .ip_throttle_detector import ThrottleAssessment


class ClassifiedError:
    """A structured representation of a classified error."""

    __slots__ = (
        "error_type",
        "original_exception",
        "status_code",
        "retry_after",
        "quota_reset_timestamp",
        "throttle_assessment",
        "quota_value",
        "quota_id",
        "reason",
        "raw_response_body",
        "attempt_number",
    )

    def __init__(
        self,
        error_type: str,
        original_exception: Optional[Exception] = None,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
        quota_reset_timestamp: Optional[float] = None,
        throttle_assessment: Optional[ThrottleAssessment] = None,
        quota_value: Optional[str] = None,
        quota_id: Optional[str] = None,
        reason: Optional[str] = None,
        raw_response_body: Optional[str] = None,
        attempt_number: int = 0,
    ):
        self.error_type = error_type
        self.original_exception = original_exception
        self.status_code = status_code
        self.retry_after = retry_after
        # Unix timestamp when quota resets (from quota_exhausted errors)
        # This is the authoritative reset time parsed from provider's error response
        self.quota_reset_timestamp = quota_reset_timestamp
        # IP throttle assessment (when multiple credentials show correlated 429s)
        self.throttle_assessment = throttle_assessment
        # Quota details from Google/Gemini API errors (quotaValue and quotaId)
        self.quota_value = quota_value
        self.quota_id = quota_id
        # Provider-specific reason (e.g., INSUFFICIENT_BALANCE, QUOTA_EXHAUSTED)
        self.reason = reason
        # Raw response body from provider (max 2KB stored internally)
        self.raw_response_body = raw_response_body
        # Which attempt number produced this error (1-based, 0 = unset)
        self.attempt_number = attempt_number

    def __str__(self):
        parts = [
            f"type={self.error_type}",
            f"status={self.status_code}",
            f"retry_after={self.retry_after}",
        ]
        if self.quota_reset_timestamp:
            parts.append(f"quota_reset_ts={self.quota_reset_timestamp}")
        if self.throttle_assessment:
            parts.append(f"throttle_scope={self.throttle_assessment.scope.value}")
        if self.quota_value:
            parts.append(f"quota_value={self.quota_value}")
        if self.quota_id:
            parts.append(f"quota_id={self.quota_id}")
        if self.reason:
            parts.append(f"reason={self.reason}")
        parts.append(f"original_exc={self.original_exception}")
        return f"ClassifiedError({', '.join(parts)})"


class NoAvailableKeysError(Exception):
    """Raised when no API keys are available for a request after waiting."""

    pass


class PreRequestCallbackError(Exception):
    """Raised when a pre-request callback fails."""

    pass


class CredentialNeedsReauthError(Exception):
    """
    Raised when a credential's refresh token is invalid and re-authentication is required.

    This is a rotatable error - the request should try the next credential while
    the broken credential is queued for re-authentication in the background.

    Unlike generic HTTPStatusError, this exception signals:
    - The credential is temporarily unavailable (needs user action)
    - Re-auth has already been queued
    - The request should rotate to the next credential without logging scary tracebacks

    Attributes:
        credential_path: Path to the credential file that needs re-auth
        message: Human-readable message about the error
    """

    def __init__(self, credential_path: str, message: str = ""):
        self.credential_path = credential_path
        self.message = (
            message or f"Credential '{credential_path}' requires re-authentication"
        )
        super().__init__(self.message)


class EmptyResponseError(Exception):
    """
    Raised when a provider returns an empty response after multiple retry attempts.

    This is a rotatable error - the request should try the next credential.
    Treated as a transient server-side issue (503 equivalent).

    Attributes:
        provider: The provider name (e.g., "antigravity")
        model: The model that was requested
        message: Human-readable message about the error
    """

    def __init__(self, provider: str, model: str, message: str = ""):
        self.provider = provider
        self.model = model
        self.message = (
            message
            or f"Empty response from {provider}/{model} after multiple retry attempts"
        )
        super().__init__(self.message)


class GarbageResponseError(Exception):
    """
    Raised when a provider returns a garbage/hallucinated response.

    Detected by content quality validation heuristics:
    - High word repetition ratio
    - Code fragment injection into non-code context
    - Mixed-script gibberish patterns
    - File path leakage into response content

    This is a rotatable error - the request should try the next credential.
    Treated as a transient server-side issue (503 equivalent).

    Attributes:
        provider: The provider name
        model: The model that was requested
        reason: Why the response was classified as garbage
    """

    def __init__(self, provider: str, model: str, reason: str = ""):
        self.provider = provider
        self.model = model
        self.reason = reason
        self.message = (
            reason or f"Garbage response from {provider}/{model}"
        )
        super().__init__(self.message)


class SchemaValidationError(GarbageResponseError):
    """Raised when a provider response fails structural schema validation.

    Unlike content-quality garbage detection, this catches malformed structure:
    missing required fields, wrong types, invalid enum values.

    Attributes:
        provider: The provider name
        model: The model that was requested
        reason: Description of the schema violation
        field_path: Dotted path to the violated field (e.g. ``choices[0].message``)
        attempt_count: How many attempts were made before giving up
    """

    def __init__(
        self,
        provider: str,
        model: str,
        reason: str = "",
        field_path: str = "",
        attempt_count: int = 0,
    ):
        self.field_path = field_path
        self.attempt_count = attempt_count
        super().__init__(provider=provider, model=model, reason=reason)


class TransientQuotaError(Exception):
    """
    Raised when a provider returns a 429 without retry timing information.

    This indicates a transient rate limit rather than true quota exhaustion.
    The request has already been retried internally; this error signals
    that the credential should be rotated to try the next one.

    Treated as a transient server-side issue (503 equivalent), same as EmptyResponseError.

    Attributes:
        provider: The provider name (e.g., "antigravity")
        model: The model that was requested
        message: Human-readable message about the error
    """

    def __init__(self, provider: str, model: str, message: str = ""):
        self.provider = provider
        self.model = model
        self.message = (
            message
            or f"Transient 429 from {provider}/{model} after multiple retry attempts"
        )
        super().__init__(self.message)


class ContextOverflowError(Exception):
    """
    Raised when input tokens exceed the model's context window.

    This is a pre-emptive rejection before sending the request to the API,
    based on token counting and model context limits.

    This is NOT a rotatable error - all credentials will fail for the same request.
    The client should reduce the input size or use a model with a larger context window.

    Attributes:
        model: The model that was requested
        tokens_over_limit: How many tokens over the limit (0 if unknown)
        current_tokens: Current token count (0 if unknown)
        context_limit: Context window limit in tokens (0 if unknown)
        message: Human-readable message about the error
    """

    def __init__(
        self,
        model: str,
        message: str = "",
        *,
        tokens_over_limit: int = 0,
        current_tokens: int = 0,
        context_limit: int = 0,
    ):
        self.model = model
        self.tokens_over_limit = tokens_over_limit
        self.current_tokens = current_tokens
        self.context_limit = context_limit
        if not message:
            if tokens_over_limit:
                message = (
                    f"Input tokens exceed context window for model {model}: "
                    f"{current_tokens} tokens, limit {context_limit}, "
                    f"{tokens_over_limit} tokens over limit"
                )
            else:
                message = f"Input tokens exceed context window for model {model}"
        self.message = message
        super().__init__(self.message)


# Abnormal errors that require attention and should always be reported to client
ABNORMAL_ERROR_TYPES = frozenset(
    {
        "forbidden",  # 403 - credential access issue
        "authentication",  # 401 - credential invalid/revoked
        "pre_request_callback_error",  # Internal proxy error
    }
)


def is_abnormal_error(classified_error: ClassifiedError) -> bool:
    """
    Check if an error is abnormal and should be reported to the client.

    Abnormal errors indicate credential issues that need attention:
    - 403 Forbidden: Credential doesn't have access
    - 401 Unauthorized: Credential is invalid/revoked

    Normal errors are expected during operation:
    - 429 Rate limit: Expected during high load
    - 5xx Server errors: Transient provider issues
    """
    return classified_error.error_type in ABNORMAL_ERROR_TYPES


def _mask_credential_uncached(credential: str) -> str:
    """
    Mask a credential for safe display in logs and error messages.

    - For API keys: shows last 6 characters (e.g., "...xyz123")
    - For OAuth file paths: shows just the filename (e.g., "antigravity_oauth_1.json")
    """
    if os.path.isfile(credential) or credential.endswith(".json"):
        return os.path.basename(credential)
    elif len(credential) > 6:
        return f"...{credential[-6:]}"
    else:
        return "***"


_mask_credential_cache = functools.lru_cache(maxsize=512)  # type: ignore[arg-type]
mask_credential = _mask_credential_cache(_mask_credential_uncached)

from .error_accumulator import RequestErrorAccumulator  # noqa: E402 \u2014 re-export for backwards compat
