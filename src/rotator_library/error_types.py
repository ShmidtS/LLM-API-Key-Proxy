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
        message: Human-readable message about the error
    """

    def __init__(self, model: str, message: str = ""):
        self.model = model
        self.message = (
            message or f"Input tokens exceed context window for model {model}"
        )
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


@functools.lru_cache(maxsize=512)
def mask_credential(credential: str) -> str:
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


class RequestErrorAccumulator:
    __slots__ = (
        "abnormal_errors", "normal_errors", "_tried_credentials",
        "timeout_occurred", "model", "provider",
    )

    """
    Tracks errors encountered during a request's credential rotation cycle.

    Used to build informative error messages for clients when all credentials
    are exhausted. Distinguishes between abnormal errors (that need attention)
    and normal errors (expected during operation).
    """

    def __init__(self):
        self.abnormal_errors: list = []  # 403, 401 - always report details
        self.normal_errors: list = []  # 429, 5xx - summarize only
        self._tried_credentials: set = set()  # Track unique credentials
        self.timeout_occurred: bool = False
        self.model: str = ""
        self.provider: str = ""

    def record_error(
        self, credential: str, classified_error: ClassifiedError, error_message: str
    ):
        """Record an error for a credential."""
        self._tried_credentials.add(credential)
        masked_cred = mask_credential(credential)

        error_record = {
            "credential": masked_cred,
            "error_type": classified_error.error_type,
            "status_code": classified_error.status_code,
            "message": self._truncate_message(error_message, 150),
        }

        if is_abnormal_error(classified_error):
            self.abnormal_errors.append(error_record)
        else:
            self.normal_errors.append(error_record)

    @property
    def total_credentials_tried(self) -> int:
        """Return the number of unique credentials tried."""
        return len(self._tried_credentials)

    def _truncate_message(self, message: str, max_length: int = 150) -> str:
        """Truncate error message for readability."""
        # Take first line and truncate
        first_line = message.partition("\n")[0]
        if len(first_line) > max_length:
            return first_line[:max_length] + "..."
        return first_line

    def has_errors(self) -> bool:
        """Check if any errors were recorded."""
        return bool(self.abnormal_errors or self.normal_errors)

    def has_abnormal_errors(self) -> bool:
        """Check if any abnormal errors were recorded."""
        return bool(self.abnormal_errors)

    def get_normal_error_summary(self) -> str:
        """Get a summary of normal errors (not individual details)."""
        if not self.normal_errors:
            return ""

        # Count by type
        counts = {}
        for err in self.normal_errors:
            err_type = err["error_type"]
            counts[err_type] = counts.get(err_type, 0) + 1

        # Build summary like "3 rate_limit, 1 server_error"
        parts = [f"{count} {err_type}" for err_type, count in counts.items()]
        return ", ".join(parts)

    def build_client_error_response(self) -> dict:
        """
        Build a structured error response for the client.

        Returns a dict suitable for JSON serialization in the error response.
        """
        # Determine the primary failure reason
        if self.timeout_occurred:
            error_type = "proxy_timeout"
            base_message = f"Request timed out after trying {self.total_credentials_tried} credential(s)"
        else:
            error_type = "proxy_all_credentials_exhausted"
            base_message = f"All {self.total_credentials_tried} credential(s) exhausted for {self.provider}"

        # Build human-readable message
        message_parts = [base_message]

        if self.abnormal_errors:
            message_parts.append("\n\nCredential issues (require attention):")
            for err in self.abnormal_errors:
                status = (
                    f"HTTP {err['status_code']}"
                    if err["status_code"] is not None
                    else err["error_type"]
                )
                message_parts.append(
                    f"\n  \u2022 {err['credential']}: {status} - {err['message']}"
                )

        normal_summary = self.get_normal_error_summary()
        if normal_summary:
            if self.abnormal_errors:
                message_parts.append(
                    f"\n\nAdditionally: {normal_summary} (expected during normal operation)"
                )
            else:
                message_parts.append(f"\n\nAll failures were: {normal_summary}")
                message_parts.append(
                    "\nThis is normal during high load - retry later or add more credentials."
                )

        response = {
            "error": {
                "message": "".join(message_parts),
                "type": error_type,
                "details": {
                    "model": self.model,
                    "provider": self.provider,
                    "credentials_tried": self.total_credentials_tried,
                    "timeout": self.timeout_occurred,
                },
            }
        }

        # Only include abnormal errors in details (they need attention)
        if self.abnormal_errors:
            response["error"]["details"]["abnormal_errors"] = self.abnormal_errors

        # Include summary of normal errors
        if normal_summary:
            response["error"]["details"]["normal_error_summary"] = normal_summary

        return response

    def build_log_message(self) -> str:
        """
        Build a concise log message for server-side logging.

        Shorter than client message, suitable for terminal display.
        """
        parts = []

        if self.timeout_occurred:
            parts.append(
                f"TIMEOUT: {self.total_credentials_tried} creds tried for {self.model}"
            )
        else:
            parts.append(
                f"ALL CREDS EXHAUSTED: {self.total_credentials_tried} tried for {self.model}"
            )

        if self.abnormal_errors:
            abnormal_summary = ", ".join(
                f"{e['credential']}={e['status_code'] or e['error_type']}"
                for e in self.abnormal_errors
            )
            parts.append(f"ISSUES: {abnormal_summary}")

        normal_summary = self.get_normal_error_summary()
        if normal_summary:
            parts.append(f"Normal: {normal_summary}")

        return " | ".join(parts)
