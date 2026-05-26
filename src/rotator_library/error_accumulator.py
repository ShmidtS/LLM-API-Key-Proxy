# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
import time

from .error_types import ClassifiedError, mask_credential, is_abnormal_error

_ENRICHED_ERRORS_ENABLED_CACHE: bool | None = None

# Mapping from error_type / condition to stable machine-readable code
_ERROR_TYPE_TO_CODE: dict[str, str] = {
    "proxy_timeout": "PROXY_TIMEOUT",
    "proxy_all_credentials_exhausted": "ALL_CREDENTIALS_EXHAUSTED",
    "context_window_exceeded": "CONTEXT_OVERFLOW",
    "authentication": "CREDENTIAL_AUTH_FAILURE",
    "forbidden": "CREDENTIAL_AUTH_FAILURE",
    "rate_limit": "RATE_LIMIT_EXHAUSTED",
    "quota_exceeded": "RATE_LIMIT_EXHAUSTED",
    "ip_rate_limit": "RATE_LIMIT_EXHAUSTED",
}

_RAW_RESPONSE_INTERNAL_MAX = 2048  # 2KB internal storage cap
_RAW_RESPONSE_SNIPPET_MAX = 500  # 500 char client-facing snippet


def _enriched_errors_enabled() -> bool:
    """Check the ENRICHED_ERRORS_ENABLED env var (cached after first read)."""
    global _ENRICHED_ERRORS_ENABLED_CACHE
    if _ENRICHED_ERRORS_ENABLED_CACHE is None:
        val = os.environ.get("ENRICHED_ERRORS_ENABLED", "true").lower()
        _ENRICHED_ERRORS_ENABLED_CACHE = val not in ("0", "false", "no", "off")
    return _ENRICHED_ERRORS_ENABLED_CACHE


class RequestErrorAccumulator:
    __slots__ = (
        "abnormal_errors", "normal_errors", "_tried_credentials",
        "timeout_occurred", "model", "provider", "_enriched_attempts",
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
        # Chronological list of enriched attempt records (when feature flag on)
        self._enriched_attempts: list = []

    def record_error(
        self,
        credential: str,
        classified_error: ClassifiedError,
        error_message: str,
        raw_response_body: str | None = None,
        attempt_number: int = 0,
    ):
        """Record an error for a credential.

        The optional ``raw_response_body`` and ``attempt_number`` kwargs are
        used to build the enriched ``details.attempts`` / ``details.causal_chain``
        arrays in the client error response. They are purely additive; omitting
        them preserves prior behaviour.
        """
        self._tried_credentials.add(credential)
        masked_cred = mask_credential(credential)

        # Prefer the raw body carried on the ClassifiedError itself (populated
        # by the classifier) when caller did not pass one explicitly.
        raw_body = raw_response_body or getattr(classified_error, "raw_response_body", None)
        attempt_num = attempt_number or getattr(classified_error, "attempt_number", 0) or 0

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

        if _enriched_errors_enabled():
            self._enriched_attempts.append({
                "credential": masked_cred,
                "error_type": classified_error.error_type,
                "status_code": classified_error.status_code,
                "attempt_number": int(attempt_num) if attempt_num else len(self._enriched_attempts) + 1,
                "timestamp": time.time(),
                "raw_response_snippet": self._build_raw_snippet(raw_body, credential),
            })

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

    @staticmethod
    def _build_raw_snippet(raw_body: str | None, credential: str) -> str:
        """Build a client-safe snippet from a raw response body.

        Truncates to ``_RAW_RESPONSE_SNIPPET_MAX`` chars and masks any
        occurrence of the credential string itself (to avoid leaking keys
        back to the client).
        """
        if not raw_body:
            return ""
        # Cap internal storage at 2KB, then take the last N chars for the snippet
        capped = raw_body[:_RAW_RESPONSE_INTERNAL_MAX]
        snippet = capped[-_RAW_RESPONSE_SNIPPET_MAX:]
        # Mask the credential if it appears verbatim in the body
        if credential and credential in snippet:
            snippet = snippet.replace(credential, mask_credential(credential))
        return snippet

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
        When ``ENRICHED_ERRORS_ENABLED`` is true (default), includes:
        - ``error.code``: stable machine-readable error code
        - ``details.attempts``: chronological array of attempt records
        - ``details.causal_chain``: array of error type labels
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
                    f"\n  • {err['credential']}: {status} - {err['message']}"
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

        # Enriched error fields (feature-flagged, additive only)
        if _enriched_errors_enabled():
            # Add machine-readable error code
            response["error"]["code"] = _ERROR_TYPE_TO_CODE.get(error_type, "SERVER_ERROR")

            # Add chronological attempts array
            if self._enriched_attempts:
                response["error"]["details"]["attempts"] = self._enriched_attempts

                # Build causal chain: chronological sequence of error types
                response["error"]["details"]["causal_chain"] = [
                    att["error_type"] for att in self._enriched_attempts
                ]

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
