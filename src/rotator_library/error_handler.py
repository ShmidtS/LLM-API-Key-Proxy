# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import re
import json
import os
import logging
from typing import Optional, Dict, Any, Tuple
import httpx

from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
    OpenAIError,
    InternalServerError,
    Timeout,
    ContextWindowExceededError,
)

from .ip_throttle_detector import (
    IPThrottleDetector,
    ThrottleAssessment,
    ThrottleScope,
    get_ip_throttle_detector,
)

lib_logger = logging.getLogger("rotator_library")

# Default cooldown for rate limits without retry_after (reduced from 60s)
RATE_LIMIT_DEFAULT_COOLDOWN = 10  # seconds

# IP-based throttle detection patterns
# These patterns indicate rate limiting at IP level rather than API key level
IP_THROTTLE_INDICATORS = frozenset(
    {
        "ip",
        "ip_address",
        "source ip",
        "client ip",
        "rate limit exceeded for your ip",
        "too many requests from your ip",
        "rate limit exceeded for ip",
        "too many requests from ip",
        "ip rate limit",
        "ip-based rate limit",
    }
)

# Patterns that indicate a GENERIC rate limit (no specific key mentioned)
# When these appear without key-specific info, it's likely IP-level throttling
GENERIC_RATE_LIMIT_PATTERNS = frozenset(
    {
        "rate limit exceeded",
        "too many requests",
        "requests per minute",
        "requests per second",
        "rate_limit_exceeded",
        "ratelimitexceeded",
        "429 too many requests",
        "usage limit reached",
        "usage limit exceeded",
        "limit reached",
    }
)

# Patterns that indicate KEY-SPECIFIC rate limiting (not IP-level)
KEY_SPECIFIC_PATTERNS = frozenset(
    {
        "api key",
        "apikey",
        "key ",
        "your key",
        "this key",
        "credential",
        "token",
        "quota",  # quota is usually per-key/account
        "resource_exhausted",  # Google's quota error
    }
)

# Providers that route through multiple backends - IP throttle detection is unreliable
# These providers aggregate multiple upstream APIs, so rate limits may vary per backend
PROXY_PROVIDERS = frozenset(
    {
        "kilocode",    # Routes to multiple providers (minimax, moonshot, z-ai, etc.)
        "openrouter",  # Routes to 100+ providers
        "requesty",    # Router/aggregator
    }
)


def _detect_ip_throttle(error_body: Optional[str], provider: Optional[str] = None) -> Optional[int]:
    """
    Detect IP-based rate limiting from error response body.

    IP throttling affects all credentials from the same IP, so rotation
    won't help. Returns a cooldown period to wait before retrying.

    Detection strategy:
    1. Explicit IP mentions -> IP throttle (high confidence)
    2. Generic rate limit WITHOUT key-specific info -> likely IP throttle
       (BUT skip for PROXY_PROVIDERS - they route to multiple backends)
    3. Key-specific rate limit info -> NOT IP throttle

    Args:
        error_body: The raw error response body (case-insensitive matching)
        provider: Optional provider name (used to skip unreliable detection for proxy providers)

    Returns:
        Cooldown seconds if IP throttle detected, None otherwise
    """
    if not error_body:
        return None

    error_body_lower = error_body.lower()

    # Check for explicit IP throttle indicators (highest confidence)
    # This is reliable even for proxy providers
    for indicator in IP_THROTTLE_INDICATORS:
        if indicator in error_body_lower:
            lib_logger.info(
                f"Detected IP-based rate limiting: found indicator '{indicator}'"
            )
            return RATE_LIMIT_DEFAULT_COOLDOWN

    # For PROXY_PROVIDERS (kilocode, openrouter), skip generic rate limit detection
    # These providers route to multiple backends, so generic rate limits may be
    # backend-specific rather than IP-specific
    if provider and provider in PROXY_PROVIDERS:
        lib_logger.debug(
            f"Skipping generic IP throttle detection for proxy provider '{provider}' "
            "- rate limits may be backend-specific"
        )
        return None

    # Check if this is a generic rate limit without key-specific info
    # This indicates IP-level throttling (provider doesn't know which key)
    has_generic_rate_limit = any(
        pattern in error_body_lower for pattern in GENERIC_RATE_LIMIT_PATTERNS
    )
    has_key_specific_info = any(
        pattern in error_body_lower for pattern in KEY_SPECIFIC_PATTERNS
    )

    if has_generic_rate_limit and not has_key_specific_info:
        lib_logger.info(
            "Detected likely IP-based rate limiting: generic rate limit message "
            "without key-specific info"
        )
        return RATE_LIMIT_DEFAULT_COOLDOWN

    return None


def _parse_duration_string(duration_str: str) -> Optional[int]:
    """
    Parse duration strings in various formats to total seconds.

    Handles:
    - Milliseconds: '290.979975ms' -> 1 second (rounds up for sub-second values)
    - Compound durations: '156h14m36.752463453s', '2h30m', '45m30s'
    - Simple durations: '562476.752463453s', '3600s', '60m', '2h'
    - Plain seconds (no unit): '562476'

    Args:
        duration_str: Duration string to parse

    Returns:
        Total seconds as integer, or None if parsing fails.
        For sub-second values, returns at least 1 to avoid retry floods.
    """
    if not duration_str:
        return None

    total_seconds = 0.0
    remaining = duration_str.strip().lower()

    # Try parsing as plain number first (no units)
    try:
        return int(float(remaining))
    except ValueError:
        pass

    # Handle pure milliseconds format: "290.979975ms"
    # MUST check this BEFORE checking 'm' for minutes to avoid misinterpreting 'ms'
    ms_match = re.match(r"^([\d.]+)ms$", remaining)
    if ms_match:
        ms_value = float(ms_match.group(1))
        seconds = ms_value / 1000.0
        # Round up to at least 1 second to avoid immediate retry floods
        return max(1, int(seconds)) if seconds > 0 else 0

    # Parse hours component
    hour_match = re.match(r"(\d+)h", remaining)
    if hour_match:
        total_seconds += int(hour_match.group(1)) * 3600
        remaining = remaining[hour_match.end() :]

    # Parse minutes component - use negative lookahead to avoid matching 'ms'
    min_match = re.match(r"(\d+)m(?!s)", remaining)
    if min_match:
        total_seconds += int(min_match.group(1)) * 60
        remaining = remaining[min_match.end() :]

    # Parse seconds component (including decimals like 36.752463453s)
    sec_match = re.match(r"([\d.]+)s", remaining)
    if sec_match:
        total_seconds += float(sec_match.group(1))

    # For sub-second values, round up to at least 1
    if total_seconds > 0:
        return max(1, int(total_seconds))
    return None


def extract_retry_after_from_body(error_body: Optional[str]) -> Optional[int]:
    """
    Extract the retry-after time from an API error response body.

    Handles various error formats including:
    - Gemini CLI: "Your quota will reset after 39s."
    - Antigravity: "quota will reset after 156h14m36s"
    - Generic: "quota will reset after 120s", "retry after 60s"

    Args:
        error_body: The raw error response body

    Returns:
        The retry time in seconds, or None if not found
    """
    if not error_body:
        return None

    # Pattern to match various "reset after" formats - capture the full duration string
    patterns = [
        r"quota will reset after\s*([\dhmso.]+)",  # Matches compound: 156h14m36s or 120s
        r"reset after\s*([\dhmso.]+)",
        r"retry after\s*([\dhmso.]+)",
        r"try again in\s*(\d+)\s*seconds?",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_body, re.IGNORECASE)
        if match:
            duration_str = match.group(1)
            result = _parse_duration_string(duration_str)
            if result is not None:
                return result

    return None


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
        self.message = message or f"Input tokens exceed context window for model {model}"
        super().__init__(self.message)


# =============================================================================
# ERROR TRACKING FOR CLIENT REPORTING
# =============================================================================

# Abnormal errors that require attention and should always be reported to client
ABNORMAL_ERROR_TYPES = frozenset(
    {
        "forbidden",  # 403 - credential access issue
        "authentication",  # 401 - credential invalid/revoked
        "pre_request_callback_error",  # Internal proxy error
    }
)

# Normal/expected errors during operation - only report if ALL credentials fail
NORMAL_ERROR_TYPES = frozenset(
    {
        "rate_limit",  # 429 - expected during high load
        "ip_rate_limit",  # 429 - IP-based rate limit (affects all credentials)
        "quota_exceeded",  # Expected when quota runs out
        "server_error",  # 5xx - transient provider issues
        "api_connection",  # Network issues - transient
    }
)


def is_abnormal_error(classified_error: "ClassifiedError") -> bool:
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
        self, credential: str, classified_error: "ClassifiedError", error_message: str
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
        first_line = message.split("\n")[0]
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


class ClassifiedError:
    """A structured representation of a classified error."""

    def __init__(
        self,
        error_type: str,
        original_exception: Optional[Exception] = None,
        status_code: Optional[int] = None,
        retry_after: Optional[int] = None,
        quota_reset_timestamp: Optional[float] = None,
        throttle_assessment: Optional[ThrottleAssessment] = None,
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
        parts.append(f"original_exc={self.original_exception}")
        return f"ClassifiedError({', '.join(parts)})"


class AllProviders:
    """
    Handles provider-specific settings and custom API bases.
    Supports custom OpenAI-compatible providers via PROVIDERNAME_API_BASE env vars.

    Usage:
        export KILOCODE_API_BASE=https://kilo.ai/api/openrouter
        # Then model "kilocode/z-ai/glm-5:free" will use this API base

    Known providers are skipped (they have native LiteLLM support):
        openai, anthropic, google, gemini, nvidia, mistral, cohere, groq, openrouter
    """

    KNOWN_PROVIDERS = frozenset({
        "openai", "anthropic", "google", "gemini", "nvidia",
        "mistral", "cohere", "groq", "openrouter"
    })

    def __init__(self):
        self.providers: Dict[str, Dict[str, Any]] = {}
        self._load_custom_providers()

    def _load_custom_providers(self) -> None:
        """Load custom providers from PROVIDERNAME_API_BASE env vars."""
        for env_var, value in os.environ.items():
            if env_var.endswith("_API_BASE") and value:
                provider = env_var[:-9].lower()  # Remove "_API_BASE"
                if provider not in self.KNOWN_PROVIDERS:
                    self.providers[provider] = {
                        "api_base": value.rstrip("/"),
                        "model_prefix": None,  # No prefix transformation
                    }
                    lib_logger.info(
                        f"AllProviders: registered custom provider '{provider}' "
                        f"with api_base={value.rstrip('/')}"
                    )

    def get_provider_kwargs(self, **kwargs) -> Dict[str, Any]:
        """
        Inject provider-specific settings into kwargs.

        Called before LiteLLM request to override api_base for custom providers.
        """
        model = kwargs.get("model", "")
        if "/" in model:
            provider = model.split("/")[0]
            settings = self.providers.get(provider, {})
            if "api_base" in settings:
                kwargs["api_base"] = settings["api_base"]
                lib_logger.debug(
                    f"AllProviders: using custom api_base={settings['api_base']} "
                    f"for provider={provider}"
                )
        return kwargs

    def is_custom_provider(self, model: str) -> bool:
        """Check if model uses a custom provider."""
        if "/" in model:
            provider = model.split("/")[0]
            return provider in self.providers
        return False


# Singleton instance
_all_providers_instance: Optional["AllProviders"] = None


def get_all_providers() -> "AllProviders":
    """Get the global AllProviders instance."""
    global _all_providers_instance
    if _all_providers_instance is None:
        _all_providers_instance = AllProviders()
    return _all_providers_instance


def _extract_retry_from_json_body(json_text: str) -> Optional[int]:
    """
    Extract retry delay from a JSON error response body.

    Handles Antigravity/Google API error formats with details array containing:
    - RetryInfo with retryDelay: "562476.752463453s"
    - ErrorInfo metadata with quotaResetDelay: "156h14m36.752463453s"

    Args:
        json_text: JSON string (original case, not lowercased)

    Returns:
        Retry delay in seconds, or None if not found
    """
    try:
        # Find JSON object in the text
        json_match = re.search(r"(\{.*\})", json_text, re.DOTALL)
        if not json_match:
            return None

        error_json = json.loads(json_match.group(1))
        details = error_json.get("error", {}).get("details", [])

        # Iterate through ALL details items (not just index 0)
        for detail in details:
            detail_type = detail.get("@type", "")

            # Check RetryInfo for retryDelay (most authoritative)
            # Note: Case-sensitive key names as returned by API
            if "google.rpc.RetryInfo" in detail_type:
                delay_str = detail.get("retryDelay")
                if delay_str:
                    # Handle both {"seconds": "123"} format and "123.456s" string format
                    if isinstance(delay_str, dict):
                        seconds = delay_str.get("seconds")
                        if seconds:
                            return int(float(seconds))
                    elif isinstance(delay_str, str):
                        result = _parse_duration_string(delay_str)
                        if result is not None:
                            return result

            # Check ErrorInfo metadata for quotaResetDelay (Antigravity-specific)
            if "google.rpc.ErrorInfo" in detail_type:
                metadata = detail.get("metadata", {})
                # Try both camelCase and lowercase variants
                quota_reset_delay = metadata.get("quotaResetDelay") or metadata.get(
                    "quotaresetdelay"
                )
                if quota_reset_delay:
                    result = _parse_duration_string(quota_reset_delay)
                    if result is not None:
                        return result

    except (json.JSONDecodeError, IndexError, KeyError, TypeError):
        pass

    return None


def _extract_quota_details(json_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract quotaValue and quotaId from Google/Gemini API errors.

    Google API errors structure:
    {
        "error": {
            "details": [{
                "violations": [{
                    "quotaValue": "60",
                    "quotaId": "GenerateRequestsPerMinutePerProjectPerRegion"
                }]
            }]
        }
    }
    """
    try:
        json_match = re.search(r"(\{.*\})", json_text, re.DOTALL)
        if not json_match:
            return None, None

        error_json = json.loads(json_match.group(1))
        details = error_json.get("error", {}).get("details", [])

        for detail in details:
            violations = detail.get("violations", [])
            for violation in violations:
                quota_value = violation.get("quotaValue")
                quota_id = violation.get("quotaId")
                if quota_value or quota_id:
                    return str(quota_value) if quota_value else None, quota_id
    except Exception:
        pass
    return None, None


def get_retry_after(error: Exception) -> Optional[int]:
    """
    Extracts the 'retry-after' duration in seconds from an exception message.
    Handles both integer and string representations of the duration, as well as JSON bodies.
    Also checks HTTP response headers for httpx.HTTPStatusError instances.

    Supports Antigravity/Google API error formats:
    - RetryInfo with retryDelay: "562476.752463453s"
    - ErrorInfo metadata with quotaResetDelay: "156h14m36.752463453s"
    - Human-readable message: "quota will reset after 156h14m36s"
    """
    # 0. For httpx errors, check response body and headers
    if isinstance(error, httpx.HTTPStatusError):
        # First, try to parse the response body JSON (contains retryDelay/quotaResetDelay)
        # This is where Antigravity puts the retry information
        try:
            response_text = error.response.text
            if response_text:
                result = _extract_retry_from_json_body(response_text)
                if result is not None:
                    return result
        except Exception:
            pass  # Response body may not be available

        # Fallback to HTTP headers
        headers = error.response.headers
        # Check standard Retry-After header (case-insensitive)
        retry_header = headers.get("retry-after") or headers.get("Retry-After")
        if retry_header:
            try:
                return int(retry_header)  # Assumes seconds format
            except ValueError:
                pass  # Might be HTTP date format, skip for now

        # Check X-RateLimit-Reset header (Unix timestamp)
        reset_header = headers.get("x-ratelimit-reset") or headers.get(
            "X-RateLimit-Reset"
        )
        if reset_header:
            try:
                import time

                reset_timestamp = int(reset_header)
                current_time = int(time.time())
                wait_seconds = reset_timestamp - current_time
                if wait_seconds > 0:
                    return wait_seconds
            except (ValueError, TypeError):
                pass

    # 1. Try to parse JSON from the error string representation
    # Some exceptions embed JSON in their string representation
    error_str = str(error)
    result = _extract_retry_from_json_body(error_str)
    if result is not None:
        return result

    # 2. Common regex patterns for 'retry-after' (with compound duration support)
    # Use lowercase for pattern matching
    error_str_lower = error_str.lower()
    patterns = [
        r"retry[-_\s]after:?\s*(\d+)",  # Matches: retry-after, retry_after, retry after
        r"retry in\s*(\d+)\s*seconds?",
        r"wait for\s*(\d+)\s*seconds?",
        r'"retrydelay":\s*"([\d.]+)s?"',  # retryDelay in JSON (lowercased)
        r"x-ratelimit-reset:?\s*(\d+)",
        # Compound duration patterns (Antigravity format)
        r"quota will reset after\s*([\dhms.]+)",  # e.g., "156h14m36s" or "120s"
        r"reset after\s*([\dhms.]+)",
        r'"quotaresetdelay":\s*"([\dhms.]+)"',  # quotaResetDelay in JSON (lowercased)
    ]

    for pattern in patterns:
        match = re.search(pattern, error_str_lower)
        if match:
            duration_str = match.group(1)
            # Try parsing as compound duration first
            result = _parse_duration_string(duration_str)
            if result is not None:
                return result
            # Fallback to simple integer
            try:
                return int(duration_str)
            except (ValueError, IndexError):
                continue

    # 3. Handle cases where the error object itself has the attribute
    if hasattr(error, "retry_after"):
        value = getattr(error, "retry_after")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            result = _parse_duration_string(value)
            if result is not None:
                return result

    return None


# SSE Stream Error Patterns
STREAM_ABORT_INDICATORS = frozenset({
    "finish_reason",  # When value is "error"
    "native_finish_reason",  # When value is "abort"
    "stream error",
    "stream aborted",
    "connection reset",
    "mid-stream error",
})


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

    finish_reason = raw_response.get('finish_reason')
    native_reason = raw_response.get('native_finish_reason')

    if finish_reason == 'error':
        return True
    if native_reason == 'abort':
        return True

    # Check for empty content with error
    choices = raw_response.get('choices', [])
    if choices:
        for choice in choices:
            if choice.get('finish_reason') == 'error':
                return True
            message = choice.get('message', {})
            delta = choice.get('delta', {})
            # Empty content with error indication
            if not message.get('content') and not delta.get('content'):
                if choice.get('finish_reason') == 'error':
                    return True

    return False


def classify_stream_error(raw_response: Dict) -> "ClassifiedError":
    """
    Classify streaming errors from provider response.

    Creates ClassifiedError appropriate for retry logic.
    """
    if is_provider_abort(raw_response):
        return ClassifiedError(
            error_type="api_connection",  # Treat as transient for retry
            status_code=503,
            original_exception=None,
            retry_after=2,  # Short retry delay
        )

    # Default to server_error for unknown stream issues
    return ClassifiedError(
        error_type="server_error",
        status_code=500,
        original_exception=None,
        retry_after=5,
    )


# =============================================================================
# Provider-Specific Backoff Configuration
# =============================================================================
# Allows tuning retry behavior per provider for better resilience.

PROVIDER_BACKOFF_CONFIGS: Dict[str, Dict[str, float]] = {
    "kilocode": {
        "server_error_base": 1.0,  # Faster retry for kilocode 500s
        "connection_base": 0.5,
        "max_backoff": 30.0,
    },
    "friendli": {  # z-ai uses Friendli backend
        "server_error_base": 1.5,
        "connection_base": 0.5,
        "max_backoff": 20.0,
    },
}


def _get_provider_backoff_config(provider: Optional[str]) -> Dict[str, float]:
    """
    Get backoff config for a provider, with env var overrides.

    Env vars:
        KILOCODE_BACKOFF_BASE - base multiplier for server errors
        KILOCODE_MAX_BACKOFF - maximum backoff in seconds

    Returns:
        Dict with server_error_base, connection_base, max_backoff
    """
    if not provider:
        return {}

    config = PROVIDER_BACKOFF_CONFIGS.get(provider, {}).copy()

    # Env var overrides for kilocode
    if provider == "kilocode":
        if "KILOCODE_BACKOFF_BASE" in os.environ:
            try:
                config["server_error_base"] = float(os.environ["KILOCODE_BACKOFF_BASE"])
            except ValueError:
                pass
        if "KILOCODE_MAX_BACKOFF" in os.environ:
            try:
                config["max_backoff"] = float(os.environ["KILOCODE_MAX_BACKOFF"])
            except ValueError:
                pass

    return config


def get_retry_backoff(
    classified_error: "ClassifiedError",
    attempt: int,
    provider: Optional[str] = None
) -> float:
    """
    Calculate retry backoff time based on error type and attempt number.

    Different strategies for different error types:
    - api_connection: More aggressive retry (network issues are transient)
    - server_error: Standard exponential backoff
    - rate_limit: Use retry_after if available, otherwise shorter default
    - ip_rate_limit: Use retry_after (from detection) or default cooldown

    Args:
        classified_error: The classified error with type and retry_after
        attempt: Current retry attempt number (0-indexed)
        provider: Optional provider name for provider-specific tuning

    Returns:
        Backoff time in seconds
    """
    import random

    # If provider specified retry_after, use it
    if classified_error.retry_after:
        return classified_error.retry_after

    error_type = classified_error.error_type

    # Provider-specific config
    config = _get_provider_backoff_config(provider)
    max_backoff = config.get("max_backoff", 60.0)

    if error_type == "api_connection":
        # More aggressive retry for network errors - they're usually transient
        # 0.5s, 0.75s, 1.1s, 1.7s, 2.5s...
        base = config.get("connection_base", 0.5)
        backoff = base * (1.5 ** attempt) + random.uniform(0, 0.5)
    elif error_type == "server_error":
        # Standard exponential backoff with provider-specific base
        # Default: 1s, 2s, 4s, 8s... (base=2)
        # Kilocode: 1s, 1s, 1s, 1s... (base=1.0, slower growth)
        base = config.get("server_error_base", 2.0)
        backoff = (base ** attempt) + random.uniform(0, 1)
    elif error_type == "rate_limit":
        # Short default for transient rate limits without retry_after
        backoff = 5 + random.uniform(0, 2)
    elif error_type == "ip_rate_limit":
        # IP throttle - use default cooldown with jitter
        backoff = RATE_LIMIT_DEFAULT_COOLDOWN + random.uniform(0, 2)
    else:
        # Default backoff
        backoff = (2 ** attempt) + random.uniform(0, 1)

    return min(backoff, max_backoff)


# =============================================================================
# Unified 429 Error Handler
# =============================================================================

from dataclasses import dataclass, field as dataclass_field
from enum import Enum


class ThrottleActionType(Enum):
    """Actions to take after processing a 429 error."""
    CREDENTIAL_COOLDOWN = "credential_cooldown"  # Single credential throttled
    PROVIDER_COOLDOWN = "provider_cooldown"      # IP-level throttle detected
    FAIL_IMMEDIATELY = "fail_immediately"        # Non-recoverable (should not happen for 429)


@dataclass
class ThrottleAction:
    """
    Result of processing a 429 error with unified handling.

    This dataclass consolidates all decisions about what to do after a 429:
    - What action to take (credential vs provider cooldown)
    - How long to wait
    - Whether to open the circuit breaker
    - Related metadata for logging/debugging
    """
    action_type: ThrottleActionType
    cooldown_seconds: int = 0
    open_circuit_breaker: bool = False
    throttle_scope: ThrottleScope = ThrottleScope.CREDENTIAL
    confidence: float = 0.0
    affected_credentials: list = dataclass_field(default_factory=list)
    reason: str = ""

    def __str__(self) -> str:
        return (
            f"ThrottleAction(action={self.action_type.value}, "
            f"cooldown={self.cooldown_seconds}s, "
            f"circuit_breaker={self.open_circuit_breaker}, "
            f"scope={self.throttle_scope.value})"
        )


async def handle_429_error(
    provider: str,
    credential: str,
    error: Exception,
    error_body: Optional[str] = None,
    retry_after: Optional[int] = None,
    ip_throttle_detector: Optional["IPThrottleDetector"] = None,
    circuit_breaker: Optional["ProviderCircuitBreaker"] = None,
    cooldown_manager: Optional["CooldownManager"] = None,
) -> ThrottleAction:
    """
    Unified handler for 429 rate limit errors.

    This function consolidates all 429 processing logic:
    1. Detects IP-level vs credential-level throttle
    2. Determines appropriate cooldown duration
    3. Decides whether to open circuit breaker
    4. Returns a ThrottleAction with all decisions

    If circuit_breaker and cooldown_manager are provided, actions are applied
    automatically. Otherwise, the caller must apply them based on the returned
    ThrottleAction.

    This replaces the duplicated logic across client.py with ~22 calls
    to open_immediately() for provider-level throttling.

    Args:
        provider: Provider name (e.g., "openai", "anthropic")
        credential: Credential identifier (for correlation and cooldown)
        error: The original exception
        error_body: Optional error response body for pattern analysis
        retry_after: Optional retry-after value from headers
        ip_throttle_detector: Optional IP throttle detector instance
                             (uses global singleton if not provided)
        circuit_breaker: Optional circuit breaker for provider-level cooldown
        cooldown_manager: Optional cooldown manager for credential-level cooldown

    Returns:
        ThrottleAction with action type, cooldown, and circuit breaker decision

    Usage:
        # With automatic action application:
        action = await handle_429_error(
            provider="openai",
            credential="sk-xxx",
            error=exc,
            error_body=response_text,
            retry_after=60,
            circuit_breaker=self.circuit_breaker,
            cooldown_manager=self.cooldown_manager,
        )
        # Actions are already applied - just check result
        if action.action_type == ThrottleActionType.PROVIDER_COOLDOWN:
            # Provider blocked, stop rotation
            pass

        # Without automatic application (manual):
        action = await handle_429_error(...)
        if action.action_type == ThrottleActionType.PROVIDER_COOLDOWN:
            await circuit_breaker.open_immediately(
                provider, reason=action.reason, duration=action.cooldown_seconds
            )
        elif action.action_type == ThrottleActionType.CREDENTIAL_COOLDOWN:
            await cooldown_manager.start_cooldown(credential, action.cooldown_seconds)
    """
    # Get or create detector
    if ip_throttle_detector is None:
        ip_throttle_detector = get_ip_throttle_detector()

    # Step 1: Check for explicit IP throttle indicators in error body
    ip_throttle_from_body = _detect_ip_throttle(error_body, provider=provider)

    if ip_throttle_from_body is not None:
        # Error body explicitly indicates IP-level throttle
        cooldown = retry_after or ip_throttle_from_body
        lib_logger.warning(
            f"IP-level throttle detected for provider '{provider}' from error body. "
            f"Blocking provider for {cooldown}s."
        )
        action = ThrottleAction(
            action_type=ThrottleActionType.PROVIDER_COOLDOWN,
            cooldown_seconds=cooldown,
            open_circuit_breaker=True,
            throttle_scope=ThrottleScope.IP,
            confidence=1.0,  # High confidence from explicit error body
            reason="IP-level throttle detected from error body",
        )
        # Auto-apply if managers provided
        if circuit_breaker is not None:
            await circuit_breaker.open_immediately(
                provider, reason=action.reason, duration=action.cooldown_seconds
            )
        return action

    # Step 2: Record 429 and correlate with other credentials
    assessment = ip_throttle_detector.record_429(
        provider=provider,
        credential=mask_credential(credential),
        error_body=error_body,
        retry_after=retry_after,
    )

    # Step 3: Determine action based on assessment scope
    cooldown = max(retry_after or 0, assessment.suggested_cooldown)
    if cooldown == 0:
        cooldown = RATE_LIMIT_DEFAULT_COOLDOWN

    if assessment.scope == ThrottleScope.IP:
        # Multiple credentials throttled - IP-level
        lib_logger.warning(
            f"IP-level throttle detected for provider '{provider}' via correlation: "
            f"{len(assessment.affected_credentials)} credentials affected, "
            f"confidence={assessment.confidence:.2f}. "
            f"Blocking provider for {cooldown}s."
        )
        action = ThrottleAction(
            action_type=ThrottleActionType.PROVIDER_COOLDOWN,
            cooldown_seconds=cooldown,
            open_circuit_breaker=True,
            throttle_scope=ThrottleScope.IP,
            confidence=assessment.confidence,
            affected_credentials=assessment.affected_credentials,
            reason="IP-level throttle detected via correlation",
        )
        # Auto-apply if managers provided
        if circuit_breaker is not None:
            await circuit_breaker.open_immediately(
                provider, reason=action.reason, duration=action.cooldown_seconds
            )
        return action

    # Step 4: Single credential throttle
    lib_logger.debug(
        f"Credential-level throttle for {mask_credential(credential)} "
        f"on provider '{provider}'. Cooldown: {cooldown}s."
    )
    action = ThrottleAction(
        action_type=ThrottleActionType.CREDENTIAL_COOLDOWN,
        cooldown_seconds=cooldown,
        open_circuit_breaker=False,
        throttle_scope=ThrottleScope.CREDENTIAL,
        confidence=assessment.confidence,
        reason="Credential-level rate limit",
    )
    # Auto-apply if managers provided
    if cooldown_manager is not None:
        await cooldown_manager.start_cooldown(credential, action.cooldown_seconds)
    return action


def classify_error(e: Exception, provider: Optional[str] = None) -> ClassifiedError:
    """
    Classifies an exception into a structured ClassifiedError object.
    Now handles both litellm and httpx exceptions.

    If provider is specified and has a parse_quota_error() method,
    attempts provider-specific error parsing first before falling back
    to generic classification.

    Error types and their typical handling:
    - rate_limit (429): Rotate key, may retry with backoff
    - server_error (5xx): Retry with backoff, then rotate
    - forbidden (403): Rotate key immediately (access denied for this credential)
    - authentication (401): Rotate key, trigger re-auth if OAuth
    - quota_exceeded: Rotate key (credential quota exhausted)
    - invalid_request (400): Don't retry - client error in request
    - context_window_exceeded: Don't retry - request too large
    - api_connection: Retry with backoff, then rotate
    - unknown: Rotate key (safer to try another)

    Args:
        e: The exception to classify
        provider: Optional provider name for provider-specific error parsing

    Returns:
        ClassifiedError with error_type, status_code, retry_after, etc.
    """
    # Try provider-specific parsing first for 429/rate limit errors
    if provider:
        try:
            from .providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider)

            if provider_class and hasattr(provider_class, "parse_quota_error"):
                # Get error body if available
                error_body = None
                if hasattr(e, "response") and hasattr(e.response, "text"):
                    try:
                        error_body = e.response.text
                    except Exception:
                        pass
                elif hasattr(e, "body"):
                    error_body = str(e.body)

                quota_info = provider_class.parse_quota_error(e, error_body)

                if quota_info and quota_info.get("retry_after"):
                    retry_after = quota_info["retry_after"]
                    reason = quota_info.get("reason", "QUOTA_EXHAUSTED")
                    reset_ts = quota_info.get("reset_timestamp")
                    quota_reset_timestamp = quota_info.get("quota_reset_timestamp")

                    # Log the parsed result with human-readable duration
                    hours = retry_after / 3600
                    lib_logger.info(
                        f"Provider '{provider}' parsed quota error: "
                        f"retry_after={retry_after}s ({hours:.1f}h), reason={reason}"
                        + (f", resets at {reset_ts}" if reset_ts else "")
                    )

                    return ClassifiedError(
                        error_type="quota_exceeded",
                        original_exception=e,
                        status_code=429,
                        retry_after=retry_after,
                        quota_reset_timestamp=quota_reset_timestamp,
                    )
        except Exception as parse_error:
            lib_logger.debug(
                f"Provider-specific error parsing failed for '{provider}': {parse_error}"
            )
            # Fall through to generic classification

    # Check for provider abort from streaming (finish_reason='error' or native_finish_reason='abort')
    # This handles StreamedAPIError.data which is a dict
    if isinstance(e, dict):
        if is_provider_abort(e):
            lib_logger.warning(
                f"Provider abort detected in stream: finish_reason={e.get('finish_reason')}, "
                f"native_finish_reason={e.get('native_finish_reason')}"
            )
            return classify_stream_error(e)
        # Also check for nested error dict
        if "error" in e and isinstance(e.get("error"), dict):
            error_obj = e.get("error", {})
            if is_provider_abort(error_obj):
                return classify_stream_error(error_obj)

    # Generic classification logic
    status_code = getattr(e, "status_code", None)

    if isinstance(e, httpx.HTTPStatusError):  # [NEW] Handle httpx errors first
        status_code = e.response.status_code

        # Try to get error body for better classification
        try:
            error_body = e.response.text.lower() if hasattr(e.response, "text") else ""
        except Exception:
            error_body = ""

        if status_code == 401:
            return ClassifiedError(
                error_type="authentication",
                original_exception=e,
                status_code=status_code,
            )
        if status_code == 403:
            # 403 Forbidden - credential doesn't have access, should rotate
            # Could be: IP restriction, account disabled, permission denied, etc.
            return ClassifiedError(
                error_type="forbidden",
                original_exception=e,
                status_code=status_code,
            )
        if status_code == 429:
            retry_after = get_retry_after(e)
            # Check if this is a quota error vs rate limit
            if "quota" in error_body or "resource_exhausted" in error_body:
                return ClassifiedError(
                    error_type="quota_exceeded",
                    original_exception=e,
                    status_code=status_code,
                    retry_after=retry_after,
                )
            # Check for IP-based rate limiting (affects all credentials)
            ip_throttle_cooldown = _detect_ip_throttle(error_body, provider=provider)
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
        if status_code == 400:
            # Check for context window / token limit errors with more specific patterns
            if any(
                pattern in error_body
                for pattern in [
                    "context_length",
                    "max_tokens",
                    "token limit",
                    "context window",
                    "too many tokens",
                    "too long",
                ]
            ):
                return ClassifiedError(
                    error_type="context_window_exceeded",
                    original_exception=e,
                    status_code=status_code,
                )

            # Provider-side transient 400s (from upstream wrappers) should rotate.
            # Keep strict fail-fast behavior for explicit policy/safety violations.
            if any(
                pattern in error_body
                for pattern in [
                    "policy",
                    "safety",
                    "content blocked",
                    "prompt blocked",
                ]
            ):
                return ClassifiedError(
                    error_type="invalid_request",
                    original_exception=e,
                    status_code=status_code,
                )

            if any(
                pattern in error_body
                for pattern in [
                    "provider returned error",
                    "upstream error",
                    "upstream temporarily unavailable",
                    "upstream service unavailable",
                ]
            ):
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
        if 400 <= status_code < 500:
            # Other 4xx errors - generally client errors
            return ClassifiedError(
                error_type="invalid_request",
                original_exception=e,
                status_code=status_code,
            )
        if 500 <= status_code:
            return ClassifiedError(
                error_type="server_error", original_exception=e, status_code=status_code
            )

    if isinstance(
        e, (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError)
    ):  # [NEW]
        return ClassifiedError(
            error_type="api_connection", original_exception=e, status_code=status_code
        )

    if isinstance(e, PreRequestCallbackError):
        return ClassifiedError(
            error_type="pre_request_callback_error",
            original_exception=e,
            status_code=400,  # Treat as a bad request
        )

    if isinstance(e, CredentialNeedsReauthError):
        # This is a rotatable error - credential is broken but re-auth is queued
        return ClassifiedError(
            error_type="credential_reauth_needed",
            original_exception=e,
            status_code=401,  # Treat as auth error for reporting purposes
        )

    if isinstance(e, EmptyResponseError):
        # Transient server-side issue - provider returned empty response
        # This is rotatable - try next credential
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=503,
        )

    if isinstance(e, TransientQuotaError):
        # Transient 429 without retry info - provider returned bare rate limit
        # This is rotatable - try next credential
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=503,
        )

    if isinstance(e, RateLimitError):
        retry_after = get_retry_after(e)
        # Check if this is a quota error vs rate limit
        error_msg = str(e).lower()
        if "quota" in error_msg or "resource_exhausted" in error_msg:
            return ClassifiedError(
                error_type="quota_exceeded",
                original_exception=e,
                status_code=status_code or 429,
                retry_after=retry_after,
            )
        # Check for IP-based rate limiting (affects all credentials)
        ip_throttle_cooldown = _detect_ip_throttle(error_msg, provider=provider)
        if ip_throttle_cooldown is not None:
            return ClassifiedError(
                error_type="ip_rate_limit",
                original_exception=e,
                status_code=status_code or 429,
                retry_after=retry_after or ip_throttle_cooldown,
            )
        return ClassifiedError(
            error_type="rate_limit",
            original_exception=e,
            status_code=status_code or 429,
            retry_after=retry_after,
        )

    if isinstance(e, (AuthenticationError,)):
        return ClassifiedError(
            error_type="authentication",
            original_exception=e,
            status_code=status_code or 401,
        )

    if isinstance(e, (InvalidRequestError, BadRequestError)):
        error_msg = str(e).lower()
        if any(
            pattern in error_msg
            for pattern in [
                "provider returned error",
                "upstream error",
                "upstream temporarily unavailable",
                "upstream service unavailable",
            ]
        ):
            return ClassifiedError(
                error_type="server_error",
                original_exception=e,
                status_code=status_code or 503,
            )

        return ClassifiedError(
            error_type="invalid_request",
            original_exception=e,
            status_code=status_code or 400,
        )

    if isinstance(e, ContextWindowExceededError):
        return ClassifiedError(
            error_type="context_window_exceeded",
            original_exception=e,
            status_code=status_code or 400,
        )

    if isinstance(e, (APIConnectionError, Timeout)):
        return ClassifiedError(
            error_type="api_connection",
            original_exception=e,
            status_code=status_code or 503,  # Treat like a server error
        )

    if isinstance(e, (ServiceUnavailableError, InternalServerError)):
        # These are often temporary server-side issues
        # Note: OpenAIError removed - it's too broad and can catch client errors
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=status_code or 503,
        )

    # Fallback for any other unclassified errors
    return ClassifiedError(
        error_type="unknown", original_exception=e, status_code=status_code
    )


def is_rate_limit_error(e: Exception) -> bool:
    """Checks if the exception is a rate limit error."""
    return isinstance(e, RateLimitError)


def is_server_error(e: Exception) -> bool:
    """Checks if the exception is a temporary server-side error."""
    return isinstance(
        e,
        (ServiceUnavailableError, APIConnectionError, InternalServerError),
    )


def is_unrecoverable_error(e: Exception) -> bool:
    """
    Checks if the exception is a non-retriable client-side error.
    These are errors that will not resolve on their own.

    NOTE: We no longer treat BadRequestError/InvalidRequestError as unrecoverable
    because "invalid_request" can come from provider-side issues (e.g., "Provider returned error")
    and should trigger rotation rather than immediate failure.
    """
    return False  # All errors are potentially recoverable via rotation


def should_rotate_on_error(classified_error: ClassifiedError) -> bool:
    """
    Determines if an error should trigger key rotation.

    Errors that SHOULD rotate (try another key):
    - rate_limit: Current key is throttled
    - quota_exceeded: Current key/account exhausted
    - forbidden: Current credential denied access
    - authentication: Current credential invalid
    - credential_reauth_needed: Credential needs interactive re-auth (queued)
    - server_error: Provider having issues (might work with different endpoint/key)
    - api_connection: Network issues (might be transient)
    - unknown: Safer to try another key

    Errors that should NOT rotate:
    - invalid_request: Client error in request payload (won't help to retry)
    - context_window_exceeded: Request too large (won't help to retry)
    - pre_request_callback_error: Internal proxy error
    - ip_rate_limit: IP-based throttle (rotation won't help, all keys share IP)

    Returns:
        True if should rotate to next key, False if should fail immediately
    """
    non_rotatable_errors = {
        "invalid_request",
        "context_window_exceeded",
        "pre_request_callback_error",
        "ip_rate_limit",
    }
    return classified_error.error_type not in non_rotatable_errors


def should_retry_same_key(classified_error: ClassifiedError) -> bool:
    """
    Determines if an error should retry with the same key (with backoff).

    Server errors, connection issues, and IP-based rate limits should retry
    the same key, as these are often transient or affect all credentials.

    Returns:
        True if should retry same key, False if should rotate immediately
    """
    retryable_errors = {
        "server_error",
        "api_connection",
        "ip_rate_limit",
    }
    return classified_error.error_type in retryable_errors


def classify_429_with_throttle_detection(
    e: Exception,
    provider: str,
    credential: str,
    error_body: Optional[str] = None,
) -> ClassifiedError:
    """
    Classify a 429 error with IP throttle detection via correlation analysis.

    This function records the 429 event in the IP throttle detector and
    returns a ClassifiedError with throttle assessment if IP-level throttling
    is detected.

    Use this function instead of classify_error() when you have access to
    the credential identifier and want IP throttle correlation.

    Args:
        e: The exception (should be a 429 error)
        provider: Provider name (e.g., "openai", "anthropic")
        credential: Credential identifier for correlation
        error_body: Optional error response body

    Returns:
        ClassifiedError with throttle_assessment populated if IP throttle detected
    """
    retry_after = get_retry_after(e)
    detector = get_ip_throttle_detector()

    # Record the 429 and get throttle assessment
    assessment = detector.record_429(
        provider=provider,
        credential=credential,
        error_body=error_body,
        retry_after=retry_after,
    )

    # Determine error type based on assessment
    if assessment.scope == ThrottleScope.IP:
        error_type = "ip_rate_limit"
        lib_logger.warning(
            f"IP-level throttle detected for {provider}: "
            f"{len(assessment.affected_credentials)} credentials affected, "
            f"confidence={assessment.confidence:.2f}, "
            f"cooldown={assessment.suggested_cooldown}s"
        )
    else:
        # Check if it's a quota error
        error_body_lower = (error_body or "").lower()
        if "quota" in error_body_lower or "resource_exhausted" in error_body_lower:
            error_type = "quota_exceeded"
        else:
            error_type = "rate_limit"

    # Use the larger of retry_after or suggested_cooldown
    final_cooldown = max(retry_after or 0, assessment.suggested_cooldown)

    return ClassifiedError(
        error_type=error_type,
        original_exception=e,
        status_code=429,
        retry_after=final_cooldown if final_cooldown > 0 else None,
        throttle_assessment=assessment,
    )
