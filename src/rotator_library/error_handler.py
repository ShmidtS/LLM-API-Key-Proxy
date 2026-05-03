# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import re
import os
import time
import logging
from typing import TYPE_CHECKING, Optional, Dict, Tuple
import httpx

from litellm.exceptions import (  # type: ignore[import-untyped]
    APIConnectionError,
    APIError as LiteLLMAPIError,
    RateLimitError,
    ServiceUnavailableError,
    AuthenticationError,
    InvalidRequestError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    Timeout,
    ContextWindowExceededError,
)
from litellm.llms.openai.common_utils import OpenAIError  # type: ignore[import-untyped]

from .ip_throttle_detector import (
    ThrottleScope,
    IPThrottleDetector,
)
if TYPE_CHECKING:
    from .circuit_breaker import ProviderCircuitBreaker
    from .cooldown_manager import CooldownManager
from .utils.json_utils import json_loads, JSONDecodeError, extract_json_object
from .utils.duration import parse_duration
from .utils.http_retry import compute_backoff_with_jitter

from .config.defaults import COOLDOWN_RATE_LIMIT_DEFAULT, env_bool, env_float
from .error_types import (
    ClassifiedError,
    GarbageResponseError,
    PreRequestCallbackError,
    CredentialNeedsReauthError,
    EmptyResponseError,
    TransientQuotaError,
    mask_credential,
)

lib_logger = logging.getLogger("rotator_library")

# Default cooldown for rate limits without retry_after
# Uses centralized value from config/defaults.py
RATE_LIMIT_DEFAULT_COOLDOWN = COOLDOWN_RATE_LIMIT_DEFAULT  # 60 seconds

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
_PROXY_PROVIDERS_DEFAULT = frozenset(
    {
        "kilocode",  # Routes to multiple providers (minimax, moonshot, z-ai, etc.)
        "openrouter",  # Routes to 100+ providers
        "requesty",  # Router/aggregator
        "opencode",  # OpenCode AI provider with quota-based rate limits
        "inception",  # Inception Labs - 429 errors should trigger rotation, not IP throttle
        "nvidia",  # NVIDIA NIM routes to multiple backends, 429 should rotate keys
        "zai",  # ZAI uses shared hourly quota across all credentials; 429 on one key means all keys are likely rate-limited, not IP throttle
        "friendli",  # FriendliAI serverless backend; 429 should trigger key rotation, not IP-level circuit breaker
    }
)

_env_providers = os.environ.get("PROXY_PROVIDERS")
PROXY_PROVIDERS = (
    frozenset(p.strip() for p in _env_providers.split(",") if p.strip())
    if _env_providers is not None
    else _PROXY_PROVIDERS_DEFAULT
)

_RETRY_AFTER_BODY_PATTERNS = (
    re.compile(r"quota will reset after\s*([\dhmso.]+)", re.IGNORECASE),
    re.compile(r"reset after\s*([\dhmso.]+)", re.IGNORECASE),
    re.compile(r"retry after\s*([\dhmso.]+)", re.IGNORECASE),
    re.compile(r"try again in\s*(\d+)\s*seconds?", re.IGNORECASE),
)

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

_NON_ROTATABLE_ERRORS = frozenset(
    {
        "invalid_request",
        "context_window_exceeded",
        "pre_request_callback_error",
        "ip_rate_limit",
    }
)

_RETRYABLE_SAME_KEY_ERRORS = frozenset(
    {
        "server_error",
        "api_connection",
        "ip_rate_limit",
    }
)

# Pre-compiled patterns for validate_response_quality
_WORD_SPLIT_RE = re.compile(r'[\s\\/"\']+')
_CODE_FENCE_RE = re.compile(r'```[\s\S]*?```')
_CODE_PATTERN_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r'\bimport\s+\w+', r'\bfrom\s+\w+\s+import\b',
    r'\bclass\s+\w+',
    r'\bdef\s+\w+', r'\breturn\s+',
))
_PATH_PATTERN_RES = tuple(re.compile(p) for p in (
    r'[A-Z]:\\[\w\s\\]+\.\w{2,4}',
    r'/home/\w', r'/usr/\w', r'/var/\w', r'/tmp/\w',
    r'C:\\Users\\',
))


def _detect_ip_throttle(
    error_body: Optional[str], provider: Optional[str] = None
) -> Optional[int]:
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
                "Detected IP-based rate limiting: found indicator '%s'",
                indicator,
            )
            return RATE_LIMIT_DEFAULT_COOLDOWN

    # For PROXY_PROVIDERS (kilocode, openrouter), skip generic rate limit detection
    # These providers route to multiple backends, so generic rate limits may be
    # backend-specific rather than IP-specific
    if provider and provider in PROXY_PROVIDERS:
        lib_logger.debug(
            "Skipping generic IP throttle detection for proxy provider '%s' "
            "- rate limits may be backend-specific",
            provider,
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

    for pattern in _RETRY_AFTER_BODY_PATTERNS:
        match = pattern.search(error_body)
        if match:
            duration_str = match.group(1)
            result = parse_duration(duration_str)
            if result is not None:
                return result

    return None




# Pre-compiled retry-after patterns for get_retry_after() (module-level, not per-call)
_RETRY_AFTER_PATTERNS = (
    re.compile(r"retry[-_\s]after:?\s*(\d+)"),
    re.compile(r"retry in\s*(\d+)\s*seconds?"),
    re.compile(r"wait for\s*(\d+)\s*seconds?"),
    re.compile(r'"retrydelay":\s*"([\d.]+)s?"'),
    re.compile(r"x-ratelimit-reset:?\s*(\d+)"),
    re.compile(r"quota will reset after\s*([\dhms.]+)"),
    re.compile(r"reset after\s*([\dhms.]+)"),
    re.compile(r'"quotaresetdelay":\s*"([\dhms.]+)"'),
)


def _classify_rate_limit(
    e: Exception,
    error_text: str,
    status_code: int,
    retry_after: Optional[int],
    provider: Optional[str] = None,
    response_text: Optional[str] = None,
) -> "ClassifiedError":
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
                quota_value, quota_id = _extract_quota_details(response_text)
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

    ip_throttle_cooldown = _detect_ip_throttle(error_text, provider=provider)
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
        json_str = extract_json_object(json_text)
        if not json_str:
            return None

        error_json = json_loads(json_str)
        error_obj = error_json.get("error")
        if not isinstance(error_obj, dict):
            return None
        details = error_obj.get("details", [])

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
                        result = parse_duration(delay_str)
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
                    result = parse_duration(quota_reset_delay)
                    if result is not None:
                        return result

    except (JSONDecodeError, IndexError, KeyError, TypeError) as e:
        lib_logger.debug("Failed to extract retry info from response body: %s", e)

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
        json_str = extract_json_object(json_text)
        if not json_str:
            return None, None

        error_json = json_loads(json_str)
        error_obj = error_json.get("error")
        if not isinstance(error_obj, dict):
            return None, None
        details = error_obj.get("details", [])

        for detail in details:
            violations = detail.get("violations", [])
            for violation in violations:
                quota_value = violation.get("quotaValue")
                quota_id = violation.get("quotaId")
                if quota_value or quota_id:
                    return str(quota_value) if quota_value else None, quota_id
    except (KeyError, TypeError, ValueError):
        lib_logger.debug("Failed to extract quota details from error body", exc_info=True)
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
        except (httpx.HTTPError, RuntimeError, AttributeError) as exc:
            lib_logger.debug("Response body unavailable for retry-after extraction (%s: %s)", type(exc).__name__, exc)

        # Fallback to HTTP headers
        headers = error.response.headers
        # Check standard Retry-After header (case-insensitive)
        retry_header = headers.get("retry-after") or headers.get("Retry-After")
        if retry_header:
            try:
                return int(retry_header)  # Assumes seconds format
            except ValueError as e:
                lib_logger.debug("Could not parse date header: %s", e)

        # Check X-RateLimit-Reset header (Unix timestamp)
        reset_header = headers.get("x-ratelimit-reset") or headers.get(
            "X-RateLimit-Reset"
        )
        if reset_header:
            try:
                reset_timestamp = int(reset_header)
                current_time = int(time.time())
                wait_seconds = reset_timestamp - current_time
                if wait_seconds > 0:
                    return wait_seconds
            except (ValueError, TypeError):
                lib_logger.debug("Invalid retry-after header value: %s", reset_header)

    # 1. Try to parse JSON from the error string representation
    # Some exceptions embed JSON in their string representation
    error_str = str(error)
    if "{" in error_str:
        error_str_lower = error_str.lower()
        if "retry" in error_str_lower or "quota" in error_str_lower or "rate" in error_str_lower:
            result = _extract_retry_from_json_body(error_str)
            if result is not None:
                return result

    # 2. Common regex patterns for 'retry-after' (with compound duration support)
    # Use lowercase for pattern matching
    error_str_lower = error_str.lower()

    for pattern in _RETRY_AFTER_PATTERNS:
        match = pattern.search(error_str_lower)
        if match:
            duration_str = match.group(1)
            # Try parsing as compound duration first
            result = parse_duration(duration_str)
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
            result = parse_duration(value)
            if result is not None:
                return result

    return None


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

    # Check for empty content with error
    choices = raw_response.get("choices", [])
    if choices:
        for choice in choices:
            if choice.get("finish_reason") == "error":
                return True
            message = choice.get("message", {})
            delta = choice.get("delta", {})
            # Empty content with error indication
            if not message.get("content") and not delta.get("content"):
                if choice.get("native_finish_reason") == "abort":
                    return True

    return False


def classify_stream_error(raw_response: Dict) -> "ClassifiedError":
    """
    Classify streaming errors from provider response.

    Creates ClassifiedError appropriate for retry logic.
    """
    # Inception Labs specific error patterns
    raw_str = str(raw_response).lower()
    if "inception" in raw_str:
        if "server had an error" in raw_str or "the server had an error" in raw_str:
            return ClassifiedError(
                error_type="server_error",
                status_code=503,
                original_exception=None,
                retry_after=int(5.0),
            )

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
    "inception": {
        "server_error_base": 2.0,  # Longer initial backoff for 500s
        "connection_base": 1.0,
        "max_backoff": 60.0,
        "max_retries": 5,
    },
}

_BACKOFF_CONFIG_CACHE: dict[str, dict] = {}
_BACKOFF_ENV_CACHE: dict[str, Tuple[Optional[str], Optional[str]]] = {}


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

    env_values: Tuple[Optional[str], Optional[str]]
    if provider == "kilocode":
        env_values = (
            os.environ.get("KILOCODE_BACKOFF_BASE"),
            os.environ.get("KILOCODE_MAX_BACKOFF"),
        )
    elif provider == 'inception':
        env_values = (
            os.environ.get('INCEPTION_BACKOFF_BASE'),
            os.environ.get('INCEPTION_MAX_BACKOFF'),
        )
    else:
        env_values = (None, None)

    cached = _BACKOFF_CONFIG_CACHE.get(provider)
    if cached is not None and _BACKOFF_ENV_CACHE.get(provider) == env_values:
        return cached

    config = PROVIDER_BACKOFF_CONFIGS.get(provider, {}).copy()

    # Env var overrides for kilocode
    if provider == "kilocode":
        backoff_base = env_values[0]
        max_backoff = env_values[1]
        if backoff_base is not None:
            try:
                config["server_error_base"] = float(backoff_base)
            except ValueError:
                lib_logger.debug("Invalid KILOCODE_BACKOFF_BASE value")
        if max_backoff is not None:
            try:
                config["max_backoff"] = float(max_backoff)
            except ValueError:
                lib_logger.debug("Invalid KILOCODE_MAX_BACKOFF value")

    # Env var overrides for inception
    if provider == 'inception':
        backoff_base = env_values[0]
        max_backoff = env_values[1]
        if backoff_base is not None:
            try:
                config['server_error_base'] = float(backoff_base)
            except ValueError:
                lib_logger.debug("Invalid INCEPTION_BACKOFF_BASE value")
        if max_backoff is not None:
            try:
                config['max_backoff'] = float(max_backoff)
            except ValueError:
                lib_logger.debug("Invalid INCEPTION_MAX_BACKOFF value")

    _BACKOFF_CONFIG_CACHE[provider] = config
    _BACKOFF_ENV_CACHE[provider] = env_values
    return config


def get_retry_backoff(
    classified_error: "ClassifiedError", attempt: int, provider: Optional[str] = None
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
    # If provider specified retry_after, use it
    if classified_error.retry_after:
        return classified_error.retry_after

    error_type = classified_error.error_type

    # Provider-specific config
    config = _get_provider_backoff_config(provider)
    max_backoff = config.get("max_backoff", 60.0)

    if error_type == "api_connection":
        # More aggressive retry for network errors - they're usually transient
        base = config.get("connection_base", 0.5)
        backoff = compute_backoff_with_jitter(attempt, base=1.5, max_wait=max_backoff, jitter=0.3, min_wait=float(base))
    elif error_type == "server_error":
        # Standard exponential backoff with provider-specific base
        base = config.get("server_error_base", 2.0)
        backoff = compute_backoff_with_jitter(attempt, base=base, max_wait=max_backoff, jitter=0.3)
    elif error_type == "rate_limit":
        # Short default for transient rate limits without retry_after
        backoff = compute_backoff_with_jitter(attempt, base=2.0, max_wait=max_backoff, jitter=0.3, retry_after=5.0)
    elif error_type == "ip_rate_limit":
        # IP throttle - use default cooldown with jitter
        backoff = compute_backoff_with_jitter(attempt, base=2.0, max_wait=max_backoff, jitter=0.3, retry_after=float(RATE_LIMIT_DEFAULT_COOLDOWN))
    else:
        # Default backoff
        backoff = compute_backoff_with_jitter(attempt, base=2.0, max_wait=max_backoff, jitter=0.3)

    return backoff


# =============================================================================
# Unified 429 Error Handler
# =============================================================================

from dataclasses import dataclass, field as dataclass_field
from enum import Enum


class ThrottleActionType(Enum):
    """Actions to take after processing a 429 error."""

    CREDENTIAL_COOLDOWN = "credential_cooldown"  # Single credential throttled
    PROVIDER_COOLDOWN = "provider_cooldown"  # IP-level throttle detected
    FAIL_IMMEDIATELY = "fail_immediately"  # Non-recoverable (should not happen for 429)


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
    detector = ip_throttle_detector if ip_throttle_detector is not None else IPThrottleDetector()

    # Step 1: Check for explicit IP throttle indicators in error body
    ip_throttle_from_body = _detect_ip_throttle(error_body, provider=provider)

    if ip_throttle_from_body is not None:
        # Error body explicitly indicates IP-level throttle
        cooldown = retry_after or ip_throttle_from_body
        lib_logger.warning(
            "IP-level throttle detected for provider '%s' from error body. "
            "Blocking provider for %ss.",
            provider, cooldown,
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
        # Apply per-credential cooldowns to all affected credentials
        # so they are properly tracked when the circuit recovers
        if cooldown_manager is not None and action.affected_credentials:
            for affected_cred in action.affected_credentials:
                await cooldown_manager.start_cooldown(affected_cred, action.cooldown_seconds)
        return action

    # Step 2: For PROXY_PROVIDERS, skip IP throttle correlation entirely.
    # These providers aggregate multiple backends or use shared quotas;
    # 429 on multiple keys does NOT mean IP-level throttle.
    if provider and provider in PROXY_PROVIDERS:
        cooldown = retry_after or RATE_LIMIT_DEFAULT_COOLDOWN
        lib_logger.debug(
            "Skipping IP throttle correlation for proxy provider '%s'. "
            "Credential-level cooldown: %ss.",
            provider, cooldown,
        )
        action = ThrottleAction(
            action_type=ThrottleActionType.CREDENTIAL_COOLDOWN,
            cooldown_seconds=cooldown,
            open_circuit_breaker=False,
            throttle_scope=ThrottleScope.CREDENTIAL,
            confidence=0.8,
            reason="Credential-level rate limit (proxy provider, IP correlation skipped)",
        )
        if cooldown_manager is not None:
            await cooldown_manager.start_cooldown(credential, action.cooldown_seconds)
        return action

    # Step 3: Record 429 and correlate with other credentials
    assessment = await detector.record_429(
        provider=provider,
        credential=mask_credential(credential),
        error_body=error_body,
        retry_after=retry_after,
    )

    # Step 4: Determine action based on assessment scope
    cooldown = max(retry_after or 0, assessment.suggested_cooldown)
    if cooldown == 0:
        cooldown = RATE_LIMIT_DEFAULT_COOLDOWN

    if assessment.scope == ThrottleScope.IP:
        # Multiple credentials throttled - IP-level
        lib_logger.warning(
            "IP-level throttle detected for provider '%s' via correlation: "
            "%s credentials affected, confidence=%.2f. Blocking provider for %ss.",
            provider, len(assessment.affected_credentials), assessment.confidence, cooldown,
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
        # Apply per-credential cooldowns to all affected credentials
        # so they are properly tracked when the circuit recovers
        if cooldown_manager is not None and action.affected_credentials:
            for affected_cred in action.affected_credentials:
                await cooldown_manager.start_cooldown(affected_cred, action.cooldown_seconds)
        return action

    # Step 5: Single credential throttle
    lib_logger.debug(
        "Credential-level throttle for %s on provider '%s'. Cooldown: %ss.",
        mask_credential(credential), provider, cooldown,
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


def _try_parse_provider_quota_error(
    e: Exception, provider: Optional[str], status_code: Optional[int] = None
) -> Optional["ClassifiedError"]:
    """Try provider-specific quota error parsing.

    Extracts error body from the exception, delegates to the provider's
    parse_quota_error method, and returns a ClassifiedError if a quota
    error is detected.

    Args:
        e: The exception to parse
        provider: Provider name for provider-specific parsing
        status_code: HTTP status code to use in the returned ClassifiedError

    Returns:
        ClassifiedError with quota_exceeded type, or None if no quota error found
    """
    if not provider:
        return None
    try:
        from .providers import get_provider

        provider_class = get_provider(provider)
        if not provider_class or not hasattr(provider_class, "parse_quota_error"):
            return None

        # Get error body if available
        error_body = None
        _resp = getattr(e, "response", None)
        if _resp is not None and hasattr(_resp, "text"):
            try:
                error_body = _resp.text
            except (AttributeError, OSError):
                lib_logger.debug("Could not read error response text", exc_info=True)
        else:
            _body = getattr(e, "body", None)
            if _body is not None:
                error_body = str(_body)
        # Fallback to full exception string
        if not error_body:
            error_body = str(e)

        quota_info = provider_class.parse_quota_error(e, error_body)
        if quota_info and quota_info.get("retry_after"):
            retry_after = quota_info["retry_after"]
            reason = quota_info.get("reason", "QUOTA_EXHAUSTED")
            reset_ts = quota_info.get("reset_timestamp")
            quota_reset_timestamp = quota_info.get("quota_reset_timestamp")

            # Log the parsed result with human-readable duration
            hours = retry_after / 3600
            lib_logger.info(
                "Provider '%s' parsed quota error: retry_after=%ss (%.1fh), reason=%s%s",
                provider, retry_after, hours, reason,
                ", resets at %s" % reset_ts if reset_ts else "",
            )

            return ClassifiedError(
                error_type="quota_exceeded",
                original_exception=e,
                status_code=status_code if status_code is not None else 429,
                retry_after=retry_after,
                quota_reset_timestamp=quota_reset_timestamp,
                reason=reason,
            )
    except (ValueError, KeyError, TypeError, AttributeError) as parse_error:
        lib_logger.debug(
            "Provider-specific error parsing failed for '%s': %s",
            provider, parse_error,
        )
    return None


def validate_response_quality(response, provider: str = "", model: str = ""):
    """
    Validate that a model response contains meaningful content, not garbage.

    Checks both OpenAI-format (ModelResponse/dict with choices) and
    Anthropic-format (dict with content blocks) responses.

    Raises GarbageResponseError if the response is detected as garbage.
    Returns True if the response appears valid.

    Garbage indicators:
    - High word repetition (unique/total ratio < threshold)
    - Code fragment injection (import, from, extension keywords in non-code context)
    - File path leakage (C:\\..., /home/, /usr/ in response)
    - Repetitive token flooding (same short token repeated many times)
    """
    if not env_bool("GARBAGE_DETECTION_ENABLED", True):
        return True

    repetition_threshold = env_float("GARBAGE_REPETITION_THRESHOLD", 0.25)

    # Extract text content from response
    text_parts = []

    if isinstance(response, dict):
        # Anthropic format: {"content": [{"type": "text"/"thinking", ...}]}
        if "content" in response and isinstance(response["content"], list):
            for block in response["content"]:
                if isinstance(block, dict):
                    for key in ("text", "thinking"):
                        val = block.get(key)
                        if isinstance(val, str) and len(val) > 50:
                            text_parts.append(val)
        # OpenAI format: {"choices": [{"message": {"content": "..."}}]}
        if "choices" in response and isinstance(response["choices"], list):
            for choice in response["choices"]:
                msg = choice.get("message", {})
                content = msg.get("content")
                if isinstance(content, str) and len(content) > 50:
                    text_parts.append(content)
    elif hasattr(response, "choices"):
        # litellm ModelResponse object
        for choice in response.choices:
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content
                if isinstance(content, str) and len(content) > 50:
                    text_parts.append(content)

    if not text_parts:
        return True

    full_text = " ".join(text_parts)

    # === Heuristic 1: Word repetition ratio ===
    words = _WORD_SPLIT_RE.split(full_text)
    words = [w for w in words if len(w) > 2]
    if len(words) >= 20:
        unique_words = set(w.lower() for w in words)
        ratio = len(unique_words) / len(words)
        if ratio < repetition_threshold:
            raise GarbageResponseError(
                provider=provider, model=model,
                reason=f"High repetition ratio: {ratio:.2f} < {repetition_threshold} "
                       f"(unique {len(unique_words)}/{len(words)} words)"
            )

    # === Heuristic 2: Code fragment injection ===
    # Only triggers for unusually high code-keyword density outside code fences.
    # Threshold is high (10) to avoid false positives on legitimate code-assistance responses.
    # Strip content inside markdown code fences before counting.
    text_outside_fences = _CODE_FENCE_RE.sub('', full_text)
    code_hits = sum(len(p.findall(text_outside_fences)) for p in _CODE_PATTERN_RES)
    if code_hits >= 10:
        raise GarbageResponseError(
            provider=provider, model=model,
            reason=f"Code fragment injection: {code_hits} code patterns detected outside fences"
        )

    # === Heuristic 3: File path leakage ===
    path_hits = sum(len(p.findall(full_text)) for p in _PATH_PATTERN_RES)
    if path_hits >= 2:
        raise GarbageResponseError(
            provider=provider, model=model,
            reason=f"File path leakage: {path_hits} paths detected in content"
        )

    # === Heuristic 4: Repetitive token flooding ===
    token_counts = {}
    for w in words:
        wl = w.lower()
        if 2 < len(wl) < 15:
            token_counts[wl] = token_counts.get(wl, 0) + 1
    if token_counts:
        max_count = max(token_counts.values())
        max_token = max(token_counts, key=lambda k: token_counts[k])
        if max_count >= 8 and max_count / max(len(words), 1) > 0.15:
            raise GarbageResponseError(
                provider=provider, model=model,
                reason=f"Token flooding: '{max_token}' repeated {max_count} times "
                       f"({max_count}/{len(words)} = {max_count/max(len(words),1):.1%})"
            )

    return True


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
    error_str = str(e)
    error_str_lower = error_str.lower()

    # Determine HTTP status early so we can avoid misclassifying a genuine 400
    # (body may contain quota-sounding keywords) as quota_exceeded. The provider
    # quota parser is only meaningful for 429 or when status is unknown.
    if isinstance(e, httpx.HTTPStatusError):
        early_status_code = e.response.status_code
    else:
        early_status_code = getattr(e, "status_code", None)

    # Try provider-specific parsing first for 429/rate limit errors
    if early_status_code in (None, 429):
        result = _try_parse_provider_quota_error(e, provider, status_code=429)
        if result is not None:
            return result

    # Check for provider abort from streaming (finish_reason='error' or native_finish_reason='abort')
    # This handles StreamedAPIError.data which is a dict
    if isinstance(e, dict):
        if is_provider_abort(e):
            lib_logger.warning(
                "Provider abort detected in stream: finish_reason=%s, native_finish_reason=%s",
                e.get("finish_reason"), e.get("native_finish_reason"),
            )
            return classify_stream_error(e)
        # Also check for nested error dict
        error_obj = e.get("error")
        if isinstance(error_obj, dict):
            if is_provider_abort(error_obj):
                return classify_stream_error(error_obj)

    # Generic classification logic
    status_code = getattr(e, "status_code", None)

    if isinstance(e, httpx.HTTPStatusError):  # [NEW] Handle httpx errors first
        status_code = e.response.status_code

        # Try to get error body for better classification
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
            )
        if status_code == 403:
            # Check for Cloudflare Edge IP Restricted (non-retryable provider issue)
            if "edge_ip_restricted" in error_body_lower or "error 1034" in error_body_lower or (
                "cloudflare" in error_body_lower and "owner_action_required" in error_body_lower
            ):
                return ClassifiedError(
                    error_type="ip_rate_limit",
                    original_exception=e,
                    status_code=status_code,
                )
            # 403 Forbidden - credential doesn't have access, should rotate
            return ClassifiedError(
                error_type="forbidden",
                original_exception=e,
                status_code=status_code,
            )
        if status_code == 429:
            retry_after = get_retry_after(e)
            return _classify_rate_limit(
                e,
                error_text=error_body_lower,
                status_code=status_code,
                retry_after=retry_after,
                provider=provider,
                response_text=response_text_raw,
            )
        if status_code == 400:
            # Check for context window / token limit errors with more specific patterns
            if any(pattern in error_body_lower for pattern in _CONTEXT_WINDOW_ERROR_PATTERNS):
                return ClassifiedError(
                    error_type="context_window_exceeded",
                    original_exception=e,
                    status_code=status_code,
                )

            # Provider-side transient 400s (from upstream wrappers) should rotate.
            # Keep strict fail-fast behavior for explicit policy/safety violations.
            if any(pattern in error_body_lower for pattern in _POLICY_ERROR_PATTERNS):
                return ClassifiedError(
                    error_type="invalid_request",
                    original_exception=e,
                    status_code=status_code,
                )

            if any(pattern in error_body_lower for pattern in _UPSTREAM_ERROR_PATTERNS):
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

    if isinstance(e, GarbageResponseError):
        # Garbage/hallucinated response - rotate to next credential
        return ClassifiedError(
            error_type="garbage_response",
            original_exception=e,
            status_code=503,
            reason=e.reason if hasattr(e, 'reason') else error_str,
        )

    if isinstance(e, RateLimitError):
        retry_after = get_retry_after(e)
        return _classify_rate_limit(
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
        if any(pattern in error_str_lower for pattern in _AUTHENTICATION_ERROR_PATTERNS):
            return ClassifiedError(
                error_type="authentication",
                original_exception=e,
                status_code=401,
            )

        if any(pattern in error_str_lower for pattern in _UPSTREAM_ERROR_PATTERNS):
            return ClassifiedError(
                error_type="server_error",
                original_exception=e,
                status_code=status_code or 503,
            )

        # Account/billing errors (e.g. Aliyun "Arrearage", "Access denied...account in good standing")
        # These are per-key/account issues — rotating to another credential may succeed.
        if any(pattern in error_str_lower for pattern in _ACCOUNT_BILLING_ERROR_PATTERNS):
            return ClassifiedError(
                error_type="quota_exceeded",
                original_exception=e,
                status_code=status_code or 402,
                retry_after=300,
                reason="account_billing_issue",
            )

        # Some providers (e.g. ZAI) return quota errors as 400 Bad Request
        # with messages like "Insufficient balance" or "Please recharge".
        # Re-check parse_quota_error for BadRequestError cases that the
        # earlier provider-specific pass may have missed (e.g. missing body).
        if provider:
            try:
                from .providers import get_provider
                provider_class = get_provider(provider)
                if provider_class and hasattr(provider_class, "parse_quota_error"):
                    error_body = None
                    _resp = getattr(e, "response", None)
                    if _resp is not None and hasattr(_resp, "text"):
                        try:
                            error_body = _resp.text
                        except (AttributeError, OSError):
                            lib_logger.debug("Could not read error response text for quota parse", exc_info=True)
                    else:
                        _body = getattr(e, "body", None)
                        if _body is not None:
                            error_body = str(_body)
                    # Also try the full string as body fallback
                    if not error_body:
                        error_body = error_str
                    quota_info = provider_class.parse_quota_error(e, error_body)
                    if quota_info and quota_info.get("retry_after"):
                        retry_after = quota_info["retry_after"]
                        quota_reset_timestamp = quota_info.get("quota_reset_timestamp")
                        return ClassifiedError(
                            error_type="quota_exceeded",
                            original_exception=e,
                            status_code=status_code or 400,
                            retry_after=retry_after,
                            quota_reset_timestamp=quota_reset_timestamp,
                            reason=quota_info.get("reason"),
                        )
            except (KeyError, TypeError, ValueError):
                lib_logger.debug("Provider-specific quota parse failed for '%s'", provider, exc_info=True)

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
        # Inception Labs server errors often manifest as connection errors with specific messages
        if 'server had an error' in error_str_lower or 'the server had an error' in error_str_lower:
            return ClassifiedError(
                error_type='api_connection',
                original_exception=e,
                status_code=503,
                retry_after=5,  # Inception needs ~5s to recover from transient overload
            )
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

    # litellm.llms.openai.common_utils.OpenAIError — upstream wrapper
    # Catches "Not Found", 404, etc. from OpenAI-compatible endpoints.
    # Treat 404 as invalid_request (fail-fast, no rotation); otherwise fall through.
    if isinstance(e, OpenAIError):
        if "not found" in error_str_lower or "404" in error_str_lower:
            return ClassifiedError(
                error_type="invalid_request",
                original_exception=e,
                status_code=status_code or 404,
                reason="not_found_openai_error",
            )
        # Non-404 OpenAIError — treat as server_error with rotation
        return ClassifiedError(
            error_type="server_error",
            original_exception=e,
            status_code=status_code or 500,
            reason="openai_error_unclassified",
        )

    # litellm NotFoundError — model/endpoint not found at provider (404)
    # Not a key issue; don't penalize credentials with escalating cooldown
    if isinstance(e, NotFoundError):
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

    # litellm.APIError wraps upstream errors (402 credits, etc.) without exposing httpx status
    if isinstance(e, LiteLLMAPIError):
        if any(p in error_str_lower for p in _LITELLM_API_CREDIT_PATTERNS):
            return ClassifiedError(
                error_type="quota_exceeded",
                original_exception=e,
                status_code=status_code or 402,
                retry_after=300,
                reason="litellm_api_credits",
            )
        if (
            "invalid api key" in error_str_lower
            or "invalid_api_key" in error_str_lower
            or any(pattern in error_str_lower for pattern in _AUTHENTICATION_ERROR_PATTERNS)
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

    # Fallback for any other unclassified errors
    return ClassifiedError(
        error_type="unknown", original_exception=e, status_code=status_code
    )


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
    return classified_error.error_type not in _NON_ROTATABLE_ERRORS


def should_retry_same_key(classified_error: ClassifiedError) -> bool:
    """
    Determines if an error should retry with the same key (with backoff).

    Server errors, connection issues, and IP-based rate limits should retry
    the same key, as these are often transient or affect all credentials.

    Returns:
        True if should retry same key, False if should rotate immediately
    """
    return classified_error.error_type in _RETRYABLE_SAME_KEY_ERRORS


