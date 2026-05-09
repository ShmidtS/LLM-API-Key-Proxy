# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
from typing import Optional, Tuple

from ..config.defaults import COOLDOWN_RATE_LIMIT_DEFAULT, PROXY_PROVIDERS
from ..error_types import ClassifiedError
from ..utils.json_utils import extract_json_object, json_loads

lib_logger = logging.getLogger("rotator_library")
RATE_LIMIT_DEFAULT_COOLDOWN = COOLDOWN_RATE_LIMIT_DEFAULT

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

KEY_SPECIFIC_PATTERNS = frozenset(
    {
        "api key",
        "apikey",
        "key ",
        "your key",
        "this key",
        "credential",
        "token",
        "quota",
        "resource_exhausted",
    }
)


def detect_ip_throttle(
    error_body: Optional[str], provider: Optional[str] = None
) -> Optional[int]:
    """
    Detect IP-based rate limiting from error response body.

    IP throttling affects all credentials from the same IP, so rotation
    won't help. Returns a cooldown period to wait before retrying.
    """
    if not error_body:
        return None

    error_body_lower = error_body.lower()

    for indicator in IP_THROTTLE_INDICATORS:
        if indicator in error_body_lower:
            lib_logger.info(
                "Detected IP-based rate limiting: found indicator '%s'",
                indicator,
            )
            return RATE_LIMIT_DEFAULT_COOLDOWN

    if provider and provider in PROXY_PROVIDERS:
        lib_logger.debug(
            "Skipping generic IP throttle detection for proxy provider '%s' "
            "- rate limits may be backend-specific",
            provider,
        )
        return None

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


def extract_quota_details(json_text: str) -> Tuple[Optional[str], Optional[str]]:
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


def try_parse_provider_quota_error(
    e: Exception, provider: Optional[str], status_code: Optional[int] = None
) -> Optional[ClassifiedError]:
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
        from ..providers import get_provider

        provider_class = get_provider(provider)
        if not provider_class or not hasattr(provider_class, "parse_quota_error"):
            return None

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
        if not error_body:
            error_body = str(e)

        quota_info = provider_class.parse_quota_error(e, error_body)
        if quota_info and quota_info.get("retry_after"):
            retry_after = quota_info["retry_after"]
            reason = quota_info.get("reason", "QUOTA_EXHAUSTED")
            reset_ts = quota_info.get("reset_timestamp")
            quota_reset_timestamp = quota_info.get("quota_reset_timestamp")

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
