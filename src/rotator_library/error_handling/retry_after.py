# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
import re
import time
from typing import Optional

import httpx

from ..utils.duration import parse_duration
from ..utils.json_utils import JSONDecodeError, extract_json_object, json_loads

lib_logger = logging.getLogger("rotator_library")

_RETRY_AFTER_BODY_PATTERNS = (
    re.compile(r"quota will reset after\s*([\dhmso.]+)", re.IGNORECASE),
    re.compile(r"reset after\s*([\dhmso.]+)", re.IGNORECASE),
    re.compile(r"retry after\s*([\dhmso.]+)", re.IGNORECASE),
    re.compile(r"try again in\s*(\d+)\s*seconds?", re.IGNORECASE),
)

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
        json_str = extract_json_object(json_text)
        if not json_str:
            return None

        error_json = json_loads(json_str)
        error_obj = error_json.get("error")
        if not isinstance(error_obj, dict):
            return None
        details = error_obj.get("details", [])

        for detail in details:
            detail_type = detail.get("@type", "")

            if "google.rpc.RetryInfo" in detail_type:
                delay_str = detail.get("retryDelay")
                if delay_str:
                    if isinstance(delay_str, dict):
                        seconds = delay_str.get("seconds")
                        if seconds:
                            return int(float(seconds))
                    elif isinstance(delay_str, str):
                        result = parse_duration(delay_str)
                        if result is not None:
                            return result

            if "google.rpc.ErrorInfo" in detail_type:
                metadata = detail.get("metadata", {})
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
    if isinstance(error, httpx.HTTPStatusError):
        try:
            response_text = error.response.text
            if response_text:
                result = _extract_retry_from_json_body(response_text)
                if result is not None:
                    return result
        except (httpx.HTTPError, RuntimeError, AttributeError) as exc:
            lib_logger.debug("Response body unavailable for retry-after extraction (%s: %s)", type(exc).__name__, exc)

        headers = error.response.headers
        retry_header = headers.get("retry-after") or headers.get("Retry-After")
        if retry_header:
            try:
                return int(retry_header)
            except ValueError as e:
                lib_logger.debug("Could not parse date header: %s", e)

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

    error_str = str(error)
    if "{" in error_str:
        error_str_lower = error_str.lower()
        if "retry" in error_str_lower or "quota" in error_str_lower or "rate" in error_str_lower:
            result = _extract_retry_from_json_body(error_str)
            if result is not None:
                return result

    error_str_lower = error_str.lower()

    for pattern in _RETRY_AFTER_PATTERNS:
        match = pattern.search(error_str_lower)
        if match:
            duration_str = match.group(1)
            result = parse_duration(duration_str)
            if result is not None:
                return result
            try:
                return int(duration_str)
            except (ValueError, IndexError):
                continue

    if hasattr(error, "retry_after"):
        value = getattr(error, "retry_after")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            result = parse_duration(value)
            if result is not None:
                return result

    return None
