# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Shared duration parsing utility."""

import re
from typing import Optional

_RE_MS = re.compile(r"^([\d.]+)ms$")
_RE_PURE_SECONDS = re.compile(r"^([\d.]+)s$")
_RE_DAYS = re.compile(r"(\d+)\s*d")
_RE_HOURS = re.compile(r"(\d+)\s*h")
_RE_MINUTES = re.compile(r"(\d+)\s*m(?!s)")
_RE_SECONDS = re.compile(r"([\d.]+)\s*s$")
_RE_PLAIN_NUM = re.compile(r"^(\d+)$")

_COMPOUND_PATTERNS = [
    (_RE_DAYS, 86400),
    (_RE_HOURS, 3600),
    (_RE_MINUTES, 60),
    (_RE_SECONDS, 1),
]


def parse_duration(text: str) -> Optional[int]:
    """Parse duration strings like '2s', '156h14m36.73s', '515092.73s', '1d' to seconds.

    Handles compound formats, pure seconds with decimals, milliseconds,
    and plain numbers (assumed seconds).

    Args:
        text: Duration string to parse

    Returns:
        Total seconds as integer, or None if parsing fails
    """
    if not text:
        return None

    text = text.strip()

    # Handle pure milliseconds format: "290.979975ms"
    # MUST check before 'm' for minutes to avoid misinterpreting 'ms'
    ms_match = _RE_MS.match(text)
    if ms_match:
        ms_value = float(ms_match.group(1))
        seconds = ms_value / 1000.0
        return max(1, int(seconds)) if seconds > 0 else 0

    # Handle pure seconds format: "515092.730699158s" or "2s"
    pure_seconds_match = _RE_PURE_SECONDS.match(text)
    if pure_seconds_match:
        seconds = float(pure_seconds_match.group(1))
        return max(1, int(seconds)) if seconds > 0 else 0

    # Handle compound format: "143h4m52.730699158s"
    total_seconds = 0.0
    for pattern, multiplier in _COMPOUND_PATTERNS:
        match = pattern.search(text)
        if match:
            total_seconds += float(match.group(1)) * multiplier

    if total_seconds > 0:
        return max(1, int(total_seconds))

    # Try plain number (assume seconds)
    match = _RE_PLAIN_NUM.match(text)
    if match:
        return int(match.group(1))

    return None
