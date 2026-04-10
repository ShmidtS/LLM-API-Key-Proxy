# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Shared duration parsing utility."""

import re
from typing import Optional


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
    ms_match = re.match(r"^([\d.]+)ms$", text)
    if ms_match:
        ms_value = float(ms_match.group(1))
        seconds = ms_value / 1000.0
        return max(1, int(seconds)) if seconds > 0 else 0

    # Handle pure seconds format: "515092.730699158s" or "2s"
    pure_seconds_match = re.match(r"^([\d.]+)s$", text)
    if pure_seconds_match:
        seconds = float(pure_seconds_match.group(1))
        return max(1, int(seconds)) if seconds > 0 else 0

    # Handle compound format: "143h4m52.730699158s"
    total_seconds = 0.0
    patterns = [
        (r"(\d+)\s*d", 86400),
        (r"(\d+)\s*h", 3600),
        (r"(\d+)\s*m(?!s)", 60),
        (r"([\d.]+)\s*s$", 1),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if match:
            total_seconds += float(match.group(1)) * multiplier

    if total_seconds > 0:
        return max(1, int(total_seconds))

    # Try plain number (assume seconds)
    match = re.match(r"^(\d+)$", text)
    if match:
        return int(match.group(1))

    return None
