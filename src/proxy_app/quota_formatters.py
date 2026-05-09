# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Pure formatting utilities for quota viewer display."""

import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional


def format_tokens(count: int) -> str:
    """Format token count for display (e.g., 125000 -> 125k)."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.0f}k"
    return str(count)


def format_cost(cost: Optional[float]) -> str:
    """Format cost for display."""
    if cost is None or cost == 0:
        return "-"
    if cost < 0.01:
        return f"${cost:.4f}"
    return f"${cost:.2f}"


def format_time_ago(timestamp: Optional[float]) -> str:
    """Format timestamp as relative time (e.g., '5 min ago')."""
    if not timestamp:
        return "Never"
    try:
        delta = time.time() - timestamp
        if delta < 60:
            return f"{int(delta)}s ago"
        elif delta < 3600:
            return f"{int(delta / 60)} min ago"
        elif delta < 86400:
            return f"{int(delta / 3600)}h ago"
        else:
            return f"{int(delta / 86400)}d ago"
    except (ValueError, OSError):
        return "Unknown"


def format_reset_time(iso_time: Optional[str]) -> str:
    """Format ISO time string for display."""
    if not iso_time:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
        local_dt = dt.astimezone()
        return local_dt.strftime("%b %d %H:%M")
    except (ValueError, AttributeError):
        return iso_time[:16] if iso_time else "-"


def create_progress_bar(percent: Optional[int], width: int = 10) -> str:
    """Create a text-based progress bar."""
    if percent is None:
        return "░" * width
    filled = int(percent / 100 * width)
    return "▓" * filled + "░" * (width - filled)


def is_local_host(host: str) -> bool:
    """Check if host is a local/private address (should use http, not https)."""
    if host in ("localhost", "127.0.0.1", "::1", "0.0.0.0", "::"):
        return True
    if host.startswith("192.168.") or host.startswith("10."):
        return True
    if host.startswith("172."):
        try:
            second_octet = int(host.split(".")[1])
            if 16 <= second_octet <= 31:
                return True
        except (ValueError, IndexError):
            pass
    return False


def normalize_host_for_connection(host: str) -> str:
    """
    Convert bind addresses to connectable addresses.

    0.0.0.0 and :: are valid for binding a server to all interfaces,
    but clients cannot connect to them. Translate to loopback addresses.
    """
    if host == "0.0.0.0":
        return "127.0.0.1"
    if host == "::":
        return "::1"
    return host


def get_scheme_for_host(host: str, port: int) -> str:
    """Determine http or https scheme based on host and port."""
    if port == 443:
        return "https"
    if is_local_host(host):
        return "http"
    if "." in host:
        return "https"
    return "http"


def is_full_url(host: str) -> bool:
    """Check if host is already a full URL (starts with http:// or https://)."""
    return host.startswith("http://") or host.startswith("https://")


def format_cooldown(seconds: int) -> str:
    """Format cooldown seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        return f"{mins}m {secs}s" if secs > 0 else f"{mins}m"
    else:
        hours = seconds // 3600
        mins = (seconds % 3600) // 60
        return f"{hours}h {mins}m" if mins > 0 else f"{hours}h"


def natural_sort_key(item: Dict[str, Any]) -> List:
    """
    Generate a sort key for natural/numeric sorting.

    Sorts credentials like proj-1, proj-2, proj-10 correctly
    instead of alphabetically (proj-1, proj-10, proj-2).
    """
    identifier = item.get("identifier", "")
    parts = re.split(r"(\d+)", identifier)
    return [int(p) if p.isdigit() else p.lower() for p in parts]
