# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/utils/model_utils.py
"""
Shared model string parsing utilities.

Eliminates duplication of provider/model string extraction across
client.py, usage_manager.py, and provider modules.
"""

from __future__ import annotations

import functools
from typing import Optional


@functools.lru_cache(maxsize=256)
def extract_provider_from_model(model: str) -> str:
    """
    Extract provider prefix from ``provider/model`` format.

    Args:
        model: Model string, optionally with ``provider/`` prefix.

    Returns:
        Lowercased provider name, or empty string if no prefix.
    """
    if not isinstance(model, str):
        return ""
    normalized = model.strip()
    if not normalized or "/" not in normalized:
        return ""
    return normalized.split("/", 1)[0].strip().lower()


@functools.lru_cache(maxsize=256)
def normalize_model_string(model: str) -> str:
    """
    Normalize incoming model string for consistent routing.

    Remaps legacy provider prefixes (e.g., nvidia_nim/ -> nvidia/) and strips whitespace.

    Args:
        model: Raw model string from request.

    Returns:
        Stripped and normalized model string, or empty string if not a string.
    """
    if not isinstance(model, str):
        return ""
    normalized = model.strip()
    # Legacy provider prefix aliases
    _PROVIDER_PREFIX_ALIASES = {
        "nvidia_nim": "nvidia",
    }
    slash_pos = normalized.find("/")
    if slash_pos > 0:
        prefix = normalized[:slash_pos]
        if prefix in _PROVIDER_PREFIX_ALIASES:
            normalized = _PROVIDER_PREFIX_ALIASES[prefix] + normalized[slash_pos:]
    return normalized


@functools.lru_cache(maxsize=256)
def parse_env_credential_path(path: str) -> Optional[str]:
    """
    Parse a virtual ``env://`` path and return the credential index.

    Supported formats:
    - ``env://provider/0`` — legacy single credential
    - ``env://provider/1`` — first numbered credential

    Args:
        path: Credential path string.

    Returns:
        Credential index as string ("0" for legacy, "1", "2", etc.)
        or None if path is not an ``env://`` path.
    """
    if not isinstance(path, str) or not path.startswith("env://"):
        return None

    parts = path[6:].split("/")
    if len(parts) >= 2:
        return parts[1]
    return "0"
