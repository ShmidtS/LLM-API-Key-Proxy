# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Cached environment variables for provider configuration. Pre-filtered at module load to avoid repeated O(P*E) scans of os.environ in provider __init__."""

import os
import threading
from typing import Dict, Optional


_PROVIDER_ENV_PREFIXES = (
    "CONCURRENCY_MULTIPLIER_",
    "CUSTOM_CAP_",
    "CUSTOM_CAP_COOLDOWN_",
    "FAIR_CYCLE_",
    "FAIR_CYCLE_TRACKING_MODE_",
    "FAIR_CYCLE_CROSS_TIER_",
    "FAIR_CYCLE_DURATION_",
    "EXHAUSTION_COOLDOWN_THRESHOLD_",
    "_API_HEADERS",
)
"""Environment variable prefixes that are cached for provider configuration lookups."""

_provider_env_cache: Optional[Dict[str, str]] = None
"""Dict of env vars matching provider prefixes, lazily computed on first access."""

_env_cache_lock = threading.Lock()
"""Lock protecting build/invalidate paths for thread safety."""


def _build_env_cache() -> Dict[str, str]:
    """Build the provider env cache by scanning os.environ for matching prefixes."""
    cache: Dict[str, str] = {
        k: v
        for k, v in os.environ.items()
        if any(k.startswith(p) or k.endswith(p) for p in _PROVIDER_ENV_PREFIXES)
    }
    if "EXHAUSTION_COOLDOWN_THRESHOLD" in os.environ:
        cache["EXHAUSTION_COOLDOWN_THRESHOLD"] = os.environ[
            "EXHAUSTION_COOLDOWN_THRESHOLD"
        ]
    return cache


def get_provider_env_cache() -> Dict[str, str]:
    """Return the provider env cache, building it on first call or after invalidation.

    Thread-safe: callers never see None, even if another thread has just
    invalidated the cache.
    """
    global _provider_env_cache
    if _provider_env_cache is not None:
        return _provider_env_cache
    with _env_cache_lock:
        if _provider_env_cache is None:
            _provider_env_cache = _build_env_cache()
        return _provider_env_cache
