# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/credential_weight_cache.py
"""
Credential weight caching for optimized credential selection.

Caches calculated weights for credential selection to avoid
recalculating on every request. Weights are invalidated
when usage changes.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

lib_logger = logging.getLogger("rotator_library")


@dataclass
class CachedWeights:
    """Cached weight calculation for a provider/model combination."""
    weights: Dict[str, float]  # credential -> weight
    total_weight: float
    credentials: List[str]  # Ordered list of available credentials
    calculated_at: float = field(default_factory=time.time)
    usage_snapshot: Dict[str, int] = field(default_factory=dict)  # credential -> usage at calc time
    invalidated: bool = False


class CredentialWeightCache:
    """
    Caches credential selection weights with automatic invalidation.

    Weight calculation is expensive when there are many credentials.
    This cache stores the calculated weights and only recalculates
    when usage changes significantly.

    Features:
    - Per-provider/model weight caching
    - Automatic invalidation on usage change
    - Background refresh for stale entries
    - Thread-safe with asyncio locks
    """

    def __init__(
        self,
        ttl_seconds: float = 60.0,  # Max age before forced refresh
        usage_change_threshold: int = 1,  # Min usage change to invalidate
    ):
        """
        Initialize the weight cache.

        Args:
            ttl_seconds: Max age of cached weights before refresh
            usage_change_threshold: Min usage delta to trigger invalidation
        """
        self._ttl = ttl_seconds
        self._threshold = usage_change_threshold
        self._cache: Dict[str, CachedWeights] = {}  # key -> CachedWeights
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "evictions": 0,
        }

    def _make_key(self, provider: str, model: str, tier: Optional[int] = None) -> str:
        """Create cache key from provider/model/tier."""
        if tier is not None:
            return f"{provider}:{model}:t{tier}"
        return f"{provider}:{model}"

    async def get(
        self,
        provider: str,
        model: str,
        tier: Optional[int] = None,
    ) -> Optional[CachedWeights]:
        """
        Get cached weights if still valid.

        Args:
            provider: Provider name
            model: Model name
            tier: Optional tier for tier-specific selection

        Returns:
            CachedWeights if valid cache exists, None otherwise
        """
        key = self._make_key(provider, model, tier)

        async with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                self._stats["misses"] += 1
                return None

            # Check if expired
            if time.time() - cached.calculated_at > self._ttl:
                self._stats["evictions"] += 1
                del self._cache[key]
                return None

            # Check if invalidated
            if cached.invalidated:
                self._stats["invalidations"] += 1
                del self._cache[key]
                return None

            self._stats["hits"] += 1
            return cached

    async def set(
        self,
        provider: str,
        model: str,
        weights: Dict[str, float],
        credentials: List[str],
        usage_snapshot: Dict[str, int],
        tier: Optional[int] = None,
    ) -> None:
        """
        Store calculated weights in cache.

        Args:
            provider: Provider name
            model: Model name
            weights: Calculated weights (credential -> weight)
            credentials: List of available credentials
            usage_snapshot: Usage at time of calculation
            tier: Optional tier for tier-specific selection
        """
        key = self._make_key(provider, model, tier)

        cached = CachedWeights(
            weights=weights,
            total_weight=sum(weights.values()),
            credentials=credentials,
            usage_snapshot=usage_snapshot,
        )

        async with self._lock:
            self._cache[key] = cached

    async def invalidate(
        self,
        provider: str,
        credential: str,
        model: Optional[str] = None,
    ) -> None:
        """
        Invalidate cache entries affected by a usage change.

        Args:
            provider: Provider name
            credential: Credential that changed
            model: Optional specific model (invalidates all if None)
        """
        async with self._lock:
            keys_to_invalidate = []

            for key, cached in self._cache.items():
                # Check if this key is for the affected provider
                if not key.startswith(f"{provider}:"):
                    continue

                # Check if this credential is in the cached entry
                if credential not in cached.usage_snapshot:
                    continue

                # If model specified, only invalidate that model's entries
                if model is not None and f":{model}:" not in key and not key.endswith(f":{model}"):
                    continue

                keys_to_invalidate.append(key)

            for key in keys_to_invalidate:
                self._cache[key].invalidated = True
                self._stats["invalidations"] += 1

    async def invalidate_all(self, provider: Optional[str] = None) -> None:
        """
        Invalidate all cache entries, optionally filtered by provider.

        Args:
            provider: Optional provider to filter by
        """
        async with self._lock:
            if provider is None:
                self._cache.clear()
            else:
                keys_to_remove = [
                    k for k in self._cache.keys()
                    if k.startswith(f"{provider}:")
                ]
                for key in keys_to_remove:
                    del self._cache[key]

            self._stats["invalidations"] += len(self._cache)

    async def check_usage_change(
        self,
        provider: str,
        model: str,
        current_usage: Dict[str, int],
        tier: Optional[int] = None,
    ) -> bool:
        """
        Check if usage has changed enough to invalidate cache.

        Args:
            provider: Provider name
            model: Model name
            current_usage: Current usage dict (credential -> count)
            tier: Optional tier

        Returns:
            True if cache should be invalidated
        """
        key = self._make_key(provider, model, tier)

        async with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return True  # No cache, needs calculation

            # Check for usage changes exceeding threshold
            for cred, usage in current_usage.items():
                cached_usage = cached.usage_snapshot.get(cred, 0)
                if abs(usage - cached_usage) >= self._threshold:
                    return True

            # Check for new credentials
            for cred in current_usage:
                if cred not in cached.usage_snapshot:
                    return True

            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            **self._stats,
            "entries": len(self._cache),
            "ttl_seconds": self._ttl,
            "threshold": self._threshold,
        }

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        now = time.time()
        removed = 0

        async with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items()
                if now - v.calculated_at > self._ttl or v.invalidated
            ]
            for key in keys_to_remove:
                del self._cache[key]
                removed += 1

        return removed


# Singleton instance
_CACHE_INSTANCE: Optional[CredentialWeightCache] = None


def get_weight_cache() -> CredentialWeightCache:
    """Get the global weight cache singleton."""
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = CredentialWeightCache()
    return _CACHE_INSTANCE
