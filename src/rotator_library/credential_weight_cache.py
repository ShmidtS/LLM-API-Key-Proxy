# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

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
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .utils.singleton import SingletonMeta

lib_logger = logging.getLogger("rotator_library")


@dataclass
class CachedWeights:
    """Cached weight calculation for a provider/model combination."""

    weights: Dict[str, float]  # credential -> weight
    total_weight: float
    credentials: List[str]  # Ordered list of available credentials
    calculated_at: float = field(default_factory=time.monotonic)
    usage_snapshot: Dict[str, int] = field(
        default_factory=dict
    )  # credential -> usage at calc time
    invalidated: bool = False


class CredentialWeightCache(metaclass=SingletonMeta):
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
        self._provider_index: Dict[str, List[str]] = {}  # provider -> [cache_keys]
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
            if time.monotonic() - cached.calculated_at > self._ttl:
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
            # Update provider index for O(1) invalidation
            idx = self._provider_index.setdefault(provider, [])
            if key not in idx:
                idx.append(key)

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

            # Use provider index for O(1) lookup instead of O(n) prefix scan
            provider_keys = self._provider_index.get(provider, [])

            for key in provider_keys:
                cached = self._cache.get(key)
                if cached is None:
                    continue

                # Check if this credential is in the cached entry
                if credential not in cached.usage_snapshot:
                    continue

                # If model specified, only invalidate that model's entries
                if (
                    model is not None
                    and f":{model}:" not in key
                    and not key.endswith(f":{model}")
                ):
                    continue

                keys_to_invalidate.append(key)

            for key in keys_to_invalidate:
                self._cache[key].invalidated = True
                self._stats["invalidations"] += 1
                # Remove from provider index to speed up cleanup_expired
                prov_keys = self._provider_index.get(provider)
                if prov_keys and key in prov_keys:
                    prov_keys.remove(key)

    async def invalidate_all(self, provider: Optional[str] = None) -> None:
        """
        Invalidate all cache entries, optionally filtered by provider.

        Args:
            provider: Optional provider to filter by
        """
        async with self._lock:
            if provider is None:
                invalidated_count = len(self._cache)
                self._cache.clear()
                self._provider_index.clear()
            else:
                # Use provider index for O(1) lookup
                keys_to_remove = self._provider_index.pop(provider, [])
                invalidated_count = len(keys_to_remove)
                for key in keys_to_remove:
                    self._cache.pop(key, None)

            self._stats["invalidations"] += invalidated_count

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
        now = time.monotonic()
        removed = 0

        async with self._lock:
            keys_to_remove = [
                k
                for k, v in self._cache.items()
                if now - v.calculated_at > self._ttl or v.invalidated
            ]
            for key in keys_to_remove:
                del self._cache[key]
                # Clean up provider index
                for prov_keys in self._provider_index.values():
                    if key in prov_keys:
                        prov_keys.remove(key)
                removed += 1

            # Clean empty provider index entries
            self._provider_index = {k: v for k, v in self._provider_index.items() if v}

        return removed

    async def warmup_weights(
        self,
        providers: List[str],
        models: List[str],
        weight_calculator: Any = None,
    ) -> int:
        """
        Pre-populate weight cache for common provider/model combinations.

        This method triggers weight calculation for specified combinations,
        ensuring the cache is warm before the first request arrives.

        Args:
            providers: List of provider names to warmup
            models: List of model names to warmup
            weight_calculator: Optional callable to calculate weights
                              (provider, model) -> (weights, credentials, usage)

        Returns:
            Number of cache entries warmed up
        """
        warmed = 0

        for provider in providers:
            for model in models:
                key = self._make_key(provider, model)
                async with self._lock:
                    if key in self._cache:
                        continue  # Already cached

                # If a weight calculator is provided, use it
                if weight_calculator:
                    try:
                        result = await weight_calculator(provider, model)
                        if result:
                            weights, credentials, usage_snapshot = result
                            await self.set(
                                provider, model, weights, credentials, usage_snapshot
                            )
                            warmed += 1
                    except Exception as e:
                        lib_logger.debug(f"Warmup failed for {provider}/{model}: {e}")

        if warmed > 0:
            lib_logger.info(f"Weight cache warmed up: {warmed} entries")

        return warmed
