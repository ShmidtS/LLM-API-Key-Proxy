# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
import time
from typing import Dict

from .utils.provider_locks import ProviderLockManager

lib_logger = logging.getLogger("rotator_library")


class CooldownManager:
    """
    Manages cooldown periods for API credentials to handle rate limiting.
    Cooldowns are applied per-credential, allowing other credentials from the
    same provider to be used while one is cooling down.

    Uses per-provider sharded locks to avoid global serialization:
    parallel requests to different providers do not block each other.
    """

    _MAX_COOLDOWNS = 10000

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._provider_lock_manager = ProviderLockManager()
        self._last_cleanup = 0.0

    def _extract_provider(self, credential: str) -> str:
        """
        Extract provider name from a credential string.
        Credentials typically follow the pattern 'provider_key_N' or
        are just the credential string itself.
        Returns a provider key suitable for lock sharding.
        """
        # NOTE: naive split — assumes credential format 'provider_key_N'.
        # Will misidentify if the provider name itself contains '_' or the
        # prefix is not the provider name.
        parts = credential.split("_")
        if len(parts) >= 2:
            return parts[0]
        return credential

    async def _cleanup_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._cooldowns.items() if now >= v]
        if expired:
            provider = self._extract_provider(expired[0])
            lock = await self._provider_lock_manager.get_lock(provider)
            async with lock:
                for k in expired:
                    if k in self._cooldowns and now >= self._cooldowns[k]:
                        del self._cooldowns[k]
        if len(self._cooldowns) > self._MAX_COOLDOWNS:
            sorted_keys = sorted(self._cooldowns, key=self._cooldowns.get)
            oversize = sorted_keys[:len(self._cooldowns) - self._MAX_COOLDOWNS]
            if oversize:
                provider = self._extract_provider(oversize[0])
                lock = await self._provider_lock_manager.get_lock(provider)
                async with lock:
                    for k in oversize:
                        if k in self._cooldowns:
                            del self._cooldowns[k]
        self._last_cleanup = now

    async def periodic_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup >= 30.0:
            await self._cleanup_expired()

    def get_available_credentials(self, credentials: list[str]) -> list[str]:
        """Return credentials not currently in cooldown. Single-pass O(N)."""
        now = time.monotonic()
        return [c for c in credentials if self._cooldowns.get(c, 0.0) <= now]

    def is_cooling_down_sync(self, credential: str) -> bool:
        """Synchronous check if a credential is cooling down. Safe in asyncio single-thread."""
        expiry = self._cooldowns.get(credential)
        if expiry is None:
            return False
        return time.monotonic() < expiry

    async def is_cooling_down(self, credential: str) -> bool:
        """Checks if a credential is currently in a cooldown period."""
        return self.is_cooling_down_sync(credential)

    async def start_cooldown(self, credential: str, duration: int):
        """
        Initiates or extends a cooldown period for a credential.
        Sets expiry to max(existing, now + duration) so concurrent 429s
        with different durations always keep the longest cooldown.
        """
        provider = self._extract_provider(credential)
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            new_expiry = time.monotonic() + duration
            existing = self._cooldowns.get(credential, 0)
            self._cooldowns[credential] = max(existing, new_expiry)

    async def get_cooldown_remaining(self, credential: str) -> float:
        """
        Returns the remaining cooldown time in seconds for a credential.
        Returns 0 if the credential is not in a cooldown period.
        """
        provider = self._extract_provider(credential)
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            expiry = self._cooldowns.get(credential)
            if expiry is None:
                return 0
            return max(0.0, expiry - time.monotonic())
