# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
import threading
import time
from typing import Dict

lib_logger = logging.getLogger("rotator_library")


class CooldownManager:
    """
    Manages cooldown periods for API credentials to handle rate limiting.
    Cooldowns are applied per-credential, allowing other credentials from the
    same provider to be used while one is cooling down.
    """

    _MAX_COOLDOWNS = 10000

    def __init__(self):
        self._cooldowns: Dict[str, float] = {}
        self._cooldowns_lock = threading.Lock()
        self._last_cleanup = 0.0

    async def _cleanup_expired(self) -> None:
        now = time.monotonic()
        with self._cooldowns_lock:
            expired = [k for k, v in self._cooldowns.items() if now >= v]
            for k in expired:
                del self._cooldowns[k]
            overflow = len(self._cooldowns) - self._MAX_COOLDOWNS
            if overflow > 0:
                sorted_items = sorted(self._cooldowns.items(), key=lambda kv: kv[1])
                for k, _ in sorted_items[:overflow]:
                    del self._cooldowns[k]
        self._last_cleanup = now

    async def periodic_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup >= 30.0:
            await self._cleanup_expired()

    async def start_cooldown(self, credential: str, duration: int):
        """
        Initiates or extends a cooldown period for a credential.
        Sets expiry to max(existing, now + duration) so concurrent 429s
        with different durations always keep the longest cooldown.
        """
        with self._cooldowns_lock:
            new_expiry = time.monotonic() + duration
            existing = self._cooldowns.get(credential, 0)
            self._cooldowns[credential] = max(existing, new_expiry)

