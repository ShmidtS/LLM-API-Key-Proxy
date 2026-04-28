# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Adaptive Rate Limiter — proactive per-provider request pacing.

Uses token bucket with AIMD rate adjustment:
- On 429: decrease rate (multiply by decrease_factor, min min_rps)
- On success: gradually increase rate (add increase_rps every increase_interval)
- Ceiling tracking: after 429, remember the rate that caused it; probe to 90%
  of ceiling before going higher
- Ceiling expiry: if no 429 for 3 * increase_interval, clear ceiling and probe higher

Disabled by default. Enable via ADAPTIVE_RATE_LIMIT_ENABLED=true.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from .utils.provider_locks import ProviderLockManager
from .config.defaults import (
    ADAPTIVE_RATE_LIMIT_ENABLED,
    ADAPTIVE_RATE_LIMIT_INITIAL_RPS,
    ADAPTIVE_RATE_LIMIT_MIN_RPS,
    ADAPTIVE_RATE_LIMIT_MAX_RPS,
    ADAPTIVE_RATE_LIMIT_DECREASE_FACTOR,
    ADAPTIVE_RATE_LIMIT_INCREASE_RPS,
    ADAPTIVE_RATE_LIMIT_INCREASE_INTERVAL,
)

lib_logger = logging.getLogger("rotator_library")


@dataclass
class _ProviderRateState:
    """Per-provider rate limiter state."""
    current_rps: float
    ceiling_rps: Optional[float] = None
    ceiling_time: Optional[float] = None
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)
    last_429: Optional[float] = None
    last_increase: Optional[float] = None
    total_requests: int = 0
    total_429s: int = 0
    next_available_at: float = 0.0


class AdaptiveRateLimiter:
    """
    Adaptive per-provider rate limiter with token bucket + AIMD.

    Disabled by default. Enable via ADAPTIVE_RATE_LIMIT_ENABLED=true.
    """

    def __init__(
        self,
        enabled: bool = ADAPTIVE_RATE_LIMIT_ENABLED,
        initial_rps: float = ADAPTIVE_RATE_LIMIT_INITIAL_RPS,
        min_rps: float = ADAPTIVE_RATE_LIMIT_MIN_RPS,
        max_rps: float = ADAPTIVE_RATE_LIMIT_MAX_RPS,
        decrease_factor: float = ADAPTIVE_RATE_LIMIT_DECREASE_FACTOR,
        increase_rps: float = ADAPTIVE_RATE_LIMIT_INCREASE_RPS,
        increase_interval: float = ADAPTIVE_RATE_LIMIT_INCREASE_INTERVAL,
    ):
        self.enabled = enabled
        self.initial_rps = initial_rps
        self.min_rps = min_rps
        self.max_rps = max_rps
        self.decrease_factor = decrease_factor
        self.increase_rps = increase_rps
        self.increase_interval = increase_interval
        self._states: Dict[str, _ProviderRateState] = {}
        self._lock_manager = ProviderLockManager()

    def _get_state(self, provider: str) -> _ProviderRateState:
        # Double-check under lock to prevent concurrent creation for same provider
        if provider not in self._states:
            raise KeyError(provider)
        return self._states[provider]

    async def _ensure_state(self, provider: str) -> _ProviderRateState:
        """Get or create per-provider state. Caller MUST hold the per-provider lock."""
        if provider not in self._states:
            self._states[provider] = _ProviderRateState(current_rps=self.initial_rps)
            self._states[provider].tokens = self.initial_rps
        return self._states[provider]

    async def acquire(self, provider: str, tokens: int = 1) -> float:
        """Acquire tokens. Returns wait time in seconds (0 if immediate)."""
        if not self.enabled:
            return 0.0

        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            state = await self._ensure_state(provider)
            now = time.monotonic()

            elapsed = now - state.last_refill
            state.tokens = min(
                state.current_rps,
                state.tokens + elapsed * state.current_rps,
            )
            state.last_refill = now

            if state.next_available_at < now:
                state.next_available_at = now

            wait = 0.0
            if state.next_available_at > now:
                wait = state.next_available_at - now

            interval = tokens / max(state.current_rps, 0.001)
            state.next_available_at = max(state.next_available_at, now) + interval

            state.tokens -= tokens
            if state.tokens < 0:
                state.tokens = 0
            state.total_requests += 1
            return wait

    async def _record_result(self, provider: str, success: bool, retry_after: Optional[int] = None) -> None:
        if not self.enabled:
            return

        lock = await self._lock_manager.get_lock(provider)

        # Capture log args outside lock to minimize critical section
        _log: Optional[tuple] = None

        async with lock:
            state = await self._ensure_state(provider)
            now = time.monotonic()

            if success:
                if state.last_429 is not None and (now - state.last_429) < self.increase_interval:
                    return

                if state.last_increase is not None and (now - state.last_increase) < self.increase_interval:
                    return

                if state.ceiling_rps is not None and state.ceiling_time is not None:
                    if (now - state.ceiling_time) >= self.increase_interval * 3:
                        _log = ("info", "ceiling_expired", provider, state.ceiling_rps)
                        state.ceiling_rps = None
                        state.ceiling_time = None

                old_rps = state.current_rps
                state.current_rps += self.increase_rps

                if state.ceiling_rps is not None and state.ceiling_time is not None:
                    cap = state.ceiling_rps * 0.9
                    state.current_rps = min(state.current_rps, cap)

                state.current_rps = min(state.current_rps, self.max_rps)
                state.last_increase = now

                if state.current_rps != old_rps:
                    _log = ("debug", "rate_increased", provider, old_rps, state.current_rps)
            else:
                state.ceiling_rps = state.current_rps
                state.ceiling_time = now

                old_rps = state.current_rps
                state.current_rps = max(self.min_rps, state.current_rps * self.decrease_factor)
                state.last_429 = now
                state.total_429s += 1
                state.tokens = 0.0

                _log = ("info", "rate_decreased", provider, old_rps, state.current_rps, state.ceiling_rps, state.total_429s)

        # Logging outside lock — I/O no longer blocks other waiters
        if _log is not None:
            level, kind = _log[0], _log[1]
            if kind == "ceiling_expired":
                _, _, _provider, _cap = _log
                lib_logger.log(
                    getattr(logging, level.upper()),
                    "AdaptiveRateLimiter: ceiling expired for %s, clearing %.1f rps cap",
                    _provider, _cap,
                )
            elif kind == "rate_increased":
                _, _, _provider, _old, _new = _log
                lib_logger.log(
                    getattr(logging, level.upper()),
                    "AdaptiveRateLimiter: %s rate increased %.1f -> %.1f rps",
                    _provider, _old, _new,
                )
            elif kind == "rate_decreased":
                _, _, _provider, _old, _new, _ceil, _count = _log
                lib_logger.log(
                    getattr(logging, level.upper()),
                    "AdaptiveRateLimiter: %s 429 received, rate decreased %.1f -> %.1f rps (ceiling=%.1f, 429s=%d)",
                    _provider, _old, _new, _ceil, _count,
                )

    async def record_success(self, provider: str) -> None:
        await self._record_result(provider, success=True)

    async def record_429(self, provider: str, retry_after: Optional[int] = None) -> None:
        await self._record_result(provider, success=False, retry_after=retry_after)

    def is_enabled(self) -> bool:
        return self.enabled

    def get_provider_info(self, provider: str) -> Dict[str, Any]:
        try:
            state = self._get_state(provider)
        except KeyError:
            return {"provider": provider, "tracked": False}
        return {
            "provider": provider,
            "tracked": True,
            "current_rps": round(state.current_rps, 2),
            "ceiling_rps": state.ceiling_rps,
            "tokens": round(state.tokens, 2),
            "total_requests": state.total_requests,
            "total_429s": state.total_429s,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "providers": {p: self.get_provider_info(p) for p in self._states},
        }
