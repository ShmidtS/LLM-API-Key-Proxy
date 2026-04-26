# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Base retry infrastructure for RotatingClient -- shared data structures,
credential selection, and cleanup logic used by both streaming and
non-streaming retry paths."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from cachetools import TTLCache

lib_logger = logging.getLogger("rotator_library")

# Deduplication cache for repeated circuit-breaker-open messages.
_CB_OPEN_DEDUP: TTLCache = TTLCache(maxsize=256, ttl=5.0)


def _should_suppress_cb_open(provider: str) -> bool:
    """Suppress repeated 'Circuit breaker OPEN' messages within TTL window."""
    if provider in _CB_OPEN_DEDUP:
        return True
    _CB_OPEN_DEDUP[provider] = True
    return False


class HalfOpenSlot:
    """Async context manager that auto-releases a half-open circuit breaker slot.

    Ensures the slot is released even on CancelledError or unexpected exceptions,
    preventing slot leaks that could permanently block a provider in HALF_OPEN.
    """

    __slots__ = ("_resilience", "_provider", "_active")

    def __init__(self, resilience, provider: str):
        self._resilience = resilience
        self._provider = provider
        self._active = False

    async def __aenter__(self):
        self._active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._active:
            self._active = False
            await self._resilience.release_half_open_slot(self._provider)
        return False


@dataclass
class _RetryContext:
    model: str
    provider: str
    credentials_for_provider: list[str]
    provider_plugin: Any
    deadline: float
    transaction_logger: Any
    tried_creds: set[str] = field(default_factory=set)
    last_exception: Optional[Exception] = None
    parent_log_dir: Optional[str] = None
    credential_priorities: dict[str, int] = field(default_factory=dict)
    credential_tier_names: dict[str, str] = field(default_factory=dict)
    error_accumulator: Any = None


@dataclass
class _ErrorDecision:
    """Decision from an error handler: what the retry loop should do next.

    action values:
        "rotate"         -- break inner loop, rotate to next credential
        "retry_same_key" -- sleep wait_time, then continue inner loop
        "fail"           -- non-recoverable, raise last_exception
    """
    action: str = "rotate"
    wait_time: float = 0.0
    classified_error: Any = None
    error_message: str = ""


@dataclass
class _KeySelectionResult:
    """Result of _select_next_key() -- selected credential and control flow hint.

    loop_action values:
        "proceed"   -- credential acquired, proceed with request
        "continue"  -- circuit breaker open or rate-limited, continue outer loop
        "break"     -- no credentials left, break outer loop
    """
    current_cred: str = ""
    cb_slot_held: bool = False
    loop_action: str = "proceed"


class RetryBaseMixin:
    """Base mixin with shared retry infrastructure used by RetryMixin."""

    async def _select_next_key(
        self,
        credentials_for_provider: list[str],
        tried_creds: set[str],
        model: str,
        provider: str,
        deadline: float,
        credential_priorities: dict[str, int],
        credential_tier_names: dict[str, str],
        suppress_cb_logging: bool = False,
    ) -> _KeySelectionResult:
        """Select and acquire the next available credential.

        Handles rate limiting wait, circuit breaker check, and key acquisition.
        Returns a _KeySelectionResult with loop_action indicating how the caller
        should proceed:
        - "proceed": credential acquired, use result.current_cred
        - "continue": circuit breaker open, continue the outer while loop
        - "break": no credentials remaining, break the outer while loop
        """
        creds_to_try = [c for c in credentials_for_provider if c not in tried_creds]
        if not creds_to_try:
            return _KeySelectionResult(loop_action="break")

        # Rate limiter backoff
        rate_wait = await self._resilience.acquire_rate(provider)
        if rate_wait > 0:
            wait = min(rate_wait, 5.0)
            lib_logger.debug(
                "AdaptiveRateLimiter: %s rate-limited, waiting %1.1fs",
                provider, wait,
            )
            if time.monotonic() + wait < deadline:
                await asyncio.sleep(wait)

        # Circuit breaker check -- back off briefly if OPEN, then retry
        if not await self._resilience.can_attempt(provider):
            remaining = await self._resilience.get_cooldown_remaining(provider)
            backoff = min(remaining, 5.0)
            if not (suppress_cb_logging and _should_suppress_cb_open(provider)):
                lib_logger.debug(
                    "Circuit breaker OPEN for provider '%s', "
                    "backing off %1.1fs (recovery in %0.0fs)",
                    provider, backoff, remaining,
                )
            if time.monotonic() + backoff < deadline:
                await asyncio.sleep(backoff)
            return _KeySelectionResult(loop_action="continue")

        # can_attempt() succeeded -- we now hold a half-open slot
        try:
            availability_stats = (
                await self.usage_manager.get_credential_availability_stats(
                    creds_to_try, model, credential_priorities
                )
            )
            available_count = availability_stats["available"]
            total_count = len(credentials_for_provider)
            on_cooldown = availability_stats["on_cooldown"]
            fc_excluded = availability_stats["fair_cycle_excluded"]

            exclusion_parts = []
            if on_cooldown > 0:
                exclusion_parts.append(f"cd:{on_cooldown}")
            if fc_excluded > 0:
                exclusion_parts.append(f"fc:{fc_excluded}")
            exclusion_str = (
                f",{','.join(exclusion_parts)}" if exclusion_parts else ""
            )

            lib_logger.info(
                "Acquiring credential for model %s. Tried: %s/%s(%s%s)",
                model, len(tried_creds), available_count, total_count, exclusion_str,
            )
            max_concurrent = self.max_concurrent_requests_per_key.get(provider, 1)

            current_cred = await self.usage_manager.acquire_key(
                available_keys=creds_to_try,
                model=model,
                deadline=deadline,
                max_concurrent=max_concurrent,
                credential_priorities=credential_priorities,
                credential_tier_names=credential_tier_names,
                all_provider_credentials=credentials_for_provider,
            )
            tried_creds.add(current_cred)
        except Exception:
            # Release the half-open slot if anything fails after can_attempt() succeeded
            await self._resilience.release_half_open_slot(provider)
            raise

        return _KeySelectionResult(
            current_cred=current_cred,
            cb_slot_held=True,
            loop_action="proceed",
        )

    async def _release_cred(
        self,
        current_cred: str,
        model: str,
        key_acquired: bool,
        provider: str,
        cb_slot_held: bool,
    ) -> None:
        """Release credential and circuit breaker slot (for use in finally blocks)."""
        if key_acquired and current_cred:
            await self.usage_manager.release_key(current_cred, model)
        if cb_slot_held:
            await self._resilience.release_half_open_slot(provider)
