# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Thin resilience facade — delegates to specialized modules.

RotatingClient imports this single class instead of 4 separate components.
The orchestrator owns the instances and passes them through; it does NOT
embed or duplicate any logic.
"""

from typing import Optional

from .circuit_breaker import ProviderCircuitBreaker
from .cooldown_manager import CooldownManager
from .error_handler import ThrottleActionType, handle_429_error
from .ip_throttle_detector import IPThrottleDetector
from .adaptive_rate_limiter import AdaptiveRateLimiter


class ResilienceOrchestrator:
    """Delegate-only facade for client-facing resilience operations."""

    __slots__ = ("cooldown", "circuit_breaker", "ip_throttle", "rate_limiter")

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_requests: int = 1,
        provider_overrides: Optional[dict] = None,
    ):
        self.cooldown = CooldownManager()
        self.ip_throttle = IPThrottleDetector()
        self.rate_limiter = AdaptiveRateLimiter()
        self.circuit_breaker = ProviderCircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            half_open_requests=half_open_requests,
            provider_overrides=provider_overrides,
        )

    async def handle_429(
        self, provider: str, credential: str, error: Exception,
        error_body: Optional[str], classified_error,
    ) -> bool:
        """Handle a 429 error via error_handler. Returns True if IP throttle detected."""
        action = await handle_429_error(
            provider=provider,
            credential=credential,
            error=error,
            error_body=error_body,
            retry_after=classified_error.retry_after,
            ip_throttle_detector=self.ip_throttle,
            circuit_breaker=self.circuit_breaker,
            cooldown_manager=self.cooldown,
        )
        return action.action_type == ThrottleActionType.PROVIDER_COOLDOWN

    async def can_attempt(self, provider: str) -> bool:
        """Check circuit breaker availability."""
        return await self.circuit_breaker.can_attempt(provider)

    async def get_cooldown_remaining(self, provider: str) -> float:
        """Get remaining cooldown from circuit breaker."""
        return await self.circuit_breaker.get_cooldown_remaining(provider)

    async def record_success(self, provider: str) -> None:
        """Record success in circuit breaker."""
        await self.circuit_breaker.record_success(provider)

    async def release_half_open_slot(self, provider: str) -> None:
        """Release a half-open slot acquired via can_attempt().

        Must be called in error paths that skip record_success() and
        record_ip_throttle(), otherwise the slot leaks and the provider
        becomes stuck in HALF_OPEN once half_open_active reaches half_open_max.
        """
        await self.circuit_breaker.release_half_open_slot(provider)

    async def acquire_rate(self, provider: str) -> float:
        """Acquire rate limiter token. Returns wait time (0 if immediate)."""
        return await self.rate_limiter.acquire(provider)

    def record_rate_429(self, provider: str, retry_after: int = None) -> None:
        """Record 429 in rate limiter for AIMD decrease."""
        self.rate_limiter.record_429(provider, retry_after)

    def record_rate_success(self, provider: str) -> None:
        """Record success in rate limiter for AIMD increase."""
        self.rate_limiter.record_success(provider)
