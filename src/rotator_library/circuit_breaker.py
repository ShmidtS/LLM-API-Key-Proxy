# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/circuit_breaker.py
"""
Circuit Breaker pattern implementation for provider resilience.

Prevents cascade exhaustion when a provider experiences IP-level throttling
by temporarily blocking requests to that provider, allowing it to recover.

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Provider is blocked, all requests fail fast
- HALF_OPEN: Testing if provider has recovered with limited requests

Uses per-provider sharded locks so parallel requests to different providers
do not block each other.

FIX: Implements in-flight request tracking to properly handle concurrent
requests in HALF_OPEN state. Uses half_open_active counter for requests
currently being executed, incremented on start and decremented on completion.
"""

import asyncio
import time
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .utils.provider_locks import ProviderLockManager
from .config.defaults import (
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
    CIRCUIT_BREAKER_HALF_OPEN_REQUESTS,
)

lib_logger = logging.getLogger("rotator_library")

# Deduplication cache for repeated circuit breaker messages.
# Prevents log spam when HALF_OPEN state rejects many concurrent requests.
_CB_DEDUP_CACHE: Dict[str, float] = {}
_CB_DEDUP_TTL = 5.0  # seconds to suppress identical messages


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed" # Normal operation
    OPEN = "open" # Provider blocked, fail fast
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitInfo:
    """Tracks circuit breaker state for a single provider."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    half_open_attempts: int = 0  # Total attempts made (for logging/metrics)
    half_open_active: int = 0    # Currently in-flight requests in HALF_OPEN state
    custom_recovery_timeout: Optional[int] = None # Per-circuit timeout (e.g., from IP throttle duration)

    def reset(self) -> None:
        """Reset circuit to initial state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_success_time = None
        self.half_open_attempts = 0
        self.half_open_active = 0
        self.custom_recovery_timeout = None


class ProviderCircuitBreaker:
    """
    Circuit breaker for preventing cascade exhaustion during IP throttling.

    Each provider has its own circuit that tracks failures and recovery.
    When a provider's circuit opens, requests to that provider fail fast
    without attempting the actual API call.

    Uses per-provider sharded locks: parallel requests to different providers
    (e.g., NVIDIA and Gemini simultaneously) do not contend for the same lock.

    Configuration:
    failure_threshold: Number of failures before opening circuit
    recovery_timeout: Seconds to wait before attempting recovery
    half_open_requests: Number of test requests in half-open state

    Usage:
    circuit = ProviderCircuitBreaker()

    if circuit.can_attempt("openai"):
    try:
    result = await make_request()
    circuit.record_success("openai")
    except IPThrottleError:
    circuit.record_ip_throttle("openai")
    raise
    else:
    raise CircuitOpenError("Provider temporarily unavailable")
    """

    def __init__(
        self,
        failure_threshold: int = CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout: int = CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
        half_open_requests: int = CIRCUIT_BREAKER_HALF_OPEN_REQUESTS,
        provider_overrides: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """
        Initialize the circuit breaker.

        Args:
        failure_threshold: Number of consecutive failures before opening
        recovery_timeout: Seconds to wait before attempting recovery
        half_open_requests: Max concurrent test requests in half-open state
        provider_overrides: Per-provider settings dict, e.g.:
        {"kilocode": {"failure_threshold": 5, "recovery_timeout": 30}}
        """
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_requests = half_open_requests
        self._provider_overrides = provider_overrides or {}
        self._circuits: Dict[str, CircuitInfo] = {}

        # Per-provider sharded locks (lazy init)
        self._provider_lock_manager = ProviderLockManager()

        lib_logger.info(
            "Circuit breaker initialized: threshold=%d, timeout=%ds, half_open=%d, overrides=%s",
            failure_threshold, recovery_timeout, half_open_requests,
            list(self._provider_overrides.keys())
        )

    @staticmethod
    def _dedup_log(key: str) -> bool:
        """Check if a deduplicated log message should be suppressed.

        Returns True if the message was logged recently (should skip).
        """
        now = time.monotonic()
        last = _CB_DEDUP_CACHE.get(key, 0.0)
        if now - last < _CB_DEDUP_TTL:
            return True  # suppress
        _CB_DEDUP_CACHE[key] = now
        # Evict stale entries on every write
        stale = [k for k, v in _CB_DEDUP_CACHE.items() if now - v > _CB_DEDUP_TTL]
        for k in stale:
            del _CB_DEDUP_CACHE[k]
        return False

    def _get_provider_threshold(self, provider: str) -> int:
        """Get failure threshold for a provider (with overrides)."""
        if provider in self._provider_overrides:
            return self._provider_overrides[provider].get(
                "failure_threshold", self._failure_threshold
            )
        return self._failure_threshold

    def _get_provider_timeout(self, provider: str) -> int:
        """Get recovery timeout for a provider (with overrides)."""
        if provider in self._provider_overrides:
            return self._provider_overrides[provider].get(
                "recovery_timeout", self._recovery_timeout
            )
        return self._recovery_timeout

    def _get_provider_half_open(self, provider: str) -> int:
        """Get half-open requests for a provider (with overrides)."""
        if provider in self._provider_overrides:
            return self._provider_overrides[provider].get(
                "half_open_requests", self._half_open_requests
            )
        return self._half_open_requests

    def _get_or_create_circuit(self, provider: str) -> CircuitInfo:
        """Get or create circuit info for a provider (must be called under provider lock)."""
        if provider not in self._circuits:
            self._circuits[provider] = CircuitInfo()
        return self._circuits[provider]

    async def can_attempt(self, provider: str) -> bool:
        """
        Check if a request can be attempted for the given provider.

        In CLOSED state: Always returns True
        In OPEN state: Returns False until recovery timeout passes,
        then transitions to HALF_OPEN
        In HALF_OPEN state: Returns True up to half_open_requests concurrent requests

        FIX: Uses half_open_active counter instead of half_open_attempts,
        properly tracking in-flight requests with increment/decrement semantics.

        Args:
        provider: Provider name to check

        Returns:
        True if request can be attempted, False otherwise
        """
        # Fast-path: CLOSED (or unseen) providers skip lock acquisition entirely.
        # dict.get() is atomic for reads in CPython, so this check is safe
        # without synchronization. A benign race exists: state may change
        # after this check, but that also applies under lock since can_attempt()
        # permission is point-in-time, not a reservation.
        circuit = self._circuits.get(provider)
        if circuit is None or circuit.state == CircuitState.CLOSED:
            return True

        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)
            current_time = time.monotonic()

            if circuit.state == CircuitState.CLOSED:
                return True

            if circuit.state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if circuit.last_failure_time is None:
                    # Should not happen, but handle gracefully
                    circuit.reset()
                    return True

                elapsed = current_time - circuit.last_failure_time
                # Use custom timeout if set (from IP throttle duration), else provider default
                recovery_timeout = circuit.custom_recovery_timeout or self._get_provider_timeout(provider)
                if elapsed >= recovery_timeout:
                    # Transition to half-open
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.half_open_attempts = 0
                    circuit.half_open_active = 0
                    lib_logger.info(
                        "Circuit for '%s' transitioned OPEN -> HALF_OPEN after %.1fs (timeout was %ds)",
                        provider, elapsed, recovery_timeout
                    )
                    # Continue to HALF_OPEN check below - don't return yet

                else:
                    # Still in cooldown
                    remaining = recovery_timeout - elapsed
                    lib_logger.debug(
                        "Circuit for '%s' is OPEN, %.1fs until recovery attempt (timeout: %ds)",
                        provider, remaining, recovery_timeout
                    )
                    return False

            if circuit.state == CircuitState.HALF_OPEN:
                # FIX: Use half_open_active for concurrent request tracking
                # This allows multiple concurrent requests up to the limit
                half_open_max = self._get_provider_half_open(provider)
                if circuit.half_open_active < half_open_max:
                    # Increment active counter - will be decremented on completion
                    circuit.half_open_active += 1
                    circuit.half_open_attempts += 1  # Keep for metrics/logging
                    if not self._dedup_log(f"ho_active:{provider}"):
                        lib_logger.debug(
                            "Circuit for '%s' in HALF_OPEN, active %d/%d (total attempts: %d)",
                            provider, circuit.half_open_active, half_open_max, circuit.half_open_attempts
                        )
                    return True

                # Exceeded half-open concurrent requests, stay blocked
                if not self._dedup_log(f"ho_max:{provider}"):
                    lib_logger.debug(
                        "Circuit for '%s' in HALF_OPEN, max concurrent requests reached (%d/%d)",
                        provider, circuit.half_open_active, half_open_max
                    )
                return False

        return True # Should never reach here

    async def record_success(self, provider: str) -> None:
        """
        Record a successful request, potentially closing the circuit.

        If the circuit was in HALF_OPEN state, a success transitions it
        back to CLOSED (normal operation).

        FIX: Decrements half_open_active counter to release slot for other requests.

        Args:
        provider: Provider name that succeeded
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)
            current_time = time.monotonic()
            circuit.last_success_time = current_time

            if circuit.state == CircuitState.HALF_OPEN:
                # FIX: Decrement active counter
                circuit.half_open_active = max(0, circuit.half_open_active - 1)

                # Only close circuit if this was the last active request
                # or if we've had enough successful recoveries
                if circuit.half_open_active == 0:
                    # Recovery successful, close the circuit
                    circuit.reset()
                    lib_logger.info(
                        "Circuit for '%s' recovered: HALF_OPEN -> CLOSED",
                        provider
                    )
                else:
                    if not self._dedup_log(f"ho_success:{provider}"):
                        lib_logger.debug(
                            "Circuit for '%s' in HALF_OPEN, request succeeded, %d active requests remaining",
                            provider, circuit.half_open_active
                        )

            elif circuit.state == CircuitState.CLOSED:
                # Reset failure count on success
                circuit.failure_count = 0

    async def record_ip_throttle(self, provider: str) -> None:
        """
        Record an IP throttle event, potentially opening the circuit.

        Increments failure count and opens circuit if threshold is reached.
        In HALF_OPEN state, immediately reopens the circuit.

        FIX: Decrements half_open_active counter to release slot for other requests.

        Args:
        provider: Provider name that was throttled
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)
            current_time = time.monotonic()
            threshold = self._get_provider_threshold(provider)

            circuit.failure_count += 1
            circuit.last_failure_time = current_time

            if circuit.state == CircuitState.HALF_OPEN:
                # FIX: Decrement active counter (only for this request, not others still in-flight)
                circuit.half_open_active = max(0, circuit.half_open_active - 1)

                # Failed during recovery, reopen circuit
                circuit.state = CircuitState.OPEN
                circuit.half_open_attempts = 0
                lib_logger.warning(
                    "Circuit for '%s' reopened: HALF_OPEN -> OPEN (failure during recovery)",
                    provider
                )

            elif circuit.state == CircuitState.CLOSED:
                if circuit.failure_count >= threshold:
                    # Threshold reached, open circuit
                    circuit.state = CircuitState.OPEN
                    lib_logger.warning(
                        "Circuit for '%s' opened: CLOSED -> OPEN after %d failures (threshold=%d)",
                        provider, circuit.failure_count, threshold
                    )
                else:
                    lib_logger.debug(
                        "Circuit for '%s' failure count: %d/%d",
                        provider, circuit.failure_count, threshold
                    )

    async def release_half_open_slot(self, provider: str) -> None:
        """
        Release a half-open slot that was acquired via can_attempt().

        Must be called in every error path that does NOT go through
        record_success() or record_ip_throttle(), otherwise the slot
        leaks and the provider becomes permanently stuck in HALF_OPEN
        once half_open_active reaches half_open_max.

        Also handles the case where another request caused the circuit to
        transition HALF_OPEN -> OPEN while this request was still in-flight:
        the counter must still be decremented to avoid slot leaks.

        Args:
        provider: Provider name whose slot to release
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)
            if circuit.half_open_active > 0:
                circuit.half_open_active = max(0, circuit.half_open_active - 1)
                if not self._dedup_log(f"ho_release:{provider}"):
                    lib_logger.debug(
                        "Circuit for '%s' half_open slot released (state=%s), active %d",
                        provider, circuit.state.value, circuit.half_open_active
                    )

    async def open_immediately(
        self,
        provider: str,
        reason: str = "IP throttle detected",
        duration: Optional[int] = None
    ) -> None:
        """
        Immediately open the circuit for a provider, bypassing failure threshold.

        Use this when we have high confidence that the provider is experiencing
        IP-level throttling. The circuit will open immediately regardless of
        the failure count.

        Args:
        provider: Provider name to block
        reason: Reason for immediate opening (for logging)
        duration: Custom recovery timeout in seconds (e.g., from retry-after header)
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)
            current_time = time.monotonic()

            if circuit.state == CircuitState.OPEN:
                lib_logger.debug(
                    "Circuit for '%s' already OPEN, updating failure time and duration",
                    provider
                )
                circuit.last_failure_time = current_time
                if duration is not None:
                    circuit.custom_recovery_timeout = duration
                return

            circuit.state = CircuitState.OPEN
            circuit.last_failure_time = current_time
            circuit.failure_count += 1
            # FIX: Reset active counter when circuit opens
            circuit.half_open_active = 0
            if duration is not None:
                circuit.custom_recovery_timeout = duration
            lib_logger.warning(
                "Circuit for '%s' immediately opened: %s (recovery in %ds)",
                provider, reason, duration or self._get_provider_timeout(provider)
            )

    async def get_state(self, provider: str) -> CircuitState:
        """
        Get the current state of the circuit for a provider.

        This method also handles state transitions based on time:
        - OPEN circuits may transition to HALF_OPEN if timeout passed

        Args:
        provider: Provider name to check

        Returns:
        Current CircuitState for the provider
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)

            # Check for automatic transition from OPEN to HALF_OPEN
            if circuit.state == CircuitState.OPEN and circuit.last_failure_time:
                elapsed = time.monotonic() - circuit.last_failure_time
                recovery_timeout = circuit.custom_recovery_timeout or self._recovery_timeout
                if elapsed >= recovery_timeout:
                    circuit.state = CircuitState.HALF_OPEN
                    circuit.half_open_attempts = 0
                    circuit.half_open_active = 0
                    lib_logger.info(
                        "Circuit for '%s' auto-transitioned OPEN -> HALF_OPEN (timeout was %ds)",
                        provider, recovery_timeout
                    )

            return circuit.state

    async def get_all_states(self) -> Dict[str, CircuitState]:
        """Get the current state of all provider circuits concurrently."""
        providers = list(self._circuits.keys())

        async def _read_state(provider: str) -> Tuple[str, Optional[CircuitState]]:
            lock = await self._provider_lock_manager.get_lock(provider)
            async with lock:
                if provider in self._circuits:
                    return provider, self._circuits[provider].state
            return provider, None

        results = await asyncio.gather(*[_read_state(p) for p in providers])
        return {p: s for p, s in results if s is not None}

    async def reset_provider(self, provider: str) -> None:
        """
        Manually reset a provider's circuit to CLOSED state.

        Args:
        provider: Provider name to reset
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            if provider in self._circuits:
                self._circuits[provider].reset()
        lib_logger.info("Circuit for '%s' manually reset to CLOSED", provider)

    async def reset_all(self) -> None:
        """Reset all provider circuits to CLOSED state."""
        providers = list(self._circuits.keys())

        for provider in providers:
            lock = await self._provider_lock_manager.get_lock(provider)
            async with lock:
                if provider in self._circuits:
                    self._circuits[provider].reset()
        lib_logger.info("All circuits reset to CLOSED")

    async def get_provider_info(self, provider: str) -> Dict:
        """
        Get detailed information about a provider's circuit.

        Args:
        provider: Provider name to query

        Returns:
        Dictionary with circuit details
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)

            info = {
                "provider": provider,
                "state": circuit.state.value,
                "failure_count": circuit.failure_count,
                "failure_threshold": self._failure_threshold,
                "recovery_timeout": self._recovery_timeout,
                "half_open_requests": self._half_open_requests,
                "half_open_attempts": circuit.half_open_attempts,
                "half_open_active": circuit.half_open_active,
            }

            now = time.monotonic()
            if circuit.last_failure_time:
                elapsed = now - circuit.last_failure_time
                info["last_failure_elapsed"] = elapsed
                info["recovery_in"] = max(0, self._recovery_timeout - elapsed)

            if circuit.last_success_time:
                info["last_success_elapsed"] = now - circuit.last_success_time

            return info

    async def get_cooldown_remaining(self, provider: str) -> float:
        """
        Get remaining cooldown time for a provider's circuit.

        Args:
        provider: Provider name to check

        Returns:
        Remaining cooldown time in seconds, or 0 if not cooling down
        """
        lock = await self._provider_lock_manager.get_lock(provider)
        async with lock:
            circuit = self._get_or_create_circuit(provider)
            if circuit.state == CircuitState.OPEN and circuit.last_failure_time:
                elapsed = time.monotonic() - circuit.last_failure_time
                recovery_timeout = circuit.custom_recovery_timeout or self._recovery_timeout
                remaining = recovery_timeout - elapsed
                return max(0, remaining)
            return 0

    async def get_all_open_circuits(self) -> List[str]:
        """
        Get list of all providers with open circuits.

        Returns:
        List of provider names with OPEN circuits
        """
        providers = list(self._circuits.keys())

        open_circuits = []
        for provider in providers:
            lock = await self._provider_lock_manager.get_lock(provider)
            async with lock:
                if provider in self._circuits and self._circuits[provider].state == CircuitState.OPEN:
                    open_circuits.append(provider)
        return open_circuits
