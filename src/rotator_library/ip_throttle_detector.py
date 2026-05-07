# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
IP Throttle Detector - Detects IP-level throttling via correlation of 429 errors.

This module analyzes 429 (rate limit) errors across multiple credentials to detect
when throttling is applied at the IP level rather than per-credential.

Detection heuristics:
- 3+ different credentials receiving 429 errors within a 30-second window
- Identical error_body hash between credentials (same error from same IP)
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

from .utils.provider_locks import ProviderLockManager
from .utils.singleton import SingletonMeta

lib_logger = logging.getLogger("rotator_library")


class ThrottleScope(Enum):
    """Scope of detected throttling."""

    CREDENTIAL = "credential"  # Per-credential rate limit
    IP = "ip"  # IP-level rate limit (affects all credentials from this IP)
    ACCOUNT = "account"  # Account-level rate limit (affects all keys in account)


class ThrottleAssessment:
    """
    Assessment of throttle scope and recommended action.

    Attributes:
        scope: Detected throttle scope (CREDENTIAL/IP/ACCOUNT)
        confidence: Confidence level 0.0-1.0
        suggested_cooldown: Recommended cooldown period in seconds
        affected_credentials: List of credentials that triggered this assessment
        error_signature: Hash of error body for correlation
        details: Additional diagnostic information
    """

    def __init__(
        self,
        scope: ThrottleScope,
        confidence: float = 0.0,
        suggested_cooldown: int = 0,
        affected_credentials: Optional[List[str]] = None,
        error_signature: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.scope = scope
        self.confidence = confidence
        self.suggested_cooldown = suggested_cooldown
        self.affected_credentials = affected_credentials if affected_credentials is not None else []
        self.error_signature = error_signature
        self.details = details if details is not None else {}

    def __str__(self) -> str:
        return (
            f"ThrottleAssessment(scope={self.scope.value}, "
            f"confidence={self.confidence:.2f}, "
            f"cooldown={self.suggested_cooldown}s, "
            f"affected={len(self.affected_credentials)} creds)"
        )

    def __repr__(self) -> str:
        return (
            f"ThrottleAssessment(scope={self.scope!r}, confidence={self.confidence!r}, "
            f"suggested_cooldown={self.suggested_cooldown!r}, "
            f"affected_credentials={self.affected_credentials!r}, "
            f"error_signature={self.error_signature!r}, details={self.details!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ThrottleAssessment):
            return NotImplemented
        return (
            self.scope == other.scope
            and self.confidence == other.confidence
            and self.suggested_cooldown == other.suggested_cooldown
            and self.affected_credentials == other.affected_credentials
            and self.error_signature == other.error_signature
            and self.details == other.details
        )

    def __hash__(self) -> int:
        return hash((
            self.scope,
            self.confidence,
            self.suggested_cooldown,
            tuple(self.affected_credentials),
            self.error_signature,
            tuple(sorted(self.details.items())),
        ))


@dataclass
class _ThrottleRecord:
    """Internal record for tracking 429 events."""

    timestamp: float
    credential: str
    error_body_hash: Optional[str]
    retry_after: Optional[int]
    error_body: Optional[str] = None


class IPThrottleDetector(metaclass=SingletonMeta):
    """
    Detects IP-level throttling by correlating 429 errors across credentials.

    When multiple credentials from the same IP receive 429 errors simultaneously,
    it indicates IP-level throttling rather than per-credential limits.

    Usage:
        detector = IPThrottleDetector()

        # Record a 429 error
        assessment = await detector.record_429(
            provider="openai",
            credential="key_abc123",
            error_body='{"error": "Rate limit exceeded"}',
            retry_after=60
        )

        if assessment.scope == ThrottleScope.IP:
            # All credentials from this IP are throttled
            # Apply cooldown to all credentials
            pass
    """

    # Configuration constants
    DEFAULT_WINDOW_SECONDS = 10
    DEFAULT_MIN_CREDENTIALS = 3
    DEFAULT_IP_COOLDOWN = 30
    DEFAULT_CREDENTIAL_COOLDOWN = 10
    MAX_RECORDS_PER_PROVIDER = 100  # Memory limit per provider

    def __init__(
        self,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        min_credentials_for_ip_throttle: int = DEFAULT_MIN_CREDENTIALS,
        ip_cooldown: int = DEFAULT_IP_COOLDOWN,
        credential_cooldown: int = DEFAULT_CREDENTIAL_COOLDOWN,
    ):
        """
        Initialize the IP throttle detector.

        Args:
            window_seconds: Time window in seconds to correlate 429 errors
            min_credentials_for_ip_throttle: Minimum credentials with 429 to detect IP throttle
            ip_cooldown: Default cooldown for IP-level throttling
            credential_cooldown: Default cooldown for credential-level throttling
        """
        self.window_seconds = window_seconds
        self.min_credentials = min_credentials_for_ip_throttle
        self.ip_cooldown = ip_cooldown
        self.credential_cooldown = credential_cooldown

        # Per-provider sharded locks (lazy init)
        self._locks = ProviderLockManager()
        self._records_lock = asyncio.Lock()

        # Per-provider tracking: provider -> list of _ThrottleRecord
        self._records: Dict[str, List[_ThrottleRecord]] = defaultdict(list)

        lib_logger.debug(
            f"IPThrottleDetector initialized: window={window_seconds}s, "
            f"min_creds={min_credentials_for_ip_throttle}"
        )

    def _hash_error_body(self, error_body: Optional[str]) -> Optional[str]:
        """Create a hash of error body for correlation."""
        if not error_body:
            return None
        # Normalize whitespace and case for consistent hashing
        normalized = "".join(error_body.split()).lower()
        return hashlib.md5(normalized.encode(), usedforsecurity=False).hexdigest()

    def _prune_records(self, records: List[_ThrottleRecord], now: Optional[float] = None) -> List[_ThrottleRecord]:
        """Remove records older than the detection window and enforce memory limit."""
        cutoff = (now if now is not None else time.monotonic()) - self.window_seconds
        records = [r for r in records if r.timestamp > cutoff]
        if len(records) > self.MAX_RECORDS_PER_PROVIDER:
            records = records[-self.MAX_RECORDS_PER_PROVIDER :]
        return records

    async def _provider_names(self) -> tuple[str, ...]:
        async with self._records_lock:
            return tuple(self._records)

    async def record_429(
        self,
        provider: str,
        credential: str,
        error_body: Optional[str] = None,
        retry_after: Optional[int] = None,
    ) -> ThrottleAssessment:
        """
        Record a 429 error and assess throttle scope.

        This is the main entry point for detecting IP-level throttling.
        Call this method whenever a 429 error is received.

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            credential: Credential identifier (masked for logging)
            error_body: Raw error response body
            retry_after: Retry-After value from headers or body

        Returns:
            ThrottleAssessment with scope, confidence, and suggested cooldown
        """
        now = time.monotonic()
        error_body_hash = self._hash_error_body(error_body)

        record = _ThrottleRecord(
            timestamp=now,
            credential=credential,
            error_body_hash=error_body_hash,
            retry_after=retry_after,
            error_body=error_body[:500] if error_body else None,
        )
        lock = await self._locks.get_lock(provider)
        async with lock:
            records = self._prune_records(self._records[provider], now)
            records.append(record)
            records = self._prune_records(records, now)
            self._records[provider] = records
            assessment = self._assess_throttle_scope_from_records(records)

        if retry_after and retry_after > assessment.suggested_cooldown:
            assessment.suggested_cooldown = retry_after

        lib_logger.debug(
            f"IPThrottleDetector.record_429: provider={provider}, "
            f"credential={credential}, assessment={assessment}"
        )

        return assessment

    async def _assess_throttle_scope(self, provider: str) -> ThrottleAssessment:
        """Assess the scope of throttling for a provider."""
        lock = await self._locks.get_lock(provider)
        async with lock:
            records = self._prune_records(self._records[provider])
            self._records[provider] = records
            return self._assess_throttle_scope_from_records(records)

    def _assess_throttle_scope_from_records(self, records: List[_ThrottleRecord]) -> ThrottleAssessment:
        """Assess the scope of throttling from a consistent records snapshot."""
        if not records:
            return ThrottleAssessment(
                scope=ThrottleScope.CREDENTIAL,
                confidence=1.0,
                suggested_cooldown=self.credential_cooldown,
            )

        unique_credentials = list(dict.fromkeys(r.credential for r in records))
        num_unique_credentials = len(unique_credentials)

        hash_counts: Dict[Optional[str], int] = defaultdict(int)
        for r in records:
            hash_counts[r.error_body_hash] += 1

        common_hash, common_hash_count = max(hash_counts.items(), key=lambda x: x[1])
        max_retry_after = max((r.retry_after or 0) for r in records)

        if num_unique_credentials >= self.min_credentials:
            confidence = min(1.0, num_unique_credentials / self.min_credentials)
            if common_hash and common_hash_count >= 2:
                confidence = min(1.0, confidence + 0.2)
            if common_hash and common_hash_count == len(records):
                confidence = min(1.0, confidence + 0.1)

            return ThrottleAssessment(
                scope=ThrottleScope.IP,
                confidence=confidence,
                suggested_cooldown=max(max_retry_after, self.ip_cooldown),
                affected_credentials=unique_credentials,
                error_signature=common_hash,
                details={
                    "credentials_throttled": num_unique_credentials,
                    "error_hash_matches": common_hash_count,
                    "window_seconds": self.window_seconds,
                },
            )

        return ThrottleAssessment(
            scope=ThrottleScope.CREDENTIAL,
            confidence=0.8,
            suggested_cooldown=max(max_retry_after, self.credential_cooldown),
            affected_credentials=unique_credentials,
            error_signature=common_hash,
            details={
                "credentials_throttled": num_unique_credentials,
            },
        )

    async def get_active_ip_throttles(self) -> Dict[str, ThrottleAssessment]:
        """
        Get all providers currently experiencing IP-level throttling.

        Returns:
            Dict mapping provider names to their ThrottleAssessment
        """
        result = {}
        for provider in await self._provider_names():
            assessment = await self._assess_throttle_scope(provider)
            if assessment.scope == ThrottleScope.IP:
                result[provider] = assessment
        return result

    async def clear_provider(self, provider: str) -> None:
        """Clear all records for a provider (e.g., after cooldown expires)."""
        lock = await self._locks.get_lock(provider)
        async with lock:
            async with self._records_lock:
                if provider in self._records:
                    del self._records[provider]
        lib_logger.debug(f"IPThrottleDetector: cleared records for {provider}")

    async def clear_all(self) -> None:
        """Clear all records."""
        for provider in await self._provider_names():
            lock = await self._locks.get_lock(provider)
            async with lock:
                async with self._records_lock:
                    self._records.pop(provider, None)
        lib_logger.debug("IPThrottleDetector: cleared all records")

    async def get_stats(self) -> Dict[str, Any]:
        """Get diagnostic statistics about the detector state."""
        per_provider = {}
        total_records = 0
        for provider in await self._provider_names():
            lock = await self._locks.get_lock(provider)
            async with lock:
                records = self._prune_records(self._records[provider])
                self._records[provider] = records
            unique_creds = len(set(r.credential for r in records))
            per_provider[provider] = {
                "records": len(records),
                "unique_credentials": unique_creds,
            }
            total_records += len(records)

        return {
            "providers_tracked": len(per_provider),
            "total_records": total_records,
            "window_seconds": self.window_seconds,
            "min_credentials": self.min_credentials,
            "per_provider": per_provider,
        }

