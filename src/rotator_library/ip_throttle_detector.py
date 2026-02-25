# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
IP Throttle Detector - Detects IP-level throttling via correlation of 429 errors.

This module analyzes 429 (rate limit) errors across multiple credentials to detect
when throttling is applied at the IP level rather than per-credential.

Detection heuristics:
- 3+ different credentials receiving 429 errors within a 30-second window
- Identical error_body hash between credentials (same error from same IP)
"""

import hashlib
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

lib_logger = logging.getLogger("rotator_library")


class ThrottleScope(Enum):
    """Scope of detected throttling."""

    CREDENTIAL = "credential"  # Per-credential rate limit
    IP = "ip"  # IP-level rate limit (affects all credentials from this IP)
    ACCOUNT = "account"  # Account-level rate limit (affects all keys in account)


@dataclass
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

    scope: ThrottleScope
    confidence: float = 0.0
    suggested_cooldown: int = 0
    affected_credentials: List[str] = field(default_factory=list)
    error_signature: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ThrottleAssessment(scope={self.scope.value}, "
            f"confidence={self.confidence:.2f}, "
            f"cooldown={self.suggested_cooldown}s, "
            f"affected={len(self.affected_credentials)} creds)"
        )


@dataclass
class _ThrottleRecord:
    """Internal record for tracking 429 events."""

    timestamp: float
    credential: str
    error_body_hash: Optional[str]
    retry_after: Optional[int]
    error_body: Optional[str] = None


class IPThrottleDetector:
    """
    Detects IP-level throttling by correlating 429 errors across credentials.

    When multiple credentials from the same IP receive 429 errors simultaneously,
    it indicates IP-level throttling rather than per-credential limits.

    Usage:
        detector = IPThrottleDetector()

        # Record a 429 error
        assessment = detector.record_429(
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
    DEFAULT_MIN_CREDENTIALS = 2
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

        # Per-provider tracking: provider -> list of _ThrottleRecord
        self._records: Dict[str, List[_ThrottleRecord]] = defaultdict(list)

        # Cache of recent assessments to avoid repeated computation
        self._assessment_cache: Dict[str, tuple] = {}

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

    def _cleanup_old_records(self, provider: str) -> None:
        """Remove records older than the detection window and enforce memory limit."""
        cutoff = time.time() - self.window_seconds
        self._records[provider] = [
            r for r in self._records[provider] if r.timestamp > cutoff
        ]
        # FIFO eviction if records exceed memory limit
        if len(self._records[provider]) > self.MAX_RECORDS_PER_PROVIDER:
            self._records[provider] = self._records[provider][
                -self.MAX_RECORDS_PER_PROVIDER :
            ]

    def record_429(
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
        now = time.time()
        error_body_hash = self._hash_error_body(error_body)

        # Create and store the record
        record = _ThrottleRecord(
            timestamp=now,
            credential=credential,
            error_body_hash=error_body_hash,
            retry_after=retry_after,
            error_body=error_body[:500] if error_body else None,  # Truncate for storage
        )
        self._records[provider].append(record)

        # Cleanup old records
        self._cleanup_old_records(provider)

        # Assess throttle scope
        assessment = self._assess_throttle_scope(provider)

        # Override cooldown with retry_after if provided
        if retry_after and retry_after > assessment.suggested_cooldown:
            assessment.suggested_cooldown = retry_after

        lib_logger.debug(
            f"IPThrottleDetector.record_429: provider={provider}, "
            f"credential={credential}, assessment={assessment}"
        )

        return assessment

    def _assess_throttle_scope(self, provider: str) -> ThrottleAssessment:
        """
        Assess the scope of throttling for a provider.

        Analyzes recent 429 records to determine if throttling is:
        - Per-credential (normal rate limit)
        - IP-level (affects all credentials)
        - Account-level (affects all keys in account)

        Detection heuristics:
        1. 3+ different credentials with 429 in window -> IP throttle (high confidence)
        2. Same error_body_hash across credentials -> Same throttle source
        3. Single credential with 429 -> Credential-level throttle

        Args:
            provider: Provider name to assess

        Returns:
            ThrottleAssessment with detected scope and recommendations
        """
        records = self._records[provider]

        if not records:
            return ThrottleAssessment(
                scope=ThrottleScope.CREDENTIAL,
                confidence=1.0,
                suggested_cooldown=self.credential_cooldown,
            )

        # Get unique credentials
        unique_credentials = list(set(r.credential for r in records))
        num_unique_credentials = len(unique_credentials)

        # Analyze error body hashes
        hash_counts: Dict[Optional[str], int] = defaultdict(int)
        for r in records:
            hash_counts[r.error_body_hash] += 1

        # Find the most common error hash
        most_common_hash = max(hash_counts.items(), key=lambda x: x[1])
        common_hash, common_hash_count = most_common_hash

        # Get the maximum retry_after from recent records
        max_retry_after = max((r.retry_after or 0) for r in records)

        # Calculate confidence based on correlation strength
        # IP throttle detection
        if num_unique_credentials >= self.min_credentials:
            # Multiple credentials throttled -> likely IP-level
            confidence = min(1.0, num_unique_credentials / self.min_credentials)

            # Higher confidence if same error body
            if common_hash and common_hash_count >= 2:
                confidence = min(1.0, confidence + 0.2)

            # Even higher confidence if same error across ALL credentials
            if common_hash and common_hash_count == len(records):
                confidence = min(1.0, confidence + 0.1)

            lib_logger.info(
                f"IP-level throttle detected: provider={provider}, "
                f"credentials={num_unique_credentials}, confidence={confidence:.2f}"
            )

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

        # Check for error body correlation with fewer credentials
        if num_unique_credentials == 2 and common_hash and common_hash_count >= 2:
            # Two credentials with same error -> likely same throttle source
            confidence = 0.7

            lib_logger.info(
                f"Possible IP-level throttle: provider={provider}, "
                f"credentials={num_unique_credentials}, same_error_body"
            )

            return ThrottleAssessment(
                scope=ThrottleScope.IP,
                confidence=confidence,
                suggested_cooldown=max(max_retry_after, self.ip_cooldown),
                affected_credentials=unique_credentials,
                error_signature=common_hash,
                details={
                    "credentials_throttled": num_unique_credentials,
                    "error_hash_matches": common_hash_count,
                    "detection_type": "error_body_correlation",
                },
            )

        # Single credential throttled -> credential-level
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

    def get_active_ip_throttles(self) -> Dict[str, ThrottleAssessment]:
        """
        Get all providers currently experiencing IP-level throttling.

        Returns:
            Dict mapping provider names to their ThrottleAssessment
        """
        result = {}
        for provider in list(self._records.keys()):
            self._cleanup_old_records(provider)
            if self._records[provider]:
                assessment = self._assess_throttle_scope(provider)
                if assessment.scope == ThrottleScope.IP:
                    result[provider] = assessment
        return result

    def clear_provider(self, provider: str) -> None:
        """Clear all records for a provider (e.g., after cooldown expires)."""
        if provider in self._records:
            del self._records[provider]
        lib_logger.debug(f"IPThrottleDetector: cleared records for {provider}")

    def clear_all(self) -> None:
        """Clear all records."""
        self._records.clear()
        self._assessment_cache.clear()
        lib_logger.debug("IPThrottleDetector: cleared all records")

    def get_stats(self) -> Dict[str, Any]:
        """Get diagnostic statistics about the detector state."""
        stats = {
            "providers_tracked": len(self._records),
            "total_records": sum(len(r) for r in self._records.values()),
            "window_seconds": self.window_seconds,
            "min_credentials": self.min_credentials,
            "per_provider": {},
        }

        for provider, records in self._records.items():
            unique_creds = len(set(r.credential for r in records))
            stats["per_provider"][provider] = {
                "records": len(records),
                "unique_credentials": unique_creds,
            }

        return stats


# Singleton instance for global use
_detector_instance: Optional[IPThrottleDetector] = None


def get_ip_throttle_detector() -> IPThrottleDetector:
    """Get the global IP throttle detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = IPThrottleDetector()
    return _detector_instance


def reset_ip_throttle_detector() -> None:
    """Reset the global IP throttle detector (mainly for testing)."""
    global _detector_instance
    if _detector_instance:
        _detector_instance.clear_all()
    _detector_instance = None
