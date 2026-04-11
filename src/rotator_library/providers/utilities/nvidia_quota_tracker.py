# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
NVIDIA NIM Quota Tracker - tracks rate limits via request counting.

NVIDIA NIM doesn't provide public API for quota monitoring, so we use
sliding window counters to estimate rate limit usage.
"""

import time
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import deque
import logging

lib_logger = logging.getLogger('rotator_library')


@dataclass
class RateLimitWindow:
    """Sliding window for rate limit tracking."""
    requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    window_seconds: int = 60  # 1 minute window

    def add_request(self, timestamp: float):
        """Add request to window."""
        self.requests.append(timestamp)

    def get_request_count(self, since: float) -> int:
        """Get request count since timestamp."""
        return sum(1 for ts in self.requests if ts >= since)

    def is_within_limit(self, limit: int) -> bool:
        """Check if within rate limit."""
        now = time.time()
        window_start = now - self.window_seconds
        return self.get_request_count(window_start) < limit


class NvidiaQuotaTracker:
    """
    Track NVIDIA NIM rate limits using sliding window counters.

    NVIDIA NIM limits:
    - Free tier: 40 RPM per model
    - Paid tier: Higher limits (not publicly documented)

    Since NVIDIA doesn't provide rate limit headers, we track requests
    locally and estimate when limits are approached.
    """

    # Rate limits by tier (requests per minute)
    RATE_LIMITS = {
        "free": {"rpm": 40, "tpm": 200000},  # Estimated
        "paid": {"rpm": 500, "tpm": 2000000},  # Estimated
    }

    def __init__(self):
        # Per-credential tracking
        self._windows: Dict[str, Dict[str, RateLimitWindow]] = {}
        self._lock = asyncio.Lock()

        # Per-credential tier detection
        self._credential_tiers: Dict[str, str] = {}

        # Model quota groups (models that share rate limits)
        self._model_quota_groups: Dict[str, List[str]] = {
            "deepseek": [
                "deepseek-ai/deepseek-v3.1",
                "deepseek-ai/deepseek-v3.1-terminus",
                "deepseek-ai/deepseek-v3.2",
                "deepseek-ai/deepseek-r1",
            ],
            "qwen": [
                "qwen/qwen3.5-397b-a17b",
                "qwen/qwen3-coder-480b-a35b-instruct",
            ],
        }

    def _get_window(self, credential: str, model: str) -> RateLimitWindow:
        """Get or create rate limit window for credential+model."""
        if credential not in self._windows:
            self._windows[credential] = {}

        if model not in self._windows[credential]:
            self._windows[credential][model] = RateLimitWindow()

        return self._windows[credential][model]

    def _get_model_group(self, model: str) -> str:
        """Get quota group for model."""
        for group, models in self._model_quota_groups.items():
            if model in models:
                return group
        return model  # Use model name as group if not in any group

    def _get_rate_limit(self, credential: str, model: str) -> int:
        """Get rate limit for credential and model."""
        tier = self._credential_tiers.get(credential, "free")
        limits = self.RATE_LIMITS.get(tier, self.RATE_LIMITS["free"])
        return limits["rpm"]

    async def track_request(self, credential: str, model: str) -> None:
        """
        Track a request for rate limit enforcement.

        Args:
            credential: Credential identifier
            model: Model name (e.g., "nvidia/deepseek-ai/deepseek-v3.1")
        """
        async with self._lock:
            # Strip provider prefix if present
            if "/" in model:
                model = model.split("/", 1)[1]

            # Get model group for quota sharing
            model_group = self._get_model_group(model)

            # Track in all models within the group
            for grouped_model in self._model_quota_groups.get(model_group, [model]):
                window = self._get_window(credential, grouped_model)
                window.add_request(time.time())

    def can_make_request(self, credential: str, model: str) -> bool:
        """
        Check if request is within rate limits.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            True if request is allowed, False otherwise
        """
        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/", 1)[1]

        # Get model group for quota sharing
        model_group = self._get_model_group(model)

        # Check all models in the group
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            window = self._get_window(credential, grouped_model)
            limit = self._get_rate_limit(credential, grouped_model)

            if not window.is_within_limit(limit):
                return False

        return True

    def get_wait_time(self, credential: str, model: str) -> float:
        """
        Get seconds to wait before next request.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            Seconds to wait (0.0 if can proceed immediately)
        """
        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/", 1)[1]

        # Get model group for quota sharing
        model_group = self._get_model_group(model)

        max_wait = 0.0

        # Check all models in the group
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            window = self._get_window(credential, grouped_model)
            limit = self._get_rate_limit(credential, grouped_model)

            if not window.is_within_limit(limit):
                # Calculate when oldest request in window will expire
                if window.requests:
                    oldest = min(window.requests)
                    wait_time = (oldest + window.window_seconds) - time.time()
                    max_wait = max(max_wait, wait_time)

        return max(0.0, max_wait)

    def get_usage_stats(self, credential: str, model: str) -> Dict:
        """
        Get usage statistics for credential and model.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            Dictionary with usage statistics
        """
        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/", 1)[1]

        # Get model group for quota sharing
        model_group = self._get_model_group(model)

        stats = {
            "credential": credential,
            "model": model,
            "model_group": model_group,
            "tier": self._credential_tiers.get(credential, "free"),
            "models": {}
        }

        # Get stats for all models in the group
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            window = self._get_window(credential, grouped_model)
            limit = self._get_rate_limit(credential, grouped_model)

            now = time.time()
            window_start = now - window.window_seconds
            request_count = window.get_request_count(window_start)

            stats["models"][grouped_model] = {
                "request_count": request_count,
                "rate_limit": limit,
                "remaining": max(0, limit - request_count),
                "window_seconds": window.window_seconds,
                "is_within_limit": window.is_within_limit(limit),
            }

        return stats

    def set_credential_tier(self, credential: str, tier: str) -> None:
        """
        Set tier for credential.

        Args:
            credential: Credential identifier
            tier: Tier name ("free" or "paid")
        """
        self._credential_tiers[credential] = tier
        lib_logger.info(f"Set NVIDIA credential {credential} tier to {tier}")

    def reset_window(self, credential: str, model: str) -> None:
        """
        Reset rate limit window for credential and model.

        Args:
            credential: Credential identifier
            model: Model name
        """
        # Strip provider prefix if present
        if "/" in model:
            model = model.split("/", 1)[1]

        # Get model group for quota sharing
        model_group = self._get_model_group(model)

        # Reset all models in the group
        for grouped_model in self._model_quota_groups.get(model_group, [model]):
            if credential in self._windows and grouped_model in self._windows[credential]:
                self._windows[credential][grouped_model] = RateLimitWindow()

    def cleanup_old_windows(self, max_age_seconds: int = 3600) -> int:
        """
        Clean up old rate limit windows.

        Args:
            max_age_seconds: Maximum age of windows to keep

        Returns:
            Number of windows cleaned up
        """
        now = time.time()
        cleaned = 0

        for credential in list(self._windows.keys()):
            for model in list(self._windows[credential].keys()):
                window = self._windows[credential][model]

                # Check if window has any recent requests
                if window.requests:
                    newest = max(window.requests)
                    if now - newest > max_age_seconds:
                        del self._windows[credential][model]
                        cleaned += 1
                else:
                    # Empty window, remove it
                    del self._windows[credential][model]
                    cleaned += 1

            # Remove credential if no models left
            if not self._windows[credential]:
                del self._windows[credential]

        if cleaned > 0:
            lib_logger.debug(f"Cleaned up {cleaned} old rate limit windows")

        return cleaned
