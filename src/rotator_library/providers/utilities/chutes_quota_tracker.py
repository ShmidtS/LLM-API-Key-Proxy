# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Chutes Quota Tracking Mixin

Provides quota tracking for the Chutes provider using their quota usage API.
Chutes tracks credit-based quotas at the credential level with daily limits:
- 1 request = 1 credit consumed
- Daily quota reset at 00:00 UTC

API Details:
- Endpoint: GET https://api.chutes.ai/users/me/quota_usage/me
- Auth: Authorization: Bearer <api_key>
- Response: { quota: int, used: float }

Required from provider:
    - self._get_api_key(credential_path) -> str
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import httpx

from .simple_quota_tracker import SimpleQuotaTrackerBase

lib_logger = logging.getLogger("rotator_library")

# Chutes API endpoint
CHUTES_QUOTA_API_URL = "https://api.chutes.ai/users/me/quota_usage/me"


class ChutesQuotaTracker(SimpleQuotaTrackerBase):
    """
    Mixin class providing quota tracking functionality for Chutes provider.

    Inherits shared cache/pool boilerplate from SimpleQuotaTrackerBase.
    """

    _quota_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # Tier thresholds
    TIER_THRESHOLDS = {200: "legacy", 300: "base", 2000: "plus", 5000: "pro"}

    # =========================================================================
    # QUOTA USAGE API
    # =========================================================================

    async def fetch_quota_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch quota usage from the Chutes API.

        Args:
            api_key: Chutes API key
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "quota": int,  # Total daily quota
                "used": float,  # Credits consumed today
                "remaining": float,  # Credits remaining
                "remaining_fraction": float,  # 0.0 to 1.0
                "tier": str,  # legacy/base/plus/pro
                "reset_at": float,  # Unix timestamp (seconds)
                "fetched_at": float,
            }
        """
        try:
            headers = self._make_bearer_header(api_key)
            response = await self._fetch_via_pool(
                CHUTES_QUOTA_API_URL, headers, client
            )
            response.raise_for_status()
            data = response.json()

            # Parse response with null safety
            quota = data.get("quota") or 0
            used = data.get("used") or 0.0
            remaining = max(0.0, quota - used)
            remaining_fraction = (remaining / quota) if quota > 0 else 0.0

            # Detect tier from quota value
            tier = self._get_tier_from_quota(quota)

            # Calculate next reset (00:00 UTC)
            reset_at = self._calculate_next_reset()

            return {
                "status": "success",
                "error": None,
                "quota": quota,
                "used": used,
                "remaining": remaining,
                "remaining_fraction": remaining_fraction,
                "tier": tier,
                "reset_at": reset_at,
                "fetched_at": time.time(),
            }

        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                httpx.WriteTimeout, httpx.PoolTimeout, httpx.ConnectTimeout) as e:
            lib_logger.warning(f"Transient error fetching Chutes quota: {e}")
            return {
                "status": "transient_error",
                "error": str(e),
                "quota": 0,
                "used": 0.0,
                "remaining": None,
                "remaining_fraction": None,
                "tier": "base",
                "reset_at": 0,
                "fetched_at": time.time(),
            }
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.text
                if error_body:
                    error_msg = f"{error_msg}: {error_body[:200]}"
            except Exception:
                pass
            lib_logger.warning(f"Failed to fetch Chutes quota: {error_msg}")
            return {
                "status": "error",
                "error": error_msg,
                "quota": 0,
                "used": 0.0,
                "remaining": 0.0,
                "remaining_fraction": 0.0,
                "tier": "base",
                "reset_at": 0,
                "fetched_at": time.time(),
            }
        except Exception as e:
            lib_logger.warning(f"Failed to fetch Chutes quota: {e}")
            return {
                "status": "error",
                "error": str(e),
                "quota": 0,
                "used": 0.0,
                "remaining": 0.0,
                "remaining_fraction": 0.0,
                "tier": "base",
                "reset_at": 0,
                "fetched_at": time.time(),
            }

    def _get_tier_from_quota(self, quota: int) -> str:
        """
        Map Chutes quota value to tier name.

        Args:
            quota: Daily quota value (200, 300, 2000, or 5000)

        Returns:
            Tier name (legacy, base, plus, or pro)
        """
        tier = self.TIER_THRESHOLDS.get(quota)
        if tier is None:
            lib_logger.warning(
                f"Unknown Chutes quota value {quota}, defaulting to 'base' tier. "
                f"Known values: {list(self.TIER_THRESHOLDS.keys())}"
            )
            return "base"
        return tier

    def _calculate_next_reset(self) -> float:
        """
        Calculate next 00:00 UTC reset timestamp.

        Returns:
            Unix timestamp when quota resets
        """
        now = datetime.now(timezone.utc)
        next_reset = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return next_reset.timestamp()
