# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
ZAI Quota Tracking Mixin

Provides quota tracking for the ZAI provider using their quota monitoring API.
ZAI tracks hourly request limits at the credential level with tier-based quotas:
- lite: 100 requests/hour
- pro: 1000 requests/hour
- max: 4000 requests/hour

API Details:
- Endpoint: GET https://api.z.ai/api/monitor/usage/quota/limit
- Auth: Authorization: Bearer <api_key>
- Response: {"code": 200, "data": {"level": "lite"|"pro"|"max", "limits": [...]}}

Required from provider:
    - self._get_api_key(credential_path) -> str
"""

import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import httpx

from .lightweight_quota_mixin import LightweightQuotaMixin

lib_logger = logging.getLogger("rotator_library")

ZAI_QUOTA_API_URL = "https://api.z.ai/api/monitor/usage/quota/limit"

ZAI_TIER_HOURLY_LIMITS = {
    "lite": 100,
    "pro": 1000,
    "max": 4000,
}

# ZAI limit unit codes (from API documentation)
ZAI_UNIT_5MIN_TOKENS = 3    # 5-minute token usage percentage
ZAI_UNIT_HOURLY_TIME = 5   # Hourly time-based request count
ZAI_UNIT_DAILY_TOKENS = 6  # Daily token usage percentage


class ZaiQuotaTracker(LightweightQuotaMixin):
    """
    Mixin class providing quota tracking functionality for ZAI provider.

    Inherits shared cache/pool boilerplate from LightweightQuotaMixin.
    """

    _quota_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int
    _tier_cache: Dict[str, str]

    async def fetch_quota_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch quota usage from the ZAI monitoring API.

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "level": str,  # lite/pro/max
                "hourly_used": int,
                "hourly_limit": int,
                "hourly_remaining": int,
                "remaining_fraction": float,  # 0.0 to 1.0 (hourly)
                "pct_5min": float,  # 5-min usage percentage
                "pct_daily": float,  # daily usage percentage
                "quota": int,  # alias for hourly_limit (LightweightQuotaMixin compat)
                "used": float,  # alias for hourly_used
                "remaining": float,  # alias for hourly_remaining
                "reset_at": float,  # Unix timestamp (next hour boundary)
                "fetched_at": float,
            }
        """
        headers = self._make_bearer_header(api_key)
        data = await self._fetch_json(ZAI_QUOTA_API_URL, headers, client)
        if data is None:
            return {
                "status": "transient_error",
                "error": "empty_or_invalid_response",
                "level": "lite",
                "hourly_used": 0,
                "hourly_limit": 0,
                "hourly_remaining": 0,
                "remaining_fraction": None,
                "pct_5min": 0.0,
                "pct_daily": 0.0,
                "quota": 0,
                "used": 0.0,
                "remaining": None,
                "reset_at": 0,
                "fetched_at": time.time(),
            }

        if data.get("code") != 200:
            error_msg = data.get("msg", "unknown error")
            return self._error_result(f"API_ERROR: {error_msg}")

        quota_data = data.get("data")
        if not quota_data:
            return self._error_result("NO_DATA")

        level = quota_data.get("level", "lite")
        limits = quota_data.get("limits", [])

        pct_5min = 0.0
        pct_daily = 0.0
        hourly_used = 0
        hourly_limit = 0

        for lim in limits:
            lim_type = lim.get("type", "")
            unit = lim.get("unit", 0)
            if lim_type == "TOKENS_LIMIT" and unit == ZAI_UNIT_5MIN_TOKENS:
                pct_5min = float(lim.get("percentage", 0))
            elif lim_type == "TOKENS_LIMIT" and unit == ZAI_UNIT_DAILY_TOKENS:
                pct_daily = float(lim.get("percentage", 0))
            elif lim_type == "TIME_LIMIT" and unit == ZAI_UNIT_HOURLY_TIME:
                hourly_used = int(lim.get("currentValue", 0))
                hourly_limit = int(lim.get("usage", 0))

        if hourly_limit <= 0:
            hourly_limit = ZAI_TIER_HOURLY_LIMITS.get(level, 100)

        hourly_remaining = max(0, hourly_limit - hourly_used)
        remaining_fraction = (
            (hourly_remaining / hourly_limit) if hourly_limit > 0 else 0.0
        )

        reset_at = self._calculate_next_hour_reset()

        return {
            "status": "success",
            "error": None,
            "level": level,
            "hourly_used": hourly_used,
            "hourly_limit": hourly_limit,
            "hourly_remaining": hourly_remaining,
            "remaining_fraction": remaining_fraction,
            "pct_5min": pct_5min,
            "pct_daily": pct_daily,
            "quota": hourly_limit,
            "used": float(hourly_used),
            "remaining": float(hourly_remaining),
            "reset_at": reset_at,
            "fetched_at": time.time(),
        }

    def _error_result(self, error: str) -> Dict[str, Any]:
        # remaining_fraction=0.0 on error: treat unknown state as exhausted
        # so that baseline filtering skips the credential rather than
        # routing requests to a potentially dead key.
        return {
            "status": "error",
            "error": error,
            "level": "lite",
            "hourly_used": 100,
            "hourly_limit": 100,
            "hourly_remaining": 0,
            "remaining_fraction": 0.0,
            "pct_5min": 0.0,
            "pct_daily": 0.0,
            "quota": 100,
            "used": 100.0,
            "remaining": 0.0,
            "reset_at": 0,
            "fetched_at": time.time(),
        }

    def _calculate_next_hour_reset(self) -> float:
        """Calculate timestamp of the next hour boundary (when hourly quota resets)."""
        now = datetime.now(timezone.utc)
        next_hour = (now + timedelta(hours=1)).replace(
            minute=0, second=0, microsecond=0
        )
        return next_hour.timestamp()
