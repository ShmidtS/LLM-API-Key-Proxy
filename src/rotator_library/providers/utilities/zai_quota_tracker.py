# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

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

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ...error_handler import mask_credential
from ...http_client_pool import get_http_pool

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


class ZaiQuotaTracker:
    """
    Mixin class providing quota tracking functionality for ZAI provider.

    Usage:
        class ZaiProvider(QuotaRefreshMixin, ZaiQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300
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
                "quota": int,  # alias for hourly_limit (QuotaRefreshMixin compat)
                "used": float,  # alias for hourly_used
                "remaining": float,  # alias for hourly_remaining
                "reset_at": float,  # Unix timestamp (next hour boundary)
                "fetched_at": float,
            }
        """
        try:
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            }

            if client is not None:
                response = await client.get(
                    ZAI_QUOTA_API_URL, headers=headers, timeout=30
                )
            else:
                pool = await get_http_pool()
                new_client = await pool.get_client_async()
                response = await new_client.get(
                    ZAI_QUOTA_API_URL, headers=headers, timeout=30
                )
            response.raise_for_status()
            data = response.json()

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

        except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadTimeout,
                httpx.WriteTimeout, httpx.PoolTimeout, httpx.ConnectTimeout) as e:
            lib_logger.warning(f"Transient error fetching ZAI quota: {e}")
            return {
                "status": "transient_error",
                "error": str(e),
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
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP {e.response.status_code}"
            try:
                error_body = e.response.text
                if error_body:
                    error_msg = f"{error_msg}: {error_body[:200]}"
            except Exception:
                pass
            lib_logger.warning(f"Failed to fetch ZAI quota: {error_msg}")
            return self._error_result(error_msg)
        except Exception as e:
            lib_logger.warning(f"Failed to fetch ZAI quota: {e}")
            return self._error_result(str(e))

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

    def get_remaining_fraction(self, usage_data: Dict[str, Any]) -> float:
        return usage_data.get("remaining_fraction", 0.0)

    def get_reset_timestamp(self, usage_data: Dict[str, Any]) -> Optional[float]:
        reset_at = usage_data.get("reset_at", 0)
        return reset_at if reset_at > 0 else None

    async def refresh_quota_usage(
        self,
        api_key: str,
        credential_identifier: str,
    ) -> Dict[str, Any]:
        """Refresh and cache quota usage for a credential."""
        usage_data = await self.fetch_quota_usage(api_key)

        if usage_data.get("status") == "success":
            self._quota_cache[credential_identifier] = usage_data
            level = usage_data.get("level")
            if level:
                self._tier_cache[credential_identifier] = level
            lib_logger.debug(
                f"ZAI quota for {mask_credential(credential_identifier)}: "
                f"{usage_data['hourly_remaining']}/{usage_data['hourly_limit']} remaining "
                f"({usage_data['remaining_fraction'] * 100:.1f}%), "
                f"level={usage_data['level']}"
            )

        return usage_data

    def get_cached_usage(self, credential_identifier: str) -> Optional[Dict[str, Any]]:
        return self._quota_cache.get(credential_identifier)

    async def get_all_quota_info(
        self,
        api_keys: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Get quota info for all credentials in parallel."""
        results = {}
        total_quota = 0
        total_used = 0.0
        total_remaining = 0.0

        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(
            identifier: str, api_key: str, client: httpx.AsyncClient
        ):
            async with semaphore:
                return identifier, await self.fetch_quota_usage(api_key, client)

        pool = await get_http_pool()
        client = await pool.get_client_async()
        tasks = [
            fetch_with_semaphore(ident, key, client) for ident, key in api_keys
        ]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"ZAI quota fetch failed: {result}")
                continue

            identifier, usage_data = result

            if usage_data.get("status") == "success":
                total_quota += usage_data.get("hourly_limit", 0)
                total_used += usage_data.get("hourly_used", 0)
                total_remaining += usage_data.get("hourly_remaining", 0)

            results[identifier] = {
                "identifier": identifier,
                "level": usage_data.get("level"),
                "status": usage_data.get("status", "error"),
                "error": usage_data.get("error"),
                "hourly_limit": usage_data.get("hourly_limit"),
                "hourly_used": usage_data.get("hourly_used"),
                "hourly_remaining": usage_data.get("hourly_remaining"),
                "remaining_fraction": usage_data.get("remaining_fraction"),
                "pct_5min": usage_data.get("pct_5min"),
                "pct_daily": usage_data.get("pct_daily"),
                "reset_at": usage_data.get("reset_at"),
                "fetched_at": usage_data.get("fetched_at"),
            }

        return {
            "credentials": results,
            "summary": {
                "total_credentials": len(api_keys),
                "total_quota": total_quota,
                "total_used": total_used,
                "total_remaining": total_remaining,
            },
            "timestamp": time.time(),
        }
