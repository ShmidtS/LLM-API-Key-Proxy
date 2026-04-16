# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
NanoGPT Quota Tracking Mixin

Provides quota tracking for the NanoGPT provider using their subscription usage API.
Unlike Gemini/Antigravity which track per-model quotas, NanoGPT tracks "usage units"
(successful operations) at the credential level with daily/monthly limits.

API Details (from https://docs.nano-gpt.com/api-reference/endpoint/subscription-usage):
- Endpoint: GET https://nano-gpt.com/api/subscription/v1/usage
- Auth: Authorization: Bearer <api_key> or x-api-key: <api_key>
- Response: { active, limits, daily, monthly, state, ... }

Required from provider:
    - self._get_api_key(credential_path) -> str
"""

import logging
import time
from typing import Any, Dict, Optional

import httpx

from .lightweight_quota_mixin import LightweightQuotaMixin

lib_logger = logging.getLogger("rotator_library")

# NanoGPT API base URL
NANOGPT_API_BASE = "https://nano-gpt.com"


class NanoGptQuotaTracker(LightweightQuotaMixin):
    """
    Mixin class providing quota tracking functionality for NanoGPT provider.

    This mixin adds the following capabilities:
    - Fetch subscription usage from the NanoGPT API
    - Track daily/monthly usage limits
    - Determine subscription tier from state field

    Usage:
        class NanoGptProvider(NanoGptQuotaTracker, ProviderInterface):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._subscription_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # Type hints for attributes from provider
    _subscription_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    # =========================================================================
    # SUBSCRIPTION USAGE API
    # =========================================================================

    async def fetch_subscription_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch subscription usage from the NanoGPT API.

        Args:
            api_key: NanoGPT API key
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "active": bool,
                "state": str,  # "active" | "grace" | "inactive"
                "limits": {"daily": int, "monthly": int},
                "daily": {
                    "used": int,
                    "remaining": int,
                    "percent_used": float,
                    "reset_at": float,  # Unix timestamp (seconds)
                },
                "monthly": {
                    "used": int,
                    "remaining": int,
                    "percent_used": float,
                    "reset_at": float,
                },
                "fetched_at": float,
            }
        """
        url = f"{NANOGPT_API_BASE}/api/subscription/v1/usage"
        headers = self._make_bearer_header(api_key)

        data = await self._fetch_json(url, headers, client)
        if data is None:
            return {
                "status": "error",
                "error": None,
                "active": False,
                "state": "unknown",
                "limits": {"daily": 0, "monthly": 0},
                "daily": {"used": 0, "remaining": 0, "percent_used": 0.0, "reset_at": 0},
                "monthly": {"used": 0, "remaining": 0, "percent_used": 0.0, "reset_at": 0},
                "fetched_at": time.time(),
            }

        # Parse response
        daily = data.get("daily", {})
        monthly = data.get("monthly", {})
        limits = data.get("limits", {})

        return {
            "status": "success",
            "error": None,
            "active": data.get("active", False),
            "state": data.get("state", "inactive"),
            "enforce_daily_limit": data.get("enforceDailyLimit", False),
            "limits": {
                "daily": limits.get("daily", 0),
                "monthly": limits.get("monthly", 0),
            },
            "daily": {
                "used": daily.get("used", 0),
                "remaining": daily.get("remaining", 0),
                "percent_used": daily.get("percentUsed", 0.0),
                # Convert epoch ms to seconds
                "reset_at": daily.get("resetAt", 0) / 1000.0,
            },
            "monthly": {
                "used": monthly.get("used", 0),
                "remaining": monthly.get("remaining", 0),
                "percent_used": monthly.get("percentUsed", 0.0),
                "reset_at": monthly.get("resetAt", 0) / 1000.0,
            },
            "fetched_at": time.time(),
        }

    def get_tier_from_state(self, state: str) -> str:
        """
        Map NanoGPT subscription state to tier name.

        Args:
            state: One of "active", "grace", "inactive"

        Returns:
            Tier name for priority mapping
        """
        state_to_tier = {
            "active": "subscription-active",
            "grace": "subscription-grace",
            "inactive": "no-subscription",
        }
        return state_to_tier.get(state, "no-subscription")

    def get_remaining_fraction(self, usage_data: Dict[str, Any]) -> float:
        """
        Calculate remaining quota fraction from usage data.

        Uses daily limit by default, unless enforceDailyLimit is False
        (in which case only monthly matters).

        Args:
            usage_data: Response from fetch_subscription_usage()

        Returns:
            Remaining fraction (0.0 to 1.0)
        """
        limits = usage_data.get("limits", {})
        daily = usage_data.get("daily", {})

        daily_limit = limits.get("daily", 0)
        daily_remaining = daily.get("remaining", 0)

        if daily_limit <= 0:
            return 1.0  # No limit configured

        return min(1.0, max(0.0, daily_remaining / daily_limit))

    # =========================================================================
    # BACKGROUND JOB SUPPORT
    # =========================================================================

    async def refresh_subscription_usage(
        self,
        api_key: str,
        credential_identifier: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Refresh and cache subscription usage for a credential.

        Args:
            api_key: NanoGPT API key
            credential_identifier: Identifier for caching
            client: Optional HTTP client for connection reuse/concurrency control

        Returns:
            Usage data from fetch_subscription_usage()
        """
        usage_data = await self.fetch_subscription_usage(api_key, client)

        if usage_data.get("status") == "success":
            self._subscription_cache[credential_identifier] = usage_data

            daily = usage_data.get("daily", {})
            limits = usage_data.get("limits", {})
            lib_logger.debug(
                f"NanoGPT subscription usage for {credential_identifier}: "
                f"daily={daily.get('remaining', 0)}/{limits.get('daily', 0)}, "
                f"state={usage_data.get('state')}"
            )

        return usage_data
