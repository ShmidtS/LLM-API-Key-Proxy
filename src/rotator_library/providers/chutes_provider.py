# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging
import time
from datetime import datetime, timezone, timedelta

import httpx
from typing import Any, Dict, List, Optional
from .provider_interface import ProviderInterface, UsageResetConfigDef, build_bearer_headers
from .utilities import fetch_provider_models
from .utilities.lightweight_quota_mixin import LightweightQuotaMixin
from ..config.defaults import env_int

lib_logger = logging.getLogger("rotator_library")


class ChutesProvider(LightweightQuotaMixin, ProviderInterface):
    """
    Provider implementation for the chutes.ai API with quota tracking.
    """

    _virtual_model_name = "chutes/_quota"
    provider_name = "chutes"
    _include_max_requests = True

    # Enable environment variable overrides (e.g., QUOTA_GROUPS_CHUTES_GLOBAL)
    provider_env_name = "chutes"

    # Quota groups for tracking daily limits
    # Uses a virtual model "_quota" for credential-level quota tracking
    model_quota_groups = {
        "chutes_global": ["_quota"],
    }

    # Usage reset configuration for daily quota
    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=86400,  # 24 hours (daily quota reset)
            mode="per_model",
            description="Chutes daily quota",
            field_name="daily",
        )
    }

    def __init__(self, *args, **kwargs):
        """Initialize ChutesProvider with quota tracking."""
        super().__init__(*args, **kwargs)

        # Quota tracking cache and refresh interval
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = env_int(
            "CHUTES_QUOTA_REFRESH_INTERVAL", 300
        )

    # --- Quota tracking (inlined from removed ChutesQuotaTracker) ---

    async def fetch_quota_usage(
        self, api_key: str, client: Optional[httpx.AsyncClient] = None
    ) -> Dict[str, Any]:
        headers = self._make_bearer_header(api_key)
        data = await self._fetch_json(
            "https://api.chutes.ai/users/me/quota_usage/me", headers, client
        )
        if data is None:
            return self._error_result(
                quota=0, used=0.0, remaining=None, remaining_fraction=None, tier="base", reset_at=0
            )

        quota = data.get("quota") or 0
        used = data.get("used") or 0.0
        remaining = max(0.0, quota - used)
        remaining_fraction = (remaining / quota) if quota > 0 else 0.0
        tier = self._get_tier_from_quota(quota)
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

    def _get_tier_from_quota(self, quota: int) -> str:
        thresholds = {200: "legacy", 300: "base", 2000: "plus", 5000: "pro"}
        tier = thresholds.get(quota)
        if tier is None:
            lib_logger.warning(
                f"Unknown Chutes quota value {quota}, defaulting to 'base' tier. "
                f"Known values: {list(thresholds.keys())}"
            )
            return "base"
        return tier

    def _calculate_next_reset(self) -> float:
        now = datetime.now(timezone.utc)
        next_reset = (now + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        return next_reset.timestamp()

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All Chutes models share the same credential-level quota pool,
        so they all belong to the same quota group.

        Args:
            model: Model name (ignored - all models share quota)

        Returns:
            Quota group identifier for shared credential-level tracking
        """
        return "chutes_global"

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from the Chutes API.

        Args:
            api_key: Chutes API key
            client: HTTP client

        Returns:
            List of model names prefixed with 'chutes/'
        """
        return await fetch_provider_models(
            client,
            "https://llm.chutes.ai/v1/models",
            build_bearer_headers(api_key, content_type=None),
            "chutes.ai",
            lambda data: [
                f"chutes/{model['id']}" for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ],
        )

