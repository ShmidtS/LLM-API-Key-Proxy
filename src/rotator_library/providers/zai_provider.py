# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import os
import re
from ..utils.json_utils import json_loads
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_streaming_provider import QuotaRefreshMixin
from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.zai_quota_tracker import ZaiQuotaTracker

ZAI_DEFAULT_API_BASE = "https://api.z.ai/api/coding/paas/v4"

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

import logging

lib_logger = logging.getLogger("rotator_library")


class ZaiProvider(QuotaRefreshMixin, ZaiQuotaTracker, ProviderInterface):
    """
    Provider implementation for the ZAI (z.ai) API with quota tracking.

    ZAI uses hourly request quotas with three tiers:
    - lite: 100 requests/hour
    - pro: 1000 requests/hour
    - max: 4000 requests/hour

    Quota monitoring API: GET https://api.z.ai/api/monitor/usage/quota/limit
    Completion API: OpenAI-compatible, default https://api.z.ai/api/coding/paas/v4
    """

    _virtual_model_name = "zai/_quota"
    provider_name = "zai"
    _include_max_requests = True

    provider_env_name = "zai"

    tier_priorities = {
        "max": 1,
        "pro": 2,
        "lite": 3,
    }

    model_quota_groups = {
        "zai_global": ["_quota"],
    }

    usage_reset_configs = {
        "default": UsageResetConfigDef(
            window_seconds=3600,
            mode="per_model",
            description="ZAI hourly quota",
            field_name="hourly",
        )
    }

    # Sequential rotation: use one key until exhausted, then switch
    default_rotation_mode = "sequential"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval: int = int(
            os.environ.get("ZAI_QUOTA_REFRESH_INTERVAL", "300")
        )
        self._tier_cache: Dict[str, str] = {}
        self._known_models: List[str] = []
        self.api_base: str = os.environ.get(
            "ZAI_API_BASE", ZAI_DEFAULT_API_BASE
        ).rstrip("/")

    def get_credential_tier_name(self, credential: str) -> Optional[str]:
        cached = self._tier_cache.get(credential)
        if cached:
            return cached
        usage = self._quota_cache.get(credential)
        if usage and usage.get("status") == "success":
            level = usage.get("level", "lite")
            self._tier_cache[credential] = level
            return level
        return None

    def get_model_quota_group(self, model: str) -> Optional[str]:
        # All ZAI models share a single hourly quota — every model
        # maps to zai_global so that cooldowns and baseline_remaining_fraction
        # set by the background refresh on zai/_quota propagate to
        # every real model (e.g. zai/glm-5.1).
        return "zai_global"

    @classmethod
    def parse_quota_error(
        cls, error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse ZAI 429 error to extract quota reset information.

        ZAI returns rate limit errors when hourly quota is exhausted.
        Error format: {'error': {'code': '1113', 'message': 'Insufficient balance...'}}
        """
        body = cls._extract_error_body(error, error_body) or ""
        try:
            data = json_loads(body) if body else {}
        except (ValueError, TypeError):
            data = {}

        # ZAI nests error info under 'error' key
        error_obj = data.get("error", {})
        code = data.get("code") or error_obj.get("code")
        msg = str(data.get("msg", "") or data.get("message", "") or error_obj.get("message", "")).lower()

        # Normalize code to int for comparison (API returns string codes like "1113")
        try:
            code_int = int(code) if code is not None else None
        except (ValueError, TypeError):
            code_int = None

        # ZAI-specific: code 1113 = insufficient balance, plus standard rate/limit/quota keywords
        is_quota = (
            code_int == 429
            or code_int == 1113
            or "rate" in msg
            or "limit" in msg
            or "quota" in msg
            or "balance" in msg
            or "recharge" in msg
            or "insufficient" in msg
        )
        if not is_quota:
            return None

        now = datetime.now(timezone.utc)

        # Code 1113 / "insufficient balance" = account has no funds,
        # won't recover at hour boundary — use 24h cooldown.
        # Standard 429 = hourly quota, resets at next hour boundary.
        if code_int == 1113 or "balance" in msg or "recharge" in msg:
            cooldown = timedelta(hours=24)
            reset_time = now + cooldown
            retry_after = int(cooldown.total_seconds())
            reason = "INSUFFICIENT_BALANCE"
        else:
            next_hour = (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )
            reset_time = next_hour
            retry_after = max(60, int(next_hour.timestamp() - now.timestamp()))
            reason = "QUOTA_EXHAUSTED"

        reset_ts = reset_time.timestamp()

        return {
            "retry_after": retry_after,
            "reason": reason,
            "reset_timestamp": reset_time.isoformat(),
            "quota_reset_timestamp": reset_ts,
        }

    def get_models_in_quota_group(self, group: str) -> List[str]:
        if group == "zai_global":
            return ["_quota"] + self._known_models
        return []

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            response = await client.get(
                f"{self.api_base}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            response.raise_for_status()
            models = [
                model["id"] for model in response.json().get("data", [])
            ]
            # Cache bare model IDs so get_models_in_quota_group can
            # propagate cooldowns/baselines to every real model.
            if models and not self._known_models:
                self._known_models = models
            return [f"zai/{m}" for m in models]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch ZAI models: {e}")
            return []

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        return {
            "interval": self._quota_refresh_interval,
            "name": "zai_quota_refresh",
            "run_on_start": True,
        }
