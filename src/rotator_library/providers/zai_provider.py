# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
import json as json_lib
import os
from ..utils.json_utils import json_loads
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.zai_quota_tracker import ZaiQuotaTracker
from ..config.defaults import env_int

ZAI_DEFAULT_API_BASE = "https://api.z.ai/api/coding/paas/v4"

import logging

lib_logger = logging.getLogger("rotator_library")


class ZaiProvider(ZaiQuotaTracker, ProviderInterface):
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

    # Body keyword patterns for quota detection (base class fallback).
    # The full override handles ZAI-specific code matching + dynamic cooldown.
    _quota_error_patterns = [
        ("body", "insufficient balance", 0, "INSUFFICIENT_BALANCE"),
        ("body", "recharge", 0, "INSUFFICIENT_BALANCE"),
        ("body", "rate", 3600, "QUOTA_EXHAUSTED"),
        ("body", "limit", 3600, "QUOTA_EXHAUSTED"),
        ("body", "quota", 3600, "QUOTA_EXHAUSTED"),
    ]

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
        self._quota_refresh_interval: int = env_int(
            "ZAI_QUOTA_REFRESH_INTERVAL", 300
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
        """Parse ZAI 429 error to extract quota reset information.

        Overrides base class because _quota_error_patterns cannot express:
        - ZAI-specific numeric error codes (1113 = insufficient balance, 429 = hourly quota)
        - Dynamic retry_after calculation based on wall-clock boundaries (next midnight UTC, next hour)
        - Computed reset_timestamp and quota_reset_timestamp from datetime arithmetic
        (Other providers parse Google RPC details; ZAI uses a flat JSON structure with numeric codes)
        """
        body = cls._extract_error_body(error, error_body) or ""
        try:
            data = json_loads(body) if body else {}
        except (ValueError, TypeError):
            data = {}

        error_obj = data.get("error", {})
        code = data.get("code") or error_obj.get("code")
        msg = str(
            data.get("msg", "") or data.get("message", "") or error_obj.get("message", "")
        ).lower()

        try:
            code_int = int(code) if code is not None else None
        except (ValueError, TypeError):
            code_int = None

        # ZAI-specific: code 1113 / balance keywords = insufficient balance
        # Standard: code 429 or rate/limit/quota keywords = hourly quota
        is_insufficient = code_int == 1113 or "balance" in msg or "recharge" in msg
        is_quota = (
            code_int == 429 or code_int == 1113
            or any(kw in msg for kw in ("rate", "limit", "quota", "balance", "recharge", "insufficient"))
        )
        if not is_quota:
            return None

        now = datetime.now(timezone.utc)

        if is_insufficient:
            next_day = (now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            reset_time = next_day
            retry_after = max(60, int(next_day.timestamp() - now.timestamp()))
            reason = "INSUFFICIENT_BALANCE"
        else:
            next_hour = (now + timedelta(hours=1)).replace(
                minute=0, second=0, microsecond=0
            )
            reset_time = next_hour
            retry_after = max(60, int(next_hour.timestamp() - now.timestamp()))
            reason = "QUOTA_EXHAUSTED"

        return {
            "retry_after": retry_after,
            "reason": reason,
            "reset_timestamp": reset_time.isoformat(),
            "quota_reset_timestamp": reset_time.timestamp(),
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
            try:
                data = response.json()
            except (json_lib.JSONDecodeError, ValueError) as e:
                lib_logger.warning(f"Invalid JSON from ZAI models: {e}, body={response.text[:200]}")
                return []
            models = [
                model["id"] for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ]
            # Cache bare model IDs so get_models_in_quota_group can
            # propagate cooldowns/baselines to every real model.
            if models and not self._known_models:
                self._known_models = models
            return [f"zai/{m}" for m in models]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                lib_logger.warning(f"Auth error fetching ZAI models: {e.response.status_code}")
            elif e.response.status_code >= 500:
                lib_logger.warning(f"Server error fetching ZAI models: {e.response.status_code}")
            else:
                lib_logger.error(f"HTTP error fetching ZAI models: {e}")
            return []
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch ZAI models: {e}")
            return []

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {credential_identifier}"}

    # --- ZAI-specific API methods (non-litellm) ---

    async def _forward_request(
        self,
        credential: str,
        client: httpx.AsyncClient,
        path: str,
        *,
        method: str = "post",
        params: Optional[Dict[str, str]] = None,
        timeout: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """Shared helper for ZAI API requests with JSON error handling."""
        headers = {"Authorization": f"Bearer {credential}"}
        if method == "post":
            response = await client.post(
                f"{self.api_base}/{path}",
                headers=headers,
                json=kwargs,
                timeout=timeout,
            )
        else:
            response = await client.get(
                f"{self.api_base}/{path}",
                headers=headers,
                params=params,
                timeout=timeout,
            )
        response.raise_for_status()
        try:
            return response.json()
        except (json_lib.JSONDecodeError, ValueError) as e:
            body_preview = response.text[:200] if response.text else "<empty>"
            lib_logger.warning(
                "Invalid JSON response from %s: %s — body: %s",
                self.provider_name, e, body_preview,
            )
            raise

    async def video_generate(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Submit an async video generation request to ZAI."""
        return await self._forward_request(
            credential, client, "video/generate", timeout=60, **kwargs
        )

    async def video_status(
        self, credential: str, client: httpx.AsyncClient, video_id: str
    ) -> Dict[str, Any]:
        """Check the status of an async video generation task."""
        return await self._forward_request(
            credential, client,
            f"video/{video_id}/status",
            method="get",
        )

    async def async_image_generate(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Submit an async image generation request to ZAI."""
        return await self._forward_request(
            credential, client, "images/generations", timeout=60, **kwargs
        )

    async def async_image_status(
        self, credential: str, client: httpx.AsyncClient, image_id: str
    ) -> Dict[str, Any]:
        """Retrieve status/result of an async image generation task."""
        return await self._forward_request(
            credential, client,
            f"images/{image_id}",
            method="get",
        )

    async def tool_tokenizer(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Call the ZAI tokenizer tool."""
        return await self._forward_request(
            credential, client, "tools/tokenizer", **kwargs
        )

    async def tool_layout_parsing(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Call the ZAI layout parsing tool."""
        return await self._forward_request(
            credential, client, "tools/layout-parsing", timeout=60, **kwargs
        )

    async def tool_web_reader(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Call the ZAI web reader tool."""
        return await self._forward_request(
            credential, client, "tools/web-reader", **kwargs
        )

    async def agent_chat(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Synchronous agent chat endpoint."""
        return await self._forward_request(
            credential, client, "agents/chat", timeout=120, **kwargs
        )

    async def agent_file_upload(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Upload a file for agent processing."""
        return await self._forward_request(
            credential, client, "agents/file-upload", timeout=60, **kwargs
        )

    async def agent_async_result(
        self, credential: str, client: httpx.AsyncClient, task_id: str
    ) -> Dict[str, Any]:
        """Retrieve async agent task result."""
        return await self._forward_request(
            credential, client,
            "agents/async-result",
            method="get",
            params={"task_id": task_id},
        )

    async def agent_conversation(
        self, credential: str, client: httpx.AsyncClient, **kwargs
    ) -> Dict[str, Any]:
        """Continue an agent conversation."""
        return await self._forward_request(
            credential, client, "agents/conversation", timeout=120, **kwargs
        )
