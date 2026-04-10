# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from .base_streaming_provider import QuotaRefreshMixin
from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.chutes_quota_tracker import ChutesQuotaTracker

if TYPE_CHECKING:
    from ..usage_manager import UsageManager

# Create a local logger for this module
import logging

lib_logger = logging.getLogger("rotator_library")


class ChutesProvider(QuotaRefreshMixin, ChutesQuotaTracker, ProviderInterface):
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
        self._quota_refresh_interval: int = int(
            os.environ.get("CHUTES_QUOTA_REFRESH_INTERVAL", "300")
        )

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
        try:
            response = await client.get(
                "https://llm.chutes.ai/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            return [
                f"chutes/{model['id']}" for model in response.json().get("data", [])
            ]
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            lib_logger.error(f"Failed to fetch chutes.ai models: {e}")
            return []

    # =========================================================================
    # BACKGROUND JOB CONFIGURATION
    # =========================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """
        Configure periodic quota usage refresh.

        Returns:
            Background job configuration for quota refresh
        """
        return {
            "interval": self._quota_refresh_interval,
            "name": "chutes_quota_refresh",
            "run_on_start": True,
        }

