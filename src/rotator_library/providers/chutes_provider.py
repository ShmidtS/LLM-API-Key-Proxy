# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
import json
from typing import Any, Dict, List, Optional
from .provider_interface import ProviderInterface, UsageResetConfigDef
from .utilities.chutes_quota_tracker import ChutesQuotaTracker
from ..config.defaults import env_int

# Create a local logger for this module
import logging

lib_logger = logging.getLogger("rotator_library")


class ChutesProvider(ChutesQuotaTracker, ProviderInterface):
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
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                lib_logger.warning(f"Invalid JSON from chutes.ai models: {e}, body={response.text[:200]}")
                return []
            return [
                f"chutes/{model['id']}" for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                lib_logger.warning(f"Auth error fetching chutes.ai models: {e.response.status_code}")
            elif e.response.status_code >= 500:
                lib_logger.warning(f"Server error fetching chutes.ai models: {e.response.status_code}")
            else:
                lib_logger.error(f"HTTP error fetching chutes.ai models: {e}")
            return []
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch chutes.ai models: {e}")
            return []

