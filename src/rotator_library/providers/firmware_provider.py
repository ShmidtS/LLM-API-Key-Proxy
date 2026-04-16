"""
Firmware.ai Provider with Quota Tracking

Provider implementation for the Firmware.ai API with 5-hour rolling window quota tracking.
Uses the FirmwareQuotaTracker mixin to fetch quota usage from their API.

Environment variables:
    FIRMWARE_API_BASE: API base URL (default: https://app.firmware.ai/api/v1)
    FIRMWARE_API_KEY: API key for authentication
    FIRMWARE_QUOTA_REFRESH_INTERVAL: Quota refresh interval in seconds (default: 300)
"""

import httpx
import os
from typing import Any, Dict, List, Optional

from .provider_interface import ProviderInterface
from .utilities.firmware_quota_tracker import FirmwareQuotaTracker
from ..config.defaults import env_int


import logging

lib_logger = logging.getLogger("rotator_library")


class FirmwareProvider(FirmwareQuotaTracker, ProviderInterface):
    """
    Provider implementation for the Firmware.ai API with quota tracking.
    """

    _virtual_model_name = "firmware/_quota"
    provider_name = "firmware"
    _include_max_requests = False

    # Quota groups for tracking 5-hour rolling window limits
    # Uses a virtual model "firmware/_quota" for credential-level quota tracking
    model_quota_groups = {
        "firmware_global": ["firmware/_quota"],
    }

    def __init__(self, *args, **kwargs):
        """Initialize FirmwareProvider with quota tracking."""
        super().__init__(*args, **kwargs)

        # Quota tracking cache and refresh interval
        self._quota_cache: Dict[str, Dict[str, Any]] = {}
        self._quota_refresh_interval = env_int(
            "FIRMWARE_QUOTA_REFRESH_INTERVAL", 300
        )

        # API base URL (default to Firmware.ai)
        self.api_base = os.environ.get(
            "FIRMWARE_API_BASE", "https://app.firmware.ai/api/v1"
        )

    def get_model_quota_group(self, model: str) -> Optional[str]:
        """
        Get the quota group for a model.

        All Firmware.ai models share the same credential-level quota pool,
        so they all belong to the same quota group.

        Args:
            model: Model name (ignored - all models share quota)

        Returns:
            Quota group identifier for shared credential-level tracking
        """
        return "firmware_global"

    def get_models_in_quota_group(self, group: str) -> List[str]:
        """
        Get all models in a quota group.

        For Firmware.ai, we use a virtual model "firmware/_quota" to track the
        credential-level 5-hour rolling window quota.

        Args:
            group: Quota group name

        Returns:
            List of model names in the group
        """
        if group == "firmware_global":
            return ["firmware/_quota"]
        return []

    def get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Return usage reset configuration for Firmware.ai credentials.

        Firmware.ai uses per_model mode to track usage at the model level,
        with 5-hour rolling window quotas managed via the background job.

        Args:
            credential: The API key (unused, same config for all)

        Returns:
            Configuration with per_model mode and 5-hour window
        """
        return {
            "mode": "per_model",
            "window_seconds": 18000,  # 5 hours (5-hour rolling window)
            "field_name": "models",
        }

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetch available models from the Firmware.ai API.

        Args:
            api_key: Firmware.ai API key
            client: HTTP client

        Returns:
            List of model names prefixed with 'firmware/'
        """
        try:
            response = await client.get(
                f"{self.api_base.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            import json as json_lib
            try:
                data = response.json()
            except (json_lib.JSONDecodeError, ValueError) as e:
                lib_logger.warning(f"Invalid JSON from Firmware.ai models: {e}, body={response.text[:200]}")
                return []
            return [
                f"firmware/{model['id']}" for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                lib_logger.warning(f"Auth error fetching Firmware.ai models: {e.response.status_code}")
            elif e.response.status_code >= 500:
                lib_logger.warning(f"Server error fetching Firmware.ai models: {e.response.status_code}")
            else:
                lib_logger.error(f"HTTP error fetching Firmware.ai models: {e}")
            return []
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Firmware.ai models: {e}")
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
            "name": "firmware_quota_refresh",
            "run_on_start": True,
        }

