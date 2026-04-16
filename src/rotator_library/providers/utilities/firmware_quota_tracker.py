"""
Firmware.ai Quota Tracking Mixin

Provides quota tracking for the Firmware.ai provider using their quota usage API.
Firmware.ai uses a 5-hour rolling window quota system where:
- `used` is already a ratio (0 to 1) indicating quota utilization
- `reset` is an ISO 8601 UTC timestamp, or null when no active window

API Details:
- Endpoint: GET https://app.firmware.ai/api/v1/quota
- Auth: Authorization: Bearer <api_key>
- Response: { used: float, reset: string|null }

Required from provider:
    - self.api_base: str (API base URL)
    - self._quota_cache: Dict[str, Dict[str, Any]] = {}
    - self._quota_refresh_interval: int = 300
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

from .lightweight_quota_mixin import LightweightQuotaMixin

lib_logger = logging.getLogger("rotator_library")


class FirmwareQuotaTracker(LightweightQuotaMixin):
    """
    Mixin class providing quota tracking functionality for Firmware.ai provider.

    Inherits shared cache/pool boilerplate from LightweightQuotaMixin.
    """

    api_base: str
    _quota_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    def _get_quota_url(self) -> str:
        """Get the quota API URL based on configured api_base."""
        return f"{self.api_base.rstrip('/')}/quota"

    # =========================================================================
    # QUOTA USAGE API
    # =========================================================================

    async def fetch_quota_usage(
        self,
        api_key: str,
        client: Optional[httpx.AsyncClient] = None,
    ) -> Dict[str, Any]:
        """
        Fetch quota usage from the Firmware.ai API.

        Args:
            api_key: Firmware.ai API key
            client: Optional HTTP client for connection reuse

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "used": float,  # 0.0 to 1.0 (from API directly)
                "remaining_fraction": float,  # 1.0 - used
                "reset_at": float | None,  # Unix timestamp (seconds)
                "has_active_window": bool,  # True if reset is not null
                "fetched_at": float,
            }
        """
        headers = self._make_bearer_header(api_key)
        quota_url = self._get_quota_url()
        data = await self._fetch_json(quota_url, headers, client)
        if data is None:
            return {
                "status": "error",
                "error": None,
                "used": None,
                "remaining_fraction": None,
                "reset_at": None,
                "has_active_window": False,
                "fetched_at": time.time(),
            }

        # Parse response - API returns ratio directly
        used_raw = data.get("used")
        # Validate used is numeric
        if not isinstance(used_raw, (int, float)):
            lib_logger.warning(
                f"Firmware.ai quota API returned non-numeric 'used' value: {used_raw}"
            )
            used = 0.0
        else:
            used = float(used_raw)
        reset_iso = data.get("reset")

        # Calculate remaining (inverse of used), clamped to 0.0-1.0
        remaining_fraction = max(0.0, min(1.0, 1.0 - used))

        # Parse ISO 8601 reset timestamp
        reset_at = None
        if reset_iso is not None:
            reset_at = self._parse_iso_timestamp(reset_iso)
        # Only mark active window if we successfully parsed the timestamp
        has_active_window = reset_at is not None

        return {
            "status": "success",
            "error": None,
            "used": used,
            "remaining_fraction": remaining_fraction,
            "reset_at": reset_at,
            "has_active_window": has_active_window,
            "fetched_at": time.time(),
        }

    def _parse_iso_timestamp(self, iso_string: str) -> Optional[float]:
        """
        Parse ISO 8601 timestamp to Unix timestamp.

        Args:
            iso_string: ISO 8601 formatted timestamp (e.g., "2026-01-20T18:12:03.000Z")

        Returns:
            Unix timestamp in seconds, or None if parsing fails
        """
        try:
            # Handle 'Z' suffix by replacing with UTC offset
            if iso_string.endswith("Z"):
                iso_string = iso_string.replace("Z", "+00:00")

            dt = datetime.fromisoformat(iso_string)
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception as e:
            lib_logger.warning(f"Failed to parse ISO timestamp '{iso_string}': {e}")
            return None
