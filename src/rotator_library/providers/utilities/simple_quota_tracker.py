# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
SimpleQuotaTrackerBase - lightweight base class for API-based quota trackers.

Provides the shared boilerplate that Chutes, Firmware, ZAI, and similar
providers duplicate: cache dict, refresh interval, and the HTTP client
pool fallback pattern.

Subclasses must implement:
    fetch_quota_usage(api_key, client) -> Dict[str, Any]

Usage:
    class MyQuotaTracker(SimpleQuotaTrackerBase):
        async def fetch_quota_usage(self, api_key, client=None):
            ...
"""

import time
import logging
from typing import Any, Dict, Optional

import httpx

from ...http_client_pool import get_http_pool

lib_logger = logging.getLogger("rotator_library")


class SimpleQuotaTrackerBase:
    """
    Lightweight base for quota trackers that fetch from an HTTP API.

    Provides:
    - _quota_cache dict (credential -> usage data)
    - _quota_refresh_interval (seconds, default 300)
    - _fetch_via_pool() helper for the client/pool fallback pattern
    """

    _quota_cache: Dict[str, Dict[str, Any]]
    _quota_refresh_interval: int

    async def _fetch_via_pool(
        self,
        url: str,
        headers: Dict[str, str],
        client: Optional[httpx.AsyncClient] = None,
        timeout: int = 30,
    ) -> httpx.Response:
        """
        Fetch a URL, reusing the provided client or falling back to the shared pool.

        Args:
            url: Request URL
            headers: Request headers
            client: Optional existing httpx client for connection reuse
            timeout: Request timeout in seconds

        Returns:
            httpx.Response

        Raises:
            httpx.HTTPStatusError: On non-2xx responses (must call raise_for_status yourself)
        """
        if client is not None:
            return await client.get(url, headers=headers, timeout=timeout)

        pool = await get_http_pool()
        new_client = await pool.get_client_async()
        return await new_client.get(url, headers=headers, timeout=timeout)

    @staticmethod
    def _make_bearer_header(api_key: str) -> Dict[str, str]:
        """Build a standard Bearer auth header."""
        return {
            "accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _error_result(self, **overrides) -> Dict[str, Any]:
        """Build a standardized error result dict. Subclasses can override defaults."""
        base = {
            "status": "error",
            "error": None,
            "fetched_at": time.time(),
        }
        base.update(overrides)
        return base
