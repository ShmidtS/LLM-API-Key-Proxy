# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
LightweightQuotaMixin - unified mixin for lightweight API-based quota trackers.

Merges the former SimpleQuotaTrackerBase (cache/pool/bearer/error helpers)
with LightweightQuotaMixin (background job boilerplate) into a single mixin.

Subclasses must implement:
    fetch_quota_usage(api_key, client) -> Dict[str, Any]

Subclasses must define:
    _virtual_model_name  (str): e.g. "chutes/_quota"
    provider_name        (str): for logging

Optional:
    _include_max_requests: whether to pass max_requests (default True)
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import json

from ...http_client_pool import get_http_pool

lib_logger = logging.getLogger("rotator_library")


class LightweightQuotaMixin:
    """
    Unified mixin for lightweight quota trackers that fetch from an HTTP API.

    Provides cache dict, refresh interval, HTTP client pool fallback,
    bearer header builder, error result builder, and background job runner.
    """

    _virtual_model_name: str = ""
    _quota_cache: Optional[Dict[str, Dict[str, Any]]] = None
    _quota_refresh_interval: int = 300
    provider_name: str = ""
    _include_max_requests: bool = True

    def __init__(self, **kwargs):
        self._quota_cache = {}
        super().__init__(**kwargs)

    # =====================================================================
    # HTTP HELPERS (from SimpleQuotaTrackerBase)
    # =====================================================================

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
        """
        if client is not None:
            return await client.get(url, headers=headers, timeout=timeout)

        pool = await get_http_pool()
        new_client = await pool.get_client_async()
        return await new_client.get(url, headers=headers, timeout=timeout)

    async def _fetch_json(
        self,
        url: str,
        headers: Dict[str, str],
        client: Optional[httpx.AsyncClient] = None,
        timeout: int = 30,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch a URL and parse the JSON response with robust error handling.

        Checks for 5xx / 401 / 403 status codes *before* raise_for_status
        so they are logged as warnings rather than raising HTTPStatusError.
        Catches JSON decode errors, HTTPStatusError, and RequestError,
        returning None (with a logged warning/error) in every failure case.

        Args:
            url: Request URL
            headers: Request headers
            client: Optional existing httpx client for connection reuse
            timeout: Request timeout in seconds

        Returns:
            Parsed dict on success, or None on any failure
        """
        provider = getattr(self, "provider_name", "unknown")
        try:
            response = await self._fetch_via_pool(url, headers, client, timeout)

            # Pre-check status codes that should not raise
            if response.status_code in (401, 403):
                lib_logger.warning(
                    f"{provider} quota API returned {response.status_code} "
                    f"(auth/forbidden), skipping"
                )
                return None
            if response.status_code >= 500:
                lib_logger.warning(
                    f"{provider} quota API returned {response.status_code} "
                    f"(server error), skipping"
                )
                return None

            response.raise_for_status()

            body = response.text
            if not body or not body.strip():
                lib_logger.debug(
                    f"{provider} quota API returned empty response (status {response.status_code})"
                )
                return None

            try:
                return response.json()
            except (json.JSONDecodeError, ValueError) as exc:
                lib_logger.warning(
                    f"{provider} quota API returned invalid JSON: {exc}"
                )
                return None

        except httpx.HTTPStatusError as exc:
            lib_logger.warning(
                f"{provider} quota API HTTP error: {exc.response.status_code}"
            )
            return None
        except httpx.RequestError as exc:
            lib_logger.error(
                f"{provider} quota API request error: {exc}"
            )
            return None

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

    # =====================================================================
    # BACKGROUND JOB (from LightweightQuotaMixin)
    # =====================================================================

    def get_background_job_config(self) -> Optional[Dict[str, Any]]:
        """Return background job config for quota refresh.

        Subclasses can override to customise interval, name, or run_on_start.
        """
        if not self.provider_name:
            return None
        return {
            "interval": self._quota_refresh_interval,
            "name": f"{self.provider_name}_quota_refresh",
            "run_on_start": True,
        }

    async def run_background_job(
        self,
        usage_manager: Any,
        credentials: List[str],
        quota_fetch_concurrency: int = 5,
        get_http_pool_fn=None,
    ) -> None:
        """
        Refresh quota usage for all credentials in parallel.

        Args:
            usage_manager: UsageManager instance
            credentials: List of API keys / credential paths
            quota_fetch_concurrency: Max concurrent quota fetch tasks
            get_http_pool_fn: Callable to get HTTP pool (injected for flexibility)
        """

        async def refresh_single_credential(
            api_key: str, client: Any, counters: Dict[str, int]
        ) -> None:
            async with semaphore:
                try:
                    usage_data = await self.fetch_quota_usage(api_key, client)

                    if usage_data.get("status") == "success":
                        if self._quota_cache is not None:
                            self._quota_cache[api_key] = usage_data
                        counters["success"] += 1

                        remaining_fraction = usage_data.get("remaining_fraction", 0.0)
                        reset_ts = usage_data.get("reset_at")

                        baseline_kwargs: Dict[str, Any] = {
                            "remaining_fraction": remaining_fraction,
                            "reset_timestamp": reset_ts,
                        }
                        if self._include_max_requests:
                            quota = usage_data.get("quota", 0)
                            baseline_kwargs["max_requests"] = quota

                        await usage_manager.update_quota_baseline(
                            api_key,
                            self._virtual_model_name,
                            **baseline_kwargs,
                        )

                    elif usage_data.get("status") == "transient_error" or usage_data.get("remaining_fraction") is None:
                        counters["transient"] += 1
                        lib_logger.debug(
                            f"Transient error refreshing {self.provider_name} quota for credential ...{api_key[-4:]} "
                            f"(error: {usage_data.get('error')}), preserving previous baseline"
                        )
                    else:
                        counters["failed"] += 1
                        if self._quota_cache is not None:
                            self._quota_cache[api_key] = usage_data
                        await usage_manager.update_quota_baseline(
                            api_key,
                            self._virtual_model_name,
                            remaining_fraction=0.0,
                            reset_timestamp=usage_data.get("reset_at"),
                        )
                        lib_logger.warning(
                            f"Failed to refresh {self.provider_name} quota for credential ...{api_key[-4:]} "
                            f"(error: {usage_data.get('error')}), marking as exhausted"
                        )

                except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
                    counters["errors"] += 1
                    lib_logger.warning(
                        f"Failed to refresh {self.provider_name} quota usage: {e}"
                    )

        if get_http_pool_fn is None:
            from ...http_client_pool import get_http_pool as _get_pool
            get_http_pool_fn = _get_pool

        counters: Dict[str, int] = {"success": 0, "transient": 0, "failed": 0, "errors": 0}
        semaphore = asyncio.Semaphore(quota_fetch_concurrency)
        pool = await get_http_pool_fn()
        client = await pool.get_client_async()
        tasks = [
            refresh_single_credential(api_key, client, counters) for api_key in credentials
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        total = len(credentials)
        if counters["success"] == total:
            lib_logger.debug(
                f"{self.provider_name} quota refresh: all {total} credentials updated"
            )
        elif counters["success"] > 0:
            lib_logger.debug(
                f"{self.provider_name} quota refresh: {counters['success']}/{total} ok, "
                f"{counters['transient']} transient, {counters['failed']} failed, "
                f"{counters['errors']} errors"
            )
