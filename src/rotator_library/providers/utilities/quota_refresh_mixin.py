# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
QuotaRefreshMixin - shared background job boilerplate for quota refresh.

Moved from base_streaming_provider.py (which should only contain streaming logic).
"""

import asyncio
import logging
from typing import Any, Dict, List

lib_logger = logging.getLogger("rotator_library")


class QuotaRefreshMixin:
    """
    Mixin providing shared background job boilerplate for quota refresh.

    Subclasses must define:
      - _virtual_model_name  (str): e.g. "chutes/_quota"
      - _quota_cache         (dict): credential -> usage_data cache
      - fetch_quota_usage(api_key, client) -> dict: provider-specific API call
      - provider_name         (str): for logging

    Optional:
      - _include_max_requests: whether to pass max_requests (default True)
    """

    _virtual_model_name: str = ""
    _quota_cache: Dict[str, Dict[str, Any]] = {}
    provider_name: str = ""
    _include_max_requests: bool = True

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
            api_key: str, client: Any
        ) -> None:
            async with semaphore:
                try:
                    usage_data = await self.fetch_quota_usage(api_key, client)

                    if usage_data.get("status") == "success":
                        self._quota_cache[api_key] = usage_data

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

                        lib_logger.debug(
                            f"Updated {self.provider_name} quota baseline for credential: "
                            f"{usage_data.get('remaining', 0):.0f}/{usage_data.get('quota', 0)} remaining "
                            f"({remaining_fraction * 100:.0f}%)"
                        )
                    elif usage_data.get("status") == "transient_error" or usage_data.get("remaining_fraction") is None:
                        lib_logger.warning(
                            f"Transient error refreshing {self.provider_name} quota for credential ...{api_key[-4:]} "
                            f"(error: {usage_data.get('error')}), preserving previous baseline"
                        )
                    else:
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

                except Exception as e:
                    lib_logger.warning(
                        f"Failed to refresh {self.provider_name} quota usage: {e}"
                    )

        if get_http_pool_fn is None:
            from ...http_client_pool import get_http_pool
            get_http_pool_fn = get_http_pool

        semaphore = asyncio.Semaphore(quota_fetch_concurrency)
        pool = await get_http_pool_fn()
        client = await pool.get_client_async()
        tasks = [
            refresh_single_credential(api_key, client) for api_key in credentials
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
