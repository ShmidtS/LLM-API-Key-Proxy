# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Google OAuth Quota Tracker Base
"""

import logging
import httpx
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import time
from .base_quota_tracker import BaseQuotaTracker
from .quota_utils import parse_iso_timestamp, post_json_with_error_handling
from ...http_client_pool import get_http_pool

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

lib_logger = logging.getLogger("rotator_library")

class GoogleQuotaTrackerBase(BaseQuotaTracker):
    """
    Intermediate base class for Google-OAuth-based quota trackers.
    """

    _use_integer_max_requests: bool = True

    # Abstract Hooks
    def _get_api_base_url(self) -> str:
        raise NotImplementedError
    def _get_api_headers(self) -> Dict[str, str]:
        return {}
    def _get_quota_endpoint_suffix(self) -> str:
        raise NotImplementedError
    def _get_quota_headers(self, auth_header: Dict[str, str]) -> Dict[str, str]:
        """Default: Bearer token + Content-Type + provider-specific API headers."""
        access_token = auth_header["Authorization"].split(" ")[1]
        return {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **self._get_api_headers(),
        }
    def _parse_quota_response(self, data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        raise NotImplementedError
    def _get_quota_error_response_key(self) -> str:
        return "models"
    def _get_empty_quota_container(self) -> Any:
        return {}

    # ------------------------------------------------------------------
    # Shared quota response parser
    # ------------------------------------------------------------------

    def _parse_remaining_fraction_response(
        self,
        data: Dict[str, Any],
        *,
        container_key: str,
        entries: Callable[[Any], Iterable[Tuple[str, Dict[str, Any], Any]]],
        extra_fields: Callable[[str, Any], Dict[str, Any]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Shared parser for quota responses built around ``remainingFraction``.

        Both Antigravity and Gemini CLI responses share the same core shape:
        iterate entries, extract ``remainingFraction``, compute ``is_exhausted``,
        and parse ``resetTime`` via :func:`parse_iso_timestamp`.

        Providers parameterize the differences through callables:

        Args:
            data: Raw API response dict.
            container_key: Top-level key in *data* holding the entries
                (``"models"`` for Antigravity, ``"buckets"`` for Gemini CLI).
            entries: Callable receiving the container value and yielding
                ``(name, quota_info, raw_entry)`` triples.  *quota_info*
                is the sub-dict containing ``remainingFraction`` / ``resetTime``.
                *raw_entry* is the full original entry (for ``extra_fields``).
            extra_fields: Callable receiving ``(name, raw_entry)`` and
                returning a dict of provider-specific fields to merge into
                the result info dict.

        Returns:
            List of ``(name, info_dict)`` tuples.
        """
        container = data.get(container_key)
        if container is None:
            container = {} if container_key == "models" else []
        results: List[Tuple[str, Dict[str, Any]]] = []
        for name, quota_info, raw_entry in entries(container):
            remaining = quota_info.get("remainingFraction")
            if remaining is None:
                remaining = 0.0
                is_exhausted = True
            else:
                is_exhausted = remaining <= 0
            reset_time_iso = quota_info.get("resetTime")
            reset_timestamp: Optional[float] = None
            if reset_time_iso:
                reset_timestamp = parse_iso_timestamp(reset_time_iso)
            info: Dict[str, Any] = {
                "remaining_fraction": remaining,
                "is_exhausted": is_exhausted,
                "reset_time_iso": reset_time_iso,
                "reset_timestamp": reset_timestamp,
                **extra_fields(name, raw_entry),
            }
            results.append((name, info))
        return results

    async def fetch_quota_from_api(self, credential_path: str) -> Dict[str, Any]:
        identifier = (
            Path(credential_path).name
            if not credential_path.startswith("env://")
            else credential_path
        )
        try:
            auth_header = await self.get_auth_header(credential_path)
            access_token = auth_header["Authorization"].split(" ")[1]
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                project_id = await self._discover_project_id(credential_path, access_token, {})
            tier = self.project_tier_cache.get(credential_path)
            url = f"{self._get_api_base_url()}:{self._get_quota_endpoint_suffix()}"
            headers = self._get_quota_headers(auth_header)
            payload = {"project": project_id} if project_id else {}
            pool = await get_http_pool()
            client = await pool.get_client_async()
            provider_name = self.cache_subdir if hasattr(self, "cache_subdir") else "google"
            data = await post_json_with_error_handling(
                client, url, headers, payload,
                timeout=30, provider_name=provider_name,
            )
            if data is None:
                return {
                    "status": "error",
                    "error": "HTTP or JSON error",
                    "identifier": identifier,
                    "tier": tier,
                    "project_id": project_id,
                    self._get_quota_error_response_key(): self._get_empty_quota_container(),
                    "fetched_at": time.time(),
                }
            parsed = self._parse_quota_response(data)
            result = {
                "status": "success",
                "error": None,
                "identifier": identifier,
                "tier": tier,
                "project_id": project_id,
                "fetched_at": time.time(),
            }
            quota_key = self._get_quota_error_response_key()
            if quota_key == "buckets":
                result["buckets"] = [info for _name, info in parsed]
            else:
                result["models"] = {name: info for name, info in parsed}
            return result
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            lib_logger.warning(f"Failed to fetch quota for {identifier}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "identifier": identifier,
                "tier": self.project_tier_cache.get(credential_path),
                "project_id": self.project_id_cache.get(credential_path),
                self._get_quota_error_response_key(): self._get_empty_quota_container(),
                "fetched_at": time.time(),
            }

    async def _make_test_request(self, credential_path: str, model: str) -> Dict[str, Any]:
        try:
            auth_header = await self.get_auth_header(credential_path)
            access_token = auth_header["Authorization"].split(" ")[1]
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                project_id = await self._discover_project_id(credential_path, access_token, {})
            internal_model = self._user_to_api_model(model)
            url = f"{self._get_api_base_url()}:generateContent"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                **self._get_api_headers(),
            }
            payload = {
                "project": project_id,
                "model": internal_model,
                "request": {
                    "contents": [{"role": "user", "parts": [{"text": "Say 'test'"}]}],
                    "generationConfig": {"maxOutputTokens": 10},
                },
            }
            pool = await get_http_pool()
            client = await pool.get_client_async()
            response = await client.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code == 200:
                return {"success": True, "error": None}
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as e:
            return {"success": False, "error": str(e)}
