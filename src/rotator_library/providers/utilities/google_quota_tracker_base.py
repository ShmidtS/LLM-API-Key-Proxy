# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Google OAuth Quota Tracker Base
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from pathlib import Path
import time
from .base_quota_tracker import BaseQuotaTracker
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
        raise NotImplementedError
    def _parse_quota_response(self, data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        raise NotImplementedError
    def _get_quota_error_response_key(self) -> str:
        return "models"
    def _get_empty_quota_container(self) -> Any:
        return {}

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
            response = await client.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
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
        except Exception as e:
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
        except Exception as e:
            return {"success": False, "error": str(e)}
