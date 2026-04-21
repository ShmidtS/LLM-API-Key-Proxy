# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Google OAuth Quota Tracker Base

Intermediate base class for providers that use Google OAuth credentials
with quota-based rate limiting (e.g., Antigravity, Gemini CLI).

Extracts shared test-request logic and class attribute configuration
that is common to all Google-OAuth-based quota trackers.

Required from provider (via mixin inheritance):
    - self.project_id_cache: Dict[str, str]
    - self.project_tier_cache: Dict[str, str]
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self._discover_project_id(cred_path, token, headers) -> str
    - self._get_api_base_url() -> str  (extension point defined here)
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...http_client_pool import get_http_pool
from .base_quota_tracker import BaseQuotaTracker


# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")


class GoogleQuotaTrackerBase(BaseQuotaTracker):
    """
    Intermediate base class for Google-OAuth-based quota trackers.

    Provides shared:
    - Integer max_requests mode configuration
    - Test request implementation (for discover_quota_costs)
    - Parameterized quota API fetch with provider-specific hooks
    - Extension points for provider-specific URL, headers, and response parsing

    Subclasses must implement:
    - _get_api_base_url() -> str
    - _get_quota_endpoint_suffix() -> str
    - _get_quota_headers(auth_header: dict) -> dict
    - _parse_quota_response(data: dict) -> list of tuples
    - _fetch_quota_for_credential()  (from BaseQuotaTracker)
    - _extract_model_quota_from_response()  (from BaseQuotaTracker)
    """

    # Integer max_requests mode - shared by all Google OAuth providers
    _use_integer_max_requests: bool = True

    # Type hints for attributes that must exist on the provider
    _learned_costs: Dict[str, Dict[str, int]]
    _learned_costs_loaded: bool
    _quota_refresh_interval: int
    project_tier_cache: Dict[str, str]
    project_id_cache: Dict[str, str]

    # =========================================================================
    # EXTENSION POINTS - Override in subclass
    # =========================================================================

    def _get_api_base_url(self) -> str:
        """Get the base URL for API requests. Override in subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_api_base_url"
        )

    def _get_api_headers(self) -> Dict[str, str]:
        """Get extra headers for API requests. Override in subclass if needed."""
        return {}

    def _get_quota_endpoint_suffix(self) -> str:
        """Get the endpoint suffix for quota API calls. Override in subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_quota_endpoint_suffix"
        )

    def _get_quota_headers(self, auth_header: Dict[str, str]) -> Dict[str, str]:
        """Build quota API request headers. Override in subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_quota_headers"
        )

    def _parse_quota_response(
        self, data: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse raw quota API response into normalized entries.

        Returns:
            List of (model_id_or_name, quota_info_dict) tuples where
            quota_info_dict contains at minimum:
            {
                "remaining_fraction": float,
                "is_exhausted": bool,
                "reset_time_iso": str | None,
                "reset_timestamp": float | None,
                ... provider-specific fields ...
            }
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _parse_quota_response"
        )

    def _get_quota_error_response_key(self) -> str:
        """Return the key used for quota data in error responses. Override if needed."""
        return "models"

    def _get_empty_quota_container(self) -> Any:
        """Return the empty container for quota data in error responses. Override if needed."""
        return {}

    # =========================================================================
    # SHARED QUOTA API FETCH
    # =========================================================================

    async def fetch_quota_from_api(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the provider's quota API.

        Uses parameterized hooks for provider-specific differences:
        - _get_quota_endpoint_suffix() for the API endpoint
        - _get_quota_headers() for request headers
        - _parse_quota_response() for response normalization

        Args:
            credential_path: Path to credential file or "env://provider/N"

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "identifier": str,
                "tier": str | None,
                "project_id": str | None,
                ... provider-specific fields ...
                "fetched_at": float,
            }
        """
        identifier = (
            Path(credential_path).name
            if not credential_path.startswith("env://")
            else credential_path
        )

        try:
            # Get auth header and project_id
            auth_header = await self.get_auth_header(credential_path)
            access_token = auth_header["Authorization"].split(" ")[1]

            # Get or discover project_id
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                project_id = await self._discover_project_id(
                    credential_path, access_token, {}
                )

            tier = self.project_tier_cache.get(credential_path)

            # Make API request
            url = f"{self._get_api_base_url()}:{self._get_quota_endpoint_suffix()}"
            headers = self._get_quota_headers(auth_header)
            payload = {"project": project_id} if project_id else {}

            pool = await get_http_pool()
            client = await pool.get_client_async()
            response = await client.post(
                url, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Parse response via provider-specific hook
            parsed = self._parse_quota_response(data)

            # Build normalized result dict
            result: Dict[str, Any] = {
                "status": "success",
                "error": None,
                "identifier": identifier,
                "tier": tier,
                "project_id": project_id,
                "fetched_at": time.time(),
            }

            # Store parsed entries under the provider's key
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

    # =========================================================================
    # TEST REQUEST (for discover_quota_costs)
    # =========================================================================

    async def _make_test_request(
        self,
        credential_path: str,
        model: str,
    ) -> Dict[str, Any]:
        """
        Make a minimal test request to consume quota.

        Args:
            credential_path: Credential to use
            model: Model to test

        Returns:
            {"success": bool, "error": str | None}
        """
        try:
            # Get auth header
            auth_header = await self.get_auth_header(credential_path)
            access_token = auth_header["Authorization"].split(" ")[1]

            # Get project_id
            project_id = self.project_id_cache.get(credential_path)
            if not project_id:
                project_id = await self._discover_project_id(
                    credential_path, access_token, {}
                )

            # Map user model to internal model name
            internal_model = self._user_to_api_model(model)

            # Build minimal request payload
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
            response = await client.post(
                url, headers=headers, json=payload, timeout=60
            )

            if response.status_code == 200:
                return {"success": True, "error": None}
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}
