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
from typing import Any, Dict

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
    - Extension points for provider-specific URL and headers

    Subclasses must implement:
    - _get_api_base_url() -> str
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
