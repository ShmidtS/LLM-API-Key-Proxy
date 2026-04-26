# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Gemini CLI Quota Tracking Mixin

Provides quota tracking and retrieval methods for the Gemini CLI provider.
Uses the Google Code Assist retrieveUserQuota API to fetch actual quota data.

This inherits from BaseQuotaTracker for shared functionality and implements
Gemini CLI-specific quota API calls.

API Details (from google-gemini/gemini-cli):
- Endpoint: https://cloudcode-pa.googleapis.com/v1internal:retrieveUserQuota
- Request: { project: string, userAgent?: string }
- Response: { buckets?: BucketInfo[] }
- BucketInfo: { remainingAmount?, remainingFraction?, resetTime?, tokenType?, modelId? }

Required from provider:
    - self.project_id_cache: Dict[str, str]
    - self.project_tier_cache: Dict[str, str]
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self._discover_project_id(cred_path, token, params) -> str
    - self._load_tier_from_file(cred_path) -> Optional[str]
    - self.list_credentials(base_dir) -> List[Dict[str, Any]]
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .google_quota_tracker_base import GoogleQuotaTrackerBase
from .gemini_shared_utils import CODE_ASSIST_ENDPOINT


# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# QUOTA LIMITS (max requests per 100% quota)
# =============================================================================
# Max requests per quota period. This is the SOURCE OF TRUTH.
# Cost percentage is derived as: 100 / max_requests
# Using integers avoids floating-point precision issues.
#
# Verified 2026-01-07 via quota verification tests (see GEMINI_CLI_QUOTA_REPORT.md)
# Learned values (from file) override these defaults if available.

DEFAULT_MAX_REQUESTS: Dict[str, Dict[str, int]] = {
    "standard-tier": {
        # Pro group (verified: 0.4% per request = 250 requests)
        "gemini-2.5-pro": 250,
        "gemini-3-pro-preview": 250,
        # Flash group - 2.5 (verified: ~0.0667% per request = 1500 requests)
        # gemini-2.0-flash shares quota with 2.5-flash models
        "gemini-2.0-flash": 1500,
        "gemini-2.5-flash": 1500,
        "gemini-2.5-flash-lite": 1500,
        # 3-Flash group (verified: ~0.0667% per request = 1500 requests)
        "gemini-3-flash-preview": 1500,
    },
    "free-tier": {
        # Pro group (verified: 1.0% per request = 100 requests)
        "gemini-2.5-pro": 100,
        "gemini-3-pro-preview": 100,
        # Flash group - 2.5 (verified: 0.1% per request = 1000 requests)
        "gemini-2.0-flash": 1000,
        "gemini-2.5-flash": 1000,
        "gemini-2.5-flash-lite": 1000,
        # 3-Flash group (verified: 0.1% per request = 1000 requests)
        "gemini-3-flash-preview": 1000,
    },
}

# Default max requests for unknown models (1% = 100 requests)
DEFAULT_MAX_REQUESTS_UNKNOWN = 1000


class GeminiCliQuotaTracker(GoogleQuotaTrackerBase):
    """
    Mixin class providing quota tracking functionality for Gemini CLI provider.

    This mixin adds the following capabilities:
    - Fetch real-time quota info from the Gemini CLI retrieveUserQuota API
    - Discover all credentials (file-based and env-based)
    - Get structured quota info for all credentials

    Usage:
        class GeminiCliProvider(GeminiAuthBase, GeminiCliQuotaTracker):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._quota_refresh_interval: int = 300  # 5 min default
        self._learned_costs: Dict[str, Dict[str, float]] = {}
        self._learned_costs_loaded: bool = False
    """

    # =========================================================================
    # CLASS ATTRIBUTES - BaseQuotaTracker configuration
    # =========================================================================

    provider_env_prefix = "GEMINI_CLI"
    cache_subdir = "gemini_cli"

    # No model name mappings needed - API names match public names
    user_to_api_model_map: Dict[str, str] = {}
    api_to_user_model_map: Dict[str, str] = {}

    default_max_requests: Dict[str, Dict[str, int]] = DEFAULT_MAX_REQUESTS
    default_max_requests_unknown: int = DEFAULT_MAX_REQUESTS_UNKNOWN

    # =========================================================================
    # GOOGLE QUOTA TRACKER BASE HOOK IMPLEMENTATIONS
    # =========================================================================

    def _get_api_base_url(self) -> str:
        """Get the base URL for Gemini CLI API requests."""
        return CODE_ASSIST_ENDPOINT

    def _get_api_headers(self) -> Dict[str, str]:
        """Get Gemini CLI-specific headers for API requests."""
        return {}

    def _get_quota_endpoint_suffix(self) -> str:
        """Get the endpoint suffix for Gemini CLI quota API."""
        return "retrieveUserQuota"

    def _get_quota_headers(self, auth_header: Dict[str, str]) -> Dict[str, str]:
        """Build headers for Gemini CLI quota API request."""
        access_token = auth_header["Authorization"].split(" ")[1]
        return {
            "Authorization": f"Bearer {access_token}",
            **self._get_gemini_cli_headers(),
        }

    def _parse_quota_response(
        self, data: Dict[str, Any]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse Gemini CLI retrieveUserQuota response into normalized entries.

        Returns:
            List of (model_id, quota_info_dict) tuples.
        """
        from datetime import datetime

        results = []
        for bucket in data.get("buckets", []):
            # Parse remaining fraction (0.0 to 1.0)
            remaining = bucket.get("remainingFraction")
            if remaining is None:
                # NULL means exhausted
                remaining = 0.0
                is_exhausted = True
            else:
                is_exhausted = remaining <= 0

            # Parse reset time
            reset_time_iso = bucket.get("resetTime")
            reset_timestamp = None
            if reset_time_iso:
                try:
                    reset_dt = datetime.fromisoformat(
                        reset_time_iso.replace("Z", "+00:00")
                    )
                    reset_timestamp = reset_dt.timestamp()
                except (ValueError, AttributeError):
                    pass

            results.append(
                (
                    bucket.get("modelId"),
                    {
                        "model_id": bucket.get("modelId"),
                        "remaining_fraction": remaining,
                        "remaining_amount": bucket.get("remainingAmount"),
                        "reset_time_iso": reset_time_iso,
                        "reset_timestamp": reset_timestamp,
                        "token_type": bucket.get("tokenType"),
                        "is_exhausted": is_exhausted,
                    },
                )
            )
        return results

    def _get_quota_error_response_key(self) -> str:
        """Return the key used for quota data in error responses."""
        return "buckets"

    def _get_empty_quota_container(self) -> Any:
        """Return the empty container for quota data in error responses."""
        return []

    # =========================================================================
    # GEMINI CLI-SPECIFIC HELPERS
    # =========================================================================

    def _get_gemini_cli_headers(self) -> Dict[str, str]:
        """Get standard headers for Gemini CLI API requests."""
        return {
            "User-Agent": "google-api-nodejs-client/9.15.1",
            "X-Goog-Api-Client": "gl-node/22.17.0",
            "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _fetch_quota_for_credential(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the Gemini CLI retrieveUserQuota API.
        """
        return await self.fetch_quota_from_api(credential_path)

    def _extract_model_quota_from_response(
        self,
        quota_data: Dict[str, Any],
        tier: str,
    ) -> List[Tuple[str, float, Optional[int]]]:
        """
        Extract model quota information from Gemini CLI bucket response.

        Returns:
            List of tuples: (model_name, remaining_fraction, max_requests)
        """
        results = []

        for bucket in quota_data.get("buckets", []):
            model_id = bucket.get("model_id")
            if not model_id:
                continue

            remaining = bucket.get("remaining_fraction")
            if remaining is None:
                remaining = 0.0

            # Convert to user-facing model name
            user_model = self._api_to_user_model(model_id)

            # Calculate max_requests from tier-based cost
            max_requests = self.get_max_requests_for_model(user_model, tier)

            results.append((user_model, remaining, max_requests))

        return results

    # NOTE: _store_baselines_to_usage_manager is inherited from BaseQuotaTracker.

    # NOTE: The following methods are now inherited from BaseQuotaTracker:
    # - _load_learned_costs()
    # - _save_learned_costs()
    # - get_quota_cost()
    # - get_max_requests_for_model()
    # - update_learned_cost()
    # - _user_to_api_model()
    # - _api_to_user_model()
    # - discover_all_credentials()
    # - fetch_initial_baselines()
    # - refresh_active_quota_baselines()
    # - discover_quota_costs()
    # - _get_quota_group_for_model()

    # NOTE: _get_effective_quota_groups() is inherited from ProviderInterface
    # The quota groups are defined on GeminiCliProvider.model_quota_groups class attribute
    # This allows .env overrides via QUOTA_GROUPS_GEMINI_CLI_{GROUP}="model1,model2"
