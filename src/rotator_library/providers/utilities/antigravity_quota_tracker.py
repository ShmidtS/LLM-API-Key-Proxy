# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Antigravity Quota Tracking Mixin

Provides quota tracking, estimation, and verification methods for the
Antigravity provider. This inherits from BaseQuotaTracker for shared
functionality and implements Antigravity-specific quota API calls.

Required from provider:
    - self._get_effective_quota_groups() -> Dict[str, List[str]]
    - self._get_available_models() -> List[str]  # User-facing model names
    - self._get_antigravity_headers() -> Dict[str, str]  # API headers for requests
    - self.list_credentials(base_dir) -> List[Dict[str, Any]]
    - self.project_tier_cache: Dict[str, str]
    - self.project_id_cache: Dict[str, str]
    - self.get_auth_header(credential_path) -> Dict[str, str]
    - self._discover_project_id(cred_path, token, headers) -> str
    - self._get_base_url() -> str
    - self._load_tier_from_file(cred_path) -> Optional[str]
"""

import asyncio
import json
import logging
from ...utils.json_utils import json_loads
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from ...http_client_pool import get_http_pool
from .base_quota_tracker import BaseQuotaTracker, QUOTA_DISCOVERY_DELAY_SECONDS

if TYPE_CHECKING:
    from ...usage_manager import UsageManager

# Use the shared rotator_library logger
lib_logger = logging.getLogger("rotator_library")


# =============================================================================
# QUOTA LIMITS (max requests per 100% quota)
# =============================================================================
# Max requests per quota period. This is the SOURCE OF TRUTH.
# Cost percentage is derived as: 100 / max_requests
# Using integers avoids floating-point precision issues (e.g., 149 vs 150).
#
# Verified empirically 2026-01-07 - see tests/quota_verification/QUOTA_TESTING_GUIDE.md
# Learned values (from file) override these defaults if available.

DEFAULT_MAX_REQUESTS: Dict[str, Dict[str, int]] = {
    "standard-tier": {
        # Claude/GPT-OSS group (verified: 0.6667% per request = 150 requests)
        "claude-sonnet-4-5": 150,
        "claude-sonnet-4-5-thinking": 150,
        "claude-opus-4-5": 150,
        "claude-opus-4-5-thinking": 150,
        "claude-sonnet-4.5": 150,
        "claude-opus-4.5": 150,
        "gpt-oss-120b-medium": 150,
        # Gemini 3 Pro group (verified: 0.3125% per request = 320 requests)
        "gemini-3-pro-high": 320,
        "gemini-3-pro-low": 320,
        "gemini-3-pro-preview": 320,
        # Gemini 3 Flash (verified: 0.25% per request = 400 requests)
        "gemini-3-flash": 400,
        # Gemini 2.5 Flash group (verified: 0.0333% per request = 3000 requests)
        "gemini-2.5-flash": 3000,
        "gemini-2.5-flash-thinking": 3000,
        # Gemini 2.5 Flash Lite - SEPARATE pool (verified: 0.02% per request = 5000 requests)
        "gemini-2.5-flash-lite": 5000,
        # Gemini 2.5 Pro - UNVERIFIED/UNUSED (assumed 0.1% = 1000 requests)
        "gemini-2.5-pro": 1,
    },
    "free-tier": {
        # Claude/GPT-OSS group (verified: 2.0% per request = 50 requests)
        "claude-sonnet-4-5": 50,
        "claude-sonnet-4-5-thinking": 50,
        "claude-opus-4-5": 50,
        "claude-opus-4-5-thinking": 50,
        "claude-sonnet-4.5": 50,
        "claude-opus-4.5": 50,
        "gpt-oss-120b-medium": 50,
        # Gemini 3 Pro group (verified: 0.6667% per request = 150 requests)
        "gemini-3-pro-high": 150,
        "gemini-3-pro-low": 150,
        "gemini-3-pro-preview": 150,
        # Gemini 3 Flash (verified: 0.2% per request = 500 requests)
        "gemini-3-flash": 500,
        # Gemini 2.5 Flash group (verified: 0.0333% per request = 3000 requests)
        "gemini-2.5-flash": 3000,
        "gemini-2.5-flash-thinking": 3000,
        # Gemini 2.5 Flash Lite - SEPARATE pool (verified: 0.02% per request = 5000 requests)
        "gemini-2.5-flash-lite": 5000,
        # Gemini 2.5 Pro - UNVERIFIED/UNUSED (assumed 0.1% = 1000 requests)
        "gemini-2.5-pro": 1,
    },
}

# Default max requests for unknown models (1% = 100 requests)
DEFAULT_MAX_REQUESTS_UNKNOWN = 100

# =============================================================================
# MODEL NAME MAPPINGS
# =============================================================================
# Some user-facing model names don't exist in the API response.
# These mappings convert between user-facing names and API names.

# User-facing name -> API name (for looking up quota in fetchAvailableModels response)
_USER_TO_API_MODEL_MAP: Dict[str, str] = {
    "claude-opus-4-5": "claude-opus-4-5-thinking",  # Opus only exists as -thinking in API (legacy)
    "claude-opus-4.5": "claude-opus-4-5-thinking",  # Opus only exists as -thinking in API (new format)
    "gemini-3-pro-preview": "gemini-3-pro-high",  # Preview maps to high by default
}

# API name -> User-facing name (for consistency when processing API responses)
_API_TO_USER_MODEL_MAP: Dict[str, str] = {
    "claude-opus-4-5-thinking": "claude-opus-4.5",  # Normalize to new user-facing name
    "claude-opus-4-5": "claude-opus-4.5",  # Normalize old format to new
    "claude-sonnet-4-5-thinking": "claude-sonnet-4.5",  # Normalize to new user-facing name
    "claude-sonnet-4-5": "claude-sonnet-4.5",  # Normalize old format to new
    "gemini-3-pro-high": "gemini-3-pro-preview",  # Could map to preview (but high is valid too)
    "gemini-3-pro-low": "gemini-3-pro-preview",  # Could map to preview (but low is valid too)
    "gemini-2.5-flash-thinking": "gemini-2.5-flash",  # Normalize to user-facing name
}


class AntigravityQuotaTracker(BaseQuotaTracker):
    """
    Mixin class providing quota tracking functionality for Antigravity provider.

    This mixin adds the following capabilities:
    - Fetch quota info from the Antigravity fetchAvailableModels API
    - Track requests locally to estimate remaining quota
    - Verify and learn quota costs adaptively
    - Discover all credentials (file-based and env-based)
    - Get structured quota info for all credentials

    Usage:
        class AntigravityProvider(GoogleOAuthBase, AntigravityQuotaTracker):
            ...

    The provider class must initialize these instance attributes in __init__:
        self._learned_costs: Dict[str, Dict[str, int]] = {}
        self._learned_costs_loaded: bool = False
        self._quota_refresh_interval: int = 300  # 5 min default
    """

    # =========================================================================
    # CLASS ATTRIBUTES - BaseQuotaTracker configuration
    # =========================================================================

    provider_env_prefix = "ANTIGRAVITY"
    cache_subdir = "antigravity"
    user_to_api_model_map = _USER_TO_API_MODEL_MAP
    api_to_user_model_map = _API_TO_USER_MODEL_MAP

    # Integer max_requests mode (source of truth = integer max, not float cost)
    _use_integer_max_requests: bool = True
    default_max_requests: Dict[str, Dict[str, int]] = DEFAULT_MAX_REQUESTS
    default_max_requests_unknown: int = DEFAULT_MAX_REQUESTS_UNKNOWN

    # Type hints for attributes that must exist on the provider
    _learned_costs: Dict[str, Dict[str, int]]
    _learned_costs_loaded: bool
    _quota_refresh_interval: int
    project_tier_cache: Dict[str, str]
    project_id_cache: Dict[str, str]

    # =========================================================================
    # ANTIGRAVITY-SPECIFIC HELPERS
    # =========================================================================

    def _get_provider_prefix(self) -> str:
        """Get the provider prefix for model names."""
        return "antigravity"

    def _get_quota_group_for_model(self, model: str) -> Optional[str]:
        """Get the quota group name for a model."""
        clean_model = model.split("/")[-1] if "/" in model else model
        groups = self._get_effective_quota_groups()
        for group_name, models in groups.items():
            if clean_model in models:
                return group_name
        return None

    # =========================================================================
    # BaseQuotaTracker ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    async def _fetch_quota_for_credential(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the Antigravity fetchAvailableModels API.
        """
        return await self.fetch_quota_from_api(credential_path)

    def _extract_model_quota_from_response(
        self,
        quota_data: Dict[str, Any],
        tier: str,
    ) -> List[Tuple[str, float, Optional[int]]]:
        """
        Extract model quota information from Antigravity models response.

        Returns:
            List of tuples: (model_name, remaining_fraction, max_requests)
        """
        results = []

        # Get user-facing model names we care about
        available_models = set(self._get_available_models())

        # Track which user-facing models we've already added to avoid duplicates
        added_models: set = set()

        for api_model_name, model_info in quota_data.get("models", {}).items():
            remaining = model_info.get("remaining_fraction")
            if remaining is None:
                continue

            # Convert API name to user-facing name
            user_model = self._api_to_user_model(api_model_name)

            # Only include if this is a model we expose to users
            if user_model not in available_models:
                continue

            # Skip duplicates (e.g., claude-sonnet-4-5 and claude-sonnet-4-5-thinking)
            if user_model in added_models:
                continue

            # Calculate max_requests for this model/tier
            max_requests = self.get_max_requests_for_model(user_model, tier)

            results.append((user_model, remaining, max_requests))
            added_models.add(user_model)

        return results

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
            url = f"{self._get_base_url()}:generateContent"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                **self._get_antigravity_headers(),
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

    # =========================================================================
    # ANTIGRAVITY-SPECIFIC QUOTA API
    # =========================================================================

    async def fetch_quota_from_api(
        self,
        credential_path: str,
    ) -> Dict[str, Any]:
        """
        Fetch quota information from the Antigravity fetchAvailableModels API.

        Args:
            credential_path: Path to credential file or "env://antigravity/N"

        Returns:
            {
                "status": "success" | "error",
                "error": str | None,
                "identifier": str,
                "tier": str | None,
                "project_id": str | None,
                "models": {
                    "model_name": {
                        "remaining_fraction": 0.95,  # None from API = 0.0 (EXHAUSTED)
                        "is_exhausted": bool,
                        "reset_time_iso": "2025-12-16T10:31:36Z" | None,
                        "reset_timestamp": float | None,
                        "display_name": str | None,
                    }
                },
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
            url = f"{self._get_base_url()}:fetchAvailableModels"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                **self._get_antigravity_headers(),
            }
            payload = {"project": project_id} if project_id else {}

            pool = await get_http_pool()
            client = await pool.get_client_async()
            response = await client.post(
                url, headers=headers, json=payload, timeout=30
            )
            response.raise_for_status()
            data = response.json()

            # Parse models
            models_data = {}
            for model_name, model_info in data.get("models", {}).items():
                quota_info = model_info.get("quotaInfo", {})

                # CRITICAL: NULL remainingFraction means EXHAUSTED (0.0)
                remaining = quota_info.get("remainingFraction")
                if remaining is None:
                    remaining = 0.0
                    is_exhausted = True
                else:
                    is_exhausted = remaining <= 0

                reset_time_iso = quota_info.get("resetTime")
                reset_timestamp = None
                if reset_time_iso:
                    try:
                        reset_dt = datetime.fromisoformat(
                            reset_time_iso.replace("Z", "+00:00")
                        )
                        reset_timestamp = reset_dt.timestamp()
                    except (ValueError, AttributeError):
                        pass

                models_data[model_name] = {
                    "remaining_fraction": remaining,
                    "is_exhausted": is_exhausted,
                    "reset_time_iso": reset_time_iso,
                    "reset_timestamp": reset_timestamp,
                    "display_name": model_info.get("displayName"),
                }

            return {
                "status": "success",
                "error": None,
                "identifier": identifier,
                "tier": tier,
                "project_id": project_id,
                "models": models_data,
                "fetched_at": time.time(),
            }

        except Exception as e:
            lib_logger.warning(f"Failed to fetch quota for {identifier}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "identifier": identifier,
                "tier": self.project_tier_cache.get(credential_path),
                "project_id": self.project_id_cache.get(credential_path),
                "models": {},
                "fetched_at": time.time(),
            }

    # =========================================================================
    # BASELINE MANAGEMENT (Override for Antigravity-specific cooldown logging)
    # =========================================================================

    async def refresh_active_quota_baselines(
        self,
        credential_paths: List[str],
        usage_data: Dict[str, Any],
        interval_seconds: Optional[int] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Refresh quota baselines for credentials with recent activity.

        Only refreshes credentials that were used within the interval.

        Args:
            credential_paths: All credential paths to consider
            usage_data: Usage data from UsageManager
            interval_seconds: Consider "active" if used within this time (default: _quota_refresh_interval)

        Returns:
            Dict mapping credential_path -> fetched quota data (for updating baselines)
        """
        if interval_seconds is None:
            interval_seconds = self._quota_refresh_interval

        now = time.time()
        active_credentials = []

        for cred_path in credential_paths:
            cred_usage = usage_data.get(cred_path, {})
            last_used = cred_usage.get("last_used_ts", 0)

            if now - last_used < interval_seconds:
                active_credentials.append(cred_path)

        if not active_credentials:
            lib_logger.debug(
                "No recently active credentials to refresh quota baselines"
            )
            return {}

        lib_logger.debug(
            f"Refreshing quota baselines for {len(active_credentials)} "
            f"recently active credentials"
        )

        results = {}
        for cred_path in active_credentials:
            quota_data = await self.fetch_quota_from_api(cred_path)
            results[cred_path] = quota_data

        return results

    async def fetch_initial_baselines(
        self,
        credential_paths: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch quota baselines for all credentials.

        Fetches quota data from the Antigravity API for all provided credentials
        with limited concurrency to avoid rate limiting.

        Args:
            credential_paths: All credential paths to fetch baselines for

        Returns:
            Dict mapping credential_path -> fetched quota data
        """
        if not credential_paths:
            return {}

        lib_logger.debug(
            f"Fetching quota baselines for {len(credential_paths)} credentials..."
        )

        results = {}

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(cred_path: str):
            async with semaphore:
                return cred_path, await self.fetch_quota_from_api(cred_path)

        # Fetch all in parallel with limited concurrency
        tasks = [fetch_with_semaphore(cred) for cred in credential_paths]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        for result in fetch_results:
            if isinstance(result, Exception):
                lib_logger.warning(f"Baseline fetch failed: {result}")
                continue

            cred_path, quota_data = result
            if quota_data["status"] == "success":
                success_count += 1
            results[cred_path] = quota_data

        lib_logger.debug(
            f"Baseline fetch complete: {success_count}/{len(credential_paths)} successful"
        )

        return results

    async def _store_baselines_to_usage_manager(
        self,
        quota_results: Dict[str, Dict[str, Any]],
        usage_manager: "UsageManager",
    ) -> int:
        """
        Store fetched quota baselines into UsageManager.

        Args:
            quota_results: Dict from fetch_quota_from_api or fetch_initial_baselines
            usage_manager: UsageManager instance to store baselines in

        Returns:
            Number of baselines successfully stored
        """
        stored_count = 0

        # Get user-facing model names we care about
        available_models = set(self._get_available_models())

        # Aggregate cooldown info for consolidated logging
        # Structure: {short_cred_name: {group_or_model: hours_until_reset}}
        cooldowns_by_cred: Dict[str, Dict[str, float]] = {}

        for cred_path, quota_data in quota_results.items():
            if quota_data.get("status") != "success":
                continue

            # Get tier for this credential (needed for max_requests calculation)
            tier = self.project_tier_cache.get(cred_path, "unknown")

            models = quota_data.get("models", {})
            # Track which user-facing models we've already stored to avoid duplicates
            stored_for_cred: set = set()

            # Short credential name for logging (strip antigravity_ prefix and .json suffix)
            if cred_path.startswith("env://"):
                short_cred = cred_path.split("/")[-1]
            else:
                short_cred = Path(cred_path).stem
                if short_cred.startswith("antigravity_"):
                    short_cred = short_cred[len("antigravity_") :]

            for api_model_name, model_info in models.items():
                remaining = model_info.get("remaining_fraction")
                if remaining is None:
                    continue

                # Convert API name to user-facing name
                user_model = self._api_to_user_model(api_model_name)

                # Only store if this is a model we expose to users
                if user_model not in available_models:
                    continue

                # Skip if we already stored this user-facing model
                # (e.g., claude-sonnet-4-5 and claude-sonnet-4-5-thinking both map to claude-sonnet-4-5)
                if user_model in stored_for_cred:
                    continue

                # Calculate max_requests for this model/tier
                max_requests = self.get_max_requests_for_model(user_model, tier)

                # Extract reset_timestamp (already parsed to float in fetch_quota_from_api)
                reset_timestamp = model_info.get("reset_timestamp")

                # Store with provider prefix for consistency with usage tracking
                prefixed_model = f"antigravity/{user_model}"
                cooldown_info = await usage_manager.update_quota_baseline(
                    cred_path, prefixed_model, remaining, max_requests, reset_timestamp
                )

                # Aggregate cooldown info if returned
                if cooldown_info:
                    group_or_model = cooldown_info["group_or_model"]
                    hours = cooldown_info["hours_until_reset"]
                    if short_cred not in cooldowns_by_cred:
                        cooldowns_by_cred[short_cred] = {}
                    # Only keep first occurrence per group/model (avoids duplicates)
                    if group_or_model not in cooldowns_by_cred[short_cred]:
                        cooldowns_by_cred[short_cred][group_or_model] = hours

                stored_for_cred.add(user_model)
                stored_count += 1

        # Log consolidated message for all cooldowns
        if cooldowns_by_cred:
            # Build message: "oauth_1[claude 3.4h, gemini-3-pro 2.1h], oauth_2[claude 5.2h]"
            parts = []
            for cred_name, groups in sorted(cooldowns_by_cred.items()):
                group_strs = [f"{g} {h:.1f}h" for g, h in sorted(groups.items())]
                parts.append(f"{cred_name}[{', '.join(group_strs)}]")
            lib_logger.info(f"Antigravity quota exhausted: {', '.join(parts)}")
        else:
            lib_logger.debug("Antigravity quota baseline refresh: no cooldowns needed")

        return stored_count
