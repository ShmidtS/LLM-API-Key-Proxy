# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

lib_logger = logging.getLogger("rotator_library")


class QuotaReporter:
    def __init__(self, usage_manager, provider_plugins, provider_instances, all_credentials):
        self.usage_manager = usage_manager
        self._provider_plugins = provider_plugins
        self._provider_instances = provider_instances
        self.all_credentials = all_credentials

    async def get_quota_stats(
        self,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get quota and usage stats for all credentials.

        This returns cached/disk data aggregated by provider.
        For provider-specific quota info (e.g., Antigravity quota groups),
        it enriches the data from provider plugins.

        Args:
            provider_filter: If provided, only return stats for this provider

        Returns:
            Complete stats dict ready for the /v1/quota-stats endpoint
        """
        # Get base stats from usage manager
        stats = await self.usage_manager.get_stats_for_endpoint(provider_filter)

        # Enrich with provider-specific quota data
        for provider, prov_stats in stats.get("providers", {}).items():
            provider_class = self._provider_plugins.get(provider)
            if not provider_class:
                continue

            # Get or create provider instance
            provider_instance = self._provider_instances.get_or_create(provider, provider_class)
            # Check if provider has quota tracking (like Antigravity)
            if hasattr(provider_instance, "_get_effective_quota_groups"):
                # Add quota group summary
                quota_groups = provider_instance._get_effective_quota_groups()
                prov_stats["quota_groups"] = {}

                for group_name, group_models in quota_groups.items():
                    group_stats = {
                        "models": group_models,
                        "credentials_total": 0,
                        "credentials_exhausted": 0,
                        "avg_remaining_pct": 0,
                        "total_remaining_pcts": [],
                        # Total requests tracking across all credentials
                        "total_requests_used": 0,
                        "total_requests_max": 0,
                        # Tier breakdown: tier_name -> {"total": N, "active": M}
                        "tiers": {},
                    }

                    # Calculate per-credential quota for this group
                    for cred in prov_stats.get("credentials", []):
                        models_data = cred.get("models", {})
                        group_stats["credentials_total"] += 1

                        # Track tier - get directly from provider cache since cred["tier"] not set yet
                        tier = cred.get("tier")
                        if not tier and hasattr(
                            provider_instance, "project_tier_cache"
                        ):
                            cred_path = cred.get("full_path", "")
                            tier = provider_instance.project_tier_cache.get(cred_path)
                        tier = tier or "unknown"

                        # Initialize tier entry if needed with priority for sorting
                        if tier not in group_stats["tiers"]:
                            priority = 10  # default
                            if hasattr(provider_instance, "_resolve_tier_priority"):
                                priority = provider_instance._resolve_tier_priority(
                                    tier
                                )
                            group_stats["tiers"][tier] = {
                                "total": 0,
                                "active": 0,
                                "priority": priority,
                            }
                        group_stats["tiers"][tier]["total"] += 1

                        # Find model with VALID baseline (not just any model with stats)
                        model_stats = None
                        for model in group_models:
                            candidate = self._find_model_stats_in_data(
                                models_data, model, provider, provider_instance
                            )
                            if candidate:
                                baseline = candidate.get("baseline_remaining_fraction")
                                if baseline is not None:
                                    model_stats = candidate
                                    break
                                # Keep first found as fallback (for request counts)
                                if model_stats is None:
                                    model_stats = candidate

                        if model_stats:
                            baseline = model_stats.get("baseline_remaining_fraction")
                            req_count = model_stats.get("request_count", 0)
                            max_req = model_stats.get("quota_max_requests") or 0

                            # Accumulate totals (one model per group per credential)
                            group_stats["total_requests_used"] += req_count
                            group_stats["total_requests_max"] += max_req

                            if baseline is not None:
                                remaining_pct = int(baseline * 100)
                                group_stats["total_remaining_pcts"].append(
                                    remaining_pct
                                )
                                if baseline <= 0:
                                    group_stats["credentials_exhausted"] += 1
                                else:
                                    # Credential is active (has quota remaining)
                                    group_stats["tiers"][tier]["active"] += 1

                    # Calculate average remaining percentage (per-credential average)
                    if group_stats["total_remaining_pcts"]:
                        group_stats["avg_remaining_pct"] = int(
                            sum(group_stats["total_remaining_pcts"])
                            / len(group_stats["total_remaining_pcts"])
                        )
                    del group_stats["total_remaining_pcts"]

                    # Calculate total remaining percentage (global)
                    if group_stats["total_requests_max"] > 0:
                        used = group_stats["total_requests_used"]
                        max_r = group_stats["total_requests_max"]
                        group_stats["total_requests_remaining"] = max(0, max_r - used)
                        group_stats["total_remaining_pct"] = max(
                            0, int((1 - used / max_r) * 100)
                        )
                    else:
                        group_stats["total_requests_remaining"] = 0
                        # Fallback to avg_remaining_pct when max_requests unavailable
                        # This handles providers like Firmware that only provide percentage
                        group_stats["total_remaining_pct"] = group_stats.get(
                            "avg_remaining_pct"
                        )

                    prov_stats["quota_groups"][group_name] = group_stats

                # Also enrich each credential with formatted quota group info
                credentials = prov_stats.get("credentials", [])
                for i, cred in enumerate(credentials):
                    cred = dict(cred)
                    cred["model_groups"] = {}
                    models_data = cred.get("models", {})

                    for group_name, group_models in quota_groups.items():
                        # Find model with VALID baseline (prefer over any model with stats)
                        # Also track the best reset_ts across all models in the group
                        model_stats = None
                        best_reset_ts = None

                        for model in group_models:
                            candidate = self._find_model_stats_in_data(
                                models_data, model, provider, provider_instance
                            )
                            if candidate:
                                # Track the best (latest) reset_ts from any model in group
                                candidate_reset_ts = candidate.get("quota_reset_ts")
                                if candidate_reset_ts:
                                    if (
                                        best_reset_ts is None
                                        or candidate_reset_ts > best_reset_ts
                                    ):
                                        best_reset_ts = candidate_reset_ts

                                baseline = candidate.get("baseline_remaining_fraction")
                                if baseline is not None:
                                    model_stats = candidate
                                    # Don't break - continue to find best reset_ts
                                # Keep first found as fallback
                                if model_stats is None:
                                    model_stats = candidate

                        if model_stats:
                            baseline = model_stats.get("baseline_remaining_fraction")
                            max_req = model_stats.get("quota_max_requests")
                            req_count = model_stats.get("request_count", 0)
                            # Use best_reset_ts from any model in the group
                            reset_ts = best_reset_ts or model_stats.get(
                                "quota_reset_ts"
                            )

                            remaining_pct = (
                                int(baseline * 100) if baseline is not None else None
                            )
                            is_exhausted = baseline is not None and baseline <= 0

                            # Format reset time
                            reset_iso = None
                            if reset_ts:
                                try:
                                    from datetime import datetime, timezone

                                    reset_iso = datetime.fromtimestamp(
                                        reset_ts, tz=timezone.utc
                                    ).isoformat()
                                except (ValueError, OSError, TypeError) as e:
                                    lib_logger.debug("Could not process timestamp: %s", e)

                            requests_remaining = (
                                max(0, max_req - req_count) if max_req else 0
                            )

                            # Determine display format
                            # Priority: requests (if max known) > percentage (if baseline available) > unknown
                            if max_req:
                                display = f"{requests_remaining}/{max_req}"
                            elif remaining_pct is not None:
                                display = f"{remaining_pct}%"
                            else:
                                display = "?/?"

                            cred["model_groups"][group_name] = {
                                "remaining_pct": remaining_pct,
                                "requests_used": req_count,
                                "requests_remaining": requests_remaining,
                                "requests_max": max_req,
                                "display": display,
                                "is_exhausted": is_exhausted,
                                "reset_time_iso": reset_iso,
                                "models": group_models,
                                "confidence": self._get_baseline_confidence(
                                    model_stats
                                ),
                            }

                    # Recalculate only current-period requests from model_groups
                    # to avoid double-counting shared quota pools while preserving lifetime totals.
                    if cred.get("model_groups"):
                        group_requests = sum(
                            g.get("requests_used", 0)
                            for g in cred["model_groups"].values()
                        )
                        cred["requests"] = group_requests

                    # Try to get email from provider's cache
                    cred_path = cred.get("full_path", "")
                    if hasattr(provider_instance, "project_tier_cache"):
                        tier = provider_instance.project_tier_cache.get(cred_path)
                        if tier:
                            cred["tier"] = tier

                    credentials[i] = cred

        return stats

    def _find_model_stats_in_data(
        self,
        models_data: Dict[str, Any],
        model: str,
        provider: str,
        provider_instance: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Find model stats in models_data, trying various name variants.

        Handles aliased model names (e.g., gemini-3-pro-preview -> gemini-3-pro-high)
        by using the provider's _user_to_api_model() mapping.

        Args:
            models_data: Dict of model_name -> stats from credential
            model: Model name to look up (user-facing name)
            provider: Provider name for prefixing
            provider_instance: Provider instance for alias methods

        Returns:
            Model stats dict if found, None otherwise
        """
        # Try direct match with and without provider prefix
        prefixed_model = f"{provider}/{model}"
        model_stats = models_data.get(prefixed_model) or models_data.get(model)

        if model_stats:
            return model_stats

        # Try with API model name (e.g., gemini-3-pro-preview -> gemini-3-pro-high)
        if hasattr(provider_instance, "_user_to_api_model"):
            api_model = provider_instance._user_to_api_model(model)
            if api_model != model:
                prefixed_api = f"{provider}/{api_model}"
                model_stats = models_data.get(prefixed_api) or models_data.get(
                    api_model
                )

        return model_stats

    def _get_baseline_confidence(self, model_stats: Dict) -> str:
        """
        Determine confidence level based on baseline age.

        Args:
            model_stats: Model statistics dict with baseline_fetched_at

        Returns:
            "high" | "medium" | "low"
        """
        baseline_fetched_at = model_stats.get("baseline_fetched_at")
        if not baseline_fetched_at:
            return "low"

        age_seconds = time.time() - baseline_fetched_at
        if age_seconds < 300:  # 5 minutes
            return "high"
        elif age_seconds < 1800:  # 30 minutes
            return "medium"
        return "low"

    async def force_refresh_quota(
        self,
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Force refresh quota from external API.

        For Antigravity, this fetches live quota data from the API.
        For other providers, this is a no-op (just reloads from disk).

        Args:
            provider: If specified, only refresh this provider
            credential: If specified, only refresh this specific credential

        Returns:
            Refresh result dict with success/failure info
        """
        result = {
            "action": "force_refresh",
            "scope": (
                "credential" if credential else ("provider" if provider else "all")
            ),
            "provider": provider,
            "credential": credential,
            "credentials_refreshed": 0,
            "success_count": 0,
            "failed_count": 0,
            "duration_ms": 0,
            "errors": [],
        }

        start_time = time.time()

        # Determine which providers to refresh
        if provider:
            providers_to_refresh = (
                [provider] if provider in self.all_credentials else []
            )
        else:
            providers_to_refresh = list(self.all_credentials.keys())

        for prov in providers_to_refresh:
            provider_class = self._provider_plugins.get(prov)
            if not provider_class:
                continue

            # Get or create provider instance
            provider_instance = self._provider_instances.get_or_create(prov, provider_class)

            # Check if provider supports quota refresh (like Antigravity)
            if hasattr(provider_instance, "fetch_initial_baselines"):
                # Get credentials to refresh
                if credential:
                    # Find full path for this credential
                    creds_to_refresh = []
                    for cred_path in self.all_credentials.get(prov, []):
                        if cred_path.endswith(credential) or cred_path == credential:
                            creds_to_refresh.append(cred_path)
                            break
                else:
                    creds_to_refresh = self.all_credentials.get(prov, [])

                if not creds_to_refresh:
                    continue

                try:
                    # Fetch live quota from API for ALL specified credentials
                    quota_results = await provider_instance.fetch_initial_baselines(
                        creds_to_refresh
                    )

                    # Store baselines in usage manager
                    if hasattr(provider_instance, "_store_baselines_to_usage_manager"):
                        stored = (
                            await provider_instance._store_baselines_to_usage_manager(
                                quota_results, self.usage_manager
                            )
                        )
                        result["success_count"] += stored

                    result["credentials_refreshed"] += len(creds_to_refresh)

                    # Count failures
                    for cred_path, data in quota_results.items():
                        if data.get("status") != "success":
                            result["failed_count"] += 1
                            result["errors"].append(
                                f"{Path(cred_path).name}: {data.get('error', 'Unknown error')}"
                            )

                except (RuntimeError, ValueError, OSError, Exception) as e:
                    lib_logger.error(f"Failed to refresh quota for {prov}: {e}")
                    result["errors"].append(f"{prov}: {str(e)}")
                    result["failed_count"] += len(creds_to_refresh)

        result["duration_ms"] = int((time.time() - start_time) * 1000)
        return result
