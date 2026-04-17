# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger, MAX_CACHE_ENTRIES
import time
from pathlib import Path
from typing import Any, Dict, Optional


class UsageManagerStatisticsMixin:
    async def get_stats_for_endpoint(
        self,
        provider_filter: Optional[str] = None,
        include_global: bool = True,
    ) -> Dict[str, Any]:
        """
        Get usage stats formatted for the /v1/quota-stats endpoint.

        Aggregates data from key_usage.json grouped by provider.
        Includes both current period stats and global (lifetime) stats.

        Args:
            provider_filter: If provided, only return stats for this provider
            include_global: If True, include global/lifetime stats alongside current

        Returns:
            {
                "providers": {
                    "provider_name": {
                        "credential_count": int,
                        "active_count": int,
                        "on_cooldown_count": int,
                        "total_requests": int,
                        "tokens": {
                            "input_cached": int,
                            "input_uncached": int,
                            "input_cache_pct": float,
                            "output": int
                        },
                        "approx_cost": float | None,
                        "credentials": [...],
                        "global": {...}  # If include_global is True
                    }
                },
                "summary": {...},
                "global_summary": {...},  # If include_global is True
                "timestamp": float
            }
        """
        await self._lazy_init()

        now_ts = time.time()
        providers: Dict[str, Dict[str, Any]] = {}
        # Track global stats separately
        global_providers: Dict[str, Dict[str, Any]] = {}

        async with self._data_lock.read():
            if not self._usage_data:
                return {
                    "providers": {},
                    "summary": {
                        "total_providers": 0,
                        "total_credentials": 0,
                        "active_credentials": 0,
                        "exhausted_credentials": 0,
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_total_cost": 0.0,
                    },
                    "global_summary": {
                        "total_providers": 0,
                        "total_credentials": 0,
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_total_cost": 0.0,
                    },
                    "data_source": "cache",
                    "timestamp": now_ts,
                }

            for credential, cred_data in self._usage_data.items():
                # Extract provider from credential path
                provider = self._get_provider_from_credential(credential)
                if not provider:
                    continue

                # Apply filter if specified
                if provider_filter and provider != provider_filter:
                    continue

                # Initialize provider entry
                if provider not in providers:
                    providers[provider] = {
                        "credential_count": 0,
                        "active_count": 0,
                        "on_cooldown_count": 0,
                        "exhausted_count": 0,
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_cost": 0.0,
                        "credentials": [],
                    }
                    global_providers[provider] = {
                        "total_requests": 0,
                        "tokens": {
                            "input_cached": 0,
                            "input_uncached": 0,
                            "input_cache_pct": 0,
                            "output": 0,
                        },
                        "approx_cost": 0.0,
                    }

                prov_stats = providers[provider]
                prov_stats["credential_count"] += 1

                # Determine credential status and cooldowns
                key_cooldown = cred_data.get("key_cooldown_until", 0) or 0
                model_cooldowns = cred_data.get("model_cooldowns", {})

                # Build active cooldowns with remaining time
                active_cooldowns = {}
                for model, cooldown_ts in model_cooldowns.items():
                    if cooldown_ts > now_ts:
                        remaining_seconds = int(cooldown_ts - now_ts)
                        active_cooldowns[model] = {
                            "until_ts": cooldown_ts,
                            "remaining_seconds": remaining_seconds,
                        }

                key_cooldown_remaining = None
                if key_cooldown > now_ts:
                    key_cooldown_remaining = int(key_cooldown - now_ts)

                has_active_cooldown = key_cooldown > now_ts or len(active_cooldowns) > 0

                # Check if exhausted (all quota groups exhausted for Antigravity)
                is_exhausted = False
                models_data = cred_data.get("models", {})
                if models_data:
                    # Check if any model has remaining quota
                    all_exhausted = True
                    for model_stats in models_data.values():
                        if isinstance(model_stats, dict):
                            baseline = model_stats.get("baseline_remaining_fraction")
                            if baseline is None or baseline > 0:
                                all_exhausted = False
                                break
                    if all_exhausted and len(models_data) > 0:
                        is_exhausted = True

                if is_exhausted:
                    prov_stats["exhausted_count"] += 1
                    status = "exhausted"
                elif has_active_cooldown:
                    prov_stats["on_cooldown_count"] += 1
                    status = "cooldown"
                else:
                    prov_stats["active_count"] += 1
                    status = "active"

                # Aggregate token stats (current period)
                cred_tokens = {
                    "input_cached": 0,
                    "input_uncached": 0,
                    "input_cache_creation": 0,
                    "output": 0,
                }
                cred_requests = 0
                cred_cost = 0.0

                # Aggregate global token stats
                cred_global_tokens = {
                    "input_cached": 0,
                    "input_uncached": 0,
                    "input_cache_creation": 0,
                    "output": 0,
                }
                cred_global_requests = 0
                cred_global_cost = 0.0

                # Handle per-model structure (current period)
                if models_data:
                    for model_name, model_stats in models_data.items():
                        if not isinstance(model_stats, dict):
                            continue
                        # Prefer request_count if available and non-zero, else fall back to success+failure
                        req_count = model_stats.get("request_count", 0)
                        if req_count > 0:
                            cred_requests += req_count
                        else:
                            cred_requests += model_stats.get("success_count", 0)
                            cred_requests += model_stats.get("failure_count", 0)
                        # Token stats - track cached separately
                        cred_tokens["input_cached"] += model_stats.get(
                            "prompt_tokens_cached", 0
                        )
                        cred_tokens["input_cache_creation"] += model_stats.get(
                            "prompt_tokens_cache_creation", 0
                        )
                        cred_tokens["input_uncached"] += model_stats.get(
                            "prompt_tokens", 0
                        )
                        cred_tokens["output"] += model_stats.get("completion_tokens", 0)
                        cred_cost += model_stats.get("approx_cost", 0.0)

                # Handle legacy daily structure
                daily_data = cred_data.get("daily", {})
                daily_models = daily_data.get("models", {})
                for model_name, model_stats in daily_models.items():
                    if not isinstance(model_stats, dict):
                        continue
                    cred_requests += model_stats.get("success_count", 0)
                    cred_tokens["input_cached"] += model_stats.get(
                        "prompt_tokens_cached", 0
                    )
                    cred_tokens["input_cache_creation"] += model_stats.get(
                        "prompt_tokens_cache_creation", 0
                    )
                    cred_tokens["input_uncached"] += model_stats.get("prompt_tokens", 0)
                    cred_tokens["output"] += model_stats.get("completion_tokens", 0)
                    cred_cost += model_stats.get("approx_cost", 0.0)

                # Handle global stats
                global_data = cred_data.get("global", {})
                global_models = global_data.get("models", {})
                for model_name, model_stats in global_models.items():
                    if not isinstance(model_stats, dict):
                        continue
                    req_count = model_stats.get("request_count", 0)
                    if req_count > 0:
                        cred_global_requests += req_count
                    else:
                        cred_global_requests += model_stats.get("success_count", 0)
                        cred_global_requests += model_stats.get("failure_count", 0)
                    cred_global_tokens["input_cached"] += model_stats.get(
                        "prompt_tokens_cached", 0
                    )
                    cred_global_tokens["input_cache_creation"] += model_stats.get(
                        "prompt_tokens_cache_creation", 0
                    )
                    cred_global_tokens["input_uncached"] += model_stats.get(
                        "prompt_tokens", 0
                    )
                    cred_global_tokens["output"] += model_stats.get(
                        "completion_tokens", 0
                    )
                    cred_global_cost += model_stats.get("approx_cost", 0.0)

                # Add current period stats to global totals
                cred_global_requests += cred_requests
                cred_global_tokens["input_cached"] += cred_tokens["input_cached"]
                cred_global_tokens["input_cache_creation"] += cred_tokens[
                    "input_cache_creation"
                ]
                cred_global_tokens["input_uncached"] += cred_tokens["input_uncached"]
                cred_global_tokens["output"] += cred_tokens["output"]
                cred_global_cost += cred_cost

                # Build credential entry
                # Mask credential identifier for display
                if credential.startswith("env://"):
                    identifier = credential
                else:
                    identifier = Path(credential).name

                cred_entry = {
                    "identifier": identifier,
                    "full_path": credential,
                    "status": status,
                    "last_used_ts": cred_data.get("last_used_ts"),
                    "requests": cred_requests,
                    "tokens": cred_tokens,
                    "approx_cost": cred_cost if cred_cost > 0 else None,
                }

                # Add cooldown info
                if key_cooldown_remaining is not None:
                    cred_entry["key_cooldown_remaining"] = key_cooldown_remaining
                if active_cooldowns:
                    cred_entry["model_cooldowns"] = active_cooldowns

                # Add global stats for this credential
                if include_global:
                    # Calculate global cache percentage
                    global_total_input = (
                        cred_global_tokens["input_cached"]
                        + cred_global_tokens["input_uncached"]
                    )
                    global_cache_pct = (
                        round(
                            cred_global_tokens["input_cached"]
                            / global_total_input
                            * 100,
                            1,
                        )
                        if global_total_input > 0
                        else 0
                    )

                    cred_entry["global"] = {
                        "requests": cred_global_requests,
                        "tokens": {
                            "input_cached": cred_global_tokens["input_cached"],
                            "input_cache_creation": cred_global_tokens[
                                "input_cache_creation"
                            ],
                            "input_uncached": cred_global_tokens["input_uncached"],
                            "input_cache_pct": global_cache_pct,
                            "output": cred_global_tokens["output"],
                        },
                        "approx_cost": (
                            cred_global_cost if cred_global_cost > 0 else None
                        ),
                    }

                # Add model-specific data for providers with per-model tracking
                if models_data:
                    cred_entry["models"] = {}
                    for model_name, model_stats in models_data.items():
                        if not isinstance(model_stats, dict):
                            continue
                        cred_entry["models"][model_name] = {
                            "requests": model_stats.get("success_count", 0)
                            + model_stats.get("failure_count", 0),
                            "request_count": model_stats.get("request_count", 0),
                            "success_count": model_stats.get("success_count", 0),
                            "failure_count": model_stats.get("failure_count", 0),
                            "prompt_tokens": model_stats.get("prompt_tokens", 0),
                            "prompt_tokens_cached": model_stats.get(
                                "prompt_tokens_cached", 0
                            ),
                            "prompt_tokens_cache_creation": model_stats.get(
                                "prompt_tokens_cache_creation", 0
                            ),
                            "completion_tokens": model_stats.get(
                                "completion_tokens", 0
                            ),
                            "approx_cost": model_stats.get("approx_cost", 0.0),
                            "window_start_ts": model_stats.get("window_start_ts"),
                            "quota_reset_ts": model_stats.get("quota_reset_ts"),
                            # Quota baseline fields (Antigravity-specific)
                            "baseline_remaining_fraction": model_stats.get(
                                "baseline_remaining_fraction"
                            ),
                            "baseline_fetched_at": model_stats.get(
                                "baseline_fetched_at"
                            ),
                            "quota_max_requests": model_stats.get("quota_max_requests"),
                            "quota_display": model_stats.get("quota_display"),
                        }

                prov_stats["credentials"].append(cred_entry)

                # Aggregate to provider totals (current period)
                prov_stats["total_requests"] += cred_requests
                prov_stats["tokens"]["input_cached"] += cred_tokens["input_cached"]
                prov_stats["tokens"]["input_uncached"] += cred_tokens["input_uncached"]
                prov_stats["tokens"]["output"] += cred_tokens["output"]
                if cred_cost > 0:
                    prov_stats["approx_cost"] += cred_cost

                # Aggregate to global provider totals
                global_providers[provider]["total_requests"] += cred_global_requests
                global_providers[provider]["tokens"][
                    "input_cached"
                ] += cred_global_tokens["input_cached"]
                global_providers[provider]["tokens"][
                    "input_uncached"
                ] += cred_global_tokens["input_uncached"]
                global_providers[provider]["tokens"]["output"] += cred_global_tokens[
                    "output"
                ]
                global_providers[provider]["approx_cost"] += cred_global_cost

        # Calculate cache percentages for each provider
        for provider, prov_stats in providers.items():
            total_input = (
                prov_stats["tokens"]["input_cached"]
                + prov_stats["tokens"]["input_uncached"]
            )
            if total_input > 0:
                prov_stats["tokens"]["input_cache_pct"] = round(
                    prov_stats["tokens"]["input_cached"] / total_input * 100, 1
                )
            # Set cost to None if 0
            if prov_stats["approx_cost"] == 0:
                prov_stats["approx_cost"] = None

            # Calculate global cache percentages
            if include_global and provider in global_providers:
                gp = global_providers[provider]
                global_total = (
                    gp["tokens"]["input_cached"] + gp["tokens"]["input_uncached"]
                )
                if global_total > 0:
                    gp["tokens"]["input_cache_pct"] = round(
                        gp["tokens"]["input_cached"] / global_total * 100, 1
                    )
                if gp["approx_cost"] == 0:
                    gp["approx_cost"] = None
                prov_stats["global"] = gp

        # Build summary (current period)
        total_creds = sum(p["credential_count"] for p in providers.values())
        active_creds = sum(p["active_count"] for p in providers.values())
        exhausted_creds = sum(p["exhausted_count"] for p in providers.values())
        total_requests = sum(p["total_requests"] for p in providers.values())
        total_input_cached = sum(
            p["tokens"]["input_cached"] for p in providers.values()
        )
        total_input_uncached = sum(
            p["tokens"]["input_uncached"] for p in providers.values()
        )
        total_output = sum(p["tokens"]["output"] for p in providers.values())
        total_cost = sum(p["approx_cost"] or 0 for p in providers.values())

        total_input = total_input_cached + total_input_uncached
        input_cache_pct = (
            round(total_input_cached / total_input * 100, 1) if total_input > 0 else 0
        )

        result = {
            "providers": providers,
            "summary": {
                "total_providers": len(providers),
                "total_credentials": total_creds,
                "active_credentials": active_creds,
                "exhausted_credentials": exhausted_creds,
                "total_requests": total_requests,
                "tokens": {
                    "input_cached": total_input_cached,
                    "input_uncached": total_input_uncached,
                    "input_cache_pct": input_cache_pct,
                    "output": total_output,
                },
                "approx_total_cost": total_cost if total_cost > 0 else None,
            },
            "data_source": "cache",
            "timestamp": now_ts,
        }

        # Build global summary
        if include_global:
            global_total_requests = sum(
                gp["total_requests"] for gp in global_providers.values()
            )
            global_total_input_cached = sum(
                gp["tokens"]["input_cached"] for gp in global_providers.values()
            )
            global_total_input_uncached = sum(
                gp["tokens"]["input_uncached"] for gp in global_providers.values()
            )
            global_total_output = sum(
                gp["tokens"]["output"] for gp in global_providers.values()
            )
            global_total_cost = sum(
                gp["approx_cost"] or 0 for gp in global_providers.values()
            )

            global_total_input = global_total_input_cached + global_total_input_uncached
            global_input_cache_pct = (
                round(global_total_input_cached / global_total_input * 100, 1)
                if global_total_input > 0
                else 0
            )

            result["global_summary"] = {
                "total_providers": len(global_providers),
                "total_credentials": total_creds,
                "total_requests": global_total_requests,
                "tokens": {
                    "input_cached": global_total_input_cached,
                    "input_uncached": global_total_input_uncached,
                    "input_cache_pct": global_input_cache_pct,
                    "output": global_total_output,
                },
                "approx_total_cost": (
                    global_total_cost if global_total_cost > 0 else None
                ),
            }

        return result
