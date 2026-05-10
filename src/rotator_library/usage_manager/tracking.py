# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Usage recording, querying, statistics, and formatting."""

import time
import asyncio
from .manager import MAX_CACHE_ENTRIES, lib_logger
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import litellm  # type: ignore[import-untyped]
from ..config.defaults import TRACE
from ..config import (
    COOLDOWN_BACKOFF_TIERS,
    COOLDOWN_BACKOFF_MAX,
    COOLDOWN_AUTH_ERROR,
    COOLDOWN_TRANSIENT_ERROR,
    COOLDOWN_RATE_LIMIT_DEFAULT,
)
from ..error_types import ClassifiedError, mask_credential
from ..utils.litellm_patches import suppress_litellm_prints
from ..utils.model_utils import extract_provider_from_model


def _new_model_data(**overrides):
    """Create a fresh per-model usage tracking dict."""
    return {
        "window_start_ts": None,
        "quota_reset_ts": None,
        "success_count": 0,
        "failure_count": 0,
        "request_count": 0,
        "prompt_tokens": 0,
        "prompt_tokens_cached": 0,
        "prompt_tokens_cache_creation": 0,
        "completion_tokens": 0,
        "thinking_tokens": 0,
        "approx_cost": 0.0,
        **overrides,
    }


def _new_key_data(reset_mode: str, today_utc_str: str) -> dict:
    """Create a fresh key-level usage tracking dict for the given reset mode."""
    if reset_mode == "per_model":
        return {
            "models": {},
            "global": {"models": {}},
            "model_cooldowns": {},
            "failures": {},
        }
    return {
        "daily": {"date": today_utc_str, "models": {}},
        "global": {"models": {}},
        "model_cooldowns": {},
        "failures": {},
    }


def _was_already_on_cooldown(
    model_cooldowns: dict, model: str, now_ts: float
) -> bool:
    """Check if a model already has an active cooldown before modification."""
    existing = model_cooldowns.get(model)
    return existing is not None and existing > now_ts


def _sync_quota_group_fields(
    usage_manager, models_data: dict, model: str, credential: str,
    fields: Dict[str, Any],
) -> list:
    """Sync specified fields to all sibling models sharing a quota group.

    Returns list of (sibling_model, sibling_data) tuples for callers that
    need additional per-sibling processing (e.g., cooldown sync).
    """
    group = usage_manager._get_model_quota_group(credential, model)
    if not group:
        return []
    grouped_models = usage_manager._get_grouped_models(credential, group)
    siblings = []
    for grouped_model in grouped_models:
        if grouped_model == model:
            continue
        other_model_data = models_data.setdefault(grouped_model, _new_model_data())
        for field_name, field_value in fields.items():
            other_model_data[field_name] = field_value
        siblings.append((grouped_model, other_model_data))
    return siblings


def _maybe_mark_fair_cycle_exhausted(
    usage_manager, credential: str, model: str, cooldown_duration: float,
    provider: Optional[str] = None, priority: Optional[int] = None,
) -> None:
    """Mark credential exhausted in fair cycle if cooldown exceeds threshold."""
    if provider is None:
        provider = usage_manager._get_provider_from_credential(credential)
    if not provider:
        return
    threshold = usage_manager._get_exhaustion_cooldown_threshold(provider)
    if cooldown_duration <= threshold:
        return
    rotation_mode = usage_manager._get_rotation_mode(provider)
    if not usage_manager._is_fair_cycle_enabled(provider, rotation_mode):
        return
    if priority is None:
        priority = usage_manager._get_credential_priority(credential, provider)
    tier_key = usage_manager._get_tier_key(provider, priority)
    tracking_key = usage_manager._get_tracking_key(credential, model, provider)
    usage_manager._mark_credential_exhausted(
        credential, provider, tier_key, tracking_key
    )


class UsageManagerRecordingMixin:
    async def record_success(
        self,
        key: str,
        model: str,
        completion_response: Optional[litellm.ModelResponse] = None,
    ):
        """
        Records a successful API call, resetting failure counters.
        It safely handles cases where token usage data is not available.

        Supports two modes based on provider configuration:
        - per_model: Each model has its own window_start_ts and stats in key_data["models"]
        - credential: Legacy mode with key_data["daily"]["models"]
        """
        await self._lazy_init()

        # Normalize model name to public-facing name for consistent tracking
        model = self._normalize_model_for_tracking(key, model)

        async with self._data_lock.write():
            now_ts = time.time()
            today_utc_str = datetime.now(timezone.utc).date().isoformat()

            reset_config = self._get_usage_reset_config(key)
            reset_mode = (
                reset_config.get("mode", "credential") if reset_config else "credential"
            )

            if reset_mode == "per_model":
                # New per-model structure
                key_data = self._usage_data.setdefault(
                    key, _new_key_data("per_model", today_utc_str),
                )

                # Ensure models dict exists
                if "models" not in key_data:
                    key_data["models"] = {}

                # Get or create per-model data with window tracking
                model_data = key_data["models"].setdefault(
                    model, _new_model_data(),
                )

                # Start window on first request for this model
                if model_data.get("window_start_ts") is None:
                    model_data["window_start_ts"] = now_ts

                    # Set expected quota reset time from provider config
                    window_seconds = (
                        reset_config.get("window_seconds", 0) if reset_config else 0
                    )
                    if window_seconds > 0:
                        model_data["quota_reset_ts"] = now_ts + window_seconds

                    window_hours = window_seconds / 3600 if window_seconds else 0
                    lib_logger.info(
                        f"Started {window_hours:.1f}h window for model {model} on {mask_credential(key)}"
                    )

                # Record stats
                model_data["success_count"] += 1
                model_data["request_count"] = model_data.get("request_count", 0) + 1

                # Sync request_count across quota group (for providers with shared quota pools)
                new_request_count = model_data["request_count"]
                sync_fields: Dict[str, Any] = {"request_count": new_request_count}
                window_start = model_data.get("window_start_ts")
                if window_start:
                    sync_fields["window_start_ts"] = window_start
                max_req = model_data.get("quota_max_requests")
                if max_req:
                    sync_fields["quota_max_requests"] = max_req
                    sync_fields["quota_display"] = f"{new_request_count}/{max_req}"
                _sync_quota_group_fields(self, key_data["models"], model, key, sync_fields)

                # Update quota_display if max_requests is set (Antigravity-specific)
                max_req = model_data.get("quota_max_requests")
                if max_req:
                    model_data["quota_display"] = (
                        f"{model_data['request_count']}/{max_req}"
                    )

                # Check custom cap
                if self._check_and_apply_custom_cap(
                    key, model, model_data["request_count"]
                ):
                    # Custom cap exceeded, cooldown applied
                    # Continue to record tokens/cost but credential will be skipped next time
                    pass

                usage_data_ref = model_data  # For token/cost recording below

            else:
                # Legacy credential-level structure
                key_data = self._usage_data.setdefault(
                    key, _new_key_data("credential", today_utc_str),
                )

                if "last_daily_reset" not in key_data:
                    key_data["last_daily_reset"] = today_utc_str

                # Get or create model data in daily structure
                usage_data_ref = key_data["daily"]["models"].setdefault(
                    model,
                    {
                        "success_count": 0,
                        "prompt_tokens": 0,
                        "prompt_tokens_cached": 0,
                        "prompt_tokens_cache_creation": 0,
                        "completion_tokens": 0,
                        "thinking_tokens": 0,
                        "approx_cost": 0.0,
                    },
                )
                usage_data_ref["success_count"] += 1

            # Reset failures for this model
            model_failures = key_data.setdefault("failures", {}).setdefault(model, {})
            model_failures["consecutive_failures"] = 0

            # Clear transient cooldown on success (but NOT quota_reset_ts)
            if model in key_data.get("model_cooldowns", {}):
                del key_data["model_cooldowns"][model]

            # Record token and cost usage
            if (
                completion_response
                and hasattr(completion_response, "usage")
                and completion_response.usage
            ):
                usage = completion_response.usage
                prompt_total = usage.prompt_tokens

                # Extract cached tokens from prompt_tokens_details if present
                cached_tokens = 0
                cache_creation_tokens = 0
                prompt_details = getattr(usage, "prompt_tokens_details", None)
                if prompt_details:
                    if isinstance(prompt_details, dict):
                        cached_tokens = prompt_details.get("cached_tokens", 0) or 0
                        cache_creation_tokens = (
                            prompt_details.get("cache_creation_tokens", 0) or 0
                        )
                    elif hasattr(prompt_details, "cached_tokens"):
                        cached_tokens = prompt_details.cached_tokens or 0
                    if hasattr(prompt_details, "cache_creation_tokens"):
                        cache_creation_tokens = (
                            prompt_details.cache_creation_tokens or 0
                        )

                # Store uncached tokens (prompt_tokens is total, subtract cached and cache_creation)
                uncached_tokens = prompt_total - cached_tokens - cache_creation_tokens
                usage_data_ref["prompt_tokens"] += uncached_tokens

                # Store cached tokens separately
                if cached_tokens > 0:
                    usage_data_ref["prompt_tokens_cached"] = (
                        usage_data_ref.get("prompt_tokens_cached", 0) + cached_tokens
                    )

                # Store cache creation tokens separately (tokens used to create the cache)
                if cache_creation_tokens > 0:
                    usage_data_ref["prompt_tokens_cache_creation"] = (
                        usage_data_ref.get("prompt_tokens_cache_creation", 0)
                        + cache_creation_tokens
                    )

                # Extract thinking/reasoning tokens from various provider formats
                thinking_tokens = 0
                # Try Anthropic/OpenAI style: completion_tokens_details.reasoning_tokens
                completion_details = getattr(usage, "completion_tokens_details", None)
                if completion_details:
                    if isinstance(completion_details, dict):
                        thinking_tokens = (
                            completion_details.get("reasoning_tokens", 0) or 0
                        )
                    elif hasattr(completion_details, "reasoning_tokens"):
                        thinking_tokens = completion_details.reasoning_tokens or 0
                # Try DeepSeek style: direct reasoning_tokens field
                if thinking_tokens == 0:
                    thinking_tokens = getattr(usage, "reasoning_tokens", 0) or 0

                raw_completion_tokens = getattr(usage, "completion_tokens", 0)
                # Effective completion tokens exclude thinking tokens (they are not charged)
                effective_completion_tokens = raw_completion_tokens - thinking_tokens
                usage_data_ref["completion_tokens"] += effective_completion_tokens

                # Store thinking tokens separately if present
                if thinking_tokens > 0:
                    usage_data_ref["thinking_tokens"] = (
                        usage_data_ref.get("thinking_tokens", 0) + thinking_tokens
                    )
                lib_logger.info(
                    f"Recorded usage from response object for key {mask_credential(key)}"
                )
                try:
                    provider_name = extract_provider_from_model(model)
                    provider_instance = self._get_provider_instance(provider_name)

                    if not provider_instance or getattr(
                        provider_instance, "skip_cost_calculation", False
                    ):
                        lib_logger.debug(
                            f"Skipping cost calculation for provider '{provider_name}'"
                            f" ({'no plugin' if not provider_instance else 'custom provider'})."
                        )
                    else:
                        # Suppress LiteLLM's direct print() statements for unknown providers
                        # LiteLLM prints "Provider List: https://..." spam for unknown models
                        with suppress_litellm_prints():
                            if isinstance(
                                completion_response, litellm.EmbeddingResponse
                            ):
                                model_info = litellm.get_model_info(model)
                                input_cost = model_info.get("input_cost_per_token")
                                if input_cost and hasattr(completion_response, "usage") and completion_response.usage:
                                    cost = (
                                        completion_response.usage.prompt_tokens
                                        * input_cost
                                    )
                                else:
                                    cost = None
                            else:
                                cost = litellm.completion_cost(
                                    completion_response=completion_response, model=model
                                )

                        if cost is not None:
                            usage_data_ref["approx_cost"] += cost
                except Exception as e:
                    # LiteLLM currently raises a generic Exception for cost map misses,
                    # so string matching is required until a typed exception is exposed.
                    if "not found in cost map" in str(e).lower():
                        lib_logger.debug(f"Cost map entry missing for model {model}, skipping cost calculation")
                    else:
                        lib_logger.debug(f"Could not calculate cost for model {model}: {e}")
            elif isinstance(completion_response, asyncio.Future) or hasattr(
                completion_response, "__aiter__"
            ):
                pass  # Stream - usage recorded from chunks
            else:
                lib_logger.warning(
                    f"No usage data found in completion response for model {model}. Recording success without token count."
                )

            key_data["last_used_ts"] = now_ts
            self._clear_provider_resolution_cache()

        await self._save_usage()

    async def record_failure(
        self,
        key: str,
        model: str,
        classified_error: ClassifiedError,
        increment_consecutive_failures: bool = True,
    ):
        """Records a failure and applies cooldowns based on error type.

        Distinguishes between:
        - quota_exceeded: Long cooldown with exact reset time (from quota_reset_timestamp)
          Sets quota_reset_ts on model (and group) - this becomes authoritative stats reset time
        - rate_limit: Short transient cooldown (just wait and retry)
          Only sets model_cooldowns - does NOT affect stats reset timing

        Args:
            key: The API key or credential identifier
            model: The model name
            classified_error: The classified error object
            increment_consecutive_failures: Whether to increment the failure counter.
                Set to False for provider-level errors that shouldn't count against the key.
        """
        # Skip usage tracking for server errors (503) to prevent incorrect quota exhaustion
        # These are transient provider issues, not actual consumed quota
        if classified_error.status_code == 503:
            lib_logger.debug(
                f"Skipping usage tracking for 503 error on {mask_credential(key)} - "
                f"transient server error, not a quota consumption"
            )
            return

        await self._lazy_init()

        # Normalize model name to public-facing name for consistent tracking
        model = self._normalize_model_for_tracking(key, model)

        async with self._data_lock.write():
            now_ts = time.time()
            today_utc_str = datetime.now(timezone.utc).date().isoformat()

            reset_config = self._get_usage_reset_config(key)
            reset_mode = (
                reset_config.get("mode", "credential") if reset_config else "credential"
            )

            # Initialize key data with appropriate structure
            if reset_mode == "per_model":
                key_data = self._usage_data.setdefault(
                    key, _new_key_data("per_model", today_utc_str),
                )
            else:
                key_data = self._usage_data.setdefault(
                    key, _new_key_data("credential", today_utc_str),
                )

            # Provider-level errors (transient issues) should not count against the key
            provider_level_errors = {"server_error", "api_connection", "ip_rate_limit"}

            # Determine if we should increment the failure counter
            should_increment = (
                increment_consecutive_failures
                and classified_error.error_type not in provider_level_errors
            )

            # Calculate cooldown duration based on error type
            cooldown_seconds = None
            model_cooldowns = key_data.setdefault("model_cooldowns", {})

            # Capture existing cooldown BEFORE we modify it
            # Used to determine if this is a fresh exhaustion vs re-processing
            was_already_on_cooldown = _was_already_on_cooldown(
                model_cooldowns, model, now_ts
            )

            if classified_error.error_type == "quota_exceeded":
                # Quota exhausted - use authoritative reset timestamp if available
                quota_reset_ts = classified_error.quota_reset_timestamp
                cooldown_seconds = (
                    classified_error.retry_after or COOLDOWN_RATE_LIMIT_DEFAULT
                )

                if quota_reset_ts and reset_mode == "per_model":
                    # Set quota_reset_ts on model - this becomes authoritative stats reset time
                    models_data = key_data.setdefault("models", {})
                    model_data = models_data.setdefault(
                        model, _new_model_data(),
                    )
                    model_data["quota_reset_ts"] = quota_reset_ts
                    # Track failure for quota estimation (request still consumes quota)
                    model_data["failure_count"] = model_data.get("failure_count", 0) + 1
                    model_data["request_count"] = model_data.get("request_count", 0) + 1

                    # Clamp request_count to quota_max_requests when quota is exhausted
                    # This prevents display overflow (e.g., 151/150) when requests are
                    # counted locally before API refresh corrects the value
                    max_req = model_data.get("quota_max_requests")
                    if max_req is not None and model_data["request_count"] > max_req:
                        model_data["request_count"] = max_req
                        # Update quota_display with clamped value
                        model_data["quota_display"] = f"{max_req}/{max_req}"
                    new_request_count = model_data["request_count"]

                    # Apply to all models in the same quota group
                    group = self._get_model_quota_group(key, model)
                    if group:
                        # Invalidate group cache so newly discovered
                        # models are included in propagation
                        group_cache = self._grouped_models_cache.get(key)
                        if group_cache and group in group_cache:
                            del group_cache[group]
                        grouped_models = self._get_grouped_models(key, group)
                        for grouped_model in grouped_models:
                            group_model_data = models_data.setdefault(
                                grouped_model, _new_model_data(),
                            )
                            group_model_data["quota_reset_ts"] = quota_reset_ts
                            # Sync request_count across quota group
                            group_model_data["request_count"] = new_request_count
                            # Also sync quota_max_requests if set
                            max_req = model_data.get("quota_max_requests")
                            if max_req:
                                group_model_data["quota_max_requests"] = max_req
                                group_model_data["quota_display"] = (
                                    f"{new_request_count}/{max_req}"
                                )
                            # Also set transient cooldown for selection logic
                            model_cooldowns[grouped_model] = quota_reset_ts

                        reset_dt = datetime.fromtimestamp(
                            quota_reset_ts, tz=timezone.utc
                        )
                        lib_logger.info(
                            f"Quota exhausted for group '{group}' ({len(grouped_models)} models) "
                            f"on {mask_credential(key)}. Resets at {reset_dt.isoformat()}"
                        )
                    else:
                        reset_dt = datetime.fromtimestamp(
                            quota_reset_ts, tz=timezone.utc
                        )
                        hours = (quota_reset_ts - now_ts) / 3600
                        lib_logger.info(
                            f"Quota exhausted for model {model} on {mask_credential(key)}. "
                            f"Resets at {reset_dt.isoformat()} ({hours:.1f}h)"
                        )

                    # Set transient cooldown for selection logic
                    model_cooldowns[model] = quota_reset_ts
                else:
                    # No authoritative timestamp or legacy mode - just use retry_after
                    model_cooldowns[model] = now_ts + cooldown_seconds
                    hours = cooldown_seconds / 3600
                    lib_logger.info(
                        f"Quota exhausted on {mask_credential(key)} for model {model}. "
                        f"Cooldown: {cooldown_seconds}s ({hours:.1f}h)"
                    )

                # Mark credential as exhausted for fair cycle if cooldown exceeds threshold
                # BUT only if this is a FRESH exhaustion (wasn't already on cooldown)
                # This prevents re-marking after cycle reset
                if not was_already_on_cooldown:
                    effective_cooldown = (
                        (quota_reset_ts - now_ts)
                        if quota_reset_ts
                        else (cooldown_seconds or 0)
                    )
                    _maybe_mark_fair_cycle_exhausted(
                        self, key, model, effective_cooldown,
                    )

            elif classified_error.error_type == "rate_limit":
                # Transient rate limit - just set short cooldown (does NOT set quota_reset_ts)
                cooldown_seconds = (
                    classified_error.retry_after or COOLDOWN_RATE_LIMIT_DEFAULT
                )
                model_cooldowns[model] = now_ts + cooldown_seconds
                lib_logger.info(
                    f"Rate limit on {mask_credential(key)} for model {model}. "
                    f"Transient cooldown: {cooldown_seconds}s"
                )

            elif classified_error.error_type == "authentication":
                # Apply a 5-minute key-level lockout for auth errors
                key_data["key_cooldown_until"] = now_ts + COOLDOWN_AUTH_ERROR
                cooldown_seconds = COOLDOWN_AUTH_ERROR
                model_cooldowns[model] = now_ts + cooldown_seconds
                lib_logger.warning(
                    f"Authentication error on key {mask_credential(key)}. Applying 5-minute key-level lockout."
                )

            # If we should increment failures, calculate escalating backoff
            if should_increment:
                failures_data = key_data.setdefault("failures", {})
                model_failures = failures_data.setdefault(
                    model, {"consecutive_failures": 0}
                )
                model_failures["consecutive_failures"] += 1
                count = model_failures["consecutive_failures"]

                # If cooldown wasn't set by specific error type, use escalating backoff
                if cooldown_seconds is None:
                    cooldown_seconds = COOLDOWN_BACKOFF_TIERS.get(
                        count, COOLDOWN_BACKOFF_MAX
                    )
                    model_cooldowns[model] = now_ts + cooldown_seconds
                    lib_logger.warning(
                        f"Failure #{count} for key {mask_credential(key)} with model {model}. "
                        f"Error type: {classified_error.error_type}, cooldown: {cooldown_seconds}s"
                    )
            else:
                # Provider-level errors: apply short cooldown but don't count against key
                if cooldown_seconds is None:
                    cooldown_seconds = COOLDOWN_TRANSIENT_ERROR
                    model_cooldowns[model] = now_ts + cooldown_seconds
                lib_logger.info(
                    f"Provider-level error ({classified_error.error_type}) for key {mask_credential(key)} "
                    f"with model {model}. NOT incrementing failures. Cooldown: {cooldown_seconds}s"
                )

            # Check for key-level lockout condition
            await self._check_key_lockout(key, key_data)

            # Track failure count for quota estimation (all failures consume quota)
            # This is separate from consecutive_failures which is for backoff logic
            if reset_mode == "per_model":
                models_data = key_data.setdefault("models", {})
                model_data = models_data.setdefault(
                    model, _new_model_data(),
                )
                # Only increment if not already incremented in quota_exceeded branch
                if classified_error.error_type != "quota_exceeded":
                    model_data["failure_count"] = model_data.get("failure_count", 0) + 1
                    model_data["request_count"] = model_data.get("request_count", 0) + 1

                    # Sync request_count across quota group
                    new_request_count = model_data["request_count"]
                    sync_fields: Dict[str, Any] = {"request_count": new_request_count}
                    max_req = model_data.get("quota_max_requests")
                    if max_req:
                        sync_fields["quota_max_requests"] = max_req
                        sync_fields["quota_display"] = f"{new_request_count}/{max_req}"
                    _sync_quota_group_fields(self, models_data, model, key, sync_fields)

            key_data["last_failure"] = {
                "timestamp": now_ts,
                "model": model,
                "error": str(classified_error.original_exception),
            }
            self._clear_provider_resolution_cache()

        await self._save_usage()

    async def update_quota_baseline(
        self,
        credential: str,
        model: str,
        remaining_fraction: float,
        max_requests: Optional[int] = None,
        reset_timestamp: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update quota baseline data for a credential/model after fetching from API.

        This stores the current quota state as a baseline, which is used to
        estimate remaining quota based on subsequent request counts.

        When quota is exhausted (remaining_fraction <= 0.0) and a valid reset_timestamp
        is provided, this also sets model_cooldowns to prevent wasted requests.

        Args:
            credential: Credential identifier (file path or env:// URI)
            model: Model name (with or without provider prefix)
            remaining_fraction: Current remaining quota as fraction (0.0 to 1.0)
            max_requests: Maximum requests allowed per quota period (e.g., 250 for Claude)
            reset_timestamp: Unix timestamp when quota resets. Only trusted when
                remaining_fraction < 1.0 (quota has been used). API returns garbage
                reset times for unused quota (100%).

        Returns:
            None if no cooldown was set/updated, otherwise:
            {
                "group_or_model": str,  # quota group name or model name if ungrouped
                "hours_until_reset": float,
            }
        """
        await self._lazy_init()
        async with self._data_lock.write():
            now_ts = time.time()

            # Get or create key data structure
            key_data = self._usage_data.setdefault(
                credential, _new_key_data("per_model", ""),
            )

            # Ensure models dict exists
            if "models" not in key_data:
                key_data["models"] = {}

            # Get or create per-model data
            model_data = key_data["models"].setdefault(
                model, _new_model_data(
                    baseline_remaining_fraction=None,
                    baseline_fetched_at=None,
                    requests_at_baseline=None,
                ),
            )

            # Calculate actual used requests from API's remaining fraction
            # The API is authoritative - sync our local count to match reality
            if max_requests is not None:
                used_requests = int((1.0 - remaining_fraction) * max_requests)
            else:
                # Estimate max_requests from provider's quota cost
                # This matches how get_max_requests_for_model() calculates it
                provider = self._get_provider_from_credential(credential)
                plugin_instance = self._get_provider_instance(provider)
                if plugin_instance and hasattr(
                    plugin_instance, "get_max_requests_for_model"
                ):
                    # Get tier from provider's cache
                    tier = getattr(plugin_instance, "project_tier_cache", {}).get(
                        credential, "standard-tier"
                    )
                    # Strip provider prefix from model if present
                    clean_model = model.split("/")[-1] if "/" in model else model
                    max_requests = plugin_instance.get_max_requests_for_model(
                        clean_model, tier
                    )
                    used_requests = int((1.0 - remaining_fraction) * (max_requests or 0))
                else:
                    # Fallback: keep existing count if we can't calculate
                    used_requests = model_data.get("request_count", 0)
                    max_requests = model_data.get("quota_max_requests")

            # Sync local request count to API's authoritative value
            # Use max() to prevent API from resetting our count if it returns stale/cached 100%
            # The API can only increase our count (if we missed requests), not decrease it
            # See: https://github.com/ShmidtS/LLM-API-Key-Proxy/issues/75
            current_count = model_data.get("request_count", 0)
            synced_count = max(current_count, used_requests)
            model_data["request_count"] = synced_count
            model_data["requests_at_baseline"] = synced_count

            # Update baseline fields
            model_data["baseline_remaining_fraction"] = remaining_fraction
            model_data["baseline_fetched_at"] = now_ts

            # Update max_requests and quota_display
            if max_requests is not None:
                model_data["quota_max_requests"] = max_requests
                model_data["quota_display"] = f"{synced_count}/{max_requests}"

            # Handle reset_timestamp: only trust it when quota has been used (< 100%)
            # API returns garbage reset times for unused quota
            valid_reset_ts = (
                reset_timestamp is not None
                and remaining_fraction < 1.0
                and reset_timestamp > now_ts
            )

            if valid_reset_ts:
                model_data["quota_reset_ts"] = reset_timestamp

            # Set cooldowns when quota is exhausted
            model_cooldowns = key_data.setdefault("model_cooldowns", {})
            is_exhausted = remaining_fraction <= 0.0
            cooldown_set_info = (
                None  # Will be returned if cooldown was newly set/updated
            )

            if is_exhausted and valid_reset_ts:
                # Check if there was an existing ACTIVE cooldown before we update
                # This distinguishes between fresh exhaustion vs refresh of existing state
                was_already_on_cooldown = _was_already_on_cooldown(
                    model_cooldowns, model, now_ts
                )

                # Only update cooldown if not set or differs by more than 5 minutes
                existing_cooldown = model_cooldowns.get(model)
                should_update = (
                    existing_cooldown is None
                    or abs(existing_cooldown - reset_timestamp) > 300
                )
                if should_update:
                    model_cooldowns[model] = reset_timestamp
                    hours_until_reset = (reset_timestamp - now_ts) / 3600  # type: ignore[operator]
                    # Determine group or model name for logging
                    group = self._get_model_quota_group(credential, model)
                    cooldown_set_info = {
                        "group_or_model": group if group else model.split("/")[-1],
                        "hours_until_reset": hours_until_reset,
                    }

                # Mark credential as exhausted in fair cycle if cooldown exceeds threshold
                # BUT only if this is a FRESH exhaustion (wasn't already on cooldown)
                # This prevents re-marking after cycle reset when quota refresh sees existing cooldown
                if not was_already_on_cooldown:
                    cooldown_duration = reset_timestamp - now_ts  # type: ignore[operator]
                    _maybe_mark_fair_cycle_exhausted(
                        self, credential, model, cooldown_duration,
                    )

                # Defensive clamp: ensure request_count doesn't exceed max when exhausted
                if (
                    max_requests is not None
                    and model_data["request_count"] > max_requests
                ):
                    model_data["request_count"] = max_requests
                    model_data["quota_display"] = f"{max_requests}/{max_requests}"

            # Sync baseline fields and quota info across quota group
            group = self._get_model_quota_group(credential, model)
            if group:
                # Invalidate group cache when updating a virtual _quota model
                # so that newly discovered models are included in propagation
                if model.endswith("/_quota"):
                    group_cache = self._grouped_models_cache.get(credential)
                    if group_cache and group in group_cache:
                        del group_cache[group]
                # Build sync fields dict
                sync_fields: Dict[str, Any] = {
                    "request_count": synced_count,
                    "baseline_remaining_fraction": remaining_fraction,
                    "baseline_fetched_at": now_ts,
                    "requests_at_baseline": synced_count,
                }
                if max_requests is not None:
                    sync_fields["quota_max_requests"] = max_requests
                    sync_fields["quota_display"] = f"{synced_count}/{max_requests}"
                if valid_reset_ts:
                    sync_fields["quota_reset_ts"] = reset_timestamp
                window_start = model_data.get("window_start_ts")
                if window_start:
                    sync_fields["window_start_ts"] = window_start
                siblings = _sync_quota_group_fields(
                    self, key_data["models"], model, credential, sync_fields,
                )
                # Sync cooldown if exhausted (with ±5 min check)
                if is_exhausted and valid_reset_ts:
                    for grouped_model, other_model_data in siblings:
                        existing_grouped = model_cooldowns.get(grouped_model)
                        should_update_grouped = (
                            existing_grouped is None
                            or abs(existing_grouped - reset_timestamp) > 300
                        )
                        if should_update_grouped:
                            model_cooldowns[grouped_model] = reset_timestamp
                        # Defensive clamp for grouped models when exhausted
                        if (
                            max_requests is not None
                            and other_model_data["request_count"] > max_requests
                        ):
                            other_model_data["request_count"] = max_requests
                            other_model_data["quota_display"] = (
                                f"{max_requests}/{max_requests}"
                            )

            lib_logger.log(
                TRACE,
                f"Updated quota baseline for {mask_credential(credential)} model={model}: "
                f"remaining={remaining_fraction:.2%}, synced_request_count={synced_count}"
            )
            self._clear_provider_resolution_cache()

        await self._save_usage()
        return cooldown_set_info

    async def _check_key_lockout(self, key: str, key_data: Dict):
        """
        Checks if a key should be locked out due to multiple model failures.

        NOTE: This check is currently disabled. The original logic counted individual
        models in long-term lockout, but this caused issues with quota groups - when
        a single quota group (e.g., "claude" with 5 models) was exhausted, it would
        count as 5 lockouts and trigger key-level lockout, blocking other quota groups
        (like gemini) that were still available.

        The per-model and per-group cooldowns already handle quota exhaustion properly.
        """
        # Disabled - see docstring above
        pass



# --- Query Mixin ---

import re
import time
from collections import OrderedDict, namedtuple
from typing import Optional, Dict, List, Any
from ..utils.model_utils import get_or_create_provider_instance
from ..error_types import mask_credential


# Combined regex for the two OAuth path patterns (previously two separate regexes):
#   /provider_oauth_N.json$   or   oauth_creds/provider_oauth_N.json$
# The _oauth_\d+\.json$ anchor forces correct backtracking so [a-z_]+ captures
# only the provider name (e.g., "antigravity") not the full stem ("antigravity_oauth").
_OAUTH_PROVIDER_RE = re.compile(
    r"(?:/|oauth_creds/)([a-z_]+)_oauth_\d+\.json$", re.IGNORECASE
)
_OAUTH_FILENAME_RE = re.compile(r"([a-z_]+)_oauth_\d+\.json$", re.IGNORECASE)


_CredentialAvailabilityState = namedtuple(
    "_CredentialAvailabilityState", "normalized_model key_cooldown_until model_cooldown_until is_on_cooldown soonest_cooldown_until key_data_exists"
)


class UsageManagerQueryMixin:
    _REQUEST_COUNT_PROVIDERS = {"antigravity", "gemini_cli", "chutes", "nanogpt", "zai"}


    def _get_provider_from_credential(self, credential: str) -> Optional[str]:
        """
        Extract provider name from credential path or identifier.

        Supports multiple credential formats:
        - OAuth: "oauth_creds/antigravity_oauth_15.json" -> "antigravity"
        - OAuth: "C:\\...\\oauth_creds\\gemini_cli_oauth_1.json" -> "gemini_cli"
        - OAuth filename only: "antigravity_oauth_1.json" -> "antigravity"
        - API key style: extracted from model names in usage data (e.g., "firmware/model" -> "firmware")

        Args:
            credential: The credential identifier (path or key)

        Returns:
            Provider name string or None if cannot be determined
        """
        # Lookup from credential-to-provider mapping (built at init from all_credentials)
        if self.credential_to_provider and credential in self.credential_to_provider:
            return self.credential_to_provider[credential]

        cached_provider = self._provider_resolution_cache.get(credential)
        if cached_provider is not None:
            self._provider_resolution_cache.move_to_end(credential)
            return cached_provider

        # Pattern: env:// URI format (e.g., "env://antigravity/1" -> "antigravity")
        if credential.startswith("env://"):
            provider = credential[6:].partition("/")[0]
            if provider:
                provider = provider.lower()
                self._cache_provider_resolution(credential, provider)
                return provider
            # Malformed env:// URI (empty provider name)
            lib_logger.warning(
                "Malformed env:// credential URI: %s", mask_credential(credential)
            )
            return None

        # Normalize path separators only when backslashes are present
        normalized = credential.replace("\\", "/") if "\\" in credential else credential

        # Combined pattern: /provider_oauth_N.json$ or oauth_creds/provider_oauth_N.json$
        match = _OAUTH_PROVIDER_RE.search(normalized)
        if match:
            provider = match.group(1).lower()
            self._cache_provider_resolution(credential, provider)
            return provider

        # Fallback: oauth_creds/provider_ without filename suffix
        if "oauth_creds/" in normalized:
            idx = normalized.index("oauth_creds/") + len("oauth_creds/")
            rest = normalized[idx:]
            underscore = rest.rfind("_")
            if underscore > 0:
                provider = rest[:underscore].lower()
                self._cache_provider_resolution(credential, provider)
                return provider

        # Pattern: filename only {provider}_oauth_{number}.json (no path)
        match = _OAUTH_FILENAME_RE.match(normalized)
        if match:
            provider = match.group(1).lower()
            self._cache_provider_resolution(credential, provider)
            return provider

        # Pattern: API key prefixes for specific providers
        # These are raw API keys with recognizable prefixes
        api_key_prefixes = {
            "sk-nano-": "nanogpt",
            "sk-or-": "openrouter",
            "sk-ant-": "anthropic",
        }
        for prefix, provider in api_key_prefixes.items():
            if credential.startswith(prefix):
                self._cache_provider_resolution(credential, provider)
                return provider

        # Fallback: For raw API keys, extract provider from model names in usage data
        # This handles providers like firmware, chutes, nanogpt that use credential-level quota
        if self._usage_data and credential in self._usage_data:
            cred_data = self._usage_data[credential]

            # Check "models" section first (for per_model mode and quota tracking)
            models_data = cred_data.get("models", {})
            if models_data:
                # Get first model name and extract provider prefix
                first_model = next(iter(models_data.keys()), None)
                if first_model and "/" in first_model:
                    provider = first_model.split("/")[0].lower()
                    self._cache_provider_resolution(credential, provider)
                    return provider

            # Fallback to "daily" section (legacy structure)
            daily_data = cred_data.get("daily", {})
            daily_models = daily_data.get("models", {})
            if daily_models:
                # Get first model name and extract provider prefix
                first_model = next(iter(daily_models.keys()), None)
                if first_model and "/" in first_model:
                    provider = first_model.split("/")[0].lower()
                    self._cache_provider_resolution(credential, provider)
                    return provider

        return None

    def _cache_provider_resolution(self, credential: str, provider: str) -> None:
        self._provider_resolution_cache[credential] = provider
        self._provider_resolution_cache.move_to_end(credential)
        while len(self._provider_resolution_cache) > MAX_CACHE_ENTRIES:
            self._provider_resolution_cache.popitem(last=False)

    def _clear_provider_resolution_cache(self) -> None:
        self._provider_resolution_cache.clear()

    def _get_provider_instance(self, provider: str) -> Optional[Any]:
        """
        Get or create a provider plugin instance.

        Args:
            provider: The provider name

        Returns:
            Provider plugin instance or None
        """
        plugin_entry = self.provider_plugins.get(provider)
        if plugin_entry is not None:
            cached_instance = self._provider_instances.get(provider)
            if isinstance(plugin_entry, type):
                if isinstance(cached_instance, plugin_entry):
                    return cached_instance
                instance = plugin_entry()
            else:
                instance = plugin_entry
            self._provider_instances.register(provider, instance)
            return instance

        return get_or_create_provider_instance(
            provider, self.provider_plugins, self._provider_instances
        )

    def _get_provider_capability(
        self, provider: Optional[str], method_name: str
    ) -> Optional[Any]:
        if not provider:
            return None

        plugin_instance = self._get_provider_instance(provider)
        if not plugin_instance:
            return None

        provider_cache = self._provider_capability_cache.setdefault(provider, {})
        if method_name not in provider_cache:
            provider_cache[method_name] = hasattr(plugin_instance, method_name)

        if provider_cache[method_name]:
            return getattr(plugin_instance, method_name)
        return None

    def _get_usage_reset_config(self, credential: str) -> Optional[Dict[str, Any]]:
        """
        Get the usage reset configuration for a credential from its provider plugin.

        Args:
            credential: The credential identifier

        Returns:
            Configuration dict with window_seconds, field_name, etc.
            or None to use default daily reset.
        """
        provider = self._get_provider_from_credential(credential)
        get_usage_reset_config = self._get_provider_capability(
            provider, "get_usage_reset_config"
        )
        if get_usage_reset_config:
            return get_usage_reset_config(credential)

        return None

    def _get_reset_mode(self, credential: str) -> str:
        """
        Get the reset mode for a credential: 'credential' or 'per_model'.

        Args:
            credential: The credential identifier

        Returns:
            "per_model" or "credential" (default)
        """
        config = self._get_usage_reset_config(credential)
        return config.get("mode", "credential") if config else "credential"

    def _get_model_quota_group(self, credential: str, model: str) -> Optional[str]:
        """
        Get the quota group for a model, if the provider defines one.

        Uses a lazy cache since quota groups are stable config that rarely
        changes. Invalidates when custom_caps are updated (call _invalidate_quota_caches
        if needed).

        Args:
            credential: The credential identifier
            model: Model name (with or without provider prefix)

        Returns:
            Group name (e.g., "claude") or None if not grouped
        """
        # Check cache
        key_cache = self._quota_group_cache.get(credential)
        if key_cache is not None and model in key_cache:
            # Move to end for LRU
            key_cache.move_to_end(model)
            return key_cache[model]

        provider = self._get_provider_from_credential(credential)
        get_model_quota_group = self._get_provider_capability(
            provider, "get_model_quota_group"
        )

        result = None
        if get_model_quota_group:
            result = get_model_quota_group(model)

        # Populate cache with eviction check
        if credential not in self._quota_group_cache:
            self._quota_group_cache[credential] = OrderedDict()
        self._quota_group_cache[credential][model] = result
        self._cleanup_caches()
        return result

    def _get_grouped_models(self, credential: str, group: str) -> List[str]:
        """
        Get all model_names in a quota group (with provider prefix), normalized.

        Returns only public-facing model names, deduplicated. Internal variants
        (e.g., claude-sonnet-4-5-thinking) are normalized to their public name
        (e.g., claude-sonnet-4.5).

        Args:
            credential: The credential identifier
            group: Group name (e.g., "claude")

        Returns:
            List of normalized, deduplicated model names with provider prefix
            (e.g., ["antigravity/claude-sonnet-4.5", "antigravity/claude-opus-4.5"])
        """
        # Check cache
        group_key_cache = self._grouped_models_cache.get(credential)
        if group_key_cache is not None and group in group_key_cache:
            # Move to end for LRU
            group_key_cache.move_to_end(group)
            models_list, _ = group_key_cache[group]
            return models_list

        provider = self._get_provider_from_credential(credential)
        get_models_in_quota_group = self._get_provider_capability(
            provider, "get_models_in_quota_group"
        )

        if get_models_in_quota_group:
            models = get_models_in_quota_group(group)

            # Normalize and deduplicate
            normalize_model_for_tracking = self._get_provider_capability(
                provider, "normalize_model_for_tracking"
            )
            if normalize_model_for_tracking:
                seen = set()
                normalized = []
                for m in models:
                    prefixed = f"{provider}/{m}"
                    norm = normalize_model_for_tracking(prefixed)
                    if norm not in seen:
                        seen.add(norm)
                        normalized.append(norm)
                if credential not in self._grouped_models_cache:
                    self._grouped_models_cache[credential] = OrderedDict()
                self._grouped_models_cache[credential][group] = (normalized, -1)
                self._cleanup_caches()
                return normalized

            # Fallback: just add provider prefix
            result = [f"{provider}/{m}" for m in models]
            if credential not in self._grouped_models_cache:
                self._grouped_models_cache[credential] = OrderedDict()
            self._grouped_models_cache[credential][group] = (result, -1)
            self._cleanup_caches()
            return result

        return []

    def _cleanup_caches(self) -> None:
        """
        Evict oldest nested cache entries in bulk if total exceeds MAX_CACHE_ENTRIES.

        Called after each cache population to prevent unbounded growth.
        Uses bulk eviction: entire credential sub-caches are removed.
        """
        if getattr(self, "_is_cleaning_caches", False):
            return

        self._is_cleaning_caches = True
        try:
            total_entries = sum(len(v) for v in self._quota_group_cache.values())
            total_entries += sum(len(v) for v in self._grouped_models_cache.values())

            if total_entries <= MAX_CACHE_ENTRIES:
                return

            margin = int(MAX_CACHE_ENTRIES * 0.1)
            excess = total_entries - (MAX_CACHE_ENTRIES - margin)

            for cache in (self._quota_group_cache, self._grouped_models_cache):
                for credential in list(cache.keys()):
                    if excess <= 0:
                        break
                    excess -= len(cache[credential])
                    del cache[credential]

                if excess <= 0:
                    break
        finally:
            self._is_cleaning_caches = False

    def _get_model_usage_weight(self, credential: str, model: str) -> int:
        """
        Get the usage weight for a model when calculating grouped usage.

        Args:
            credential: The credential identifier
            model: Model name (with or without provider prefix)

        Returns:
            Weight multiplier (default 1 if not configured)
        """
        provider = self._get_provider_from_credential(credential)
        get_model_usage_weight = self._get_provider_capability(
            provider, "get_model_usage_weight"
        )
        if get_model_usage_weight:
            return get_model_usage_weight(model)

        return 1

    def _normalize_model_for_tracking(self, credential: str, model: str) -> str:
        """
        Normalize model name using provider's mapping.

        Converts internal model names (e.g., claude-sonnet-4-5-thinking) to
        public-facing names (e.g., claude-sonnet-4.5) for consistent storage.

        Args:
            credential: The credential identifier
            model: Model name (with or without provider prefix)

        Returns:
            Normalized model name (provider prefix preserved if present)
        """
        provider = self._get_provider_from_credential(credential)
        normalize_model_for_tracking = self._get_provider_capability(
            provider, "normalize_model_for_tracking"
        )
        if normalize_model_for_tracking:
            return normalize_model_for_tracking(model)

        return model

    def _get_credential_availability_state(
        self,
        key: str,
        model: str,
        now: float,
        key_data: Optional[Dict[str, Any]] = None,
        provider: Optional[str] = None,
        normalized_model: Optional[str] = None,
    ) -> _CredentialAvailabilityState:
        if key_data is None:
            key_data = getattr(self, "_usage_data", {}).get(key)
        if normalized_model is None:
            normalized_model = self._normalize_model_for_tracking(key, model)

        if key_data is None:
            return _CredentialAvailabilityState(normalized_model, 0, 0, False, None, False)

        key_cooldown_until = key_data.get("key_cooldown_until") or 0
        model_cooldowns = key_data.get("model_cooldowns") or {}
        direct_model_cooldown_until = model_cooldowns.get(normalized_model) or 0
        quota_group_cooldown_until = 0
        if model_cooldowns:
            quota_group_cooldown_until = self._check_quota_group_cooldown(
                key, model_cooldowns, normalized_model, provider=provider
            )
        model_cooldown_until = max(
            direct_model_cooldown_until, quota_group_cooldown_until
        )

        soonest_cooldown_until = None
        for cooldown in (
            key_cooldown_until,
            direct_model_cooldown_until,
            quota_group_cooldown_until,
        ):
            if cooldown > now and (
                soonest_cooldown_until is None or cooldown < soonest_cooldown_until
            ):
                soonest_cooldown_until = cooldown

        return _CredentialAvailabilityState(
            normalized_model,
            key_cooldown_until,
            model_cooldown_until,
            soonest_cooldown_until is not None,
            soonest_cooldown_until,
            True,
        )

    def _check_quota_group_cooldown(
        self,
        credential: str,
        model_cooldowns: Dict[str, float],
        normalized_model: str,
        provider: Optional[str] = None,
    ) -> float:
        """Check if a virtual _quota model cooldown applies to this model.

        Background quota refresh sets cooldowns on the virtual
        {provider}/_quota model.  Before real models are discovered
        and added to the quota group, the real model won't have its
        own cooldown entry.  This helper returns the _quota cooldown
        if the requested model belongs to a quota group and has no
        direct cooldown, so callers can treat it as an effective
        model-level cooldown.

        Returns the _quota cooldown timestamp if applicable, else 0.
        """
        if normalized_model in model_cooldowns:
            return 0  # model has its own cooldown entry — use that
        if provider is None:
            provider = self._get_provider_from_credential(credential)
        if not provider:
            return 0
        get_model_quota_group = self._get_provider_capability(
            provider, "get_model_quota_group"
        )
        if not get_model_quota_group:
            return 0
        group = get_model_quota_group(normalized_model)
        if not group:
            return 0
        virtual_name = f"{provider}/_quota"
        return model_cooldowns.get(virtual_name) or 0

    def _get_baseline_remaining(self, key: str, model: str) -> Optional[float]:
        """
        Get baseline_remaining_fraction for a credential's model, checking quota groups.

        Background refresh stores baseline on the virtual _quota model and syncs
        it across the quota group. This method finds the baseline by checking
        the requested model first, then falling back to any grouped model.

        Returns None if no baseline data exists (provider not yet refreshed).
        """
        key_data = self._usage_data.get(key)
        if not key_data:
            return None
        models_data = key_data.get("models", {})
        # Direct lookup on the requested model
        model_stats = models_data.get(model)
        if model_stats:
            baseline = model_stats.get("baseline_remaining_fraction")
            if baseline is not None:
                return baseline
        # Check quota group: look for _quota virtual model or any grouped model
        group = self._get_model_quota_group(key, model)
        if group:
            for model_name, stats in models_data.items():
                if model_name == model:
                    continue
                other_group = self._get_model_quota_group(key, model_name)
                if other_group == group:
                    baseline = stats.get("baseline_remaining_fraction")
                    if baseline is not None:
                        return baseline
        return None

    def _get_grouped_usage_count(self, key: str, model: str) -> int:
        """
        Get usage count for credential selection, considering quota groups.

        For providers in _REQUEST_COUNT_PROVIDERS (e.g., antigravity), uses
        request_count instead of success_count since failed requests also
        consume quota.

        If the model belongs to a quota group, the request_count is already
        synced across all models in the group (by record_success/record_failure),
        so we just read from the requested model directly.

        Args:
            key: Credential identifier
            model: Model name (with provider prefix, e.g., "antigravity/claude-sonnet-4-5")

        Returns:
            Usage count for the model (synced across group if applicable)
        """
        # Determine usage field based on provider
        # Some providers (antigravity) count failed requests against quota
        provider = self._get_provider_from_credential(key)
        usage_field = (
            "request_count"
            if provider in self._REQUEST_COUNT_PROVIDERS
            else "success_count"
        )

        # For providers with synced quota groups (antigravity), request_count
        # is already synced across all models in the group, so just read directly.
        # For other providers, we still need to sum success_count across group.
        if provider in self._REQUEST_COUNT_PROVIDERS:
            # request_count is synced - just read the model's value
            return self._get_usage_count(key, model, usage_field)

        # For non-synced providers, check if model is in a quota group and sum
        group = self._get_model_quota_group(key, model)

        if group:
            # Get all models in the group
            grouped_models = self._get_grouped_models(key, group)

            # Sum weighted usage across all models in the group
            total_weighted_usage = 0
            for grouped_model in grouped_models:
                usage = self._get_usage_count(key, grouped_model, usage_field)
                weight = self._get_model_usage_weight(key, grouped_model)
                total_weighted_usage += usage * weight
            return total_weighted_usage

        # Not grouped - return individual model usage (no weight applied)
        return self._get_usage_count(key, model, usage_field)

    def _get_quota_display(self, key: str, model: str) -> str:
        """
        Get a formatted quota display string for logging.

        For antigravity (providers in _REQUEST_COUNT_PROVIDERS), returns:
            "quota: 170/250 [32%]" format

        For other providers, returns:
            "usage: 170" format (no max available)

        Args:
            key: Credential identifier
            model: Model name (with provider prefix)

        Returns:
            Formatted string for logging
        """
        provider = self._get_provider_from_credential(key)

        if provider not in self._REQUEST_COUNT_PROVIDERS:
            # Non-antigravity: just show usage count
            usage = self._get_usage_count(key, model, "success_count")
            return f"usage: {usage}"

        # Antigravity: show quota display with remaining percentage
        if self._usage_data is None:
            return "quota: 0/? [100%]"

        # Normalize model name for consistent lookup (data is stored under normalized names)
        model = self._normalize_model_for_tracking(key, model)

        usage_data = self._usage_data
        key_data = usage_data.get(key)
        if not key_data:
            return "quota: 0"
        models = key_data.get("models")
        if not models:
            return "quota: 0"
        model_data = models.get(model)
        if not model_data:
            return "quota: 0"

        request_count = model_data.get("request_count", 0)
        max_requests = model_data.get("quota_max_requests")

        if max_requests:
            remaining = max_requests - request_count
            remaining_pct = (
                int((remaining / max_requests) * 100) if max_requests > 0 else 0
            )
            return f"quota: {request_count}/{max_requests} [{remaining_pct}%]"
        else:
            return f"quota: {request_count}"

    def _get_usage_field_name(self, credential: str) -> str:
        """
        Get the usage tracking field name for a credential.

        Returns the provider-specific field name if configured,
        otherwise falls back to "daily".

        Args:
            credential: The credential identifier

        Returns:
            Field name string (e.g., "5h_window", "weekly", "daily")
        """
        config = self._get_usage_reset_config(credential)
        if config and "field_name" in config:
            return config["field_name"]

        # Check provider default
        provider = self._get_provider_from_credential(credential)
        get_default_usage_field_name = self._get_provider_capability(
            provider, "get_default_usage_field_name"
        )
        if get_default_usage_field_name:
            return get_default_usage_field_name()

        return "daily"

    def _get_usage_count(
        self, key: str, model: str, field: str = "success_count"
    ) -> int:
        """
        Get the current usage count for a model from the appropriate usage structure.

        Supports both:
        - New per-model structure: {"models": {"model_name": {"success_count": N, ...}}}
        - Legacy structure: {"daily": {"models": {"model_name": {"success_count": N, ...}}}}

        Args:
            key: Credential identifier
            model: Model name
            field: The field to read for usage count (default: "success_count").
                   Use "request_count" for providers where failed requests also
                   consume quota (e.g., antigravity).

        Returns:
            Usage count for the model in the current window/period
        """
        if self._usage_data is None:
            return 0

        # Normalize model name for consistent lookup (data is stored under normalized names)
        model = self._normalize_model_for_tracking(key, model)

        try:
            key_data = self._usage_data[key]
        except KeyError:
            return 0

        reset_mode = self._get_reset_mode(key)

        if reset_mode == "per_model":
            # New per-model structure: key_data["models"][model][field]
            try:
                return key_data["models"][model][field]
            except KeyError:
                return 0
        else:
            # Legacy structure: key_data["daily"]["models"][model][field]
            try:
                return key_data["daily"]["models"][model][field]
            except KeyError:
                return 0

    async def get_available_credentials_for_model(
        self, credentials: List[str], model: str
    ) -> List[str]:
        """
        Get credentials that are not on cooldown for a specific model.

        Filters out credentials where:
        - key_cooldown_until > now (key-level cooldown)
        - model_cooldowns[model] > now (model-specific cooldown, includes quota exhausted)

        Args:
            credentials: List of credential identifiers to check
            model: Model name to check cooldowns for

        Returns:
            List of credentials that are available (not on cooldown) for this model
        """
        await self._lazy_init()
        now = time.time()
        available = []

        usage_data = self._usage_data
        # Pre-compute provider and normalized model outside the lock
        precomputed = []
        for key in credentials:
            provider = self._get_provider_from_credential(key)
            normalized_model = self._normalize_model_for_tracking(key, model)
            precomputed.append((key, provider, normalized_model))

        async with self._data_lock.read():
            for key, provider, normalized_model in precomputed:
                state = self._get_credential_availability_state(
                    key, model, now, usage_data.get(key),
                    provider=provider, normalized_model=normalized_model
                )
                if not state.key_data_exists or state.is_on_cooldown:
                    continue

                # Skip if quota confirmed exhausted via background refresh
                baseline = self._get_baseline_remaining(key, state.normalized_model)
                if baseline is not None and baseline <= 0:
                    continue

                available.append(key)

        return available

    async def get_credential_availability_stats(
        self,
        credentials: List[str],
        model: str,
        credential_priorities: Optional[Dict[str, int]] = None,
    ) -> Dict[str, int]:
        """
        Get credential availability statistics including cooldown and fair cycle exclusions.

        This is used for logging to show why credentials are excluded.

        Args:
            credentials: List of credential identifiers to check
            model: Model name to check
            credential_priorities: Optional dict mapping credentials to priorities

        Returns:
            Dict with:
                "total": Total credentials
                "on_cooldown": Count on cooldown
                "fair_cycle_excluded": Count excluded by fair cycle
                "available": Count available for selection
        """
        await self._lazy_init()
        now = time.time()

        total = len(credentials)
        on_cooldown = 0
        not_on_cooldown = []

        # First pass: check cooldowns
        usage_data = self._usage_data
        # Pre-compute provider and normalized model outside the lock
        precomputed = []
        for key in credentials:
            provider = self._get_provider_from_credential(key)
            normalized_model = self._normalize_model_for_tracking(key, model)
            precomputed.append((key, provider, normalized_model))

        async with self._data_lock.read():
            for key, provider, normalized_model in precomputed:
                state = self._get_credential_availability_state(
                    key, model, now, usage_data.get(key),
                    provider=provider, normalized_model=normalized_model
                )
                if not state.key_data_exists:
                    continue

                if state.is_on_cooldown:
                    on_cooldown += 1
                else:
                    not_on_cooldown.append(key)

        # Second pass: check fair cycle exclusions (only for non-cooldown credentials)
        fair_cycle_excluded = 0
        if not_on_cooldown:
            provider = self._get_provider_from_credential(not_on_cooldown[0])
            if provider:
                rotation_mode = self._get_rotation_mode(provider)
                if self._is_fair_cycle_enabled(provider, rotation_mode):
                    # Check each credential against its own tier's exhausted set
                    for key in not_on_cooldown:
                        key_priority = (
                            credential_priorities.get(key, 999)
                            if credential_priorities
                            else 999
                        )
                        tier_key = self._get_tier_key(provider, key_priority)
                        tracking_key = self._get_tracking_key(key, model, provider)

                        if self._is_credential_exhausted_in_cycle(
                            key, provider, tier_key, tracking_key
                        ):
                            fair_cycle_excluded += 1

        available = total - on_cooldown - fair_cycle_excluded

        return {
            "total": total,
            "on_cooldown": on_cooldown,
            "fair_cycle_excluded": fair_cycle_excluded,
            "available": available,
        }

    async def get_soonest_cooldown_end(
        self,
        credentials: List[str],
        model: str,
    ) -> Optional[float]:
        """
        Find the soonest time when any credential will come off cooldown.

        This is used for smart waiting logic - if no credentials are available,
        we can determine whether to wait (if soonest cooldown < deadline) or
        fail fast (if soonest cooldown > deadline).

        Args:
            credentials: List of credential identifiers to check
            model: Model name to check cooldowns for

        Returns:
            Timestamp of soonest cooldown end, or None if no credentials are on cooldown
        """
        await self._lazy_init()
        now = time.time()
        soonest_end = None

        # Pre-compute provider and normalized model outside the lock
        precomputed = []
        for key in credentials:
            provider = self._get_provider_from_credential(key)
            normalized_model = self._normalize_model_for_tracking(key, model)
            precomputed.append((key, provider, normalized_model))

        async with self._data_lock.read():
            for key, provider, normalized_model in precomputed:
                state = self._get_credential_availability_state(
                    key, model, now,
                    provider=provider, normalized_model=normalized_model
                )
                if state.soonest_cooldown_until is not None:
                    if soonest_end is None or state.soonest_cooldown_until < soonest_end:
                        soonest_end = state.soonest_cooldown_until

        return soonest_end

    # =========================================================================
    # TIMESTAMP FORMATTING HELPERS
    # =========================================================================





# --- Statistics Mixin (extracted to statistics.py) ---

from .statistics import UsageManagerStatisticsMixin, get_stats_for_endpoint
