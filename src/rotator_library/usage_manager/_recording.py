# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
import time
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional
import litellm
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
                    key,
                    {
                        "models": {},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
                )

                # Ensure models dict exists
                if "models" not in key_data:
                    key_data["models"] = {}

                # Get or create per-model data with window tracking
                model_data = key_data["models"].setdefault(
                    model,
                    {
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
                    },
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
                # Uses cached grouped models list (quota groups are stable config).
                # Only writes siblings whose request_count actually differs to avoid
                # unnecessary setdefault + dict mutations on hot path.
                new_request_count = model_data["request_count"]
                group = self._get_model_quota_group(key, model)
                if group:
                    grouped_models = self._get_grouped_models(key, group)
                    window_start = model_data.get("window_start_ts")
                    max_req = model_data.get("quota_max_requests")
                    for grouped_model in grouped_models:
                        if grouped_model == model:
                            continue
                        other_model_data = key_data["models"].setdefault(
                            grouped_model,
                            {
                                "window_start_ts": None,
                                "quota_reset_ts": None,
                                "success_count": 0,
                                "failure_count": 0,
                                "request_count": 0,
                                "prompt_tokens": 0,
                                "prompt_tokens_cached": 0,
                                "completion_tokens": 0,
                                "thinking_tokens": 0,
                                "approx_cost": 0.0,
                            },
                        )
                        # Skip write if value already matches
                        if other_model_data.get("request_count") == new_request_count:
                            continue
                        other_model_data["request_count"] = new_request_count
                        # Sync window timing (shared quota pool = shared window)
                        if window_start:
                            other_model_data["window_start_ts"] = window_start
                        # Also sync quota_max_requests if set
                        if max_req:
                            other_model_data["quota_max_requests"] = max_req
                            other_model_data["quota_display"] = (
                                f"{new_request_count}/{max_req}"
                            )

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
                    key,
                    {
                        "daily": {"date": today_utc_str, "models": {}},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
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
                                if input_cost:
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
                    key,
                    {
                        "models": {},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
                )
            else:
                key_data = self._usage_data.setdefault(
                    key,
                    {
                        "daily": {"date": today_utc_str, "models": {}},
                        "global": {"models": {}},
                        "model_cooldowns": {},
                        "failures": {},
                    },
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
            existing_cooldown_before = model_cooldowns.get(model)
            was_already_on_cooldown = (
                existing_cooldown_before is not None
                and existing_cooldown_before > now_ts
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
                        model,
                        {
                            "window_start_ts": None,
                            "quota_reset_ts": None,
                            "success_count": 0,
                            "failure_count": 0,
                            "request_count": 0,
                            "prompt_tokens": 0,
                            "prompt_tokens_cached": 0,
                            "completion_tokens": 0,
                            "thinking_tokens": 0,
                            "approx_cost": 0.0,
                        },
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
                                grouped_model,
                                {
                                    "window_start_ts": None,
                                    "quota_reset_ts": None,
                                    "success_count": 0,
                                    "failure_count": 0,
                                    "request_count": 0,
                                    "prompt_tokens": 0,
                                    "prompt_tokens_cached": 0,
                                    "completion_tokens": 0,
                                    "thinking_tokens": 0,
                                    "approx_cost": 0.0,
                                },
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
                    provider = self._get_provider_from_credential(key)
                    if provider:
                        threshold = self._get_exhaustion_cooldown_threshold(provider)
                        if effective_cooldown > threshold:
                            rotation_mode = self._get_rotation_mode(provider)
                            if self._is_fair_cycle_enabled(provider, rotation_mode):
                                priority = self._get_credential_priority(key, provider)
                                tier_key = self._get_tier_key(provider, priority)
                                tracking_key = self._get_tracking_key(
                                    key, model, provider
                                )
                                self._mark_credential_exhausted(
                                    key, provider, tier_key, tracking_key
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
                    model,
                    {
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
                    },
                )
                # Only increment if not already incremented in quota_exceeded branch
                if classified_error.error_type != "quota_exceeded":
                    model_data["failure_count"] = model_data.get("failure_count", 0) + 1
                    model_data["request_count"] = model_data.get("request_count", 0) + 1

                    # Sync request_count across quota group
                    new_request_count = model_data["request_count"]
                    group = self._get_model_quota_group(key, model)
                    if group:
                        grouped_models = self._get_grouped_models(key, group)
                        for grouped_model in grouped_models:
                            if grouped_model != model:
                                other_model_data = models_data.setdefault(
                                    grouped_model,
                                    {
                                        "window_start_ts": None,
                                        "quota_reset_ts": None,
                                        "success_count": 0,
                                        "failure_count": 0,
                                        "request_count": 0,
                                        "prompt_tokens": 0,
                                        "prompt_tokens_cached": 0,
                                        "completion_tokens": 0,
                                        "thinking_tokens": 0,
                                        "approx_cost": 0.0,
                                    },
                                )
                                other_model_data["request_count"] = new_request_count
                                # Also sync quota_max_requests if set
                                max_req = model_data.get("quota_max_requests")
                                if max_req:
                                    other_model_data["quota_max_requests"] = max_req
                                    other_model_data["quota_display"] = (
                                        f"{new_request_count}/{max_req}"
                                    )

            key_data["last_failure"] = {
                "timestamp": now_ts,
                "model": model,
                "error": str(classified_error.original_exception),
            }

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
                credential,
                {
                    "models": {},
                    "global": {"models": {}},
                    "model_cooldowns": {},
                    "failures": {},
                },
            )

            # Ensure models dict exists
            if "models" not in key_data:
                key_data["models"] = {}

            # Get or create per-model data
            model_data = key_data["models"].setdefault(
                model,
                {
                    "window_start_ts": None,
                    "quota_reset_ts": None,
                    "success_count": 0,
                    "failure_count": 0,
                    "request_count": 0,
                    "prompt_tokens": 0,
                    "prompt_tokens_cached": 0,
                    "completion_tokens": 0,
                    "thinking_tokens": 0,
                    "approx_cost": 0.0,
                    "baseline_remaining_fraction": None,
                    "baseline_fetched_at": None,
                    "requests_at_baseline": None,
                },
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
                    used_requests = int((1.0 - remaining_fraction) * max_requests)
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
                existing_cooldown = model_cooldowns.get(model)
                was_already_on_cooldown = (
                    existing_cooldown is not None and existing_cooldown > now_ts
                )

                # Only update cooldown if not set or differs by more than 5 minutes
                should_update = (
                    existing_cooldown is None
                    or abs(existing_cooldown - reset_timestamp) > 300
                )
                if should_update:
                    model_cooldowns[model] = reset_timestamp
                    hours_until_reset = (reset_timestamp - now_ts) / 3600
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
                    cooldown_duration = reset_timestamp - now_ts
                    provider = self._get_provider_from_credential(credential)
                    if provider:
                        threshold = self._get_exhaustion_cooldown_threshold(provider)
                        if cooldown_duration > threshold:
                            rotation_mode = self._get_rotation_mode(provider)
                            if self._is_fair_cycle_enabled(provider, rotation_mode):
                                priority = self._get_credential_priority(
                                    credential, provider
                                )
                                tier_key = self._get_tier_key(provider, priority)
                                tracking_key = self._get_tracking_key(
                                    credential, model, provider
                                )
                                self._mark_credential_exhausted(
                                    credential, provider, tier_key, tracking_key
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
                grouped_models = self._get_grouped_models(credential, group)
                for grouped_model in grouped_models:
                    if grouped_model != model:
                        other_model_data = key_data["models"].setdefault(
                            grouped_model,
                            {
                                "window_start_ts": None,
                                "quota_reset_ts": None,
                                "success_count": 0,
                                "failure_count": 0,
                                "request_count": 0,
                                "prompt_tokens": 0,
                                "prompt_tokens_cached": 0,
                                "completion_tokens": 0,
                                "thinking_tokens": 0,
                                "approx_cost": 0.0,
                            },
                        )
                        # Sync request tracking (use synced_count to prevent reset bug)
                        other_model_data["request_count"] = synced_count
                        if max_requests is not None:
                            other_model_data["quota_max_requests"] = max_requests
                            other_model_data["quota_display"] = (
                                f"{synced_count}/{max_requests}"
                            )
                        # Sync baseline fields
                        other_model_data["baseline_remaining_fraction"] = (
                            remaining_fraction
                        )
                        other_model_data["baseline_fetched_at"] = now_ts
                        other_model_data["requests_at_baseline"] = synced_count
                        # Sync reset timestamp if valid
                        if valid_reset_ts:
                            other_model_data["quota_reset_ts"] = reset_timestamp
                        # Sync window start time
                        window_start = model_data.get("window_start_ts")
                        if window_start:
                            other_model_data["window_start_ts"] = window_start
                        # Sync cooldown if exhausted (with В±5 min check)
                        if is_exhausted and valid_reset_ts:
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

