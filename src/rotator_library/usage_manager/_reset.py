# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
import time
from datetime import datetime, timezone
from typing import Dict, Any
from ..error_types import mask_credential


class UsageManagerResetMixin:
    async def _reset_daily_stats_if_needed(self):
        """
        Checks if usage stats need to be reset for any key.

        Supports three reset modes:
        1. per_model: Each model has its own window, resets based on quota_reset_ts or fallback window
        2. credential: One window per credential (legacy with custom window duration)
        3. daily: Legacy daily reset at daily_reset_time_utc
        """
        if self._usage_data is None:
            return

        now_utc = datetime.now(timezone.utc)
        now_ts = time.time()
        today_str = now_utc.date().isoformat()
        needs_saving = False

        async with self._data_lock.write():
            for key, data in list(self._usage_data.items()):
                reset_config = self._get_usage_reset_config(key)

                if reset_config:
                    reset_mode = reset_config.get("mode", "credential")

                    if reset_mode == "per_model":
                        # Per-model window reset
                        needs_saving |= await self._check_per_model_resets(
                            key, data, reset_config, now_ts
                        )
                    else:
                        # Credential-level window reset (legacy)
                        needs_saving |= await self._check_window_reset(
                            key, data, reset_config, now_ts
                        )
                elif self.daily_reset_time_utc:
                    # Legacy daily reset
                    needs_saving |= await self._check_daily_reset(
                        key, data, now_utc, today_str, now_ts
                    )

        if needs_saving:
            await self._save_usage()

    async def _check_per_model_resets(
        self,
        key: str,
        data: Dict[str, Any],
        reset_config: Dict[str, Any],
        now_ts: float,
    ) -> bool:
        """
        Check and perform per-model resets for a credential.

        Each model resets independently based on:
        1. quota_reset_ts (authoritative, from quota exhausted error) if set
        2. window_start_ts + window_seconds (fallback) otherwise

        Grouped models reset together - all models in a group must be ready.

        Args:
            key: Credential identifier
            data: Usage data for this credential
            reset_config: Provider's reset configuration
            now_ts: Current timestamp

        Returns:
            True if data was modified and needs saving
        """
        window_seconds = reset_config.get("window_seconds", 86400)
        models_data = data.get("models", {})

        if not models_data:
            return False

        modified = False
        processed_groups = set()

        for model, model_data in list(models_data.items()):
            # Check if this model is in a quota group
            group = self._get_model_quota_group(key, model)

            if group:
                if group in processed_groups:
                    continue  # Already handled this group

                # Check if entire group should reset
                if self._should_group_reset(
                    key, group, models_data, window_seconds, now_ts
                ):
                    # Archive and reset all models in group
                    grouped_models = self._get_grouped_models(key, group)
                    archived_count = 0

                    for grouped_model in grouped_models:
                        if grouped_model in models_data:
                            gm_data = models_data[grouped_model]
                            self._archive_model_to_global(data, grouped_model, gm_data)
                            self._reset_model_data(gm_data)
                            archived_count += 1

                    if archived_count > 0:
                        lib_logger.info(
                            f"Reset model group '{group}' ({archived_count} models) for {mask_credential(key)}"
                        )
                        modified = True

                processed_groups.add(group)

            else:
                # Ungrouped model - check individually
                if self._should_model_reset(model_data, window_seconds, now_ts):
                    self._archive_model_to_global(data, model, model_data)
                    self._reset_model_data(model_data)
                    lib_logger.info(f"Reset model {model} for {mask_credential(key)}")
                    modified = True

        # Preserve unexpired cooldowns
        if modified:
            self._preserve_unexpired_cooldowns(key, data, now_ts)
            if "failures" in data:
                data["failures"] = {}

        return modified

    def _should_model_reset(
        self, model_data: Dict[str, Any], window_seconds: int, now_ts: float
    ) -> bool:
        """
        Check if a single model should reset.

        Returns True if:
        - quota_reset_ts is set AND now >= quota_reset_ts, OR
        - quota_reset_ts is NOT set AND now >= window_start_ts + window_seconds
        """
        quota_reset = model_data.get("quota_reset_ts")
        window_start = model_data.get("window_start_ts")

        if quota_reset:
            return now_ts >= quota_reset
        elif window_start:
            return now_ts >= window_start + window_seconds
        return False

    def _should_group_reset(
        self,
        key: str,
        group: str,
        models_data: Dict[str, Dict],
        window_seconds: int,
        now_ts: float,
    ) -> bool:
        """
        Check if all models in a group should reset.

        All models in the group must be ready to reset.
        If any model has an active cooldown/window, the whole group waits.
        """
        grouped_models = self._get_grouped_models(key, group)

        # Track if any model in group has data
        any_has_data = False

        for grouped_model in grouped_models:
            model_data = models_data.get(grouped_model, {})

            if not model_data or (
                model_data.get("window_start_ts") is None
                and model_data.get("success_count", 0) == 0
            ):
                continue  # No stats for this model yet

            any_has_data = True

            if not self._should_model_reset(model_data, window_seconds, now_ts):
                return False  # At least one model not ready

        return any_has_data

    def _archive_model_to_global(
        self, data: Dict[str, Any], model: str, model_data: Dict[str, Any]
    ) -> None:
        """Archive a single model's stats to global."""
        global_data = data.setdefault("global", {"models": {}})
        global_model = global_data["models"].setdefault(
            model,
            {
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

        global_model["success_count"] += model_data.get("success_count", 0)
        global_model["failure_count"] += model_data.get("failure_count", 0)
        global_model["request_count"] += model_data.get("request_count", 0)
        global_model["prompt_tokens"] += model_data.get("prompt_tokens", 0)
        global_model["prompt_tokens_cached"] = global_model.get(
            "prompt_tokens_cached", 0
        ) + model_data.get("prompt_tokens_cached", 0)
        global_model["prompt_tokens_cache_creation"] = global_model.get(
            "prompt_tokens_cache_creation", 0
        ) + model_data.get("prompt_tokens_cache_creation", 0)
        global_model["completion_tokens"] += model_data.get("completion_tokens", 0)
        global_model["approx_cost"] += model_data.get("approx_cost", 0.0)

    def _reset_model_data(self, model_data: Dict[str, Any]) -> None:
        """Reset a model's window and stats."""
        model_data["window_start_ts"] = None
        model_data["quota_reset_ts"] = None
        model_data["success_count"] = 0
        model_data["failure_count"] = 0
        model_data["request_count"] = 0
        model_data["prompt_tokens"] = 0
        model_data["completion_tokens"] = 0
        model_data["approx_cost"] = 0.0
        # Reset quota baseline fields only if they exist (Antigravity-specific)
        # These are added by update_quota_baseline(), only called for Antigravity
        if "baseline_remaining_fraction" in model_data:
            model_data["baseline_remaining_fraction"] = None
            model_data["baseline_fetched_at"] = None
            model_data["requests_at_baseline"] = None
            # Reset quota display but keep max_requests (it doesn't change between periods)
            max_req = model_data.get("quota_max_requests")
            if max_req:
                model_data["quota_display"] = f"0/{max_req}"

    async def _check_window_reset(
        self,
        key: str,
        data: Dict[str, Any],
        reset_config: Dict[str, Any],
        now_ts: float,
    ) -> bool:
        """
        Check and perform rolling window reset for a credential.

        Args:
            key: Credential identifier
            data: Usage data for this credential
            reset_config: Provider's reset configuration
            now_ts: Current timestamp

        Returns:
            True if data was modified and needs saving
        """
        window_seconds = reset_config.get("window_seconds", 86400)  # Default 24h
        field_name = reset_config.get("field_name", "window")
        description = reset_config.get("description", "rolling window")

        # Get current window data
        window_data = data.get(field_name, {})
        window_start = window_data.get("start_ts")

        # No window started yet - nothing to reset
        if window_start is None:
            return False

        # Check if window has expired
        window_end = window_start + window_seconds
        if now_ts < window_end:
            # Window still active
            return False

        # Window expired - perform reset
        hours_elapsed = (now_ts - window_start) / 3600
        lib_logger.info(
            f"Resetting {field_name} for {mask_credential(key)} - "
            f"{description} expired after {hours_elapsed:.1f}h"
        )

        # Archive to global
        self._archive_to_global(data, window_data)

        # Preserve unexpired cooldowns
        self._preserve_unexpired_cooldowns(key, data, now_ts)

        # Reset window stats (but don't start new window until first request)
        data[field_name] = {"start_ts": None, "models": {}}

        # Reset consecutive failures
        if "failures" in data:
            data["failures"] = {}

        return True

    async def _check_daily_reset(
        self,
        key: str,
        data: Dict[str, Any],
        now_utc: datetime,
        today_str: str,
        now_ts: float,
    ) -> bool:
        """
        Check and perform legacy daily reset for a credential.

        Args:
            key: Credential identifier
            data: Usage data for this credential
            now_utc: Current datetime in UTC
            today_str: Today's date as ISO string
            now_ts: Current timestamp

        Returns:
            True if data was modified and needs saving
        """
        last_reset_str = data.get("last_daily_reset", "")

        if last_reset_str == today_str:
            return False

        last_reset_dt = None
        if last_reset_str:
            try:
                last_reset_dt = datetime.fromisoformat(last_reset_str).replace(
                    tzinfo=timezone.utc
                )
            except ValueError as e:
                lib_logger.debug("Could not parse value: %s", e)

        # Determine the reset threshold for today
        reset_threshold_today = datetime.combine(
            now_utc.date(), self.daily_reset_time_utc
        )

        if not (
            last_reset_dt is None or last_reset_dt < reset_threshold_today <= now_utc
        ):
            return False

        lib_logger.debug(f"Performing daily reset for key {mask_credential(key)}")

        # Preserve unexpired cooldowns
        self._preserve_unexpired_cooldowns(key, data, now_ts)

        # Reset consecutive failures
        if "failures" in data:
            data["failures"] = {}

        # Archive daily stats to global
        daily_data = data.get("daily", {})
        if daily_data:
            self._archive_to_global(data, daily_data)

        # Reset daily stats
        data["daily"] = {"date": today_str, "models": {}}
        data["last_daily_reset"] = today_str

        return True

    def _archive_to_global(
        self, data: Dict[str, Any], source_data: Dict[str, Any]
    ) -> None:
        """
        Archive usage stats from a source field (daily/window) to global.

        Args:
            data: The credential's usage data
            source_data: The source field data to archive (has "models" key)
        """
        global_data = data.setdefault("global", {"models": {}})
        for model, stats in source_data.get("models", {}).items():
            global_model_stats = global_data["models"].setdefault(
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
            global_model_stats["success_count"] += stats.get("success_count", 0)
            global_model_stats["prompt_tokens"] += stats.get("prompt_tokens", 0)
            global_model_stats["prompt_tokens_cached"] = global_model_stats.get(
                "prompt_tokens_cached", 0
            ) + stats.get("prompt_tokens_cached", 0)
            global_model_stats["prompt_tokens_cache_creation"] = global_model_stats.get(
                "prompt_tokens_cache_creation", 0
            ) + stats.get("prompt_tokens_cache_creation", 0)
            global_model_stats["completion_tokens"] += stats.get("completion_tokens", 0)
            global_model_stats["approx_cost"] += stats.get("approx_cost", 0.0)

    def _preserve_unexpired_cooldowns(
        self, key: str, data: Dict[str, Any], now_ts: float
    ) -> None:
        """
        Preserve unexpired cooldowns during reset (important for long quota cooldowns).

        Args:
            key: Credential identifier (for logging)
            data: The credential's usage data
            now_ts: Current timestamp
        """
        # Preserve unexpired model cooldowns
        if "model_cooldowns" in data:
            active_cooldowns = {
                model: end_time
                for model, end_time in data["model_cooldowns"].items()
                if end_time > now_ts
            }
            if active_cooldowns:
                max_remaining = max(
                    end_time - now_ts for end_time in active_cooldowns.values()
                )
                hours_remaining = max_remaining / 3600
                lib_logger.info(
                    f"Preserving {len(active_cooldowns)} active cooldown(s) "
                    f"for key {mask_credential(key)} during reset "
                    f"(longest: {hours_remaining:.1f}h remaining)"
                )
            data["model_cooldowns"] = active_cooldowns
        else:
            data["model_cooldowns"] = {}

        # Preserve unexpired key-level cooldown
        if data.get("key_cooldown_until"):
            if data["key_cooldown_until"] <= now_ts:
                data["key_cooldown_until"] = None
            else:
                hours_remaining = (data["key_cooldown_until"] - now_ts) / 3600
                lib_logger.info(
                    f"Preserving key-level cooldown for {mask_credential(key)} "
                    f"during reset ({hours_remaining:.1f}h remaining)"
                )
        else:
            data["key_cooldown_until"] = None

