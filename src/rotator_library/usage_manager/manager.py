# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Core UsageManager orchestrator and shared constants."""

import logging
import asyncio
import time
from datetime import timezone, time as dt_time
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from ..providers import PROVIDER_PLUGINS
from ..async_locks import ReadWriteLock
from ..utils.resilient_io import ResilientStateWriter
from ..utils.provider_locks import ProviderLockManager
from ..batched_persistence import UsagePersistenceManager
from ..config.defaults import USAGE_BATCH_PERSISTENCE
from ..utils.paths import get_data_file
from ..utils.provider_registry import get_provider_registry

lib_logger = logging.getLogger("rotator_library")
MAX_CACHE_ENTRIES = 10_000

from ..config import DEFAULT_FAIR_CYCLE_DURATION, DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD, DEFAULT_CUSTOM_CAP_COOLDOWN_MODE, DEFAULT_CUSTOM_CAP_COOLDOWN_VALUE
from ..error_types import mask_credential


class UsageManagerConfigMixin:
    def _get_rotation_mode(self, provider: str) -> str:
        """
        Get the rotation mode for a provider.

        Args:
            provider: Provider name (e.g., "antigravity", "gemini_cli")

        Returns:
            "balanced" or "sequential"
        """
        return self.provider_rotation_modes.get(provider, "balanced")

    # =========================================================================
    # FAIR CYCLE ROTATION HELPERS
    # =========================================================================

    def _is_fair_cycle_enabled(self, provider: str, rotation_mode: str) -> bool:
        """
        Check if fair cycle rotation is enabled for a provider.

        Args:
            provider: Provider name
            rotation_mode: Current rotation mode ("balanced" or "sequential")

        Returns:
            True if fair cycle is enabled
        """
        # Check provider-specific setting first
        if provider in self.fair_cycle_enabled:
            return self.fair_cycle_enabled[provider]
        # Default: enabled only for sequential mode
        return rotation_mode == "sequential"

    def _get_fair_cycle_tracking_mode(self, provider: str) -> str:
        """
        Get fair cycle tracking mode for a provider.

        Returns:
            "model_group" or "credential"
        """
        return self.fair_cycle_tracking_mode.get(provider, "model_group")

    def _is_fair_cycle_cross_tier(self, provider: str) -> bool:
        """
        Check if fair cycle tracks across all tiers (ignoring priority boundaries).

        Returns:
            True if cross-tier tracking is enabled
        """
        return self.fair_cycle_cross_tier.get(provider, False)

    def _get_fair_cycle_duration(self, provider: str) -> int:
        """
        Get fair cycle duration in seconds for a provider.

        Returns:
            Duration in seconds (default 86400 = 24 hours)
        """
        return self.fair_cycle_duration.get(provider, DEFAULT_FAIR_CYCLE_DURATION)

    def _get_exhaustion_cooldown_threshold(self, provider: str) -> int:
        """
        Get exhaustion cooldown threshold in seconds for a provider.

        A cooldown must exceed this duration to qualify as "exhausted" for fair cycle.

        Returns:
            Threshold in seconds (default 300 = 5 minutes)
        """
        return self.exhaustion_cooldown_threshold.get(
            provider, DEFAULT_EXHAUSTION_COOLDOWN_THRESHOLD
        )

    # =========================================================================
    # CUSTOM CAPS HELPERS
    # =========================================================================

    def _get_custom_cap_config(
        self,
        provider: str,
        tier_priority: int,
        model: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get custom cap config for a provider/tier/model combination.

        Resolution order:
        1. tier + model (exact match)
        2. tier + group (model's quota group)
        3. "default" + model
        4. "default" + group

        Args:
            provider: Provider name
            tier_priority: Credential's priority level
            model: Model name (with provider prefix)

        Returns:
            Cap config dict or None if no custom cap applies
        """
        provider_caps = self.custom_caps.get(provider)
        if not provider_caps:
            return None

        # Strip provider prefix from model
        clean_model = model.split("/")[-1] if "/" in model else model

        # Get quota group for this model
        group = self._get_model_quota_group_by_provider(provider, model)

        # Try to find matching tier config
        tier_config = None
        default_config = None

        for tier_key, models_config in provider_caps.items():
            if tier_key == "default":
                default_config = models_config
                continue

            # Check if this tier_key matches our priority
            if isinstance(tier_key, int) and tier_key == tier_priority:
                tier_config = models_config
                break
            elif isinstance(tier_key, tuple) and tier_priority in tier_key:
                tier_config = models_config
                break

        # Resolution order for tier config
        if tier_config:
            # Try model first
            if clean_model in tier_config:
                return tier_config[clean_model]
            # Try group
            if group and group in tier_config:
                return tier_config[group]

        # Resolution order for default config
        if default_config:
            # Try model first
            if clean_model in default_config:
                return default_config[clean_model]
            # Try group
            if group and group in default_config:
                return default_config[group]

        return None

    def _get_model_quota_group_by_provider(
        self, provider: str, model: str
    ) -> Optional[str]:
        """
        Get quota group for a model using provider name instead of credential.

        Args:
            provider: Provider name
            model: Model name

        Returns:
            Group name or None
        """
        plugin_instance = self._get_provider_instance(provider)
        if plugin_instance and hasattr(plugin_instance, "get_model_quota_group"):
            return plugin_instance.get_model_quota_group(model)
        return None

    def _resolve_custom_cap_max(
        self,
        provider: str,
        model: str,
        cap_config: Dict[str, Any],
        actual_max: Optional[int],
    ) -> Optional[int]:
        """
        Resolve custom cap max_requests value, handling percentages and clamping.

        Args:
            provider: Provider name
            model: Model name (for logging)
            cap_config: Custom cap configuration
            actual_max: Actual API max requests (may be None if unknown)

        Returns:
            Resolved cap value (clamped), or None if can't be calculated
        """
        max_requests = cap_config.get("max_requests")
        if max_requests is None:
            return None

        # Handle percentage
        if isinstance(max_requests, str) and max_requests.endswith("%"):
            if actual_max is None:
                lib_logger.warning(
                    f"Custom cap '{max_requests}' for {provider}/{model} requires known max_requests. "
                    f"Skipping until quota baseline is fetched. Use absolute value for immediate enforcement."
                )
                return None
            try:
                percentage = float(max_requests.rstrip("%")) / 100.0
                calculated = int(actual_max * percentage)
            except ValueError:
                lib_logger.warning(
                    f"Invalid percentage cap '{max_requests}' for {provider}/{model}"
                )
                return None
        else:
            # Absolute value
            try:
                calculated = int(max_requests)
            except (ValueError, TypeError):
                lib_logger.warning(
                    f"Invalid cap value '{max_requests}' for {provider}/{model}"
                )
                return None

        # Clamp to actual max (can only be MORE restrictive)
        if actual_max is not None:
            return min(calculated, actual_max)
        return calculated

    def _calculate_custom_cooldown_until(
        self,
        cap_config: Dict[str, Any],
        window_start_ts: Optional[float],
        natural_reset_ts: Optional[float],
    ) -> Optional[float]:
        """
        Calculate when custom cap cooldown should end, clamped to natural reset.

        Args:
            cap_config: Custom cap configuration
            window_start_ts: When first request was made (for fixed mode)
            natural_reset_ts: Natural quota reset timestamp

        Returns:
            Cooldown end timestamp (clamped), or None if can't calculate
        """
        mode = cap_config.get("cooldown_mode", DEFAULT_CUSTOM_CAP_COOLDOWN_MODE)
        value = cap_config.get("cooldown_value", DEFAULT_CUSTOM_CAP_COOLDOWN_VALUE)

        if mode == "quota_reset":
            calculated = natural_reset_ts
        elif mode == "offset":
            if natural_reset_ts is None:
                return None
            calculated = natural_reset_ts + value
        elif mode == "fixed":
            if window_start_ts is None:
                return None
            calculated = window_start_ts + value
        else:
            lib_logger.warning(f"Unknown cooldown_mode '{mode}', using quota_reset")
            calculated = natural_reset_ts

        if calculated is None:
            return None

        # Clamp to natural reset (can only be MORE restrictive = longer cooldown)
        if natural_reset_ts is not None:
            return max(calculated, natural_reset_ts)
        return calculated

    def _check_and_apply_custom_cap(
        self,
        credential: str,
        model: str,
        request_count: int,
    ) -> bool:
        """
        Check if custom cap is exceeded and apply cooldown if so.

        This should be called after incrementing request_count in record_success().

        Args:
            credential: Credential identifier
            model: Model name (with provider prefix)
            request_count: Current request count for this model

        Returns:
            True if cap exceeded and cooldown applied, False otherwise
        """
        provider = self._get_provider_from_credential(credential)
        if not provider:
            return False

        priority = self._get_credential_priority(credential, provider)
        cap_config = self._get_custom_cap_config(provider, priority, model)
        if not cap_config:
            return False

        # Get model data for actual max and timing info
        key_data = self._usage_data.get(credential, {})
        model_data = key_data.get("models", {}).get(model, {})
        actual_max = model_data.get("quota_max_requests")
        window_start_ts = model_data.get("window_start_ts")
        natural_reset_ts = model_data.get("quota_reset_ts")

        # Resolve custom cap max
        custom_max = self._resolve_custom_cap_max(
            provider, model, cap_config, actual_max
        )
        if custom_max is None:
            return False

        # Check if exceeded
        if request_count < custom_max:
            return False

        # Calculate cooldown end time
        cooldown_until = self._calculate_custom_cooldown_until(
            cap_config, window_start_ts, natural_reset_ts
        )
        if cooldown_until is None:
            # Can't calculate cooldown, use natural reset if available
            if natural_reset_ts:
                cooldown_until = natural_reset_ts
            else:
                lib_logger.warning(
                    f"Custom cap hit for {mask_credential(credential)}/{model} but can't calculate cooldown. "
                    f"Skipping cooldown application."
                )
                return False

        now_ts = time.time()

        # Apply cooldown
        model_cooldowns = key_data.setdefault("model_cooldowns", {})
        model_cooldowns[model] = cooldown_until

        # Store custom cap info in model data for reference
        model_data["custom_cap_max"] = custom_max
        model_data["custom_cap_hit_at"] = now_ts
        model_data["custom_cap_cooldown_until"] = cooldown_until

        hours_until = (cooldown_until - now_ts) / 3600
        lib_logger.info(
            f"Custom cap hit: {mask_credential(credential)} reached {request_count}/{custom_max} "
            f"for {model}. Cooldown for {hours_until:.1f}h"
        )

        # Sync cooldown across quota group
        group = self._get_model_quota_group(credential, model)
        if group:
            grouped_models = self._get_grouped_models(credential, group)
            for grouped_model in grouped_models:
                if grouped_model != model:
                    model_cooldowns[grouped_model] = cooldown_until

        # Check if this should trigger fair cycle exhaustion
        cooldown_duration = cooldown_until - now_ts
        threshold = self._get_exhaustion_cooldown_threshold(provider)
        if cooldown_duration > threshold:
            rotation_mode = self._get_rotation_mode(provider)
            if self._is_fair_cycle_enabled(provider, rotation_mode):
                tier_key = self._get_tier_key(provider, priority)
                tracking_key = self._get_tracking_key(credential, model, provider)
                self._mark_credential_exhausted(
                    credential, provider, tier_key, tracking_key
                )

        return True


from .tracking import (
    UsageManagerQueryMixin,
    UsageManagerRecordingMixin,
    UsageManagerStatisticsMixin,
)
from .rotation import (
    UsageManagerCycleMixin,
    UsageManagerSelectionMixin,
    UsageManagerAcquireMixin,
)
from .persistence import (
    UsageManagerSerializationMixin,
    UsageManagerPersistenceMixin,
    UsageManagerResetMixin,
)


class UsageManagerCore(
    UsageManagerConfigMixin,
    UsageManagerCycleMixin,
    UsageManagerQueryMixin,
    UsageManagerSerializationMixin,
    UsageManagerPersistenceMixin,
    UsageManagerResetMixin,
    UsageManagerSelectionMixin,
    UsageManagerAcquireMixin,
    UsageManagerRecordingMixin,
    UsageManagerStatisticsMixin,
):
    """
    Manages usage statistics and cooldowns for API keys with asyncio-safe locking,
    asynchronous file I/O, lazy-loading mechanism, and weighted random credential rotation.
    """

    def __init__(
        self,
        file_path: Optional[Union[str, Path]] = None,
        daily_reset_time_utc: Optional[str] = "03:00",
        rotation_tolerance: float = 0.0,
        provider_rotation_modes: Optional[Dict[str, str]] = None,
        provider_plugins: Optional[Dict[str, Any]] = None,
        priority_multipliers: Optional[Dict[str, Dict[int, int]]] = None,
        priority_multipliers_by_mode: Optional[
            Dict[str, Dict[str, Dict[int, int]]]
        ] = None,
        sequential_fallback_multipliers: Optional[Dict[str, int]] = None,
        fair_cycle_enabled: Optional[Dict[str, bool]] = None,
        fair_cycle_tracking_mode: Optional[Dict[str, str]] = None,
        fair_cycle_cross_tier: Optional[Dict[str, bool]] = None,
        fair_cycle_duration: Optional[Dict[str, int]] = None,
        exhaustion_cooldown_threshold: Optional[Dict[str, int]] = None,
        custom_caps: Optional[
            Dict[str, Dict[Union[int, Tuple[int, ...], str], Dict[str, Dict[str, Any]]]]
        ] = None,
        credential_to_provider: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the UsageManager.

        Args:
            file_path: Path to the usage data JSON file. If None, uses get_data_file("key_usage.json").
                       Can be absolute Path, relative Path, or string.
            daily_reset_time_utc: Time in UTC when daily stats should reset (HH:MM format)
            rotation_tolerance: Tolerance for weighted random credential rotation.
                - 0.0: Deterministic, least-used credential always selected
                - tolerance = 2.0 - 4.0 (default, recommended): Balanced randomness, can pick credentials within 2 uses of max
                - 5.0+: High randomness, more unpredictable selection patterns
            provider_rotation_modes: Dict mapping provider names to rotation modes.
                - "balanced": Rotate credentials to distribute load evenly (default)
                - "sequential": Use one credential until exhausted (preserves caching)
            provider_plugins: Dict mapping provider names to provider plugin instances.
                Used for per-provider usage reset configuration (window durations, field names).
            priority_multipliers: Dict mapping provider -> priority -> multiplier.
                Universal multipliers that apply regardless of rotation mode.
                Example: {"antigravity": {1: 5, 2: 3}}
            priority_multipliers_by_mode: Dict mapping provider -> mode -> priority -> multiplier.
                Mode-specific overrides. Example: {"antigravity": {"balanced": {3: 1}}}
            sequential_fallback_multipliers: Dict mapping provider -> fallback multiplier.
                Used in sequential mode when priority not in priority_multipliers.
                Example: {"antigravity": 2}
            fair_cycle_enabled: Dict mapping provider -> bool to enable fair cycle rotation.
                When enabled, credentials must all exhaust before any can be reused.
                Default: enabled for sequential mode only.
            fair_cycle_tracking_mode: Dict mapping provider -> tracking mode.
                - "model_group": Track per quota group or model (default)
                - "credential": Track per credential globally
            fair_cycle_cross_tier: Dict mapping provider -> bool for cross-tier tracking.
                - False: Each tier cycles independently (default)
                - True: All credentials must exhaust regardless of tier
            fair_cycle_duration: Dict mapping provider -> cycle duration in seconds.
                Default: 86400 (24 hours)
            exhaustion_cooldown_threshold: Dict mapping provider -> threshold in seconds.
                A cooldown must exceed this to qualify as "exhausted". Default: 300 (5 min)
            custom_caps: Dict mapping provider -> tier -> model/group -> cap config.
                Allows setting custom usage limits per tier, per model or quota group.
                See ProviderInterface.default_custom_caps for format details.
        """
        self._init_config_state(
            file_path,
            rotation_tolerance,
            provider_rotation_modes,
            provider_plugins,
            priority_multipliers,
            priority_multipliers_by_mode,
            sequential_fallback_multipliers,
            credential_to_provider,
        )
        self._init_fair_cycle_state(
            fair_cycle_enabled,
            fair_cycle_tracking_mode,
            fair_cycle_cross_tier,
            fair_cycle_duration,
            exhaustion_cooldown_threshold,
            custom_caps,
        )
        self._init_synchronization_state()
        self._init_usage_caches()
        self._init_persistence_state(file_path)
        self._init_daily_reset_time(daily_reset_time_utc)

    def _init_config_state(
        self,
        file_path: Optional[Union[str, Path]],
        rotation_tolerance: float,
        provider_rotation_modes: Optional[Dict[str, str]],
        provider_plugins: Optional[Dict[str, Any]],
        priority_multipliers: Optional[Dict[str, Dict[int, int]]],
        priority_multipliers_by_mode: Optional[
            Dict[str, Dict[str, Dict[int, int]]]
        ],
        sequential_fallback_multipliers: Optional[Dict[str, int]],
        credential_to_provider: Optional[Dict[str, str]],
    ) -> None:
        # Resolve file_path - use default if not provided
        if file_path is None:
            self.file_path = str(get_data_file("key_usage.json"))
        elif isinstance(file_path, Path):
            self.file_path = str(file_path)
        else:
            # String path - could be relative or absolute
            self.file_path = file_path
        self.rotation_tolerance = rotation_tolerance
        self.provider_rotation_modes = provider_rotation_modes or {}
        self.provider_plugins = provider_plugins or PROVIDER_PLUGINS
        self.credential_to_provider = credential_to_provider or {}
        self.priority_multipliers = priority_multipliers or {}
        self.priority_multipliers_by_mode = priority_multipliers_by_mode or {}
        self.sequential_fallback_multipliers = sequential_fallback_multipliers or {}
        self._provider_instances = get_provider_registry()  # Shared singleton registry
        self._provider_capability_cache: dict[str, dict[str, Any]] = {}
        self.key_states: Dict[str, Dict[str, Any]] = {}

    def _init_fair_cycle_state(
        self,
        fair_cycle_enabled: Optional[Dict[str, bool]],
        fair_cycle_tracking_mode: Optional[Dict[str, str]],
        fair_cycle_cross_tier: Optional[Dict[str, bool]],
        fair_cycle_duration: Optional[Dict[str, int]],
        exhaustion_cooldown_threshold: Optional[Dict[str, int]],
        custom_caps: Optional[
            Dict[str, Dict[Union[int, Tuple[int, ...], str], Dict[str, Dict[str, Any]]]]
        ],
    ) -> None:
        # Fair cycle rotation configuration
        self.fair_cycle_enabled = fair_cycle_enabled or {}
        self.fair_cycle_tracking_mode = fair_cycle_tracking_mode or {}
        self.fair_cycle_cross_tier = fair_cycle_cross_tier or {}
        self.fair_cycle_duration = fair_cycle_duration or {}
        self.exhaustion_cooldown_threshold = exhaustion_cooldown_threshold or {}
        self.custom_caps = custom_caps or {}
        # In-memory cycle state: {provider: {tier_key: {tracking_key: {"cycle_started_at": float, "exhausted": Set[str]}}}}
        self._cycle_exhausted: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

    def _init_synchronization_state(self) -> None:
        # Per-provider locks for parallel access (sharded locking)
        # This allows concurrent operations on different providers
        self._provider_lock_manager = ProviderLockManager()

        # Read-write lock for usage data: allows parallel reads, exclusive writes
        self._data_lock = ReadWriteLock()
        self._usage_data: Optional[Dict] = None
        self._initialized = asyncio.Event()
        self._init_lock = asyncio.Lock()

    def _init_usage_caches(self) -> None:
        # Lazy caches for stable quota group config (OrderedDict for LRU eviction)
        # (key, model) -> group_name or None
        self._quota_group_cache: "OrderedDict[str, OrderedDict[str, Optional[str]]]" = OrderedDict()
        # credential -> provider for parsed/derived provider resolution hits
        self._provider_resolution_cache: "OrderedDict[str, str]" = OrderedDict()
        # (key, group) -> (models_list, request_count_at_snapshot)
        # Used by record_success to skip syncing siblings when count unchanged
        self._grouped_models_cache: "OrderedDict[str, OrderedDict[str, Tuple[List[str], int]]]" = OrderedDict()

    def _init_persistence_state(
        self, file_path: Optional[Union[str, Path]]
    ) -> None:
        # Resilient writer for usage data persistence
        self._state_writer = ResilientStateWriter(file_path or "", lib_logger)

        # Batch persistence manager for high-throughput scenarios
        # Enabled via USAGE_BATCH_PERSISTENCE=true environment variable
        self._batch_persistence: Optional[UsagePersistenceManager] = None
        self._use_batch_persistence = USAGE_BATCH_PERSISTENCE

    def _init_daily_reset_time(self, daily_reset_time_utc: Optional[str]) -> None:
        if daily_reset_time_utc:
            hour, minute = map(int, daily_reset_time_utc.split(":"))
            self.daily_reset_time_utc = dt_time(
                hour=hour, minute=minute, tzinfo=timezone.utc
            )
        else:
            self.daily_reset_time_utc = None
