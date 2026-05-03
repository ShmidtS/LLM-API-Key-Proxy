# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
from ._config import UsageManagerConfigMixin
from ._cycle import UsageManagerCycleMixin
from ._query import UsageManagerQueryMixin
from ._serialization import UsageManagerSerializationMixin
from ._persistence import UsageManagerPersistenceMixin
from ._reset import UsageManagerResetMixin
from ._selection import UsageManagerSelectionMixin
from ._acquire import UsageManagerAcquireMixin
from ._recording import UsageManagerRecordingMixin
from ._statistics import UsageManagerStatisticsMixin
import os
import asyncio
from datetime import timezone, time as dt_time
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from ..providers import PROVIDER_PLUGINS
from ..async_locks import ReadWriteLock
from ..utils.resilient_io import ResilientStateWriter
from ..utils.provider_locks import ProviderLockManager
from ..batched_persistence import UsagePersistenceManager
from ..utils.paths import get_data_file
from ..utils.provider_registry import get_provider_registry


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

        # Fair cycle rotation configuration
        self.fair_cycle_enabled = fair_cycle_enabled or {}
        self.fair_cycle_tracking_mode = fair_cycle_tracking_mode or {}
        self.fair_cycle_cross_tier = fair_cycle_cross_tier or {}
        self.fair_cycle_duration = fair_cycle_duration or {}
        self.exhaustion_cooldown_threshold = exhaustion_cooldown_threshold or {}
        self.custom_caps = custom_caps or {}
        # In-memory cycle state: {provider: {tier_key: {tracking_key: {"cycle_started_at": float, "exhausted": Set[str]}}}}
        self._cycle_exhausted: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}

        # Per-provider locks for parallel access (sharded locking)
        # This allows concurrent operations on different providers
        self._provider_lock_manager = ProviderLockManager()

        # Read-write lock for usage data: allows parallel reads, exclusive writes
        self._data_lock = ReadWriteLock()
        self._usage_data: Optional[Dict] = None
        self._initialized = asyncio.Event()
        self._init_lock = asyncio.Lock()


        # Lazy caches for stable quota group config (OrderedDict for LRU eviction)
        # (key, model) -> group_name or None
        self._quota_group_cache: "OrderedDict[str, OrderedDict[str, Optional[str]]]" = OrderedDict()
        # credential -> provider for parsed/derived provider resolution hits
        self._provider_resolution_cache: "OrderedDict[str, str]" = OrderedDict()
        # (key, group) -> (models_list, request_count_at_snapshot)
        # Used by record_success to skip syncing siblings when count unchanged
        self._grouped_models_cache: "OrderedDict[str, OrderedDict[str, Tuple[List[str], int]]]" = OrderedDict()

        # Resilient writer for usage data persistence
        self._state_writer = ResilientStateWriter(file_path or "", lib_logger)

        # Batch persistence manager for high-throughput scenarios
        # Enabled via USAGE_PERSISTENCE_ENABLE=true environment variable
        self._batch_persistence: Optional[UsagePersistenceManager] = None
        self._use_batch_persistence = os.getenv(
            "USAGE_BATCH_PERSISTENCE", "true"
        ).lower() in ("true", "1", "yes")

        if daily_reset_time_utc:
            hour, minute = map(int, daily_reset_time_utc.split(":"))
            self.daily_reset_time_utc = dt_time(
                hour=hour, minute=minute, tzinfo=timezone.utc
            )
        else:
            self.daily_reset_time_utc = None
