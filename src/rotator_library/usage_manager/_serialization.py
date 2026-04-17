# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
import time
from datetime import datetime
import logging
from typing import Optional, Dict, List, Tuple, Any
from ..error_types import mask_credential


class UsageManagerSerializationMixin:
    def _format_timestamp_local(self, ts: Optional[float]) -> Optional[str]:
        """
        Format Unix timestamp as local time string with timezone offset.

        Args:
            ts: Unix timestamp or None

        Returns:
            Formatted string like "2025-12-07 14:30:17 +0100" or None
        """
        if ts is None:
            return None
        try:
            dt = datetime.fromtimestamp(ts).astimezone()  # Local timezone
            # Use UTC offset for conciseness (works on all platforms)
            return dt.strftime("%Y-%m-%d %H:%M:%S %z")
        except (OSError, ValueError, OverflowError):
            return None

    def _add_readable_timestamps(self, data: Dict) -> Dict:
        """
        Add human-readable timestamp fields to usage data before saving.

        Adds 'window_started' and 'quota_resets' fields derived from
        Unix timestamps for easier debugging and monitoring.

        Args:
            data: The usage data dict to enhance

        Returns:
            The same dict with readable timestamp fields added
        """
        for key, key_data in data.items():
            # Handle per-model structure
            models = key_data.get("models", {})
            for model_name, model_stats in models.items():
                if not isinstance(model_stats, dict):
                    continue

                # Add readable window start time
                window_start = model_stats.get("window_start_ts")
                if window_start:
                    model_stats["window_started"] = self._format_timestamp_local(
                        window_start
                    )
                elif "window_started" in model_stats:
                    del model_stats["window_started"]

                # Add readable reset time
                quota_reset = model_stats.get("quota_reset_ts")
                if quota_reset:
                    model_stats["quota_resets"] = self._format_timestamp_local(
                        quota_reset
                    )
                elif "quota_resets" in model_stats:
                    del model_stats["quota_resets"]

        return data

    def _sort_sequential(
        self,
        candidates: List[Tuple[str, int]],
        credential_priorities: Optional[Dict[str, int]] = None,
    ) -> List[Tuple[str, int]]:
        """
        Sort credentials for sequential mode with position retention.

        Credentials maintain their position based on established usage patterns,
        ensuring that actively-used credentials remain primary until exhausted.

        Sorting order (within each sort key, lower value = higher priority):
        1. Priority tier (lower number = higher priority)
        2. Usage count (higher = more established in rotation, maintains position)
        3. Last used timestamp (higher = more recent, tiebreaker for stickiness)
        4. Credential ID (alphabetical, stable ordering)

        Args:
            candidates: List of (credential_id, usage_count) tuples
            credential_priorities: Optional dict mapping credentials to priority levels

        Returns:
            Sorted list of candidates (same format as input)
        """
        if not candidates:
            return []

        if len(candidates) == 1:
            return candidates

        def sort_key(item: Tuple[str, int]) -> Tuple[int, int, float, str]:
            cred, usage_count = item
            priority = (
                credential_priorities.get(cred, 999) if credential_priorities else 999
            )
            last_used = (
                self._usage_data.get(cred, {}).get("last_used_ts", 0)
                if self._usage_data
                else 0
            )
            return (
                priority,  # ASC: lower priority number = higher priority
                -usage_count,  # DESC: higher usage = more established
                -last_used,  # DESC: more recent = preferred for ties
                cred,  # ASC: stable alphabetical ordering
            )

        sorted_candidates = sorted(candidates, key=sort_key)

        # Debug logging - show top 3 credentials in ordering
        if lib_logger.isEnabledFor(logging.DEBUG):
            order_info = [
                f"{mask_credential(c)}(p={credential_priorities.get(c, 999) if credential_priorities else 'N/A'}, u={u})"
                for c, u in sorted_candidates[:3]
            ]
            lib_logger.debug(f"Sequential ordering: {' → '.join(order_info)}")

        return sorted_candidates

    # =========================================================================
    # FAIR CYCLE PERSISTENCE
    # =========================================================================

    def _serialize_cycle_state(self) -> Dict[str, Any]:
        """
        Serialize in-memory cycle state for JSON persistence.

        Converts sets to lists for JSON compatibility.
        """
        result: Dict[str, Any] = {}
        for provider, tier_data in self._cycle_exhausted.items():
            result[provider] = {}
            for tier_key, tracking_data in tier_data.items():
                result[provider][tier_key] = {}
                for tracking_key, cycle_data in tracking_data.items():
                    result[provider][tier_key][tracking_key] = {
                        "cycle_started_at": cycle_data.get("cycle_started_at"),
                        "exhausted": list(cycle_data.get("exhausted", set())),
                    }
        return result

    def _deserialize_cycle_state(self, data: Dict[str, Any]) -> None:
        """
        Deserialize cycle state from JSON and populate in-memory structure.

        Converts lists back to sets and validates expired cycles.
        """
        self._cycle_exhausted = {}
        now_ts = time.time()

        for provider, tier_data in data.items():
            if not isinstance(tier_data, dict):
                continue
            self._cycle_exhausted[provider] = {}

            for tier_key, tracking_data in tier_data.items():
                if not isinstance(tracking_data, dict):
                    continue
                self._cycle_exhausted[provider][tier_key] = {}

                for tracking_key, cycle_data in tracking_data.items():
                    if not isinstance(cycle_data, dict):
                        continue

                    cycle_started = cycle_data.get("cycle_started_at")
                    exhausted_list = cycle_data.get("exhausted", [])

                    # Check if cycle has expired
                    if cycle_started is not None:
                        duration = self._get_fair_cycle_duration(provider)
                        if now_ts >= cycle_started + duration:
                            # Cycle expired - skip (don't restore)
                            lib_logger.debug(
                                f"Fair cycle expired for {provider}/{tier_key}/{tracking_key} - not restoring"
                            )
                            continue

                    # Restore valid cycle
                    self._cycle_exhausted[provider][tier_key][tracking_key] = {
                        "cycle_started_at": cycle_started,
                        "exhausted": set(exhausted_list) if exhausted_list else set(),
                    }

        # Log restoration summary
        total_cycles = sum(
            len(tracking)
            for tier in self._cycle_exhausted.values()
            for tracking in tier.values()
        )
        if total_cycles > 0:
            lib_logger.info(f"Restored {total_cycles} active fair cycle(s) from disk")

