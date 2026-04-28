# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import asyncio
import random
from typing import Dict, List, Tuple


class UsageManagerSelectionMixin:
    def _initialize_key_states(self, keys: List[str]):
        """Initializes state tracking for all provided keys if not already present."""
        for key in keys:
            self.key_states.setdefault(key, {
                "lock": asyncio.Lock(),
                "condition": asyncio.Condition(),
                "models_in_use": {},  # Dict[model_name, concurrent_count]
                "total_in_use": 0,
            })

    def _select_weighted_random(self, candidates: List[tuple], tolerance: float) -> str:
        """
        Selects a credential using weighted random selection based on usage counts.

        Args:
            candidates: List of (credential_id, usage_count) tuples
            tolerance: Tolerance value for weight calculation

        Returns:
            Selected credential ID

        Formula:
            weight = (max_usage - credential_usage) + tolerance + 1

        This formula ensures:
            - Lower usage = higher weight = higher selection probability
            - Tolerance adds variability: higher tolerance means more randomness
            - The +1 ensures all credentials have at least some chance of selection
        """
        if not candidates:
            raise ValueError("Cannot select from empty candidate list")

        if len(candidates) == 1:
            return candidates[0][0]

        # Extract usage counts
        usage_counts = [usage for _, usage in candidates]
        max_usage = max(usage_counts)

        # Calculate weights using the formula: (max - current) + tolerance + 1
        weights = []
        for credential, usage in candidates:
            weight = (max_usage - usage) + tolerance + 1
            weights.append(weight)

        # Random selection with weights
        selected_credential = random.choices(
            [cred for cred, _ in candidates], weights=weights, k=1
        )[0]

        return selected_credential

    async def _get_cooldown_snapshot(
        self, keys: List[str], normalized_model: str
    ) -> Dict[str, Tuple[float, float]]:
        """Snapshot cooldowns for all keys with a single read lock."""
        async with self._data_lock.read():
            snapshot = {}
            for key in keys:
                key_data = self._usage_data.get(key, {})
                key_cd = key_data.get("key_cooldown_until") or 0
                model_cooldowns = key_data.get("model_cooldowns", {})
                model_cd = model_cooldowns.get(normalized_model) or 0

                if model_cd == 0:
                    quota_cd = self._check_quota_group_cooldown(
                        key, model_cooldowns, normalized_model
                    )
                    if quota_cd > model_cd:
                        model_cd = quota_cd

                snapshot[key] = (key_cd, model_cd)

            return snapshot
