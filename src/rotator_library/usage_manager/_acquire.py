# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
import asyncio
import logging
import time
from typing import Dict, List, Optional, Tuple
from ..error_types import NoAvailableKeysError, mask_credential
from ..utils.model_utils import extract_provider_from_model


class UsageManagerAcquireMixin:
    async def _get_key_cooldown_async(
        self, key: str, normalized_model: str
    ) -> Tuple[str, Tuple[float, float]]:
        """Helper for parallel cooldown lookup.

        Checks cooldown for the requested model AND for the virtual
        quota-baseline model (e.g. zai/_quota) when the model belongs
        to a quota group.  This ensures that cooldowns placed by the
        background quota refresh on the virtual model are respected
        even before real models are discovered and added to the group.
        """
        async with self._data_lock.read():
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

            return key, (key_cd, model_cd)

    async def acquire_key(
        self,
        available_keys: List[str],
        model: str,
        deadline: float,
        max_concurrent: int = 1,
        credential_priorities: Optional[Dict[str, int]] = None,
        credential_tier_names: Optional[Dict[str, str]] = None,
        all_provider_credentials: Optional[List[str]] = None,
    ) -> str:
        """
        Acquires the best available key using a tiered, model-aware locking strategy,
        respecting a global deadline and credential priorities.

        Priority Logic:
        - Groups credentials by priority level (1=highest, 2=lower, etc.)
        - Always tries highest priority (lowest number) first
        - Within same priority, sorts by usage count (load balancing)
        - Only moves to next priority if all higher-priority keys exhausted/busy

        Args:
            available_keys: List of credential identifiers to choose from
            model: Model name being requested
            deadline: Timestamp after which to stop trying
            max_concurrent: Maximum concurrent requests allowed per credential
            credential_priorities: Optional dict mapping credentials to priority levels (1=highest)
            credential_tier_names: Optional dict mapping credentials to tier names (for logging)
            all_provider_credentials: Full list of provider credentials (used for cycle reset checks)

        Returns:
            Selected credential identifier

        Raises:
            NoAvailableKeysError: If no key could be acquired within the deadline
        """
        await self._lazy_init()
        self._initialize_key_states(available_keys)

        # FAST PATH: Single credential case - skip complex logic
        if len(available_keys) == 1:
            key = available_keys[0]
            state = self.key_states[key]
            async with state["lock"]:
                total_in_use = state.get("total_in_use", 0)
                if total_in_use < max_concurrent:
                    state["models_in_use"][model] = (
                        state["models_in_use"].get(model, 0) + 1
                    )
                    state["total_in_use"] = total_in_use + 1
                    if lib_logger.isEnabledFor(logging.INFO):
                        lib_logger.info(
                            f"Acquired key {mask_credential(key)} for model {model} "
                            f"(fast path: concurrent {total_in_use + 1}/{max_concurrent})"
                        )
                    return key
            # If we get here, the single key is at capacity - fall through to waiting logic

        # Normalize model name for consistent cooldown lookup
        # (cooldowns are stored under normalized names by record_failure)
        # Use first credential for provider detection; all credentials passed here
        # are for the same provider (filtered by client.py before calling acquire_key).
        # For providers without normalize_model_for_tracking (non-Antigravity),
        # this returns the model unchanged, so cooldown lookups work as before.
        normalized_model = (
            self._normalize_model_for_tracking(available_keys[0], model) if available_keys else model
        )

        # This loop continues as long as the global deadline has not been met.
        while time.monotonic() < deadline:
            now = time.time()

            # Group credentials by priority level (if priorities provided)
            if credential_priorities:
                # Snapshot cooldown data in parallel, then release immediately
                cooldown_snapshot = dict(
                await asyncio.gather(
                *[
                self._get_key_cooldown_async(key, normalized_model)
                for key in available_keys
                ]
                )
                )
                                # Group keys by priority level — OUTSIDE the read lock
                priority_groups = {}
                for key in available_keys:
                    key_cd, model_cd = cooldown_snapshot.get(key, (0, 0))

                    # Skip keys on cooldown (use normalized model for lookup)
                    if key_cd > now or model_cd > now:
                        continue

                    # Skip keys with confirmed exhausted quota (baseline_remaining_fraction <= 0)
                    # Background refresh sets this from the provider's quota API,
                    # so we avoid keys that are known-exhausted even before cooldown kicks in.
                    baseline = self._get_baseline_remaining(key, normalized_model)
                    if baseline is not None and baseline <= 0:
                        continue

                    # Get priority for this key (default to 999 if not specified)
                    priority = credential_priorities.get(key, 999)

                    # Get usage count for load balancing within priority groups
                    # Uses grouped usage if model is in a quota group
                    usage_count = self._get_grouped_usage_count(key, model)

                    # Group by priority
                    if priority not in priority_groups:
                        priority_groups[priority] = []
                    priority_groups[priority].append((key, usage_count))

                # Try priority groups in order (1, 2, 3, ...)
                sorted_priorities = sorted(priority_groups.keys())

                for priority_level in sorted_priorities:
                    keys_in_priority = priority_groups[priority_level]

                    # Determine selection method based on provider's rotation mode
                    provider = extract_provider_from_model(model)
                    rotation_mode = self._get_rotation_mode(provider)

                    # Fair cycle filtering
                    if provider and self._is_fair_cycle_enabled(
                        provider, rotation_mode
                    ):
                        tier_key = self._get_tier_key(provider, priority_level)
                        tracking_key = self._get_tracking_key(
                            keys_in_priority[0][0] if keys_in_priority else "",
                            model,
                            provider,
                        )

                        # Get all credentials for this tier (for cycle completion check)
                        all_tier_creds = self._get_all_credentials_for_tier_key(
                            provider,
                            tier_key,
                            all_provider_credentials or available_keys,
                            credential_priorities,
                        )

                        # Check if cycle should reset (all exhausted, expired, or none available)
                        if self._should_reset_cycle(
                            provider,
                            tier_key,
                            tracking_key,
                            all_tier_creds,
                            available_not_on_cooldown=[
                                key for key, _ in keys_in_priority
                            ],
                        ):
                            self._reset_cycle(provider, tier_key, tracking_key)

                        # Filter out exhausted credentials
                        filtered_keys = []
                        for key, usage_count in keys_in_priority:
                            if not self._is_credential_exhausted_in_cycle(
                                key, provider, tier_key, tracking_key
                            ):
                                filtered_keys.append((key, usage_count))

                        keys_in_priority = filtered_keys

                    # Calculate effective concurrency based on priority tier
                    multiplier = self._get_priority_multiplier(
                        provider, priority_level, rotation_mode
                    )
                    effective_max_concurrent = max_concurrent * multiplier

                    # Within each priority group, use existing tier1/tier2 logic
                    tier1_keys, tier2_keys = [], []
                    for key, usage_count in keys_in_priority:
                        key_state = self.key_states[key]

                        # Tier 1: Keys already active for this model that can accept more concurrent requests.
                        if (
                            key_state["models_in_use"].get(model, 0)
                            < effective_max_concurrent
                            and key_state["models_in_use"]
                        ):
                            tier1_keys.append((key, usage_count))
                        # Tier 2: Completely idle keys.
                        elif not key_state["models_in_use"]:
                            tier2_keys.append((key, usage_count))

                    if rotation_mode == "sequential":
                        # Sequential mode: sort credentials by priority, usage, recency
                        # Keep all candidates in sorted order (no filtering to single key)
                        selection_method = "sequential"
                        all_seq_keys = tier2_keys + tier1_keys
                        if all_seq_keys:
                            seq_scores = {
                                cred: self._score_key(cred, usage, credential_priorities)
                                for cred, usage in all_seq_keys
                            }
                            tier1_keys = self._sort_sequential(
                                tier1_keys, credential_priorities, scores=seq_scores
                            ) if tier1_keys else []
                            tier2_keys = self._sort_sequential(
                                tier2_keys, credential_priorities, scores=seq_scores
                            ) if tier2_keys else []
                        all_available_keys = tier2_keys + tier1_keys
                    elif self.rotation_tolerance > 0:
                        # Balanced mode with weighted randomness across ALL candidates
                        selection_method = "weighted-random"
                        all_candidates = tier2_keys + tier1_keys  # idle first for better distribution
                        if all_candidates:
                            selected_key = self._select_weighted_random(
                                all_candidates, self.rotation_tolerance
                            )
                            all_available_keys = [
                                (k, u) for k, u in all_candidates if k == selected_key
                            ]
                        else:
                            all_available_keys = []
                    else:
                        # Deterministic: sort by usage, idle keys first
                        selection_method = "least-used"
                        all_available_keys = sorted(
                            tier2_keys + tier1_keys, key=lambda x: x[1]
                        )

                    for key, usage in all_available_keys:
                        state = self.key_states[key]
                        async with state["lock"]:
                            current_count = state["models_in_use"].get(model, 0)
                            if current_count < effective_max_concurrent:
                                # Track whether this is a reused-active or idle key
                                is_active = bool(state["models_in_use"])
                                state["models_in_use"][model] = current_count + 1
                                state["total_in_use"] = state.get("total_in_use", 0) + 1
                                if lib_logger.isEnabledFor(logging.INFO):
                                    tier_name = (
                                        credential_tier_names.get(key, "unknown")
                                        if credential_tier_names
                                        else "unknown"
                                    )
                                    quota_display = self._get_quota_display(key, model)
                                    selection_source = "reused-active" if is_active else "idle"
                                    lib_logger.info(
                                        f"Acquired key {mask_credential(key)} for model {model} "
                                        f"(tier: {tier_name}, priority: {priority_level}, selection: {selection_method}, selection_source: {selection_source}, concurrent: {state['models_in_use'][model]}/{effective_max_concurrent}, {quota_display})"
                                    )
                                return key

                # If we get here, all priority groups were exhausted but keys might become available
                # Collect all keys across all priorities for waiting
                all_potential_keys = []
                for keys_list in priority_groups.values():
                    all_potential_keys.extend(keys_list)

                if not all_potential_keys:
                    # All credentials are on cooldown or locked - check if waiting makes sense
                    soonest_end = await self.get_soonest_cooldown_end(
                        available_keys, model
                    )

                    if soonest_end is None:
                        # No cooldowns active but no keys available - all are locked by concurrent requests
                        # Wait on any key's condition variable to be notified when a key is released
                        lib_logger.debug(
                            "All keys are busy. Waiting for a key to be released..."
                        )
                        # Pick any available key to wait on (they're all locked)
                        if available_keys:
                            wait_condition = self.key_states[available_keys[0]][
                                "condition"
                            ]
                            try:
                                async with wait_condition:
                                    remaining_budget = deadline - time.monotonic()
                                    if remaining_budget <= 0:
                                        break
                                    await asyncio.wait_for(
                                        wait_condition.wait(),
                                        timeout=min(0.5, remaining_budget),
                                    )
                            except asyncio.TimeoutError:
                                pass  # Continue loop and re-evaluate
                        else:
                            await asyncio.sleep(0.1)
                        continue

                    remaining_budget = deadline - time.monotonic()
                    wait_needed = soonest_end - time.time()

                    if wait_needed > remaining_budget:
                        # Fail fast - no credential will be available in time
                        lib_logger.warning(
                            f"All credentials on cooldown. Soonest available in {wait_needed:.1f}s, "
                            f"but only {remaining_budget:.1f}s budget remaining. Failing fast."
                        )
                        break  # Exit loop, will raise NoAvailableKeysError

                    # Wait for the credential to become available
                    lib_logger.info(
                        f"All credentials on cooldown. Waiting {wait_needed:.1f}s for soonest credential..."
                    )
                    await asyncio.sleep(min(wait_needed + 0.1, remaining_budget))
                    continue

                # Wait for the highest priority key with lowest usage
                best_priority = min(priority_groups.keys())
                best_priority_keys = priority_groups[best_priority]
                best_wait_key = min(best_priority_keys, key=lambda x: x[1])[0]
                wait_condition = self.key_states[best_wait_key]["condition"]

                lib_logger.info(
                    f"All Priority-{best_priority} keys are busy. Waiting for highest priority credential to become available..."
                )

            else:
                # Original logic when no priorities specified

                # Determine selection method based on provider's rotation mode
                provider = extract_provider_from_model(model)
                rotation_mode = self._get_rotation_mode(provider)

                # Calculate effective concurrency for default priority (999)
                # When no priorities are specified, all credentials get default priority
                default_priority = 999
                multiplier = self._get_priority_multiplier(
                    provider, default_priority, rotation_mode
                )
                effective_max_concurrent = max_concurrent * multiplier

                tier1_keys, tier2_keys = [], []

                # Snapshot cooldown data in parallel, then release immediately
                cooldown_snapshot = dict(
                await asyncio.gather(
                *[
                self._get_key_cooldown_async(key, normalized_model)
                for key in available_keys
                ]
                )
                )

                # Filter and tier keys — OUTSIDE the read lock
                for key in available_keys:
                    key_cd, model_cd = cooldown_snapshot.get(key, (0, 0))

                    # Skip keys on cooldown (use normalized model for lookup)
                    if key_cd > now or model_cd > now:
                        continue

                    # Skip keys with confirmed exhausted quota (baseline_remaining_fraction <= 0)
                    # Background refresh sets this from the provider's quota API,
                    # so we avoid keys that are known-exhausted even before cooldown kicks in.
                    baseline = self._get_baseline_remaining(key, normalized_model)
                    if baseline is not None and baseline <= 0:
                        continue

                    # Prioritize keys based on their current usage to ensure load balancing.
                    # Uses grouped usage if model is in a quota group
                    usage_count = self._get_grouped_usage_count(key, model)
                    key_state = self.key_states[key]

                    # Tier 1: Keys already active for this model that can accept more concurrent requests.
                    if (
                        key_state["models_in_use"].get(model, 0)
                        < effective_max_concurrent
                        and key_state["models_in_use"]
                    ):
                        tier1_keys.append((key, usage_count))
                    # Tier 2: Completely idle keys.
                    elif not key_state["models_in_use"]:
                        tier2_keys.append((key, usage_count))

                # Fair cycle filtering (non-priority case)
                if provider and self._is_fair_cycle_enabled(provider, rotation_mode):
                    tier_key = self._get_tier_key(provider, default_priority)
                    tracking_key = self._get_tracking_key(
                        available_keys[0] if available_keys else "",
                        model,
                        provider,
                    )

                    # Get all credentials for this tier (for cycle completion check)
                    all_tier_creds = self._get_all_credentials_for_tier_key(
                        provider,
                        tier_key,
                        all_provider_credentials or available_keys,
                        None,
                    )

                    # Check if cycle should reset (all exhausted, expired, or none available)
                    if self._should_reset_cycle(
                        provider,
                        tier_key,
                        tracking_key,
                        all_tier_creds,
                        available_not_on_cooldown=[
                            key for key, _ in (tier1_keys + tier2_keys)
                        ],
                    ):
                        self._reset_cycle(provider, tier_key, tracking_key)

                    # Filter out exhausted credentials from both tiers
                    tier1_keys = [
                        (key, usage)
                        for key, usage in tier1_keys
                        if not self._is_credential_exhausted_in_cycle(
                            key, provider, tier_key, tracking_key
                        )
                    ]
                    tier2_keys = [
                        (key, usage)
                        for key, usage in tier2_keys
                        if not self._is_credential_exhausted_in_cycle(
                            key, provider, tier_key, tracking_key
                        )
                    ]

                if rotation_mode == "sequential":
                    # Sequential mode: sort credentials by priority, usage, recency
                    # Keep all candidates in sorted order (no filtering to single key)
                    selection_method = "sequential"
                    all_seq_keys = tier2_keys + tier1_keys
                    if all_seq_keys:
                        seq_scores = {
                            cred: self._score_key(cred, usage, credential_priorities)
                            for cred, usage in all_seq_keys
                        }
                        tier1_keys = self._sort_sequential(
                            tier1_keys, credential_priorities, scores=seq_scores
                        ) if tier1_keys else []
                        tier2_keys = self._sort_sequential(
                            tier2_keys, credential_priorities, scores=seq_scores
                        ) if tier2_keys else []
                    # Combine idle keys first, then active, for better distribution.
                    all_available_keys = tier2_keys + tier1_keys
                elif self.rotation_tolerance > 0:
                    # Balanced mode with weighted randomness across ALL candidates
                    selection_method = "weighted-random"
                    all_candidates = tier2_keys + tier1_keys  # idle first for better distribution
                    if all_candidates:
                        selected_key = self._select_weighted_random(
                            all_candidates, self.rotation_tolerance
                        )
                        all_available_keys = [
                            (k, u) for k, u in all_candidates if k == selected_key
                        ]
                    else:
                        all_available_keys = []
                else:
                    # Deterministic: sort by usage, idle keys first
                    selection_method = "least-used"
                    all_available_keys = sorted(
                        tier2_keys + tier1_keys, key=lambda x: x[1]
                    )

                # Attempt to acquire a key, preferring idle keys for better distribution.
                for key, usage in all_available_keys:
                    state = self.key_states[key]
                    async with state["lock"]:
                        current_count = state["models_in_use"].get(model, 0)
                        is_currently_active = bool(state["models_in_use"])
                        if is_currently_active:
                            if current_count < effective_max_concurrent:
                                state["models_in_use"][model] = current_count + 1
                            else:
                                continue
                        else:
                            state["models_in_use"][model] = 1
                        state["total_in_use"] = state.get("total_in_use", 0) + 1
                        if lib_logger.isEnabledFor(logging.INFO):
                            tier_name = (
                                credential_tier_names.get(key)
                                if credential_tier_names
                                else None
                            )
                            tier_info = f"tier: {tier_name}, " if tier_name else ""
                            quota_display = self._get_quota_display(key, model)
                            selection_source = "reused-active" if is_currently_active else "idle"
                            lib_logger.info(
                                f"Acquired key {mask_credential(key)} for model {model} "
                                f"({tier_info}selection: {selection_method}, selection_source: {selection_source}, concurrent: {state['models_in_use'][model]}/{effective_max_concurrent}, {quota_display})"
                            )
                        return key

                # If all eligible keys are locked, wait for a key to be released.
                lib_logger.info(
                    "All keys are busy with concurrent requests. Waiting for one to become available..."
                )

                all_potential_keys = tier1_keys + tier2_keys
                if not all_potential_keys:
                    # All credentials are on cooldown or locked - check if waiting makes sense
                    soonest_end = await self.get_soonest_cooldown_end(
                        available_keys, model
                    )

                    if soonest_end is None:
                        # No cooldowns active but no keys available - all are locked by concurrent requests
                        # Wait on any key's condition variable to be notified when a key is released
                        lib_logger.debug(
                            "All keys are busy. Waiting for a key to be released..."
                        )
                        # Pick any available key to wait on (they're all locked)
                        if available_keys:
                            wait_condition = self.key_states[available_keys[0]][
                                "condition"
                            ]
                            try:
                                async with wait_condition:
                                    remaining_budget = deadline - time.monotonic()
                                    if remaining_budget <= 0:
                                        break
                                    await asyncio.wait_for(
                                        wait_condition.wait(),
                                        timeout=min(0.5, remaining_budget),
                                    )
                            except asyncio.TimeoutError:
                                pass  # Continue loop and re-evaluate
                        else:
                            await asyncio.sleep(0.1)
                        continue

                    remaining_budget = deadline - time.monotonic()
                    wait_needed = soonest_end - time.time()

                    if wait_needed > remaining_budget:
                        # Fail fast - no credential will be available in time
                        lib_logger.warning(
                            f"All credentials on cooldown. Soonest available in {wait_needed:.1f}s, "
                            f"but only {remaining_budget:.1f}s budget remaining. Failing fast."
                        )
                        break  # Exit loop, will raise NoAvailableKeysError

                    # Wait for the credential to become available
                    lib_logger.info(
                        f"All credentials on cooldown. Waiting {wait_needed:.1f}s for soonest credential..."
                    )
                    await asyncio.sleep(min(wait_needed + 0.1, remaining_budget))
                    continue

                # Wait on the condition of the key with the lowest current usage.
                best_wait_key = min(all_potential_keys, key=lambda x: x[1])[0]
                wait_condition = self.key_states[best_wait_key]["condition"]

            try:
                async with wait_condition:
                    remaining_budget = deadline - time.monotonic()
                    if remaining_budget <= 0:
                        break  # Exit if the budget has already been exceeded.
                    # Wait for a notification, but no longer than the remaining budget or 1 second.
                    await asyncio.wait_for(
                        wait_condition.wait(), timeout=min(1, remaining_budget)
                    )
                lib_logger.debug("Notified that a key was released. Re-evaluating...")
            except asyncio.TimeoutError:
                # This is not an error, just a timeout for the wait. The main loop will re-evaluate.
                lib_logger.debug("Wait timed out. Re-evaluating for any available key.")

        # If the loop exits, it means the deadline was exceeded.
        raise NoAvailableKeysError(
            f"Could not acquire a key for model {model} within the global time budget."
        )

    async def release_key(self, key: str, model: str):
        """Releases a key's lock for a specific model and notifies waiting tasks."""
        if key not in self.key_states:
            return

        state = self.key_states[key]
        async with state["lock"]:
            if model in state["models_in_use"]:
                state["models_in_use"][model] -= 1
                state["total_in_use"] = max(0, state.get("total_in_use", 1) - 1)
                remaining = state["models_in_use"][model]
                if remaining <= 0:
                    del state["models_in_use"][model]  # Clean up when count reaches 0
                lib_logger.info(
                    f"Released credential {mask_credential(key)} from model {model} "
                    f"(remaining concurrent: {max(0, remaining)})"
                )
            else:
                lib_logger.warning(
                    f"Attempted to release credential {mask_credential(key)} for model {model}, but it was not in use."
                )

        # Notify all tasks waiting on this key's condition
        async with state["condition"]:
            state["condition"].notify_all()
