# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
import time
from typing import Optional, Dict, List, Any
from ..error_types import mask_credential


class UsageManagerCycleMixin:
    def _get_tier_key(self, provider: str, priority: int) -> str:
        """
        Get the tier key for cycle tracking based on cross_tier setting.

        Args:
            provider: Provider name
            priority: Credential priority level

        Returns:
            "__all_tiers__" if cross-tier enabled, else str(priority)
        """
        if self._is_fair_cycle_cross_tier(provider):
            return "__all_tiers__"
        return str(priority)

    def _get_tracking_key(self, credential: str, model: str, provider: str) -> str:
        """
        Get the key for exhaustion tracking based on tracking mode.

        Args:
            credential: Credential identifier
            model: Model name (with provider prefix)
            provider: Provider name

        Returns:
            Tracking key string (quota group name, model name, or "__credential__")
        """
        mode = self._get_fair_cycle_tracking_mode(provider)
        if mode == "credential":
            return "__credential__"
        # model_group mode: use quota group if exists, else model
        group = self._get_model_quota_group(credential, model)
        return group if group else model

    def _get_credential_priority(self, credential: str, provider: str) -> int:
        """
        Get the priority level for a credential.

        Args:
            credential: Credential identifier
            provider: Provider name

        Returns:
            Priority level (default 999 if unknown)
        """
        plugin_instance = self._get_provider_instance(provider)
        if plugin_instance and hasattr(plugin_instance, "get_credential_priority"):
            priority = plugin_instance.get_credential_priority(credential)
            if priority is not None:
                return priority
        return 999

    def _get_cycle_data(
        self, provider: str, tier_key: str, tracking_key: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cycle data for a provider/tier/tracking key combination.

        Returns:
            Cycle data dict or None if not exists
        """
        return (
            self._cycle_exhausted.get(provider, {}).get(tier_key, {}).get(tracking_key)
        )

    def _ensure_cycle_structure(
        self, provider: str, tier_key: str, tracking_key: str
    ) -> Dict[str, Any]:
        """
        Ensure the nested cycle structure exists and return the cycle data dict.
        Uses setdefault for atomic check-and-create (avoids check-then-act race).
        """
        provider_dict = self._cycle_exhausted.setdefault(provider, {})
        tier_dict = provider_dict.setdefault(tier_key, {})
        cycle = tier_dict.setdefault(tracking_key, {
            "cycle_started_at": None,
            "exhausted": set(),
        })
        return cycle

    def _mark_credential_exhausted(
        self,
        credential: str,
        provider: str,
        tier_key: str,
        tracking_key: str,
    ) -> None:
        """
        Mark a credential as exhausted for fair cycle tracking.

        Starts the cycle timer on first exhaustion.
        Skips if credential is already in the exhausted set (prevents duplicate logging).
        """
        cycle_data = self._ensure_cycle_structure(provider, tier_key, tracking_key)

        # Skip if already exhausted in this cycle (prevents duplicate logging)
        if credential in cycle_data.get("exhausted", set()):
            return

        # Start cycle timer on first exhaustion
        if cycle_data["cycle_started_at"] is None:
            cycle_data["cycle_started_at"] = time.time()
            lib_logger.info(
                f"Fair cycle started for {provider} tier={tier_key} tracking='{tracking_key}'"
            )

        cycle_data["exhausted"].add(credential)
        lib_logger.info(
            f"Fair cycle: marked {mask_credential(credential)} exhausted "
            f"for {tracking_key} ({len(cycle_data['exhausted'])} total)"
        )

    def _is_credential_exhausted_in_cycle(
        self,
        credential: str,
        provider: str,
        tier_key: str,
        tracking_key: str,
    ) -> bool:
        """
        Check if a credential was exhausted in the current cycle.
        """
        cycle_data = self._get_cycle_data(provider, tier_key, tracking_key)
        if cycle_data is None:
            return False
        return credential in cycle_data.get("exhausted", set())

    def _is_cycle_expired(
        self, provider: str, tier_key: str, tracking_key: str
    ) -> bool:
        """
        Check if the current cycle has exceeded its duration.
        """
        cycle_data = self._get_cycle_data(provider, tier_key, tracking_key)
        if cycle_data is None:
            return False
        cycle_started = cycle_data.get("cycle_started_at")
        if cycle_started is None:
            return False
        duration = self._get_fair_cycle_duration(provider)
        return time.time() >= cycle_started + duration

    def _should_reset_cycle(
        self,
        provider: str,
        tier_key: str,
        tracking_key: str,
        all_credentials_in_tier: List[str],
        available_not_on_cooldown: Optional[List[str]] = None,
    ) -> bool:
        """
        Check if cycle should reset.

        Returns True if:
        1. Cycle duration has expired, OR
        2. No credentials remain available (after cooldown + fair cycle exclusion), OR
        3. All credentials in the tier have been marked exhausted (fallback)
        """
        # Check duration first
        if self._is_cycle_expired(provider, tier_key, tracking_key):
            return True

        cycle_data = self._get_cycle_data(provider, tier_key, tracking_key)
        if cycle_data is None:
            return False

        # If available credentials are provided, reset when none remain usable
        if available_not_on_cooldown is not None:
            has_available = any(
                not self._is_credential_exhausted_in_cycle(
                    cred, provider, tier_key, tracking_key
                )
                for cred in available_not_on_cooldown
            )
            if not has_available and len(all_credentials_in_tier) > 0:
                return True

        exhausted = cycle_data.get("exhausted", set())
        # All must be exhausted (and there must be at least one credential)
        return (
            len(exhausted) >= len(all_credentials_in_tier)
            and len(all_credentials_in_tier) > 0
        )

    def _reset_cycle(self, provider: str, tier_key: str, tracking_key: str) -> None:
        """
        Reset exhaustion tracking for a completed cycle.
        """
        cycle_data = self._get_cycle_data(provider, tier_key, tracking_key)
        if cycle_data:
            exhausted_count = len(cycle_data.get("exhausted", set()))
            lib_logger.info(
                f"Fair cycle complete for {provider} tier={tier_key} "
                f"tracking='{tracking_key}' - resetting ({exhausted_count} credentials cycled)"
            )
            cycle_data["cycle_started_at"] = None
            cycle_data["exhausted"] = set()

    def _get_all_credentials_for_tier_key(
        self,
        provider: str,
        tier_key: str,
        available_keys: List[str],
        credential_priorities: Optional[Dict[str, int]],
    ) -> List[str]:
        """
        Get all credentials that belong to a tier key.

        Args:
            provider: Provider name
            tier_key: Either "__all_tiers__" or str(priority)
            available_keys: List of available credential identifiers
            credential_priorities: Dict mapping credentials to priorities

        Returns:
            List of credentials belonging to this tier key
        """
        if tier_key == "__all_tiers__":
            # Cross-tier: all credentials for this provider
            return list(available_keys)
        else:
            # Within-tier: only credentials with matching priority
            priority = int(tier_key)
            if credential_priorities:
                return [
                    k
                    for k in available_keys
                    if credential_priorities.get(k, 999) == priority
                ]
            return list(available_keys)

    def _count_fair_cycle_excluded(
        self,
        provider: str,
        tier_key: str,
        tracking_key: str,
        candidates: List[str],
    ) -> int:
        """
        Count how many candidates are excluded by fair cycle.

        Args:
            provider: Provider name
            tier_key: Tier key for tracking
            tracking_key: Model/group tracking key
            candidates: List of candidate credentials (not on cooldown)

        Returns:
            Number of candidates excluded by fair cycle
        """
        count = 0
        for cred in candidates:
            if self._is_credential_exhausted_in_cycle(
                cred, provider, tier_key, tracking_key
            ):
                count += 1
        return count

    def _get_priority_multiplier(
        self, provider: str, priority: int, rotation_mode: str
    ) -> int:
        """
        Get the concurrency multiplier for a provider/priority/mode combination.

        Lookup order:
        1. Mode-specific tier override: priority_multipliers_by_mode[provider][mode][priority]
        2. Universal tier multiplier: priority_multipliers[provider][priority]
        3. Sequential fallback (if mode is sequential): sequential_fallback_multipliers[provider]
        4. Global default: 1 (no multiplier effect)

        Args:
            provider: Provider name (e.g., "antigravity")
            priority: Priority level (1 = highest priority)
            rotation_mode: Current rotation mode ("sequential" or "balanced")

        Returns:
            Multiplier value
        """
        provider_lower = provider.lower()

        # 1. Check mode-specific override
        if provider_lower in self.priority_multipliers_by_mode:
            mode_multipliers = self.priority_multipliers_by_mode[provider_lower]
            if rotation_mode in mode_multipliers:
                if priority in mode_multipliers[rotation_mode]:
                    return mode_multipliers[rotation_mode][priority]

        # 2. Check universal tier multiplier
        if provider_lower in self.priority_multipliers:
            if priority in self.priority_multipliers[provider_lower]:
                return self.priority_multipliers[provider_lower][priority]

        # 3. Sequential fallback (only for sequential mode)
        if rotation_mode == "sequential":
            if provider_lower in self.sequential_fallback_multipliers:
                return self.sequential_fallback_multipliers[provider_lower]

        # 4. Global default
        return 1

