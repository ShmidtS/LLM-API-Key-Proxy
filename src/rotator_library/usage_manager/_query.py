# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger, MAX_CACHE_ENTRIES
import time
from collections import OrderedDict
from typing import Optional, Dict, List, Any
from ..utils.model_utils import get_or_create_provider_instance


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
        import re

        # Pattern: env:// URI format (e.g., "env://antigravity/1" -> "antigravity")
        if credential.startswith("env://"):
            parts = credential[6:].split("/")  # Remove "env://" prefix
            if parts and parts[0]:
                return parts[0].lower()
            # Malformed env:// URI (empty provider name)
            lib_logger.warning(f"Malformed env:// credential URI: {credential}")
            return None

        # Normalize path separators
        normalized = credential.replace("\\", "/")

        # Pattern: path ending with {provider}_oauth_{number}.json
        match = re.search(r"/([a-z_]+)_oauth_\d+\.json$", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Pattern: oauth_creds/{provider}_...
        match = re.search(r"oauth_creds/([a-z_]+)_", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Pattern: filename only {provider}_oauth_{number}.json (no path)
        match = re.match(r"([a-z_]+)_oauth_\d+\.json$", normalized, re.IGNORECASE)
        if match:
            return match.group(1).lower()

        # Pattern: API key prefixes for specific providers
        # These are raw API keys with recognizable prefixes
        api_key_prefixes = {
            "sk-nano-": "nanogpt",
            "sk-or-": "openrouter",
            "sk-ant-": "anthropic",
        }
        for prefix, provider in api_key_prefixes.items():
            if credential.startswith(prefix):
                return provider

        # Lookup from credential-to-provider mapping (built at init from all_credentials)
        if self.credential_to_provider and credential in self.credential_to_provider:
            return self.credential_to_provider[credential]

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
                    return provider

            # Fallback to "daily" section (legacy structure)
            daily_data = cred_data.get("daily", {})
            daily_models = daily_data.get("models", {})
            if daily_models:
                # Get first model name and extract provider prefix
                first_model = next(iter(daily_models.keys()), None)
                if first_model and "/" in first_model:
                    provider = first_model.split("/")[0].lower()
                    return provider

        return None

    def _get_provider_instance(self, provider: str) -> Optional[Any]:
        """
        Get or create a provider plugin instance.

        Args:
            provider: The provider name

        Returns:
            Provider plugin instance or None
        """
        return get_or_create_provider_instance(
            provider, self.provider_plugins, self._provider_instances
        )

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
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_usage_reset_config"):
            return plugin_instance.get_usage_reset_config(credential)

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
        plugin_instance = self._get_provider_instance(provider)

        result = None
        if plugin_instance and hasattr(plugin_instance, "get_model_quota_group"):
            result = plugin_instance.get_model_quota_group(model)

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
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_models_in_quota_group"):
            models = plugin_instance.get_models_in_quota_group(group)

            # Normalize and deduplicate
            if hasattr(plugin_instance, "normalize_model_for_tracking"):
                seen = set()
                normalized = []
                for m in models:
                    prefixed = f"{provider}/{m}"
                    norm = plugin_instance.normalize_model_for_tracking(prefixed)
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
        Evict oldest 20% of cache entries if total exceeds MAX_CACHE_ENTRIES.

        Called after each cache population to prevent unbounded growth.
        Uses LRU eviction: oldest entries (at the front of OrderedDict) are removed.
        """
        total_entries = sum(len(v) for v in self._quota_group_cache.values())
        total_entries += sum(len(v) for v in self._grouped_models_cache.values())

        if total_entries <= MAX_CACHE_ENTRIES:
            return

        # Evict 20% from each cache
        evict_ratio = 0.2

        for cache in (self._quota_group_cache, self._grouped_models_cache):
            entries_to_evict = int(len(cache) * evict_ratio)
            if entries_to_evict > 0:
                for _ in range(entries_to_evict):
                    if cache:
                        cache.popitem(last=False)

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
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_model_usage_weight"):
            return plugin_instance.get_model_usage_weight(model)

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
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "normalize_model_for_tracking"):
            return plugin_instance.normalize_model_for_tracking(model)

        return model

    def _check_quota_group_cooldown(
        self,
        credential: str,
        model_cooldowns: Dict[str, float],
        normalized_model: str,
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
        provider = self._get_provider_from_credential(credential)
        if not provider:
            return 0
        plugin = self._get_provider_instance(provider)
        if not plugin or not hasattr(plugin, "get_model_quota_group"):
            return 0
        group = plugin.get_model_quota_group(normalized_model)
        if not group:
            return 0
        virtual_name = f"{provider}/_quota"
        return model_cooldowns.get(virtual_name) or 0

    # Providers where request_count should be used for credential selection
    # instead of success_count (because failed requests also consume quota)
    _REQUEST_COUNT_PROVIDERS = {"antigravity", "gemini_cli", "chutes", "nanogpt", "zai"}

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

        key_data = self._usage_data.get(key, {})
        model_data = key_data.get("models", {}).get(model, {})

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
        plugin_instance = self._get_provider_instance(provider)

        if plugin_instance and hasattr(plugin_instance, "get_default_usage_field_name"):
            return plugin_instance.get_default_usage_field_name()

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

        key_data = self._usage_data.get(key, {})
        reset_mode = self._get_reset_mode(key)

        if reset_mode == "per_model":
            # New per-model structure: key_data["models"][model][field]
            return key_data.get("models", {}).get(model, {}).get(field, 0)
        else:
            # Legacy structure: key_data["daily"]["models"][model][field]
            return (
                key_data.get("daily", {}).get("models", {}).get(model, {}).get(field, 0)
            )

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

        async with self._data_lock.read():
            for key in credentials:
                key_data = self._usage_data.get(key, {})

                # Skip if key-level cooldown is active
                if (key_data.get("key_cooldown_until") or 0) > now:
                    continue

                # Normalize model name for consistent cooldown lookup
                # (cooldowns are stored under normalized names by record_failure)
                # For providers without normalize_model_for_tracking (non-Antigravity),
                # this returns the model unchanged, so cooldown lookups work as before.
                normalized_model = self._normalize_model_for_tracking(key, model)

                # Skip if model-specific cooldown is active
                model_cooldowns = key_data.get("model_cooldowns", {})
                model_cd = model_cooldowns.get(normalized_model) or 0
                if model_cd == 0:
                    model_cd = self._check_quota_group_cooldown(
                        key, model_cooldowns, normalized_model
                    )
                if model_cd > now:
                    continue

                # Skip if quota confirmed exhausted via background refresh
                baseline = self._get_baseline_remaining(key, normalized_model)
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
        async with self._data_lock.read():
            for key in credentials:
                key_data = self._usage_data.get(key, {})
                model_cooldowns = key_data.get("model_cooldowns", {})

                # Check if key-level or model-level cooldown is active
                normalized_model = self._normalize_model_for_tracking(key, model)
                model_cd = model_cooldowns.get(normalized_model) or 0
                if model_cd == 0:
                    model_cd = self._check_quota_group_cooldown(
                        key, model_cooldowns, normalized_model
                    )
                is_on_cooldown = (
                    (key_data.get("key_cooldown_until") or 0) > now
                    or model_cd > now
                )

                if is_on_cooldown:
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

        async with self._data_lock.read():
            for key in credentials:
                key_data = self._usage_data.get(key, {})
                normalized_model = self._normalize_model_for_tracking(key, model)

                # Check key-level cooldown
                key_cooldown = key_data.get("key_cooldown_until") or 0
                if key_cooldown > now:
                    if soonest_end is None or key_cooldown < soonest_end:
                        soonest_end = key_cooldown

                # Check model-level cooldown
                model_cooldowns_map = key_data.get("model_cooldowns", {})
                model_cooldown = model_cooldowns_map.get(normalized_model) or 0
                if model_cooldown == 0:
                    model_cooldown = self._check_quota_group_cooldown(
                        key, model_cooldowns_map, normalized_model
                    )

                if model_cooldown > now:
                    if soonest_end is None or model_cooldown < soonest_end:
                        soonest_end = model_cooldown

        return soonest_end

    # =========================================================================
    # TIMESTAMP FORMATTING HELPERS
    # =========================================================================

