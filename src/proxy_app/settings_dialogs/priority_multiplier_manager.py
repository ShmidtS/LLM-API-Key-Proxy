# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Manages CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N> settings"""

import logging
import os
from typing import Dict

from proxy_app.settings_dialogs.advanced_settings import AdvancedSettings

logger = logging.getLogger(__name__)


class PriorityMultiplierManager:
    """Manages CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N> settings"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_provider_defaults(self, provider: str) -> Dict[int, int]:
        """Get default priority multipliers from provider class"""
        try:
            from rotator_library.providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if provider_class and hasattr(
                provider_class, "default_priority_multipliers"
            ):
                return dict(provider_class.default_priority_multipliers)
        except ImportError:
            logger.debug("get_default_priority_multipliers: provider %s not importable", provider)
        return {}

    def get_sequential_fallback(self, provider: str) -> int:
        """Get sequential fallback multiplier from provider class"""
        try:
            from rotator_library.providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if provider_class and hasattr(
                provider_class, "default_sequential_fallback_multiplier"
            ):
                return provider_class.default_sequential_fallback_multiplier
        except ImportError:
            logger.debug("get_sequential_fallback: provider %s not importable", provider)
        return 1

    def get_current_multipliers(self) -> Dict[str, Dict[int, int]]:
        """Get currently configured priority multipliers from env vars"""
        multipliers: Dict[str, Dict[int, int]] = {}
        for key, value in os.environ.items():
            if key.startswith("CONCURRENCY_MULTIPLIER_") and "_PRIORITY_" in key:
                try:
                    # Parse: CONCURRENCY_MULTIPLIER_<PROVIDER>_PRIORITY_<N>
                    parts = key.split("_PRIORITY_")
                    provider = parts[0].replace("CONCURRENCY_MULTIPLIER_", "").lower()
                    remainder = parts[1]

                    # Check if mode-specific (has _SEQUENTIAL or _BALANCED suffix)
                    if "_" in remainder:
                        continue  # Skip mode-specific for now (show in separate view)

                    priority = int(remainder)
                    multiplier = int(value)

                    if provider not in multipliers:
                        multipliers[provider] = {}
                    multipliers[provider][priority] = multiplier
                except (ValueError, IndexError):
                    logger.debug("get_current_multipliers: failed to parse multiplier env var for %s", provider)
        return multipliers

    def get_effective_multiplier(self, provider: str, priority: int) -> int:
        """Get effective multiplier (configured, provider default, or 1)"""
        # Check env var override
        current = self.get_current_multipliers()
        if provider.lower() in current:
            if priority in current[provider.lower()]:
                return current[provider.lower()][priority]

        # Check provider defaults
        defaults = self.get_provider_defaults(provider)
        if priority in defaults:
            return defaults[priority]

        # Return 1 (no multiplier)
        return 1

    def set_multiplier(self, provider: str, priority: int, multiplier: int):
        """Set priority multiplier for a provider"""
        if multiplier < 1:
            raise ValueError("Multiplier must be >= 1")
        key = f"CONCURRENCY_MULTIPLIER_{provider.upper()}_PRIORITY_{priority}"
        self.settings.set(key, str(multiplier))

    def remove_multiplier(self, provider: str, priority: int):
        """Remove multiplier (reset to provider default)"""
        key = f"CONCURRENCY_MULTIPLIER_{provider.upper()}_PRIORITY_{priority}"
        self.settings.remove(key)
