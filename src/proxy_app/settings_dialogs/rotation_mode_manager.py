# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Manages ROTATION_MODE_PROVIDER settings for sequential/balanced credential rotation"""

import logging
import os
from typing import Dict

from proxy_app.settings_dialogs.advanced_settings import AdvancedSettings

logger = logging.getLogger(__name__)


class RotationModeManager:
    """Manages ROTATION_MODE_PROVIDER settings for sequential/balanced credential rotation"""

    VALID_MODES = ["balanced", "sequential"]

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_modes(self) -> Dict[str, str]:
        """Get currently configured rotation modes"""
        modes = {}
        for key, value in os.environ.items():
            if key.startswith("ROTATION_MODE_"):
                provider = key.replace("ROTATION_MODE_", "").lower()
                if value.lower() in self.VALID_MODES:
                    modes[provider] = value.lower()
        return modes

    def get_default_mode(self, provider: str) -> str:
        """Get the default rotation mode for a provider"""
        try:
            from rotator_library.providers import PROVIDER_PLUGINS

            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if provider_class and hasattr(provider_class, "default_rotation_mode"):
                return provider_class.default_rotation_mode
            return "balanced"
        except ImportError:
            # Fallback defaults if import fails
            if provider.lower() == "antigravity":
                return "sequential"
            return "balanced"

    def get_effective_mode(self, provider: str) -> str:
        """Get the effective rotation mode (configured or default)"""
        configured = self.get_current_modes().get(provider.lower())
        if configured:
            return configured
        return self.get_default_mode(provider)

    def set_mode(self, provider: str, mode: str):
        """Set rotation mode for a provider"""
        if mode.lower() not in self.VALID_MODES:
            raise ValueError(
                f"Invalid rotation mode: {mode}. Must be one of {self.VALID_MODES}"
            )
        key = f"ROTATION_MODE_{provider.upper()}"
        self.settings.set(key, mode.lower())

    def remove_mode(self, provider: str):
        """Remove rotation mode (reset to provider default)"""
        key = f"ROTATION_MODE_{provider.upper()}"
        self.settings.remove(key)
