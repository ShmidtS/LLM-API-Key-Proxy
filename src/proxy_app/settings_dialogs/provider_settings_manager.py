# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Manages provider-specific configuration settings"""

import os
from typing import Dict, Any, List

from proxy_app.settings_dialogs.advanced_settings import AdvancedSettings
from proxy_app.settings_dialogs._common import PROVIDER_SETTINGS_MAP


class ProviderSettingsManager:
    """Manages provider-specific configuration settings"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_available_providers(self) -> List[str]:
        """Get list of providers with specific settings available"""
        return list(PROVIDER_SETTINGS_MAP.keys())

    def get_provider_settings_definitions(
        self, provider: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get settings definitions for a provider"""
        return PROVIDER_SETTINGS_MAP.get(provider, {})

    def get_current_value(self, key: str, definition: Dict[str, Any]) -> Any:
        """Get current value of a setting from environment"""
        env_value = os.getenv(key)
        if env_value is None:
            return definition.get("default")

        setting_type = definition.get("type", "str")
        try:
            if setting_type == "bool":
                return env_value.lower() in ("true", "1", "yes")
            elif setting_type == "int":
                return int(env_value)
            else:
                return env_value
        except (ValueError, AttributeError):
            return definition.get("default")

    def get_all_current_values(self, provider: str) -> Dict[str, Any]:
        """Get all current values for a provider"""
        definitions = self.get_provider_settings_definitions(provider)
        values = {}
        for key, definition in definitions.items():
            values[key] = self.get_current_value(key, definition)
        return values

    def set_value(self, key: str, value: Any, definition: Dict[str, Any]):
        """Set a setting value, converting to string for .env storage"""
        setting_type = definition.get("type", "str")
        if setting_type == "bool":
            str_value = "true" if value else "false"
        else:
            str_value = str(value)
        self.settings.set(key, str_value)

    def reset_to_default(self, key: str):
        """Remove a setting to reset it to default"""
        self.settings.remove(key)

    def get_modified_settings(self, provider: str) -> Dict[str, Any]:
        """Get settings that differ from defaults"""
        definitions = self.get_provider_settings_definitions(provider)
        modified = {}
        for key, definition in definitions.items():
            current = self.get_current_value(key, definition)
            default = definition.get("default")
            if current != default:
                modified[key] = current
        return modified
