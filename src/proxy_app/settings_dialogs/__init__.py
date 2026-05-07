# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Settings dialogs package — decomposed manager classes for the settings tool."""

from proxy_app.settings_dialogs._common import _NOT_FOUND, console
from proxy_app.settings_dialogs.advanced_settings import AdvancedSettings
from proxy_app.settings_dialogs.custom_provider_manager import CustomProviderManager
from proxy_app.settings_dialogs.model_definition_manager import ModelDefinitionManager
from proxy_app.settings_dialogs.concurrency_manager import ConcurrencyManager
from proxy_app.settings_dialogs.rotation_mode_manager import RotationModeManager
from proxy_app.settings_dialogs.priority_multiplier_manager import PriorityMultiplierManager
from proxy_app.settings_dialogs.provider_settings_manager import ProviderSettingsManager

__all__ = [
    "_NOT_FOUND",
    "console",
    "AdvancedSettings",
    "CustomProviderManager",
    "ModelDefinitionManager",
    "ConcurrencyManager",
    "RotationModeManager",
    "PriorityMultiplierManager",
    "ProviderSettingsManager",
]
