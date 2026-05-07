# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Manages custom provider API bases"""

import os
from typing import Dict

from proxy_app.settings_dialogs.advanced_settings import AdvancedSettings


class CustomProviderManager:
    """Manages custom provider API bases"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_providers(self) -> Dict[str, str]:
        """Get currently configured custom providers"""
        from proxy_app.provider_urls import PROVIDER_URL_MAP

        providers = {}
        for key, value in os.environ.items():
            if key.endswith("_API_BASE"):
                provider = key.replace("_API_BASE", "").lower()
                # Only include if NOT in hardcoded map
                if provider not in PROVIDER_URL_MAP:
                    providers[provider] = value
        return providers

    def add_provider(self, name: str, api_base: str):
        """Add PROVIDER_API_BASE"""
        key = f"{name.upper()}_API_BASE"
        self.settings.set(key, api_base)

    def edit_provider(self, name: str, api_base: str):
        """Edit PROVIDER_API_BASE"""
        self.add_provider(name, api_base)

    def remove_provider(self, name: str):
        """Remove PROVIDER_API_BASE"""
        key = f"{name.upper()}_API_BASE"
        self.settings.remove(key)
