# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Manages PROVIDER_MODELS"""

import json
import logging
import os
from typing import Dict, Any, Optional, List

import orjson

from proxy_app.settings_dialogs.advanced_settings import AdvancedSettings

logger = logging.getLogger(__name__)


class ModelDefinitionManager:
    """Manages PROVIDER_MODELS"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_provider_models(self, provider: str) -> Optional[Dict | List]:
        """Get currently configured models for a provider"""
        key = f"{provider.upper()}_MODELS"
        value = os.getenv(key)
        if value:
            try:
                return orjson.loads(value)
            except (json.JSONDecodeError, ValueError):
                logger.debug("get_current_provider_models: invalid JSON for provider %s", provider, exc_info=True)
                return None
        return None

    def get_all_providers_with_models(self) -> Dict[str, int]:
        """Get all providers with model definitions"""
        providers = {}
        for key, value in os.environ.items():
            if key.endswith("_MODELS"):
                provider = key.replace("_MODELS", "").lower()
                try:
                    parsed = orjson.loads(value)
                    if isinstance(parsed, dict):
                        providers[provider] = len(parsed)
                    elif isinstance(parsed, list):
                        providers[provider] = len(parsed)
                except (json.JSONDecodeError, ValueError):
                    logger.debug("detect_provider_counts: invalid JSON for provider %s", provider)
        return providers

    def set_models(self, provider: str, models: Dict[str, Dict[str, Any]]):
        """Set PROVIDER_MODELS"""
        key = f"{provider.upper()}_MODELS"
        value = orjson.dumps(models).decode()
        self.settings.set(key, value)

    def remove_models(self, provider: str):
        """Remove PROVIDER_MODELS"""
        key = f"{provider.upper()}_MODELS"
        self.settings.remove(key)
