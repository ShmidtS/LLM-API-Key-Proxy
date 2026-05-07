# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Manages MAX_CONCURRENT_REQUESTS_PER_KEY_PROVIDER"""

import json
import logging
import os
from typing import Dict

from proxy_app.settings_dialogs.advanced_settings import AdvancedSettings

logger = logging.getLogger(__name__)


class ConcurrencyManager:
    """Manages MAX_CONCURRENT_REQUESTS_PER_KEY_PROVIDER"""

    def __init__(self, settings: AdvancedSettings):
        self.settings = settings

    def get_current_limits(self) -> Dict[str, int]:
        """Get currently configured concurrency limits"""
        limits = {}
        for key, value in os.environ.items():
            if key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
                provider = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
                try:
                    limits[provider] = int(value)
                except (json.JSONDecodeError, ValueError):
                    logger.debug("detect_concurrency_limits: invalid value for %s", provider)
        return limits

    def set_limit(self, provider: str, limit: int):
        """Set concurrency limit"""
        key = f"MAX_CONCURRENT_REQUESTS_PER_KEY_{provider.upper()}"
        self.settings.set(key, str(limit))

    def remove_limit(self, provider: str):
        """Remove concurrency limit (reset to default)"""
        key = f"MAX_CONCURRENT_REQUESTS_PER_KEY_{provider.upper()}"
        self.settings.remove(key)
