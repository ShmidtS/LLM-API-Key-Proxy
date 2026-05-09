# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Settings detection utilities for the launcher TUI."""

import os
import re
import json
import logging

logger = logging.getLogger(__name__)
from pathlib import Path

import orjson
from rotator_library.utils.paths import get_data_file


class SettingsDetector:
    """Detects settings from .env for display"""

    @staticmethod
    def _load_local_env() -> dict[str, str]:
        """Load environment variables from local .env file only"""
        env_file = get_data_file(".env")
        env_dict: dict[str, str] = {}
        if not env_file.exists():
            return env_dict
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip()
                        if value and value[0] in ('"', "'") and value[-1] == value[0]:
                            value = value[1:-1]
                        env_dict[key] = value
        except (IOError, OSError):
            logger.debug("load_env_dict: failed to read .env file")
        return env_dict

    @staticmethod
    def get_all_settings() -> dict:
        """Returns comprehensive settings overview (includes provider_settings which triggers heavy imports)"""
        return {
            "credentials": SettingsDetector.detect_credentials(),
            "custom_bases": SettingsDetector.detect_custom_api_bases(),
            "model_definitions": SettingsDetector.detect_model_definitions(),
            "concurrency_limits": SettingsDetector.detect_concurrency_limits(),
            "model_filters": SettingsDetector.detect_model_filters(),
            "provider_settings": SettingsDetector.detect_provider_settings(),
        }

    @staticmethod
    def get_basic_settings() -> dict:
        """Returns basic settings overview without provider_settings (avoids heavy imports)"""
        return {
            "credentials": SettingsDetector.detect_credentials(),
            "custom_bases": SettingsDetector.detect_custom_api_bases(),
            "model_definitions": SettingsDetector.detect_model_definitions(),
            "concurrency_limits": SettingsDetector.detect_concurrency_limits(),
            "model_filters": SettingsDetector.detect_model_filters(),
        }

    @staticmethod
    def detect_credentials() -> dict:
        """Detect API keys and OAuth credentials"""
        import re
        from pathlib import Path

        providers = {}

        # Scan for API keys
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if "_API_KEY" in key and key != "PROXY_API_KEY":
                provider = key.split("_API_KEY")[0].lower()
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["api_keys"] += 1

        # Scan for file-based OAuth credentials
        oauth_dir = Path("oauth_creds")
        if oauth_dir.exists():
            for file in oauth_dir.glob("*_oauth_*.json"):
                provider = file.name.split("_oauth_")[0]
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["oauth"] += 1

        # Scan for env-based OAuth credentials
        # Maps provider name to the ENV_PREFIX used by the provider
        # (duplicated from credential_manager to avoid heavy imports)
        env_oauth_providers = {
            "gemini_cli": "GEMINI_CLI",
            "antigravity": "ANTIGRAVITY",
            "qwen_code": "QWEN_CODE",
            "iflow": "IFLOW",
        }

        for provider, env_prefix in env_oauth_providers.items():
            oauth_count = 0

            # Check numbered credentials (PROVIDER_N_ACCESS_TOKEN pattern)
            numbered_pattern = re.compile(rf"^{env_prefix}_(\d+)_ACCESS_TOKEN$")
            for key in env_vars.keys():
                match = numbered_pattern.match(key)
                if match:
                    index = match.group(1)
                    refresh_key = f"{env_prefix}_{index}_REFRESH_TOKEN"
                    if refresh_key in env_vars and env_vars[refresh_key]:
                        oauth_count += 1

            # Check legacy single credential (if no numbered found)
            if oauth_count == 0:
                access_key = f"{env_prefix}_ACCESS_TOKEN"
                refresh_key = f"{env_prefix}_REFRESH_TOKEN"
                if env_vars.get(access_key) and env_vars.get(refresh_key):
                    oauth_count = 1

            if oauth_count > 0:
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["oauth"] += oauth_count

        # Mark custom providers (have API_BASE set)
        for provider in providers:
            if os.getenv(f"{provider.upper()}_API_BASE"):
                providers[provider]["custom"] = True

        return providers

    @staticmethod
    def detect_custom_api_bases() -> dict:
        """Detect custom API base URLs (not in hardcoded map)"""
        from proxy_app.provider_urls import PROVIDER_URL_MAP

        bases = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.endswith("_API_BASE"):
                provider = key.replace("_API_BASE", "").lower()
                # Only include if NOT in hardcoded map
                if provider not in PROVIDER_URL_MAP:
                    bases[provider] = value
        return bases

    @staticmethod
    def detect_model_definitions() -> dict:
        """Detect provider model definitions"""
        models = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.endswith("_MODELS"):
                provider = key.replace("_MODELS", "").lower()
                try:
                    parsed = orjson.loads(value)
                    if isinstance(parsed, dict):
                        models[provider] = len(parsed)
                    elif isinstance(parsed, list):
                        models[provider] = len(parsed)
                except (json.JSONDecodeError, ValueError):
                    logger.debug("detect_model_counts: invalid JSON for provider %s", provider)
        return models

    @staticmethod
    def detect_concurrency_limits() -> dict:
        """Detect max concurrent requests per key"""
        limits = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
                provider = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
                try:
                    limits[provider] = int(value)
                except ValueError:
                    logger.debug("detect_concurrency_limits: invalid value for %s", provider)
        return limits

    @staticmethod
    def detect_model_filters() -> dict:
        """Detect active model filters (basic info only: defined or not)"""
        filters = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.startswith("IGNORE_MODELS_") or key.startswith("WHITELIST_MODELS_"):
                filter_type = "ignore" if key.startswith("IGNORE") else "whitelist"
                provider = key.replace(f"{filter_type.upper()}_MODELS_", "").lower()
                if provider not in filters:
                    filters[provider] = {"has_ignore": False, "has_whitelist": False}
                if filter_type == "ignore":
                    filters[provider]["has_ignore"] = True
                else:
                    filters[provider]["has_whitelist"] = True
        return filters

    @staticmethod
    def detect_provider_settings() -> dict:
        """Detect provider-specific settings (Antigravity, Gemini CLI)"""
        try:
            from proxy_app._provider_settings import PROVIDER_SETTINGS_MAP
        except ImportError:
            # Fallback for direct execution or testing
            from ._provider_settings import PROVIDER_SETTINGS_MAP

        provider_settings = {}
        env_vars = SettingsDetector._load_local_env()

        for provider, definitions in PROVIDER_SETTINGS_MAP.items():
            modified_count = 0
            for key, definition in definitions.items():
                env_value = env_vars.get(key)
                if env_value is not None:
                    # Check if value differs from default
                    default = definition.get("default")
                    setting_type = definition.get("type", "str")

                    try:
                        if setting_type == "bool":
                            current = env_value.lower() in ("true", "1", "yes")
                        elif setting_type == "int":
                            current = int(env_value)
                        else:
                            current = env_value

                        if current != default:
                            modified_count += 1
                    except (ValueError, AttributeError):
                        logger.debug("detect_modified_settings: failed to parse setting for %s", provider)

            if modified_count > 0:
                provider_settings[provider] = modified_count

        return provider_settings
