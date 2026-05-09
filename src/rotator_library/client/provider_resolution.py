# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Provider and credential resolution helpers for RotatingClient."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..providers.openai_compatible_provider import OpenAICompatibleProvider
from ..utils.model_utils import extract_provider_from_model
from ..utils.model_utils import get_or_create_provider_instance

lib_logger = logging.getLogger("rotator_library")

_PROVIDER_METHOD_CACHE_MISS = object()
_PROVIDER_METHOD_NO_PROVIDER = object()
_PROVIDER_METHOD_NO_METHOD = object()


class ProviderResolutionError(ValueError):
    """Raised when provider or credential resolution fails."""


def filter_configured_credentials(
    api_keys: Optional[dict[str, list[str]]],
    oauth_credentials: Optional[dict[str, list[str]]],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Drop providers that have no configured credentials."""
    filtered_api_keys = {
        provider: keys for provider, keys in (api_keys or {}).items() if keys
    }
    filtered_oauth_credentials = {
        provider: paths
        for provider, paths in (oauth_credentials or {}).items()
        if paths
    }
    return filtered_api_keys, filtered_oauth_credentials


def combine_provider_credentials(
    api_keys: dict[str, list[str]],
    oauth_credentials: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Combine API keys and OAuth account paths by provider."""
    all_credentials: dict[str, list[str]] = {}
    for provider, keys in api_keys.items():
        all_credentials.setdefault(provider, []).extend(keys)
    for provider, paths in oauth_credentials.items():
        all_credentials.setdefault(provider, []).extend(paths)
    return all_credentials


def build_credential_to_provider_map(all_credentials: dict[str, list[str]]) -> dict[str, str]:
    """Build a reverse mapping from credential identifier to provider name."""
    mapping: dict[str, str] = {}
    for provider, creds in all_credentials.items():
        for cred in creds:
            mapping[cred] = provider
    return mapping


class ProviderResolver:
    """Resolve providers, provider instances, and credentials for requests."""

    def __init__(
        self,
        provider_config: Any,
        all_credentials: dict[str, list[str]],
        oauth_providers: set[str],
        provider_plugins: dict[str, Any],
        provider_instances: Any,
        provider_method_cache: dict[tuple[str, str], Any] | None = None,
    ):
        self.provider_config = provider_config
        self.all_credentials = all_credentials
        self.oauth_providers = oauth_providers
        self._provider_plugins = provider_plugins
        self._provider_instances = provider_instances
        self._provider_method_cache = provider_method_cache if provider_method_cache is not None else {}

    @property
    def provider_plugins(self):
        return self._provider_plugins

    @property
    def provider_instances(self):
        return self._provider_instances

    def provider_from_model(self, model: str) -> str:
        provider = extract_provider_from_model(model)
        if not provider:
            raise ProviderResolutionError("'model' must be in 'provider/model' format.")
        return provider

    def credentials_for_provider(self, provider: str) -> list[str]:
        credentials = self.all_credentials.get(provider)
        if not credentials:
            raise ProviderResolutionError(
                f"No API keys or OAuth credentials configured for provider: {provider}"
            )
        return list(credentials)

    def credential_key_for_provider(self, provider_name: str) -> str:
        if provider_name.endswith("_oauth"):
            base_name = provider_name[:-6]
            if base_name in self.oauth_providers:
                return base_name
        return provider_name

    def has_credentials(self, provider_name: str) -> bool:
        return self.credential_key_for_provider(provider_name) in self.all_credentials

    def is_custom_openai_compatible_provider(self, provider_name: str) -> bool:
        """
        Checks if a provider is a custom OpenAI-compatible provider.

        Custom providers are identified by:
        1. Having a _API_BASE environment variable set, AND
        2. NOT being in the list of known LiteLLM providers
        """
        return self.provider_config.is_custom_provider(provider_name)

    def build_credential_to_provider_map(self) -> dict[str, str]:
        """Build a reverse mapping from credential identifier to provider name."""
        mapping: dict[str, str] = {}
        for provider, creds in self.all_credentials.items():
            for cred in creds:
                mapping[cred] = provider
        return mapping

    def get_provider_method(
        self, provider_name: str, method_name: str, required: bool = False
    ):
        cache_key = (provider_name, method_name)
        cached = self._provider_method_cache.get(cache_key, _PROVIDER_METHOD_CACHE_MISS)
        if cached is _PROVIDER_METHOD_NO_PROVIDER:
            if required:
                raise ValueError(f"No provider instance for '{provider_name}'")
            return None
        if cached is _PROVIDER_METHOD_NO_METHOD:
            if required:
                raise AttributeError(
                    f"Provider '{provider_name}' has no method '{method_name}'"
                )
            return None
        if cached is not _PROVIDER_METHOD_CACHE_MISS:
            # Dynamically resolve on current provider instance to avoid stale references
            provider_instance = self.get_provider_instance(provider_name)
            if provider_instance is None:
                if required:
                    raise ValueError(f"No provider instance for '{provider_name}'")
                return None
            method = getattr(provider_instance, method_name, None)
            if method is None and required:
                raise AttributeError(
                    f"Provider '{provider_name}' has no method '{method_name}'"
                )
            return method

        provider_instance = self.get_provider_instance(provider_name)
        cache_optional_missing = not required and method_name == "get_model_options"
        if provider_instance is None:
            if required:
                raise ValueError(f"No provider instance for '{provider_name}'")
            if cache_optional_missing:
                self._provider_method_cache[cache_key] = _PROVIDER_METHOD_NO_PROVIDER
            return None

        method = getattr(provider_instance, method_name, None)
        if method is None:
            if required:
                raise AttributeError(
                    f"Provider '{provider_name}' has no method '{method_name}'"
                )
            if cache_optional_missing:
                self._provider_method_cache[cache_key] = _PROVIDER_METHOD_NO_METHOD
            return None

        # Cache sentinel to avoid repeated introspection; resolve dynamically on hit
        self._provider_method_cache[cache_key] = True
        return method

    def get_provider_instance(self, provider_name: str):
        """
        Lazily initializes and returns a provider instance.
        Only initializes providers that have configured credentials.

        Args:
            provider_name: The name of the provider to get an instance for.
                          For OAuth providers, this may include "_oauth" suffix
                          (e.g., "antigravity_oauth"), but credentials are stored
                          under the base name (e.g., "antigravity").

        Returns:
            Provider instance if credentials exist, None otherwise.
        """
        # For OAuth providers, credentials are stored under base name (without _oauth suffix)
        # e.g., "antigravity_oauth" plugin -> credentials under "antigravity"
        credential_key = provider_name
        if provider_name.endswith("_oauth"):
            base_name = provider_name[:-6]  # Remove "_oauth"
            if base_name in self.oauth_providers:
                credential_key = base_name

        # Only initialize providers for which we have credentials
        if credential_key not in self.all_credentials:
            lib_logger.debug(
                "Skipping provider '%s' initialization: no credentials configured",
                provider_name,
            )
            return None

        # Try shared lazy-load path first
        result = get_or_create_provider_instance(
            provider_name, self._provider_plugins, self._provider_instances
        )
        if result is not None:
            return result

        # Client-specific fallback: custom OpenAI-compatible providers
        if self.is_custom_openai_compatible_provider(provider_name):
            try:
                instance = OpenAICompatibleProvider(provider_name)
                self._provider_instances.register(provider_name, instance)
                return instance
            except ValueError:
                return None

        # Fallback: known providers with api_base but no plugin get OpenAICompatibleProvider
        # This fixes providers like openrouter, xai, openai, moonshot that have keys
        # and api_base but no dedicated provider plugin class
        api_base = self.provider_config.api_bases.get(provider_name)
        if api_base:
            try:
                instance = OpenAICompatibleProvider(provider_name)
                self._provider_instances.register(provider_name, instance)
                return instance
            except ValueError:
                return None

        # Check if already registered (e.g. by usage_manager)
        return self._provider_instances.get(provider_name)
