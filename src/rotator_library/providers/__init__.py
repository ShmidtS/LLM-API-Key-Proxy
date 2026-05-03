# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import importlib
import logging
import pkgutil
import os
from typing import Dict, Type, Union

from .provider_interface import ProviderInterface as _ProviderInterface

ProviderInterface = _ProviderInterface

lib_logger = logging.getLogger("rotator_library")

__all__ = [
    "ProviderInterface",
    "PROVIDER_PLUGINS",
    "PROVIDER_AUTH_MAP",
    "get_provider",
    "list_providers",
    "get_all_providers",
]

# --- Provider Plugin System ---

_LazyEntry = tuple[str, str]
_ProviderEntry = Union[Type[_ProviderInterface], _LazyEntry]


class _LazyProviderPlugins(dict):
    def __getitem__(self, key: str) -> _ProviderEntry:
        value = super().__getitem__(key)
        if isinstance(value, tuple):
            value = _resolve_lazy_entry(value)
            super().__setitem__(key, value)
        return value

    def get(self, key: str, default: _ProviderEntry | None = None) -> _ProviderEntry | None:
        if key not in self:
            return default
        return self[key]

    def items(self):
        for key in list(super().keys()):
            yield key, self[key]

    def values(self):
        for key in list(super().keys()):
            yield self[key]


# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, _ProviderEntry] = _LazyProviderPlugins()

# Compatibility registry for auth/credential tooling imports.
PROVIDER_AUTH_MAP: Dict[str, Union[_ProviderEntry, type]] = {
    "gemini_cli": (".gemini_auth_base", "GeminiAuthBase"),
    "qwen_code": (".qwen_auth_base", "QwenAuthBase"),
    "iflow": (".iflow_auth_base", "IFlowAuthBase"),
    "antigravity": (".antigravity_auth_base", "AntigravityAuthBase"),
    "colin": (".colin_provider", "ColinProvider"),
    "elysiver": (".elysiver_provider", "ElysiverProvider"),
}

_LAZY_IMPORTS = {
    "OpenAICompatibleProvider": (".openai_compatible_provider", "OpenAICompatibleProvider"),
    "ColinProvider": (".colin_provider", "ColinProvider"),
    "ElysiverProvider": (".elysiver_provider", "ElysiverProvider"),
}


# --- Pre-register providers with custom logic ---
# These providers implement has_custom_logic() = True and need early registration
# to bypass the standard litellm flow
PROVIDER_PLUGINS["colin"] = PROVIDER_AUTH_MAP["colin"]
PROVIDER_PLUGINS["elysiver"] = PROVIDER_AUTH_MAP["elysiver"]


# --- Lazy Provider Loading ---


def _resolve_lazy_entry(entry: _LazyEntry):
    module_path, class_name = entry
    module = importlib.import_module(module_path, __name__)
    return getattr(module, class_name)


def _resolve_auth_class(name: str):
    entry = PROVIDER_AUTH_MAP[name]
    if isinstance(entry, tuple):
        provider_class = _resolve_lazy_entry(entry)
        PROVIDER_AUTH_MAP[name] = provider_class
        if name in PROVIDER_PLUGINS and PROVIDER_PLUGINS[name] == entry:
            PROVIDER_PLUGINS[name] = provider_class
        return provider_class
    return entry


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        value = _resolve_lazy_entry(_LAZY_IMPORTS[name])
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_provider_module_name(provider_name: str) -> str:
    """Convert provider name to module name."""
    return f"{provider_name}_provider"


def _try_load_from_module(module_path: str, provider_name: str):
    """Try to load a ProviderInterface subclass from a module."""
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return None

    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if (
            isinstance(attribute, type)
            and issubclass(attribute, _ProviderInterface)
            and attribute is not _ProviderInterface
            and attribute.__module__ == module.__name__
        ):
            PROVIDER_PLUGINS[provider_name] = attribute
            lib_logger.debug(
                f"Lazy-loaded provider: {provider_name}"
            )
            return attribute

    return None


def _load_provider(provider_name: str):
    """
    Lazily load a single provider by name.
    Returns the provider class or None if not found.

    Tries two patterns:
    1. Package: {provider_name}/__init__.py  (e.g., antigravity/)
    2. Module: {provider_name}_provider.py   (e.g., openai_provider.py)
    """
    if provider_name in PROVIDER_PLUGINS:
        return PROVIDER_PLUGINS[provider_name]

    # Try package-based provider first (e.g., antigravity/)
    result = _try_load_from_module(f"{__name__}.{provider_name}", provider_name)
    if result:
        return result

    # Try module-based provider (e.g., openai_provider.py)
    module_name = _get_provider_module_name(provider_name)
    result = _try_load_from_module(f"{__name__}.{module_name}", provider_name)
    return result


def get_provider(name: str):
    """
    Lazy load provider by name.

    Args:
        name: Provider name (e.g., 'openai', 'anthropic', 'gemini_cli')

    Returns:
        Provider class or None if not found
    """
    return _load_provider(name)


def list_providers():
    """
    List available provider names without loading them.

    Returns:
        List of provider names discovered from provider files
    """
    providers = set()

    # Scan for file-based providers (both packages and modules)
    for _, module_name, is_pkg in pkgutil.iter_modules(__path__):
        if module_name.endswith("_provider"):
            provider_name = module_name[:-9]  # Remove '_provider' suffix
            providers.add(provider_name)
        elif is_pkg:
            # Package-based provider (e.g., antigravity/)
            providers.add(module_name)

    # Add dynamic providers from environment variables
    from ..provider_routing_config import KNOWN_PROVIDERS

    for env_var in os.environ:
        if env_var.endswith("_API_BASE"):
            provider_name = env_var[:-9].lower()  # Remove '_API_BASE' suffix
            if provider_name not in KNOWN_PROVIDERS:
                providers.add(provider_name)

    return sorted(providers)


def _ensure_dynamic_providers():
    """
    Register dynamic OpenAI-compatible providers from environment variables.
    Called lazily when first provider access happens.
    """
    from ..provider_routing_config import KNOWN_PROVIDERS

    for env_var in os.environ:
        if env_var.endswith("_API_BASE"):
            provider_name = env_var[:-9].lower()  # Remove '_API_BASE' suffix

            # Skip if this is a known LiteLLM provider (not a custom provider)
            if provider_name in KNOWN_PROVIDERS:
                continue

            # Skip if this provider already exists (file-based plugin)
            if provider_name in PROVIDER_PLUGINS:
                continue

            # Create a dynamic plugin class
            def create_plugin_class(name):
                class DynamicPlugin(__getattr__("OpenAICompatibleProvider")):
                    def __init__(self):
                        super().__init__(name)

                return DynamicPlugin

            # Create and register the plugin class
            plugin_class = create_plugin_class(provider_name)
            PROVIDER_PLUGINS[provider_name] = plugin_class
            lib_logger.debug(
                f"Registered dynamic provider: {provider_name}"
            )


def get_all_providers() -> Dict[str, _ProviderEntry]:
    """
    Get all available providers, loading them lazily as needed.
    Maintains backward compatibility with code expecting PROVIDER_PLUGINS.

    Returns:
        Dictionary mapping provider names to provider classes
    """
    # Register dynamic providers first
    _ensure_dynamic_providers()

    # Load any file-based providers that haven't been loaded yet
    for provider_name in list_providers():
        if provider_name not in PROVIDER_PLUGINS:
            _load_provider(provider_name)

    return PROVIDER_PLUGINS
