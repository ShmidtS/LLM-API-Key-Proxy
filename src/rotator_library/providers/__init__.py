# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import importlib
import logging
import pkgutil
import os
from typing import Dict, Type

lib_logger = logging.getLogger("rotator_library")

from .provider_interface import ProviderInterface as _ProviderInterface
from .gemini_auth_base import GeminiAuthBase as _GeminiAuthBase
from .qwen_auth_base import QwenAuthBase as _QwenAuthBase
from .iflow_auth_base import IFlowAuthBase as _IFlowAuthBase
from .antigravity_auth_base import AntigravityAuthBase as _AntigravityAuthBase
from .colin_provider import ColinProvider as _ColinProvider
from .elysiver_provider import ElysiverProvider as _ElysiverProvider
from .openai_compatible_provider import OpenAICompatibleProvider

__all__ = [
    "PROVIDER_PLUGINS",
    "PROVIDER_AUTH_MAP",
    "OpenAICompatibleProvider",
    "get_provider",
    "list_providers",
    "get_all_providers",
]

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[_ProviderInterface]] = {}

# Compatibility registry for auth/credential tooling imports.
PROVIDER_AUTH_MAP: Dict[str, type] = {
    "gemini_cli": _GeminiAuthBase,
    "qwen_code": _QwenAuthBase,
    "iflow": _IFlowAuthBase,
    "antigravity": _AntigravityAuthBase,
    "colin": _ColinProvider,
    "elysiver": _ElysiverProvider,
}


# --- Pre-register providers with custom logic ---
# These providers implement has_custom_logic() = True and need early registration
# to bypass the standard litellm flow
PROVIDER_PLUGINS["colin"] = _ColinProvider
PROVIDER_PLUGINS["elysiver"] = _ElysiverProvider


# --- Lazy Provider Loading ---


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
                class DynamicPlugin(OpenAICompatibleProvider):
                    def __init__(self):
                        super().__init__(name)

                return DynamicPlugin

            # Create and register the plugin class
            plugin_class = create_plugin_class(provider_name)
            PROVIDER_PLUGINS[provider_name] = plugin_class
            lib_logger.debug(
                f"Registered dynamic provider: {provider_name}"
            )


def get_all_providers() -> Dict[str, Type[_ProviderInterface]]:
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
