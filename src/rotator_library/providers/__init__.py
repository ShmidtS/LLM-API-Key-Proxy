# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import importlib
import pkgutil
import os
from typing import Dict, Type
from .provider_interface import ProviderInterface
from .gemini_auth_base import GeminiAuthBase
from .qwen_auth_base import QwenAuthBase
from .iflow_auth_base import IFlowAuthBase
from .antigravity_auth_base import AntigravityAuthBase
from .colin_provider import ColinProvider

# Shared base class for streaming response deduplication
from .base_streaming_provider import StreamingResponseMixin

# --- Provider Plugin System ---

# Dictionary to hold discovered provider classes, mapping provider name to class
PROVIDER_PLUGINS: Dict[str, Type[ProviderInterface]] = {}

# Compatibility registry for auth/credential tooling imports.
PROVIDER_AUTH_MAP: Dict[str, type] = {
    "gemini_cli": GeminiAuthBase,
    "qwen_code": QwenAuthBase,
    "iflow": IFlowAuthBase,
    "antigravity": AntigravityAuthBase,
    "colin": ColinProvider,
}


class DynamicOpenAICompatibleProvider:
    """
    Dynamic provider class for custom OpenAI-compatible providers.
    Created at runtime for providers with _API_BASE environment variables
    that are NOT known LiteLLM providers.

    Environment variable pattern:
    <NAME>_API_BASE - The API base URL (required)
    <NAME>_API_KEY - The API key

    Example:
    MYSERVER_API_BASE=http://localhost:8000/v1
    MYSERVER_API_KEY=sk-xxx

    Note: For known providers (openai, anthropic, etc.), setting _API_BASE
    will override their default endpoint without creating a custom provider.
    """

    # Class attribute - no need to instantiate
    skip_cost_calculation: bool = True

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        # Get API base URL from environment (using _API_BASE pattern)
        self.api_base = os.getenv(f"{provider_name.upper()}_API_BASE")
        if not self.api_base:
            raise ValueError(
                f"Environment variable {provider_name.upper()}_API_BASE is required for custom OpenAI-compatible provider"
            )

        # Import model definitions
        from ..model_definitions import ModelDefinitions

        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client):
        """Delegate to OpenAI-compatible provider implementation."""
        from .openai_compatible_provider import OpenAICompatibleProvider

        # Create temporary instance to reuse logic
        temp_provider = OpenAICompatibleProvider(self.provider_name)
        return await temp_provider.get_models(api_key, client)

    def get_model_options(self, model_name: str) -> Dict[str, any]:
        """Get model options from static definitions."""
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """Returns False since we want to use the standard litellm flow."""
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Returns the standard Bearer token header."""
        return {"Authorization": f"Bearer {credential_identifier}"}


# --- Pre-register providers with custom logic ---
# These providers implement has_custom_logic() = True and need early registration
# to bypass the standard litellm flow
PROVIDER_PLUGINS["colin"] = ColinProvider


# --- Lazy Provider Loading ---

# Cache for loaded provider modules
_provider_registry: Dict[str, object] = {}

# Track which providers have been fully registered
_registered_providers: set = set()


def _get_provider_module_name(provider_name: str) -> str:
    """Convert provider name to module name."""
    # Handle special case: nvidia_nim -> nvidia_provider
    if provider_name == "nvidia_nim":
        return "nvidia_provider"
    return f"{provider_name}_provider"


def _load_provider(provider_name: str):
    """
    Lazily load a single provider by name.
    Returns the provider class or None if not found.
    """
    if provider_name in _provider_registry:
        return _provider_registry[provider_name]

    module_name = _get_provider_module_name(provider_name)
    full_module_path = f"{__name__}.{module_name}"

    try:
        module = importlib.import_module(full_module_path)
    except ImportError:
        return None

    # Look for a class that inherits from ProviderInterface
    for attribute_name in dir(module):
        attribute = getattr(module, attribute_name)
        if (
            isinstance(attribute, type)
            and issubclass(attribute, ProviderInterface)
            and attribute is not ProviderInterface
        ):
            _provider_registry[provider_name] = attribute
            PROVIDER_PLUGINS[provider_name] = attribute
            _registered_providers.add(provider_name)
            import logging
            logging.getLogger("rotator_library").debug(
                f"Lazy-loaded provider: {provider_name}"
            )
            return attribute

    return None


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

    # Scan for file-based providers
    for _, module_name, _ in pkgutil.iter_modules(__path__):
        if module_name.endswith("_provider"):
            provider_name = module_name[:-9]  # Remove '_provider' suffix
            # Remap 'nvidia' to 'nvidia_nim' to align with litellm's provider name
            if provider_name == "nvidia":
                provider_name = "nvidia_nim"
            providers.add(provider_name)

    # Add dynamic providers from environment variables
    from ..provider_config import KNOWN_PROVIDERS

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
    from ..provider_config import KNOWN_PROVIDERS

    for env_var in os.environ:
        if env_var.endswith("_API_BASE"):
            provider_name = env_var[:-9].lower()  # Remove '_API_BASE' suffix

            # Skip if this is a known LiteLLM provider (not a custom provider)
            if provider_name in KNOWN_PROVIDERS:
                continue

            # Skip if this provider already exists (file-based plugin)
            if provider_name in _provider_registry or provider_name in _registered_providers:
                continue

            # Create a dynamic plugin class
            def create_plugin_class(name):
                class DynamicPlugin(DynamicOpenAICompatibleProvider):
                    def __init__(self):
                        super().__init__(name)

                return DynamicPlugin

            # Create and register the plugin class
            plugin_class = create_plugin_class(provider_name)
            _provider_registry[provider_name] = plugin_class
            PROVIDER_PLUGINS[provider_name] = plugin_class
            _registered_providers.add(provider_name)
            import logging
            logging.getLogger("rotator_library").debug(
                f"Registered dynamic provider: {provider_name}"
            )


def get_all_providers() -> Dict[str, Type[ProviderInterface]]:
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
        if provider_name not in _provider_registry:
            _load_provider(provider_name)

    return PROVIDER_PLUGINS
