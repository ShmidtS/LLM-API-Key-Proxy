# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
from typing import Dict

from .provider_interface import ProviderInterface


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
