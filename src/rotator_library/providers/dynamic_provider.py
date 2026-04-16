# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from .openai_compatible_provider import OpenAICompatibleProvider


class DynamicOpenAICompatibleProvider(OpenAICompatibleProvider):
    """
    Dynamic provider for custom OpenAI-compatible providers.
    Inherits all logic from OpenAICompatibleProvider.

    Created at runtime for providers with _API_BASE environment variables
    that are NOT known LiteLLM providers.
    """

    skip_cost_calculation: bool = True

    def __init__(self, provider_name: str):
        super().__init__(provider_name)
