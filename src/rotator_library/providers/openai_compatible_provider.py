# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
import httpx
import logging
from typing import List, Dict, Any
from .provider_interface import ProviderInterface
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")


class OpenAICompatibleProvider(ProviderInterface):
    """
    Generic provider implementation for any OpenAI-compatible API.
    This provider can be configured via environment variables to support
    custom OpenAI-compatible endpoints without requiring code changes.
    Supports both dynamic model discovery and static model definitions.

    Environment variable pattern:
        <NAME>_API_BASE - The API base URL (required)
        <NAME>_API_KEY  - The API key (optional for some providers)

    Example:
        MYSERVER_API_BASE=http://localhost:8000/v1
        MYSERVER_API_KEY=sk-xxx

    Note: This is only used for providers NOT in the known LiteLLM providers list.
    For known providers, setting _API_BASE will override their default endpoint.
    """

    skip_cost_calculation: bool = True  # Skip cost calculation for custom providers

    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        # Get API base URL from environment (using _API_BASE pattern)
        self.api_base = os.getenv(f"{provider_name.upper()}_API_BASE")
        if not self.api_base:
            raise ValueError(
                f"Environment variable {provider_name.upper()}_API_BASE is required for custom OpenAI-compatible provider"
            )

        # Initialize model definitions loader
        self.model_definitions = ModelDefinitions()

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the OpenAI-compatible API.
        Combines dynamic discovery with static model definitions.
        """
        import json as json_lib

        models = []

        # First, try to get static model definitions
        static_models = self.model_definitions.get_all_provider_models(
            self.provider_name
        )
        static_model_ids = {m.split("/")[-1] for m in static_models}
        if static_models:
            models.extend(static_models)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for {self.provider_name}"
            )

        # Then, try dynamic discovery to get additional models
        try:
            models_url = f"{self.api_base.rstrip('/')}/models"
            response = await client.get(
                models_url, headers={"Authorization": f"Bearer {api_key}"}
            )

            # Handle HTTP errors with appropriate severity
            if response.status_code >= 500:
                lib_logger.warning(
                    "Model discovery server error %d for %s, returning static models only",
                    response.status_code, self.provider_name,
                )
                return models
            if response.status_code in (401, 403):
                lib_logger.warning(
                    "Model discovery auth error %d for %s",
                    response.status_code, self.provider_name,
                )
                return models
            response.raise_for_status()

            try:
                response_data = response.json()
            except (json_lib.JSONDecodeError, ValueError) as e:
                body_preview = response.text[:200] if response.text else "<empty>"
                lib_logger.warning(
                    "Invalid JSON in model discovery for %s: %s — body: %s",
                    self.provider_name, e, body_preview,
                )
                return models

            dynamic_models = [
                f"{self.provider_name}/{model['id']}"
                for model in response_data.get("data", [])
                if isinstance(model, dict)
                and "id" in model
                and model["id"] not in static_model_ids
            ]

            if dynamic_models:
                models.extend(dynamic_models)
                lib_logger.debug(
                    f"Discovered {len(dynamic_models)} additional models for {self.provider_name}"
                )

        except httpx.HTTPStatusError as e:
            lib_logger.warning(
                "Model discovery HTTP %d for %s",
                e.response.status_code, self.provider_name,
            )
        except httpx.RequestError:
            lib_logger.debug("Dynamic model discovery request failed for %s", self.provider_name, exc_info=True)

        return models

    def get_model_options(self, model_name: str) -> Dict[str, Any]:
        """
        Get options for a specific model from static definitions or environment variables.

        Args:
            model_name: Model name (without provider prefix)

        Returns:
            Dictionary of model options
        """
        # Extract model name without provider prefix if present
        if "/" in model_name:
            model_name = model_name.split("/")[-1]

        return self.model_definitions.get_model_options(self.provider_name, model_name)

    def has_custom_logic(self) -> bool:
        """
        Returns False since we want to use the standard litellm flow
        with just custom API base configuration.
        """
        return False

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns the standard Bearer token header for API key authentication.
        """
        return {"Authorization": f"Bearer {credential_identifier}"}
