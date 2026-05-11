# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
import json
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


class SimpleModelProvider(ProviderInterface):
    """
    Base class for providers that only need model discovery + error pattern config.

    Subclasses set class attributes for URL, prefix, and error patterns,
    and optionally override _resolve_models_url() for dynamic URLs
    or get_auth_header() for custom authentication.
    """

    _models_url: str = ""
    _provider_prefix: str = ""

    def _resolve_models_url(self) -> str:
        """Return the models discovery URL. Override for dynamic URLs."""
        return self._models_url

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """Fetch the list of available models from the provider's API."""
        url = self._resolve_models_url()
        try:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code >= 500:
                lib_logger.warning(
                    "%s model discovery server error %d",
                    self._provider_prefix, response.status_code,
                )
                return []
            if response.status_code in (401, 403):
                lib_logger.warning(
                    "%s model discovery auth error %d",
                    self._provider_prefix, response.status_code,
                )
                return []
            response.raise_for_status()

            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                body_preview = response.text[:200] if response.text else "<empty>"
                lib_logger.warning(
                    "Invalid JSON in %s model discovery: %s — body: %s",
                    self._provider_prefix, e, body_preview,
                )
                return []

            return [
                f"{self._provider_prefix}/{model['id']}"
                for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ]
        except httpx.HTTPStatusError as e:
            lib_logger.warning(
                "%s model discovery HTTP %d",
                self._provider_prefix, e.response.status_code,
            )
            return []
        except httpx.RequestError as e:
            lib_logger.error(
                "Failed to fetch %s models: %s", self._provider_prefix, e,
            )
            return []


_SIMPLE_PROVIDER_REGISTRY: dict[str, dict] = {
    "openrouter": {
        "models_url": "https://openrouter.ai/api/v1/models",
        "prefix": "openrouter",
    },
    "xai": {
        "models_url": "https://api.x.ai/v1/models",
        "prefix": "xai",
    },
    "kilocode": {
        "models_url": "https://kilo.ai/api/openrouter/models",
        "prefix": "kilocode",
        "quota_error_patterns": [
            ("extract", "error.metadata.retry_after", "RATE_LIMIT_EXCEEDED"),
            ("json", "error.code", 429, 30, "RATE_LIMIT_EXCEEDED"),
            ("body", "upstream error", 5, "UPSTREAM_ERROR"),
            ("body", "provider error", 5, "UPSTREAM_ERROR"),
        ],
    },
}


def get_simple_provider(name: str) -> type[SimpleModelProvider] | None:
    config = _SIMPLE_PROVIDER_REGISTRY.get(name)
    if config is None:
        return None
    attrs = {
        "_models_url": config["models_url"],
        "_provider_prefix": config["prefix"],
    }
    if "quota_error_patterns" in config:
        attrs["_quota_error_patterns"] = config["quota_error_patterns"]
    return type(f"{name.title()}Provider", (SimpleModelProvider,), attrs)
