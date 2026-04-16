# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
import logging
from typing import List
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


class KilocodeProvider(ProviderInterface):
    """
    Provider implementation for the Kilocode API.

    Kilocode routes requests to various providers through model prefixes:
    - minimax/minimax-m2.1:free
    - moonshotai/kimi-k2.5:free
    - z-ai/glm-4.7:free
    - And other provider/model combinations
    """

    _quota_error_patterns = [
        ("extract", "error.metadata.retry_after", "RATE_LIMIT_EXCEEDED"),
        ("json", "error.code", 429, 30, "RATE_LIMIT_EXCEEDED"),
        ("body", "upstream error", 5, "UPSTREAM_ERROR"),
        ("body", "provider error", 5, "UPSTREAM_ERROR"),
    ]

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Kilocode API.
        """
        import json as json_lib

        try:
            response = await client.get(
                "https://kilo.ai/api/openrouter/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            if response.status_code >= 500:
                lib_logger.warning(
                    "Kilocode model discovery server error %d", response.status_code,
                )
                return []
            if response.status_code in (401, 403):
                lib_logger.warning(
                    "Kilocode model discovery auth error %d", response.status_code,
                )
                return []
            response.raise_for_status()

            try:
                data = response.json()
            except (json_lib.JSONDecodeError, ValueError) as e:
                body_preview = response.text[:200] if response.text else "<empty>"
                lib_logger.warning(
                    "Invalid JSON in Kilocode model discovery: %s — body: %s",
                    e, body_preview,
                )
                return []

            return [
                f"kilocode/{model['id']}"
                for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ]
        except httpx.HTTPStatusError as e:
            lib_logger.warning(
                "Kilocode model discovery HTTP %d", e.response.status_code,
            )
            return []
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch Kilocode models: {e}")
            return []
