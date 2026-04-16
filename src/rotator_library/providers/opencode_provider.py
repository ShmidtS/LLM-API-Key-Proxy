# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
import logging
from typing import List, Dict
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")


class OpencodeProvider(ProviderInterface):
    """
    Provider implementation for the OpenCode API.

    OpenCode is an OpenAI-compatible API provider that offers access to
    various models including glm-5-free and others.

    Configuration:
        OPENCODE_API_BASE - The API base URL (default: https://opencode.ai/zen/v1)
        OPENCODE_API_KEY  - The API key for authentication
    """

    skip_cost_calculation = True  # Skip cost calculation for OpenCode

    _quota_error_patterns = [
        ("json", "error.code", 1113, 3600, "QUOTA_EXHAUSTED"),
        ("json", "error.code", 429, 60, "RATE_LIMIT_EXCEEDED"),
        ("json", "error.message*", "insufficient balance", 3600, "QUOTA_EXHAUSTED"),
        ("json", "error.message*", "no resource package", 3600, "QUOTA_EXHAUSTED"),
        ("json", "error.message*", "rate limit", 60, "RATE_LIMIT_EXCEEDED"),
        ("body", "rate limit", 60, "RATE_LIMIT_EXCEEDED"),
        ("body", "too many requests", 60, "RATE_LIMIT_EXCEEDED"),
    ]

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """
        Returns authentication headers for OpenCode API.

        OpenCode requires HTTP-Referer and X-Title headers for free tier models
        like glm-5-free to work properly. Without these headers, free models
        return error 1113 "Insufficient balance or no resource package".
        """
        return {
            "Authorization": f"Bearer {credential_identifier}",
            "HTTP-Referer": "https://opencode.ai",
            "X-Title": "Roo Code",
        }

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the OpenCode API.
        """
        import os

        api_base = os.getenv("OPENCODE_API_BASE", "https://opencode.ai/zen/v1").rstrip(
            "/"
        )

        try:
            response = await client.get(
                f"{api_base}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            import json as json_lib
            try:
                data = response.json()
            except (json_lib.JSONDecodeError, ValueError) as e:
                lib_logger.warning(f"Invalid JSON from OpenCode models: {e}, body={response.text[:200]}")
                return []
            return [
                f"opencode/{model['id']}" for model in data.get("data", [])
                if isinstance(model, dict) and "id" in model
            ]
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                lib_logger.warning(f"Auth error fetching OpenCode models: {e.response.status_code}")
            elif e.response.status_code >= 500:
                lib_logger.warning(f"Server error fetching OpenCode models: {e.response.status_code}")
            else:
                lib_logger.error(f"HTTP error fetching OpenCode models: {e}")
            return []
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch OpenCode models: {e}")
            return []
