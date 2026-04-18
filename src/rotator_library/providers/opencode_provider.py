# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
from typing import Dict
from ._simple_model_base import SimpleModelProvider


class OpencodeProvider(SimpleModelProvider):
    """
    Provider implementation for the OpenCode API.

    OpenCode is an OpenAI-compatible API provider that offers access to
    various models including glm-5-free and others.

    Configuration:
        OPENCODE_API_BASE - The API base URL (default: https://opencode.ai/zen/v1)
        OPENCODE_API_KEY  - The API key for authentication
    """

    skip_cost_calculation = True

    _models_url = "https://opencode.ai/zen/v1/models"
    _provider_prefix = "opencode"

    _quota_error_patterns = [
        ("json", "error.code", 1113, 3600, "QUOTA_EXHAUSTED"),
        ("json", "error.code", 429, 60, "RATE_LIMIT_EXCEEDED"),
        ("json", "error.message*", "insufficient balance", 3600, "QUOTA_EXHAUSTED"),
        ("json", "error.message*", "no resource package", 3600, "QUOTA_EXHAUSTED"),
        ("json", "error.message*", "rate limit", 60, "RATE_LIMIT_EXCEEDED"),
        ("body", "rate limit", 60, "RATE_LIMIT_EXCEEDED"),
        ("body", "too many requests", 60, "RATE_LIMIT_EXCEEDED"),
    ]

    def _resolve_models_url(self) -> str:
        """Resolve models URL from environment variable."""
        api_base = os.getenv("OPENCODE_API_BASE", "https://opencode.ai/zen/v1").rstrip("/")
        return f"{api_base}/models"

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
