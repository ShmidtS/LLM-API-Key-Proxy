# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import json
import httpx
import logging
from typing import List, Dict, Any, Optional
from .provider_interface import ProviderInterface
from ..error_handler import extract_retry_after_from_body

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

    def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
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
            return [
                f"opencode/{model['id']}" for model in response.json().get("data", [])
            ]
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch OpenCode models: {e}")
            return []

    @staticmethod
    def parse_quota_error(
        error: Exception, error_body: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Parse OpenCode rate limit and quota errors.

        OpenCode error format:
        {
            "error": {
                "code": 1113,
                "message": "Insufficient balance or no resource package"
            }
        }

        Common error codes:
        - 1113: Insufficient balance or no resource package (quota exhausted)
        - 429: Rate limit exceeded
        """
        body = error_body
        if not body:
            if hasattr(error, "response") and hasattr(error.response, "text"):
                try:
                    body = error.response.text
                except Exception:
                    pass
            if not body and hasattr(error, "body"):
                body = str(error.body) if error.body else None

        if not body:
            return None

        # Try extract_retry_after_from_body first
        retry_after = extract_retry_after_from_body(body)
        if retry_after:
            return {
                "retry_after": retry_after,
                "reason": "RATE_LIMIT_EXCEEDED",
            }

        # Try to parse JSON for OpenCode-specific format
        try:
            data = json.loads(body)
            error_obj = data.get("error", data)

            error_code = error_obj.get("code")
            error_message = error_obj.get("message", "").lower()

            # Code 1113: Insufficient balance or no resource package
            if error_code == 1113:
                return {
                    "retry_after": 3600,  # 1 hour default for quota issues
                    "reason": "QUOTA_EXHAUSTED",
                }

            # Code 429: Rate limit exceeded
            if error_code == 429:
                return {
                    "retry_after": 60,  # Default 60s for rate limit
                    "reason": "RATE_LIMIT_EXCEEDED",
                }

            # Check message content for specific errors
            if (
                "insufficient balance" in error_message
                or "no resource package" in error_message
            ):
                return {
                    "retry_after": 3600,  # 1 hour default
                    "reason": "QUOTA_EXHAUSTED",
                }

            if "rate limit" in error_message:
                return {
                    "retry_after": 60,
                    "reason": "RATE_LIMIT_EXCEEDED",
                }

        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Check for generic rate limit indicators
        body_lower = body.lower()
        if "rate limit" in body_lower or "too many requests" in body_lower:
            return {
                "retry_after": 60,
                "reason": "RATE_LIMIT_EXCEEDED",
            }

        return None
