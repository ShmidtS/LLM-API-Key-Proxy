# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/qwen_code_provider.py

import time
import os
import httpx
import logging
from typing import List, Dict, Any
from .provider_interface import ProviderInterface, strip_provider_prefix, build_bearer_headers
from .qwen_auth_base import QwenAuthBase
from .acompletion_mixin import ACompletionMixin
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")


HARDCODED_MODELS = ["qwen3-coder-plus", "qwen3-coder-flash"]

# OpenAI-compatible parameters supported by Qwen Code API
SUPPORTED_PARAMS = {
    "model",
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "stream",
    "tools",
    "tool_choice",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "stop",
    "seed",
    "response_format",
}


class QwenCodeProvider(QwenAuthBase, ACompletionMixin, ProviderInterface):
    skip_cost_calculation = True
    REASONING_START_MARKER = "THINK||"
    provider_name = "Qwen Code"
    llm_provider = "qwen_code"

    def _get_stream_endpoint(self, model: str) -> str:
        return "/v1/chat/completions"

    def _get_extra_headers(self) -> dict:
        return {
            "User-Agent": "google-api-nodejs-client/9.15.1",
            "X-Goog-Api-Client": "gl-node/22.17.0",
            "Client-Metadata": "ideType=IDE_UNSPECIFIED,platform=PLATFORM_UNSPECIFIED,pluginType=GEMINI",
        }

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a merged list of Qwen Code models from three sources:
        1. Environment variable models (via QWEN_CODE_MODELS) - ALWAYS included, take priority
        2. Hardcoded models (fallback list) - added only if ID not in env vars
        3. Dynamic discovery from Qwen API (if supported) - added only if ID not in env vars

        Environment variable models always win and are never deduplicated, even if they
        share the same ID (to support different configs like temperature, etc.)

        Validates OAuth credentials if applicable.
        """
        models = []
        env_var_ids = (
            set()
        )  # Track IDs from env vars to prevent hardcoded/dynamic duplicates

        def extract_model_id(item) -> str:
            """Extract model ID from various formats (dict, string with/without provider prefix)."""
            if isinstance(item, dict):
                # Dict format: extract 'id' or 'name' field
                return item.get("id") or item.get("name", "")
            elif isinstance(item, str):
                # String format: extract ID from "provider/id" or just "id"
                return item.split("/")[-1] if "/" in item else item
            return str(item)

        # Source 1: Load environment variable models (ALWAYS include ALL of them)
        static_models = self.model_definitions.get_all_provider_models("qwen_code")
        if static_models:
            for model in static_models:
                # Extract model name from "qwen_code/ModelName" format
                model_name = strip_provider_prefix(model)
                # Get the actual model ID from definitions (which may differ from the name)
                model_id = self.model_definitions.get_model_id("qwen_code", model_name)

                # ALWAYS add env var models (no deduplication)
                models.append(model)
                # Track the ID to prevent hardcoded/dynamic duplicates
                if model_id:
                    env_var_ids.add(model_id)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for qwen_code from environment variables"
            )

        # Source 2: Add hardcoded models (only if ID not already in env vars)
        for model_id in HARDCODED_MODELS:
            if model_id not in env_var_ids:
                models.append(f"qwen_code/{model_id}")
                env_var_ids.add(model_id)

        # Source 3: Try dynamic discovery from Qwen Code API (only if ID not already in env vars)
        try:
            # Validate OAuth credentials and get API details
            if os.path.isfile(credential):
                await self.initialize_token(credential)

            api_base, access_token = await self.get_api_details(credential)
            models_url = f"{api_base.rstrip('/')}/v1/models"

            response = await client.get(
                models_url, headers=build_bearer_headers(access_token)
            )
            response.raise_for_status()

            import json as json_lib
            try:
                dynamic_data = response.json()
            except (json_lib.JSONDecodeError, ValueError) as e:
                lib_logger.warning(f"Invalid JSON from qwen_code models: {e}, body={response.text[:200]}")
                dynamic_data = {}

            # Handle both {data: [...]} and direct [...] formats
            model_list = (
                dynamic_data.get("data", dynamic_data)
                if isinstance(dynamic_data, dict)
                else dynamic_data
            )

            dynamic_count = 0
            for model in model_list:
                model_id = extract_model_id(model)
                if model_id and model_id not in env_var_ids:
                    models.append(f"qwen_code/{model_id}")
                    env_var_ids.add(model_id)
                    dynamic_count += 1

            if dynamic_count > 0:
                lib_logger.debug(
                    f"Discovered {dynamic_count} additional models for qwen_code from API"
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                lib_logger.warning(f"Auth error fetching qwen_code models: {e.response.status_code}")
            elif e.response.status_code >= 500:
                lib_logger.warning(f"Server error fetching qwen_code models: {e.response.status_code}")
            else:
                lib_logger.debug(f"HTTP error fetching qwen_code models: {e}")
        except httpx.RequestError as e:
            lib_logger.debug(f"Request error fetching qwen_code models: {e}")
        except Exception as e:
            # Silently ignore dynamic discovery errors
            lib_logger.debug(f"Dynamic model discovery failed for qwen_code: {e}")

        return models

    def _build_request_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Builds a clean request payload with only supported parameters.
        This prevents 400 Bad Request errors from litellm-internal parameters.
        """
        # Extract only supported OpenAI parameters
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        # Always force streaming for internal processing
        payload["stream"] = True

        # Always include usage data in stream
        payload["stream_options"] = {"include_usage": True}

        # Handle tool schema cleaning
        if "tools" in payload and payload["tools"]:
            payload["tools"] = self._clean_tool_schemas(payload["tools"])
            lib_logger.debug(f"Cleaned {len(payload['tools'])} tool schemas")
        elif not payload.get("tools"):
            # Per Qwen Code API bug (see: https://github.com/qianwen-team/flash-dance/issues/2),
            # injecting a dummy tool prevents stream corruption when no tools are provided
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "do_not_call_me",
                        "description": "Do not call this tool.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
            lib_logger.debug(
                "Injected dummy tool to prevent Qwen API stream corruption"
            )

        return payload

    def _convert_chunk_to_openai(self, chunk: Dict[str, Any], model_id: str):
        """
        Converts a raw Qwen SSE chunk to an OpenAI-compatible chunk.

        CRITICAL FIX: Handle chunks with BOTH usage and choices (final chunk)
        by combining them into a single chunk with both finish_reason and usage.
        The client expects finish_reason and usage with completion_tokens > 0
        to be in the same chunk to properly detect the final chunk.
        """
        if not isinstance(chunk, dict):
            return

        # Get choices and usage data
        choices = chunk.get("choices", [])
        usage_data = chunk.get("usage")
        chunk_id = chunk.get("id", f"chatcmpl-qwen-{time.time()}")
        chunk_created = chunk.get("created", int(time.time()))

        # Handle chunks with BOTH choices and usage (typical for final chunk)
        # CRITICAL: Combine into ONE chunk with both finish_reason and usage
        # The client detects final chunk by usage.completion_tokens > 0 AND choices
        if choices and usage_data:
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Yield a SINGLE combined chunk with both choices and usage
            yield {
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk_id,
                "created": chunk_created,
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
            }
            return

        # Handle usage-only chunks (without choices)
        # These cannot carry finish_reason since client requires choices to set it
        if usage_data:
            yield {
                "choices": [],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk_id,
                "created": chunk_created,
                "usage": {
                    "prompt_tokens": usage_data.get("prompt_tokens", 0),
                    "completion_tokens": usage_data.get("completion_tokens", 0),
                    "total_tokens": usage_data.get("total_tokens", 0),
                },
            }
            return

        # Handle content-only chunks
        if not choices:
            return

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle <think> tags for reasoning content
        content = delta.get("content")
        if content and ("<think>" in content or "</think>" in content):
            parts = (
                content.replace("<think>", f"||{self.REASONING_START_MARKER}")
                .replace("</think>", f"||/{self.REASONING_START_MARKER}")
                .split("||")
            )
            for part in parts:
                if not part:
                    continue

                new_delta = {}
                if part.startswith(self.REASONING_START_MARKER):
                    new_delta["reasoning_content"] = part.replace(
                        self.REASONING_START_MARKER, ""
                    )
                elif part.startswith(f"/{self.REASONING_START_MARKER}"):
                    continue
                else:
                    new_delta["content"] = part

                yield {
                    "choices": [
                        {"index": 0, "delta": new_delta, "finish_reason": None}
                    ],
                    "model": model_id,
                    "object": "chat.completion.chunk",
                    "id": chunk_id,
                    "created": chunk_created,
                }
        else:
            # Standard content chunk
            yield {
                "choices": [
                    {"index": 0, "delta": delta, "finish_reason": finish_reason}
                ],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk_id,
                "created": chunk_created,
            }

