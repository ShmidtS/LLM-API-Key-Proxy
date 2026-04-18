# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/iflow_provider.py

import json
import time
import os
import httpx
import logging
from typing import List, Dict, Any
from .provider_interface import ProviderInterface, strip_provider_prefix, build_bearer_headers
from .iflow_auth_base import IFlowAuthBase
from .acompletion_mixin import ACompletionMixin
from ..model_definitions import ModelDefinitions

lib_logger = logging.getLogger("rotator_library")


# Model list can be expanded as iFlow supports more models
HARDCODED_MODELS = [
    "glm-4.6",
    "minimax-m2",
    "qwen3-coder-plus",
    "kimi-k2",
    "kimi-k2-0905",
    "kimi-k2-thinking",
    "qwen3-max",
    "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3.2-chat",
    "deepseek-v3.2",
    "deepseek-v3.1",
    "deepseek-v3",
    "deepseek-r1",
    "qwen3-vl-plus",
    "qwen3-235b-a22b-instruct",
    "qwen3-235b",
]

# OpenAI-compatible parameters supported by iFlow API
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
    "thinking",
    "reasoning_effort",
}

# Models that support extended thinking/reasoning
THINKING_SUPPORTED_MODELS = {
    "deepseek-r1",
    "kimi-k2-thinking",
    "qwen3-235b-a22b-thinking-2507",
}

# Reasoning effort token budgets (for models that support it)
REASONING_BUDGETS = {
    "low": 4096,
    "medium": 8192,
    "high": 16384,
}


class IFlowProvider(IFlowAuthBase, ACompletionMixin, ProviderInterface):
    """
    iFlow provider using OAuth authentication with local callback server.
    API requests use the derived API key (NOT OAuth access_token).
    """

    skip_cost_calculation = True
    provider_name = "iFlow"
    llm_provider = "iflow"

    def _get_stream_endpoint(self, model: str) -> str:
        return "/chat/completions"

    def _get_extra_headers(self) -> dict:
        return {"User-Agent": "iFlow-Cli"}

    def __init__(self):
        super().__init__()
        self.model_definitions = ModelDefinitions()

    def has_custom_logic(self) -> bool:
        return True

    async def get_models(self, credential: str, client: httpx.AsyncClient) -> List[str]:
        """
        Returns a merged list of iFlow models from three sources:
        1. Environment variable models (via IFLOW_MODELS) - ALWAYS included, take priority
        2. Hardcoded models (fallback list) - added only if ID not in env vars
        3. Dynamic discovery from iFlow API (if supported) - added only if ID not in env vars

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
        static_models = self.model_definitions.get_all_provider_models("iflow")
        if static_models:
            for model in static_models:
                # Extract model name from "iflow/ModelName" format
                model_name = strip_provider_prefix(model)
                # Get the actual model ID from definitions (which may differ from the name)
                model_id = self.model_definitions.get_model_id("iflow", model_name)

                # ALWAYS add env var models (no deduplication)
                models.append(model)
                # Track the ID to prevent hardcoded/dynamic duplicates
                if model_id:
                    env_var_ids.add(model_id)
            lib_logger.info(
                f"Loaded {len(static_models)} static models for iflow from environment variables"
            )

        # Source 2: Add hardcoded models (only if ID not already in env vars)
        for model_id in HARDCODED_MODELS:
            if model_id not in env_var_ids:
                models.append(f"iflow/{model_id}")
                env_var_ids.add(model_id)

        # Source 3: Try dynamic discovery from iFlow API (only if ID not already in env vars)
        try:
            # Validate OAuth credentials and get API details
            if os.path.isfile(credential):
                await self.initialize_token(credential)

            api_base, api_key = await self.get_api_details(credential)
            models_url = f"{api_base.rstrip('/')}/models"

            response = await client.get(
                models_url, headers=build_bearer_headers(api_key)
            )
            response.raise_for_status()

            try:
                dynamic_data = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                lib_logger.warning(f"Invalid JSON from iflow models: {e}, body={response.text[:200]}")
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
                    models.append(f"iflow/{model_id}")
                    env_var_ids.add(model_id)
                    dynamic_count += 1

            if dynamic_count > 0:
                lib_logger.debug(
                    f"Discovered {dynamic_count} additional models for iflow from API"
                )

        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                lib_logger.warning(f"Auth error fetching iflow models: {e.response.status_code}")
            elif e.response.status_code >= 500:
                lib_logger.warning(f"Server error fetching iflow models: {e.response.status_code}")
            else:
                lib_logger.debug(f"HTTP error fetching iflow models: {e}")
        except httpx.RequestError as e:
            lib_logger.debug(f"Request error fetching iflow models: {e}")
        except Exception as e:
            # Silently ignore dynamic discovery errors
            lib_logger.debug(f"Dynamic model discovery failed for iflow: {e}")

        return models

    def _handle_thinking_parameter(self, payload: Dict[str, Any], model: str) -> None:
        """
        Handles thinking/reasoning parameters for iFlow models.

        Supports two modes:
        1. Explicit `thinking` parameter (OpenAI-style): {"type": "enabled", "budget_tokens": N}
        2. `reasoning_effort` parameter: "low", "medium", "high" -> maps to token budgets

        Only applies to models that support extended thinking.
        """
        model_id = strip_provider_prefix(model)

        # Check if model supports thinking
        if model_id not in THINKING_SUPPORTED_MODELS:
            # Clean up reasoning params for non-thinking models
            payload.pop("thinking", None)
            payload.pop("reasoning_effort", None)
            return

        # If explicit thinking is already set, validate and keep it
        if "thinking" in payload:
            thinking = payload["thinking"]
            if isinstance(thinking, dict):
                # Ensure required fields
                if thinking.get("type") != "enabled":
                    thinking["type"] = "enabled"
                # Validate budget_tokens
                budget = thinking.get("budget_tokens")
                if budget is not None and not isinstance(budget, int):
                    try:
                        thinking["budget_tokens"] = int(budget)
                    except (ValueError, TypeError):
                        thinking["budget_tokens"] = 8192  # Default
            else:
                # Convert to proper format
                payload["thinking"] = {"type": "enabled", "budget_tokens": 8192}
            # Remove reasoning_effort if thinking is explicitly set
            payload.pop("reasoning_effort", None)
            return

        # Handle reasoning_effort -> thinking conversion
        reasoning_effort = payload.get("reasoning_effort")
        if reasoning_effort:
            budget = REASONING_BUDGETS.get(reasoning_effort, 8192)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
            payload.pop("reasoning_effort", None)
            lib_logger.debug(f"Converted reasoning_effort={reasoning_effort} to thinking budget={budget}")
        else:
            # Default thinking for supported models (auto-enable)
            payload["thinking"] = {"type": "enabled", "budget_tokens": -1}  # -1 = auto
            lib_logger.debug(f"Auto-enabled thinking for model {model_id}")

    def _build_request_payload(self, **kwargs) -> Dict[str, Any]:
        """
        Builds a clean request payload with only supported parameters.
        This prevents 400 Bad Request errors from litellm-internal parameters.
        """
        # Extract only supported OpenAI parameters
        payload = {k: v for k, v in kwargs.items() if k in SUPPORTED_PARAMS}

        # Always force streaming for internal processing
        payload["stream"] = True

        # NOTE: iFlow API does not support stream_options parameter
        # Unlike other providers, we don't include it to avoid HTTP 406 errors

        # Handle tool schema cleaning
        if "tools" in payload and payload["tools"]:
            payload["tools"] = self._clean_tool_schemas(payload["tools"])
            lib_logger.debug(f"Cleaned {len(payload['tools'])} tool schemas")
        elif (
            "tools" in payload
            and isinstance(payload["tools"], list)
            and len(payload["tools"]) == 0
        ):
            # Inject dummy tool for empty arrays to prevent streaming issues (similar to Qwen's behavior)
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "Placeholder tool to stabilise streaming",
                        "parameters": {"type": "object"},
                    },
                }
            ]
            lib_logger.debug("Injected placeholder tool for empty tools array")

        # Handle thinking/reasoning parameters
        model = kwargs.get("model", "")
        self._handle_thinking_parameter(payload, model)

        return payload

    # Marker for parsing reasoning content from <think> tags
    REASONING_START_MARKER = "THINK||"

    def _convert_chunk_to_openai(self, chunk: Dict[str, Any], model_id: str):
        """
        Converts a raw iFlow SSE chunk to an OpenAI-compatible chunk.
        Since iFlow is OpenAI-compatible, minimal conversion is needed.

        CRITICAL FIX: Handle chunks with BOTH usage and choices (final chunk)
        without early return to ensure finish_reason is properly processed.

        Also handles reasoning content from models like DeepSeek-R1 that use
        <think> tags to separate reasoning from final response.
        """
        if not isinstance(chunk, dict):
            return

        # Get choices and usage data
        choices = chunk.get("choices", [])
        usage_data = chunk.get("usage")
        chunk_id = chunk.get("id", f"chatcmpl-iflow-{time.time()}")
        chunk_created = chunk.get("created", int(time.time()))

        # Handle chunks with BOTH choices and usage (typical for final chunk)
        # CRITICAL: Process choices FIRST to capture finish_reason, then yield usage
        if choices and usage_data:
            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Check for reasoning content in the delta
            content = delta.get("content")
            if content and ("<think>" in content or "</think>" in content):
                # Parse thinking tags and yield multiple chunks
                for parsed_chunk in self._parse_thinking_content(
                    content, delta, finish_reason, model_id, chunk_id, chunk_created
                ):
                    yield parsed_chunk
            else:
                # Yield the choice chunk first (contains finish_reason)
                yield {
                    "choices": choices,
                    "model": model_id,
                    "object": "chat.completion.chunk",
                    "id": chunk_id,
                    "created": chunk_created,
                }

            # Then yield the usage chunk
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

        # Handle usage-only chunks
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

        # Handle <think> tags for reasoning content (e.g., DeepSeek-R1)
        content = delta.get("content")
        if content and ("<think>" in content or "</think>" in content):
            for parsed_chunk in self._parse_thinking_content(
                content, delta, finish_reason, model_id, chunk_id, chunk_created
            ):
                yield parsed_chunk
        else:
            # Standard content chunk - pass through
            yield {
                "choices": choices,
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk_id,
                "created": chunk_created,
            }

    def _parse_thinking_content(
        self, content: str, delta: Dict[str, Any], finish_reason: Any,
        model_id: str, chunk_id: str, chunk_created: int
    ):
        """
        Parses content with <think> tags and yields separate chunks for
        reasoning_content and content.

        Similar to Qwen Code implementation but adapted for iFlow.
        """
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
                # Reasoning content (inside <think> tags)
                new_delta["reasoning_content"] = part.replace(
                    self.REASONING_START_MARKER, ""
                )
            elif part.startswith(f"/{self.REASONING_START_MARKER}"):
                # End of thinking - skip this marker
                continue
            else:
                # Regular content (outside <think> tags)
                new_delta["content"] = part

            yield {
                "choices": [
                    {"index": 0, "delta": new_delta, "finish_reason": finish_reason}
                ],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk_id,
                "created": chunk_created,
            }

