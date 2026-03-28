# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
COLIN API Provider - OpenAI Responses API format.

This provider implements the OpenAI Responses API format used by COLIN.
Unlike standard chat/completions, this uses the /responses endpoint.

API Documentation:
- Endpoint: https://claude.colin1112.tech/v1/responses
- Model: gpt-5.3-codex
- Format: OpenAI Responses API (not chat/completions)

Key differences from chat/completions:
- Uses 'input' instead of 'messages'
- Response structure: output[].content[].text
- Supports streaming via SSE
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator, Union
import httpx
import litellm
from litellm.types.utils import Delta as DeltaType, ChatCompletionMessageToolCall

from .provider_interface import ProviderInterface

lib_logger = logging.getLogger("rotator_library")

# Default API configuration
COLIN_DEFAULT_API_BASE = "https://claude.colin1112.tech/v1"


class ColinProvider(ProviderInterface):
    """
    Provider for COLIN API using OpenAI Responses API format.

    Environment variables:
    COLIN_API_BASE: API base URL (default: https://claude.colin1112.tech/v1)
    COLIN_API_KEY_N: API key(s) for authentication

    Usage in .env:
    COLIN_API_BASE=https://claude.colin1112.tech/v1
    COLIN_API_KEY_1=sk-xxx
    """

    # Provider identification
    provider_env_name: str = "COLIN"

    # Skip cost calculation for custom providers
    skip_cost_calculation: bool = True

    # Default rotation mode
    default_rotation_mode: str = "balanced"

    # Tier configuration (single tier for simplicity)
    tier_priorities: Dict[str, int] = {
        "default": 1,
    }

    def __init__(self):
        """Initialize COLIN provider."""
        self.api_base = os.getenv("COLIN_API_BASE", COLIN_DEFAULT_API_BASE)

        # Ensure no trailing slash
        self.api_base = self.api_base.rstrip("/")

        lib_logger.info(f"COLIN Provider initialized with base: {self.api_base}")

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Return available models for COLIN provider.

        Note: COLIN's /models endpoint returns 403, so we return static list.
        The primary model is gpt-5.3-codex.
        """
        # Static model list since /models returns 403
        # These are known working models on COLIN
        return [
            "colin/gpt-5.3-codex",
        ]

    def has_custom_logic(self) -> bool:
        """
        Return True since we implement custom acompletion logic.

        COLIN uses the Responses API format which is different from
        standard chat/completions, so we need custom handling.
        """
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        """
        Handle completion request using COLIN's Responses API.

        Converts chat/completions format to Responses API format:
        - messages -> input (last user message)
        - Standard response -> litellm.ModelResponse
        - Streaming -> AsyncGenerator of ModelResponse chunks
        """
        # Extract parameters
        model = kwargs.get("model", "gpt-5.3-codex")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        # credential_identifier is passed by the rotator, contains the API key
        credential_identifier = kwargs.pop("credential_identifier", "")
        transaction_context = kwargs.pop("transaction_context", None)

        # Remove provider prefix from model if present
        if "/" in model:
            model = model.split("/")[-1]

        # Build request payload for Responses API
        # Convert messages to input format
        # For multi-turn, we use the last user message as input
        # and include conversation history in metadata if needed

        # Extract the conversation into a single input
        input_text = self._messages_to_input(messages)

        payload = {
            "model": model,
            "input": input_text,
        }

        # Add optional parameters
        if kwargs.get("max_tokens"):
            payload["max_output_tokens"] = kwargs["max_tokens"]

        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]

        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]

        # Handle response format for JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}

        # Handle tools (function calling)
        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]
            lib_logger.info(f"COLIN: tools parameter present, count={len(kwargs['tools'])}")

        lib_logger.debug(f"COLIN request: model={model}, stream={stream}")

        # Make request
        url = f"{self.api_base}/responses"
        headers = {
            "Authorization": f"Bearer {credential_identifier}",
            "Content-Type": "application/json",
        }

        if stream:
            payload["stream"] = True
            return self._stream_response(client, url, headers, payload, model)
        else:
            return await self._non_stream_response(client, url, headers, payload, model)

    def _messages_to_input(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert chat messages to Responses API input format.

        The Responses API expects a single 'input' field. We concatenate
        the conversation into a formatted string that preserves context.
        """
        if not messages:
            return ""

        # For single user message, use directly
        if len(messages) == 1 and messages[0].get("role") == "user":
            return messages[0].get("content", "")

        # For multi-turn, format as conversation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"[SYSTEM]: {content}")
            elif role == "user":
                parts.append(f"[USER]: {content}")
            elif role == "assistant":
                parts.append(f"[ASSISTANT]: {content}")

        return "\n".join(parts)

    async def _non_stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        """
        Handle non-streaming response.

        Note: COLIN API always returns SSE stream, even without stream=True.
        We need to consume the entire stream and accumulate the response.
        """
        # Use streaming request even for non-streaming API call
        # COLIN always returns SSE format
        async with client.stream(
            "POST", url, headers=headers, json=payload
        ) as response:
            # Check status code
            if response.status_code >= 400:
                error_body = await response.aread()
                response.raise_for_status()

            # Accumulate content and metadata from SSE stream
            full_content = ""
            response_id = f"colin-{model}"
            created_at = 0
            usage = {}
            tool_calls = []

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Handle SSE format: "event: ..." or "data: ..."
                if line.startswith("event:"):
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)

                        # Capture response metadata from response.created event
                        if data.get("type") == "response.created":
                            resp = data.get("response", {})
                            response_id = resp.get("id", response_id)
                            created_at = resp.get("created_at", 0)

                        # Capture response.completed for final metadata
                        elif data.get("type") == "response.completed":
                            resp = data.get("response", {})
                            usage = resp.get("usage", {})
                            if resp.get("id"):
                                response_id = resp["id"]

                        # Accumulate text from response.output_text.delta
                        elif data.get("type") == "response.output_text.delta":
                            full_content += data.get("delta", "")

                        # Handle function calls from response.output_item.done
                        elif data.get("type") == "response.output_item.done":
                            output_item = data.get("output", {})
                            if output_item.get("type") == "function_call":
                                tool_calls.append(
                                    ChatCompletionMessageToolCall(
                                        id=output_item.get("id", f"call_{len(tool_calls)}"),
                                        type="function",
                                        function={
                                            "name": output_item.get("name", ""),
                                            "arguments": output_item.get("arguments", "{}"),
                                        },
                                    )
                                )

                        # Alternative: output_text.delta
                        elif "delta" in data:
                            delta = data["delta"]
                            if isinstance(delta, str):
                                full_content += delta
                            elif isinstance(delta, dict):
                                full_content += delta.get("content", "") or delta.get(
                                    "text", ""
                                )

                    except json.JSONDecodeError:
                        lib_logger.debug(f"Failed to parse SSE data: {data_str[:100]}")
                        continue

            # Determine finish reason
            finish_reason = "stop"
            if tool_calls:
                finish_reason = "tool_calls"

            # Create litellm response
            return litellm.ModelResponse(
                id=response_id,
                choices=[
                    litellm.Choices(
                        index=0,
                        message=litellm.Message(
                            role="assistant",
                            content=full_content if not tool_calls else None,
                            tool_calls=tool_calls if tool_calls else None,
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                created=created_at,
                model=model,
                object="chat.completion",
                usage=litellm.Usage(
                    prompt_tokens=usage.get("input_tokens", 0),
                    completion_tokens=usage.get("output_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
            )

    async def _stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> AsyncGenerator[litellm.ModelResponse, None]:
        """Handle streaming response."""
        # Use async with for proper context management
        async with client.stream(
            "POST", url, headers=headers, json=payload
        ) as response:
            # Check status code after response starts
            if response.status_code >= 400:
                # Read error body before raising
                error_body = await response.aread()
                response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                # Handle SSE format: "event: ..." or "data: ..."
                if line.startswith("event:"):
                    # Event type line, skip
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)

                        # Parse COLIN SSE format
                        content_delta = ""

                        # Pattern 1: response.output_item.added
                        if data.get("type") == "response.output_item.added":
                            # Output item started, no content yet
                            pass

                        # Pattern 2: response.content_part.added
                        elif data.get("type") == "response.content_part.added":
                            # Content part started
                            pass

                        # Pattern 3: response.output_text.delta
                        elif data.get("type") == "response.output_text.delta":
                            content_delta = data.get("delta", "")

                        # Handle function calls in streaming
                        elif data.get("type") == "response.output_item.done":
                            output_item = data.get("output", {})
                            if output_item.get("type") == "function_call":
                                # Yield tool call chunk
                                yield litellm.ModelResponse(
                                    id=data.get("id", f"colin-{model}"),
                                    choices=[
                                        litellm.Choices(
                                            index=0,
                                            delta=DeltaType(
                                                role="assistant",
                                                content=None,
                                                tool_calls=[
                                                    {
                                                        "id": output_item.get("id", f"call_0"),
                                                        "type": "function",
                                                        "function": {
                                                            "name": output_item.get("name", ""),
                                                            "arguments": output_item.get("arguments", "{}"),
                                                        },
                                                    }
                                                ],
                                            ),
                                            finish_reason="tool_calls",
                                        )
                                    ],
                                    created=data.get("created_at", 0),
                                    model=model,
                                    object="chat.completion.chunk",
                                )

                        # Pattern 4: output_text.delta (alternative)
                        elif "delta" in data:
                            delta = data["delta"]
                            if isinstance(delta, dict):
                                content_delta = delta.get("content", "") or delta.get(
                                    "text", ""
                                )
                            elif isinstance(delta, str):
                                content_delta = delta

                        # Pattern 5: output.content.text delta
                        if not content_delta and data.get("output"):
                            for output in data.get("output", []):
                                if output.get("type") == "output_text":
                                    content_delta = output.get(
                                        "delta", ""
                                    ) or output.get("text", "")
                                    break

                        if content_delta:
                            yield litellm.ModelResponse(
                                id=data.get("id", f"colin-{model}"),
                                choices=[
                                    litellm.Choices(
                                        index=0,
                                        delta=DeltaType(
                                            role="assistant",
                                            content=content_delta,
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                                created=data.get("created_at", 0),
                                model=model,
                                object="chat.completion.chunk",
                            )
                    except json.JSONDecodeError:
                        lib_logger.debug(f"Failed to parse SSE data: {data_str[:100]}")
                        continue

            # Send final chunk
            yield litellm.ModelResponse(
                id=f"colin-{model}-done",
                choices=[
                    litellm.Choices(
                        index=0,
                        delta=DeltaType(),
                        finish_reason="stop",
                    )
                ],
                created=0,
                model=model,
                object="chat.completion.chunk",
            )

    async def get_auth_header(self, credential_identifier: str) -> Dict[str, str]:
        """Return Bearer token header for API key authentication."""
        return {"Authorization": f"Bearer {credential_identifier}"}
