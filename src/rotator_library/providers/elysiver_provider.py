# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Elysiver API Provider - OpenAI Responses API format.

This provider implements the OpenAI Responses API format used by Elysiver.
Unlike standard chat/completions, this uses the /responses endpoint.

Key differences from chat/completions:
- Uses 'input' instead of 'messages'
- Response structure: output[].content[].text
- Supports streaming via SSE
"""

import json
import os
import logging
from typing import List, Dict, Any, AsyncGenerator, Union
import aiohttp
import httpx
import litellm  # type: ignore[import-untyped]
import orjson
from litellm.types.utils import Delta as DeltaType, ChatCompletionMessageToolCall  # type: ignore[import-untyped]

from .provider_interface import ProviderInterface, strip_provider_prefix, build_bearer_headers
from .base_streaming_provider import parse_sse_stream
from ..timeout_config import TimeoutConfig
from ..utils.json_utils import json_loads

lib_logger = logging.getLogger("rotator_library")

ELYSIVER_DEFAULT_API_BASE = "https://elysiver.h-e.top/v1"


class ElysiverProvider(ProviderInterface):
    """
    Provider for Elysiver API using OpenAI Responses API format.

    Environment variables:
    ELYSIVER_API_BASE: API base URL (default: https://elysiver.h-e.top/v1)
    ELYSIVER_API_KEY_N: API key(s) for authentication
    """

    provider_env_name: str = "ELYSIVER"
    skip_cost_calculation: bool = True
    default_rotation_mode: str = "balanced"

    tier_priorities: Dict[str, int] = {
        "default": 1,
    }

    def __init__(self):
        self.api_base = os.getenv("ELYSIVER_API_BASE", ELYSIVER_DEFAULT_API_BASE)
        self.api_base = self.api_base.rstrip("/")
        lib_logger.info("Elysiver Provider initialized with base: %s", self.api_base)

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        try:
            url = f"{self.api_base}/models"
            headers = build_bearer_headers(api_key)
            response = await client.get(url, headers=headers, timeout=TimeoutConfig.provider_request())
            if response.status_code < 400:
                data = response.json()
                return [
                    f"elysiver/{m['id']}"
                    for m in data.get("data", [])
                    if isinstance(m, dict) and "id" in m
                ]
            lib_logger.warning(
                "Elysiver model discovery HTTP %d", response.status_code,
            )
        except aiohttp.ClientResponseError as ex:
            lib_logger.warning("Elysiver model discovery HTTP %d", ex.status)
        except aiohttp.ClientError as ex:
            lib_logger.warning("Elysiver model discovery connection issue: %s", ex)
        except httpx.HTTPStatusError as ex:
            lib_logger.warning("Elysiver model discovery HTTP %d", ex.response.status_code)
        except httpx.RequestError as ex:
            lib_logger.warning("Elysiver model discovery connection issue: %s", ex)
        except (orjson.JSONDecodeError, ValueError) as ex:
            lib_logger.warning("Elysiver model discovery parse failure: %s", ex)
        except Exception:
            lib_logger.warning("Elysiver model discovery failed", exc_info=True)
        return []

    def has_custom_logic(self) -> bool:
        return True

    async def acompletion(
        self, client: httpx.AsyncClient, **kwargs
    ) -> Union[litellm.ModelResponse, AsyncGenerator[litellm.ModelResponse, None]]:
        model = kwargs.get("model", "gpt-5.5")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)
        credential_identifier = kwargs.pop("credential_identifier", "")

        model = strip_provider_prefix(model)

        input_data = self._messages_to_input(messages)

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_data,
        }

        if kwargs.get("max_tokens"):
            payload["max_output_tokens"] = kwargs["max_tokens"]
        if kwargs.get("max_completion_tokens") and "max_output_tokens" not in payload:
            payload["max_output_tokens"] = kwargs["max_completion_tokens"]
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("top_p") is not None:
            payload["top_p"] = kwargs["top_p"]

        if kwargs.get("response_format", {}).get("type") == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}

        if kwargs.get("tools"):
            payload["tools"] = kwargs["tools"]

        if kwargs.get("instructions") or self._extract_system_message(messages):
            instructions = kwargs.get("instructions") or self._extract_system_message(messages)
            if instructions:
                payload["instructions"] = instructions

        url = f"{self.api_base}/responses"
        headers = build_bearer_headers(credential_identifier)

        lib_logger.debug("Elysiver request: model=%s stream=%s", model, stream)

        payload["stream"] = True

        if stream:
            return self._stream_response(client, url, headers, payload, model)
        else:
            return await self._non_stream_response(client, url, headers, payload, model)

    def _extract_system_message(self, messages: List[Dict[str, Any]]) -> str:
        parts = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
        return "\n".join(parts) if parts else ""

    def _messages_to_input(self, messages: List[Dict[str, Any]]) -> Any:
        """Convert chat messages to Responses API input format.

        Elysiver expects messages-style list: [{'role': 'user', 'content': '...'}]
        String input is NOT supported (causes convert_request_failed).
        """
        if not messages:
            return []

        non_system = [m for m in messages if m.get("role") != "system"]

        items = []
        for msg in non_system:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "tool":
                call_id = msg.get("tool_call_id", "")
                items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": content if isinstance(content, str) else json.dumps(content),
                })
            elif role == "assistant":
                tool_calls = msg.get("tool_calls")
                if tool_calls:
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        items.append({
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        })
                else:
                    text = content if isinstance(content, str) else str(content)
                    if text:
                        items.append({"role": "assistant", "content": text})
            else:
                text = content if isinstance(content, str) else str(content)
                items.append({"role": "user", "content": text})

        return items

    async def _non_stream_response(
        self,
        client: httpx.AsyncClient,
        url: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        model: str,
    ) -> litellm.ModelResponse:
        async with client.stream(
            "POST", url, headers=headers, json=payload
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                response.raise_for_status()

            full_content = ""
            response_id = f"elysiver-{model}"
            created_at = 0
            usage = {}
            tool_calls = []

            async for line in response.aiter_lines():
                if not line:
                    continue

                if line.startswith("event:"):
                    continue

                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json_loads(data_str)

                        if data.get("type") == "response.created":
                            resp = data.get("response", {})
                            response_id = resp.get("id", response_id)
                            created_at = resp.get("created_at", 0)

                        elif data.get("type") == "response.completed":
                            resp = data.get("response", {})
                            usage = resp.get("usage", {})
                            if resp.get("id"):
                                response_id = resp["id"]

                        elif data.get("type") == "response.output_text.delta":
                            full_content += data.get("delta", "")

                        elif data.get("type") == "response.output_item.done":
                            output_item = data.get("item", data.get("output", {}))
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

                    except (json.JSONDecodeError, ValueError):
                        lib_logger.debug("Failed to parse SSE data: %s", data_str[:100])
                        continue

            finish_reason = "tool_calls" if tool_calls else "stop"

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
        async with client.stream(
            "POST", url, headers=headers, json=payload
        ) as response:
            if response.status_code >= 400:
                await response.aread()
                response.raise_for_status()

            async for data in parse_sse_stream(response, provider_name="Elysiver"):
                content_delta = ""

                if data.get("type") == "response.output_text.delta":
                    content_delta = data.get("delta", "")

                elif data.get("type") == "response.output_item.done":
                    output_item = data.get("item", data.get("output", {}))
                    if output_item.get("type") == "function_call":
                        yield litellm.ModelResponse(
                            id=data.get("id", f"elysiver-{model}"),
                            choices=[
                                litellm.Choices(
                                    index=0,
                                    delta=DeltaType(
                                        role="assistant",
                                        content=None,
                                        tool_calls=[
                                            {
                                                "id": output_item.get("id", "call_0"),
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

                elif "delta" in data:
                    delta = data["delta"]
                    if isinstance(delta, dict):
                        content_delta = delta.get("content", "") or delta.get("text", "")
                    elif isinstance(delta, str):
                        content_delta = delta

                if content_delta:
                    yield litellm.ModelResponse(
                        id=data.get("id", f"elysiver-{model}"),
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

            yield litellm.ModelResponse(
                id=f"elysiver-{model}-done",
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
        return {"Authorization": f"Bearer {credential_identifier}"}
