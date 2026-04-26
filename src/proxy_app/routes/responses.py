# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging
import time
import uuid
from typing import Any, AsyncGenerator

import orjson
from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.streaming import streaming_response_wrapper, make_sse_response
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors
from rotator_library.utils.json_utils import sse_data_event

router = APIRouter(tags=["responses"])
logger = logging.getLogger(__name__)

_RESPONSES_ONLY_FIELDS = {
    "background",
    "client_metadata",
    "conversation",
    "include",
    "metadata",
    "output_text",
    "parallel_tool_calls",
    "previous_response_id",
    "prompt",
    "prompt_cache_key",
    "reasoning",
    "safety_identifier",
    "service_tier",
    "store",
    "truncation",
}


@router.post("/v1/responses")
@handle_route_errors(error_format="openai", log_context="Responses API request failed")
async def create_response(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI Responses API endpoint.

    Accepts OpenAI Responses API input, translates it to Chat Completions for
    the rotator/LiteLLM pipeline, then translates the result back to Responses
    API shape for clients such as Codex CLI.
    """
    request_data = orjson.loads(await request.body())
    chat_request_data = _responses_request_to_chat_request(request_data)

    logger.info(
        "Responses request normalized: model=%s stream=%s max_tokens=%s max_completion_tokens=%s has_max_output_tokens=%s tools=%s input_type=%s",
        chat_request_data.get("model"),
        chat_request_data.get("stream"),
        chat_request_data.get("max_tokens"),
        chat_request_data.get("max_completion_tokens"),
        "max_output_tokens" in request_data,
        len(chat_request_data.get("tools", [])) if isinstance(chat_request_data.get("tools"), list) else 0,
        type(request_data.get("input")).__name__ if request_data.get("input") is not None else "none",
    )

    log_request_to_console(
        url=str(request.url),
        client_info=(request.client.host, request.client.port),
        request_data=chat_request_data,
    )

    is_streaming = chat_request_data.get("stream", False)

    if is_streaming:
        response_generator = client.acompletion(request=request, **chat_request_data)
        chat_sse_stream = streaming_response_wrapper(request, response_generator)
        return make_sse_response(
            _chat_sse_to_responses_sse(
                chat_sse_stream,
                model=chat_request_data.get("model", ""),
            )
        )

    response = await client.acompletion(request=request, **chat_request_data)
    return _chat_completion_to_response(
        response,
        model=chat_request_data.get("model", ""),
    )


def _responses_request_to_chat_request(request_data: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI Responses API request into Chat Completions kwargs."""
    chat_data = dict(request_data)
    _normalize_response_model(chat_data)
    _normalize_response_max_tokens(chat_data)

    input_data = chat_data.pop("input", None)
    messages = chat_data.pop("messages", None) or _input_to_messages(input_data)

    instructions = chat_data.pop("instructions", None)
    if instructions and isinstance(instructions, str):
        messages.insert(0, {"role": "system", "content": instructions})

    chat_data["messages"] = messages

    max_output_tokens = chat_data.pop("max_output_tokens", None)
    if (
        max_output_tokens is not None
        and "max_tokens" not in chat_data
        and "max_completion_tokens" not in chat_data
    ):
        chat_data["max_tokens"] = max_output_tokens

    text_config = chat_data.pop("text", None)
    response_format = _text_config_to_response_format(text_config)
    if response_format and "response_format" not in chat_data:
        chat_data["response_format"] = response_format

    if "tools" in chat_data:
        tools = _responses_tools_to_chat_tools(chat_data.get("tools"))
        if tools:
            chat_data["tools"] = tools
        else:
            chat_data.pop("tools", None)
            chat_data.pop("tool_choice", None)

    if "tool_choice" in chat_data:
        chat_data["tool_choice"] = _responses_tool_choice_to_chat_tool_choice(
            chat_data["tool_choice"]
        )

    for field in _RESPONSES_ONLY_FIELDS:
        chat_data.pop(field, None)

    return chat_data


def _normalize_response_model(chat_data: dict[str, Any]) -> None:
    """Codex/OpenAI clients send bare model IDs; the rotator requires provider/model."""
    model = chat_data.get("model")
    if isinstance(model, str) and model.strip() and "/" not in model:
        chat_data["model"] = f"openai/{model.strip()}"


def _normalize_response_max_tokens(chat_data: dict[str, Any]) -> None:
    """Prevent Responses requests from inheriting an oversized auto-generated max_tokens.

    The rotator auto-fills max_tokens from model context when the client omits it.
    Responses API callers such as Codex CLI often omit token limits, which can lead
    to upstream 400 errors on chat/completions-compatible backends. Use a conservative
    default only when no explicit limit was provided.
    """
    if (
        "max_output_tokens" in chat_data
        or "max_tokens" in chat_data
        or "max_completion_tokens" in chat_data
    ):
        return
    chat_data["max_tokens"] = 4096


def _input_to_messages(input_data) -> list:
    """
    Convert Responses API 'input' field to chat/completions 'messages' format.

    The Responses API accepts strings, message items, content-part lists,
    function_call items, and function_call_output items.
    """
    if input_data is None:
        return []

    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    if isinstance(input_data, list):
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                _append_response_input_item(messages, item)
        return messages

    return [{"role": "user", "content": str(input_data)}]


def _append_response_input_item(messages: list[dict[str, Any]], item: dict[str, Any]) -> None:
    item_type = item.get("type")

    if item_type == "message":
        role = item.get("role", "user")
        _append_chat_message(messages, role, item.get("content", ""), item)
        return

    if item_type in ("input_text", "output_text"):
        role = "assistant" if item_type == "output_text" else "user"
        messages.append({"role": role, "content": item.get("text", "")})
        return

    if item_type == "function_call_output":
        call_id = item.get("call_id") or item.get("id") or item.get("tool_call_id")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _stringify_tool_output(item.get("output", "")),
            }
        )
        return

    if item_type == "function_call":
        call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                ],
            }
        )
        return

    role = item.get("role", "user")
    _append_chat_message(messages, role, item.get("content", item.get("text", "")), item)


def _append_chat_message(
    messages: list[dict[str, Any]],
    role: str,
    content: Any,
    source: dict[str, Any],
) -> None:
    if role not in ("user", "assistant", "system", "developer", "tool"):
        return

    mapped_role = "system" if role == "developer" else role
    message = {"role": mapped_role, "content": _content_to_chat_content(content)}

    if mapped_role == "tool":
        message["tool_call_id"] = (
            source.get("tool_call_id") or source.get("call_id") or source.get("id")
        )
        message["content"] = _stringify_tool_output(content)

    messages.append(message)


def _content_to_chat_content(content: Any) -> Any:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        text_parts = []
        has_non_text = False

        for part in content:
            converted = _content_part_to_chat_part(part)
            if converted is None:
                continue
            if isinstance(converted, str):
                text_parts.append(converted)
                parts.append({"type": "text", "text": converted})
            else:
                has_non_text = True
                parts.append(converted)

        if not has_non_text:
            return "\n".join(text_parts)
        return parts

    if isinstance(content, dict):
        converted = _content_part_to_chat_part(content)
        if isinstance(converted, str):
            return converted
        if converted is not None:
            return [converted]

    return "" if content is None else str(content)


def _content_part_to_chat_part(part: Any) -> Any:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return None

    part_type = part.get("type")
    if part_type in ("input_text", "output_text", "text"):
        return part.get("text", "")

    if part_type == "input_image":
        image_url = part.get("image_url") or part.get("url")
        if isinstance(image_url, dict):
            return {"type": "image_url", "image_url": image_url}
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}

    return None


def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    if not isinstance(tools, list):
        return []

    chat_tools = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue

        if isinstance(tool.get("function"), dict):
            chat_tools.append(tool)
            continue

        function = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
        }
        if "strict" in tool:
            function["strict"] = tool["strict"]

        chat_tools.append({"type": "function", "function": function})

    return chat_tools


def _responses_tool_choice_to_chat_tool_choice(tool_choice: Any) -> Any:
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") != "function":
        return tool_choice
    if isinstance(tool_choice.get("function"), dict):
        return tool_choice

    name = tool_choice.get("name")
    if not name:
        return tool_choice
    return {"type": "function", "function": {"name": name}}


def _text_config_to_response_format(text_config: Any) -> dict[str, Any] | None:
    if not isinstance(text_config, dict):
        return None

    text_format = text_config.get("format")
    if not isinstance(text_format, dict):
        return None

    format_type = text_format.get("type")
    if format_type in (None, "text"):
        return None
    return text_format


def _stringify_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    try:
        return orjson.dumps(output).decode("utf-8")
    except TypeError:
        return str(output)


async def _chat_sse_to_responses_sse(
    chat_sse_stream: AsyncGenerator[str | bytes, None],
    *,
    model: str,
) -> AsyncGenerator[bytes, None]:
    state = _new_responses_stream_state(model)

    try:
        async for event in chat_sse_stream:
            payload = _parse_sse_payload(event)
            if payload is None:
                continue
            if payload == "[DONE]":
                for response_event in _complete_response_stream_events(state):
                    yield _responses_sse_event(response_event)
                yield b"data: [DONE]\n\n"
                return

            for response_event in _chat_chunk_to_response_stream_events(payload, state):
                yield _responses_sse_event(response_event)

        for response_event in _complete_response_stream_events(state):
            yield _responses_sse_event(response_event)
        yield b"data: [DONE]\n\n"
    except GeneratorExit:
        if hasattr(chat_sse_stream, "aclose"):
            await chat_sse_stream.aclose()
        return
    finally:
        if hasattr(chat_sse_stream, "aclose"):
            try:
                await chat_sse_stream.aclose()
            except Exception:
                pass


def _parse_sse_payload(event: str | bytes) -> dict[str, Any] | str | None:
    text = event.decode("utf-8") if isinstance(event, bytes) else event
    data_lines = []
    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if not data_lines:
        return None

    data = "\n".join(data_lines)
    if data == "[DONE]":
        return data
    return orjson.loads(data)


def _new_responses_stream_state(model: str) -> dict[str, Any]:
    return {
        "response_id": f"resp_{uuid.uuid4().hex}",
        "created_at": int(time.time()),
        "model": model,
        "started": False,
        "finished": False,
        "message_item_id": f"msg_{uuid.uuid4().hex}",
        "message_output_index": None,
        "text_started": False,
        "text_parts": [],
        "tool_calls": {},
        "output_sequence": [],
        "usage": None,
        "finish_reason": None,
    }


def _chat_chunk_to_response_stream_events(
    chunk: dict[str, Any],
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    events = _ensure_response_stream_started(state, chunk)

    if "error" in chunk and "choices" not in chunk:
        state["finished"] = True
        events.append(
            {
                "type": "response.failed",
                "response": _responses_stream_response(
                    state,
                    status="failed",
                    output=_build_response_output(state),
                    error=chunk["error"],
                ),
            }
        )
        return events

    usage = chunk.get("usage")
    if usage:
        state["usage"] = _convert_usage(usage)

    choices = chunk.get("choices") or []
    if not choices:
        return events

    choice = choices[0]
    delta = choice.get("delta") or {}
    content_delta = delta.get("content")
    if content_delta:
        events.extend(_append_text_delta_events(state, content_delta))

    tool_calls = delta.get("tool_calls") or []
    if tool_calls:
        events.extend(_append_tool_call_delta_events(state, tool_calls))

    if choice.get("finish_reason"):
        state["finish_reason"] = choice["finish_reason"]

    return events


def _ensure_response_stream_started(
    state: dict[str, Any],
    chunk: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if state["started"]:
        return []

    if chunk:
        state["created_at"] = chunk.get("created") or state["created_at"]
        state["model"] = chunk.get("model") or state["model"]

    state["started"] = True
    response = _responses_stream_response(state, status="in_progress", output=[])
    return [
        {"type": "response.created", "response": response},
        {"type": "response.in_progress", "response": response},
    ]


def _append_text_delta_events(state: dict[str, Any], delta: str) -> list[dict[str, Any]]:
    events = []
    if not state["text_started"]:
        state["text_started"] = True
        state["message_output_index"] = len(state["output_sequence"])
        state["output_sequence"].append(("message", None))
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": state["message_output_index"],
                    "item": _message_output_item(state, status="in_progress"),
                },
                {
                    "type": "response.content_part.added",
                    "item_id": state["message_item_id"],
                    "output_index": state["message_output_index"],
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            ]
        )

    state["text_parts"].append(delta)
    events.append(
        {
            "type": "response.output_text.delta",
            "item_id": state["message_item_id"],
            "output_index": state["message_output_index"],
            "content_index": 0,
            "delta": delta,
        }
    )
    return events


def _append_tool_call_delta_events(
    state: dict[str, Any],
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    events = []
    for tool_call in tool_calls:
        index = tool_call.get("index", len(state["tool_calls"]))
        function = tool_call.get("function") or {}
        tool_state = state["tool_calls"].get(index)

        if tool_state is None:
            tool_state = {
                "id": tool_call.get("id") or f"fc_{uuid.uuid4().hex}",
                "name_parts": [],
                "arguments_parts": [],
                "output_index": len(state["output_sequence"]),
            }
            state["tool_calls"][index] = tool_state
            state["output_sequence"].append(("tool", index))
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": tool_state["output_index"],
                    "item": _tool_output_item(tool_state, status="in_progress"),
                }
            )

        if function.get("name"):
            tool_state["name_parts"].append(function["name"])

        arguments_delta = function.get("arguments") or ""
        if arguments_delta:
            tool_state["arguments_parts"].append(arguments_delta)
            events.append(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": tool_state["id"],
                    "output_index": tool_state["output_index"],
                    "delta": arguments_delta,
                }
            )

    return events


def _complete_response_stream_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    if state["finished"]:
        return []

    events = _ensure_response_stream_started(state)

    if state["text_started"]:
        text = "".join(state["text_parts"])
        part = {"type": "output_text", "text": text, "annotations": []}
        events.extend(
            [
                {
                    "type": "response.output_text.done",
                    "item_id": state["message_item_id"],
                    "output_index": state["message_output_index"],
                    "content_index": 0,
                    "text": text,
                },
                {
                    "type": "response.content_part.done",
                    "item_id": state["message_item_id"],
                    "output_index": state["message_output_index"],
                    "content_index": 0,
                    "part": part,
                },
                {
                    "type": "response.output_item.done",
                    "output_index": state["message_output_index"],
                    "item": _message_output_item(state, status="completed"),
                },
            ]
        )

    for _, index in state["output_sequence"]:
        if index is None:
            continue
        tool_state = state["tool_calls"][index]
        events.extend(
            [
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": tool_state["id"],
                    "output_index": tool_state["output_index"],
                    "arguments": "".join(tool_state["arguments_parts"]),
                },
                {
                    "type": "response.output_item.done",
                    "output_index": tool_state["output_index"],
                    "item": _tool_output_item(tool_state, status="completed"),
                },
            ]
        )

    state["finished"] = True
    events.append(
        {
            "type": "response.completed",
            "response": _responses_stream_response(
                state,
                status="completed",
                output=_build_response_output(state),
            ),
        }
    )
    return events


def _responses_stream_response(
    state: dict[str, Any],
    *,
    status: str,
    output: list[dict[str, Any]],
    error: Any = None,
) -> dict[str, Any]:
    return {
        "id": state["response_id"],
        "object": "response",
        "created_at": state["created_at"],
        "status": status,
        "model": state["model"],
        "output": output,
        "parallel_tool_calls": True,
        "error": error,
        "incomplete_details": None,
        "usage": state.get("usage"),
    }


def _build_response_output(state: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for kind, index in state["output_sequence"]:
        if kind == "message":
            output.append(_message_output_item(state, status="completed"))
        elif kind == "tool":
            output.append(_tool_output_item(state["tool_calls"][index], status="completed"))
    return output


def _message_output_item(state: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "id": state["message_item_id"],
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [
            {
                "type": "output_text",
                "text": "".join(state["text_parts"]),
                "annotations": [],
            }
        ],
    }


def _tool_output_item(tool_state: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "id": tool_state["id"],
        "type": "function_call",
        "status": status,
        "call_id": tool_state["id"],
        "name": "".join(tool_state["name_parts"]),
        "arguments": "".join(tool_state["arguments_parts"]),
    }


def _responses_sse_event(payload: dict[str, Any]) -> bytes:
    event_type = payload.get("type")
    if not event_type:
        return sse_data_event(payload)
    return b"".join(
        (b"event: ", event_type.encode("utf-8"), b"\n", sse_data_event(payload))
    )


def _chat_completion_to_response(response: Any, *, model: str) -> dict[str, Any]:
    data = response.model_dump() if hasattr(response, "model_dump") else response
    if isinstance(data, dict) and "error" in data and "choices" not in data:
        return data
    if not isinstance(data, dict):
        data = {}

    response_id = data.get("id") or f"resp_{uuid.uuid4().hex}"
    created_at = data.get("created") or int(time.time())
    response_model = data.get("model") or model
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}

    output = []
    output_text = _message_content_to_output_text(message.get("content"))
    if output_text:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": output_text, "annotations": []}
                ],
            }
        )

    for tool_call in message.get("tool_calls") or []:
        function = tool_call.get("function") or {}
        tool_id = tool_call.get("id") or f"fc_{uuid.uuid4().hex}"
        output.append(
            {
                "id": tool_id,
                "type": "function_call",
                "status": "completed",
                "call_id": tool_id,
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            }
        )

    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": "completed",
        "model": response_model,
        "output": output,
        "output_text": output_text,
        "parallel_tool_calls": True,
        "error": None,
        "incomplete_details": None,
        "usage": _convert_usage(data.get("usage")),
    }


def _message_content_to_output_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _convert_usage(usage: Any) -> dict[str, Any] | None:
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return None

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
