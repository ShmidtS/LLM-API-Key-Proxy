# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Optimized streaming wrapper for converting OpenAI streaming format to Anthropic streaming format.

This module provides a framework-agnostic streaming wrapper that converts
OpenAI SSE (Server-Sent Events) format to Anthropic's streaming format.

Performance optimizations:
- Uses orjson for fast JSON parsing (3-5x faster than stdlib json)
- Reuses parsed objects where possible
- Minimizes allocations in hot paths
- Pre-built templates for common events
"""

import logging
import uuid
from typing import AsyncGenerator, Callable, Optional, Awaitable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..transaction_logger import TransactionLogger

# Try to import orjson for faster JSON handling
try:
    import orjson

    def json_dumps(obj: Any) -> str:
        """Fast JSON serialization using orjson."""
        return orjson.dumps(obj).decode('utf-8')

    def json_loads(s: str) -> Any:
        """Fast JSON parsing using orjson."""
        return orjson.loads(s)

    HAS_ORJSON = True
except ImportError:
    import json

    def json_dumps(obj: Any) -> str:
        """Fallback JSON serialization using stdlib."""
        return json.dumps(obj)

    def json_loads(s: str) -> Any:
        """Fallback JSON parsing using stdlib."""
        return json.loads(s)

    HAS_ORJSON = False

logger = logging.getLogger("rotator_library.anthropic_compat")


# Pre-built event templates for common operations (reduces allocations)
def _make_message_start_event(request_id: str, model: str, input_tokens: int = 0, cached_tokens: int = 0) -> str:
    """Build message_start event string."""
    usage = {
        "input_tokens": input_tokens - cached_tokens,
        "output_tokens": 0,
    }
    if cached_tokens > 0:
        usage["cache_read_input_tokens"] = cached_tokens
        usage["cache_creation_input_tokens"] = 0

    message_start = {
        "type": "message_start",
        "message": {
            "id": request_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": usage,
        },
    }
    return f"event: message_start\ndata: {json_dumps(message_start)}\n\n"


def _make_content_block_start_event(index: int, block_type: str, **extra) -> str:
    """Build content_block_start event string."""
    block = {"type": block_type}
    if block_type == "text":
        block["text"] = ""
    elif block_type == "thinking":
        block["thinking"] = ""
    elif block_type == "tool_use":
        block.update(extra)

    event = {
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    }
    return f"event: content_block_start\ndata: {json_dumps(event)}\n\n"


def _make_text_delta_event(index: int, text: str) -> str:
    """Build text_delta event string."""
    event = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    return f"event: content_block_delta\ndata: {json_dumps(event)}\n\n"


def _make_thinking_delta_event(index: int, thinking: str) -> str:
    """Build thinking_delta event string."""
    event = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "thinking_delta", "thinking": thinking},
    }
    return f"event: content_block_delta\ndata: {json_dumps(event)}\n\n"


def _make_input_json_delta_event(index: int, partial_json: str) -> str:
    """Build input_json_delta event string."""
    event = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    }
    return f"event: content_block_delta\ndata: {json_dumps(event)}\n\n"


def _make_content_block_stop_event(index: int) -> str:
    """Build content_block_stop event string."""
    return f'event: content_block_stop\ndata: {{"type": "content_block_stop", "index": {index}}}\n\n'


def _make_message_delta_event(stop_reason: str, output_tokens: int = 0, cached_tokens: int = 0) -> str:
    """Build message_delta event string."""
    usage = {"output_tokens": output_tokens}
    if cached_tokens > 0:
        usage["cache_read_input_tokens"] = cached_tokens
        usage["cache_creation_input_tokens"] = 0

    event = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage,
    }
    return f"event: message_delta\ndata: {json_dumps(event)}\n\n"


def _make_message_stop_event() -> str:
    """Build message_stop event string."""
    return 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


async def anthropic_streaming_wrapper_fast(
    openai_stream: AsyncGenerator[str, None],
    original_model: str,
    request_id: Optional[str] = None,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    transaction_logger: Optional["TransactionLogger"] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert OpenAI streaming format to Anthropic streaming format (optimized version).

    This is a framework-agnostic wrapper that can be used with any async web framework.
    Instead of taking a FastAPI Request object, it accepts an optional callback function
    to check for client disconnection.

    Anthropic SSE events:
    - message_start: Initial message metadata
    - content_block_start: Start of a content block
    - content_block_delta: Content chunk
    - content_block_stop: End of a content block
    - message_delta: Final message metadata (stop_reason, usage)
    - message_stop: End of message

    Args:
        openai_stream: AsyncGenerator yielding OpenAI SSE format strings
        original_model: The model name to include in responses
        request_id: Optional request ID (auto-generated if not provided)
        is_disconnected: Optional async callback that returns True if client disconnected
        transaction_logger: Optional TransactionLogger for logging the final Anthropic response

    Yields:
        SSE format strings in Anthropic's streaming format
    """
    if request_id is None:
        request_id = f"msg_{uuid.uuid4().hex[:24]}"

    # State tracking
    message_started = False
    content_block_started = False
    thinking_block_started = False
    current_block_index = 0

    # Tool calls tracking
    tool_calls_by_index: dict = {}  # Track tool calls by their index
    tool_block_indices: dict = {}  # Track which block index each tool call uses

    # Token tracking
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0

    # Accumulated content for logging
    accumulated_text = ""
    accumulated_thinking = ""
    stop_reason_final = "end_turn"

    try:
        async for chunk_str in openai_stream:
            # Check for client disconnection if callback provided
            if is_disconnected is not None and await is_disconnected():
                break

            # Fast path: skip empty chunks and non-data lines
            if not chunk_str or not chunk_str.startswith("data:"):
                continue

            data_content = chunk_str[5:].strip()  # Skip "data:" prefix

            # Handle stream end
            if data_content == "[DONE]":
                # CRITICAL: Send message_start if we haven't yet
                if not message_started:
                    yield _make_message_start_event(request_id, original_model, input_tokens, cached_tokens)
                    message_started = True

                # Close any open thinking block
                if thinking_block_started:
                    yield _make_content_block_stop_event(current_block_index)
                    current_block_index += 1
                    thinking_block_started = False

                # Close any open text block
                if content_block_started:
                    yield _make_content_block_stop_event(current_block_index)
                    current_block_index += 1
                    content_block_started = False

                # Close all open tool_use blocks
                for tc_index in sorted(tool_block_indices.keys()):
                    block_idx = tool_block_indices[tc_index]
                    yield _make_content_block_stop_event(block_idx)

                # Determine stop_reason
                stop_reason = "tool_use" if tool_calls_by_index else "end_turn"
                stop_reason_final = stop_reason

                # Send final events
                yield _make_message_delta_event(stop_reason, output_tokens, cached_tokens)
                yield _make_message_stop_event()

                # Log if needed
                if transaction_logger:
                    _log_anthropic_response(
                        transaction_logger, request_id, original_model,
                        accumulated_thinking, accumulated_text,
                        tool_calls_by_index, input_tokens, output_tokens,
                        cached_tokens, stop_reason_final
                    )
                break

            # Parse chunk (fast path with orjson)
            try:
                chunk = json_loads(data_content)
            except Exception:
                continue

            # Extract usage if present
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)
                # Extract cached tokens from prompt_tokens_details
                if usage.get("prompt_tokens_details"):
                    cached_tokens = usage["prompt_tokens_details"].get(
                        "cached_tokens", cached_tokens
                    )

            # Send message_start on first chunk
            if not message_started:
                yield _make_message_start_event(request_id, original_model, input_tokens, cached_tokens)
                message_started = True

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # Handle reasoning/thinking content
            reasoning_content = delta.get("reasoning_content")
            if reasoning_content:
                if not thinking_block_started:
                    yield _make_content_block_start_event(current_block_index, "thinking")
                    thinking_block_started = True

                yield _make_thinking_delta_event(current_block_index, reasoning_content)
                accumulated_thinking += reasoning_content

            # Handle text content
            content = delta.get("content")
            if content:
                # Close thinking block if we were in one
                if thinking_block_started and not content_block_started:
                    yield _make_content_block_stop_event(current_block_index)
                    current_block_index += 1
                    thinking_block_started = False

                if not content_block_started:
                    yield _make_content_block_start_event(current_block_index, "text")
                    content_block_started = True

                yield _make_text_delta_event(current_block_index, content)
                accumulated_text += content

            # Handle tool calls
            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                tc_index = tc.get("index", 0)

                if tc_index not in tool_calls_by_index:
                    # Close previous blocks
                    if thinking_block_started:
                        yield _make_content_block_stop_event(current_block_index)
                        current_block_index += 1
                        thinking_block_started = False

                    if content_block_started:
                        yield _make_content_block_stop_event(current_block_index)
                        current_block_index += 1
                        content_block_started = False

                    # Start new tool use block
                    tool_calls_by_index[tc_index] = {
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": "",
                    }
                    tool_block_indices[tc_index] = current_block_index

                    yield _make_content_block_start_event(
                        current_block_index,
                        "tool_use",
                        id=tool_calls_by_index[tc_index]["id"],
                        name=tool_calls_by_index[tc_index]["name"],
                        input={},
                    )
                    current_block_index += 1

                # Accumulate arguments
                func = tc.get("function", {})
                if func.get("name"):
                    tool_calls_by_index[tc_index]["name"] = func["name"]
                if func.get("arguments"):
                    tool_calls_by_index[tc_index]["arguments"] += func["arguments"]
                    yield _make_input_json_delta_event(
                        tool_block_indices[tc_index], func["arguments"]
                    )

    except Exception as e:
        logger.error(f"Error in Anthropic streaming wrapper: {e}")

        # Send error as visible text
        if not message_started:
            yield _make_message_start_event(request_id, original_model, input_tokens, cached_tokens)

        error_message = f"Error: {str(e)}"
        yield _make_content_block_start_event(current_block_index, "text")
        yield _make_text_delta_event(current_block_index, error_message)
        yield _make_content_block_stop_event(current_block_index)
        yield _make_message_delta_event("end_turn", 0, cached_tokens)
        yield _make_message_stop_event()

        # Send formal error event
        error_event = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        yield f"event: error\ndata: {json_dumps(error_event)}\n\n"


def _log_anthropic_response(
    transaction_logger: "TransactionLogger",
    request_id: str,
    model: str,
    accumulated_thinking: str,
    accumulated_text: str,
    tool_calls_by_index: dict,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int,
    stop_reason: str,
) -> None:
    """Log the final Anthropic response."""
    # Build content blocks
    content_blocks = []

    if accumulated_thinking:
        content_blocks.append({
            "type": "thinking",
            "thinking": accumulated_thinking,
        })

    if accumulated_text:
        content_blocks.append({
            "type": "text",
            "text": accumulated_text,
        })

    # Add tool use blocks
    for tc_index in sorted(tool_calls_by_index.keys()):
        tc = tool_calls_by_index[tc_index]
        try:
            input_data = json_loads(tc.get("arguments", "{}"))
        except Exception:
            input_data = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": tc.get("name", ""),
            "input": input_data,
        })

    # Build usage
    log_usage = {
        "input_tokens": input_tokens - cached_tokens,
        "output_tokens": output_tokens,
    }
    if cached_tokens > 0:
        log_usage["cache_read_input_tokens"] = cached_tokens
        log_usage["cache_creation_input_tokens"] = 0

    anthropic_response = {
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": log_usage,
    }

    transaction_logger.log_response(
        anthropic_response,
        filename="anthropic_response.json",
    )


# Export the fast version as the default wrapper
anthropic_streaming_wrapper = anthropic_streaming_wrapper_fast
