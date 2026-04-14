# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

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
import time
from time import monotonic
import uuid
from typing import AsyncGenerator, Callable, Optional, Awaitable, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..transaction_logger import TransactionLogger

from ..utils.json_utils import json_dumps_str as json_dumps, json_loads, STREAM_DONE

logger = logging.getLogger("rotator_library.anthropic_compat")


class ChunkBatcher:
    """
    Batch SSE events for improved throughput.
    
    Buffers small events and flushes when:
    - Buffer size exceeds max_size
    - Time since last flush exceeds max_delay_ms
    """
    
    def __init__(self, max_size: int = 4096, max_delay_ms: int = 1):
        self.buffer = []
        self.current_size = 0
        self.max_size = max_size
        self.max_delay_ms = max_delay_ms
        self.last_flush = monotonic()
        self._first_event = True  # Flush first event immediately for low TTFT

    def add(self, event: str) -> Optional[str]:
        """
        Add event to buffer.

        Returns:
            Flushed buffer if threshold reached, None otherwise
        """
        self.buffer.append(event)
        self.current_size += len(event)

        # First event flushes immediately to minimize time-to-first-token
        if self._first_event:
            self._first_event = False
            return self.flush()

        if self.current_size >= self.max_size:
            return self.flush()

        # Check time-based flush
        elapsed_ms = (monotonic() - self.last_flush) * 1000
        if elapsed_ms >= self.max_delay_ms:
            return self.flush()

        return None
    
    def flush(self) -> str:
        """Flush buffer and return concatenated events."""
        if not self.buffer:
            return ""
        
        result = "".join(self.buffer)
        self.buffer.clear()
        self.current_size = 0
        self.last_flush = monotonic()
        return result


# Pre-built event templates for common operations (reduces allocations)
def _make_message_start_event(request_id: str, model: str, input_tokens: int = 0, cached_tokens: int = 0, cache_creation_tokens: int = 0) -> str:
    """Build message_start event string."""
    usage = {
        "input_tokens": input_tokens - cached_tokens - cache_creation_tokens,
        "output_tokens": 0,
    }
    if cached_tokens > 0:
        usage["cache_read_input_tokens"] = cached_tokens
    if cache_creation_tokens > 0:
        usage["cache_creation_input_tokens"] = cache_creation_tokens

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


def _make_message_delta_event(stop_reason: str, output_tokens: int = 0, cached_tokens: int = 0, cache_creation_tokens: int = 0) -> str:
    """Build message_delta event string."""
    usage = {"output_tokens": output_tokens}
    if cached_tokens > 0:
        usage["cache_read_input_tokens"] = cached_tokens
    if cache_creation_tokens > 0:
        usage["cache_creation_input_tokens"] = cache_creation_tokens

    event = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": usage,
    }
    return f"event: message_delta\ndata: {json_dumps(event)}\n\n"


def _make_message_stop_event() -> str:
    """Build message_stop event string."""
    return 'event: message_stop\ndata: {"type": "message_stop"}\n\n'


async def _finalize_anthropic_stream(
    batcher: ChunkBatcher,
    thinking_block_started: bool,
    content_block_started: bool,
    current_block_index: int,
    tool_block_indices: dict,
    tool_calls_by_index: dict,
    output_tokens: int,
    cached_tokens: int,
    cache_creation_tokens: int,
    transaction_logger: Optional["TransactionLogger"],
    request_id: str,
    original_model: str,
    _thinking_parts: list,
    _text_parts: list,
    input_tokens: int,
) -> AsyncGenerator[str, None]:
    """Close all open content blocks and send final Anthropic stream events."""
    if thinking_block_started:
        event_str = _make_content_block_stop_event(current_block_index)
        if event := batcher.add(event_str):
            yield event
        current_block_index += 1

    if content_block_started:
        event_str = _make_content_block_stop_event(current_block_index)
        if event := batcher.add(event_str):
            yield event
        current_block_index += 1

    for tc_index in sorted(tool_block_indices.keys()):
        block_idx = tool_block_indices[tc_index]
        event_str = _make_content_block_stop_event(block_idx)
        if event := batcher.add(event_str):
            yield event

    stop_reason = "tool_use" if tool_calls_by_index else "end_turn"

    event_str = _make_message_delta_event(stop_reason, output_tokens, cached_tokens, cache_creation_tokens)
    if event := batcher.add(event_str):
        yield event
    event_str = _make_message_stop_event()
    if event := batcher.add(event_str):
        yield event

    if remaining := batcher.flush():
        yield remaining

    if transaction_logger:
        _log_anthropic_response(
            transaction_logger, request_id, original_model,
            "".join(_thinking_parts), "".join(_text_parts),
            tool_calls_by_index, input_tokens, output_tokens,
            cached_tokens, cache_creation_tokens, stop_reason
        )


async def anthropic_streaming_wrapper_fast(
    openai_stream: AsyncGenerator[Any, None],
    original_model: str,
    request_id: Optional[str] = None,
    is_disconnected: Optional[Callable[[], Awaitable[bool]]] = None,
    transaction_logger: Optional["TransactionLogger"] = None,
    precomputed_input_tokens: Optional[int] = None,
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
        openai_stream: AsyncGenerator yielding OpenAI chunks (dict) or SSE strings
        original_model: The model name to include in responses
        request_id: Optional request ID (auto-generated if not provided)
        is_disconnected: Optional async callback that returns True if client disconnected
        transaction_logger: Optional TransactionLogger for logging the final Anthropic response
        precomputed_input_tokens: Optional pre-computed input token count. Used as fallback
            when provider doesn't return usage in stream (e.g., Kilocode without stream_options).
            This is critical for Claude Code's context management to work correctly.

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

    # Token tracking - use precomputed input tokens as fallback
    # This is critical for providers that don't return usage in stream (e.g., Kilocode)
    input_tokens = precomputed_input_tokens if precomputed_input_tokens is not None else 0
    output_tokens = 0
    cached_tokens = 0
    cache_creation_tokens = 0
    usage_received_from_provider = False  # Track if we got usage from provider

    # Accumulated content for logging (list + join for O(n) instead of += O(n^2))
    _text_parts: list[str] = []
    _thinking_parts: list[str] = []
    stop_reason_final = "end_turn"
    
    # Initialize chunk batcher for improved throughput
    batcher = ChunkBatcher(max_size=4096, max_delay_ms=1)
    
    # Heartbeat tracking to prevent connection timeouts
    last_event_time = monotonic()
    HEARTBEAT_INTERVAL = 30  # seconds

    _chunk_count = 0
    try:
        async for raw_chunk in openai_stream:
            # Check for client disconnection every 20 chunks (avoid per-chunk syscall overhead)
            _chunk_count += 1
            if is_disconnected is not None and _chunk_count % 20 == 0 and await is_disconnected():
                break

            # STREAM_DONE sentinel: stream is complete
            if raw_chunk is STREAM_DONE:
                async for event in _finalize_anthropic_stream(
                    batcher, thinking_block_started, content_block_started,
                    current_block_index, tool_block_indices, tool_calls_by_index,
                    output_tokens, cached_tokens, cache_creation_tokens,
                    transaction_logger, request_id, original_model,
                    _thinking_parts, _text_parts, input_tokens,
                ):
                    yield event
                stop_reason_final = "tool_use" if tool_calls_by_index else "end_turn"
                break

            # Dict chunk (new internal pipeline format)
            if isinstance(raw_chunk, dict):
                chunk = raw_chunk
            elif isinstance(raw_chunk, str):
                # Legacy SSE string format (backward compat)
                if not raw_chunk or not raw_chunk.startswith("data:"):
                    continue
                data_content = raw_chunk[5:].strip()
                if data_content == "[DONE]":
                    async for event in _finalize_anthropic_stream(
                        batcher, thinking_block_started, content_block_started,
                        current_block_index, tool_block_indices, tool_calls_by_index,
                        output_tokens, cached_tokens, cache_creation_tokens,
                        transaction_logger, request_id, original_model,
                        _thinking_parts, _text_parts, input_tokens,
                    ):
                        yield event
                    stop_reason_final = "tool_use" if tool_calls_by_index else "end_turn"
                    break

                try:
                    chunk = json_loads(data_content)
                except Exception:
                    continue
            else:
                continue

            # Send heartbeat if no events for a while
            current_time = monotonic()
            if current_time - last_event_time > HEARTBEAT_INTERVAL:
                yield ": heartbeat\n\n"
                last_event_time = current_time

            # Extract usage if present
            if "usage" in chunk and chunk["usage"]:
                usage = chunk["usage"]
                # Provider returned usage - use it (overrides precomputed)
                if usage.get("prompt_tokens"):
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    usage_received_from_provider = True
                output_tokens = usage.get("completion_tokens", output_tokens)
                # Extract cached tokens from prompt_tokens_details
                if usage.get("prompt_tokens_details"):
                    prompt_details = usage["prompt_tokens_details"]
                    cached_tokens = prompt_details.get(
                        "cached_tokens", cached_tokens
                    )
                    cache_creation_tokens = prompt_details.get(
                        "cache_creation_tokens", cache_creation_tokens
                    )

            # Send message_start on first chunk
            if not message_started:
                event_str = _make_message_start_event(request_id, original_model, input_tokens, cached_tokens, cache_creation_tokens)
                if event := batcher.add(event_str):
                    yield event
                message_started = True
                last_event_time = current_time

            choices = chunk.get("choices") or []
            if not choices:
                continue

            delta = choices[0].get("delta", {})

            # Handle reasoning/thinking content
            reasoning_content = delta.get("reasoning_content")
            if reasoning_content:
                if not thinking_block_started:
                    event_str = _make_content_block_start_event(current_block_index, "thinking")
                    if event := batcher.add(event_str):
                        yield event
                    thinking_block_started = True

                event_str = _make_thinking_delta_event(current_block_index, reasoning_content)
                if event := batcher.add(event_str):
                    yield event
                _thinking_parts.append(reasoning_content)
                last_event_time = current_time

            # Handle text content
            content = delta.get("content")
            if content:
                # Close thinking block if we were in one
                if thinking_block_started and not content_block_started:
                    event_str = _make_content_block_stop_event(current_block_index)
                    if event := batcher.add(event_str):
                        yield event
                    current_block_index += 1
                    thinking_block_started = False

                if not content_block_started:
                    event_str = _make_content_block_start_event(current_block_index, "text")
                    if event := batcher.add(event_str):
                        yield event
                    content_block_started = True

                event_str = _make_text_delta_event(current_block_index, content)
                if event := batcher.add(event_str):
                    yield event
                _text_parts.append(content)
                last_event_time = current_time

            # Handle tool calls
            tool_calls = delta.get("tool_calls") or []
            for tc in tool_calls:
                tc_index = tc.get("index", 0)

                if tc_index not in tool_calls_by_index:
                    # Close previous blocks
                    if thinking_block_started:
                        event_str = _make_content_block_stop_event(current_block_index)
                        if event := batcher.add(event_str):
                            yield event
                        current_block_index += 1
                        thinking_block_started = False

                    if content_block_started:
                        event_str = _make_content_block_stop_event(current_block_index)
                        if event := batcher.add(event_str):
                            yield event
                        current_block_index += 1
                        content_block_started = False

                    # Start new tool use block
                    tool_calls_by_index[tc_index] = {
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": tc.get("function", {}).get("name", ""),
                        "arguments": "",
                    }
                    tool_block_indices[tc_index] = current_block_index

                    event_str = _make_content_block_start_event(
                        current_block_index,
                        "tool_use",
                        id=tool_calls_by_index[tc_index]["id"],
                        name=tool_calls_by_index[tc_index]["name"],
                        input={},
                    )
                    if event := batcher.add(event_str):
                        yield event
                    current_block_index += 1

                # Accumulate arguments
                func = tc.get("function", {})
                if func.get("name"):
                    tool_calls_by_index[tc_index]["name"] = func["name"]
                args_entry = tool_calls_by_index[tc_index]
                if func.get("arguments"):
                    if "chunks" not in args_entry:
                        args_entry["chunks"] = []
                    args_entry["chunks"].append(func["arguments"])
                    event_str = _make_input_json_delta_event(
                        tool_block_indices[tc_index], func["arguments"]
                    )
                    if event := batcher.add(event_str):
                        yield event
                    last_event_time = current_time

    except Exception as e:
        logger.error(f"Error in Anthropic streaming wrapper: {e}")

        # Send error as visible text
        if not message_started:
            event_str = _make_message_start_event(request_id, original_model, input_tokens, cached_tokens, cache_creation_tokens)
            if event := batcher.add(event_str):
                yield event

            error_message = f"Error: {str(e)}"
        event_str = _make_content_block_start_event(current_block_index, "text")
        if event := batcher.add(event_str):
            yield event
        event_str = _make_text_delta_event(current_block_index, error_message)
        if event := batcher.add(event_str):
            yield event
        event_str = _make_content_block_stop_event(current_block_index)
        if event := batcher.add(event_str):
            yield event
        event_str = _make_message_delta_event("end_turn", 0, cached_tokens, cache_creation_tokens)
        if event := batcher.add(event_str):
            yield event
        event_str = _make_message_stop_event()
        if event := batcher.add(event_str):
            yield event

        # Flush any remaining events
        if remaining := batcher.flush():
            yield remaining

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
    cache_creation_tokens: int,
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
            args_str = "".join(tc.get("chunks", [])) if "chunks" in tc else tc.get("arguments", "{}")
            if not args_str:
                args_str = "{}"
            input_data = json_loads(args_str)
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
        "input_tokens": input_tokens - cached_tokens - cache_creation_tokens,
        "output_tokens": output_tokens,
    }
    if cached_tokens > 0:
        log_usage["cache_read_input_tokens"] = cached_tokens
    if cache_creation_tokens > 0:
        log_usage["cache_creation_input_tokens"] = cache_creation_tokens

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
