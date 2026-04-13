# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import time
import logging
from typing import AsyncGenerator, Any, Optional

from fastapi import HTTPException, Request
from rotator_library import STREAM_DONE
from rotator_library.utils.json_utils import sse_data_event
from proxy_app.detailed_logger import RawIOLogger

import litellm


async def streaming_response_wrapper(
    request: Request,
    request_data: dict,
    response_stream: AsyncGenerator[Any, None],
    logger: Optional[RawIOLogger] = None,
) -> AsyncGenerator[str, None]:
    """
    Wraps a streaming response to log the full response after completion
    and ensures any errors during the stream are sent to the client.

    Receives dicts + STREAM_DONE sentinel from the client pipeline,
    serializes to SSE only at the HTTP yield boundary (single serialize).
    """
    # --- Inline aggregation state (eliminates second pass over response_chunks) ---
    final_message = {"role": "assistant"}
    _content_parts: list = []
    _generic_str_parts: dict = {}
    aggregated_tool_calls = {}  # index -> {type, id, function: {name_parts, args_parts}}
    usage_data = None
    finish_reason = None
    first_chunk_meta = None  # {id, created, model} from first chunk
    _chunk_count = 0
    _last_disconnect_check = time.monotonic()

    try:
        async for chunk in response_stream:
            _chunk_count += 1
            now = time.monotonic()
            if now - _last_disconnect_check > 1.0:
                if await request.is_disconnected():
                    logging.warning("Client disconnected, stopping stream.")
                    break
                _last_disconnect_check = now

            # STREAM_DONE sentinel: emit SSE [DONE] and stop
            if chunk is STREAM_DONE:
                yield b"data: [DONE]\n\n"
                continue

            # chunk is a dict — serialize to SSE only here (single serialize point)
            chunk_str = sse_data_event(chunk)
            yield chunk_str

            if not isinstance(chunk, dict):
                continue

            if logger:
                logger.log_stream_chunk(chunk)
            else:
                continue

            # --- Inline aggregation (single pass, no second iteration) ---
            # Capture metadata from first chunk
            if first_chunk_meta is None:
                first_chunk_meta = {
                    "id": chunk.get("id"),
                    "created": chunk.get("created"),
                    "model": chunk.get("model"),
                }

            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                for key, value in delta.items():
                    if value is None:
                        continue

                    if key == "content":
                        if value:
                            _content_parts.append(value)

                    elif key == "tool_calls":
                        for tc_chunk in value:
                            index = tc_chunk["index"]
                            if index not in aggregated_tool_calls:
                                aggregated_tool_calls[index] = {
                                    "type": "function",
                                    "function": {
                                        "name_parts": [],
                                        "args_parts": [],
                                    },
                                }
                            tc = aggregated_tool_calls[index]
                            if tc_chunk.get("id"):
                                tc["id"] = tc_chunk["id"]
                            if "function" in tc_chunk:
                                fn = tc_chunk["function"]
                                if fn.get("name") is not None:
                                    tc["function"]["name_parts"].append(fn["name"])
                                if fn.get("arguments") is not None:
                                    tc["function"]["args_parts"].append(
                                        fn["arguments"]
                                    )

                    elif key == "function_call":
                        if "function_call" not in final_message:
                            final_message["function_call"] = {
                                "_name_parts": [],
                                "_args_parts": [],
                            }
                        if value.get("name") is not None:
                            final_message["function_call"]["_name_parts"].append(
                                value["name"]
                            )
                        if value.get("arguments") is not None:
                            final_message["function_call"]["_args_parts"].append(
                                value["arguments"]
                            )

                    else:  # Generic key handling for other data like 'reasoning'
                        if key == "role":
                            final_message[key] = value
                        elif isinstance(value, str):
                            if key not in _generic_str_parts:
                                _generic_str_parts[key] = []
                            _generic_str_parts[key].append(value)
                        else:
                            final_message[key] = value

                if "finish_reason" in choice and choice["finish_reason"]:
                    finish_reason = choice["finish_reason"]

            if "usage" in chunk and chunk["usage"]:
                usage_data = chunk["usage"]
    except Exception as e:
        logging.error(f"An error occurred during the response stream: {e}")
        # Yield a final error message to the client to ensure they are not left hanging.
        error_payload = {
            "error": {
                "message": f"An unexpected error occurred during the stream: {str(e)}",
                "type": "proxy_internal_error",
                "code": 500,
            }
        }
        yield sse_data_event(error_payload)
        yield b"data: [DONE]\n\n"
        # Also log this as a failed request
        if logger:
            logger.log_final_response(
                status_code=500, headers=None, body={"error": str(e)}
            )
        return  # Stop further processing
    finally:
        if logger:
            if first_chunk_meta is not None:
                # --- Join accumulated string parts ---
                if _content_parts:
                    final_message["content"] = "".join(_content_parts)

                for key, parts in _generic_str_parts.items():
                    final_message[key] = "".join(parts)

                # Flatten tool_calls: convert name_parts/args_parts to strings
                if aggregated_tool_calls:
                    tool_calls_list = []
                    for tc in aggregated_tool_calls.values():
                        fn = tc["function"]
                        tool_calls_list.append(
                            {
                                "id": tc.get("id"),
                                "type": tc["type"],
                                "function": {
                                    "name": "".join(fn["name_parts"]),
                                    "arguments": "".join(fn["args_parts"]),
                                },
                            }
                        )
                    final_message["tool_calls"] = tool_calls_list
                    # CRITICAL FIX: Override finish_reason when tool_calls exist
                    # This ensures OpenCode and other agentic systems continue the conversation loop
                    finish_reason = "tool_calls"

                if "function_call" in final_message:
                    fc = final_message["function_call"]
                    final_message["function_call"] = {
                        "name": "".join(fc["_name_parts"]),
                        "arguments": "".join(fc["_args_parts"]),
                    }

                # Ensure standard fields are present for consistent logging
                for field in ["content", "tool_calls", "function_call"]:
                    if field not in final_message:
                        final_message[field] = None

                final_choice = {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": finish_reason,
                }

                full_response = {
                    "id": first_chunk_meta.get("id"),
                    "object": "chat.completion",
                    "created": first_chunk_meta.get("created"),
                    "model": first_chunk_meta.get("model"),
                    "choices": [final_choice],
                    "usage": usage_data,
                }
            else:
                full_response = {}

            logger.log_final_response(
                status_code=200,
                headers=None,  # Headers are not available at this stage
                body=full_response,
            )


def handle_litellm_error(e: Exception, error_format: str = "openai") -> HTTPException:
    """Map litellm exceptions to HTTPException with OpenAI or Anthropic error format."""
    _ERROR_MAP = [
        (
            (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError),
            400,
            "Invalid Request",
            "invalid_request_error",
        ),
        ((litellm.AuthenticationError,), 401, "Authentication Error", "authentication_error"),
        ((litellm.RateLimitError,), 429, "Rate Limit Exceeded", "rate_limit_error"),
        (
            (litellm.ServiceUnavailableError, litellm.APIConnectionError),
            503,
            "Service Unavailable",
            "api_error",
        ),
        ((litellm.Timeout,), 504, "Gateway Timeout", "api_error"),
        ((litellm.InternalServerError, litellm.OpenAIError), 502, "Bad Gateway", "api_error"),
    ]

    for exc_types, status_code, openai_label, anthropic_error_type in _ERROR_MAP:
        if isinstance(e, exc_types):
            if error_format == "openai":
                detail = f"{openai_label}: {str(e)}"
            else:
                message = f"Request timed out: {str(e)}" if isinstance(e, litellm.Timeout) else str(e)
                detail = {
                    "type": "error",
                    "error": {"type": anthropic_error_type, "message": message},
                }
            return HTTPException(status_code=status_code, detail=detail)

    # Fallback for unmatched litellm errors
    if error_format == "openai":
        return HTTPException(status_code=500, detail=str(e))
    return HTTPException(
        status_code=500,
        detail={"type": "error", "error": {"type": "api_error", "message": str(e)}},
    )
