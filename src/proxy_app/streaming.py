# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
import logging
from typing import AsyncGenerator, Any, Optional

import litellm  # type: ignore[import-untyped]
from litellm.exceptions import (  # type: ignore[import-untyped]
    InvalidRequestError,
    BadRequestError,
    ContextWindowExceededError,
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    APIConnectionError,
    Timeout,
    InternalServerError,
    OpenAIError,
)

_GENERIC_STREAM_ERROR_MESSAGE = "An unexpected error occurred during the stream"

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from rotator_library import STREAM_DONE
from rotator_library.error_types import ClassifiedError
from rotator_library.utils.json_utils import sse_data_event
from rotator_library.utils.chunk_aggregator import ChunkAggregator
from proxy_app.dependencies import _inc_streams, _dec_streams
from proxy_app.detailed_logger import RawIOLogger

logger = logging.getLogger(__name__)


def _get_litellm_error_map():
    return [
        (
            (
                InvalidRequestError,
                BadRequestError,
                ValueError,
                ContextWindowExceededError,
            ),
            400,
            "Invalid Request",
            "invalid_request_error",
        ),
        ((AuthenticationError,), 401, "Authentication Error", "authentication_error"),
        ((NotFoundError,), 404, "Not Found", "invalid_request_error"),
        ((RateLimitError,), 429, "Rate Limit Exceeded", "rate_limit_error"),
        (
            (ServiceUnavailableError, APIConnectionError),
            503,
            "Service Unavailable",
            "api_error",
        ),
        ((Timeout,), 504, "Gateway Timeout", "api_error"),
        ((InternalServerError, OpenAIError), 502, "Bad Gateway", "api_error"),
    ]


def get_litellm_error_types():
    return tuple(exc_type for row in _get_litellm_error_map() for exc_type in row[0])


async def streaming_response_wrapper(
    request: Request,
    response_stream: AsyncGenerator[Any, None],
    logger: Optional[RawIOLogger] = None,
) -> AsyncGenerator[str | bytes, None]:
    """
    Wraps a streaming response to log the full response after completion
    and ensures any errors during the stream are sent to the client.

    Receives dicts + STREAM_DONE sentinel from the client pipeline,
    serializes to SSE only at the HTTP yield boundary (single serialize).
    """
    # --- Aggregation state: only needed when logger is active ---
    aggregator = ChunkAggregator() if logger is not None else None
    final_status_code = 200
    final_error_log_context = None
    log_task: Optional[asyncio.Task] = None

    if logger is not None:
        log_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        log_done = asyncio.Event()

        async def _log_drain():
            while not log_done.is_set() or not log_queue.empty():
                try:
                    chunk = await asyncio.wait_for(log_queue.get(), timeout=0.1)
                    await logger.log_stream_chunk(chunk)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

        log_task = asyncio.create_task(_log_drain())

    # Track active streaming connections for graceful shutdown
    # NOTE: track_stream() in dependencies.py also increments the counter.
    # When both wrappers are used, only track_stream should own the counter.
    # Here we only increment if track_stream is NOT in the call chain.
    _owns_counter = not getattr(request, '_stream_tracked', False)
    if _owns_counter:
        try:
            await _inc_streams(request)
        except AttributeError:
            logging.debug("stream_response: request lacks stream counter attribute on increment")

    try:
        _chunk_count = 0
        _bytes_since_yield = 0
        _bytes_streamed = 0
        _buffer: list[bytes] = []
        _buffer_size = 0
        _FIRST_IMMEDIATE_BYTES = 2048
        _ADAPTIVE_CHUNK_INTERVAL = 16
        _ADAPTIVE_FLUSH_BYTES = 32768
        async for chunk in response_stream:

            # STREAM_DONE sentinel: flush buffer, emit SSE [DONE] and stop
            if chunk is STREAM_DONE:
                if _buffer:
                    yield b"".join(_buffer)
                    _buffer.clear()
                    _buffer_size = 0
                yield b"data: [DONE]\n\n"
                return

            if not isinstance(chunk, dict):
                logging.warning("Unexpected chunk type %s in stream, skipping", type(chunk).__name__)
                continue

            # chunk is a dict — serialize to SSE only here (single serialize point)
            chunk_str = sse_data_event(chunk)
            _buffer.append(chunk_str)
            _buffer_size += len(chunk_str)
            _chunk_count += 1
            _bytes_since_yield += len(chunk_str)

            if _chunk_count == 1 or _bytes_streamed < _FIRST_IMMEDIATE_BYTES:
                yield b"".join(_buffer)
                _bytes_streamed += _buffer_size
                _buffer.clear()
                _buffer_size = 0
                _bytes_since_yield = 0
            elif _chunk_count % _ADAPTIVE_CHUNK_INTERVAL == 0 or _bytes_since_yield >= _ADAPTIVE_FLUSH_BYTES:
                yield b"".join(_buffer)
                _bytes_streamed += _buffer_size
                _buffer.clear()
                _buffer_size = 0
                _bytes_since_yield = 0

            if logger is not None:
                try:
                    log_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass
                if aggregator is not None:
                    aggregator.add_chunk(chunk)
    except (GeneratorExit, asyncio.CancelledError):
        if hasattr(response_stream, "aclose"):
            await response_stream.aclose()
        _buffer.clear()
        raise
    except Exception as e:
        logging.exception("Error during response stream")
        # Flush any buffered chunks before sending error
        if _buffer:
            yield b"".join(_buffer)
            _buffer.clear()
        # Propagate classified error type so clients can distinguish 429/503/502
        if isinstance(e, ClassifiedError) and e.status_code:
            final_status_code = e.status_code
            final_error_log_context = {
                "type": e.error_type,
                "code": e.status_code,
            }
            error_payload = {
                "error": {
                    "message": _GENERIC_STREAM_ERROR_MESSAGE,
                    "type": e.error_type,
                    "code": e.status_code,
                }
            }
        else:
            final_status_code = 500
            final_error_log_context = {
                "type": "proxy_internal_error",
                "code": 500,
            }
            error_payload = {
                "error": {
                    "message": _GENERIC_STREAM_ERROR_MESSAGE,
                    "type": "proxy_internal_error",
                    "code": 500,
                }
            }
        logging.error(
            "Stream failed: type=%s code=%s",
            final_error_log_context["type"] if final_error_log_context else "proxy_internal_error",
            final_error_log_context["code"] if final_error_log_context else 500,
        )
        yield sse_data_event(error_payload)
        yield b"data: [DONE]\n\n"
        return  # Stop further processing
    finally:
        if log_task is not None:
            log_done.set()
            try:
                await log_task
            except Exception:
                pass
        if _owns_counter:
            try:
                await _dec_streams(request)
            except AttributeError:
                logging.debug("stream_response: request lacks stream counter attribute on decrement")
        if logger and aggregator is not None:
            try:
                await logger.log_final_response(
                    status_code=final_status_code,
                    headers=None,
                    body=aggregator.build_response_dict() if final_status_code == 200 else {"error": {"message": _GENERIC_STREAM_ERROR_MESSAGE, "type": final_error_log_context["type"] if final_error_log_context else "proxy_internal_error", "code": final_status_code}},
                )
            except Exception as e:
                logging.exception("Error during stream finalization logging: %s", e)


SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def make_sse_response(generator) -> StreamingResponse:
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers=SSE_RESPONSE_HEADERS,
    )


LITELLM_ERROR_MAP = [
    (
        (
            InvalidRequestError,
            BadRequestError,
            ValueError,
            ContextWindowExceededError,
        ),
        400,
        "Invalid Request",
        "invalid_request_error",
    ),
    ((AuthenticationError,), 401, "Authentication Error", "authentication_error"),
    ((NotFoundError,), 404, "Not Found", "invalid_request_error"),
    ((RateLimitError,), 429, "Rate Limit Exceeded", "rate_limit_error"),
    (
        (ServiceUnavailableError, APIConnectionError),
        503,
        "Service Unavailable",
        "api_error",
    ),
    ((Timeout,), 504, "Gateway Timeout", "api_error"),
    ((InternalServerError, OpenAIError), 502, "Bad Gateway", "api_error"),
]


def handle_litellm_error(e: Exception, error_format: str = "openai") -> HTTPException:
    """Map litellm exceptions to HTTPException with OpenAI or Anthropic error format."""
    for exc_types, status_code, openai_label, anthropic_error_type in LITELLM_ERROR_MAP:
        if isinstance(e, exc_types):
            if error_format == "openai":
                detail = f"{openai_label}: {str(e)}"
            else:
                message = f"Request timed out: {str(e)}" if isinstance(e, Timeout) else str(e)
                detail = {
                    "type": "error",
                    "error": {"type": anthropic_error_type, "message": message},
                }
            return HTTPException(status_code=status_code, detail=detail)

    # Fallback for unmatched litellm errors — use generic message to avoid information leakage
    if error_format == "openai":
        return HTTPException(status_code=500, detail="Internal server error")
    return HTTPException(
        status_code=500,
        detail={"type": "error", "error": {"type": "api_error", "message": "Internal server error"}},
    )
