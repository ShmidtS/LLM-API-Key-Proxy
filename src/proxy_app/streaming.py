# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, AsyncGenerator, Any, Optional

if TYPE_CHECKING:
    from proxy_app.detailed_logger import RawIOLogger

_GENERIC_STREAM_ERROR_MESSAGE = "An unexpected error occurred during the stream"

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from proxy_app.dependencies import _inc_streams, _dec_streams

logger = logging.getLogger(__name__)
_STREAM_DONE_FALLBACK = object()


def _stream_done():
    try:
        from rotator_library.utils.json_utils import STREAM_DONE
    except ImportError:
        return _STREAM_DONE_FALLBACK
    return STREAM_DONE


def _sse_data_event(payload: Any) -> bytes:
    from rotator_library.utils.json_utils import sse_data_event

    return sse_data_event(payload)


def _is_classified_error(error: Exception) -> bool:
    return error.__class__.__name__ == "ClassifiedError"


async def safe_aclose(stream: Any) -> None:
    """Close async streams that expose aclose(), ignoring close failures."""
    if hasattr(stream, "aclose"):
        try:
            await stream.aclose()
        except Exception as e:
            logger.debug("Error closing async stream: %s", e)


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
    if logger is not None:
        from rotator_library.utils.chunk_aggregator import ChunkAggregator

        aggregator = ChunkAggregator()
    else:
        aggregator = None
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
            if chunk is _stream_done():
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
            chunk_str = _sse_data_event(chunk)
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
        await safe_aclose(response_stream)
        _buffer.clear()
        raise
    except Exception as e:
        logging.exception("Error during response stream")
        # Flush any buffered chunks before sending error
        if _buffer:
            yield b"".join(_buffer)
            _buffer.clear()
        # Propagate classified error type so clients can distinguish 429/503/502
        if _is_classified_error(e) and getattr(e, "status_code", None):
            error_type = getattr(e, "error_type", "api_error")
            status_code = getattr(e, "status_code", 500)
            final_status_code = status_code
            final_error_log_context = {
                "type": error_type,
                "code": status_code,
            }
            error_payload = {
                "error": {
                    "message": _GENERIC_STREAM_ERROR_MESSAGE,
                    "type": error_type,
                    "code": status_code,
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
        yield _sse_data_event(error_payload)
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


def get_litellm_error_map():
    from litellm.exceptions import (  # type: ignore[import-untyped]
        APIConnectionError,
        AuthenticationError,
        BadRequestError,
        ContextWindowExceededError,
        InternalServerError,
        InvalidRequestError,
        NotFoundError,
        OpenAIError,
        RateLimitError,
        ServiceUnavailableError,
        Timeout,
    )

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


def handle_litellm_error(e: Exception, error_format: str = "openai") -> HTTPException:
    """Map litellm exceptions to HTTPException with OpenAI or Anthropic error format."""
    from litellm.exceptions import Timeout  # type: ignore[import-untyped]

    for exc_types, status_code, openai_label, anthropic_error_type in get_litellm_error_map():
        if isinstance(e, exc_types):
            if error_format == "openai":
                detail = f"{openai_label}: {e!s}"
            else:
                message = f"Request timed out: {e!s}" if isinstance(e, Timeout) else f"{e!s}"
                detail = {
                    "type": "error",
                    "error": {"type": anthropic_error_type, "message": message},
                }
            return HTTPException(status_code=status_code, detail=detail)

    # Fallback for unmatched litellm errors — use generic message to avoid information leakage
    from proxy_app.routes.error_handler import internal_server_error_payload

    detail = (
        "Internal server error"
        if error_format == "openai"
        else internal_server_error_payload(error_format)
    )
    return HTTPException(status_code=500, detail=detail)
