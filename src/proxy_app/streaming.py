# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
import logging
from typing import AsyncGenerator, Any, Optional


_GENERIC_STREAM_ERROR_MESSAGE = "An unexpected error occurred during the stream"

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse
from rotator_library import STREAM_DONE
from rotator_library.error_types import ClassifiedError
from rotator_library.utils.json_utils import sse_data_event
from rotator_library.utils.chunk_aggregator import ChunkAggregator
from proxy_app.dependencies import _inc_streams, _dec_streams
from proxy_app.detailed_logger import RawIOLogger

import litellm

logger = logging.getLogger(__name__)


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
        async for chunk in response_stream:

            # STREAM_DONE sentinel: emit SSE [DONE] and stop
            if chunk is STREAM_DONE:
                yield b"data: [DONE]\n\n"
                return

            # chunk is a dict — serialize to SSE only here (single serialize point)
            chunk_str = sse_data_event(chunk)
            yield chunk_str

            if not isinstance(chunk, dict):
                continue

            if logger is not None:
                logger.log_stream_chunk(chunk)
                if aggregator is not None:
                    aggregator.add_chunk(chunk)
    except (GeneratorExit, asyncio.CancelledError):
        if hasattr(response_stream, "aclose"):
            await response_stream.aclose()
        raise
    except Exception as e:
        logging.exception("Error during response stream")
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
        if _owns_counter:
            try:
                await _dec_streams(request)
            except AttributeError:
                logging.debug("stream_response: request lacks stream counter attribute on decrement")
        if logger and aggregator is not None:
            try:
                logger.log_final_response(
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


def handle_litellm_error(e: Exception, error_format: str = "openai") -> HTTPException:
    """Map litellm exceptions to HTTPException with OpenAI or Anthropic error format."""
    for exc_types, status_code, openai_label, anthropic_error_type in LITELLM_ERROR_MAP:
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

    # Fallback for unmatched litellm errors — use generic message to avoid information leakage
    if error_format == "openai":
        return HTTPException(status_code=500, detail="Internal server error")
    return HTTPException(
        status_code=500,
        detail={"type": "error", "error": {"type": "api_error", "message": "Internal server error"}},
    )
