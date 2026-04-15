# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import hmac
import asyncio
import logging
import threading
from typing import AsyncGenerator, Any

logger = logging.getLogger(__name__)

from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader

from rotator_library import RotatingClient
from proxy_app.batch_manager import EmbeddingBatcher

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# These are set once by main.py after environment loading, before any requests are served.
# They are read-only during request handling, so no race condition exists in the
# single-threaded asyncio event loop. Moving to app.state would be cleaner but
# requires refactoring all route imports that reference these module-level names.
PROXY_API_KEY: str = None
_BEARER_PROXY_API_KEY: str = None

_streams_lock = threading.Lock()


def _inc_streams(request):
    with _streams_lock:
        request.app.state.active_streams += 1


def _dec_streams(request):
    with _streams_lock:
        request.app.state.active_streams -= 1


def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client


def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    return request.app.state.embedding_batcher


def make_error_response(message: str, error_type: str = "api_error", code: str | None = None) -> dict:
    """Build a consistent error JSON payload for HTTPException.detail."""
    err: dict = {"message": message, "type": error_type}
    if code is not None:
        err["code"] = code
    return {"error": err}


async def track_stream(request: Request, stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
    """Wrap an async generator to track active streaming connections for graceful shutdown."""
    try:
        _inc_streams(request)
    except AttributeError:
        logger.debug("track_stream: request lacks stream counter attribute")
    try:
        async for chunk in stream:
            yield chunk
    except (GeneratorExit, asyncio.CancelledError):
        if hasattr(stream, "aclose"):
            await stream.aclose()
        raise
    finally:
        try:
            _dec_streams(request)
        except AttributeError:
            logger.debug("track_stream: request lacks stream counter attribute on decrement")


async def verify_api_key(auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not PROXY_API_KEY:
        return auth
    if not auth or not hmac.compare_digest(auth, _BEARER_PROXY_API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth


async def verify_anthropic_api_key(
    x_api_key: str = Depends(anthropic_api_key_header),
    auth: str = Depends(api_key_header),
):
    """
    Dependency to verify API key for Anthropic endpoints.
    Accepts either x-api-key header (Anthropic style) or Authorization Bearer (OpenAI style).
    """
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not PROXY_API_KEY:
        return x_api_key or auth
    # Check x-api-key first (Anthropic style)
    if x_api_key and hmac.compare_digest(x_api_key, PROXY_API_KEY):
        return x_api_key
    # Fall back to Bearer token (OpenAI style)
    if auth and hmac.compare_digest(auth, _BEARER_PROXY_API_KEY):
        return auth
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")
