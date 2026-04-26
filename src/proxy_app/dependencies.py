# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import hmac
import asyncio
import logging
from typing import AsyncGenerator, Any

logger = logging.getLogger(__name__)

from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader

from rotator_library import RotatingClient
from proxy_app.batch_manager import EmbeddingBatcher

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def _inc_streams(request):
    async with request.app.state.stream_lock:
        request.app.state.active_streams += 1


async def _dec_streams(request):
    async with request.app.state.stream_lock:
        cur = getattr(request.app.state, "active_streams", 0)
        request.app.state.active_streams = max(0, cur - 1)


def _register_stream_gen(request, gen):
    """Register a stream generator for graceful shutdown cancellation."""
    if getattr(request.app.state, "_shutting_down", False):
        return
    try:
        request.app.state.active_stream_gens.add(gen)
    except AttributeError:
        request.app.state.active_stream_gens = {gen}


def _unregister_stream_gen(request, gen):
    """Unregister a stream generator after completion."""
    gens = getattr(request.app.state, "active_stream_gens", None)
    if gens:
        gens.discard(gen)


def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    try:
        client = request.app.state.rotating_client
    except AttributeError as e:
        logger.error("Failed to get rotating client from app state: %s", e)
        client = None
    if client is None:
        raise HTTPException(status_code=500, detail="Server not initialized: rotating client unavailable")
    return client


def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    try:
        batcher = request.app.state.embedding_batcher
    except AttributeError as e:
        logger.error("Failed to get embedding batcher from app state: %s", e)
        batcher = None
    if batcher is None:
        raise HTTPException(status_code=500, detail="Server not initialized: embedding batcher unavailable")
    return batcher


def make_error_response(message: str, error_type: str = "api_error", code: str | None = None) -> dict:
    """Build a consistent error JSON payload for HTTPException.detail."""
    err: dict = {"message": message, "type": error_type}
    if code is not None:
        err["code"] = code
    return {"error": err}


async def track_stream(request: Request, stream: AsyncGenerator[Any, None]) -> AsyncGenerator[Any, None]:
    """Wrap an async generator to track active streaming connections for graceful shutdown."""
    request._stream_tracked = True
    try:
        await _inc_streams(request)
    except AttributeError:
        logger.error("track_stream: request lacks stream counter attribute")
    try:
        _register_stream_gen(request, stream)
    except AttributeError:
        logger.error("track_stream: request lacks stream generator registry")
    try:
        async for chunk in stream:
            yield chunk
    except (GeneratorExit, asyncio.CancelledError):
        if hasattr(stream, "aclose"):
            await stream.aclose()
        raise
    except Exception:
        if hasattr(stream, "aclose"):
            await stream.aclose()
        raise
    finally:
        _unregister_stream_gen(request, stream)
        try:
            await _dec_streams(request)
        except AttributeError:
            logger.error("track_stream: request lacks stream counter attribute on decrement")


async def verify_api_key(request: Request, auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    proxy_api_key = getattr(request.app.state, "proxy_api_key", None)
    bearer_key = getattr(request.app.state, "bearer_proxy_api_key", None)
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not proxy_api_key:
        logger.warning("PROXY_API_KEY not set - authentication disabled")
        return auth
    if not auth or not bearer_key or not hmac.compare_digest(auth, bearer_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth


async def verify_anthropic_api_key(
    request: Request,
    x_api_key: str = Depends(anthropic_api_key_header),
    auth: str = Depends(api_key_header),
):
    """
    Dependency to verify API key for Anthropic endpoints.
    Accepts either x-api-key header (Anthropic style) or Authorization Bearer (OpenAI style).
    """
    proxy_api_key = getattr(request.app.state, "proxy_api_key", None)
    bearer_key = getattr(request.app.state, "bearer_proxy_api_key", None)
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not proxy_api_key:
        logger.warning("PROXY_API_KEY not set - authentication disabled")
        return x_api_key or auth
    # Check x-api-key first (Anthropic style)
    if x_api_key and hmac.compare_digest(x_api_key, proxy_api_key):
        return x_api_key
    # Fall back to Bearer token (OpenAI style)
    if auth and bearer_key and hmac.compare_digest(auth, bearer_key):
        return auth
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")
