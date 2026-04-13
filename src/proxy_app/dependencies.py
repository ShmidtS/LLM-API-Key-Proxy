# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader

from rotator_library import RotatingClient
from proxy_app.batch_manager import EmbeddingBatcher

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)

# These are set by main.py after environment loading
PROXY_API_KEY: str = None
_BEARER_PROXY_API_KEY: str = None


def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client


def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    return request.app.state.embedding_batcher


async def verify_api_key(auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not PROXY_API_KEY:
        return auth
    if not auth or auth != _BEARER_PROXY_API_KEY:
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
    if x_api_key and x_api_key == PROXY_API_KEY:
        return x_api_key
    # Fall back to Bearer token (OpenAI style)
    if auth and auth == _BEARER_PROXY_API_KEY:
        return auth
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")
