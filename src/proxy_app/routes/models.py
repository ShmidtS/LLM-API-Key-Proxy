# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
import time

from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from rotator_library import PROVIDER_PLUGINS

router = APIRouter()

# Module-level cache shared across requests
_models_cache: dict = {"data": None, "expires": 0.0}
_models_cache_lock = asyncio.Lock()


def invalidate_models_cache():
    _models_cache["data"] = None
    _models_cache["expires"] = 0.0


@router.get("/v1/models")
async def list_models(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    enriched: bool = True,
):
    """
    Returns a list of available models in the OpenAI-compatible format.

    Query Parameters:
        enriched: If True (default), returns detailed model info with pricing and capabilities.
                  If False, returns minimal OpenAI-compatible response.
    """
    now = time.monotonic()
    if _models_cache["data"] is not None and _models_cache["expires"] > now:
        return JSONResponse(content=_models_cache["data"], headers={"Cache-Control": "max-age=60"})

    model_ids = await client.get_all_available_models(grouped=False)

    if enriched and hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            enriched_data = model_info_service.enrich_model_list(model_ids)
            response_data = {"object": "list", "data": enriched_data}
            async with _models_cache_lock:
                _models_cache["data"] = response_data
                _models_cache["expires"] = now + 60
            return JSONResponse(content=response_data, headers={"Cache-Control": "max-age=60"})

    # Fallback to basic model cards
    model_cards = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "Mirro-Proxy",
        }
        for model_id in model_ids
    ]
    response_data = {"object": "list", "data": model_cards}
    async with _models_cache_lock:
        _models_cache["data"] = response_data
        _models_cache["expires"] = now + 60
    return JSONResponse(content=response_data, headers={"Cache-Control": "max-age=60"})


@router.get("/v1/models/{model_id:path}")
async def get_model(
    model_id: str,
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns detailed information about a specific model.

    Path Parameters:
        model_id: The model ID (e.g., "anthropic/claude-3-opus", "openrouter/openai/gpt-4")
    """
    if hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            info = model_info_service.get_model_info(model_id)
            if info:
                return info.to_dict()

    # Return basic info if service not ready or model not found
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": model_id.split("/")[0] if "/" in model_id else "unknown",
    }


@router.get("/v1/model-info/stats")
async def model_info_stats(
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns statistics about the model info service (for monitoring/debugging).
    """
    if hasattr(request.app.state, "model_info_service"):
        return request.app.state.model_info_service.get_stats()
    return {"error": "Model info service not initialized"}


@router.get("/v1/providers")
async def list_providers(_=Depends(verify_api_key)):
    """
    Returns a list of all available providers.
    """
    return list(PROVIDER_PLUGINS.keys())
