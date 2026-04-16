# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
import logging
import time

import orjson
from fastapi import APIRouter, Request, HTTPException, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key, make_error_response
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter()


@router.get("/v1/quota-stats")
@handle_route_errors(error_format="simple", log_context="Failed to get quota stats")
async def get_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    provider: str = None,
):
    """
    Returns quota and usage statistics for all credentials.

    This returns cached data from the proxy without making external API calls.
    Use POST to reload from disk or force refresh from external APIs.

    Query Parameters:
        provider: Optional filter to return stats for a specific provider only

    Returns:
        {
            "providers": {
                "provider_name": {
                    "credential_count": int,
                    "active_count": int,
                    "on_cooldown_count": int,
                    "exhausted_count": int,
                    "total_requests": int,
                    "tokens": {...},
                    "approx_cost": float | null,
                    "quota_groups": {...},  // For Antigravity
                    "credentials": [...]
                }
            },
            "summary": {...},
            "data_source": "cache",
            "timestamp": float
        }
    """
    stats = await client.get_quota_stats(provider_filter=provider)
    return stats


@router.post("/v1/quota-stats")
@handle_route_errors(error_format="simple", log_context="Failed to refresh quota stats")
async def refresh_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Refresh quota and usage statistics.

    Request body:
        {
            "action": "reload" | "force_refresh",
            "scope": "all" | "provider" | "credential",
            "provider": "antigravity",  // required if scope != "all"
            "credential": "antigravity_oauth_1.json"  // required if scope == "credential"
        }

    Actions:
        - reload: Re-read data from disk (no external API calls)
        - force_refresh: For Antigravity, fetch live quota from API.
                        For other providers, same as reload.

    Returns:
        Same as GET, plus a "refresh_result" field with operation details.
    """
    data = orjson.loads(await request.body())
    action = data.get("action", "reload")
    scope = data.get("scope", "all")
    provider = data.get("provider")
    credential = data.get("credential")

    # Validate parameters
    if action not in ("reload", "force_refresh"):
        raise HTTPException(
            status_code=400,
            detail=make_error_response("action must be 'reload' or 'force_refresh'", "invalid_request_error"),
        )

    if scope not in ("all", "provider", "credential"):
        raise HTTPException(
            status_code=400,
            detail=make_error_response("scope must be 'all', 'provider', or 'credential'", "invalid_request_error"),
        )

    if scope in ("provider", "credential") and not provider:
        raise HTTPException(
            status_code=400,
            detail=make_error_response("'provider' is required when scope is 'provider' or 'credential'", "invalid_request_error"),
        )

    if scope == "credential" and not credential:
        raise HTTPException(
            status_code=400,
            detail=make_error_response("'credential' is required when scope is 'credential'", "invalid_request_error"),
        )

    refresh_result = {
        "action": action,
        "scope": scope,
        "provider": provider,
        "credential": credential,
    }

    if action == "reload":
        # Just reload from disk
        start_time = time.time()
        await client.reload_usage_from_disk()
        refresh_result["duration_ms"] = int((time.time() - start_time) * 1000)
        refresh_result["success"] = True
        refresh_result["message"] = "Reloaded usage data from disk"

    elif action == "force_refresh":
        # Force refresh from external API (for supported providers like Antigravity)
        result = await client.force_refresh_quota(
            provider=provider if scope in ("provider", "credential") else None,
            credential=credential if scope == "credential" else None,
        )
        refresh_result.update(result)
        refresh_result["success"] = result["failed_count"] == 0

    # Get updated stats
    stats = await client.get_quota_stats(provider_filter=provider)
    stats["refresh_result"] = refresh_result
    stats["data_source"] = "refreshed"

    return stats


@router.post("/v1/token-count")
@handle_route_errors(error_format="simple", log_context="Token count failed")
async def token_count(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Calculates the token count for a given list of messages and a model.
    """
    data = orjson.loads(await request.body())
    model = data.get("model")
    messages = data.get("messages")

    if not model or not messages:
        raise HTTPException(
            status_code=400, detail=make_error_response("'model' and 'messages' are required.", "invalid_request_error")
        )

    count = await asyncio.to_thread(client.token_count, **data)
    return {"token_count": count}


@router.post("/v1/cost-estimate")
@handle_route_errors(error_format="simple", log_context="Cost estimate failed")
async def cost_estimate(request: Request, _=Depends(verify_api_key)):
    """
    Estimates the cost for a request based on token counts and model pricing.

    Request body:
        {
            "model": "anthropic/claude-3-opus",
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache_read_tokens": 0,       # optional
            "cache_creation_tokens": 0    # optional
        }

    Returns:
        {
            "model": "anthropic/claude-3-opus",
            "cost": 0.0375,
            "currency": "USD",
            "pricing": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075
            },
            "source": "model_info_service"  # or "litellm_fallback"
        }
    """
    data = orjson.loads(await request.body())
    model = data.get("model")
    prompt_tokens = data.get("prompt_tokens", 0)
    completion_tokens = data.get("completion_tokens", 0)
    cache_read_tokens = data.get("cache_read_tokens", 0)
    cache_creation_tokens = data.get("cache_creation_tokens", 0)

    if not model:
        raise HTTPException(status_code=400, detail=make_error_response("'model' is required.", "invalid_request_error"))

    result = {
        "model": model,
        "cost": None,
        "currency": "USD",
        "pricing": {},
        "source": None,
    }

    # Try model info service first
    if hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            cost = model_info_service.calculate_cost(
                model,
                prompt_tokens,
                completion_tokens,
                cache_read_tokens,
                cache_creation_tokens,
            )
            if cost is not None:
                cost_info = model_info_service.get_cost_info(model)
                result["cost"] = cost
                result["pricing"] = cost_info or {}
                result["source"] = "model_info_service"
                return result

    # Fallback to litellm
    try:
        import litellm

        # Create a mock response for cost calculation
        model_info = litellm.get_model_info(model)
        input_cost = model_info.get("input_cost_per_token", 0)
        output_cost = model_info.get("output_cost_per_token", 0)
        cache_read_cost = model_info.get("cache_read_input_token_cost", 0) or input_cost * 0.1
        cache_creation_cost = model_info.get("cache_creation_input_token_cost", 0) or input_cost * 1.25

        if input_cost or output_cost:
            non_cached_input = max(prompt_tokens - cache_read_tokens - cache_creation_tokens, 0)
            cost = (
                non_cached_input * input_cost
                + cache_read_tokens * cache_read_cost
                + cache_creation_tokens * cache_creation_cost
                + completion_tokens * output_cost
            )
            result["cost"] = cost
            result["pricing"] = {
                "input_cost_per_token": input_cost,
                "output_cost_per_token": output_cost,
                "cache_read_input_token_cost": cache_read_cost,
                "cache_creation_input_token_cost": cache_creation_cost,
            }
            result["source"] = "litellm_fallback"
            return result
    except Exception as e:
        logging.debug("Pricing lookup failed for model %s: %s", result.get("model", "?"), e)

    result["source"] = "unknown"
    result["error"] = "Pricing data not available for this model"
    return result
