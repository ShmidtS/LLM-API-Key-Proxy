# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rotator_library import RotatingClient

from fastapi import APIRouter, Request, Depends

from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes.error_handler import handle_route_errors
from proxy_app.routes._helpers import (
    _parse_and_log,
    proxy_provider_status_call,
    proxy_zai_route,
)

router = APIRouter(tags=["images"])
_async_image_generate = proxy_zai_route("async_image_generate")


@router.post("/v1/images/generations")
@handle_route_errors(error_format="openai", log_context="Image generation failed")
async def image_generations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for image generation.

    Accepts model, prompt, n, size, quality, response_format and other
    parameters, proxies through litellm.aimage_generation with key rotation.
    """
    request_data = await _parse_and_log(request, "aimage_generation")
    return await client.aimage_generation(request=request, **request_data)


@router.post("/v1/images/edits")
@handle_route_errors(error_format="openai", log_context="Image edit failed")
async def image_edits(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for image editing.

    Accepts image, mask, prompt and other parameters, proxies through
    litellm.aimage_edit with key rotation.
    """
    request_data = await _parse_and_log(request, "aimage_edit")
    return await client.aimage_edit(request=request, **request_data)


@router.post("/v1/images/variations")
@handle_route_errors(error_format="openai", log_context="Image variation failed")
async def image_variations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for generating image variations.

    Accepts image and other parameters, proxies through
    litellm.aimage_variation with key rotation.
    """
    request_data = await _parse_and_log(request, "aimage_variation")
    return await client.aimage_variation(request=request, **request_data)


@router.post("/v1/images/generations/async")
@handle_route_errors(error_format="openai", log_context="Image upscale failed")
async def async_image_generations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Async image generation endpoint (ZAI-specific). Returns a task ID for polling."""
    return await _async_image_generate(request, client)


@router.get("/v1/images/{image_id}")
@handle_route_errors(error_format="openai", log_context="Image generation failed")
async def get_image_status(
    image_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Retrieve status/result of an async image generation task (ZAI-specific)."""
    return await proxy_provider_status_call(
        client, "zai", "async_image_status", "image_id", image_id
    )
