# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from typing import Any

from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes.error_handler import handle_route_errors
from proxy_app.routes._helpers import proxy_provider_call, proxy_client_method

router = APIRouter(tags=["images"])


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
    return await proxy_client_method(request, client, "aimage_generation")


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
    return await proxy_client_method(request, client, "aimage_edit")


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
    return await proxy_client_method(request, client, "aimage_variation")


@router.post("/v1/images/generations/async")
@handle_route_errors(error_format="openai", log_context="Image upscale failed")
async def async_image_generations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Async image generation endpoint (ZAI-specific). Returns a task ID for polling."""
    return await proxy_provider_call(request, client, "zai", "async_image_generate")


@router.get("/v1/images/{image_id}")
@handle_route_errors(error_format="openai", log_context="Image generation failed")
async def get_image_status(
    image_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Retrieve status/result of an async image generation task (ZAI-specific)."""
    return await client.call_provider_method(
        "zai", "async_image_status", image_id=image_id
    )
