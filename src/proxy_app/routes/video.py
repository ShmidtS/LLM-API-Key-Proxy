# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes.error_handler import handle_route_errors
from proxy_app.routes._helpers import proxy_provider_call

router = APIRouter(tags=["video"])


@router.post("/v1/video/generate")
@handle_route_errors(error_format="openai", log_context="Video generation failed")
async def video_generate(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Submit an async video generation request (ZAI-specific)."""
    return await proxy_provider_call(request, client, "zai", "video_generate")


@router.get("/v1/video/{video_id}/status")
@handle_route_errors(error_format="openai", log_context="Video status failed")
async def video_status(
    video_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Check the status of an async video generation task (ZAI-specific)."""
    return await client.call_provider_method(
        "zai", "video_status", video_id=video_id
    )
