# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rotator_library import RotatingClient

from fastapi import APIRouter, Request, Depends

from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes.error_handler import handle_route_errors
from proxy_app.routes._helpers import proxy_provider_status_call, proxy_zai_route

router = APIRouter(tags=["video"])
_video_generate = proxy_zai_route("video_generate")


@router.post("/v1/video/generate")
@handle_route_errors(error_format="openai", log_context="Video generation failed")
async def video_generate(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Submit an async video generation request (ZAI-specific)."""
    return await _video_generate(request, client)


@router.get("/v1/video/{video_id}/status")
@handle_route_errors(error_format="openai", log_context="Video status failed")
async def video_status(
    video_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Check the status of an async video generation task (ZAI-specific)."""
    return await proxy_provider_status_call(
        client, "zai", "video_status", "video_id", video_id
    )
