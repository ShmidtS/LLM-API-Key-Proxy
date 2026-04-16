# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import orjson
from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.request_logger import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["video"])


@router.post("/v1/video/generate")
@handle_route_errors(error_format="openai", log_context="Video generation failed")
async def video_generate(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Submit an async video generation request (ZAI-specific)."""
    request_data = orjson.loads(await request.body())

    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )

    return await client.call_provider_method(
        "zai", "video_generate", **request_data
    )


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
