# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import orjson
from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["moderation"])


@router.post("/v1/moderations")
@handle_route_errors(error_format="openai", log_context="Moderation request failed")
async def moderations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for content moderation.

    Analyzes text for policy violations (hate, violence, sexual content, etc.).
    Supports providers via LiteLLM: OpenAI, Azure, etc.
    """
    request_data = orjson.loads(await request.body())
    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )
    response = await client.amoderation(request=request, **request_data)
    return response
