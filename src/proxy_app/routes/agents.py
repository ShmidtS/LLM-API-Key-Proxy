# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import orjson
from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.request_logger import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["agents"])


@router.post("/v1/agents/chat")
@handle_route_errors(error_format="openai", log_context="Agents request failed")
async def agent_chat(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Synchronous agent chat endpoint (ZAI-specific)."""
    request_data = orjson.loads(await request.body())

    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )

    return await client.call_provider_method(
        "zai", "agent_chat", **request_data
    )


@router.post("/v1/agents/file-upload")
@handle_route_errors(error_format="openai", log_context="Agent details failed")
async def agent_file_upload(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Upload a file for agent processing (ZAI-specific)."""
    request_data = orjson.loads(await request.body())

    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )

    return await client.call_provider_method(
        "zai", "agent_file_upload", **request_data
    )


@router.get("/v1/agents/async-result")
@handle_route_errors(error_format="openai", log_context="Agent thread failed")
async def agent_async_result(
    task_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Retrieve async agent task result (ZAI-specific)."""
    return await client.call_provider_method(
        "zai", "agent_async_result", task_id=task_id
    )


@router.post("/v1/agents/conversation")
@handle_route_errors(error_format="openai", log_context="Agents request failed")
async def agent_conversation(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Continue an agent conversation (ZAI-specific)."""
    request_data = orjson.loads(await request.body())

    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )

    return await client.call_provider_method(
        "zai", "agent_conversation", **request_data
    )
