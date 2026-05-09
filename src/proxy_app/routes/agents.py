# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from rotator_library import RotatingClient

from fastapi import APIRouter, Request, Depends

from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes.error_handler import handle_route_errors
from proxy_app.routes._helpers import proxy_provider_status_call, proxy_zai_route

router = APIRouter(tags=["agents"])
_agent_chat = proxy_zai_route("agent_chat")
_agent_file_upload = proxy_zai_route("agent_file_upload")
_agent_conversation = proxy_zai_route("agent_conversation")


@router.post("/v1/agents/chat")
@handle_route_errors(error_format="openai", log_context="Agents request failed")
async def agent_chat(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Synchronous agent chat endpoint (ZAI-specific)."""
    return await _agent_chat(request, client)


@router.post("/v1/agents/file-upload")
@handle_route_errors(error_format="openai", log_context="Agent details failed")
async def agent_file_upload(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Upload a file for agent processing (ZAI-specific)."""
    return await _agent_file_upload(request, client)


@router.get("/v1/agents/async-result")
@handle_route_errors(error_format="openai", log_context="Agent thread failed")
async def agent_async_result(
    task_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Retrieve async agent task result (ZAI-specific)."""
    return await proxy_provider_status_call(
        client, "zai", "agent_async_result", "task_id", task_id
    )


@router.post("/v1/agents/conversation")
@handle_route_errors(error_format="openai", log_context="Agents request failed")
async def agent_conversation(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """Continue an agent conversation (ZAI-specific)."""
    return await _agent_conversation(request, client)
