# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rotator_library import RotatingClient

from fastapi import APIRouter, Request, Depends

from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes._helpers import _parse_and_log
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["moderation"])


@router.post("/v1/moderations")
@handle_route_errors(error_format="openai", log_context="Moderation request failed")
async def moderations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for content moderation.

    Analyzes text for policy violations (hate, violence, sexual content, etc.).
    Supports providers via LiteLLM: OpenAI, Azure, etc.
    """
    request_data = await _parse_and_log(request, "amoderation")
    return await client.amoderation(request=request, **request_data)
