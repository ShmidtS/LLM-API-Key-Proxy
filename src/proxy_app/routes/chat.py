# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

import logging

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rotator_library import RotatingClient

from fastapi import APIRouter, Request, Depends

logger = logging.getLogger(__name__)
rotator_logger = logging.getLogger("rotator_library")

from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.routes._helpers import complete_or_stream, resolve_request_context
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["chat"])


@router.post("/v1/chat/completions")
@handle_route_errors(error_format="openai", log_context="Request failed after all retries")
async def chat_completions(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses and logs them.
    """
    ctx = await resolve_request_context(request, client)
    request_data = ctx.request_data

    if (
        ctx.override_temp_zero in ("remove", "set", "true", "1", "yes")
        and "temperature" in request_data
        and request_data["temperature"] == 0
    ):
        if ctx.override_temp_zero == "remove":
            del request_data["temperature"]
            logging.debug(
                "OVERRIDE_TEMPERATURE_ZERO=remove: Removed temperature=0 from request"
            )
        else:
            request_data["temperature"] = 1.0
            logging.debug(
                "OVERRIDE_TEMPERATURE_ZERO=set: Converting temperature=0 to temperature=1.0"
            )

    model = request_data.get("model")
    generation_cfg = (
        request_data.get("generationConfig", {})
        or request_data.get("generation_config", {})
        or {}
    )
    reasoning_effort = request_data.get("reasoning_effort") or generation_cfg.get(
        "reasoning_effort"
    )

    rotator_logger.debug(
        "Handling reasoning parameters: model=%s, reasoning_effort=%s",
        model,
        reasoning_effort,
    )

    try:
        return await complete_or_stream(
            request,
            request_data,
            client,
            request_data.get("stream", False),
            ctx.raw_logger,
        )
    except Exception as e:
        logger.exception("Chat completions error: %s", e)
        raise
