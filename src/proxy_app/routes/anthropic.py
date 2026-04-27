# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

logger = logging.getLogger(__name__)

import orjson
import litellm
from typing import Any, AsyncGenerator
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from rotator_library import RotatingClient
from rotator_library.anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
)
from proxy_app.dependencies import get_rotating_client, verify_anthropic_api_key, track_stream
from proxy_app.detailed_logger import RawIOLogger
from proxy_app.streaming import make_sse_response
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["anthropic"])



@router.post("/v1/messages")
@handle_route_errors(error_format="anthropic", log_context="Anthropic messages endpoint error")
async def anthropic_messages(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
) -> JSONResponse:
    """
    Anthropic-compatible Messages API endpoint.

    Accepts requests in Anthropic's format and returns responses in Anthropic's format.
    Internally translates to OpenAI format for processing via LiteLLM.

    This endpoint is compatible with Claude Code and other Anthropic API clients.
    """
    # Initialize raw I/O logger if enabled (for debugging proxy boundary)
    enable_raw_logging = getattr(request.app.state, "enable_raw_logging", False)
    logger = RawIOLogger() if enable_raw_logging else None

    # Parse raw body — use model_validate_json (single parse) unless raw logging
    # needs the original dict, in which case parse twice for correctness.
    raw_body = await request.body()
    if enable_raw_logging:
        body_data = orjson.loads(raw_body)
        body = AnthropicMessagesRequest.model_validate(body_data)
    else:
        body = AnthropicMessagesRequest.model_validate_json(raw_body)
        body_data = body.model_dump()

    # Log raw Anthropic request if raw logging is enabled
    if logger:
        await logger.log_request(
            headers=dict(request.headers),
            body=body_data,
        )

    log_request_to_console(
        url=str(request.url),
        client_info=(
            request.client.host if request.client else "unknown",
            request.client.port if request.client else 0,
        ),
        request_data=body_data,
    )

    result = await client.anthropic_messages(body, raw_request=request, raw_body_data=body_data)

    if body.stream:
        # Streaming response
        return make_sse_response(track_stream(request, result))
    else:
        # Non-streaming response
        if logger:
            logger.log_final_response(
                status_code=200,
                headers=None,
                body=result,
            )
        return JSONResponse(content=result)


@router.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request,
    body: AnthropicCountTokensRequest,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
) -> JSONResponse:
    """
    Anthropic-compatible count_tokens endpoint.

    Counts the number of tokens that would be used by a Messages API request.
    This is useful for estimating costs and managing context windows.

    Accepts requests in Anthropic's format and returns token count in Anthropic's format.
    """
    try:
        # Use the library method to handle the request
        result = await client.anthropic_count_tokens(body)
        return JSONResponse(content=result)

    except (
        litellm.InvalidRequestError,
        litellm.ContextWindowExceededError,
        ValueError,
    ) as e:
        error_response = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": str(e)},
        }
        raise HTTPException(status_code=400, detail=error_response)
    except litellm.AuthenticationError as e:
        error_response = {
            "type": "error",
            "error": {"type": "authentication_error", "message": str(e)},
        }
        raise HTTPException(status_code=401, detail=error_response)
    except Exception as e:
        logger.exception("Anthropic count_tokens endpoint error: %s", e)
        error_response = {
            "type": "error",
            "error": {"type": "api_error", "message": "Internal server error"},
        }
        raise HTTPException(status_code=500, detail=error_response)
