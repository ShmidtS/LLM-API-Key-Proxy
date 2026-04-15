# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
import litellm
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from rotator_library import RotatingClient
from rotator_library.anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
)
from proxy_app.dependencies import get_rotating_client, verify_anthropic_api_key, make_error_response, track_stream
from proxy_app.streaming import handle_litellm_error
from proxy_app.detailed_logger import RawIOLogger
from proxy_app.request_logger import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["anthropic"])

# Set by main.py after config loading
ENABLE_RAW_LOGGING: bool = False


@router.post("/v1/messages")
@handle_route_errors(error_format="anthropic", log_context="Anthropic messages endpoint error")
async def anthropic_messages(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
):
    """
    Anthropic-compatible Messages API endpoint.

    Accepts requests in Anthropic's format and returns responses in Anthropic's format.
    Internally translates to OpenAI format for processing via LiteLLM.

    This endpoint is compatible with Claude Code and other Anthropic API clients.
    """
    # Initialize raw I/O logger if enabled (for debugging proxy boundary)
    logger = RawIOLogger() if ENABLE_RAW_LOGGING else None

    # Parse raw body once with orjson — avoid double Pydantic validate-then-dump round-trip
    body_data = orjson.loads(await request.body())

    # Construct Pydantic model without validation (data already validated by orjson parsing)
    body = AnthropicMessagesRequest.model_construct(**body_data)

    # Log raw Anthropic request if raw logging is enabled
    if logger:
        logger.log_request(
            headers=dict(request.headers),
            body=body_data,
        )

    try:
        # Log the request to console
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(
                request.client.host if request.client else "unknown",
                request.client.port if request.client else 0,
            ),
            request_data=body_data,
        )

        # Use the library method to handle the request
        result = await client.anthropic_messages(body, raw_request=request, raw_body_data=body_data)

        if body.stream:
            # Streaming response
            return StreamingResponse(
                track_stream(request, result),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            if logger:
                logger.log_final_response(
                    status_code=200,
                    headers=None,
                    body=result,
                )
            return JSONResponse(content=result)
    except Exception as e:
        # Raw I/O logger: log the failed response if enabled
        if logger:
            logger.log_final_response(
                status_code=500,
                headers=None,
                body={"error": "Internal server error"},
            )
        raise


@router.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request,
    body: AnthropicCountTokensRequest,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
):
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
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        error_response = {
            "type": "error",
            "error": make_error_response(str(e), "invalid_request_error"),
        }
        raise HTTPException(status_code=400, detail=error_response)
    except litellm.AuthenticationError as e:
        error_response = {
            "type": "error",
            "error": make_error_response(str(e), "authentication_error"),
        }
        raise HTTPException(status_code=401, detail=error_response)
    except Exception:
        logging.error("Anthropic count_tokens endpoint error", exc_info=True)
        error_response = {
            "type": "error",
            "error": make_error_response("Internal server error", "api_error"),
        }
        raise HTTPException(status_code=500, detail=error_response)
