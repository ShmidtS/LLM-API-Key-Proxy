# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
import litellm
from fastapi import APIRouter, Request, Depends, HTTPException

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key, make_error_response
from proxy_app.streaming import handle_litellm_error
from proxy_app.request_logger import log_request_to_console

router = APIRouter(tags=["video"])


@router.post("/v1/video/generate")
async def video_generate(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Submit an async video generation request (ZAI-specific)."""
    try:
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

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Video operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.get("/v1/video/{video_id}/status")
async def video_status(
    video_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Check the status of an async video generation task (ZAI-specific)."""
    try:
        return await client.call_provider_method(
            "zai", "video_status", video_id=video_id
        )

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.AuthenticationError, litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.APIConnectionError, litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Video operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))
