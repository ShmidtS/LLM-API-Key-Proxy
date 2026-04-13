# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
import litellm
from fastapi import APIRouter, Request, Depends, HTTPException

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.streaming import handle_litellm_error
from proxy_app.request_logger import log_request_to_console

router = APIRouter(tags=["moderation"])


@router.post("/v1/moderations")
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
    try:
        request_data = orjson.loads(await request.body())
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        response = await client.amoderation(request=request, **request_data)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Moderation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
