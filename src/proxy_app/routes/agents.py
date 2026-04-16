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

router = APIRouter(tags=["agents"])


@router.post("/v1/agents/chat")
async def agent_chat(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Synchronous agent chat endpoint (ZAI-specific)."""
    try:
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

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Agent chat request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.post("/v1/agents/file-upload")
async def agent_file_upload(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Upload a file for agent processing (ZAI-specific)."""
    try:
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

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Agent file upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.get("/v1/agents/async-result")
async def agent_async_result(
    task_id: str,
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Retrieve async agent task result (ZAI-specific)."""
    try:
        return await client.call_provider_method(
            "zai", "agent_async_result", task_id=task_id
        )

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Agent async result retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.post("/v1/agents/conversation")
async def agent_conversation(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """Continue an agent conversation (ZAI-specific)."""
    try:
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

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Agent conversation request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))
