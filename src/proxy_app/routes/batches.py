# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
import litellm
from fastapi import APIRouter, Request, Depends, HTTPException

from proxy_app.dependencies import verify_api_key
from proxy_app.streaming import handle_litellm_error
from proxy_app.request_logger import log_request_to_console

router = APIRouter(tags=["batches"])


@router.post("/v1/batches")
async def create_batch(
    request: Request,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for creating a batch job.

    Submits multiple API requests for asynchronous processing at reduced cost.
    Supports providers via LiteLLM: OpenAI, Azure, Anthropic, etc.
    """
    try:
        request_data = orjson.loads(await request.body())
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        response = await litellm.acreate_batch(**request_data)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Create batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/batches/{batch_id}")
async def retrieve_batch(
    batch_id: str,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for retrieving a batch job status.
    """
    try:
        response = await litellm.aretrieve_batch(batch_id=batch_id)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Retrieve batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(
    batch_id: str,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for cancelling a batch job.
    """
    try:
        response = await litellm.acancel_batch(batch_id=batch_id)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Cancel batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/batches")
async def list_batches(
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for listing batch jobs.
    """
    try:
        response = await litellm.alist_batches()
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"List batches failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
