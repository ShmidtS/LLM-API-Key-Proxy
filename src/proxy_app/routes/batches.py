# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import orjson
import litellm
from fastapi import APIRouter, Request, Depends

from proxy_app.dependencies import verify_api_key
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["batches"])


@router.post("/v1/batches")
@handle_route_errors(error_format="openai", log_context="Batch request failed")
async def create_batch(
    request: Request,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for creating a batch job.

    Submits multiple API requests for asynchronous processing at reduced cost.
    Supports providers via LiteLLM: OpenAI, Azure, Anthropic, etc.
    """
    request_data = orjson.loads(await request.body())
    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )
    response = await litellm.acreate_batch(**request_data)
    return response


@router.get("/v1/batches/{batch_id}")
@handle_route_errors(error_format="openai", log_context="Batch request failed")
async def retrieve_batch(
    batch_id: str,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for retrieving a batch job status.
    """
    response = await litellm.aretrieve_batch(batch_id=batch_id)
    return response


@router.post("/v1/batches/{batch_id}/cancel")
@handle_route_errors(error_format="openai", log_context="Batch request failed")
async def cancel_batch(
    batch_id: str,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for cancelling a batch job.
    """
    response = await litellm.acancel_batch(batch_id=batch_id)
    return response


@router.get("/v1/batches")
@handle_route_errors(error_format="openai", log_context="Batch request failed")
async def list_batches(
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for listing batch jobs.
    """
    response = await litellm.alist_batches()
    return response
