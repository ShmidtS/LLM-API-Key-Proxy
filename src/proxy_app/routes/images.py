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

router = APIRouter(tags=["images"])


@router.post("/v1/images/generations")
async def image_generations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for image generation.

    Accepts model, prompt, n, size, quality, response_format and other
    parameters, proxies through litellm.aimage_generation with key rotation.
    """
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        response = await client.aimage_generation(request=request, **request_data)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Image generation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/edits")
async def image_edits(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for image editing.

    Accepts image, mask, prompt and other parameters, proxies through
    litellm.aimage_edit with key rotation.
    """
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        response = await client.aimage_edit(request=request, **request_data)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Image edit request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/images/variations")
async def image_variations(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for generating image variations.

    Accepts image and other parameters, proxies through
    litellm.aimage_variation with key rotation.
    """
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        response = await client.aimage_variation(request=request, **request_data)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Image variation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
