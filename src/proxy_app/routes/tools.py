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

router = APIRouter(tags=["tools"])


@router.post("/v1/tools/web-search")
async def web_search(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Web search tool endpoint.

    Accepts a query and returns search results from providers
    that support web search (e.g. Z.ai style).
    The request is forwarded to the appropriate provider based on
    the model prefix in the request body.
    """
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        model = request_data.get("model")
        query = request_data.get("query") or request_data.get("input")
        if not query:
            raise HTTPException(status_code=400, detail="'query' or 'input' is required")

        # Route web search through acompletion with web_search tool
        # This leverages providers that support built-in web search tools
        messages = [{"role": "user", "content": query}]
        tools = [{
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }]

        completion_kwargs = {
            "model": model,
            "messages": messages,
            "tools": tools,
        }

        # Pass through any additional kwargs from the request
        for key in ("temperature", "max_tokens", "top_p", "stream"):
            if key in request_data:
                completion_kwargs[key] = request_data[key]

        is_streaming = completion_kwargs.get("stream", False)

        if is_streaming:
            from proxy_app.streaming import streaming_response_wrapper
            from fastapi.responses import StreamingResponse

            response_generator = client.acompletion(request=request, **completion_kwargs)
            return StreamingResponse(
                streaming_response_wrapper(request, completion_kwargs, response_generator),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await client.acompletion(request=request, **completion_kwargs)
            return response

    except HTTPException:
        raise
    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Tool operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/v1/tools/tokenizer")
async def tool_tokenizer(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """ZAI tokenizer tool endpoint."""
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        return await client.call_provider_method(
            "zai", "tool_tokenizer", **request_data
        )

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.AuthenticationError, litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.APIConnectionError, litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Tool operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/v1/tools/layout-parsing")
async def tool_layout_parsing(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """ZAI layout parsing tool endpoint."""
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        return await client.call_provider_method(
            "zai", "tool_layout_parsing", **request_data
        )

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.AuthenticationError, litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.APIConnectionError, litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Tool operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/v1/tools/web-reader")
async def tool_web_reader(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """ZAI web reader tool endpoint."""
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        return await client.call_provider_method(
            "zai", "tool_web_reader", **request_data
        )

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.AuthenticationError, litellm.RateLimitError, litellm.ServiceUnavailableError, litellm.APIConnectionError, litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Tool operation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
