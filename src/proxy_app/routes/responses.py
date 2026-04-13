# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
import litellm
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.streaming import handle_litellm_error, streaming_response_wrapper
from proxy_app.request_logger import log_request_to_console

router = APIRouter(tags=["responses"])


@router.post("/v1/responses")
async def create_response(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI Responses API endpoint.

    Accepts the OpenAI Responses API format (with 'input' instead of 'messages')
    and routes to providers that support this format. For providers that only
    support chat/completions, the request is converted automatically.
    """
    try:
        request_data = orjson.loads(await request.body())

        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )

        is_streaming = request_data.get("stream", False)

        # Convert Responses API format to chat/completions format for
        # providers that use the standard pipeline. Providers with custom
        # logic (e.g. ColinProvider) handle the format internally.
        messages = _input_to_messages(request_data.get("input"))
        request_data["messages"] = messages

        if is_streaming:
            response_generator = client.acompletion(request=request, **request_data)
            return StreamingResponse(
                streaming_response_wrapper(request, request_data, response_generator),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await client.acompletion(request=request, **request_data)
            return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Responses API request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _input_to_messages(input_data) -> list:
    """
    Convert Responses API 'input' field to chat/completions 'messages' format.

    The Responses API accepts:
    - str: single user message
    - list of dicts with role/content
    - list of mixed items (messages + tool results)
    """
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]

    if isinstance(input_data, list):
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", item.get("text", ""))
                if role in ("user", "assistant", "system", "developer"):
                    # Map developer role to system for providers that don't support it
                    mapped_role = "system" if role == "developer" else role
                    messages.append({"role": mapped_role, "content": content})
        return messages

    return [{"role": "user", "content": str(input_data)}]
