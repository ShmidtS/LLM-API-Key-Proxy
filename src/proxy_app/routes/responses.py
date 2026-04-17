# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import orjson
from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.streaming import streaming_response_wrapper, make_sse_response
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["responses"])


@router.post("/v1/responses")
@handle_route_errors(error_format="openai", log_context="Responses API request failed")
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
    request_data = orjson.loads(await request.body())

    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )

    is_streaming = request_data.get("stream", False)

    messages = _input_to_messages(request_data.get("input"))
    request_data["messages"] = messages

    if is_streaming:
        response_generator = client.acompletion(request=request, **request_data)
        return make_sse_response(
            streaming_response_wrapper(request, response_generator)
        )
    else:
        response = await client.acompletion(request=request, **request_data)
        return response


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
