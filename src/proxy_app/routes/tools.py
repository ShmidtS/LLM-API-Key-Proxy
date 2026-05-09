# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rotator_library import RotatingClient

import orjson
from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from proxy_app.dependencies import get_rotating_client, verify_api_key, make_error_response
from proxy_app.routes.error_handler import handle_route_errors
from proxy_app.routes._helpers import _parse_and_log, complete_or_stream, proxy_zai_route

router = APIRouter(tags=["tools"])
_tool_tokenizer = proxy_zai_route("tool_tokenizer")
_tool_layout_parsing = proxy_zai_route("tool_layout_parsing")
_tool_web_reader = proxy_zai_route("tool_web_reader")


@router.post("/v1/tools/web-search")
@handle_route_errors(error_format="openai", log_context="Tool call failed")
async def web_search(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """
    Web search tool endpoint.

    Accepts a query and returns search results from providers
    that support web search (e.g. Z.ai style).
    The request is forwarded to the appropriate provider based on
    the model prefix in the request body.
    """
    try:
        request_data = await _parse_and_log(request)
    except orjson.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid JSON in request body",
                    "type": "invalid_request_error",
                    "code": "invalid_json",
                }
            },
        )

    model = request_data.get("model")
    query = request_data.get("query") or request_data.get("input")
    if not query:
        raise HTTPException(status_code=400, detail=make_error_response("'query' or 'input' is required", "invalid_request_error"))

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

    for key in ("temperature", "max_tokens", "top_p", "stream"):
        if key in request_data:
            completion_kwargs[key] = request_data[key]

    return await complete_or_stream(
        request,
        completion_kwargs,
        client,
        bool(completion_kwargs.get("stream", False)),
    )


@router.post("/v1/tools/tokenizer")
@handle_route_errors(error_format="openai", log_context="Tool list failed")
async def tool_tokenizer(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """ZAI tokenizer tool endpoint."""
    return await _tool_tokenizer(request, client)


@router.post("/v1/tools/layout-parsing")
@handle_route_errors(error_format="openai", log_context="Tool create failed")
async def tool_layout_parsing(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """ZAI layout parsing tool endpoint."""
    return await _tool_layout_parsing(request, client)


@router.post("/v1/tools/web-reader")
@handle_route_errors(error_format="openai", log_context="Tool update failed")
async def tool_web_reader(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
) -> Any:
    """ZAI web reader tool endpoint."""
    return await _tool_web_reader(request, client)
