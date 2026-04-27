# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Optional

import orjson
from fastapi import HTTPException, Request

from rotator_library import RotatingClient
from proxy_app.dependencies import make_error_response
from proxy_app.provider_urls import get_provider_endpoint


_COMMON_ALLOWED_FIELDS = {
    "model",
    "timeout",
    "user",
    "metadata",
    "extra_body",
}

_ENDPOINT_RULES = {
    "aimage_generation": {
        "required": {"model", "prompt"},
        "allowed": _COMMON_ALLOWED_FIELDS
        | {"prompt", "n", "quality", "response_format", "size", "style"},
    },
    "aimage_edit": {
        "required": {"model", "image", "prompt"},
        "allowed": _COMMON_ALLOWED_FIELDS
        | {"image", "mask", "prompt", "n", "response_format", "size"},
    },
    "aimage_variation": {
        "required": {"model", "image"},
        "allowed": _COMMON_ALLOWED_FIELDS | {"image", "n", "response_format", "size"},
    },
    "async_image_generate": {
        "required": {"model", "prompt"},
        "allowed": _COMMON_ALLOWED_FIELDS
        | {"prompt", "image", "n", "quality", "response_format", "size"},
    },
    "video_generate": {
        "required": {"model", "prompt"},
        "allowed": _COMMON_ALLOWED_FIELDS | {"prompt", "image", "duration", "fps", "quality", "size"},
    },
    "tool_tokenizer": {
        "required": {"model", "input"},
        "allowed": _COMMON_ALLOWED_FIELDS | {"input", "messages", "tools"},
    },
    "tool_layout_parsing": {
        "required": {"model"},
        "allowed": _COMMON_ALLOWED_FIELDS | {"file", "image", "options", "url"},
    },
    "tool_web_reader": {
        "required": {"model"},
        "allowed": _COMMON_ALLOWED_FIELDS | {"query", "url", "urls"},
    },
    "agent_chat": {
        "required": {"model"},
        "allowed": _COMMON_ALLOWED_FIELDS
        | {"conversation_id", "input", "messages", "stream", "tools"},
    },
    "agent_file_upload": {
        "required": {"file"},
        "allowed": _COMMON_ALLOWED_FIELDS | {"file", "purpose"},
    },
    "agent_conversation": {
        "required": {"model", "conversation_id"},
        "allowed": _COMMON_ALLOWED_FIELDS | {"conversation_id", "input", "messages", "stream"},
    },
}


def validate_request_data(request_data: dict, endpoint_type: str | None = None) -> dict:
    """Validate required top-level fields for this endpoint."""
    if not isinstance(request_data, dict):
        raise HTTPException(
            status_code=400,
            detail=make_error_response("Request body must be a JSON object", "invalid_request_error"),
        )

    if endpoint_type is None:
        return request_data

    rules = _ENDPOINT_RULES.get(endpoint_type)
    if not rules:
        return request_data

    missing = sorted(
        field
        for field in rules["required"]
        if field not in request_data or request_data[field] is None
    )
    if missing:
        fields = ", ".join(f"'{field}'" for field in missing)
        raise HTTPException(
            status_code=400,
            detail=make_error_response(f"Missing required field(s): {fields}", "invalid_request_error"),
        )

    return request_data


@dataclass
class RequestContext:
    """Pre-resolved request state to avoid repeated getattr/orjson on hot paths."""
    client: RotatingClient
    request_data: dict
    enable_raw_logging: bool
    raw_logger: Optional[object]  # RawIOLogger or None
    override_temp_zero: str


async def resolve_request_context(
    request: Request, client: RotatingClient, endpoint_type: str | None = None
) -> RequestContext:
    """Parse body, log request, and resolve app state in a single pass."""
    request_data = await _parse_and_log(request, endpoint_type)
    enable_raw_logging = getattr(request.app.state, "enable_raw_logging", False)
    if enable_raw_logging:
        from proxy_app.detailed_logger import RawIOLogger
        raw_logger = RawIOLogger()
    else:
        raw_logger = None
    override_temp_zero = getattr(request.app.state, "override_temp_zero", "false")
    return RequestContext(
        client=client,
        request_data=request_data,
        enable_raw_logging=enable_raw_logging,
        raw_logger=raw_logger,
        override_temp_zero=override_temp_zero,
    )


def log_request_to_console(url: str, client_info: tuple, request_data: dict):
    """
    Logs a concise, single-line summary of an incoming request to the console.
    """
    time_str = datetime.now().strftime("%H:%M")
    model_full = request_data.get("model", "N/A")

    provider = "N/A"
    model_name = model_full
    endpoint_url = "N/A"

    if '/' in model_full:
        parts = model_full.split('/', 1)
        provider = parts[0]
        model_name = parts[1]
        endpoint_url = get_provider_endpoint(provider, model_name, url) or "N/A"

    log_message = f"{time_str} - {client_info[0]}:{client_info[1]} - provider: {provider}, model: {model_name} - {endpoint_url}"
    logging.info(log_message)


async def _parse_and_log(request: Request, endpoint_type: str | None = None) -> dict:
    """Parse JSON body, validate it, and log the incoming request."""
    request_data = orjson.loads(await request.body())
    request_data = validate_request_data(request_data, endpoint_type)
    client_info = (request.client.host, request.client.port) if request.client else ("unknown", 0)
    log_request_to_console(
        url=str(request.url),
        client_info=client_info,
        request_data=request_data,
    )
    return request_data


async def proxy_provider_call(
    request: Request, client: RotatingClient, provider: str, method: str
):
    """Parse request body, log it, and forward via ``client.call_provider_method``."""
    request_data = await _parse_and_log(request, method)
    return await client.call_provider_method(provider, method, **request_data)


async def proxy_client_method(
    request: Request, client: RotatingClient, method_name: str
):
    """Parse request body, log it, and forward via an arbitrary ``client`` method."""
    request_data = await _parse_and_log(request, method_name)
    handler = getattr(client, method_name)
    return await handler(request=request, **request_data)
