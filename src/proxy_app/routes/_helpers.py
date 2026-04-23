# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from dataclasses import dataclass
from datetime import datetime
import logging
from typing import Optional

import orjson
from fastapi import Request

from rotator_library import RotatingClient
from proxy_app.provider_urls import get_provider_endpoint


@dataclass
class RequestContext:
    """Pre-resolved request state to avoid repeated getattr/orjson on hot paths."""
    client: RotatingClient
    request_data: dict
    enable_raw_logging: bool
    raw_logger: Optional[object]  # RawIOLogger or None
    override_temp_zero: str


async def resolve_request_context(request: Request, client: RotatingClient) -> RequestContext:
    """Parse body, log request, and resolve app state in a single pass."""
    request_data = orjson.loads(await request.body())
    client_info = (request.client.host, request.client.port) if request.client else ("unknown", 0)
    log_request_to_console(
        url=str(request.url),
        client_info=client_info,
        request_data=request_data,
    )
    enable_raw_logging = getattr(request.app.state, "enable_raw_logging", False)
    raw_logger = None
    if enable_raw_logging:
        from proxy_app.detailed_logger import RawIOLogger
        raw_logger = RawIOLogger()
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


async def _parse_and_log(request: Request) -> dict:
    """Parse JSON body and log the incoming request."""
    request_data = orjson.loads(await request.body())
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
    request_data = await _parse_and_log(request)
    return await client.call_provider_method(provider, method, **request_data)


async def proxy_client_method(
    request: Request, client: RotatingClient, method_name: str
):
    """Parse request body, log it, and forward via an arbitrary ``client`` method."""
    request_data = await _parse_and_log(request)
    handler = getattr(client, method_name)
    return await handler(request=request, **request_data)
