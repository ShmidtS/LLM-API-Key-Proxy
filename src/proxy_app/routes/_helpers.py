# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from datetime import datetime
import logging

import orjson
from fastapi import Request

from rotator_library import RotatingClient
from proxy_app.provider_urls import get_provider_endpoint


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
