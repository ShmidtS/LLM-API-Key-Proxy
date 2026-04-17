# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import orjson
from fastapi import Request

from rotator_library import RotatingClient
from proxy_app.request_logger import log_request_to_console


async def _parse_and_log(request: Request) -> dict:
    """Parse JSON body and log the incoming request."""
    request_data = orjson.loads(await request.body())
    log_request_to_console(
        url=str(request.url),
        headers=request.headers,
        client_info=(request.client.host, request.client.port),
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
