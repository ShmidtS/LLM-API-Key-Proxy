# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Shared response parsing helpers for providers."""

import json
import logging
from typing import Any

import httpx

lib_logger = logging.getLogger("rotator_library")


async def parse_bearer_json(
    client: httpx.AsyncClient, url: str, api_key: str, timeout: int = 30
) -> dict:
    """GET JSON from a Bearer-authenticated endpoint."""
    response = await client.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=timeout,
    )
    try:
        return await parse_post_json_response(response, url)
    except (json.JSONDecodeError, ValueError):
        return {}


async def parse_post_json_response(response: httpx.Response, provider_name: str) -> dict:
    """Parse JSON from a response, handling decode errors with provider context."""
    if response.status_code >= 400:
        await response.aread()
    response.raise_for_status()
    try:
        data: Any = response.json()
    except (json.JSONDecodeError, ValueError) as e:
        body_preview = response.text[:200] if response.text else "<empty>"
        lib_logger.warning(
            "Invalid JSON from %s: %s, status=%s, body=%s",
            provider_name,
            e,
            response.status_code,
            body_preview,
        )
        raise
    return data
