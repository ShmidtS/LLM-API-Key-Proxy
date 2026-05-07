# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Shared quota tracking utilities.

Contains common helper functions used across quota tracker implementations,
deduplicated from BaseQuotaTracker and LightweightQuotaMixin.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx
import json

lib_logger = logging.getLogger("rotator_library")


def parse_iso_timestamp(iso_string: str) -> Optional[float]:
    """Parse ISO 8601 timestamp to Unix timestamp.

    Args:
        iso_string: ISO 8601 formatted timestamp (e.g., "2026-01-20T18:12:03.000Z")

    Returns:
        Unix timestamp in seconds, or None if parsing fails
    """
    try:
        if iso_string.endswith("Z"):
            iso_string = iso_string.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_string)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError, KeyError) as e:
        lib_logger.warning(f"Failed to parse ISO timestamp '{iso_string}': {e}", exc_info=True)
        return None


def make_bearer_header(api_key: str) -> Dict[str, str]:
    """Build a standard Bearer auth header.

    Args:
        api_key: API key for Authorization header

    Returns:
        Dict with accept and Authorization headers
    """
    return {
        "accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


async def post_json_with_error_handling(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    *,
    timeout: int = 30,
    provider_name: str = "unknown",
) -> Optional[Dict[str, Any]]:
    """POST JSON with robust error handling for quota APIs.

    Checks for 5xx / 401 / 403 status codes before raise_for_status
    so they are logged as warnings rather than raising HTTPStatusError.
    Catches JSON decode errors, HTTPStatusError, and RequestError,
    returning None (with a logged warning/error) in every failure case.

    Args:
        client: httpx.AsyncClient to use for the request
        url: Request URL
        headers: Request headers
        payload: JSON payload for POST body
        timeout: Request timeout in seconds
        provider_name: Provider name for log messages

    Returns:
        Parsed dict on success, or None on any failure
    """
    try:
        response = await client.post(url, headers=headers, json=payload, timeout=timeout)

        if response.status_code in (401, 403):
            lib_logger.warning(
                f"{provider_name} quota API returned {response.status_code} "
                f"(auth/forbidden), skipping"
            )
            return None
        if response.status_code >= 500:
            lib_logger.warning(
                f"{provider_name} quota API returned {response.status_code} "
                f"(server error), skipping"
            )
            return None

        response.raise_for_status()

        body = response.text
        if not body or not body.strip():
            lib_logger.debug(
                f"{provider_name} quota API returned empty response (status {response.status_code})"
            )
            return None

        try:
            return response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            lib_logger.warning(
                f"{provider_name} quota API returned invalid JSON: {exc}"
            )
            return None

    except httpx.HTTPStatusError as exc:
        lib_logger.warning(
            f"{provider_name} quota API HTTP error: {exc.response.status_code}"
        )
        return None
    except httpx.RequestError as exc:
        lib_logger.error(
            f"{provider_name} quota API request error: {exc}"
        )
        return None
