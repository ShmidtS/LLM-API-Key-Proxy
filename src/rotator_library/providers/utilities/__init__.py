# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# Utilities for provider implementations

import json
import logging
from typing import Any, Callable, Dict, List

import httpx

from .gemini_shared_utils import (
    DEFAULT_GENERIC_SAFETY_SETTINGS,
    DEFAULT_SAFETY_SETTINGS,
    DEFAULT_GEMINI_SAFETY_SETTINGS_MAP,
)
from .google_quota_tracker_base import GoogleQuotaTrackerBase

lib_logger = logging.getLogger("rotator_library")


async def fetch_provider_models(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    provider_label: str,
    parse_fn: Callable[[Any], List[str]],
) -> List[str]:
    try:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            lib_logger.warning(f"Invalid JSON from {provider_label} models: {e}, body={response.text[:200]}")
            return []
        return parse_fn(data)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (401, 403):
            lib_logger.warning(f"Auth error fetching {provider_label} models: {e.response.status_code}")
        elif e.response.status_code >= 500:
            lib_logger.warning(f"Server error fetching {provider_label} models: {e.response.status_code}")
        else:
            lib_logger.error(f"HTTP error fetching {provider_label} models: {e}")
        return []
    except httpx.RequestError as e:
        lib_logger.error(f"Failed to fetch {provider_label} models: {e}")
        return []


__all__ = [
    "DEFAULT_GENERIC_SAFETY_SETTINGS",
    "DEFAULT_SAFETY_SETTINGS",
    "DEFAULT_GEMINI_SAFETY_SETTINGS_MAP",
    "GoogleQuotaTrackerBase",
    "fetch_provider_models",
]
