# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from ..config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_GLOBAL_TIMEOUT,
    DEFAULT_ROTATION_TOLERANCE,
)


@dataclass
class ClientConfig:
    """Configuration for RotatingClient.

    All fields correspond to RotatingClient constructor keyword arguments.
    Can be passed directly via ``RotatingClient(config=...)`` or constructed
    implicitly from keyword arguments for backward compatibility.
    """

    api_keys: Optional[dict[str, list[str]]] = None
    oauth_credentials: Optional[dict[str, list[str]]] = None
    max_retries: int = DEFAULT_MAX_RETRIES
    usage_file_path: Optional[Union[str, Path]] = None
    configure_logging: bool = True
    global_timeout: int = DEFAULT_GLOBAL_TIMEOUT
    abort_on_callback_error: bool = True
    litellm_provider_params: Optional[dict[str, Any]] = None
    ignore_models: Optional[dict[str, list[str]]] = None
    whitelist_models: Optional[dict[str, list[str]]] = None
    enable_request_logging: bool = False
    max_concurrent_requests_per_key: Optional[dict[str, int]] = None
    rotation_tolerance: float = DEFAULT_ROTATION_TOLERANCE
    data_dir: Optional[Union[str, Path]] = None

    def __repr__(self) -> str:
        # Mask credential fields to prevent accidental secret leakage in logs
        fields = []
        for key, value in self.__dict__.items():
            if key in ("api_keys", "oauth_credentials") and value:
                masked = {k: "***" for k in value}
                fields.append(f"{key}={masked!r}")
            else:
                fields.append(f"{key}={value!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"
