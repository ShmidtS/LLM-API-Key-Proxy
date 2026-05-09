# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Centralized defaults for proxy application runtime tuning."""

import logging
import os


_logger = logging.getLogger(__name__)


def env_int(key: str, default: int) -> int:
    value = os.getenv(key, str(default))
    try:
        return int(value)
    except (TypeError, ValueError):
        _logger.warning("Invalid integer for %s=%r; using default %s", key, value, default)
        return default


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_CONCURRENT_REQUESTS = 1000
DEFAULT_UVICORN_BACKLOG = 2048
DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT = 15
DEFAULT_GLOBAL_TIMEOUT = 30
MIN_GLOBAL_TIMEOUT = 5
MAX_GLOBAL_TIMEOUT = 600
DEFAULT_GZIP_MIN_SIZE = 2048
DEFAULT_GZIP_COMPRESSION_LEVEL = 3
