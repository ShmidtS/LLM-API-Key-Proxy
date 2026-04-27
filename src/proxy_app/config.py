# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""Centralized defaults for proxy application runtime tuning."""

import os


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_CONCURRENT_REQUESTS = 1000
DEFAULT_UVICORN_BACKLOG = 2048
DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT = 15
DEFAULT_GZIP_MIN_SIZE = 2048
DEFAULT_GZIP_COMPRESSION_LEVEL = 3
