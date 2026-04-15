# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/timeout_config.py
"""
Centralized timeout configuration for HTTP requests.

All values can be overridden via environment variables:
    TIMEOUT_CONNECT - Connection establishment timeout (default: 30s)
    TIMEOUT_WRITE - Request body send timeout (default: 30s)
    TIMEOUT_POOL - Connection pool acquisition timeout (default: 15s)
    TIMEOUT_READ_STREAMING - Read timeout between chunks for streaming (default: 300s / 5 min)
    TIMEOUT_READ_NON_STREAMING - Read timeout for non-streaming responses (default: 600s / 10 min)
"""

import os
import logging
from typing import Optional

import httpx

lib_logger = logging.getLogger("rotator_library")


class TimeoutConfig:
    """
    Centralized timeout configuration for HTTP requests.

    All values can be overridden via environment variables.
    """

    # Default values (in seconds)
    _CONNECT = 15.0
    _WRITE = 30.0
    _POOL = 15.0  # Reduced from 60s for faster failure detection
    _READ_STREAMING = 300.0  # 5 minutes between chunks
    _READ_NON_STREAMING = 300.0  # 5 minutes for full response (was 600s)

    # Cached httpx.Timeout instances
    _STREAMING_TIMEOUT: Optional[httpx.Timeout] = None
    _DEFAULT_TIMEOUT: Optional[httpx.Timeout] = None

    @classmethod
    def _get_env_float(cls, key: str, default: float) -> float:
        """Get a float value from environment variable, or return default."""
        value = os.environ.get(key)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                lib_logger.warning(
                    f"Invalid value for {key}: {value}. Using default: {default}"
                )
        return default

    @classmethod
    def connect(cls) -> float:
        """Connection establishment timeout."""
        return cls._get_env_float("TIMEOUT_CONNECT", cls._CONNECT)

    @classmethod
    def write(cls) -> float:
        """Request body send timeout."""
        return cls._get_env_float("TIMEOUT_WRITE", cls._WRITE)

    @classmethod
    def pool(cls) -> float:
        """Connection pool acquisition timeout."""
        return cls._get_env_float("TIMEOUT_POOL", cls._POOL)

    @classmethod
    def read_streaming(cls) -> float:
        """Read timeout between chunks for streaming requests."""
        return cls._get_env_float("TIMEOUT_READ_STREAMING", cls._READ_STREAMING)

    @classmethod
    def read_non_streaming(cls) -> float:
        """Read timeout for non-streaming responses."""
        return cls._get_env_float("TIMEOUT_READ_NON_STREAMING", cls._READ_NON_STREAMING)

    @classmethod
    def create_timeout(cls, is_streaming: bool = False) -> httpx.Timeout:
        """
        Return a cached httpx.Timeout instance based on streaming mode.

        Uses lazy initialization: the Timeout object is created once from
        current env vars and reused on subsequent calls. Changing TIMEOUT_*
        env vars at runtime has no effect until process restart — this is
        intentional, as timeouts are deployment-time configuration.
        """
        if is_streaming:
            if cls._STREAMING_TIMEOUT is None:
                cls._STREAMING_TIMEOUT = httpx.Timeout(
                    connect=cls.connect(),
                    read=cls.read_streaming(),
                    write=cls.write(),
                    pool=cls.pool(),
                )
            return cls._STREAMING_TIMEOUT
        else:
            if cls._DEFAULT_TIMEOUT is None:
                cls._DEFAULT_TIMEOUT = httpx.Timeout(
                    connect=cls.connect(),
                    read=cls.read_non_streaming(),
                    write=cls.write(),
                    pool=cls.pool(),
                )
            return cls._DEFAULT_TIMEOUT

    @classmethod
    def streaming(cls) -> httpx.Timeout:
        """
        Timeout configuration for streaming LLM requests.

        Uses a shorter read timeout (default 3 min) since we expect
        periodic chunks. If no data arrives for this duration, the
        connection is considered stalled.
        """
        return cls.create_timeout(is_streaming=True)

    @classmethod
    def non_streaming(cls) -> httpx.Timeout:
        """
        Timeout configuration for non-streaming LLM requests.

        Uses a longer read timeout (default 10 min) since the server
        may take significant time to generate the complete response
        before sending anything back.
        """
        return cls.create_timeout(is_streaming=False)
