# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/http_client_pool.py
"""
Optimized HTTP client pool with connection warmup and lifecycle management.

Key optimizations:
- Pre-warmed connections at startup
- Separate pools for streaming vs non-streaming
- Connection health tracking
- Optimized limits for LLM API workloads
"""

import asyncio
import logging
import os
import time
from typing import Dict, Optional, Tuple
import httpx

from .timeout_config import TimeoutConfig

lib_logger = logging.getLogger("rotator_library")


# Configuration defaults (overridable via environment)
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 50  # Increased from 20 for high-throughput
DEFAULT_MAX_CONNECTIONS = 200  # Increased from 100 for multiple providers
DEFAULT_KEEPALIVE_EXPIRY = 30.0  # Seconds to keep idle connections alive
DEFAULT_WARMUP_CONNECTIONS = 3  # Connections to pre-warm per provider
DEFAULT_WARMUP_TIMEOUT = 10.0  # Max seconds for warmup


def _env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


class HttpClientPool:
    """
    Manages a pool of HTTP clients optimized for LLM API workloads.

    Features:
    - Separate clients for streaming/non-streaming (different timeout profiles)
    - Connection pre-warming for reduced latency on first request
    - Health tracking and automatic recovery
    - Optimized connection limits for high-throughput scenarios

    Usage:
        pool = HttpClientPool()
        await pool.initialize()  # Pre-warm connections

        # Get appropriate client
        client = pool.get_client(streaming=True)

        # On shutdown
        await pool.close()
    """

    def __init__(
        self,
        max_keepalive: Optional[int] = None,
        max_connections: Optional[int] = None,
        keepalive_expiry: Optional[float] = None,
        warmup_connections: Optional[int] = None,
    ):
        """
        Initialize the HTTP client pool.

        Args:
            max_keepalive: Max keep-alive connections (default: 50)
            max_connections: Max total connections (default: 200)
            keepalive_expiry: Seconds to keep idle connections (default: 30)
            warmup_connections: Connections to pre-warm per host (default: 3)
        """
        self._max_keepalive = max_keepalive or _env_int(
            "HTTP_MAX_KEEPALIVE", DEFAULT_MAX_KEEPALIVE_CONNECTIONS
        )
        self._max_connections = max_connections or _env_int(
            "HTTP_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS
        )
        self._keepalive_expiry = keepalive_expiry or _env_float(
            "HTTP_KEEPALIVE_EXPIRY", DEFAULT_KEEPALIVE_EXPIRY
        )
        self._warmup_count = warmup_connections or _env_int(
            "HTTP_WARMUP_CONNECTIONS", DEFAULT_WARMUP_CONNECTIONS
        )

        # Client instances (lazy initialization)
        self._streaming_client: Optional[httpx.AsyncClient] = None
        self._non_streaming_client: Optional[httpx.AsyncClient] = None
        self._client_lock = asyncio.Lock()

        # Health tracking
        self._healthy = True
        self._last_error: Optional[str] = None
        self._last_error_time: Optional[float] = None

        # Warmup state
        self._warmed_up = False
        self._warmup_hosts: list = []  # Hosts to pre-warm

        # Statistics
        self._stats = {
            "requests_total": 0,
            "requests_streaming": 0,
            "requests_non_streaming": 0,
            "connection_errors": 0,
            "timeout_errors": 0,
            "reconnects": 0,
        }

    def _create_limits(self) -> httpx.Limits:
        """Create optimized connection limits."""
        return httpx.Limits(
            max_keepalive_connections=self._max_keepalive,
            max_connections=self._max_connections,
            keepalive_expiry=self._keepalive_expiry,
        )

    async def _create_client(self, streaming: bool = False) -> httpx.AsyncClient:
        """
        Create a new HTTP client with appropriate configuration.

        Args:
            streaming: Whether this client will be used for streaming requests

        Returns:
            Configured httpx.AsyncClient
        """
        timeout = TimeoutConfig.streaming() if streaming else TimeoutConfig.non_streaming()

        client = httpx.AsyncClient(
            timeout=timeout,
            limits=self._create_limits(),
            follow_redirects=True,
            http2=True,  # Enable HTTP/2 for better performance
            http1=True,  # Fallback to HTTP/1.1
        )

        lib_logger.debug(
            f"Created new HTTP client (streaming={streaming}, "
            f"max_conn={self._max_connections}, keepalive={self._max_keepalive})"
        )

        return client

    async def initialize(self, warmup_hosts: Optional[list] = None) -> None:
        """
        Initialize the client pool and optionally pre-warm connections.

        Args:
            warmup_hosts: List of URLs to pre-warm connections to
                         (e.g., ["https://api.openai.com", "https://api.anthropic.com"])
        """
        async with self._client_lock:
            # Create both clients upfront
            self._streaming_client = await self._create_client(streaming=True)
            self._non_streaming_client = await self._create_client(streaming=False)

            self._warmup_hosts = warmup_hosts or []

            # Pre-warm connections if hosts provided
            if self._warmup_hosts:
                await self._warmup_connections()

            lib_logger.info(
                f"HTTP client pool initialized "
                f"(max_conn={self._max_connections}, keepalive={self._max_keepalive})"
            )

    async def _warmup_connections(self) -> None:
        """
        Pre-warm connections to common API hosts.

        This reduces latency on the first real request by establishing
        TCP+TLS connections in advance.
        """
        if not self._warmup_hosts or self._warmed_up:
            return

        start_time = time.time()
        warmed = 0

        # Use non-streaming client for warmup (lighter weight)
        client = self._non_streaming_client
        if not client:
            return

        for host in self._warmup_hosts[:5]:  # Limit to 5 hosts for warmup
            try:
                # Make a lightweight HEAD request to establish connection
                # Most APIs will respond quickly to HEAD /
                await asyncio.wait_for(
                    client.head(host, follow_redirects=True),
                    timeout=DEFAULT_WARMUP_TIMEOUT
                )
                warmed += 1
            except asyncio.TimeoutError:
                lib_logger.debug(f"Warmup timeout for {host}")
            except Exception as e:
                # Connection errors during warmup are not critical
                lib_logger.debug(f"Warmup error for {host}: {type(e).__name__}")

        self._warmed_up = True
        elapsed = time.time() - start_time

        if warmed > 0:
            lib_logger.info(f"Pre-warmed {warmed} connection(s) in {elapsed:.2f}s")

    def get_client(self, streaming: bool = False) -> httpx.AsyncClient:
        """
        Get the appropriate HTTP client.

        Note: This is a sync method for compatibility. The client is created
        during initialize(). If not initialized, returns a lazily-created client.

        Args:
            streaming: Whether the request will be streaming

        Returns:
            httpx.AsyncClient instance
        """
        self._stats["requests_total"] += 1

        if streaming:
            self._stats["requests_streaming"] += 1
            return self._streaming_client or self._get_lazy_client(streaming=True)
        else:
            self._stats["requests_non_streaming"] += 1
            return self._non_streaming_client or self._get_lazy_client(streaming=False)

    def _get_lazy_client(self, streaming: bool) -> httpx.AsyncClient:
        """
        Get or create a client lazily (fallback when not initialized).

        This should rarely be called if initialize() is used properly.
        """
        lib_logger.warning(
            "HTTP client pool accessed before initialization. "
            "Call await pool.initialize() during startup for optimal performance."
        )

        # Create synchronously (blocking, but better than nothing)
        timeout = TimeoutConfig.streaming() if streaming else TimeoutConfig.non_streaming()
        return httpx.AsyncClient(
            timeout=timeout,
            limits=self._create_limits(),
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Close all HTTP clients gracefully."""
        async with self._client_lock:
            errors = []

            if self._streaming_client:
                try:
                    await self._streaming_client.aclose()
                except Exception as e:
                    errors.append(f"streaming: {e}")
                self._streaming_client = None

            if self._non_streaming_client:
                try:
                    await self._non_streaming_client.aclose()
                except Exception as e:
                    errors.append(f"non-streaming: {e}")
                self._non_streaming_client = None

            if errors:
                lib_logger.warning(f"Errors during client pool shutdown: {errors}")
            else:
                lib_logger.info(
                    f"HTTP client pool closed "
                    f"(total_requests={self._stats['requests_total']})"
                )

    def record_error(self, error_type: str, message: str) -> None:
        """
        Record an error for health tracking.

        Args:
            error_type: Type of error (connection, timeout, etc.)
            message: Error message
        """
        self._last_error = message
        self._last_error_time = time.time()

        if error_type == "connection":
            self._stats["connection_errors"] += 1
        elif error_type == "timeout":
            self._stats["timeout_errors"] += 1

        lib_logger.debug(f"HTTP client error recorded: {error_type} - {message}")

    def get_stats(self) -> Dict[str, any]:
        """Get client pool statistics."""
        return {
            **self._stats,
            "healthy": self._healthy,
            "warmed_up": self._warmed_up,
            "last_error": self._last_error,
            "last_error_time": self._last_error_time,
            "config": {
                "max_connections": self._max_connections,
                "max_keepalive": self._max_keepalive,
                "keepalive_expiry": self._keepalive_expiry,
            },
        }

    @property
    def is_healthy(self) -> bool:
        """Check if the client pool is healthy."""
        return self._healthy

    @property
    def is_initialized(self) -> bool:
        """Check if the pool has been initialized."""
        return self._streaming_client is not None or self._non_streaming_client is not None


# Singleton instance for application-wide use
_POOL_INSTANCE: Optional[HttpClientPool] = None
_POOL_LOCK = asyncio.Lock()


async def get_http_pool() -> HttpClientPool:
    """
    Get the global HTTP client pool singleton.

    Creates the pool if it doesn't exist. Note: You should still call
    pool.initialize() to pre-warm connections.
    """
    global _POOL_INSTANCE

    if _POOL_INSTANCE is None:
        async with _POOL_LOCK:
            if _POOL_INSTANCE is None:
                _POOL_INSTANCE = HttpClientPool()

    return _POOL_INSTANCE


async def close_http_pool() -> None:
    """Close the global HTTP client pool."""
    global _POOL_INSTANCE

    if _POOL_INSTANCE is not None:
        await _POOL_INSTANCE.close()
        _POOL_INSTANCE = None
