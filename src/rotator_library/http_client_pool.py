# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/http_client_pool.py
"""
Optimized HTTP client pool with connection warmup and lifecycle management.

Key optimizations:
- Pre-warmed connections at startup
- Separate pools for streaming vs non-streaming
- Connection health tracking
- Optimized limits for LLM API workloads
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import os
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union
import httpx

from .config.defaults import (
    HTTP_COMPRESS_MIN_SIZE,
    HTTP_COMPRESS_REQUESTS,
    HTTP_COMPRESSION_THRESHOLD,
    HTTP_GZIP_MAX_WORKERS,
    HTTP_KEEPALIVE_EXPIRY,
    HTTP_MAX_CONNECTIONS_POSIX,
    HTTP_MAX_CONNECTIONS_WINDOWS,
    HTTP_MAX_KEEPALIVE_POSIX,
    HTTP_MAX_KEEPALIVE_WINDOWS,
    HTTP_SSL_VERIFY_DEFAULT,
    HTTP_STREAMING_KEEPALIVE_EXPIRY,
    HTTP_STREAMING_MAX_CONNECTIONS_POSIX,
    HTTP_STREAMING_MAX_CONNECTIONS_WINDOWS,
    HTTP_STREAMING_MAX_KEEPALIVE_POSIX,
    HTTP_STREAMING_MAX_KEEPALIVE_WINDOWS,
    HTTP_WARMUP_CONNECTIONS,
    HTTP_WARMUP_HOST_LIMIT,
)
from .utils.singleton import SingletonMeta

from .timeout_config import TimeoutConfig
from .config import env_bool as _env_bool, env_float as _env_float, env_int as _env_int

lib_logger = logging.getLogger("rotator_library")


# Shared ThreadPoolExecutor for gzip compression across all GzipRequestTransport instances
_gzip_executor: Optional[ThreadPoolExecutor] = None
_gzip_executor_lock = threading.Lock()


def _get_gzip_executor() -> ThreadPoolExecutor:
    global _gzip_executor
    if _gzip_executor is None:
        with _gzip_executor_lock:
            if _gzip_executor is None:
                _gzip_executor = ThreadPoolExecutor(
                    max_workers=HTTP_GZIP_MAX_WORKERS,
                    thread_name_prefix='gzip-compress',
                )
                lib_logger.debug(
                    'Created shared gzip executor (max_workers=%s, id=%s)',
                    HTTP_GZIP_MAX_WORKERS,
                    id(_gzip_executor),
                )
    return _gzip_executor


def shutdown_gzip_executor() -> None:
    """Shut down the shared gzip compression executor."""
    global _gzip_executor
    if _gzip_executor is not None:
        with _gzip_executor_lock:
            if _gzip_executor is not None:
                executor = _gzip_executor
                _gzip_executor = None
                executor.shutdown(wait=False)
                lib_logger.debug(
                    'Shutdown shared gzip executor (id=%s)',
                    id(executor),
                )


# Platform-aware connection pool limits
# Windows SelectorEventLoop has much lower file descriptor limits than Linux
_IS_WIN = os.name == "nt"
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = (
    HTTP_MAX_KEEPALIVE_WINDOWS if _IS_WIN else HTTP_MAX_KEEPALIVE_POSIX
)
DEFAULT_MAX_CONNECTIONS = HTTP_MAX_CONNECTIONS_WINDOWS if _IS_WIN else HTTP_MAX_CONNECTIONS_POSIX
DEFAULT_KEEPALIVE_EXPIRY = HTTP_KEEPALIVE_EXPIRY
DEFAULT_WARMUP_CONNECTIONS = HTTP_WARMUP_CONNECTIONS
DEFAULT_STREAMING_MAX_CONNECTIONS = (
    HTTP_STREAMING_MAX_CONNECTIONS_WINDOWS if _IS_WIN else HTTP_STREAMING_MAX_CONNECTIONS_POSIX
)
DEFAULT_STREAMING_MAX_KEEPALIVE = (
    HTTP_STREAMING_MAX_KEEPALIVE_WINDOWS if _IS_WIN else HTTP_STREAMING_MAX_KEEPALIVE_POSIX
)
DEFAULT_STREAMING_KEEPALIVE_EXPIRY = HTTP_STREAMING_KEEPALIVE_EXPIRY
DEFAULT_SSL_VERIFY = HTTP_SSL_VERIFY_DEFAULT
DEFAULT_HTTP2_ENABLED = not _IS_WIN
_COMPRESSION_THRESHOLD = HTTP_COMPRESSION_THRESHOLD


from .ssl_patch import AZURE_COMPATIBLE_CIPHERS


class GzipRequestTransport(httpx.AsyncHTTPTransport):
    """Compress large JSON request bodies before sending them."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._compress_min_size = HTTP_COMPRESS_MIN_SIZE
        self._compress_enabled = HTTP_COMPRESS_REQUESTS

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Compress request body if eligible before sending."""
        if self._compress_enabled and request.content:
            content_len = len(request.content)

            if content_len >= self._compress_min_size:
                content_type = request.headers.get("content-type", "")
                if "application/json" in content_type:
                    if "content-encoding" not in {k.lower() for k in request.headers}:
                        loop = asyncio.get_running_loop()
                        compressed = await loop.run_in_executor(
                            _get_gzip_executor(), gzip.compress, request.content
                        )

                        if len(compressed) < content_len * _COMPRESSION_THRESHOLD:
                            # Create new request with compressed content
                            # (request.content is read-only in httpx)
                            headers = dict(request.headers)
                            headers["content-encoding"] = "gzip"
                            headers["content-length"] = str(len(compressed))
                            request = httpx.Request(
                                method=request.method,
                                url=request.url,
                                content=compressed,
                                headers=headers,
                            )

                            lib_logger.debug(
                                "Gzip compressed: %d -> %d bytes (%.1f%% reduction)",
                                content_len, len(compressed),
                                100 * (1 - len(compressed) / content_len),
                            )

        return await super().handle_async_request(request)


def _env_ssl_verify() -> Union[bool, List[str]]:
    """
    Parse SSL verification configuration from environment.

    Returns:
    True: Standard SSL verification (default)
    False: Disable SSL verification globally
    List[str]: List of hosts to skip SSL verification for
    """
    # Check global SSL verification setting
    if _env_bool("DISABLE_TLS_VERIFY", False) or not _env_bool("HTTP_SSL_VERIFY", DEFAULT_SSL_VERIFY):
        lib_logger.warning(
            "SSL certificate verification is DISABLED globally via DISABLE_TLS_VERIFY or HTTP_SSL_VERIFY. "
            "This is insecure and should only be used for testing."
        )
        return False

    # Check per-host SSL verification overrides
    hosts_str = os.getenv("HTTP_SSL_VERIFY_HOSTS", "").strip()

    # Default hosts that need SSL bypass due to Azure SSLV3_ALERT_HANDSHAKE_FAILURE
    DEFAULT_SSL_BYPASS_HOSTS = [
        "chatgpt.com",
    ]

    hosts = []
    if hosts_str:
        hosts = [h.strip() for h in hosts_str.split(",") if h.strip()]

    # Add default hosts if not already present
    for default_host in DEFAULT_SSL_BYPASS_HOSTS:
        if default_host not in hosts:
            hosts.append(default_host)

    if hosts:
        lib_logger.info(
            "SSL certificate verification DISABLED for hosts: %s. "
            "These hosts will skip SSL verification.",
            hosts,
        )
        return hosts

    return True


class HttpClientPool(metaclass=SingletonMeta):
    """Manage reusable HTTP clients for streaming and non-streaming requests."""

    def __init__(
        self,
        max_keepalive: Optional[int] = None,
        max_connections: Optional[int] = None,
        keepalive_expiry: Optional[float] = None,
        warmup_connections: Optional[int] = None,
        ssl_verify: Optional[Union[bool, List[str]]] = None,
    ) -> None:
        """
        Initialize the HTTP client pool.

        Args:
        max_keepalive: Max keep-alive connections (default: 50)
        max_connections: Max total connections (default: 200)
        keepalive_expiry: Seconds to keep idle connections (default: 30)
        warmup_connections: Connections to pre-warm per host (default: 3)
        ssl_verify: SSL verification setting (default: from env or True)
        - True: Standard SSL verification
        - False: Disable SSL verification globally
        - List[str]: List of hosts to skip SSL verification for
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

        # Streaming-specific limits (longer keepalive, fewer concurrent connections)
        self._streaming_max_connections = _env_int(
            "HTTP_STREAMING_MAX_CONNECTIONS", DEFAULT_STREAMING_MAX_CONNECTIONS
        )
        self._streaming_max_keepalive = _env_int(
            "HTTP_STREAMING_MAX_KEEPALIVE", DEFAULT_STREAMING_MAX_KEEPALIVE
        )
        self._streaming_keepalive_expiry = _env_float(
            "HTTP_STREAMING_KEEPALIVE_EXPIRY", DEFAULT_STREAMING_KEEPALIVE_EXPIRY
        )

        # SSL configuration
        self._ssl_verify = ssl_verify if ssl_verify is not None else _env_ssl_verify()

        # Log SSL configuration
        if isinstance(self._ssl_verify, bool):
            if not self._ssl_verify:
                lib_logger.warning(
                    "HTTP client pool: SSL verification DISABLED globally"
                )
        else:
            lib_logger.info(
                "HTTP client pool: SSL verification disabled for hosts: %s",
                self._ssl_verify,
            )

        # Create SSL context once (optimized - reused for all clients)
        self._ssl_context = self._create_ssl_context()

        # HTTP/2 configuration (can be disabled for problematic providers)
        self._http2_enabled = _env_bool("HTTP2_ENABLED", DEFAULT_HTTP2_ENABLED)
        if not self._http2_enabled:
            reason = (
                "SelectorEventLoop incompatibility"
                if _IS_WIN
                else "HTTP2_ENABLED env var"
            )
            lib_logger.warning("HTTP/2 is DISABLED (%s). Using HTTP/1.1 only.", reason)

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
        self._warmup_task: Optional[asyncio.Task] = None

        # Orphan close task tracking (prevents GC from collecting fire-and-forget tasks)
        self._orphan_close_tasks: set[asyncio.Task] = set()

        # Statistics
        # No lock needed: asyncio is single-threaded, so integer counter
        # mutations are atomic between yield points (read-modify-write of
        # plain ints never interleaves with another coroutine).
        self._stats = {
            "requests_total": 0,
            "requests_streaming": 0,
            "requests_non_streaming": 0,
            "connection_errors": 0,
            "timeout_errors": 0,
            "reconnects": 0,
        }

    def _create_ssl_context(self) -> ssl.SSLContext:
        """
        Create SSL context with TLS 1.2+ for Azure compatibility.

        This is called once in __init__ and reused for all clients.
        Use _refresh_ssl_context() if you need to update SSL settings.
        """
        ssl_context = ssl.create_default_context()
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2

        # Set Azure-compatible cipher suites to fix SSLV3_ALERT_HANDSHAKE_FAILURE
        try:
            ssl_context.set_ciphers(AZURE_COMPATIBLE_CIPHERS)
        except ssl.SSLError:
            lib_logger.debug("SSL cipher configuration failed, using defaults", exc_info=True)
            pass  # Use default ciphers if set_ciphers fails

        if not self._ssl_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        return ssl_context

    def _refresh_ssl_context(self) -> None:
        """
        Refresh the SSL context (e.g., after SSL verify config changes).

        Note: This only affects new clients. Existing clients retain
        their original SSL context until they are recreated.
        """
        self._ssl_context = self._create_ssl_context()
        lib_logger.info("SSL context refreshed")

    def _create_limits(self, streaming: bool = False) -> httpx.Limits:
        """Create optimized connection limits.

        Streaming connections are long-lived (minutes), so they need fewer
        total slots but longer keepalive.  Non-streaming connections are
        short-lived (seconds), so they benefit from more slots and shorter
        keepalive to free resources quickly.
        """
        if streaming:
            return httpx.Limits(
                max_connections=self._streaming_max_connections,
                max_keepalive_connections=self._streaming_max_keepalive,
                keepalive_expiry=self._streaming_keepalive_expiry,
            )
        return httpx.Limits(
            max_connections=self._max_connections,
            max_keepalive_connections=self._max_keepalive,
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
        timeout = (
            TimeoutConfig.streaming() if streaming else TimeoutConfig.non_streaming()
        )

        # Use pre-created SSL context (optimized - created once in __init__)
        ssl_context = self._ssl_context

        # Build client kwargs
        client_kwargs = {
            "timeout": timeout,
            "limits": self._create_limits(streaming=streaming),
            "follow_redirects": True,
            "http2": self._http2_enabled,
            "http1": True,
            "verify": ssl_context,
        }

        # Use GzipRequestTransport for request body compression
        # HTTP/2 already provides header compression (HPACK), so gzip on top
        # is unnecessary and silently overrides the HTTP/2 transport.
        if HTTP_COMPRESS_REQUESTS and not self._http2_enabled:
            client_kwargs["transport"] = GzipRequestTransport(
                limits=self._create_limits(streaming=streaming),
                verify=ssl_context,
                http2=False,
            )

        # Note: httpx does not support custom DNS resolver.
        # DNS resolution is handled by the OS/aiohttp resolver.
        # If custom DNS is needed, set DNS env vars before process start.

        client = httpx.AsyncClient(**client_kwargs)

        lib_logger.debug(
            "Created new HTTP client (streaming=%s, "
            "max_conn=%d, keepalive=%d, "
            "ssl_verify=%s, http2=%s)",
            streaming, self._max_connections, self._max_keepalive,
            self._ssl_verify, self._http2_enabled,
        )

        return client

    async def initialize(self, warmup_hosts: Optional[list] = None) -> None:
        """Initialize clients and optionally pre-warm configured connections."""
        async with self._client_lock:
            # Create both clients upfront
            self._streaming_client = await self._create_client(streaming=True)
            self._non_streaming_client = await self._create_client(streaming=False)

            self._warmup_hosts = warmup_hosts or []

            # Pre-warm connections if hosts provided (background task)
            if self._warmup_hosts:
                self._warmup_task = asyncio.create_task(self._warmup_connections())

            lib_logger.info(
                "HTTP client pool initialized "
                "(max_conn=%d, keepalive=%d)",
                self._max_connections, self._max_keepalive,
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
        ssl_errors = []

        # Use non-streaming client for warmup (lighter weight)
        client = self._non_streaming_client
        if not client:
            return

        # Build list of all warmup tasks (parallel execution)
        warmup_tasks = []
        for host in self._warmup_hosts[:HTTP_WARMUP_HOST_LIMIT]:  # Limit to 5 hosts for warmup
            for _ in range(self._warmup_count):
                warmup_tasks.append(client.get(host, follow_redirects=True))

        # Execute all warmup requests in parallel with graceful error handling
        results = await asyncio.gather(*warmup_tasks, return_exceptions=True)

        # Process results and track errors
        task_idx = 0
        for host in self._warmup_hosts[:HTTP_WARMUP_HOST_LIMIT]:
            for _ in range(self._warmup_count):
                result = results[task_idx]
                task_idx += 1
                if isinstance(result, Exception):
                    if isinstance(result, asyncio.TimeoutError):
                        lib_logger.debug("Warmup timeout for %s", host)
                    elif isinstance(result, httpx.ConnectError):
                        error_str = str(result).lower()
                        if (
                            "ssl" in error_str
                            or "certificate" in error_str
                            or "tls" in error_str
                        ):
                            ssl_errors.append((host, str(result)))
                            lib_logger.warning(
                                "SSL/TLS connection error during warmup for %s: %s. "
                                "Consider adding '%s' to HTTP_SSL_VERIFY_HOSTS environment variable.",
                                host, result, host,
                            )
                        else:
                            lib_logger.debug(
                                "Warmup connection error for %s: %s: %s",
                                host, type(result).__name__, result,
                            )
                    else:
                        lib_logger.debug(
                            "Warmup error for %s: %s: %s",
                            host, type(result).__name__, type(result).__name__ + ": " + str(result),
                        )
                else:
                    warmed += 1

        self._warmed_up = True
        elapsed = time.time() - start_time

        if warmed > 0:
            lib_logger.info("Pre-warmed %d connection(s) in %.2fs", warmed, elapsed)

        # Log summary of SSL errors if any occurred
        if ssl_errors:
            lib_logger.warning(
                "SSL/TLS errors occurred during warmup for %d host(s). "
                "To disable SSL verification for specific hosts, set: "
                "HTTP_SSL_VERIFY_HOSTS=%s",
                len(ssl_errors),
                ','.join(h for h, _ in ssl_errors),
            )

    def _is_client_closed(self, client: Optional[httpx.AsyncClient]) -> bool:
        """
        Check if a client is closed or unusable.

        Args:
        client: The client to check

        Returns:
        True if the client is closed or None, False otherwise
        """
        if client is None:
            return True
        return client.is_closed

    async def _ensure_client(self, streaming: bool) -> httpx.AsyncClient:
        """
        Ensure a valid client exists for the given mode, recreating if necessary.

        Creates clients outside the lock to avoid blocking concurrent access
        during expensive _create_client() (SSL/DNS setup).

        Args:
        streaming: Whether to get streaming client

        Returns:
        Valid httpx.AsyncClient instance
        """
        # Fast path: check under lock if client is already valid
        async with self._client_lock:
            client = self._streaming_client if streaming else self._non_streaming_client
            if not self._is_client_closed(client):
                return client

        # Slow path: recreate outside lock so other requests aren't blocked
        lib_logger.warning(
            "%s HTTP client was closed, recreating...",
            'Streaming' if streaming else 'Non-streaming',
        )
        new_client = await self._create_client(streaming=streaming)

        # Assign under lock — cheap pointer swap
        async with self._client_lock:
            if streaming:
                # Another coroutine may have already recreated it
                if self._is_client_closed(self._streaming_client):
                    self._streaming_client = new_client
                    self._stats["reconnects"] += 1
                else:
                    self._schedule_orphan_close(new_client)
                return self._streaming_client
            else:
                if self._is_client_closed(self._non_streaming_client):
                    self._non_streaming_client = new_client
                    self._stats["reconnects"] += 1
                else:
                    self._schedule_orphan_close(new_client)
                return self._non_streaming_client

    def get_client(self, streaming: bool = False) -> httpx.AsyncClient:
        """Return the HTTP client for the requested streaming mode."""
        # Note: _stats mutations in sync methods are safe in single-threaded asyncio;
        # no yield point exists between read and write of these integers.
        self._stats["requests_total"] += 1

        if streaming:
            self._stats["requests_streaming"] += 1
            client = self._streaming_client or self._get_lazy_client(streaming=True)
        else:
            self._stats["requests_non_streaming"] += 1
            client = self._non_streaming_client or self._get_lazy_client(
                streaming=False
            )

        # NOTE: We deliberately do not attempt to recreate closed clients here.
        # Sync lock-checking (self._client_lock.locked()) is a TOCTOU race, and
        # creating clients without the lock can produce duplicates.  Callers that
        # need automatic recovery should use get_client_async() instead.
        if client.is_closed:
            lib_logger.debug(
                "get_client() returned a closed client — "
                "use get_client_async() for automatic recovery"
            )

        return client

    async def get_client_async(self, streaming: bool = False) -> httpx.AsyncClient:
        """Return a healthy HTTP client for the requested streaming mode."""
        # _stats mutations are safe without a lock in single-threaded asyncio
        self._stats["requests_total"] += 1
        if streaming:
            self._stats["requests_streaming"] += 1
        else:
            self._stats["requests_non_streaming"] += 1

        return await self._ensure_client(streaming)

    def _schedule_orphan_close(self, client: httpx.AsyncClient) -> None:
        """Schedule closure of an orphaned client created by a concurrent lazy-init race."""
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(client.aclose())
            self._orphan_close_tasks.add(task)
            task.add_done_callback(self._orphan_close_tasks.discard)
        except RuntimeError:
            lib_logger.debug("Could not schedule orphan client close")

    def _get_lazy_client(self, streaming: bool) -> httpx.AsyncClient:
        """
        Get or create a client lazily (fallback when not initialized).

        The created client is stored in the pool so that close() can shut it
        down gracefully.  This prevents connection leaks when initialize() is
        not called before get_client().
        """
        lib_logger.warning(
            "HTTP client pool accessed before initialization. "
            "Call await pool.initialize() during startup for optimal performance."
        )

        # Create synchronously (blocking, but better than nothing)
        timeout = (
            TimeoutConfig.streaming() if streaming else TimeoutConfig.non_streaming()
        )

        # Build client kwargs consistent with _create_client()
        client_kwargs = {
            "timeout": timeout,
            "limits": self._create_limits(streaming=streaming),
            "follow_redirects": True,
            "http2": self._http2_enabled,
            "http1": True,
            "verify": self._ssl_context,
        }

        if HTTP_COMPRESS_REQUESTS and not self._http2_enabled:
            client_kwargs["transport"] = GzipRequestTransport(
                limits=self._create_limits(streaming=streaming),
                verify=self._ssl_context,
                http2=False,
            )
        elif HTTP_COMPRESS_REQUESTS and self._http2_enabled:
            lib_logger.info(
                "HTTP/2 takes priority over gzip request compression "
                "(HTTP/2 already provides HPACK header compression). "
                "Set HTTP2_ENABLED=false if you need gzip compression instead."
            )

        client = httpx.AsyncClient(**client_kwargs)

        # Store in pool so close() can clean it up
        # Guard: close() may have set the attr to None concurrently
        if streaming:
            if self._streaming_client is None:
                self._streaming_client = client
            else:
                self._schedule_orphan_close(client)
            return self._streaming_client
        else:
            if self._non_streaming_client is None:
                self._non_streaming_client = client
            else:
                self._schedule_orphan_close(client)
            return self._non_streaming_client

    async def close(self) -> None:
        """Close all HTTP clients gracefully."""
        # Cancel warmup task before acquiring lock (it holds no lock itself)
        if self._warmup_task is not None and not self._warmup_task.done():
            self._warmup_task.cancel()
            try:
                await self._warmup_task
            except asyncio.CancelledError:
                lib_logger.debug("Warmup task cancelled during close", exc_info=True)
            self._warmup_task = None

        async with self._client_lock:
            errors = []

            if self._streaming_client:
                try:
                    await self._streaming_client.aclose()
                except (httpx.HTTPError, ConnectionError, OSError, TimeoutError) as e:
                    errors.append(f"streaming: {e}")
                self._streaming_client = None

            if self._non_streaming_client:
                try:
                    await self._non_streaming_client.aclose()
                except (httpx.HTTPError, ConnectionError, OSError, TimeoutError) as e:
                    errors.append(f"non-streaming: {e}")
                self._non_streaming_client = None

            if errors:
                lib_logger.warning("Errors during client pool shutdown: %s", errors)
            else:
                lib_logger.info(
                    "HTTP client pool closed "
                    "(total_requests=%d)",
                    self._stats['requests_total'],
                )

    def record_error(self, error_type: str, message: str) -> None:
        """Record an HTTP client error for health tracking."""
        self._last_error = message
        self._last_error_time = time.time()

        if error_type == "connection":
            self._stats["connection_errors"] += 1
        elif error_type == "timeout":
            self._stats["timeout_errors"] += 1

        lib_logger.debug("HTTP client error recorded: %s - %s", error_type, message)

    def get_stats(self) -> Dict[str, Any]:
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
                "ssl_verify": self._ssl_verify,
                "http2_enabled": self._http2_enabled,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """Report health status for each managed HTTP client."""
        health = {
            "streaming_client": "unknown",
            "non_streaming_client": "unknown",
            "overall_healthy": True,
        }

        # Check streaming client
        if self._streaming_client is None:
            health["streaming_client"] = "not_initialized"
        elif self._is_client_closed(self._streaming_client):
            health["streaming_client"] = "closed"
            health["overall_healthy"] = False
        else:
            health["streaming_client"] = "healthy"

        # Check non-streaming client
        if self._non_streaming_client is None:
            health["non_streaming_client"] = "not_initialized"
        elif self._is_client_closed(self._non_streaming_client):
            health["non_streaming_client"] = "closed"
            health["overall_healthy"] = False
        else:
            health["non_streaming_client"] = "healthy"

        self._healthy = health["overall_healthy"]
        return health

    async def recover(self) -> bool:
        """Recreate closed or unhealthy HTTP clients."""
        recovered = []
        new_streaming = None
        new_non_streaming = None

        # Snapshot client state under lock to avoid TOCTOU race
        async with self._client_lock:
            need_streaming = self._is_client_closed(self._streaming_client)
            need_non_streaming = self._is_client_closed(self._non_streaming_client)

        # Create clients outside lock — SSL/DNS setup can take seconds
        if need_streaming:
            try:
                new_streaming = await self._create_client(streaming=True)
                recovered.append("streaming")
            except (httpx.HTTPError, ConnectionError, OSError, TimeoutError) as e:
                lib_logger.error(f"Failed to recover streaming client: {e}")

        if need_non_streaming:
            try:
                new_non_streaming = await self._create_client(streaming=False)
                recovered.append("non-streaming")
            except (httpx.HTTPError, ConnectionError, OSError, TimeoutError) as e:
                lib_logger.error(f"Failed to recover non-streaming client: {e}")

        # Assign under lock — cheap pointer swaps, no I/O
        # Use same guard as _ensure_client: only assign if still closed
        async with self._client_lock:
            if new_streaming is not None:
                if self._is_client_closed(self._streaming_client):
                    self._streaming_client = new_streaming
                    self._stats["reconnects"] += 1
                else:
                    self._schedule_orphan_close(new_streaming)
            if new_non_streaming is not None:
                if self._is_client_closed(self._non_streaming_client):
                    self._non_streaming_client = new_non_streaming
                    self._stats["reconnects"] += 1
                else:
                    self._schedule_orphan_close(new_non_streaming)

            if recovered:
                lib_logger.info("HTTP client pool recovered: %s", ', '.join(recovered))
                self._healthy = True

            return len(recovered) > 0 or (
                self._streaming_client is not None
                and self._non_streaming_client is not None
            )

    @property
    def is_healthy(self) -> bool:
        """Check if the client pool is healthy."""
        # Quick synchronous check - for async health check use health_check()
        if self._is_client_closed(self._streaming_client):
            return False
        if self._is_client_closed(self._non_streaming_client):
            return False
        return self._healthy

    @property
    def is_initialized(self) -> bool:
        """Check if the pool has been initialized."""
        return (
            self._streaming_client is not None or self._non_streaming_client is not None
        )


# Singleton via SingletonMeta
async def get_http_pool() -> HttpClientPool:
    """Return the global HTTP client pool singleton."""
    return HttpClientPool()


async def close_http_pool() -> None:
    """Close the global HTTP client pool and reset the singleton."""
    pool = HttpClientPool.get_instance()
    if pool is not None:
        await pool.close()
        HttpClientPool.reset()
    shutdown_gzip_executor()
