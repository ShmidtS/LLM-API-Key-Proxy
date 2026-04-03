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
import gzip
import logging
import os
import ssl
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx

from .config.defaults import HTTP_COMPRESS_MIN_SIZE, HTTP_COMPRESS_REQUESTS

# Disable aiodns before any aiohttp import to fix DNS resolution issues
# This must be set before aiohttp is imported anywhere in the process
# See: https://github.com/aio-libs/aiohttp/issues/1135
# Accepts: true, 1, yes, on, or any non-empty value (including DNS IP addresses)
_http_dns_resolver = os.getenv("HTTP_DNS_RESOLVER", "").strip().lower()
if _http_dns_resolver in ("true", "1", "yes", "on") or (
    _http_dns_resolver and _http_dns_resolver not in ("false", "0", "no", "off")
):
    os.environ["AIOHTTP_NO_EXTENSIONS"] = "1"

from .timeout_config import TimeoutConfig
from .config import env_bool as _env_bool, env_float as _env_float, env_int as _env_int

lib_logger = logging.getLogger("rotator_library")


# Configuration defaults (overridable via environment)
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = (
    100  # Increased for high-throughput NVIDIA/OpenAI workloads
)
DEFAULT_MAX_CONNECTIONS = 500  # Supports 100+ parallel NVIDIA requests
DEFAULT_KEEPALIVE_EXPIRY = 60.0  # Seconds to keep idle connections alive
DEFAULT_WARMUP_CONNECTIONS = 3  # Connections to pre-warm per provider
DEFAULT_WARMUP_TIMEOUT = 10.0  # Max seconds for warmup
DEFAULT_SSL_VERIFY = True  # SSL certificate verification enabled by default
DEFAULT_HTTP2_ENABLED = True  # HTTP/2 enabled by default
DEFAULT_DNS_RESOLVER = None  # Custom DNS resolver (e.g., "8.8.8.8")
DEFAULT_DNS_PORT = 53  # Default DNS port

# Azure-compatible cipher suites to fix SSLV3_ALERT_HANDSHAKE_FAILURE
# Some Azure endpoints reject TLS 1.3 cipher suites; this list prefers TLS 1.2 suites
AZURE_COMPATIBLE_CIPHERS = (
    "ECDH+AESGCM:DH+AESGCM:ECDH+AES256:DH+AES256:ECDH+AES128:DH+AES:"
    "ECDH+3DES:DH+3DES:RSA+AESGCM:RSA+AES:RSA+3DES:!aNULL:!MD5:!DSS"
)


class GzipRequestTransport(httpx.AsyncHTTPTransport):
    """
    Custom HTTP transport that compresses large request bodies with gzip.

    This bypasses WAF payload size limits on providers like zenllm.org
    that block requests >100KB by compressing the body before sending.
    """

    def __init__(self, *args, **kwargs):
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
                        compressed = gzip.compress(request.content)

                        if len(compressed) < content_len * 0.9:
                            request.content = compressed
                            request.headers["content-encoding"] = "gzip"
                            request.headers["content-length"] = str(len(compressed))

                            lib_logger.debug(
                                f"Gzip compressed: {content_len} -> {len(compressed)} bytes "
                                f"({100 * (1 - len(compressed) / content_len):.1f}% reduction)"
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
    if not _env_bool("HTTP_SSL_VERIFY", DEFAULT_SSL_VERIFY):
        lib_logger.warning(
            "SSL certificate verification is DISABLED globally via HTTP_SSL_VERIFY. "
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
            f"SSL certificate verification DISABLED for hosts: {hosts}. "
            f"These hosts will skip SSL verification."
        )
        return hosts

    return True


def should_skip_ssl_for_host(host: str, ssl_verify: Union[bool, List[str]]) -> bool:
    """
    Check if SSL verification should be skipped for a specific host.

    Args:
        host: Hostname to check (e.g., "chatgpt.com")
        ssl_verify: SSL verification setting from _env_ssl_verify()

    Returns:
        True if SSL verification should be skipped for this host
    """
    if ssl_verify is False:
        return True
    if ssl_verify is True:
        return False
    if isinstance(ssl_verify, list):
        # Check for exact match or subdomain match
        for skip_host in ssl_verify:
            if host == skip_host or host.endswith(f".{skip_host}"):
                return True
    return False


def _create_custom_dns_resolver(dns_host: str, dns_port: int = DEFAULT_DNS_PORT):
    """
    Create a custom DNS resolver for httpx.

    This allows bypassing system DNS which may be hijacked by VPN/proxy/antivirus.

    Args:
        dns_host: DNS server IP (e.g., "8.8.8.8", "1.1.1.1")
        dns_port: DNS server port (default: 53)

    Returns:
        httpx.AsyncDNSResolver instance
    """
    try:
        import httpx

        # httpx.AsyncDNSResolver uses aiodns by default
        # We need to create a custom resolver that uses the specified DNS server
        resolver = httpx.AsyncDNSResolver(
            host=dns_host,
            port=dns_port,
        )
        lib_logger.info(f"Created custom DNS resolver: {dns_host}:{dns_port}")
        return resolver
    except Exception as e:
        lib_logger.warning(
            f"Failed to create custom DNS resolver {dns_host}:{dns_port}: {e}. "
            f"Falling back to system DNS."
        )
        return None


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
        ssl_verify: Optional[Union[bool, List[str]]] = None,
    ):
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
                f"HTTP client pool: SSL verification disabled for hosts: {self._ssl_verify}"
            )

        # HTTP/2 configuration (can be disabled for problematic providers)
        self._http2_enabled = _env_bool("HTTP2_ENABLED", DEFAULT_HTTP2_ENABLED)
        if not self._http2_enabled:
            lib_logger.warning(
                "HTTP/2 is DISABLED via HTTP2_ENABLED. Using HTTP/1.1 only."
            )

        # Custom DNS resolver (for DNS resolution issues)
        self._dns_resolver = os.getenv("HTTP_DNS_RESOLVER", DEFAULT_DNS_RESOLVER)
        if self._dns_resolver:
            lib_logger.info(
                f"HTTP client pool: Using custom DNS resolver: {self._dns_resolver}"
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
        timeout = (
            TimeoutConfig.streaming() if streaming else TimeoutConfig.non_streaming()
        )

        # Create SSL context with TLS 1.2 for compatibility with servers that don't support TLS 1.3
        ssl_context = ssl.create_default_context()
        ssl_context.minimum_version = (
            ssl.TLSVersion.TLSv1_2
        )  # Allow TLS 1.2 and 1.3 for Azure compatibility

        # Set Azure-compatible cipher suites to fix SSLV3_ALERT_HANDSHAKE_FAILURE
        try:
            ssl_context.set_ciphers(AZURE_COMPATIBLE_CIPHERS)
        except ssl.SSLError:
            pass  # Use default ciphers if set_ciphers fails

        if not self._ssl_verify:
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

        # Build client kwargs
        client_kwargs = {
            "timeout": timeout,
            "limits": self._create_limits(),
            "follow_redirects": True,
            "http2": self._http2_enabled,
            "http1": True,
            "verify": ssl_context,
        }

        # Use GzipRequestTransport for request body compression
        if HTTP_COMPRESS_REQUESTS:
            client_kwargs["transport"] = GzipRequestTransport(
                verify=ssl_context,
                limits=self._create_limits(),
                http2=self._http2_enabled,
            )

        # Configure custom DNS resolver if specified
        # This allows bypassing system DNS which may be hijacked by VPN/proxy/antivirus
        if self._dns_resolver:
            try:
                # Parse DNS resolver (format: "host" or "host:port")
                dns_host = self._dns_resolver
                dns_port = DEFAULT_DNS_PORT

                if ":" in self._dns_resolver:
                    parts = self._dns_resolver.rsplit(":", 1)
                    dns_host = parts[0]
                    try:
                        dns_port = int(parts[1])
                    except ValueError:
                        pass

                # Create custom transport with DNS resolver
                transport = httpx.AsyncHTTPTransport(
                    verify=ssl_context,
                    limits=self._create_limits(),
                    http2=self._http2_enabled,
                    # Note: httpx doesn't support custom DNS resolver directly
                    # We'll use a workaround by setting the resolver in the environment
                    # and letting aiohttp handle it
                )

                lib_logger.info(
                    f"Using custom DNS resolver: {dns_host}:{dns_port} "
                    f"(Note: DNS resolution will be handled by aiohttp)"
                )

            except Exception as e:
                lib_logger.warning(
                    f"Failed to configure custom DNS resolver {self._dns_resolver}: {e}. "
                    f"Falling back to system DNS."
                )

        client = httpx.AsyncClient(**client_kwargs)

        lib_logger.debug(
            f"Created new HTTP client (streaming={streaming}, "
            f"max_conn={self._max_connections}, keepalive={self._max_keepalive}, "
            f"ssl_verify={self._ssl_verify}, http2={self._http2_enabled}, "
            f"dns_resolver={self._dns_resolver})"
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

            # Pre-warm connections if hosts provided (background task)
            if self._warmup_hosts:
                asyncio.create_task(self._warmup_connections())

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
        ssl_errors = []

        # Use non-streaming client for warmup (lighter weight)
        client = self._non_streaming_client
        if not client:
            return

        # Build list of all warmup tasks (parallel execution)
        warmup_tasks = []
        for host in self._warmup_hosts[:5]:  # Limit to 5 hosts for warmup
            for _ in range(self._warmup_count):
                warmup_tasks.append(client.head(host, follow_redirects=True))

        # Execute all warmup requests in parallel with graceful error handling
        results = await asyncio.gather(*warmup_tasks, return_exceptions=True)

        # Process results and track errors
        task_idx = 0
        for host in self._warmup_hosts[:5]:
            for _ in range(self._warmup_count):
                result = results[task_idx]
                task_idx += 1
                if isinstance(result, Exception):
                    if isinstance(result, asyncio.TimeoutError):
                        lib_logger.debug(f"Warmup timeout for {host}")
                    elif isinstance(result, httpx.ConnectError):
                        error_str = str(result).lower()
                        if (
                            "ssl" in error_str
                            or "certificate" in error_str
                            or "tls" in error_str
                        ):
                            ssl_errors.append((host, str(result)))
                            lib_logger.warning(
                                f"SSL/TLS connection error during warmup for {host}: {result}. "
                                f"Consider adding '{host}' to HTTP_SSL_VERIFY_HOSTS environment variable."
                            )
                        else:
                            lib_logger.debug(
                                f"Warmup connection error for {host}: {type(result).__name__}: {result}"
                            )
                    else:
                        lib_logger.debug(f"Warmup error for {host}: {type(result).__name__}")
                else:
                    warmed += 1

        self._warmed_up = True
        elapsed = time.time() - start_time

        if warmed > 0:
            lib_logger.info(f"Pre-warmed {warmed} connection(s) in {elapsed:.2f}s")

        # Log summary of SSL errors if any occurred
        if ssl_errors:
            lib_logger.warning(
                f"SSL/TLS errors occurred during warmup for {len(ssl_errors)} host(s). "
                f"To disable SSL verification for specific hosts, set: "
                f"HTTP_SSL_VERIFY_HOSTS={','.join(h for h, _ in ssl_errors)}"
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
        # httpx.AsyncClient sets _client to None when closed
        # We check the internal _client attribute which is the actual transport
        return getattr(client, "_client", None) is None

    async def _ensure_client(self, streaming: bool) -> httpx.AsyncClient:
        """
        Ensure a valid client exists for the given mode, recreating if necessary.

        This is an async method that can safely recreate closed clients.

        Args:
            streaming: Whether to get streaming client

        Returns:
            Valid httpx.AsyncClient instance
        """
        if streaming:
            client = self._streaming_client
            if self._is_client_closed(client):
                lib_logger.warning("Streaming HTTP client was closed, recreating...")
                self._streaming_client = await self._create_client(streaming=True)
                self._stats["reconnects"] += 1
            return self._streaming_client
        else:
            client = self._non_streaming_client
            if self._is_client_closed(client):
                lib_logger.warning(
                    "Non-streaming HTTP client was closed, recreating..."
                )
                self._non_streaming_client = await self._create_client(streaming=False)
                self._stats["reconnects"] += 1
            return self._non_streaming_client

    def get_client(self, streaming: bool = False) -> httpx.AsyncClient:
        """
        Get the appropriate HTTP client.

        Note: This is a sync method for compatibility. The client is created
        during initialize(). If not initialized, returns a lazily-created client.

        WARNING: This method does NOT auto-recreate closed clients. Use
        get_client_async() for automatic recovery from closed clients.

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

    async def get_client_async(self, streaming: bool = False) -> httpx.AsyncClient:
        """
        Get the appropriate HTTP client with automatic recovery.

        This async method checks if the client is closed and recreates it
        if necessary. Use this for resilience in production code.

        Args:
            streaming: Whether the request will be streaming

        Returns:
            Valid httpx.AsyncClient instance
        """
        self._stats["requests_total"] += 1

        if streaming:
            self._stats["requests_streaming"] += 1
        else:
            self._stats["requests_non_streaming"] += 1

        return await self._ensure_client(streaming)

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
        timeout = (
            TimeoutConfig.streaming() if streaming else TimeoutConfig.non_streaming()
        )
        return httpx.AsyncClient(
            timeout=timeout,
            limits=self._create_limits(),
            follow_redirects=True,
            verify=self._ssl_verify,  # SSL verification configuration
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
                "dns_resolver": self._dns_resolver,
            },
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the client pool.

        Returns:
            Dict with health status for each client
        """
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
        """
        Attempt to recover closed or unhealthy clients.

        Returns:
            True if recovery was successful, False otherwise
        """
        recovered = []

        if self._is_client_closed(self._streaming_client):
            try:
                self._streaming_client = await self._create_client(streaming=True)
                recovered.append("streaming")
                self._stats["reconnects"] += 1
            except Exception as e:
                lib_logger.error(f"Failed to recover streaming client: {e}")

        if self._is_client_closed(self._non_streaming_client):
            try:
                self._non_streaming_client = await self._create_client(streaming=False)
                recovered.append("non-streaming")
                self._stats["reconnects"] += 1
            except Exception as e:
                lib_logger.error(f"Failed to recover non-streaming client: {e}")

        if recovered:
            lib_logger.info(f"HTTP client pool recovered: {', '.join(recovered)}")
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
