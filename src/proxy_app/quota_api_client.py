# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""HTTP/API client for quota stats endpoints."""

import atexit
import logging
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import httpx

from rotator_library.timeout_config import TimeoutConfig
from .quota_formatters import get_scheme_for_host, is_full_url, normalize_host_for_connection

logger = logging.getLogger(__name__)
_http_client: Optional[httpx.Client] = None


def get_http_client() -> httpx.Client:
    """Return the module-level shared httpx.Client, creating it lazily."""
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client()
    return _http_client


def _close_http_client() -> None:
    """Close the shared client on process exit."""
    global _http_client
    if _http_client is not None:
        _http_client.close()
        _http_client = None


atexit.register(_close_http_client)


class QuotaApiClient:
    """HTTP/API client for quota stats endpoints."""

    def __init__(self):
        self.current_remote: Optional[Dict[str, Any]] = None
        self.cached_stats: Optional[Dict[str, Any]] = None
        self.last_error: Optional[str] = None

    def get_headers(self) -> Dict[str, str]:
        """Get HTTP headers including auth if configured."""
        headers = {}
        if self.current_remote and self.current_remote.get("api_key"):
            headers["Authorization"] = f"Bearer {self.current_remote['api_key']}"
        return headers

    def get_base_url(self) -> str:
        """Get base URL for the current remote."""
        if not self.current_remote:
            return "http://127.0.0.1:8000"
        host = self.current_remote.get("host", "127.0.0.1")
        host = normalize_host_for_connection(host)

        if is_full_url(host):
            return host.rstrip("/")

        port = self.current_remote.get("port", 8000)
        scheme = get_scheme_for_host(host, port)
        return f"{scheme}://{host}:{port}"

    def build_endpoint_url(self, endpoint: str) -> str:
        """
        Build a full endpoint URL with smart path handling.

        Handles cases where base URL already contains a path (e.g., /v1):
        - Base: "https://api.example.com/v1", Endpoint: "/v1/quota-stats"
          -> "https://api.example.com/v1/quota-stats" (no duplication)
        - Base: "http://localhost:8000", Endpoint: "/v1/quota-stats"
          -> "http://localhost:8000/v1/quota-stats"
        """
        base_url = self.get_base_url()
        endpoint = endpoint.lstrip("/")

        parsed = urlparse(base_url)
        base_path = parsed.path.rstrip("/")

        if base_path:
            base_segments = base_path.split("/")
            endpoint_segments = endpoint.split("/")

            if base_segments and endpoint_segments:
                if base_segments[-1] == endpoint_segments[0]:
                    endpoint = "/".join(endpoint_segments[1:])

        return f"{base_url}/{endpoint}"

    def handle_httpx_error(self, exc: Exception) -> None:
        """Handle httpx exceptions by setting last_error and returning None."""
        if isinstance(exc, httpx.ConnectError):
            self.last_error = "Connection failed. Is the proxy running?"
        elif isinstance(exc, httpx.TimeoutException):
            self.last_error = "Request timed out."
        else:
            self.last_error = str(exc)
        return None

    def check_connection(
        self, remote: Dict[str, Any], timeout: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Check if a remote proxy is reachable."""
        if timeout is None:
            timeout = TimeoutConfig.quota_viewer_connect()
        host = remote.get("host", "127.0.0.1")
        host = normalize_host_for_connection(host)

        if is_full_url(host):
            parsed = urlparse(host)
            url = f"{parsed.scheme}://{parsed.netloc}/"
        else:
            port = remote.get("port", 8000)
            scheme = get_scheme_for_host(host, port)
            url = f"{scheme}://{host}:{port}/"

        headers = {}
        if remote.get("api_key"):
            headers["Authorization"] = f"Bearer {remote['api_key']}"

        try:
            client = get_http_client()
            response = client.get(url, headers=headers, timeout=timeout)
            if response.status_code == 200:
                return True, "Online"
            elif response.status_code == 401:
                return False, "Auth failed"
            else:
                return False, f"HTTP {response.status_code}"
        except httpx.ConnectError as e:
            logger.debug(f"Connection failed to {remote.get('name', 'unknown')}: {e}")
            return False, "Offline"
        except httpx.TimeoutException as e:
            logger.debug(f"Timeout connecting to {remote.get('name', 'unknown')}: {e}")
            return False, "Timeout"
        except Exception as e:
            logger.error(f"Unexpected error checking connection to {remote.get('name', 'unknown')}: {e}", exc_info=True)
            return False, str(e)[:20]

    def fetch_stats(self, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch quota stats from the current remote."""
        url = self.build_endpoint_url("/v1/quota-stats")
        if provider:
            url += f"?provider={provider}"

        try:
            client = get_http_client()
            response = client.get(url, headers=self.get_headers(), timeout=TimeoutConfig.quota_viewer_fetch())

            if response.status_code == 401:
                self.last_error = "Authentication failed. Check API key."
                return None
            elif response.status_code != 200:
                self.last_error = (
                    f"HTTP {response.status_code}: {response.text[:100]}"
                )
                return None

            try:
                self.cached_stats = response.json()
            except ValueError:
                self.last_error = f"Invalid JSON in response (HTTP {response.status_code})"
                return None
            self.last_error = None
            return self.cached_stats

        except httpx.ConnectError as e:
            return self.handle_httpx_error(e)
        except httpx.TimeoutException as e:
            return self.handle_httpx_error(e)
        except Exception as e:
            return self.handle_httpx_error(e)

    def merge_provider_stats(self, provider: str, result: Dict[str, Any]) -> None:
        """Merge provider-specific stats into the existing cache."""
        if not self.cached_stats:
            self.cached_stats = result
            return

        if "providers" in result and provider in result["providers"]:
            if "providers" not in self.cached_stats:
                self.cached_stats["providers"] = {}
            self.cached_stats["providers"][provider] = result["providers"][provider]

        if "timestamp" in result:
            self.cached_stats["timestamp"] = result["timestamp"]

        self.recalculate_summary()

    def recalculate_summary(self) -> None:
        """Recalculate summary fields from all provider data in cache."""
        stats = self.cached_stats
        if not stats:
            return
        providers = stats.get("providers", {})
        if not providers:
            return

        total_creds = 0
        active_creds = 0
        exhausted_creds = 0
        total_requests = 0
        total_input_cached = 0
        total_input_uncached = 0
        total_output = 0
        total_cost = 0.0

        for prov_stats in providers.values():
            total_creds += prov_stats.get("credential_count", 0)
            active_creds += prov_stats.get("active_count", 0)
            exhausted_creds += prov_stats.get("exhausted_count", 0)
            total_requests += prov_stats.get("total_requests", 0)

            tokens = prov_stats.get("tokens", {})
            total_input_cached += tokens.get("input_cached", 0)
            total_input_uncached += tokens.get("input_uncached", 0)
            total_output += tokens.get("output", 0)

            cost = prov_stats.get("approx_cost")
            if cost:
                total_cost += cost

        total_input = total_input_cached + total_input_uncached
        input_cache_pct = (
            round(total_input_cached / total_input * 100, 1) if total_input > 0 else 0
        )

        stats["summary"] = {
            "total_providers": len(providers),
            "total_credentials": total_creds,
            "active_credentials": active_creds,
            "exhausted_credentials": exhausted_creds,
            "total_requests": total_requests,
            "tokens": {
                "input_cached": total_input_cached,
                "input_uncached": total_input_uncached,
                "input_cache_pct": input_cache_pct,
                "output": total_output,
            },
            "approx_total_cost": total_cost if total_cost > 0 else None,
        }

        if "global_summary" in stats:
            global_total_requests = 0
            global_input_cached = 0
            global_input_uncached = 0
            global_output = 0
            global_cost = 0.0

            for prov_stats in providers.values():
                global_data = prov_stats.get("global", prov_stats)
                global_total_requests += global_data.get("total_requests", 0)

                tokens = global_data.get("tokens", {})
                global_input_cached += tokens.get("input_cached", 0)
                global_input_uncached += tokens.get("input_uncached", 0)
                global_output += tokens.get("output", 0)

                cost = global_data.get("approx_cost")
                if cost:
                    global_cost += cost

            global_total_input = global_input_cached + global_input_uncached
            global_cache_pct = (
                round(global_input_cached / global_total_input * 100, 1)
                if global_total_input > 0
                else 0
            )

            stats["global_summary"] = {
                "total_providers": len(providers),
                "total_credentials": total_creds,
                "total_requests": global_total_requests,
                "tokens": {
                    "input_cached": global_input_cached,
                    "input_uncached": global_input_uncached,
                    "input_cache_pct": global_cache_pct,
                    "output": global_output,
                },
                "approx_total_cost": global_cost if global_cost > 0 else None,
            }

    def post_action(
        self,
        action: str,
        scope: str = "all",
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Post a refresh action to the proxy."""
        url = self.build_endpoint_url("/v1/quota-stats")
        payload = {
            "action": action,
            "scope": scope,
        }
        if provider:
            payload["provider"] = provider
        if credential:
            payload["credential"] = credential

        try:
            client = get_http_client()
            response = client.post(url, headers=self.get_headers(), json=payload, timeout=TimeoutConfig.quota_viewer_action())

            if response.status_code == 401:
                self.last_error = "Authentication failed. Check API key."
                return None
            elif response.status_code != 200:
                self.last_error = (
                    f"HTTP {response.status_code}: {response.text[:100]}"
                )
                return None

            try:
                result = response.json()
            except ValueError:
                self.last_error = f"Invalid JSON in response (HTTP {response.status_code})"
                return None

            if scope == "provider" and provider and self.cached_stats:
                self.merge_provider_stats(provider, result)
            else:
                self.cached_stats = result

            self.last_error = None
            return result

        except httpx.ConnectError as e:
            return self.handle_httpx_error(e)
        except httpx.TimeoutException as e:
            return self.handle_httpx_error(e)
        except Exception as e:
            return self.handle_httpx_error(e)
