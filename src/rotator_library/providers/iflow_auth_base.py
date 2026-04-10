# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

# src/rotator_library/providers/iflow_auth_base.py

import secrets
import base64
import time
import asyncio
import logging
import webbrowser
import socket
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar, Dict, Any, Tuple, Union, Optional, List
from urllib.parse import urlencode, parse_qs, urlparse

import httpx

from ..http_client_pool import get_http_pool
from aiohttp import web
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape as rich_escape
from ..utils.headless_detection import is_headless_environment
from ..utils.json_utils import json_loads
from ..error_handler import CredentialNeedsReauthError

lib_logger = logging.getLogger("rotator_library")

IFLOW_OAUTH_AUTHORIZE_ENDPOINT = "https://iflow.cn/oauth"
IFLOW_OAUTH_TOKEN_ENDPOINT = "https://iflow.cn/oauth/token"
IFLOW_USER_INFO_ENDPOINT = "https://iflow.cn/api/oauth/getUserInfo"
IFLOW_SUCCESS_REDIRECT_URL = "https://iflow.cn/oauth/success"
IFLOW_ERROR_REDIRECT_URL = "https://iflow.cn/oauth/error"

# Client credentials provided by iFlow
IFLOW_CLIENT_ID = "10009311001"
IFLOW_CLIENT_SECRET = "4Z3YjXycVsQvyGF1etiNlIBB4RsqSDtW"

# Local callback server port
CALLBACK_PORT = 11451




def get_callback_port() -> int:
    """
    Get the OAuth callback port, checking environment variable first.

    Reads from IFLOW_OAUTH_PORT environment variable, falling back
    to the default CALLBACK_PORT if not set.
    """
    env_value = os.getenv("IFLOW_OAUTH_PORT")
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            logging.getLogger("rotator_library").warning(
                f"Invalid IFLOW_OAUTH_PORT value: {env_value}, using default {CALLBACK_PORT}"
            )
    return CALLBACK_PORT


console = Console()


class OAuthCallbackServer:
    """
    Minimal HTTP server for handling iFlow OAuth callbacks.
    """

    def __init__(self, port: int = CALLBACK_PORT):
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.result_future: Optional[asyncio.Future] = None
        self.expected_state: Optional[str] = None

    def _is_port_available(self) -> bool:
        """Checks if the callback port is available."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", self.port))
            sock.close()
            return True
        except OSError:
            return False

    async def start(self, expected_state: str):
        """Starts the OAuth callback server."""
        if not self._is_port_available():
            raise RuntimeError(f"Port {self.port} is already in use")

        self.expected_state = expected_state
        self.result_future = asyncio.Future()

        # Setup route
        self.app.router.add_get("/oauth2callback", self._handle_callback)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "localhost", self.port)
        await self.site.start()

        lib_logger.debug(f"iFlow OAuth callback server started on port {self.port}")

    async def stop(self):
        """Stops the OAuth callback server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        lib_logger.debug("iFlow OAuth callback server stopped")

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handles the OAuth callback request."""
        query = request.query

        # Check for error parameter
        if "error" in query:
            error = query.get("error", "unknown_error")
            lib_logger.error(f"iFlow OAuth callback received error: {error}")
            if not self.result_future.done():
                self.result_future.set_exception(ValueError(f"OAuth error: {error}"))
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Check for authorization code
        code = query.get("code")
        if not code:
            lib_logger.error("iFlow OAuth callback missing authorization code")
            if not self.result_future.done():
                self.result_future.set_exception(
                    ValueError("Missing authorization code")
                )
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Validate state parameter
        state = query.get("state", "")
        if state != self.expected_state:
            lib_logger.error(
                f"iFlow OAuth state mismatch. Expected: {self.expected_state}, Got: {state}"
            )
            if not self.result_future.done():
                self.result_future.set_exception(ValueError("State parameter mismatch"))
            return web.Response(
                status=302, headers={"Location": IFLOW_ERROR_REDIRECT_URL}
            )

        # Success - set result and redirect to success page
        if not self.result_future.done():
            self.result_future.set_result(code)

        return web.Response(
            status=302, headers={"Location": IFLOW_SUCCESS_REDIRECT_URL}
        )

    async def wait_for_callback(self, timeout: float = 300.0) -> str:
        """Waits for the OAuth callback and returns the authorization code."""
        try:
            code = await asyncio.wait_for(self.result_future, timeout=timeout)
            return code
        except asyncio.TimeoutError:
            raise TimeoutError("Timeout waiting for OAuth callback")


from .google_oauth_base import GoogleOAuthBase


class IFlowAuthBase(GoogleOAuthBase):
    """
    iFlow OAuth authentication base class.
    Implements authorization code flow with local callback server.
    """

    # Class attributes required by GoogleOAuthBase
    CLIENT_ID = IFLOW_CLIENT_ID
    CLIENT_SECRET = IFLOW_CLIENT_SECRET
    OAUTH_SCOPES = ["read", "write"]
    ENV_PREFIX = "IFLOW"
    REFRESH_EXPIRY_BUFFER_SECONDS = 24 * 60 * 60
    _cache_default_ttl: float = 90000.0  # 25hr: aligns with 24hr buffer + 1hr token lifetime
    BUFFER_ON_FAILURE: ClassVar[bool] = False

    def __init__(self):
        super().__init__()
        self._api_key_cache: dict[str, tuple[float, str]] = {}

    def _load_from_env(
        self, credential_index: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load OAuth credentials from environment variables for stateless deployments.

        Supports two formats:
        1. Legacy (credential_index="0" or None): IFLOW_ACCESS_TOKEN
        2. Numbered (credential_index="1", "2", etc.): IFLOW_1_ACCESS_TOKEN, etc.

        Expected environment variables (for numbered format with index N):
        - IFLOW_{N}_ACCESS_TOKEN (required)
        - IFLOW_{N}_REFRESH_TOKEN (required)
        - IFLOW_{N}_API_KEY (required - critical for iFlow!)
        - IFLOW_{N}_EXPIRY_DATE (optional, defaults to empty string)
        - IFLOW_{N}_EMAIL (optional, defaults to "env-user-{N}")
        - IFLOW_{N}_TOKEN_TYPE (optional, defaults to "Bearer")
        - IFLOW_{N}_SCOPE (optional, defaults to "read write")

        Returns:
            Dict with credential structure if env vars present, None otherwise
        """
        # Determine the env var prefix based on credential index
        if credential_index and credential_index != "0":
            prefix = f"IFLOW_{credential_index}"
            default_email = f"env-user-{credential_index}"
        else:
            prefix = "IFLOW"
            default_email = "env-user"

        access_token = os.getenv(f"{prefix}_ACCESS_TOKEN")
        refresh_token = os.getenv(f"{prefix}_REFRESH_TOKEN")
        api_key = os.getenv(f"{prefix}_API_KEY")

        # All three are required for iFlow
        if not (access_token and refresh_token and api_key):
            return None

        lib_logger.debug(
            f"Loading iFlow credentials from environment variables (prefix: {prefix})"
        )

        # Parse expiry_date as string (ISO 8601 format)
        expiry_str = os.getenv(f"{prefix}_EXPIRY_DATE", "")

        creds = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "api_key": api_key,  # Critical for iFlow!
            "expiry_date": expiry_str,
            "email": os.getenv(f"{prefix}_EMAIL", default_email),
            "token_type": os.getenv(f"{prefix}_TOKEN_TYPE", "Bearer"),
            "scope": os.getenv(f"{prefix}_SCOPE", "read write"),
            "_proxy_metadata": {
                "email": os.getenv(f"{prefix}_EMAIL", default_email),
                "last_check_timestamp": time.time(),
                "loaded_from_env": True,
                "env_credential_index": credential_index or "0",
            },
        }

        return creds

    async def _read_creds_from_file(self, path: str) -> Dict[str, Any]:
        """Reads credentials from file and populates the cache. No locking."""
        try:
            lib_logger.debug(f"Reading iFlow credentials from file: {path}")
            with open(path, "r") as f:
                creds = json_loads(f.read())
            self._credentials_cache[path] = creds
            return creds
        except FileNotFoundError:
            raise IOError(f"iFlow OAuth credential file not found at '{path}'")
        except Exception as e:
            raise IOError(f"Failed to load iFlow OAuth credentials from '{path}': {e}")

    async def _load_credentials(self, path: str) -> Dict[str, Any]:
        """Loads credentials from cache, environment variables, or file."""
        if path in self._credentials_cache:
            return self._credentials_cache[path]

        async with self._get_lock(path):
            # Re-check cache after acquiring lock
            if path in self._credentials_cache:
                return self._credentials_cache[path]

            # Check if this is a virtual env:// path
            credential_index = self._parse_env_credential_path(path)
            if credential_index is not None:
                env_creds = self._load_from_env(credential_index)
                if env_creds:
                    lib_logger.info(
                        f"Using iFlow credentials from environment variables (index: {credential_index})"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                else:
                    raise IOError(
                        f"Environment variables for iFlow credential index {credential_index} not found"
                    )

            # Try file-based loading first (preferred for explicit file paths)
            try:
                return await self._read_creds_from_file(path)
            except IOError:
                # File not found - fall back to legacy env vars for backwards compatibility
                env_creds = self._load_from_env()
                if env_creds:
                    lib_logger.info(
                        f"File '{path}' not found, using iFlow credentials from environment variables"
                    )
                    self._credentials_cache[path] = env_creds
                    return env_creds
                raise  # Re-raise the original file not found error

    def _is_token_expired(self, creds: Dict[str, Any]) -> bool:
        """Checks if the token is expired (with buffer for proactive refresh).

        Override required: iFlow uses ISO 8601 string format for expiry_date
        (e.g. "2025-01-17T12:00:00Z") while GoogleOAuthBase expects gcloud
        token_expiry strings or numeric milliseconds. The parent's logic would
        attempt string division (/ 1000) and crash on this format.
        """
        # Try to parse expiry_date as ISO 8601 string
        expiry_str = creds.get("expiry_date")
        if not expiry_str:
            return True

        expiry_timestamp = creds.get("_parsed_expiry")
        if expiry_timestamp is None:
            try:
                expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                expiry_timestamp = expiry_dt.timestamp()
                creds["_parsed_expiry"] = expiry_timestamp
            except (ValueError, AttributeError):
                # Fallback: treat as numeric timestamp
                try:
                    expiry_timestamp = float(expiry_str)
                except (ValueError, TypeError):
                    lib_logger.warning(f"Could not parse expiry_date: {expiry_str}")
                    return True

        return expiry_timestamp < time.time() + self.REFRESH_EXPIRY_BUFFER_SECONDS

    def _is_token_truly_expired(self, creds: Dict[str, Any]) -> bool:
        """Check if token is TRULY expired (past actual expiry, not just threshold).

        Override required for the same reason as _is_token_expired: iFlow uses
        ISO 8601 string format for expiry_date, incompatible with the parent's
        gcloud/milliseconds parsing logic.
        """
        expiry_str = creds.get("expiry_date")
        if not expiry_str:
            return True

        expiry_timestamp = creds.get("_parsed_expiry")
        if expiry_timestamp is None:
            try:
                expiry_dt = datetime.fromisoformat(expiry_str.replace("Z", "+00:00"))
                expiry_timestamp = expiry_dt.timestamp()
                creds["_parsed_expiry"] = expiry_timestamp
            except (ValueError, AttributeError):
                try:
                    expiry_timestamp = float(expiry_str)
                except (ValueError, TypeError):
                    return True

        return expiry_timestamp < time.time()

    async def _fetch_user_info(self, access_token: str) -> Dict[str, Any]:
        """
        Fetches user info (including API key) from iFlow API.
        This is critical: iFlow uses a separate API key for actual API calls.
        """
        if not access_token or not access_token.strip():
            raise ValueError("Access token is empty")

        url = f"{IFLOW_USER_INFO_ENDPOINT}?accessToken={access_token}"
        headers = {"Accept": "application/json"}

        pool = await get_http_pool()
        client = await pool.get_client_async()
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            raise ValueError("iFlow user info request not successful")

        data = result.get("data", {})
        api_key = data.get("apiKey", "").strip()
        if not api_key:
            raise ValueError("Missing API key in user info response")

        email = data.get("email", "").strip()
        if not email:
            email = data.get("phone", "").strip()
        if not email:
            raise ValueError("Missing email/phone in user info response")

        return {"api_key": api_key, "email": email}

    async def _exchange_code_for_tokens(
        self, code: str, redirect_uri: str
    ) -> Dict[str, Any]:
        """
        Exchanges authorization code for access and refresh tokens.
        Uses Basic Auth with client credentials.
        """
        # Create Basic Auth header
        auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
        basic_auth = base64.b64encode(auth_string.encode()).decode()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "Authorization": f"Basic {basic_auth}",
        }

        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": IFLOW_CLIENT_ID,
            "client_secret": IFLOW_CLIENT_SECRET,
        }

        pool = await get_http_pool()
        client = await pool.get_client_async()
        response = await client.post(
            IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data
        )

        if response.status_code != 200:
            error_text = response.text
            lib_logger.error(
                f"iFlow token exchange failed: {response.status_code} {error_text}"
            )
            raise ValueError(
                f"Token exchange failed: {response.status_code} {error_text}"
            )

        token_data = response.json()

        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("Missing access_token in token response")

        refresh_token = token_data.get("refresh_token", "")
        expires_in = token_data.get("expires_in", 3600)
        token_type = token_data.get("token_type", "Bearer")
        scope = token_data.get("scope", "")

        # Fetch user info to get API key
        user_info = await self._fetch_user_info(access_token)

        # Calculate expiry date
        expiry_date = (
            datetime.utcnow() + timedelta(seconds=expires_in)
        ).isoformat() + "Z"

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "api_key": user_info["api_key"],
            "email": user_info["email"],
            "expiry_date": expiry_date,
            "token_type": token_type,
            "scope": scope,
        }

    async def _refresh_token(self, path: str, creds: Optional[Dict[str, Any]] = None, force: bool = False) -> Dict[str, Any]:
        """
        Refreshes the OAuth tokens and re-fetches the API key.
        CRITICAL: Must re-fetch user info to get potentially updated API key.
        """
        async with self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # [ROTATING TOKEN FIX] Always read fresh from disk before refresh.
            # iFlow may use rotating refresh tokens - each refresh could invalidate the previous token.
            # If we use a stale cached token, refresh will fail.
            # Reading fresh from disk ensures we have the latest token.
            await self._read_creds_from_file(path)
            creds_from_file = self._credentials_cache[path]

            lib_logger.debug(f"Refreshing iFlow OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                raise ValueError("No refresh_token found in iFlow credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None

            # Create Basic Auth header
            auth_string = f"{IFLOW_CLIENT_ID}:{IFLOW_CLIENT_SECRET}"
            basic_auth = base64.b64encode(auth_string.encode()).decode()

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "Authorization": f"Basic {basic_auth}",
            }

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": IFLOW_CLIENT_ID,
                "client_secret": IFLOW_CLIENT_SECRET,
            }

            pool = await get_http_pool()
            client = await pool.get_client_async()
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        IFLOW_OAUTH_TOKEN_ENDPOINT, headers=headers, data=data
                    )
                    response.raise_for_status()
                    new_token_data = response.json()

                    # [FIX] Handle wrapped response format: {success: bool, data: {...}}
                    # iFlow API may return tokens nested inside a 'data' key
                    if (
                        isinstance(new_token_data, dict)
                        and "data" in new_token_data
                    ):
                        lib_logger.debug(
                            f"iFlow refresh response wrapped in 'data' key, extracting..."
                        )
                        # Check for error in wrapped response
                        if not new_token_data.get("success", True):
                            error_msg = new_token_data.get(
                                "message", "Unknown error"
                            )
                            raise ValueError(
                                f"iFlow token refresh failed: {error_msg}"
                            )
                        new_token_data = new_token_data.get("data", {})

                    break  # Success

                except httpx.HTTPStatusError as e:
                    last_error = e
                    status_code = e.response.status_code
                    error_body = e.response.text

                    lib_logger.error(
                        f"[REFRESH HTTP ERROR] HTTP {status_code} for '{Path(path).name}': {error_body}"
                    )

                    # [STATUS CODE HANDLING]
                    # [INVALID GRANT HANDLING] Handle 400/401/403 by raising
                    # Queue for re-auth in background so credential gets fixed automatically
                    if status_code == 400:
                        # Check if this is an invalid refresh token error
                        try:
                            error_data = e.response.json()
                            error_type = error_data.get("error", "")
                            error_desc = error_data.get("error_description", "")
                            if not error_desc:
                                error_desc = error_data.get("message", error_body)
                        except Exception:
                            error_type = ""
                            error_desc = error_body

                        if (
                            "invalid" in error_desc.lower()
                            or error_type == "invalid_request"
                        ):
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP 400: {error_desc}). "
                                f"Queued for re-authentication, rotating to next credential."
                            )
                            # Queue for re-auth in background (non-blocking, fire-and-forget)
                            # This ensures credential gets fixed even if caller doesn't handle it
                            asyncio.create_task(
                                self._queue_refresh(
                                    path, force=True, needs_reauth=True
                                )
                            )
                            # Raise rotatable error instead of raw HTTPStatusError
                            raise CredentialNeedsReauthError(
                                credential_path=path,
                                message=f"Refresh token invalid for '{Path(path).name}'. Re-auth queued.",
                            )
                        else:
                            # Other 400 error - raise it
                            raise

                    elif status_code in (401, 403):
                        lib_logger.info(
                            f"Credential '{Path(path).name}' needs re-auth (HTTP {status_code}). "
                            f"Queued for re-authentication, rotating to next credential."
                        )
                        # Queue for re-auth in background (non-blocking, fire-and-forget)
                        asyncio.create_task(
                            self._queue_refresh(path, force=True, needs_reauth=True)
                        )
                        # Raise rotatable error instead of raw HTTPStatusError
                        raise CredentialNeedsReauthError(
                            credential_path=path,
                            message=f"Token invalid for '{Path(path).name}' (HTTP {status_code}). Re-auth queued.",
                        )

                    elif status_code == 429:
                        retry_after = int(e.response.headers.get("Retry-After", 60))
                        lib_logger.warning(
                            f"Rate limited (HTTP 429), retry after {retry_after}s"
                        )
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_after)
                            continue
                        raise

                    elif 500 <= status_code < 600:
                        if attempt < max_retries - 1:
                            wait_time = 2**attempt
                            lib_logger.warning(
                                f"Server error (HTTP {status_code}), retry {attempt + 1}/{max_retries} in {wait_time}s"
                            )
                            await asyncio.sleep(wait_time)
                            continue
                        raise

                    else:
                        raise

                except (httpx.RequestError, httpx.TimeoutException) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = 2**attempt
                        lib_logger.warning(
                            f"Network error during refresh: {e}, retry {attempt + 1}/{max_retries} in {wait_time}s"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    raise

            if new_token_data is None:
                # [BACKOFF TRACKING] Increment failure count and set backoff timer
                self._refresh_failures[path] = self._refresh_failures.get(path, 0) + 1
                backoff_seconds = min(
                    300, 30 * (2 ** self._refresh_failures[path])
                )  # Max 5 min backoff
                self._next_refresh_after[path] = time.time() + backoff_seconds
                lib_logger.debug(
                    f"Setting backoff for '{Path(path).name}': {backoff_seconds}s"
                )
                raise last_error or Exception("Token refresh failed after all retries")

            # Update tokens
            access_token = new_token_data.get("access_token")
            if not access_token:
                # Log response keys for debugging
                response_keys = (
                    list(new_token_data.keys())
                    if isinstance(new_token_data, dict)
                    else type(new_token_data).__name__
                )
                lib_logger.error(
                    f"Missing access_token in refresh response for '{Path(path).name}'. "
                    f"Response keys: {response_keys}"
                )
                raise ValueError("Missing access_token in refresh response")

            creds_from_file["access_token"] = access_token
            creds_from_file["refresh_token"] = new_token_data.get(
                "refresh_token", creds_from_file["refresh_token"]
            )

            expires_in = new_token_data.get("expires_in", 3600)

            creds_from_file["expiry_date"] = (
                datetime.utcnow() + timedelta(seconds=expires_in)
            ).isoformat() + "Z"

            creds_from_file["token_type"] = new_token_data.get(
                "token_type", creds_from_file.get("token_type", "Bearer")
            )
            creds_from_file["scope"] = new_token_data.get(
                "scope", creds_from_file.get("scope", "")
            )

            # CRITICAL: Re-fetch user info to get potentially updated API key
            try:
                cached_api_key = self._api_key_cache.get(path)
                if cached_api_key and (time.time() - cached_api_key[0]) < 86400:
                    user_info = {"api_key": cached_api_key[1]}
                else:
                    user_info = await self._fetch_user_info(access_token)
                    if user_info.get("api_key"):
                        self._api_key_cache[path] = (time.time(), user_info["api_key"])
                if user_info.get("api_key"):
                    creds_from_file["api_key"] = user_info["api_key"]
                if user_info.get("email"):
                    creds_from_file["email"] = user_info["email"]
            except Exception as e:
                lib_logger.warning(
                    f"Failed to update API key during token refresh: {e}"
                )

            # Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds_from_file:
                creds_from_file["_proxy_metadata"] = {}
            creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # [VALIDATION] Verify required fields exist after refresh
            required_fields = ["access_token", "refresh_token", "api_key"]
            missing_fields = [
                field for field in required_fields if not creds_from_file.get(field)
            ]
            if missing_fields:
                raise ValueError(
                    f"Refreshed credentials missing required fields: {missing_fields}"
                )

            # [BACKOFF TRACKING] Clear failure count on successful refresh
            self._refresh_failures.pop(path, None)
            self._next_refresh_after.pop(path, None)

            # Save credentials - MUST succeed for rotating token providers
            if not await self._save_credentials(path, creds_from_file):
                # CRITICAL: If we can't persist the new token, the old token may be
                # invalidated. This is a critical failure - raise so retry logic kicks in.
                raise IOError(
                    f"Failed to persist refreshed credentials for '{Path(path).name}'. "
                    f"Disk write failed - refresh will be retried."
                )

            lib_logger.debug(
                f"Successfully refreshed iFlow OAuth token for '{Path(path).name}'."
            )
            return self._credentials_cache[path]  # Return from cache (synced with disk)

    async def get_api_details(self, credential_identifier: str) -> Tuple[str, str]:
        """
        Returns the API base URL and API key (NOT access_token).
        CRITICAL: iFlow uses the api_key for API requests, not the OAuth access_token.

        Supports both credential types:
        - OAuth: credential_identifier is a file path to JSON credentials
        - API Key: credential_identifier is the API key string itself
        """
        # Detect credential type
        if os.path.isfile(credential_identifier):
            # OAuth credential: file path to JSON
            lib_logger.debug(
                f"Using OAuth credentials from file: {credential_identifier}"
            )
            creds = await self._load_credentials(credential_identifier)

            # Check if token needs refresh
            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)

            api_key = creds.get("api_key")
            if not api_key:
                raise ValueError("Missing api_key in iFlow OAuth credentials")
        else:
            # Direct API key: use as-is
            lib_logger.debug("Using direct API key for iFlow")
            api_key = credential_identifier

        base_url = "https://apis.iflow.cn/v1"
        return base_url, api_key

    # proactively_refresh inherited from GoogleOAuthBase (with IOError handling)

    def _is_invalid_grant_error(self, error_body: str, status_code: int, error_type: str = "") -> bool:
        """iFlow uses 'invalid' in error description instead of 'invalid_grant'."""
        if status_code == 400:
            return "invalid" in error_body.lower() or error_type == "invalid_request"
        return False

    # _process_refresh_queue inherited from GoogleOAuthBase

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth authorization code flow (browser-based authentication).

        This method is called via the global ReauthCoordinator to ensure
        only one interactive OAuth flow runs at a time across all providers.

        Args:
            path: Credential file path
            creds: Current credentials dict (will be updated)
            display_name: Display name for logging/UI

        Returns:
            Updated credentials dict with new tokens
        """
        # [HEADLESS DETECTION] Check if running in headless environment
        is_headless = is_headless_environment()

        # Generate random state for CSRF protection
        state = secrets.token_urlsafe(32)

        # Build authorization URL
        callback_port = get_callback_port()
        redirect_uri = f"http://localhost:{callback_port}/oauth2callback"
        auth_params = {
            "loginMethod": "phone",
            "type": "phone",
            "redirect": redirect_uri,
            "state": state,
            "client_id": IFLOW_CLIENT_ID,
        }
        auth_url = f"{IFLOW_OAUTH_AUTHORIZE_ENDPOINT}?{urlencode(auth_params)}"

        # Start OAuth callback server
        callback_server = OAuthCallbackServer(port=callback_port)
        try:
            await callback_server.start(expected_state=state)

            # [HEADLESS SUPPORT] Display appropriate instructions
            if is_headless:
                auth_panel_text = Text.from_markup(
                    "Running in headless environment (no GUI detected).\n"
                    "Please open the URL below in a browser on another machine to authorize:\n"
                    "1. Visit the URL below to sign in with your phone number.\n"
                    "2. [bold]Authorize the application[/bold] to access your account.\n"
                    "3. You will be automatically redirected after authorization."
                )
            else:
                auth_panel_text = Text.from_markup(
                    "1. Visit the URL below to sign in with your phone number.\n"
                    "2. [bold]Authorize the application[/bold] to access your account.\n"
                    "3. You will be automatically redirected after authorization."
                )

            console.print(
                Panel(
                    auth_panel_text,
                    title=f"iFlow OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                    style="bold blue",
                )
            )
            escaped_url = rich_escape(auth_url)
            console.print(f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n")

            # [HEADLESS SUPPORT] Only attempt browser open if NOT headless
            if not is_headless:
                try:
                    webbrowser.open(auth_url)
                    lib_logger.info("Browser opened successfully for iFlow OAuth flow")
                except Exception as e:
                    lib_logger.warning(
                        f"Failed to open browser automatically: {e}. Please open the URL manually."
                    )

            # Wait for callback
            with console.status(
                "[bold green]Waiting for authorization in the browser...[/bold green]",
                spinner="dots",
            ):
                # Note: The 300s timeout here is handled by the ReauthCoordinator
                # We use a slightly longer internal timeout to let the coordinator handle it
                code = await callback_server.wait_for_callback(timeout=310.0)

            lib_logger.info("Received authorization code, exchanging for tokens...")

            # Exchange code for tokens and API key
            token_data = await self._exchange_code_for_tokens(code, redirect_uri)

            # Update credentials
            creds.update(
                {
                    "access_token": token_data["access_token"],
                    "refresh_token": token_data["refresh_token"],
                    "api_key": token_data["api_key"],
                    "email": token_data["email"],
                    "expiry_date": token_data["expiry_date"],
                    "token_type": token_data["token_type"],
                    "scope": token_data["scope"],
                }
            )

            # Create metadata object
            if not creds.get("_proxy_metadata"):
                creds["_proxy_metadata"] = {
                    "email": token_data["email"],
                    "last_check_timestamp": time.time(),
                }

            if path:
                if not await self._save_credentials(path, creds):
                    raise IOError(
                        f"Failed to save OAuth credentials to disk for '{display_name}'. "
                        f"Please retry authentication."
                    )

            lib_logger.info(
                f"iFlow OAuth initialized successfully for '{display_name}'."
            )
            return creds

        finally:
            await callback_server.stop()

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        """
        Initialize OAuth token, triggering interactive authorization flow if needed.

        If interactive OAuth is required (expired refresh token, missing credentials, etc.),
        the flow is coordinated globally via ReauthCoordinator to ensure only one
        interactive OAuth flow runs at a time across all providers.

        Args:
            creds_or_path: Either a credentials dict or path to credentials file.
            force_interactive: If True, skip expiry checks and force interactive OAuth.
                               Use this when the refresh token is known to be invalid
                               (e.g., after HTTP 400 from token endpoint).
        """
        path = creds_or_path if isinstance(creds_or_path, str) else None

        display_name = self._get_display_name(creds_or_path)

        lib_logger.debug(f"Initializing iFlow token for '{display_name}'...")

        try:
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            token_expired = self._is_token_expired(creds)
            needs_interactive, reason = self._should_force_interactive(
                creds, force_interactive=force_interactive
            )

            # Expired tokens: try refresh before interactive (unless already
            # forcing interactive, which means refresh already failed)
            if token_expired and not needs_interactive:
                needs_interactive = True
                reason = "token is expired"

            if needs_interactive:
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        return await self._refresh_token(path)
                    except Exception as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                lib_logger.warning(
                    f"iFlow OAuth token for '{display_name}' needs setup: {reason}."
                )

                return await self._execute_interactive_oauth(
                    path=path,
                    creds=creds,
                    display_name=display_name,
                    provider_name="IFLOW",
                    timeout=300.0,
                )

            lib_logger.info(f"iFlow OAuth token at '{display_name}' is valid.")
            return creds

        except Exception as e:
            raise ValueError(f"Failed to initialize iFlow OAuth for '{path}': {e}")

    async def get_auth_header(self, credential_path: str) -> Dict[str, str]:
        """
        Returns auth header with API key (NOT OAuth access_token).
        CRITICAL: iFlow API requests use the api_key, not the OAuth tokens.
        Includes graceful degradation if refresh fails.
        """
        try:
            creds = await self._load_credentials(credential_path)
            if self._is_token_expired(creds):
                try:
                    creds = await self._refresh_token(credential_path, creds)
                    self._staleness_counter.pop(credential_path, None)
                except Exception as e:
                    cached = self._credentials_cache.get(credential_path)
                    if cached and cached.get("api_key"):
                        stale_count = self._staleness_counter.get(credential_path, 0)
                        if stale_count < 1:
                            self._staleness_counter[credential_path] = stale_count + 1
                            lib_logger.warning(
                                f"Token refresh failed for {Path(credential_path).name}: {e}. "
                                "Using cached api_key (token may be expired)."
                            )
                            creds = cached
                        else:
                            lib_logger.error(
                                f"Token refresh failed for {Path(credential_path).name}: {e}. "
                                "Stale token served too many times, raising."
                            )
                            raise
                    else:
                        raise

            api_key = creds.get("api_key")
            if not api_key:
                raise ValueError("Missing api_key in iFlow credentials")

            return {"Authorization": f"Bearer {api_key}"}
        except Exception as e:
            cached = self._credentials_cache.get(credential_path)
            if cached and cached.get("api_key"):
                stale_count = self._staleness_counter.get(credential_path, 0)
                if stale_count < 1:
                    self._staleness_counter[credential_path] = stale_count + 1
                    lib_logger.error(
                        f"Credential load failed for {credential_path}: {e}. "
                        "Using stale cached api_key as last resort."
                    )
                    return {"Authorization": f"Bearer {cached['api_key']}"}
                else:
                    lib_logger.error(
                        f"Credential load failed for {credential_path}: {e}. "
                        "Stale token served too many times, raising."
                    )
            raise

    async def get_user_info(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """Retrieves user info from the _proxy_metadata in the credential file."""
        try:
            path = creds_or_path if isinstance(creds_or_path, str) else None
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            # Ensure the token is valid
            if path:
                await self.initialize_token(path)
                creds = await self._load_credentials(path)

            email = creds.get("email") or creds.get("_proxy_metadata", {}).get("email")

            if not email:
                lib_logger.warning(
                    f"No email found in iFlow credentials for '{path or 'in-memory object'}'."
                )

            # Update timestamp in cache only (not disk) to avoid overwriting
            # potentially newer tokens that were saved by another process/refresh.
            # The timestamp is non-critical metadata - losing it on restart is fine.
            if path and "_proxy_metadata" in creds:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()
                # Note: We intentionally don't save to disk here because:
                # 1. The cache may have older tokens than disk (if external refresh occurred)
                # 2. Saving would overwrite the newer disk tokens with stale cached ones
                # 3. The timestamp is non-critical and will be updated on next refresh

            return {"email": email}
        except Exception as e:
            lib_logger.error(f"Failed to get iFlow user info from credentials: {e}")
            return {"email": None}

    # =========================================================================
    # CREDENTIAL MANAGEMENT METHODS
    # =========================================================================

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """Generate .env file lines for an iFlow credential."""
        email = creds.get("email") or creds.get("_proxy_metadata", {}).get(
            "email", "unknown"
        )
        prefix = f"IFLOW_{cred_number}"

        lines = [
            f"# IFLOW Credential #{cred_number} for: {email}",
            f"# Exported from: iflow_oauth_{cred_number}.json",
            f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "#",
            "# To combine multiple credentials into one .env file, copy these lines",
            "# and ensure each credential has a unique number (1, 2, 3, etc.)",
            "",
            f"{prefix}_ACCESS_TOKEN={creds.get('access_token', '')}",
            f"{prefix}_REFRESH_TOKEN={creds.get('refresh_token', '')}",
            f"{prefix}_API_KEY={creds.get('api_key', '')}",
            f"{prefix}_EXPIRY_DATE={creds.get('expiry_date', '')}",
            f"{prefix}_EMAIL={email}",
            f"{prefix}_TOKEN_TYPE={creds.get('token_type', 'Bearer')}",
            f"{prefix}_SCOPE={creds.get('scope', 'read write')}",
        ]

        return lines
