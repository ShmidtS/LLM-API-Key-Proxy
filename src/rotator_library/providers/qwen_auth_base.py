# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/qwen_auth_base.py

import secrets
import hashlib
import base64
import time
import asyncio
import logging
import webbrowser
import os
from pathlib import Path
from typing import ClassVar, Dict, Any, Tuple, Union, Optional, List

import httpx

from ..http_client_pool import get_http_pool
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.json_utils import json_loads
from ..error_types import CredentialNeedsReauthError

lib_logger = logging.getLogger("rotator_library")

CLIENT_ID = (
    "f0304373b74a44d2b584a3fb70ca9e56"  # https://api.kilocode.ai/extension-config.json
)
SCOPE = "openid profile email model.completion"
TOKEN_ENDPOINT = "https://chat.qwen.ai/api/v1/oauth2/token"
console = Console()


from .google_oauth_base import GoogleOAuthBase


class QwenAuthBase(GoogleOAuthBase):
    CLIENT_ID = "f0304373b74a44d2b584a3fb70ca9e56"
    CLIENT_SECRET = ""  # Qwen uses public client (no secret)
    OAUTH_SCOPES = ["openid", "profile", "email", "model.completion"]
    ENV_PREFIX = "QWEN_CODE"
    TOKEN_URI = TOKEN_ENDPOINT
    REFRESH_EXPIRY_BUFFER_SECONDS = 3 * 60 * 60  # 3 hours buffer
    _cache_default_ttl: float = 14400.0  # 4hr: aligns with 3hr buffer + 1hr token lifetime
    BUFFER_ON_FAILURE: ClassVar[bool] = False

    def __init__(self):
        super().__init__()

    def _load_from_env(
        self, credential_index: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        # Use parent's generic implementation, then add Qwen-specific fields
        creds = super()._load_from_env(credential_index)
        if creds is None:
            return None

        if credential_index and credential_index != "0":
            prefix = f"QWEN_CODE_{credential_index}"
        else:
            prefix = "QWEN_CODE"

        # Add Qwen-specific fields not present in generic Google OAuth creds
        creds["resource_url"] = os.getenv(
            f"{prefix}_RESOURCE_URL", "https://portal.qwen.ai/v1"
        )
        # Remove Google-specific fields that don't apply to Qwen
        creds.pop("client_id", None)
        creds.pop("client_secret", None)
        creds.pop("token_uri", None)
        creds.pop("universe_domain", None)

        return creds

    # _read_creds_from_file: removed — inherited from GoogleOAuthBase._load_credentials
    # _load_credentials: removed — inherited from GoogleOAuthBase (calls self._load_from_env)

    async def _refresh_token(self, path: str, creds: Optional[Dict[str, Any]] = None, force: bool = False) -> Dict[str, Any]:
        async with self._get_lock(path):
            cached_creds = self._credentials_cache.get(path)
            if not force and cached_creds and not self._is_token_expired(cached_creds):
                return cached_creds

            # [ROTATING TOKEN FIX] Always read fresh from disk before refresh.
            # Qwen uses rotating refresh tokens - each refresh invalidates the previous token.
            if path in self._credentials_cache:
                try:
                    creds_raw = await asyncio.to_thread(Path(path).read_text, encoding="utf-8")
                    self._credentials_cache[path] = json_loads(creds_raw)
                except FileNotFoundError:
                    pass
            creds_from_file = self._credentials_cache[path]

            lib_logger.debug(f"Refreshing Qwen OAuth token for '{Path(path).name}'...")
            refresh_token = creds_from_file.get("refresh_token")
            if not refresh_token:
                lib_logger.error(f"No refresh_token found in '{Path(path).name}'")
                raise ValueError("No refresh_token found in Qwen credentials file.")

            # [RETRY LOGIC] Implement exponential backoff for transient errors
            max_retries = 3
            new_token_data = None
            last_error = None

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            }

            pool = await get_http_pool()
            client = await pool.get_client_async()
            for attempt in range(max_retries):
                try:
                    response = await client.post(
                        TOKEN_ENDPOINT,
                        headers=headers,
                        data={
                            "grant_type": "refresh_token",
                            "refresh_token": refresh_token,
                            "client_id": CLIENT_ID,
                        },
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    new_token_data = response.json()
                    break  # Success

                except httpx.HTTPStatusError as e:
                    last_error = e
                    status_code = e.response.status_code
                    error_body = e.response.text
                    lib_logger.error(
                        f"HTTP {status_code} for '{Path(path).name}': {error_body}"
                    )

                    # [INVALID GRANT HANDLING] Handle 400/401/403 by raising
                    # The caller (_process_refresh_queue or initialize_token) will handle re-auth
                    # We must NOT call initialize_token from here as we hold a lock (would deadlock)
                    if status_code == 400:
                        # Check if this is an invalid refresh token error
                        try:
                            error_data = e.response.json()
                            error_type = error_data.get("error", "")
                            error_desc = error_data.get("error_description", "")
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

            creds_from_file["access_token"] = new_token_data["access_token"]
            creds_from_file["refresh_token"] = new_token_data.get(
                "refresh_token", creds_from_file["refresh_token"]
            )
            creds_from_file["expiry_date"] = (
                time.time() + new_token_data["expires_in"]
            ) * 1000
            creds_from_file["resource_url"] = new_token_data.get(
                "resource_url", creds_from_file.get("resource_url")
            )

            # Ensure _proxy_metadata exists and update timestamp
            if "_proxy_metadata" not in creds_from_file:
                creds_from_file["_proxy_metadata"] = {}
            creds_from_file["_proxy_metadata"]["last_check_timestamp"] = time.time()

            # [VALIDATION] Verify required fields exist after refresh
            required_fields = ["access_token", "refresh_token"]
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
                # CRITICAL: For rotating tokens, if we can't persist the new token,
                # the old token is already invalidated by Qwen. This is a critical failure.
                # Raise an error so retry logic kicks in.
                raise IOError(
                    f"Failed to persist refreshed credentials for '{Path(path).name}'. "
                    f"Disk write failed - refresh will be retried."
                )

            lib_logger.debug(
                f"Successfully refreshed Qwen OAuth token for '{Path(path).name}'."
            )
            return self._credentials_cache[path]  # Return from cache (synced with disk)

    async def get_api_details(self, credential_identifier: str) -> Tuple[str, str]:
        """
        Returns the API base URL and access token.

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

            if self._is_token_expired(creds):
                creds = await self._refresh_token(credential_identifier)

            base_url = creds.get("resource_url", "https://portal.qwen.ai/v1")
            if not base_url.startswith("http"):
                base_url = f"https://{base_url}"
            access_token = creds["access_token"]
        else:
            # Direct API key: use as-is
            lib_logger.debug("Using direct API key for Qwen Code")
            base_url = "https://portal.qwen.ai/v1"
            access_token = credential_identifier

        return base_url, access_token

    # proactively_refresh inherited from GoogleOAuthBase (with IOError handling)

    def _is_invalid_grant_error(self, error_body: str, status_code: int, error_type: str = "") -> bool:
        """Qwen uses 'invalid' in error description instead of 'invalid_grant'."""
        if status_code == 400:
            return "invalid" in error_body.lower() or error_type == "invalid_request"
        return False

    # _process_refresh_queue inherited from GoogleOAuthBase

    async def _perform_interactive_oauth(
        self, path: str, creds: Dict[str, Any], display_name: str
    ) -> Dict[str, Any]:
        """
        Perform interactive OAuth device flow (browser-based authentication).

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

        code_verifier = (
            base64.urlsafe_b64encode(secrets.token_bytes(32))
            .decode("utf-8")
            .rstrip("=")
        )
        code_challenge = (
            base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode("utf-8")).digest()
            )
            .decode("utf-8")
            .rstrip("=")
        )

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        pool = await get_http_pool()
        client = await pool.get_client_async()
        request_data = {
            "client_id": CLIENT_ID,
            "scope": SCOPE,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        lib_logger.debug(f"Qwen device code request data: {request_data}")
        try:
            dev_response = await client.post(
                "https://chat.qwen.ai/api/v1/oauth2/device/code",
                headers=headers,
                data=request_data,
            )
            dev_response.raise_for_status()
            dev_data = dev_response.json()
            lib_logger.debug(f"Qwen device auth response: {dev_data}")
        except httpx.HTTPStatusError as e:
            lib_logger.error(
                f"Qwen device code request failed with status {e.response.status_code}: {e.response.text}"
            )
            raise e

        # [HEADLESS SUPPORT] Display appropriate instructions
        if is_headless:
            auth_panel_text = Text.from_markup(
                "Running in headless environment (no GUI detected).\n"
                "Please open the URL below in a browser on another machine to authorize:\n"
                "1. Visit the URL below to sign in.\n"
                "2. [bold]Copy your email[/bold] or another unique identifier and authorize the application.\n"
                "3. You will be prompted to enter your identifier after authorization."
            )
        else:
            auth_panel_text = Text.from_markup(
                "1. Visit the URL below to sign in.\n"
                "2. [bold]Copy your email[/bold] or another unique identifier and authorize the application.\n"
                "3. You will be prompted to enter your identifier after authorization."
            )

        console.print(
            Panel(
                auth_panel_text,
                title=f"Qwen OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                style="bold blue",
            )
        )
        verification_url = dev_data["verification_uri_complete"]
        escaped_url = rich_escape(verification_url)
        console.print(
            f"[bold]URL:[/bold] [link={verification_url}]{escaped_url}[/link]\n"
        )

        # [HEADLESS SUPPORT] Only attempt browser open if NOT headless
        if not is_headless:
            try:
                webbrowser.open(dev_data["verification_uri_complete"])
                lib_logger.info("Browser opened successfully for Qwen OAuth flow")
            except Exception as e:
                lib_logger.warning(
                    f"Failed to open browser automatically: {e}. Please open the URL manually."
                )

        token_data = None
        start_time = time.time()
        interval = dev_data.get("interval", 5)

        with console.status(
            "[bold green]Polling for token, please complete authentication in the browser...[/bold green]",
            spinner="dots",
        ):
            while time.time() - start_time < dev_data["expires_in"]:
                poll_response = await client.post(
                    TOKEN_ENDPOINT,
                    headers=headers,
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": dev_data["device_code"],
                        "client_id": CLIENT_ID,
                        "code_verifier": code_verifier,
                    },
                )
                if poll_response.status_code == 200:
                    token_data = poll_response.json()
                    lib_logger.info("Successfully received token.")
                    break
                elif poll_response.status_code == 400:
                    poll_data = poll_response.json()
                    error_type = poll_data.get("error")
                    if error_type == "authorization_pending":
                        lib_logger.debug(
                            f"Polling status: {error_type}, waiting {interval}s"
                        )
                    elif error_type == "slow_down":
                        interval = int(interval * 1.5)
                        if interval > 10:
                            interval = 10
                        lib_logger.debug(
                            f"Polling status: {error_type}, waiting {interval}s"
                        )
                    else:
                        raise ValueError(
                            f"Token polling failed: {poll_data.get('error_description', error_type)}"
                        )
                else:
                    poll_response.raise_for_status()

                await asyncio.sleep(interval)

        if not token_data:
            raise TimeoutError("Qwen device flow timed out.")

        creds.update(
            {
                "access_token": token_data["access_token"],
                "refresh_token": token_data.get("refresh_token"),
                "expiry_date": (time.time() + token_data["expires_in"]) * 1000,
                "resource_url": token_data.get("resource_url"),
            }
        )

        # Prompt for user identifier and create metadata object if needed
        if not creds.get("_proxy_metadata", {}).get("email"):
            try:
                prompt_text = Text.from_markup(
                    f"\\n[bold]Please enter your email or a unique identifier for [yellow]'{display_name}'[/yellow][/bold]"
                )
                email = Prompt.ask(prompt_text)
                creds["_proxy_metadata"] = {
                    "email": email.strip(),
                    "last_check_timestamp": time.time(),
                }
            except (EOFError, KeyboardInterrupt):
                console.print(
                    "\\n[bold yellow]No identifier provided. Deduplication will not be possible.[/bold yellow]"
                )
                creds["_proxy_metadata"] = {
                    "email": None,
                    "last_check_timestamp": time.time(),
                }

        if path:
            if not await self._save_credentials(path, creds):
                raise IOError(
                    f"Failed to save OAuth credentials to disk for '{display_name}'. "
                    f"Please retry authentication."
                )
        lib_logger.info(
            f"Qwen OAuth initialized successfully for '{display_name}'."
        )
        return creds

    # get_auth_header: inherited from GoogleOAuthBase (uses _primary_token_key="access_token" by default)

    async def get_user_info(
        self, creds_or_path: Union[Dict[str, Any], str]
    ) -> Dict[str, Any]:
        """
        Retrieves user info from the _proxy_metadata in the credential file.
        Overrides parent to avoid Google UserInfo API call (Qwen uses metadata only).
        """
        try:
            path = creds_or_path if isinstance(creds_or_path, str) else None
            creds = (
                await self._load_credentials(creds_or_path) if path else creds_or_path
            )

            # This will ensure the token is valid and metadata exists if the flow was just run
            if path:
                await self.initialize_token(path)
                creds = await self._load_credentials(
                    path
                )  # Re-load after potential init

            metadata = creds.get("_proxy_metadata", {"email": None})
            email = metadata.get("email")

            if not email:
                lib_logger.warning(
                    f"No email found in _proxy_metadata for '{path or 'in-memory object'}'."
                )

            # Update timestamp in cache only (not disk) to avoid overwriting
            # potentially newer tokens that were saved by another process/refresh.
            # The timestamp is non-critical metadata - losing it on restart is fine.
            if path and "_proxy_metadata" in creds:
                creds["_proxy_metadata"]["last_check_timestamp"] = time.time()

            return {"email": email}
        except Exception as e:
            lib_logger.error(f"Failed to get Qwen user info from credentials: {e}")
            return {"email": None}

    def build_env_lines(self, creds: Dict[str, Any], cred_number: int) -> List[str]:
        """Generate .env file lines for a Qwen credential."""
        email = self._get_env_email(creds)
        prefix = f"{self.ENV_PREFIX}_{cred_number}"
        lines = self._build_env_header(email, cred_number)

        lines.extend([
            f"{prefix}_ACCESS_TOKEN={creds.get('access_token', '')}",
            f"{prefix}_REFRESH_TOKEN={creds.get('refresh_token', '')}",
            f"{prefix}_EXPIRY_DATE={creds.get('expiry_date', 0)}",
            f"{prefix}_RESOURCE_URL={creds.get('resource_url', 'https://portal.qwen.ai/v1')}",
            f"{prefix}_EMAIL={email}",
        ])

        return lines
