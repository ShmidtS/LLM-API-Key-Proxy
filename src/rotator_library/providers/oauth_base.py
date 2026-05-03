# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/oauth_base.py

"""
Consolidated OAuth base classes.

Provides BaseTokenManager, AuthQueueMixin, and OAuthFlowMixin.
OAuthMixin is retained as a backward-compatible alias of OAuthFlowMixin.

All classes are imported by google_oauth_base.py and inherited
by GoogleOAuthBase -> {GeminiAuthBase, AntigravityAuthBase, QwenAuthBase, IFlowAuthBase}.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import httpx

from ..utils.reauth_coordinator import get_reauth_coordinator
from ..utils.ttl_dict import TTLDict

lib_logger = logging.getLogger("rotator_library")


# =========================================================================
# BaseTokenManager - common state infrastructure
# =========================================================================


class BaseTokenManager:
    """Common state and infrastructure for token refresh queue management.

    Shared by GoogleOAuthBase, QwenAuthBase, IFlowAuthBase.
    Contains only the __init__ state initialization that is identical across all.
    """

    # Subclasses override to match their REFRESH_EXPIRY_BUFFER_SECONDS + token lifetime.
    # Default 3600s works for Google (30min buffer, 1hr tokens).
    # Qwen (3hr buffer) and iFlow (24hr buffer) override with longer TTLs.
    _cache_default_ttl: float = 3600.0

    def __init__(self):
        # Cache and lock management
        self._credentials_cache: TTLDict = TTLDict(maxsize=200, default_ttl=self._cache_default_ttl)
        self._refresh_locks: Dict[str, asyncio.Lock] = {}
        self._locks_lock = asyncio.Lock()

        # Refresh failure tracking
        self._refresh_failures: Dict[str, int] = {}
        self._next_refresh_after: Dict[str, float] = {}

        # Queue infrastructure (Python 3.10+ required: asyncio.Queue in __init__ binds to the running loop lazily)
        self._refresh_queue: asyncio.Queue = asyncio.Queue()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._reauth_queue: asyncio.Queue = asyncio.Queue()
        self._reauth_processor_task: Optional[asyncio.Task] = None

        # Queue deduplication and unavailability tracking
        self._queued_credentials: set = set()
        self._unavailable_credentials: Dict[str, float] = {}
        self._unavailable_ttl_seconds: int = 360
        self._queue_tracking_lock = asyncio.Lock()
        self._queue_retry_count: Dict[str, int] = {}

        # Timing constants
        self._refresh_timeout_seconds: int = 15
        self._refresh_interval_seconds: int = 30
        self._refresh_max_retries: int = 3
        self._reauth_timeout_seconds: int = 300

    def reset_auth_caches(self, credential: str) -> None:
        """Clear all cached state for a credential after an auth error.

        Subclasses with additional caches should override and call super()
        to ensure all caches are cleared consistently.
        """
        self._credentials_cache.pop(credential, None)
        self._unavailable_credentials.pop(credential, None)
        self._refresh_locks.pop(credential, None)


# =========================================================================
# AuthQueueMixin - deduplicated refresh/re-auth queue processing
# =========================================================================


class AuthQueueMixin:
    """Shared mixin providing deduplicated refresh/re-auth queue processing methods.

    Used by GoogleOAuthBase, IFlowAuthBase, and QwenAuthBase.
    Requires the consuming class to provide these attributes/methods:
    - _queue_retry_count: dict tracking retry counts per credential path
    - _refresh_max_retries: int, maximum retry attempts
    - _queued_credentials: set of currently queued credential paths
    - _refresh_queue: asyncio.Queue for normal refresh operations
    - _reauth_queue: asyncio.Queue for re-auth operations
    - _queue_tracking_lock: asyncio.Lock for queue tracking
    - _reauth_processor_task: task reference for the re-auth processor
    - _queue_processor_task: task reference for the refresh processor
    - _unavailable_credentials: dict tracking unavailable credentials
    - _unavailable_ttl_seconds: int, TTL for unavailable credential entries
    - _locks_lock: asyncio.Lock for protecting lock creation
    - _refresh_locks: dict of asyncio.Lock per credential path
    - _next_refresh_after: dict tracking backoff times
    - _refresh_timeout_seconds: int, timeout for refresh operations
    - _refresh_interval_seconds: int, delay between processing credentials
    - _credentials_cache: dict caching loaded credentials
    - initialize_token(path, force_interactive=True): async method
    - _refresh_token(path, creds=None, force=False): async method
    - _load_credentials(path): async method
    - _is_token_expired(creds): method to check if token needs refresh
    - _is_token_truly_expired(creds): method to check if token is actually expired
    """

    def _is_invalid_grant_error(self, error_body: str, status_code: int, error_type: str = "") -> bool:
        """Check if an HTTP error indicates an invalid/expired refresh token needing re-auth.

        Default: checks for 'invalid' in error body or 'invalid_request' error type.
        Override in subclasses for provider-specific error patterns (e.g. Google checks 'invalid_grant').
        """
        if status_code == 400:
            return "invalid" in error_body.lower() or error_type == "invalid_request"
        return False

    # =========================================================================
    # LOCK MANAGEMENT
    # =========================================================================

    async def _get_lock(self, path: str) -> asyncio.Lock:
        """Gets or creates a lock for the given credential path.

        Uses an asyncio.Lock to prevent TOCTOU race conditions during lock creation.
        """
        # Fast path - lock already exists
        if path in self._refresh_locks:
            return self._refresh_locks[path]

        # Async-safe creation
        async with self._locks_lock:
            if path not in self._refresh_locks:
                self._refresh_locks[path] = asyncio.Lock()
            return self._refresh_locks[path]

    # =========================================================================
    # CREDENTIAL AVAILABILITY CHECK
    # =========================================================================

    def is_credential_available(self, path: str) -> bool:
        """Check if a credential is available for rotation.

        Credentials are unavailable if:
        1. In re-auth queue (token is truly broken, requires user interaction)
        2. Token is TRULY expired (past actual expiry, not just threshold)

        Note: Credentials in normal refresh queue are still available because
        the old token is valid until actual expiry.

        TTL cleanup (defense-in-depth): If a credential has been in the re-auth
        queue longer than _unavailable_ttl_seconds without being processed, it's
        cleaned up. This should only happen if the re-auth processor crashes or
        is cancelled without proper cleanup.
        """
        # Check if in re-auth queue (truly unavailable)
        if path in self._unavailable_credentials:
            marked_time = self._unavailable_credentials.get(path)
            if marked_time is not None:
                now = time.time()
                if now - marked_time > self._unavailable_ttl_seconds:
                    # Entry is stale - clean it up and return available
                    lib_logger.warning(
                        f"Credential '{Path(path).name}' stuck in re-auth queue for "
                        f"{int(now - marked_time)}s (TTL: {self._unavailable_ttl_seconds}s). "
                        f"Re-auth processor may have crashed. Auto-cleaning stale entry."
                    )
                    # Clean up both tracking structures for consistency
                    self._unavailable_credentials.pop(path, None)
                    self._queued_credentials.discard(path)
                else:
                    return False  # Still in re-auth, not available

        # Check if token is TRULY expired (not just threshold-expired)
        creds = self._credentials_cache.get(path)
        if creds and self._is_token_truly_expired(creds):
            # Token is actually expired - should not be used
            # Queue for refresh if not already queued
            if path not in self._queued_credentials:
                asyncio.create_task(
                    self._queue_refresh(path, force=True, needs_reauth=False)
                )
            return False

        return True

    # =========================================================================
    # QUEUE PROCESSOR LAZY START
    # =========================================================================

    async def _ensure_queue_processor_running(self):
        """Lazily starts the queue processor if not already running."""
        if self._queue_processor_task is None or self._queue_processor_task.done():
            self._queue_processor_task = asyncio.create_task(
                self._process_refresh_queue()
            )

    async def _ensure_reauth_processor_running(self):
        """Lazily starts the re-auth queue processor if not already running."""
        if self._reauth_processor_task is None or self._reauth_processor_task.done():
            self._reauth_processor_task = asyncio.create_task(
                self._process_reauth_queue()
            )

    # =========================================================================
    # QUEUE REFRESH ROUTING
    # =========================================================================

    async def _queue_refresh(
        self, path: str, force: bool = False, needs_reauth: bool = False
    ):
        """Add a credential to the appropriate refresh queue if not already queued.

        Args:
            path: Credential file path
            force: Force refresh even if not expired
            needs_reauth: True if full re-authentication needed (routes to re-auth queue)

        Queue routing:
        - needs_reauth=True: Goes to re-auth queue, marks as unavailable
        - needs_reauth=False: Goes to normal refresh queue, does NOT mark unavailable
          (old token is still valid until actual expiry)
        """
        # IMPORTANT: Only check backoff for simple automated refreshes
        # Re-authentication (interactive OAuth) should BYPASS backoff since it needs user input
        if not needs_reauth:
            now = time.time()
            if path in self._next_refresh_after:
                backoff_until = self._next_refresh_after[path]
                if now < backoff_until:
                    # Credential is in backoff for automated refresh, do not queue
                    return

        async with self._queue_tracking_lock:
            if path not in self._queued_credentials:
                self._queued_credentials.add(path)

                if needs_reauth:
                    # Re-auth queue: mark as unavailable (token is truly broken)
                    self._unavailable_credentials[path] = time.time()
                    await self._reauth_queue.put(path)
                    await self._ensure_reauth_processor_running()
                else:
                    # Normal refresh queue: do NOT mark unavailable (old token still valid)
                    await self._refresh_queue.put((path, force))
                    await self._ensure_queue_processor_running()

    # =========================================================================
    # REFRESH QUEUE PROCESSOR
    # =========================================================================

    async def _process_refresh_queue(self):
        """Background worker that processes normal refresh requests sequentially.

        Key behaviors:
        - 15s timeout per refresh operation
        - 30s delay between processing credentials (prevents thundering herd)
        - On failure: back of queue, max 3 retries before kicked
        - If 401/403 detected: routes to re-auth queue
        - Does NOT mark credentials unavailable (old token still valid)
        """
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path, force = await asyncio.wait_for(
                        self._refresh_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Queue is empty and idle for 60s - clean up and exit
                    async with self._queue_tracking_lock:
                        # Clear any stale retry counts
                        self._queue_retry_count.clear()
                        self._queue_processor_task = None
                    return

                try:
                    # Quick check if still expired (optimization to avoid unnecessary refresh)
                    creds = self._credentials_cache.get(path)
                    if creds and not self._is_token_expired(creds):
                        # No longer expired, skip refresh
                        # Clear retry count on skip (not a failure)
                        self._queue_retry_count.pop(path, None)
                        continue

                    # Perform refresh with timeout
                    if not creds:
                        creds = await self._load_credentials(path)

                    # Try-lock: if a user request holds the per-path lock,
                    # skip this round — background refresh is best-effort and
                    # must not block incoming API requests.
                    lock = await self._get_lock(path)
                    if lock.locked():
                        lib_logger.debug(
                            f"Skipping refresh for '{Path(path).name}': "
                            f"lock held by user request"
                        )
                        self._queue_retry_count.pop(path, None)
                        continue

                    try:
                        async with asyncio.timeout(self._refresh_timeout_seconds):
                            await self._refresh_token(path, creds, force=force)

                        # SUCCESS: Clear retry count and lock
                        self._queue_retry_count.pop(path, None)
                        self._refresh_locks.pop(path, None)

                    except asyncio.TimeoutError:
                        lib_logger.warning(
                            f"Refresh timeout ({self._refresh_timeout_seconds}s) for '{Path(path).name}'"
                        )
                        await self._handle_refresh_failure(path, force, "timeout")

                    except httpx.HTTPStatusError as e:
                        status_code = e.response.status_code
                        # Check for invalid refresh token errors (400/401/403)
                        # These need to be routed to re-auth queue for interactive OAuth
                        needs_reauth = False

                        if status_code == 400:
                            # Check if this is an invalid refresh token error
                            try:
                                error_data = e.response.json()
                                error_type = error_data.get("error", "")
                                error_desc = error_data.get("error_description", "")
                                if not error_desc:
                                    error_desc = error_data.get("message", str(e))
                            except (httpx.HTTPStatusError, ValueError) as e:
                                lib_logger.debug("Failed to parse OAuth error response JSON: %s", e)
                                error_type = ""
                                error_desc = str(e)

                            if self._is_invalid_grant_error(error_desc, status_code, error_type):
                                needs_reauth = True
                                lib_logger.info(
                                    f"Credential '{Path(path).name}' needs re-auth (HTTP 400: {error_desc}). "
                                    f"Routing to re-auth queue."
                                )
                        elif status_code in (401, 403):
                            needs_reauth = True
                            lib_logger.info(
                                f"Credential '{Path(path).name}' needs re-auth (HTTP {status_code}). "
                                f"Routing to re-auth queue."
                            )

                        if needs_reauth:
                            self._queue_retry_count.pop(path, None)  # Clear retry count
                            async with self._queue_tracking_lock:
                                self._queued_credentials.discard(path)
                            await self._queue_refresh(path, force=True, needs_reauth=True)
                        else:
                            await self._handle_refresh_failure(
                                path, force, f"HTTP {status_code}"
                            )

                    except asyncio.CancelledError:
                        raise
                    except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError) as e:
                        await self._handle_refresh_failure(path, force, str(e))

                finally:
                    # Remove from queued set (unless re-queued by failure handler)
                    async with self._queue_tracking_lock:
                        # Only discard if not re-queued (check if still in queue set from retry)
                        if (
                            path in self._queued_credentials
                            and self._queue_retry_count.get(path, 0) == 0
                        ):
                            self._queued_credentials.discard(path)
                    self._refresh_queue.task_done()

                    # Wait between credentials to spread load
                    await asyncio.sleep(self._refresh_interval_seconds)

            except asyncio.CancelledError:
                break
            except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError, OSError) as e:
                lib_logger.error(f"Error in refresh queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)

    # =========================================================================
    # RE-AUTH QUEUE PROCESSOR
    # =========================================================================

    async def _process_reauth_queue(self):
        """Background worker that processes re-auth requests.

        Key behaviors:
        - Credentials ARE marked unavailable (token is truly broken)
        - Uses ReauthCoordinator for interactive OAuth
        - No automatic retry (requires user action)
        - Cleans up unavailable status when done
        """
        while True:
            path = None
            try:
                # Wait for an item with timeout to allow graceful shutdown
                try:
                    path = await asyncio.wait_for(
                        self._reauth_queue.get(), timeout=60.0
                    )
                except asyncio.TimeoutError:
                    # Queue is empty and idle for 60s - exit
                    self._reauth_processor_task = None
                    return

                try:
                    lib_logger.info(f"Starting re-auth for '{Path(path).name}'...")
                    await self.initialize_token(path, force_interactive=True)
                    lib_logger.info(f"Re-auth SUCCESS for '{Path(path).name}'")

                except asyncio.CancelledError:
                    raise
                except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError, OSError) as e:
                    lib_logger.error(f"Re-auth FAILED for '{Path(path).name}': {e}")
                    # No automatic retry for re-auth (requires user action)

                finally:
                    # Always clean up
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                    self._reauth_queue.task_done()

            except asyncio.CancelledError:
                # Clean up current credential before breaking
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)
                break
            except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError, OSError) as e:
                lib_logger.error(f"Error in re-auth queue processor: {e}")
                if path:
                    async with self._queue_tracking_lock:
                        self._queued_credentials.discard(path)
                        self._unavailable_credentials.pop(path, None)

    # =========================================================================
    # REFRESH FAILURE HANDLER
    # =========================================================================

    async def _handle_refresh_failure(self, path: str, force: bool, error: str):
        """Handle a refresh failure with back-of-line retry logic.

        - Increments retry count
        - If under max retries: re-adds to END of queue
        - If at max retries: kicks credential out (retried next BackgroundRefresher cycle)
        """
        retry_count = self._queue_retry_count.get(path, 0) + 1
        self._queue_retry_count[path] = retry_count

        if retry_count >= self._refresh_max_retries:
            # Kicked out until next BackgroundRefresher cycle
            lib_logger.error(
                f"Max retries ({self._refresh_max_retries}) reached for '{Path(path).name}' "
                f"(last error: {error}). Will retry next refresh cycle."
            )
            self._queue_retry_count.pop(path, None)
            async with self._queue_tracking_lock:
                self._queued_credentials.discard(path)
            return

        # Re-add to END of queue for retry
        lib_logger.warning(
            f"Refresh failed for '{Path(path).name}' ({error}). "
            f"Retry {retry_count}/{self._refresh_max_retries}, back of queue."
        )
        # Keep in queued_credentials set, add back to queue
        await self._refresh_queue.put((path, force))


# =========================================================================
# OAuthFlowMixin - shared OAuth token initialization logic + coordination
# =========================================================================


class OAuthFlowMixin:
    """Mixin providing shared OAuth token initialization logic and coordination.

    This mixin handles:
    - Determining if interactive OAuth is needed
    - Global reauth coordination to prevent concurrent auth flows
    - Token initialization with automatic refresh fallback
    - Common credential validation patterns
    """

    # ========================================================================
    # REAUTH COORDINATION
    # ========================================================================

    async def _execute_interactive_oauth(
        self,
        path: str,
        creds: Dict[str, Any],
        display_name: str,
        provider_name: str,
        oauth_func=None,
        timeout: float = 300.0,
    ) -> Dict[str, Any]:
        """Execute interactive OAuth with global coordination."""
        coordinator = get_reauth_coordinator()

        if oauth_func is None:
            oauth_func = self._perform_interactive_oauth

        async def _do_interactive_oauth():
            return await oauth_func(path, creds, display_name)

        return await coordinator.execute_reauth(
            credential_path=path or display_name,
            provider_name=provider_name,
            reauth_func=_do_interactive_oauth,
            timeout=timeout,
        )

    # ========================================================================
    # CREDENTIAL VALIDATION HELPERS
    # ========================================================================

    def _should_force_interactive(
        self,
        creds: Dict[str, Any],
        force_interactive: bool = False,
    ) -> tuple:
        """Determine if interactive OAuth is needed and return reason."""
        if force_interactive:
            return True, "re-authentication was explicitly requested (refresh token invalid)"

        if not creds.get("refresh_token"):
            return True, "refresh token is missing"

        return False, ""

    # ========================================================================
    # PROVIDER DISPLAY NAME HELPERS
    # ========================================================================

    def _get_display_name(
        self,
        creds_or_path: Union[Dict[str, Any], str],
    ) -> str:
        """Get display name from credentials or path."""
        if isinstance(creds_or_path, dict):
            return creds_or_path.get("_proxy_metadata", {}).get(
                "display_name", "in-memory object"
            )
        else:
            path = creds_or_path
            return Path(path).name if path else "in-memory object"

    # ========================================================================
    # TOKEN INITIALIZATION
    # ========================================================================

    async def initialize_token(
        self,
        creds_or_path: Union[Dict[str, Any], str],
        force_interactive: bool = False,
    ) -> Dict[str, Any]:
        path: Optional[str] = creds_or_path if isinstance(creds_or_path, str) else None

        display_name = self._get_display_name(creds_or_path)

        lib_logger.debug(f"Initializing {self.ENV_PREFIX} token for '{display_name}'...")
        try:
            creds: Dict[str, Any] = (
                await self._load_credentials(creds_or_path) if path else creds_or_path  # type: ignore[arg-type]
            )
            if not isinstance(creds, dict):
                raise ValueError(f"Expected dict credentials, got {type(creds).__name__}")
            token_expired = self._is_token_expired(creds)
            needs_interactive, reason = self._should_force_interactive(
                creds, force_interactive=force_interactive
            )

            if token_expired and not needs_interactive:
                needs_interactive = True
                reason = "token is expired"

            if needs_interactive:
                if reason == "token is expired" and creds.get("refresh_token"):
                    try:
                        if not path:
                            raise ValueError("path required for token refresh")
                        return await self._refresh_token(path, creds)
                    except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError) as e:
                        lib_logger.warning(
                            f"Automatic token refresh for '{display_name}' failed: {e}. Proceeding to interactive login."
                        )

                lib_logger.warning(
                    f"{self.ENV_PREFIX} OAuth token for '{display_name}' needs setup: {reason}."
                )

                return await self._execute_interactive_oauth(
                    path=path or "",
                    creds=creds,
                    display_name=display_name,
                    provider_name=self.ENV_PREFIX,
                    timeout=300.0,
                )

            lib_logger.info(
                f"{self.ENV_PREFIX} OAuth token at '{display_name}' is valid."
            )
            return creds
        except (asyncio.CancelledError, ValueError):
            raise
        except (httpx.HTTPError, httpx.TimeoutException, KeyError, OSError) as e:
            lib_logger.error("Failed to initialize %s OAuth for '%s': %s", self.ENV_PREFIX, path, e)
            raise ValueError(
                f"Failed to initialize {self.ENV_PREFIX} OAuth for '{path}': {e}"
            ) from e


# =========================================================================
# OAuthMixin - backward-compatible alias for OAuthFlowMixin
# =========================================================================


class OAuthMixin(OAuthFlowMixin):
    """Backward-compatible alias for OAuthFlowMixin.

    Previously contained browser/headless helpers and coordination methods.
    Those helpers were dead code (subclasses call is_headless_environment()
    and webbrowser.open() directly). The remaining coordination methods now
    live in OAuthFlowMixin where initialize_token can access them without
    cross-mixin coupling.
    """
    pass
