# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/providers/oauth_mixin.py

"""
OAuth Mixin - Common OAuth flow UI and coordination logic.

This mixin provides shared functionality for OAuth providers:
- Headless environment detection
- Browser opening with fallback
- Rich console UI for auth instructions
- Global reauth coordination via ReauthCoordinator

Providers using this mixin:
- GoogleOAuthBase (gemini_cli_provider, antigravity_provider)
- IFlowAuthBase
- QwenAuthBase
"""

import logging
import webbrowser
from pathlib import Path
from typing import Dict, Any, Union

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markup import escape as rich_escape

from ..utils.headless_detection import is_headless_environment
from ..utils.reauth_coordinator import get_reauth_coordinator

lib_logger = logging.getLogger("rotator_library")
console = Console()


class OAuthMixin:
    """
    Mixin providing common OAuth flow UI and coordination.

    This mixin handles:
    - Displaying auth instructions to users (headless vs GUI)
    - Opening browsers with graceful fallback
    - Global reauth coordination to prevent concurrent auth flows
    - Common credential validation patterns
    """

    # ========================================================================
    # HEADLESS DETECTION
    # ========================================================================

    def _is_headless(self) -> bool:
        """Check if running in a headless environment (no GUI)."""
        return is_headless_environment()

    # ========================================================================
    # BROWSER UTILITIES
    # ========================================================================

    def _open_browser(self, url: str) -> bool:
        """Attempt to open URL in browser with graceful fallback."""
        if self._is_headless():
            lib_logger.debug("Skipping browser open - headless environment")
            return False

        try:
            webbrowser.open(url)
            lib_logger.info("Browser opened successfully for OAuth flow")
            return True
        except Exception as e:
            lib_logger.warning(
                f"Failed to open browser automatically: {e}. "
                "Please open the URL manually."
            )
            return False

    # ========================================================================
    # RICH UI DISPLAY
    # ========================================================================

    def _display_auth_instructions(
        self,
        auth_url: str,
        display_name: str,
        provider_name: str,
        instructions: str = None,
        headless_extra: str = None,
    ) -> None:
        """Display OAuth authentication instructions to user."""
        is_headless = self._is_headless()

        if instructions is None:
            if is_headless:
                instructions = (
                    "Running in headless environment (no GUI detected).\n"
                    "Please open the URL below in a browser on another machine to authorize:"
                )
            else:
                instructions = (
                    "1. Your browser will now open to log in and authorize the application.\n"
                    "2. If it doesn't open automatically, please open the URL below manually."
                )

        if headless_extra and is_headless:
            instructions = f"{instructions}\n{headless_extra}"

        console.print(
            Panel(
                Text.from_markup(instructions),
                title=f"{provider_name} OAuth Setup for [bold yellow]{display_name}[/bold yellow]",
                style="bold blue",
            )
        )

        escaped_url = rich_escape(auth_url)
        console.print(f"[bold]URL:[/bold] [link={auth_url}]{escaped_url}[/link]\n")

    def _display_waiting_status(self, message: str = None) -> None:
        """Display a waiting spinner status."""
        if message is None:
            message = "[bold green]Waiting for authorization in the browser...[/bold green]"
        console.print(message)

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

    async def _try_refresh_before_interactive(
        self,
        path: str,
        creds: Dict[str, Any],
        reason: str,
        display_name: str,
    ) -> Dict[str, Any]:
        """Attempt automatic token refresh before falling back to interactive OAuth."""
        if reason == "token is expired" and creds.get("refresh_token"):
            try:
                return await self._refresh_token(path, creds)
            except Exception as e:
                lib_logger.warning(
                    f"Automatic token refresh for '{display_name}' failed: {e}. "
                    "Proceeding to interactive login."
                )
                raise
        return None

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
