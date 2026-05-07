# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/credential_tool.py
#
# Thin orchestrator: entry points + re-exports.
# Logic lives in credential_io, credential_providers, credential_display,
# credential_setup, and credential_exports.

import asyncio
import time

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .utils.terminal_utils import clear_screen
from .credential_providers import (
    OAUTH_FRIENDLY_NAMES,
    _ensure_providers_loaded,
)
from .credential_display import (
    _display_credentials_summary,
    _display_oauth_providers_summary,
    _display_provider_credentials,
    _view_api_keys_detail,
    _view_oauth_credentials_detail,
)
from .credential_setup import (
    ensure_env_defaults,
    setup_api_key,
    setup_custom_openai_provider,
    setup_new_credential,
    manage_credentials_submenu,
)
from .credential_exports import export_credentials_submenu

console = Console()

# Re-export public names consumed by external modules:
#   from rotator_library.credential_tool import run_credential_tool
#   from rotator_library.credential_tool import ensure_env_defaults
#   from rotator_library.credential_tool import _ensure_providers_loaded
__all__ = ["run_credential_tool", "ensure_env_defaults", "_ensure_providers_loaded"]


async def view_credentials_menu():
    """
    Menu for viewing credentials. Shows summary first, then allows drilling
    down to view detailed credentials for a specific provider.
    """
    from .credential_providers import _get_api_keys_from_env, _get_oauth_credentials_summary

    while True:
        clear_screen("View Credentials")

        # Display summary
        _display_credentials_summary()

        # Build list of all providers with credentials
        api_keys = await asyncio.to_thread(_get_api_keys_from_env)
        oauth_creds = await asyncio.to_thread(_get_oauth_credentials_summary)

        all_providers = []

        # Add API key providers
        for provider in sorted(api_keys.keys()):
            count = len(api_keys[provider])
            all_providers.append(("api", provider, count))

        # Add OAuth providers with credentials
        for provider in sorted(oauth_creds.keys()):
            if oauth_creds[provider]:
                count = len(oauth_creds[provider])
                display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
                all_providers.append(("oauth", provider, count, display_name))

        if not all_providers:
            console.print("[bold yellow]No credentials configured.[/bold yellow]")
            console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            input()
            break

        # Display provider selection menu
        console.print(
            Panel(
                Text.from_markup("[bold]Select a provider to view details:[/bold]"),
                title="View Provider Credentials",
                style="bold blue",
            )
        )

        for i, provider_info in enumerate(all_providers, 1):
            if provider_info[0] == "api":
                _, provider, count = provider_info
                console.print(f"  {i}. [cyan]API:[/cyan] {provider} ({count} key(s))")
            else:
                _, provider, count, display_name = provider_info
                console.print(
                    f"  {i}. [cyan]OAuth:[/cyan] {display_name} ({count} credential(s))"
                )

        choice = Prompt.ask(
            Text.from_markup(
                "\n[bold]Select provider or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=[str(i) for i in range(1, len(all_providers) + 1)] + ["b"],
            show_choices=False,
        )

        if choice.lower() == "b":
            break

        try:
            idx = int(choice) - 1
            provider_info = all_providers[idx]

            if provider_info[0] == "api":
                _, provider, _ = provider_info
                await _view_api_keys_detail(provider)
            else:
                _, provider, _, _ = provider_info
                await _view_oauth_credentials_detail(provider)

        except (ValueError, IndexError):
            console.print("[bold red]Invalid choice.[/bold red]")
            await asyncio.sleep(1)


async def main(clear_on_start=True):
    """
    An interactive CLI tool to add new credentials.

    Args:
        clear_on_start: If False, skip initial screen clear (used when called from launcher
                       to preserve the loading screen)
    """
    ensure_env_defaults()

    # Only show header if we're clearing (standalone mode)
    if clear_on_start:
        clear_screen("Interactive Credential Setup")

    while True:
        # Clear screen between menu selections for cleaner UX
        clear_screen("Interactive Credential Setup")

        # Display credentials summary at the top
        _display_credentials_summary()

        console.print(
            Panel(
                Text.from_markup(
                    "1. Add OAuth Credential\n"
                    "2. Add API Key\n"
                    "3. Add Custom OpenAI-Compatible Provider\n"
                    "4. Export Credentials\n"
                    "5. View Credentials\n"
                    "6. Manage Credentials"
                ),
                title="Choose action",
                style="bold blue",
            )
        )

        setup_type = Prompt.ask(
            Text.from_markup(
                "[bold]Please select an option or type [red]'q'[/red] to quit[/bold]"
            ),
            choices=["1", "2", "3", "4", "5", "6", "q"],
            show_choices=False,
        )

        if setup_type.lower() == "q":
            break

        if setup_type == "1":
            # Clear and show OAuth providers summary before listing providers
            clear_screen("Add OAuth Credential")
            _display_oauth_providers_summary()

            auth_map, _ = _ensure_providers_loaded()
            available_providers = list(auth_map.keys())

            provider_text = Text()
            for i, provider in enumerate(available_providers):
                display_name = OAUTH_FRIENDLY_NAMES.get(
                    provider, provider.replace("_", " ").title()
                )
                provider_text.append(f"  {i + 1}. {display_name}\n")

            console.print(
                Panel(
                    provider_text,
                    title="Available Providers for OAuth",
                    style="bold blue",
                )
            )

            choice = Prompt.ask(
                Text.from_markup(
                    "[bold]Please select a provider or type [red]'b'[/red] to go back[/bold]"
                ),
                choices=[str(i + 1) for i in range(len(available_providers))] + ["b"],
                show_choices=False,
            )

            if choice.lower() == "b":
                continue

            try:
                choice_index = int(choice) - 1
                if 0 <= choice_index < len(available_providers):
                    provider_name = available_providers[choice_index]
                    display_name = OAUTH_FRIENDLY_NAMES.get(
                        provider_name, provider_name.replace("_", " ").title()
                    )

                    # Show existing credentials for this provider before proceeding
                    _display_provider_credentials(provider_name)

                    console.print(
                        f"Starting OAuth setup for [bold cyan]{display_name}[/bold cyan]..."
                    )
                    await setup_new_credential(provider_name)
                    # Don't clear after OAuth - user needs to see full flow
                    console.print("\n[dim]Press Enter to return to main menu...[/dim]")
                    input()
                else:
                    console.print(
                        "[bold red]Invalid choice. Please try again.[/bold red]"
                    )
                    await asyncio.sleep(1.5)
            except ValueError:
                console.print(
                    "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
                )
                await asyncio.sleep(1.5)

        elif setup_type == "2":
            await setup_api_key()
            # console.print("\n[dim]Press Enter to return to main menu...[/dim]")
            # input()

        elif setup_type == "3":
            await setup_custom_openai_provider()

        elif setup_type == "4":
            await export_credentials_submenu()

        elif setup_type == "5":
            await view_credentials_menu()

        elif setup_type == "6":
            await manage_credentials_submenu()


def run_credential_tool(from_launcher=False):
    """
    Entry point for credential tool.

    Args:
        from_launcher: If True, skip loading screen (launcher already showed it)
    """
    # Check if we need to show loading screen
    if not from_launcher:
        # Standalone mode - show full loading UI
        print("\033[2J\033[H", end="", flush=True)

        _start_time = time.time()

        # Phase 1: Show initial message
        print("━" * 70)
        print("Interactive Credential Setup Tool")
        print("GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
        print("━" * 70)
        print("Loading credential management components...")

        # Phase 2: Load dependencies with spinner
        with console.status("Loading authentication providers...", spinner="dots"):
            _ensure_providers_loaded()
        console.print("✓ Authentication providers loaded")

        with console.status("Initializing credential tool...", spinner="dots"):
            time.sleep(0.2)  # Brief pause for UI consistency
        console.print("✓ Credential tool initialized")

        _elapsed = time.time() - _start_time
        _, _plugins = _ensure_providers_loaded()
        print(f"✓ Tool ready in {_elapsed:.2f}s ({len(_plugins) if _plugins is not None else 0} providers available)")

        # Small delay to let user see the ready message
        time.sleep(0.5)

    # Run the main async event loop
    # If from launcher, don't clear screen at start to preserve loading messages
    try:
        asyncio.run(main(clear_on_start=not from_launcher))
        clear_screen()  # Clear terminal when credential tool exits
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Exiting setup.[/bold yellow]")
        clear_screen()  # Clear terminal on keyboard interrupt too
