# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/credential_display.py

import asyncio
import json
import logging
import httpx
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .utils.terminal_utils import clear_screen
from .credential_io import _get_oauth_base_dir, _read_json_file, _write_json_file
from .credential_providers import (
    OAUTH_FRIENDLY_NAMES,
    _OAUTH_PROVIDERS,
    _ensure_providers_loaded,
    _get_provider_auth_class,
    _normalize_tier_name,
    _count_tiers,
    _format_tier_counts,
    _get_api_keys_from_env,
    _get_oauth_credentials_summary,
    _get_all_credentials_summary,
    _get_existing_custom_providers,
)

console = Console()


def _display_custom_providers_summary():
    """
    Display a summary of existing custom OpenAI-compatible providers.
    """
    custom_providers = _get_existing_custom_providers()

    if not custom_providers:
        console.print(
            "[dim]No custom OpenAI-compatible providers configured yet.[/dim]\n"
        )
        return

    table = Table(
        title="Existing Custom Providers",
        box=None,
        padding=(0, 2),
        title_style="bold cyan",
    )
    table.add_column("Provider", style="yellow", no_wrap=True)
    table.add_column("API Base", style="dim")
    table.add_column("API Key", style="green", justify="center")

    for provider in custom_providers:
        name = provider["name"].upper()
        api_base = provider["api_base"]
        # Truncate long URLs
        if len(api_base) > 40:
            api_base = api_base[:37] + "..."
        has_key = "✓" if provider["has_key"] else "✗"
        key_style = "green" if provider["has_key"] else "red"
        table.add_row(name, api_base, Text(has_key, style=key_style))

    console.print(table)
    console.print()


def _display_credentials_summary():
    """
    Display a compact 2-column summary of all configured credentials.
    API Keys on the left, OAuth credentials on the right.
    Handles cases where only one type exists or neither.
    """
    from rich.columns import Columns

    summary = _get_all_credentials_summary()
    api_keys = summary["api_keys"]
    oauth_creds = summary["oauth"]

    # Calculate totals
    total_api_keys = sum(len(keys) for keys in api_keys.values())
    total_oauth = sum(len(creds) for creds in oauth_creds.values() if creds)

    # Handle empty case
    if total_api_keys == 0 and total_oauth == 0:
        console.print("[dim]No credentials configured yet.[/dim]\n")
        return

    # Build API Keys table (left column)
    api_table = None
    if total_api_keys > 0:
        api_table = Table(
            title="API Keys", box=None, padding=(0, 1), title_style="bold cyan"
        )
        api_table.add_column("Provider", style="yellow", no_wrap=True)
        api_table.add_column("Count", style="green", justify="right")

        for provider, keys in sorted(api_keys.items()):
            api_table.add_row(provider, str(len(keys)))

        # Add total row
        api_table.add_row("─" * 12, "─" * 5, style="dim")
        api_table.add_row("Total", str(total_api_keys), style="bold")

    # Build OAuth table (right column)
    oauth_table = None
    if total_oauth > 0:
        oauth_table = Table(
            title="OAuth Credentials", box=None, padding=(0, 1), title_style="bold cyan"
        )
        oauth_table.add_column("Provider", style="yellow", no_wrap=True)
        oauth_table.add_column("Count", style="green", justify="right")
        oauth_table.add_column("Tiers", style="dim", no_wrap=True)

        for provider, creds in sorted(oauth_creds.items()):
            if not creds:
                continue
            display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
            count = len(creds)

            # Count and format tiers for providers that have tier info
            tier_counts = _count_tiers(creds)
            tier_str = _format_tier_counts(tier_counts)

            oauth_table.add_row(display_name, str(count), tier_str)

        # Add total row
        oauth_table.add_row("─" * 12, "─" * 5, "", style="dim")
        oauth_table.add_row("Total", str(total_oauth), "", style="bold")

    # Display based on what's available
    if api_table and oauth_table:
        # Both columns - use Columns for side-by-side layout
        console.print(Columns([api_table, oauth_table], padding=(0, 4), expand=False))
    elif api_table:
        # Only API keys
        console.print(api_table)
    elif oauth_table:
        # Only OAuth
        console.print(oauth_table)

    console.print("")  # Blank line after summary


def _display_oauth_providers_summary():
    """
    Display a compact summary of OAuth providers only (used when adding OAuth credentials).
    """
    oauth_summary = _get_oauth_credentials_summary()

    total = sum(len(creds) for creds in oauth_summary.values())

    # Build compact table
    table = Table(
        title="Current OAuth Credentials",
        box=None,
        padding=(0, 1),
        title_style="bold cyan",
    )
    table.add_column("Provider", style="yellow", no_wrap=True)
    table.add_column("Count", style="green", justify="right")

    for provider, creds in sorted(oauth_summary.items()):
        display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
        table.add_row(display_name, str(len(creds)))

    if total > 0:
        table.add_row("─" * 12, "─" * 5, style="dim")
        table.add_row("Total", str(total), style="bold")

    console.print(table)
    console.print("")


def _display_provider_credentials(provider_name: str):
    """
    Display all credentials for a specific OAuth provider.

    Args:
        provider_name: The provider key (e.g., "gemini_cli", "qwen_code")
    """
    _ensure_providers_loaded()

    try:
        auth_class = _get_provider_auth_class(provider_name)
        auth_instance = auth_class()
        credentials = auth_instance.list_credentials(_get_oauth_base_dir())
    except (httpx.HTTPError, ValueError, KeyError, OSError, json.JSONDecodeError) as e:  # non-critical: credential listing failed
        logging.debug("Credential listing failed for %s: %s", provider_name, e)
        credentials = []

    display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())

    if not credentials:
        console.print(f"\n[dim]No existing credentials for {display_name}[/dim]\n")
        return

    console.print(f"\n[bold cyan]Existing {display_name} Credentials:[/bold cyan]")

    table = Table(box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("File", style="yellow")
    table.add_column("Email/Identifier", style="cyan")

    # Add tier/project columns for Google OAuth providers
    if provider_name in _OAUTH_PROVIDERS:
        table.add_column("Tier", style="green")
        table.add_column("Project", style="dim")

    for i, cred in enumerate(credentials, 1):
        file_name = Path(cred["file_path"]).name
        email = cred.get("email", "unknown")

        if provider_name in _OAUTH_PROVIDERS:
            tier = cred.get("tier", "-")
            project = cred.get("project_id", "-")
            if project and len(project) > 20:
                project = project[:17] + "..."
            table.add_row(str(i), file_name, email, tier or "-", project or "-")
        else:
            table.add_row(str(i), file_name, email)

    console.print(table)
    console.print("")


async def _edit_oauth_credential_email(provider_name: str):
    """
    Edit the email field of an OAuth credential.

    Args:
        provider_name: The provider key (e.g., "qwen_code")
    """
    _ensure_providers_loaded()

    try:
        auth_class = _get_provider_auth_class(provider_name)
        auth_instance = auth_class()
        credentials = await asyncio.to_thread(auth_instance.list_credentials, _get_oauth_base_dir())
    except (httpx.HTTPError, ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error loading credentials: {e}[/bold red]")
        return

    display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())

    if not credentials:
        console.print(
            f"[bold yellow]No {display_name} credentials found.[/bold yellow]"
        )
        return

    # Display credentials for selection
    _display_provider_credentials(provider_name)

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Select credential to edit or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(credentials) + 1)] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        idx = int(choice) - 1
        cred_info = credentials[idx]
        cred_path = cred_info["file_path"]
        current_email = cred_info.get("email", "unknown")

        console.print(f"\nCurrent email: [cyan]{current_email}[/cyan]")
        new_email = Prompt.ask("Enter new email/identifier")

        if not new_email.strip():
            console.print("[bold yellow]No changes made (empty input).[/bold yellow]")
            return

        # Load and update the credential file
        creds = await asyncio.to_thread(_read_json_file, cred_path)

        if "_proxy_metadata" not in creds:
            creds["_proxy_metadata"] = {}

        old_email = creds["_proxy_metadata"].get("email")
        creds["_proxy_metadata"]["email"] = new_email.strip()

        # Save the updated credentials
        await asyncio.to_thread(_write_json_file, cred_path, creds)

        console.print(
            Panel(
                f"Email updated from [yellow]'{old_email}'[/yellow] to [green]'{new_email.strip()}'[/green]",
                style="bold green",
                title="Success",
                expand=False,
            )
        )

    except (httpx.HTTPError, ValueError, KeyError, OSError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error editing credential: {e}[/bold red]")


async def _view_api_keys_detail(provider_name: str):
    """Display detailed view of API keys for a specific provider."""
    clear_screen(f"View {provider_name} API Keys")

    api_keys = await asyncio.to_thread(_get_api_keys_from_env)
    keys = api_keys.get(provider_name, [])

    if not keys:
        console.print(
            f"[bold yellow]No API keys found for {provider_name}.[/bold yellow]"
        )
        console.print("\n[dim]Press Enter to go back...[/dim]")
        input()
        return

    # Display detailed table
    table = Table(title=f"{provider_name} API Keys", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=4)
    table.add_column("Key Name", style="yellow")
    table.add_column("Value (masked)", style="dim")

    for i, (key_name, key_value) in enumerate(keys, 1):
        masked = f"****{key_value[-4:]}" if len(key_value) > 4 else "****"
        table.add_row(str(i), key_name, masked)

    console.print(table)
    console.print(f"\n[dim]Total: {len(keys)} key(s)[/dim]")
    console.print("\n[dim]Press Enter to go back...[/dim]")
    input()


async def _view_oauth_credentials_detail(provider_name: str):
    """Display detailed view of OAuth credentials for a specific provider."""
    display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())
    clear_screen(f"View {display_name} Credentials")

    _ensure_providers_loaded()

    try:
        auth_class = _get_provider_auth_class(provider_name)
        auth_instance = auth_class()
        credentials = await asyncio.to_thread(auth_instance.list_credentials, _get_oauth_base_dir())
    except (httpx.HTTPError, ValueError, KeyError, OSError, json.JSONDecodeError) as e:  # non-critical: credential listing failed
        logging.debug("Credential listing failed for %s: %s", provider_name, e)
        credentials = []

    if not credentials:
        console.print(
            f"[bold yellow]No credentials found for {display_name}.[/bold yellow]"
        )
        console.print("\n[dim]Press Enter to go back...[/dim]")
        input()
        return

    # Display detailed table
    table = Table(title=f"{display_name} Credentials", box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="yellow")
    table.add_column("Email/Identifier", style="cyan")

    # Add tier/project columns for Google OAuth providers
    if provider_name in _OAUTH_PROVIDERS:
        table.add_column("Tier", style="green")
        table.add_column("Project", style="dim")

    for i, cred in enumerate(credentials, 1):
        file_name = Path(cred["file_path"]).name
        email = cred.get("email", "unknown")

        if provider_name in _OAUTH_PROVIDERS:
            tier = _normalize_tier_name(cred.get("tier")) if cred.get("tier") else "-"
            project = cred.get("project_id", "-")
            if project and len(project) > 25:
                project = project[:22] + "..."
            table.add_row(str(i), file_name, email, tier, project or "-")
        else:
            table.add_row(str(i), file_name, email)

    console.print(table)
    console.print(f"\n[dim]Total: {len(credentials)} credential(s)[/dim]")
    console.print("\n[dim]Press Enter to go back...[/dim]")
    input()
