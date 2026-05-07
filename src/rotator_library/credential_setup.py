# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/credential_setup.py

import asyncio
import json
import logging
import httpx
import secrets
from pathlib import Path
from dotenv import set_key

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.text import Text

from .utils.terminal_utils import clear_screen
from .provider_ui_config import LITELLM_PROVIDERS, PROVIDER_CATEGORIES
from .provider_routing_config import PROVIDER_BLACKLIST
from .litellm_providers import SCRAPED_PROVIDERS
from .credential_io import _get_env_file, _get_oauth_base_dir, _read_text_file
from .credential_providers import (
    OAUTH_FRIENDLY_NAMES,
    _ensure_providers_loaded,
    _get_provider_auth_class,
    _env_key_exists,
    _get_api_keys_from_env,
    _get_oauth_credentials_summary,
    _delete_api_key_from_env,
)
from .credential_display import (
    _display_credentials_summary,
    _display_custom_providers_summary,
    _display_oauth_providers_summary,
    _display_provider_credentials,
    _edit_oauth_credential_email,
)

console = Console()


def ensure_env_defaults():
    """
    Ensures the .env file exists and contains essential default values like PROXY_API_KEY.
    """
    from dotenv import get_key

    if not _get_env_file().is_file():
        _get_env_file().touch()
        console.print(
            f"Creating a new [bold yellow]{_get_env_file().name}[/bold yellow] file..."
        )

    # Check for PROXY_API_KEY, similar to setup_env.bat
    if get_key(str(_get_env_file()), "PROXY_API_KEY") is None:
        default_key = secrets.token_urlsafe(32)
        console.print(
            f"Adding generated [bold cyan]PROXY_API_KEY[/bold cyan] to [bold yellow]{_get_env_file().name}[/bold yellow]..."
        )
        set_key(str(_get_env_file()), "PROXY_API_KEY", default_key)


def _search_providers(query: str, providers: dict) -> list:
    """Search providers by substring match (case-insensitive).

    Searches both the provider key and display name.
    """
    query_lower = query.lower()
    matches = []
    for provider_key, config in providers.items():
        display_name = config.get("display_name", provider_key)
        if query_lower in provider_key.lower() or query_lower in display_name.lower():
            matches.append((provider_key, config))
    return matches


def _get_providers_by_category(providers: dict) -> dict:
    """Group providers by category."""
    by_category = {}
    for name, config in providers.items():
        category = config.get("category", "other")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((name, config))
    return by_category


async def setup_api_key():
    """
    Interactively sets up a new API key for a provider.
    Supports search, categorized display, and additional configuration variables.
    """
    clear_screen("Add API Key")

    # Show info panel
    console.print(
        Panel(
            Text.from_markup(
                "[bold]This list is powered by the LiteLLM library.[/bold]\n"
                "Some providers require additional configuration (API base URL, etc.)\n\n"
                "[dim]Full documentation: https://docs.litellm.ai/docs/providers[/dim]\n"
                "[dim]Note: Adding multiple API base URLs per provider is not yet supported.[/dim]"
            ),
            style="blue",
            title="Provider Information",
            expand=False,
        )
    )
    console.print()

    # -------------------------------------------------------------------------
    # Discover custom providers from project's provider registry
    # -------------------------------------------------------------------------
    _, PROVIDER_PLUGINS = _ensure_providers_loaded()
    from .providers import OpenAICompatibleProvider

    # Build a set of API key env vars already in SCRAPED_PROVIDERS
    litellm_api_keys = set()
    for info in SCRAPED_PROVIDERS.values():
        for api_key_var in info.get("api_key_env_vars", []):
            litellm_api_keys.add(api_key_var)

    # OAuth-only providers to exclude entirely from API key setup
    oauth_only_providers = {
        "gemini_cli",  # OAuth-only
        "antigravity",  # OAuth-only
        "qwen_code",  # OAuth is primary, don't advertise API key
        "iflow",  # OAuth is primary
    }

    # Base classes to exclude
    base_classes = {
        "openai_compatible",
    }

    # Create combined providers dict with scraped data + UI config
    # Key is the provider route key, value includes display_name, api_key, category, etc.
    all_providers = {}

    # Add all scraped providers with their UI config
    for provider_key in SCRAPED_PROVIDERS:
        # Skip blacklisted providers
        if provider_key in PROVIDER_BLACKLIST:
            continue

        scraped_info = SCRAPED_PROVIDERS[provider_key]
        ui_config = LITELLM_PROVIDERS.get(provider_key, {"category": "other"})

        # Skip providers without API keys (OAuth-only or no auth)
        api_key_vars = scraped_info.get("api_key_env_vars", [])
        if not api_key_vars:
            continue

        # Prefer *_API_KEY pattern, fall back to first
        api_key_var = None
        for var in api_key_vars:
            if var.endswith("_API_KEY"):
                api_key_var = var
                break
        if not api_key_var:
            api_key_var = api_key_vars[0]

        all_providers[provider_key] = {
            "display_name": scraped_info.get("display_name", provider_key),
            "api_key": api_key_var,
            "category": ui_config.get("category", "other"),
            "note": ui_config.get("note"),
            "extra_vars": ui_config.get("extra_vars", []),
        }

    # Add custom providers from PROVIDER_PLUGINS
    provider_plugins = PROVIDER_PLUGINS or {}
    for provider_key, provider_class in provider_plugins.items():
        # Skip OAuth-only providers
        if provider_key in oauth_only_providers:
            continue

        # Skip base classes
        if provider_key in base_classes:
            continue

        # Skip if already in scraped providers
        if provider_key in all_providers:
            continue

        # Check if this is a dynamic OpenAI-compatible provider
        try:
            is_dynamic = isinstance(provider_class, type) and issubclass(
                provider_class, OpenAICompatibleProvider
            )
        except TypeError:
            is_dynamic = False

        env_var = f"{provider_key.upper()}_API_KEY"

        # Skip if API key already covered
        if env_var in litellm_api_keys:
            continue

        display_name = provider_key.replace("_", " ").title()

        if is_dynamic:
            # Dynamic OpenAI-compatible provider uses _API_BASE pattern
            all_providers[provider_key] = {
                "display_name": display_name,
                "api_key": env_var,
                "category": "custom_openai",
                "note": "Custom OpenAI-compatible provider.",
                "extra_vars": [
                    (f"{provider_key.upper()}_API_BASE", "API Base URL", None),
                ],
            }
        else:
            # First-party file-based provider
            all_providers[provider_key] = {
                "display_name": display_name,
                "api_key": env_var,
                "category": "custom",
                "note": "First-party provider from the library.",
            }

    # Search prompt
    search_query = Prompt.ask(
        "[bold]Search providers[/bold] [dim](or press Enter to see all)[/dim]",
        default="",
    )

    # Build provider list based on search
    if search_query.strip():
        # Search mode
        matches = _search_providers(search_query, all_providers)
        if not matches:
            console.print(
                f"[bold yellow]No providers found matching '{search_query}'[/bold yellow]"
            )
            console.print("[dim]Press Enter to continue...[/dim]")
            input()
            return

        # Build numbered list from search results
        provider_list = []
        provider_text = Text()
        provider_text.append(
            f"\nMatching providers for '{search_query}':\n\n", style="bold cyan"
        )

        for i, (provider_key, config) in enumerate(matches, 1):
            provider_list.append((provider_key, config))
            display_name = config.get("display_name", provider_key)
            category = config.get("category", "other")
            category_label = next(
                (label for cat, label in PROVIDER_CATEGORIES if cat == category),
                "Other",
            )
            api_key_var = config.get("api_key")
            if api_key_var:
                key_prefix = (
                    api_key_var.replace("_API_KEY", "")
                    .replace("_TOKEN", "")
                    .replace("_", " ")
                )
                provider_text.append(
                    f"  {i}. {display_name} ({key_prefix}) ", style="white"
                )
            else:
                provider_text.append(f"  {i}. {display_name} ", style="white")
            provider_text.append(f"[{category_label}]\n", style="dim")

        console.print(provider_text)

    else:
        # Full categorized list mode
        by_category = _get_providers_by_category(all_providers)
        provider_list = []
        provider_text = Text()

        for category_key, category_label in PROVIDER_CATEGORIES:
            if category_key not in by_category:
                continue

            providers_in_cat = by_category[category_key]
            provider_text.append(f"\n--- {category_label} ---\n", style="bold cyan")

            for provider_key, config in providers_in_cat:
                idx = len(provider_list) + 1
                provider_list.append((provider_key, config))
                display_name = config.get("display_name", provider_key)
                api_key_var = config.get("api_key")
                if api_key_var:
                    key_prefix = (
                        api_key_var.replace("_API_KEY", "")
                        .replace("_TOKEN", "")
                        .replace("_", " ")
                    )
                    provider_text.append(f"  {idx}. {display_name} ({key_prefix})\n")
                else:
                    provider_text.append(
                        f"  {idx}. {display_name} [dim](no API key)[/dim]\n"
                    )

        console.print(provider_text)

    # Provider selection
    console.print()
    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Select a provider number or type [red]'b'[/red] to go back[/bold]"
        ),
        default="b",
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if choice_index < 0 or choice_index >= len(provider_list):
            console.print("[bold red]Invalid choice.[/bold red]")
            return

        provider_key, provider_config = provider_list[choice_index]
        display_name = provider_config.get("display_name", provider_key)
        api_key_var = provider_config.get("api_key")
        note = provider_config.get("note")
        extra_vars = provider_config.get("extra_vars", [])

        # Get additional info from scraped data
        scraped_info = SCRAPED_PROVIDERS.get(provider_key, {})
        route = scraped_info.get("route", "").rstrip("/")
        api_base_url = scraped_info.get("api_base_url")

        console.print()

        # Build and show provider info panel
        info_lines = []
        if route:
            info_lines.append(f"Route: [cyan]{route}/[/cyan]")
            info_lines.append(f"Example: [dim]{route}/model-name[/dim]")
        if api_base_url:
            info_lines.append(f"API Base: [dim]{api_base_url}[/dim]")
        if api_key_var:
            info_lines.append(f"Env Variable: [green]{api_key_var}[/green]")

        if info_lines:
            console.print(
                Panel(
                    "\n".join(info_lines),
                    title=f"[bold]{display_name}[/bold]",
                    expand=False,
                    border_style="blue",
                )
            )
            console.print()

        # Show provider note if exists
        if note:
            console.print(
                Panel(
                    note,
                    style="yellow",
                    title="Configuration Note",
                    expand=False,
                )
            )
            console.print()

        saved_vars = []

        # Prompt for API key (if provider has one)
        if api_key_var:
            api_key = Prompt.ask(
                f"[bold]Enter {api_key_var}[/bold] [dim](or press Enter to skip)[/dim]",
                default="",
            )

            if api_key.strip():
                # Find next available key index
                key_index = 1
                while True:
                    key_name = f"{api_key_var}_{key_index}"
                    if not await asyncio.to_thread(_env_key_exists, key_name):
                        break
                    key_index += 1

                key_name = f"{api_key_var}_{key_index}"
                await asyncio.to_thread(set_key, str(_get_env_file()), key_name, api_key.strip())
                saved_vars.append((key_name, api_key.strip()))

        # Prompt for extra variables
        if extra_vars:
            console.print("\n[bold]Additional configuration:[/bold]")
            for env_var_name, label, default_value in extra_vars:
                if default_value:
                    # Pre-fill with default
                    value = Prompt.ask(
                        f"  {label}",
                        default=default_value,
                    )
                else:
                    value = Prompt.ask(
                        f"  {label} [dim](or press Enter to skip)[/dim]",
                        default="",
                    )

                if value.strip():
                    await asyncio.to_thread(set_key, str(_get_env_file()), env_var_name, value.strip())
                    saved_vars.append((env_var_name, value.strip()))

        # Show success message
        if saved_vars:
            success_lines = [f"Successfully configured [bold]{display_name}[/bold]:\n"]
            for var_name, var_value in saved_vars:
                if len(var_value) > 8:
                    masked = f"{var_value[:4]}...{var_value[-4:]}"
                elif len(var_value) > 4:
                    masked = f"****{var_value[-4:]}"
                else:
                    masked = "****"
                success_lines.append(f"  [yellow]{var_name}[/yellow] = {masked}")

            console.print(
                Panel(
                    Text.from_markup("\n".join(success_lines)),
                    style="bold green",
                    title="Success",
                    expand=False,
                )
            )
        else:
            console.print("[dim]No values configured (all skipped).[/dim]")

        # Wait for user to read the result
        console.print("\n[dim]Press Enter to continue...[/dim]")
        input()

    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )
        console.print("\n[dim]Press Enter to continue...[/dim]")
        input()


async def setup_custom_openai_provider():
    """
    Interactively sets up a custom OpenAI-compatible provider.

    This adds a new provider that uses the standard OpenAI API format but points
    to a custom endpoint (LM Studio, Ollama, vLLM, custom server, etc.).
    """
    clear_screen("Add Custom OpenAI-Compatible Provider")

    # Show info panel
    console.print(
        Panel(
            Text.from_markup(
                "[bold]Custom OpenAI-Compatible Providers[/bold]\n\n"
                "Add a custom endpoint that uses the OpenAI API format.\n"
                "This works with: LM Studio, Ollama, vLLM, text-generation-webui, "
                "and other OpenAI-compatible servers.\n\n"
                "[dim]The library will automatically discover available models from your endpoint.[/dim]\n"
                "[dim]You can also override built-in providers (e.g., OPENAI) to route traffic elsewhere.[/dim]\n\n"
                "[yellow]Please consult the provider's documentation for the correct API base URL.[/yellow]"
            ),
            style="blue",
            title="Custom Provider Setup",
            expand=False,
        )
    )
    console.print()

    # Show existing custom providers
    _display_custom_providers_summary()

    # Prompt for provider name
    console.print("[dim]Provider name will be used for environment variables.[/dim]")
    console.print(
        "[dim]Use alphanumeric characters and underscores only (e.g., MY_LOCAL_LLM).[/dim]\n"
    )

    while True:
        provider_name = Prompt.ask(
            "[bold]Enter provider name[/bold] [dim](or 'b' to go back)[/dim]",
            default="",
        )

        if provider_name.lower() == "b" or not provider_name.strip():
            return

        provider_name = provider_name.strip().upper()

        # Validate name (alphanumeric + underscores only)
        import re

        if not re.match(r"^[A-Z][A-Z0-9_]*$", provider_name):
            console.print(
                "[bold red]Invalid name. Use letters, numbers, and underscores only. "
                "Must start with a letter.[/bold red]"
            )
            continue

        # Check for conflict with built-in LiteLLM providers
        conflict_provider = None
        for litellm_name, config in LITELLM_PROVIDERS.items():
            api_key_var = config.get("api_key", "")
            if api_key_var:
                # Extract prefix from API key var (e.g., OPENAI_API_KEY -> OPENAI)
                prefix = api_key_var.replace("_API_KEY", "").replace("_TOKEN", "")
                if prefix == provider_name:
                    conflict_provider = litellm_name
                    break

        if conflict_provider:
            console.print(
                f"\n[bold yellow]Warning:[/bold yellow] '{provider_name}' matches the built-in "
                f"'{conflict_provider}' provider."
            )
            console.print(
                "If you continue, requests to this provider will be routed to your custom endpoint "
                "instead of the official API.\n"
            )
            override_confirm = Prompt.ask(
                "[bold]Do you want to override the built-in provider?[/bold]",
                choices=["y", "n"],
                default="n",
            )
            if override_confirm.lower() != "y":
                continue

        break

    # Prompt for API Base URL (required)
    console.print()
    console.print("[dim]The API base URL is where requests will be sent.[/dim]")
    console.print(
        "[dim]Common examples: http://localhost:1234/v1, http://localhost:11434/v1[/dim]\n"
    )

    while True:
        api_base = Prompt.ask(
            "[bold]Enter API Base URL[/bold] [dim](required)[/dim]",
            default="",
        )

        if not api_base.strip():
            console.print("[bold red]API Base URL is required.[/bold red]")
            continue

        api_base = api_base.strip()

        # Validate URL format
        if not api_base.startswith(("http://", "https://")):
            console.print(
                "[bold red]Invalid URL. Must start with http:// or https://[/bold red]"
            )
            continue

        break

    # Prompt for API Key (required)
    console.print()
    console.print("[dim]Enter the API key for authentication.[/dim]")
    console.print(
        "[dim]If your server doesn't require authentication, enter any placeholder value.[/dim]\n"
    )

    while True:
        api_key = Prompt.ask(
            "[bold]Enter API Key[/bold] [dim](required)[/dim]",
            default="",
        )

        if not api_key.strip():
            console.print("[bold red]API Key is required.[/bold red]")
            continue

        api_key = api_key.strip()
        break

    # Save to .env file
    env_file = _get_env_file()

    # Save API Base URL
    api_base_var = f"{provider_name}_API_BASE"
    await asyncio.to_thread(set_key, str(env_file), api_base_var, api_base)

    # Save API Key (find next available index)
    api_key_var_base = f"{provider_name}_API_KEY"
    key_index = 1
    if env_file.is_file():
        content = await asyncio.to_thread(_read_text_file, env_file)
        while f"{api_key_var_base}_{key_index}=" in content:
            key_index += 1

    api_key_var = f"{api_key_var_base}_{key_index}"
    await asyncio.to_thread(set_key, str(env_file), api_key_var, api_key)

    # Mask the API key for display
    if len(api_key) > 8:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    elif len(api_key) > 4:
        masked_key = f"****{api_key[-4:]}"
    else:
        masked_key = "****"

    # Show success message
    console.print(
        Panel(
            Text.from_markup(
                f"Successfully configured custom provider [bold]{provider_name}[/bold]:\n\n"
                f"  [yellow]{api_base_var}[/yellow] = {api_base}\n"
                f"  [yellow]{api_key_var}[/yellow] = {masked_key}\n\n"
                "[dim]The library will automatically fetch available models from your endpoint.[/dim]\n"
                "[dim]Use launcher menu option 4 'List Available Models' to verify the setup.[/dim]"
            ),
            style="bold green",
            title="Success",
            expand=False,
        )
    )

    console.print("\n[dim]Press Enter to continue...[/dim]")
    input()


async def setup_new_credential(provider_name: str):
    """
    Interactively sets up a new OAuth credential for a given provider.

    Delegates all credential management logic to the auth class's setup_credential() method.
    """
    try:
        _ensure_providers_loaded()
        auth_class = _get_provider_auth_class(provider_name)
        auth_instance = auth_class()

        # Call the auth class's setup_credential() method which handles the entire flow:
        # - OAuth authentication
        # - Email extraction for deduplication
        # - File path determination (new or existing)
        # - Credential file saving
        # - Post-auth discovery (tier/project for Google OAuth providers)
        result = await auth_instance.setup_credential(_get_oauth_base_dir())

        if not result.success:
            console.print(
                Panel(
                    f"Credential setup failed: {result.error}",
                    style="bold red",
                    title="Error",
                )
            )
            return

        # Display success message with details
        if result.is_update:
            success_text = Text.from_markup(
                f"Successfully updated credential at [bold yellow]'{Path(result.file_path).name}'[/bold yellow] "
                f"for user [bold cyan]'{result.email}'[/bold cyan]."
            )
        else:
            success_text = Text.from_markup(
                f"Successfully created new credential at [bold yellow]'{Path(result.file_path).name}'[/bold yellow] "
                f"for user [bold cyan]'{result.email}'[/bold cyan]."
            )

        # Add tier/project info if available (Google OAuth providers)
        if hasattr(result, "tier") and result.tier:
            success_text.append(f"\nTier: {result.tier}")
        if hasattr(result, "project_id") and result.project_id:
            success_text.append(f"\nProject: {result.project_id}")

        console.print(Panel(success_text, style="bold green", title="Success"))

    except (
        httpx.HTTPError,
        httpx.TimeoutException,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as e:
        console.print(
            Panel(
                f"An error occurred during setup for {provider_name}: {e}",
                style="bold red",
                title="Error",
            )
        )


async def manage_credentials_submenu():
    """
    Submenu for viewing and managing all credentials (API keys and OAuth).
    Allows deletion of any credential and editing email for OAuth credentials.
    """
    while True:
        clear_screen("Manage Credentials")

        # Display full summary
        _display_credentials_summary()

        console.print(
            Panel(
                Text.from_markup(
                    "[bold]Actions:[/bold]\n"
                    "1. Delete an API Key\n"
                    "2. Delete an OAuth Credential\n"
                    "3. Edit OAuth Credential Email [dim](Qwen Code recommended)[/dim]"
                ),
                title="Choose action",
                style="bold blue",
            )
        )

        action = Prompt.ask(
            Text.from_markup(
                "[bold]Select an option or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=["1", "2", "3", "b"],
            show_choices=False,
        )

        if action.lower() == "b":
            break

        if action == "1":
            # Delete API Key
            await _delete_api_key_menu()
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

        elif action == "2":
            # Delete OAuth Credential
            await _delete_oauth_credential_menu()
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()

        elif action == "3":
            # Edit OAuth Credential Email
            await _edit_oauth_credential_menu()
            console.print("\n[dim]Press Enter to continue...[/dim]")
            input()


async def _delete_api_key_menu():
    """Menu for deleting an API key from the .env file."""
    clear_screen("Delete API Key")
    api_keys = await asyncio.to_thread(_get_api_keys_from_env)

    if not api_keys:
        console.print("[bold yellow]No API keys configured.[/bold yellow]")
        return

    # Build a flat list of all keys for selection
    all_keys = []
    console.print("\n[bold cyan]Configured API Keys:[/bold cyan]")

    table = Table(box=None, padding=(0, 2))
    table.add_column("#", style="dim", width=3)
    table.add_column("Key Name", style="yellow")
    table.add_column("Provider", style="cyan")
    table.add_column("Value", style="dim")

    idx = 1
    for provider, keys in sorted(api_keys.items()):
        for key_name, key_value in keys:
            masked = f"****{key_value[-4:]}" if len(key_value) > 4 else "****"
            table.add_row(str(idx), key_name, provider, masked)
            all_keys.append((key_name, key_value, provider))
            idx += 1

    console.print(table)

    choice = Prompt.ask(
        Text.from_markup(
            "\n[bold]Select API key to delete or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(all_keys) + 1)] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        idx = int(choice) - 1
        key_name, key_value, provider = all_keys[idx]

        # Confirmation prompt
        masked = f"****{key_value[-4:]}" if len(key_value) > 4 else "****"
        confirmed = Confirm.ask(
            f"[bold red]Delete[/bold red] [yellow]{key_name}[/yellow] ({masked})?"
        )

        if not confirmed:
            console.print("[dim]Deletion cancelled.[/dim]")
            return

        if await asyncio.to_thread(_delete_api_key_from_env, key_name):
            console.print(
                Panel(
                    f"Successfully deleted [yellow]{key_name}[/yellow]",
                    style="bold green",
                    title="Success",
                    expand=False,
                )
            )
        else:
            console.print(
                Panel(
                    f"Failed to delete [yellow]{key_name}[/yellow]",
                    style="bold red",
                    title="Error",
                    expand=False,
                )
            )

    except (ValueError, OSError) as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


async def _delete_oauth_credential_menu():
    """Menu for deleting an OAuth credential file."""
    clear_screen("Delete OAuth Credential")
    oauth_summary = await asyncio.to_thread(_get_oauth_credentials_summary)

    # Check if there are any credentials
    total = sum(len(creds) for creds in oauth_summary.values())
    if total == 0:
        console.print("[bold yellow]No OAuth credentials configured.[/bold yellow]")
        return

    # First, select provider
    console.print("\n[bold cyan]Select OAuth Provider:[/bold cyan]")

    providers_with_creds = [(p, c) for p, c in oauth_summary.items() if c]
    for i, (provider, creds) in enumerate(providers_with_creds, 1):
        display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
        console.print(f"  {i}. {display_name} ({len(creds)} credential(s))")

    provider_choice = Prompt.ask(
        Text.from_markup(
            "\n[bold]Select provider or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(providers_with_creds) + 1)] + ["b"],
        show_choices=False,
    )

    if provider_choice.lower() == "b":
        return

    try:
        provider_idx = int(provider_choice) - 1
        provider_name, credentials = providers_with_creds[provider_idx]
        display_name = OAUTH_FRIENDLY_NAMES.get(provider_name, provider_name.title())

        # Now select credential
        _display_provider_credentials(provider_name)

        cred_choice = Prompt.ask(
            Text.from_markup(
                "[bold]Select credential to delete or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=[str(i) for i in range(1, len(credentials) + 1)] + ["b"],
            show_choices=False,
        )

        if cred_choice.lower() == "b":
            return

        cred_idx = int(cred_choice) - 1
        cred_info = credentials[cred_idx]
        cred_path = cred_info["file_path"]
        email = cred_info.get("email", "unknown")

        # Confirmation prompt
        confirmed = Confirm.ask(
            f"[bold red]Delete[/bold red] credential for [cyan]{email}[/cyan] from {display_name}?"
        )

        if not confirmed:
            console.print("[dim]Deletion cancelled.[/dim]")
            return

        # Use the auth class's delete method
        _ensure_providers_loaded()
        auth_class = _get_provider_auth_class(provider_name)
        auth_instance = auth_class()

        if await asyncio.to_thread(auth_instance.delete_credential, cred_path):
            console.print(
                Panel(
                    f"Successfully deleted credential for [cyan]{email}[/cyan]",
                    style="bold green",
                    title="Success",
                    expand=False,
                )
            )
        else:
            console.print(
                Panel(
                    f"Failed to delete credential for [cyan]{email}[/cyan]",
                    style="bold red",
                    title="Error",
                    expand=False,
                )
            )

    except (ValueError, OSError) as e:
        console.print(f"[bold red]Error: {e}[/bold red]")


async def _edit_oauth_credential_menu():
    """Menu for editing an OAuth credential's email field."""
    clear_screen("Edit OAuth Credential")
    oauth_summary = await asyncio.to_thread(_get_oauth_credentials_summary)

    # Check if there are any credentials
    total = sum(len(creds) for creds in oauth_summary.values())
    if total == 0:
        console.print("[bold yellow]No OAuth credentials configured.[/bold yellow]")
        return

    # Show warning about editing
    console.print(
        Panel(
            Text.from_markup(
                "[bold yellow]Warning:[/bold yellow] Editing OAuth credentials is generally not recommended.\n"
                "This is mainly useful for [bold]Qwen Code[/bold] where you manually enter an email identifier.\n\n"
                "For Google OAuth providers (Gemini CLI, Antigravity), the email is automatically\n"
                "retrieved during authentication and changing it may cause confusion."
            ),
            style="yellow",
            title="Edit OAuth Credential",
            expand=False,
        )
    )

    # First, select provider
    console.print("\n[bold cyan]Select OAuth Provider:[/bold cyan]")

    providers_with_creds = [(p, c) for p, c in oauth_summary.items() if c]
    for i, (provider, creds) in enumerate(providers_with_creds, 1):
        display_name = OAUTH_FRIENDLY_NAMES.get(provider, provider.title())
        recommended = " [green](recommended)[/green]" if provider == "qwen_code" else ""
        console.print(
            f"  {i}. {display_name} ({len(creds)} credential(s)){recommended}"
        )

    provider_choice = Prompt.ask(
        Text.from_markup(
            "\n[bold]Select provider or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i) for i in range(1, len(providers_with_creds) + 1)] + ["b"],
        show_choices=False,
    )

    if provider_choice.lower() == "b":
        return

    try:
        provider_idx = int(provider_choice) - 1
        provider_name, _ = providers_with_creds[provider_idx]
        await _edit_oauth_credential_email(provider_name)

    except (ValueError, OSError) as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
