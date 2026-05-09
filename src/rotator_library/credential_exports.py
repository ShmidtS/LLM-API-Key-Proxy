# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/credential_exports.py

import asyncio
import json
import logging
import time
import httpx
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from .utils.terminal_utils import clear_screen
from .credential_io import _get_oauth_base_dir, _read_json_file, _write_text_file
from .credential_providers import (
    OAUTH_FRIENDLY_NAMES,
    _ensure_providers_loaded,
    _get_provider_auth_class,
)

console = Console()


_INDIVIDUAL_EXPORT_PROVIDERS = {
    "gemini_cli": {
        "display_name": "Gemini CLI",
        "screen_title": "Export Gemini CLI Credential",
        "env_prefix": "GEMINI_CLI",
    },
    "qwen_code": {
        "display_name": "Qwen Code",
        "screen_title": "Export Qwen Code Credential",
        "env_prefix": "QWEN_CODE",
    },
    "iflow": {
        "display_name": "iFlow",
        "screen_title": "Export iFlow Credential",
        "env_prefix": "IFLOW",
    },
    "antigravity": {
        "display_name": "Antigravity",
        "screen_title": "Export Antigravity Credential",
        "env_prefix": "ANTIGRAVITY",
    },
}


def _get_provider_auth_instance(provider_name: str):
    _ensure_providers_loaded()
    auth_class = _get_provider_auth_class(provider_name)
    return auth_class()


async def _export_credentials_to_env(export_config: dict[str, str], source_getter):
    display_name = export_config["display_name"]
    clear_screen(export_config["screen_title"])

    auth_instance = source_getter()

    credentials = await asyncio.to_thread(auth_instance.list_credentials, _get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                f"No {display_name} credentials found. Please add one first using 'Add OAuth Credential'.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    cred_text = Text()
    for i, cred_info in enumerate(credentials):
        cred_text.append(
            f"  {i + 1}. {Path(cred_info['file_path']).name} ({cred_info['email']})\n"
        )

    console.print(
        Panel(
            cred_text,
            title=f"Available {display_name} Credentials",
            style="bold blue",
        )
    )

    choice = Prompt.ask(
        Text.from_markup(
            "[bold]Please select a credential to export or type [red]'b'[/red] to go back[/bold]"
        ),
        choices=[str(i + 1) for i in range(len(credentials))] + ["b"],
        show_choices=False,
    )

    if choice.lower() == "b":
        return

    try:
        choice_index = int(choice) - 1
        if 0 <= choice_index < len(credentials):
            cred_info = credentials[choice_index]
            env_path = auth_instance.export_credential_to_env(
                cred_info["file_path"], _get_oauth_base_dir()
            )

            if env_path:
                numbered_prefix = f"{export_config['env_prefix']}_{cred_info['number']}"
                success_text = Text.from_markup(
                    f"Successfully exported credential to [bold yellow]'{Path(env_path).name}'[/bold yellow]\n\n"
                    f"[bold]Environment variable prefix:[/bold] [cyan]{numbered_prefix}_*[/cyan]\n\n"
                    f"[bold]To use this credential:[/bold]\n"
                    f"1. Copy the contents to your main .env file, OR\n"
                    f"2. Source it: [bold cyan]source {Path(env_path).name}[/bold cyan] (Linux/Mac)\n"
                    f"3. Or on Windows PowerShell: [bold cyan]$env:VAR = \"value\"[/bold cyan] for each variable\n\n"
                    f"[bold]To combine multiple credentials:[/bold]\n"
                    f"Copy lines from multiple .env files into one file.\n"
                    f"Each credential uses a unique number ({numbered_prefix}_*)."
                )
                console.print(Panel(success_text, style="bold green", title="Success"))
            else:
                console.print(
                    Panel(
                        "Failed to export credential", style="bold red", title="Error"
                    )
                )
        else:
            console.print("[bold red]Invalid choice. Please try again.[/bold red]")
    except ValueError:
        console.print(
            "[bold red]Invalid input. Please enter a number or 'b'.[/bold red]"
        )
    except (
        httpx.HTTPError,
        httpx.TimeoutException,
        OSError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as e:
        console.print(
            Panel(
                f"An error occurred during export: {e}", style="bold red", title="Error"
            )
        )


async def export_gemini_cli_to_env():
    """Export a Gemini CLI credential JSON file to .env format."""
    await _export_credentials_to_env(
        _INDIVIDUAL_EXPORT_PROVIDERS["gemini_cli"],
        lambda: _get_provider_auth_instance("gemini_cli"),
    )


async def export_qwen_code_to_env():
    """Export a Qwen Code credential JSON file to .env format."""
    await _export_credentials_to_env(
        _INDIVIDUAL_EXPORT_PROVIDERS["qwen_code"],
        lambda: _get_provider_auth_instance("qwen_code"),
    )


async def export_iflow_to_env():
    """Export an iFlow credential JSON file to .env format."""
    await _export_credentials_to_env(
        _INDIVIDUAL_EXPORT_PROVIDERS["iflow"],
        lambda: _get_provider_auth_instance("iflow"),
    )


async def export_antigravity_to_env():
    """Export an Antigravity credential JSON file to .env format."""
    await _export_credentials_to_env(
        _INDIVIDUAL_EXPORT_PROVIDERS["antigravity"],
        lambda: _get_provider_auth_instance("antigravity"),
    )


async def export_all_provider_credentials(provider_name: str):
    """
    Export all credentials for a specific provider to individual .env files.
    Uses the auth class's list_credentials() and export_credential_to_env() methods.
    """
    display_name = provider_name.replace("_", " ").title()
    clear_screen(f"Export All {display_name} Credentials")
    # Get auth instance for this provider
    _ensure_providers_loaded()
    try:
        auth_class = _get_provider_auth_class(provider_name)
        auth_instance = auth_class()
    except (ValueError, TypeError, OSError) as e:  # non-critical: provider auth unavailable
        logging.debug("Provider auth instantiation failed for %s: %s", provider_name, e)
        console.print(f"[bold red]Unknown provider: {provider_name}[/bold red]")
        return

    display_name = provider_name.replace("_", " ").title()

    console.print(
        Panel(
            f"[bold cyan]Export All {display_name} Credentials[/bold cyan]",
            expand=False,
        )
    )

    # List all credentials using auth class
    credentials = await asyncio.to_thread(auth_instance.list_credentials, _get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                f"No {display_name} credentials found.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    exported_count = 0
    for cred_info in credentials:
        try:
            # Use auth class to export
            env_path = auth_instance.export_credential_to_env(
                cred_info["file_path"], _get_oauth_base_dir()
            )

            if env_path:
                console.print(
                    f"  ✓ Exported [cyan]{Path(cred_info['file_path']).name}[/cyan] → [yellow]{Path(env_path).name}[/yellow]"
                )
                exported_count += 1
            else:
                console.print(
                    f"  ✗ Failed to export {Path(cred_info['file_path']).name}"
                )

        except (
            httpx.HTTPError,
            httpx.TimeoutException,
            OSError,
            KeyError,
            TypeError,
            json.JSONDecodeError,
        ) as e:
            console.print(
                f"  ✗ Failed to export {Path(cred_info['file_path']).name}: {e}"
            )

    console.print(
        Panel(
            f"Successfully exported {exported_count}/{len(credentials)} {display_name} credentials to individual .env files.",
            style="bold green",
            title="Export Complete",
        )
    )


async def combine_provider_credentials(provider_name: str):
    """
    Combine all credentials for a specific provider into a single .env file.
    Uses the auth class's list_credentials() and build_env_lines() methods.
    """
    display_name = provider_name.replace("_", " ").title()
    clear_screen(f"Combine {display_name} Credentials")
    # Get auth instance for this provider
    _ensure_providers_loaded()
    try:
        auth_class = _get_provider_auth_class(provider_name)
        auth_instance = auth_class()
    except (ValueError, TypeError, OSError) as e:  # non-critical: provider auth unavailable
        logging.debug("Provider auth instantiation failed for %s: %s", provider_name, e)
        console.print(f"[bold red]Unknown provider: {provider_name}[/bold red]")
        return

    display_name = provider_name.replace("_", " ").title()

    console.print(
        Panel(
            f"[bold cyan]Combine All {display_name} Credentials[/bold cyan]",
            expand=False,
        )
    )

    # List all credentials using auth class
    credentials = await asyncio.to_thread(auth_instance.list_credentials, _get_oauth_base_dir())

    if not credentials:
        console.print(
            Panel(
                f"No {display_name} credentials found.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    combined_lines = [
        f"# Combined {display_name} Credentials",
        f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total credentials: {len(credentials)}",
        "#",
        "# Copy all lines below into your main .env file",
        "",
    ]

    combined_count = 0
    for cred_info in credentials:
        try:
            # Load credential file
            creds = await asyncio.to_thread(_read_json_file, cred_info["file_path"])

            # Use auth class to build env lines
            env_lines = auth_instance.build_env_lines(creds, cred_info["number"])

            combined_lines.extend(env_lines)
            combined_lines.append("")  # Blank line between credentials
            combined_count += 1

        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            console.print(
                f"  ✗ Failed to process {Path(cred_info['file_path']).name}: {e}"
            )

    # Write combined file
    combined_filename = f"{provider_name}_all_combined.env"
    combined_filepath = _get_oauth_base_dir() / combined_filename

    await asyncio.to_thread(_write_text_file, combined_filepath, "\n".join(combined_lines))

    console.print(
        Panel(
            Text.from_markup(
                f"Successfully combined {combined_count} {display_name} credentials into:\n"
                f"[bold yellow]{combined_filepath}[/bold yellow]\n\n"
                f"[bold]To use:[/bold] Copy the contents into your main .env file."
            ),
            style="bold green",
            title="Combine Complete",
        )
    )


async def combine_all_credentials():
    """
    Combine ALL credentials from ALL providers into a single .env file.
    Uses auth class list_credentials() and build_env_lines() methods.
    """
    clear_screen("Combine All Credentials")

    # List of providers that support OAuth credentials
    oauth_providers = ["gemini_cli", "qwen_code", "iflow", "antigravity"]

    _ensure_providers_loaded()

    combined_lines = [
        "# Combined All Provider Credentials",
        f"# Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "#",
        "# Copy all lines below into your main .env file",
        "",
    ]

    total_count = 0
    provider_counts = {}

    for provider_name in oauth_providers:
        try:
            auth_class = _get_provider_auth_class(provider_name)
            auth_instance = auth_class()
        except (ValueError, TypeError, OSError) as e:  # non-critical: provider auth unavailable
            logging.debug("Provider auth instantiation skipped for %s: %s", provider_name, e)
            continue  # Skip providers that don't have auth classes

        credentials = await asyncio.to_thread(auth_instance.list_credentials, _get_oauth_base_dir())

        if not credentials:
            continue

        display_name = provider_name.replace("_", " ").title()
        combined_lines.append(f"# ===== {display_name} Credentials =====")
        combined_lines.append("")

        provider_count = 0
        for cred_info in credentials:
            try:
                # Load credential file
                creds = await asyncio.to_thread(_read_json_file, cred_info["file_path"])

                # Use auth class to build env lines
                env_lines = auth_instance.build_env_lines(creds, cred_info["number"])

                combined_lines.extend(env_lines)
                combined_lines.append("")
                provider_count += 1
                total_count += 1

            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                console.print(
                    f"  ✗ Failed to process {Path(cred_info['file_path']).name}: {e}"
                )

        provider_counts[display_name] = provider_count

    if total_count == 0:
        console.print(
            Panel(
                "No credentials found to combine.",
                style="bold red",
                title="No Credentials",
            )
        )
        return

    # Write combined file
    combined_filename = "all_providers_combined.env"
    combined_filepath = _get_oauth_base_dir() / combined_filename

    await asyncio.to_thread(_write_text_file, combined_filepath, "\n".join(combined_lines))

    # Build summary
    summary_lines = [
        f"  • {name}: {count} credential(s)" for name, count in provider_counts.items()
    ]
    summary = "\n".join(summary_lines)

    console.print(
        Panel(
            Text.from_markup(
                f"Successfully combined {total_count} credentials from {len(provider_counts)} providers:\n"
                f"{summary}\n\n"
                f"[bold]Output file:[/bold] [yellow]{combined_filepath}[/yellow]\n\n"
                f"[bold]To use:[/bold] Copy the contents into your main .env file."
            ),
            style="bold green",
            title="Combine Complete",
        )
    )


async def export_credentials_submenu():
    """
    Submenu for credential export options.
    """
    while True:
        clear_screen("Export Credentials")

        console.print(
            Panel(
                Text.from_markup(
                    "[bold]Individual Exports:[/bold]\n"
                    "1. Export Gemini CLI credential\n"
                    "2. Export Qwen Code credential\n"
                    "3. Export iFlow credential\n"
                    "4. Export Antigravity credential\n"
                    "\n"
                    "[bold]Bulk Exports (per provider):[/bold]\n"
                    "5. Export ALL Gemini CLI credentials\n"
                    "6. Export ALL Qwen Code credentials\n"
                    "7. Export ALL iFlow credentials\n"
                    "8. Export ALL Antigravity credentials\n"
                    "\n"
                    "[bold]Combine Credentials:[/bold]\n"
                    "9. Combine all Gemini CLI into one file\n"
                    "10. Combine all Qwen Code into one file\n"
                    "11. Combine all iFlow into one file\n"
                    "12. Combine all Antigravity into one file\n"
                    "13. Combine ALL providers into one file"
                ),
                title="Choose export option",
                style="bold blue",
            )
        )

        export_choice = Prompt.ask(
            Text.from_markup(
                "[bold]Please select an option or type [red]'b'[/red] to go back[/bold]"
            ),
            choices=[
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "11",
                "12",
                "13",
                "b",
            ],
            show_choices=False,
        )

        if export_choice.lower() == "b":
            break

        # Individual exports
        if export_choice == "1":
            await export_gemini_cli_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "2":
            await export_qwen_code_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "3":
            await export_iflow_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "4":
            await export_antigravity_to_env()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        # Bulk exports (all credentials for a provider)
        elif export_choice == "5":
            await export_all_provider_credentials("gemini_cli")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "6":
            await export_all_provider_credentials("qwen_code")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "7":
            await export_all_provider_credentials("iflow")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "8":
            await export_all_provider_credentials("antigravity")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        # Combine per provider
        elif export_choice == "9":
            await combine_provider_credentials("gemini_cli")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "10":
            await combine_provider_credentials("qwen_code")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "11":
            await combine_provider_credentials("iflow")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        elif export_choice == "12":
            await combine_provider_credentials("antigravity")
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
        # Combine all providers
        elif export_choice == "13":
            await combine_all_credentials()
            console.print("\n[dim]Press Enter to return to export menu...[/dim]")
            input()
