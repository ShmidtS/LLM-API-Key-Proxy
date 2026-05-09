# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rotator_library.utils.terminal_utils import clear_screen

from proxy_app.bootstrap import bootstrap, configure_app
from rotator_library.utils.paths import get_data_file
from rotator_library.credential_tool import ensure_env_defaults, run_credential_tool

if TYPE_CHECKING:
    from fastapi import FastAPI


def needs_onboarding(env_file) -> bool:
    """Check if the proxy needs onboarding (first-time setup)."""
    return not env_file.is_file()


def show_onboarding_message() -> None:
    """Display clear explanatory message for why onboarding is needed."""
    clear_screen()
    console = Console()
    console.print(
        Panel.fit(
            "[bold cyan]LLM API Key Proxy - First Time Setup[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print("[bold yellow]Configuration Required[/bold yellow]\n")
    console.print("The proxy needs initial configuration:")
    console.print("  [red]No .env file found[/red]")
    console.print("\n[bold]Why this matters:[/bold]")
    console.print("  • The .env file stores your credentials and settings")
    console.print("  • PROXY_API_KEY protects your proxy from unauthorized access")
    console.print("  • Provider API keys enable LLM access")
    console.print("\n[bold]What happens next:[/bold]")
    console.print("  1. We'll create a .env file with PROXY_API_KEY")
    console.print("  2. You can add LLM provider credentials (API keys or OAuth)")
    console.print("  3. The proxy will then start normally")
    console.print(
        "\n[bold yellow]Note:[/bold yellow] The credential tool adds PROXY_API_KEY by default."
    )
    console.print("   You can remove it later if you want an unsecured proxy.\n")
    console.input("[bold green]Press Enter to launch the credential setup tool...[/bold green]")


def run_onboarding_if_needed(app: "FastAPI", parsed: argparse.Namespace) -> None:
    startup_state = getattr(parsed, "startup_state", None)
    if startup_state is None:
        startup_state = bootstrap(parsed)
        parsed.startup_state = startup_state

    configure_app(app, parsed)

    env_file = get_data_file(".env")

    if parsed.add_credential:
        ensure_env_defaults()
        if startup_state.load_dotenv is not None:
            startup_state.load_dotenv(env_file, override=True)
        run_credential_tool()
    elif needs_onboarding(env_file):
        show_onboarding_message()
        ensure_env_defaults()
        if startup_state.load_dotenv is not None:
            startup_state.load_dotenv(env_file, override=True)
        run_credential_tool()
        if startup_state.load_dotenv is not None:
            startup_state.load_dotenv(env_file, override=True)

        if needs_onboarding(env_file):
            console = Console()
            console.print("\n[bold red]Configuration incomplete.[/bold red]")
            console.print(
                "The proxy still cannot start. Please ensure PROXY_API_KEY is set in .env\n"
            )
            sys.exit(1)
