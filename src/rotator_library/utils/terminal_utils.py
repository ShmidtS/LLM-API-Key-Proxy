# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os

from rich.console import Console
from rich.panel import Panel

_console = Console()


def clear_screen(subtitle: str = ""):
    """
    Cross-platform terminal clear with optional header.

    Uses native OS commands instead of ANSI escape sequences:
    - Windows (conhost & Windows Terminal): cls
    - Unix-like systems (Linux, Mac): clear

    Args:
        subtitle: If provided, displays a header panel with this subtitle.
                  If empty/None, just clears the screen.
    """
    print("\033[2J\033[H", end="", flush=True)
    if subtitle:
        _console.print(
            Panel(
                f"[bold cyan]{subtitle}[/bold cyan]",
                title="--- API Key Proxy ---",
            )
        )
