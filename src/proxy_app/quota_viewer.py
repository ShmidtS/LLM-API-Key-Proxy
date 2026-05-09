# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Lightweight Quota Stats Viewer TUI.

Connects to a running proxy to display quota and usage statistics.
Uses only httpx + rich (no heavy rotator_library imports).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .quota_formatters import (
    format_tokens,
    format_cost,
    format_time_ago,
    format_reset_time,
    create_progress_bar,
    is_full_url,
    format_cooldown,
    natural_sort_key,
)
from .quota_api_client import QuotaApiClient
from .quota_viewer_config import QuotaViewerConfig
from rotator_library.utils.terminal_utils import clear_screen
from proxy_app.ui_constants import SEPARATOR_78


class QuotaRenderer:
    """Rendering helpers for quota stats screens."""

    def __init__(self, console: Console):
        self.console = console

    def render_credential_panel(
        self, idx: int, cred: Dict[str, Any], provider: str, view_mode: str
    ) -> None:
        """Render a single credential as a panel."""
        identifier = cred.get("identifier", f"credential {idx}")
        email = cred.get("email")
        tier = cred.get("tier", "")
        status = cred.get("status", "unknown")

        key_cooldown = cred.get("key_cooldown_remaining")
        model_cooldowns = cred.get("model_cooldowns", {})
        has_cooldown = key_cooldown or model_cooldowns

        if status == "exhausted":
            status_icon = "[red]⛔ Exhausted[/red]"
        elif status == "cooldown" or has_cooldown:
            if key_cooldown:
                status_icon = f"[yellow]⚠️ Cooldown ({format_cooldown(int(key_cooldown))})[/yellow]"
            else:
                status_icon = "[yellow]⚠️ Cooldown[/yellow]"
        else:
            status_icon = "[green]✅ Active[/green]"

        display_name = email if email else identifier
        tier_str = f" ({tier})" if tier else ""
        header = f"[{idx}] {display_name}{tier_str} {status_icon}"

        if view_mode == "global":
            stats_source = cred.get("global", cred)
        else:
            stats_source = cred

        last_used = format_time_ago(cred.get("last_used_ts"))
        requests = stats_source.get("requests", 0)
        tokens = stats_source.get("tokens", {})
        input_total = tokens.get("input_cached", 0) + tokens.get("input_uncached", 0)
        output = tokens.get("output", 0)
        cost = format_cost(stats_source.get("approx_cost"))

        stats_line = (
            f"Last used: {last_used} | Requests: {requests} | "
            f"Tokens: {format_tokens(input_total)}/{format_tokens(output)}"
        )
        if cost != "-":
            stats_line += f" | Cost: {cost}"

        content_lines = [
            f"[dim]{stats_line}[/dim]",
        ]

        model_groups = cred.get("model_groups", {})

        if model_cooldowns:
            if model_groups:
                group_cooldowns: Dict[str, int] = {}
                ungrouped_cooldowns: List[Tuple[str, int]] = []

                for model_name, cooldown_info in model_cooldowns.items():
                    remaining = cooldown_info.get("remaining_seconds", 0)
                    if remaining <= 0:
                        continue

                    clean_model = model_name.split("/")[-1]
                    found_group = None
                    for group_name, group_info in model_groups.items():
                        group_models = group_info.get("models", [])
                        if clean_model in group_models:
                            found_group = group_name
                            break

                    if found_group:
                        group_cooldowns[found_group] = max(
                            group_cooldowns.get(found_group, 0), remaining
                        )
                    else:
                        ungrouped_cooldowns.append((model_name, remaining))

                if group_cooldowns or ungrouped_cooldowns:
                    content_lines.append("")
                    content_lines.append("[yellow]Active Cooldowns:[/yellow]")

                    for group_name in sorted(group_cooldowns.keys()):
                        remaining = group_cooldowns[group_name]
                        content_lines.append(
                            f"  [yellow]⏱️ {group_name}: {format_cooldown(remaining)}[/yellow]"
                        )

                    for model_name, remaining in ungrouped_cooldowns:
                        short_model = model_name.split("/")[-1][:35]
                        content_lines.append(
                            f"  [yellow]⏱️ {short_model}: {format_cooldown(remaining)}[/yellow]"
                        )
            else:
                content_lines.append("")
                content_lines.append("[yellow]Active Cooldowns:[/yellow]")
                for model_name, cooldown_info in model_cooldowns.items():
                    remaining = cooldown_info.get("remaining_seconds", 0)
                    if remaining > 0:
                        short_model = model_name.split("/")[-1][:35]
                        content_lines.append(
                            f"  [yellow]⏱️ {short_model}: {format_cooldown(int(remaining))}[/yellow]"
                        )

        if model_groups:
            content_lines.append("")
            for group_name, group_stats in model_groups.items():
                remaining_pct = group_stats.get("remaining_pct")
                requests_used = group_stats.get("requests_used", 0)
                requests_max = group_stats.get("requests_max")
                requests_remaining = group_stats.get("requests_remaining")
                is_exhausted = group_stats.get("is_exhausted", False)
                reset_time = format_reset_time(group_stats.get("reset_time_iso"))
                confidence = group_stats.get("confidence", "low")

                if requests_remaining is None and requests_max:
                    requests_remaining = max(0, requests_max - requests_used)
                display = group_stats.get(
                    "display", f"{requests_remaining or 0}/{requests_max or '?'}")
                bar = create_progress_bar(remaining_pct)

                has_reset_time = reset_time and reset_time != "-"

                if is_exhausted:
                    color = "red"
                    if has_reset_time:
                        status_text = f"⛔ Resets: {reset_time}"
                    else:
                        status_text = "⛔ EXHAUSTED"
                elif remaining_pct is not None and remaining_pct < 20:
                    color = "yellow"
                    if has_reset_time:
                        status_text = f"⚠️ Resets: {reset_time}"
                    else:
                        status_text = "⚠️ LOW"
                else:
                    color = "green"
                    if has_reset_time:
                        status_text = f"Resets: {reset_time}"
                    else:
                        status_text = ""

                conf_indicator = ""
                if confidence == "low":
                    conf_indicator = " [dim](~)[/dim]"
                elif confidence == "medium":
                    conf_indicator = " [dim](?)[/dim]"

                pct_str = f"{remaining_pct}%" if remaining_pct is not None else "?%"
                content_lines.append(
                    f"  [{color}]{group_name:<18} {display:<10} {pct_str:>4} {bar}[/{color}]  {status_text}{conf_indicator}"
                )
        else:
            models = cred.get("models", {})
            if models:
                content_lines.append("")
                content_lines.append("  [dim]Models used:[/dim]")
                for model_name, model_stats in models.items():
                    req_count = model_stats.get("success_count", 0)
                    model_cost = format_cost(model_stats.get("approx_cost"))
                    short_name = model_name.split("/")[-1][:30]
                    content_lines.append(
                        f"    {short_name}: {req_count} requests, {model_cost}"
                    )

        self.console.print(
            Panel(
                "\n".join(content_lines),
                title=header,
                title_align="left",
                border_style="dim",
                expand=True,
            )
        )


class QuotaViewer:
    """Main Quota Viewer TUI class."""

    def __init__(self, config: Optional[QuotaViewerConfig] = None):
        """
        Initialize the viewer.

        Args:
            config: Optional config object. If not provided, one will be created.
        """
        self.console = Console()
        self.config = config or QuotaViewerConfig()
        self.config.sync_with_launcher_config()
        self.api_client = QuotaApiClient()
        self.renderer = QuotaRenderer(self.console)

        self._current_remote: Optional[Dict[str, Any]] = None
        self._cached_stats: Optional[Dict[str, Any]] = None
        self._last_error: Optional[str] = None
        self.running = True
        self.view_mode = "current"  # "current" or "global"

    @property
    def current_remote(self) -> Optional[Dict[str, Any]]:
        """Current remote proxy config."""
        return self.api_client.current_remote

    @current_remote.setter
    def current_remote(self, value: Optional[Dict[str, Any]]) -> None:
        self.api_client.current_remote = value

    @property
    def cached_stats(self) -> Optional[Dict[str, Any]]:
        """Cached quota stats."""
        return self.api_client.cached_stats

    @cached_stats.setter
    def cached_stats(self, value: Optional[Dict[str, Any]]) -> None:
        self.api_client.cached_stats = value

    @property
    def last_error(self) -> Optional[str]:
        """Last API error message."""
        return self.api_client.last_error

    @last_error.setter
    def last_error(self, value: Optional[str]) -> None:
        self.api_client.last_error = value

    def check_connection(
        self, remote: Dict[str, Any], timeout: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Check if a remote proxy is reachable."""
        return self.api_client.check_connection(remote, timeout)

    def fetch_stats(self, provider: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetch quota stats from the current remote."""
        return self.api_client.fetch_stats(provider)

    def post_action(
        self,
        action: str,
        scope: str = "all",
        provider: Optional[str] = None,
        credential: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Post a refresh action to the proxy."""
        return self.api_client.post_action(action, scope, provider, credential)

    # =========================================================================
    # DISPLAY SCREENS
    # =========================================================================

    def show_connection_error(self) -> str:
        """
        Display connection error screen with options to configure remotes.

        Returns:
            User choice: 's' (switch), 'm' (manage), 'r' (retry), 'b' (back/exit)
        """
        clear_screen()

        remote_name = (
            self.current_remote.get("name", "Unknown")
            if self.current_remote
            else "None"
        )
        remote_host = self.current_remote.get("host", "") if self.current_remote else ""
        remote_port = self.current_remote.get("port", "") if self.current_remote else ""

        # Format connection display - handle full URLs
        if is_full_url(remote_host):
            connection_display = remote_host
        elif remote_port:
            connection_display = f"{remote_host}:{remote_port}"
        else:
            connection_display = remote_host

        self.console.print(
            Panel(
                Text.from_markup(
                    "[bold red]Connection Error[/bold red]\n\n"
                    f"Remote: [bold]{remote_name}[/bold] ({connection_display})\n"
                    f"Error: {self.last_error or 'Unknown error'}\n\n"
                    "[bold]This tool requires the proxy to be running.[/bold]\n"
                    "Start the proxy first, or configure a different remote.\n\n"
                    "[dim]Tip: Select option 1 from the main menu to run the proxy.[/dim]"
                ),
                border_style="red",
                expand=False,
            )
        )

        self.console.print()
        self.console.print(SEPARATOR_78)
        self.console.print()
        self.console.print("   S. Switch to a different remote")
        self.console.print("   M. Manage remotes (add/edit/delete)")
        self.console.print("   R. Retry connection")
        self.console.print("   B. Back to main menu")
        self.console.print()
        self.console.print(SEPARATOR_78)

        choice = Prompt.ask("Select option", default="B").strip().lower()

        if choice in ("s", "m", "r", "b"):
            return choice
        return "b"  # Default to back for invalid input

    def show_summary_screen(self):
        """Display the main summary screen with all providers."""
        clear_screen()

        # Header
        remote_name = (
            self.current_remote.get("name", "Unknown")
            if self.current_remote
            else "None"
        )
        remote_host = self.current_remote.get("host", "") if self.current_remote else ""
        remote_port = self.current_remote.get("port", "") if self.current_remote else ""

        # Format connection display - handle full URLs
        if is_full_url(remote_host):
            connection_display = remote_host
        elif remote_port:
            connection_display = f"{remote_host}:{remote_port}"
        else:
            connection_display = remote_host

        # Calculate data age
        data_age = ""
        if self.cached_stats and self.cached_stats.get("timestamp"):
            age_seconds = int(time.time() - self.cached_stats["timestamp"])
            data_age = f"Data age: {age_seconds}s"

        # View mode indicator
        if self.view_mode == "global":
            view_label = "[magenta]📊 Global/Lifetime[/magenta]"
        else:
            view_label = "[cyan]📈 Current Period[/cyan]"

        self.console.print(SEPARATOR_78)
        self.console.print(
            f"[bold cyan]📈 Quota & Usage Statistics[/bold cyan]  |  {view_label}"
        )
        self.console.print(SEPARATOR_78)
        self.console.print(
            f"Connected to: [bold]{remote_name}[/bold] ({connection_display}) "
            f"[green]✅[/green] | {data_age}"
        )
        self.console.print()

        if not self.cached_stats:
            self.console.print("[yellow]No data available. Press R to reload.[/yellow]")
        else:
            # Build provider table
            table = Table(
                box=None, show_header=True, header_style="bold", padding=(0, 1)
            )
            table.add_column("Provider", style="cyan", min_width=10)
            table.add_column("Creds", justify="center", min_width=5)
            table.add_column("Quota Status", min_width=28)
            table.add_column("Requests", justify="right", min_width=8)
            table.add_column("Tokens (in/out)", min_width=20)
            table.add_column("Cost", justify="right", min_width=6)

            providers = self.cached_stats.get("providers", {})
            provider_list = list(providers.keys())

            for idx, (provider, prov_stats) in enumerate(providers.items(), 1):
                cred_count = prov_stats.get("credential_count", 0)

                # Use global stats if in global mode
                if self.view_mode == "global":
                    stats_source = prov_stats.get("global", prov_stats)
                    total_requests = stats_source.get("total_requests", 0)
                    tokens = stats_source.get("tokens", {})
                    cost_value = stats_source.get("approx_cost")
                else:
                    total_requests = prov_stats.get("total_requests", 0)
                    tokens = prov_stats.get("tokens", {})
                    cost_value = prov_stats.get("approx_cost")

                # Format tokens
                input_total = tokens.get("input_cached", 0) + tokens.get(
                    "input_uncached", 0
                )
                output = tokens.get("output", 0)
                cache_pct = tokens.get("input_cache_pct", 0)
                token_str = f"{format_tokens(input_total)}/{format_tokens(output)} ({cache_pct}% cached)"

                # Format cost
                cost_str = format_cost(cost_value)

                # Build quota status string (for providers with quota groups)
                quota_groups = prov_stats.get("quota_groups", {})
                if quota_groups:
                    quota_lines = []
                    for group_name, group_stats in quota_groups.items():
                        # Use remaining requests (not used) so percentage matches displayed value
                        total_remaining = group_stats.get("total_requests_remaining", 0)
                        total_max = group_stats.get("total_requests_max", 0)
                        total_pct = group_stats.get("total_remaining_pct")
                        tiers = group_stats.get("tiers", {})

                        # Format tier info: "5(15)f/2s" = 5 active out of 15 free, 2 standard all active
                        # Sort by priority (lower number = higher priority, appears first)
                        tier_parts = []
                        sorted_tiers = sorted(
                            tiers.items(), key=lambda x: x[1].get("priority", 10)
                        )
                        for tier_name, tier_info in sorted_tiers:
                            if tier_name == "unknown":
                                continue  # Skip unknown tiers in display
                            total_t = tier_info.get("total", 0)
                            active_t = tier_info.get("active", 0)
                            # Use first letter: standard-tier -> s, free-tier -> f
                            short = tier_name.replace("-tier", "")[0]

                            if active_t < total_t:
                                # Some exhausted - show active(total)
                                tier_parts.append(f"{active_t}({total_t}){short}")
                            else:
                                # All active - just show total
                                tier_parts.append(f"{total_t}{short}")
                        tier_str = "/".join(tier_parts) if tier_parts else ""

                        # Determine color based purely on remaining percentage
                        if total_pct is not None:
                            if total_pct <= 10:
                                color = "red"
                            elif total_pct < 30:
                                color = "yellow"
                            else:
                                color = "green"
                        else:
                            color = "dim"

                        bar = create_progress_bar(total_pct)
                        pct_str = f"{total_pct}%" if total_pct is not None else "?"

                        # Build status suffix (just tiers now, no outer parens)
                        status = tier_str

                        # Fixed-width format for aligned bars
                        # Adjust these to change column spacing:
                        QUOTA_NAME_WIDTH = 10  # name + colon, left-aligned
                        QUOTA_USAGE_WIDTH = (
                            12  # remaining/max ratio, right-aligned (handles 100k+)
                        )
                        display_name = group_name[: QUOTA_NAME_WIDTH - 1]
                        usage_str = f"{total_remaining}/{total_max}"
                        quota_lines.append(
                            f"[{color}]{display_name + ':':<{QUOTA_NAME_WIDTH}}{usage_str:>{QUOTA_USAGE_WIDTH}} {pct_str:>4} {bar}[/{color}] {status}"
                        )

                    # First line goes in the main row
                    first_quota = quota_lines[0] if quota_lines else "-"
                    table.add_row(
                        provider,
                        str(cred_count),
                        first_quota,
                        str(total_requests),
                        token_str,
                        cost_str,
                    )
                    # Additional quota lines as sub-rows
                    for quota_line in quota_lines[1:]:
                        table.add_row("", "", quota_line, "", "", "")
                else:
                    # No quota groups
                    table.add_row(
                        provider,
                        str(cred_count),
                        "-",
                        str(total_requests),
                        token_str,
                        cost_str,
                    )

                # Add separator between providers (except last)
                if idx < len(providers):
                    table.add_row(
                        "─" * 10, "─" * 4, "─" * 26, "─" * 7, "─" * 20, "─" * 6
                    )

            self.console.print(table)

            # Summary line - use global_summary if in global mode
            if self.view_mode == "global":
                summary = self.cached_stats.get(
                    "global_summary", self.cached_stats.get("summary", {})
                )
            else:
                summary = self.cached_stats.get("summary", {})

            total_creds = summary.get("total_credentials", 0)
            total_requests = summary.get("total_requests", 0)
            total_tokens = summary.get("tokens", {})
            total_input = total_tokens.get("input_cached", 0) + total_tokens.get(
                "input_uncached", 0
            )
            total_output = total_tokens.get("output", 0)
            total_cost = format_cost(summary.get("approx_total_cost"))

            self.console.print()
            self.console.print(
                f"[bold]Total:[/bold] {total_creds} credentials | "
                f"{total_requests} requests | "
                f"{format_tokens(total_input)}/{format_tokens(total_output)} tokens | "
                f"{total_cost} cost"
            )

        # Menu
        self.console.print()
        self.console.print(SEPARATOR_78)
        self.console.print()

        # Build provider menu options
        providers = self.cached_stats.get("providers", {}) if self.cached_stats else {}
        provider_list = list(providers.keys())

        for idx, provider in enumerate(provider_list, 1):
            self.console.print(f"   {idx}. View [cyan]{provider}[/cyan] details")

        self.console.print()
        self.console.print("   G. Toggle view mode (current/global)")
        self.console.print("   R. Reload all stats (re-read from proxy)")
        self.console.print("   S. Switch remote")
        self.console.print("   M. Manage remotes")
        self.console.print("   B. Back to main menu")
        self.console.print()
        self.console.print(SEPARATOR_78)

        # Get input
        valid_choices = [str(i) for i in range(1, len(provider_list) + 1)]
        valid_choices.extend(["r", "R", "s", "S", "m", "M", "b", "B", "g", "G"])

        choice = Prompt.ask("Select option", default="").strip()

        if choice.lower() == "b":
            self.running = False
        elif choice == "":
            # Empty input - just refresh the screen
            pass
        elif choice.lower() == "g":
            # Toggle view mode
            self.view_mode = "global" if self.view_mode == "current" else "current"
        elif choice.lower() == "r":
            with self.console.status("[bold]Reloading stats...", spinner="dots"):
                self.post_action("reload", scope="all")
        elif choice.lower() == "s":
            self.show_switch_remote_screen()
        elif choice.lower() == "m":
            self.show_manage_remotes_screen()
        elif choice.isdigit() and 1 <= int(choice) <= len(provider_list):
            provider = provider_list[int(choice) - 1]
            self.show_provider_detail_screen(provider)

    def show_provider_detail_screen(self, provider: str):
        """Display detailed stats for a specific provider."""
        while True:
            clear_screen()

            # View mode indicator
            if self.view_mode == "global":
                view_label = "[magenta]Global/Lifetime[/magenta]"
            else:
                view_label = "[cyan]Current Period[/cyan]"

            self.console.print(SEPARATOR_78)
            self.console.print(
                f"[bold cyan]📊 {provider.title()} - Detailed Stats[/bold cyan]  |  {view_label}"
            )
            self.console.print(SEPARATOR_78)
            self.console.print()

            if not self.cached_stats:
                self.console.print("[yellow]No data available.[/yellow]")
            else:
                prov_stats = self.cached_stats.get("providers", {}).get(provider, {})
                credentials = prov_stats.get("credentials", [])

                # Sort credentials naturally (1, 2, 10 not 1, 10, 2)
                credentials = sorted(credentials, key=natural_sort_key)

                if not credentials:
                    self.console.print(
                        "[dim]No credentials configured for this provider.[/dim]"
                    )
                else:
                    for idx, cred in enumerate(credentials, 1):
                        self.renderer.render_credential_panel(idx, cred, provider, self.view_mode)
                        self.console.print()

            # Menu
            self.console.print(SEPARATOR_78)
            self.console.print()
            self.console.print("   G.  Toggle view mode (current/global)")
            self.console.print("   R.  Reload stats (from proxy cache)")
            self.console.print("   RA. Reload all stats")

            # Force refresh options (only for providers that support it)
            has_quota_groups = bool(
                self.cached_stats
                and self.cached_stats.get("providers", {})
                .get(provider, {})
                .get("quota_groups")
            )

            if has_quota_groups:
                self.console.print()
                self.console.print(
                    f"   F.  [yellow]Force refresh ALL {provider} quotas from API[/yellow]"
                )
                credentials = (
                    self.cached_stats.get("providers", {})
                    .get(provider, {})
                    .get("credentials", [])
                    if self.cached_stats
                    else []
                )
                # Sort credentials naturally
                credentials = sorted(credentials, key=natural_sort_key)
                for idx, cred in enumerate(credentials, 1):
                    identifier = cred.get("identifier", f"credential {idx}")
                    email = cred.get("email", identifier)
                    self.console.print(
                        f"   F{idx}. Force refresh [{idx}] only ({email})"
                    )

            self.console.print()
            self.console.print("   B.  Back to summary")
            self.console.print()
            self.console.print(SEPARATOR_78)

            choice = Prompt.ask("Select option", default="B").strip().upper()

            if choice == "B":
                break
            elif choice == "G":
                # Toggle view mode
                self.view_mode = "global" if self.view_mode == "current" else "current"
            elif choice == "R":
                with self.console.status(
                    f"[bold]Reloading {provider} stats...", spinner="dots"
                ):
                    self.post_action("reload", scope="provider", provider=provider)
            elif choice == "RA":
                with self.console.status(
                    "[bold]Reloading all stats...", spinner="dots"
                ):
                    self.post_action("reload", scope="all")
            elif choice == "F" and has_quota_groups:
                result = None
                with self.console.status(
                    f"[bold]Fetching live quota for ALL {provider} credentials...",
                    spinner="dots",
                ):
                    result = self.post_action(
                        "force_refresh", scope="provider", provider=provider
                    )
                # Handle result OUTSIDE spinner
                if result and result.get("refresh_result"):
                    rr = result["refresh_result"]
                    self.console.print(
                        f"\n[green]Refreshed {rr.get('credentials_refreshed', 0)} credentials "
                        f"in {rr.get('duration_ms', 0)}ms[/green]"
                    )
                    if rr.get("errors"):
                        for err in rr["errors"]:
                            self.console.print(f"[red]  Error: {err}[/red]")
                    Prompt.ask("Press Enter to continue", default="")
            elif choice.startswith("F") and choice[1:].isdigit() and has_quota_groups:
                idx = int(choice[1:])
                credentials = (
                    self.cached_stats.get("providers", {})
                    .get(provider, {})
                    .get("credentials", [])
                    if self.cached_stats
                    else []
                )
                # Sort credentials naturally to match display order
                credentials = sorted(credentials, key=natural_sort_key)
                if 1 <= idx <= len(credentials):
                    cred = credentials[idx - 1]
                    cred_id = cred.get("identifier", "")
                    email = cred.get("email", cred_id)
                    result = None
                    with self.console.status(
                        f"[bold]Fetching live quota for {email}...", spinner="dots"
                    ):
                        result = self.post_action(
                            "force_refresh",
                            scope="credential",
                            provider=provider,
                            credential=cred_id,
                        )
                    # Handle result OUTSIDE spinner
                    if result and result.get("refresh_result"):
                        rr = result["refresh_result"]
                        self.console.print(
                            f"\n[green]Refreshed in {rr.get('duration_ms', 0)}ms[/green]"
                        )
                        if rr.get("errors"):
                            for err in rr["errors"]:
                                self.console.print(f"[red]  Error: {err}[/red]")
                        Prompt.ask("Press Enter to continue", default="")

    def show_switch_remote_screen(self):
        """Display remote selection screen."""
        clear_screen()

        self.console.print(SEPARATOR_78)
        self.console.print("[bold cyan]🔄 Switch Remote[/bold cyan]")
        self.console.print(SEPARATOR_78)
        self.console.print()

        current_name = self.current_remote.get("name") if self.current_remote else None
        self.console.print(f"Current: [bold]{current_name}[/bold]")
        self.console.print()
        self.console.print("Available remotes:")

        remotes = self.config.get_remotes()
        remote_status: List[Tuple[Dict, bool, str]] = []

        # Check status of all remotes
        with self.console.status("[dim]Checking remote status...", spinner="dots"):
            for remote in remotes:
                is_online, status_msg = self.check_connection(remote)
                remote_status.append((remote, is_online, status_msg))

        for idx, (remote, is_online, status_msg) in enumerate(remote_status, 1):
            name = remote.get("name", "Unknown")
            host = remote.get("host", "")
            port = remote.get("port", "")

            # Format connection display - handle full URLs
            if is_full_url(host):
                connection_display = host
            elif port:
                connection_display = f"{host}:{port}"
            else:
                connection_display = host

            is_current = name == current_name
            current_marker = " (current)" if is_current else ""

            if is_online:
                status_icon = "[green]✅ Online[/green]"
            else:
                status_icon = f"[red]⚠️ {status_msg}[/red]"

            self.console.print(
                f"   {idx}. {name:<20} {connection_display:<30} {status_icon}{current_marker}"
            )

        self.console.print()
        self.console.print(SEPARATOR_78)
        self.console.print()

        choice = Prompt.ask(
            f"Select remote (1-{len(remotes)}) or B to go back", default="B"
        ).strip()

        if choice.lower() == "b":
            return

        if choice.isdigit() and 1 <= int(choice) <= len(remotes):
            selected = remotes[int(choice) - 1]
            self.current_remote = selected
            self.config.set_last_used(selected["name"])
            self.cached_stats = None  # Clear cache

            # Try to fetch stats from new remote
            with self.console.status("[bold]Connecting...", spinner="dots"):
                stats = self.fetch_stats()
                if stats is None:
                    # Try with API key from .env for Local
                    if selected["name"] == "Local" and not selected.get("api_key"):
                        env_key = self.config.get_api_key_from_env()
                        current_remote = self.current_remote
                        if env_key and current_remote:
                            current_remote["api_key"] = env_key
                            stats = self.fetch_stats()

            if stats is None:
                self.show_api_key_prompt()

    def show_api_key_prompt(self):
        """Prompt for API key when authentication fails."""
        self.console.print()
        self.console.print(
            "[yellow]Authentication required or connection failed.[/yellow]"
        )
        self.console.print(f"Error: {self.last_error}")
        self.console.print()

        api_key = Prompt.ask(
            "Enter API key (or press Enter to cancel)", default=""
        ).strip()

        if api_key and self.current_remote:
            remote = self.current_remote
            remote["api_key"] = api_key
            # Update config with new API key
            self.config.update_remote(remote["name"], api_key=api_key)

            # Try again
            with self.console.status("[bold]Reconnecting...", spinner="dots"):
                if self.fetch_stats() is None:
                    self.console.print(f"[red]Still failed: {self.last_error}[/red]")
                    Prompt.ask("Press Enter to continue", default="")
        else:
            self.console.print("[dim]Cancelled.[/dim]")
            Prompt.ask("Press Enter to continue", default="")

    def show_manage_remotes_screen(self):
        """Display remote management screen."""
        while True:
            clear_screen()

            self.console.print(SEPARATOR_78)
            self.console.print("[bold cyan]⚙️ Manage Remotes[/bold cyan]")
            self.console.print(SEPARATOR_78)
            self.console.print()

            remotes = self.config.get_remotes()

            table = Table(box=None, show_header=True, header_style="bold")
            table.add_column("#", style="dim", width=3)
            table.add_column("Name", min_width=16)
            table.add_column("Host", min_width=24)
            table.add_column("Port", justify="right", width=6)
            table.add_column("Default", width=8)

            for idx, remote in enumerate(remotes, 1):
                is_default = "★" if remote.get("is_default") else ""
                table.add_row(
                    str(idx),
                    remote.get("name", ""),
                    remote.get("host", ""),
                    str(remote.get("port", 8000)),
                    is_default,
                )

            self.console.print(table)

            self.console.print()
            self.console.print(SEPARATOR_78)
            self.console.print()
            self.console.print("   A. Add new remote")
            self.console.print("   E. Edit remote (enter number, e.g., E1)")
            self.console.print("   D. Delete remote (enter number, e.g., D1)")
            self.console.print("   S. Set default remote")
            self.console.print("   B. Back")
            self.console.print()
            self.console.print(SEPARATOR_78)

            choice = Prompt.ask("Select option", default="B").strip().upper()

            if choice == "B":
                break
            elif choice == "A":
                self._add_remote_dialog()
            elif choice == "S":
                self._set_default_dialog(remotes)
            elif choice.startswith("E") and choice[1:].isdigit():
                idx = int(choice[1:])
                if 1 <= idx <= len(remotes):
                    self._edit_remote_dialog(remotes[idx - 1])
            elif choice.startswith("D") and choice[1:].isdigit():
                idx = int(choice[1:])
                if 1 <= idx <= len(remotes):
                    self._delete_remote_dialog(remotes[idx - 1])

    def _add_remote_dialog(self):
        """Dialog to add a new remote."""
        self.console.print()
        self.console.print("[bold]Add New Remote[/bold]")
        self.console.print(
            "[dim]For full URLs (e.g., https://api.example.com/v1), leave port empty[/dim]"
        )
        self.console.print()

        name = Prompt.ask("Name", default="").strip()
        if not name:
            self.console.print("[dim]Cancelled.[/dim]")
            return

        host = Prompt.ask("Host (or full URL)", default="").strip()
        if not host:
            self.console.print("[dim]Cancelled.[/dim]")
            return

        # For full URLs, default to empty port
        if is_full_url(host):
            port_default = ""
        else:
            port_default = "8000"

        port_str = Prompt.ask(
            "Port (empty for full URLs)", default=port_default
        ).strip()
        if port_str == "":
            port = ""
        else:
            try:
                port = int(port_str)
            except ValueError:
                port = 8000

        api_key = Prompt.ask("API Key (optional)", default="").strip() or None

        if self.config.add_remote(name, host, port, api_key):
            self.console.print(f"[green]Added remote '{name}'.[/green]")
        else:
            self.console.print(f"[red]Remote '{name}' already exists.[/red]")

        Prompt.ask("Press Enter to continue", default="")

    def _edit_remote_dialog(self, remote: Dict[str, Any]):
        """Dialog to edit an existing remote."""
        self.console.print()
        self.console.print(f"[bold]Edit Remote: {remote['name']}[/bold]")
        self.console.print(
            "[dim]Press Enter to keep current value. For full URLs, leave port empty.[/dim]"
        )
        self.console.print()

        new_name = Prompt.ask("Name", default=remote["name"]).strip()
        new_host = Prompt.ask(
            "Host (or full URL)", default=remote.get("host", "")
        ).strip()

        # Get current port, handle empty string
        current_port = remote.get("port", "")
        port_default = str(current_port) if current_port != "" else ""

        new_port_str = Prompt.ask(
            "Port (empty for full URLs)", default=port_default
        ).strip()
        if new_port_str == "":
            new_port = ""
        else:
            try:
                new_port = int(new_port_str)
            except ValueError:
                new_port = current_port if current_port != "" else 8000

        current_key = remote.get("api_key", "") or ""
        display_key = f"{current_key[:8]}..." if len(current_key) > 8 else current_key
        new_key = Prompt.ask(
            f"API Key (current: {display_key or 'none'})", default=""
        ).strip()

        updates = {}
        if new_name != remote["name"]:
            updates["new_name"] = new_name
        if new_host != remote.get("host"):
            updates["host"] = new_host
        if new_port != remote.get("port"):
            updates["port"] = new_port
        if new_key:
            updates["api_key"] = new_key

        if updates:
            if self.config.update_remote(remote["name"], **updates):
                self.console.print("[green]Remote updated.[/green]")
                # Update current_remote if it was the one being edited
                if (
                    self.current_remote
                    and self.current_remote["name"] == remote["name"]
                ):
                    self.current_remote.update(updates)
                    if "new_name" in updates:
                        self.current_remote["name"] = updates["new_name"]
            else:
                self.console.print("[red]Failed to update remote.[/red]")
        else:
            self.console.print("[dim]No changes made.[/dim]")

        Prompt.ask("Press Enter to continue", default="")

    def _delete_remote_dialog(self, remote: Dict[str, Any]):
        """Dialog to delete a remote."""
        self.console.print()
        self.console.print(f"[yellow]Delete remote '{remote['name']}'?[/yellow]")

        confirm = Prompt.ask("Type 'yes' to confirm", default="no").strip().lower()

        if confirm == "yes":
            if self.config.delete_remote(remote["name"]):
                self.console.print(f"[green]Deleted remote '{remote['name']}'.[/green]")
                # If deleted current remote, switch to another
                if (
                    self.current_remote
                    and self.current_remote["name"] == remote["name"]
                ):
                    self.current_remote = self.config.get_default_remote()
                    self.cached_stats = None
            else:
                self.console.print(
                    "[red]Cannot delete. At least one remote must exist.[/red]"
                )
        else:
            self.console.print("[dim]Cancelled.[/dim]")

        Prompt.ask("Press Enter to continue", default="")

    def _set_default_dialog(self, remotes: List[Dict[str, Any]]):
        """Dialog to set the default remote."""
        self.console.print()
        choice = Prompt.ask(f"Set default (1-{len(remotes)})", default="").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(remotes):
            remote = remotes[int(choice) - 1]
            if self.config.set_default_remote(remote["name"]):
                self.console.print(
                    f"[green]'{remote['name']}' is now the default.[/green]"
                )
            else:
                self.console.print("[red]Failed to set default.[/red]")
            Prompt.ask("Press Enter to continue", default="")

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """Main viewer loop."""
        # Get initial remote
        self.current_remote = self.config.get_last_used_remote()

        if not self.current_remote:
            self.console.print("[red]No remotes configured.[/red]")
            return

        # Connection loop - allows retry after configuring remotes
        while True:
            # For Local remote, try to get API key from .env if not set
            current_remote = self.current_remote
            if current_remote and current_remote["name"] == "Local" and not current_remote.get(
                "api_key"
            ):
                env_key = self.config.get_api_key_from_env()
                if env_key:
                    current_remote["api_key"] = env_key

            # Try to connect
            with self.console.status("[bold]Connecting to proxy...", spinner="dots"):
                stats = self.fetch_stats()

            if stats is not None:
                break  # Connected successfully

            # Connection failed - show error with options
            choice = self.show_connection_error()

            if choice == "b":
                return  # Exit to main menu
            elif choice == "s":
                self.show_switch_remote_screen()
            elif choice == "m":
                self.show_manage_remotes_screen()
            elif choice == "r":
                continue  # Retry connection

            # After switch/manage, refresh current_remote from config
            # (it may have been changed)
            if self.current_remote:
                updated = self.config.get_remote_by_name(self.current_remote["name"])
                if updated:
                    self.current_remote = updated

        # Main loop
        while self.running:
            self.show_summary_screen()


def run_quota_viewer():
    """Entry point for the quota viewer."""
    viewer = QuotaViewer()
    viewer.run()


if __name__ == "__main__":
    run_quota_viewer()
