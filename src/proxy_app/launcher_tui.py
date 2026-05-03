# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Interactive TUI launcher for the LLM API Key Proxy.
Provides a beautiful Rich-based interface for configuration and execution.
"""

import os
import secrets
import sys

import orjson
import json
import logging

logger = logging.getLogger(__name__)
from pathlib import Path
from rich.console import Console
from rich.prompt import IntPrompt, Prompt
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv, set_key
from rotator_library.utils.paths import get_data_file
from rotator_library.utils.terminal_utils import clear_screen
from proxy_app.config import DEFAULT_HOST, DEFAULT_PORT, env_int

console = Console()

LauncherConfigValue = str | int | bool


class LauncherConfig:
    """Manages launcher_config.json (host, port, logging only)"""

    def __init__(self, config_path: Path = Path("launcher_config.json")):
        self.config_path = config_path
        self.defaults: dict[str, LauncherConfigValue] = {
            "host": DEFAULT_HOST,
            "port": DEFAULT_PORT,
            "enable_request_logging": False,
            "enable_raw_logging": False,
        }
        self.config: dict[str, LauncherConfigValue] = self.load()

    def load(self) -> dict[str, LauncherConfigValue]:
        """Load config from file or create with defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                # Merge with defaults for any missing keys
                for key, value in self.defaults.items():
                    if key not in config:
                        config[key] = value
                return config
            except (json.JSONDecodeError, IOError):
                return self.defaults.copy()
        return self.defaults.copy()

    def save(self):
        """Save current config to file."""
        import datetime

        self.config["last_updated"] = datetime.datetime.now().isoformat()
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)
        except IOError as e:
            console.print(f"[red]Error saving config: {e}[/red]")

    def update(self, **kwargs):
        """Update config values."""
        self.config.update(kwargs)
        self.save()

    @staticmethod
    def update_proxy_api_key(new_key: str):
        """Update PROXY_API_KEY in .env only"""
        env_file = get_data_file(".env")
        set_key(str(env_file), "PROXY_API_KEY", new_key)
        load_dotenv(dotenv_path=env_file, override=True)


class SettingsDetector:
    """Detects settings from .env for display"""

    @staticmethod
    def _load_local_env() -> dict[str, str]:
        """Load environment variables from local .env file only"""
        env_file = get_data_file(".env")
        env_dict: dict[str, str] = {}
        if not env_file.exists():
            return env_dict
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" in line:
                        key, _, value = line.partition("=")
                        key, value = key.strip(), value.strip()
                        if value and value[0] in ('"', "'") and value[-1] == value[0]:
                            value = value[1:-1]
                        env_dict[key] = value
        except (IOError, OSError):
            logger.debug("load_env_dict: failed to read .env file")
        return env_dict

    @staticmethod
    def get_all_settings() -> dict:
        """Returns comprehensive settings overview (includes provider_settings which triggers heavy imports)"""
        return {
            "credentials": SettingsDetector.detect_credentials(),
            "custom_bases": SettingsDetector.detect_custom_api_bases(),
            "model_definitions": SettingsDetector.detect_model_definitions(),
            "concurrency_limits": SettingsDetector.detect_concurrency_limits(),
            "model_filters": SettingsDetector.detect_model_filters(),
            "provider_settings": SettingsDetector.detect_provider_settings(),
        }

    @staticmethod
    def get_basic_settings() -> dict:
        """Returns basic settings overview without provider_settings (avoids heavy imports)"""
        return {
            "credentials": SettingsDetector.detect_credentials(),
            "custom_bases": SettingsDetector.detect_custom_api_bases(),
            "model_definitions": SettingsDetector.detect_model_definitions(),
            "concurrency_limits": SettingsDetector.detect_concurrency_limits(),
            "model_filters": SettingsDetector.detect_model_filters(),
        }

    @staticmethod
    def detect_credentials() -> dict:
        """Detect API keys and OAuth credentials"""
        import re
        from pathlib import Path

        providers = {}

        # Scan for API keys
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if "_API_KEY" in key and key != "PROXY_API_KEY":
                provider = key.split("_API_KEY")[0].lower()
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["api_keys"] += 1

        # Scan for file-based OAuth credentials
        oauth_dir = Path("oauth_creds")
        if oauth_dir.exists():
            for file in oauth_dir.glob("*_oauth_*.json"):
                provider = file.name.split("_oauth_")[0]
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["oauth"] += 1

        # Scan for env-based OAuth credentials
        # Maps provider name to the ENV_PREFIX used by the provider
        # (duplicated from credential_manager to avoid heavy imports)
        env_oauth_providers = {
            "gemini_cli": "GEMINI_CLI",
            "antigravity": "ANTIGRAVITY",
            "qwen_code": "QWEN_CODE",
            "iflow": "IFLOW",
        }

        for provider, env_prefix in env_oauth_providers.items():
            oauth_count = 0

            # Check numbered credentials (PROVIDER_N_ACCESS_TOKEN pattern)
            numbered_pattern = re.compile(rf"^{env_prefix}_(\d+)_ACCESS_TOKEN$")
            for key in env_vars.keys():
                match = numbered_pattern.match(key)
                if match:
                    index = match.group(1)
                    refresh_key = f"{env_prefix}_{index}_REFRESH_TOKEN"
                    if refresh_key in env_vars and env_vars[refresh_key]:
                        oauth_count += 1

            # Check legacy single credential (if no numbered found)
            if oauth_count == 0:
                access_key = f"{env_prefix}_ACCESS_TOKEN"
                refresh_key = f"{env_prefix}_REFRESH_TOKEN"
                if env_vars.get(access_key) and env_vars.get(refresh_key):
                    oauth_count = 1

            if oauth_count > 0:
                if provider not in providers:
                    providers[provider] = {"api_keys": 0, "oauth": 0, "custom": False}
                providers[provider]["oauth"] += oauth_count

        # Mark custom providers (have API_BASE set)
        for provider in providers:
            if os.getenv(f"{provider.upper()}_API_BASE"):
                providers[provider]["custom"] = True

        return providers

    @staticmethod
    def detect_custom_api_bases() -> dict:
        """Detect custom API base URLs (not in hardcoded map)"""
        from proxy_app.provider_urls import PROVIDER_URL_MAP

        bases = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.endswith("_API_BASE"):
                provider = key.replace("_API_BASE", "").lower()
                # Only include if NOT in hardcoded map
                if provider not in PROVIDER_URL_MAP:
                    bases[provider] = value
        return bases

    @staticmethod
    def detect_model_definitions() -> dict:
        """Detect provider model definitions"""
        models = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.endswith("_MODELS"):
                provider = key.replace("_MODELS", "").lower()
                try:
                    parsed = orjson.loads(value)
                    if isinstance(parsed, dict):
                        models[provider] = len(parsed)
                    elif isinstance(parsed, list):
                        models[provider] = len(parsed)
                except (json.JSONDecodeError, ValueError):
                    logger.debug("detect_model_counts: invalid JSON for provider %s", provider)
        return models

    @staticmethod
    def detect_concurrency_limits() -> dict:
        """Detect max concurrent requests per key"""
        limits = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
                provider = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
                try:
                    limits[provider] = int(value)
                except ValueError:
                    logger.debug("detect_concurrency_limits: invalid value for %s", provider)
        return limits

    @staticmethod
    def detect_model_filters() -> dict:
        """Detect active model filters (basic info only: defined or not)"""
        filters = {}
        env_vars = SettingsDetector._load_local_env()
        for key, value in env_vars.items():
            if key.startswith("IGNORE_MODELS_") or key.startswith("WHITELIST_MODELS_"):
                filter_type = "ignore" if key.startswith("IGNORE") else "whitelist"
                provider = key.replace(f"{filter_type.upper()}_MODELS_", "").lower()
                if provider not in filters:
                    filters[provider] = {"has_ignore": False, "has_whitelist": False}
                if filter_type == "ignore":
                    filters[provider]["has_ignore"] = True
                else:
                    filters[provider]["has_whitelist"] = True
        return filters

    @staticmethod
    def detect_provider_settings() -> dict:
        """Detect provider-specific settings (Antigravity, Gemini CLI)"""
        try:
            from proxy_app._provider_settings import PROVIDER_SETTINGS_MAP
        except ImportError:
            # Fallback for direct execution or testing
            from ._provider_settings import PROVIDER_SETTINGS_MAP

        provider_settings = {}
        env_vars = SettingsDetector._load_local_env()

        for provider, definitions in PROVIDER_SETTINGS_MAP.items():
            modified_count = 0
            for key, definition in definitions.items():
                env_value = env_vars.get(key)
                if env_value is not None:
                    # Check if value differs from default
                    default = definition.get("default")
                    setting_type = definition.get("type", "str")

                    try:
                        if setting_type == "bool":
                            current = env_value.lower() in ("true", "1", "yes")
                        elif setting_type == "int":
                            current = int(env_value)
                        else:
                            current = env_value

                        if current != default:
                            modified_count += 1
                    except (ValueError, AttributeError):
                        logger.debug("detect_modified_settings: failed to parse setting for %s", provider)

            if modified_count > 0:
                provider_settings[provider] = modified_count

        return provider_settings


class LauncherTUI:
    """Main launcher interface"""

    def __init__(self):
        self.console = Console()
        self.config = LauncherConfig()
        self.running = True
        self.env_file = get_data_file(".env")
        # Load .env file to ensure environment variables are available
        load_dotenv(dotenv_path=self.env_file, override=True)

    def needs_onboarding(self) -> bool:
        """Check if onboarding is needed"""
        return not self.env_file.exists() or not os.getenv("PROXY_API_KEY")

    def run(self):
        """Main TUI loop"""
        while self.running:
            self.show_main_menu()

    def show_main_menu(self):
        """Display main menu and handle selection"""
        clear_screen()
        settings = SettingsDetector.get_basic_settings()
        show_warning = self.needs_onboarding()

        self._show_main_header()
        self._show_main_warnings(show_warning)
        self._show_main_config_summary()
        self._show_main_status_summary(settings)
        self._show_main_menu_options(show_warning)
        self._handle_main_menu_choice(self._prompt_main_menu_choice())

    def _show_main_header(self):
        self.console.print(
            Panel.fit(
                "[bold cyan]🚀 LLM API Key Proxy - Interactive Launcher[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print(
            "[dim]GitHub: [blue underline]https://github.com/ShmidtS/LLM-API-Key-Proxy[/blue underline][/dim]"
        )

    def _show_main_warnings(self, show_warning: bool):
        if show_warning:
            self._show_initial_setup_warning()
        elif not os.getenv("PROXY_API_KEY"):
            self._show_proxy_key_security_warning()

    def _show_initial_setup_warning(self):
        self.console.print()
        self.console.print(
            Panel(
                Text.from_markup(
                    "⚠️  [bold yellow]INITIAL SETUP REQUIRED[/bold yellow]\n\n"
                    "The proxy needs initial configuration:\n"
                    "  ❌ No .env file found\n\n"
                    "Why this matters:\n"
                    "  • The .env file stores your credentials and settings\n"
                    "  • PROXY_API_KEY protects your proxy from unauthorized access\n"
                    "  • Provider API keys enable LLM access\n\n"
                    "What to do:\n"
                    '  1. Select option "3. Manage Credentials" to launch the credential tool\n'
                    "  2. The tool will create .env and set up PROXY_API_KEY automatically\n"
                    "  3. You can add provider credentials (API keys or OAuth)\n\n"
                    "⚠️  Note: The credential tool adds PROXY_API_KEY by default.\n"
                    "   You can remove it later if you want an unsecured proxy."
                ),
                border_style="yellow",
                expand=False,
            )
        )

    def _show_proxy_key_security_warning(self):
        self.console.print()
        self.console.print(
            Panel(
                Text.from_markup(
                    "⚠️  [bold red]SECURITY WARNING: PROXY_API_KEY Not Set[/bold red]\n\n"
                    "Your proxy is currently UNSECURED!\n"
                    "Anyone can access it without authentication.\n\n"
                    "This is a serious security risk if your proxy is accessible\n"
                    "from the internet or untrusted networks.\n\n"
                    "👉 [bold]Recommended:[/bold] Set PROXY_API_KEY in .env file\n"
                    '   Use option "2. Configure Proxy Settings" → "3. Set Proxy API Key"\n'
                    '   or option "3. Manage Credentials"'
                ),
                border_style="red",
                expand=False,
            )
        )

    def _show_main_config_summary(self):
        self.console.print()
        self.console.print("[bold]📋 Proxy Configuration[/bold]")
        self.console.print("━" * 70)
        self.console.print(f"   Host:                {self.config.config['host']}")
        self.console.print(f"   Port:                {self.config.config['port']}")
        self.console.print(
            f"   Transaction Logging: {'✅ Enabled' if self.config.config['enable_request_logging'] else '❌ Disabled'}"
        )
        self.console.print(
            f"   Raw I/O Logging:     {'✅ Enabled' if self.config.config.get('enable_raw_logging', False) else '❌ Disabled'}"
        )
        proxy_key = os.getenv("PROXY_API_KEY")
        if proxy_key:
            _masked = proxy_key[:4] + "..." + proxy_key[-4:] if len(proxy_key) > 8 else "***"
            self.console.print(f"   Proxy API Key:       {_masked}")
        else:
            self.console.print("   Proxy API Key:       [red]Not Set (INSECURE!)[/red]")

    def _show_main_status_summary(self, settings: dict):
        credentials = settings["credentials"]
        custom_bases = settings["custom_bases"]

        self.console.print()
        self.console.print("[bold]📊 Status Summary[/bold]")
        self.console.print("━" * 70)
        provider_count = len(credentials)
        custom_count = len(custom_bases)

        self.console.print(f"   Providers:           {provider_count} configured")
        self.console.print(f"   Custom Providers:    {custom_count} configured")
        # Note: provider_settings detection is deferred to avoid heavy imports on startup
        has_advanced = bool(
            settings["model_definitions"]
            or settings["concurrency_limits"]
            or settings["model_filters"]
        )
        self.console.print(
            f"   Advanced Settings:   {'Active (view in menu 4)' if has_advanced else 'None (view menu 4 for details)'}"
        )

    def _show_main_menu_options(self, show_warning: bool):
        self.console.print()
        self.console.print("━" * 70)
        self.console.print()
        self.console.print("[bold]🎯 Main Menu[/bold]")
        self.console.print()
        if show_warning:
            self.console.print("   1. ▶️  Run Proxy Server")
            self.console.print("   2. ⚙️  Configure Proxy Settings")
            self.console.print(
                "   3. 🔑 Manage Credentials            ⬅️  [bold yellow]Start here![/bold yellow]"
            )
        else:
            self.console.print("   1. ▶️  Run Proxy Server")
            self.console.print("   2. ⚙️  Configure Proxy Settings")
            self.console.print("   3. 🔑 Manage Credentials")

        self.console.print("   4. 📊 View Provider & Advanced Settings")
        self.console.print("   5. 📈 View Quota & Usage Stats (Alpha)")
        self.console.print("   6. 🔄 Reload Configuration")
        self.console.print("   7. ℹ️  About")
        self.console.print("   8. 🚪 Exit")

        self.console.print()
        self.console.print("━" * 70)
        self.console.print()

    def _prompt_main_menu_choice(self) -> str:
        return Prompt.ask(
            "Select option",
            choices=["1", "2", "3", "4", "5", "6", "7", "8"],
            show_choices=False,
        )

    def _handle_main_menu_choice(self, choice: str):
        handlers = {
            "1": self._handle_start_server,
            "2": self._handle_configure_proxy,
            "3": self._handle_manage_credentials,
            "4": self._handle_view_provider_settings,
            "5": self._handle_view_quota_stats,
            "6": self._handle_reload_configuration,
            "7": self._handle_about,
            "8": self._handle_exit,
        }
        handlers[choice]()

    def _handle_start_server(self):
        self.run_proxy()

    def _handle_configure_proxy(self):
        self.show_config_menu()

    def _handle_manage_credentials(self):
        self.launch_credential_tool()

    def _handle_view_provider_settings(self):
        self.show_provider_settings_menu()

    def _handle_view_quota_stats(self):
        self.launch_quota_viewer()

    def _handle_reload_configuration(self):
        load_dotenv(dotenv_path=get_data_file(".env"), override=True)
        self.config = LauncherConfig()  # Reload config
        self.console.print("\n[green]✅ Configuration reloaded![/green]")

    def _handle_about(self):
        self.show_about()

    def _handle_exit(self):
        self.running = False
        sys.exit(0)

    def confirm_setting_change(self, setting_name: str, warning_lines: list) -> bool:
        """
        Display a warning and require Y/N (case-sensitive) confirmation.
        Re-prompts until user enters exactly 'Y' or 'N'.
        Returns True only if user enters 'Y'.
        """
        clear_screen()
        self.console.print()
        self.console.print(
            Panel(
                Text.from_markup(
                    f"[bold yellow]⚠️  WARNING: You are about to change the {setting_name}[/bold yellow]\n\n"
                    + "\n".join(warning_lines)
                    + "\n\n[bold]If you are not sure about changing this - don't.[/bold]"
                ),
                border_style="yellow",
                expand=False,
            )
        )

        while True:
            response = Prompt.ask(
                "Enter [bold]Y[/bold] to confirm, [bold]N[/bold] to cancel (case-sensitive)"
            )
            if response == "Y":
                return True
            elif response == "N":
                self.console.print("\n[dim]Operation cancelled.[/dim]")
                return False
            else:
                self.console.print(
                    "[red]Please enter exactly 'Y' or 'N' (case-sensitive)[/red]"
                )

    def show_config_menu(self):
        """Display configuration sub-menu"""
        while True:
            clear_screen()
            self._show_config_header()
            self._show_config_current_settings()
            self._show_config_options()
            choice = self._prompt_config_menu_choice()
            if choice == "7":
                break
            self._handle_config_menu_choice(choice)

    def _show_config_header(self):
        self.console.print(
            Panel.fit(
                "[bold cyan]⚙️  Proxy Configuration[/bold cyan]", border_style="cyan"
            )
        )

    def _show_config_current_settings(self):
        self.console.print()
        self.console.print("[bold]📋 Current Settings[/bold]")
        self.console.print("━" * 70)
        self.console.print(f"   Host:                {self.config.config['host']}")
        self.console.print(f"   Port:                {self.config.config['port']}")
        self.console.print(
            f"   Transaction Logging: {'✅ Enabled' if self.config.config['enable_request_logging'] else '❌ Disabled'}"
        )
        self.console.print(
            f"   Raw I/O Logging:     {'✅ Enabled' if self.config.config.get('enable_raw_logging', False) else '❌ Disabled'}"
        )
        self.console.print(
            f"   Proxy API Key:       {'✅ Set' if os.getenv('PROXY_API_KEY') else '❌ Not Set'}"
        )

    def _show_config_options(self):
        self.console.print()
        self.console.print("━" * 70)
        self.console.print()
        self.console.print("[bold]⚙️  Configuration Options[/bold]")
        self.console.print()
        self.console.print("   1. 🌐 Set Host IP")
        self.console.print("   2. 🔌 Set Port")
        self.console.print("   3. 🔑 Set Proxy API Key")
        self.console.print("   4. 📝 Toggle Transaction Logging")
        self.console.print("   5. 📋 Toggle Raw I/O Logging")
        self.console.print("   6. 🔄 Reset to Default Settings")
        self.console.print("   7. ↩️  Back to Main Menu")

        self.console.print()
        self.console.print("━" * 70)
        self.console.print()

    def _prompt_config_menu_choice(self) -> str:
        return Prompt.ask(
            "Select option",
            choices=["1", "2", "3", "4", "5", "6", "7"],
            show_choices=False,
        )

    def _handle_config_menu_choice(self, choice: str):
        handlers = {
            "1": self._handle_set_host,
            "2": self._handle_set_port,
            "3": self._handle_set_proxy_api_key,
            "4": self._handle_toggle_transaction_logging,
            "5": self._handle_toggle_raw_logging,
            "6": self._handle_reset_defaults,
        }
        handlers[choice]()

    def _handle_set_host(self):
        confirmed = self.confirm_setting_change(
            "Host IP",
            [
                "Changing the host IP affects which network interfaces the proxy listens on:",
                "  • [cyan]127.0.0.1[/cyan] = Local access only (recommended for development)",
                "  • [cyan]0.0.0.0[/cyan] = Accessible from all network interfaces",
                "",
                "Applications configured to connect to the old host may fail to connect.",
            ],
        )
        if not confirmed:
            return

        new_host = Prompt.ask(
            "Enter new host IP", default=self.config.config["host"]
        )
        self.config.update(host=new_host)
        self.console.print(f"\n[green]✅ Host updated to: {new_host}[/green]")

    def _handle_set_port(self):
        confirmed = self.confirm_setting_change(
            "Port",
            [
                "Changing the port will affect all applications currently configured",
                "to connect to your proxy on the existing port.",
                "",
                "Applications using the old port will fail to connect.",
            ],
        )
        if not confirmed:
            return

        new_port = int(IntPrompt.ask(
            "Enter new port", default=self.config.config["port"]
        ))
        if 1 <= new_port <= 65535:
            self.config.update(port=new_port)
            self.console.print(
                f"\n[green]✅ Port updated to: {new_port}[/green]"
            )
        else:
            self.console.print("\n[red]❌ Port must be between 1-65535[/red]")

    def _handle_set_proxy_api_key(self):
        confirmed = self.confirm_setting_change(
            "Proxy API Key",
            [
                "This is the authentication key that applications use to access your proxy.",
                "",
                "[bold red]⚠️  Changing this will BREAK all applications currently configured",
                "   with the existing API key![/bold red]",
                "",
                "[bold cyan]💡 If you want to add provider API keys (OpenAI, Gemini, etc.),",
                '   go to "3. 🔑 Manage Credentials" in the main menu instead.[/bold cyan]',
            ],
        )
        if not confirmed:
            return

        current = os.getenv("PROXY_API_KEY") or ""
        new_key = Prompt.ask(
            "Enter new Proxy API Key (leave empty to disable authentication)",
            default=current,
        )

        if new_key != current:
            # If setting to empty, show additional warning
            if not new_key:
                self.console.print(
                    "\n[bold red]⚠️  Authentication will be DISABLED - anyone can access your proxy![/bold red]"
                )
                Prompt.ask("Press Enter to continue", default="")

            LauncherConfig.update_proxy_api_key(new_key)

            if new_key:
                self.console.print(
                    "\n[green]✅ Proxy API Key updated successfully![/green]"
                )
                self.console.print("   Updated in .env file")
            else:
                self.console.print(
                    "\n[yellow]⚠️  Proxy API Key cleared - authentication disabled![/yellow]"
                )
                self.console.print("   Updated in .env file")
        else:
            self.console.print("\n[yellow]No changes made[/yellow]")

    def _handle_toggle_transaction_logging(self):
        current = self.config.config["enable_request_logging"]
        self.config.update(enable_request_logging=not current)
        self.console.print(
            f"\n[green]✅ Transaction Logging {'enabled' if not current else 'disabled'}![/green]"
        )

    def _handle_toggle_raw_logging(self):
        current = self.config.config.get("enable_raw_logging", False)
        self.config.update(enable_raw_logging=not current)
        self.console.print(
            f"\n[green]✅ Raw I/O Logging {'enabled' if not current else 'disabled'}![/green]"
        )

    def _handle_reset_defaults(self):
        default_host = os.environ.get("PROXY_HOST", DEFAULT_HOST)
        default_port = env_int("PROXY_PORT", DEFAULT_PORT)
        default_logging = os.environ.get("PROXY_ENABLE_REQUEST_LOGGING", "false").lower() == "true"
        default_raw_logging = os.environ.get("PROXY_ENABLE_RAW_LOGGING", "false").lower() == "true"
        # Use secrets.token_urlsafe(32) only if PROXY_API_KEY is not already set in environment
        default_api_key = os.environ.get("PROXY_API_KEY") or secrets.token_urlsafe(32)

        current_host = self.config.config["host"]
        current_port = self.config.config["port"]
        current_logging = self.config.config["enable_request_logging"]
        current_raw_logging = self.config.config.get(
            "enable_raw_logging", False
        )
        current_api_key = os.getenv("PROXY_API_KEY") or ""

        warning_lines = [
            "This will reset ALL proxy settings to their defaults:",
            "",
            "[bold]   Setting              Current Value         →  Default Value[/bold]",
            "   " + "─" * 62,
            f"   Host IP              {current_host:20} →  {default_host}",
            f"   Port                 {str(current_port):20} →  {default_port}",
            f"   Transaction Logging  {'Enabled':20} →  Disabled"
            if current_logging
            else f"   Transaction Logging  {'Disabled':20} →  Disabled",
            f"   Raw I/O Logging      {'Enabled':20} →  Disabled"
            if current_raw_logging
            else f"   Raw I/O Logging      {'Disabled':20} →  Disabled",
            f"   Proxy API Key        {current_api_key[:20]:20} →  {default_api_key}",
            "",
            "[bold red]⚠️  This may break applications configured with current settings![/bold red]",
        ]

        confirmed = self.confirm_setting_change(
            "Settings (Reset to Defaults)", warning_lines
        )
        if not confirmed:
            return

        self.config.update(
            host=default_host,
            port=default_port,
            enable_request_logging=default_logging,
            enable_raw_logging=default_raw_logging,
        )
        LauncherConfig.update_proxy_api_key(default_api_key)

        self.console.print(
            "\n[green]✅ All settings have been reset to defaults![/green]"
        )
        self.console.print(f"   Host:               {default_host}")
        self.console.print(f"   Port:               {default_port}")
        self.console.print("   Transaction Logging: Disabled")
        self.console.print("   Raw I/O Logging:    Disabled")
        _default_masked = default_api_key[:4] + "..." + default_api_key[-4:] if len(default_api_key) > 8 else "***"
        self.console.print(f"   Proxy API Key:      {_default_masked}")

    def show_provider_settings_menu(self):
        """Display provider/advanced settings (read-only + launch tool)"""
        clear_screen()

        # Use basic settings to avoid heavy imports - provider_settings deferred to Settings Tool
        settings = SettingsDetector.get_basic_settings()

        credentials = settings["credentials"]
        custom_bases = settings["custom_bases"]
        model_defs = settings["model_definitions"]
        concurrency = settings["concurrency_limits"]
        filters = settings["model_filters"]

        self.console.print(
            Panel.fit(
                "[bold cyan]📊 Provider & Advanced Settings[/bold cyan]",
                border_style="cyan",
            )
        )

        # Configured Providers
        self.console.print()
        self.console.print("[bold]📊 Configured Providers[/bold]")
        self.console.print("━" * 70)
        if credentials:
            for provider, info in credentials.items():
                provider_name = provider.title()
                parts = []
                if info["api_keys"] > 0:
                    parts.append(
                        f"{info['api_keys']} API key{'s' if info['api_keys'] > 1 else ''}"
                    )
                if info["oauth"] > 0:
                    parts.append(
                        f"{info['oauth']} OAuth credential{'s' if info['oauth'] > 1 else ''}"
                    )

                display = " + ".join(parts)
                if info["custom"]:
                    display += " (Custom)"

                self.console.print(f"   ✅ {provider_name:20} {display}")
        else:
            self.console.print("   [dim]No providers configured[/dim]")

        # Custom API Bases
        if custom_bases:
            self.console.print()
            self.console.print("[bold]🌐 Custom API Bases[/bold]")
            self.console.print("━" * 70)
            for provider, base in custom_bases.items():
                self.console.print(f"   • {provider:15} {base}")

        # Model Definitions
        if model_defs:
            self.console.print()
            self.console.print("[bold]📦 Provider Model Definitions[/bold]")
            self.console.print("━" * 70)
            for provider, count in model_defs.items():
                self.console.print(
                    f"   • {provider:15} {count} model{'s' if count > 1 else ''} configured"
                )

        # Concurrency Limits
        if concurrency:
            self.console.print()
            self.console.print("[bold]⚡ Concurrency Limits[/bold]")
            self.console.print("━" * 70)
            for provider, limit in concurrency.items():
                self.console.print(f"   • {provider:15} {limit} requests/key")
            self.console.print("   • Default:        1 request/key (all others)")

        # Model Filters (basic info only)
        if filters:
            self.console.print()
            self.console.print("[bold]🎯 Model Filters[/bold]")
            self.console.print("━" * 70)
            for provider, filter_info in filters.items():
                status_parts = []
                if filter_info["has_whitelist"]:
                    status_parts.append("Whitelist")
                if filter_info["has_ignore"]:
                    status_parts.append("Ignore list")
                status = " + ".join(status_parts) if status_parts else "None"
                self.console.print(f"   • {provider:15} ✅ {status}")

        # Provider-Specific Settings (deferred to Settings Tool to avoid heavy imports)
        self.console.print()
        self.console.print("[bold]🔬 Provider-Specific Settings[/bold]")
        self.console.print("━" * 70)
        self.console.print(
            "   [dim]Launch Settings Tool to view/configure provider-specific settings[/dim]"
        )

        # Actions
        self.console.print()
        self.console.print("━" * 70)
        self.console.print()
        self.console.print("[bold]💡 Actions[/bold]")
        self.console.print()
        self.console.print(
            "   1. 🔧 Launch Settings Tool      (configure advanced settings)"
        )
        self.console.print("   2. ↩️  Back to Main Menu")

        self.console.print()
        self.console.print("━" * 70)
        self.console.print(
            "[dim]ℹ️  Advanced settings are stored in .env file.\n   Use the Settings Tool to configure them interactively.[/dim]"
        )
        self.console.print()
        self.console.print(
            "[dim]⚠️  Note: Settings Tool supports only common configuration types.\n   For complex settings, edit .env directly.[/dim]"
        )
        self.console.print()

        choice = Prompt.ask("Select option", choices=["1", "2"], show_choices=False)

        if choice == "1":
            self.launch_settings_tool()
        # choice == "2" returns to main menu

    def launch_credential_tool(self):
        """Launch credential management tool"""
        import time

        # CRITICAL: Show full loading UI to replace the 6-7 second blank wait
        clear_screen()

        _start_time = time.time()

        # Show the same header as standalone mode
        self.console.print("━" * 70)
        self.console.print("Interactive Credential Setup Tool")
        self.console.print("GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
        self.console.print("━" * 70)
        self.console.print("Loading credential management components...")

        # Now import with spinner (this is where the 6-7 second delay happens)
        with self.console.status("Initializing credential tool...", spinner="dots"):
            from rotator_library.credential_tool import (
                run_credential_tool,
                _ensure_providers_loaded,
            )

            _, PROVIDER_PLUGINS = _ensure_providers_loaded()
        self.console.print("✓ Credential tool initialized")

        _elapsed = time.time() - _start_time
        self.console.print(
            f"✓ Tool ready in {_elapsed:.2f}s ({len(PROVIDER_PLUGINS) if PROVIDER_PLUGINS is not None else 0} providers available)"
        )

        # Small delay to let user see the ready message
        time.sleep(0.5)

        # Run the tool with from_launcher=True to skip duplicate loading screen
        run_credential_tool(from_launcher=True)
        # Reload environment after credential tool
        load_dotenv(dotenv_path=get_data_file(".env"), override=True)

    def launch_settings_tool(self):
        """Launch settings configuration tool"""
        import time

        clear_screen()

        self.console.print("━" * 70)
        self.console.print("Advanced Settings Configuration Tool")
        self.console.print("━" * 70)

        _start_time = time.time()

        with self.console.status("Initializing settings tool...", spinner="dots"):
            from proxy_app.settings_tool import run_settings_tool

        _elapsed = time.time() - _start_time
        self.console.print(f"✓ Settings tool ready in {_elapsed:.2f}s")

        time.sleep(0.3)

        run_settings_tool()
        # Reload environment after settings tool
        load_dotenv(dotenv_path=get_data_file(".env"), override=True)

    def launch_quota_viewer(self):
        """Launch the quota stats viewer"""
        clear_screen()

        self.console.print("━" * 70)
        self.console.print("Quota & Usage Statistics Viewer")
        self.console.print("━" * 70)
        self.console.print()

        # Import the lightweight viewer (no heavy imports)
        from proxy_app.quota_viewer import run_quota_viewer

        run_quota_viewer()

    def show_about(self):
        """Display About page with project information"""
        clear_screen()

        self.console.print(
            Panel.fit(
                "[bold cyan]ℹ️  About LLM API Key Proxy[/bold cyan]", border_style="cyan"
            )
        )

        self.console.print()
        self.console.print("[bold]📦 Project Information[/bold]")
        self.console.print("━" * 70)
        self.console.print("   [bold cyan]LLM API Key Proxy[/bold cyan]")
        self.console.print(
            "   A lightweight, high-performance proxy server for managing"
        )
        self.console.print("   LLM API keys with automatic rotation and OAuth support")
        self.console.print()
        self.console.print(
            "   [dim]GitHub:[/dim] [blue underline]https://github.com/ShmidtS/LLM-API-Key-Proxy[/blue underline]"
        )

        self.console.print()
        self.console.print("[bold]✨ Key Features[/bold]")
        self.console.print("━" * 70)
        self.console.print(
            "   • [green]Smart Key Rotation[/green] - Automatic rotation across multiple API keys"
        )
        self.console.print(
            "   • [green]OAuth Support[/green] - Automated OAuth flows for supported providers"
        )
        self.console.print(
            "   • [green]Multiple Providers[/green] - Support for 10+ LLM providers"
        )
        self.console.print(
            "   • [green]Custom Providers[/green] - Easy integration of custom OpenAI-compatible APIs"
        )
        self.console.print(
            "   • [green]Advanced Filtering[/green] - Model whitelists and ignore lists per provider"
        )
        self.console.print(
            "   • [green]Concurrency Control[/green] - Per-key rate limiting and request management"
        )
        self.console.print(
            "   • [green]Cost Tracking[/green] - Track usage and costs across all providers"
        )
        self.console.print(
            "   • [green]Interactive TUI[/green] - Beautiful terminal interface for easy configuration"
        )

        self.console.print()
        self.console.print("[bold]📝 License & Credits[/bold]")
        self.console.print("━" * 70)
        self.console.print("   Made with ❤️  by the community")
        self.console.print("   Open source - contributions welcome!")

        self.console.print()
        self.console.print("━" * 70)
        self.console.print()

        Prompt.ask("Press Enter to return to main menu", default="")

    def run_proxy(self):
        """Prepare and launch proxy in same window"""
        # Check if forced onboarding needed
        if self.needs_onboarding():
            clear_screen()
            self.console.print(
                Panel(
                    Text.from_markup(
                        "⚠️  [bold yellow]Setup Required[/bold yellow]\n\n"
                        "Cannot start without .env.\n"
                        "Launching credential tool..."
                    ),
                    border_style="yellow",
                )
            )

            # Force credential tool
            from rotator_library.credential_tool import (
                ensure_env_defaults,
                run_credential_tool,
            )

            ensure_env_defaults()
            load_dotenv(dotenv_path=get_data_file(".env"), override=True)
            run_credential_tool()
            load_dotenv(dotenv_path=get_data_file(".env"), override=True)
    
            # Check again after credential tool
            if not os.getenv("PROXY_API_KEY"):
                self.console.print(
                    "\n[red]❌ PROXY_API_KEY still not set. Cannot start proxy.[/red]"
                )
                return

        # Clear console and modify sys.argv
        clear_screen()
        self.console.print(
            f"\n[bold green]🚀 Starting proxy on {self.config.config['host']}:{self.config.config['port']}...[/bold green]\n"
        )

        # Brief pause so user sees the message before main.py takes over
        import time

        time.sleep(0.5)

        # Reconstruct sys.argv for main.py
        sys.argv = [
            "main.py",
            "--host",
            self.config.config["host"],
            "--port",
            str(self.config.config["port"]),
        ]
        if self.config.config["enable_request_logging"]:
            sys.argv.append("--enable-request-logging")
        if self.config.config.get("enable_raw_logging", False):
            sys.argv.append("--enable-raw-logging")

        # Exit TUI - main.py will continue execution
        self.running = False


def run_launcher_tui():
    """Entry point for launcher TUI"""
    tui = LauncherTUI()
    tui.run()
