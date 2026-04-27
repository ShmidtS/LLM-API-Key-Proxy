# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import os
import sys

import time

# Phase 1: Minimal imports for arg parsing and TUI
import asyncio
from pathlib import Path

# Set Windows Selector event loop policy BEFORE any async operations.
# ProactorEventLoop (default on Windows Python 3.12+) causes:
# - ConnectionResetError spam after streaming
# - File descriptor issues with selector-based libraries (aiohttp)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import argparse
import atexit
import logging
import logging.handlers
import queue
import re

# Add the 'src' directory to the Python path before importing local modules.
sys.path.append(str(Path(__file__).resolve().parent.parent))

from proxy_app.config import DEFAULT_HOST, DEFAULT_PORT

logger = logging.getLogger(__name__)

# Early startup messages need a basic handler before full config is set up.
# Without this, logger.info() calls before line ~246 are silently dropped.
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io

    if hasattr(sys.stdout, "buffer") and sys.stdout.buffer is not None:
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
    if hasattr(sys.stderr, "buffer") and sys.stderr.buffer is not None:
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        logger.debug("stdout/stderr reconfigure failed, ignoring", exc_info=True)

# --- Argument Parsing (BEFORE heavy imports) ---
parser = argparse.ArgumentParser(description="API Key Proxy Server")
parser.add_argument(
    "--host",
    type=str,
    default=DEFAULT_HOST,
    help="Host to bind the server to. Use 0.0.0.0 to expose to all interfaces.",
)
parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run the server on.")
parser.add_argument(
    "--enable-request-logging",
    action="store_true",
    help="Enable transaction logging in the library (logs request/response with provider correlation).",
)
parser.add_argument(
    "--enable-raw-logging",
    action="store_true",
    help="Enable raw I/O logging at proxy boundary (captures unmodified HTTP data, disabled by default).",
)
parser.add_argument(
    "--add-credential",
    action="store_true",
    help="Launch the interactive tool to add a new OAuth credential.",
)
args, _ = parser.parse_known_args()

# Check if we should launch TUI (no arguments = TUI mode)
if len(sys.argv) == 1:
    # TUI MODE - Load ONLY what's needed for the launcher (fast path!)
    from proxy_app.launcher_tui import run_launcher_tui

    run_launcher_tui()
    # Launcher modifies sys.argv and returns, or exits if user chose Exit
    # If we get here, user chose "Run Proxy" and sys.argv is modified
    # Re-parse arguments with modified sys.argv
    args = parser.parse_args()

# Check if credential tool mode (also doesn't need heavy proxy imports)
if args.add_credential:
    from rotator_library.credential_tool import run_credential_tool

    run_credential_tool()
    sys.exit(0)

# If we get here, we're ACTUALLY running the proxy - NOW show startup messages and start timer
_start_time = time.time()

# Load all .env files from root folder (main .env first, then any additional *.env files)
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    logger.warning("python-dotenv not installed; .env files will not be loaded")

# Get the application root directory (EXE dir if frozen, else CWD)
# Inlined here to avoid triggering heavy rotator_library imports before loading screen
if getattr(sys, "frozen", False):
    _root_dir = Path(sys.executable).parent
else:
    _root_dir = Path.cwd()

# Load main .env first
if load_dotenv is not None:
    load_dotenv(_root_dir / ".env")

# Load any additional .env files (e.g., antigravity_all_combined.env, gemini_cli_all_combined.env)
_env_files_found = list(_root_dir.glob("*.env"))
if load_dotenv is not None:
    for _env_file in sorted(_env_files_found):  # reuse already-computed list
        if _env_file.name != ".env":  # Skip main .env (already loaded)
            load_dotenv(_env_file, override=False)  # Don't override existing values

# Log discovered .env files for deployment verification
if _env_files_found:
    _env_names = [_ef.name for _ef in _env_files_found]
    logger.info(
        "Loaded %d .env file(s): %s", len(_env_files_found), ", ".join(_env_names)
    )

# Get proxy API key for display
_early_proxy_api_key = os.getenv("PROXY_API_KEY")
if _early_proxy_api_key:
    _masked = "***"
    key_display = f"✓ {_masked}"
else:
    key_display = "✗ Not Set"

logger.info("━" * 70)
logger.info("Starting proxy on %s:%s", args.host, args.port)
# Do not log the actual API key, only the status
logger.info("Proxy API Key status: %s", "Set" if _early_proxy_api_key else "Not Set")
logger.info("GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
logger.info("━" * 70)
logger.info("Loading server components...")


# Phase 2: Load Rich for loading spinner (lightweight)
from rich.console import Console

_console = Console()

# Phase 3: Heavy dependencies with granular loading messages
logger.info("Loading FastAPI framework...")
with _console.status("[dim]Loading FastAPI framework...", spinner="dots"):
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

logger.info("Loading core dependencies...")
with _console.status("[dim]Loading core dependencies...", spinner="dots"):
    import colorlog

    # --- Early Log Level Configuration ---
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

logger.info("Loading LiteLLM library...")
with _console.status("[dim]Loading LiteLLM library...", spinner="dots"):
    import litellm
    import httpx

    # CRITICAL: Apply SSL patches IMMEDIATELY after litellm import
    # This must happen BEFORE rotator_library import and BEFORE any API calls
    from rotator_library.ssl_patch import _patch_litellm_ssl

    _patch_litellm_ssl()

# Phase 4: Application imports
from proxy_app.middleware import _NoGzipForSSE
from proxy_app.logging_config import RotatorDebugFilter, NoLiteLLMLogFilter

# Anthropic API Models (imported from library)
logger.info("Discovering provider plugins...")
# Provider lazy loading happens during import, so time it here
_provider_start = time.time()
with _console.status("[dim]Discovering provider plugins...", spinner="dots"):
    from rotator_library import (
        PROVIDER_PLUGINS,
    )  # This triggers lazy load via __getattr__
_provider_time = time.time() - _provider_start

# Get count after import (without timing to avoid double-counting)
_plugin_count = len(PROVIDER_PLUGINS)


# Calculate total loading time
_elapsed = time.time() - _start_time
logger.info(
    "Server ready in %.2fs (%d providers discovered in %.2fs)",
    _elapsed,
    _plugin_count,
    _provider_time,
)

# Clear screen and reprint header for clean startup view
# This pushes loading messages up (still in scroll history) but shows a clean final screen
sys.stdout.write("\033[2J\033[H")
sys.stdout.flush()

# Reprint header
logger.info("━" * 70)
logger.info("Starting proxy on %s:%s", args.host, args.port)
logger.info("Proxy API Key status: %s", "Set" if _early_proxy_api_key else "Not Set")
logger.info("GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
logger.info("━" * 70)
logger.info(
    "Server ready in %.2fs (%d providers discovered in %.2fs)",
    _elapsed,
    _plugin_count,
    _provider_time,
)


# Note: Debug logging will be added after logging configuration below

# --- Logging Configuration ---
# Import path utilities here (after loading screen) to avoid triggering heavy imports early
from rotator_library.utils.paths import get_logs_dir, get_data_file
from rotator_library.utils.terminal_utils import clear_screen

LOG_DIR = get_logs_dir(_root_dir)

# Configure a file handler for INFO-level logs and higher
_info_file_handler = logging.FileHandler(LOG_DIR / "proxy.log", encoding="utf-8")
_info_file_handler.setLevel(logging.INFO)
_info_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
_info_queue = queue.Queue(-1)
info_file_handler = logging.handlers.QueueHandler(_info_queue)
_info_queue_listener = logging.handlers.QueueListener(
    _info_queue, _info_file_handler, respect_handler_level=True
)
_info_queue_listener.start()
atexit.register(_info_queue_listener.stop)

# Configure a dedicated file handler for all DEBUG-level logs
_debug_file_handler = logging.FileHandler(LOG_DIR / "proxy_debug.log", encoding="utf-8")
_debug_file_handler.setLevel(logging.DEBUG)
_debug_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
_debug_file_handler.addFilter(RotatorDebugFilter())
_debug_queue = queue.Queue(-1)
debug_file_handler = logging.handlers.QueueHandler(_debug_queue)
_debug_queue_listener = logging.handlers.QueueListener(
    _debug_queue, _debug_file_handler, respect_handler_level=True
)
_debug_queue_listener.start()
atexit.register(_debug_queue_listener.stop)

# Configure a console handler with color
console_handler = colorlog.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(message)s",
    log_colors={
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white",
    },
)
console_handler.setFormatter(formatter)

console_handler.addFilter(NoLiteLLMLogFilter())

# Get the root logger and set it to DEBUG to capture all messages
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Remove the basicConfig StreamHandler added at startup to avoid duplicate output
root_logger.handlers.clear()

# Add all handlers to the root logger
root_logger.addHandler(info_file_handler)
root_logger.addHandler(console_handler)
root_logger.addHandler(debug_file_handler)

# Silence other noisy loggers by setting their level higher than root
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Isolate LiteLLM's logger to prevent it from reaching the console.
# We will capture its logs via the logger_fn callback in the client instead.
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.handlers = []
litellm_logger.propagate = False

# Now that logging is configured, log the module load time to debug file only
logger.debug(f"Modules loaded in {_elapsed:.2f}s")


# --- Configuration ---
USE_EMBEDDING_BATCHER = False
ENABLE_REQUEST_LOGGING = args.enable_request_logging
ENABLE_RAW_LOGGING = args.enable_raw_logging
if ENABLE_REQUEST_LOGGING:
    logger.info(
        "Transaction logging is enabled (library-level with provider correlation)."
    )
if ENABLE_RAW_LOGGING:
    logger.info("Raw I/O logging is enabled (proxy boundary, unmodified HTTP data).")
PROXY_API_KEY = _early_proxy_api_key
# Note: PROXY_API_KEY validation moved to server startup to allow credential tool to run first
# Pre-build Bearer string once to avoid f-string on every request
_BEARER_PROXY_API_KEY = f"Bearer {PROXY_API_KEY}" if PROXY_API_KEY else None

# Cache OVERRIDE_TEMPERATURE_ZERO at module load time (stored on app.state during lifespan)
OVERRIDE_TEMP_ZERO = os.getenv("OVERRIDE_TEMPERATURE_ZERO", "false").lower()


def _parse_env_prefix_map(prefix: str, parser=None):
    """Parse env vars matching PREFIX into {suffix_lower: parser(value)} dict."""
    result = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            suffix = key[len(prefix) :].lower()
            result[suffix] = parser(value) if parser else value
    return result


def _parse_comma_list(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


# Discover API keys from environment variables
api_keys = {}
for key, value in os.environ.items():
    if "_API_KEY" in key and key != "PROXY_API_KEY":
        match = re.match(r"^([A-Z0-9]+)_API_KEY(?:_\d+)?$", key)
        if match:
            provider = match.group(1).lower()
            if provider not in api_keys:
                api_keys[provider] = []
            api_keys[provider].append(value)

# Legacy provider name aliases: remap old env var prefixes to canonical names
_PROVIDER_CREDENTIAL_ALIASES = {
    "nvidia_nim": "nvidia",
}
for _old, _new in _PROVIDER_CREDENTIAL_ALIASES.items():
    if _old in api_keys and _new not in api_keys:
        api_keys[_new] = api_keys.pop(_old)
    elif _old in api_keys and _new in api_keys:
        api_keys[_new].extend(api_keys.pop(_old))

# Load model ignore/whitelist lists and max concurrent from environment variables
ignore_models = _parse_env_prefix_map("IGNORE_MODELS_", _parse_comma_list)
whitelist_models = _parse_env_prefix_map("WHITELIST_MODELS_", _parse_comma_list)


def _parse_max_concurrent(value: str):
    try:
        v = int(value)
        return v if v >= 1 else 1
    except ValueError:
        return 1


max_concurrent_requests_per_key = _parse_env_prefix_map(
    "MAX_CONCURRENT_REQUESTS_PER_KEY_", _parse_max_concurrent
)


# --- Lifespan Management (delegated to _lifecycle module) ---
from proxy_app._lifecycle import LifespanConfig, create_lifespan
from proxy_app.config import (
    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_UVICORN_BACKLOG,
    env_int,
)

lifespan = create_lifespan(
    LifespanConfig(
        api_keys=api_keys,
        proxy_api_key=PROXY_API_KEY,
        bearer_proxy_api_key=_BEARER_PROXY_API_KEY,
        override_temp_zero=OVERRIDE_TEMP_ZERO,
        enable_request_logging=ENABLE_REQUEST_LOGGING,
        enable_raw_logging=ENABLE_RAW_LOGGING,
        use_embedding_batcher=USE_EMBEDDING_BATCHER,
        max_concurrent_requests_per_key=max_concurrent_requests_per_key,
        ignore_models=ignore_models,
        whitelist_models=whitelist_models,
    )
)


# --- FastAPI App Setup ---

app = FastAPI(lifespan=lifespan)

# Add CORS middleware with env-configured origins
_cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _cors_origins_env.strip():
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins == ["*"]:
        logger.warning(
            "CORS_ALLOWED_ORIGINS='*' — allowing all origins (dev-mode only, restrict in production!)"
        )
else:
    logger.warning(
        "CORS_ALLOWED_ORIGINS not set — defaulting to same-origin only (set CORS_ALLOWED_ORIGINS to allow specific origins)"
    )
    _cors_origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=[
        "X-Accel-Buffering",
        "X-Request-Id",
        "X-Provider",
        "Retry-After",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
    ],  # Custom headers visible to JS clients
)

# SSE-aware gzip: compresses non-streaming responses >= minimum_size, passes SSE through raw
app.add_middleware(_NoGzipForSSE)


# --- Register route modules ---
from proxy_app.routes import all_routers

for _router in all_routers:
    app.include_router(_router)


@app.api_route("/", methods=["GET", "HEAD"])
async def read_root():
    return {"Status": "API Key Proxy is running"}


if __name__ == "__main__":
    # Define ENV_FILE for onboarding checks using centralized path
    ENV_FILE = get_data_file(".env")

    # Check if launcher TUI should be shown (no arguments provided)
    if len(sys.argv) == 1:
        # No arguments - show launcher TUI (lazy import)
        from proxy_app.launcher_tui import run_launcher_tui

        run_launcher_tui()
        # Launcher modifies sys.argv and returns, or exits if user chose Exit
        # If we get here, user chose "Run Proxy" and sys.argv is modified
        # Re-parse arguments with modified sys.argv
        args = parser.parse_args()

    def needs_onboarding() -> bool:
        """
        Check if the proxy needs onboarding (first-time setup).
        Returns True if onboarding is needed, False otherwise.
        """
        # Only check if .env file exists
        # PROXY_API_KEY is optional (will show warning if not set)
        if not ENV_FILE.is_file():
            return True

        return False

    def show_onboarding_message():
        """Display clear explanatory message for why onboarding is needed."""
        clear_screen()
        console.print(
            Panel.fit(
                "[bold cyan]🚀 LLM API Key Proxy - First Time Setup[/bold cyan]",
                border_style="cyan",
            )
        )
        console.print("[bold yellow]⚠️  Configuration Required[/bold yellow]\n")

        console.print("The proxy needs initial configuration:")
        console.print("  [red]❌ No .env file found[/red]")

        console.print("\n[bold]Why this matters:[/bold]")
        console.print("  • The .env file stores your credentials and settings")
        console.print("  • PROXY_API_KEY protects your proxy from unauthorized access")
        console.print("  • Provider API keys enable LLM access")

        console.print("\n[bold]What happens next:[/bold]")
        console.print("  1. We'll create a .env file with PROXY_API_KEY")
        console.print("  2. You can add LLM provider credentials (API keys or OAuth)")
        console.print("  3. The proxy will then start normally")

        console.print(
            "\n[bold yellow]⚠️  Note:[/bold yellow] The credential tool adds PROXY_API_KEY by default."
        )
        console.print("   You can remove it later if you want an unsecured proxy.\n")

        console.input(
            "[bold green]Press Enter to launch the credential setup tool...[/bold green]"
        )

    # Check if user explicitly wants to add credentials
    if args.add_credential:
        # Import and call ensure_env_defaults to create .env and PROXY_API_KEY if needed
        from rotator_library.credential_tool import ensure_env_defaults

        ensure_env_defaults()
        # Reload environment variables after ensure_env_defaults creates/updates .env
        load_dotenv(ENV_FILE, override=True)
        run_credential_tool()
    else:
        # Check if onboarding is needed
        if needs_onboarding():
            # Import console from rich for better messaging
            from rich.console import Console
            from rich.panel import Panel

            console = Console()

            # Show clear explanatory message
            show_onboarding_message()

            # Launch credential tool automatically
            from rotator_library.credential_tool import ensure_env_defaults

            ensure_env_defaults()
            load_dotenv(ENV_FILE, override=True)
            run_credential_tool()

            # After credential tool exits, reload and re-check
            load_dotenv(ENV_FILE, override=True)
            # Re-read PROXY_API_KEY from environment (may have changed after credential tool)
            _early_proxy_api_key = os.getenv("PROXY_API_KEY")
            PROXY_API_KEY = _early_proxy_api_key
            _BEARER_PROXY_API_KEY = f"Bearer {PROXY_API_KEY}" if PROXY_API_KEY else None

            # Verify onboarding is complete
            if needs_onboarding():
                console.print("\n[bold red]❌ Configuration incomplete.[/bold red]")
                console.print(
                    "The proxy still cannot start. Please ensure PROXY_API_KEY is set in .env\n"
                )
                sys.exit(1)
            else:
                console.print("\n[bold green]✅ Configuration complete![/bold green]")
                console.print("\nStarting proxy server...\n")

        import uvicorn

        if sys.platform == "win32":
            import signal as _signal

            if hasattr(_signal, "SIGBREAK"):
                _signal.signal(
                    _signal.SIGBREAK, lambda *_: _signal.raise_signal(_signal.SIGINT)
                )

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            limit_concurrency=env_int(
                "MAX_CONCURRENT_REQUESTS", DEFAULT_MAX_CONCURRENT_REQUESTS
            ),
            backlog=env_int("UVICORN_BACKLOG", DEFAULT_UVICORN_BACKLOG),
            timeout_graceful_shutdown=env_int(
                "TIMEOUT_GRACEFUL_SHUTDOWN", DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT
            ),
        )
