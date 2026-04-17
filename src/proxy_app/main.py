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
import logging
import re

logger = logging.getLogger(__name__)

# Early startup messages need a basic handler before full config is set up.
# Without this, logger.info() calls before line ~246 are silently dropped.
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io

    if hasattr(sys.stdout, 'buffer') and sys.stdout.buffer is not None:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, 'buffer') and sys.stderr.buffer is not None:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, io.UnsupportedOperation):
        pass

# --- Argument Parsing (BEFORE heavy imports) ---
parser = argparse.ArgumentParser(description="API Key Proxy Server")
parser.add_argument(
    "--host", type=str, default="127.0.0.1",
    help="Host to bind the server to. Use 0.0.0.0 to expose to all interfaces."
)
parser.add_argument("--port", type=int, default=8000, help="Port to run the server on.")
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

# Add the 'src' directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

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
    logger.info("Loaded %d .env file(s): %s", len(_env_files_found), ', '.join(_env_names))

# Get proxy API key for display
_early_proxy_api_key = os.getenv("PROXY_API_KEY")
if _early_proxy_api_key:
    _masked = _early_proxy_api_key[:4] + "..." + _early_proxy_api_key[-4:] if len(_early_proxy_api_key) > 8 else "***"
    key_display = f"✓ {_masked}"
else:
    key_display = "✗ Not Set (INSECURE - anyone can access!)"

logger.info("━" * 70)
logger.info("Starting proxy on %s:%s", args.host, args.port)
logger.info("Proxy API Key: %s", key_display)
logger.info("GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
logger.info("━" * 70)
logger.info("Loading server components...")


# Phase 2: Load Rich for loading spinner (lightweight)
from rich.console import Console

_console = Console()

# Phase 3: Heavy dependencies with granular loading messages
logger.info("Loading FastAPI framework...")
with _console.status("[dim]Loading FastAPI framework...", spinner="dots"):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

logger.info("Loading core dependencies...")
with _console.status("[dim]Loading core dependencies...", spinner="dots"):
    import colorlog
    import json
    import orjson
    # --- Early Log Level Configuration ---
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

logger.info("Loading LiteLLM library...")
with _console.status("[dim]Loading LiteLLM library...", spinner="dots"):
    import litellm
    import httpx

    # CRITICAL: Apply SSL patches IMMEDIATELY after litellm import
    # This must happen BEFORE rotator_library import and BEFORE any API calls

    _ssl_verify_env = os.environ.get("HTTP_SSL_VERIFY", "true").lower()
    if _ssl_verify_env == "false":
        logger.info("[SSL-FIX-MAIN] HTTP_SSL_VERIFY=false - Applying SSL patches in main.py")

        # 1. Set litellm's SSL verification to False
        litellm.ssl_verify = False
        logger.info("[SSL-FIX-MAIN] Set litellm.ssl_verify = False")

        # 2. Create pre-configured httpx clients with SSL verification disabled
        # This is the MOST RELIABLE way to disable SSL in litellm
        from rotator_library.timeout_config import TimeoutConfig
        _litellm_timeout = TimeoutConfig.non_streaming()
        litellm.client_session = httpx.Client(verify=False, timeout=_litellm_timeout)
        litellm.aclient_session = httpx.AsyncClient(verify=False, timeout=_litellm_timeout)
        logger.info("[SSL-FIX-MAIN] Created litellm.client_session and aclient_session with verify=False")

        # 3. Set environment variable for litellm
        os.environ["SSL_VERIFY"] = "False"
        logger.info("[SSL-FIX-MAIN] Set SSL_VERIFY=False environment variable")

# Phase 4: Application imports with granular loading messages
logger.info("Initializing proxy core...")
with _console.status("[dim]Initializing proxy core...", spinner="dots"):
    from rotator_library import RotatingClient
    from rotator_library.credential_manager import CredentialManager
    from rotator_library.dns_fix import close_doh_client, close_dns_executor
    from rotator_library.model_info_service import init_model_info_service
    from proxy_app.batch_manager import EmbeddingBatcher

# Import extracted modules
from proxy_app.middleware import _NoGzipForSSE, SecurityHeadersMiddleware
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
logger.info("Server ready in %.2fs (%d providers discovered in %.2fs)", _elapsed, _plugin_count, _provider_time)

# Clear screen and reprint header for clean startup view
# This pushes loading messages up (still in scroll history) but shows a clean final screen
sys.stdout.write("\033[2J\033[H")
sys.stdout.flush()

# Reprint header
logger.info("━" * 70)
logger.info("Starting proxy on %s:%s", args.host, args.port)
logger.info("Proxy API Key: %s", key_display)
logger.info("GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
logger.info("━" * 70)
logger.info("Server ready in %.2fs (%d providers discovered in %.2fs)", _elapsed, _plugin_count, _provider_time)


# Note: Debug logging will be added after logging configuration below

# --- Logging Configuration ---
# Import path utilities here (after loading screen) to avoid triggering heavy imports early
from rotator_library.utils.paths import get_logs_dir, get_data_file
from rotator_library.utils.terminal_utils import clear_screen

LOG_DIR = get_logs_dir(_root_dir)

# Configure a file handler for INFO-level logs and higher
info_file_handler = logging.FileHandler(LOG_DIR / "proxy.log", encoding="utf-8")
info_file_handler.setLevel(logging.INFO)
info_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

# Configure a dedicated file handler for all DEBUG-level logs
debug_file_handler = logging.FileHandler(LOG_DIR / "proxy_debug.log", encoding="utf-8")
debug_file_handler.setLevel(logging.DEBUG)
debug_file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)

debug_file_handler.addFilter(RotatorDebugFilter())

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
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
# Note: PROXY_API_KEY validation moved to server startup to allow credential tool to run first
# Pre-build Bearer string once to avoid f-string on every request
_BEARER_PROXY_API_KEY = f"Bearer {PROXY_API_KEY}" if PROXY_API_KEY else None

# Cache OVERRIDE_TEMPERATURE_ZERO at module load time (stored on app.state during lifespan)
OVERRIDE_TEMP_ZERO = os.getenv("OVERRIDE_TEMPERATURE_ZERO", "false").lower()

# Discover API keys from environment variables
api_keys = {}
for key, value in os.environ.items():
    if "_API_KEY" in key and key != "PROXY_API_KEY":
        # Parse provider name from key like KILOCODE_API_KEY or KILOCODE_API_KEY_1
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

# Load model ignore lists from environment variables
ignore_models = {}
for key, value in os.environ.items():
    if key.startswith("IGNORE_MODELS_"):
        provider = key.replace("IGNORE_MODELS_", "").lower()
        models_to_ignore = [
            model.strip() for model in value.split(",") if model.strip()
        ]
        ignore_models[provider] = models_to_ignore
        logger.debug(
            f"Loaded ignore list for provider '{provider}': {models_to_ignore}"
        )

# Load model whitelist from environment variables
whitelist_models = {}
for key, value in os.environ.items():
    if key.startswith("WHITELIST_MODELS_"):
        provider = key.replace("WHITELIST_MODELS_", "").lower()
        models_to_whitelist = [
            model.strip() for model in value.split(",") if model.strip()
        ]
        whitelist_models[provider] = models_to_whitelist
        logger.debug(
            f"Loaded whitelist for provider '{provider}': {models_to_whitelist}"
        )

# Load max concurrent requests per key from environment variables
max_concurrent_requests_per_key = {}
for key, value in os.environ.items():
    if key.startswith("MAX_CONCURRENT_REQUESTS_PER_KEY_"):
        provider = key.replace("MAX_CONCURRENT_REQUESTS_PER_KEY_", "").lower()
        try:
            max_concurrent = int(value)
            if max_concurrent < 1:
                logger.warning(
                    f"Invalid max_concurrent value for provider '{provider}': {value}. Must be >= 1. Using default (1)."
                )
                max_concurrent = 1
            max_concurrent_requests_per_key[provider] = max_concurrent
            logger.debug(
                f"Loaded max concurrent requests for provider '{provider}': {max_concurrent}"
            )
        except ValueError:
            logger.warning(
                f"Invalid max_concurrent value for provider '{provider}': {value}. Using default (1)."
            )


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the RotatingClient's lifecycle with the app's lifespan."""
    # Startup guard: warn if PROXY_API_KEY is missing and auth not explicitly disabled
    if not os.getenv("PROXY_API_KEY") and os.getenv("ALLOW_NO_AUTH", "").lower() != "true":
        logger.warning("=" * 70)
        logger.warning("SECURITY: PROXY_API_KEY is not set and ALLOW_NO_AUTH is not enabled!")
        logger.warning("Your proxy is running WITHOUT authentication — anyone can access it.")
        logger.warning("Set PROXY_API_KEY in .env or set ALLOW_NO_AUTH=true to suppress this warning.")
        logger.warning("=" * 70)

    # Suppress noisy ConnectionResetError from Windows ProactorEventLoop
    # High-TPS providers (fireworks, friendli) forcefully close connections
    # after streaming, causing socket.shutdown() to throw in cleanup callbacks.
    # Scope: only suppress transport-level errors (proactor/send/socket context),
    # not arbitrary business logic errors that happen to be ConnectionResetError.
    _original_handler = None
    if sys.platform == "win32":
        loop = asyncio.get_running_loop()
        _original_handler = loop.get_exception_handler()

        def _suppress_connection_reset(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
                msg = str(exc).lower()
                # Only suppress transport-level resets, not provider auth failures
                context_msg = context.get("message", "").lower()
                if ("send" in msg or "socket" in msg
                        or "transport" in context_msg or "proactor" in context_msg
                        or "fatal write error" in context_msg
                        or "write error" in context_msg):
                    return  # Disconnected client / transport cleanup
            if _original_handler:
                _original_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        loop.set_exception_handler(_suppress_connection_reset)
    # [MODIFIED] Perform skippable OAuth initialization at startup
    skip_oauth_init = os.getenv("SKIP_OAUTH_INIT_CHECK", "false").lower() == "true"

    # The CredentialManager now handles all discovery, including .env overrides.
    # We pass all environment variables to it for this purpose.
    cred_manager = CredentialManager(os.environ)
    oauth_credentials = cred_manager.discover_and_prepare()

    if not skip_oauth_init and oauth_credentials:
        logger.info("Starting OAuth credential validation and deduplication...")
        processed_emails = {}  # email -> {provider: path}
        credentials_to_initialize = {}  # provider -> [paths]
        final_oauth_credentials = {}

        # --- Pass 1: Pre-initialization Scan & Deduplication ---
        for provider, paths in oauth_credentials.items():
            if provider not in credentials_to_initialize:
                credentials_to_initialize[provider] = []
            for path in paths:
                # Skip env-based credentials (virtual paths) - they don't have metadata files
                if path.startswith("env://"):
                    credentials_to_initialize[provider].append(path)
                    continue

                try:
                    def _read_json(p):
                        with open(p, "r", encoding="utf-8") as f:
                            return orjson.loads(f.read())

                    data = await asyncio.get_running_loop().run_in_executor(None, _read_json, path)
                    metadata = data.get("_proxy_metadata", {})
                    email = metadata.get("email")

                    if email:
                        if email not in processed_emails:
                            processed_emails[email] = {}

                        if provider in processed_emails[email]:
                            original_path = processed_emails[email][provider]
                            logger.warning(
                                f"Duplicate for '{email}' on '{provider}' found in pre-scan: '{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                            )
                            continue
                        else:
                            processed_emails[email][provider] = path

                    credentials_to_initialize[provider].append(path)

                except (FileNotFoundError, orjson.JSONDecodeError) as e:
                    logger.warning(
                        f"Could not pre-read metadata from '{path}': {e}. Will process during initialization."
                    )
                    credentials_to_initialize[provider].append(path)

        # --- Pass 2: Parallel Initialization of Filtered Credentials ---
        async def process_credential(provider: str, path: str, provider_instance):
            """Process a single credential: initialize and fetch user info."""
            try:
                await provider_instance.initialize_token(path)

                if not hasattr(provider_instance, "get_user_info"):
                    return (provider, path, None, None)

                user_info = await provider_instance.get_user_info(path)
                email = user_info.get("email")
                return (provider, path, email, None)

            except Exception as e:
                logger.error(
                    f"Failed to process OAuth token for {provider} at '{path}': {e}"
                )
                return (provider, path, None, e)

        # Collect all tasks for parallel execution
        tasks = []
        for provider, paths in credentials_to_initialize.items():
            if not paths:
                continue

            provider_plugin_class = PROVIDER_PLUGINS.get(provider)
            if not provider_plugin_class:
                continue

            provider_instance = provider_plugin_class()

            if hasattr(provider_instance, "preload_credentials"):
                await provider_instance.preload_credentials(paths)

            for path in paths:
                tasks.append(process_credential(provider, path, provider_instance))

        # Execute all credential processing tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # --- Pass 3: Sequential Deduplication and Final Assembly ---
        for result in results:
            # Handle exceptions from gather
            if isinstance(result, Exception):
                logger.error(f"Credential processing raised exception: {result}")
                continue

            provider, path, email, error = result

            # Skip if there was an error
            if error:
                continue

            # If provider doesn't support get_user_info, add directly
            if email is None:
                if provider not in final_oauth_credentials:
                    final_oauth_credentials[provider] = []
                final_oauth_credentials[provider].append(path)
                continue

            # Handle empty email
            if not email:
                logger.warning(
                    f"Could not retrieve email for '{path}'. Treating as unique."
                )
                if provider not in final_oauth_credentials:
                    final_oauth_credentials[provider] = []
                final_oauth_credentials[provider].append(path)
                continue

            # Deduplication check
            if email not in processed_emails:
                processed_emails[email] = {}

            if (
                provider in processed_emails[email]
                and processed_emails[email][provider] != path
            ):
                original_path = processed_emails[email][provider]
                logger.warning(
                    f"Duplicate for '{email}' on '{provider}' found post-init: '{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                )
                continue
            else:
                processed_emails[email][provider] = path
                if provider not in final_oauth_credentials:
                    final_oauth_credentials[provider] = []
                final_oauth_credentials[provider].append(path)

                # Update metadata (skip for env-based credentials - they don't have files)
                if not path.startswith("env://"):
                    def _update_metadata(p, eml, ts):
                        with open(p, "r+", encoding="utf-8") as f:
                            data = json.load(f)
                            metadata = data.get("_proxy_metadata", {})
                            metadata["email"] = eml
                            metadata["last_check_timestamp"] = ts
                            data["_proxy_metadata"] = metadata
                            f.seek(0)
                            json.dump(data, f, indent=2)
                            f.truncate()

                    try:
                        await asyncio.get_running_loop().run_in_executor(
                            None, _update_metadata, path, email, time.time()
                        )
                    except Exception as e:
                        logger.error(f"Failed to update metadata for '{path}': {e}")

        logger.info("OAuth credential processing complete.")
        oauth_credentials = final_oauth_credentials

    # [NEW] Load provider-specific params
    litellm_provider_params = {
        "gemini_cli": {"project_id": os.getenv("GEMINI_CLI_PROJECT_ID")}
    }

    # Load global timeout from environment (default 30 seconds)
    global_timeout = int(os.getenv("GLOBAL_TIMEOUT", "30"))
    if global_timeout < 5:
        logger.warning("GLOBAL_TIMEOUT=%d is too low, clamping to 5", global_timeout)
        global_timeout = 5
    elif global_timeout > 600:
        logger.warning("GLOBAL_TIMEOUT=%d is too high, clamping to 600", global_timeout)
        global_timeout = 600

    # The client now uses the root logger configuration
    client = RotatingClient(
        api_keys=api_keys,
        oauth_credentials=oauth_credentials,  # Pass OAuth config
        configure_logging=True,
        global_timeout=global_timeout,
        litellm_provider_params=litellm_provider_params,
        ignore_models=ignore_models,
        whitelist_models=whitelist_models,
        enable_request_logging=ENABLE_REQUEST_LOGGING,
        max_concurrent_requests_per_key=max_concurrent_requests_per_key,
    )

    # [OPTIMIZED] Parallel initialization of HTTP pool, model info service, and background refresher
    # This reduces startup time by ~200-500ms compared to sequential execution
    async def init_http_pool():
        """Initialize HTTP pool with pre-warmed connections."""
        await client._ensure_http_pool()
        return len(client._provider_endpoints)

    async def init_model_info():
        """Initialize model info service."""
        return await init_model_info_service()

    # Run HTTP pool init and model info service in parallel
    init_results = await asyncio.gather(
        init_http_pool(),
        init_model_info(),
        return_exceptions=True,
    )

    endpoint_count = (
        init_results[0] if not isinstance(init_results[0], Exception) else 0
    )
    model_info_service = (
        init_results[1] if not isinstance(init_results[1], Exception) else None
    )

    if not isinstance(init_results[0], Exception):
        logger.info(f"HTTP pool initialized with {endpoint_count} endpoints")

    # Log loaded credentials summary (compact, always visible for deployment verification)
    client.background_refresher.start()  # Start the background task
    app.state.rotating_client = client
    app.state.active_streams = 0
    app.state.proxy_api_key = PROXY_API_KEY
    app.state.bearer_proxy_api_key = _BEARER_PROXY_API_KEY
    app.state.override_temp_zero = OVERRIDE_TEMP_ZERO
    app.state.enable_raw_logging = ENABLE_RAW_LOGGING
    app.state.use_embedding_batcher = USE_EMBEDDING_BATCHER

    # Warn if no provider credentials are configured
    if not client.all_credentials:
        logger.warning("=" * 70)
        logger.warning("NO PROVIDER CREDENTIALS CONFIGURED")
        logger.warning("The proxy is running but cannot serve any LLM requests.")
        logger.warning(
            "Launch the credential tool to add API keys or OAuth credentials."
        )
        logger.warning("  * Executable: Run with --add-credential flag")
        logger.warning("  * Source: python src/proxy_app/main.py --add-credential")
        logger.warning("=" * 70)

    os.environ["LITELLM_LOG"] = "ERROR"
    litellm.set_verbose = False
    if USE_EMBEDDING_BATCHER:
        batcher = EmbeddingBatcher(client=client)
        app.state.embedding_batcher = batcher
        logger.info("RotatingClient and EmbeddingBatcher initialized.")
    else:
        app.state.embedding_batcher = None
        logger.info("RotatingClient initialized (EmbeddingBatcher disabled).")

    app.state.model_info_service = model_info_service
    if model_info_service:
        logger.info(
            "Model info service started (fetching pricing data in background)."
        )

    yield

    # Restore original exception handler on shutdown
    if sys.platform == "win32" and _original_handler is not None:
        try:
            loop = asyncio.get_running_loop()
            loop.set_exception_handler(_original_handler)
        except RuntimeError:
            pass

    # Grace period: allow in-flight streaming responses to complete
    try:
        logger.info("Shutdown requested, waiting up to 5s for active streams...")
        for _ in range(50):
            if not getattr(app.state, "active_streams", 0):
                break
            await asyncio.sleep(0.1)
        remaining = getattr(app.state, "active_streams", 0)
        if remaining:
            logger.warning("Cancelling %d remaining active streams", remaining)
            # Cancel remaining in-flight stream generators
            active_stream_gens = getattr(app.state, "active_stream_gens", None)
            if active_stream_gens:
                for stream_gen in list(active_stream_gens):
                    try:
                        if hasattr(stream_gen, "aclose"):
                            await stream_gen.aclose()
                    except Exception as e:
                        logger.debug("Error during stream cleanup: %s", e)
                active_stream_gens.clear()
    except Exception as e:
        logger.debug("Error waiting for active streams during shutdown: %s", e)

    await client.background_refresher.stop()  # Stop the background task on shutdown
    close_doh_client()  # Close persistent DoH httpx.Client
    close_dns_executor()  # Shutdown DNS thread pool
    if app.state.embedding_batcher:
        await app.state.embedding_batcher.stop()
    await client.close()  # Also calls close_http_pool()

    # Close litellm's internal aiohttp/httpx sessions to prevent
    # "Unclosed client session" warnings on shutdown
    try:
        await litellm.close_litellm_async_clients()
    except Exception as e:
        logger.debug("Error closing litellm async clients: %s", e)

    # Clear litellm's internal httpx handler cache (creates unclosed sessions)
    try:
        from litellm.llms import custom_httpx as _custom_httpx
        _handler = getattr(_custom_httpx, "httpx_handler", None)
        if _handler is not None:
            for _attr in ("_async_client", "_client", "client", "async_client"):
                _obj = getattr(_handler, _attr, None)
                if _obj is not None:
                    if hasattr(_obj, "aclose"):
                        await _obj.aclose()
                    elif hasattr(_obj, "close"):
                        _obj.close()
            _custom_httpx.httpx_handler = None
    except Exception as e:
        logger.debug("Error clearing custom_httpx handler: %s", e)

    if hasattr(litellm, "aclient_session") and litellm.aclient_session is not None:
        try:
            await litellm.aclient_session.aclose()
            litellm.aclient_session = None
        except Exception as e:
            logger.debug("Error closing litellm aclient_session: %s", e)
    if hasattr(litellm, "client_session") and litellm.client_session is not None:
        try:
            litellm.client_session.close()
            litellm.client_session = None
        except Exception as e:
            logger.debug("Error closing litellm client_session: %s", e)

    # Stop model info service
    if hasattr(app.state, "model_info_service") and app.state.model_info_service:
        await app.state.model_info_service.stop()

    if app.state.embedding_batcher:
        logger.info("RotatingClient and EmbeddingBatcher closed.")
    else:
        logger.info("RotatingClient closed.")


# --- FastAPI App Setup ---

app = FastAPI(lifespan=lifespan)

# Add CORS middleware with env-configured origins
_cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "")
if _cors_origins_env.strip():
    _cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins == ["*"]:
        logger.warning("CORS_ALLOWED_ORIGINS='*' — allowing all origins (dev-mode only, restrict in production!)")
else:
    logger.warning("CORS_ALLOWED_ORIGINS not set — defaulting to same-origin only (set CORS_ALLOWED_ORIGINS to allow specific origins)")
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
app.add_middleware(_NoGzipForSSE, minimum_size=1000)

# Security headers: X-Content-Type-Options, X-Frame-Options, Referrer-Policy
app.add_middleware(SecurityHeadersMiddleware)


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
            # Re-read PROXY_API_KEY from environment
            PROXY_API_KEY = os.getenv("PROXY_API_KEY")

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
                _signal.signal(_signal.SIGBREAK, lambda *_: _signal.raise_signal(_signal.SIGINT))

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            limit_concurrency=int(os.getenv("MAX_CONCURRENT_REQUESTS", "1000")),
            backlog=2048,
        )
