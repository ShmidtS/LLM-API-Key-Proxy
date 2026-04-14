# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

# Disable aiodns BEFORE any aiohttp/litellm imports to fix DNS resolution issues
# This must be set before aiohttp is imported anywhere in the process
# See: https://github.com/aio-libs/aiohttp/issues/1135
# When aiodns is installed, aiohttp uses it by default but it may fail to resolve
# domains that work fine with system DNS (ping works but aiohttp fails)
import os
import sys

# Disable aiodns C extensions on Windows only (breaks DNS resolution there).
# Linux/macOS keep C extensions for performance.
if sys.platform == "win32":
    os.environ["AIOHTTP_NO_EXTENSIONS"] = "1"

import time
import orjson

# Phase 1: Minimal imports for arg parsing and TUI
import asyncio
from pathlib import Path
import argparse
import logging
import re

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# --- Argument Parsing (BEFORE heavy imports) ---
parser = argparse.ArgumentParser(description="API Key Proxy Server")
parser.add_argument(
    "--host", type=str, default="0.0.0.0", help="Host to bind the server to."
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
from dotenv import load_dotenv
from glob import glob

# Get the application root directory (EXE dir if frozen, else CWD)
# Inlined here to avoid triggering heavy rotator_library imports before loading screen
if getattr(sys, "frozen", False):
    _root_dir = Path(sys.executable).parent
else:
    _root_dir = Path.cwd()

# Load main .env first
load_dotenv(_root_dir / ".env")

# Load any additional .env files (e.g., antigravity_all_combined.env, gemini_cli_all_combined.env)
_env_files_found = list(_root_dir.glob("*.env"))
for _env_file in sorted(_env_files_found):  # reuse already-computed list
    if _env_file.name != ".env":  # Skip main .env (already loaded)
        load_dotenv(_env_file, override=False)  # Don't override existing values

# Log discovered .env files for deployment verification
if _env_files_found:
    _env_names = [_ef.name for _ef in _env_files_found]
    print(f"📁 Loaded {len(_env_files_found)} .env file(s): {', '.join(_env_names)}")

# Get proxy API key for display
proxy_api_key = os.getenv("PROXY_API_KEY")
if proxy_api_key:
    key_display = f"✓ {proxy_api_key}"
else:
    key_display = "✗ Not Set (INSECURE - anyone can access!)"

print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print(f"GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
print("━" * 70)
print("Loading server components...")


# Phase 2: Load Rich for loading spinner (lightweight)
from rich.console import Console

_console = Console()

# Phase 3: Heavy dependencies with granular loading messages
print("  → Loading FastAPI framework...")
with _console.status("[dim]Loading FastAPI framework...", spinner="dots"):
    from contextlib import asynccontextmanager
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse, JSONResponse
    from fastapi.security import APIKeyHeader

print("  → Loading core dependencies...")
with _console.status("[dim]Loading core dependencies...", spinner="dots"):
    from dotenv import load_dotenv
    import colorlog
    import json
    from typing import AsyncGenerator, Any, List, Optional, Union

    # --- Early Log Level Configuration ---
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

print(" → Loading LiteLLM library...")
with _console.status("[dim]Loading LiteLLM library...", spinner="dots"):
    import litellm
    import httpx

    # CRITICAL: Apply SSL patches IMMEDIATELY after litellm import
    # This must happen BEFORE rotator_library import and BEFORE any API calls

    _ssl_verify_env = os.environ.get("HTTP_SSL_VERIFY", "true").lower()
    if _ssl_verify_env == "false":
        print("[SSL-FIX-MAIN] HTTP_SSL_VERIFY=false - Applying SSL patches in main.py")

        # 1. Set litellm's SSL verification to False
        litellm.ssl_verify = False
        print(f"[SSL-FIX-MAIN] Set litellm.ssl_verify = False")

        # 2. Create pre-configured httpx clients with SSL verification disabled
        # This is the MOST RELIABLE way to disable SSL in litellm
        litellm.client_session = httpx.Client(verify=False)
        litellm.aclient_session = httpx.AsyncClient(verify=False)
        print(
            f"[SSL-FIX-MAIN] Created litellm.client_session and aclient_session with verify=False"
        )

        # 3. Set environment variable for litellm
        os.environ["SSL_VERIFY"] = "False"
        print(f"[SSL-FIX-MAIN] Set SSL_VERIFY=False environment variable")

# Phase 4: Application imports with granular loading messages
print("  → Initializing proxy core...")
with _console.status("[dim]Initializing proxy core...", spinner="dots"):
    from rotator_library import RotatingClient, STREAM_DONE
    from rotator_library.credential_manager import CredentialManager
    from rotator_library.background_refresher import BackgroundRefresher
    from rotator_library.dns_fix import close_doh_client
    from rotator_library.http_client_pool import close_http_pool
    from rotator_library.model_info_service import init_model_info_service
    from proxy_app.request_logger import log_request_to_console
    from proxy_app.batch_manager import EmbeddingBatcher
    from proxy_app.detailed_logger import RawIOLogger

# Import extracted modules
from proxy_app.models import (
    EmbeddingRequest,
    ModelCard,
    ModelCapabilities,
    EnrichedModelCard,
    ModelList,
    EnrichedModelList,
)
from proxy_app.middleware import _NoGzipForSSE, RotatorDebugFilter, NoLiteLLMLogFilter
from proxy_app.dependencies import (
    api_key_header,
    get_rotating_client,
    get_embedding_batcher,
    verify_api_key,
    verify_anthropic_api_key,
    _streams_lock,
)
from proxy_app.streaming import streaming_response_wrapper, handle_litellm_error

# Anthropic API Models (imported from library)
from rotator_library.anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
)

print("  → Discovering provider plugins...")
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
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)

# Clear screen and reprint header for clean startup view
# This pushes loading messages up (still in scroll history) but shows a clean final screen
os.system("cls" if os.name == "nt" else "clear")

# Reprint header
print("━" * 70)
print(f"Starting proxy on {args.host}:{args.port}")
print(f"Proxy API Key: {key_display}")
print(f"GitHub: https://github.com/ShmidtS/LLM-API-Key-Proxy")
print("━" * 70)
print(
    f"✓ Server ready in {_elapsed:.2f}s ({_plugin_count} providers discovered in {_provider_time:.2f}s)"
)


# Note: Debug logging will be added after logging configuration below

# --- Logging Configuration ---
# Import path utilities here (after loading screen) to avoid triggering heavy imports early
from rotator_library.utils.paths import get_logs_dir, get_data_file
from rotator_library.utils.json_utils import sse_data_event
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
logging.debug(f"Modules loaded in {_elapsed:.2f}s")


# --- Configuration ---
USE_EMBEDDING_BATCHER = False
ENABLE_REQUEST_LOGGING = args.enable_request_logging
ENABLE_RAW_LOGGING = args.enable_raw_logging
if ENABLE_REQUEST_LOGGING:
    logging.info(
        "Transaction logging is enabled (library-level with provider correlation)."
    )
if ENABLE_RAW_LOGGING:
    logging.info("Raw I/O logging is enabled (proxy boundary, unmodified HTTP data).")
PROXY_API_KEY = os.getenv("PROXY_API_KEY")
# Note: PROXY_API_KEY validation moved to server startup to allow credential tool to run first
# Pre-build Bearer string once to avoid f-string on every request
_BEARER_PROXY_API_KEY = f"Bearer {PROXY_API_KEY}" if PROXY_API_KEY else None

# Inject API key config into dependencies module so route handlers can use it
from proxy_app import dependencies as _deps

_deps.PROXY_API_KEY = PROXY_API_KEY
_deps._BEARER_PROXY_API_KEY = _BEARER_PROXY_API_KEY

# Cache OVERRIDE_TEMPERATURE_ZERO at module load time (called on every request otherwise)
OVERRIDE_TEMP_ZERO = os.getenv("OVERRIDE_TEMPERATURE_ZERO", "false").lower()

# Inject into chat route module
from proxy_app.routes import chat as _chat_mod

_chat_mod.OVERRIDE_TEMP_ZERO = OVERRIDE_TEMP_ZERO
_chat_mod.ENABLE_RAW_LOGGING = ENABLE_RAW_LOGGING

# Inject into embeddings route module
from proxy_app.routes import embeddings as _emb_mod

_emb_mod.USE_EMBEDDING_BATCHER = USE_EMBEDDING_BATCHER

# Inject into anthropic route module
from proxy_app.routes import anthropic as _anthro_mod

_anthro_mod.ENABLE_RAW_LOGGING = ENABLE_RAW_LOGGING

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
        logging.debug(
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
        logging.debug(
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
                logging.warning(
                    f"Invalid max_concurrent value for provider '{provider}': {value}. Must be >= 1. Using default (1)."
                )
                max_concurrent = 1
            max_concurrent_requests_per_key[provider] = max_concurrent
            logging.debug(
                f"Loaded max concurrent requests for provider '{provider}': {max_concurrent}"
            )
        except ValueError:
            logging.warning(
                f"Invalid max_concurrent value for provider '{provider}': {value}. Using default (1)."
            )


# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage the RotatingClient's lifecycle with the app's lifespan."""
    # Suppress noisy ConnectionResetError from Windows ProactorEventLoop
    # High-TPS providers (fireworks, friendli) forcefully close connections
    # after streaming, causing socket.shutdown() to throw in cleanup callbacks.
    if sys.platform == "win32":
        loop = asyncio.get_running_loop()

        def _suppress_connection_reset(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
                msg = str(exc).lower()
                if "send" in msg or "socket" in msg:
                    return  # Disconnected client, not a provider auth failure
            loop.default_exception_handler(context)

        loop.set_exception_handler(_suppress_connection_reset)
    # [MODIFIED] Perform skippable OAuth initialization at startup
    skip_oauth_init = os.getenv("SKIP_OAUTH_INIT_CHECK", "false").lower() == "true"

    # The CredentialManager now handles all discovery, including .env overrides.
    # We pass all environment variables to it for this purpose.
    cred_manager = CredentialManager(os.environ)
    oauth_credentials = cred_manager.discover_and_prepare()

    if not skip_oauth_init and oauth_credentials:
        logging.info("Starting OAuth credential validation and deduplication...")
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
                    with open(path, "r") as f:
                        data = json.load(f)
                    metadata = data.get("_proxy_metadata", {})
                    email = metadata.get("email")

                    if email:
                        if email not in processed_emails:
                            processed_emails[email] = {}

                        if provider in processed_emails[email]:
                            original_path = processed_emails[email][provider]
                            logging.warning(
                                f"Duplicate for '{email}' on '{provider}' found in pre-scan: '{Path(path).name}'. Original: '{Path(original_path).name}'. Skipping."
                            )
                            continue
                        else:
                            processed_emails[email][provider] = path

                    credentials_to_initialize[provider].append(path)

                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logging.warning(
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
                logging.error(
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
                logging.error(f"Credential processing raised exception: {result}")
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
                logging.warning(
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
                logging.warning(
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
                    try:
                        with open(path, "r+") as f:
                            data = json.load(f)
                            metadata = data.get("_proxy_metadata", {})
                            metadata["email"] = email
                            metadata["last_check_timestamp"] = time.time()
                            data["_proxy_metadata"] = metadata
                            f.seek(0)
                            json.dump(data, f, indent=2)
                            f.truncate()
                    except Exception as e:
                        logging.error(f"Failed to update metadata for '{path}': {e}")

        logging.info("OAuth credential processing complete.")
        oauth_credentials = final_oauth_credentials

    # [NEW] Load provider-specific params
    litellm_provider_params = {
        "gemini_cli": {"project_id": os.getenv("GEMINI_CLI_PROJECT_ID")}
    }

    # Load global timeout from environment (default 30 seconds)
    global_timeout = int(os.getenv("GLOBAL_TIMEOUT", "30"))

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
        endpoints = client._get_provider_endpoints()
        await client._ensure_http_pool()
        return len(endpoints)

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
        logging.info(f"HTTP pool initialized with {endpoint_count} endpoints")

    # Log loaded credentials summary (compact, always visible for deployment verification)
    client.background_refresher.start()  # Start the background task
    app.state.rotating_client = client
    app.state.active_streams = 0

    # Warn if no provider credentials are configured
    if not client.all_credentials:
        logging.warning("=" * 70)
        logging.warning("NO PROVIDER CREDENTIALS CONFIGURED")
        logging.warning("The proxy is running but cannot serve any LLM requests.")
        logging.warning(
            "Launch the credential tool to add API keys or OAuth credentials."
        )
        logging.warning("  * Executable: Run with --add-credential flag")
        logging.warning("  * Source: python src/proxy_app/main.py --add-credential")
        logging.warning("=" * 70)

    os.environ["LITELLM_LOG"] = "ERROR"
    litellm.set_verbose = False
    litellm.drop_params = True
    if USE_EMBEDDING_BATCHER:
        batcher = EmbeddingBatcher(client=client)
        app.state.embedding_batcher = batcher
        logging.info("RotatingClient and EmbeddingBatcher initialized.")
    else:
        app.state.embedding_batcher = None
        logging.info("RotatingClient initialized (EmbeddingBatcher disabled).")

    app.state.model_info_service = model_info_service
    if model_info_service:
        logging.info(
            "Model info service started (fetching pricing data in background)."
        )

    yield

    # Grace period: allow in-flight streaming responses to complete
    try:
        logging.info("Shutdown requested, waiting up to 5s for active streams...")
        for _ in range(50):
            with _streams_lock:
                if not getattr(app.state, "active_streams", 0):
                    break
            await asyncio.sleep(0.1)
        with _streams_lock:
            remaining = getattr(app.state, "active_streams", 0)
        if remaining:
            logging.warning("Forcing shutdown with %d active streams", remaining)
    except Exception:
        pass

    await client.background_refresher.stop()  # Stop the background task on shutdown
    close_doh_client()  # Close persistent DoH httpx.Client
    if app.state.embedding_batcher:
        await app.state.embedding_batcher.stop()
    await client.close()
    await close_http_pool()

    # Close litellm's internal aiohttp/httpx sessions to prevent
    # "Unclosed client session" warnings on shutdown
    try:
        await litellm.close_litellm_async_clients()
    except Exception:
        pass
    if hasattr(litellm, "aclient_session") and litellm.aclient_session is not None:
        try:
            await litellm.aclient_session.aclose()
        except Exception:
            pass
    if hasattr(litellm, "client_session") and litellm.client_session is not None:
        try:
            litellm.client_session.close()
        except Exception:
            pass

    # Stop model info service
    if hasattr(app.state, "model_info_service") and app.state.model_info_service:
        await app.state.model_info_service.stop()

    if app.state.embedding_batcher:
        logging.info("RotatingClient and EmbeddingBatcher closed.")
    else:
        logging.info("RotatingClient closed.")


# --- FastAPI App Setup ---

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow all origins, methods, and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# SSE-aware gzip: compresses non-streaming responses >= minimum_size, passes SSE through raw
app.add_middleware(_NoGzipForSSE, minimum_size=1000)


# --- Register route modules ---
from proxy_app.routes import all_routers

for _router in all_routers:
    app.include_router(_router)


@app.get("/")
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

        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            limit_concurrency=1000,
            backlog=2048,
        )
