# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

# Disable aiodns BEFORE any aiohttp/litellm imports to fix DNS resolution issues
# This must be set before aiohttp is imported anywhere in the process
# See: https://github.com/aio-libs/aiohttp/issues/1135
# When aiodns is installed, aiohttp uses it by default but it may fail to resolve
# domains that work fine with system DNS (ping works but aiohttp fails)
import os

# Always disable aiodns to use system DNS resolver
os.environ["AIOHTTP_NO_EXTENSIONS"] = "1"

import time
import orjson

# Phase 1: Minimal imports for arg parsing and TUI
import asyncio
from pathlib import Path
import sys
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

_models_cache: dict = {"data": None, "expires": 0.0}

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
    from pydantic import BaseModel, ConfigDict, Field

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


# --- Pydantic Models ---
class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    input_type: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None


class ModelCard(BaseModel):
    """Basic model card for minimal response."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "Mirro-Proxy"


class ModelCapabilities(BaseModel):
    """Model capability flags."""

    tool_choice: bool = False
    function_calling: bool = False
    reasoning: bool = False
    vision: bool = False
    system_messages: bool = True
    prompt_caching: bool = False
    assistant_prefill: bool = False


class EnrichedModelCard(BaseModel):
    """Extended model card with pricing and capabilities."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "unknown"
    # Pricing (optional - may not be available for all models)
    input_cost_per_token: Optional[float] = None
    output_cost_per_token: Optional[float] = None
    cache_read_input_token_cost: Optional[float] = None
    cache_creation_input_token_cost: Optional[float] = None
    # Limits (optional)
    max_input_tokens: Optional[int] = None
    max_output_tokens: Optional[int] = None
    context_window: Optional[int] = None
    # Capabilities
    mode: str = "chat"
    supported_modalities: List[str] = Field(default_factory=lambda: ["text"])
    supported_output_modalities: List[str] = Field(default_factory=lambda: ["text"])
    capabilities: Optional[ModelCapabilities] = None
    # Debug info (optional)
    _sources: Optional[List[str]] = None
    _match_type: Optional[str] = None

    model_config = ConfigDict(extra="allow")  # Allow extra fields from the service


class ModelList(BaseModel):
    """List of models response."""

    object: str = "list"
    data: List[ModelCard]


class EnrichedModelList(BaseModel):
    """List of enriched models with pricing and capabilities."""

    object: str = "list"
    data: List[EnrichedModelCard]


# --- Anthropic API Models (imported from library) ---
from rotator_library.anthropic_compat import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
)


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


# Create a filter to ensure the debug handler ONLY gets DEBUG messages from the rotator_library
class RotatorDebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG and record.name.startswith(
            "rotator_library"
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


# Add a filter to prevent any LiteLLM logs from cluttering the console
class NoLiteLLMLogFilter(logging.Filter):
    def filter(self, record):
        return not record.name.startswith("LiteLLM")


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

# Cache OVERRIDE_TEMPERATURE_ZERO at module load time (called on every request otherwise)
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
        loop = asyncio.get_event_loop()

        def _suppress_connection_reset(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
                return  # Silently ignore - remote closed connection, not an error
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


class _NoGzipForSSE:
    """GZip middleware that skips compression for streaming responses.

    Detects streaming via content-type (text/event-stream) or more_body=True
    on the first body chunk. Passes streaming chunks through immediately
    without buffering, avoiding TTFT overhead from GZipMiddleware.
    """

    _ACCEPT_ENCODING = b"accept-encoding"
    _GZIP = b"gzip"

    def __init__(self, app, minimum_size=1000):
        self._app = app
        self._minimum_size = minimum_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        accept_gzip = any(
            _NoGzipForSSE._GZIP in v
            for k, v in scope.get("headers", [])
            if k == _NoGzipForSSE._ACCEPT_ENCODING
        )

        if not accept_gzip:
            await self._app(scope, receive, send)
            return

        import gzip as _gzip

        initial_message = None
        compressor = None
        skip = False
        started = False

        async def _send(message):
            nonlocal initial_message, compressor, skip, started

            if message["type"] == "http.response.start":
                initial_message = message
                headers = message.get("headers", [])
                for h in headers:
                    hk = (h[0].decode("latin-1") if isinstance(h[0], bytes) else h[0]).lower()
                    hv = h[1].decode("latin-1") if isinstance(h[1], bytes) else h[1]
                    if hk == "content-type" and "text/event-stream" in hv:
                        skip = True
                        break
                    if hk == "content-length" and int(hv) < self._minimum_size:
                        skip = True
                        break
                return

            if message["type"] != "http.response.body":
                await send(message)
                return

            body = message.get("body", b"")
            more = message.get("more_body", False)

            if not started:
                started = True
                if skip or more:
                    skip = True
                    await send(initial_message)
                    await send(message)
                    return

                if len(body) < self._minimum_size:
                    await send(initial_message)
                    await send(message)
                    return

                out = _gzip.compress(body, compresslevel=6)
                if len(out) >= len(body):
                    await send(initial_message)
                    await send(message)
                    return

                headers = [
                    (hk, hv)
                    for hk, hv in initial_message.get("headers", [])
                    if (hk.decode("latin-1") if isinstance(hk, bytes) else hk).lower()
                    != "content-length"
                ]
                headers.append((b"content-encoding", b"gzip"))
                headers.append((b"vary", b"accept-encoding"))
                headers.append((b"content-length", str(len(out)).encode()))
                await send({**initial_message, "headers": headers})
                await send({"type": "http.response.body", "body": out, "more_body": False})
                return

            if skip or compressor is None:
                await send(message)
                return

            chunks = []
            if body:
                chunks.append(compressor.compress(body))
            if not more:
                chunks.append(compressor.flush())
                compressor = None
            out = b"".join(chunks)
            await send({"type": "http.response.body", "body": out, "more_body": more})

        await self._app(scope, receive, _send)


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
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def get_rotating_client(request: Request) -> RotatingClient:
    """Dependency to get the rotating client instance from the app state."""
    return request.app.state.rotating_client


def get_embedding_batcher(request: Request) -> EmbeddingBatcher:
    """Dependency to get the embedding batcher instance from the app state."""
    return request.app.state.embedding_batcher


async def verify_api_key(auth: str = Depends(api_key_header)):
    """Dependency to verify the proxy API key."""
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not PROXY_API_KEY:
        return auth
    if not auth or auth != _BEARER_PROXY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return auth


# --- Anthropic API Key Header ---
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_anthropic_api_key(
    x_api_key: str = Depends(anthropic_api_key_header),
    auth: str = Depends(api_key_header),
):
    """
    Dependency to verify API key for Anthropic endpoints.
    Accepts either x-api-key header (Anthropic style) or Authorization Bearer (OpenAI style).
    """
    # If PROXY_API_KEY is not set or empty, skip verification (open access)
    if not PROXY_API_KEY:
        return x_api_key or auth
    # Check x-api-key first (Anthropic style)
    if x_api_key and x_api_key == PROXY_API_KEY:
        return x_api_key
    # Fall back to Bearer token (OpenAI style)
    if auth and auth == _BEARER_PROXY_API_KEY:
        return auth
    raise HTTPException(status_code=401, detail="Invalid or missing API Key")


async def streaming_response_wrapper(
    request: Request,
    request_data: dict,
    response_stream: AsyncGenerator[Any, None],
    logger: Optional[RawIOLogger] = None,
) -> AsyncGenerator[str, None]:
    """
    Wraps a streaming response to log the full response after completion
    and ensures any errors during the stream are sent to the client.

    Receives dicts + STREAM_DONE sentinel from the client pipeline,
    serializes to SSE only at the HTTP yield boundary (single serialize).
    """
    # --- Inline aggregation state (eliminates second pass over response_chunks) ---
    final_message = {"role": "assistant"}
    _content_parts: list = []
    _generic_str_parts: dict = {}
    aggregated_tool_calls = {}  # index -> {type, id, function: {name_parts, args_parts}}
    usage_data = None
    finish_reason = None
    first_chunk_meta = None  # {id, created, model} from first chunk
    _chunk_count = 0
    _last_disconnect_check = time.monotonic()

    try:
        async for chunk in response_stream:
            _chunk_count += 1
            now = time.monotonic()
            if now - _last_disconnect_check > 1.0:
                if await request.is_disconnected():
                    logging.warning("Client disconnected, stopping stream.")
                    break
                _last_disconnect_check = now

            # STREAM_DONE sentinel: emit SSE [DONE] and stop
            if chunk is STREAM_DONE:
                yield b"data: [DONE]\n\n"
                continue

            # chunk is a dict — serialize to SSE only here (single serialize point)
            chunk_str = sse_data_event(chunk)
            yield chunk_str

            if not isinstance(chunk, dict):
                continue

            if logger:
                logger.log_stream_chunk(chunk)
            else:
                continue

            # --- Inline aggregation (single pass, no second iteration) ---
            # Capture metadata from first chunk
            if first_chunk_meta is None:
                first_chunk_meta = {
                    "id": chunk.get("id"),
                    "created": chunk.get("created"),
                    "model": chunk.get("model"),
                }

            if "choices" in chunk and chunk["choices"]:
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})

                for key, value in delta.items():
                    if value is None:
                        continue

                    if key == "content":
                        if value:
                            _content_parts.append(value)

                    elif key == "tool_calls":
                        for tc_chunk in value:
                            index = tc_chunk["index"]
                            if index not in aggregated_tool_calls:
                                aggregated_tool_calls[index] = {
                                    "type": "function",
                                    "function": {
                                        "name_parts": [],
                                        "args_parts": [],
                                    },
                                }
                            tc = aggregated_tool_calls[index]
                            if tc_chunk.get("id"):
                                tc["id"] = tc_chunk["id"]
                            if "function" in tc_chunk:
                                fn = tc_chunk["function"]
                                if fn.get("name") is not None:
                                    tc["function"]["name_parts"].append(fn["name"])
                                if fn.get("arguments") is not None:
                                    tc["function"]["args_parts"].append(
                                        fn["arguments"]
                                    )

                    elif key == "function_call":
                        if "function_call" not in final_message:
                            final_message["function_call"] = {
                                "_name_parts": [],
                                "_args_parts": [],
                            }
                        if value.get("name") is not None:
                            final_message["function_call"]["_name_parts"].append(
                                value["name"]
                            )
                        if value.get("arguments") is not None:
                            final_message["function_call"]["_args_parts"].append(
                                value["arguments"]
                            )

                    else:  # Generic key handling for other data like 'reasoning'
                        if key == "role":
                            final_message[key] = value
                        elif isinstance(value, str):
                            if key not in _generic_str_parts:
                                _generic_str_parts[key] = []
                            _generic_str_parts[key].append(value)
                        else:
                            final_message[key] = value

                if "finish_reason" in choice and choice["finish_reason"]:
                    finish_reason = choice["finish_reason"]

            if "usage" in chunk and chunk["usage"]:
                usage_data = chunk["usage"]
    except Exception as e:
        logging.error(f"An error occurred during the response stream: {e}")
        # Yield a final error message to the client to ensure they are not left hanging.
        error_payload = {
            "error": {
                "message": f"An unexpected error occurred during the stream: {str(e)}",
                "type": "proxy_internal_error",
                "code": 500,
            }
        }
        yield sse_data_event(error_payload)
        yield b"data: [DONE]\n\n"
        # Also log this as a failed request
        if logger:
            logger.log_final_response(
                status_code=500, headers=None, body={"error": str(e)}
            )
        return  # Stop further processing
    finally:
        if logger:
            if first_chunk_meta is not None:
                # --- Join accumulated string parts ---
                if _content_parts:
                    final_message["content"] = "".join(_content_parts)

                for key, parts in _generic_str_parts.items():
                    final_message[key] = "".join(parts)

                # Flatten tool_calls: convert name_parts/args_parts to strings
                if aggregated_tool_calls:
                    tool_calls_list = []
                    for tc in aggregated_tool_calls.values():
                        fn = tc["function"]
                        tool_calls_list.append(
                            {
                                "id": tc.get("id"),
                                "type": tc["type"],
                                "function": {
                                    "name": "".join(fn["name_parts"]),
                                    "arguments": "".join(fn["args_parts"]),
                                },
                            }
                        )
                    final_message["tool_calls"] = tool_calls_list
                    # CRITICAL FIX: Override finish_reason when tool_calls exist
                    # This ensures OpenCode and other agentic systems continue the conversation loop
                    finish_reason = "tool_calls"

                if "function_call" in final_message:
                    fc = final_message["function_call"]
                    final_message["function_call"] = {
                        "name": "".join(fc["_name_parts"]),
                        "arguments": "".join(fc["_args_parts"]),
                    }

                # Ensure standard fields are present for consistent logging
                for field in ["content", "tool_calls", "function_call"]:
                    if field not in final_message:
                        final_message[field] = None

                final_choice = {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": finish_reason,
                }

                full_response = {
                    "id": first_chunk_meta.get("id"),
                    "object": "chat.completion",
                    "created": first_chunk_meta.get("created"),
                    "model": first_chunk_meta.get("model"),
                    "choices": [final_choice],
                    "usage": usage_data,
                }
            else:
                full_response = {}

            logger.log_final_response(
                status_code=200,
                headers=None,  # Headers are not available at this stage
                body=full_response,
            )


def handle_litellm_error(e: Exception, error_format: str = "openai") -> HTTPException:
    """Map litellm exceptions to HTTPException with OpenAI or Anthropic error format."""
    _ERROR_MAP = [
        (
            (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError),
            400,
            "Invalid Request",
            "invalid_request_error",
        ),
        ((litellm.AuthenticationError,), 401, "Authentication Error", "authentication_error"),
        ((litellm.RateLimitError,), 429, "Rate Limit Exceeded", "rate_limit_error"),
        (
            (litellm.ServiceUnavailableError, litellm.APIConnectionError),
            503,
            "Service Unavailable",
            "api_error",
        ),
        ((litellm.Timeout,), 504, "Gateway Timeout", "api_error"),
        ((litellm.InternalServerError, litellm.OpenAIError), 502, "Bad Gateway", "api_error"),
    ]

    for exc_types, status_code, openai_label, anthropic_error_type in _ERROR_MAP:
        if isinstance(e, exc_types):
            if error_format == "openai":
                detail = f"{openai_label}: {str(e)}"
            else:
                message = f"Request timed out: {str(e)}" if isinstance(e, litellm.Timeout) else str(e)
                detail = {
                    "type": "error",
                    "error": {"type": anthropic_error_type, "message": message},
                }
            return HTTPException(status_code=status_code, detail=detail)

    # Fallback for unmatched litellm errors
    if error_format == "openai":
        return HTTPException(status_code=500, detail=str(e))
    return HTTPException(
        status_code=500,
        detail={"type": "error", "error": {"type": "api_error", "message": str(e)}},
    )


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses and logs them.
    """
    # Raw I/O logger captures unmodified HTTP data at proxy boundary (disabled by default)
    raw_logger = RawIOLogger() if ENABLE_RAW_LOGGING else None
    request_data: dict = {}
    try:
        # Read and parse the request body only once at the beginning.
        try:
            request_data = orjson.loads(await request.body())
        except orjson.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body.")

        # Global temperature=0 override (controlled by .env variable, default: OFF)
        # Low temperature makes models deterministic and prone to following training data
        # instead of actual schemas, which can cause tool hallucination
        # Modes: "remove" = delete temperature key, "set" = change to 1.0, "false" = disabled
        override_temp_zero = OVERRIDE_TEMP_ZERO

        if (
            override_temp_zero in ("remove", "set", "true", "1", "yes")
            and "temperature" in request_data
            and request_data["temperature"] == 0
        ):
            if override_temp_zero == "remove":
                # Remove temperature key entirely
                del request_data["temperature"]
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=remove: Removed temperature=0 from request"
                )
            else:
                # Set to 1.0 (for "set", "true", "1", "yes")
                request_data["temperature"] = 1.0
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=set: Converting temperature=0 to temperature=1.0"
                )

        # If raw logging is enabled, capture the unmodified request data.
        if raw_logger:
            raw_logger.log_request(headers=request.headers, body=request_data)

        # Extract and log specific reasoning parameters for monitoring.
        model = request_data.get("model")
        generation_cfg = (
            request_data.get("generationConfig", {})
            or request_data.get("generation_config", {})
            or {}
        )
        reasoning_effort = request_data.get("reasoning_effort") or generation_cfg.get(
            "reasoning_effort"
        )

        logging.getLogger("rotator_library").debug(
            f"Handling reasoning parameters: model={model}, reasoning_effort={reasoning_effort}"
        )

        # Log basic request info to console (this is a separate, simpler logger).
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            response_generator = client.acompletion(request=request, **request_data)
            return StreamingResponse(
                streaming_response_wrapper(
                    request, request_data, response_generator, raw_logger
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            response = await client.acompletion(request=request, **request_data)
            if raw_logger:
                # Assuming response has status_code and headers attributes
                # This might need adjustment based on the actual response object
                response_headers = (
                    response.headers if hasattr(response, "headers") else None
                )
                status_code = (
                    response.status_code if hasattr(response, "status_code") else 200
                )
                raw_logger.log_final_response(
                    status_code=status_code,
                    headers=response_headers,
                    body=response.model_dump(),
                )
            return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Request failed after all retries: {e}")
        # Optionally log the failed request (request_data already parsed above)
        if ENABLE_REQUEST_LOGGING:
            if raw_logger:
                raw_logger.log_final_response(
                    status_code=500, headers=None, body={"error": str(e)}
                )
        raise HTTPException(status_code=500, detail=str(e))


# --- Anthropic Messages API Endpoint ---
@app.post("/v1/messages")
async def anthropic_messages(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
):
    """
    Anthropic-compatible Messages API endpoint.

    Accepts requests in Anthropic's format and returns responses in Anthropic's format.
    Internally translates to OpenAI format for processing via LiteLLM.

    This endpoint is compatible with Claude Code and other Anthropic API clients.
    """
    # Initialize raw I/O logger if enabled (for debugging proxy boundary)
    logger = RawIOLogger() if ENABLE_RAW_LOGGING else None

    # Parse raw body once with orjson — avoid double Pydantic validate-then-dump round-trip
    body_data = orjson.loads(await request.body())

    # Construct Pydantic model without validation (data already validated by orjson parsing)
    body = AnthropicMessagesRequest.model_construct(**body_data)

    # Log raw Anthropic request if raw logging is enabled
    if logger:
        logger.log_request(
            headers=dict(request.headers),
            body=body_data,
        )

    try:
        # Log the request to console
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(
                request.client.host if request.client else "unknown",
                request.client.port if request.client else 0,
            ),
            request_data=body_data,
        )

        # Use the library method to handle the request
        result = await client.anthropic_messages(body, raw_request=request, raw_body_data=body_data)

        if body.stream:
            # Streaming response
            return StreamingResponse(
                result,
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            # Non-streaming response
            if logger:
                logger.log_final_response(
                    status_code=200,
                    headers=None,
                    body=result,
                )
            return JSONResponse(content=result)

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError, litellm.ContextWindowExceededError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="anthropic")
        logging.error(f"Anthropic messages endpoint error: {e}")
        if logger:
            logger.log_final_response(
                status_code=500,
                headers=None,
                body={"error": str(e)},
            )
        error_response = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        raise HTTPException(status_code=500, detail=error_response)


# --- Anthropic Count Tokens Endpoint ---
@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(
    request: Request,
    body: AnthropicCountTokensRequest,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_anthropic_api_key),
):
    """
    Anthropic-compatible count_tokens endpoint.

    Counts the number of tokens that would be used by a Messages API request.
    This is useful for estimating costs and managing context windows.

    Accepts requests in Anthropic's format and returns token count in Anthropic's format.
    """
    try:
        # Use the library method to handle the request
        result = await client.anthropic_count_tokens(body)
        return JSONResponse(content=result)

    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        error_response = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": str(e)},
        }
        raise HTTPException(status_code=400, detail=error_response)
    except litellm.AuthenticationError as e:
        error_response = {
            "type": "error",
            "error": {"type": "authentication_error", "message": str(e)},
        }
        raise HTTPException(status_code=401, detail=error_response)
    except Exception as e:
        logging.error(f"Anthropic count_tokens endpoint error: {e}")
        error_response = {
            "type": "error",
            "error": {"type": "api_error", "message": str(e)},
        }
        raise HTTPException(status_code=500, detail=error_response)


@app.post("/v1/embeddings")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    client: RotatingClient = Depends(get_rotating_client),
    batcher: Optional[EmbeddingBatcher] = Depends(get_embedding_batcher),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for creating embeddings.
    Supports two modes based on the USE_EMBEDDING_BATCHER flag:
    - True: Uses a server-side batcher for high throughput.
    - False: Passes requests directly to the provider.
    """
    try:
        # Serialize body once — reused for logging and both code paths below
        request_data = body.model_dump(exclude_none=True)
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        if USE_EMBEDDING_BATCHER and batcher:
            # --- Server-Side Batching Logic ---
            inputs = request_data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]

            tasks = []
            for single_input in inputs:
                individual_request = request_data.copy()
                individual_request["input"] = single_input
                tasks.append(batcher.add_request(individual_request))

            results = await asyncio.gather(*tasks)

            all_data = []
            total_prompt_tokens = 0
            total_tokens = 0
            for i, result in enumerate(results):
                result["data"][0]["index"] = i
                all_data.extend(result["data"])
                total_prompt_tokens += result["usage"]["prompt_tokens"]
                total_tokens += result["usage"]["total_tokens"]

            final_response_data = {
                "object": "list",
                "model": results[0]["model"],
                "data": all_data,
                "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "total_tokens": total_tokens,
                },
            }
            response = litellm.EmbeddingResponse(**final_response_data)

        else:
            # --- Direct Pass-Through Logic ---
            if isinstance(request_data.get("input"), str):
                request_data["input"] = [request_data["input"]]

            response = await client.aembedding(request=request, **request_data)

        return response

    except HTTPException as e:
        # Re-raise HTTPException to ensure it's not caught by the generic Exception handler
        raise e
    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ) as e:
        raise HTTPException(status_code=400, detail=f"Invalid Request: {str(e)}")
    except litellm.AuthenticationError as e:
        raise HTTPException(status_code=401, detail=f"Authentication Error: {str(e)}")
    except litellm.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Rate Limit Exceeded: {str(e)}")
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError) as e:
        raise HTTPException(status_code=503, detail=f"Service Unavailable: {str(e)}")
    except litellm.Timeout as e:
        raise HTTPException(status_code=504, detail=f"Gateway Timeout: {str(e)}")
    except (litellm.InternalServerError, litellm.OpenAIError) as e:
        raise HTTPException(status_code=502, detail=f"Bad Gateway: {str(e)}")
    except Exception as e:
        logging.error(f"Embedding request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    return {"Status": "API Key Proxy is running"}


@app.get("/v1/models")
async def list_models(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    enriched: bool = True,
):
    """
    Returns a list of available models in the OpenAI-compatible format.

    Query Parameters:
        enriched: If True (default), returns detailed model info with pricing and capabilities.
                  If False, returns minimal OpenAI-compatible response.
    """
    now = time.monotonic()
    if _models_cache["data"] is not None and _models_cache["expires"] > now:
        return _models_cache["data"]

    model_ids = await client.get_all_available_models(grouped=False)

    if enriched and hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            enriched_data = model_info_service.enrich_model_list(model_ids)
            response_data = {"object": "list", "data": enriched_data}
            _models_cache["data"] = response_data
            _models_cache["expires"] = now + 60
            return response_data

    # Fallback to basic model cards
    model_cards = [
        {
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "Mirro-Proxy",
        }
        for model_id in model_ids
    ]
    response_data = {"object": "list", "data": model_cards}
    _models_cache["data"] = response_data
    _models_cache["expires"] = now + 60
    return response_data


@app.get("/v1/models/{model_id:path}")
async def get_model(
    model_id: str,
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns detailed information about a specific model.

    Path Parameters:
        model_id: The model ID (e.g., "anthropic/claude-3-opus", "openrouter/openai/gpt-4")
    """
    if hasattr(request.app.state, "model_info_service"):
        model_info_service = request.app.state.model_info_service
        if model_info_service.is_ready:
            info = model_info_service.get_model_info(model_id)
            if info:
                return info.to_dict()

    # Return basic info if service not ready or model not found
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": model_id.split("/")[0] if "/" in model_id else "unknown",
    }


@app.get("/v1/model-info/stats")
async def model_info_stats(
    request: Request,
    _=Depends(verify_api_key),
):
    """
    Returns statistics about the model info service (for monitoring/debugging).
    """
    if hasattr(request.app.state, "model_info_service"):
        return request.app.state.model_info_service.get_stats()
    return {"error": "Model info service not initialized"}


@app.get("/v1/providers")
async def list_providers(_=Depends(verify_api_key)):
    """
    Returns a list of all available providers.
    """
    return list(PROVIDER_PLUGINS.keys())


@app.get("/v1/quota-stats")
async def get_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
    provider: str = None,
):
    """
    Returns quota and usage statistics for all credentials.

    This returns cached data from the proxy without making external API calls.
    Use POST to reload from disk or force refresh from external APIs.

    Query Parameters:
        provider: Optional filter to return stats for a specific provider only

    Returns:
        {
            "providers": {
                "provider_name": {
                    "credential_count": int,
                    "active_count": int,
                    "on_cooldown_count": int,
                    "exhausted_count": int,
                    "total_requests": int,
                    "tokens": {...},
                    "approx_cost": float | null,
                    "quota_groups": {...},  // For Antigravity
                    "credentials": [...]
                }
            },
            "summary": {...},
            "data_source": "cache",
            "timestamp": float
        }
    """
    try:
        stats = await client.get_quota_stats(provider_filter=provider)
        return stats
    except Exception as e:
        logging.error(f"Failed to get quota stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/quota-stats")
async def refresh_quota_stats(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Refresh quota and usage statistics.

    Request body:
        {
            "action": "reload" | "force_refresh",
            "scope": "all" | "provider" | "credential",
            "provider": "antigravity",  // required if scope != "all"
            "credential": "antigravity_oauth_1.json"  // required if scope == "credential"
        }

    Actions:
        - reload: Re-read data from disk (no external API calls)
        - force_refresh: For Antigravity, fetch live quota from API.
                        For other providers, same as reload.

    Returns:
        Same as GET, plus a "refresh_result" field with operation details.
    """
    try:
        data = orjson.loads(await request.body())
        action = data.get("action", "reload")
        scope = data.get("scope", "all")
        provider = data.get("provider")
        credential = data.get("credential")

        # Validate parameters
        if action not in ("reload", "force_refresh"):
            raise HTTPException(
                status_code=400,
                detail="action must be 'reload' or 'force_refresh'",
            )

        if scope not in ("all", "provider", "credential"):
            raise HTTPException(
                status_code=400,
                detail="scope must be 'all', 'provider', or 'credential'",
            )

        if scope in ("provider", "credential") and not provider:
            raise HTTPException(
                status_code=400,
                detail="'provider' is required when scope is 'provider' or 'credential'",
            )

        if scope == "credential" and not credential:
            raise HTTPException(
                status_code=400,
                detail="'credential' is required when scope is 'credential'",
            )

        refresh_result = {
            "action": action,
            "scope": scope,
            "provider": provider,
            "credential": credential,
        }

        if action == "reload":
            # Just reload from disk
            start_time = time.time()
            await client.reload_usage_from_disk()
            refresh_result["duration_ms"] = int((time.time() - start_time) * 1000)
            refresh_result["success"] = True
            refresh_result["message"] = "Reloaded usage data from disk"

        elif action == "force_refresh":
            # Force refresh from external API (for supported providers like Antigravity)
            result = await client.force_refresh_quota(
                provider=provider if scope in ("provider", "credential") else None,
                credential=credential if scope == "credential" else None,
            )
            refresh_result.update(result)
            refresh_result["success"] = result["failed_count"] == 0

        # Get updated stats
        stats = await client.get_quota_stats(provider_filter=provider)
        stats["refresh_result"] = refresh_result
        stats["data_source"] = "refreshed"

        return stats

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to refresh quota stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/token-count")
async def token_count(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    Calculates the token count for a given list of messages and a model.
    """
    try:
        data = orjson.loads(await request.body())
        model = data.get("model")
        messages = data.get("messages")

        if not model or not messages:
            raise HTTPException(
                status_code=400, detail="'model' and 'messages' are required."
            )

        count = await asyncio.to_thread(client.token_count, **data)
        return {"token_count": count}

    except Exception as e:
        logging.error(f"Token count failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/cost-estimate")
async def cost_estimate(request: Request, _=Depends(verify_api_key)):
    """
    Estimates the cost for a request based on token counts and model pricing.

    Request body:
        {
            "model": "anthropic/claude-3-opus",
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "cache_read_tokens": 0,       # optional
            "cache_creation_tokens": 0    # optional
        }

    Returns:
        {
            "model": "anthropic/claude-3-opus",
            "cost": 0.0375,
            "currency": "USD",
            "pricing": {
                "input_cost_per_token": 0.000015,
                "output_cost_per_token": 0.000075
            },
            "source": "model_info_service"  # or "litellm_fallback"
        }
    """
    try:
        data = orjson.loads(await request.body())
        model = data.get("model")
        prompt_tokens = data.get("prompt_tokens", 0)
        completion_tokens = data.get("completion_tokens", 0)
        cache_read_tokens = data.get("cache_read_tokens", 0)
        cache_creation_tokens = data.get("cache_creation_tokens", 0)

        if not model:
            raise HTTPException(status_code=400, detail="'model' is required.")

        result = {
            "model": model,
            "cost": None,
            "currency": "USD",
            "pricing": {},
            "source": None,
        }

        # Try model info service first
        if hasattr(request.app.state, "model_info_service"):
            model_info_service = request.app.state.model_info_service
            if model_info_service.is_ready:
                cost = model_info_service.calculate_cost(
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                )
                if cost is not None:
                    cost_info = model_info_service.get_cost_info(model)
                    result["cost"] = cost
                    result["pricing"] = cost_info or {}
                    result["source"] = "model_info_service"
                    return result

        # Fallback to litellm
        try:
            import litellm

            # Create a mock response for cost calculation
            model_info = litellm.get_model_info(model)
            input_cost = model_info.get("input_cost_per_token", 0)
            output_cost = model_info.get("output_cost_per_token", 0)
            cache_read_cost = model_info.get("cache_read_input_token_cost", 0) or input_cost * 0.1
            cache_creation_cost = model_info.get("cache_creation_input_token_cost", 0) or input_cost * 1.25

            if input_cost or output_cost:
                non_cached_input = max(prompt_tokens - cache_read_tokens - cache_creation_tokens, 0)
                cost = (
                    non_cached_input * input_cost
                    + cache_read_tokens * cache_read_cost
                    + cache_creation_tokens * cache_creation_cost
                    + completion_tokens * output_cost
                )
                result["cost"] = cost
                result["pricing"] = {
                    "input_cost_per_token": input_cost,
                    "output_cost_per_token": output_cost,
                    "cache_read_input_token_cost": cache_read_cost,
                    "cache_creation_input_token_cost": cache_creation_cost,
                }
                result["source"] = "litellm_fallback"
                return result
        except Exception:
            pass

        result["source"] = "unknown"
        result["error"] = "Pricing data not available for this model"
        return result

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Cost estimate failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        os.system(
            "cls" if os.name == "nt" else "clear"
        )  # Clear terminal for clean presentation
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
