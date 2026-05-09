# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import sys
from pathlib import Path
from contextlib import asynccontextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from proxy_app.bootstrap import (
    bootstrap,
    configure_app,
    create_arg_parser,
    default_args,
)
from proxy_app.config import (
    DEFAULT_GRACEFUL_SHUTDOWN_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_UVICORN_BACKLOG,
    env_int,
)
from proxy_app.middleware import _NoGzipForSSE


@asynccontextmanager
async def lifespan(app: FastAPI):
    from proxy_app.bootstrap import create_lifespan_from_environment

    if not getattr(app.state, "routes_registered", False):
        _register_routes(app)
        app.state.routes_registered = True
    runtime_lifespan = create_lifespan_from_environment(
        getattr(app.state, "cli_args", default_args())
    )
    async with runtime_lifespan(app):
        yield


# --- FastAPI App Setup ---

app = FastAPI(lifespan=lifespan)
app.state.cli_args = default_args()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Accel-Buffering",
        "X-Request-Id",
        "X-Provider",
        "Retry-After",
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
    ],
)
app.add_middleware(_NoGzipForSSE)

def _register_routes(app: FastAPI) -> None:
    from proxy_app.routes import all_routers

    for router in all_routers:
        app.include_router(router)


if __name__ == "__main__":
    _register_routes(app)
    app.state.routes_registered = True


@app.api_route("/", methods=["GET", "HEAD"])
async def read_root():
    return {"Status": "API Key Proxy is running"}


def needs_onboarding(env_file) -> bool:
    """Check if the proxy needs onboarding (first-time setup)."""
    return not env_file.is_file()


def show_onboarding_message() -> None:
    """Display clear explanatory message for why onboarding is needed."""
    from rich.console import Console
    from rich.panel import Panel
    from rotator_library.utils.terminal_utils import clear_screen

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


def run_server() -> None:
    parser = create_arg_parser()
    args = default_args()

    if len(sys.argv) == 1:
        startup_state = bootstrap(args)
        from proxy_app.launcher_tui import run_launcher_tui

        run_launcher_tui()
        args = parser.parse_args()
    else:
        args = parser.parse_args()
        startup_state = bootstrap(args)

    configure_app(app, args)

    from rotator_library.utils.paths import get_data_file

    env_file = get_data_file(".env")

    if args.add_credential:
        from rotator_library.credential_tool import ensure_env_defaults, run_credential_tool

        ensure_env_defaults()
        if startup_state.load_dotenv is not None:
            startup_state.load_dotenv(env_file, override=True)
        run_credential_tool()
    elif needs_onboarding(env_file):
        from rich.console import Console
        from rotator_library.credential_tool import ensure_env_defaults, run_credential_tool

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


if __name__ == "__main__":
    run_server()
