# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Model fetcher - handles async model fetching from providers.

Runs fetching in a background thread to avoid blocking the GUI.
Includes caching to avoid refetching on every provider switch.
No tkinter dependencies.
"""

import asyncio
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from rotator_library.timeout_config import TimeoutConfig

logger = logging.getLogger(__name__)

# Global cache for fetched models (persists across provider switches)
_model_cache: Dict[str, List[str]] = {}


class ModelFetcher:
    """
    Handles async model fetching from providers.

    Runs fetching in a background thread to avoid blocking the GUI.
    Includes caching to avoid refetching on every provider switch.
    """

    @staticmethod
    def get_cached_models(provider: str) -> Optional[List[str]]:
        """Get cached models for a provider, if available."""
        return _model_cache.get(provider)

    @staticmethod
    def clear_cache(provider: Optional[str] = None):
        """Clear model cache. If provider specified, only clear that provider."""
        if provider:
            _model_cache.pop(provider, None)
        else:
            _model_cache.clear()

    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of providers that have credentials configured."""
        providers = set()
        load_dotenv(override=True)

        # Scan environment for API keys (handles numbered keys like GEMINI_API_KEY_1)
        for key in os.environ:
            if "_API_KEY" in key and "PROXY_API_KEY" not in key:
                # Extract provider: NVIDIA_API_KEY_1 -> nvidia
                provider = key.split("_API_KEY")[0].lower()
                providers.add(provider)

        # Check for OAuth providers
        oauth_dir = Path("oauth_creds")
        if oauth_dir.exists():
            for file in oauth_dir.glob("*_oauth_*.json"):
                provider = file.name.split("_oauth_")[0]
                providers.add(provider)

        return sorted(list(providers))

    @staticmethod
    def _find_credential(provider: str) -> Optional[str]:
        """Find a credential for a provider (handles numbered keys)."""
        load_dotenv(override=True)
        provider_upper = provider.upper()

        # Try exact match first (e.g., GEMINI_API_KEY)
        exact_key = f"{provider_upper}_API_KEY"
        if os.getenv(exact_key):
            return os.getenv(exact_key)

        # Look for numbered keys (e.g., GEMINI_API_KEY_1, NVIDIA_NIM_API_KEY_1)
        for key, value in os.environ.items():
            if key.startswith(f"{provider_upper}_API_KEY") and value:
                return value

        # Check for OAuth credentials
        oauth_dir = Path("oauth_creds")
        if oauth_dir.exists():
            oauth_files = list(oauth_dir.glob(f"{provider}_oauth_*.json"))
            if oauth_files:
                return str(oauth_files[0])

        return None

    @staticmethod
    async def _fetch_models_async(provider: str) -> Tuple[List[str], Optional[str]]:
        """
        Async implementation of model fetching.
        Returns: (models_list, error_message_or_none)
        """
        try:
            import httpx
            from rotator_library.providers import PROVIDER_PLUGINS

            # Get credential
            credential = ModelFetcher._find_credential(provider)
            if not credential:
                return [], f"No credentials found for '{provider}'"

            # Get provider class
            provider_class = PROVIDER_PLUGINS.get(provider.lower())
            if not provider_class:
                return [], f"Unknown provider: '{provider}'"

            # Fetch models
            async with httpx.AsyncClient(timeout=TimeoutConfig.model_filter_fetch()) as client:
                from typing import cast
                instance = cast(Any, provider_class)()
                models = await instance.get_models(credential, client)
                return models, None

        except ImportError as e:
            logger.error(f"Failed to import requirements for model fetching: {e}", exc_info=True)
            return [], f"Import error: {e}"
        except Exception as e:
            logger.error(f"Unexpected error during model fetch for {provider}: {e}", exc_info=True)
            return [], f"Failed to fetch: {str(e)}"

    @staticmethod
    def fetch_models(
        provider: str,
        on_success: Callable[[List[str]], None],
        on_error: Callable[[str], None],
        on_start: Optional[Callable[[], None]] = None,
        force_refresh: bool = False,
    ):
        """
        Fetch models in a background thread.

        Args:
            provider: Provider name (e.g., 'openai', 'gemini')
            on_success: Callback with list of model IDs
            on_error: Callback with error message
            on_start: Optional callback when fetching starts
            force_refresh: If True, bypass cache and fetch fresh
        """
        # Check cache first (unless force refresh)
        if not force_refresh:
            cached = ModelFetcher.get_cached_models(provider)
            if cached is not None:
                on_success(cached)
                return

        def run_fetch():
            if on_start:
                on_start()

            try:
                # Run async fetch in new event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    models, error = loop.run_until_complete(
                        ModelFetcher._fetch_models_async(provider)
                    )
                    # Clean up any pending tasks to avoid warnings
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                finally:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

                if error:
                    on_error(error)
                else:
                    # Cache the results
                    _model_cache[provider] = models
                    on_success(models)

            except Exception as e:
                logger.error(f"Error in background model fetch thread: {e}", exc_info=True)
                on_error(str(e))

        thread = threading.Thread(target=run_fetch, daemon=True)
        thread.start()
