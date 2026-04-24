# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/provider_routing_config.py
"""
Provider routing configuration for the rotator library.

This module handles:
- API base overrides for known providers
- Custom OpenAI-compatible provider detection and routing
- Provider blacklist and known provider set
"""

import os
import logging
from .config.defaults import TRACE

from typing import Dict, Any, Set, Optional

from .litellm_providers import SCRAPED_PROVIDERS
from .provider_ui_config import LITELLM_PROVIDERS

lib_logger = logging.getLogger("rotator_library")

# =============================================================================
# Provider Blacklist
# =============================================================================
# Providers that are in LiteLLM but should be excluded from:
# - KNOWN_PROVIDERS (so _API_BASE for them creates a "custom" provider)
# - Credential tool UI (won't show up in provider selection)
#
# Reasons for blacklisting:
# - No standard API key authentication (requires OAuth, token files, etc.)
# - Not actual LLM providers (protocols, templates, etc.)
# - Legacy/deprecated APIs
# - Complex auth requiring multiple credentials
# - Non-standard API key patterns (proxy only supports *_API_KEY)
# =============================================================================

PROVIDER_BLACKLIST: Set[str] = {
    # Not standard LLM providers / protocols
    "a2a",  # Pydantic AI agent-to-agent protocol
    "my-custom-llm",  # Template, not a real provider
    "text-completion-openai",  # Legacy text completion API
    # Require special auth (token files, OAuth, etc.)
    "github_copilot",  # Requires token file configuration
    "vercel_ai_gateway",  # Requires OIDC token
    # No API key authentication (use custom provider instead)
    "ollama",  # Local, no API key
    "llamafile",  # Local, no API key
    "petals",  # Distributed network, no API key
    "triton",  # NVIDIA Triton server, no API key
    "lemonade",  # Local, no API key
    "oci",  # OCI SDK auth only
    # Complex multi-credential auth (proxy only supports API_KEY + API_BASE)
    "azure",  # Requires API key + endpoint + API version
    "bedrock",  # Requires AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + region
    "sagemaker",  # Same as bedrock
    "vertex_ai",  # File path credential + project + location
    "sap",  # Multiple service credentials required
    "cloudflare",  # Requires API key + account ID
    "snowflake",  # Requires JWT + account ID
    "watsonx",  # Requires API key + special URL parameter
    # Non-standard API key patterns (uses *_TOKEN or *_KEY, not *_API_KEY)
    "cometapi",  # Uses COMETAPI_KEY
    "friendliai",  # Uses FRIENDLI_TOKEN
    "huggingface",  # Uses HF_TOKEN
}


def _build_known_providers_set() -> Set[str]:
    """
    Build set of known provider routes from scraped LiteLLM data.

    Uses routes as the primary identifier (authoritative from LiteLLM docs).
    Only uses API key prefix as fallback when provider has no route.

    Excludes providers in PROVIDER_BLACKLIST.

    Returns:
        Set of lowercase provider route identifiers known to LiteLLM.
    """
    known = set()

    for provider_key, info in SCRAPED_PROVIDERS.items():
        # Skip blacklisted providers
        if provider_key in PROVIDER_BLACKLIST:
            continue

        route = info.get("route", "").rstrip("/").lower()

        if route:
            # Provider has a route - use it as the canonical key
            known.add(route)
        else:
            # No route - fall back to API key prefix
            for api_key_var in info.get("api_key_env_vars", []):
                prefix = _extract_api_key_prefix(api_key_var)
                if prefix:
                    known.add(prefix)
                    break  # Only need one fallback

    return known


def _extract_api_key_prefix(api_key_var: str) -> Optional[str]:
    """Extract provider prefix from an API key environment variable name.

    Examples:
        OPENAI_API_KEY -> openai
        HF_TOKEN -> hf
        WATSONX_APIKEY -> watsonx
    """
    if not api_key_var:
        return None

    api_key_var = api_key_var.upper()

    if api_key_var.endswith("_API_KEY"):
        return api_key_var[:-8].lower()
    elif api_key_var.endswith("_TOKEN"):
        return api_key_var[:-6].lower()
    elif api_key_var.endswith("_APIKEY"):
        return api_key_var[:-7].lower()
    elif api_key_var.endswith("_KEY"):
        return api_key_var[:-4].lower()
    elif api_key_var.endswith("_JWT"):
        return api_key_var[:-4].lower()

    return None


# Pre-computed set of known provider names
KNOWN_PROVIDERS: Set[str] = _build_known_providers_set()

# Manually add providers with custom plugins that aren't in scraped LiteLLM data
KNOWN_PROVIDERS.add("trybons")
KNOWN_PROVIDERS.add("colin")  # COLIN uses OpenAI Responses API format
KNOWN_PROVIDERS.add("zai")  # ZAI has custom quota tracking provider
KNOWN_PROVIDERS.add("fireworks")  # fireworks_ai alias — users specify "fireworks/" prefix


class ProviderConfig:
    """
    Centralized provider configuration handling.

    Handles:
    - API base overrides for known LiteLLM providers
    - Custom OpenAI-compatible providers (unknown provider names)

    Usage patterns:

    1. Override existing provider's API base:
       Set OPENAI_API_BASE=http://my-local-llm/v1
       Request: openai/gpt-4 -> LiteLLM gets model="openai/gpt-4", api_base="http://..."

    2. Custom OpenAI-compatible provider:
       Set MYSERVER_API_BASE=http://myserver:8000/v1
       Request: myserver/llama-3 -> LiteLLM gets model="openai/llama-3",
                api_base="http://...", custom_llm_provider="openai"
    """

    def __init__(self):
        self._api_bases: Dict[str, str] = {}
        self._custom_providers: Set[str] = set()
        self._load_api_bases()

    def _load_api_bases(self) -> None:
        """
        Load all <PROVIDER>_API_BASE environment variables.

        Detects whether each is an override for a known provider
        or defines a new custom provider.
        """
        for key, value in os.environ.items():
            if key.endswith("_API_BASE") and value:
                provider = key[:-9].lower()  # Remove _API_BASE
                self._api_bases[provider] = value.rstrip("/")

                # Track if this is a custom provider (not known to LiteLLM)
                if provider not in KNOWN_PROVIDERS:
                    self._custom_providers.add(provider)
                    lib_logger.info(
                        f"Detected custom OpenAI-compatible provider: {provider} "
                        f"(api_base: {value})"
                    )
                else:
                    lib_logger.info(
                        f"Detected API base override for {provider}: {value}"
                    )

        # Then, apply defaults for providers with extra_vars default API_BASE
        # This handles providers like kilocode that are not known to LiteLLM
        for provider, config in LITELLM_PROVIDERS.items():
            if provider in self._api_bases:
                continue  # Already configured via env var

            extra_vars = config.get("extra_vars", [])
            for var_name, var_label, var_default in extra_vars:
                if var_name.endswith("_API_BASE") and var_default:
                    # Provider has a default API_BASE and is not known to LiteLLM
                    if provider not in KNOWN_PROVIDERS:
                        self._api_bases[provider] = var_default.rstrip("/")
                        self._custom_providers.add(provider)
                        lib_logger.info(
                            f"Applied default API_BASE for custom provider '{provider}': {var_default}"
                        )
                    break

    def is_known_provider(self, provider: str) -> bool:
        """Check if provider is known to LiteLLM."""
        return provider.lower() in KNOWN_PROVIDERS

    def is_custom_provider(self, provider: str) -> bool:
        """Check if provider is a custom OpenAI-compatible provider."""
        return provider.lower() in self._custom_providers

    def get_api_base(self, provider: str) -> Optional[str]:
        """Get configured API base for a provider, if any."""
        return self._api_bases.get(provider.lower())

    def get_custom_providers(self) -> Set[str]:
        """Get the set of detected custom provider names."""
        return self._custom_providers.copy()

    @property
    def api_bases(self) -> Dict[str, str]:
        """Get the dictionary of configured API bases (read-only view)."""
        return self._api_bases.copy()

    def convert_for_litellm(self, **kwargs) -> Dict[str, Any]:
        """
        Convert model params for LiteLLM call.

        Handles:
        - Known provider with _API_BASE: pass api_base as override
        - Unknown provider with _API_BASE: convert to openai/, set custom_llm_provider
        - No _API_BASE configured: pass through unchanged

        Args:
            **kwargs: LiteLLM call kwargs including 'model'

        Returns:
            Modified kwargs dict ready for LiteLLM
        """
        model = kwargs.get("model")
        if not model:
            return kwargs

        # Extract provider from model string (e.g., "openai/gpt-4" -> "openai")
        provider = model.split("/")[0].lower()
        api_base = self._api_bases.get(provider)

        # TryBons: Anthropic-compatible API, route through LiteLLM's anthropic support
        if provider == "trybons":
            trybons_base = api_base or "https://go.trybons.ai"
            model_name = model.split("/", 1)[1] if "/" in model else model
            kwargs = kwargs.copy()
            kwargs["model"] = f"anthropic/{model_name}"
            kwargs["api_base"] = trybons_base
            lib_logger.debug(
                f"Routing trybons model through anthropic: "
                f"model={kwargs['model']}, api_base={trybons_base}"
            )
            return kwargs

        # Fireworks: litellm registers as "fireworks_ai" but users use "fireworks/" prefix
        if provider == "fireworks":
            fireworks_base = api_base or self._api_bases.get("fireworks_ai")
            model_name = model.split("/", 1)[1] if "/" in model else model
            kwargs = kwargs.copy()
            kwargs["model"] = f"fireworks_ai/{model_name}"
            if fireworks_base:
                kwargs["api_base"] = fireworks_base
            lib_logger.debug(
                f"Routing fireworks model through fireworks_ai: "
                f"model={kwargs['model']}, api_base={fireworks_base}"
            )
            return kwargs

        if not api_base:
            # No override configured for this provider
            return kwargs

        # Create a copy to avoid modifying the original
        kwargs = kwargs.copy()

        if provider in KNOWN_PROVIDERS:
            # Known provider - just add api_base override
            # Special case: Inception Labs requires model name without provider prefix
            # Their API expects "mercury-2" not "inception/mercury-2"
            if provider == "inception":
                model_name = model.split("/", 1)[1] if "/" in model else model
                kwargs["model"] = model_name
                lib_logger.debug(
                    f"Stripping provider prefix for inception model: {model} -> {model_name}"
                )
            kwargs["api_base"] = api_base
            lib_logger.log(
                TRACE,
                f"Applying api_base override for known provider {provider}: {api_base}"
            )
        else:
            # Custom provider - route through OpenAI-compatible endpoint
            model_name = model.split("/", 1)[1] if "/" in model else model

            # Handle models with embedded slashes (e.g., "openai/gpt-5.4" from "noob/openai/gpt-5.4")
            # If model_name contains a slash, it's already in provider/model format
            if "/" in model_name:
                kwargs["model"] = model_name  # Use as-is: "openai/gpt-5.4"
            else:
                kwargs["model"] = (
                    f"openai/{model_name}"  # Add prefix: "gpt-4" -> "openai/gpt-4"
                )
            kwargs["api_base"] = api_base
            kwargs["custom_llm_provider"] = "openai"
            lib_logger.debug(
                f"Routing custom provider {provider} through openai: "
                f"model={kwargs['model']}, api_base={api_base}"
            )

        return kwargs
