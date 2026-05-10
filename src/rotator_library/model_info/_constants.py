# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Provider priority configuration and lookup helpers.

Extracted from model_info_service for shared use across adapters and merger.
"""

# Native/authoritative providers - prefer these over proxy/aggregator providers
# Lower index = higher priority
NATIVE_PROVIDER_PRIORITY = [
    "anthropic",
    "openai",
    "google",
    "google-vertex",
    "mistral",
    "mistralai",
    "cohere",
    "deepseek",
    "deepseek-ai",  # Used in nvidia_nim/deepseek-ai/model format
    "qwen",
    "alibaba",
    "alibaba-cn",
    "meta-llama",
    "nvidia",
    "moonshotai",  # Used in nvidia_nim/moonshotai/model format
    "iflow",
    "iflowcn",
    # These are aggregators/proxies - lower priority
    "openrouter",
    "azure",
    "azure-cognitive-services",
    "aws-bedrock",
    "github-copilot",
    "opencode",
    "requesty",
    "helicone",
    "vercel",
    "aihubmix",
    "venice",
    "poe",
    "cortecs",
    "fastrouter",
    "ollama-cloud",
    "nebius",
    "fireworks-ai",
    "groq",
    "sap-ai-core",
    "zenmux",
]

# Maps custom/proxy provider names to their canonical equivalents in data sources.
# When looking up "nvidia_nim/org/model", we first try "nvidia/org/model" directly.
# This allows direct matches before falling back to fuzzy suffix matching.
#
# Format: "custom_provider": ["canonical_provider1", "canonical_provider2", ...]
# Multiple aliases are tried in order until a match is found.
PROVIDER_ALIASES = {
    "nvidia": ["nvidia_nim"],
    "gemini_cli": ["google"],
    "gemini": ["google"],
    "iflow": ["iflow", "iflowcn"],  # iflow may exist as either
}


def _get_provider_priority(provider: str) -> int:
    """
    Get priority score for a provider (lower = better).
    Native providers get priority over proxy/aggregator providers.
    """
    try:
        return NATIVE_PROVIDER_PRIORITY.index(provider.lower())
    except ValueError:
        # Unknown providers get lowest priority
        return len(NATIVE_PROVIDER_PRIORITY) + 1


def _extract_provider_from_source_id(source_id: str) -> str:
    """
    Extract the actual data provider from a source model ID.

    Examples:
        "anthropic/claude-opus-4.5" -> "anthropic"
        "openrouter/google/gemini-2.5-pro" -> "google" (skip openrouter prefix)
        "nvidia/mistralai/mistral-large" -> "mistralai" (3-segment, use middle)
    """
    parts = source_id.split("/")
    if len(parts) >= 2:
        # Skip openrouter prefix if present
        if parts[0].lower() == "openrouter" and len(parts) >= 3:
            return parts[1].lower()
        # For 3-segment IDs like nvidia/mistralai/model, use middle segment
        if len(parts) == 3:
            return parts[1].lower()
        return parts[0].lower()
    return source_id.lower()
