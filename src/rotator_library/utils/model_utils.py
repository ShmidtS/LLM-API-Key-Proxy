# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/utils/model_utils.py
"""
Shared model string parsing utilities.

Eliminates duplication of provider/model string extraction across
client.py, usage_manager.py, and provider modules.
"""

import fnmatch
import functools
import os
import re
from typing import Any, Optional


CompiledModelPatterns = dict[str, tuple[set[str], tuple[re.Pattern[str], ...], bool]]


@functools.lru_cache(maxsize=4096)
def _cached_match_model_pattern(
    _provider: str,
    model_id: str,
    patterns_id: int,
    wildcard_return: bool,
    provider_model_name: str,
    model_provider: str,
) -> bool:
    patterns = _MODEL_PATTERN_REGISTRY.get(patterns_id)
    if patterns is None:
        return False

    compiled = patterns.get(model_provider)
    if compiled is None:
        return False

    exact_patterns, wildcard_patterns, match_all = compiled
    if match_all:
        return wildcard_return
    normalized_model_name = os.path.normcase(provider_model_name)
    normalized_model_id = os.path.normcase(model_id)
    if normalized_model_name in exact_patterns or normalized_model_id in exact_patterns:
        return True
    return any(
        pattern.match(normalized_model_name) is not None
        or pattern.match(normalized_model_id) is not None
        for pattern in wildcard_patterns
    )


_MODEL_PATTERN_REGISTRY: dict[int, CompiledModelPatterns] = {}


def compile_model_patterns(pattern_dict: dict[str, list[str]]) -> CompiledModelPatterns:
    compiled: CompiledModelPatterns = {}
    for provider, patterns in pattern_dict.items():
        exact_patterns: set[str] = set()
        wildcard_patterns: list[re.Pattern[str]] = []
        for pattern in patterns:
            normalized_pattern = os.path.normcase(pattern)
            if normalized_pattern == "*" and patterns == ["*"]:
                continue
            if any(char in normalized_pattern for char in "*?["):
                wildcard_patterns.append(re.compile(fnmatch.translate(normalized_pattern)))
            else:
                exact_patterns.add(normalized_pattern)
        compiled[provider] = (
            exact_patterns,
            tuple(wildcard_patterns),
            patterns == ["*"],
        )
    return compiled


def register_model_patterns(patterns: CompiledModelPatterns) -> None:
    _MODEL_PATTERN_REGISTRY[id(patterns)] = patterns


def clear_model_match_cache() -> None:
    _cached_match_model_pattern.cache_clear()


def match_model_pattern(
    provider: str,
    model_id: str,
    patterns: CompiledModelPatterns,
    wildcard_return: bool = False,
) -> bool:
    model_provider = model_id.split("/")[0]
    if model_provider not in patterns:
        return False
    try:
        provider_model_name = model_id.split("/", 1)[1]
    except IndexError:
        provider_model_name = model_id
    return _cached_match_model_pattern(
        provider,
        model_id,
        id(patterns),
        wildcard_return,
        provider_model_name,
        model_provider,
    )


@functools.lru_cache(maxsize=256)
def extract_provider_from_model(model: str) -> str:
    """
    Extract provider prefix from ``provider/model`` format.

    Args:
        model: Model string, optionally with ``provider/`` prefix.

    Returns:
        Lowercased provider name, or empty string if no prefix.
    """
    if not isinstance(model, str):
        return ""
    normalized = model.strip()
    if not normalized or "/" not in normalized:
        return ""
    return normalized.split("/", 1)[0].strip().lower()


@functools.lru_cache(maxsize=256)
def normalize_model_string(model: str) -> str:
    """
    Normalize incoming model string for consistent routing.

    Remaps legacy provider prefixes (e.g., nvidia_nim/ -> nvidia/) and strips whitespace.

    Args:
        model: Raw model string from request.

    Returns:
        Stripped and normalized model string, or empty string if not a string.
    """
    if not isinstance(model, str):
        return ""
    normalized = model.strip()
    # Legacy provider prefix aliases
    _PROVIDER_PREFIX_ALIASES = {
        "nvidia_nim": "nvidia",
    }
    slash_pos = normalized.find("/")
    if slash_pos > 0:
        prefix = normalized[:slash_pos]
        if prefix in _PROVIDER_PREFIX_ALIASES:
            normalized = _PROVIDER_PREFIX_ALIASES[prefix] + normalized[slash_pos:]
    return normalized


@functools.lru_cache(maxsize=256)
def parse_env_credential_path(path: str) -> Optional[str]:
    """
    Parse a virtual ``env://`` path and return the credential index.

    Supported formats:
    - ``env://provider/0`` — legacy single credential
    - ``env://provider/1`` — first numbered credential

    Args:
        path: Credential path string.

    Returns:
        Credential index as string ("0" for legacy, "1", "2", etc.)
        or None if path is not an ``env://`` path.
    """
    if not isinstance(path, str) or not path.startswith("env://"):
        return None

    parts = path[6:].split("/")
    if len(parts) >= 2:
        return parts[1]
    return "0"


def get_or_create_provider_instance(
    provider_name: str,
    provider_plugins: dict,
    provider_registry: Any,
) -> Any:
    """
    Get or create a provider instance via the shared plugin/registry system.

    This consolidates the lazy-load pattern shared between client.py and
    usage_manager.py: look up the plugin class in *provider_plugins*,
    falling back to the lazy-load ``providers.get_provider()`` entry point,
    then delegate creation to *provider_registry*.

    Args:
        provider_name: Provider identifier (e.g. ``"antigravity"``).
        provider_plugins: Dict mapping provider names to plugin classes.
        provider_registry: A ProviderRegistry (or compatible) instance that
                           supports ``get_or_create(name, entry)``.

    Returns:
        Provider instance, or None if the provider is unknown.
    """
    if not provider_name:
        return None

    plugin_class = provider_plugins.get(provider_name)
    if not plugin_class:
        from ..providers import get_provider as _lazy_get_provider
        plugin_class = _lazy_get_provider(provider_name)
        if not plugin_class:
            return None
        provider_plugins[provider_name] = plugin_class

    return provider_registry.get_or_create(provider_name, plugin_class)
