# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/credential_providers.py

import functools
import json
import logging
import httpx
import re

from rich.console import Console
from dotenv import get_key

from .credential_io import (
    _get_env_file,
    _get_oauth_base_dir,
    _read_json_file,
)
from .provider_routing_config import PROVIDER_BLACKLIST

console = Console()

# Global variables for lazily loaded modules
_provider_auth_map = None
_provider_plugins = None


def _ensure_providers_loaded():
    """Lazy load provider modules only when needed"""
    global _provider_auth_map, _provider_plugins
    if _provider_auth_map is None:
        from .providers import PROVIDER_AUTH_MAP as pam
        from .providers import PROVIDER_PLUGINS as pp

        _provider_auth_map = pam
        _provider_plugins = pp
    return _provider_auth_map, _provider_plugins


def _get_provider_auth_class(provider_name: str):
    """Get auth class for provider, raising ValueError if unknown."""
    auth_map, _ = _ensure_providers_loaded()
    provider_key = provider_name.lower()
    provider_class = auth_map.get(provider_key)
    if not provider_class:
        raise ValueError(f"Unknown provider: {provider_name}")
    if isinstance(provider_class, tuple):
        from .providers import _resolve_auth_class

        provider_class = _resolve_auth_class(provider_key)
    return provider_class


# OAuth provider display names mapping (no "(OAuth)" suffix - context makes it clear)
OAUTH_FRIENDLY_NAMES = {
    "gemini_cli": "Gemini CLI",
    "qwen_code": "Qwen Code",
    "iflow": "iFlow",
    "antigravity": "Antigravity",
}

_OAUTH_PROVIDERS = frozenset({"gemini_cli", "antigravity"})


@functools.lru_cache(maxsize=128)
def _extract_key_number(key_name: str) -> int:
    """Extract the numeric suffix from a key name for proper sorting.

    Examples:
        GEMINI_API_KEY_1 -> 1
        GEMINI_API_KEY_10 -> 10
        GEMINI_API_KEY -> 0
    """
    match = re.search(r"_(\d+)$", key_name)
    return int(match.group(1)) if match else 0


@functools.lru_cache(maxsize=128)
def _normalize_tier_name(tier: str) -> str:
    """Normalize tier names for consistent display.

    Examples:
        "free-tier" -> "free"
        "FREE_TIER" -> "free"
        "PAID" -> "paid"
        "standard" -> "standard"
        None -> "unknown"
    """
    if not tier:
        return "unknown"

    # Lowercase and remove common suffixes/prefixes
    normalized = tier.lower().strip()
    normalized = normalized.replace("-tier", "").replace("_tier", "")
    normalized = normalized.replace("-", "").replace("_", "")

    return normalized


def _count_tiers(credentials: list) -> dict:
    """Count credentials by tier.

    Args:
        credentials: List of credential info dicts with optional 'tier' key

    Returns:
        Dict mapping normalized tier names to counts, e.g. {"free": 15, "paid": 2}
    """
    tier_counts = {}
    for cred in credentials:
        tier = cred.get("tier")
        if tier:
            normalized = _normalize_tier_name(tier)
            tier_counts[normalized] = tier_counts.get(normalized, 0) + 1
    return tier_counts


def _format_tier_counts(tier_counts: dict) -> str:
    """Format tier counts as a compact string.

    Examples:
        {"free": 15, "paid": 2} -> "(15 free, 2 paid)"
        {"free": 5} -> "(5 free)"
        {} -> ""
    """
    if not tier_counts:
        return ""

    # Sort by count descending, then alphabetically
    sorted_tiers = sorted(tier_counts.items(), key=lambda x: (-x[1], x[0]))
    parts = [f"{count} {tier}" for tier, count in sorted_tiers]
    return f"({', '.join(parts)})"


def _env_key_exists(key_name: str) -> bool:
    env_file = _get_env_file()
    if not env_file.is_file():
        return False
    with open(env_file, "r", encoding="utf-8") as f:
        return any(line.startswith(f"{key_name}=") for line in f)


def _get_api_keys_from_env() -> dict:
    """
    Parse the .env file and return a dictionary of API keys grouped by provider.
    Keys are sorted numerically within each provider.

    Returns:
        Dict mapping provider names to lists of (key_name, key_value) tuples.
        Example: {"GEMINI": [("GEMINI_API_KEY_1", "abc123"), ("GEMINI_API_KEY_2", "def456")]}
    """
    api_keys = {}
    env_file = _get_env_file()

    if not env_file.is_file():
        return api_keys

    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Look for lines with API_KEY pattern
                if "_API_KEY" in line and "=" in line:
                    key_name, _, key_value = line.partition("=")
                    key_name = key_name.strip()
                    key_value = key_value.strip().strip('"').strip("'")

                    # Skip PROXY_API_KEY and empty values
                    if key_name == "PROXY_API_KEY" or not key_value:
                        continue

                    # Skip placeholder values
                    if key_value.startswith("YOUR_") or key_value == "":
                        continue

                    # Extract provider name (everything before _API_KEY)
                    # Handle cases like GEMINI_API_KEY_1 -> GEMINI
                    parts = key_name.split("_API_KEY")
                    if parts:
                        provider_name = parts[0]
                        if provider_name not in api_keys:
                            api_keys[provider_name] = []
                        api_keys[provider_name].append((key_name, key_value))

        # Sort keys numerically within each provider
        for provider_name in api_keys:
            api_keys[provider_name].sort(key=lambda x: _extract_key_number(x[0]))

    except (ValueError, OSError, KeyError, httpx.HTTPError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error reading .env file: {e}[/bold red]")

    return api_keys


def _delete_api_key_from_env(key_name: str) -> bool:
    """
    Delete an API key from the .env file with safety backup and comparison.

    This function creates a backup of all API keys before deletion,
    performs the deletion, and then verifies no unintended keys were lost.

    Args:
        key_name: The exact key name to delete (e.g., "GEMINI_API_KEY_2")

    Returns:
        True if deletion was successful and verified, False otherwise
    """
    env_file = _get_env_file()

    if not env_file.is_file():
        console.print("[bold red]Error: .env file not found[/bold red]")
        return False

    try:
        # Step 1: Read all lines and backup all API keys
        with open(env_file, "r", encoding="utf-8") as f:
            original_lines = f.readlines()

        # Create backup of all API keys before modification
        api_keys_before = _get_api_keys_from_env()
        all_keys_before = set()
        for provider_keys in api_keys_before.values():
            for kn, kv in provider_keys:
                all_keys_before.add((kn, kv))

        # Step 2: Find and remove the target key
        new_lines = []
        key_found = False
        deleted_key_value = None

        for line in original_lines:
            stripped = line.strip()
            # Check if this line contains our target key
            if stripped.startswith(f"{key_name}="):
                key_found = True
                # Store the value being deleted for verification
                _, _, deleted_key_value = stripped.partition("=")
                deleted_key_value = deleted_key_value.strip().strip('"').strip("'")
                continue  # Skip this line (delete it)
            new_lines.append(line)

        if not key_found:
            console.print(
                f"[bold red]Error: Key '{key_name}' not found in .env file[/bold red]"
            )
            return False

        # Step 3: Write the modified content
        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        # Step 4: Verify the deletion - compare before and after
        api_keys_after = _get_api_keys_from_env()
        all_keys_after = set()
        for provider_keys in api_keys_after.values():
            for kn, kv in provider_keys:
                all_keys_after.add((kn, kv))

        # Check that only the intended key was removed
        expected_remaining = all_keys_before - {(key_name, deleted_key_value)}

        if all_keys_after != expected_remaining:
            # Something went wrong - restore from backup
            console.print(
                "[bold red]Error: Unexpected keys were affected during deletion![/bold red]"
            )
            console.print("[bold yellow]Restoring original file...[/bold yellow]")
            with open(env_file, "w", encoding="utf-8") as f:
                f.writelines(original_lines)
            return False

        return True

    except (ValueError, OSError) as e:
        console.print(f"[bold red]Error during API key deletion: {e}[/bold red]")
        return False


def _get_oauth_credentials_summary() -> dict:
    """
    Get a summary of all OAuth credentials for all providers.

    Returns:
        Dict mapping provider names to lists of credential info dicts.
        Example: {"gemini_cli": [{"email": "user@example.com", "tier": "free-tier", ...}, ...]}
    """
    _ensure_providers_loaded()
    oauth_providers = ["gemini_cli", "qwen_code", "iflow", "antigravity"]
    oauth_summary = {}

    for provider_name in oauth_providers:
        try:
            auth_class = _get_provider_auth_class(provider_name)
            auth_instance = auth_class()
            credentials = auth_instance.list_credentials(_get_oauth_base_dir())
            oauth_summary[provider_name] = credentials
        except (ValueError, KeyError, OSError) as e:  # non-critical: provider auth unavailable
            logging.debug("Provider auth listing failed for %s: %s", provider_name, e)
            oauth_summary[provider_name] = []

    return oauth_summary


def _get_all_credentials_summary() -> dict:
    """
    Get a complete summary of all credentials (API keys and OAuth).

    Returns:
        Dict with "api_keys" and "oauth" sections containing credential summaries.
    """
    return {
        "api_keys": _get_api_keys_from_env(),
        "oauth": _get_oauth_credentials_summary(),
    }


def _get_existing_custom_providers() -> list:
    """
    Scan the .env file for existing custom OpenAI-compatible providers.

    Custom providers are identified by *_API_BASE entries where the provider
    name is NOT a known LiteLLM provider.

    Returns:
        List of dicts with provider info:
        [{"name": "myserver", "api_base": "http://...", "has_key": True}, ...]
    """
    from .provider_routing_config import KNOWN_PROVIDERS

    custom_providers = []
    env_file = _get_env_file()

    if not env_file.is_file():
        return custom_providers

    try:
        # First pass: collect all _API_BASE entries
        api_bases = {}
        api_keys = set()

        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key_name, _, value = line.partition("=")
                key_name = key_name.strip()
                value = value.strip().strip('"').strip("'")

                if key_name.endswith("_API_BASE") and value:
                    provider_name = key_name[:-9].lower()  # Remove _API_BASE
                    # Only include if NOT a known provider
                    if provider_name not in KNOWN_PROVIDERS:
                        api_bases[provider_name] = value
                elif "_API_KEY" in key_name and value:
                    # Extract provider name from API key
                    provider_prefix = key_name.split("_API_KEY")[0].lower()
                    api_keys.add(provider_prefix)

        # Build result list
        for provider_name, api_base in sorted(api_bases.items()):
            custom_providers.append(
                {
                    "name": provider_name,
                    "api_base": api_base,
                    "has_key": provider_name in api_keys,
                }
            )

    except (ValueError, OSError, KeyError, httpx.HTTPError, json.JSONDecodeError) as e:
        console.print(f"[bold red]Error reading .env file: {e}[/bold red]")

    return custom_providers
