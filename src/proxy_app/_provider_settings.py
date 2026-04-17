# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

"""
Lightweight module exposing PROVIDER_SETTINGS_MAP without loading
the full settings_tool (2456-line) module. Used by launcher_tui
to avoid eager import during TUI startup.
"""

# Import default OAuth port values from provider modules
try:
    from rotator_library.providers.gemini_auth_base import GeminiAuthBase

    GEMINI_CLI_DEFAULT_OAUTH_PORT = GeminiAuthBase.CALLBACK_PORT
except ImportError:
    GEMINI_CLI_DEFAULT_OAUTH_PORT = 8085

try:
    from rotator_library.providers.antigravity_auth_base import AntigravityAuthBase

    ANTIGRAVITY_DEFAULT_OAUTH_PORT = AntigravityAuthBase.CALLBACK_PORT
except ImportError:
    ANTIGRAVITY_DEFAULT_OAUTH_PORT = 51121

try:
    from rotator_library.providers.iflow_auth_base import (
        CALLBACK_PORT as IFLOW_DEFAULT_OAUTH_PORT,
    )
except ImportError:
    IFLOW_DEFAULT_OAUTH_PORT = 11451

# Antigravity provider environment variables
ANTIGRAVITY_SETTINGS = {
    "ANTIGRAVITY_SIGNATURE_CACHE_TTL": {
        "type": "int",
        "default": 3600,
        "description": "Memory cache TTL for Gemini 3 thought signatures (seconds)",
    },
    "ANTIGRAVITY_SIGNATURE_DISK_TTL": {
        "type": "int",
        "default": 86400,
        "description": "Disk cache TTL for Gemini 3 thought signatures (seconds)",
    },
    "ANTIGRAVITY_PRESERVE_THOUGHT_SIGNATURES": {
        "type": "bool",
        "default": True,
        "description": "Preserve thought signatures in client responses",
    },
    "ANTIGRAVITY_ENABLE_SIGNATURE_CACHE": {
        "type": "bool",
        "default": True,
        "description": "Enable signature caching for multi-turn conversations",
    },
    "ANTIGRAVITY_ENABLE_DYNAMIC_MODELS": {
        "type": "bool",
        "default": False,
        "description": "Enable dynamic model discovery from API",
    },
    "ANTIGRAVITY_GEMINI3_TOOL_FIX": {
        "type": "bool",
        "default": True,
        "description": "Enable Gemini 3 tool hallucination prevention",
    },
    "ANTIGRAVITY_CLAUDE_TOOL_FIX": {
        "type": "bool",
        "default": True,
        "description": "Enable Claude tool hallucination prevention",
    },
    "ANTIGRAVITY_CLAUDE_THINKING_SANITIZATION": {
        "type": "bool",
        "default": True,
        "description": "Sanitize thinking blocks for Claude multi-turn conversations",
    },
    "ANTIGRAVITY_GEMINI3_TOOL_PREFIX": {
        "type": "str",
        "default": "gemini3_",
        "description": "Prefix added to tool names for Gemini 3 disambiguation",
    },
    "ANTIGRAVITY_GEMINI3_DESCRIPTION_PROMPT": {
        "type": "str",
        "default": "\n\nSTRICT PARAMETERS: {params}.",
        "description": "Template for strict parameter hints in tool descriptions",
    },
    "ANTIGRAVITY_CLAUDE_DESCRIPTION_PROMPT": {
        "type": "str",
        "default": "\n\nSTRICT PARAMETERS: {params}.",
        "description": "Template for Claude strict parameter hints in tool descriptions",
    },
    "ANTIGRAVITY_OAUTH_PORT": {
        "type": "int",
        "default": ANTIGRAVITY_DEFAULT_OAUTH_PORT,
        "description": "Local port for OAuth callback server during authentication",
    },
}

# Gemini CLI provider environment variables
GEMINI_CLI_SETTINGS = {
    "GEMINI_CLI_SIGNATURE_CACHE_TTL": {
        "type": "int",
        "default": 3600,
        "description": "Memory cache TTL for thought signatures (seconds)",
    },
    "GEMINI_CLI_SIGNATURE_DISK_TTL": {
        "type": "int",
        "default": 86400,
        "description": "Disk cache TTL for thought signatures (seconds)",
    },
    "GEMINI_CLI_PRESERVE_THOUGHT_SIGNATURES": {
        "type": "bool",
        "default": True,
        "description": "Preserve thought signatures in client responses",
    },
    "GEMINI_CLI_ENABLE_SIGNATURE_CACHE": {
        "type": "bool",
        "default": True,
        "description": "Enable signature caching for multi-turn conversations",
    },
    "GEMINI_CLI_GEMINI3_TOOL_FIX": {
        "type": "bool",
        "default": True,
        "description": "Enable Gemini 3 tool hallucination prevention",
    },
    "GEMINI_CLI_GEMINI3_TOOL_PREFIX": {
        "type": "str",
        "default": "gemini3_",
        "description": "Prefix added to tool names for Gemini 3 disambiguation",
    },
    "GEMINI_CLI_GEMINI3_DESCRIPTION_PROMPT": {
        "type": "str",
        "default": "\n\nSTRICT PARAMETERS: {params}.",
        "description": "Template for strict parameter hints in tool descriptions",
    },
    "GEMINI_CLI_PROJECT_ID": {
        "type": "str",
        "default": "",
        "description": "GCP Project ID for paid tier users (required for paid tiers)",
    },
    "GEMINI_CLI_OAUTH_PORT": {
        "type": "int",
        "default": GEMINI_CLI_DEFAULT_OAUTH_PORT,
        "description": "Local port for OAuth callback server during authentication",
    },
}

# iFlow provider environment variables
IFLOW_SETTINGS = {
    "IFLOW_OAUTH_PORT": {
        "type": "int",
        "default": IFLOW_DEFAULT_OAUTH_PORT,
        "description": "Local port for OAuth callback server during authentication",
    },
}

# Map provider names to their settings definitions
PROVIDER_SETTINGS_MAP = {
    "antigravity": ANTIGRAVITY_SETTINGS,
    "gemini_cli": GEMINI_CLI_SETTINGS,
    "iflow": IFLOW_SETTINGS,
}
