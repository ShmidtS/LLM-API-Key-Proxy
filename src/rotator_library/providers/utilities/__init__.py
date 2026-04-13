# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# Utilities for provider implementations
from .base_quota_tracker import BaseQuotaTracker
from .antigravity_quota_tracker import AntigravityQuotaTracker
from .gemini_cli_quota_tracker import GeminiCliQuotaTracker

# Shared utilities for Gemini-based providers
from ...config import env_bool, env_int
from .gemini_shared_utils import (
    inline_schema_refs,
    normalize_type_arrays,
    clean_gemini_schema,
    recursively_parse_json_strings,
    GEMINI3_TOOL_RENAMES,
    GEMINI3_TOOL_RENAMES_REVERSE,
    FINISH_REASON_MAP,
    DEFAULT_GENERIC_SAFETY_SETTINGS,
    DEFAULT_SAFETY_SETTINGS,
    DEFAULT_GEMINI_SAFETY_SETTINGS_MAP,
)
from .gemini_tool_handler import GeminiToolHandler
from .gemini_credential_manager import GeminiCredentialManager
from .google_project_discovery import GoogleProjectDiscoveryMixin

# Message transformer for Gemini format conversion
from .message_transformer import (
    transform_messages_for_gemini,
    build_tool_call_id_to_name_mapping,
    ensure_user_first,
)

# Re-export loggers from transaction_logger for backward compatibility
from ...transaction_logger import (
    ProviderLogger,
    AntigravityProviderLogger,
)

__all__ = [
    # Quota trackers
    "BaseQuotaTracker",
    "AntigravityQuotaTracker",
    "GeminiCliQuotaTracker",
    # Shared utilities
    "env_bool",
    "env_int",
    "inline_schema_refs",
    "normalize_type_arrays",
    "clean_gemini_schema",
    "recursively_parse_json_strings",
    "GEMINI3_TOOL_RENAMES",
    "GEMINI3_TOOL_RENAMES_REVERSE",
    "FINISH_REASON_MAP",
    "DEFAULT_GENERIC_SAFETY_SETTINGS",
    "DEFAULT_SAFETY_SETTINGS",
    "DEFAULT_GEMINI_SAFETY_SETTINGS_MAP",
    # Loggers (from transaction_logger)
    "ProviderLogger",
    "AntigravityProviderLogger",
    # Mixins
    "GeminiToolHandler",
    "GeminiCredentialManager",
    "GoogleProjectDiscoveryMixin",
    # Message transformer
    "transform_messages_for_gemini",
    "build_tool_call_id_to_name_mapping",
    "ensure_user_first",
]
