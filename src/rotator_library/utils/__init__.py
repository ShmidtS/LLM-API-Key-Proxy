# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

# src/rotator_library/utils/__init__.py

from .headless_detection import is_headless_environment
from .paths import (
    get_default_root,
    get_logs_dir,
    get_cache_dir,
    get_oauth_dir,
    get_data_file,
)
from .provider_locks import ProviderLockManager
from .reauth_coordinator import get_reauth_coordinator, ReauthCoordinator
from .resilient_io import (
    ResilientStateWriter,
    safe_write_json,
    safe_log_write,
    safe_mkdir,
)
from .litellm_patches import patch_litellm_finish_reason, suppress_litellm_serialization_warnings, suppress_litellm_prints
from .http_retry import exponential_backoff_with_jitter
from .json_utils import json_dumps, json_dumps_str, json_loads, sse_data_event, STREAM_DONE
from .duration import parse_duration

__all__ = [
    "is_headless_environment",
    "get_default_root",
    "get_logs_dir",
    "get_cache_dir",
    "get_oauth_dir",
    "get_data_file",
    "ProviderLockManager",
    "get_reauth_coordinator",
    "ReauthCoordinator",
    "ResilientStateWriter",
    "safe_write_json",
    "safe_log_write",
    "safe_mkdir",
    "patch_litellm_finish_reason",
    "suppress_litellm_serialization_warnings",
    "suppress_litellm_prints",
    "exponential_backoff_with_jitter",
    "json_dumps",
    "json_dumps_str",
    "json_loads",
    "sse_data_event",
    "STREAM_DONE",
    "parse_duration",
]
