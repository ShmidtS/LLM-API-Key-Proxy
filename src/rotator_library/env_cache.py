# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import os
from typing import Dict


# Cached environment variables for provider configuration
# Pre-filtered at module load to avoid repeated O(P*E) scans in __init__
_PROVIDER_ENV_PREFIXES = (
    "CONCURRENCY_MULTIPLIER_",
    "CUSTOM_CAP_",
    "CUSTOM_CAP_COOLDOWN_",
    "FAIR_CYCLE_",
    "FAIR_CYCLE_TRACKING_MODE_",
    "FAIR_CYCLE_CROSS_TIER_",
    "FAIR_CYCLE_DURATION_",
    "EXHAUSTION_COOLDOWN_THRESHOLD_",
    "_API_HEADERS",
)

_provider_env_cache: Dict[str, str] = {
    k: v for k, v in os.environ.items()
    if any(k.startswith(p) or k.endswith(p) for p in _PROVIDER_ENV_PREFIXES)
}

# Add global exhaustion cooldown threshold (no provider suffix)
if "EXHAUSTION_COOLDOWN_THRESHOLD" in os.environ:
    _provider_env_cache["EXHAUSTION_COOLDOWN_THRESHOLD"] = os.environ["EXHAUSTION_COOLDOWN_THRESHOLD"]
