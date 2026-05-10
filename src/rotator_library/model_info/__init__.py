# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
model_info -- extracted components from model_info_service.

Public API remains on model_info_service; this package provides
modular imports for internal use.
"""

from ._types import (
    ModelPricing,
    ModelLimits,
    ModelCapabilities,
    ModelInfo,
    ModelMetadata,
)
from ._constants import (
    NATIVE_PROVIDER_PRIORITY,
    PROVIDER_ALIASES,
    _get_provider_priority,
    _extract_provider_from_source_id,
)
from .model_index import ModelIndex
from .data_merger import DataMerger

__all__ = [
    "ModelPricing",
    "ModelLimits",
    "ModelCapabilities",
    "ModelInfo",
    "ModelMetadata",
    "NATIVE_PROVIDER_PRIORITY",
    "PROVIDER_ALIASES",
    "ModelIndex",
    "DataMerger",
]
