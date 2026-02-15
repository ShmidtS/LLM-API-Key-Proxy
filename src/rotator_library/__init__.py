# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

from typing import TYPE_CHECKING, Dict, Type

from .client import RotatingClient

# For type checkers (Pylint, mypy), import PROVIDER_PLUGINS statically
# At runtime, it's lazy-loaded via __getattr__
if TYPE_CHECKING:
    from .providers import PROVIDER_PLUGINS
    from .providers.provider_interface import ProviderInterface
    from .model_info_service import ModelInfoService, ModelInfo, ModelMetadata
    from . import anthropic_compat
    from .http_client_pool import HttpClientPool, get_http_pool, close_http_pool
    from .credential_weight_cache import CredentialWeightCache, get_weight_cache
    from .batched_persistence import BatchedPersistence, UsagePersistenceManager

__all__ = [
    "RotatingClient",
    "PROVIDER_PLUGINS",
    "ModelInfoService",
    "ModelInfo",
    "ModelMetadata",
    "anthropic_compat",
    # Performance optimization modules
    "HttpClientPool",
    "get_http_pool",
    "close_http_pool",
    "CredentialWeightCache",
    "get_weight_cache",
    "BatchedPersistence",
    "UsagePersistenceManager",
]


def __getattr__(name):
    """Lazy-load PROVIDER_PLUGINS, ModelInfoService, and anthropic_compat to speed up module import."""
    if name == "PROVIDER_PLUGINS":
        from .providers import PROVIDER_PLUGINS

        return PROVIDER_PLUGINS
    if name == "ModelInfoService":
        from .model_info_service import ModelInfoService

        return ModelInfoService
    if name == "ModelInfo":
        from .model_info_service import ModelInfo

        return ModelInfo
    if name == "ModelMetadata":
        from .model_info_service import ModelMetadata

        return ModelMetadata
    if name == "anthropic_compat":
        from . import anthropic_compat

        return anthropic_compat
    # Performance optimization modules
    if name == "HttpClientPool":
        from .http_client_pool import HttpClientPool
        return HttpClientPool
    if name == "get_http_pool":
        from .http_client_pool import get_http_pool
        return get_http_pool
    if name == "close_http_pool":
        from .http_client_pool import close_http_pool
        return close_http_pool
    if name == "CredentialWeightCache":
        from .credential_weight_cache import CredentialWeightCache
        return CredentialWeightCache
    if name == "get_weight_cache":
        from .credential_weight_cache import get_weight_cache
        return get_weight_cache
    if name == "BatchedPersistence":
        from .batched_persistence import BatchedPersistence
        return BatchedPersistence
    if name == "UsagePersistenceManager":
        from .batched_persistence import UsagePersistenceManager
        return UsagePersistenceManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
