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
    from .circuit_breaker import ProviderCircuitBreaker, CircuitState
    from .ip_throttle_detector import IPThrottleDetector, ThrottleScope
    from .error_handler import get_retry_backoff

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
    # Resilience modules
    "ProviderCircuitBreaker",
    "CircuitState",
    "IPThrottleDetector",
    "ThrottleScope",
    "get_retry_backoff",
    # Custom provider support
    "AllProviders",
    "get_all_providers",
    "is_provider_abort",
    "classify_stream_error",
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
    # Resilience modules
    if name == "ProviderCircuitBreaker":
        from .circuit_breaker import ProviderCircuitBreaker
        return ProviderCircuitBreaker
    if name == "CircuitState":
        from .circuit_breaker import CircuitState
        return CircuitState
    # IP throttle detection
    if name == "IPThrottleDetector":
        from .ip_throttle_detector import IPThrottleDetector
        return IPThrottleDetector
    if name == "ThrottleScope":
        from .ip_throttle_detector import ThrottleScope
        return ThrottleScope
    if name == "get_retry_backoff":
        from .error_handler import get_retry_backoff
        return get_retry_backoff
    # Custom provider support
    if name == "AllProviders":
        from .error_handler import AllProviders
        return AllProviders
    if name == "get_all_providers":
        from .error_handler import get_all_providers
        return get_all_providers
    if name == "is_provider_abort":
        from .error_handler import is_provider_abort
        return is_provider_abort
    if name == "classify_stream_error":
        from .error_handler import classify_stream_error
        return classify_stream_error
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
