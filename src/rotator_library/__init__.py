# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from typing import TYPE_CHECKING, Dict

from .client import RotatingClient
from .utils.json_utils import STREAM_DONE

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
    "STREAM_DONE",
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
    "is_provider_abort",
    "classify_stream_error",
]


_LAZY_IMPORTS = {
    "PROVIDER_PLUGINS": (".providers", "PROVIDER_PLUGINS"),
    "ModelInfoService": (".model_info_service", "ModelInfoService"),
    "ModelInfo": (".model_info_service", "ModelInfo"),
    "ModelMetadata": (".model_info_service", "ModelMetadata"),
    "anthropic_compat": (".anthropic_compat", None),
    "HttpClientPool": (".http_client_pool", "HttpClientPool"),
    "get_http_pool": (".http_client_pool", "get_http_pool"),
    "close_http_pool": (".http_client_pool", "close_http_pool"),
    "CredentialWeightCache": (".credential_weight_cache", "CredentialWeightCache"),
    "get_weight_cache": (".credential_weight_cache", "get_weight_cache"),
    "BatchedPersistence": (".batched_persistence", "BatchedPersistence"),
    "UsagePersistenceManager": (".batched_persistence", "UsagePersistenceManager"),
    "ProviderCircuitBreaker": (".circuit_breaker", "ProviderCircuitBreaker"),
    "CircuitState": (".circuit_breaker", "CircuitState"),
    "IPThrottleDetector": (".ip_throttle_detector", "IPThrottleDetector"),
    "ThrottleScope": (".ip_throttle_detector", "ThrottleScope"),
    "get_retry_backoff": (".error_handler", "get_retry_backoff"),
    "is_provider_abort": (".error_handler", "is_provider_abort"),
    "classify_stream_error": (".error_handler", "classify_stream_error"),
}


def __getattr__(name):
    """Lazy-load heavy modules to speed up initial import."""
    entry = _LAZY_IMPORTS.get(name)
    if entry is not None:
        import importlib

        module_path, attr_name = entry
        module = importlib.import_module(module_path, __name__)
        value = module if attr_name is None else getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
