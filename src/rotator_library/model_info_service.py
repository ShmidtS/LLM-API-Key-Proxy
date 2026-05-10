# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Unified Model Registry

Provides aggregated model metadata from external catalogs (OpenRouter, Models.dev)
for pricing calculations and the /v1/models endpoint.

Data retrieval happens asynchronously post-startup to keep initialization fast.
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

from .utils.json_utils import json_loads
from .config.defaults import env_int
from .utils.singleton import SingletonMeta

# Re-export from extracted modules for backward compatibility
from .model_info._types import (  # noqa: F401
    ModelPricing,
    ModelLimits,
    ModelCapabilities,
    ModelInfo,
    ModelMetadata,
)
from .model_info._constants import (  # noqa: F401
    NATIVE_PROVIDER_PRIORITY,
    PROVIDER_ALIASES,
    _get_provider_priority,
    _extract_provider_from_source_id,
)
from .model_info.model_index import ModelIndex  # noqa: F401
from .model_info.data_merger import DataMerger  # noqa: F401

logger = logging.getLogger(__name__)


# ============================================================================
# Data Source Adapters
# ============================================================================


class DataSourceAdapter:
    """Base interface for external data sources."""

    source_name: str = "unknown"
    endpoint: str = ""

    def fetch(self) -> Dict[str, Dict]:
        """Retrieve and normalize data. Returns {model_id: raw_data}."""
        raise NotImplementedError

    def _http_get(self, url: str, timeout: int = 30) -> Any:
        """Execute HTTP GET with standard headers."""
        req = Request(url, headers={"User-Agent": "ModelRegistry/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            return json_loads(resp.read().decode("utf-8"))

    def _fetch_json(self) -> Any:
        """
        Fetch JSON from self.endpoint with unified error handling.

        Wraps _http_get with the try/except + ConnectionError rewrite
        shared by all adapters.
        """
        try:
            return self._http_get(self.endpoint)
        except (URLError, json.JSONDecodeError, TimeoutError, ConnectionError) as err:
            logger.error("%s fetch failed: %s", self.source_name, err)
            raise ConnectionError(f"{self.source_name} unavailable: {err}") from err


class OpenRouterAdapter(DataSourceAdapter):
    """Fetches model data from OpenRouter's public API."""

    source_name = "openrouter"
    endpoint = "https://openrouter.ai/api/v1/models"

    def fetch(self) -> Dict[str, Dict]:
        raw = self._fetch_json()
        entries = raw.get("data", [])

        catalog = {}
        for entry in entries:
            mid = entry.get("id")
            if not mid:
                continue

            full_id = f"openrouter/{mid}"
            catalog[full_id] = self._normalize(entry)

        return catalog

    def _normalize(self, raw: Dict) -> Dict:
        """Transform OpenRouter schema to internal format."""
        prices = raw.get("pricing", {})
        arch = raw.get("architecture", {})
        top = raw.get("top_provider", {})
        params = raw.get("supported_parameters", [])

        tokenizer = arch.get("tokenizer", "")
        category = "embedding" if "embedding" in tokenizer.lower() else "chat"

        # Extract cache pricing
        cache_read = prices.get("input_cache_read", 0)
        cache_write = prices.get("input_cache_write", 0)

        return {
            # Basic info
            "name": raw.get("name", ""),
            "original_id": raw.get("id", ""),
            "provider": "openrouter",
            "source": "openrouter",
            "category": category,
            # Pricing (already per-token from OpenRouter)
            "prompt_cost": float(prices.get("prompt", 0)),
            "completion_cost": float(prices.get("completion", 0)),
            "cache_read_cost": float(cache_read) if cache_read else None,
            "cache_write_cost": float(cache_write) if cache_write else None,
            # Limits
            "context": top.get("context_length", 0),
            "max_out": top.get("max_completion_tokens", 0),
            # Modalities
            "inputs": arch.get("input_modalities", ["text"]),
            "outputs": arch.get("output_modalities", ["text"]),
            # Capabilities
            "has_tools": "tool_choice" in params or "tools" in params,
            "has_functions": "tools" in params or "function_calling" in params,
            "has_reasoning": "reasoning" in params or "include_reasoning" in params,
            "has_vision": "image" in arch.get("input_modalities", []),
            "has_structured_output": "structured_outputs" in params
            or "response_format" in params,
            "has_temperature": "temperature" in params,
            "has_attachments": "file" in arch.get("input_modalities", []),
            "has_interleaved": False,  # Not available from OpenRouter
            # Extended model info
            "description": raw.get("description", ""),
            "tokenizer": tokenizer,
            "huggingface_id": raw.get("hugging_face_id", ""),
            "supported_parameters": params,
            # OpenRouter doesn't provide these, leave empty
            "family": "",
            "knowledge_cutoff": "",
            "release_date": "",
            "open_weights": False,
            "status": "active",
        }


class ModelsDevAdapter(DataSourceAdapter):
    """Fetches model data from Models.dev catalog."""

    source_name = "modelsdev"
    endpoint = "https://models.dev/api.json"

    def __init__(self, skip_providers: Optional[List[str]] = None):
        self.skip_providers = skip_providers or []

    def fetch(self) -> Dict[str, Dict]:
        raw = self._fetch_json()

        catalog = {}
        for provider_key, provider_block in raw.items():
            if not isinstance(provider_block, dict):
                continue
            if provider_key in self.skip_providers:
                continue

            models_block = provider_block.get("models", {})
            if not isinstance(models_block, dict):
                continue

            for model_key, model_data in models_block.items():
                if not isinstance(model_data, dict):
                    continue

                full_id = f"{provider_key}/{model_key}"
                catalog[full_id] = self._normalize(model_data, provider_key)

        return catalog

    def _normalize(self, raw: Dict, provider_key: str) -> Dict:
        """Transform Models.dev schema to internal format."""
        costs = raw.get("cost", {})
        mods = raw.get("modalities", {})
        lims = raw.get("limit", {})

        outputs = mods.get("output", ["text"])
        if "image" in outputs:
            category = "image"
        elif "audio" in outputs:
            category = "audio"
        else:
            category = "chat"

        # Models.dev uses per-million pricing, convert to per-token
        divisor = 1_000_000

        cache_read = costs.get("cache_read")
        cache_write = costs.get("cache_write")

        return {
            # Basic info
            "name": raw.get("name", ""),
            "original_id": raw.get("id", ""),
            "provider": provider_key,
            "source": "modelsdev",
            "category": category,
            # Pricing (converted to per-token)
            "prompt_cost": float(costs.get("input", 0)) / divisor,
            "completion_cost": float(costs.get("output", 0)) / divisor,
            "cache_read_cost": float(cache_read) / divisor if cache_read else None,
            "cache_write_cost": float(cache_write) / divisor if cache_write else None,
            # Limits
            "context": lims.get("context", 0),
            "max_out": lims.get("output", 0),
            # Modalities
            "inputs": mods.get("input", ["text"]),
            "outputs": outputs,
            # Capabilities
            "has_tools": raw.get("tool_call", False),
            "has_functions": raw.get("tool_call", False),
            "has_reasoning": raw.get("reasoning", False),
            "has_vision": "image" in mods.get("input", []),
            "has_structured_output": raw.get("structured_output", False),
            "has_temperature": raw.get("temperature", True),
            "has_attachments": raw.get("attachment", False),
            "has_interleaved": raw.get("interleaved", False),
            # Extended model info
            "family": raw.get("family", ""),
            "knowledge_cutoff": raw.get("knowledge", ""),
            "release_date": raw.get("release_date", ""),
            "open_weights": raw.get("open_weights", False),
            "status": raw.get("status", "active"),
        }


# ============================================================================
# Main Registry Service
# ============================================================================


class ModelRegistry(metaclass=SingletonMeta):
    """
    Central registry for model metadata from external catalogs.

    Manages background data refresh and provides lookup/pricing APIs.
    """

    REFRESH_INTERVAL_DEFAULT = 6 * 60 * 60  # 6 hours

    def __init__(
        self,
        refresh_seconds: Optional[int] = None,
        skip_modelsdev_providers: Optional[List[str]] = None,
    ):
        self._refresh_interval = refresh_seconds or env_int(
            "MODEL_INFO_REFRESH_INTERVAL", self.REFRESH_INTERVAL_DEFAULT
        )

        # Configure adapters
        self._adapters: List[DataSourceAdapter] = [
            OpenRouterAdapter(),
            ModelsDevAdapter(skip_providers=skip_modelsdev_providers or []),
        ]

        # Raw data stores
        self._openrouter_store: Dict[str, Dict] = {}
        self._modelsdev_store: Dict[str, Dict] = {}

        # Lookup infrastructure
        self._index = ModelIndex()
        self._result_cache: OrderedDict[str, ModelMetadata] = OrderedDict()
        self._negative_cache: OrderedDict[str, float] = OrderedDict()
        self._negative_cache_ttl = 300.0
        self._cache_maxsize = 2048
        self._raw_models_cache: Optional[Dict[str, Dict]] = None

        # Async coordination
        self._ready = asyncio.Event()
        self._mutex = asyncio.Lock()
        self._worker: Optional[asyncio.Task] = None
        self._last_refresh: float = 0

    # ---------- Lifecycle ----------

    async def start(self):
        """Begin background refresh worker."""
        if self._worker is None:
            self._worker = asyncio.create_task(self._refresh_worker())
            logger.info(
                "ModelRegistry started (refresh every %ds)", self._refresh_interval
            )

    async def stop(self):
        """Halt background worker."""
        if self._worker:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                logger.debug("Model registry worker cancelled during stop", exc_info=True)
            self._worker = None
            logger.info("ModelRegistry stopped")

    async def await_ready(self, timeout_secs: float = 30.0) -> bool:
        """Block until initial data load completes."""
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout_secs)
            return True
        except asyncio.TimeoutError:
            logger.warning("ModelRegistry ready timeout after %.1fs", timeout_secs)
            return False

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    # ---------- Background Worker ----------

    async def _refresh_worker(self):
        """Periodic refresh loop."""
        await self._load_all_sources()
        self._ready.set()

        while True:
            try:
                await asyncio.sleep(self._refresh_interval)
                logger.info("Scheduled registry refresh...")
                await self._load_all_sources()
                logger.info("Registry refresh complete")
            except asyncio.CancelledError:
                break
            except Exception as ex:
                logger.error("Registry refresh error: %s", ex, exc_info=True)

    async def _load_all_sources(self):
        """Fetch from all adapters concurrently."""
        loop = asyncio.get_running_loop()

        tasks = [
            loop.run_in_executor(None, adapter.fetch) for adapter in self._adapters
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        async with self._mutex:
            for adapter, result in zip(self._adapters, results):
                if isinstance(result, Exception):
                    logger.error("%s fetch failed: %s", adapter.source_name, result)
                    continue

                if adapter.source_name == "openrouter":
                    self._openrouter_store = result
                    logger.info("OpenRouter: %d models loaded", len(result) if isinstance(result, dict) else 0)
                elif adapter.source_name == "modelsdev":
                    self._modelsdev_store = result
                    logger.info("Models.dev: %d models loaded", len(result) if isinstance(result, dict) else 0)

            self._rebuild_index()
            self._last_refresh = time.time()

    def _rebuild_index(self):
        """Reconstruct lookup index from current stores."""
        self._index.clear()
        self._result_cache.clear()
        self._negative_cache.clear()
        self._raw_models_cache = None

        for model_id in self._openrouter_store:
            self._index.add(model_id)

        for model_id in self._modelsdev_store:
            self._index.add(model_id)

    # ---------- Query API ----------

    def lookup(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Retrieve model metadata by ID.

        Matching strategy:
        1. Exact match against known IDs
        2. Fuzzy match by model name suffix
        3. Aggregate if multiple sources match
        """
        if model_id in self._result_cache:
            self._result_cache.move_to_end(model_id)
            return self._result_cache[model_id]

        now = time.time()
        cached_miss = self._negative_cache.get(model_id)
        if cached_miss is not None and (now - cached_miss) < self._negative_cache_ttl:
            self._negative_cache.move_to_end(model_id)
            return None

        metadata = self._resolve_model(model_id)
        if metadata:
            if len(self._result_cache) >= self._cache_maxsize:
                self._result_cache.popitem(last=False)
            self._result_cache[model_id] = metadata
        else:
            if len(self._negative_cache) >= self._cache_maxsize:
                self._negative_cache.popitem(last=False)
            self._negative_cache[model_id] = time.time()
        return metadata

    def _resolve_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Build ModelMetadata by matching source data."""
        records: List[Tuple[Dict, str]] = []
        quality = "none"

        # Step 1: Check exact matches first
        or_key = (
            f"openrouter/{model_id}"
            if not model_id.startswith("openrouter/")
            else model_id
        )
        if or_key in self._openrouter_store:
            records.append(
                (self._openrouter_store[or_key], f"openrouter:exact:{or_key}")
            )
            quality = "exact"

        if model_id in self._modelsdev_store:
            records.append(
                (self._modelsdev_store[model_id], f"modelsdev:exact:{model_id}")
            )
            quality = "exact"

        # Step 2: Try provider alias substitution for direct match
        if not records:
            alias_candidates = self._get_alias_candidates(model_id)
            for alias_id in alias_candidates:
                # Try Models.dev first (usually more complete)
                if alias_id in self._modelsdev_store:
                    records.append(
                        (self._modelsdev_store[alias_id], f"modelsdev:alias:{alias_id}")
                    )
                    quality = "alias"
                    break  # Take first match
                # Try OpenRouter with prefix
                or_alias = f"openrouter/{alias_id}"
                if or_alias in self._openrouter_store:
                    records.append(
                        (
                            self._openrouter_store[or_alias],
                            f"openrouter:alias:{or_alias}",
                        )
                    )
                    quality = "alias"
                    break

        # Step 3: Fall back to fuzzy index search
        if not records:
            candidates = self._index.resolve(model_id)
            for cid in candidates:
                if cid in self._openrouter_store:
                    records.append(
                        (self._openrouter_store[cid], f"openrouter:fuzzy:{cid}")
                    )
                elif cid in self._modelsdev_store:
                    records.append(
                        (self._modelsdev_store[cid], f"modelsdev:fuzzy:{cid}")
                    )

            if records:
                quality = "fuzzy"

        if not records:
            return None

        return DataMerger.combine(model_id, records, quality)

    def _get_alias_candidates(self, model_id: str) -> List[str]:
        """
        Generate alternative model IDs by substituting provider aliases.

        Examples:
            nvidia_nim/mistralai/model -> nvidia/mistralai/model
            gemini_cli/gemini-2.5-flash -> google/gemini-2.5-flash
            gemini/gemini-2.5-pro -> google/gemini-2.5-pro
        """
        parts = model_id.split("/")
        if len(parts) < 2:
            return []

        provider = parts[0]
        rest = "/".join(parts[1:])

        candidates = []

        # Check if provider has aliases defined
        if provider in PROVIDER_ALIASES:
            for alias in PROVIDER_ALIASES[provider]:
                candidates.append(f"{alias}/{rest}")

        return candidates

    def get_pricing(self, model_id: str) -> Optional[Dict[str, float]]:
        """Extract just pricing info for cost calculations."""
        meta = self.lookup(model_id)
        if not meta:
            return None

        result = {}
        if meta.pricing.prompt is not None:
            result["input_cost_per_token"] = meta.pricing.prompt
        if meta.pricing.completion is not None:
            result["output_cost_per_token"] = meta.pricing.completion
        if meta.pricing.cached_input is not None:
            result["cache_read_input_token_cost"] = meta.pricing.cached_input
        if meta.pricing.cache_write is not None:
            result["cache_creation_input_token_cost"] = meta.pricing.cache_write

        return result if result else None

    def compute_cost(
        self,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> Optional[float]:
        """
        Calculate total request cost.

        Returns None if pricing unavailable.
        """
        pricing = self.get_pricing(model_id)
        if not pricing:
            return None

        in_rate = pricing.get("input_cost_per_token")
        out_rate = pricing.get("output_cost_per_token")

        if in_rate is None or out_rate is None:
            return None

        total = (input_tokens * in_rate) + (output_tokens * out_rate)

        cache_read_rate = pricing.get("cache_read_input_token_cost")
        if cache_read_rate and cache_hit_tokens:
            total += cache_hit_tokens * cache_read_rate

        cache_write_rate = pricing.get("cache_creation_input_token_cost")
        if cache_write_rate and cache_miss_tokens:
            total += cache_miss_tokens * cache_write_rate

        return total

    def enrich_models(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Attach metadata to a list of model IDs.

        Used by /v1/models endpoint.
        """
        enriched = []
        for mid in model_ids:
            meta = self.lookup(mid)
            if meta:
                enriched.append(meta.as_api_response())
            else:
                # Fallback minimal entry
                enriched.append(
                    {
                        "id": mid,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": mid.split("/")[0] if "/" in mid else "unknown",
                    }
                )
        return enriched

    def all_raw_models(self) -> Dict[str, Dict]:
        """Return all raw source data (for debugging)."""
        if self._raw_models_cache is not None:
            return self._raw_models_cache
        result = {**self._openrouter_store, **self._modelsdev_store}
        self._raw_models_cache = result
        return result

    def diagnostics(self) -> Dict[str, Any]:
        """Return service health/stats."""
        return {
            "ready": self._ready.is_set(),
            "last_refresh": self._last_refresh,
            "openrouter_count": len(self._openrouter_store),
            "modelsdev_count": len(self._modelsdev_store),
            "cached_lookups": len(self._result_cache),
            "index_entries": self._index.entry_count(),
            "refresh_interval": self._refresh_interval,
        }

    # ---------- Backward Compatibility Methods ----------

    def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        """Alias for lookup() - backward compatibility."""
        return self.lookup(model_id)

    def get_cost_info(self, model_id: str) -> Optional[Dict[str, float]]:
        """Alias for get_pricing() - backward compatibility."""
        return self.get_pricing(model_id)

    def calculate_cost(
        self,
        model_id: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> Optional[float]:
        """Alias for compute_cost() - backward compatibility."""
        return self.compute_cost(
            model_id,
            prompt_tokens,
            completion_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        )

    def enrich_model_list(self, model_ids: List[str]) -> List[Dict[str, Any]]:
        """Alias for enrich_models() - backward compatibility."""
        return self.enrich_models(model_ids)

    def get_all_source_models(self) -> Dict[str, Dict]:
        """Alias for all_raw_models() - backward compatibility."""
        return self.all_raw_models()

    def get_stats(self) -> Dict[str, Any]:
        """Alias for diagnostics() - backward compatibility."""
        return self.diagnostics()

    def wait_for_ready(self, timeout: float = 30.0):
        """Sync wrapper for await_ready() - for compatibility."""
        return self.await_ready(timeout)


# ============================================================================
# Backward Compatibility Layer
# ============================================================================

# Compat wrapper — SingletonMeta handles singleton lifecycle
def get_model_info_service() -> ModelRegistry:
    """Get or create the global registry instance (delegates to SingletonMeta)."""
    return ModelRegistry()


async def init_model_info_service() -> ModelRegistry:
    """Initialize and start the global registry."""
    registry = get_model_info_service()
    await registry.start()
    return registry
