# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Data merger -- selects best source and creates ModelMetadata.

Extracted from model_info_service for modularity.
"""

from typing import Dict, List, Optional, Tuple

from ._types import ModelMetadata, ModelPricing, ModelLimits, ModelCapabilities, ModelInfo
from ._constants import _get_provider_priority, _extract_provider_from_source_id


class DataMerger:
    """
    Selects best source and creates ModelMetadata for queried model.

    Key principle: For custom provider models (like antigravity/claude-opus-4-5),
    we inherit technical specs from the best matching native provider source
    (like anthropic/claude-opus-4.5), but keep the queried model's identity.
    """

    @staticmethod
    def create_metadata(
        queried_model_id: str,
        records: List[Tuple[Dict, str]],
        quality: str,
    ) -> ModelMetadata:
        """
        Create ModelMetadata for the queried model.

        For fuzzy matches, picks the best source based on provider priority
        rather than merging multiple sources (which would average pricing incorrectly).

        The queried model's provider is preserved in owned_by, while technical
        specs come from the best matching source.
        """
        if not records:
            raise ValueError("No records to create metadata from")

        # Extract the queried provider from the model ID
        queried_parts = queried_model_id.split("/")
        queried_provider = queried_parts[0] if queried_parts else "unknown"

        # Pick the best source based on provider priority
        best_record, best_origin = DataMerger._select_best_source(records)

        # Extract parent model ID from origin for transparency
        parent_model_id = DataMerger._extract_model_id_from_origin(best_origin)

        return ModelMetadata(
            model_id=queried_model_id,
            display_name=best_record.get("name", queried_model_id.split("/")[-1]),
            # Use QUERIED provider, not source provider
            provider=queried_provider,
            category=best_record.get("category", "chat"),
            pricing=ModelPricing(
                prompt=best_record.get("prompt_cost"),
                completion=best_record.get("completion_cost"),
                cached_input=best_record.get("cache_read_cost"),
                cache_write=best_record.get("cache_write_cost"),
            ),
            limits=ModelLimits(
                context_window=best_record.get("context") or None,
                max_output=best_record.get("max_out") or None,
            ),
            capabilities=ModelCapabilities(
                tools=best_record.get("has_tools", False),
                functions=best_record.get("has_functions", False),
                reasoning=best_record.get("has_reasoning", False),
                vision=best_record.get("has_vision", False),
                # Extended capabilities
                structured_output=best_record.get("has_structured_output", False),
                temperature=best_record.get("has_temperature", True),
                attachments=best_record.get("has_attachments", False),
                interleaved=best_record.get("has_interleaved", False),
            ),
            info=ModelInfo(
                family=best_record.get("family", ""),
                description=best_record.get("description", ""),
                knowledge_cutoff=best_record.get("knowledge_cutoff", ""),
                release_date=best_record.get("release_date", ""),
                open_weights=best_record.get("open_weights", False),
                status=best_record.get("status", "active"),
                tokenizer=best_record.get("tokenizer", ""),
                huggingface_id=best_record.get("huggingface_id", ""),
            ),
            input_types=best_record.get("inputs", ["text"]),
            output_types=best_record.get("outputs", ["text"]),
            supported_parameters=best_record.get("supported_parameters", []),
            origin=f"{best_origin}|parent:{parent_model_id}"
            if parent_model_id
            else best_origin,
            match_quality=quality,
        )

    @staticmethod
    def _select_best_source(records: List[Tuple[Dict, str]]) -> Tuple[Dict, str]:
        """
        Select the best source from multiple candidates based on provider priority.

        Prefers native providers (anthropic, openai, google) over proxy/aggregator
        providers (azure, openrouter, requesty, etc.).

        When multiple sources have the same extracted provider (e.g., both
        requesty/anthropic/model and anthropic/model extract to anthropic),
        prefer the source where the first segment is the native provider
        (i.e., anthropic/model is preferred over requesty/anthropic/model).
        """
        if len(records) == 1:
            return records[0]

        def get_sort_key(record_tuple: Tuple[Dict, str]) -> Tuple[int, int, int]:
            data, origin = record_tuple
            # Extract source_id from origin string like "modelsdev:fuzzy:anthropic/claude-opus-4.5"
            source_id = origin.split(":")[-1] if ":" in origin else origin

            # Primary: priority of extracted provider (handles nested paths)
            provider = _extract_provider_from_source_id(source_id)
            primary_priority = _get_provider_priority(provider)

            # Secondary: prefer sources where first segment is a native provider
            # This ensures anthropic/model wins over requesty/anthropic/model
            parts = source_id.split("/")
            first_segment = parts[0].lower() if parts else ""
            first_segment_priority = _get_provider_priority(first_segment)

            # Tertiary: prefer shorter paths (2-segment over 3-segment)
            # This is a tiebreaker when both have same first segment priority
            path_length = len(parts)

            return (primary_priority, first_segment_priority, path_length)

        # Sort by priority tuple (lower is better) and return the best
        sorted_records = sorted(records, key=get_sort_key)
        return sorted_records[0]

    @staticmethod
    def _extract_model_id_from_origin(origin: str) -> Optional[str]:
        """
        Extract the source model ID from an origin string.

        Examples:
            "modelsdev:fuzzy:anthropic/claude-opus-4.5" -> "anthropic/claude-opus-4.5"
            "openrouter:exact:openrouter/google/gemini-2.5-pro" -> "google/gemini-2.5-pro"
        """
        if ":" not in origin:
            return None

        parts = origin.split(":")
        if len(parts) >= 3:
            source_id = parts[-1]
            # Remove openrouter prefix if present
            if source_id.startswith("openrouter/"):
                source_id = source_id[len("openrouter/") :]
            return source_id
        return None

    # Legacy method for backward compatibility
    @staticmethod
    def single(model_id: str, data: Dict, origin: str, quality: str) -> ModelMetadata:
        """Create ModelMetadata from a single source record. Legacy method."""
        return DataMerger.create_metadata(model_id, [(data, origin)], quality)

    # Legacy method for backward compatibility
    @staticmethod
    def combine(
        model_id: str, records: List[Tuple[Dict, str]], quality: str
    ) -> ModelMetadata:
        """Create ModelMetadata from records. Now uses best-source selection."""
        return DataMerger.create_metadata(model_id, records, quality)
