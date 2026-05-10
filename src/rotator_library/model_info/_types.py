# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Data structures for model metadata.

Extracted from model_info_service to allow focused imports.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelPricing:
    """Token-level pricing information."""

    prompt: Optional[float] = None
    completion: Optional[float] = None
    cached_input: Optional[float] = None
    cache_write: Optional[float] = None


@dataclass
class ModelLimits:
    """Context and output token limits."""

    context_window: Optional[int] = None
    max_output: Optional[int] = None


@dataclass
class ModelCapabilities:
    """Feature flags for model capabilities."""

    tools: bool = False
    functions: bool = False
    reasoning: bool = False
    vision: bool = False
    system_prompt: bool = True
    caching: bool = False
    prefill: bool = False
    # Extended capabilities from Models.dev
    structured_output: bool = False
    temperature: bool = True  # Most models support temperature
    attachments: bool = False  # File/document attachments
    interleaved: bool = False  # Interleaved content support


@dataclass
class ModelInfo:
    """Extended model information and metadata."""

    family: str = ""  # Model family (e.g., "claude-opus", "gpt-4")
    description: str = ""  # Model description
    knowledge_cutoff: str = ""  # Knowledge cutoff date (e.g., "2025-03-31")
    release_date: str = ""  # Model release date
    open_weights: bool = False  # Whether model weights are open
    status: str = "active"  # Model status: active, deprecated, preview
    tokenizer: str = ""  # Tokenizer type
    huggingface_id: str = ""  # HuggingFace model ID


@dataclass
class ModelMetadata:
    """Complete model information record."""

    model_id: str
    display_name: str = ""
    provider: str = ""
    category: str = "chat"  # chat, embedding, image, audio

    pricing: ModelPricing = field(default_factory=ModelPricing)
    limits: ModelLimits = field(default_factory=ModelLimits)
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    info: ModelInfo = field(default_factory=ModelInfo)  # Extended info

    input_types: List[str] = field(default_factory=lambda: ["text"])
    output_types: List[str] = field(default_factory=lambda: ["text"])
    supported_parameters: List[str] = field(
        default_factory=list
    )  # Supported API params

    timestamp: int = field(default_factory=lambda: int(time.time()))
    origin: str = ""
    match_quality: str = "unknown"

    def as_api_response(self) -> Dict[str, Any]:
        """
        Format for OpenAI-compatible /v1/models response.

        Standard OpenAI fields come first, then extended fields,
        then debug/meta fields prefixed with underscore.
        """
        # === Core OpenAI-compatible fields ===
        response: Dict[str, Any] = {
            "id": self.model_id,
            "object": "model",
            "created": self.timestamp,
            "owned_by": self.provider or "proxy",
        }

        # === Token limits (standard) ===
        if self.limits.context_window:
            response["context_length"] = self.limits.context_window
        if self.limits.max_output:
            response["max_completion_tokens"] = self.limits.max_output

        # === Pricing fields (extended but common) ===
        if self.pricing.prompt is not None:
            response["pricing"] = {"prompt": self.pricing.prompt}
            if self.pricing.completion is not None:
                response["pricing"]["completion"] = self.pricing.completion
            if self.pricing.cached_input is not None:
                response["pricing"]["cached_input"] = self.pricing.cached_input
            if self.pricing.cache_write is not None:
                response["pricing"]["cache_write"] = self.pricing.cache_write

        # === Architecture/modalities (OpenRouter-style) ===
        response["architecture"] = {
            "input_modalities": self.input_types,
            "output_modalities": self.output_types,
        }
        if self.info.tokenizer:
            response["architecture"]["tokenizer"] = self.info.tokenizer  # type: ignore[assignment]

        # === Capabilities (extended) ===
        response["capabilities"] = {
            "tool_choice": self.capabilities.tools,
            "function_calling": self.capabilities.functions,
            "reasoning": self.capabilities.reasoning,
            "vision": self.capabilities.vision,
            "system_messages": self.capabilities.system_prompt,
            "prompt_caching": self.capabilities.caching,
            "assistant_prefill": self.capabilities.prefill,
            "structured_output": self.capabilities.structured_output,
            "temperature": self.capabilities.temperature,
            "attachments": self.capabilities.attachments,
            "interleaved": self.capabilities.interleaved,
        }

        # === Supported parameters (if available) ===
        if self.supported_parameters:
            response["supported_parameters"] = self.supported_parameters

        # === Extended model info ===
        if self.info.family:
            response["family"] = self.info.family
        if self.info.description:
            response["description"] = self.info.description
        if self.info.knowledge_cutoff:
            response["knowledge_cutoff"] = self.info.knowledge_cutoff
        if self.info.release_date:
            response["release_date"] = self.info.release_date
        if self.info.open_weights:
            response["open_weights"] = self.info.open_weights
        if self.info.status and self.info.status != "active":
            response["status"] = self.info.status
        if self.info.huggingface_id:
            response["huggingface_id"] = self.info.huggingface_id

        # === Legacy fields for backward compatibility ===
        if self.limits.context_window:
            response["max_input_tokens"] = self.limits.context_window
            response["context_window"] = self.limits.context_window
        if self.limits.max_output:
            response["max_output_tokens"] = self.limits.max_output
        if self.pricing.prompt is not None:
            response["input_cost_per_token"] = self.pricing.prompt
        if self.pricing.completion is not None:
            response["output_cost_per_token"] = self.pricing.completion
        if self.pricing.cached_input is not None:
            response["cache_read_input_token_cost"] = self.pricing.cached_input
        if self.pricing.cache_write is not None:
            response["cache_creation_input_token_cost"] = self.pricing.cache_write
        response["mode"] = self.category
        response["supported_modalities"] = self.input_types
        response["supported_output_modalities"] = self.output_types

        # === Debug/meta fields (underscore prefix) ===
        if self.origin:
            origin_parts = self.origin.split("|")
            main_origin = origin_parts[0]

            response["_sources"] = [main_origin]
            response["_match_type"] = self.match_quality

            for part in origin_parts[1:]:
                if part.startswith("parent:"):
                    response["_parent_model"] = part[len("parent:") :]
                    break

        return response

    def as_minimal(self) -> Dict[str, Any]:
        """Minimal OpenAI format."""
        return {
            "id": self.model_id,
            "object": "model",
            "created": self.timestamp,
            "owned_by": self.provider or "proxy",
        }

    def to_dict(self) -> Dict[str, Any]:
        """Alias for as_api_response() - backward compatibility."""
        return self.as_api_response()

    def to_openai_format(self) -> Dict[str, Any]:
        """Alias for as_minimal() - backward compatibility."""
        return self.as_minimal()

    # Backward-compatible property aliases
    @property
    def id(self) -> str:
        return self.model_id

    @property
    def name(self) -> str:
        return self.display_name

    @property
    def input_cost_per_token(self) -> Optional[float]:
        return self.pricing.prompt

    @property
    def output_cost_per_token(self) -> Optional[float]:
        return self.pricing.completion

    @property
    def cache_read_input_token_cost(self) -> Optional[float]:
        return self.pricing.cached_input

    @property
    def cache_creation_input_token_cost(self) -> Optional[float]:
        return self.pricing.cache_write

    @property
    def max_input_tokens(self) -> Optional[int]:
        return self.limits.context_window

    @property
    def max_output_tokens(self) -> Optional[int]:
        return self.limits.max_output

    @property
    def mode(self) -> str:
        return self.category

    @property
    def supported_modalities(self) -> List[str]:
        return self.input_types

    @property
    def supported_output_modalities(self) -> List[str]:
        return self.output_types

    @property
    def supports_tool_choice(self) -> bool:
        return self.capabilities.tools

    @property
    def supports_function_calling(self) -> bool:
        return self.capabilities.functions

    @property
    def supports_reasoning(self) -> bool:
        return self.capabilities.reasoning

    @property
    def supports_vision(self) -> bool:
        return self.capabilities.vision

    @property
    def supports_system_messages(self) -> bool:
        return self.capabilities.system_prompt

    @property
    def supports_prompt_caching(self) -> bool:
        return self.capabilities.caching

    @property
    def supports_assistant_prefill(self) -> bool:
        return self.capabilities.prefill

    @property
    def litellm_provider(self) -> str:
        return self.provider

    @property
    def created(self) -> int:
        return self.timestamp

    @property
    def _sources(self) -> List[str]:
        return [self.origin] if self.origin else []

    @property
    def _match_type(self) -> str:
        return self.match_quality
