# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Message transform mixin for GeminiCliProvider.

Extracts message transformation, tool schema transformation, reasoning parameter
handling, and streaming chunk conversion from GeminiCliProvider into a reusable mixin.

Follows the same pattern as AntigravityProvider's MessageTransformMixin.

Providers using this mixin must also inherit from GeminiToolHandler (for
_enforce_strict_schema, _inject_signature_into_description, _strip_gemini3_prefix)
and must define these instance attributes:
- _is_gemini_3(model: str) -> bool
- _gemini3_tool_prefix: str
- _enable_gemini3_tool_fix: bool
- _enable_signature_cache: bool
- _signature_cache: ProviderCache-like (with .retrieve method)
- _preserve_signatures_in_client: bool
- _gemini3_enforce_strict_schema: bool
- _gemini3_description_prompt: str
- _gemini3_system_instruction: str
- _enable_json_string_parsing: bool
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from ...utils.json_utils import json_dumps_str
from .gemini_shared_utils import (
    GEMINI3_TOOL_RENAMES,
    clean_gemini_schema,
    inline_schema_refs,
    recursively_parse_json_strings,
)

lib_logger = logging.getLogger("rotator_library")


class GeminiCliMessageTransformMixin:
    """
    Mixin providing message and schema transformation methods for GeminiCliProvider.

    Methods:
    - _transform_messages: OpenAI messages -> Gemini contents + system instruction
    - _handle_reasoning_parameters: reasoning_effort -> thinkingConfig
    - _convert_chunk_to_openai: Gemini response chunks -> OpenAI streaming format
    - _gemini_cli_transform_schema: JSON schema cleanup for Gemini CLI
    - _transform_tool_schemas: OpenAI tool definitions -> Gemini functionDeclarations
    - _inject_gemini3_system_instruction: Gemini 3 tool-fix system prompt injection
    """

    def _transform_messages(
        self, messages: List[Dict[str, Any]], model: str = ""
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Gemini CLI format.

        Handles:
        - System instruction extraction
        - Multi-part content (text, images)
        - Tool calls and responses
        - Gemini 3 thoughtSignature preservation
        """
        from .message_transformer import transform_messages_for_gemini

        return transform_messages_for_gemini(
            messages=messages,
            model=model,
            is_gemini_3=self._is_gemini_3(model),
            gemini3_tool_prefix=self._gemini3_tool_prefix,
            enable_gemini3_tool_fix=self._enable_gemini3_tool_fix,
            enable_signature_cache=self._enable_signature_cache,
            signature_cache_retrieve=self._signature_cache.retrieve if self._enable_signature_cache else None,
            gemini3_tool_renames=GEMINI3_TOOL_RENAMES,
        )

    def _handle_reasoning_parameters(
        self, payload: Dict[str, Any], model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration for Gemini models.

        - Gemini 2.5: thinkingBudget (integer tokens)
        - Gemini 3 Pro: thinkingLevel (string: "low"/"high")
        - Gemini 3 Flash: thinkingLevel (string: "minimal"/"low"/"medium"/"high")
        """
        reasoning_effort = payload.pop("reasoning_effort", None)

        if "thinkingConfig" in payload.get("generationConfig", {}):
            return None

        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = self._is_gemini_3(model)
        is_gemini_3_flash = "gemini-3-flash" in model

        if not (is_gemini_25 or is_gemini_3):
            return None

        # Normalize and validate upfront
        if reasoning_effort is None:
            effort = "auto"
        elif isinstance(reasoning_effort, str):
            effort = reasoning_effort.strip().lower() or "auto"
        else:
            lib_logger.warning(
                f"[GeminiCLI] Invalid reasoning_effort type: {type(reasoning_effort).__name__}, using auto"
            )
            effort = "auto"

        valid_efforts = {
            "auto",
            "disable",
            "off",
            "none",
            "minimal",
            "low",
            "low_medium",
            "medium",
            "medium_high",
            "high",
        }
        if effort not in valid_efforts:
            lib_logger.warning(
                f"[GeminiCLI] Unknown reasoning_effort: '{reasoning_effort}', using auto"
            )
            effort = "auto"

        # Gemini 3 Flash: minimal/low/medium/high
        if is_gemini_3_flash:
            if effort in ("disable", "off", "none"):
                return {"thinkingLevel": "minimal", "include_thoughts": True}
            if effort in ("minimal", "low"):
                return {"thinkingLevel": "low", "include_thoughts": True}
            if effort in ("low_medium", "medium"):
                return {"thinkingLevel": "medium", "include_thoughts": True}
            # auto, medium_high, high -> high
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 3 Pro: only low/high
        if is_gemini_3:
            if effort in ("disable", "off", "none", "minimal", "low", "low_medium"):
                return {"thinkingLevel": "low", "include_thoughts": True}
            # auto, medium, medium_high, high -> high
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 2.5: Integer thinkingBudget
        if effort in ("disable", "off", "none"):
            return {"thinkingBudget": 0, "include_thoughts": False}

        if effort == "auto":
            return {"thinkingBudget": -1, "include_thoughts": True}

        # Model-specific budgets
        if "gemini-2.5-flash" in model:
            budgets = {
                "minimal": 3072,
                "low": 6144,
                "low_medium": 9216,
                "medium": 12288,
                "medium_high": 18432,
                "high": 24576,
            }
        else:
            budgets = {
                "minimal": 4096,
                "low": 8192,
                "low_medium": 12288,
                "medium": 16384,
                "medium_high": 24576,
                "high": 32768,
            }

        return {"thinkingBudget": budgets[effort], "include_thoughts": True}

    def _convert_chunk_to_openai(
        self,
        chunk: Dict[str, Any],
        model_id: str,
        accumulator: Optional[Dict[str, Any]] = None,
    ):
        """
        Convert Gemini response chunk to OpenAI streaming format.

        Args:
            chunk: Gemini API response chunk
            model_id: Model name
            accumulator: Optional dict to accumulate data for post-processing (signatures, etc.)
        """
        response_data = chunk.get("response", chunk)
        candidates = response_data.get("candidates", [])
        if not candidates:
            return

        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts", [])
        is_gemini_3 = self._is_gemini_3(model_id)

        for part in parts:
            delta = {}

            has_func = "functionCall" in part
            has_text = "text" in part
            has_sig = bool(part.get("thoughtSignature"))
            is_thought = part.get("thought") is True or (
                isinstance(part.get("thought"), str)
                and str(part.get("thought")).lower() == "true"
            )

            # Skip standalone signature parts (no function, no meaningful text)
            if has_sig and not has_func and (not has_text or not part.get("text")):
                continue

            if has_func:
                function_call = part["functionCall"]
                function_name = function_call.get("name", "unknown")

                # Strip Gemini 3 prefix from tool name
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    function_name = self._strip_gemini3_prefix(function_name)

                # Use provided ID or generate unique one with nanosecond precision
                tool_call_id = (
                    function_call.get("id")
                    or f"call_{function_name}_{int(time.time() * 1_000_000_000)}"
                )

                # Get current tool index from accumulator (default 0) and increment
                current_tool_idx = accumulator.get("tool_idx", 0) if accumulator else 0

                # Optionally parse JSON strings in tool args
                # NOTE: This is very possibly redundant
                raw_args = function_call.get("args", {})
                if self._enable_json_string_parsing:
                    tool_args = recursively_parse_json_strings(raw_args)
                else:
                    tool_args = raw_args

                # Strip _confirm ONLY if it's the sole parameter
                # This ensures we only strip our injection, not legitimate user params
                if isinstance(tool_args, dict) and "_confirm" in tool_args:
                    if len(tool_args) == 1:
                        # _confirm is the only param - this was our injection
                        tool_args.pop("_confirm")

                tool_call = {
                    "index": current_tool_idx,
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": json_dumps_str(tool_args),
                    },
                }

                # Handle thoughtSignature for Gemini 3
                # Store signature for each tool call (needed for parallel tool calls)
                if is_gemini_3 and has_sig:
                    sig = part["thoughtSignature"]

                    if self._enable_signature_cache:
                        self._signature_cache.store(tool_call_id, sig)
                        lib_logger.debug(f"Stored signature for {tool_call_id}")

                    if self._preserve_signatures_in_client:
                        tool_call["thought_signature"] = sig

                delta["tool_calls"] = [tool_call]
                # Mark that we've sent tool calls and increment tool_idx
                if accumulator is not None:
                    accumulator["has_tool_calls"] = True
                    accumulator["tool_idx"] = current_tool_idx + 1

            elif has_text:
                # Use an explicit check for the 'thought' flag, as its type can be inconsistent
                if is_thought:
                    delta["reasoning_content"] = part["text"]
                else:
                    delta["content"] = part["text"]

            if not delta:
                continue

            # Mark that we have tool calls for accumulator tracking
            # finish_reason determination is handled by the client

            # Mark stream complete if we have usageMetadata
            is_final_chunk = "usageMetadata" in response_data
            if is_final_chunk and accumulator is not None:
                accumulator["is_complete"] = True

            # Build choice - don't include finish_reason, let client handle it
            choice = {"index": 0, "delta": delta}

            openai_chunk = {
                "choices": [choice],
                "model": model_id,
                "object": "chat.completion.chunk",
                "id": chunk.get("responseId", f"chatcmpl-geminicli-{time.time()}"),
                "created": int(time.time()),
            }

            if "usageMetadata" in response_data:
                usage = response_data["usageMetadata"]
                prompt_tokens = usage.get("promptTokenCount", 0)  # Input
                thoughts_tokens = usage.get(
                    "thoughtsTokenCount", 0
                )  # Output (thinking)
                candidate_tokens = usage.get(
                    "candidatesTokenCount", 0
                )  # Output (content)
                cached_tokens = usage.get("cachedContentTokenCount", 0)  # Input subset

                openai_chunk["usage"] = {
                    "prompt_tokens": prompt_tokens,  # Input only
                    "completion_tokens": candidate_tokens
                    + thoughts_tokens,  # All output
                    "total_tokens": usage.get("totalTokenCount", 0),
                }

                # Add input breakdown: cached tokens
                if cached_tokens > 0:
                    openai_chunk["usage"]["prompt_tokens_details"] = {
                        "cached_tokens": cached_tokens
                    }

                # Add output breakdown: reasoning tokens
                if thoughts_tokens > 0:
                    openai_chunk["usage"]["completion_tokens_details"] = {
                        "reasoning_tokens": thoughts_tokens
                    }

            yield openai_chunk

    def _gemini_cli_transform_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively transforms a JSON schema to be compatible with the Gemini CLI endpoint.
        - Converts `type: ["type", "null"]` to `type: "type", nullable: true`
        - Removes unsupported properties like `strict`.
        - Preserves `additionalProperties` for _enforce_strict_schema to handle.
        """
        return clean_gemini_schema(schema)

    def _transform_tool_schemas(
        self, tools: List[Dict[str, Any]], model: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Transforms a list of OpenAI-style tool schemas into the format required by the Gemini CLI API.
        This uses a custom schema transformer instead of litellm's generic one.

        For Gemini 3 models, also applies:
        - Namespace prefix to tool names
        - Parameter signature injection into descriptions
        - Strict schema enforcement (additionalProperties: false)
        """
        transformed_declarations = []
        is_gemini_3 = self._is_gemini_3(model)

        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                new_function = dict(tool["function"])

                # The Gemini CLI API does not support the 'strict' property.
                new_function.pop("strict", None)

                # Gemini CLI expects 'parametersJsonSchema' instead of 'parameters'
                if "parameters" in new_function:
                    # Inline $ref definitions first
                    schema = inline_schema_refs(new_function["parameters"])
                    schema = self._gemini_cli_transform_schema(schema)
                    # Workaround: Gemini fails to emit functionCall for tools
                    # with empty properties {}. Inject a required confirmation param.
                    # Using a required parameter forces the model to commit to
                    # the tool call rather than just thinking about it.
                    props = schema.get("properties", {})
                    if not props:
                        schema["properties"] = {
                            "_confirm": {
                                "type": "string",
                                "description": "Enter 'yes' to proceed",
                            }
                        }
                        schema["required"] = ["_confirm"]
                    new_function["parametersJsonSchema"] = schema
                    del new_function["parameters"]
                elif "parametersJsonSchema" not in new_function:
                    # Set default schema with required confirm param if neither exists
                    new_function["parametersJsonSchema"] = {
                        "type": "object",
                        "properties": {
                            "_confirm": {
                                "type": "string",
                                "description": "Enter 'yes' to proceed",
                            }
                        },
                        "required": ["_confirm"],
                    }

                # Gemini 3 specific transformations
                if is_gemini_3 and self._enable_gemini3_tool_fix:
                    # Add namespace prefix to tool names (and rename problematic tools)
                    name = new_function.get("name", "")
                    if name:
                        name = GEMINI3_TOOL_RENAMES.get(name, name)
                        new_function["name"] = f"{self._gemini3_tool_prefix}{name}"

                    # Enforce strict schema (additionalProperties: false)
                    if (
                        self._gemini3_enforce_strict_schema
                        and "parametersJsonSchema" in new_function
                    ):
                        new_function["parametersJsonSchema"] = (
                            self._enforce_strict_schema(
                                new_function["parametersJsonSchema"]
                            )
                        )

                    # Inject parameter signature into description
                    new_function = self._inject_signature_into_description(
                        new_function, self._gemini3_description_prompt
                    )

                transformed_declarations.append(new_function)

        return transformed_declarations

    def _inject_gemini3_system_instruction(
        self, request_payload: Dict[str, Any]
    ) -> None:
        """Inject Gemini 3 tool fix system instruction if tools are present."""
        if not request_payload.get("request", {}).get("tools"):
            return

        existing_system = request_payload.get("request", {}).get("systemInstruction")

        if existing_system:
            # Prepend to existing system instruction
            existing_parts = existing_system.get("parts", [])
            if existing_parts and existing_parts[0].get("text"):
                existing_parts[0]["text"] = (
                    self._gemini3_system_instruction
                    + "\n\n"
                    + existing_parts[0]["text"]
                )
            else:
                existing_parts.insert(0, {"text": self._gemini3_system_instruction})
        else:
            # Create new system instruction
            request_payload["request"]["systemInstruction"] = {
                "role": "user",
                "parts": [{"text": self._gemini3_system_instruction}],
            }
