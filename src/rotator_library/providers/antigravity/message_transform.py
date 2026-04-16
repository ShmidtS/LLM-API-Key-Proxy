# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Message transform mixin for AntigravityProvider."""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import orjson
from ...config import env_bool
from ...utils.json_utils import json_deep_copy
from .constants import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    PREPEND_INSTRUCTION,
    INJECT_IDENTITY_OVERRIDE,
    USE_SHORT_ANTIGRAVITY_PROMPTS,
    ANTIGRAVITY_AGENT_SYSTEM_INSTRUCTION,
    ANTIGRAVITY_AGENT_SYSTEM_INSTRUCTION_SHORT,
    ANTIGRAVITY_IDENTITY_OVERRIDE_INSTRUCTION,
    ANTIGRAVITY_IDENTITY_OVERRIDE_INSTRUCTION_SHORT,
    _generate_request_id,
    _generate_stable_session_id,
    _clean_claude_schema,
    lib_logger,
)
from ..utilities.gemini_shared_utils import (
    map_finish_reason,
    inline_schema_refs,
    normalize_type_arrays,
    recursively_parse_json_strings,
    GEMINI3_TOOL_RENAMES,
    DEFAULT_SAFETY_SETTINGS,
)
from ..utilities.message_transformer import (
    _transform_user_content_default,
    _transform_tool_message as _base_transform_tool_message,
)


class MessageTransformMixin:
    """Mixin providing message format transformation methods."""

    async def _transform_messages(
        self, messages: List[Dict[str, Any]], model: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transform OpenAI messages to Gemini CLI format.

        Handles:
        - System instruction extraction
        - Multi-part content (text, images)
        - Tool calls and responses
        - Claude thinking injection from cache
        - Gemini 3 thoughtSignature preservation
        """
        # Deep copy required: _inject_interleaved_thinking_reminder mutates
        # messages[i]["parts"].append(...) which would corrupt caller data
        messages = json_deep_copy(messages)
        system_instruction = None
        gemini_contents = []

        # Extract system prompts (handle multiple consecutive system messages)
        system_parts = []
        while messages and messages[0].get("role") == "system":
            system_content = messages.pop(0).get("content", "")
            if system_content:
                new_parts = _transform_user_content_default(system_content)
                system_parts.extend(new_parts)

        if system_parts:
            system_instruction = {"role": "user", "parts": system_parts}

        # Build tool_call_id → name mapping
        tool_id_to_name = {}
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("type") == "function":
                        tc_id = tc["id"]
                        tc_name = tc["function"]["name"]
                        tool_id_to_name[tc_id] = tc_name

        # Convert each message, consolidating consecutive tool responses
        # Per Gemini docs: parallel function responses must be in a single user message
        pending_tool_parts = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            parts = []

            # Flush pending tool parts before non-tool message
            if pending_tool_parts and role != "tool":
                gemini_contents.append({"role": "user", "parts": pending_tool_parts})
                pending_tool_parts = []

            if role == "user":
                parts = _transform_user_content_default(content)
            elif role == "assistant":
                parts = await self._transform_assistant_message(msg, model, tool_id_to_name)
            elif role == "tool":
                is_gem3 = self._is_gemini_3(model)
                tool_parts = _base_transform_tool_message(
                    msg=msg,
                    tool_call_id_to_name=tool_id_to_name,
                    is_gemini_3=is_gem3,
                    gemini3_tool_prefix=self._gemini3_tool_prefix if is_gem3 else "",
                    enable_gemini3_tool_fix=self._enable_gemini3_tool_fix if is_gem3 else False,
                    gemini3_tool_renames=GEMINI3_TOOL_RENAMES if is_gem3 else None,
                )
                # Accumulate tool responses instead of adding individually
                pending_tool_parts.extend(tool_parts)
                continue

            if parts:
                gemini_role = "model" if role == "assistant" else "user"
                gemini_contents.append({"role": gemini_role, "parts": parts})

        # Flush any remaining tool parts
        if pending_tool_parts:
            gemini_contents.append({"role": "user", "parts": pending_tool_parts})

        return system_instruction, gemini_contents

    async def _transform_assistant_message(
        self, msg: Dict[str, Any], model: str, _tool_id_to_name: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Transform assistant message including tool calls and thinking injection."""
        parts = []
        content = msg.get("content")
        tool_calls = msg.get("tool_calls", [])
        reasoning_content = msg.get("reasoning_content")

        # Handle reasoning_content if present (from original Claude response with thinking)
        if reasoning_content and self._is_claude(model):
            # Add thinking part with cached signature
            thinking_part = {
                "text": reasoning_content,
                "thought": True,
            }
            # Try to get signature from cache
            cache_key = self._generate_thinking_cache_key(
                content if isinstance(content, str) else "", tool_calls
            )
            cached_sig = None
            if cache_key:
                cached_json = await self._thinking_cache.retrieve_async(cache_key)
                if cached_json:
                    try:
                        cached_data = orjson.loads(cached_json)
                        cached_sig = cached_data.get("thought_signature", "")
                    except orjson.JSONDecodeError:
                        pass

            if cached_sig:
                thinking_part["thoughtSignature"] = cached_sig
                parts.append(thinking_part)
                lib_logger.debug(
                    f"Added reasoning_content with cached signature ({len(reasoning_content)} chars)"
                )
            else:
                # No cached signature - skip the thinking block
                # This can happen if context was compressed and signature was lost
                lib_logger.warning(
                    "Skipping reasoning_content - no valid signature found. "
                    "This may cause issues if thinking is enabled."
                )
        elif (
            self._is_claude(model)
            and self._enable_signature_cache
            and not reasoning_content
        ):
            # Fallback: Try to inject cached thinking for Claude (original behavior)
            thinking_parts = await self._get_cached_thinking(content, tool_calls)
            parts.extend(thinking_parts)

        # Add regular content
        if isinstance(content, str) and content:
            parts.append({"text": content})

        # Add tool calls
        # Track if we've seen the first function call in this message
        # Per Gemini docs: Only the FIRST parallel function call gets a signature
        first_func_in_msg = True
        for tc in tool_calls:
            if tc.get("type") != "function":
                continue

            try:
                args = orjson.loads(tc["function"]["arguments"])
            except (orjson.JSONDecodeError, TypeError):
                args = {}

            tool_id = tc.get("id", "")
            func_name = tc["function"]["name"]

            # Add prefix for Gemini 3 (and rename problematic tools)
            if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                func_name = GEMINI3_TOOL_RENAMES.get(func_name, func_name)
                func_name = f"{self._gemini3_tool_prefix}{func_name}"

            func_part = {
                "functionCall": {"name": func_name, "args": args, "id": tool_id}
            }

            # Add thoughtSignature for Gemini 3
            # Per Gemini docs: Only the FIRST parallel function call gets a signature.
            # Subsequent parallel calls should NOT have a thoughtSignature field.
            if self._is_gemini_3(model):
                sig = tc.get("thought_signature")
                if not sig and tool_id and self._enable_signature_cache:
                    sig = await self._signature_cache.retrieve_async(tool_id)

                if sig:
                    func_part["thoughtSignature"] = sig
                elif first_func_in_msg:
                    # Only add bypass to the first function call if no sig available
                    func_part["thoughtSignature"] = "skip_thought_signature_validator"
                    lib_logger.debug(
                        f"Missing thoughtSignature for first func call {tool_id}, using bypass"
                    )
                # Subsequent parallel calls: no signature field at all

                first_func_in_msg = False

            parts.append(func_part)

        # Safety: ensure we return at least one part to maintain role alternation
        # This handles edge cases like assistant messages that had only thinking content
        # which got stripped, leaving the message otherwise empty
        if not parts:
            # Use a minimal text part - can happen after thinking is stripped
            parts.append({"text": ""})
            lib_logger.debug(
                "[Transform] Added empty text part to maintain role alternation"
            )

        return parts

    async def _get_cached_thinking(
        self, content: Any, tool_calls: List[Dict]
    ) -> List[Dict[str, Any]]:
        """Retrieve and format cached thinking content for Claude."""
        parts = []
        msg_text = content if isinstance(content, str) else ""
        cache_key = self._generate_thinking_cache_key(msg_text, tool_calls)

        if not cache_key:
            return parts

        cached_json = await self._thinking_cache.retrieve_async(cache_key)
        if not cached_json:
            return parts

        try:
            thinking_data = orjson.loads(cached_json)
            thinking_text = thinking_data.get("thinking_text", "")
            sig = thinking_data.get("thought_signature", "")

            if thinking_text:
                thinking_part = {
                    "text": thinking_text,
                    "thought": True,
                    "thoughtSignature": sig or "skip_thought_signature_validator",
                }
                parts.append(thinking_part)
                lib_logger.debug(f"Injected {len(thinking_text)} chars of thinking")
        except orjson.JSONDecodeError:
            lib_logger.warning(f"Failed to parse cached thinking: {cache_key}")

        return parts

    # =========================================================================
    # GEMINI 3 TOOL TRANSFORMATIONS
    # =========================================================================

    def _apply_gemini3_namespace(
        self, tools: List[Dict[str, Any]], copy_tools: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Add namespace prefix to tool names for Gemini 3.

        Also renames certain tools that conflict with Gemini's internal behavior
        (e.g., "batch" triggers MALFORMED_FUNCTION_CALL errors).

        Args:
            tools: List of tool definitions to modify.
            copy_tools: If True, create a deep copy before modifying.
                       Set False when called after other tool transformers
                       to avoid redundant copying.
        """
        if not tools:
            return tools

        modified = json_deep_copy(tools) if copy_tools else tools
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                name = func_decl.get("name", "")
                if name:
                    # Rename problematic tools first
                    name = GEMINI3_TOOL_RENAMES.get(name, name)
                    # Then add prefix
                    func_decl["name"] = f"{self._gemini3_tool_prefix}{name}"

        return modified

    def _enforce_strict_schema_on_tools(
        self, tools: List[Dict[str, Any]], copy_tools: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Apply strict schema enforcement to all tools in a list.

        Wraps the mixin's _enforce_strict_schema() method to operate on a list of tools,
        applying 'additionalProperties: false' to each tool's schema.
        Supports both 'parametersJsonSchema' and 'parameters' keys.

        Args:
            tools: List of tool definitions to modify.
            copy_tools: If True, create a deep copy before modifying.
                       Set False when called after other tool transformers
                       to avoid redundant copying.
        """
        if not tools:
            return tools

        modified = json_deep_copy(tools) if copy_tools else tools
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                # Support both parametersJsonSchema and parameters keys
                for schema_key in ("parametersJsonSchema", "parameters"):
                    if schema_key in func_decl:
                        # Delegate to mixin's singular _enforce_strict_schema method
                        func_decl[schema_key] = self._enforce_strict_schema(
                            func_decl[schema_key]
                        )
                        break  # Only process one schema key per function

        return modified

    def _inject_signature_into_descriptions(
        self, tools: List[Dict[str, Any]], description_prompt: Optional[str] = None, copy_tools: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Apply signature injection to all tools in a list.

        Wraps the mixin's _inject_signature_into_description() method to operate
        on a list of tools, injecting parameter signatures into each tool's description.

        Args:
            tools: List of tool definitions to modify.
            description_prompt: Optional prompt template for signature injection.
            copy_tools: If True, create a deep copy before modifying.
                       Set False when called after other tool transformers
                       to avoid redundant copying.
        """
        if not tools:
            return tools

        # Use provided prompt or default to Gemini 3 prompt
        prompt_template = description_prompt or self._gemini3_description_prompt

        modified = json_deep_copy(tools) if copy_tools else tools
        for tool in modified:
            for func_decl in tool.get("functionDeclarations", []):
                # Delegate to mixin's singular _inject_signature_into_description method
                self._inject_signature_into_description(func_decl, prompt_template)

        return modified


    def _build_tools_payload(
        self, tools: Optional[List[Dict[str, Any]]], model: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Build Gemini-format tools from OpenAI tools.

        For Gemini models, all tools are placed in a SINGLE functionDeclarations array.
        This matches the format expected by Gemini CLI and prevents MALFORMED_FUNCTION_CALL errors.

        Uses 'parameters' key for all models. The Antigravity API backend expects this format.
        Schema cleaning is applied based on target model (Claude vs Gemini).
        """
        if not tools:
            return None

        function_declarations = []

        # Always use 'parameters' key - Antigravity API expects this for all models
        # Previously used 'parametersJsonSchema' but this caused MALFORMED_FUNCTION_CALL
        # errors with Gemini 3 Pro models. Using 'parameters' works for all backends.
        schema_key = "parameters"

        for tool in tools:
            if tool.get("type") != "function":
                continue

            func = tool.get("function", {})
            params = func.get("parameters")

            func_decl = {
                "name": self._sanitize_tool_name(func.get("name", "")),
                "description": func.get("description", ""),
            }

            if params and isinstance(params, dict):
                schema = dict(params)
                schema.pop("strict", None)
                # Inline $ref definitions, then strip unsupported keywords
                schema = inline_schema_refs(schema)
                # For Gemini models, use for_gemini=True to:
                # - Preserve truthy additionalProperties (for freeform param objects)
                # - Strip false values (let _enforce_strict_schema add them)
                is_gemini = not self._is_claude(model)
                schema = _clean_claude_schema(schema, for_gemini=is_gemini)
                schema = normalize_type_arrays(schema)

                # Workaround: Antigravity/Gemini fails to emit functionCall
                # when tool has empty properties {}. Inject a dummy optional
                # parameter to ensure the tool call is emitted.
                # Using a required confirmation parameter forces the model to
                # commit to the tool call rather than just thinking about it.
                props = schema.get("properties", {})
                if not props:
                    schema["properties"] = {
                        "_confirm": {
                            "type": "string",
                            "description": "Enter 'yes' to proceed",
                        }
                    }
                    schema["required"] = ["_confirm"]

                func_decl[schema_key] = schema
            else:
                # No parameters provided - use default with required confirm param
                # to ensure the tool call is emitted properly
                func_decl[schema_key] = {
                    "type": "object",
                    "properties": {
                        "_confirm": {
                            "type": "string",
                            "description": "Enter 'yes' to proceed",
                        }
                    },
                    "required": ["_confirm"],
                }

            function_declarations.append(func_decl)

        if not function_declarations:
            return None

        # Return all tools in a SINGLE functionDeclarations array
        # This is the format Gemini CLI uses and prevents MALFORMED_FUNCTION_CALL errors
        return [{"functionDeclarations": function_declarations}]

    def _transform_to_antigravity_format(
        self,
        gemini_payload: Dict[str, Any],
        model: str,
        project_id: str,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[Union[str, float, int]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Transform Gemini CLI payload to complete Antigravity format.

        Args:
            gemini_payload: Request in Gemini CLI format
            model: Model name (public alias)
            max_tokens: Max output tokens (including thinking)
            reasoning_effort: Reasoning effort level (determines -thinking variant for Claude)
        """
        internal_model = self._alias_to_internal(model)

        # Map Claude models to their -thinking variant
        # claude-opus-4-5: ALWAYS use -thinking (non-thinking variant doesn't exist)
        # claude-sonnet-4-5: only use -thinking when reasoning_effort is provided
        if self._is_claude(internal_model) and not internal_model.endswith("-thinking"):
            if internal_model == "claude-opus-4-5":
                # Opus 4.5 ALWAYS requires -thinking variant
                internal_model = "claude-opus-4-5-thinking"
            elif internal_model == "claude-sonnet-4-5" and reasoning_effort:
                # Sonnet 4.5 uses -thinking only when reasoning_effort is provided
                internal_model = "claude-sonnet-4-5-thinking"

        # Map gemini-2.5-flash to -thinking variant when reasoning_effort is provided
        if internal_model == "gemini-2.5-flash" and reasoning_effort:
            internal_model = "gemini-2.5-flash-thinking"

        # Map gemini-3-pro-preview to -low/-high variant based on thinking config
        if model == "gemini-3-pro-preview" or internal_model == "gemini-3-pro-preview":
            # Check thinking config to determine variant
            thinking_config = gemini_payload.get("generationConfig", {}).get(
                "thinkingConfig", {}
            )
            thinking_level = thinking_config.get("thinkingLevel", "high")
            if thinking_level == "low":
                internal_model = "gemini-3-pro-low"
            else:
                internal_model = "gemini-3-pro-high"

        # Wrap in Antigravity envelope
        # Per CLIProxyAPI commit 67985d8: added requestType: "agent"
        antigravity_payload = {
            "project": project_id,  # Will be passed as parameter
            "userAgent": "antigravity",
            "requestType": "agent",  # Required for agent-style requests
            "requestId": _generate_request_id(),
            "model": internal_model,
            "request": json_deep_copy(gemini_payload),
        }

        # Add stable session ID based on first user message
        contents = antigravity_payload["request"].get("contents", [])
        antigravity_payload["request"]["sessionId"] = _generate_stable_session_id(
            contents
        )

        # Prepend Antigravity agent system instruction to existing system instruction
        # Sets request.systemInstruction.role = "user"
        # and sets parts.0.text to the agent identity/guidelines
        # We preserve any existing parts by shifting them (Antigravity = parts[0], existing = parts[1:])
        #
        # Controlled by environment variables:
        # - ANTIGRAVITY_PREPEND_INSTRUCTION: Skip prepending agent instruction entirely
        # - ANTIGRAVITY_PRESERVE_SYSTEM_INSTRUCTION_CASE: Keep original field casing
        request = antigravity_payload["request"]

        # Determine which field name to use (snake_case vs camelCase)
        has_snake_case = "system_instruction" in request
        has_camel_case = "systemInstruction" in request

        # Get existing system instruction (check both formats)
        if has_camel_case:
            existing_sys_inst = request.get("systemInstruction", {})
        elif has_snake_case:
            existing_sys_inst = request.get("system_instruction", {})
        else:
            existing_sys_inst = {}

        existing_parts = existing_sys_inst.get("parts", [])

        # Always normalize to camelCase (Antigravity API requirement)
        target_key = "systemInstruction"
        # Remove snake_case version if present (avoid duplicate fields)
        if has_snake_case:
            del request["system_instruction"]

        # Build new parts array
        if not PREPEND_INSTRUCTION:
            # Skip prepending agent instruction, just use existing parts
            new_parts = existing_parts if existing_parts else []
        else:
            # Choose prompt versions based on USE_SHORT_ANTIGRAVITY_PROMPTS setting
            # Short prompts significantly reduce context/token usage while maintaining API compatibility
            if USE_SHORT_ANTIGRAVITY_PROMPTS:
                agent_instruction = ANTIGRAVITY_AGENT_SYSTEM_INSTRUCTION_SHORT
                override_instruction = ANTIGRAVITY_IDENTITY_OVERRIDE_INSTRUCTION_SHORT
            else:
                agent_instruction = ANTIGRAVITY_AGENT_SYSTEM_INSTRUCTION
                override_instruction = ANTIGRAVITY_IDENTITY_OVERRIDE_INSTRUCTION

            # Antigravity instruction first (parts[0])
            new_parts = [{"text": agent_instruction}]

            # If override is enabled, inject it as parts[1] to neutralize Antigravity identity
            if INJECT_IDENTITY_OVERRIDE:
                new_parts.append({"text": override_instruction})

            # Then add existing parts (shifted to later positions)
            new_parts.extend(existing_parts)

        # Set the combined system instruction with role "user" (per Go implementation)
        if new_parts:
            request[target_key] = {
                "role": "user",
                "parts": new_parts,
            }

        # Add default safety settings to prevent content filtering
        # Only add if not already present in the payload
        if "safetySettings" not in antigravity_payload["request"]:
            # DEFAULT_SAFETY_SETTINGS is a list of simple flat dicts with
            # string keys/values — shallow list copy is sufficient
            antigravity_payload["request"]["safetySettings"] = [
                dict(item) for item in DEFAULT_SAFETY_SETTINGS
            ]

        # Handle max_tokens and thinking budget clamping/expansion
        # For Claude: expand max_tokens to accommodate thinking (default) or clamp thinking to max_tokens
        # Controlled by ANTIGRAVITY_CLAMP_THINKING_TO_OUTPUT env var (default: false = expand)
        gen_config = antigravity_payload["request"].get("generationConfig", {})
        is_claude = self._is_claude(model)

        # Get thinking budget from config (if present)
        thinking_config = gen_config.get("thinkingConfig", {})
        thinking_budget = thinking_config.get("thinkingBudget", -1)

        # Determine effective max_tokens
        if max_tokens is not None:
            effective_max = max_tokens
        elif is_claude:
            effective_max = DEFAULT_MAX_OUTPUT_TOKENS
        else:
            effective_max = None

        # Apply clamping or expansion if thinking budget exceeds max_tokens
        if (
            thinking_budget > 0
            and effective_max is not None
            and thinking_budget >= effective_max
        ):
            clamp_mode = env_bool("ANTIGRAVITY_CLAMP_THINKING_TO_OUTPUT", False)

            if clamp_mode:
                # CLAMP: Reduce thinking budget to fit within max_tokens
                clamped_budget = max(0, effective_max - 1)
                lib_logger.warning(
                    f"[Antigravity] thinkingBudget ({thinking_budget}) >= maxOutputTokens ({effective_max}). "
                    f"Clamping thinkingBudget to {clamped_budget}. "
                    "Set ANTIGRAVITY_CLAMP_THINKING_TO_OUTPUT=false to expand output instead."
                )
                thinking_config["thinkingBudget"] = clamped_budget
                gen_config["thinkingConfig"] = thinking_config
            else:
                # EXPAND (default): Increase max_tokens to accommodate thinking
                # Add buffer for actual response content (1024 tokens)
                expanded_max = thinking_budget + 1024
                lib_logger.warning(
                    f"[Antigravity] thinkingBudget ({thinking_budget}) >= maxOutputTokens ({effective_max}). "
                    f"Expanding maxOutputTokens to {expanded_max}. "
                    "Set ANTIGRAVITY_CLAMP_THINKING_TO_OUTPUT=true to clamp thinking instead."
                )
                effective_max = expanded_max

        # Set maxOutputTokens
        if effective_max is not None:
            gen_config["maxOutputTokens"] = effective_max

        antigravity_payload["request"]["generationConfig"] = gen_config

        # Set toolConfig based on tool_choice parameter
        tool_config_result = self._translate_tool_choice(tool_choice, model)
        if tool_config_result:
            antigravity_payload["request"]["toolConfig"] = tool_config_result
        else:
            # Default to AUTO if no tool_choice specified
            tool_config = antigravity_payload["request"].setdefault("toolConfig", {})
            func_config = tool_config.setdefault("functionCallingConfig", {})
            func_config["mode"] = "AUTO"

        # Handle Gemini 3 thinking logic
        if not internal_model.startswith("gemini-3-"):
            thinking_config = gen_config.get("thinkingConfig", {})
            if "thinkingLevel" in thinking_config:
                del thinking_config["thinkingLevel"]
                thinking_config["thinkingBudget"] = -1

        # Ensure first function call in each model message has a thoughtSignature for Gemini 3
        # Per Gemini docs: Only the FIRST parallel function call gets a signature
        if internal_model.startswith("gemini-3-"):
            for content in antigravity_payload["request"].get("contents", []):
                if content.get("role") == "model":
                    first_func_seen = False
                    for part in content.get("parts", []):
                        if "functionCall" in part:
                            if not first_func_seen:
                                # First function call in this message - needs a signature
                                if "thoughtSignature" not in part:
                                    part["thoughtSignature"] = (
                                        "skip_thought_signature_validator"
                                    )
                                first_func_seen = True
                            # Subsequent parallel calls: leave as-is (no signature)

        return antigravity_payload

    # =========================================================================
    # RESPONSE TRANSFORMATION
    # =========================================================================

    def _unwrap_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Extract Gemini response from Antigravity envelope."""
        return response.get("response", response)

    def _gemini_to_openai_chunk(
        self,
        chunk: Dict[str, Any],
        model: str,
        accumulator: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convert Gemini response chunk to OpenAI streaming format.

        Args:
            chunk: Gemini API response chunk
            model: Model name
            accumulator: Optional dict to accumulate data for post-processing
        """
        candidates = chunk.get("candidates", [])
        if not candidates:
            return {}

        candidate = candidates[0]
        content_parts = candidate.get("content", {}).get("parts", [])

        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls = []
        # Use accumulator's tool_idx if available, otherwise use local counter
        tool_idx = accumulator.get("tool_idx", 0) if accumulator else 0

        for part in content_parts:
            has_func = "functionCall" in part
            has_text = "text" in part
            has_sig = bool(part.get("thoughtSignature"))
            is_thought = (
                part.get("thought") is True
                or str(part.get("thought")).lower() == "true"
            )

            # Accumulate signature for Claude caching
            if has_sig and is_thought and accumulator is not None:
                accumulator["thought_signature"] = part["thoughtSignature"]

            # Skip standalone signature parts
            if has_sig and not has_func and (not has_text or not part.get("text")):
                continue

            if has_text:
                text = part["text"]
                if is_thought:
                    reasoning_parts.append(text)
                    if accumulator is not None:
                        accumulator["reasoning_content"] += text
                else:
                    text_parts.append(text)
                    if accumulator is not None:
                        accumulator["text_content"] += text

            if has_func:
                # Get tool_schemas from accumulator for schema-aware parsing
                tool_schemas = accumulator.get("tool_schemas") if accumulator else None
                tool_call = self._extract_tool_call(
                    part, model, tool_idx, accumulator, tool_schemas
                )

                # Store signature for each tool call (needed for parallel tool calls)
                if has_sig:
                    self._handle_tool_signature(tool_call, part["thoughtSignature"])

                tool_calls.append(tool_call)
                tool_idx += 1

        # Build delta
        text_content = "".join(text_parts)
        reasoning_content = "".join(reasoning_parts)
        delta = {}
        if text_content:
            delta["content"] = text_content
        if reasoning_content:
            delta["reasoning_content"] = reasoning_content
        if tool_calls:
            delta["tool_calls"] = tool_calls
            delta["role"] = "assistant"
            # Update tool_idx for next chunk
            if accumulator is not None:
                accumulator["tool_idx"] = tool_idx
        elif text_content or reasoning_content:
            delta["role"] = "assistant"

        # Build usage if present
        usage = self._build_usage(chunk.get("usageMetadata", {}))

        # Store last received usage for final chunk
        if usage and accumulator is not None:
            accumulator["last_usage"] = usage

        # Mark completion when we see usageMetadata
        if chunk.get("usageMetadata") and accumulator is not None:
            accumulator["is_complete"] = True

        # Build choice - just translate, don't include finish_reason
        # Client will handle finish_reason logic
        choice = {"index": 0, "delta": delta}

        response = {
            "id": chunk.get("responseId", f"chatcmpl-{uuid.uuid4().hex[:24]}"),
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
        }

        if usage:
            response["usage"] = usage

        return response

    def _build_tool_schema_map(
        self, tools: Optional[List[Dict[str, Any]]], model: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build a mapping of tool name -> parameter schema from tools payload.

        Used for schema-aware JSON string parsing to avoid corrupting
        string content that looks like JSON (e.g., write tool's content field).
        """
        if not tools:
            return {}

        schema_map = {}
        for tool in tools:
            for func_decl in tool.get("functionDeclarations", []):
                name = func_decl.get("name", "")
                # Strip gemini3 prefix if applicable
                if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
                    name = self._strip_gemini3_prefix(name)

                # Check both parametersJsonSchema (Gemini native) and parameters (Claude/OpenAI)
                schema = func_decl.get("parametersJsonSchema") or func_decl.get(
                    "parameters", {}
                )

                if name and schema:
                    schema_map[name] = schema

        return schema_map

    def _extract_tool_call(
        self,
        part: Dict[str, Any],
        model: str,
        index: int,
        accumulator: Optional[Dict[str, Any]] = None,
        tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Extract and format a tool call from a response part."""
        func_call = part["functionCall"]
        tool_id = func_call.get("id") or f"call_{uuid.uuid4().hex[:24]}"

        # lib_logger.debug(f"[ID Extraction] Extracting tool call: id={tool_id}, raw_id={func_call.get('id')}")

        tool_name = func_call.get("name", "")
        if self._is_gemini_3(model) and self._enable_gemini3_tool_fix:
            tool_name = self._strip_gemini3_prefix(tool_name)

        # Restore original tool name after stripping any prefixes
        tool_name = self._restore_tool_name(tool_name)

        raw_args = func_call.get("args", {})

        # Optionally parse JSON strings (handles escaped control chars, malformed JSON)
        # NOTE: Gemini 3 sometimes returns stringified arrays for array parameters
        # (e.g., batch, todowrite). Schema-aware parsing prevents corrupting string
        # content that looks like JSON (e.g., write tool's content field).
        if self._enable_json_string_parsing:
            # Get schema for this tool if available
            tool_schema = tool_schemas.get(tool_name) if tool_schemas else None
            parsed_args = recursively_parse_json_strings(
                raw_args, schema=tool_schema, parse_json_objects=True
            )
        else:
            parsed_args = raw_args

        # Strip the injected _confirm parameter ONLY if it's the sole parameter
        # This ensures we only strip our injection, not legitimate user params
        if isinstance(parsed_args, dict) and "_confirm" in parsed_args:
            if len(parsed_args) == 1:
                # _confirm is the only param - this was our injection
                parsed_args.pop("_confirm")

        tool_call = {
            "id": tool_id,
            "type": "function",
            "index": index,
            "function": {"name": tool_name, "arguments": orjson.dumps(parsed_args).decode()},
        }

        if accumulator is not None:
            accumulator["tool_calls"].append(tool_call)

        return tool_call

    def _handle_tool_signature(self, tool_call: Dict, signature: str) -> None:
        """Handle thoughtSignature for a tool call."""
        tool_id = tool_call["id"]

        if self._enable_signature_cache:
            self._signature_cache.store(tool_id, signature)
            lib_logger.debug(f"Stored signature for {tool_id}")

        if self._preserve_signatures_in_client:
            tool_call["thought_signature"] = signature

    def _map_finish_reason(
        self, gemini_reason: Optional[str], has_tool_calls: bool
    ) -> Optional[str]:
        """Map Gemini finish reason to OpenAI format."""
        return map_finish_reason(gemini_reason, has_tool_calls)

    def _build_usage(self, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build usage dict from Gemini usage metadata.

        Token accounting:
        - prompt_tokens: Input tokens sent to model (promptTokenCount)
        - completion_tokens: Output tokens received (candidatesTokenCount + thoughtsTokenCount)
        - prompt_tokens_details.cached_tokens: Cached input tokens subset
        - completion_tokens_details.reasoning_tokens: Thinking tokens subset of output
        """
        if not metadata:
            return None

        prompt = metadata.get("promptTokenCount", 0)  # Input tokens
        thoughts = metadata.get("thoughtsTokenCount", 0)  # Output (thinking)
        completion = metadata.get("candidatesTokenCount", 0)  # Output (content)
        cached = metadata.get("cachedContentTokenCount", 0)  # Input subset (cached)

        usage = {
            "prompt_tokens": prompt,  # Input only
            "completion_tokens": completion + thoughts,  # All output
            "total_tokens": metadata.get("totalTokenCount", 0),
        }

        # Input breakdown: cached tokens (subset of prompt_tokens)
        if cached > 0:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}

        # Output breakdown: reasoning/thinking tokens (subset of completion_tokens)
        if thoughts > 0:
            usage["completion_tokens_details"] = {"reasoning_tokens": thoughts}

        return usage
