# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Thinking cache mixin for AntigravityProvider."""

from __future__ import annotations

import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import orjson
from ...utils.json_utils import json_deep_copy
from .constants import (
    ENABLE_INTERLEAVED_THINKING,
    CLAUDE_INTERLEAVED_THINKING_HINT,
    CLAUDE_USER_INTERLEAVED_THINKING_REMINDER,
    ANTIGRAVITY_IDENTITY_OVERRIDE_INSTRUCTION_SHORT,
    _get_gemini3_signature_cache_file,
    _get_claude_thinking_cache_file,
    lib_logger,
)


class ThinkingCacheMixin:
    """Mixin providing thinking/signature caching methods."""

    def _generate_thinking_cache_key(
        self, text_content: str, tool_calls: List[Dict]
    ) -> Optional[str]:
        """
        Generate stable cache key from response content for Claude thinking preservation.

        Uses composite key:
        - Tool call IDs (most stable)
        - Text hash (for text-only responses)
        """
        key_parts = []

        if tool_calls:
            first_id = tool_calls[0].get("id", "")
            if first_id:
                key_parts.append(f"tool_{first_id.replace('call_', '')}")

        if text_content:
            text_hash = hashlib.md5(text_content[:200].encode()).hexdigest()[:16]
            key_parts.append(f"text_{text_hash}")

        return "thinking_" + "_".join(key_parts) if key_parts else None

    # =========================================================================
    # THINKING MODE SANITIZATION
    # =========================================================================

    def _analyze_conversation_state(
        self, messages: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze conversation state to detect tool use loops and thinking mode issues.

        Key insight: A "turn" can span multiple assistant messages in a tool-use loop.
        We need to find the TURN START (first assistant message after last real user message)
        and check if THAT message had thinking, not just the last assistant message.

        Returns:
            {
                "in_tool_loop": bool - True if we're in an incomplete tool use loop
                "turn_start_idx": int - Index of first model message in current turn
                "turn_has_thinking": bool - Whether the TURN started with thinking
                "last_model_idx": int - Index of last model message
                "last_model_has_thinking": bool - Whether last model msg has thinking
                "last_model_has_tool_calls": bool - Whether last model msg has tool calls
                "pending_tool_results": bool - Whether there are tool results after last model
                "thinking_block_indices": List[int] - Indices of messages with thinking/reasoning
            }

        NOTE: This now operates on Gemini-format messages (after transformation):
        - Role "model" instead of "assistant"
        - Role "user" for both user messages AND tool results (with functionResponse)
        - "parts" array with "thought": true for thinking
        - "parts" array with "functionCall" for tool calls
        - "parts" array with "functionResponse" for tool results
        """
        state = {
            "in_tool_loop": False,
            "turn_start_idx": -1,
            "turn_has_thinking": False,
            "last_assistant_idx": -1,  # Keep name for compatibility
            "last_assistant_has_thinking": False,
            "last_assistant_has_tool_calls": False,
            "pending_tool_results": False,
            "thinking_block_indices": [],
        }

        # First pass: Find the last "real" user message (not a tool result)
        # In Gemini format, tool results are "user" role with functionResponse parts
        last_real_user_idx = -1
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == "user":
                # Check if this is a real user message or a tool result container
                parts = msg.get("parts", [])
                is_tool_result_msg = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )

                if not is_tool_result_msg:
                    last_real_user_idx = i

        # Second pass: Analyze conversation and find turn boundaries
        for i, msg in enumerate(messages):
            role = msg.get("role")

            if role == "model":
                # Check for thinking/reasoning content (Gemini format)
                has_thinking = self._message_has_thinking(msg)

                # Check for tool calls (functionCall in parts)
                parts = msg.get("parts", [])
                has_tool_calls = any(
                    isinstance(p, dict) and "functionCall" in p for p in parts
                )

                # Track if this is the turn start
                if i > last_real_user_idx and state["turn_start_idx"] == -1:
                    state["turn_start_idx"] = i
                    state["turn_has_thinking"] = has_thinking

                state["last_assistant_idx"] = i
                state["last_assistant_has_tool_calls"] = has_tool_calls
                state["last_assistant_has_thinking"] = has_thinking

                if has_thinking:
                    state["thinking_block_indices"].append(i)

            elif role == "user":
                # Check if this is a tool result (functionResponse in parts)
                parts = msg.get("parts", [])
                is_tool_result = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )

                if is_tool_result and state["last_assistant_has_tool_calls"]:
                    state["pending_tool_results"] = True

        # We're in a tool loop if:
        # 1. There are pending tool results
        # 2. The conversation ends with tool results (last message is user with functionResponse)
        if state["pending_tool_results"] and messages:
            last_msg = messages[-1]
            if last_msg.get("role") == "user":
                parts = last_msg.get("parts", [])
                ends_with_tool_result = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )
                if ends_with_tool_result:
                    state["in_tool_loop"] = True

        return state

    def _message_has_thinking(self, msg: Dict[str, Any]) -> bool:
        """
        Check if a message contains thinking/reasoning content.

        Handles GEMINI format (after transformation):
        - "parts" array with items having "thought": true
        """
        parts = msg.get("parts", [])
        for part in parts:
            if isinstance(part, dict) and part.get("thought") is True:
                return True
        return False

    def _message_has_tool_calls(self, msg: Dict[str, Any]) -> bool:
        """Check if a message contains tool calls (Gemini format)."""
        parts = msg.get("parts", [])
        return any(isinstance(p, dict) and "functionCall" in p for p in parts)

    async def _sanitize_thinking_for_claude(
        self, messages: List[Dict[str, Any]], thinking_enabled: bool
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Sanitize thinking blocks in conversation history for Claude compatibility.

        For interleaved thinking:
        1. If thinking disabled: strip ALL thinking blocks
        2. If thinking enabled:
           a. Recover thinking from cache for ALL model messages in current turn
           b. If first model message has thinking after recovery: valid turn, continue
           c. If first model message has NO thinking: close loop with synthetic messages

        Per Claude docs:
        - "If thinking is enabled, the final assistant turn must start with a thinking block"
        - Tool use loops are part of a single assistant turn
        - You CANNOT toggle thinking mid-turn

        Returns:
            Tuple of (sanitized_messages, force_disable_thinking)
            - sanitized_messages: The cleaned message list
            - force_disable_thinking: If True, thinking must be disabled for this request
        """
        # gemini_contents is a local variable freshly created by _transform_messages,
        # not shared across coroutines — no need to deepcopy before in-place mutation
        state = self._analyze_conversation_state(messages)

        lib_logger.debug(
            f"[Thinking Sanitization] thinking_enabled={thinking_enabled}, "
            f"in_tool_loop={state['in_tool_loop']}, "
            f"turn_has_thinking={state['turn_has_thinking']}, "
            f"turn_start_idx={state['turn_start_idx']}"
        )

        if not thinking_enabled:
            # Thinking disabled - strip ALL thinking blocks
            return self._strip_all_thinking_blocks(messages), False

        # Thinking is enabled
        # Always try to recover thinking for ALL model messages in current turn
        if state["turn_start_idx"] >= 0:
            recovered = await self._recover_all_turn_thinking(
                messages, state["turn_start_idx"]
            )
            if recovered > 0:
                lib_logger.debug(
                    f"[Thinking Sanitization] Recovered {recovered} thinking blocks from cache"
                )
                # Re-analyze state after recovery
                state = self._analyze_conversation_state(messages)

        if state["in_tool_loop"]:
            # In tool loop - first model message MUST have thinking
            if state["turn_has_thinking"]:
                # Valid: first message has thinking, continue
                lib_logger.debug(
                    "[Thinking Sanitization] Tool loop with thinking at turn start - valid"
                )
                return messages, False
            else:
                # Invalid: first message has no thinking, close loop
                lib_logger.info(
                    "[Thinking Sanitization] Closing tool loop - turn has no thinking at start"
                )
                return self._close_tool_loop_for_thinking(messages), False
        else:
            # Not in tool loop - just return messages as-is
            return messages, False

    def _remove_empty_messages(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove empty messages from conversation history.

        A message is considered empty if it has no parts, or all parts are:
        - Empty/whitespace-only text
        - No thinking blocks
        - No functionCall
        - No functionResponse

        This cleans up after compaction or stripping operations that may leave
        hollow message structures.
        """
        cleaned = []
        for msg in messages:
            parts = msg.get("parts", [])

            if not parts:
                # No parts at all - skip
                lib_logger.debug(
                    f"[Cleanup] Removing message with no parts: role={msg.get('role')}"
                )
                continue

            has_content = False
            for part in parts:
                if isinstance(part, dict):
                    # Check for non-empty text (empty string or whitespace-only is invalid)
                    if "text" in part and part["text"].strip():
                        has_content = True
                        break
                    # Check for thinking
                    if part.get("thought") is True:
                        has_content = True
                        break
                    # Check for function call
                    if "functionCall" in part:
                        has_content = True
                        break
                    # Check for function response
                    if "functionResponse" in part:
                        has_content = True
                        break

            if has_content:
                cleaned.append(msg)
            else:
                lib_logger.debug(
                    f"[Cleanup] Removing empty message: role={msg.get('role')}, "
                    f"parts_count={len(parts)}"
                )

        return cleaned

    def _inject_interleaved_thinking_reminder(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Inject interleaved thinking reminder into the last real user message.

        Appends an additional text part to the last user message that contains
        actual text (not just functionResponse). This is the same anchor message
        used for tool loop detection - the start of the current turn.

        If no real user message exists, no injection occurs.
        """
        # Find last real user message (same logic as _analyze_conversation_state)
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.get("role") == "user":
                parts = msg.get("parts", [])

                # Check if this is a real user message (has text, not just functionResponse)
                has_text = any(
                    isinstance(p, dict) and "text" in p and p.get("text", "").strip()
                    for p in parts
                )
                has_function_response = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )

                if has_text and not has_function_response:
                    # This is the last real user message - append reminder
                    messages[i]["parts"].append(
                        {"text": CLAUDE_USER_INTERLEAVED_THINKING_REMINDER}
                    )
                    lib_logger.debug(
                        f"[Interleaved Thinking] Injected reminder to user message at index {i}"
                    )
                    return messages

        # No real user message found - no injection
        lib_logger.debug(
            "[Interleaved Thinking] No real user message found for reminder injection"
        )
        return messages

    def _strip_all_thinking_blocks(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Remove all thinking/reasoning content from messages.

        Handles GEMINI format (after transformation):
        - Role "model" instead of "assistant"
        - "parts" array with "thought": true for thinking
        """
        for msg in messages:
            if msg.get("role") == "model":
                parts = msg.get("parts", [])
                if parts:
                    # Filter out thinking parts (those with "thought": true)
                    filtered = [
                        p
                        for p in parts
                        if not (isinstance(p, dict) and p.get("thought") is True)
                    ]

                    # Check if there are still functionCalls remaining
                    has_function_calls = any(
                        isinstance(p, dict) and "functionCall" in p for p in filtered
                    )

                    if not filtered:
                        # All parts were thinking - need placeholder for valid structure
                        if not has_function_calls:
                            msg["parts"] = [{"text": ""}]
                        else:
                            msg["parts"] = []  # Will be invalid, but shouldn't happen
                    else:
                        msg["parts"] = filtered
        return messages

    def _strip_old_turn_thinking(
        self, messages: List[Dict[str, Any]], last_model_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Strip thinking from old turns but preserve for the last model turn.

        Per Claude docs: "thinking blocks from previous turns are removed from context"
        This mimics the API behavior and prevents issues.

        Handles GEMINI format: role "model", "parts" with "thought": true
        """
        for i, msg in enumerate(messages):
            if msg.get("role") == "model" and i < last_model_idx:
                # Old turn - strip thinking parts
                parts = msg.get("parts", [])
                if parts:
                    filtered = [
                        p
                        for p in parts
                        if not (isinstance(p, dict) and p.get("thought") is True)
                    ]

                    has_function_calls = any(
                        isinstance(p, dict) and "functionCall" in p for p in filtered
                    )

                    if not filtered:
                        msg["parts"] = [{"text": ""}] if not has_function_calls else []
                    else:
                        msg["parts"] = filtered
        return messages

    def _preserve_current_turn_thinking(
        self, messages: List[Dict[str, Any]], last_model_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Preserve thinking only for the current (last) model turn.
        Strip from all previous turns.
        """
        # Same as strip_old_turn_thinking - we keep the last turn intact
        return self._strip_old_turn_thinking(messages, last_model_idx)

    def _preserve_turn_start_thinking(
        self, messages: List[Dict[str, Any]], turn_start_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Preserve thinking at the turn start message.

        In multi-message tool loops, the thinking block is at the FIRST model
        message of the turn (turn_start_idx), not the last one. We need to preserve
        thinking from the turn start, and strip it from all older turns.

        Handles GEMINI format: role "model", "parts" with "thought": true
        """
        for i, msg in enumerate(messages):
            if msg.get("role") == "model" and i < turn_start_idx:
                # Old turn - strip thinking parts
                parts = msg.get("parts", [])
                if parts:
                    filtered = [
                        p
                        for p in parts
                        if not (isinstance(p, dict) and p.get("thought") is True)
                    ]

                    has_function_calls = any(
                        isinstance(p, dict) and "functionCall" in p for p in filtered
                    )

                    if not filtered:
                        msg["parts"] = [{"text": ""}] if not has_function_calls else []
                    else:
                        msg["parts"] = filtered
        return messages

    def _looks_like_compacted_thinking_turn(self, msg: Dict[str, Any]) -> bool:
        """
        Detect if a message looks like it was compacted from a thinking-enabled turn.

        Heuristics (GEMINI format):
        1. Has functionCall parts (typical thinking flow produces tool calls)
        2. No thinking parts (thought: true)
        3. No text content before functionCall (thinking responses usually have text)

        This is imperfect but helps catch common compaction scenarios.
        """
        parts = msg.get("parts", [])
        if not parts:
            return False

        has_function_call = any(
            isinstance(p, dict) and "functionCall" in p for p in parts
        )

        if not has_function_call:
            return False

        # Check for text content (not thinking)
        has_text = any(
            isinstance(p, dict)
            and "text" in p
            and p.get("text", "").strip()
            and not p.get("thought")  # Exclude thinking text
            for p in parts
        )

        # If we have functionCall but no non-thinking text, likely compacted
        if not has_text:
            return True

        return False

    async def _try_recover_thinking_from_cache(
        self, messages: List[Dict[str, Any]], turn_start_idx: int
    ) -> bool:
        """
        Try to recover thinking content from cache for a compacted turn.

        Handles GEMINI format: extracts functionCall for cache key lookup,
        injects thinking as a part with thought: true.

        Returns True if thinking was successfully recovered and injected, False otherwise.
        """
        if turn_start_idx < 0 or turn_start_idx >= len(messages):
            return False

        msg = messages[turn_start_idx]
        parts = msg.get("parts", [])

        # Extract text content and build tool_calls structure for cache key lookup
        text_content = ""
        tool_calls = []

        for part in parts:
            if isinstance(part, dict):
                if "text" in part and not part.get("thought"):
                    text_content = part["text"]
                elif "functionCall" in part:
                    fc = part["functionCall"]
                    # Convert to OpenAI tool_calls format for cache key compatibility
                    tool_calls.append(
                        {
                            "id": fc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": fc.get("name", ""),
                                "arguments": orjson.dumps(fc.get("args", {})).decode(),
                            },
                        }
                    )

        # Generate cache key and try to retrieve
        cache_key = self._generate_thinking_cache_key(text_content, tool_calls)
        if not cache_key:
            return False

        cached_json = await self._thinking_cache.retrieve_async(cache_key)
        if not cached_json:
            lib_logger.debug(
                f"[Thinking Sanitization] No cached thinking found for key: {cache_key}"
            )
            return False

        try:
            thinking_data = orjson.loads(cached_json)
            thinking_text = thinking_data.get("thinking_text", "")
            signature = thinking_data.get("thought_signature", "")

            if not thinking_text or not signature:
                lib_logger.debug(
                    "[Thinking Sanitization] Cached thinking missing text or signature"
                )
                return False

            # Inject the recovered thinking part at the beginning (Gemini format)
            thinking_part = {
                "text": thinking_text,
                "thought": True,
                "thoughtSignature": signature,
            }

            msg["parts"] = [thinking_part] + parts

            lib_logger.debug(
                f"[Thinking Sanitization] Recovered thinking from cache: {len(thinking_text)} chars"
            )
            return True

        except orjson.JSONDecodeError:
            lib_logger.warning(
                f"[Thinking Sanitization] Failed to parse cached thinking"
            )
            return False

    async def _recover_all_turn_thinking(
        self, messages: List[Dict[str, Any]], turn_start_idx: int
    ) -> int:
        """
        Recover thinking from cache for ALL model messages in current turn.

        For interleaved thinking, every model response in the turn may have thinking.
        Clients strip thinking content, so we restore from cache.
        Always overwrites existing thinking (safer - ensures signature is valid).

        Args:
            messages: Gemini-format messages
            turn_start_idx: Index of first model message in current turn

        Returns:
            Count of messages where thinking was recovered.
        """
        if turn_start_idx < 0:
            return 0

        recovered_count = 0

        for i in range(turn_start_idx, len(messages)):
            msg = messages[i]
            if msg.get("role") != "model":
                continue

            parts = msg.get("parts", [])

            # Extract text content and tool_calls for cache lookup
            # Also collect non-thinking parts to rebuild the message
            text_content = ""
            tool_calls = []
            non_thinking_parts = []

            for part in parts:
                if isinstance(part, dict):
                    if part.get("thought") is True:
                        # Skip existing thinking - we'll overwrite with cached version
                        continue
                    if "text" in part:
                        text_content = part["text"]
                        non_thinking_parts.append(part)
                    elif "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append(
                            {
                                "id": fc.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": fc.get("name", ""),
                                    "arguments": orjson.dumps(fc.get("args", {})).decode(),
                                },
                            }
                        )
                        non_thinking_parts.append(part)
                    else:
                        non_thinking_parts.append(part)

            # Try cache recovery
            cache_key = self._generate_thinking_cache_key(text_content, tool_calls)
            if not cache_key:
                continue

            cached_json = await self._thinking_cache.retrieve_async(cache_key)
            if not cached_json:
                continue

            try:
                thinking_data = orjson.loads(cached_json)
                thinking_text = thinking_data.get("thinking_text", "")
                signature = thinking_data.get("thought_signature", "")

                if thinking_text and signature:
                    # Inject recovered thinking at beginning
                    thinking_part = {
                        "text": thinking_text,
                        "thought": True,
                        "thoughtSignature": signature,
                    }
                    msg["parts"] = [thinking_part] + non_thinking_parts
                    recovered_count += 1
                    lib_logger.debug(
                        f"[Thinking Recovery] Recovered thinking for msg {i}: "
                        f"{len(thinking_text)} chars"
                    )
            except orjson.JSONDecodeError:
                pass

        return recovered_count

    def _close_tool_loop_for_thinking(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Close an incomplete tool loop by injecting synthetic messages to start a new turn.

        This is used when:
        - We're in a tool loop (conversation ends with functionResponse)
        - The tool call was made WITHOUT thinking (e.g., by Gemini, non-thinking Claude, or compaction stripped it)
        - We NOW want to enable thinking

        Per Claude docs on toggling thinking modes:
        - "If thinking is enabled, the final assistant turn must start with a thinking block"
        - "To toggle thinking, you must complete the assistant turn first"
        - A non-tool-result user message ends the turn and allows a fresh start

        Solution (GEMINI format):
        1. Add synthetic MODEL message to complete the non-thinking turn
        2. Add synthetic USER message to start a NEW turn
        3. Claude will generate thinking for its response to the new turn

        The synthetic messages are minimal and unobtrusive - they just satisfy the
        turn structure requirements without influencing model behavior.
        """
        # Strip any old thinking first
        messages = self._strip_all_thinking_blocks(messages)

        # Count tool results from the end of the conversation (Gemini format)
        tool_result_count = 0
        for msg in reversed(messages):
            if msg.get("role") == "user":
                parts = msg.get("parts", [])
                has_function_response = any(
                    isinstance(p, dict) and "functionResponse" in p for p in parts
                )
                if has_function_response:
                    tool_result_count += len(
                        [
                            p
                            for p in parts
                            if isinstance(p, dict) and "functionResponse" in p
                        ]
                    )
                else:
                    break  # Real user message, stop counting
            elif msg.get("role") == "model":
                break  # Stop at the model that made the tool calls

        # Safety check: if no tool results found, this shouldn't have been called
        # But handle gracefully with a generic message
        if tool_result_count == 0:
            lib_logger.warning(
                "[Thinking Sanitization] _close_tool_loop_for_thinking called but no tool results found. "
                "This may indicate malformed conversation history."
            )
            synthetic_model_content = "[Processing previous context.]"
        elif tool_result_count == 1:
            synthetic_model_content = "[Tool execution completed.]"
        else:
            synthetic_model_content = (
                f"[{tool_result_count} tool executions completed.]"
            )

        # Step 1: Inject synthetic MODEL message to complete the non-thinking turn (Gemini format)
        synthetic_model = {
            "role": "model",
            "parts": [{"text": synthetic_model_content}],
        }
        messages.append(synthetic_model)

        # Step 2: Inject synthetic USER message to start a NEW turn (Gemini format)
        # This allows Claude to generate thinking for its response
        # The message is minimal and unobtrusive - just triggers a new turn
        synthetic_user = {
            "role": "user",
            "parts": [{"text": "[Continue]"}],
        }
        messages.append(synthetic_user)

        lib_logger.info(
            f"[Thinking Sanitization] Closed tool loop with synthetic messages. "
            f"Model: '{synthetic_model_content}', User: '[Continue]'. "
            f"Claude will now start a fresh turn with thinking enabled."
        )

        return messages

    # =========================================================================
    # REASONING CONFIGURATION
    # =========================================================================

    def _get_thinking_config(
        self,
        reasoning_effort: Optional[str],
        model: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Map reasoning_effort to thinking configuration.

        - Gemini 2.5 & Claude: thinkingBudget (integer tokens)
        - Gemini 3 Pro: thinkingLevel (string: "low"/"high")
        - Gemini 3 Flash: thinkingLevel (string: "minimal"/"low"/"medium"/"high")
        """
        internal = self._alias_to_internal(model)
        is_gemini_25 = "gemini-2.5" in model
        is_gemini_3 = internal.startswith("gemini-3-")
        is_gemini_3_flash = "gemini-3-flash" in model or "gemini-3-flash" in internal
        is_claude = self._is_claude(model)

        if not (is_gemini_25 or is_gemini_3 or is_claude):
            return None

        # Normalize and validate upfront
        if reasoning_effort is None:
            effort = "auto"
        elif isinstance(reasoning_effort, str):
            effort = reasoning_effort.strip().lower() or "auto"
        else:
            lib_logger.warning(
                f"[Antigravity] Invalid reasoning_effort type: {type(reasoning_effort).__name__}, using auto"
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
                f"[Antigravity] Unknown reasoning_effort: '{reasoning_effort}', using auto"
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
            # auto, medium_high, high → high
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 3 Pro: only low/high
        if is_gemini_3:
            if effort in ("disable", "off", "none", "minimal", "low", "low_medium"):
                return {"thinkingLevel": "low", "include_thoughts": True}
            # auto, medium, medium_high, high → high
            return {"thinkingLevel": "high", "include_thoughts": True}

        # Gemini 2.5 & Claude: Integer thinkingBudget
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
            if is_claude:
                budgets["high"] = 31999  # Claude max budget

        return {"thinkingBudget": budgets[effort], "include_thoughts": True}


    def _cache_thinking(
        self, reasoning: str, signature: str, text: str, tool_calls: List[Dict]
    ) -> None:
        """Cache Claude thinking content."""
        cache_key = self._generate_thinking_cache_key(text, tool_calls)
        if not cache_key:
            return

        data = {
            "thinking_text": reasoning,
            "thought_signature": signature,
            "text_preview": text[:100] if text else "",
            "tool_ids": [tc.get("id", "") for tc in tool_calls],
            "timestamp": time.time(),
        }

        self._thinking_cache.store(cache_key, orjson.dumps(data).decode())
        lib_logger.debug(f"Cached thinking: {cache_key[:50]}...")

    # =========================================================================
    # PROVIDER INTERFACE IMPLEMENTATION
    # =========================================================================
