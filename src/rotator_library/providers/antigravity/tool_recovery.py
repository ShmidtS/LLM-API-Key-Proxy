# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Tool recovery mixin for AntigravityProvider."""

import time
import uuid
from typing import Any, Dict, Optional, Tuple

import litellm
import orjson
from .constants import (
    lib_logger,
)


class ToolRecoveryMixin:
    """Mixin providing malformed function call recovery methods."""

    # MALFORMED FUNCTION CALL HANDLING
    # =========================================================================

    def _check_for_malformed_call(self, response: Dict[str, Any]) -> Optional[str]:
        """
        Check if response contains MALFORMED_FUNCTION_CALL.

        Returns finishMessage if malformed, None otherwise.
        """
        candidates = response.get("candidates", [])
        if not candidates:
            return None

        candidate = candidates[0]
        if candidate.get("finishReason") == "MALFORMED_FUNCTION_CALL":
            return candidate.get("finishMessage", "Unknown malformed call error")

        return None

    def _parse_malformed_call_message(
        self, finish_message: str, model: str
    ) -> Optional[Dict[str, Any]]:
        """
        Parse MALFORMED_FUNCTION_CALL finishMessage to extract tool info.

        Input format: "Malformed function call: call:namespace:tool_name{raw_args}"

        Returns:
            {"tool_name": "read", "prefixed_name": "gemini3_read",
             "raw_args": "{filePath: \"...\"}"}
            or None if unparseable
        """
        import re

        # Pattern: "Malformed function call: call:namespace:tool_name{args}"
        pattern = r"Malformed function call:\s*call:[^:]+:([^{]+)(\{.+\})$"
        match = re.match(pattern, finish_message, re.DOTALL)

        if not match:
            lib_logger.warning(
                f"[Antigravity] Could not parse MALFORMED_FUNCTION_CALL: {finish_message[:100]}"
            )
            return None

        prefixed_name = match.group(1).strip()  # "gemini3_read"
        raw_args = match.group(2)  # "{filePath: \"...\"}"

        # Strip our prefix to get original tool name
        tool_name = self._strip_gemini3_prefix(prefixed_name)

        return {
            "tool_name": tool_name,
            "prefixed_name": prefixed_name,
            "raw_args": raw_args,
        }

    def _analyze_json_error(self, raw_args: str) -> Dict[str, Any]:
        """
        Analyze malformed JSON to detect specific errors and attempt to fix it.

        Combines orjson.JSONDecodeError with heuristic pattern detection
        to provide actionable error information.

        Returns:
            {
                "json_error": str or None,  # Python's JSON error message
                "json_position": int or None,  # Position of error
                "issues": List[str],  # Human-readable issues detected
                "unquoted_keys": List[str],  # Specific unquoted key names
                "fixed_json": str or None,  # Corrected JSON if we could fix it
            }
        """
        import re as re_module

        result = {
            "json_error": None,
            "json_position": None,
            "issues": [],
            "unquoted_keys": [],
            "fixed_json": None,
        }

        # Option 1: Try orjson.loads to get exact error
        try:
            orjson.loads(raw_args)
            return result  # Valid JSON, no errors
        except orjson.JSONDecodeError as e:
            result["json_error"] = e.msg
            result["json_position"] = e.pos

        # Option 2: Heuristic pattern detection for specific issues
        # Detect unquoted keys: {word: or ,word:
        unquoted_key_pattern = r"[{,]\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:"
        unquoted_keys = re_module.findall(unquoted_key_pattern, raw_args)
        if unquoted_keys:
            result["unquoted_keys"] = unquoted_keys
            if len(unquoted_keys) == 1:
                result["issues"].append(f"Unquoted key: '{unquoted_keys[0]}'")
            else:
                result["issues"].append(
                    f"Unquoted keys: {', '.join(repr(k) for k in unquoted_keys)}"
                )

        # Detect single quotes
        if "'" in raw_args:
            result["issues"].append("Single quotes used instead of double quotes")

        # Detect trailing comma
        if re_module.search(r",\s*[}\]]", raw_args):
            result["issues"].append("Trailing comma before closing bracket")

        # Option 3: Try to fix the JSON and validate
        fixed = raw_args
        # Add quotes around unquoted keys
        fixed = re_module.sub(
            r"([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:",
            r'\1"\2":',
            fixed,
        )
        # Replace single quotes with double quotes
        fixed = fixed.replace("'", '"')
        # Remove trailing commas
        fixed = re_module.sub(r",(\s*[}\]])", r"\1", fixed)

        try:
            # Validate the fix works
            parsed = orjson.loads(fixed)
            # Use compact JSON format (matches what model should produce)
            result["fixed_json"] = orjson.dumps(parsed).decode()
        except orjson.JSONDecodeError:
            # First fix didn't work - try more aggressive cleanup
            pass

        # Option 4: If first attempt failed, try more aggressive fixes
        if result["fixed_json"] is None:
            try:
                # Normalize all whitespace (collapse newlines/multiple spaces)
                aggressive_fix = re_module.sub(r"\s+", " ", fixed)
                # Try parsing again
                parsed = orjson.loads(aggressive_fix)
                result["fixed_json"] = orjson.dumps(parsed).decode()
                lib_logger.debug(
                    "[Antigravity] Fixed malformed JSON with aggressive whitespace normalization"
                )
            except orjson.JSONDecodeError:
                pass

        # Option 5: If still failing, try fixing unquoted string values
        if result["fixed_json"] is None:
            try:
                # Some models produce unquoted string values like {key: value}
                # Try to quote values that look like unquoted strings
                # Match : followed by unquoted word (not a number, bool, null, or object/array)
                aggressive_fix = re_module.sub(
                    r":\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}\]])",
                    r': "\1"\2',
                    fixed,
                )
                parsed = orjson.loads(aggressive_fix)
                result["fixed_json"] = orjson.dumps(parsed).decode()
                lib_logger.debug(
                    "[Antigravity] Fixed malformed JSON by quoting unquoted string values"
                )
            except orjson.JSONDecodeError:
                # All fixes failed, leave as None
                pass

        return result

    def _build_malformed_call_retry_messages(
        self,
        parsed_call: Dict[str, Any],
        tool_schema: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build synthetic Gemini-format messages for malformed call retry.

        Returns: (assistant_message, user_message) in Gemini format
        """
        tool_name = parsed_call["tool_name"]
        raw_args = parsed_call["raw_args"]

        # Analyze the JSON error and try to fix it
        error_info = self._analyze_json_error(raw_args)

        # Assistant message: Show what it tried to do
        assistant_msg = {
            "role": "model",
            "parts": [{"text": f"I'll call the '{tool_name}' function."}],
        }

        # Build a concise error message
        if error_info["fixed_json"]:
            # We successfully fixed the JSON - show the corrected version
            error_text = f"""[FUNCTION CALL ERROR - INVALID JSON]

Your call to '{tool_name}' failed. All JSON keys must be double-quoted.

INVALID: {raw_args}

CORRECTED: {error_info["fixed_json"]}

Retry the function call now using the corrected JSON above. Output ONLY the tool call, no text."""
        else:
            # Couldn't auto-fix - give hints
            error_text = f"""[FUNCTION CALL ERROR - INVALID JSON]

Your call to '{tool_name}' failed due to malformed JSON.

You provided: {raw_args}

Fix: All JSON keys must be double-quoted. Example: {{"key":"value"}} not {{key:"value"}}

Analyze what you did wrong, correct it, and retry the function call. Output ONLY the tool call, no text."""

        # Add schema if available (strip $schema reference)
        if tool_schema:
            clean_schema = {k: v for k, v in tool_schema.items() if k != "$schema"}
            schema_str = orjson.dumps(clean_schema).decode()
            error_text += f"\n\nSchema: {schema_str}"

        user_msg = {"role": "user", "parts": [{"text": error_text}]}

        return assistant_msg, user_msg

    def _build_malformed_fallback_response(
        self, model: str, error_details: str
    ) -> litellm.ModelResponse:
        """
        Build error response when malformed call retries are exhausted.

        Uses finish_reason=None to indicate the response didn't complete normally,
        allowing clients to detect the incomplete state and potentially retry.
        """
        return litellm.ModelResponse(
            **{
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "[TOOL CALL ERROR] I attempted to call a function but "
                                "repeatedly produced malformed syntax. This may be a model issue.\n\n"
                                f"Last error: {error_details}\n\n"
                                "Please try rephrasing your request or try a different approach."
                            ),
                        },
                        "finish_reason": None,
                    }
                ],
            }
        )

    def _build_malformed_fallback_chunk(
        self,
        model: str,
        error_details: str,
        response_id: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> litellm.ModelResponse:
        """
        Build streaming chunk error response when malformed call retries are exhausted.

        Uses streaming format (delta instead of message) for consistency with streaming responses.
        Includes usage with completion_tokens > 0 so client.py recognizes it as a final chunk.
        """
        chunk_id = response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Ensure usage has completion_tokens > 0 for client to recognize as final chunk
        if not usage or usage.get("completion_tokens", 0) <= 0:
            prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,
                "total_tokens": prompt_tokens + 1,
            }

        return litellm.ModelResponse(
            **{
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": (
                                "[TOOL CALL ERROR] I attempted to call a function but "
                                "repeatedly produced malformed syntax. This may be a model issue.\n\n"
                                f"Last error: {error_details}\n\n"
                                "Please try rephrasing your request or try a different approach."
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": usage,
            }
        )

    def _build_fixed_tool_call_response(
        self,
        model: str,
        parsed_call: Dict[str, Any],
        error_info: Dict[str, Any],
    ) -> Optional[litellm.ModelResponse]:
        """
        Build a synthetic valid tool call response from auto-fixed malformed JSON.

        When Gemini 3 produces malformed JSON (e.g., unquoted keys), this method
        takes the auto-corrected JSON from _analyze_json_error() and builds a
        proper OpenAI-format tool call response.

        Returns None if the JSON couldn't be fixed.
        """
        fixed_json = error_info.get("fixed_json")
        if not fixed_json:
            return None

        # Validate the fixed JSON is actually valid
        try:
            orjson.loads(fixed_json)
        except orjson.JSONDecodeError:
            return None

        tool_name = parsed_call["tool_name"]
        tool_id = f"call_{uuid.uuid4().hex[:24]}"

        return litellm.ModelResponse(
            **{
                "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": fixed_json,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        )

    def _build_fixed_tool_call_chunk(
        self,
        model: str,
        parsed_call: Dict[str, Any],
        error_info: Dict[str, Any],
        response_id: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> Optional[litellm.ModelResponse]:
        """
        Build a streaming chunk with the auto-fixed tool call.

        Similar to _build_fixed_tool_call_response but uses streaming format:
        - object: "chat.completion.chunk" instead of "chat.completion"
        - delta: {...} instead of message: {...}
        - tool_calls items include "index" field

        Args:
            response_id: Optional original response ID to maintain stream continuity
            usage: Optional usage from previous chunks. Must include completion_tokens > 0
                   for client to recognize this as a final chunk.

        Returns None if the JSON couldn't be fixed.
        """
        fixed_json = error_info.get("fixed_json")
        if not fixed_json:
            return None

        # Validate the fixed JSON is actually valid
        try:
            orjson.loads(fixed_json)
        except orjson.JSONDecodeError:
            return None

        tool_name = parsed_call["tool_name"]
        tool_id = f"call_{uuid.uuid4().hex[:24]}"
        # Use original response ID if provided, otherwise generate new one
        chunk_id = response_id or f"chatcmpl-{uuid.uuid4().hex[:24]}"

        # Ensure usage has completion_tokens > 0 for client to recognize as final chunk
        # Client.py's _safe_streaming_wrapper uses completion_tokens > 0 to detect final chunks
        if not usage or usage.get("completion_tokens", 0) <= 0:
            prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": 1,  # Minimum to signal final chunk
                "total_tokens": prompt_tokens + 1,
            }

        return litellm.ModelResponse(
            **{
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": fixed_json,
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": usage,
            }
        )

    # =========================================================================
    # REQUEST TRANSFORMATION
    # =========================================================================
