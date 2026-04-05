# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Message Transformer for Gemini API format.

Extracts common message transformation logic from providers that convert
OpenAI-format messages to Gemini-format messages.

Handles:
- System instruction extraction
- Tool call ID to name mapping
- User/assistant/tool message conversion
- Pending tool parts consolidation
- Gemini 3 thoughtSignature preservation
"""

from __future__ import annotations

import json
import logging
import orjson
from typing import Any, Callable, Dict, List, Optional, Tuple

lib_logger = logging.getLogger("rotator_library")


def transform_messages_for_gemini(
    messages: List[Dict[str, Any]],
    model: str = "",
    is_gemini_3: bool = False,
    gemini3_tool_prefix: str = "",
    enable_gemini3_tool_fix: bool = True,
    enable_signature_cache: bool = True,
    signature_cache_retrieve: Optional[Callable[[str], Optional[str]]] = None,
    gemini3_tool_renames: Optional[Dict[str, str]] = None,
    transform_user_content: Optional[Callable[[Any], List[Dict[str, Any]]]] = None,
    transform_assistant_msg: Optional[Callable[[Dict[str, Any], str], List[Dict[str, Any]]]] = None,
    transform_tool_msg: Optional[Callable[[Dict[str, Any], str], List[Dict[str, Any]]]] = None,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Transform OpenAI-format messages to Gemini format.

    This is the core transformation function used by both gemini_cli_provider
    and antigravity_provider. It handles the common logic for message conversion.

    Args:
        messages: List of OpenAI-format messages to transform.
        model: Model name for context-specific transformations.
        is_gemini_3: Whether the target model is Gemini 3.
        gemini3_tool_prefix: Prefix to add to tool names for Gemini 3.
        enable_gemini3_tool_fix: Whether to apply Gemini 3 tool fixes.
        enable_signature_cache: Whether to use signature caching.
        signature_cache_retrieve: Function to retrieve cached signatures by ID.
        gemini3_tool_renames: Mapping of tool names to rename for Gemini 3.
        transform_user_content: Optional custom user content transformer.
        transform_assistant_msg: Optional custom assistant message transformer.
        transform_tool_msg: Optional custom tool message transformer.

    Returns:
        Tuple of (system_instruction, gemini_contents).
        system_instruction is a dict with role and parts, or None.
        gemini_contents is a list of message dicts in Gemini format.
    """
    # Don't mutate original messages
    messages = orjson.loads(orjson.dumps(messages))

    system_instruction = None
    gemini_contents = []

    # Separate system prompt from other messages
    # Handle multiple consecutive system messages (antigravity style)
    system_parts = []
    while messages and messages[0].get("role") == "system":
        system_content = messages.pop(0).get("content", "")
        if system_content:
            # Simple string content - convert to parts
            if isinstance(system_content, str):
                system_parts.append({"text": system_content})
            elif isinstance(system_content, list):
                # Multi-part content
                for item in system_content:
                    if item.get("type") == "text":
                        text = item.get("text", "")
                        if text:
                            system_parts.append({"text": text})

    if system_parts:
        system_instruction = {"role": "user", "parts": system_parts}

    # Build tool_call_id -> name mapping
    tool_call_id_to_name: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                if tool_call.get("type") == "function":
                    tool_call_id_to_name[tool_call["id"]] = tool_call["function"]["name"]

    # Process messages and consolidate consecutive tool responses
    # Per Gemini docs: parallel function responses must be in a single user message,
    # not interleaved as separate messages
    pending_tool_parts: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        parts: List[Dict[str, Any]] = []
        gemini_role = "model" if role == "assistant" else "user"

        # If we have pending tool parts and hit a non-tool message, flush them first
        if pending_tool_parts and role != "tool":
            gemini_contents.append({"role": "user", "parts": pending_tool_parts})
            pending_tool_parts = []

        if role == "user":
            if transform_user_content:
                parts = transform_user_content(content)
            else:
                parts = _transform_user_content_default(content)

        elif role == "assistant":
            if transform_assistant_msg:
                parts = transform_assistant_msg(msg, model)
            else:
                parts = _transform_assistant_message_default(
                    msg=msg,
                    model=model,
                    is_gemini_3=is_gemini_3,
                    gemini3_tool_prefix=gemini3_tool_prefix,
                    enable_gemini3_tool_fix=enable_gemini3_tool_fix,
                    enable_signature_cache=enable_signature_cache,
                    signature_cache_retrieve=signature_cache_retrieve,
                    gemini3_tool_renames=gemini3_tool_renames,
                )

        elif role == "tool":
            tool_parts = _transform_tool_message(
                msg=msg,
                tool_call_id_to_name=tool_call_id_to_name,
                is_gemini_3=is_gemini_3,
                gemini3_tool_prefix=gemini3_tool_prefix,
                enable_gemini3_tool_fix=enable_gemini3_tool_fix,
                gemini3_tool_renames=gemini3_tool_renames,
            )
            pending_tool_parts.extend(tool_parts)
            continue

        if parts:
            gemini_contents.append({"role": gemini_role, "parts": parts})

    # Flush any remaining tool parts at end of messages
    if pending_tool_parts:
        gemini_contents.append({"role": "user", "parts": pending_tool_parts})

    # Ensure conversation starts with user role
    if not gemini_contents or gemini_contents[0]["role"] != "user":
        gemini_contents.insert(0, {"role": "user", "parts": [{"text": ""}]})

    return system_instruction, gemini_contents


def _transform_user_content_default(content: Any) -> List[Dict[str, Any]]:
    """Transform user message content to Gemini parts (default implementation)."""
    parts: List[Dict[str, Any]] = []

    if isinstance(content, str):
        if content:
            parts.append({"text": content})
    elif isinstance(content, list):
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                if text:
                    parts.append({"text": text})
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {}).get("url", "")
                if image_url.startswith("data:"):
                    try:
                        header, data = image_url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        parts.append({
                            "inlineData": {
                                "mimeType": mime_type,
                                "data": data,
                            }
                        })
                    except Exception as e:
                        lib_logger.warning(f"Failed to parse image data URL: {e}")
                else:
                    lib_logger.warning(
                        f"Non-data-URL images not supported: {image_url[:50]}..."
                    )

    return parts


def _transform_assistant_message_default(
    msg: Dict[str, Any],
    model: str,
    is_gemini_3: bool,
    gemini3_tool_prefix: str,
    enable_gemini3_tool_fix: bool,
    enable_signature_cache: bool,
    signature_cache_retrieve: Optional[Callable[[str], Optional[str]]],
    gemini3_tool_renames: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Transform assistant message including tool calls (default implementation)."""
    parts: List[Dict[str, Any]] = []
    content = msg.get("content")

    if isinstance(content, str) and content:
        parts.append({"text": content})

    tool_calls = msg.get("tool_calls", [])
    if tool_calls:
        # Track if we've seen the first function call in this message
        # Per Gemini docs: Only the FIRST parallel function call gets a signature
        first_func_in_msg = True

        for tool_call in tool_calls:
            if tool_call.get("type") != "function":
                continue

            try:
                args_dict = orjson.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError, orjson.JSONDecodeError):
                args_dict = {}

            tool_id = tool_call.get("id", "")
            func_name = tool_call["function"]["name"]

            # Apply Gemini 3 namespace prefix
            if is_gemini_3 and enable_gemini3_tool_fix:
                if gemini3_tool_renames:
                    func_name = gemini3_tool_renames.get(func_name, func_name)
                if gemini3_tool_prefix:
                    func_name = f"{gemini3_tool_prefix}{func_name}"

            func_part: Dict[str, Any] = {
                "functionCall": {
                    "name": func_name,
                    "args": args_dict,
                    "id": tool_id,
                }
            }

            # Add thoughtSignature for Gemini 3
            if is_gemini_3:
                sig = tool_call.get("thought_signature")
                if not sig and tool_id and enable_signature_cache and signature_cache_retrieve:
                    sig = signature_cache_retrieve(tool_id)

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

    return parts


def _transform_tool_message(
    msg: Dict[str, Any],
    tool_call_id_to_name: Dict[str, str],
    is_gemini_3: bool,
    gemini3_tool_prefix: str,
    enable_gemini3_tool_fix: bool,
    gemini3_tool_renames: Optional[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Transform tool response message to Gemini functionResponse format."""
    tool_call_id = msg.get("tool_call_id", "")
    function_name = tool_call_id_to_name.get(tool_call_id, "unknown_function")
    content = msg.get("content", "{}")

    # Log warning if tool_call_id not found in mapping
    if tool_call_id not in tool_call_id_to_name:
        lib_logger.warning(
            f"[ID Mismatch] Tool response has ID '{tool_call_id}' which was not found in "
            f"tool_id_to_name map. Available IDs: {list(tool_call_id_to_name.keys())}. "
            f"Using 'unknown_function' as fallback."
        )

    # Apply Gemini 3 namespace prefix
    if is_gemini_3 and enable_gemini3_tool_fix:
        if gemini3_tool_renames:
            function_name = gemini3_tool_renames.get(function_name, function_name)
        if gemini3_tool_prefix:
            function_name = f"{gemini3_tool_prefix}{function_name}"

    # Try to parse content as JSON, fall back to string
    try:
        parsed_content = orjson.loads(content) if isinstance(content, str) else content
    except (json.JSONDecodeError, TypeError, orjson.JSONDecodeError):
        parsed_content = content

    # Wrap the tool response in a 'result' object
    return [
        {
            "functionResponse": {
                "name": function_name,
                "response": {"result": parsed_content},
                "id": tool_call_id,
            }
        }
    ]


def build_tool_call_id_to_name_mapping(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Build a mapping from tool_call_id to function name.

    This is useful for providers that need the mapping separately from
    the full message transformation.

    Args:
        messages: List of OpenAI-format messages.

    Returns:
        Dict mapping tool_call_id to function name.
    """
    tool_call_id_to_name: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tool_call in msg["tool_calls"]:
                if tool_call.get("type") == "function":
                    tool_call_id_to_name[tool_call["id"]] = tool_call["function"]["name"]
    return tool_call_id_to_name


def ensure_user_first(gemini_contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure the Gemini conversation starts with a user message.

    Gemini requires the conversation to start with a user role.
    If the first message is not from user, insert an empty user message.

    Args:
        gemini_contents: List of Gemini-format messages.

    Returns:
        Modified list (may have empty user message prepended).
    """
    if not gemini_contents or gemini_contents[0]["role"] != "user":
        gemini_contents.insert(0, {"role": "user", "parts": [{"text": ""}]})
    return gemini_contents
