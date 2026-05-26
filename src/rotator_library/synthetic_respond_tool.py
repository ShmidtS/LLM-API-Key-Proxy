# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Synthetic ``_proxy_respond`` tool adapter.

Small models (e.g. Gemini Flash, Haiku-class) often lack native tool-use
support.  Some clients (Claude Code, Cline, etc.) rely on a ``_proxy_respond``
tool to receive structured text replies.

This adapter:

1. **Injects** a synthetic ``_proxy_respond`` tool definition into outgoing
   requests when the target model matches ``SYNTHETIC_RESPOND_TOOL_MODELS``.
2. **Extracts** ``_proxy_respond`` tool calls from the provider response and
   rewrites them into plain text completions so downstream consumers see
   a normal assistant message.

The adapter is purely additive and opt-in.  When
``SYNTHETIC_RESPOND_TOOL_MODELS`` is empty (default), no code path is
affected.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from .config import SYNTHETIC_RESPOND_TOOL_MODELS

logger = logging.getLogger("rotator_library")

# Maximum size for respond tool arguments buffer (1 MB)
_MAX_RESPOND_ARGS = 1_048_576

# ---------------------------------------------------------------------------
# Synthetic tool definition (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

RESPOND_TOOL_DEFINITION: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "_proxy_respond",
        "description": (
            "Send a text response to the user. "
            "Use this tool to provide your final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "The text response to send to the user.",
                },
            },
            "required": ["response"],
        },
    },
}


# ---------------------------------------------------------------------------
# Model matching
# ---------------------------------------------------------------------------

def _compile_patterns(model_patterns: Iterable[str]) -> List[re.Pattern[str]]:
    """Convert glob-like model patterns to compiled regexes.

    Supports ``*`` as wildcard and exact string matches.
    """
    compiled: List[re.Pattern[str]] = []
    for pat in model_patterns:
        if "*" in pat:
            regex = "^" + re.escape(pat).replace(r"\*", ".*") + "$"
            compiled.append(re.compile(regex, re.IGNORECASE))
        else:
            compiled.append(re.compile("^" + re.escape(pat) + "$", re.IGNORECASE))
    return compiled


_MODEL_PATTERNS: Optional[List[re.Pattern[str]]] = None


def _get_patterns() -> List[re.Pattern[str]]:
    """Return lazily-compiled model patterns from config."""
    global _MODEL_PATTERNS
    if _MODEL_PATTERNS is None:
        _MODEL_PATTERNS = _compile_patterns(SYNTHETIC_RESPOND_TOOL_MODELS)
    return _MODEL_PATTERNS


def reset_patterns_cache() -> None:
    """Reset the cached model patterns (for testing)."""
    global _MODEL_PATTERNS
    _MODEL_PATTERNS = None


def model_matches(model: str) -> bool:
    """Return True if *model* matches any configured respond-tool pattern."""
    if not SYNTHETIC_RESPOND_TOOL_MODELS:
        return False
    for pat in _get_patterns():
        if pat.search(model):
            return True
    return False


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------

def inject_respond_tool(kwargs: Dict[str, Any]) -> bool:
    """Inject the ``respond`` tool definition into *kwargs* in-place.

    Returns True if the tool was injected (and the caller should expect
    to extract it from the response later).

    - Appends the tool to an existing ``tools`` list.
    - Creates a new ``tools`` list if none exists.
    - Forces ``tool_choice`` to ``"auto"`` only when the caller did not
      already set it (to avoid overriding explicit choices).
    """
    model = kwargs.get("model", "")
    if not model_matches(model):
        return False

    tools = kwargs.get("tools")
    if tools is None:
        kwargs["tools"] = [RESPOND_TOOL_DEFINITION]
    elif isinstance(tools, list):
        # Avoid duplicate injection on retry
        for t in tools:
            if (
                isinstance(t, dict)
                and t.get("type") == "function"
                and isinstance(t.get("function"), dict)
                and t["function"].get("name") == "_proxy_respond"
            ):
                return True  # already injected
        tools.append(RESPOND_TOOL_DEFINITION)
    else:
        return False

    # Ensure tool_choice allows the model to call respond
    if "tool_choice" not in kwargs:
        kwargs["tool_choice"] = "auto"

    logger.debug(
        "Synthetic respond tool injected for model %s", model,
    )
    return True


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_respond_call(tool_call: Dict[str, Any]) -> Optional[str]:
    """Extract the ``response`` argument from a respond tool call.

    Returns the text string, or None if the tool call is not a valid
    respond invocation.
    """
    if not isinstance(tool_call, dict):
        return None

    # OpenAI format: {"function": {"name": "_proxy_respond", "arguments": "{...}"}}
    fn = tool_call.get("function")
    if not isinstance(fn, dict):
        return None
    if fn.get("name") != "_proxy_respond":
        return None

    arguments = fn.get("arguments", "")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            # Best-effort: treat raw string as the response text
            return arguments if arguments else None
    if isinstance(arguments, dict):
        return arguments.get("response")
    return None


def _find_respond_tool_call(message: Dict[str, Any]) -> Optional[str]:
    """Search a message dict for a respond tool call and return its text."""
    # Check tool_calls array (OpenAI format)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            text = _extract_text_from_respond_call(tc)
            if text is not None:
                return text

    # Check legacy function_call format
    function_call = message.get("function_call")
    if isinstance(function_call, dict) and function_call.get("name") == "_proxy_respond":
        args = function_call.get("arguments", "")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                return args if args else None
        if isinstance(args, dict):
            return args.get("response")

    return None


# ---------------------------------------------------------------------------
# Non-streaming extraction
# ---------------------------------------------------------------------------

def extract_respond_from_response(response: Any) -> Any:
    """Rewrite a respond tool call in *response* into a plain text completion.

    If the response contains a ``respond`` tool call, the first choice's
    message is rewritten to ``content=<text>`` with ``finish_reason="stop"``,
    and tool_calls/function_call are removed.

    Returns the (possibly modified) response.  When no respond tool call
    is found, the response is returned unchanged.
    """
    # Work with dict or pydantic model
    is_pydantic = hasattr(response, "model_dump")
    if is_pydantic:
        resp_dict = response.model_dump()
    elif isinstance(response, dict):
        resp_dict = response
    else:
        return response

    choices = resp_dict.get("choices")
    if not isinstance(choices, list) or not choices:
        return response

    modified = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        text = _find_respond_tool_call(message)
        if text is not None:
            message["content"] = text
            message["role"] = "assistant"
            message.pop("tool_calls", None)
            message.pop("function_call", None)
            choice["finish_reason"] = "stop"
            modified = True

    if not modified:
        return response

    # For pydantic models, we return a dict (callers handle both)
    return resp_dict


# ---------------------------------------------------------------------------
# History cleanup
# ---------------------------------------------------------------------------

def strip_respond_tool_from_history(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of *request_data* with `_proxy_respond` tool stripped.

    Removes:
    - `_proxy_respond` tool definition from `tools` list
    - Assistant messages containing `_proxy_respond` tool_calls
    - Tool role messages that are responses to `_proxy_respond` (match by tool_call_id)
    """
    data = copy.deepcopy(request_data)

    # Strip from tools list
    tools = data.get("tools")
    if isinstance(tools, list):
        data["tools"] = [
            t for t in tools
            if not (
                isinstance(t, dict)
                and t.get("type") == "function"
                and isinstance(t.get("function"), dict)
                and t["function"].get("name") == "_proxy_respond"
            )
        ]

    # Strip from messages
    messages = data.get("messages")
    if isinstance(messages, list):
        removed_tool_call_ids: set[str] = set()
        cleaned_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                cleaned_messages.append(msg)
                continue

            role = msg.get("role")

            # Handle assistant messages with tool_calls
            if role == "assistant":
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    new_tool_calls = []
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            new_tool_calls.append(tc)
                            continue
                        fn = tc.get("function")
                        if isinstance(fn, dict) and fn.get("name") == "_proxy_respond":
                            tc_id = tc.get("id")
                            if tc_id:
                                removed_tool_call_ids.add(tc_id)
                        else:
                            new_tool_calls.append(tc)
                    if new_tool_calls:
                        msg_copy = dict(msg)
                        msg_copy["tool_calls"] = new_tool_calls
                        cleaned_messages.append(msg_copy)
                    elif msg.get("content"):
                        # Keep message if it has content but no more tool_calls
                        msg_copy = dict(msg)
                        msg_copy.pop("tool_calls", None)
                        cleaned_messages.append(msg_copy)
                    # else: drop empty assistant message
                else:
                    cleaned_messages.append(msg)

            # Handle tool response messages
            elif role == "tool":
                tool_call_id = msg.get("tool_call_id")
                if tool_call_id and tool_call_id in removed_tool_call_ids:
                    continue  # Skip this tool response
                cleaned_messages.append(msg)

            else:
                cleaned_messages.append(msg)

        data["messages"] = cleaned_messages

    return data


# ---------------------------------------------------------------------------
# Streaming extraction
# ---------------------------------------------------------------------------

class StreamingRespondExtractor:
    """Accumulates tool call chunks across a stream and extracts respond text.

    Streaming providers send tool_calls incrementally:
    - First chunk: ``{"index": 0, "id": "...", "function": {"name": "_proxy_respond"}}``
    - Subsequent chunks: ``{"index": 0, "function": {"arguments": "..."}}``

    This extractor buffers argument fragments and emits the assembled text
    when the stream ends or a non-respond tool call is detected.
    """

    def __init__(self) -> None:
        self._arguments_buffer: List[str] = []
        self._is_respond: bool = False
        self._saw_tool_call: bool = False
        self._buffer_overflow: bool = False

    @property
    def saw_respond_tool(self) -> bool:
        return self._is_respond

    def process_chunk(self, chunk_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a streaming chunk.

        Returns:
            - The chunk dict (possibly rewritten) if it should be yielded.
            - None if the chunk should be suppressed (intermediate tool call chunks).
        """
        choices = chunk_dict.get("choices")
        if not isinstance(choices, list) or not choices:
            return chunk_dict

        choice = choices[0]
        if not isinstance(choice, dict):
            return chunk_dict

        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            return chunk_dict

        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            # No tool calls in this chunk
            if self._saw_tool_call and self._is_respond:
                # We were accumulating a respond tool call — suppress empty deltas
                return None
            return chunk_dict

        self._saw_tool_call = True

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue

            # First chunk may set the name
            name = fn.get("name")
            if name == "_proxy_respond":
                self._is_respond = True

            # Accumulate arguments
            args_fragment = fn.get("arguments")
            if args_fragment and self._is_respond:
                current_size = sum(len(f) for f in self._arguments_buffer)
                if current_size + len(args_fragment) <= _MAX_RESPOND_ARGS:
                    self._arguments_buffer.append(args_fragment)
                else:
                    self._buffer_overflow = True
                    logger.warning("Respond tool arguments exceeded 1MB buffer cap; truncating.")

        # Suppress intermediate tool call chunks for respond
        if self._is_respond:
            return None

        return chunk_dict

    def extract_text(self) -> Optional[str]:
        """Extract the assembled respond text from accumulated arguments.

        Call this after the stream ends to get the final text.
        """
        if not self._is_respond or not self._arguments_buffer:
            return None

        raw = "".join(self._arguments_buffer)
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                text = parsed.get("response")
                if text and self._buffer_overflow:
                    text += " [truncated]"
                return text
        except (json.JSONDecodeError, TypeError):
            # Best-effort: return raw string
            if raw:
                return raw + " [truncated]" if self._buffer_overflow else raw
            return None
        return None

    def build_final_text_chunk(
        self, text: str, model: str
    ) -> Dict[str, Any]:
        """Build a synthetic content chunk for the extracted respond text."""
        return {
            "id": "",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }
