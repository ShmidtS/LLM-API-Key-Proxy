# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""
Monkey-patch to normalize invalid finish_reason values from providers.

Some providers (e.g., Z.AI) return finish_reason values that don't match
OpenAI/LiteLLM schemas. This causes issues:

1. OpenAI SDK's Choice class uses Literal type for finish_reason
2. LiteLLM's streaming_handler.py explicitly checks for finish_reason == "error"
   and raises an exception (line 1317).
3. LiteLLM's stream_chunk_builder uses litellm.types.utils.Choices which also
   validates finish_reason with Pydantic Literal types.

Valid finish_reason values (Pydantic Literal in LiteLLM):
- 'stop', 'length', 'tool_calls', 'content_filter', 'function_call'
- 'guardrail_intervened', 'eos', 'finish_reason_unspecified', 'malformed_function_call'

Invalid values we've seen from providers:
- 'error', 'unknown', 'abort' (Z.AI/Novita)

This patch wraps:
1. ChatCompletionChunk.model_validate (OpenAI SDK)
2. litellm.types.utils.Choices.__init__ (LiteLLM internal)

to normalize invalid values before Pydantic validation runs.

Usage:
    from rotator_library.utils.patch_litellm_finish_reason import patch_litellm_finish_reason
    patch_litellm_finish_reason()  # Call once at startup

Can be disabled with environment variable: PATCH_LITELLM_FINISH_REASON=0
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("rotator_library.patch_litellm_finish_reason")

# Mapping of invalid finish_reason values to valid ones
FINISH_REASON_MAP = {
    "error": "stop",
    "abort": "stop",
    "unknown": "stop",
}

# Valid finish_reason values per LiteLLM Pydantic schema (from error message)
LITELLM_VALID_FINISH_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "function_call",
    "guardrail_intervened",
    "eos",
    "finish_reason_unspecified",
    "malformed_function_call",
}

_original_chat_completion_chunk_model_validate = None
_original_litellm_choices_init = None
_patched: bool = False


def _normalize_finish_reason(finish_reason: Optional[str]) -> Optional[str]:
    """Normalize an invalid finish_reason to a valid one."""
    if finish_reason is None:
        return None

    if finish_reason in LITELLM_VALID_FINISH_REASONS:
        return finish_reason

    if finish_reason in FINISH_REASON_MAP:
        normalized = FINISH_REASON_MAP[finish_reason]
        logger.debug(f"Normalized finish_reason: '{finish_reason}' -> '{normalized}'")
        return normalized

    # Unknown value - log warning and map to 'stop'
    logger.warning(f"Unknown finish_reason '{finish_reason}', mapping to 'stop'")
    return "stop"


def _normalize_chunk_data(data):
    """Normalize finish_reason in chunk data (dict)."""
    if not isinstance(data, dict):
        return data

    # Normalize finish_reason in choices
    if "choices" in data and isinstance(data["choices"], list):
        for choice in data["choices"]:
            if isinstance(choice, dict) and "finish_reason" in choice:
                original = choice["finish_reason"]
                normalized = _normalize_finish_reason(original)
                if normalized != original:
                    logger.debug(f"Patched finish_reason: {original} -> {normalized}")
                choice["finish_reason"] = normalized

    return data


def _patch_litellm_choices():
    """
    Patch litellm.types.utils.Choices to normalize finish_reason before validation.

    This is needed because LiteLLM's stream_chunk_builder creates Choices objects
    directly, bypassing ChatCompletionChunk.model_validate.
    """
    global _original_litellm_choices_init

    try:
        from litellm.types.utils import Choices

        _original_litellm_choices_init = Choices.__init__

        def patched_choices_init(self, **kwargs):
            # Normalize finish_reason if present
            if "finish_reason" in kwargs:
                original = kwargs["finish_reason"]
                normalized = _normalize_finish_reason(original)
                if normalized != original:
                    logger.debug(f"Choices: normalized finish_reason {original} -> {normalized}")
                kwargs["finish_reason"] = normalized
            return _original_litellm_choices_init(self, **kwargs)

        Choices.__init__ = patched_choices_init
        logger.info("Applied finish_reason patch to litellm.types.utils.Choices")

    except ImportError as e:
        logger.warning(f"Could not patch litellm.types.utils.Choices: {e}")
    except Exception as e:
        logger.error(f"Unexpected error patching LiteLLM Choices: {e}")


def patch_litellm_finish_reason():
    """
    Apply monkey-patches to normalize finish_reason values.

    Patches:
    1. OpenAI SDK's ChatCompletionChunk.model_validate
    2. LiteLLM's litellm.types.utils.Choices.__init__
    """
    global _original_chat_completion_chunk_model_validate, _patched

    if os.getenv("PATCH_LITELLM_FINISH_REASON", "1") == "0":
        logger.info("finish_reason patch disabled by environment variable")
        return

    if _patched:
        logger.debug("finish_reason patch already applied")
        return

    # Patch OpenAI ChatCompletionChunk
    try:
        from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

        _original_chat_completion_chunk_model_validate = ChatCompletionChunk.model_validate

        def patched_model_validate(obj, *args, **kwargs):
            obj = _normalize_chunk_data(obj)
            return _original_chat_completion_chunk_model_validate(obj, *args, **kwargs)

        ChatCompletionChunk.model_validate = staticmethod(patched_model_validate)
        logger.info("Applied finish_reason patch to ChatCompletionChunk.model_validate")

    except ImportError as e:
        logger.warning(f"Could not patch OpenAI ChatCompletionChunk: {e}")
    except Exception as e:
        logger.error(f"Unexpected error patching OpenAI: {e}")

    # Patch LiteLLM Choices (critical for stream_chunk_builder)
    _patch_litellm_choices()

    _patched = True


def unpatch_litellm_finish_reason():
    """Remove all monkey-patches (OpenAI ChatCompletionChunk and LiteLLM Choices)."""
    global _original_chat_completion_chunk_model_validate, _original_litellm_choices_init, _patched

    if not _patched:
        return

    # Restore OpenAI ChatCompletionChunk
    try:
        from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

        if _original_chat_completion_chunk_model_validate is not None:
            ChatCompletionChunk.model_validate = _original_chat_completion_chunk_model_validate
            _original_chat_completion_chunk_model_validate = None
    except Exception as e:
        logger.error(f"Error removing OpenAI patch: {e}")

    # Restore LiteLLM Choices
    try:
        from litellm.types.utils import Choices

        if _original_litellm_choices_init is not None:
            Choices.__init__ = _original_litellm_choices_init
            _original_litellm_choices_init = None
    except Exception as e:
        logger.error(f"Error removing LiteLLM Choices patch: {e}")

    _patched = False
    logger.info("Removed all finish_reason patches")