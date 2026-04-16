# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""
Monkey-patches to normalize litellm behavior.

1. finish_reason patch: Normalizes invalid finish_reason values from providers
   (e.g., Z.AI returns "error"/"abort"/"unknown" instead of valid OpenAI literals).

2. serialization-warning suppression: Silences harmless Pydantic serialization
   warnings from litellm's internal type mismatches.

3. print suppression: Context manager to suppress litellm's direct print()
   statements for unknown providers.

Usage:
    from rotator_library.utils.litellm_patches import patch_litellm_finish_reason
    patch_litellm_finish_reason()  # Call once at startup, BEFORE importing litellm/openai

    from rotator_library.utils.litellm_patches import suppress_litellm_serialization_warnings
    suppress_litellm_serialization_warnings()

    from rotator_library.utils.litellm_patches import suppress_litellm_prints
    with suppress_litellm_prints():
        cost = litellm.completion_cost(completion_response, model=model)

Environment variables:
    PATCH_LITELLM_FINISH_REASON=0   — disable finish_reason patch
    SUPPRESS_LITELLM_SERIALIZATION_WARNINGS=0 — disable warning suppression
"""

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from io import StringIO
from typing import Optional

logger = logging.getLogger("rotator_library.litellm_patches")

# ---------------------------------------------------------------------------
# finish_reason patch
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Serialization-warning suppression
# ---------------------------------------------------------------------------

def suppress_litellm_serialization_warnings():
    """
    Suppress litellm's internal Pydantic serialization warnings.

    Scoped to only silence:
    - UserWarning category
    - From pydantic.main module
    - Matching "Pydantic serializer warnings: PydanticSerializationUnexpectedValue"

    Can be disabled by setting SUPPRESS_LITELLM_SERIALIZATION_WARNINGS=0
    """
    if os.getenv("SUPPRESS_LITELLM_SERIALIZATION_WARNINGS", "1") == "1":
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module=r"pydantic\.main",
            message=r"Pydantic serializer warnings:\s+PydanticSerializationUnexpectedValue",
        )


# ---------------------------------------------------------------------------
# Print suppression (context manager)
# ---------------------------------------------------------------------------

@contextmanager
def suppress_litellm_prints():
    """
    Context manager to suppress LiteLLM's direct print() statements.

    LiteLLM uses print() directly for "Provider List" messages when it encounters
    unknown providers. This context manager temporarily redirects stdout to prevent
    this spam from appearing in logs/console.

    Usage:
        with suppress_litellm_prints():
            cost = litellm.completion_cost(completion_response, model=model)
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
