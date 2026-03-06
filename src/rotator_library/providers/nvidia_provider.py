# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

import httpx
import logging
from typing import List, Dict, Any
import litellm
from .provider_interface import ProviderInterface

lib_logger = logging.getLogger('rotator_library')

# NVIDIA NIM models that support native thinking via chat_template_kwargs
_NVIDIA_DEEPSEEK_THINKING_MODELS = frozenset([
    "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-ai/deepseek-v3.2",
])

# Top-level payload fields that are Anthropic-specific and not supported by NVIDIA NIM
_ANTHROPIC_UNSUPPORTED_FIELDS = frozenset([
    "thinking",          # Anthropic thinking control {type: "adaptive"} / {type: "enabled", budget_tokens: N}
    "betas",             # Anthropic beta features list
])


class NvidiaProvider(ProviderInterface):
    skip_cost_calculation = True
    """
    Provider implementation for the NVIDIA NIM API.

    Handles translation of Anthropic-format requests (as sent by Claude Code)
    into NVIDIA-compatible OpenAI-format requests.
    """

    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the NVIDIA NIM API.
        """
        try:
            response = await client.get(
                "https://integrate.api.nvidia.com/v1/models",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            response.raise_for_status()
            models = [f"nvidia_nim/{model['id']}" for model in response.json().get("data", [])]
            return models
        except httpx.RequestError as e:
            lib_logger.error(f"Failed to fetch NVIDIA models: {e}")
            return []

    def _strip_cache_control_from_content(self, content: Any) -> Any:
        """
        Recursively strips 'cache_control' fields from message content blocks.

        Claude Code sends cache_control: {type: "ephemeral"} on system message blocks
        and user message content blocks. NVIDIA NIM does not support this field and
        may reject requests containing it.

        Args:
            content: The content field of a message — either a string (returned as-is)
                     or a list of content blocks (each block is cleaned in place).

        Returns:
            Cleaned content with all cache_control fields removed.
        """
        if isinstance(content, list):
            cleaned = []
            for block in content:
                if isinstance(block, dict):
                    block = {k: v for k, v in block.items() if k != "cache_control"}
                cleaned.append(block)
            return cleaned
        return content

    def _sanitize_messages(self, payload: Dict[str, Any]) -> None:
        """
        Strips Anthropic-specific fields from all messages in the payload.

        - Removes 'cache_control' from every content block in every message.
        - Converts system messages with list content to plain strings if all
          blocks are text-only (NVIDIA expects string system prompts).
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if content is None:
                continue

            # Strip cache_control from content blocks
            cleaned = self._strip_cache_control_from_content(content)

            # For system messages with list content, collapse to a single string
            # NVIDIA NIM (OpenAI-compatible) expects system content as a plain string
            role = msg.get("role", "")
            if role == "system" and isinstance(cleaned, list):
                text_parts = []
                all_text = True
                for block in cleaned:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    else:
                        all_text = False
                        break
                if all_text and text_parts:
                    cleaned = "\n\n".join(text_parts)

            msg["content"] = cleaned

    def _strip_anthropic_fields(self, payload: Dict[str, Any]) -> None:
        """
        Removes top-level Anthropic-specific payload fields that NVIDIA NIM does not support.

        Fields removed:
            - 'thinking': Anthropic adaptive/enabled thinking parameter
            - 'betas': Anthropic beta features list
        """
        for field in _ANTHROPIC_UNSUPPORTED_FIELDS:
            if field in payload:
                lib_logger.debug(
                    f"[NvidiaProvider] Stripping unsupported Anthropic field '{field}' from payload."
                )
                del payload[field]

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str) -> None:
        """
        Handles reasoning/thinking parameters for NVIDIA NIM models.

        Behavior per model family:

        DeepSeek models (deepseek-v3.1, deepseek-v3.2, etc.):
            - When 'reasoning_effort' is set (low/medium/high), enables native thinking
              via 'extra_body.chat_template_kwargs.thinking = True'.
            - Strips the Anthropic 'thinking' field regardless.

        All other models (Qwen, Llama, Mistral, etc.):
            - Strips 'thinking' field entirely (not supported by NVIDIA NIM for these models).
            - Strips 'cache_control' from all content blocks (Anthropic prompt caching).
            - Collapses list-format system messages into plain strings.

        This method is called from client.py for both 'nvidia_nim' and 'nvidia' providers
        before the litellm call is made.

        Args:
            payload: The litellm_kwargs dict (modified in-place).
            model: Full model string, e.g. 'nvidia_nim/qwen/qwen3.5-397b-a17b'.
        """
        # Extract the bare model name without provider prefix
        # model can be 'nvidia_nim/qwen/qwen3.5-397b-a17b' or 'nvidia/qwen/qwen3.5-397b-a17b'
        # Strip the first component (provider prefix) to get the actual model ID
        model_name = model.split('/', 1)[1] if '/' in model else model

        # --- Step 1: Strip all Anthropic-specific top-level fields ---
        self._strip_anthropic_fields(payload)

        # --- Step 2: Strip cache_control and normalize messages ---
        self._sanitize_messages(payload)

        # --- Step 3: Handle reasoning/thinking for DeepSeek models ---
        reasoning_effort = payload.get("reasoning_effort")

        if model_name in _NVIDIA_DEEPSEEK_THINKING_MODELS:
            if reasoning_effort in ("low", "medium", "high"):
                payload.setdefault("extra_body", {})
                payload["extra_body"].setdefault("chat_template_kwargs", {})
                payload["extra_body"]["chat_template_kwargs"]["thinking"] = True
                lib_logger.info(
                    f"[NvidiaProvider] Enabled native thinking for DeepSeek model "
                    f"'{model_name}' (reasoning_effort='{reasoning_effort}')."
                )
            else:
                lib_logger.debug(
                    f"[NvidiaProvider] DeepSeek model '{model_name}' without reasoning_effort; "
                    f"thinking not enabled."
                )
        else:
            # Non-DeepSeek models: log if reasoning_effort was present but not actionable
            if reasoning_effort:
                lib_logger.debug(
                    f"[NvidiaProvider] Model '{model_name}' does not support native thinking; "
                    f"reasoning_effort='{reasoning_effort}' will be left for litellm to handle or ignore."
                )
