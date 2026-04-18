# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
import json
import logging
from typing import List, Dict, Any
from .provider_interface import ProviderInterface  # strip_provider_prefix intentionally not used — nvidia preserves org prefix
from .utilities.nvidia_quota_tracker import NvidiaQuotaTracker

lib_logger = logging.getLogger('rotator_library')

# NVIDIA NIM models that support native thinking via chat_template_kwargs
_NVIDIA_DEEPSEEK_THINKING_MODELS = frozenset([
    "deepseek-ai/deepseek-v3.1",
    "deepseek-ai/deepseek-v3.1-terminus",
    "deepseek-ai/deepseek-v3.2",
])

# Qwen models that support thinking via enable_thinking parameter
_QWEN_THINKING_MODELS = frozenset([
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-coder-480b-a35b-instruct",
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

    # Provider name for env var lookups
    provider_env_name = "NVIDIA_NIM"

    def __init__(self):
        super().__init__()
        self._quota_tracker = NvidiaQuotaTracker()

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
            try:
                data = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                lib_logger.warning(f"Invalid JSON from NVIDIA models: {e}, body={response.text[:200]}")
                return []
            models = [f"nvidia/{model['id']}" for model in data.get("data", []) if isinstance(model, dict) and "id" in model]
            return models
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (401, 403):
                lib_logger.warning(f"Auth error fetching NVIDIA models: {e.response.status_code}")
            elif e.response.status_code >= 500:
                lib_logger.warning(f"Server error fetching NVIDIA models: {e.response.status_code}")
            else:
                lib_logger.error(f"HTTP error fetching NVIDIA models: {e}")
            return []
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

        Qwen models (qwen3.5, qwen3-coder, etc.):
            - When 'reasoning_effort' is set, enables native thinking
              via 'extra_body.chat_template_kwargs.enable_thinking = True'.
            - Strips the Anthropic 'thinking' field regardless.

        All other models (Llama, Mistral, etc.):
            - Strips 'thinking' field entirely (not supported by NVIDIA NIM for these models).
            - Strips 'cache_control' from all content blocks (Anthropic prompt caching).
            - Collapses list-format system messages into plain strings.

        This method is called from client.py for the 'nvidia' provider
        before the litellm call is made.

        Args:
            payload: The litellm_kwargs dict (modified in-place).
            model: Full model string, e.g. 'nvidia/qwen/qwen3.5-397b-a17b'.
        """
        # Extract the bare model name without provider prefix
        # model can be 'nvidia/qwen/qwen3.5-397b-a17b'
        # Strip the first component (provider prefix) to get the actual model ID
        model_name = model.split('/', 1)[1] if '/' in model else model

        # --- Step 1: Strip all Anthropic-specific top-level fields ---
        self._strip_anthropic_fields(payload)

        # --- Step 2: Strip cache_control and normalize messages ---
        self._sanitize_messages(payload)

        # --- Step 3: Handle reasoning/thinking for supported models ---
        reasoning_effort = payload.get("reasoning_effort")

        # DeepSeek models
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
        
        # Qwen models
        elif model_name in _QWEN_THINKING_MODELS:
            if reasoning_effort:
                payload.setdefault("extra_body", {})
                payload["extra_body"].setdefault("chat_template_kwargs", {})
                payload["extra_body"]["chat_template_kwargs"]["enable_thinking"] = True
                lib_logger.info(
                    f"[NvidiaProvider] Enabled native thinking for Qwen model "
                    f"'{model_name}' (reasoning_effort='{reasoning_effort}')."
                )
            else:
                lib_logger.debug(
                    f"[NvidiaProvider] Qwen model '{model_name}' without reasoning_effort; "
                    f"thinking not enabled."
                )
        
        # Other models
        else:
            # Non-DeepSeek/Qwen models: log if reasoning_effort was present but not actionable
            if reasoning_effort:
                lib_logger.debug(
                    f"[NvidiaProvider] Model '{model_name}' does not support native thinking; "
                    f"reasoning_effort='{reasoning_effort}' will be left for litellm to handle or ignore."
                )

    # =========================================================================
    # QUOTA TRACKING METHODS
    # =========================================================================

    async def track_request(self, credential: str, model: str) -> None:
        """
        Track a request for rate limit enforcement.

        Args:
            credential: Credential identifier
            model: Model name (e.g., "nvidia/deepseek-ai/deepseek-v3.1")
        """
        await self._quota_tracker.track_request(credential, model)

    def can_make_request(self, credential: str, model: str) -> bool:
        """
        Check if request is within rate limits.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            True if request is allowed, False otherwise
        """
        return self._quota_tracker.can_make_request(credential, model)

    def get_wait_time(self, credential: str, model: str) -> float:
        """
        Get seconds to wait before next request.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            Seconds to wait (0.0 if can proceed immediately)
        """
        return self._quota_tracker.get_wait_time(credential, model)

    def get_quota_info(self, credential: str, model: str) -> Dict:
        """
        Get quota information for credential and model.

        Args:
            credential: Credential identifier
            model: Model name

        Returns:
            Dictionary with quota information
        """
        return self._quota_tracker.get_usage_stats(credential, model)

    # =========================================================================
    # BACKGROUND JOB FOR QUOTA SYNC
    # =========================================================================

    def get_background_job_config(self) -> Dict[str, Any]:
        """
        Get background job configuration for NVIDIA quota sync.

        Returns:
            Dictionary with job configuration or None if no job needed
        """
        return {
            "name": "nvidia_quota_sync",
            "interval": 600,  # Run every 10 minutes
            "run_on_start": False,
        }

    async def run_background_job(
        self, usage_manager: Any, credentials: List[str]
    ) -> None:
        """
        Background job to sync NVIDIA quota windows.

        Cleans up old rate limit windows to prevent memory leaks.

        Args:
            usage_manager: UsageManager instance (not used for NVIDIA)
            credentials: List of credential paths (not used for NVIDIA)
        """
        try:
            cleaned = self._quota_tracker.cleanup_old_windows()
            if cleaned > 0:
                lib_logger.debug(
                    f"NVIDIA quota sync: cleaned {cleaned} old windows"
                )
        except Exception as e:
            lib_logger.error(f"Error in NVIDIA quota sync job: {e}")
