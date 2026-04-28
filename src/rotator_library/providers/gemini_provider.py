# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import httpx
from typing import List, Dict, Any
from .provider_interface import ProviderInterface
from .utilities import (
    DEFAULT_GEMINI_SAFETY_SETTINGS_MAP,
    DEFAULT_SAFETY_SETTINGS,
    fetch_provider_models,
)

class GeminiProvider(ProviderInterface):
    """
    Provider implementation for the Google Gemini API.
    """
    async def get_models(self, api_key: str, client: httpx.AsyncClient) -> List[str]:
        """
        Fetches the list of available models from the Google Gemini API.
        """
        return await fetch_provider_models(
            client,
            "https://generativelanguage.googleapis.com/v1beta/models",
            {"x-goog-api-key": api_key},
            "Gemini",
            lambda data: [
                f"gemini/{model['name'].replace('models/', '')}"
                for model in data.get("models", [])
                if isinstance(model, dict) and "name" in model
            ],
        )

    def convert_safety_settings(self, settings: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Converts generic safety settings to the Gemini-specific format.
        """
        if not settings:
            # Return full defaults if nothing provided
            return [dict(item) for item in DEFAULT_SAFETY_SETTINGS]

        # Default gemini-format settings for merging
        default_gemini = DEFAULT_GEMINI_SAFETY_SETTINGS_MAP

        # If the caller already provided Gemini-style list, merge defaults without overwriting
        if isinstance(settings, list):
            existing = {item.get("category"): item for item in settings if isinstance(item, dict) and item.get("category")}
            merged = list(settings)
            for cat, thr in default_gemini.items():
                if cat not in existing:
                    merged.append({"category": cat, "threshold": thr})
            return merged

        # Otherwise assume a generic mapping (dict) and convert
        gemini_settings = []
        category_map = {
            "harassment": "HARM_CATEGORY_HARASSMENT",
            "hate_speech": "HARM_CATEGORY_HATE_SPEECH",
            "sexually_explicit": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "dangerous_content": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "civic_integrity": "HARM_CATEGORY_CIVIC_INTEGRITY",
        }

        for generic_category, threshold in settings.items():
            if generic_category in category_map:
                thr = (threshold or "").upper()
                gemini_settings.append({
                    "category": category_map[generic_category],
                    "threshold": thr if thr else default_gemini[category_map[generic_category]]
                })

        # Add any missing defaults
        present = {s["category"] for s in gemini_settings}
        for cat, thr in default_gemini.items():
            if cat not in present:
                gemini_settings.append({"category": cat, "threshold": thr})

        return gemini_settings

    def handle_thinking_parameter(self, payload: Dict[str, Any], model: str):
        """
        Handles reasoning parameters for Gemini models, with three distinct paths:
        1. Applies a non-standard, high-value token budget if 'custom_reasoning_budget' is true.
        2. Leaves the 'reasoning_effort' parameter alone for LiteLLM to handle if it's present
           without the custom flag.
        3. Applies a default 'thinking' value for specific models if no other reasoning
           parameters are provided, ensuring they 'think' by default.
        """
        # Set default temperature to 1 if not provided
        if "temperature" not in payload:
            payload["temperature"] = 1

        custom_reasoning_budget = payload.get("custom_reasoning_budget", False)
        reasoning_effort = payload.get("reasoning_effort")

        # If 'thinking' is already explicitly set, do nothing to avoid overriding it.
        if "thinking" in payload:
            return

        # Path 1: Custom budget is explicitly requested.
        if custom_reasoning_budget:
            # Case 1a: Both params are present, so we can apply the custom budget.
            if reasoning_effort:
                if "gemini-2.5-pro" in model:
                    budgets = {"low": 8192, "medium": 16384, "high": 32768}
                elif "gemini-2.5-flash" in model:
                    budgets = {"low": 6144, "medium": 12288, "high": 24576}
                else: # Fallback for other models if the custom flag is still used
                    budgets = {"low": 1024, "medium": 2048, "high": 4096}
                
                budget = budgets.get(reasoning_effort)
                if budget is not None:
                    payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
                elif reasoning_effort == "disable":
                    payload["thinking"] = {"type": "enabled", "budget_tokens": 0}
                
                # Clean up the handled 'reasoning_effort' parameter.
                payload.pop("reasoning_effort", None)

            # Case 1b: In all cases where the custom flag was present, remove it
            # as it's not a standard LiteLLM parameter.
            payload.pop("custom_reasoning_budget", None)
            return

        # Path 2: No custom budget. Now check for standard or default behavior.
        # If 'reasoning_effort' is present, we do nothing, allowing LiteLLM to handle it.
        # If 'reasoning_effort' is NOT present, then we apply the default thinking behavior.
        if not reasoning_effort:
            if "gemini-2.5-pro" in model or "gemini-2.5-flash" in model:
                payload["thinking"] = {"type": "enabled", "budget_tokens": -1}
