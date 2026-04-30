import asyncio
import logging
import time
from typing import Dict, List, Optional, Union

from litellm.litellm_core_utils.token_counter import token_counter

from ..error_handler import classify_error
from ..error_types import mask_credential
from ..utils.model_utils import extract_provider_from_model, match_model_pattern

lib_logger = logging.getLogger("rotator_library")


class ModelsMixin:
    def _match_model_pattern(
        self,
        provider: str,
        model_id: str,
        pattern_dict: dict,
        wildcard_return: bool = False,
    ) -> bool:
        """
        Checks if a model matches any pattern in the given dict.

        Args:
            provider: Provider name
            model_id: Full model ID (e.g., "openai/gpt-4")
            pattern_dict: Dict mapping provider -> list of fnmatch patterns
            wildcard_return: Return value when pattern list is ["*"] (True for ignore, False for whitelist)

        Pattern examples:
        - "gpt-4" - exact match
        - "gpt-4*" - prefix wildcard (matches gpt-4, gpt-4-turbo, etc.)
        - "*-preview" - suffix wildcard
        - "*" - match all
        """
        return match_model_pattern(provider, model_id, pattern_dict, wildcard_return)

    def _is_model_ignored(self, provider: str, model_id: str) -> bool:
        """Checks if a model should be ignored based on the ignore list."""
        return self._match_model_pattern(
            provider, model_id, self.ignore_models, wildcard_return=True
        )

    def _is_model_whitelisted(self, provider: str, model_id: str) -> bool:
        """Checks if a model is explicitly whitelisted."""
        return self._match_model_pattern(
            provider, model_id, self.whitelist_models, wildcard_return=False
        )

    async def _resolve_model_id(self, model: str, provider: str) -> str:
        """
        Resolves the actual model ID to send to the provider.

        For custom models with name/ID mappings, returns the ID.
        Otherwise, returns the model name unchanged.

        Results are cached with TTL to avoid repeated provider lookups.
        Cache is invalidated when providers are refreshed.

        Args:
            model: Full model string with provider (e.g., "iflow/DS-v3.2")
            provider: Provider name (e.g., "iflow")

        Returns:
            Full model string with ID (e.g., "iflow/deepseek-v3.2")
        """
        cache_key = (model, provider)
        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            cached = self._resolve_model_id_cache.get(cache_key)
            if cached is not None:
                return cached

            # Extract model name from "provider/model_name" format
            model_name = model.split("/")[-1] if "/" in model else model

            # Try to get provider instance to check for model definitions
            provider_plugin = self._get_provider_instance(provider)

            # Check if provider has model definitions
            if provider_plugin and hasattr(provider_plugin, "model_definitions"):
                model_id = provider_plugin.model_definitions.get_model_id(
                    provider, model_name
                )
                if model_id and model_id != model_name:
                    result = f"{provider}/{model_id}"
                    self._resolve_model_id_cache[cache_key] = result
                    return result

            # Fallback: use client's own model definitions
            model_id = self.model_definitions.get_model_id(provider, model_name)
            if model_id and model_id != model_name:
                result = f"{provider}/{model_id}"
                self._resolve_model_id_cache[cache_key] = result
                return result

            # No conversion needed, return original
            self._resolve_model_id_cache[cache_key] = model
            return model

    def token_count(self, **kwargs) -> int:
        """Calculates the number of tokens for a given text or list of messages.

        For Antigravity provider models, this also includes the preprompt tokens
        that get injected during actual API calls (agent instruction + identity override).
        This ensures token counts match actual usage.
        """
        model = kwargs.get("model")
        text = kwargs.get("text")
        messages = kwargs.get("messages")

        if not model:
            raise ValueError("'model' is a required parameter.")

        # Calculate base token count
        if messages:
            base_count = token_counter(model=model, messages=messages)
        elif text:
            base_count = token_counter(model=model, text=text)
        else:
            raise ValueError("Either 'text' or 'messages' must be provided.")

        # Add preprompt tokens for Antigravity provider
        # The Antigravity provider injects system instructions during actual API calls,
        # so we need to account for those tokens in the count
        provider = extract_provider_from_model(model)
        if provider == "antigravity":
            try:
                from ..providers.antigravity.constants import (
                    get_antigravity_preprompt_text,
                )

                preprompt_text = get_antigravity_preprompt_text()
                if preprompt_text:
                    preprompt_tokens = token_counter(model=model, text=preprompt_text)
                    base_count += preprompt_tokens
            except ImportError:
                # Provider not available, skip preprompt token counting
                lib_logger.debug(
                    "Provider not available, skip preprompt token counting"
                )

        return base_count

    async def get_available_models(self, provider: str) -> List[str]:
        """Returns a list of available models for a specific provider, with caching."""
        lib_logger.info("Getting available models for provider: %s", provider)
        provider_instance = self._get_provider_instance(provider)

        def get_static_fallback() -> List[str]:
            static_fallback = []
            try:
                static_fallback = self.model_definitions.get_all_provider_models(provider)
            except (OSError, IOError, ValueError) as e:
                lib_logger.error(
                    "Failed to get static models for provider %s: %s: %s",
                    provider,
                    type(e).__name__,
                    e,
                )
                static_fallback = []
            if (
                not static_fallback
                and provider_instance
                and hasattr(provider_instance, "get_static_models")
            ):
                try:
                    static_fallback = provider_instance.get_static_models()
                except (OSError, IOError, ValueError, TypeError) as e:
                    lib_logger.error(
                        "Failed to get provider static models for %s: %s: %s",
                        provider,
                        type(e).__name__,
                        e,
                    )
                    static_fallback = []
            return static_fallback

        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            if provider in self._model_list_cache:
                cached_models, cached_at = self._model_list_cache[provider]
                if time.monotonic() - cached_at < self._MODEL_LIST_CACHE_TTL:
                    lib_logger.debug("Returning cached models for provider: %s", provider)
                    return cached_models

            credentials_for_provider = self.all_credentials.get(provider)
            if not credentials_for_provider:
                lib_logger.warning("No credentials for provider: %s", provider)
                static_fallback = get_static_fallback()
                lib_logger.warning(
                    "No credentials for provider %s; using %d static models",
                    provider,
                    len(static_fallback),
                )
                return static_fallback

            shuffled_credentials = list(credentials_for_provider)
            offset = self._cred_offset.get(provider, 0)
            self._cred_offset[provider] = (offset + 1) % len(shuffled_credentials)
            shuffled_credentials = (
                shuffled_credentials[offset:] + shuffled_credentials[:offset]
            )

        if provider_instance:
            # For providers with hardcoded models (like gemini_cli), we only need to call once.
            # For others, we might need to try multiple keys if one is invalid.
            # The current logic of iterating works for both, as the credential is not
            # always used in get_models.
            consecutive_auth_errors = 0
            for credential in shuffled_credentials:
                try:
                    # Display last 6 chars for API keys, or the filename for OAuth paths
                    cred_display = mask_credential(credential)
                    lib_logger.debug(
                        f"Attempting to get models for {provider} with credential {cred_display}"
                    )
                    models = await provider_instance.get_models(
                        credential, await self._get_http_client_async(streaming=False)
                    )

                    consecutive_auth_errors = 0  # Reset on success

                    lib_logger.info(
                        f"Got {len(models)} models for provider: {provider}"
                    )

                    # Whitelist and blacklist logic
                    final_models = []
                    for m in models:
                        is_whitelisted = self._is_model_whitelisted(provider, m)
                        is_blacklisted = self._is_model_ignored(provider, m)

                        if is_whitelisted:
                            final_models.append(m)
                            continue

                        if not is_blacklisted:
                            final_models.append(m)

                    if len(final_models) != len(models):
                        lib_logger.info(
                            f"Filtered out {len(models) - len(final_models)} models for provider {provider}."
                        )

                    async with lock:
                        self._model_list_cache[provider] = (final_models, time.monotonic())
                    return final_models
                except Exception as e:
                    classified_error = classify_error(e, provider=provider)
                    cred_display = mask_credential(credential)
                    is_auth_error = classified_error.error_type in (
                        "authentication",
                        "forbidden",
                    )
                    if is_auth_error:
                        consecutive_auth_errors += 1
                        lib_logger.warning(
                            f"Auth error for {provider} with {cred_display}: {classified_error.error_type} ({consecutive_auth_errors} consecutive)"
                        )
                        if consecutive_auth_errors >= 2:
                            lib_logger.warning(
                                f"Stopping model discovery for {provider}: {consecutive_auth_errors} consecutive auth errors"
                            )
                            break
                    else:
                        lib_logger.debug(
                            f"Failed to get models for provider {provider} with credential {cred_display}: {classified_error.error_type}. Trying next credential."
                        )
                    continue  # Try the next credential

        # Discovery failure is a degradation (static models still usable),
        # not a hard failure; downgrade to warning and include static count.
        static_fallback = get_static_fallback()
        lib_logger.warning(
            "Failed to get models for provider %s after trying all credentials; "
            "provider unreachable, using %d static models",
            provider,
            len(static_fallback),
        )
        return static_fallback

    async def get_all_available_models(
        self, grouped: bool = True
    ) -> Union[Dict[str, List[str]], List[str]]:
        """Returns a list of all available models, either grouped by provider or as a flat list."""
        lib_logger.info("Getting all available models...")

        all_providers = list(self.all_credentials.keys())
        tasks = [self.get_available_models(provider) for provider in all_providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_provider_models = {}
        for provider, result in zip(all_providers, results):
            if isinstance(result, Exception):
                lib_logger.error(
                    f"Failed to get models for provider {provider}: {result}"
                )
                all_provider_models[provider] = []
            else:
                all_provider_models[provider] = result

        lib_logger.info("Finished getting all available models.")
        if grouped:
            return all_provider_models
        else:
            flat_models = []
            for models in all_provider_models.values():
                flat_models.extend(models)
            return flat_models
