# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

"""Request context builder for retry paths."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..retry_base import _RetryContext
from ...config.defaults import TRACE
from ...error_accumulator import RequestErrorAccumulator
from ...utils.model_utils import (
    extract_provider_from_model,
    normalize_model_string,
)

lib_logger = logging.getLogger("rotator_library")


class RetryContextBuilderMixin:
    """Request context builder for retry paths."""

    async def _prepare_request_context(
        self,
        model: str,
        provider: str,
        credentials_for_provider: list,
        provider_plugin: Any,
    ) -> dict:
        """
        Prepare common request context shared by streaming and non-streaming retry paths.

        Handles:
        - Model ID resolution
        - Credential tier filtering (keep compatible/unknown, filter incompatible)
        - Building credential priority cache
        - Initializing RequestErrorAccumulator
        - Circuit breaker check

        Returns:
            dict with keys:
                - model: resolved model string
                - credentials: filtered credential list
                - credential_priorities: priority map (may be None)
                - credential_tier_names: tier names map
                - error_accumulator: RequestErrorAccumulator instance
        """
        # Resolve model ID early, before any credential operations
        # This ensures consistent model ID usage for acquisition, release, and tracking
        resolved_model = await self._resolve_model_id(model, provider)
        if resolved_model != model:
            lib_logger.info("Resolved model '%s' to '%s'", model, resolved_model)
            model = resolved_model

        # [NEW] Filter by model tier requirement and build priority map
        credential_priorities = None
        if provider_plugin and hasattr(provider_plugin, "get_model_tier_requirement"):
            required_tier = provider_plugin.get_model_tier_requirement(model)
            if required_tier is not None:
                # Filter OUT only credentials we KNOW are too low priority
                # Keep credentials with unknown priority (None) - they might be high priority
                incompatible_creds = []
                compatible_creds = []
                unknown_creds = []

                has_priority = hasattr(provider_plugin, "get_credential_priority")
                if has_priority:
                    get_priority = provider_plugin.get_credential_priority
                for cred in credentials_for_provider:
                    if has_priority:
                        priority = get_priority(cred)
                        if priority is None:
                            # Unknown priority - keep it, will be discovered on first use
                            unknown_creds.append(cred)
                        elif priority <= required_tier:
                            # Known compatible priority
                            compatible_creds.append(cred)
                        else:
                            # Known incompatible priority (too low)
                            incompatible_creds.append(cred)
                    else:
                        # Provider doesn't support priorities - keep all
                        unknown_creds.append(cred)

                # If we have any known-compatible or unknown credentials, use them
                tier_compatible_creds = compatible_creds + unknown_creds
                if tier_compatible_creds:
                    credentials_for_provider = tier_compatible_creds
                    if compatible_creds and unknown_creds:
                        lib_logger.info(
                            "Model %s requires priority <= %s. "
                            "Using %s known-compatible + %s unknown-tier credentials.",
                            model, required_tier,
                            len(compatible_creds), len(unknown_creds),
                        )
                    elif compatible_creds:
                        lib_logger.info(
                            "Model %s requires priority <= %s. "
                            "Using %s known-compatible credentials.",
                            model, required_tier, len(compatible_creds),
                        )
                    else:
                        lib_logger.info(
                            "Model %s requires priority <= %s. "
                            "Using %s unknown-tier credentials (will discover on use).",
                            model, required_tier, len(unknown_creds),
                        )
                elif incompatible_creds:
                    # Only known-incompatible credentials remain
                    lib_logger.warning(
                        "Model %s requires priority <= %s credentials, "
                        "but all %s known credentials have priority > %s. "
                        "Request will likely fail.",
                        model, required_tier,
                        len(incompatible_creds), required_tier,
                    )

        # Build priority map and tier names map for usage_manager (using cache)
        credential_priorities, credential_tier_names = (
            await self._build_credential_priority_cache(provider, credentials_for_provider)
        )

        if credential_priorities:
            lib_logger.log(
                TRACE,
                "Credential priorities for %s: %s",
                provider,
                ', '.join(f'P{p}={len([c for c in credentials_for_provider if credential_priorities.get(c) == p])}' for p in sorted(set(credential_priorities.values()))),
            )

        # Initialize error accumulator for tracking errors across credential rotation
        error_accumulator = RequestErrorAccumulator()
        error_accumulator.model = model
        error_accumulator.provider = provider

        # Circuit breaker check moved to retry loop - allows rotation to other keys
        # when circuit is OPEN instead of failing immediately on all provider requests

        return {
            "model": model,
            "credentials": credentials_for_provider,
            "credential_priorities": credential_priorities,
            "credential_tier_names": credential_tier_names,
            "error_accumulator": error_accumulator,
        }

    async def _prepare_retry_context(self, **kwargs) -> _RetryContext:
        model = normalize_model_string(kwargs.get("model"))
        if not model:
            raise ValueError("'model' is a required parameter.")
        kwargs["model"] = model

        provider = extract_provider_from_model(model)
        if not provider:
            raise ValueError("'model' must be in 'provider/model' format.")
        if provider not in self.all_credentials:
            raise ValueError(
                f"No API keys or OAuth credentials configured for provider: {provider}"
            )

        parent_log_dir = kwargs.pop("_parent_log_dir", None)
        _override_timeout = kwargs.pop("_global_timeout", None)
        deadline = time.monotonic() + (_override_timeout or self.global_timeout)

        transaction_logger = None
        if self.enable_request_logging:
            from ...transaction_logger import TransactionLogger

            transaction_logger = TransactionLogger(
                provider,
                model,
                enabled=True,
                api_format="oai",
                parent_dir=parent_log_dir,
            )
            await transaction_logger.log_request(kwargs)

        credentials_for_provider = list(self.all_credentials[provider])
        lock = await self._lock_manager.get_lock(provider)
        async with lock:
            offset = self._cred_offset.get(provider, 0)
            self._cred_offset[provider] = (offset + 1) % len(credentials_for_provider)
        credentials_for_provider = (
            credentials_for_provider[offset:] + credentials_for_provider[:offset]
        )

        provider_plugin = self._get_provider_instance(provider)

        ctx = await self._prepare_request_context(
            model, provider, credentials_for_provider, provider_plugin
        )

        return _RetryContext(
            model=ctx["model"],
            provider=provider,
            credentials_for_provider=ctx["credentials"],
            provider_plugin=provider_plugin,
            deadline=deadline,
            transaction_logger=transaction_logger,
            tried_creds=set(),
            last_exception=None,
            parent_log_dir=parent_log_dir,
            credential_priorities=ctx["credential_priorities"],
            credential_tier_names=ctx["credential_tier_names"],
            error_accumulator=ctx["error_accumulator"],
        )
