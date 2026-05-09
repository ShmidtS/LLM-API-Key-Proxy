# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._constants import lib_logger
import asyncio
import logging
import time
from typing import Dict, List, Optional
from ..error_types import NoAvailableKeysError, mask_credential
from ..utils.model_utils import extract_provider_from_model


class UsageManagerAcquireMixin:
    from ._acquire_strategies import acquire_key


    async def release_key(self, key: str, model: str):
        """Releases a key's lock for a specific model and notifies waiting tasks."""
        if key not in self.key_states:
            return

        state = self.key_states[key]
        async with state["lock"]:
            if model in state["models_in_use"]:
                state["models_in_use"][model] -= 1
                state["total_in_use"] = max(0, state.get("total_in_use", 1) - 1)
                remaining = state["models_in_use"][model]
                if remaining <= 0:
                    del state["models_in_use"][model]  # Clean up when count reaches 0
                lib_logger.info(
                    f"Released credential {mask_credential(key)} from model {model} "
                    f"(remaining concurrent: {max(0, remaining)})"
                )
            else:
                lib_logger.warning(
                    f"Attempted to release credential {mask_credential(key)} for model {model}, but it was not in use."
                )

        # Notify all tasks waiting on this key's condition
        async with state["condition"]:
            state["condition"].notify_all()
