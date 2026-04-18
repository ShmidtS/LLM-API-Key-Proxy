# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._simple_model_base import SimpleModelProvider


class KilocodeProvider(SimpleModelProvider):
    """
    Provider implementation for the Kilocode API.

    Kilocode routes requests to various providers through model prefixes:
    - minimax/minimax-m2.1:free
    - moonshotai/kimi-k2.5:free
    - z-ai/glm-4.7:free
    - And other provider/model combinations
    """

    _models_url = "https://kilo.ai/api/openrouter/models"
    _provider_prefix = "kilocode"

    _quota_error_patterns = [
        ("extract", "error.metadata.retry_after", "RATE_LIMIT_EXCEEDED"),
        ("json", "error.code", 429, 30, "RATE_LIMIT_EXCEEDED"),
        ("body", "upstream error", 5, "UPSTREAM_ERROR"),
        ("body", "provider error", 5, "UPSTREAM_ERROR"),
    ]
