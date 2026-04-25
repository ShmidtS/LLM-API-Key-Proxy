# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

from ._simple_model_base import SimpleModelProvider


class XaiProvider(SimpleModelProvider):
    _models_url = "https://api.x.ai/v1/models"
    _provider_prefix = "xai"
