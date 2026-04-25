# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import asyncio
import base64
import re
from typing import Any, Dict, List

import httpx

from ._simple_model_base import SimpleModelProvider


class FireworksProvider(SimpleModelProvider):
    provider_name = "fireworks"
    _models_url = "https://api.fireworks.ai/inference/v1/models"
    _provider_prefix = "fireworks"

    async def native_image_generation(
        self, client: httpx.AsyncClient, api_key: str, **kwargs
    ) -> Dict[str, Any]:
        model = str(kwargs.get("model", ""))
        model_name = model.split("/", 1)[1] if "/" in model else model
        model_path = self._image_model_path(model_name)
        if "flux-kontext" in model_name:
            url = f"https://api.fireworks.ai/inference/v1/workflows/{model_path}"
        else:
            url = f"https://api.fireworks.ai/inference/v1/workflows/{model_path}/text_to_image"
        payload = {"prompt": kwargs.get("prompt", "")}
        aspect_ratio = self._image_aspect_ratio(kwargs.get("size"))
        if aspect_ratio:
            payload["aspect_ratio"] = aspect_ratio
        if kwargs.get("quality"):
            payload["output_format"] = "png"
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Accept": "image/png"},
            json=payload,
            timeout=kwargs.get("timeout"),
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if content_type.startswith("image/"):
            return {"b64_json": base64.b64encode(response.content).decode("ascii")}
        data = response.json()
        if "request_id" in data:
            data = await self._poll_result(client, api_key, model_path, data["request_id"], kwargs.get("timeout"))
        return data

    def _image_model_path(self, model_name: str) -> str:
        if model_name.startswith("accounts/"):
            if not re.fullmatch(r"accounts/[A-Za-z0-9_.-]+/models/[A-Za-z0-9_.-]+", model_name):
                raise ValueError("Invalid Fireworks image model path")
            return model_name
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_name):
            raise ValueError("Invalid Fireworks image model name")
        return f"accounts/fireworks/models/{model_name}"

    async def _poll_result(
        self, client: httpx.AsyncClient, api_key: str, model_path: str, request_id: str, timeout: Any
    ) -> Dict[str, Any]:
        url = f"https://api.fireworks.ai/inference/v1/workflows/{model_path}/get_result"
        deadline = asyncio.get_running_loop().time() + float(timeout or 120)
        while asyncio.get_running_loop().time() < deadline:
            per_request_timeout = min(10.0, max(1.0, deadline - asyncio.get_running_loop().time()))
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"},
                json={"id": request_id},
                timeout=per_request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            status = str(data.get("status", "")).lower()
            if status in {"completed", "succeeded", "success", "done"} or data.get("result"):
                return data
            if status in {"failed", "error", "cancelled"}:
                raise httpx.HTTPStatusError(f"Fireworks workflow failed: {data}", request=response.request, response=response)
            await asyncio.sleep(1)
        raise TimeoutError("Fireworks image workflow timed out")

    def _image_aspect_ratio(self, size: Any) -> str | None:
        if not isinstance(size, str):
            return None
        return {
            "1024x1024": "1:1",
            "1024x1536": "2:3",
            "1536x1024": "3:2",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
        }.get(size.lower())
