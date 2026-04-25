# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import asyncio
import os
from typing import Any, Dict

import httpx

from ._simple_model_base import SimpleModelProvider


class QwenProvider(SimpleModelProvider):
    provider_name = "qwen"
    _models_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/models"
    _provider_prefix = "qwen"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_base = os.environ.get(
            "QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")

    def _resolve_models_url(self) -> str:
        base = self.api_base
        if base.endswith("/models"):
            return base
        return f"{base}/models"

    async def native_image_generation(
        self, client: httpx.AsyncClient, api_key: str, **kwargs
    ) -> Dict[str, Any]:
        model = str(kwargs.get("model", ""))
        model_name = model.split("/", 1)[1] if "/" in model else model
        if model_name.startswith("openai/"):
            model_name = model_name.split("/", 1)[1]
        if model_name.startswith("z-image"):
            return await self._z_image_generation(client, api_key, model_name, **kwargs)
        return await self._dashscope_image_synthesis(client, api_key, model_name, **kwargs)

    async def _z_image_generation(
        self, client: httpx.AsyncClient, api_key: str, model_name: str, **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "model": model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": str(kwargs.get("prompt", ""))[:800]}],
                    }
                ]
            },
            "parameters": {"prompt_extend": False},
        }
        size = kwargs.get("size")
        if isinstance(size, str) and size.lower() != "auto":
            payload["parameters"]["size"] = size.replace("x", "*")
        elif size and not isinstance(size, str):
            payload["parameters"]["size"] = str(size).replace("x", "*")
        response = await client.post(
            f"{self._dashscope_api_base(kwargs.get('api_base'))}/services/aigc/multimodal-generation/generation",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=kwargs.get("timeout"),
        )
        response.raise_for_status()
        return response.json()

    async def _dashscope_image_synthesis(
        self, client: httpx.AsyncClient, api_key: str, model_name: str, **kwargs
    ) -> Dict[str, Any]:
        payload = {
            "model": model_name,
            "input": {"prompt": kwargs.get("prompt", "")},
            "parameters": {"size": kwargs.get("size", "1024x1024"), "n": kwargs.get("n", 1)},
        }
        qwen_base = self._dashscope_api_base(kwargs.get("api_base"))
        response = await client.post(
            f"{qwen_base}/services/aigc/text2image/image-synthesis",
            headers={"Authorization": f"Bearer {api_key}", "X-DashScope-Async": "enable"},
            json=payload,
            timeout=kwargs.get("timeout"),
        )
        response.raise_for_status()
        data = response.json()
        task_id = data.get("output", {}).get("task_id") or data.get("task_id")
        if task_id:
            data = await self._poll_result(client, api_key, qwen_base, task_id, kwargs.get("timeout"))
        return data

    def _dashscope_api_base(self, api_base: Any = None) -> str:
        base = str(api_base or self.api_base or "").rstrip("/")
        if not base:
            return "https://dashscope.aliyuncs.com/api/v1"
        if base.endswith("/api/v1"):
            return base
        for suffix in ("/compatible-mode/v1", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}/api/v1"

    async def _poll_result(
        self, client: httpx.AsyncClient, api_key: str, qwen_base: str, task_id: str, timeout: Any
    ) -> Dict[str, Any]:
        url = f"{qwen_base}/tasks/{task_id}"
        deadline = asyncio.get_running_loop().time() + float(timeout or 120)
        while asyncio.get_running_loop().time() < deadline:
            per_request_timeout = min(10.0, max(1.0, deadline - asyncio.get_running_loop().time()))
            response = await client.get(
                url, headers={"Authorization": f"Bearer {api_key}"}, timeout=per_request_timeout
            )
            response.raise_for_status()
            data = response.json()
            status = str(data.get("output", {}).get("task_status", data.get("task_status", ""))).upper()
            if status in {"SUCCEEDED", "SUCCESS", "COMPLETED"}:
                return data
            if status in {"FAILED", "CANCELED", "UNKNOWN"}:
                raise httpx.HTTPStatusError(f"DashScope image task failed: {data}", request=response.request, response=response)
            await asyncio.sleep(1)
        raise TimeoutError("DashScope image task timed out")
