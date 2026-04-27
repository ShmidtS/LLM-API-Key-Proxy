# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
from typing import Optional, Any

import litellm
from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.models import EmbeddingRequest
from proxy_app.dependencies import get_rotating_client, get_embedding_batcher, verify_api_key
from proxy_app.batch_manager import EmbeddingBatcher
from proxy_app.routes._helpers import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["embeddings"])



@router.post("/v1/embeddings")
@handle_route_errors(error_format="openai", log_context="Embedding request failed")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    client: RotatingClient = Depends(get_rotating_client),
    batcher: Optional[EmbeddingBatcher] = Depends(get_embedding_batcher),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for creating embeddings.
    Supports two modes based on the USE_EMBEDDING_BATCHER flag:
    - True: Uses a server-side batcher for high throughput.
    - False: Passes requests directly to the provider.
    """
    request_data = body.model_dump(exclude_none=True)
    log_request_to_console(
        url=str(request.url),
        client_info=(request.client.host, request.client.port),
        request_data=request_data,
    )
    if getattr(request.app.state, "use_embedding_batcher", False) and batcher:
        inputs = request_data.get("input", [])
        if isinstance(inputs, str):
            inputs = [inputs]

        tasks = []
        for single_input in inputs:
            individual_request = request_data.copy()
            individual_request["input"] = single_input
            tasks.append(batcher.add_request(individual_request))

        results = await asyncio.gather(*tasks)

        all_data = []
        total_prompt_tokens = 0
        total_tokens = 0
        for i, result in enumerate(results):
            result["data"][0]["index"] = i
            all_data.extend(result["data"])
            total_prompt_tokens += result["usage"]["prompt_tokens"]
            total_tokens += result["usage"]["total_tokens"]

        final_response_data = {
            "object": "list",
            "model": results[0]["model"],
            "data": all_data,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "total_tokens": total_tokens,
            },
        }
        return litellm.EmbeddingResponse(**final_response_data)

    if isinstance(request_data.get("input"), str):
        request_data["input"] = [request_data["input"]]

    return await client.aembedding(request=request, **request_data)
