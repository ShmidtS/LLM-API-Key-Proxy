# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import asyncio
import logging
from typing import Optional

import litellm
from fastapi import APIRouter, Request, Depends, HTTPException

from rotator_library import RotatingClient
from proxy_app.models import EmbeddingRequest
from proxy_app.dependencies import get_rotating_client, get_embedding_batcher, verify_api_key, make_error_response
from proxy_app.batch_manager import EmbeddingBatcher
from proxy_app.routes._helpers import log_request_to_console

router = APIRouter(tags=["embeddings"])



@router.post("/v1/embeddings")
async def embeddings(
    request: Request,
    body: EmbeddingRequest,
    client: RotatingClient = Depends(get_rotating_client),
    batcher: Optional[EmbeddingBatcher] = Depends(get_embedding_batcher),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for creating embeddings.
    Supports two modes based on the USE_EMBEDDING_BATCHER flag:
    - True: Uses a server-side batcher for high throughput.
    - False: Passes requests directly to the provider.
    """
    try:
        # Serialize body once — reused for logging and both code paths below
        request_data = body.model_dump(exclude_none=True)
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        if getattr(request.app.state, "use_embedding_batcher", False) and batcher:
            # --- Server-Side Batching Logic ---
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
            response = litellm.EmbeddingResponse(**final_response_data)

        else:
            # --- Direct Pass-Through Logic ---
            if isinstance(request_data.get("input"), str):
                request_data["input"] = [request_data["input"]]

            response = await client.aembedding(request=request, **request_data)

        return response

    except HTTPException:
        # Re-raise HTTPException to ensure it's not caught by the generic Exception handler
        raise
    except (
        litellm.InvalidRequestError,
        ValueError,
        litellm.ContextWindowExceededError,
    ):
        raise HTTPException(status_code=400, detail=make_error_response("Invalid Request", "invalid_request_error"))
    except litellm.AuthenticationError:
        raise HTTPException(status_code=401, detail=make_error_response("Authentication Error", "authentication_error"))
    except litellm.RateLimitError:
        raise HTTPException(status_code=429, detail=make_error_response("Rate Limit Exceeded", "rate_limit_error"))
    except (litellm.ServiceUnavailableError, litellm.APIConnectionError):
        raise HTTPException(status_code=503, detail=make_error_response("Service Unavailable", "server_error"))
    except litellm.Timeout:
        raise HTTPException(status_code=504, detail=make_error_response("Gateway Timeout", "timeout_error"))
    except (litellm.InternalServerError, litellm.OpenAIError):
        raise HTTPException(status_code=502, detail=make_error_response("Bad Gateway", "server_error"))
    except Exception as e:
        logging.error(f"Embedding request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))
