# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import orjson
from fastapi import APIRouter, Request, Depends

from rotator_library import RotatingClient
from proxy_app.dependencies import get_rotating_client, verify_api_key
from proxy_app.streaming import streaming_response_wrapper, make_sse_response
from proxy_app.detailed_logger import RawIOLogger
from proxy_app.request_logger import log_request_to_console
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["chat"])

ENABLE_REQUEST_LOGGING: bool = False


@router.post("/v1/chat/completions")
@handle_route_errors(error_format="openai", log_context="Request failed after all retries")
async def chat_completions(
    request: Request,
    client: RotatingClient = Depends(get_rotating_client),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint powered by the RotatingClient.
    Handles both streaming and non-streaming responses and logs them.
    """
    # Raw I/O logger captures unmodified HTTP data at proxy boundary (disabled by default)
    enable_raw_logging = getattr(request.app.state, "enable_raw_logging", False)
    raw_logger = RawIOLogger() if enable_raw_logging else None
    request_data: dict = {}
    try:
        # Read and parse the request body only once at the beginning.
        request_data = orjson.loads(await request.body())

        # Global temperature=0 override (controlled by .env variable, default: OFF)
        # Low temperature makes models deterministic and prone to following training data
        # instead of actual schemas, which can cause tool hallucination
        # Modes: "remove" = delete temperature key, "set" = change to 1.0, "false" = disabled
        override_temp_zero = getattr(request.app.state, "override_temp_zero", "false")

        if (
            override_temp_zero in ("remove", "set", "true", "1", "yes")
            and "temperature" in request_data
            and request_data["temperature"] == 0
        ):
            if override_temp_zero == "remove":
                # Remove temperature key entirely
                del request_data["temperature"]
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=remove: Removed temperature=0 from request"
                )
            else:
                # Set to 1.0 (for "set", "true", "1", "yes")
                request_data["temperature"] = 1.0
                logging.debug(
                    "OVERRIDE_TEMPERATURE_ZERO=set: Converting temperature=0 to temperature=1.0"
                )

        # If raw logging is enabled, capture the unmodified request data.
        if raw_logger:
            await raw_logger.log_request(headers=request.headers, body=request_data)

        # Extract and log specific reasoning parameters for monitoring.
        model = request_data.get("model")
        generation_cfg = (
            request_data.get("generationConfig", {})
            or request_data.get("generation_config", {})
            or {}
        )
        reasoning_effort = request_data.get("reasoning_effort") or generation_cfg.get(
            "reasoning_effort"
        )

        logging.getLogger("rotator_library").debug(
            f"Handling reasoning parameters: model={model}, reasoning_effort={reasoning_effort}"
        )

        # Log basic request info to console (this is a separate, simpler logger).
        log_request_to_console(
            url=str(request.url),
            headers=request.headers,
            client_info=(request.client.host, request.client.port),
            request_data=request_data,
        )
        is_streaming = request_data.get("stream", False)

        if is_streaming:
            response_generator = client.acompletion(request=request, **request_data)
            return make_sse_response(
                streaming_response_wrapper(request, response_generator, raw_logger)
            )
        else:
            response = await client.acompletion(request=request, **request_data)
            if raw_logger:
                # Assuming response has status_code and headers attributes
                # This might need adjustment based on the actual response object
                response_headers = (
                    response.headers if hasattr(response, "headers") else None
                )
                status_code = (
                    response.status_code if hasattr(response, "status_code") else 200
                )
                raw_logger.log_final_response(
                    status_code=status_code,
                    headers=response_headers,
                    body=response.model_dump(),
                )
            return response
    except Exception as e:
        # Raw I/O logger: log the failed response if request logging is enabled
        if ENABLE_REQUEST_LOGGING and raw_logger:
            raw_logger.log_final_response(
                status_code=500, headers=None, body={"error": str(e)}
            )
        raise
