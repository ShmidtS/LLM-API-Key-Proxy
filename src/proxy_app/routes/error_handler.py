# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import functools
import logging
from typing import Callable

import json
from fastapi import HTTPException
from pydantic import ValidationError

from proxy_app.dependencies import make_error_response
from proxy_app.streaming import handle_litellm_error, LITELLM_ERROR_MAP

logger = logging.getLogger(__name__)

LITELLM_ERROR_TYPES = tuple(
    exc_type for row in LITELLM_ERROR_MAP for exc_type in row[0]
)


def handle_route_errors(
    error_format: str = "openai",
    log_context: str = "",
) -> Callable:
    """Decorator that wraps async route handlers with consistent error handling.

    Catches exceptions raised by route handlers and returns consistent error
    responses, avoiding the need for repeated try/except blocks in every route.

    Args:
        error_format: Controls error response formatting.
            "openai"    - litellm errors via handle_litellm_error("openai"),
                         500 detail = make_error_response("Internal server error", "api_error")
            "anthropic" - litellm errors via handle_litellm_error("anthropic"),
                          500 detail = Anthropic-shaped error dict
            "simple"    - no litellm error handling,
                         500 detail = "Internal server error"
        log_context: Prefix for the error log message (e.g. "Quota stats").
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail=make_error_response("Invalid JSON in request body", "invalid_request_error"),
                )
            except ValidationError as e:
                if error_format == "anthropic":
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": str(e),
                            },
                        },
                    )
                raise HTTPException(
                    status_code=400,
                    detail=make_error_response(str(e), "invalid_request_error"),
                )
            except Exception as e:
                if error_format in ("openai", "anthropic"):
                    if isinstance(e, LITELLM_ERROR_TYPES):
                        logger.error(
                            "Route litellm error: context=%s type=%s detail=%s",
                            log_context,
                            type(e).__name__,
                            str(e),
                        )
                        raise handle_litellm_error(e, error_format=error_format)

                    detail = getattr(e, "args", [None])[0]
                    if isinstance(detail, dict) and "error" in detail:
                        status_code = 500
                        if error_format == "openai":
                            details_obj = detail.get("error", {}).get("details")
                            if isinstance(details_obj, dict) and details_obj.get("timeout"):
                                status_code = 504
                            elif detail.get("error", {}).get("type") == "invalid_request_error":
                                status_code = 400
                            raise HTTPException(status_code=status_code, detail=detail)

                        raise HTTPException(status_code=status_code, detail=detail)

                prefix = f"{log_context}: " if log_context else ""
                logging.error(f"{prefix}{e}", exc_info=True)

                if error_format == "anthropic":
                    raise HTTPException(
                        status_code=500,
                        detail={
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "Internal server error",
                            },
                        },
                    )
                elif error_format == "openai":
                    raise HTTPException(
                        status_code=500,
                        detail=make_error_response("Internal server error", "api_error"),
                    )
                else:
                    raise HTTPException(
                        status_code=500,
                        detail=make_error_response("Internal server error", "api_error"),
                    )

        return wrapper

    return decorator
