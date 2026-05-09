# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from proxy_app.dependencies import make_error_response


def internal_server_error_payload(format: str) -> dict:
    if format == "anthropic":
        return {
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "Internal server error",
            },
        }
    if format == "log":
        return {"error": "Internal server error"}
    return make_error_response("Internal server error", "api_error")
