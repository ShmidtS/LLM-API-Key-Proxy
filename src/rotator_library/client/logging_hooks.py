# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS

import logging

from ..utils.json_utils import json_deep_copy

lib_logger = logging.getLogger("rotator_library")


class StreamedAPIError(Exception):
    """Custom exception to signal an API error received over a stream."""

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data


def _sanitize_litellm_log(self, log_data: dict) -> dict:
    """
    Recursively removes large data fields and sensitive information from litellm log
    dictionaries to keep debug logs clean and secure.
    """
    if not isinstance(log_data, dict):
        return log_data

    # Keys to remove at any level of the dictionary
    keys_to_pop = [
        "messages",
        "input",
        "response",
        "data",
        "api_key",
        "api_base",
        "original_response",
        "additional_args",
    ]

    # Keys that might contain nested dictionaries to clean
    nested_keys = ["kwargs", "litellm_params", "model_info", "proxy_server_request"]

    # Create a deep copy to avoid modifying the original log object in memory
    clean_data = json_deep_copy(log_data)

    def clean_recursively(data_dict):
        if not isinstance(data_dict, dict):
            return

        # Remove sensitive/large keys
        for key in keys_to_pop:
            data_dict.pop(key, None)

        # Recursively clean nested dictionaries
        for key in nested_keys:
            if key in data_dict and isinstance(data_dict[key], dict):
                clean_recursively(data_dict[key])

        # Also iterate through all values to find any other nested dicts
        for key, value in list(data_dict.items()):
            if isinstance(value, dict):
                clean_recursively(value)

    clean_recursively(clean_data)
    return clean_data


def _litellm_logger_callback(self, log_data: dict):
    """
    Callback function to redirect litellm's logs to the library's logger.
    This allows us to control the log level and destination of litellm's output.
    It also cleans up error logs for better readability in debug files.
    """
    # Filter out verbose pre_api_call and post_api_call logs
    log_event_type = log_data.get("log_event_type")
    if log_event_type in ["pre_api_call", "post_api_call"]:
        return  # Skip these verbose logs entirely

    # For successful calls or pre-call logs, a simple debug message is enough.
    if not log_data.get("exception"):
        sanitized_log = self._sanitize_litellm_log(log_data)
        # We log it at the DEBUG level to ensure it goes to the debug file
        # and not the console, based on the main.py configuration.
        lib_logger.debug(f"LiteLLM Log: {sanitized_log}")
        return

    # For failures, extract key info to make debug logs more readable.
    model = log_data.get("model", "N/A")
    call_id = log_data.get("litellm_call_id", "N/A")
    error_info = log_data.get("standard_logging_object", {}).get(
        "error_information", {}
    )
    error_class = error_info.get("error_class", "UnknownError")
    error_message = error_info.get(
        "error_message", str(log_data.get("exception", ""))
    )
    error_message = " ".join(error_message.split())  # Sanitize

    lib_logger.debug(
        f"LiteLLM Callback Handled Error: Model={model} | "
        f"Type={error_class} | Message='{error_message}'"
    )
