# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 ShmidtS


class StreamedAPIError(Exception):
    """Custom exception to signal an API error received over a stream."""

    def __init__(self, message, data=None):
        super().__init__(message)
        self.data = data
