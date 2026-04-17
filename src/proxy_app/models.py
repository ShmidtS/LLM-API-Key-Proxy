# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from typing import List, Optional, Union

from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    model: str
    input: Union[str, List[str]]
    input_type: Optional[str] = None
    dimensions: Optional[int] = None
    user: Optional[str] = None
