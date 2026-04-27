# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

from typing import Any

import litellm
from fastapi import APIRouter, Depends, UploadFile, File, Form

from proxy_app.dependencies import verify_api_key
from proxy_app.routes.error_handler import handle_route_errors

router = APIRouter(tags=["files"])


@router.post("/v1/files")
@handle_route_errors(error_format="openai", log_context="File upload failed")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form(...),
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for uploading files.

    Used for batch processing and fine-tuning. Accepts multipart form data
    with a file and purpose parameter.
    """
    file_tuple = (file.filename or "upload.jsonl", file.file)
    response = await litellm.acreate_file(file=file_tuple, purpose=purpose)
    return response


@router.get("/v1/files")
@handle_route_errors(error_format="openai", log_context="List files failed")
async def list_files(
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for listing uploaded files.
    """
    response = await litellm.afile_list()
    return response


@router.get("/v1/files/{file_id}")
@handle_route_errors(error_format="openai", log_context="Retrieve file failed")
async def retrieve_file(
    file_id: str,
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for retrieving file metadata.
    """
    response = await litellm.afile_retrieve(file_id=file_id)
    return response


@router.delete("/v1/files/{file_id}")
@handle_route_errors(error_format="openai", log_context="Delete file failed")
async def delete_file(
    file_id: str,
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for deleting an uploaded file.
    """
    response = await litellm.afile_delete(file_id=file_id)
    return response


@router.get("/v1/files/{file_id}/content")
@handle_route_errors(error_format="openai", log_context="Retrieve file content failed")
async def retrieve_file_content(
    file_id: str,
    _=Depends(verify_api_key),
) -> Any:
    """
    OpenAI-compatible endpoint for retrieving file content.
    """
    response = await litellm.afile_content(file_id=file_id)
    return response
