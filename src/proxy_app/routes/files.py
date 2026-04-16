# SPDX-License-Identifier: MIT
# Copyright (c) 2026 ShmidtS

import logging

import litellm
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from proxy_app.dependencies import verify_api_key, make_error_response
from proxy_app.streaming import handle_litellm_error

router = APIRouter(tags=["files"])


@router.post("/v1/files")
async def upload_file(
    file: UploadFile = File(...),
    purpose: str = Form(...),
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for uploading files.

    Used for batch processing and fine-tuning. Accepts multipart form data
    with a file and purpose parameter.
    """
    try:
        file_bytes = await file.read()
        file_tuple = (file.filename or "upload.jsonl", file_bytes)
        response = await litellm.acreate_file(file=file_tuple, purpose=purpose)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"File upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.get("/v1/files")
async def list_files(
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for listing uploaded files.
    """
    try:
        response = await litellm.afile_list()
        return response

    except Exception as e:
        if isinstance(e, (litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"List files failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.get("/v1/files/{file_id}")
async def retrieve_file(
    file_id: str,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for retrieving file metadata.
    """
    try:
        response = await litellm.afile_retrieve(file_id=file_id)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Retrieve file failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.delete("/v1/files/{file_id}")
async def delete_file(
    file_id: str,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for deleting an uploaded file.
    """
    try:
        response = await litellm.afile_delete(file_id=file_id)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Delete file failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))


@router.get("/v1/files/{file_id}/content")
async def retrieve_file_content(
    file_id: str,
    _=Depends(verify_api_key),
):
    """
    OpenAI-compatible endpoint for retrieving file content.
    """
    try:
        response = await litellm.afile_content(file_id=file_id)
        return response

    except Exception as e:
        if isinstance(e, (litellm.InvalidRequestError, ValueError,
                          litellm.AuthenticationError, litellm.RateLimitError,
                          litellm.ServiceUnavailableError, litellm.APIConnectionError,
                          litellm.Timeout, litellm.InternalServerError, litellm.OpenAIError)):
            raise handle_litellm_error(e, error_format="openai")
        logging.error(f"Retrieve file content failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=make_error_response("Internal server error", "api_error"))
