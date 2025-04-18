import os
import hashlib
import uuid

from datetime import datetime
from pathlib import Path
from typing import Optional, List
from urllib.parse import unquote

import aiofiles

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

from pydantic import BaseModel, Field

from core.globals import FILES_DIR
from core.logger import exception
from core.repositories.repo_files import FilesRepository, FileItem
from core.routers.schemas import error_constructor, ErrorResponse


class DeleteFilePost(BaseModel):
    file_name: str = Field(..., description="Name of the file to delete")


class FilesListPost(BaseModel):
    user_id: Optional[int] = Field(None, description="User ID to filter files by (optional)")


class FileResponse(BaseModel):
    file_name: str = Field(..., description="Hashed filename stored in the system")
    file_name_orig: str = Field(..., description="Original filename provided by the user")
    user_id: int = Field(..., description="ID of the user whose file it is")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")


class FilesListResponse(BaseModel):
    files: List[FileResponse]


class FileUploadResponse(BaseModel):
    status: str
    file_name: str
    stored_as: str


class FilesRouter(APIRouter):
    def __init__(
            self,
            files_repository: FilesRepository,
            *args, **kwargs
    ):
        # Define tags for this router
        kwargs["tags"] = ["Files"]

        super().__init__(*args, **kwargs)
        self._files_repository = files_repository

        self.add_api_route(
            "/v1/file-create",
            self._create_file,
            methods=["POST"],
            status_code=status.HTTP_201_CREATED,
            responses={
                201: {
                    "description": "File record created successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "added record of file in DB"}
                        }
                    }
                },
                400: {
                    "description": "Failed to add record of file in database",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "Failed to add record of file in database",
                                    "type": "files_error",
                                    "code": "file_record_creation_failed"
                                }
                            }
                        }
                    }
                },
                404: {
                    "description": "File not found in the file system",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "File not found in the file system",
                                    "type": "files_error",
                                    "code": "file_not_found"
                                }
                            }
                        }
                    }
                }
            },
            summary="Create File Record",
            description="Create a database record for a file that already exists in the file system."
        )

        self.add_api_route(
            "/v1/file-delete",
            self._delete_file,
            methods=["POST"],
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "File record deleted successfully",
                    "content": {
                        "application/json": {
                            "example": {"message": "record of file deleted"}
                        }
                    }
                },
                404: {
                    "description": "Failed to delete record of file",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "Failed to delete record of file",
                                    "type": "files_error",
                                    "code": "file_record_deletion_failed"
                                }
                            }
                        }
                    }
                }
            },
            summary="Delete File Record",
            description="Delete a file record from the database (does not delete the actual file)."
        )

        self.add_api_route(
            "/v1/files-list",
            self._list_files,
            methods=["POST"],
            response_model=FilesListResponse,
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "List of files retrieved successfully",
                    "model": FilesListResponse
                },
                500: {
                    "description": "Failed to retrieve files",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "An error occurred while retrieving files",
                                    "type": "files_error",
                                    "code": "files_retrieval_failed"
                                }
                            }
                        }
                    }
                }
            },
            summary="List Files",
            description="Retrieve a list of files from the database, optionally filtered by user ID."
        )

        self.add_api_route(
            "/v1/file-upload",
            self._file_upload,
            methods=["POST"],
            response_model=FileUploadResponse,
            status_code=status.HTTP_200_OK,
            responses={
                200: {
                    "description": "File uploaded successfully",
                    "model": FileUploadResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "status": "success",
                                "file_name": "example_document.pdf",
                                "stored_as": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6.pdf"
                            }
                        }
                    }
                },
                400: {
                    "description": "Invalid request parameters",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "X-File-Name header is either missing or empty",
                                    "type": "files_error",
                                    "code": "invalid_request"
                                }
                            }
                        }
                    }
                },
                500: {
                    "description": "Server error during file upload",
                    "model": ErrorResponse,
                    "content": {
                        "application/json": {
                            "example": {
                                "error": {
                                    "message": "Failed to save file information to database",
                                    "type": "files_error",
                                    "code": "file_upload_failed"
                                }
                            }
                        }
                    }
                }
            },
            summary="Upload File",
            description="Upload a file to the server with a hashed name and create a database record for it."
        )

    async def _create_file(self, file: FileItem):
        """
        Create a record for an existing file in the database.

        This endpoint creates a database record for a file that already exists in the file system.

        Parameters:
        - **file**: FileItem object containing file information

        Returns:
            Response with status code 201 if successful

        Raises:
        ```
        - 400: If the file record could not be added to the database
        - 404: If the file does not exist in the file system
        ```
        """
        if file.processing_status != "":
            return error_constructor(
                message="processing_status field is not allowed during file creation",
                error_type="files_error",
                code="invalid_request_parameters",
                status_code=400
            )

        if file.vector_store_id != "":
            return error_constructor(
                message="vector_store_id field is not allowed during file creation",
                error_type="files_error",
                code="invalid_request_parameters",
                status_code=400
            )

        if not FILES_DIR.joinpath(file.file_name).is_file():
            return error_constructor(
                message="File not found in the file system",
                error_type="files_error",
                code="file_not_found",
                status_code=404
            )

        resp = await self._files_repository.create_file(file)
        if resp:
            return JSONResponse(status_code=201, content={"message": "added record of file in DB"})
        else:
            return error_constructor(
                message="Failed to add record of file in database",
                error_type="files_error",
                code="file_record_creation_failed",
                status_code=400
            )

    async def _delete_file(self, post: DeleteFilePost):
        """
        Delete a file record from the database.

        This endpoint removes a file record from the database but does not delete the actual file.

        Parameters:
        - **post**: DeleteFilePost object containing the file name to delete

        Returns:
            Response with status code 200 if successful

        Raises:
        ```
        - 404: If the file record could not be deleted or was not found
        ```
        """
        resp = await self._files_repository.delete_file(post.file_name)
        if resp:
            content = {"message": "record of file deleted"}
            return JSONResponse(status_code=200, content=content)
        else:
            return error_constructor(
                message="Failed to delete record of file",
                error_type="files_error",
                code="file_record_deletion_failed",
                status_code=404
            )

    async def _list_files(self, post: FilesListPost):
        """
        List files from the database.

        This endpoint retrieves a list of files from the database, optionally filtered by user ID.

        Parameters:
        - **post**: FilesListPost object containing optional user_id filter

        Returns:
            FilesListResponse containing the list of files

        Raises:
        ```
        - 500: If an error occurs while retrieving files
        ```
        """
        try:
            if post.user_id is not None:
                files = await self._files_repository.get_files_by_filter("user_id = ?", (post.user_id,))
            else:
                files = await self._files_repository.get_all_files()
            return FilesListResponse(
                files=[FileResponse(**f.model_dump()) for f in files]
            )
        except Exception as e:
            return error_constructor(
                message=f"An error occurred while retrieving files: {e}",
                error_type="files_error",
                code="files_retrieval_failed",
                status_code=500
            )

    async def _file_upload(self, request: Request):
        """
        Upload a file to the server.

        This endpoint handles file uploads, storing the file with a hashed name and creating
        a database record for it.

        Headers:
        - **X-File-Name**: Original name of the file being uploaded (required)
        - **X-User-Id**: ID of the user uploading the file (required)

        Body:
        - Binary file content

        Returns:
            FileUploadResponse with upload status and file information

        Raises:
        ```
        - 400: If required headers are missing or invalid
        - 500: If an error occurs during file upload or database operation
        ```
        """
        file_name = request.headers.get('X-File-Name')
        user_id = request.headers.get('X-User-Id')

        if user_id is None:
            return error_constructor(
                message="X-User-Id header is either missing or empty",
                error_type="files_error",
                code="invalid_request",
                status_code=400
            )

        try:
            user_id = int(user_id)
        except Exception as e:
            return error_constructor(
                message=f"X-User-Id header should be convertable to int type: {str(e)}",
                error_type="files_error",
                code="invalid_request",
                status_code=400
            )

        if not file_name:
            return error_constructor(
                message="X-File-Name header is either missing or empty",
                error_type="files_error",
                code="invalid_request",
                status_code=400
            )

        file_name = unquote(file_name)

        # Create a unique hash based on the original filename and a random UUID
        random_hash = hashlib.sha256(f"{file_name}{uuid.uuid4()}".encode()).hexdigest()  # type: ignore

        # Keep the original file extension if it exists
        original_extension = Path(file_name).suffix
        hashed_filename = f"{random_hash}{original_extension}"

        file_path = Path(FILES_DIR) / hashed_filename

        temp_file_path = file_path.with_suffix(file_path.suffix + '.tmp')

        if temp_file_path.exists():
            os.remove(temp_file_path)

        temp_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(temp_file_path, 'wb') as f:
                async for chunk in request.stream():
                    await f.write(chunk)

            os.rename(temp_file_path, file_path)

            file_item = FileItem(
                file_name=hashed_filename,
                file_name_orig=file_name,
                user_id=user_id,
                created_at=datetime.now(),
            )

            if not await self._files_repository.create_file(file_item):
                raise Exception("Failed to save file information to database")

            return FileUploadResponse(
                status="success",
                file_name=file_name,
                stored_as=hashed_filename
            )

        except Exception as e:
            exception("exception")
            if temp_file_path.exists():
                os.remove(temp_file_path)

            return error_constructor(
                message=str(e),
                error_type="files_error",
                code="file_upload_failed",
                status_code=500
            )
