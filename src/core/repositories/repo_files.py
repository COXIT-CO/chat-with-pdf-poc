import sqlite3
from datetime import datetime
from typing import List

from pydantic import BaseModel, Field

from core.repositories.repo_abstract import AbstractRepository


class FileItem(BaseModel):
    file_name: str = Field( # type: ignore
        ...,
        description="Hashed filename stored in the system",
        example="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6.pdf"
    )
    file_name_orig: str = Field( # type: ignore
        ...,
        description="Original filename provided by the user",
        example="important_document.pdf"
    )
    user_id: int = Field( # type: ignore
        ...,
        description="ID of the user who owns the file",
        example=12345
    )
    created_at: datetime = Field( # type: ignore
        default_factory=datetime.now,
        description="Timestamp when the file was created",
        example="2023-09-15T14:30:00"
    )
    processing_status: str = Field( # type: ignore
        default="",
        description="Current status of file processing (empty, 'processing', 'completed', 'error')",
        example="completed"
    )
    vector_store_id: str = Field( # type: ignore
        default="",
        description="ID of the associated vector store for search/retrieval",
        example="vs_123456789"
    )


class FilesRepository(AbstractRepository):
    """
    Repository for managing user files in a SQLite database.

    This repository provides methods to create, retrieve, update, and delete file records.
    It supports both synchronous and asynchronous operations, with async methods delegating
    to their sync counterparts through the AbstractRepository's thread pool.

    The repository stores file metadata including:
    - file_name: Unique identifier for the file (primary key)
    - file_name_orig: Original name of the file
    - user_id: ID of the user who owns the file
    - created_at: Timestamp when the file was created
    - processing_status: Current status of file processing
    - vector_store_id: ID of the associated vector store (for search/retrieval)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_db()

    def _init_db(self):
        with self._get_db_connection() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS user_files (
                file_name TEXT PRIMARY KEY,
                file_name_orig TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                processing_status TEXT DEFAULT '',
                vector_store_id TEXT DEFAULT ''
            )
            """)

            # Add vector_store_id column if it doesn't exist
            try:
                conn.execute("ALTER TABLE user_files ADD COLUMN vector_store_id TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass

            conn.commit()

    def create_file_sync(self, file: FileItem) -> bool:
        with self._get_db_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO user_files 
                    (file_name, file_name_orig, user_id, created_at, processing_status, vector_store_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file.file_name,
                        file.file_name_orig,
                        file.user_id,
                        file.created_at.isoformat(),
                        file.processing_status,
                        file.vector_store_id
                    )
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_files_by_filter_sync(self, filter: str, params: tuple = ()) -> List[FileItem]:
        """
        Get files based on a custom filter expression.

        Args:
            filter: SQL WHERE clause (without the 'WHERE' keyword)
            params: Parameters to be used with the filter expression

        Returns:
            List of FileItem objects matching the filter
        """
        with self._get_db_connection() as conn:
            query = f"""
            SELECT file_name, file_name_orig, user_id, created_at, processing_status, vector_store_id
            FROM user_files
            WHERE {filter}
            ORDER BY created_at DESC
            """
            cursor = conn.execute(query, params)

            files = []
            for row in cursor.fetchall():
                files.append(FileItem(
                    file_name=row[0],
                    file_name_orig=row[1],
                    user_id=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    processing_status=row[4],
                    vector_store_id=row[5]
                ))

            return files

    def delete_file_sync(self, file_name: str) -> bool:
        with self._get_db_connection() as conn:
            try:
                cursor = conn.execute(
                    "DELETE FROM user_files WHERE file_name = ?",
                    (file_name,)
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error:
                return False

    def get_all_files_sync(self) -> List[FileItem]:
        """
        Get all files in the repository.

        Returns:
            List of all FileItem objects ordered by creation date (newest first)
        """
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT file_name, file_name_orig, user_id, created_at, processing_status, vector_store_id
                FROM user_files
                ORDER BY created_at DESC
                """
            )

            files = []
            for row in cursor.fetchall():
                files.append(FileItem(
                    file_name=row[0],
                    file_name_orig=row[1],
                    user_id=row[2],
                    created_at=datetime.fromisoformat(row[3]),
                    processing_status=row[4],
                    vector_store_id=row[5]
                ))

            return files

    def update_file_sync(self, file_name: str, file_item: FileItem) -> bool:
        """
        Update file information in the repository.

        Args:
            file_name: The file name that serves as the primary key
            file_item: FileItem object containing the updated information

        Returns:
            True if the update was successful, False otherwise
        """
        with self._get_db_connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    UPDATE user_files 
                    SET file_name_orig = ?, user_id = ?, created_at = ?, processing_status = ?, vector_store_id = ?
                    WHERE file_name = ?
                    """,
                    (
                        file_item.file_name_orig,
                        file_item.user_id,
                        file_item.created_at.isoformat(),
                        file_item.processing_status,
                        file_item.vector_store_id,
                        file_name
                    )
                )
                conn.commit()
                return cursor.rowcount > 0
            except sqlite3.Error:
                return False

    def cleanup_missing_files_sync(self, existing_files: List[str]) -> int:
        """
        Remove database records for files that no longer exist on disk.

        Args:
            existing_files: List of file names that exist on disk

        Returns:
            Number of records removed
        """
        with self._get_db_connection() as conn:
            try:
                # Convert list to comma-separated string of quoted values
                if existing_files:
                    placeholders = ','.join(['?' for _ in existing_files])
                    query = f"""
                    DELETE FROM user_files 
                    WHERE file_name NOT IN ({placeholders})
                    """
                    cursor = conn.execute(query, existing_files)
                else:
                    # If no files exist, clear the entire table
                    cursor = conn.execute("DELETE FROM user_files")

                conn.commit()
                return cursor.rowcount
            except sqlite3.Error:
                return 0

    async def create_file(self, file: FileItem) -> bool:
        return await self._run_in_thread(self.create_file_sync, file)

    async def get_files_by_filter(self, filter: str, params: tuple = ()) -> List[FileItem]:
        """
        Async version of get_files_by_filter_sync.

        Args:
            filter: SQL WHERE clause (without the 'WHERE' keyword)
            params: Parameters to be used with the filter expression

        Returns:
            List of FileItem objects matching the filter

        Examples:
            Get all PDF files:
            files = await repo.get_files_by_filter("file_ext = ?", ("pdf",))

            Get files with a specific processing status for a user:
            files = await repo.get_files_by_filter("user_id = ? AND processing_status = ?", (user_id, "completed"))
        """
        return await self._run_in_thread(self.get_files_by_filter_sync, filter, params)

    async def delete_file(self, file_name: str) -> bool:
        return await self._run_in_thread(self.delete_file_sync, file_name)

    async def get_all_files(self) -> List[FileItem]:
        """
        Async version of get_all_files_sync.

        Returns:
            List of all FileItem objects ordered by creation date (newest first)
        """
        return await self._run_in_thread(self.get_all_files_sync)


    async def update_file(self, file_name: str, file_item: FileItem) -> bool:
        """
        Async version of update_file_sync.

        Args:
            file_name: The file name that serves as the primary key
            file_item: FileItem object containing the updated information

        Returns:
            True if the update was successful, False otherwise

        Examples:
            Update file information:
            updated_file = FileItem(
                file_name="document.pdf",
                file_name_orig="new_name.pdf",
                user_id=123,
                created_at=datetime.now(),
                processing_status="completed",
                vector_store_id="vs_123456"
            )
            success = await repo.update_file("document.pdf", updated_file)
        """
        return await self._run_in_thread(self.update_file_sync, file_name, file_item)
