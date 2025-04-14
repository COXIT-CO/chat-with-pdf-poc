import hashlib
import string
import random

from typing import Iterator, Dict, List, Optional
from pathlib import Path

import ujson as json

from core.logger import exception
from core.repositories.repo_files import FileItem, FilesRepository
from telemetry.aggregations.requests_stats import RequestStats, aggr_requests_stats
from telemetry.models import RequestResult


def try_aggr_requests_stats(reqs: List[RequestResult]) -> Optional[RequestStats]:
    try:
        return aggr_requests_stats(reqs)
    except Exception as e:
        exception(e)
        return


def get_files_to_process(files_repository: FilesRepository) -> List[FileItem]:
    """Get files that need processing from the repository."""
    return files_repository.get_files_by_filter_sync(
        "processing_status IN (?, ?)",
        ("extracted", "incomplete")
    )


def reset_stuck_files(files_repository: FilesRepository) -> None:
    """
    Reset any files stuck in 'processing' status to 'incomplete'.

    Args:
        files_repository: Repository for accessing and updating file data
    """
    processing_files = files_repository.get_files_by_filter_sync(
        "processing_status IN (?)",
        ("processing",)
    )
    for file in processing_files:
        file.processing_status = "incomplete"
        files_repository.update_file_sync(file.file_name, file)


def jsonl_reader(file: Path) -> Iterator[Dict]:
    with file.open("r") as f:
        for line in f:
            yield json.loads(line)


def generate_paragraph_id(paragraph_text: str) -> str:
    """
    Generate a unique paragraph ID based on the paragraph text with added randomness.

    Args:
        paragraph_text: The text of the paragraph

    Returns:
        A unique paragraph ID with format pid-{hash}-{random}
    """
    # Generate a random string of 4 characters
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
    # Combine content hash with random suffix
    return f"pid-{generate_content_hash(paragraph_text, length=6)}-{random_suffix}"


def generate_content_hash(content: str, salt: str = "", length: int = 16) -> str:
    """
    Generate a hash from content with optional salt.

    Args:
        content: The content to hash
        salt: Optional salt to add to the content before hashing
        length: Length of the hash to return

    Returns:
        A hexadecimal hash string of the specified length
    """
    # noinspection PyTypeChecker
    return hashlib.md5((salt + content).encode()).hexdigest()[:length]


def generate_hashed_filename(
        base_name: str,
        content: str,
        extension: str
) -> str:
    """
    Generate a filename using a hash of the base_name and content to ensure uniqueness.

    Args:
        base_name: The base name of the file
        content: The content to hash (typically paragraph text)
        extension: The file extension including the dot

    Returns:
        A unique filename with format {base_name}_{hash}{extension}
    """
    content_hash = generate_content_hash(content, base_name)
    return f"{base_name}_{content_hash}{extension}"


def generate_vector_store_file_name(file: FileItem) -> str:
    """
    Generate a filename for vector store that includes the original filename stem and user ID,
    followed by the original extension.

    Args:
        file: The FileItem object containing file metadata

    Returns:
        A string with format "{filename_stem}__userid{user_id}__{extension}"
    """
    # Assuming file_name_orig is a string that includes the extension
    stem = Path(file.file_name_orig).stem
    extension = Path(file.file_name_orig).suffix
    return f"{stem}__userid{file.user_id}__{extension}"
