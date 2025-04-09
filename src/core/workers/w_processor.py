import json
import asyncio
import threading

from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set

from pydantic import BaseModel
from openai import OpenAI
from more_itertools import chunked

from core.globals import FILES_DIR
from core.logger import info, error, exception
from core.repositories.repo_files import FilesRepository, FileItem
from core.workers.w_abstract import Worker
from core.workers.w_utils import jsonl_reader, generate_paragraph_id, \
    generate_vector_store_file_name, generate_hashed_filename
from openai_wrappers.api_files import files_list, FileUpload, async_file_upload
from openai_wrappers.api_vector_store import (
    vector_stores_list, VectorStoreCreate, vector_store_create,
    vector_store_files_list, VectorStoreFilesList,
    VectorStoreFileCreate, async_vector_store_file_create
)


class ParagraphData(BaseModel):
    page_n: int
    section_number: Optional[str] = None
    paragraph_text: str
    paragraph_box: Tuple[float, float, float, float]


def worker(
        stop_event: threading.Event,
        files_repository: FilesRepository
) -> None:
    """
    Main worker function that processes files in a continuous loop until stopped.

    This worker handles the entire file processing pipeline:
    1. Initializes OpenAI client and event loop
    2. Resets any files stuck in "processing" status to "incomplete"
    3. Continuously polls for files that need processing
    4. Retrieves necessary OpenAI resources (files list and vector stores)
    5. Processes each file through the complete pipeline

    The worker will pause between iterations and can be gracefully stopped
    using the provided stop_event.

    Args:
        stop_event: Threading event used to signal the worker to stop
        files_repository: Repository for accessing and updating file data

    Returns:
        None
    """
    client = OpenAI()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    processing_files = files_repository.get_files_by_filter_sync(
        "processing_status IN (?)",
        ("processing",)
    )
    for file in processing_files:
        file.processing_status = "incomplete"
        files_repository.update_file_sync(file.file_name, file)

    try:
        while not stop_event.is_set():
            process_files = get_files_to_process(files_repository)
            if not process_files:
                stop_event.wait(3)
                continue

            openai_resources = get_openai_resources(client)
            if not openai_resources:
                continue

            openai_files_list, openai_vector_stores = openai_resources

            for file in process_files:
                # todo: execute in parallel when needed
                process_single_file(loop, client, file, files_repository, openai_files_list, openai_vector_stores)

            stop_event.wait(1)
    finally:
        loop.close()


def get_files_to_process(files_repository: FilesRepository) -> List[FileItem]:
    """Get files that need processing from the repository."""
    return files_repository.get_files_by_filter_sync(
        "processing_status IN (?, ?)",
        ("extracted", "incomplete")
    )


def get_openai_resources(client: OpenAI) -> Optional[Tuple[List[Any], List[Any]]]:
    """Retrieve necessary OpenAI resources for processing."""
    try:
        openai_files_list = files_list(client)
        openai_vector_stores = vector_stores_list(client)
        return openai_files_list, openai_vector_stores
    except Exception as e:
        error(f"Error retrieving OpenAI files_list or vector_stores_list: {str(e)}")
        return None


def process_single_file(
        loop: asyncio.AbstractEventLoop,
        client: OpenAI,
        file: FileItem,
        files_repository: FilesRepository,
        openai_files_list: List[Any],
        openai_vector_stores: List[Any]
) -> None:
    """Process a single file through the entire pipeline."""
    info(f"Processing file: {file.file_name_orig} STATUS={file.processing_status}")
    file.processing_status = "processing"

    jsonl_file_path = get_jsonl_file_path(file)
    if not jsonl_file_path or not jsonl_file_path.is_file():
        mark_file_as_error(file, files_repository, "Error: jsonl file not found on disk")
        return

    vector_store = ensure_vector_store_exists(client, file, files_repository, openai_vector_stores)
    if not vector_store:
        return

    vector_store_files = get_vector_store_files(client, vector_store)
    if vector_store_files is None:
        return

    # Generate expected filenames from disk content
    expected_filenames = generate_expected_filenames(
        jsonl_file_path,
        Path(file.file_name_orig).stem,
        Path(file.file_name).suffix
    )

    # Process paragraphs
    process_file_paragraphs(
        loop,
        client,
        file,
        jsonl_file_path,
        openai_files_list,
        vector_store,
        vector_store_files,
        files_repository
    )

    if file.processing_status != "incomplete":
        file.processing_status = "complete"
        files_repository.update_file_sync(file.file_name, file)

    info(f"{file.processing_status}; File {file.file_name}")


def get_jsonl_file_path(file: FileItem) -> Path:
    """Get the path to the JSONL file for a given file."""
    file_path: Path = FILES_DIR.joinpath(file.file_name)
    return file_path.with_suffix('.jsonl')


def mark_file_as_error(file: FileItem, files_repository: FilesRepository, error_message: str) -> None:
    """Mark a file as having an error in processing."""
    file.processing_status = error_message
    files_repository.update_file_sync(file.file_name, file)


def ensure_vector_store_exists(
        client: OpenAI,
        file: FileItem,
        files_repository: FilesRepository,
        openai_vector_stores: List[Any]
) -> Optional[Any]:
    """Ensure a vector store exists for the file, creating one if needed."""
    vs_file_name = generate_vector_store_file_name(file)
    vector_store = next((s for s in openai_vector_stores if s.name == vs_file_name), None)
    if not vector_store:
        payload = VectorStoreCreate(
            name=vs_file_name
        )
        try:
            vector_store = vector_store_create(client, payload)
            info(f"Vector store created for {file.file_name}")
        except Exception as e:
            error(f"Error while creating vector store for {file.file_name}: {str(e)}")
            return None

    if file.vector_store_id != vector_store.id:
        file.vector_store_id = vector_store.id
        files_repository.update_file_sync(file.file_name, file)

    return vector_store


def get_vector_store_files(client: OpenAI, vector_store: Any) -> Optional[List[Any]]:
    """Get the files associated with a vector store."""
    try:
        return vector_store_files_list(
            client,
            VectorStoreFilesList(
                vector_store_id=vector_store.id
            )
        )
    except Exception as e:
        error(f"Error retrieving vector store files for {vector_store.name}: {str(e)}")
        return None


def generate_expected_filenames(
        jsonl_file_path: Path,
        base_name: str,
        extension: str
) -> Set[str]:
    """Generate the set of expected filenames based on disk content."""
    expected_filenames = set()

    for data_dict in jsonl_reader(jsonl_file_path):
        try:
            para = ParagraphData(**data_dict)
            filename = generate_hashed_filename(base_name, para.paragraph_text, extension)
            expected_filenames.add(filename)
        except Exception as e:
            error(f"Error generating expected filename: {str(e)}")

    return expected_filenames


def process_file_paragraphs(
        loop: asyncio.AbstractEventLoop,
        client: OpenAI,
        file: Any,
        jsonl_file_path: Path,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any],
        files_repository: FilesRepository
) -> None:
    """Process all paragraphs in a file in batches of 10 concurrently."""

    base_name = Path(file.file_name_orig).stem
    extension = Path(file.file_name).suffix

    try:

        # Process in batches of 10
        for batch in chunked(jsonl_reader(jsonl_file_path), 10):

            # Create tasks for concurrent execution
            tasks = []
            for data_dict in batch:
                tasks.append(process_paragraph(
                    client,
                    data_dict,
                    base_name,
                    extension,
                    openai_files_list,
                    vector_store,
                    vector_store_files
                ))

            # Execute batch concurrently
            loop.run_until_complete(asyncio.gather(*tasks))

    except Exception as e:
        exception(f"Error processing paragraphs:\n{file}\n{str(e)}")
        file.processing_status = "incomplete"
        files_repository.update_file_sync(file.file_name, file)


async def process_paragraph(
        client: OpenAI,
        data_dict: Dict[str, Any],
        base_name: str,
        extension: str,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any]
) -> None:
    """Process a single paragraph from a file."""
    para = ParagraphData(**data_dict)
    filename = generate_hashed_filename(base_name, para.paragraph_text, extension)

    file_data = await upload_paragraph_if_needed(client, para, filename, openai_files_list)
    if not file_data:
        return

    await add_to_vector_store_if_needed(
        client,
        para,
        filename,
        file_data,
        vector_store,
        vector_store_files,
    )


async def upload_paragraph_if_needed(
        client: OpenAI,
        para: ParagraphData,
        filename: str,
        openai_files_list: List[Any]
) -> Optional[Any]:
    """Upload a paragraph as a file if it doesn't already exist."""
    file_data = next((f for f in openai_files_list if f.filename == filename), None)
    if not file_data:
        info(f"Uploading file: {filename}")
        try:
            file2upload = FileUpload(
                para.paragraph_text.encode(),
                filename,
                "assistants"
            )
            file_data = await async_file_upload(client, file2upload)
            info(f"OK -- file uploaded: {filename} with file_id: {file_data.id}")
        except Exception as e:
            error(f"Error uploading file {filename}: {str(e)}")
            return None
    else:
        info(f"File {filename} already exists with file_id: {file_data.id}")

    return file_data


async def add_to_vector_store_if_needed(
        client: OpenAI,
        para: ParagraphData,
        filename: str,
        file_data: Any,
        vector_store: Any,
        vector_store_files: List[Any],
) -> None:
    """Add a file to the vector store if it's not already there."""
    chunking_strategy = None  # todo: implement

    vector_store_file = next((f for f in vector_store_files if f.id == file_data.id), None)
    if vector_store_file:
        info(f"File {filename} already exists in vector store {vector_store.id}")
        return

    info(f"Adding File {filename} to vector store {vector_store.id}")

    paragraph_id = generate_paragraph_id(para.paragraph_text)

    attributes = {
        "page_n": para.page_n,
        "paragraph_id": paragraph_id,
        "paragraph_box": json.dumps(list(para.paragraph_box)),
    }

    if para.section_number:
        attributes["section_number"] = para.section_number

    try:
        vector_store_file = await async_vector_store_file_create(
            client,
            VectorStoreFileCreate(
                vector_store_id=vector_store.id,
                file_id=file_data.id,
                attributes=attributes,
                chunking_strategy=chunking_strategy,
            )
        )
        info(f"OK -- File {filename} added to vector store {vector_store.id} with id: {vector_store_file.id}")
    except Exception as e:
        error(f"Error adding file {filename} to vector store: {str(e)}")


def spawn_worker(
        files_repository: FilesRepository,
) -> Worker:
    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=worker,
        args=(stop_event, files_repository),
        daemon=True
    )
    worker_thread.start()

    return Worker("worker_processor", worker_thread, stop_event)
