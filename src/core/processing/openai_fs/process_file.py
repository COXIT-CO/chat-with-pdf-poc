import time
from pathlib import Path
from typing import Any, List

from core.globals import FILES_DIR
from core.logger import info, error, exception
from core.processing.openai_fs.process_paragraphs import process_file_paragraphs
from core.processing.p_models import WorkerContext
from core.processing.p_utils import generate_vector_store_file_name
from core.repositories.repo_files import FileItem, FilesRepository
from telemetry.models import TeleWProcessor, TeleItemStatus
from openai_wrappers.api_vector_store import VectorStoreCreate, vector_store_create, vector_store_files_list, \
    VectorStoreFilesList


def get_jsonl_file_path(file: FileItem) -> Path:
    """Get the path to the JSONL file for a given file."""
    file_path: Path = FILES_DIR.joinpath(file.file_name)
    return file_path.with_suffix('.jsonl')


def mark_file_as_error(file: FileItem, files_repository: FilesRepository, error_message: str) -> None:
    """Mark a file as having an error in processing."""
    file.processing_status = error_message
    files_repository.update_file_sync(file.file_name, file)


def ensure_vector_store_exists(
        ctx: WorkerContext,
        file: FileItem,
        files_repository: FilesRepository,
        openai_vector_stores: List[Any]
) -> Any:
    """Ensure a vector store exists for the file, creating one if needed."""
    vs_file_name = generate_vector_store_file_name(file)
    vector_store = next((s for s in openai_vector_stores if s.name == vs_file_name), None)
    if not vector_store:
        payload = VectorStoreCreate(
            name=vs_file_name
        )
        vector_store = vector_store_create(ctx.client, payload)
        info(f"Vector store created for {file.file_name}")

    if file.vector_store_id != vector_store.id:
        file.vector_store_id = vector_store.id
        files_repository.update_file_sync(file.file_name, file)

    return vector_store


def get_vector_store_files(ctx: WorkerContext, vector_store: Any) -> List[Any]:
    """Get the files associated with a vector store."""
    return vector_store_files_list(
        ctx.client,
        VectorStoreFilesList(
            vector_store_id=vector_store.id
        )
    )


def process_single_file(
        ctx: WorkerContext,
        file: FileItem,
        openai_files_list: List[Any],
        openai_vector_stores: List[Any]
) -> None:
    """Process a single file through the entire pipeline."""
    info(f"Processing file: {file.file_name_orig} STATUS={file.processing_status}")
    file.processing_status = "processing"
    ctx.files_repository.update_file_sync(file.file_name, file)

    ts_process_single_file = time.time()

    jsonl_file_path = get_jsonl_file_path(file)
    if not jsonl_file_path or not jsonl_file_path.is_file():
        error_message = "Error: jsonl file not found on disk"
        error(error_message)
        mark_file_as_error(file, ctx.files_repository, error_message)
        TeleWProcessor(
            proc_strategy="openai_fs",
            event="get_jsonl_file_path",
            status=TeleItemStatus.FAILURE,
            error_message=error_message,
            error_recoverable=False,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
        ).write(ctx.tele)
        return

    t0 = time.time()
    try:
        vector_store = ensure_vector_store_exists(ctx, file, ctx.files_repository, openai_vector_stores)
    except Exception as e:
        error_message = f"Error while creating vector store for {file.file_name}: {str(e)}"
        mark_file_as_error(file, ctx.files_repository, error_message)
        error(error_message)
        TeleWProcessor(
            proc_strategy="openai_fs",
            event="ensure_vector_store_exists",
            status=TeleItemStatus.FAILURE,
            error_message=error_message,
            error_recoverable=False,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)
        return

    t0 = time.time()
    try:
        vector_store_files = get_vector_store_files(ctx, vector_store)
    except Exception as e:
        error_message = f"Error retrieving vector store files for {vector_store.name}: {str(e)}"
        mark_file_as_error(file, ctx.files_repository, error_message)
        error(error_message)
        TeleWProcessor(
            proc_strategy="openai_fs",
            error_message=error_message,
            error_recoverable=False,
            event="get_vector_store_files",
            status=TeleItemStatus.FAILURE,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            vector_store=vector_store.name,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)
        return

    # Process paragraphs
    t0 = time.time()
    try:
        # Use ctx.loop to run the coroutine
        ctx.loop.run_until_complete(process_file_paragraphs(
            ctx,
            file,
            jsonl_file_path,
            openai_files_list,
            vector_store,
            vector_store_files
        ))
    except Exception as e:
        error_message = f"Error processing paragraphs:\n{file}\n{str(e)}"
        exception(error_message)
        file.processing_status = "incomplete"
        ctx.files_repository.update_file_sync(file.file_name, file)
        TeleWProcessor(
            proc_strategy="openai_fs",
            error_message=error_message,
            error_recoverable=True,
            event="process_file_paragraphs",
            status=TeleItemStatus.FAILURE,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            vector_store=vector_store.name,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)
    else:
        TeleWProcessor(
            proc_strategy="openai_fs",
            event="process_file_paragraphs",
            status=TeleItemStatus.SUCCESS,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            vector_store=vector_store.name,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)

    if file.processing_status != "incomplete":
        file.processing_status = "complete"
        ctx.files_repository.update_file_sync(file.file_name, file)

    TeleWProcessor(
        proc_strategy="openai_fs",
        event="process_file_done",
        status=TeleItemStatus.INFO,
        user_id=file.user_id,
        file_name=file.file_name,
        file_name_orig=file.file_name_orig,
        vector_store=vector_store.name,
        attributes={
            "processing_status": file.processing_status
        },
        duration_seconds=time.time() - ts_process_single_file
    ).write(ctx.tele)

    info(f"{file.processing_status}; File {file.file_name}")
