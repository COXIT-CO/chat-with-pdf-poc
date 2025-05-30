import time

from core.logger import info, error, exception
from processing.local_fs.process_paragraphs import process_file_paragraphs
from processing.local_fs.models import WorkerContext
from processing.openai_fs.process_file import get_jsonl_file_path, mark_file_as_error
from core.repositories.repo_files import FileItem
from telemetry.models import (
    TeleWProcessor, TeleItemStatus
)
from vectors.repositories.repo_milvus import collection_from_file_name


def process_single_file(ctx: WorkerContext, file: FileItem) -> None:
    info(f"Processing file: {file.file_name_orig} STATUS={file.processing_status}")
    file.processing_status = "processing"
    ctx.files_repository.update_file_sync(file.file_name, file)

    processed_paragraphs = set()
    if ctx.repo_redis is not None:
        processed_paragraphs = set(ctx.repo_redis.get_all_vector_ids(file.file_name))

    if ctx.repo_milvus is not None:
        col_name = collection_from_file_name(file.file_name)

        collections = ctx.repo_milvus.list_collections()
        if col_name in collections:
            try:
                processed_paragraphs = set(ctx.repo_milvus.get_all_vector_par_ids(
                    collection_from_file_name(file.file_name)
                ))
            except Exception as e:
                error(f"couldn't retrieve processed_paragraphs from milvus: {e}")

    info(f"Found {len(processed_paragraphs)} processed paragraphs")

    ts_process_single_file = time.time()

    jsonl_file_path = get_jsonl_file_path(file)
    if not jsonl_file_path or not jsonl_file_path.is_file():
        error_message = "Error: jsonl file not found on disk"
        error(error_message)
        mark_file_as_error(file, ctx.files_repository, error_message)
        TeleWProcessor(
            proc_strategy="local_fs",
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
        ctx.loop.run_until_complete(process_file_paragraphs(
            ctx,
            file,
            jsonl_file_path,
            processed_paragraphs,
        ))
    except Exception as e:
        error_message = f"Error processing paragraphs:\n{file}\n{str(e)}"
        exception(error_message)
        file.processing_status = "incomplete"
        ctx.files_repository.update_file_sync(file.file_name, file)
        TeleWProcessor(
            proc_strategy="local_fs",
            error_message=error_message,
            error_recoverable=True,
            event="process_file_paragraphs",
            status=TeleItemStatus.FAILURE,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)
    else:
        TeleWProcessor(
            proc_strategy="local_fs",
            event="process_file_paragraphs",
            status=TeleItemStatus.SUCCESS,
            user_id=file.user_id,
            file_name=file.file_name,
            file_name_orig=file.file_name_orig,
            duration_seconds=time.time() - t0
        ).write(ctx.tele)

    if file.processing_status != "incomplete":
        file.processing_status = "complete"
        ctx.files_repository.update_file_sync(file.file_name, file)

    TeleWProcessor(
        proc_strategy="local_fs",
        event="process_file_done",
        status=TeleItemStatus.INFO,
        user_id=file.user_id,
        file_name=file.file_name,
        file_name_orig=file.file_name_orig,
        attributes={
            "processing_status": file.processing_status
        },
        duration_seconds=time.time() - ts_process_single_file
    ).write(ctx.tele)

    info(f"{file.processing_status}; File {file.file_name}")
