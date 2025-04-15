import asyncio

from pathlib import Path
from typing import Dict, Any, List, Optional

from core.logger import error, exception
from core.processing.openai_fs.const import SEMAPHORE_LIMIT
from core.processing.openai_fs.uploads import upload_paragraph_if_needed, add_to_vector_store_if_needed
from core.processing.p_models import WorkerContext, ParagraphData
from core.processing.p_utils import generate_hashed_filename, jsonl_reader, try_aggr_requests_stats
from core.repositories.repo_files import FileItem
from telemetry.models import RequestResult, RequestStatus, TeleWProcessor, TeleItemStatus


async def process_file_paragraphs(
        ctx: WorkerContext,
        file: FileItem,
        jsonl_file_path: Path,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any],
) -> None:
    """Process all paragraphs in a file with controlled concurrency."""
    base_name = Path(file.file_name_orig).stem
    extension = Path(file.file_name).suffix

    t0 = ctx.loop.time()
    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    async def process_with_semaphore(d_data_dict):
        async with semaphore:
            return await process_paragraph(
                ctx,
                file,
                d_data_dict,
                base_name,
                extension,
                openai_files_list,
                vector_store,
                vector_store_files
            )

    tasks = [
        process_with_semaphore(data_dict)
        for data_dict in jsonl_reader(jsonl_file_path)
    ]

    # Process all paragraphs with controlled concurrency
    results: List[List[RequestResult]] = await asyncio.gather(*tasks)
    results: List[RequestResult] = [result for sublist in results for result in sublist]

    stats = {
        "upload_paragraph": None,
        "add_to_vector_store": None
    }

    r_upload_paragraph = [r for r in results if r.event == "upload_paragraph"]
    stats["upload_paragraph"] = try_aggr_requests_stats(r_upload_paragraph).to_dict()

    r_add_to_vector_store = [r for r in results if r.event == "add_to_vector_store"]
    stats["add_to_vector_store"] = try_aggr_requests_stats(r_add_to_vector_store).to_dict()

    TeleWProcessor(
        proc_strategy="openai_fs",
        event="process_paragraphs_done",
        status=TeleItemStatus.INFO,
        user_id=file.user_id,
        file_name=file.file_name,
        file_name_orig=file.file_name_orig,
        vector_store=vector_store.name,
        attributes={
            "stats": stats
        },
        duration_seconds=ctx.loop.time() - t0
    ).write(ctx.tele)


async def process_paragraph(
        ctx: WorkerContext,
        file: FileItem,
        data_dict: Dict[str, Any],
        base_name: str,
        extension: str,
        openai_files_list: List[Any],
        vector_store: Any,
        vector_store_files: List[Any]
) -> List[RequestResult]:
    """
    Process a single paragraph from a document file.

    This function handles the complete paragraph processing pipeline:
    1. Converts raw dictionary data to a structured ParagraphData object
    2. Generates a unique filename for the paragraph
    3. Uploads the paragraph to OpenAI (with 5s timeout)
    4. Adds the paragraph to the vector store (with 5s timeout)

    Both operations are performed with error handling and telemetry reporting.
    If any step fails, the file is marked as incomplete for future retry.

    Args:
        ctx: Worker context containing client, loop, telemetry and repository
        file: The file item being processed
        data_dict: Dictionary containing paragraph data from the JSONL file
        base_name: Base name of the original file
        extension: File extension
        openai_files_list: List of existing OpenAI files
        vector_store: The vector store to add paragraphs to
        vector_store_files: List of files already in the vector store

    Returns:
        Optional tuple containing upload duration and vector store addition duration,
        or None if any operation failed
    """
    para = ParagraphData(**data_dict)
    filename = generate_hashed_filename(base_name, para.paragraph_text, extension)

    results = []

    t0 = ctx.loop.time()
    try:
        file_data, dur_upload_para = await asyncio.wait_for(
            upload_paragraph_if_needed(ctx, para, filename, openai_files_list),
            timeout=5.0
        )
    except Exception as e:
        error_message = f"Error uploading paragraph: {str(e)}"
        if isinstance(e, asyncio.TimeoutError):
            error_message = "Timeout uploading paragraph (exceeded 5s)"
        error(error_message)

        if file.processing_status != "incomplete":
            file.processing_status = "incomplete"
            await ctx.files_repository.update_file(file.file_name, file)

        results.append(RequestResult(
            "upload_paragraph",
            status=RequestStatus.NOT_OK,
            ts_created=t0,
            duration_seconds=ctx.loop.time() - t0,
            error_message=str(e),
            attributes={
                "filename": filename,
                "paragraph_length": len(para.paragraph_text),
                "page_n": para.page_n,
                "section_number": para.section_number
            }
        ))
        return results
    else:
        if dur_upload_para != 0:
            results.append(RequestResult(
                "upload_paragraph",
                status=RequestStatus.OK,
                ts_created=t0,
                duration_seconds=dur_upload_para,
                attributes={
                    "filename": filename,
                    "paragraph_length": len(para.paragraph_text),
                    "page_n": para.page_n,
                    "section_number": para.section_number
                }
            ))


    t0 = ctx.loop.time()
    try:
        dur_add_to_vs = await asyncio.wait_for(
            add_to_vector_store_if_needed(
                ctx,
                para,
                filename,
                file_data,
                vector_store,
                vector_store_files,
            ),
            timeout=5.0
        )
    except Exception as e:
        error_message = f"Error adding paragraph to vector store: {str(e)}"
        if isinstance(e, asyncio.TimeoutError):
            error_message = "Timeout adding paragraph to vector store (exceeded 5s)"
        error(error_message)

        if file.processing_status != "incomplete":
            file.processing_status = "incomplete"
            await ctx.files_repository.update_file(file.file_name, file)

        results.append(RequestResult(
            "add_to_vector_store",
            status=RequestStatus.NOT_OK,
            ts_created=t0,
            duration_seconds=ctx.loop.time() - t0,
            error_message=str(e),
            attributes={
                "filename": filename,
                "paragraph_length": len(para.paragraph_text),
                "page_n": para.page_n,
                "section_number": para.section_number,
                "vector_store": vector_store.name,
            }
        ))
    else:
        if dur_add_to_vs != 0:
            results.append(RequestResult(
                "add_to_vector_store",
                status=RequestStatus.OK,
                ts_created=t0,
                duration_seconds=dur_add_to_vs,
                attributes={
                    "filename": filename,
                    "paragraph_length": len(para.paragraph_text),
                    "page_n": para.page_n,
                    "section_number": para.section_number,
                    "vector_store": vector_store.name,
                }
            ))

    return results
