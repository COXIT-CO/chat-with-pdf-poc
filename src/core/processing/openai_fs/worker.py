import asyncio
import threading
import time

from typing import List, Tuple, Any

from openai import OpenAI

from core.logger import error
from core.processing.openai_fs.process_file import process_single_file
from core.processing.p_models import WorkerContext
from core.processing.p_utils import get_files_to_process, reset_stuck_files
from core.repositories.repo_files import FilesRepository, FileItem
from openai_wrappers.api_files import files_list
from openai_wrappers.api_vector_store import vector_stores_list
from telemetry.models import TelemetryScope, TeleWProcessor, TeleItemStatus
from telemetry.tele_writer import TeleWriter


def get_openai_resources(client: OpenAI) -> Tuple[List[Any], List[Any]]:
    """Retrieve necessary OpenAI resources for processing."""
    openai_files_list = files_list(client)
    openai_vector_stores = vector_stores_list(client)
    return openai_files_list, openai_vector_stores


def p_openai_fs_worker(
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
    tele = TeleWriter(TelemetryScope.W_PROCESSOR)

    ctx = WorkerContext(
        client=client,
        loop=loop,
        tele=tele,
        files_repository=files_repository
    )

    reset_stuck_files(ctx.files_repository)

    try:
        while not stop_event.is_set():
            process_files = get_files_to_process(files_repository)

            if not process_files:
                stop_event.wait(3)
                continue

            t0 = time.time()
            try:
                openai_files_list, openai_vector_stores = get_openai_resources(client)
            except Exception as e:
                error(f"Error retrieving OpenAI files_list or vector_stores_list: {str(e)}")
                TeleWProcessor(
                    proc_strategy="openai_fs",
                    event="get_openai_resources",
                    status=TeleItemStatus.FAILURE,
                    error_message=str(e),
                    error_recoverable=True,
                    duration_seconds=time.time() - t0,
                ).write(ctx.tele)
                continue
            else:
                TeleWProcessor(
                    proc_strategy="openai_fs",
                    event="get_openai_resources",
                    status=TeleItemStatus.SUCCESS,
                    duration_seconds = time.time() - t0,
                ).write(ctx.tele)

            for file in process_files:
                process_single_file(
                    ctx,
                    file,
                    openai_files_list,
                    openai_vector_stores
                )

            stop_event.wait(1)
    finally:
        loop.close()
