import asyncio
import threading

from openai import OpenAI

from core.processing.local_fs.process_file import process_single_file
from core.processing.p_models import WorkerContext
from core.processing.p_utils import (
    reset_stuck_files, get_files_to_process
)
from core.repositories.repo_files import FilesRepository
from telemetry.models import (
    TelemetryScope
)
from telemetry.tele_writer import TeleWriter


def p_local_fs_worker(
        stop_event: threading.Event,
        files_repository: FilesRepository
) -> None:
    client = OpenAI()
    loop = asyncio.new_event_loop()
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
            process_files = get_files_to_process(ctx.files_repository)

            if not process_files:
                stop_event.wait(3)
                continue

            for file in process_files:
                process_single_file(ctx, file)

    finally:
        ctx.loop.close()
