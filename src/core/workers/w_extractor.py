import ujson as json
import threading

from pathlib import Path

from core.globals import FILES_DIR
from core.logger import info
from core.repositories.repo_files import FilesRepository
from core.workers.w_abstract import Worker
from coxit.extractor.file_reader import FileReader


VISUALIZE = True


def worker(
        stop_event: threading.Event,
        stats_repository: FilesRepository
):
    """
    Worker function that continuously processes files for text extraction.

    This worker monitors the repository for files with empty processing_status,
    extracts paragraphs from these files using FileReader, and saves the results
    as JSONL files. The processing status of each file is updated accordingly.

    The worker runs in a loop until the stop_event is set, with pauses between
    iterations to prevent excessive CPU usage.

    Args:
        stop_event (threading.Event): Event to signal the worker to stop processing
        stats_repository (FilesRepository): Repository for accessing and updating file metadata

    Flow:
        1. Get files with empty processing_status
        2. For each file:
           - Read file content
           - Extract paragraphs using FileReader
           - Save extracted paragraphs as JSONL
           - Update file processing status to "extracted"
        3. Wait before next iteration
    """
    while not stop_event.is_set():
        process_files = stats_repository.get_files_by_filter_sync("processing_status = ?", ("",))
        if not process_files:
            stop_event.wait(3)
            continue

        for file in process_files:
            info(f"Extracting file: {file.file_name_orig}")
            file_path: Path = FILES_DIR.joinpath(file.file_name)

            if not file_path.is_file():
                file.processing_status = f"Error: file is missing on disk"
                stats_repository.update_file_sync(file.file_name, file)
                continue

            with file_path.open("rb") as f:
                file_content = f.read()

            file_reader = FileReader(file_content, file.file_name_orig)
            extracted_paragraphs = file_reader.extract_paragraphs(visualize=VISUALIZE)

            if not extracted_paragraphs:
                file.processing_status = "Error: no paragraphs extracted"
                stats_repository.update_file_sync(file.file_name, file)
                continue

            jsonl_file = file_path.with_suffix('.jsonl')

            with jsonl_file.open("w") as f:
                for par in extracted_paragraphs:
                    f.write(json.dumps(par.to_dict()) + "\n")

            file.processing_status = "extracted"
            stats_repository.update_file_sync(file.file_name, file)
            info(f"Extracting file {file.file_name_orig} OK")

        stop_event.wait(1)


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

    return Worker("worker_extractor", worker_thread, stop_event)
