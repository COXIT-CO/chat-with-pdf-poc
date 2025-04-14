import threading

from core.processing.openai_fs.worker import p_openai_fs_worker
from core.processing.local_fs.worker import p_local_fs_worker
from core.repositories.repo_files import FilesRepository
from core.workers.w_abstract import Worker


def spawn_worker(
        files_repository: FilesRepository,
        p_strategy: str = "openai_fs",
) -> Worker:
    stop_event = threading.Event()

    if p_strategy == "openai_fs":
        worker = p_openai_fs_worker
    elif p_strategy == "local_fs":
        worker = p_local_fs_worker
    else:
        raise ValueError(f"Unknown P_STRATEGY: {p_strategy}")

    worker_thread = threading.Thread(
        target=worker,
        args=(stop_event, files_repository),
        daemon=True
    )
    worker_thread.start()

    return Worker("worker_processor", worker_thread, stop_event)
