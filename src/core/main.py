import asyncio

import uvloop

from core.globals import BASE_DIR, FILES_DIR, PROCESSING_STRATEGY
from core.logger import init_logger, info
from core.app import App
from core.repositories.repo_files import FilesRepository
from core.server import Server, setup_signal_handlers

from core.workers.w_extractor import spawn_worker as spawn_worker_doc_extractor
from core.workers.w_processor import spawn_worker as spawn_worker_doc_processor
from core.workers.w_watchdog import spawn_worker as spawn_worker_watchdog


def main():
    init_logger(True)
    info("Logger initialized")

    db_dir = BASE_DIR / "db"
    db_dir.mkdir(parents=True, exist_ok=True)

    files_repository = FilesRepository(db_dir / "files.db")

    watchdog_worker = spawn_worker_watchdog(FILES_DIR, files_repository)
    doc_e_worker = spawn_worker_doc_extractor(files_repository)
    doc_p_worker = spawn_worker_doc_processor(
        files_repository, PROCESSING_STRATEGY
    )

    app = App(
        files_repository,
        docs_url=None,
        redoc_url=None,
        openapi_url="/v1/openapi.json",
        openapi_tags=[
            {
                "name": "Files",
                "description": "Operations related to file management, including uploading, listing, and deleting files."
            },
            {
                "name": "MCPL",
                "description": "Model Context Protocol Like operations and tools."
            },
        ]
    )

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

    server = Server(
        app=app,
        host="0.0.0.0",
        port=8011, # todo: use env values
        workers=[
            watchdog_worker,
            doc_e_worker,
            doc_p_worker,
        ]
    )

    setup_signal_handlers(server)

    server.run()
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
