import asyncio

import uvloop

from core.configs import EvalConfig
from core.globals import FILES_DIR, SAVE_STRATEGY, PROCESSING_STRATEGY, DB_DIR
from core.logger import init_logger, info
from core.app import App
from core.repositories.repo_files import FilesRepository
from core.server import Server, setup_signal_handlers

from core.workers.w_extractor import spawn_worker as spawn_worker_doc_extractor
from core.workers.w_processor import spawn_worker as spawn_worker_doc_processor
from core.workers.w_watchdog import spawn_worker as spawn_worker_watchdog
from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository


def main():
    init_logger(True)
    info("Logger initialized")

    eval_config = EvalConfig()
    if not eval_config.exists():
        eval_config.save_to_disk()

    files_repository = FilesRepository(DB_DIR / "files.db")
    redis_repository = None
    milvus_repository = None

    if PROCESSING_STRATEGY == "local_fs" and SAVE_STRATEGY == "redis":
        redis_repository = RedisRepository()
        redis_repository.connect()

    elif PROCESSING_STRATEGY == "local_fs" and SAVE_STRATEGY == "milvus":
        milvus_repository = MilvusRepository(DB_DIR / "milvus.db")

    watchdog_worker = spawn_worker_watchdog(FILES_DIR, files_repository)
    doc_e_worker = spawn_worker_doc_extractor(files_repository)
    doc_p_worker = spawn_worker_doc_processor(
        files_repository=files_repository,
        redis_repository=redis_repository,
        milvus_repository=milvus_repository,
    )

    app = App(
        files_repository,
        redis_repository,
        milvus_repository,

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
        port=8011,
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
