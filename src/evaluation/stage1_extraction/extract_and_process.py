import asyncio
import time
from typing import List, Set, Tuple, Dict

from more_itertools import chunked

from core.configs import EvalConfig
from core.globals import SAVE_STRATEGY
from core.tools.tool_context import ToolContext
from evaluation.dataset.dataset_metadata import DatasetFiles
from core.logger import info
from core.repositories.repo_files import FileItem
from core.workers.w_extractor import get_file_paragraphs
from processing.local_fs.models import WorkerContext
from processing.local_fs.process_paragraphs import process_paragraphs_batch
from processing.p_models import ParagraphData, ParagraphVectorData
from processing.p_utils import generate_paragraph_id
from telemetry.models import RequestResult
from vectors.repositories.repo_milvus import collection_from_file_name
from vectors.save_strategies.save_milvus import save_vectors_to_milvus
from vectors.save_strategies.save_redis import save_vectors_to_redis


async def process_file_paragraphs(
        ctx: WorkerContext,
        file: FileItem,
        paragraphs_list: List[ParagraphData],
        eval_config: EvalConfig,
):
    semaphore = asyncio.Semaphore(eval_config.semaphore_embeddings_limit)

    async def process_paragraphs_batch_with_semaphore(
            _paragraphs: List[ParagraphData]
    ):
        async with semaphore:
            return await process_paragraphs_batch(ctx, file, _paragraphs)

    tasks = []
    for data_dict_batch in chunked(paragraphs_list, eval_config.embedding_batch_size):
        tasks.append(asyncio.create_task(
            process_paragraphs_batch_with_semaphore(data_dict_batch)
        ))

    results: List[Tuple[List[RequestResult], List[ParagraphVectorData]]] = await asyncio.gather(*tasks)
    p_vecs: List[ParagraphVectorData] = [r_i for r in results for r_i in r[1]]

    match SAVE_STRATEGY:
        case "redis" | "milvus":
            if SAVE_STRATEGY == "redis":
                assert ctx.repo_redis is not None
                save_func = save_vectors_to_redis
            else:  # milvus
                assert ctx.repo_milvus is not None
                save_func = save_vectors_to_milvus
            save_func(ctx, file, p_vecs)
        case _:
            raise NotImplementedError()


def get_processed_paragraphs(ctx: WorkerContext, file: FileItem) -> Set[str]:
    processed_paragraphs = set()
    if ctx.repo_redis is not None:
        processed_paragraphs = set(ctx.repo_redis.get_all_vector_ids(file.file_name))

    if ctx.repo_milvus is not None:
        col_name = collection_from_file_name(file.file_name)

        collections = ctx.repo_milvus.list_collections()
        if col_name in collections:
            processed_paragraphs = set(ctx.repo_milvus.get_all_vector_par_ids(
                collection_from_file_name(file.file_name)
            ))

    return processed_paragraphs


def process_file_local(
        loop: asyncio.AbstractEventLoop,
        tool_context: ToolContext,
        paragraphs_list: List[ParagraphData],
        file: FileItem,
        eval_config: EvalConfig,
):
    ctx = WorkerContext(
        client=tool_context.openai,
        loop=loop,
        repo_redis=tool_context.redis_repository,
        repo_milvus=tool_context.milvus_repository,
    )

    loop_n = 0
    while True:
        if loop_n >= 10:
            raise Exception("process_file_local: Infinite loop: >=10")

        processed_paragraphs = get_processed_paragraphs(ctx, file)
        tasks = [p for p in paragraphs_list if p.paragraph_id not in processed_paragraphs]
        if not tasks:
            break

        info(f"LOOP {loop_n}: TASKS_CNT: {len(tasks)}")
        loop.run_until_complete(
            process_file_paragraphs(ctx, file, tasks, eval_config)
        )
        loop_n += 1


def extract_and_process_files(
        loop: asyncio.AbstractEventLoop,
        tool_context: ToolContext,
        eval_files: List[FileItem],
        dataset_files: DatasetFiles,
        eval_config: EvalConfig,
) -> Dict[str, List[ParagraphData]]:
    file_paragraphs = {file.file_name_orig: [] for file in eval_files}

    for file in eval_files:
        info(f"FILE: {file.file_name_orig}")
        t0 = time.time()
        # should panic if error
        extracted_paragraphs = get_file_paragraphs(
            file,
            dataset_files.dataset_dir.joinpath(file.file_name_orig),
        )
        extracted_paragraphs = [
            ParagraphData(
                page_n=p.page_n,
                section_number=None,
                paragraph_text=p.paragraph_text,
                paragraph_box=p.paragraph_box,
                paragraph_id=generate_paragraph_id(p.paragraph_text)
            )
            for p in extracted_paragraphs
        ]
        file_paragraphs[file.file_name_orig] = extracted_paragraphs
        info(f"FILE: {file.file_name_orig} EXTRACT: {time.time() - t0:.2f}s")

        t0 = time.time()
        process_file_local(
            loop,
            tool_context,
            extracted_paragraphs,
            file,
            eval_config,
        )
        info(f"FILE: {file.file_name_orig} PROCESS: {time.time() - t0:.2f}s")

    return file_paragraphs
