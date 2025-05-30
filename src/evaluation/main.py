import asyncio
import time
from asyncio import AbstractEventLoop
from typing import List, Tuple, Dict

import aiohttp
from openai import OpenAI

from core.configs import EvalConfig
from core.globals import DB_DIR, PROCESSING_STRATEGY, SAVE_STRATEGY
from core.logger import init_logger, info
from core.repositories.repo_files import FileItem, FilesRepository
from core.tools.tool_context import ToolContext
from evaluation.args import parse_arguments
from evaluation.dataset.dataset_init import init_dataset_eval
from evaluation.metering import Metering
from evaluation.save_results import get_next_evaluation_directory, dump_eval_params, dump_stage1_extraction, \
    dump_stage2_answers, dump_stage3_llm_judge, dump_stage3_metrics, dump_metering, dump_stage4_analysis, \
    dump_stage4_anal_md, dump_stage4_anal_overall
from evaluation.stage2_answers.ans_golden import produce_golden_answers
from evaluation.stage3_evaluation.eval_collect_metrics import collect_eval_metrics
from evaluation.stage3_evaluation.llm_judge import evaluate_model_outputs
from evaluation.stage1_extraction.extract_and_process import extract_and_process_files
from evaluation.globals import EVAL_USER_ID, DB_EVAL_DIR
from evaluation.stage2_answers.ans_rag import produce_rag_answers
from evaluation.stage4_analysis.anal_all_reports import analyse_all_reports
from evaluation.stage4_analysis.anal_reports_md import analyse_reports_into_md
from evaluation.stage4_analysis.anal_results import analyse_results
from evaluation.tui import prompt_user_for_evaluation_details
from processing.p_models import ParagraphData
from vectors.repositories.repo_milvus import MilvusRepository
from vectors.repositories.repo_redis import RedisRepository


__all__ = []


def compose_tool_context(
        loop: AbstractEventLoop,
):
    files_repository = FilesRepository(DB_DIR / "files.db")
    files_repository.delete_user_files_sync(EVAL_USER_ID)

    redis_repository = None
    milvus_repository = None

    if PROCESSING_STRATEGY == "local_fs" and SAVE_STRATEGY == "redis":
        redis_repository = RedisRepository()
        redis_repository.connect()

    elif PROCESSING_STRATEGY == "local_fs" and SAVE_STRATEGY == "milvus":
        milvus_repository = MilvusRepository(DB_EVAL_DIR / "milvus.db")

    client = OpenAI()
    http_session = aiohttp.ClientSession(loop=loop)

    return ToolContext(
        http_session=http_session,
        user_id=EVAL_USER_ID,
        files_repository=files_repository,
        redis_repository=redis_repository,
        milvus_repository=milvus_repository,
        openai=client,
    )


def main():
    args = parse_arguments()

    init_logger(False)
    info("Logger initialized")

    eval_config = EvalConfig()
    eval_config.read_from_disk()
    assert eval_config.chat_endpoint
    assert eval_config.chat_endpoint_api_key

    metering = Metering()

    loop = asyncio.new_event_loop()
    tool_context = compose_tool_context(loop)
    dataset_files, dataset_eval = init_dataset_eval(
        loop, tool_context.http_session, metering, args, eval_config
    )

    for file in dataset_eval.eval_files:
        assert tool_context.files_repository.create_file_sync(file), f"Failed to create record of file={file} in DB"

    eval_dir = get_next_evaluation_directory()
    eval_details = args.eval_details
    if eval_details is None:
        eval_details = prompt_user_for_evaluation_details()

    dump_eval_params(eval_dir, eval_details, dataset_eval, dataset_files, eval_config)

    try:
        file_paragraphs_dict = extract_and_process_files(
            loop,
            tool_context,
            dataset_eval.eval_files,
            dataset_files,
            eval_config
        )

        file_paragraphs: List[Tuple[FileItem, List[ParagraphData]]] = [
            (f, file_paragraphs_dict[f.file_name_orig]) for f in dataset_eval.eval_files
        ]
        dump_stage1_extraction(eval_dir, file_paragraphs)

        golden_answers = produce_golden_answers(
            loop,
            tool_context.http_session,
            metering, file_paragraphs,
            dataset_eval.questions,
            eval_config,
        )

        rag_results = produce_rag_answers(
            tool_context,
            loop,
            metering,
            dataset_eval,
            eval_config,
        )
        rag_answers: Dict[str, Dict[int, str]] = {
            file_name: {
                k: v[-1].content for k, v in fn_results.items()
            } for file_name, fn_results in rag_results.items()
        }
        dump_stage2_answers(eval_dir, golden_answers, rag_results, rag_answers, dataset_eval.questions)

        eval_golden = evaluate_model_outputs(
            loop, tool_context.http_session, metering, dataset_eval.questions, golden_answers, eval_config
        )
        eval_rag = evaluate_model_outputs(
            loop, tool_context.http_session, metering, dataset_eval.questions, rag_answers, eval_config
        )
        dump_stage3_llm_judge(
            eval_dir, eval_golden, eval_rag, dataset_eval.questions
        )

        t0 = time.time()
        eval_metrics = collect_eval_metrics(
            eval_golden, eval_rag
        )
        dump_stage3_metrics(eval_dir, eval_metrics)
        info(f"collect_eval_metrics: {time.time() - t0:.2f}s")

        # passed_overall_df = passed_overall_to_dataframe(eval_metrics)

        anal_results, anal_user_messages = analyse_results(
            loop, tool_context.http_session, metering, eval_dir, dataset_eval.eval_files, eval_config
        )
        dump_stage4_analysis(eval_dir, anal_results, anal_user_messages)

        anal_md, anal_md_user_messages = analyse_reports_into_md(
            loop, tool_context.http_session, metering, eval_dir, dataset_eval.eval_files, eval_config
        )
        dump_stage4_anal_md(eval_dir, anal_md, anal_md_user_messages)

        anal_overall_md, anal_overall_user = analyse_all_reports(
            loop, tool_context.http_session, metering, eval_dir, dataset_eval.eval_files, eval_config
        )
        dump_stage4_anal_overall(eval_dir, anal_overall_md, anal_overall_user)

        dump_metering(eval_dir, metering)

    finally:
        tool_context.files_repository.delete_user_files_sync(EVAL_USER_ID)
        loop.run_until_complete(tool_context.http_session.close())
        loop.close()


if __name__ == "__main__":
    main()
