import asyncio
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple

from aiohttp import ClientSession

from core.configs import EvalConfig
from core.repositories.repo_files import FileItem
from evaluation.args import CMDArgs
from evaluation.dataset.dataset_metadata import DatasetFiles
from evaluation.dataset.split_questions_create import create_split_questions_if_not_exist
from evaluation.dataset.eval_questions_load import load_combined_questions, EvalQuestionCombined
from evaluation.globals import EVAL_USER_ID
from evaluation.metering import Metering


@dataclass
class DatasetEval:
    eval_files: List[FileItem]
    questions: List[EvalQuestionCombined]


def init_dataset_eval(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        metering: Metering,
        args: CMDArgs,
        eval_config: EvalConfig,
) -> Tuple[DatasetFiles, DatasetEval]:
    dataset_files = DatasetFiles.new(args)

    eval_files: Iterator[Path] = (
        f for f in args.dataset_dir.iterdir()
        if f.is_file() and f.suffix == ".pdf"
    )
    eval_files: List[FileItem] = [
        FileItem(
            file_name=f"file_{uuid.uuid4().hex[:24]}",
            file_name_orig=f.name,
            user_id=EVAL_USER_ID,
            processing_status="completed",
        )
        for f in eval_files
    ]
    if len(eval_files) == 0:
        raise Exception(f"no *.pdf found in given dataset: {args.dataset_dir}")

    if not dataset_files.questions_str_file.is_file():
        raise Exception(f"{dataset_files.questions_str_file} does not exist")

    create_split_questions_if_not_exist(
        loop, http_session, eval_files, metering, dataset_files, eval_config
    )

    questions = load_combined_questions(dataset_files)

    return dataset_files, DatasetEval(
        eval_files=eval_files,
        questions=questions
    )
