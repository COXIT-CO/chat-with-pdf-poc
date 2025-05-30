import json
import asyncio

from typing import List, Optional

from aiohttp import ClientSession
from openai import BaseModel

from core.configs import EvalConfig
from core.logger import info, error
from core.repositories.repo_files import FileItem
from evaluation.dataset.dataset_metadata import verify_dataset_integrity_or_create_metadata, DatasetFiles
from evaluation.metering import Metering, MeteringItem
from evaluation.stage3_evaluation.eval_chat import call_chat_completions_non_streaming, try_get_usage
from evaluation.stage3_evaluation.eval_utils import parse_model_output_json
from openai_wrappers.types import ChatMessage, ChatMessageUser


class SplitQuestions(BaseModel):
    questions: List[List[str]]


def create_split_questions_if_not_exist(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        eval_files: List[FileItem],
        metering: Metering,
        dataset_files: DatasetFiles,
        eval_config: EvalConfig,
):
    if dataset_files.questions_split_file.is_file():
        verify_dataset_integrity_or_create_metadata(dataset_files, eval_files)
        return

    info(f"{dataset_files.questions_split_file} does not exists: creating from questions_str.json")

    question_str_json: List[str] = json.loads(dataset_files.questions_str_file.read_text())
    assert isinstance(question_str_json, list), "questions_str.json: must be a List"
    assert all(isinstance(item, str) for item in question_str_json), "questions_str.json: All items in the list must be strings"
    assert all(len(item) > 0 for item in question_str_json), "all elements in questions_str.json must be non-empty"
    tokens_cnt = sum(len(item) / 4. for item in question_str_json)
    assert tokens_cnt < 16_000, f"too many tokens in questions_str.json: {tokens_cnt} > 16_000"

    split_questions: SplitQuestions = compose_split_questions(loop, http_session, metering, question_str_json, eval_config)

    dataset_files.questions_split_file.write_text(json.dumps(split_questions.questions, indent=2))

    verify_dataset_integrity_or_create_metadata(dataset_files, eval_files)

    if not dataset_files.questions_split_file.is_file():
        raise Exception(f"something went wrong, couldn't create {dataset_files.questions_split_file}")


async def compose_split_questions_worker(
        http_session: ClientSession,
        metering: Metering,
        messages: List[ChatMessage],
        questions_cnt: int,
        eval_config: EvalConfig,
) -> Optional[SplitQuestions]:
    try:
        resp = await call_chat_completions_non_streaming(
            http_session,
            messages,
            eval_config.chat_eval_model,
            eval_config,
        )

        usage = try_get_usage(resp)
        metering_item = metering.dataset_compose.setdefault(eval_config.chat_eval_model, MeteringItem())
        metering_item.requests_cnt += 1
        metering_item.messages_sent_cnt += len(messages)
        metering_item.tokens_in += usage.prompt_tokens
        metering_item.tokens_out += usage.completion_tokens

        answer: str = resp["choices"][0]["message"]["content"]
        res: SplitQuestions = parse_model_output_json(answer, SplitQuestions)

        assert len(res.questions) == questions_cnt

        return res
    except Exception as e:
        error(f"Error composing split questions: {e}")
        return None


def compose_split_questions(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        metering: Metering,
        question_str_json: List[str],
        eval_config: EvalConfig,
) -> SplitQuestions:
    max_iters = 5
    iters = 0

    messages = [
        ChatMessageUser(
            role="user",
            content=PROMPT.replace("%input%", json.dumps(question_str_json, indent=4))
        )
    ]

    info(f"Compose Split Questions")
    while True:
        info(f"{iters=}")

        if iters >= max_iters:
            raise Exception("too many iters")

        result: Optional[SplitQuestions] = loop.run_until_complete(
            compose_split_questions_worker(
                http_session,
                metering,
                messages,
                len(question_str_json),
                eval_config
            )
        )

        iters += 1

        if not result:
            continue

        return result


PROMPT = """
You are given a List[str] of question texts.
Some of the question texts are composite: contain multiple questions in their text.
Therefore, I ask you to transform that List[str] into List[List[str]] following the logic:

Input Example:
[
    "What are the manufacturers specified in the document? Is XXX a valid manufacturer?",
    "What materials should be used for desks? Is material XXX approved? What options are there?"
]

Output example:
```json
{
    "questions": [
        [
            "What are the manufacturers specified in the document?",
            "Is XXX a valid manufacturer?"
        ],
        [
            "What materials should be used for desks?",
            "Is material XXX approved?",
            "What options for materials are mentioned?",
        ]
    ]
}
```

Make each subquestion in output to always end with a "?" 
Subquestion must also be fully defined

Input:
%input%

Provide output in a valid machine-readable JSON format
"""
