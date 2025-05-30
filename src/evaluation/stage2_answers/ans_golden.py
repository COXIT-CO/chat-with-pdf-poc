import asyncio
from typing import Tuple, List, Dict, Optional

from aiohttp import ClientSession

from core.configs import EvalConfig
from core.logger import exception, info
from core.repositories.repo_files import FileItem
from evaluation.metering import Metering, MeteringItem
from evaluation.dataset.eval_questions_load import EvalQuestionCombined
from evaluation.stage3_evaluation.eval_chat import call_chat_completions_non_streaming, try_get_usage
from processing.p_models import ParagraphData
from openai_wrappers.types import ChatMessage, ChatMessageUser, ChatMessageSystem


__all__= ["SYSTEM", "produce_golden_answers"]


SYSTEM = """
You are a helpful assistant that is utmost concise, yet precise, in its responses.
You should use the context provided to construct clear, hallucinations-robust and confident answer.
"""


async def golden_answers_worker(
        http_session: ClientSession,
        metering: Metering,
        messages: List[ChatMessage],
        question_id: int,
        eval_config: EvalConfig,
) -> Optional[Tuple[int, str]]:
    try:
        resp = await call_chat_completions_non_streaming(
            http_session,
            messages,
            eval_config.chat_model,
            eval_config,
        )
        usage = try_get_usage(resp)
        metering_item = metering.stage2.setdefault(eval_config.chat_model, MeteringItem())
        metering_item.requests_cnt += 1
        metering_item.messages_sent_cnt += len(messages)
        metering_item.tokens_in += usage.prompt_tokens
        metering_item.tokens_out += usage.completion_tokens

        try:
            answer: str = resp["choices"][0]["message"]["content"]
        except Exception as e:
            info(resp)
            exception(f"Error while getting golden answers: {e}")
            return None

        return question_id, answer

    except Exception as e:
        exception(f"Error while getting golden answers: {e}")
        return None


async def golden_answers_for_doc(
        http_session: ClientSession,
        metering: Metering,
        doc_text: str,
        questions: List[EvalQuestionCombined],
        eval_config: EvalConfig,
) -> Dict[int, str]:
    semaphore = asyncio.Semaphore(eval_config.semaphore_chat_limit)

    async def golden_answers_with_semaphore(
            _messages: List[ChatMessage],
            question_id: int,
    ):
        async with semaphore:
            return await golden_answers_worker(
                http_session,
                metering,
                _messages,
                question_id,
                eval_config,
            )

    init_messages = [
        ChatMessageSystem(
            role="system",
            content=SYSTEM,
        ),
        ChatMessageUser(role="user", content=doc_text),
    ]

    tasks = []
    for question in questions:
        q_messages = [
            *init_messages,
            ChatMessageUser(role="user", content=question.question_text),
        ]
        tasks.append(asyncio.create_task(
            golden_answers_with_semaphore(q_messages, question.id)
        ))

    results: List[Optional[Tuple[int, str]]] = await asyncio.gather(*tasks)
    results: List[Tuple[int, str]] = [r for r in results if r is not None]

    golden_answers = {
        q_id: answer for q_id, answer in results
    }

    return golden_answers


def produce_golden_answers(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        metering: Metering,
        file_paragraphs: List[Tuple[FileItem, List[ParagraphData]]],
        questions: List[EvalQuestionCombined],
        eval_config: EvalConfig,
) -> Dict[str, Dict[int, str]]:
    results = {}

    for file, paragraphs in file_paragraphs:
        info(f"GOLDEN FOR FILE: {file.file_name_orig}")
        assert len(paragraphs)
        assert sum([(len(p.paragraph_text) or 1) / 4. for p in paragraphs]) <= 60_000, f"file {file.file_name} is too big: >60k tok"

        doc_text = "\n".join([p.paragraph_text for p in paragraphs])

        max_iters = 5
        iters = 0
        not_answered = questions.copy()
        answers_for_doc: Dict[int, str] = {}

        while True:
            info(f"{iters=}; TASKS: {len(not_answered)}")

            if iters == max_iters:
                raise Exception(f"Failed to produce golden answers: too many tries")

            answers_for_doc_iter: Dict[int, str] = loop.run_until_complete(
                golden_answers_for_doc(
                    http_session,
                    metering,
                    doc_text,
                    questions,
                    eval_config,
                )
            )
            answers_for_doc.update(answers_for_doc_iter)

            not_answered = [q for q in not_answered if q.id not in answers_for_doc.keys()]
            if not not_answered:
                break

            iters += 1

        results[file.file_name_orig] = answers_for_doc

    return results
