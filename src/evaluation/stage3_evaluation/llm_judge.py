import asyncio
import json
from typing import List, Dict, Optional, Tuple

from aiohttp import ClientSession
from pydantic import BaseModel

from core.configs import EvalConfig
from core.logger import exception, info
from evaluation.metering import Metering, MeteringItem
from evaluation.stage3_evaluation.eval_utils import parse_model_output_json
from evaluation.dataset.eval_questions_load import EvalQuestionCombined
from evaluation.stage3_evaluation.eval_chat import call_chat_completions_non_streaming, try_get_usage
from openai_wrappers.types import ChatMessage, ChatMessageUser


__all__ = ["evaluate_model_outputs", "EvaluationResult", "QuestionEval"]


class QuestionEval(BaseModel):
    id: int
    question: str
    answer: str
    is_question_answered: bool
    requires_additional_information: bool
    is_speculative: bool
    is_confident: bool

    @property
    def comprehensive_answer(self) -> bool:
        return (
                self.is_question_answered and
                not self.requires_additional_information and
                not self.is_speculative and
                self.is_confident
        )


class EvaluationResult(BaseModel):
    answer: Optional[str] = None
    questions: List[QuestionEval]


async def evaluate_answer_worker(
        http_session: ClientSession,
        metering: Metering,
        messages: List[ChatMessage],
        question_id: int,
        orig_answer: str,
        eval_config: EvalConfig,
) -> Optional[Tuple[int, EvaluationResult]]:
    try:
        resp = await call_chat_completions_non_streaming(
            http_session,
            messages,
            eval_config.chat_eval_model,
            eval_config,
        )

        usage = try_get_usage(resp)
        metering_item = metering.stage3.setdefault(eval_config.chat_eval_model, MeteringItem())
        metering_item.requests_cnt += 1
        metering_item.messages_sent_cnt += len(messages)
        metering_item.tokens_in += usage.prompt_tokens
        metering_item.tokens_out += usage.completion_tokens

        answer: str = resp["choices"][0]["message"]["content"]
        e_res: EvaluationResult = parse_model_output_json(answer, EvaluationResult)
        e_res.answer = orig_answer
        return question_id, e_res

    except Exception as e:
        exception(f"Error evaluating answer: {e}")
        return None


async def evaluate_answers_for_doc(
        http_session: ClientSession,
        metering: Metering,
        questions: List[EvalQuestionCombined],
        doc_answers: Dict[int, str],
        eval_config: EvalConfig,
) -> Dict[int, EvaluationResult]:
    semaphore = asyncio.Semaphore(eval_config.semaphore_eval_limit)

    async def evaluate_answer_with_semaphore(
            _messages: List[ChatMessage],
            question_id: int,
            _answer: str,
    ):
        async with semaphore:
            return await evaluate_answer_worker(
                http_session,
                metering,
                _messages,
                question_id,
                _answer,
                eval_config,
            )

    tasks = []
    for question in questions:
        # todo: investigate KeyError
        answer = doc_answers[question.id]

        prompt = PROMPT.replace(
            "%questions%",
            "\n".join([
                json.dumps(q.model_dump())
                for q in question.questions_split
            ])
        ).replace(
            "%answer%", answer
        )

        messages = [
            ChatMessageUser(role="user", content=prompt)
        ]
        tasks.append(asyncio.create_task(
            evaluate_answer_with_semaphore(messages, question.id, answer)
        ))

    results: List[Optional[Tuple[int, EvaluationResult]]] = await asyncio.gather(*tasks)
    results: List[Tuple[int, EvaluationResult]] = [r for r in results if r is not None]

    results: Dict[int, EvaluationResult] = {r[0]: r[1] for r in results}

    return results


def evaluate_model_outputs(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        metering: Metering,
        questions: List[EvalQuestionCombined],
        answers: Dict[str, Dict[int, str]],
        eval_config: EvalConfig,
):
    results = {}

    for doc_name, doc_answers in answers.items():
        info(f"Evaluating results for {doc_name}")
        assert len(doc_answers)

        max_iters = 5
        iters = 0
        not_answered = doc_answers.copy()
        eval_results_for_doc: Dict[int, EvaluationResult] = {}

        while True:
            info(f"{iters=}; TASKS: {len(not_answered)}")

            if iters >= max_iters:
                raise Exception("Failed to eval: too many tries")

            eval_results_for_doc_iter: Dict[int, EvaluationResult] = loop.run_until_complete(
                evaluate_answers_for_doc(
                    http_session,
                    metering,
                    questions,
                    not_answered,
                    eval_config,
                )
            )
            eval_results_for_doc.update(eval_results_for_doc_iter)

            not_answered = {
                k: v for k, v in not_answered.items()
                if k not in eval_results_for_doc.keys()
            }
            if not not_answered:
                break

            iters += 1

        results[doc_name] = eval_results_for_doc

    return results


PROMPT = """
QUESTIONS: 
%questions%


ANSWER: 
%answer%


You are given a list of one or several questions.
You have to evaluate the answer.
Question text could be complex and contain several questions.

Detect Questions (one or several) in the question text.
For each question in Questions:
    Check if the question is answered AND
    Check if the question requires additional information AND
    Check if the answer is speculative: if answer is speculative due to insufficient context AND
    Detect is_confident: was answer confident or not. 
    

CRITERIA:
is_question_answered -- question has a reasonable answer, supplemented with details.  
requires_additional_information -- whether additional information is needed to properly answer
is_speculative -- whether answer is speculative due to insufficient context.
is_confident -- whether answer is confident or not.

Example:
QUESTION: Is Stevens listed as an Approved Manufacturer or is a Substitution Request needed for this?
ANSWER: Yes, "Stevens Industries, Inc." is listed as an approved manufacturer in the document under the acceptable products for casework.

Example above gets is_answered=true, as it matches criteria.

Output format:
```json
{
    "questions": [
        {
            "id": 0
            "question": "Is Stevens an Approved Manufacturer?",
            "answer": "Yes",
            "is_question_answered": boolean,
            "requires_additional_information": boolean,
            "is_speculative": boolean,
            "is_confident": boolean
        }
        ...
    ]
}
```

WHERE:
id: id taken from QUESTIONS section
question: very compact question text
answer: very compact summary of the actual answer

Provide output in a valid machine-readable JSON format.
"""
