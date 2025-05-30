import asyncio

from typing import List, Dict, Any, Optional, Tuple

from core.configs import EvalConfig
from core.logger import info, exception
from core.repositories.repo_files import FileItem
from core.tools.tool_context import ToolContext
from core.tools.tool_search_in_file import ToolSearchInFile
from core.tools.tools import execute_tools
from evaluation.dataset.dataset_init import DatasetEval
from evaluation.metering import Metering, MeteringItem
from evaluation.stage2_answers.ans_golden import SYSTEM
from evaluation.dataset.eval_questions_load import EvalQuestionCombined
from evaluation.stage3_evaluation.eval_chat import call_chat_completions_non_streaming, try_get_usage
from openai_wrappers.types import ChatMessage, ChatMessageAssistant, ToolCall, ToolCallFunction, ChatMessageUser, \
    ChatMessageSystem


async def recursive_chat_worker(
        ctx: ToolContext,
        metering: Metering,
        messages: List[ChatMessage],
        question: EvalQuestionCombined,
        eval_config: EvalConfig,
) -> Optional[Tuple[int, List[ChatMessage]]]:
    try:
        tools = [
            ToolSearchInFile().as_chat_tool().model_dump()
        ]

        max_iters = 5
        iters = 0

        while True:
            if iters == max_iters:
                tools = None

                messages.append(ChatMessageUser(
                    role="user", content=f"Give your final answer\n----\n{question.question_text}"
                ))

            resp = await call_chat_completions_non_streaming(
                ctx.http_session,
                messages,
                eval_config.chat_model,
                eval_config,
                tools,
            )

            usage = try_get_usage(resp)
            metering_item = metering.stage2.setdefault(eval_config.chat_model, MeteringItem())
            metering_item.requests_cnt += 1
            metering_item.messages_sent_cnt += len(messages)
            metering_item.tokens_in += usage.prompt_tokens
            metering_item.tokens_out += usage.completion_tokens

            content: str = resp["choices"][0]["message"]["content"] or ""

            new_message = ChatMessageAssistant(
                role="assistant",
                content=content
            )

            tool_calls: List[Dict[str, Any]] = resp["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                tool_calls: List[ToolCall] = [
                    ToolCall(
                        id=t["id"],
                        type=t["type"],
                        function=ToolCallFunction(
                            name=t["function"]["name"],
                            arguments=t["function"]["arguments"],
                        )
                    )
                    for t in tool_calls
                ]
                new_message.tool_calls = tool_calls

            messages.append(new_message)

            if not new_message.tool_calls:
                break

            tool_results = await execute_tools(ctx, messages)
            messages.extend(tool_results)

            if iters == max_iters:
                break

            iters += 1

        assert messages[-1].role == "assistant"

        return question.id, messages

    except Exception as e:
        exception(f"Exception in recursive_chat_worker: {e}")
        return None



async def recursive_chat(
        ctx: ToolContext,
        metering: Metering,
        questions: List[EvalQuestionCombined],
        file: FileItem,
        eval_config: EvalConfig,
):
    semaphore = asyncio.Semaphore(eval_config.semaphore_chat_limit)

    async def recursive_chat_with_semaphore(
            _messages: List[ChatMessage],
            _question: EvalQuestionCombined,
    ):
        async with semaphore:
            return await recursive_chat_worker(
                ctx,
                metering,
                _messages,
                _question,
                eval_config,
            )

    init_messages = [
        ChatMessageSystem(
            role="system", content=f"{SYSTEM}\n----\nDOC NAME: {file.file_name_orig}"
        )
    ]

    tasks = []
    for question in questions:
        messages = [
            *init_messages,
            ChatMessageUser(
                role="user", content=USER_MESSAGE.replace("%question%", question.question_text)
            )
        ]
        tasks.append(asyncio.create_task(
            recursive_chat_with_semaphore(messages, question)
        ))

    results: List[Optional[Tuple[int, List[ChatMessage]]]] = await asyncio.gather(*tasks)
    results: List[Tuple[int, List[ChatMessage]]] = [r for r in results if r is not None]

    answers = {
        i[0]: i[1] for i in results
    }

    return answers


def produce_rag_answers(
        ctx: ToolContext,
        loop: asyncio.AbstractEventLoop,
        metering: Metering,
        dataset_eval: DatasetEval,
        eval_config: EvalConfig,
) -> Dict[str, Dict[int, List[ChatMessage]]]:
    results = {}

    for file in dataset_eval.eval_files:
        max_iters = 5
        iters = 0
        not_answered = dataset_eval.questions.copy()
        answers_for_doc: Dict[int, List[ChatMessage]] = {}

        while True:
            info(f"{iters=}; TASKS: {len(not_answered)}")

            if iters == max_iters:
                raise Exception(f"Failed to produce RAG answers: too many tries")

            answers_for_doc_iter: Dict[int, List[ChatMessage]] = loop.run_until_complete(
                recursive_chat(
                    ctx,
                    metering,
                    not_answered,
                    file,
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


USER_MESSAGE = """
I need you to answer a question:
%question%

To retrieve additional context that will help you to answer questions -- use tools.
You can execute document search as many times as you need to answer question properly.
You need to respond with utmost confidence in your tone of voice
"""
