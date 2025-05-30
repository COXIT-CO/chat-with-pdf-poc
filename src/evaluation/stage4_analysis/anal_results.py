import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from aiohttp import ClientSession

from core.configs import EvalConfig
from core.logger import exception, info
from core.repositories.repo_files import FileItem
from evaluation.metering import Metering, MeteringItem
from evaluation.stage3_evaluation.eval_chat import call_chat_completions_non_streaming, try_get_usage
from openai_wrappers.types import ChatMessageUser, ChatMessage


async def analyse_results_tasks_worker(
        http_session: ClientSession,
        metering: Metering,
        messages: List[ChatMessage],
        file_name: str,
        eval_config: EvalConfig,
) -> Optional[Tuple[str, str]]:
    try:
        resp = await call_chat_completions_non_streaming(
            http_session,
            messages,
            eval_config.chat_analyse_model,
            eval_config,
        )

        usage = try_get_usage(resp)
        metering_item = metering.stage4.setdefault(eval_config.chat_analyse_model, MeteringItem())
        metering_item.requests_cnt += 1
        metering_item.messages_sent_cnt += len(messages)
        metering_item.tokens_in += usage.prompt_tokens
        metering_item.tokens_out += usage.completion_tokens

        answer: str = resp["choices"][0]["message"]["content"]

        return file_name, answer
    except Exception as e:
        exception(e)
        return None


async def analyse_results_tasks(
        http_session: ClientSession,
        metering: Metering,
        tasks: List[Dict[str, Any]],
        eval_config: EvalConfig,
) -> Dict[str, str]:
    semaphore = asyncio.Semaphore(eval_config.semaphore_eval_limit)

    async def analyse_results_with_semaphore(
            _messages: List[ChatMessage],
            file_name: str
    ):
        async with semaphore:
            return await analyse_results_tasks_worker(
                http_session,
                metering,
                _messages,
                file_name,
                eval_config
            )

    tasks_to_process = [
        asyncio.create_task(
            analyse_results_with_semaphore(t["messages"], t["file_name"])
        )
        for t in tasks
    ]

    results: List[Optional[Tuple[str, str]]] = await asyncio.gather(*tasks_to_process)
    results: List[Tuple[str, str]] = [r for r in results if r is not None]
    results: Dict[str, str] = {r[0]: r[1] for r in results}

    return results


def analyse_results(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        metering: Metering,
        eval_dir: Path,
        eval_files: List[FileItem],
        eval_config: EvalConfig,
):
    comprehensive_answer_json = eval_dir / "stage3_evaluation" / "metrics" / "comprehensive_answer.json"
    comprehensive_answer: Dict[str, Any] = json.loads(comprehensive_answer_json.read_text())

    passed_overall_json = eval_dir / "stage3_evaluation" / "metrics" / "passed_overall.json"
    passed_overall: Dict[str, Any] = json.loads(passed_overall_json.read_text())

    user_messages: Dict[str, str] = {}
    tasks = []

    for file in eval_files:
        file_name = file.file_name_orig
        file_comprehensive_answer: str = comprehensive_answer["per_file"][file_name]
        file_passed_overall = passed_overall[file_name]

        s3_golden_answers_file = eval_dir / "stage3_evaluation" / "llm_judge" / "golden_evals" / Path(Path(file_name).stem).with_suffix(".txt")
        s3_golden_answers = s3_golden_answers_file.read_text()

        s3_rag_answers_file = eval_dir / "stage3_evaluation" / "llm_judge" / "rag_evals" / Path(Path(file_name).stem).with_suffix(".txt")
        s3_rag_answers = s3_rag_answers_file.read_text()

        user_message = (USER_MESSAGE
                        .replace("%golden_answers%", s3_golden_answers)
                        .replace("%rag_answers%", s3_rag_answers)
                        .replace("%passed_overall%", json.dumps(file_passed_overall, indent=2))
                        .replace("%comprehensive_answer%", json.dumps(file_comprehensive_answer, indent=2)))

        user_messages[file_name] = user_message

        messages = [
            ChatMessageUser(role="user", content=user_message)
        ]

        tasks.append(
            {"file_name": file_name, "messages": messages}
        )

    results: Dict[str, str] = {}
    max_iters = 5
    iters = 0

    while True:
        info(f"{iters=}; TASKS: {len(tasks)}")

        if iters >= max_iters:
            raise Exception("Failed to analyse_results: too many tries")

        results_local: Dict[str, str] = loop.run_until_complete(
            analyse_results_tasks(
                http_session,
                metering,
                tasks,
                eval_config,
            )
        )
        results.update(results_local)
        tasks = [t for t in tasks if t["file_name"] not in results]
        if not tasks:
            break

        iters += 1

    return results, user_messages


USER_MESSAGE = """
I am evaluating my RAG system. 
Please, conduct a research through evaluation results to provide insights about this experiment's results.
I need you to compare RAG answers with Golden Answers, Where Golden Answers = ground truth answers

To receive ground truth answer we provide the whole document to the model, and ask it a single question.
Assuming the model has all needed context, it will produce a correct answer to the question. 

On the other hand RAG answers produced when model did not have access to the whole document, 
but that model was able to call tools, and retrieve context through search system using vector similarities.

The point of the experiment is to compare, how close RAG Answers are to Golden Answers.

When we collected dataset with Golden Answers and RAG Answers,
we ask LLM-as-a-Judge system to mark each answer with labels. There are 4 of them initially:

"is_question_answered": boolean -- question has a reasonable answer, supplemented with details
"requires_additional_information": boolean -- whether additional information is needed to properly answer
"is_speculative": boolean -- whether answer is speculative due to insufficient context
"is_confident": boolean -- whether answer is confident or not

On top of those 4 labels we compose our target label: comprehensive_answer
It is composed in a following manner:
```python
( is_question_answered and
not requires_additional_information and
not is_speculative and
is_confident )
```

Here's answers from Golden, and RAG approaches with Evaluations from Evaluator LLM-as-a-Judge model
Each question can be complex, and is split into several sub-questions e.g: QID: 0 has 0_0, 0_1 sub-questions
Evaluator model marked each sub-question with 4 labels above.

Golden Answers:
```
%golden_answers%
```

RAG Answers:
```
%rag_answers%
```

And here is a Dict, where each sub-questions corresponds to boolean value -- if question passed or not,
by golden_comprehensive_answer == rag_comprehensive_answer 
```
%passed_overall%
```

Additionally, you may use metrics: accuracy, precision, recall, f1, kappa for comprehensive_answer,
those metrics received via comparison 2 List[bool]: golden and RAG for one single file.
Note, that for each metric there's Confidence Interval, which you should operate, when presenting results about a metric
```
%comprehensive_answer%
```

Having all that information, you should be able to conduct analysis of results of that experiment. 

In that analysis:

* highlight cases where RAG failed to work. 
  Why didn't it work in that specific case? 
  Do you see systematic failure?
  What could help RAG in that specific case to work better?
* Lookup on metrics, what do they say? 
  Provide your interpretation of the metric, having all the data and cases.
  What insights do those metric show? 
  How to apply those insights to make RAG system perform better?
* Compose a conclusion, having both highlighted cases, and metrics.
  Summarize results of experiments, and the insights.
  How well performed RAG system? 
  Operate with metrics and specific cases to supplement your conclusions.  
"""
