import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List

from aiohttp import ClientSession

from core.configs import EvalConfig
from core.logger import info
from core.repositories.repo_files import FileItem
from evaluation.metering import Metering
from evaluation.stage4_analysis.anal_results import analyse_results_tasks
from openai_wrappers.types import ChatMessageUser


def analyse_reports_into_md(
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

        file_anal_text = (eval_dir /  "stage4_analysis" / "analysis_results" / Path(file_name).stem).with_suffix(".txt").read_text()

        user_message = (USER_MESSAGE
                        .replace("%analysis%", file_anal_text)
                        .replace("%metrics%", json.dumps(file_comprehensive_answer, indent=2))
                        .replace("%passed_overall%", json.dumps(file_passed_overall, indent=2)))

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
            raise Exception("Failed to analyse_per_file_reports: too many tries")

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
I am evaluating my RAG System
A Large Language Model has already analyzed results, logs and provided insights for a file I was evaluating RAG on

Here's its analysis:
%analysis%

Here's metrics for that file in JSON format
%metrics%

Passed overall:
%passed_overall%

What I ask you to do is: condense all the information above in a concise, readable MD format
Please follow the structure defined bellow while constructing your report:

Summary:
<provide a brief summary of experiment's results -- 1 brief paragraph>

Questions passed:
<md table; rows: question_id, columns: question_id, sub_question_0..sub_question_N> (use passed_overall)
Example:
question_id sub_0, sub_1, sub_2
0           true true true
1           false true false
2           true false
4           false false true

Metrics:
<md table with columns: metric, value, CI_lower, CI_upper>

Interpretation:
<your interpretation of the metrics above; md table with columns: metric, interpretation; keep interpretations brief>

Opinion:
<in one word: is a system we've been evaluating proved itself good, bad, moderate etc., on the file?>
<explanation: 1 paragraph, brief>

Systematic failures:
<if any: md table with following columns: failure, description, advice>

Recommendations:
<if any, summarize recommendations, 1 paragraph tops>
"""
