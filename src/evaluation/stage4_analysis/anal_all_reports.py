import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from aiohttp import ClientSession

from core.configs import EvalConfig
from core.logger import info
from core.repositories.repo_files import FileItem
from evaluation.metering import Metering
from evaluation.stage3_evaluation.eval_utils import parse_language_block
from evaluation.stage4_analysis.anal_results import analyse_results_tasks
from openai_wrappers.types import ChatMessageUser


def analyse_all_reports(
        loop: asyncio.AbstractEventLoop,
        http_session: ClientSession,
        metering: Metering,
        eval_dir: Path,
        eval_files: List[FileItem],
        eval_config: EvalConfig,
) -> Tuple[str, str]:
    comprehensive_answer_json = eval_dir / "stage3_evaluation" / "metrics" / "comprehensive_answer.json"
    comprehensive_answer_overall: Dict[str, Any] = json.loads(comprehensive_answer_json.read_text())["overall"]

    constants: Dict[str, Any] = json.loads(eval_dir.joinpath("params.json").read_text())

    file_reports = {}

    for file in eval_files:
        file_name = file.file_name_orig
        file_report: str = (eval_dir / "stage4_analysis" / "analysis_results_md" / Path(file_name).stem).with_suffix(".md").read_text()
        file_reports[file_name] = file_report

    user_message = (USER_MESSAGE
                    .replace("%constants%", json.dumps(constants, indent=2))
                    .replace("%overall_metrics%", json.dumps(comprehensive_answer_overall, indent=2))
                    .replace("%reports_per_file%", json.dumps(file_reports, indent=2)))

    messages = [ChatMessageUser(role="user", content=user_message)]

    max_iters = 5
    iters = 0

    while True:
        info(f"{iters=}")

        if iters >= max_iters:
            raise Exception("Failed to analyse_all_reports: too many tries")

        results: Dict[str, str] = loop.run_until_complete(
            analyse_results_tasks(
                http_session,
                metering,
                [
                    {"file_name": "overall", "messages": messages}
                ],
                eval_config,
            )
        )

        result = results["overall"]
        if not results:
            iters += 1
            continue

        result = parse_language_block(result, ["markdown", "md"])

        return result, user_message


USER_MESSAGE = """
I am evaluating my RAG System
It evaluates how well RAG System deals with different files with given questions
I already received detailed metrics and analysis per each file
You need to summarise and aggregate all file results into a final report

Here are constants of the experiment
%constants%

Here are overall metrics:
%overall_metrics%

Here are detailed reports, how system shown itself on each file in a dataset:
%reports_per_file%

What I ask you to do is: condense all the information above in a concise, readable MD format
Please follow the structure defined bellow while constructing your report:

Summary:
<provide a brief summary of experiment's results -- 1 brief paragraph>

Metrics:
<md table with columns: metric, value, CI_lower, CI_upper> (overall metrics)

Interpretation:
<your interpretation of the metrics above; md table with columns: metric, interpretation; keep interpretations brief>

Opinion:
<in one word: is a system we've been evaluating proved itself good, bad, moderate etc., overall?>
<explanation: 1 paragraph, brief>

Systematic failures:
<if any: md table with following columns: failure, description, advice>

Most difficult files for the system:
<md table with columns: file_name, how_difficult, why_difficult; keep it brief>
how difficult -- reason for considering to put the file into the table
why difficult -- potential explanation what caused the system perform bad on a file
put in the table only the files system that have significantly lower metrics then the others 

Recommendations:
<if any, summarize recommendations, 1 paragraph tops>

The .md you are writing is getting published on the site as is, so keep the style
"""
