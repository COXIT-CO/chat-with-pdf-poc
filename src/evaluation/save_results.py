import json
from pathlib import Path
from typing import List, Tuple, Dict

from pydantic import BaseModel

from core.configs import EvalConfig
from core.globals import EVALUATIONS_DIR, PROCESSING_STRATEGY, SAVE_STRATEGY
from core.logger import error
from evaluation.dataset.dataset_init import DatasetEval
from evaluation.dataset.dataset_metadata import DatasetFiles
from evaluation.metering import Metering
from evaluation.dataset.eval_questions_load import EvalQuestionCombined
from evaluation.stage3_evaluation.eval_collect_metrics import CategoryMetrics
from evaluation.stage3_evaluation.llm_judge import EvaluationResult
from openai_wrappers.types import ChatMessage
from processing.p_models import ParagraphData
from core.repositories.repo_files import FileItem


def get_next_evaluation_directory() -> Path:
    """
    Find the latest evaluation directory and create a new one with an incremented number.
    Evaluation directories follow the pattern "0001", "0002", etc.

    Returns:
        Path: The path to the newly created evaluation directory
    """
    # Get all existing evaluation directories with 4-character names (like "0001", "0002", etc.)
    existing_dirs = [
        i for i in sorted(list(EVALUATIONS_DIR.iterdir()))
        if i.is_dir() and len(i.name) == 4 and i.name.isdigit()
    ]

    tries_cnt = 1
    if existing_dirs:
        last_dir = existing_dirs[-1]
        tries_cnt = int(last_dir.name) + 1

    eval_dir = EVALUATIONS_DIR / f"{tries_cnt:04d}"
    eval_dir.mkdir(exist_ok=True, parents=True)

    return eval_dir


class EvalParams(BaseModel):
    dataset_name: str
    description: str
    processing_strategy: str
    save_strategy: str
    chat_model: str
    chat_eval_model: str

    eval_documents: List[str]

    def to_json(self) -> str:
        """
        Convert the model to a JSON string with proper error handling.
        """
        try:
            # Use Pydantic's built-in JSON serialization
            return self.model_dump_json(indent=2)
        except Exception as e:
            error(f"Failed to serialize EvalParams to JSON: {e}")

            # Fallback approach - try to serialize each field individually
            result = {}
            for field_name in self.model_fields.keys():
                try:
                    field_value = getattr(self, field_name)
                    # Test if the field can be serialized
                    json.dumps(field_value)
                    result[field_name] = field_value
                except Exception as e:
                    error(f"Failed to serialize field {field_name}: {e}")
                    result[field_name] = f"Error: {e}"

            return json.dumps(result, indent=2)


def dump_eval_params(
        eval_dir: Path,
        eval_details: str,
        dataset_eval: DatasetEval,
        dataset_files: DatasetFiles,
        eval_config: EvalConfig,
):
    eval_params = EvalParams(
        dataset_name=dataset_files.dataset_dir.name,
        description=eval_details,
        processing_strategy=PROCESSING_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        chat_model=eval_config.chat_model,
        chat_eval_model=eval_config.chat_eval_model,
        eval_documents=[d.file_name_orig for d in dataset_eval.eval_files],
    )
    eval_dir.joinpath("params.json").write_text(eval_params.to_json())

    eval_dir.joinpath("QID2Questions.json").write_text(
        json.dumps({q.id: q.question_text for q in dataset_eval.questions}, indent=2)
    )
    questions_split_dicts = [
        {f"{q.id}_{qs.id}": qs.question}
        for q in dataset_eval.questions for qs in q.questions_split
    ]
    eval_dir.joinpath("QID2QuestionsSplit.json").write_text(json.dumps(questions_split_dicts, indent=2))


def dump_stage1_extraction(
        eval_dir: Path,
        file_paragraphs: List[Tuple[FileItem, List[ParagraphData]]]
):
    dir_name = eval_dir.joinpath("stage1_extraction")
    paragraphs_raw_dir = dir_name / "paragraphs_raw"
    paragraphs_raw_dir.mkdir(exist_ok=True, parents=True)
    paragraphs_readable_dir = dir_name / "paragraphs_readable"
    paragraphs_readable_dir.mkdir(exist_ok=True, parents=True)

    for file, paragraphs in file_paragraphs:
        readable_text = ""
        with paragraphs_raw_dir.joinpath(file.file_name_orig).with_suffix(".jsonl").open("w") as f_raw:
            prev_page_n = None
            for p in paragraphs:
                f_raw.write(p.model_dump_json() + "\n")
                if p.page_n != prev_page_n:
                    readable_text += f"\n\n==== PAGE {p.page_n} =====\n\n"
                prev_page_n = p.page_n
                readable_text += f"\nID: {p.paragraph_id}\n{p.paragraph_text}\n"

        paragraphs_readable_dir.joinpath(file.file_name_orig).with_suffix(".txt").write_text(readable_text)


def dump_stage2_answers(
        eval_dir: Path,
        golden_answers_dicts: Dict[str, Dict[int, str]],
        rag_results_dicts: Dict[str, Dict[int, List[ChatMessage]]],
        rag_answers_dicts: Dict[str, Dict[int, str]],
        questions: List[EvalQuestionCombined],
):
    dir_name = eval_dir.joinpath("stage2_answers")
    golden_answers_dir = dir_name / "golden_answers"
    golden_answers_dir.mkdir(exist_ok=True, parents=True)
    rag_results_raw = dir_name / "rag_results"
    rag_results_raw.mkdir(exist_ok=True, parents=True)
    rag_answers_dir = dir_name / "rag_answers"
    rag_answers_dir.mkdir(exist_ok=True, parents=True)

    for file_name, golden_answers in golden_answers_dicts.items():
        file_golden_text = f"FN: {file_name}\n\n\n\n"

        for q_id, g_answer in golden_answers.items():
            question_text: str = [q.question_text for q in questions if q.id == q_id][0]
            file_golden_text += f"Q;ID={q_id}:\n{question_text}\n\nA:\n{g_answer}\n\n\n\n"

        golden_answers_dir.joinpath(file_name).with_suffix(".txt").write_text(file_golden_text)

    for file_name, rag_results in rag_results_dicts.items():
        file_dir_raw = rag_results_raw / file_name
        file_dir_raw.mkdir(exist_ok=True, parents=True)

        for q_id, rag_messages in rag_results.items():
            file_dir_raw.joinpath(f"{q_id}.json").write_text(
                json.dumps([m.model_dump() for m in rag_messages], indent=2))

    for file_name, rag_answers in rag_answers_dicts.items():
        file_answers_text = f"FN: {file_name}\n\n\n\n"

        for q_id, rag_answer in rag_answers.items():
            question_text: str = [q.question_text for q in questions if q.id == q_id][0]
            file_answers_text += f"Q;ID={q_id}:\n{question_text}\n\nA:\n{rag_answer}\n\n\n\n"

        rag_answers_dir.joinpath(file_name).with_suffix(".txt").write_text(file_answers_text)


def save_evaluation_results(
        evals: Dict[int, EvaluationResult],
        questions: List[EvalQuestionCombined],
        file_name: str
) -> str:
    eval_text = f"FN: {file_name}\n\n\n\n"

    for q_id, eval_res in evals.items():
        question_text: str = [q.question_text for q in questions if q.id == q_id][0]
        eval_text += f"Q;ID={q_id}:\n{question_text}\n\n"
        eval_text += f"ANSWER: {eval_res.answer}\n\n"

        for quest_eval in eval_res.questions:
            sub_q_id = f"{q_id}_{quest_eval.id}"
            sub_q_question_text = [sq.question for q in questions for sq in q.questions_split if
                                   q.id == q_id and sq.id == quest_eval.id][0]
            eval_text += f"SUBQ; ID={sub_q_id}:\n{sub_q_question_text}\n\n"
            eval_text += f"EVAL:\n{quest_eval.model_dump_json(indent=2)}\n\n"

    return eval_text


def dump_stage3_llm_judge(
        eval_dir: Path,
        golden_evals: Dict[str, Dict[int, EvaluationResult]],
        rag_evals: Dict[str, Dict[int, EvaluationResult]],
        questions: List[EvalQuestionCombined],
):
    s3_dir = eval_dir.joinpath("stage3_evaluation")
    llm_judge_dir = s3_dir / "llm_judge"
    golden_evals_dir = llm_judge_dir / "golden_evals"
    golden_evals_dir.mkdir(exist_ok=True, parents=True)
    rag_evals_dir = llm_judge_dir / "rag_evals"
    rag_evals_dir.mkdir(exist_ok=True, parents=True)

    for file_name, g_evals in golden_evals.items():
        eval_text = save_evaluation_results(g_evals, questions, file_name)
        golden_evals_dir.joinpath(file_name).with_suffix(".txt").write_text(eval_text)

    for file_name, r_evals in rag_evals.items():
        eval_text = save_evaluation_results(r_evals, questions, file_name)
        rag_evals_dir.joinpath(file_name).with_suffix(".txt").write_text(eval_text)


def dump_stage3_metrics(
        eval_dir: Path,
        metrics: CategoryMetrics,
):
    s3_dir = eval_dir.joinpath("stage3_evaluation")
    metrics_dir = s3_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)

    for k in metrics.model_fields.keys():
        if k != "passed_overall":
            v = getattr(metrics, k)
            metrics_dir.joinpath(k).with_suffix(".json").write_text(v.model_dump_json(indent=2))

    metrics_dir.joinpath("passed_overall.json").write_text(json.dumps(metrics.passed_overall, indent=2))


def dump_stage4_analysis(
        eval_dir: Path,
        anal_results: Dict[str, str],
        anal_user_messages: Dict[str, str],
):
    s4_dir = eval_dir.joinpath("stage4_analysis")
    anal_results_dir = s4_dir / "analysis_results"
    anal_results_dir.mkdir(exist_ok=True, parents=True)
    user_messages_dir = s4_dir / "user_messages"
    user_messages_dir.mkdir(exist_ok=True, parents=True)

    for file_name, analysis_result in anal_results.items():
        anal_results_dir.joinpath(Path(file_name).stem).with_suffix(".txt").write_text(analysis_result)

    for file_name, user_message in anal_user_messages.items():
        user_messages_dir.joinpath(Path(file_name).stem).with_suffix(".txt").write_text(user_message)


def dump_stage4_anal_md(
        eval_dir: Path,
        anal_results: Dict[str, str],
        anal_user_messages: Dict[str, str],
):
    s4_dir = eval_dir.joinpath("stage4_analysis")
    anal_results_dir = s4_dir / "analysis_results_md"
    anal_results_dir.mkdir(exist_ok=True, parents=True)
    user_messages_dir = s4_dir / "user_messages_md"
    user_messages_dir.mkdir(exist_ok=True, parents=True)

    for file_name, analysis_result in anal_results.items():
        anal_results_dir.joinpath(Path(file_name).stem).with_suffix(".md").write_text(analysis_result)

    for file_name, user_message in anal_user_messages.items():
        user_messages_dir.joinpath(Path(file_name).stem).with_suffix(".md").write_text(user_message)


def dump_stage4_anal_overall(
        eval_dir: Path,
        anal_overall: str,
        user_message: str,
):
    s4_dir = eval_dir.joinpath("stage4_analysis")
    eval_dir.joinpath("analysis_overall.md").write_text(anal_overall)

    s4_dir.joinpath("analysis_overall_user.md").write_text(user_message)


def dump_metering(
        eval_dir: Path,
        metering: Metering,
):
    eval_dir.joinpath("metering.json").write_text(metering.model_dump_json(indent=2))
