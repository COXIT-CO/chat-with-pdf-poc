import json

from typing import List, Any
from pathlib import Path
from pydantic import BaseModel

from evaluation.dataset.dataset_metadata import DatasetFiles


__all__ = ["load_combined_questions", "EvalQuestionCombined"]



class EvalQuestion(BaseModel):
    id: int
    question: str


class EvalQuestionSplit(BaseModel):
    id: int
    question: str


class EvalQuestionsSplit(BaseModel):
    id: int
    questions: List[EvalQuestionSplit]


class EvalQuestionCombined(BaseModel):
    id: int
    question_text: str
    questions_split: List[EvalQuestionSplit]


def load_json_file(file_path: Path) -> Any:
    """Load and parse a JSON file with standardized error handling."""
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"File '{file_path}' contains invalid JSON: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Error loading file: {str(e)}")


def load_questions_str(file_path: Path) -> List[EvalQuestion]:
    """Load questions from a JSON file as EvalQuestion objects."""
    questions_str = load_json_file(file_path)

    if not isinstance(questions_str, list):
        raise TypeError(f"Expected a JSON array in '{file_path}'")

    return [EvalQuestion(id=idx, question=text) for idx, text in enumerate(questions_str)]


def load_questions_split(file_path: Path) -> List[EvalQuestionsSplit]:
    """Load split questions from a JSON file as EvalQuestionSplit objects."""
    questions_split: List[List[str]] = load_json_file(file_path)

    if not isinstance(questions_split, list):
        raise TypeError(f"Expected a JSON array in '{file_path}'")

    # Validate structure
    for i, group in enumerate(questions_split):
        if not isinstance(group, list):
            raise TypeError(f"Item {i} in '{file_path}' is not a list")
        for j, question in enumerate(group):
            if not isinstance(question, str):
                raise TypeError(f"Question {j} in group {i} in '{file_path}' is not a string")

    return [
        EvalQuestionsSplit(
            id=idx,
            questions=[
                EvalQuestionSplit(
                    id=q_idx,
                    question=q
                )
                for (q_idx, q) in enumerate(question_group)
            ]
        )
        for idx, question_group in enumerate(questions_split)
    ]


def load_combined_questions(
        dataset_files: DatasetFiles,
) -> List[EvalQuestionCombined]:
    """
    Load both question text and split questions from separate JSON files and combine them.
    """
    # Load both question formats
    str_questions = load_questions_str(dataset_files.questions_str_file)
    split_questions = load_questions_split(dataset_files.questions_split_file)

    # Ensure we have the same number of questions in both files
    if len(str_questions) != len(split_questions):
        raise ValueError(
            f"Mismatch in question counts: {len(str_questions)} texts vs {len(split_questions)} splits"
        )

    # Combine them into the new model
    combined_questions = []
    for text_q, split_q in zip(str_questions, split_questions):
        # Verify that IDs match
        if text_q.id != split_q.id:
            raise ValueError(f"Question ID mismatch: {text_q.id} vs {split_q.id}")

        combined_questions.append(
            EvalQuestionCombined(
                id=text_q.id,
                question_text=text_q.question,
                questions_split=split_q.questions
            )
        )

    return combined_questions
