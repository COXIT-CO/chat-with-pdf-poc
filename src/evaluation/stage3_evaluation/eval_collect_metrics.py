from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel

from evaluation.stage3_evaluation.llm_judge import EvaluationResult
from evaluation.stage3_evaluation.metrics import calculate_binary_metrics, BinaryMetrics


class PerCategoryMetrics(BaseModel):
    """Metrics for a single file"""
    per_file: Dict[str, BinaryMetrics]
    overall: BinaryMetrics


class CategoryMetrics(BaseModel):
    is_question_answered: Optional[PerCategoryMetrics] = None
    requires_additional_information: Optional[PerCategoryMetrics] = None
    is_speculative: Optional[PerCategoryMetrics] = None
    is_confident: Optional[PerCategoryMetrics] = None
    comprehensive_answer: Optional[PerCategoryMetrics] = None

    passed_overall: Optional[Dict[str, Dict[str, bool]]] = None


def extract_boolean_field_by_file_and_question(
        eval_results: Dict[str, Dict[int, EvaluationResult]],
        field_name: str
) -> Dict[str, Dict[str, bool]]:
    """
    Extract a specific boolean field from evaluation results, organized by file and question.

    Args:
        eval_results: Dictionary of evaluation results by filename and question ID
        field_name: Name of the boolean field to extract

    Returns:
        Dictionary mapping filenames to dictionaries of question_ids to boolean values
    """
    result = {}

    for filename, questions in eval_results.items():
        result[filename] = {}

        for q_id, eval_result in questions.items():
            for question in eval_result.questions:
                key = f"{q_id}_{question.id}"
                result[filename][key] = getattr(question, field_name)

    return result


field_extractors = {
    "is_question_answered": lambda results: extract_boolean_field_by_file_and_question(
        results, "is_question_answered"
    ),
    "requires_additional_information": lambda results: extract_boolean_field_by_file_and_question(
        results, "requires_additional_information"
    ),
    "is_speculative": lambda results: extract_boolean_field_by_file_and_question(
        results, "is_speculative"
    ),
    "is_confident": lambda results: extract_boolean_field_by_file_and_question(
        results, "is_confident"
    ),
    "comprehensive_answer": lambda results: extract_boolean_field_by_file_and_question(
        results, "comprehensive_answer"
    )
}


def get_all_boolean_fields_by_file_and_question(
        eval_results: Dict[str, Dict[int, EvaluationResult]]
) -> Dict[str, Dict[str, Dict[str, bool]]]:
    """
    Extract all boolean fields from evaluation results, organized by field name, file, and question.

    Args:
        eval_results: Dictionary of evaluation results by filename and question ID

    Returns:
        Dictionary mapping field names to dictionaries of filenames to dictionaries of question_ids to boolean values
    """
    all_fields = {}

    for field_name, extractor in field_extractors.items():
        all_fields[field_name] = extractor(eval_results)

    return all_fields


def get_passed_overall(
        dict_a: Dict[str, Dict[str, bool]],
        dict_b: Dict[str, Dict[str, bool]]
) -> Dict[str, Dict[str, bool]]:
    """
    Compare two boolean field dictionaries and return a new dictionary indicating equality.

    Args:
        dict_a: First dictionary of boolean fields by filename and question ID
        dict_b: Second dictionary of boolean fields by filename and question ID

    Returns:
        Dictionary mapping filenames to dictionaries of question_ids to boolean values
        where each value indicates if dict_a[filename][question_id] == dict_b[filename][question_id]
    """
    result = {}

    # Ensure both dictionaries have the same filenames
    assert set(dict_a.keys()) == set(dict_b.keys()), "Filenames don't match between dictionaries"

    for filename in dict_a:
        result[filename] = {}

        # Ensure both dictionaries have the same question IDs for this file
        assert set(dict_a[filename].keys()) == set(dict_b[filename].keys()), f"Question IDs don't match for file {filename}"

        for question_id in dict_a[filename]:
            # Compare the boolean values and store the result
            result[filename][question_id] = dict_a[filename][question_id] == dict_b[filename][question_id]

    return result


def passed_overall_to_dataframe(results: CategoryMetrics) -> pd.DataFrame:
    """
    Convert the passed_overall field of CategoryMetrics to a pandas DataFrame.

    Args:
        results: CategoryMetrics instance containing passed_overall data

    Returns:
        DataFrame with filenames as index, question IDs as columns, and boolean values
    """
    if results.passed_overall is None:
        return pd.DataFrame()

    # Initialize an empty dictionary to store data for the DataFrame
    data = {}

    # Get all unique question IDs across all files
    all_question_ids = set()
    for file_data in results.passed_overall.values():
        all_question_ids.update(file_data.keys())

    # For each file, create a row with values for each question ID
    for filename, file_data in results.passed_overall.items():
        row = {}
        for q_id in all_question_ids:
            row[q_id] = file_data.get(q_id, None)  # Use None if question ID not present
        data[filename] = row

    # Create DataFrame with filenames as index and question IDs as columns
    df = pd.DataFrame.from_dict(data, orient='index')

    return df


def per_category_metrics(
        dict_a: Dict[str, Dict[str, bool]],
        dict_b: Dict[str, Dict[str, bool]]
) -> PerCategoryMetrics:
    """
    Calculate binary metrics between two boolean dictionaries at two levels:
    1. Overall (all values across all files)
    2. Per file (all values within each file)

    Args:
        dict_a: First dictionary of boolean fields by filename and question ID
        dict_b: Second dictionary of boolean fields by filename and question ID

    Returns:
        FileMetrics object with metrics at overall and per-file levels
    """
    # Ensure both dictionaries have the same structure
    assert set(dict_a.keys()) == set(dict_b.keys()), "Filenames don't match between dictionaries"

    per_file_metrics = {}

    # Collect all values for overall metrics
    all_values_a = []
    all_values_b = []

    # Calculate per-file metrics and collect values for overall metrics
    for filename in dict_a:
        file_values_a = []
        file_values_b = []

        assert set(dict_a[filename].keys()) == set(dict_b[filename].keys()), f"Question IDs don't match for file {filename}"

        for question_id in dict_a[filename]:
            value_a = dict_a[filename][question_id]
            value_b = dict_b[filename][question_id]

            file_values_a.append(value_a)
            file_values_b.append(value_b)

            all_values_a.append(value_a)
            all_values_b.append(value_b)

        per_file_metrics[filename] = calculate_binary_metrics(
            file_values_a, file_values_b
        )

    overall_metrics = calculate_binary_metrics(
        all_values_a, all_values_b
    )

    # Return as Pydantic model
    return PerCategoryMetrics(
        per_file=per_file_metrics,
        overall=overall_metrics
    )


def collect_eval_metrics(
        a_golden: Dict[str, Dict[int, EvaluationResult]],
        a_pred: Dict[str, Dict[int, EvaluationResult]],
) -> CategoryMetrics:
    assert len(a_golden) == len(a_pred)
    assert set(a_golden.keys()) == set(a_pred.keys())

    results = CategoryMetrics()

    golden_bool_fields: Dict[str, Dict[str, Dict[str,  bool]]] = get_all_boolean_fields_by_file_and_question(a_golden)
    pred_bool_fields: Dict[str, Dict[str, Dict[str,  bool]]] = get_all_boolean_fields_by_file_and_question(a_pred)

    for field_name in golden_bool_fields.keys():
        field_dict_golden = golden_bool_fields[field_name]
        field_dict_pred = pred_bool_fields[field_name]

        per_cat_metrics = per_category_metrics(field_dict_golden, field_dict_pred)

        # Assign the calculated metrics to the appropriate field in the results
        setattr(results, field_name, per_cat_metrics)

    passed_overall = get_passed_overall(
        golden_bool_fields["comprehensive_answer"],
        pred_bool_fields["comprehensive_answer"]
    )
    results.passed_overall = passed_overall

    return results
