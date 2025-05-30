from typing import List, Tuple, Optional

import numpy as np
from numba import jit
from pydantic import BaseModel


class BinaryMetric(BaseModel):
    value: float = 0.
    ci: Optional[Tuple[float, float]] = None


class BinaryMetrics(BaseModel):
    accuracy: BinaryMetric
    precision: BinaryMetric
    recall: BinaryMetric
    f1: BinaryMetric
    kappa: BinaryMetric
    n_samples: int = 0


class BooleanFields(BaseModel):
    is_question_answered: BinaryMetrics
    requires_additional_information: BinaryMetrics
    is_speculative: BinaryMetrics
    is_confident: BinaryMetrics
    comprehensive_answer: BinaryMetrics


@jit(nopython=True)
def _bootstrap_sample_indices(n, n_bootstrap):
    """Generate bootstrap sample indices using Numba for performance"""
    all_indices = np.zeros((n_bootstrap, n), dtype=np.int32)
    for i in range(n_bootstrap):
        all_indices[i] = np.random.choice(n, size=n, replace=True)
    return all_indices


def bootstrap_confidence_interval(
        y_true: List[bool],
        y_pred: List[bool],
        metric_func,
        # number of bootstrap samples to generate during the confidence interval calculation
        n_bootstrap=2000,
        # confidence level for the interval
        # typical 0.95 for scientific research
        confidence=0.95
) -> Optional[Tuple[float, float]]:
    """
    Calculate confidence interval using bootstrap resampling.

    Bootstrap resampling is a statistical method that estimates the sampling distribution
    of a statistic by repeatedly sampling with replacement from the original dataset.
    This function uses bootstrap to estimate confidence intervals for classification metrics.

    Interpretation of confidence intervals:
    - The interval (lower_bound, upper_bound) represents the range where the true metric
      value is expected to lie with the specified confidence level.
    - Narrower intervals indicate more precise estimates.
    - If intervals from two different models don't overlap, there's statistical evidence
      that one model performs better than the other.
    - Wider intervals suggest higher uncertainty, often due to limited sample size or
      high variability in the data.

    Args:
        y_true: List of true binary labels
        y_pred: List of predicted binary labels
        metric_func: Function that calculates the desired metric (e.g., calculate_f1, calculate_kappa)
        n_bootstrap: Number of bootstrap samples to generate (default: 2000)
        confidence: Confidence level for the interval (default: 0.95 for 95% confidence)

    Returns:
        Tuple containing the lower and upper bounds of the confidence interval
    """
    n = len(y_true)
    if n == 0:
        return None

    # Convert inputs to numpy arrays for faster processing
    y_true_arr = np.array(y_true, dtype=bool)
    y_pred_arr = np.array(y_pred, dtype=bool)

    # Generate bootstrap sample indices using Numba-accelerated function
    all_indices = _bootstrap_sample_indices(n, n_bootstrap)

    # Calculate metric on each bootstrap sample
    bootstrap_metrics = []
    for i in range(n_bootstrap):
        indices = all_indices[i]
        bootstrap_true = y_true_arr[indices]
        bootstrap_pred = y_pred_arr[indices]

        # Calculate metric on this bootstrap sample
        metric_value = metric_func(bootstrap_true, bootstrap_pred)
        bootstrap_metrics.append(metric_value)

    # Calculate confidence interval
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100
    lower_bound = np.percentile(bootstrap_metrics, lower_percentile)
    upper_bound = np.percentile(bootstrap_metrics, upper_percentile)

    return float(lower_bound), float(upper_bound)


@jit(nopython=True)
def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy: (TP + TN) / (TP + TN + FP + FN)"""
    if len(y_true) == 0:
        return 0.0
    return float(np.sum(y_true == y_pred) / len(y_true))


@jit(nopython=True)
def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """Calculate precision: TP / (TP + FP)"""
    true_positives = np.sum(np.logical_and(y_pred, y_true))
    predicted_positives = np.sum(y_pred)

    if predicted_positives == 0:
        return zero_division
    return float(true_positives / predicted_positives)


@jit(nopython=True)
def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """Calculate recall: TP / (TP + FN)"""
    true_positives = np.sum(np.logical_and(y_pred, y_true))
    actual_positives = np.sum(y_true)

    if actual_positives == 0:
        return zero_division
    return float(true_positives / actual_positives)


@jit(nopython=True)
def calculate_f1(y_true: np.ndarray, y_pred: np.ndarray, zero_division: float = 0.0) -> float:
    """Calculate F1 score: 2 * (precision * recall) / (precision + recall)"""
    precision = calculate_precision(y_true, y_pred, zero_division)
    recall = calculate_recall(y_true, y_pred, zero_division)

    if precision + recall == 0:
        return zero_division
    return float(2 * precision * recall / (precision + recall))


@jit(nopython=True)
def calculate_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Cohen's kappa coefficient"""
    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    # Calculate observed agreement (accuracy)
    observed_agreement = np.sum(y_true == y_pred) / n_samples

    # Calculate expected agreement (probability of random agreement)
    true_pos_rate = np.mean(y_true)
    true_neg_rate = 1 - true_pos_rate
    pred_pos_rate = np.mean(y_pred)
    pred_neg_rate = 1 - pred_pos_rate

    expected_agreement = (true_pos_rate * pred_pos_rate) + (true_neg_rate * pred_neg_rate)

    # Calculate kappa
    if expected_agreement == 1:
        return 1.0

    return float((observed_agreement - expected_agreement) / (1 - expected_agreement))


def calculate_binary_metrics(
        y_true: List[bool],
        y_pred: List[bool]
) -> BinaryMetrics:
    """
    Calculate binary classification metrics using custom implementations.

    This function computes various classification metrics and their confidence intervals
    for evaluating binary classification performance.

    Args:
        y_true: List of true binary labels
        y_pred: List of predicted binary labels

    Returns:
        BinaryMetrics object containing point estimates and confidence intervals
    """
    y_true_arr = np.array(y_true, dtype=bool)
    y_pred_arr = np.array(y_pred, dtype=bool)

    # Calculate point estimates using custom functions
    accuracy = calculate_accuracy(y_true_arr, y_pred_arr)
    precision = calculate_precision(y_true_arr, y_pred_arr, zero_division=0.0)
    recall = calculate_recall(y_true_arr, y_pred_arr, zero_division=0.0)
    f1 = calculate_f1(y_true_arr, y_pred_arr, zero_division=0.0)
    kappa = calculate_kappa(y_true_arr, y_pred_arr)

    # Calculate confidence intervals using bootstrap
    accuracy_ci = bootstrap_confidence_interval(
        y_true, y_pred,
        lambda t, p: calculate_accuracy(np.array(t, dtype=bool), np.array(p, dtype=bool))
    )

    precision_ci = bootstrap_confidence_interval(
        y_true, y_pred,
        lambda t, p: calculate_precision(np.array(t, dtype=bool), np.array(p, dtype=bool), zero_division=0.0)
    )

    recall_ci = bootstrap_confidence_interval(
        y_true, y_pred,
        lambda t, p: calculate_recall(np.array(t, dtype=bool), np.array(p, dtype=bool), zero_division=0.0)
    )

    f1_ci = bootstrap_confidence_interval(
        y_true, y_pred,
        lambda t, p: calculate_f1(np.array(t, dtype=bool), np.array(p, dtype=bool), zero_division=0.0)
    )

    kappa_ci = bootstrap_confidence_interval(
        y_true, y_pred,
        lambda t, p: calculate_kappa(np.array(t, dtype=bool), np.array(p, dtype=bool))
    )

    return BinaryMetrics(
        accuracy=BinaryMetric(value=float(accuracy), ci=accuracy_ci),
        precision=BinaryMetric(value=float(precision), ci=precision_ci),
        recall=BinaryMetric(value=float(recall), ci=recall_ci),
        f1=BinaryMetric(value=float(f1), ci=f1_ci),
        kappa=BinaryMetric(value=float(kappa), ci=kappa_ci),
        n_samples=len(y_true),
    )
