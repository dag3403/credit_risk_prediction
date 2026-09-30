"""Operational ranking and validation-threshold evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def capacity_metrics(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    capacity: int,
) -> dict[str, int | float]:
    """Evaluate exact Top-K ranking performance, independent of a threshold."""
    actuals = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if actuals.ndim != 1 or scores.ndim != 1 or actuals.size != scores.size:
        raise ValueError("y_true and probabilities must be one-dimensional and aligned.")
    if actuals.size == 0:
        raise ValueError("At least one record is required for capacity evaluation.")
    if capacity <= 0:
        raise ValueError("Review capacity must be positive.")

    effective_capacity = min(int(capacity), actuals.size)
    ranking = np.argsort(-scores, kind="stable")
    selected_indices = ranking[:effective_capacity]
    selected = np.zeros(actuals.size, dtype=bool)
    selected[selected_indices] = True

    positives = actuals == 1
    negatives = actuals == 0
    tp = int(np.count_nonzero(selected & positives))
    fp = int(np.count_nonzero(selected & negatives))
    fn = int(np.count_nonzero(~selected & positives))
    tn = int(np.count_nonzero(~selected & negatives))
    total_positives = int(np.count_nonzero(positives))
    total_negatives = int(np.count_nonzero(negatives))

    return {
        "Capacity": effective_capacity,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Recall@K": tp / total_positives if total_positives else 0.0,
        "Precision@K": tp / effective_capacity,
        "Specificity": tn / total_negatives if total_negatives else 0.0,
        "Review Rate": effective_capacity / actuals.size,
        "Problematic Captured": tp,
        "Problematic Missed": fn,
        "False Positives / Extra Reviews": fp,
    }


def build_capacity_table(
    y_true: Sequence[int] | np.ndarray,
    probabilities_by_model: Mapping[str, Sequence[float] | np.ndarray],
    capacities: Sequence[int],
    *,
    split: str,
) -> pd.DataFrame:
    rows = []
    for model_name, probabilities in probabilities_by_model.items():
        for capacity in capacities:
            rows.append(
                {
                    "Model": model_name,
                    "Split": split,
                    **capacity_metrics(y_true, probabilities, capacity),
                }
            )
    return pd.DataFrame(rows)


def choose_threshold_for_recall(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    target_recall: float,
) -> tuple[float | None, dict[str, int | float] | None]:
    """Return the highest validation threshold meeting the requested recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1].")

    actuals = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    if actuals.ndim != 1 or scores.ndim != 1 or actuals.size != scores.size:
        raise ValueError("y_true and probabilities must be one-dimensional and aligned.")
    if actuals.size == 0:
        raise ValueError("At least one record is required for threshold selection.")

    positive_count = int(np.count_nonzero(actuals == 1))
    if positive_count == 0:
        return None, None

    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_actuals = actuals[order]
    group_starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    cumulative_true_positives = np.cumsum(sorted_actuals == 1)
    recall_by_threshold = cumulative_true_positives[group_starts] / positive_count
    eligible = np.flatnonzero(recall_by_threshold >= target_recall)
    if eligible.size == 0:
        return None, None

    threshold = float(sorted_scores[group_starts[eligible[0]]])
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actuals, predictions, labels=[0, 1]).ravel()
    recall = float(recall_score(actuals, predictions, zero_division=0))
    precision = float(precision_score(actuals, predictions, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    return threshold, {
        "Threshold": threshold,
        "Alerts": int(tp + fp),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Recall": recall,
        "Precision": precision,
        "Specificity": specificity,
        "F2": float(fbeta_score(actuals, predictions, beta=2, zero_division=0)),
        "PR-AUC": float(average_precision_score(actuals, scores)),
        "ROC-AUC": float(roc_auc_score(actuals, scores))
        if np.unique(actuals).size == 2
        else float("nan"),
        "Brier Score": float(brier_score_loss(actuals, scores)),
    }


def capacity_threshold(
    probabilities: Sequence[float] | np.ndarray, capacity: int
) -> float:
    """Use the validation K-th highest score as the capacity-derived threshold."""
    scores = np.asarray(probabilities, dtype=float)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("probabilities must contain at least one score.")
    if capacity <= 0:
        raise ValueError("Review capacity must be positive.")
    effective_capacity = min(int(capacity), scores.size)
    return float(np.sort(scores, kind="stable")[::-1][effective_capacity - 1])


def metrics_at_threshold(
    y_true: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    threshold: float,
) -> dict[str, int | float]:
    actuals = np.asarray(y_true, dtype=int)
    scores = np.asarray(probabilities, dtype=float)
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(actuals, predictions, labels=[0, 1]).ravel()
    total_negatives = int(np.count_nonzero(actuals == 0))
    return {
        "Threshold": float(threshold),
        "Alerts": int(tp + fp),
        "Review Rate": float((tp + fp) / actuals.size) if actuals.size else 0.0,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
        "Recall": float(recall_score(actuals, predictions, zero_division=0)),
        "Precision": float(precision_score(actuals, predictions, zero_division=0)),
        "Specificity": float(tn / total_negatives) if total_negatives else 0.0,
        "F2": float(fbeta_score(actuals, predictions, beta=2, zero_division=0)),
        "PR-AUC": float(average_precision_score(actuals, scores)),
        "ROC-AUC": float(roc_auc_score(actuals, scores))
        if np.unique(actuals).size == 2
        else float("nan"),
        "Brier Score": float(brier_score_loss(actuals, scores)),
    }


def build_ranked_predictions(
    row_indices: Sequence[int],
    y_true: Sequence[int] | np.ndarray,
    probabilities_by_model: Mapping[str, Sequence[float] | np.ndarray],
    *,
    split: str,
) -> pd.DataFrame:
    actuals = np.asarray(y_true, dtype=int)
    indices = np.asarray(row_indices)
    rows = []
    for model_name, raw_scores in probabilities_by_model.items():
        scores = np.asarray(raw_scores, dtype=float)
        order = np.argsort(-scores, kind="stable")
        for rank, position in enumerate(order, start=1):
            rows.append(
                {
                    "Model": model_name,
                    "Split": split,
                    "row_index": int(indices[position]),
                    "probability_not_fully_paid": float(scores[position]),
                    "actual_not_fully_paid": int(actuals[position]),
                    "rank": rank,
                }
            )
    return pd.DataFrame(rows)


def run_operational_evaluation(
    *,
    candidate_models: Sequence[str],
    models: Mapping[str, Mapping[str, object]],
    y_train: Sequence[int] | np.ndarray,
    y_validation: Sequence[int] | np.ndarray,
    y_test: Sequence[int] | np.ndarray,
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    cv: object,
    capacities: Sequence[int],
    target_recalls: Sequence[float],
    evaluation: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Compute validation-only selection inputs and locked test evaluations."""
    missing_models = set(candidate_models) - set(models)
    if missing_models:
        raise KeyError(f"Candidate models are missing from models: {sorted(missing_models)}")
    missing_results = set(candidate_models) - set(evaluation)
    if missing_results:
        raise KeyError(
            f"Candidate models are missing from evaluation: {sorted(missing_results)}"
        )

    validation_probabilities = {
        model_name: models[model_name]["estimator"].predict_proba(X_validation)[:, 1]
        for model_name in candidate_models
    }
    test_probabilities = {
        model_name: np.asarray(evaluation[model_name]["probabilities"], dtype=float)
        for model_name in candidate_models
    }
    cv_scores = {
        model_name: cross_val_score(
            models[model_name]["estimator"],
            X_train,
            y_train,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
        )
        for model_name in candidate_models
    }

    effective_capacities = sorted(
        {
            min(int(capacity), len(y_validation), len(y_test))
            for capacity in capacities
            if int(capacity) > 0
        }
    )
    if not effective_capacities:
        raise ValueError("At least one positive review capacity is required.")

    validation_capacity = build_capacity_table(
        y_validation,
        validation_probabilities,
        effective_capacities,
        split="validation",
    )
    test_capacity = build_capacity_table(
        y_test,
        test_probabilities,
        effective_capacities,
        split="test",
    )
    selection_capacity = min(1000, len(y_validation))
    selection_summary = validation_selection_table(
        y_validation,
        validation_probabilities,
        selection_capacity,
        cv_scores,
    )
    selected_model_name = str(selection_summary.iloc[0]["Model"])
    selected_validation_threshold = capacity_threshold(
        validation_probabilities[selected_model_name], selection_capacity
    )

    threshold_rows = []
    for model_name in candidate_models:
        validation_scores = validation_probabilities[model_name]
        test_scores = test_probabilities[model_name]
        for target_recall in target_recalls:
            threshold, validation_metrics = choose_threshold_for_recall(
                y_validation, validation_scores, target_recall
            )
            if threshold is None or validation_metrics is None:
                threshold_rows.append(
                    {
                        "Model": model_name,
                        "Target Recall": target_recall,
                        "Threshold (validation)": np.nan,
                        "Note": "No positive examples in validation.",
                    }
                )
                continue
            test_metrics = metrics_at_threshold(y_test, test_scores, threshold)
            threshold_rows.append(
                {
                    "Model": model_name,
                    "Target Recall": target_recall,
                    **{
                        f"Validation {key}": value
                        for key, value in validation_metrics.items()
                    },
                    **{
                        f"Test {key}": value
                        for key, value in test_metrics.items()
                    },
                    "Threshold (validation)": threshold,
                    "Note": "",
                }
            )
    recall_threshold_summary = pd.DataFrame(threshold_rows)

    capacity_threshold_rows = []
    for model_name in candidate_models:
        for capacity in effective_capacities:
            threshold = capacity_threshold(
                validation_probabilities[model_name], capacity
            )
            validation_metrics = metrics_at_threshold(
                y_validation, validation_probabilities[model_name], threshold
            )
            test_metrics = metrics_at_threshold(
                y_test, test_probabilities[model_name], threshold
            )
            capacity_threshold_rows.append(
                {
                    "Model": model_name,
                    "Capacity": capacity,
                    **{
                        f"Validation {key}": value
                        for key, value in validation_metrics.items()
                    },
                    **{
                        f"Test {key}": value
                        for key, value in test_metrics.items()
                    },
                }
            )
    capacity_threshold_summary = pd.DataFrame(capacity_threshold_rows)

    selection_validation_metrics = selection_summary.iloc[0].to_dict()
    selected_test_capacity = test_capacity.loc[
        (test_capacity["Model"] == selected_model_name)
        & (test_capacity["Capacity"] == selection_capacity)
    ].iloc[0]
    production_test_metrics = metrics_at_threshold(
        y_test, test_probabilities[selected_model_name], selected_validation_threshold
    )
    return {
        "candidate_models": list(candidate_models),
        "capacities": effective_capacities,
        "selection_capacity": selection_capacity,
        "validation_probabilities": validation_probabilities,
        "test_probabilities": test_probabilities,
        "cv_average_precision": cv_scores,
        "validation_capacity": validation_capacity,
        "test_capacity": test_capacity,
        "selection_summary": selection_summary,
        "selected_model_name": selected_model_name,
        "selected_validation_threshold": selected_validation_threshold,
        "selected_validation_metrics": selection_validation_metrics,
        "selected_test_capacity_metrics": selected_test_capacity.to_dict(),
        "production_test_metrics": production_test_metrics,
        "recall_threshold_summary": recall_threshold_summary,
        "capacity_threshold_summary": capacity_threshold_summary,
        "ranked_validation": build_ranked_predictions(
            validation_indices,
            y_validation,
            validation_probabilities,
            split="validation",
        ),
        "ranked_test": build_ranked_predictions(
            test_indices,
            y_test,
            test_probabilities,
            split="test",
        ),
    }


def validation_selection_table(
    y_validation: Sequence[int] | np.ndarray,
    validation_probabilities: Mapping[str, Sequence[float] | np.ndarray],
    capacity: int,
    cv_average_precision: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Rank model candidates on validation only; test metrics are excluded."""
    rows = []
    for model_name, scores in validation_probabilities.items():
        metrics = capacity_metrics(y_validation, scores, capacity)
        scores_array = np.asarray(scores, dtype=float)
        actuals = np.asarray(y_validation, dtype=int)
        cv_scores = np.asarray(cv_average_precision[model_name], dtype=float)
        rows.append(
            {
                "Model": model_name,
                **metrics,
                "PR-AUC": float(average_precision_score(actuals, scores_array)),
                "ROC-AUC": float(roc_auc_score(actuals, scores_array)),
                "CV PR-AUC mean": float(cv_scores.mean()),
                "CV PR-AUC std": float(cv_scores.std(ddof=1))
                if cv_scores.size > 1
                else 0.0,
                "F2@K": float(
                    fbeta_score(
                        actuals,
                        np.isin(
                            np.arange(actuals.size),
                            np.argsort(-scores_array, kind="stable")[
                                : min(int(capacity), actuals.size)
                            ],
                        ).astype(int),
                        beta=2,
                        zero_division=0,
                    )
                ),
                "Brier Score": float(brier_score_loss(actuals, scores_array)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        [
            "Recall@K",
            "FN",
            "FP",
            "PR-AUC",
            "Specificity",
            "Precision@K",
            "F2@K",
            "ROC-AUC",
            "Brier Score",
            "CV PR-AUC mean",
            "CV PR-AUC std",
        ],
        ascending=[
            False,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)
