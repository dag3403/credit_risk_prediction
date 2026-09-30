import numpy as np
import pytest

from src.credit_risk.evaluation import (
    build_capacity_table,
    capacity_metrics,
    capacity_threshold,
    choose_threshold_for_recall,
)


def test_capacity_metrics_are_exact_top_k_and_independent_of_threshold():
    actuals = np.array([1, 0, 1, 0, 1])
    scores = np.array([0.4, 0.99, 0.8, 0.2, 0.7])

    result = capacity_metrics(actuals, scores, 2)

    assert result["TP"] == 1
    assert result["FP"] == 1
    assert result["FN"] == 2
    assert result["TN"] == 1
    assert result["Recall@K"] == pytest.approx(1 / 3)
    assert result["Precision@K"] == pytest.approx(1 / 2)
    assert result["Review Rate"] == pytest.approx(2 / 5)


def test_capacity_metrics_caps_capacity_to_available_records():
    result = capacity_metrics([1, 0], [0.2, 0.1], 500)

    assert result["Capacity"] == 2
    assert result["Review Rate"] == 1
    assert result["FN"] == 0


def test_capacity_metrics_rejects_invalid_capacity():
    with pytest.raises(ValueError, match="positive"):
        capacity_metrics([1], [0.4], 0)


def test_capacity_table_keeps_model_split_and_capacity_columns():
    result = build_capacity_table(
        [1, 0, 1],
        {"model-a": [0.8, 0.7, 0.6]},
        [1, 2],
        split="validation",
    )

    assert list(result["Split"].unique()) == ["validation"]
    assert list(result["Capacity"]) == [1, 2]
    assert list(result["Model"].unique()) == ["model-a"]


def test_recall_threshold_uses_highest_validation_cutoff_meeting_goal():
    threshold, metrics = choose_threshold_for_recall(
        [1, 1, 0, 0],
        [0.9, 0.7, 0.8, 0.1],
        0.5,
    )

    assert threshold == pytest.approx(0.9)
    assert metrics is not None
    assert metrics["Recall"] >= 0.5
    assert metrics["Alerts"] == 1


def test_no_positive_validation_samples_cannot_meet_positive_recall_goal():
    threshold, metrics = choose_threshold_for_recall(
        [0, 0, 0],
        [0.1, 0.9, 0.8],
        0.9,
    )

    assert threshold is None
    assert metrics is None


def test_capacity_threshold_is_the_kth_largest_score():
    assert capacity_threshold([0.1, 0.7, 0.2, 0.9], 2) == pytest.approx(0.7)
