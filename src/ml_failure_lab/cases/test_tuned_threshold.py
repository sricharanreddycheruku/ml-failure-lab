"""Tuning classification thresholds on final test labels contaminates evaluation."""

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Record:
    record_id: int
    score: float
    label: int


@dataclass(frozen=True)
class ThresholdSelectionResult:
    selected_threshold: float
    evaluated_ids: frozenset[int]
    selection_accuracy: float


GRID: tuple[float, ...] = (0.3, 0.5, 0.7)


def default_validation_records() -> tuple[Record, ...]:
    return (
        Record(1, 0.1, 0),
        Record(2, 0.4, 0),
        Record(3, 0.6, 1),
        Record(4, 0.9, 1),
    )


def default_test_records(test_labels: Sequence[int] | None = None) -> tuple[Record, ...]:
    # Default test distribution: label 1 occurs at score 0.4 and above.
    labels = test_labels if test_labels is not None else (0, 1, 1, 1)
    if len(labels) != 4:
        raise ValueError("Test records require exactly 4 labels.")
    return (
        Record(101, 0.2, labels[0]),
        Record(102, 0.4, labels[1]),
        Record(103, 0.6, labels[2]),
        Record(104, 0.8, labels[3]),
    )


def accuracy_at_threshold(records: Sequence[Record], threshold: float) -> float:
    """Classify as 1 if score >= threshold, else 0."""
    if not records:
        raise ValueError("Cannot calculate accuracy for empty records.")
    correct = sum((1 if r.score >= threshold else 0) == r.label for r in records)
    return correct / len(records)


def select_threshold(
    records: Sequence[Record], grid: Sequence[float] = GRID
) -> ThresholdSelectionResult:
    """Select the threshold with highest accuracy on records.

    Deterministic tie-breaking policy: If multiple thresholds achieve the same
    maximum accuracy, the lowest threshold is selected.
    """
    if not records:
        raise ValueError("Threshold selection requires at least one record.")
    if not grid:
        raise ValueError("Threshold selection requires a non-empty grid.")

    record_ids = [r.record_id for r in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Duplicate record IDs provided to threshold selection.")

    best_threshold = grid[0]
    best_accuracy = -1.0

    for threshold in grid:
        acc = accuracy_at_threshold(records, threshold)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    return ThresholdSelectionResult(
        selected_threshold=best_threshold,
        evaluated_ids=frozenset(record_ids),
        selection_accuracy=best_accuracy,
    )


def require_untouched_test_data(
    selection_result: ThresholdSelectionResult, test_records: Sequence[Record]
) -> None:
    """Audit the records accessed during threshold selection."""
    test_ids = {r.record_id for r in test_records}
    leaked_ids = selection_result.evaluated_ids & test_ids
    if leaked_ids:
        raise ValueError(
            f"Threshold selection must not access final test records. "
            f"Found test IDs in selection: {sorted(leaked_ids)}"
        )


def experiment(
    approach: str, test_labels: Sequence[int] | None = None
) -> tuple[ThresholdSelectionResult, tuple[Record, ...], float]:
    if approach not in {"bad", "fixed"}:
        raise ValueError(f"Unknown approach: {approach}")

    val_records = default_validation_records()
    test_records = default_test_records(test_labels)

    if approach == "bad":
        # Bad workflow: tunes threshold directly using final test records & labels.
        selection = select_threshold(test_records, GRID)
    else:
        # Fixed workflow: tunes threshold using only validation records.
        selection = select_threshold(val_records, GRID)

    # Freeze chosen threshold and evaluate final test records.
    final_test_accuracy = accuracy_at_threshold(test_records, selection.selected_threshold)
    return selection, test_records, final_test_accuracy


def run() -> dict:
    measurements = []
    for approach in ("bad", "fixed"):
        selection, test_records, final_test_acc = experiment(approach)
        try:
            require_untouched_test_data(selection, test_records)
            valid = True
        except ValueError:
            valid = False

        measurements.append(
            {
                "approach": approach,
                "selected_threshold": selection.selected_threshold,
                "selection_accuracy": selection.selection_accuracy,
                "final_test_accuracy": final_test_acc,
                "test_records_used_in_selection": len(
                    selection.evaluated_ids & {r.record_id for r in test_records}
                ),
                "evaluation_contract_met": valid,
            }
        )

    return {
        "case": "test-tuned-threshold",
        "question": "Was the classification threshold tuned without touching final test labels?",
        "measurements": measurements,
        "requirement": (
            "Threshold selection must use validation records only; final test records "
            "must remain completely untouched until evaluation."
        ),
        "interpretation": (
            "The bad workflow inspects final test labels to pick a threshold (0.3), "
            "producing an artificially inflated 100% test accuracy via test set leakage. "
            "The fixed workflow tunes the threshold on validation records alone (0.5), "
            "freezes it, and honestly evaluates final test records (75% accuracy)."
        ),
        "limitation": (
            "The synthetic probabilities and grid values illustrate evaluation data access. "
            "Fixing data access does not guarantee higher test performance; it provides an "
            "unbiased estimate."
        ),
    }


def verify(approach: str) -> None:
    selection, test_records, _ = experiment(approach)
    require_untouched_test_data(selection, test_records)
