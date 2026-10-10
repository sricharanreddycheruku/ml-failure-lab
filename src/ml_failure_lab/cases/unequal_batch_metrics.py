"""A mean of batch accuracies does not generally measure record accuracy."""

import math

import numpy as np
from sklearn.metrics import accuracy_score


def fixture() -> list[tuple[np.ndarray, np.ndarray]]:
    return [(np.zeros(9, dtype=int), np.zeros(9, dtype=int)), (np.array([1]), np.array([0]))]


def report(approach: str, batches: list[tuple[np.ndarray, np.ndarray]] | None = None) -> dict:
    if approach not in {"bad", "fixed"}:
        raise ValueError("Choose 'bad' or 'fixed'.")
    batches = fixture() if batches is None else batches
    correct = records = 0
    batch_accuracies = []
    for targets, predictions in batches:
        targets, predictions = np.asarray(targets), np.asarray(predictions)
        if targets.ndim != 1 or predictions.shape != targets.shape:
            raise ValueError("Each batch needs matching one-dimensional targets and predictions.")
        if not targets.size:
            continue
        count = int(np.count_nonzero(targets == predictions))
        correct += count
        records += targets.size
        batch_accuracies.append(count / targets.size)
    if not records:
        raise ValueError("Evaluation needs at least one record; empty batches are ignored.")
    accuracy = np.mean(batch_accuracies) if approach == "bad" else correct / records
    return {
        "approach": approach,
        "accuracy": float(accuracy),
        "correct_records": correct,
        "total_records": int(records),
        "batch_accuracies": batch_accuracies,
    }


def require_record_accuracy(result: dict, batches: list[tuple[np.ndarray, np.ndarray]]) -> None:
    targets = np.concatenate([batch[0] for batch in batches])
    predictions = np.concatenate([batch[1] for batch in batches])
    if not targets.size:
        raise ValueError("Evaluation needs at least one record.")
    expected = float(accuracy_score(targets, predictions))
    if not math.isclose(result["accuracy"], expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(
            f"Record-weighted accuracy must be {expected:.0%}, not {result['accuracy']:.0%}."
        )


def run() -> dict:
    return {
        "case": "unequal-batch-metrics",
        "question": "Does averaging unequal batch accuracies give accuracy over all records?",
        "measurements": [report("bad"), report("fixed")],
        "requirement": "Accuracy must count correct predictions over all evaluated records.",
        "interpretation": (
            "Nine correct predictions in a batch of nine and zero in a batch of one give "
            "50% mean batch accuracy, but 9/10 = 90% record accuracy. The predictions did "
            "not change; only the aggregation did."
        ),
        "limitation": (
            "Equal weighting of batches can answer a different question. This correction "
            "applies to record accuracy, not averaging non-additive metrics such as F1 or AUC."
        ),
    }


def verify(approach: str) -> None:
    batches = fixture()
    require_record_accuracy(report(approach, batches), batches)
