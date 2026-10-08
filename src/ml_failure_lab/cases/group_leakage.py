"""Repeated records can answer the wrong generalization question."""

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Sample:
    entity: int
    feature: float
    label: int
    repeat: int


def samples() -> list[Sample]:
    rng = random.Random(42)
    seen_labels = [0, 1] * 15
    new_labels = [0, 1] * 5
    rng.shuffle(seen_labels)
    rng.shuffle(new_labels)
    return [
        Sample(entity, float(entity), label, repeat)
        for entity, label in enumerate(seen_labels + new_labels)
        for repeat in range(2)
    ]


def row_split(rows: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    return [row for row in rows if row.repeat == 0], [row for row in rows if row.repeat == 1]


def group_split(rows: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    return [row for row in rows if row.entity < 30], [row for row in rows if row.entity >= 30]


def overlapping_entities(train: list[Sample], test: list[Sample]) -> set[int]:
    return {row.entity for row in train} & {row.entity for row in test}


def require_new_entities(train: list[Sample], test: list[Sample]) -> None:
    if overlapping_entities(train, test):
        raise ValueError("Evaluation of new entities requires disjoint train and test entity IDs.")


def accuracy(train: list[Sample], test: list[Sample]) -> float:
    predictions = [
        min(train, key=lambda row: abs(row.feature - item.feature)).label for item in test
    ]
    return sum(label == item.label for label, item in zip(predictions, test, strict=True)) / len(
        test
    )


def run() -> dict:
    measurements = []
    for approach, split in [("bad", row_split), ("fixed", group_split)]:
        train, test = split(samples())
        measurements.append(
            {
                "approach": approach,
                "accuracy": accuracy(train, test),
                "train_records": len(train),
                "test_records": len(test),
                "overlapping_entities": len(overlapping_entities(train, test)),
            }
        )
    return {
        "case": "group-leakage",
        "question": "Does this evaluation measure predictions for previously unseen entities?",
        "measurements": measurements,
        "requirement": "Train and test entity IDs must be disjoint for this task.",
        "interpretation": (
            "The record split recognizes entities already present in training. "
            "The group split evaluates new entities. These scores answer different questions."
        ),
        "limitation": (
            "The exact 100% and 50% scores belong to this deliberately balanced toy fixture."
        ),
    }


def verify(approach: str) -> None:
    split = row_split if approach == "bad" else group_split
    require_new_entities(*split(samples()))
