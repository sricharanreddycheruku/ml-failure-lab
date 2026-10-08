"""A high accuracy score can coexist with zero detection of the target class."""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import train_test_split


def report(approach: str) -> dict:
    features = np.zeros((1000, 1))
    targets = np.array([0] * 990 + [1] * 10)
    train_x, test_x, train_y, test_y = train_test_split(
        features, targets, test_size=0.5, random_state=42, stratify=targets
    )
    model = DummyClassifier(strategy="most_frequent").fit(train_x, train_y)
    predictions = model.predict(test_x)
    result = {"approach": approach, "accuracy": float(accuracy_score(test_y, predictions))}
    if approach == "fixed":
        result.update(
            minority_recall=float(recall_score(test_y, predictions, pos_label=1)),
            balanced_accuracy=float(balanced_accuracy_score(test_y, predictions)),
            actual_minority_records=int(np.sum(test_y == 1)),
            detected_minority_records=int(np.sum((test_y == 1) & (predictions == 1))),
        )
    return result


def require_minority_detection_report(result: dict) -> None:
    if "minority_recall" not in result or "actual_minority_records" not in result:
        raise ValueError("This task requires reporting recall and support for the minority class.")


def run() -> dict:
    return {
        "case": "minority-recall",
        "question": "Does this accuracy report tell us whether the rare target class is detected?",
        "measurements": [report("bad"), report("fixed")],
        "requirement": (
            "For this detection task, report minority recall and the number of examples."
        ),
        "interpretation": (
            "Both reports describe the same majority-class model. It gets 99% accuracy "
            "and detects none of the five minority examples. The correction improves "
            "the report; it does not improve the classifier."
        ),
        "limitation": (
            "Metric choice depends on the task and error costs. This example does not "
            "claim that accuracy is always unsuitable for imbalanced data."
        ),
    }


def verify(approach: str) -> None:
    require_minority_detection_report(report(approach))
