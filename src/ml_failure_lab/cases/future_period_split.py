"""A model frozen before a future period cannot learn from labels that arrive later."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LinearRegression

FREEZE_TIME = 8
SHUFFLE_SEED = 3
APPROACHES = ("bad", "row-time", "fixed")


@dataclass(frozen=True)
class Record:
    record_id: str
    prediction_time: int
    feature: float
    label: float
    label_time: int


def records(label_overrides: dict[str, float] | None = None) -> list[Record]:
    """Twelve daily records. A label normally arrives one day after its prediction.

    A change on day 9 raises every outcome recorded from then on by 6, so the target is
    2 * feature before it and 2 * feature + 6 after it. Record r06 is predicted on day 6, but
    its label is only recorded on day 11, after the model is frozen.
    """
    overrides = label_overrides or {}
    features = [3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0, 7.0, 9.0, 8.0, 10.0]
    rows = []
    for day, feature in enumerate(features, start=1):
        record_id = f"r{day:02d}"
        label_time = 11 if record_id == "r06" else day + 1
        label = 2.0 * feature + (6.0 if label_time > FREEZE_TIME else 0.0)
        rows.append(Record(record_id, day, feature, overrides.get(record_id, label), label_time))
    unknown = set(overrides) - {row.record_id for row in rows}
    if unknown:
        raise ValueError(f"Unknown record IDs in label overrides: {sorted(unknown)}.")
    return rows


def shuffled_split(rows: list[Record]) -> tuple[list[Record], list[Record]]:
    """Misleading split: a random permutation, as if records were exchangeable."""
    order = np.random.default_rng(SHUFFLE_SEED).permutation(len(rows))
    shuffled = [rows[index] for index in order]
    return shuffled[:8], shuffled[8:]


def row_time_split(rows: list[Record]) -> tuple[list[Record], list[Record]]:
    """Chronological by prediction time only; it ignores when each label became known."""
    train = [row for row in rows if row.prediction_time <= FREEZE_TIME]
    evaluation = [row for row in rows if row.prediction_time > FREEZE_TIME]
    return train, evaluation


def frozen_model_split(rows: list[Record]) -> tuple[list[Record], list[Record]]:
    """Correction: fit only labels known by the freeze time; evaluate the later period."""
    train = [row for row in rows if row.label_time <= FREEZE_TIME]
    evaluation = [row for row in rows if row.prediction_time > FREEZE_TIME]
    return train, evaluation


SPLITS = {"bad": shuffled_split, "row-time": row_time_split, "fixed": frozen_model_split}


class AuditedRegression:
    """Linear regression that records exactly which rows were passed to ``fit``."""

    def __init__(self) -> None:
        self.model = LinearRegression()
        self.fit_rows: list[Record] = []

    def fit(self, rows: list[Record]) -> "AuditedRegression":
        if len(rows) < 2:
            raise ValueError("At least two training rows are needed to fit the model.")
        self.fit_rows = list(rows)
        features = np.array([[row.feature] for row in rows])
        labels = np.array([row.label for row in rows])
        self.model.fit(features, labels)
        return self

    def predict(self, rows: list[Record]) -> list[float]:
        return self.model.predict(np.array([[row.feature] for row in rows])).tolist()


def report(approach: str, label_overrides: dict[str, float] | None = None) -> dict:
    if approach not in SPLITS:
        raise ValueError(f"Choose one of {', '.join(APPROACHES)}.")
    train, evaluation = SPLITS[approach](records(label_overrides))
    regression = AuditedRegression().fit(train)
    predictions = regression.predict(evaluation)
    errors = [abs(pred - row.label) for pred, row in zip(predictions, evaluation, strict=True)]
    fitted = regression.fit_rows
    return {
        "approach": approach,
        "freeze_time": FREEZE_TIME,
        "fit_ids": [row.record_id for row in fitted],
        "fit_prediction_times": [row.prediction_time for row in fitted],
        "fit_label_times": [row.label_time for row in fitted],
        "evaluation_ids": [row.record_id for row in evaluation],
        "evaluation_prediction_times": [row.prediction_time for row in evaluation],
        "slope": round(float(regression.model.coef_[0]), 6) + 0.0,
        "intercept": round(float(regression.model.intercept_), 6) + 0.0,
        "predictions": [round(value, 6) + 0.0 for value in predictions],
        "labels": [row.label for row in evaluation],
        "mae": round(sum(errors) / len(errors), 6),
    }


def require_labels_known_at_freeze(result: dict) -> None:
    """Every fitted label must be known by the freeze time, before any evaluated prediction."""
    freeze = result["freeze_time"]
    late = [
        f"{record_id} (label at {label_time})"
        for record_id, label_time in zip(result["fit_ids"], result["fit_label_times"], strict=True)
        if label_time > freeze
    ]
    if late:
        raise ValueError(
            f"The model frozen at time {freeze} was fitted with labels recorded later: "
            f"{', '.join(late)}."
        )
    early = [
        f"{record_id} (predicted at {time})"
        for record_id, time in zip(
            result["evaluation_ids"], result["evaluation_prediction_times"], strict=True
        )
        if time <= freeze
    ]
    if early:
        raise ValueError(
            f"Evaluated records must be predicted after the freeze time {freeze}, "
            f"but these were not: {', '.join(early)}."
        )
    overlap = set(result["fit_ids"]) & set(result["evaluation_ids"])
    if overlap:
        raise ValueError(f"Records were both fitted and evaluated: {sorted(overlap)}.")


def run() -> dict:
    return {
        "case": "future-period-split",
        "question": (
            "Does the evaluation describe a model frozen at time 8 and used for later predictions?"
        ),
        "measurements": [report(approach) for approach in APPROACHES],
        "requirement": (
            "Every label used to fit must be recorded by the freeze time, and every evaluated "
            "record must be predicted after it."
        ),
        "interpretation": (
            "The shuffled split fits r08, r11 and r12, whose labels arrive after the freeze, and "
            "evaluates r04 and r06, which were predicted before it. Splitting by prediction time "
            "still fits r06 and r08, so the model already reflects the day 9 change. The "
            "correction fits the six labels known by time 8 and evaluates days 9 to 12. Its "
            "larger error is the honest result for this model, not a defect of the split."
        ),
        "limitation": (
            "This checks one frozen model and one forecast period using declared timestamps. It "
            "cannot discover wrong timestamps, and it does not describe walk-forward refitting. "
            "A random split can be valid when records are exchangeable."
        ),
    }


def verify(approach: str) -> None:
    require_labels_known_at_freeze(report(approach))
