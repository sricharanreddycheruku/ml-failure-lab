"""A paired model comparison needs the same held-out records for each candidate."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Record:
    record_id: int
    target: float


@dataclass(frozen=True)
class EvaluatedRecord:
    record_id: int
    target: float
    prediction: float


@dataclass(frozen=True)
class CandidateResult:
    candidate: str
    evaluated_records: tuple[EvaluatedRecord, ...]
    mse: float


def evaluate(
    candidate: str, records: Sequence[Record], predictions_by_id: Mapping[int, float]
) -> CandidateResult:
    """Keep each prediction aligned with its target by record ID, not list position."""
    if not records:
        raise ValueError("Evaluation requires at least one record.")
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Evaluation cannot contain duplicate record IDs.")
    evaluated = tuple(
        EvaluatedRecord(record.record_id, record.target, predictions_by_id[record.record_id])
        for record in records
    )
    mse = sum((row.target - row.prediction) ** 2 for row in evaluated) / len(evaluated)
    return CandidateResult(candidate, evaluated, mse)


def targets_by_id(result: CandidateResult) -> dict[int, float]:
    """Build an auditable ID-to-target mapping; reject ambiguous duplicate IDs."""
    targets = {}
    for row in result.evaluated_records:
        if row.record_id in targets:
            raise ValueError("A candidate result contains duplicate record IDs.")
        targets[row.record_id] = row.target
    if not targets:
        raise ValueError("A candidate result must contain evaluated records.")
    return targets


def require_same_evaluation_records(first: CandidateResult, second: CandidateResult) -> None:
    """Require identical held-out ID sets and the same target for each ID.

    Results may list records in different orders, because evaluations are aligned
    by record ID rather than by position.
    """
    first_targets = targets_by_id(first)
    second_targets = targets_by_id(second)
    if first_targets.keys() != second_targets.keys():
        raise ValueError("A paired comparison must evaluate the same held-out record IDs.")
    if first_targets != second_targets:
        raise ValueError("A paired comparison must use the same target for each record ID.")


def experiment(approach: str) -> tuple[CandidateResult, CandidateResult]:
    if approach not in {"bad", "fixed"}:
        raise ValueError(f"Unknown approach: {approach}")

    # Original synthetic predictions, not results from training a model.
    records = tuple(Record(record_id, 0.0) for record_id in (101, 102, 103, 104))
    predictions_a = {101: 0.0, 102: 0.0, 103: 0.0, 104: 3.0}
    predictions_b = {101: 1.0, 102: 1.0, 103: 1.0, 104: 1.0}

    # Cherry-picking A's first three easy records makes its error look perfect.
    evaluated_a = records[:3] if approach == "bad" else records
    return (
        evaluate("A", evaluated_a, predictions_a),
        evaluate("B", records, predictions_b),
    )


def run() -> dict:
    measurements = []
    for approach in ("bad", "fixed"):
        candidate_a, candidate_b = experiment(approach)
        try:
            require_same_evaluation_records(candidate_a, candidate_b)
        except ValueError:
            comparable = False
        else:
            comparable = True
        measurements.append(
            {
                "approach": approach,
                "candidate_a": asdict(candidate_a),
                "candidate_b": asdict(candidate_b),
                "paired_comparison_valid": comparable,
                "apparent_winner": "A" if candidate_a.mse < candidate_b.mse else "B",
            }
        )

    return {
        "case": "different-comparison-records",
        "question": "Were candidates A and B scored on the same held-out records?",
        "measurements": measurements,
        "requirement": (
            "A paired comparison must use the same evaluated record IDs and the same target "
            "for each ID, regardless of record order."
        ),
        "interpretation": (
            "Comparing A on only three easy records (MSE 0) against B on four (MSE 1) "
            "misleadingly favors A. On all four records, A's MSE is 2.25 and B's is 1; "
            "B performs better for this declared paired benchmark."
        ),
        "limitation": (
            "These are synthetic stored predictions, not trained models. Separate independent "
            "test sets can be valid for other study designs; this check applies to a paired "
            "comparison on the same held-out records."
        ),
    }


def verify(approach: str) -> None:
    require_same_evaluation_records(*experiment(approach))
