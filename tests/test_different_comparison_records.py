"""Regressions for comparisons on different held-out records."""

import pytest

from ml_failure_lab.cases import different_comparison_records as case


def test_bad_comparison_uses_different_records_and_is_rejected():
    a, b = case.experiment("bad")
    assert [row.record_id for row in a.evaluated_records] == [101, 102, 103]
    assert [row.record_id for row in b.evaluated_records] == [101, 102, 103, 104]
    assert a.mse == pytest.approx(0.0)
    assert b.mse == pytest.approx(1.0)
    with pytest.raises(ValueError, match="same held-out record IDs"):
        case.verify("bad")


def test_fixed_comparison_scores_exactly_the_same_records():
    a, b = case.experiment("fixed")
    assert case.targets_by_id(a) == case.targets_by_id(b) == {
        101: 0.0,
        102: 0.0,
        103: 0.0,
        104: 0.0,
    }
    assert [row.prediction for row in a.evaluated_records] == [0.0, 0.0, 0.0, 3.0]
    assert [row.prediction for row in b.evaluated_records] == [1.0] * 4
    assert a.mse == pytest.approx((0 + 0 + 0 + 9) / 4)
    assert b.mse == pytest.approx((1 + 1 + 1 + 1) / 4)
    case.verify("fixed")


def test_equal_length_but_different_ids_is_still_invalid():
    predictions = {101: 0.0, 102: 0.0, 103: 0.0}
    a = case.evaluate("A", [case.Record(101, 0.0), case.Record(102, 0.0)], predictions)
    b = case.evaluate("B", [case.Record(101, 0.0), case.Record(103, 0.0)], predictions)
    assert len(a.evaluated_records) == len(b.evaluated_records)
    with pytest.raises(ValueError, match="same held-out record IDs"):
        case.require_same_evaluation_records(a, b)


def test_record_order_does_not_change_truth_or_prediction_alignment():
    original = [case.Record(10, 2.0), case.Record(20, -1.0), case.Record(30, 5.0)]
    reordered = [original[2], original[0], original[1]]
    a = case.evaluate("A", original, {10: 3.0, 20: -1.0, 30: 3.0})
    b = case.evaluate("B", reordered, {10: 2.0, 20: -2.0, 30: 5.0})
    # Unlike the all-zero demonstration, these distinct targets expose bad alignment.
    assert [(row.record_id, row.target, row.prediction) for row in b.evaluated_records] == [
        (30, 5.0, 5.0),
        (10, 2.0, 2.0),
        (20, -1.0, -2.0),
    ]
    assert a.mse == pytest.approx(5 / 3)
    assert b.mse == pytest.approx(1 / 3)
    case.require_same_evaluation_records(a, b)


def test_different_targets_for_same_id_are_rejected():
    a = case.evaluate("A", [case.Record(5, 0.0)], {5: 1.0})
    b = case.evaluate("B", [case.Record(5, 1.0)], {5: 1.0})
    with pytest.raises(ValueError, match="same target"):
        case.require_same_evaluation_records(a, b)


def test_duplicate_or_empty_evaluated_ids_are_rejected():
    with pytest.raises(ValueError, match="at least one"):
        case.evaluate("A", [], {})
    with pytest.raises(ValueError, match="duplicate record IDs"):
        case.evaluate("A", [case.Record(1, 0.0), case.Record(1, 0.0)], {1: 0.0})


def test_run_shows_misleading_and_corrected_comparisons_side_by_side():
    result = case.run()
    assert result["case"] == "different-comparison-records"
    bad, fixed = result["measurements"]
    assert bad["approach"] == "bad"
    assert bad["paired_comparison_valid"] is False
    assert bad["apparent_winner"] == "A"
    assert bad["candidate_a"]["mse"] == pytest.approx(0.0)
    assert bad["candidate_b"]["mse"] == pytest.approx(1.0)
    assert fixed["approach"] == "fixed"
    assert fixed["paired_comparison_valid"] is True
    assert fixed["apparent_winner"] == "B"
    assert fixed["candidate_a"]["mse"] == pytest.approx(2.25)
    assert fixed["candidate_b"]["mse"] == pytest.approx(1.0)
    for comparison in (bad, fixed):
        for result_key in ("candidate_a", "candidate_b"):
            for row in comparison[result_key]["evaluated_records"]:
                assert {"record_id", "target", "prediction"} <= row.keys()
