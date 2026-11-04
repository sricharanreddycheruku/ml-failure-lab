"""Regressions for threshold tuning on test records."""

import pytest

from ml_failure_lab.cases import test_tuned_threshold as case


def test_bad_approach_uses_test_records_and_is_rejected():
    selection, test_records, final_acc = case.experiment("bad")
    test_ids = {r.record_id for r in test_records}
    assert selection.evaluated_ids == test_ids == {101, 102, 103, 104}
    assert selection.selected_threshold == 0.3
    assert selection.selection_accuracy == 1.0
    assert final_acc == 1.0

    with pytest.raises(ValueError, match="must not access final test records"):
        case.verify("bad")


def test_fixed_approach_uses_only_validation_records_and_passes():
    selection, test_records, final_acc = case.experiment("fixed")
    test_ids = {r.record_id for r in test_records}
    assert selection.evaluated_ids == {1, 2, 3, 4}
    assert not (selection.evaluated_ids & test_ids)
    assert selection.selected_threshold == 0.5
    assert selection.selection_accuracy == 1.0
    assert final_acc == 0.75

    case.verify("fixed")


def test_select_threshold_tie_breaking_picks_lowest_threshold():
    # If thresholds 0.3 and 0.5 give same accuracy, 0.3 should be chosen
    records = (
        case.Record(1, 0.2, 0),
        case.Record(2, 0.6, 1),
    )
    result = case.select_threshold(records, grid=(0.3, 0.5, 0.7))
    # At 0.3: (0, 1) -> 2/2 = 1.0
    # At 0.5: (0, 1) -> 2/2 = 1.0
    # At 0.7: (0, 0) -> 1/2 = 0.5
    assert result.selected_threshold == 0.3
    assert result.selection_accuracy == 1.0


def test_empty_or_duplicate_records_in_selection_are_rejected():
    with pytest.raises(ValueError, match="at least one record"):
        case.select_threshold([])

    with pytest.raises(ValueError, match="non-empty grid"):
        case.select_threshold(case.default_validation_records(), grid=())

    duplicate_records = (
        case.Record(1, 0.2, 0),
        case.Record(1, 0.4, 1),
    )
    with pytest.raises(ValueError, match="Duplicate record IDs"):
        case.select_threshold(duplicate_records)


def test_accuracy_at_threshold_requires_records():
    with pytest.raises(ValueError, match="Cannot calculate accuracy"):
        case.accuracy_at_threshold([], 0.5)


def test_default_test_records_requires_exactly_four_labels():
    with pytest.raises(ValueError, match="require exactly 4 labels"):
        case.default_test_records([0, 1])


def test_unknown_approach_is_rejected():
    with pytest.raises(ValueError, match="Unknown approach"):
        case.experiment("invalid")


def test_run_reports_bad_and_fixed_measurements():
    result = case.run()
    assert result["case"] == "test-tuned-threshold"
    assert "without touching final test labels" in result["question"]

    measurements = {m["approach"]: m for m in result["measurements"]}
    bad = measurements["bad"]
    fixed = measurements["fixed"]

    assert bad["selected_threshold"] == 0.3
    assert bad["selection_accuracy"] == 1.0
    assert bad["final_test_accuracy"] == 1.0
    assert bad["test_records_used_in_selection"] == 4
    assert bad["evaluation_contract_met"] is False

    assert fixed["selected_threshold"] == 0.5
    assert fixed["selection_accuracy"] == 1.0
    assert fixed["final_test_accuracy"] == 0.75
    assert fixed["test_records_used_in_selection"] == 0
    assert fixed["evaluation_contract_met"] is True


def test_changing_test_labels_does_not_affect_fixed_threshold_selection():
    # In the bad approach, changing test labels alters the selected threshold (leakage)
    bad_orig, _, _ = case.experiment("bad", test_labels=[0, 1, 1, 1])
    bad_alt, _, _ = case.experiment("bad", test_labels=[0, 0, 0, 0])
    assert bad_orig.selected_threshold == 0.3
    assert bad_alt.selected_threshold == 0.7

    # In the fixed approach, the selected threshold is decoupled from test labels
    fixed_orig, _, acc_orig = case.experiment("fixed", test_labels=[0, 1, 1, 1])
    fixed_alt, _, acc_alt = case.experiment("fixed", test_labels=[1, 1, 1, 1])
    assert fixed_orig.selected_threshold == 0.5
    assert fixed_alt.selected_threshold == 0.5
    assert acc_orig == 0.75
    assert acc_alt == 0.50
