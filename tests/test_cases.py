import pytest

from ml_failure_lab.cases import feature_selection, group_leakage, minority_recall
from ml_failure_lab.registry import CASES


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_bad_approach_violates_declared_requirement(case):
    with pytest.raises(ValueError):
        case.verify("bad")


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_fixed_approach_meets_declared_requirement(case):
    case.verify("fixed")


def test_group_fixture_exposes_recognition_instead_of_new_entity_prediction():
    result = group_leakage.run()["measurements"]
    assert result[0]["overlapping_entities"] == 40
    assert result[0]["accuracy"] == 1.0
    assert result[1]["overlapping_entities"] == 0
    assert result[1]["accuracy"] == 0.5


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_selector_audit_tracks_actual_test_rows_in_fit_input(seed):
    bad, test_ids, bad_score = feature_selection.experiment("bad", seed)
    fixed, fixed_test_ids, fixed_score = feature_selection.experiment("fixed", seed)
    assert len(bad.fitted_rows & set(test_ids)) == 50
    assert not fixed.fitted_rows & set(fixed_test_ids)
    assert len(fixed.fitted_rows) == 150
    assert 0 <= bad_score <= 1
    assert 0 <= fixed_score <= 1
    # A score ordering is deliberately not part of the evaluation requirement.


def test_minority_report_exposes_zero_detected_examples_without_changing_accuracy():
    bad = minority_recall.report("bad")
    fixed = minority_recall.report("fixed")
    assert bad["accuracy"] == fixed["accuracy"] == 0.99
    assert fixed["minority_recall"] == 0.0
    assert fixed["balanced_accuracy"] == 0.5
    assert fixed["actual_minority_records"] == 5
    assert fixed["detected_minority_records"] == 0
