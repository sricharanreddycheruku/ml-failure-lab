"""Tests for the future-period-split case."""

import pytest

from ml_failure_lab.cases import future_period_split as case


def test_fixture_declares_a_late_label_and_a_label_known_exactly_at_freeze():
    rows = {row.record_id: row for row in case.records()}
    assert case.FREEZE_TIME == 8
    assert rows["r06"].prediction_time == 6
    assert rows["r06"].label_time == 11
    assert rows["r07"].label_time == case.FREEZE_TIME
    assert rows["r08"].label_time == 9
    assert [row.label for row in case.records()] == [
        6.0,
        10.0,
        8.0,
        12.0,
        10.0,
        20.0,
        12.0,
        22.0,
        20.0,
        24.0,
        22.0,
        26.0,
    ]


def test_shuffled_split_fits_later_labels_and_evaluates_earlier_records():
    result = case.report("bad")
    assert sorted(result["fit_ids"]) == ["r01", "r02", "r03", "r05", "r07", "r08", "r11", "r12"]
    assert sorted(result["evaluation_ids"]) == ["r04", "r06", "r09", "r10"]
    assert max(result["fit_label_times"]) == 13
    assert min(result["evaluation_prediction_times"]) == 4

    with pytest.raises(ValueError, match=r"r12 \(label at 13\), r08 \(label at 9\)"):
        case.verify("bad")


def test_prediction_time_split_is_rejected_because_of_late_labels():
    result = case.report("row-time")
    assert result["fit_prediction_times"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert result["evaluation_prediction_times"] == [9, 10, 11, 12]

    with pytest.raises(ValueError, match=r"r06 \(label at 11\), r08 \(label at 9\)"):
        case.verify("row-time")


def test_frozen_model_fits_only_labels_known_by_freeze_time():
    result = case.report("fixed")
    assert result["fit_ids"] == ["r01", "r02", "r03", "r04", "r05", "r07"]
    assert max(result["fit_label_times"]) <= case.FREEZE_TIME
    assert result["evaluation_ids"] == ["r09", "r10", "r11", "r12"]
    assert min(result["evaluation_prediction_times"]) > case.FREEZE_TIME
    case.verify("fixed")


def test_frozen_model_values_match_manual_least_squares():
    # Fit rows lie exactly on label = 2 * feature, so the slope is 2 and the intercept 0.
    # Evaluation features [7, 9, 8, 10] then give predictions [14, 18, 16, 20] against
    # labels [20, 24, 22, 26], an absolute error of 6 for each record.
    result = case.report("fixed")
    assert result["slope"] == pytest.approx(2.0, abs=1e-9)
    assert result["intercept"] == pytest.approx(0.0, abs=1e-9)
    assert result["predictions"] == pytest.approx([14.0, 18.0, 16.0, 20.0], abs=1e-9)
    assert result["labels"] == [20.0, 24.0, 22.0, 26.0]
    assert result["mae"] == pytest.approx(6.0, abs=1e-9)


def test_labels_recorded_after_freeze_cannot_change_the_frozen_model():
    changed = {"r06": -50.0, "r08": -50.0, "r09": 0.0, "r12": 99.0}
    original = case.report("fixed")
    altered = case.report("fixed", label_overrides=changed)
    assert altered["slope"] == pytest.approx(original["slope"], abs=1e-9)
    assert altered["intercept"] == pytest.approx(original["intercept"], abs=1e-9)
    assert altered["predictions"] == pytest.approx(original["predictions"], abs=1e-9)

    # The prediction-time split reads r06 and r08, so the same change moves its model.
    leaky = case.report("row-time")
    leaky_altered = case.report("row-time", label_overrides=changed)
    assert leaky_altered["slope"] != pytest.approx(leaky["slope"], abs=1e-6)


def test_label_known_exactly_at_freeze_is_allowed_and_one_step_later_is_not():
    result = case.report("fixed")
    assert "r07" in result["fit_ids"]
    case.require_labels_known_at_freeze(result)

    late = dict(result, fit_label_times=[*result["fit_label_times"][:-1], case.FREEZE_TIME + 1])
    with pytest.raises(ValueError, match=r"r07 \(label at 9\)"):
        case.require_labels_known_at_freeze(late)


def test_requirement_rejects_evaluation_at_or_before_freeze_and_shared_records():
    result = case.report("fixed")
    early = dict(
        result,
        evaluation_prediction_times=[case.FREEZE_TIME, *result["evaluation_prediction_times"][1:]],
    )
    with pytest.raises(ValueError, match=r"r09 \(predicted at 8\)"):
        case.require_labels_known_at_freeze(early)

    shared = dict(result, evaluation_ids=["r01", "r10", "r11", "r12"])
    with pytest.raises(ValueError, match="both fitted and evaluated"):
        case.require_labels_known_at_freeze(shared)


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError, match="Choose one of"):
        case.report("random")
    with pytest.raises(ValueError, match="Unknown record IDs"):
        case.records({"r99": 1.0})
    with pytest.raises(ValueError, match="At least two training rows"):
        case.AuditedRegression().fit(case.records()[:1])
