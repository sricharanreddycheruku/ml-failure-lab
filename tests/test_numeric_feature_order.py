"""Tests for the numeric feature order case."""

import pytest

from ml_failure_lab.cases import numeric_feature_order as case


def test_bad_approach_produces_wrong_prediction_and_is_rejected():
    result = case.report("bad")
    assert result["input_columns"] == ["b", "a"]
    assert result["model_columns"] == ["a", "b"]
    assert result["prediction"] == pytest.approx(12.0)
    assert result["expected_prediction"] == pytest.approx(21.0)
    assert not result["columns_aligned"]

    with pytest.raises(ValueError, match="aligned to model schema"):
        case.verify("bad")


def test_fixed_approach_reorders_columns_and_meets_requirement():
    result = case.report("fixed")
    assert result["input_columns"] == ["b", "a"]
    assert result["model_columns"] == ["a", "b"]
    assert result["prediction"] == pytest.approx(21.0)
    assert result["expected_prediction"] == pytest.approx(21.0)
    assert result["columns_aligned"]

    case.verify("fixed")


def test_align_features_handles_arbitrary_key_order():
    # Input provided with columns reversed
    input_dict = {"b": 5.0, "a": 3.0}
    cols, array = case.align_features(input_dict, ["a", "b"])
    assert cols == ["a", "b"]
    assert array.tolist() == [[3.0, 5.0]]


def test_align_features_rejects_missing_column():
    input_dict = {"a": 1.0}
    with pytest.raises(ValueError, match="Missing required feature columns: \\['b'\\]"):
        case.align_features(input_dict, ["a", "b"])


def test_align_features_rejects_unrecognized_column():
    input_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    with pytest.raises(ValueError, match="Unrecognized feature columns: \\['c'\\]"):
        case.align_features(input_dict, ["a", "b"])


def test_align_features_rejects_duplicate_columns_in_sequence():
    pairs = [("a", 1.0), ("b", 2.0), ("a", 3.0)]
    with pytest.raises(ValueError, match="Duplicate feature names are not permitted"):
        case.align_features(pairs, ["a", "b"])


def test_manual_calculation_matches_aligned_prediction():
    test_cases = [
        ({"a": 2.0, "b": 4.0}, 42.0),
        ({"b": 10.0, "a": 0.5}, 100.5),
        ({"a": -1.0, "b": 3.0}, 29.0),
    ]
    for test_row, expected_val in test_cases:
        res = case.report("fixed", test_input=test_row)
        assert res["prediction"] == pytest.approx(expected_val, abs=1e-5)
        assert res["columns_aligned"]
