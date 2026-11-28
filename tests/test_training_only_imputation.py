"""Tests for the training-only-imputation case."""

import numpy as np
import pytest

from ml_failure_lab.cases import training_only_imputation as case


def test_bad_approach_fits_all_rows_and_is_rejected():
    result = case.report("bad")
    assert result["raw_test_values"] == [100.0, None]
    assert result["applied_statistic"] == pytest.approx(43.333333333333336)
    assert result["training_statistic"] == pytest.approx(15.0)
    assert result["imputed_test_values"] == pytest.approx([100.0, 43.333333333333336])
    assert result["expected_test_values"] == pytest.approx([100.0, 15.0])
    assert not result["reused_training_parameters"]

    with pytest.raises(ValueError, match="training-only statistic is 15.0"):
        case.verify("bad")


def test_fixed_approach_fits_training_only_and_meets_requirement():
    result = case.report("fixed")
    assert result["raw_test_values"] == [100.0, None]
    assert result["applied_statistic"] == pytest.approx(15.0)
    assert result["training_statistic"] == pytest.approx(15.0)
    assert result["imputed_test_values"] == pytest.approx([100.0, 15.0])
    assert result["expected_test_values"] == pytest.approx([100.0, 15.0])
    assert result["reused_training_parameters"]

    case.verify("fixed")


def test_shifted_fixture_preserves_training_imputer_statistic():
    # Shifted test fixture [50.0, np.nan]
    result = case.report("fixed", test_data=[[50.0], [np.nan]])
    assert result["raw_test_values"] == [50.0, None]
    assert result["imputed_test_values"] == pytest.approx([50.0, 15.0])
    assert result["expected_test_values"] == pytest.approx([50.0, 15.0])
    assert result["reused_training_parameters"]
    case.require_training_imputer_statistics(result)


def test_invalid_input_dimensions_are_rejected():
    with pytest.raises(ValueError, match="two-dimensional array with one column"):
        case.report("fixed", test_data=[])
    with pytest.raises(ValueError, match="two-dimensional array with one column"):
        case.report("fixed", test_data=np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_mismatched_statistic_raises_in_require():
    mismatched_result = {
        "applied_statistic": 25.0,
        "training_statistic": 15.0,
        "imputed_test_values": [100.0, 15.0],
        "expected_test_values": [100.0, 15.0],
    }
    with pytest.raises(ValueError, match="training-only statistic is 15.0"):
        case.require_training_imputer_statistics(mismatched_result)


def test_imputed_values_mismatch_raises_in_require():
    invalid_result = {
        "applied_statistic": 15.0,
        "training_statistic": 15.0,
        "imputed_test_values": [100.0, 99.0],
        "expected_test_values": [100.0, 15.0],
    }
    with pytest.raises(ValueError, match="do not match expected values"):
        case.require_training_imputer_statistics(invalid_result)
