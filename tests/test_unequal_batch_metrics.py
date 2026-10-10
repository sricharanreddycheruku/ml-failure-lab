import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from ml_failure_lab.cases import unequal_batch_metrics as case


def test_actual_fixture_exposes_the_misleading_mean():
    batches = case.fixture()
    targets = np.concatenate([batch[0] for batch in batches])
    predictions = np.concatenate([batch[1] for batch in batches])
    manual = sum(
        int(target == prediction) for target, prediction in zip(targets, predictions, strict=True)
    )
    bad, fixed = case.run()["measurements"]
    assert bad["accuracy"] == 0.5
    assert fixed["accuracy"] == pytest.approx(manual / len(targets), abs=1e-12)
    assert fixed["accuracy"] == accuracy_score(targets, predictions) == 0.9
    assert fixed["correct_records"] == 9
    assert fixed["total_records"] == 10


@pytest.mark.parametrize("sizes", [[10], [5, 5], [1] * 10, [2, 3, 5], [0, 9, 0, 1]])
def test_fixed_accuracy_is_invariant_to_repartitioning(sizes):
    targets, predictions = (np.concatenate([b[i] for b in case.fixture()]) for i in (0, 1))
    boundaries = np.cumsum([0, *sizes])
    batches = [
        (targets[a:b], predictions[a:b])
        for a, b in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    result = case.report("fixed", batches)
    assert result["accuracy"] == pytest.approx(0.9, abs=1e-12)
    case.require_record_accuracy(result, batches)


@pytest.mark.parametrize("batches", [[], [(np.array([]), np.array([]))]])
@pytest.mark.parametrize("approach", ["bad", "fixed"])
def test_an_entirely_empty_evaluation_is_rejected(batches, approach):
    with pytest.raises(ValueError, match="at least one record"):
        case.report(approach, batches)


def test_verification_checks_values_not_the_approach_label():
    result = case.report("fixed")
    result["accuracy"] = 0.5
    with pytest.raises(ValueError, match="Record-weighted accuracy"):
        case.require_record_accuracy(result, case.fixture())
    result["accuracy"] = 0.9
    result["approach"] = "bad"
    case.require_record_accuracy(result, case.fixture())


@pytest.mark.parametrize("targets,predictions", [([0, 1], [0]), ([[0]], [[0]])])
def test_invalid_batch_shapes_are_rejected(targets, predictions):
    with pytest.raises(ValueError, match="matching one-dimensional"):
        case.report("fixed", [(np.array(targets), np.array(predictions))])
