# Tuning classification thresholds on test records

## Question and prerequisites

Was the classification threshold tuned without touching final test labels?

A **threshold** converts a continuous model score into a binary decision: predictions are class 1 if `score >= threshold` and 0 otherwise. **Accuracy** is the fraction of records correctly classified. Tuning a threshold is a model selection step that optimizes performance over candidate values. This case needs only arithmetic comparison and record IDs; it does not train a model.

## Synthetic fixture

Candidate thresholds are evaluated over the grid `(0.3, 0.5, 0.7)`. The experiment provides distinct validation and held-out test records:

**Validation records:**

| Record ID | Score | Label |
|---|---:|---:|
| 1 | 0.1 | 0 |
| 2 | 0.4 | 0 |
| 3 | 0.6 | 1 |
| 4 | 0.9 | 1 |

**Final test records:**

| Record ID | Score | Label |
|---|---:|---:|
| 101 | 0.2 | 0 |
| 102 | 0.4 | 1 |
| 103 | 0.6 | 1 |
| 104 | 0.8 | 1 |

## Misleading threshold tuning

The incorrect workflow evaluates the candidate threshold grid directly on the **final test records**.

On the test set:
- Threshold `0.3` classifies all 4 records correctly (100% accuracy).
- Threshold `0.5` classifies 3 of 4 records correctly (75% accuracy).
- Threshold `0.7` classifies 2 of 4 records correctly (50% accuracy).

The bad workflow selects `0.3`, producing an artificially inflated **100% test accuracy**. Evaluating candidate thresholds directly on final test labels leaks held-out ground truth into hyperparameter tuning, compromising the integrity of the evaluation.

## Correction

Tune the threshold strictly on **validation records**, freeze the chosen value, and only then evaluate final test records.

On the validation set:
- Threshold `0.3` achieves 75% accuracy (record 2 is misclassified).
- Threshold `0.5` achieves 100% accuracy (all 4 records correct).
- Threshold `0.7` achieves 75% accuracy (record 3 is misclassified).

The fixed workflow selects `0.5` based on validation performance alone. Applying this frozen threshold to the untouched test records yields an honest **75% test accuracy**.

The audit checks the record IDs accessed during threshold selection against the test record IDs. If any test ID appears in the selection audit, verification fails.

## Run and verify

```text
python -m ml_failure_lab run test-tuned-threshold
python -m ml_failure_lab run test-tuned-threshold --json
python -m ml_failure_lab verify test-tuned-threshold --approach bad
python -m ml_failure_lab verify test-tuned-threshold --approach fixed
```

The bad verification exits with code 1 (`FAIL`), reporting the leaked test record IDs. The fixed verification exits with code 0 (`PASS`).

## Limits and references

This synthetic fixture demonstrates the evaluation contract for threshold tuning. Using a separate validation split does not guarantee higher performance on the test set; it ensures that the test metric provides an unbiased estimate of generalization error.

See [scikit-learn's common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) and [TunedThresholdClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TunedThresholdClassifierCV.html) for production threshold tuning workflows.

