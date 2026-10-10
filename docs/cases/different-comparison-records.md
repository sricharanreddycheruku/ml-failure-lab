# Comparing models on different held-out records

## Question and prerequisites

Does a paired comparison of two candidate models measure their error on **the same held-out records**?

A **target** is the actual value for a record, a **prediction** is a candidate's estimate, and a **record ID** identifies which example was evaluated. The mean squared error (MSE) averages `(target - prediction) ** 2` over the evaluated records, so lower is better on the same test set. This case needs only arithmetic and record IDs; it does not train any models.

## Synthetic fixture

All four original held-out records have target zero. The values below are stored predictions, not results claimed from training.

| Record ID | Target | Candidate A prediction | Candidate B prediction |
|---|---:|---:|---:|
| 101 | 0 | 0 | 1 |
| 102 | 0 | 0 | 1 |
| 103 | 0 | 0 | 1 |
| 104 | 0 | 3 | 1 |

## Misleading comparison

Evaluate A on only record IDs 101, 102 and 103, but evaluate B on IDs 101, 102, 103 and 104. A's measured MSE is `(0 + 0 + 0) / 3 = 0`; B's is `(1 + 1 + 1 + 1) / 4 = 1`. A appears better, but the result **does not support a paired comparison**. A never faced the difficult fourth record.

The executable case keeps each candidate's actual evaluated record IDs, targets and predictions alongside its measured score. The check rejects the mismatched ID sets. Equal sample counts are **not enough**: two equally long results with different record IDs also fail.

## Correction

Evaluate **both** candidates on all four records. A now has MSE `(0 + 0 + 0 + 9) / 4 = 2.25`; B still has MSE `1`. On this shared evaluation set, B has lower error.

Predictions are looked up by **record ID**, not aligned by list position. The comparison also checks that the target for a given ID agrees between results. Therefore, two results with the same IDs in different orders are comparable and cannot silently pair a prediction with the wrong target. Tests use a separate fixture with distinct target values to verify this ordering behavior.

## Run and verify

```text
python -m ml_failure_lab run different-comparison-records
python -m ml_failure_lab run different-comparison-records --json
python -m ml_failure_lab verify different-comparison-records --approach bad
python -m ml_failure_lab verify different-comparison-records --approach fixed
```

The incorrect check prints `FAIL` and exits with code 1. The corrected check prints `PASS` and exits with code 0. The run prints both comparisons and their actual measurements. Tests check the manually calculated MSE values and the evaluated IDs, rather than merely checking the approach label.

## Limits and references

This is a deliberately tiny paired benchmark with synthetic stored predictions and four records. It illustrates **comparison-set alignment**, not which model will generalize to other data. Studies using separate independent samples may be valid for other questions and statistical designs; those are not paired comparisons of predictions on identical held-out records.

See [scikit-learn's cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) and [model evaluation guide](https://scikit-learn.org/stable/modules/model_evaluation.html) for broader evaluation practices.
