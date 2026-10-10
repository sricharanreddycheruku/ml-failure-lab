# Unequal batch accuracies

## Question and prerequisites

What fraction of all evaluated records was predicted correctly? Accuracy is the number of correct predictions divided by the number of records. A batch is a group of records processed together. You need only fractions and an arithmetic mean to follow this example.

## Original synthetic input

The first batch has targets `[0, 0, 0, 0, 0, 0, 0, 0, 0]` and identical predictions. The second has target `[1]` and prediction `[0]`. These original arrays require no external dataset or account. No model is trained: the same predictions are used in both workflows.

## Misleading operation and correction

The incorrect workflow computes `(1.0 + 0.0) / 2 = 0.5`. It gives the single incorrect record the same aggregate weight as the nine correct records. For the declared record-level question, count nine correct predictions across ten records: `9 / 10 = 0.9`.

Empty batches contribute no records and are ignored by both workflows. An entirely empty evaluation raises `ValueError` rather than returning a score or dividing by zero. Batches must contain matching one-dimensional target/prediction arrays.

## Run and verification

```text
python -m ml_failure_lab run unequal-batch-metrics
python -m ml_failure_lab verify unequal-batch-metrics --approach bad
python -m ml_failure_lab verify unequal-batch-metrics --approach fixed
```

Measured output for these arrays is 50% for `bad` and 90% for `fixed`, with nine correct records out of ten in both reports. The bad verification exits 1; the fixed verification exits 0. Verification independently compares the reported value with scikit-learn accuracy over concatenated arrays, using absolute tolerance `1e-12` and zero relative tolerance for floating-point arithmetic. Tests also count matches manually and repartition the same records into batches of different sizes, including empty batches. Correct record accuracy must stay at 90%.

## Limits and reference

Equal weighting of batches is a valid alternative estimand (the quantity being measured) when the question is the average batch's accuracy. It is not the requested record accuracy here. This correction does not imply that batch F1 or AUC should be averaged with record weights: those metrics are not additive counts. The tiny fixture demonstrates an aggregation error, not real-world model performance.

Reference: [scikit-learn accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html).
