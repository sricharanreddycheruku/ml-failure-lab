# High accuracy with no minority detections

## Question

Does the accuracy report tell us whether the model detects the rare target class?

Accuracy is the fraction of all predictions that are correct. Recall for class 1 is the fraction of actual class-1 examples that the model predicts as class 1. Support is the number of actual examples of that class in the evaluation.

## The fixture

There are 990 class-0 records and 10 class-1 records. A stratified split keeps the label proportions on both sides, leaving 495 class-0 and 5 class-1 records in the test set.

A majority-class baseline always predicts class 0. A baseline is a simple method we can compare with a more elaborate model. Scikit-learn's `DummyClassifier` implements this one.

## What accuracy conceals

The model correctly predicts 495 of the 500 test records:

```text
accuracy = 495 / 500 = 0.99
```

It detects none of the five class-1 records:

```text
minority recall = 0 / 5 = 0
```

For a task that needs class-1 detection, the accuracy number alone leaves out the relevant failure.

## Correction

Report accuracy, minority recall and support. The corrected report also includes balanced accuracy, the average of the recall for each class. Class-0 recall is 1 and class-1 recall is 0, so balanced accuracy is 0.5.

The model is identical in both reports. Better reporting exposes its weakness; it does not fix that weakness.

```text
python -m ml_failure_lab run minority-recall
python -m ml_failure_lab verify minority-recall --approach bad
python -m ml_failure_lab verify minority-recall --approach fixed
```

The check requires minority recall and support for this declared task. It rejects the incomplete report.

## Limits and reference

Metric choice depends on the intended use and the costs of errors. This case does not establish that accuracy is always unsuitable for imbalanced data. Five minority examples are also too few to make a strong real-world performance claim.

See [scikit-learn's classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).
