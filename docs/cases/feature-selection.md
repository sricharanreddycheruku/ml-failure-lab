# Choosing features using held-out labels

## Question

Did the feature-selection step read information reserved for final evaluation?

A feature is an input column. A label is the value we want to predict. Feature selection chooses a subset of columns before fitting the classifier. Here, it scores each column using the labels supplied to `fit`.

## The fixture

The case generates 200 rows, 2,000 independent random features and random binary labels. There is no designed predictive relationship. It holds out 50 rows and selects 20 features.

The classifier is logistic regression. Its precise scores are secondary to the data-access requirement.

## Incorrect approach

Fit `SelectKBest` on all 200 rows and their labels, then train the classifier on the training subset. The selector has already used the 50 held-out labels to choose the columns. The resulting test score no longer measures a procedure that was independent of those labels.

## Correction

Fit the selector on the 150 training rows. Apply that fitted selector to both training and test features without fitting it again. Train the classifier on the selected training data.

The code records the exact row IDs used to slice the selector's fitting input. The check rejects any overlap between those IDs and the test IDs. This is a requirement check, not a claim that a particular score ordering must always occur.

```text
python -m ml_failure_lab run feature-selection
python -m ml_failure_lab verify feature-selection --approach bad
python -m ml_failure_lab verify feature-selection --approach fixed
```

The run compares three seeds. Read the reported scores from your actual execution. Library versions and numerical behavior can change the values; the fit-membership requirement remains the same.

## Limits and reference

This case uses one holdout split per seed. With cross-validation, feature selection and other learned preprocessing must happen inside each training fold. The example does not certify a notebook or detect arbitrary data flows.

The mechanism is explained in [scikit-learn's common pitfalls chapter](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage-during-pre-processing). This project uses its own random fixture, classifier and membership check.
