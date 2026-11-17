# Numeric feature order

## Question and prerequisites

Did passing a NumPy array to prediction preserve the meaning of each numeric column? Estimators in scikit-learn often fit and predict on two-dimensional arrays of numerical values. Raw NumPy arrays index features by column position, not by name. When incoming features are ordered differently from the training schema, columns can silently exchange meanings without raising any error. You need basic familiarity with linear regression and NumPy indexing to follow this example.

## Original synthetic input

The training fixture fits a simple linear model with zero intercept on two independent numeric features $a$ and $b$:

$$y = 1 \cdot a + 10 \cdot b$$

The training schema establishes that column 0 corresponds to $a$ and column 1 corresponds to $b$. The evaluation provides a test row with feature values $a = 1.0$ and $b = 2.0$. The true intended calculation is $1(1.0) + 10(2.0) = 21.0$.

## Misleading operation and correction

In the incorrect workflow, the named record is supplied in key order `['b', 'a']` and extracted directly into a raw array without checking or aligning column names against the model's training schema:

$$\text{array} = [[2.0, 1.0]]$$

The fitted linear model interprets position 0 as feature $a$ and position 1 as feature $b$, computing:

$$1 \cdot 2.0 + 10 \cdot 1.0 = 12.0$$

The prediction silently runs to completion and produces 12.0 instead of 21.0.

The correction (`align_features`) records the expected feature order `['a', 'b']` from training. Before converting inputs to an array for prediction, it validates the schema:
- Duplicate column names in sequential inputs are rejected with a clear error.
- Missing required columns raise a missing-column error.
- Unrecognized or extra columns raise an unexpected-column error.
- Existing columns are reordered to match `['a', 'b']` exactly: $[[1.0, 2.0]]$.

With aligned columns, the model outputs the correct prediction $21.0$.

## Run and verification

```text
python -m ml_failure_lab run numeric-feature-order
python -m ml_failure_lab verify numeric-feature-order --approach bad
python -m ml_failure_lab verify numeric-feature-order --approach fixed
```

The `bad` verification exits with code 1, reporting that column schema alignment was violated and the prediction was $12.0$. The `fixed` verification exits with code 0. Tests verify that the corrected prediction matches manual ground truth ($21.0$) within floating-point tolerance $10^{-5}$, and assert rejection policies for missing, duplicate, and extra feature names.

## Limits and reference

This case demonstrates silent semantic failure when passing raw arrays to numeric estimators. Some pipelines or estimators (such as pandas DataFrame inputs in modern scikit-learn) track feature names internally; however, converting to raw NumPy arrays drops metadata and reintroduces this risk. This case applies to column order alignment; it does not replace categorical encoding validation or train-test row split checks.

Reference: [scikit-learn LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html).
