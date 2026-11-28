# Training-only imputation

## Question and prerequisites

Did missing-value imputation fit statistics on training rows only? Missing-value transformers such as `SimpleImputer` compute summary statistics (such as sample mean $\mu$) across observed records during fitting. When missing values are imputed by fitting across all records (pooling train and held-out test data), test observations influence the training statistics. You need basic familiarity with mean imputation to follow this example.

## Original synthetic input

The training fixture defines a single continuous feature with two observed values and one missing entry:

$$X_{\text{train}} = [[10.0], [20.0], [\text{nan}]]$$

Computing the mean over observed training records gives:

$$\mu_{\text{train}} = \frac{10.0 + 20.0}{2} = 15.0$$

The primary test set introduces an out-of-distribution observation alongside a missing entry:

$$X_{\text{test}} = [[100.0], [\text{nan}]]$$

In a valid workflow, the missing test entry must be imputed using the statistic learned on training records ($\mu_{\text{train}} = 15.0$). A second shifted test fixture evaluates $X_{\text{shift}} = [[50.0], [\text{nan}]]$, which also must be imputed using $\mu_{\text{train}} = 15.0$.

## Misleading operation and correction

In the incorrect workflow, the imputer is fitted on all data by concatenating training and test rows:

$$\text{all\_data} = \begin{bmatrix} X_{\text{train}} \\ X_{\text{test}} \end{bmatrix}$$

Fitting on all rows computes:

$$\mu_{\text{leaked}} = \frac{10.0 + 20.0 + 100.0}{3} \approx 43.333$$

This leaks information from held-out test records into the learned imputation statistic. The imputed values become $43.333$ instead of $15.0$, distorting feature values across both sets.

The correction fits `SimpleImputer(strategy="mean")` strictly on the training set:

$$\text{imputer} = \text{SimpleImputer}(\text{strategy}=\text{"mean"}).\text{fit}(X_{\text{train}})$$

The pre-fitted imputer is then applied to transform test rows:

$$X_{\text{test, imputed}} = \text{imputer}.\text{transform}(X_{\text{test}})$$

This transforms $[[100.0], [\text{nan}]]$ into $[[100.0], [15.0]]$, ensuring that imputed values depend solely on training records without leakage.

## Run and verification

```text
python -m ml_failure_lab run training-only-imputation
python -m ml_failure_lab verify training-only-imputation --approach bad
python -m ml_failure_lab verify training-only-imputation --approach fixed
```

The `bad` verification exits with code 1, reporting that the imputer statistic was fitted as $43.333$ rather than the training-only statistic $15.0$. The `fixed` verification exits with code 0. Tests verify that imputed test values match expected ground truth under relative and absolute tolerances of $10^{-5}$ across multiple shifted fixtures.

## Limits and reference

Fitting imputation statistics strictly on training rows prevents held-out data leakage. However, mean imputation assumes missingness occurs completely at random (MCAR) and reduces overall feature variance.

Reference: [Common pitfalls and recommended practices in scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
