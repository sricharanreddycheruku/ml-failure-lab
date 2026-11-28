# Future period split

## Question and prerequisites

Does the evaluation describe a model that was frozen at time 8 and then used for predictions after time 8? You need to know what a train/test split is and how a linear regression is fitted.

Two times matter for each record. The **prediction time** is when the model is asked for an output. The **label time** is when the true outcome for that record is recorded. A label is usually known some time after its prediction, and sometimes much later. A model **frozen** at time 8 is fitted once at that time and never updated, so it can only learn from labels recorded by time 8.

## Original synthetic input

The case defines twelve fictional daily records, `r01` to `r12`, with one numeric feature. Each label is normally recorded one day after its prediction. Record `r06` is predicted on day 6, but its label is recorded on day 11. A change takes effect on day 9, so every outcome recorded from day 9 onwards is 6 higher:

| Record | Prediction time | Feature | Label | Label time |
|---|---|---|---|---|
| r01 | 1 | 3 | 6 | 2 |
| r02 | 2 | 5 | 10 | 3 |
| r03 | 3 | 4 | 8 | 4 |
| r04 | 4 | 6 | 12 | 5 |
| r05 | 5 | 5 | 10 | 6 |
| r06 | 6 | 7 | 20 | 11 |
| r07 | 7 | 6 | 12 | 8 |
| r08 | 8 | 8 | 22 | 9 |
| r09 | 9 | 7 | 20 | 10 |
| r10 | 10 | 9 | 24 | 11 |
| r11 | 11 | 8 | 22 | 12 |
| r12 | 12 | 10 | 26 | 13 |

The declared task is one model frozen at time 8 and evaluated on the forecast period, days 9 to 12. A label recorded exactly at time 8, such as `r07`'s, counts as known at the freeze.

## Misleading operation and correction

The incorrect workflow shuffles the twelve records with a fixed seed, fits the first eight and evaluates the last four, as if the records were exchangeable:

```text
fit:      r12 r08 r03 r11 r01 r02 r05 r07   label times 13 9 4 12 2 3 6 8
evaluate: r10 r06 r04 r09                   prediction times 10 6 4 9
```

The model learns from `r08`, `r11` and `r12`, whose labels did not exist at time 8, and it is scored on `r04` and `r06`, which were predicted before the freeze. The result describes interpolation between known days, not the frozen model's future use.

A common partial fix splits by prediction time: fit days 1 to 8 and evaluate days 9 to 12. This still fits `r06` and `r08`, whose labels arrive on days 11 and 9. The fitted line (slope 3.33, intercept -5.83) already reflects the day 9 change. A check on prediction time alone would accept it, which is why the case includes the late `r06` label.

The correction fits only rows whose label time is at most 8 (`r01` to `r05` and `r07`) and evaluates days 9 to 12. Those six rows lie on `label = 2 * feature`, so the fitted slope is 2 and the intercept is 0. The predictions for features `[7, 9, 8, 10]` are `[14, 18, 16, 20]`. The labels are `[20, 24, 22, 26]`, so every error is 6.

| Workflow | Fitted label times after 8 | Evaluated predictions at or before 8 | MAE |
|---|---|---|---|
| bad (shuffled) | r12, r08, r11 | r06, r04 | 1.98 |
| row-time | r06, r08 | none | 1.33 |
| fixed | none | none | 6.00 |

The corrected error is the largest. That is the honest result: a model frozen at time 8 could not know about the day 9 change. The leaky workflows score better only because they fitted outcomes from after the freeze.

## Run and verification

```text
python -m ml_failure_lab run future-period-split
python -m ml_failure_lab verify future-period-split --approach bad
python -m ml_failure_lab verify future-period-split --approach fixed
```

The check reads the label times and prediction times of the rows actually passed to `fit` and to evaluation. It fails when a fitted label was recorded after the freeze time, when an evaluated record was predicted at or before it, or when a record is both fitted and evaluated. The `bad` verification exits with code 1:

```text
FAIL: The model frozen at time 8 was fitted with labels recorded later: r12 (label at 13), r08 (label at 9), r11 (label at 12).
```

The `fixed` verification exits with code 0. The tests also reject the prediction-time split, check the boundary at exactly time 8, and change labels recorded after the freeze to confirm that the frozen model's coefficients and predictions stay the same. The comparisons use an absolute tolerance of `1e-9` because least squares returns floating-point values close to the exact 2 and 0.

## Limits and reference

The check trusts the declared timestamps. It cannot discover a wrong label time in real data. The case covers one model frozen once and one forecast period. Walk-forward evaluation, where the model is refitted as new labels arrive, needs a separate contract with one freeze time per refit. A random split can be valid when records are exchangeable and the model will not be used for a later period. The test does not require the corrected workflow to score higher or lower.

Reference: [scikit-learn cross-validation, including time series splits](https://scikit-learn.org/stable/modules/cross_validation.html).
