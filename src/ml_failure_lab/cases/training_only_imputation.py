"""Fitting an imputer on test rows leaks unseen distribution statistics into training."""

import math

import numpy as np
from sklearn.impute import SimpleImputer


def training_fixture() -> np.ndarray:
    # Single feature with observed values [10.0, 20.0] and one missing value
    return np.array([[10.0], [20.0], [np.nan]], dtype=float)


def report(approach: str, test_data: np.ndarray | list[float] | None = None) -> dict:
    if approach not in {"bad", "fixed"}:
        raise ValueError("Choose 'bad' or 'fixed'.")

    X_train = training_fixture()
    if test_data is None:
        X_test = np.array([[100.0], [np.nan]], dtype=float)
    else:
        raw = np.asarray(test_data, dtype=float)
        X_test = raw.reshape(-1, 1) if raw.ndim == 1 else raw

    if X_test.ndim != 2 or X_test.shape[1] != 1 or len(X_test) == 0:
        raise ValueError("Test data must be a non-empty two-dimensional array with one column.")

    train_valid = X_train[~np.isnan(X_train[:, 0])]
    training_statistic = float(np.mean(train_valid[:, 0]))
    expected_test_imputed = np.where(np.isnan(X_test), training_statistic, X_test)

    if approach == "bad":
        # Misleading operation: concatenate train and test rows and fit imputer on all records
        all_data = np.vstack([X_train, X_test])
        imputer = SimpleImputer(strategy="mean").fit(all_data)
        applied_statistic = float(imputer.statistics_[0])
        test_imputed = imputer.transform(X_test)
    else:
        # Correction: fit imputer strictly on training rows and transform test records
        imputer = SimpleImputer(strategy="mean").fit(X_train)
        applied_statistic = float(imputer.statistics_[0])
        test_imputed = imputer.transform(X_test)

    reused = math.isclose(applied_statistic, training_statistic, rel_tol=1e-5, abs_tol=1e-5)

    return {
        "approach": approach,
        "raw_test_values": [None if np.isnan(v) else v for v in X_test.flatten().tolist()],
        "imputed_test_values": test_imputed.flatten().tolist(),
        "expected_test_values": expected_test_imputed.flatten().tolist(),
        "training_statistic": training_statistic,
        "applied_statistic": applied_statistic,
        "reused_training_parameters": reused,
    }


def require_training_imputer_statistics(result: dict) -> None:
    if not math.isclose(
        result["applied_statistic"],
        result["training_statistic"],
        rel_tol=1e-5,
        abs_tol=1e-5,
    ):
        raise ValueError(
            f"Imputer statistic was fitted as {result['applied_statistic']}, "
            f"but training-only statistic is {result['training_statistic']}."
        )
    for actual, expected in zip(
        result["imputed_test_values"],
        result["expected_test_values"],
        strict=True,
    ):
        if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"Imputed test values {result['imputed_test_values']} do not match "
                f"expected values {result['expected_test_values']} under training statistic."
            )


def run() -> dict:
    return {
        "case": "training-only-imputation",
        "question": "Did missing-value imputation fit statistics on training rows only?",
        "measurements": [report("bad"), report("fixed")],
        "requirement": (
            "Imputation statistics must be fitted on training rows only without "
            "observing test records."
        ),
        "interpretation": (
            "Fitting an imputer on all rows pools test observations with training rows, "
            "leaking the test mean (43.33) into the imputed values. Fitting on training rows "
            "only uses the training mean (15.00), preventing held-out data leakage."
        ),
        "limitation": (
            "Fitting statistics on training rows only prevents data leakage, but mean "
            "imputation assumes missingness is completely at random and reduces feature variance."
        ),
    }


def verify(approach: str) -> None:
    result = report(approach)
    require_training_imputer_statistics(result)
