"""Passing a raw NumPy array to an estimator can silently scramble feature meaning."""

import math
from collections.abc import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

TRAINING_COLUMNS = ["a", "b"]


def fit_fixture(seed: int = 42) -> tuple[LinearRegression, list[str]]:
    rng = np.random.default_rng(seed)
    # Generate independent numeric features a and b
    features = rng.uniform(1.0, 5.0, size=(100, 2))
    # Target: y = 1 * a + 10 * b
    targets = 1.0 * features[:, 0] + 10.0 * features[:, 1]
    model = LinearRegression(fit_intercept=False)
    model.fit(features, targets)
    return model, list(TRAINING_COLUMNS)


def align_features(
    record: Mapping[str, float] | Sequence[tuple[str, float]],
    expected_columns: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Validate and align named features to the expected schema order.

    Rejection policy:
    - Sequence with duplicate feature names raises ValueError.
    - Missing required features raises ValueError.
    - Extra / unknown features raises ValueError.
    """
    if isinstance(record, Mapping):
        mapping = dict(record)
    else:
        names = [name for name, _ in record]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate feature names are not permitted in input.")
        mapping = dict(record)

    expected_set = set(expected_columns)
    given_set = set(mapping.keys())

    missing = sorted(expected_set - given_set)
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    extra = sorted(given_set - expected_set)
    if extra:
        raise ValueError(f"Unrecognized feature columns: {extra}")

    aligned_values = [float(mapping[col]) for col in expected_columns]
    return list(expected_columns), np.array([aligned_values], dtype=float)


def report(
    approach: str,
    test_input: Mapping[str, float] | Sequence[tuple[str, float]] | None = None,
) -> dict:
    if approach not in {"bad", "fixed"}:
        raise ValueError("Choose 'bad' or 'fixed'.")

    model, expected_columns = fit_fixture()
    # Default test row: a=1.0, b=2.0; supplied in swapped order [('b', 2.0), ('a', 1.0)]
    if test_input is None:
        input_pairs = [("b", 2.0), ("a", 1.0)]
    elif isinstance(test_input, Mapping):
        input_pairs = list(test_input.items())
    else:
        input_pairs = list(test_input)

    # Manual ground truth calculation: 1.0 * a + 10.0 * b
    input_dict = dict(input_pairs)
    expected_prediction = 1.0 * input_dict["a"] + 10.0 * input_dict["b"]

    if approach == "bad":
        # Misleading operation: convert directly to a NumPy array in incoming key order
        input_columns = [k for k, _ in input_pairs]
        raw_array = np.array([[v for _, v in input_pairs]], dtype=float)
        prediction = float(model.predict(raw_array)[0])
        used_columns = input_columns
    else:
        # Correction: validate schema and align columns to expected order
        used_columns, aligned_array = align_features(input_pairs, expected_columns)
        prediction = float(model.predict(aligned_array)[0])

    is_aligned = used_columns == expected_columns
    return {
        "approach": approach,
        "input_columns": [k for k, _ in input_pairs],
        "model_columns": list(expected_columns),
        "prediction": prediction,
        "expected_prediction": expected_prediction,
        "columns_aligned": is_aligned,
    }


def require_column_schema_alignment(result: dict) -> None:
    if not result["columns_aligned"]:
        raise ValueError(
            f"Input features {result['input_columns']} must be aligned to "
            f"model schema {result['model_columns']} before prediction."
        )
    if not math.isclose(result["prediction"], result["expected_prediction"], rel_tol=1e-5):
        raise ValueError(
            f"Prediction {result['prediction']} does not match expected "
            f"{result['expected_prediction']}."
        )


def run() -> dict:
    return {
        "case": "numeric-feature-order",
        "question": (
            "Did passing a NumPy array to prediction preserve the meaning of each numeric column?"
        ),
        "measurements": [report("bad"), report("fixed")],
        "requirement": (
            "Input features must be aligned to the training schema column order before prediction."
        ),
        "interpretation": (
            "Passing test features as raw NumPy arrays silently drops column names. "
            "When the incoming column order ['b', 'a'] differs from training ['a', 'b'], "
            "the model computes 1*2 + 10*1 = 12 instead of 1*1 + 10*2 = 21. Aligning columns "
            "to the schema restores the correct prediction 21."
        ),
        "limitation": (
            "This case concerns column alignment for numeric estimators accepting raw NumPy arrays."
            " It does not address categorical encoding, feature scaling, or row shuffling."
        ),
    }


def verify(approach: str) -> None:
    result = report(approach)
    require_column_schema_alignment(result)
