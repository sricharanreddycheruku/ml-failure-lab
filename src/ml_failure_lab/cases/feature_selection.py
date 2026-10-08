"""A selector fitted using test labels violates the held-out evaluation contract."""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class SelectorFit:
    selector: SelectKBest
    fitted_rows: frozenset[int]


def experiment(approach: str, seed: int = 42) -> tuple[SelectorFit, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    # There is no designed relationship between the features and labels.
    features = rng.normal(size=(200, 2000))
    targets = rng.integers(0, 2, size=200)
    train_ids, test_ids = train_test_split(
        np.arange(len(targets)), test_size=0.25, random_state=seed, stratify=targets
    )
    fit_ids = np.arange(len(targets)) if approach == "bad" else train_ids
    selector = SelectKBest(score_func=f_classif, k=20)
    # The recorded IDs are the actual indices used to slice the fit input.
    selector.fit(features[fit_ids], targets[fit_ids])
    fitted = SelectorFit(selector, frozenset(int(row) for row in fit_ids))
    model = LogisticRegression(solver="liblinear", random_state=seed)
    model.fit(selector.transform(features[train_ids]), targets[train_ids])
    predictions = model.predict(selector.transform(features[test_ids]))
    return fitted, test_ids, float(accuracy_score(targets[test_ids], predictions))


def require_unseen_test_labels(fitted: SelectorFit, test_ids: np.ndarray) -> None:
    if fitted.fitted_rows & {int(row) for row in test_ids}:
        raise ValueError("Feature selection must not fit using held-out test rows or labels.")


def run() -> dict:
    measurements = []
    for seed in [42, 43, 44]:
        for approach in ["bad", "fixed"]:
            fitted, test_ids, score = experiment(approach, seed)
            measurements.append(
                {
                    "approach": approach,
                    "seed": seed,
                    "accuracy": score,
                    "test_rows_used_to_fit_selector": len(fitted.fitted_rows & set(test_ids)),
                }
            )
    return {
        "case": "feature-selection",
        "question": "Did choosing features use any labels reserved for final evaluation?",
        "measurements": measurements,
        "requirement": "The selector must fit only on training data.",
        "interpretation": (
            "The bad selector examines held-out labels while choosing features. "
            "The fixed selector learns from training rows only. The requirement is about "
            "data access, not which run happens to produce the larger score."
        ),
        "limitation": (
            "Three seeds illustrate one synthetic setup. They do not prove a universal score "
            "gap or replace checking preprocessing inside each cross-validation fold."
        ),
    }


def verify(approach: str) -> None:
    fitted, test_ids, _ = experiment(approach)
    require_unseen_test_labels(fitted, test_ids)
