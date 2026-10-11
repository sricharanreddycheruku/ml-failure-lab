"""Inspect declared identifiers and exact feature rows across two CSV files."""

import csv
from collections import Counter
from pathlib import Path


def read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        header = reader.fieldnames
        if not header or len(header) != len(set(header)):
            raise ValueError(f"{path.name}: CSV needs a nonempty, unique header.")
        missing = set(columns) - set(header)
        if missing:
            raise ValueError(f"{path.name}: missing columns: {', '.join(sorted(missing))}.")
        rows = []
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}: malformed CSV record at line {line}.")
            if any(not row[column].strip() for column in columns):
                raise ValueError(f"{path.name}: blank checked value at line {line}.")
            rows.append(row)
        if not rows:
            raise ValueError(f"{path.name}: evaluation needs nonempty train and test files.")
        return rows


def check_split(
    train_path: Path,
    test_path: Path,
    id_column: str,
    features: list[str] | None = None,
    require_new_ids: bool = False,
) -> dict:
    features = features or []
    if len(features) != len(set(features)) or id_column in features:
        raise ValueError("Feature columns must be unique and must exclude the ID column.")
    columns = [id_column, *features]
    train = read_rows(train_path, columns)
    test = read_rows(test_path, columns)
    train_ids = {row[id_column] for row in train}
    test_ids = {row[id_column] for row in test}
    shared = train_ids & test_ids
    feature_overlap = None
    if features:
        train_counts = Counter(tuple(row[column] for column in features) for row in train)
        test_counts = Counter(tuple(row[column] for column in features) for row in test)
        repeated = train_counts.keys() & test_counts.keys()
        feature_overlap = {
            "distinct_shared_rows": len(repeated),
            "affected_test_rows": sum(test_counts[row] for row in repeated),
            "matching_train_test_pairs": sum(
                train_counts[row] * test_counts[row] for row in repeated
            ),
        }
    return {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_unique_ids": len(train_ids),
        "test_unique_ids": len(test_ids),
        "shared_ids": len(shared),
        "affected_test_rows": sum(row[id_column] in shared for row in test),
        "exact_feature_overlap": feature_overlap,
        "requirement": "new IDs" if require_new_ids else "inspect only",
        "status": "FAIL" if require_new_ids and shared else "PASS" if require_new_ids else "INFO",
        "limitation": (
            "Only the selected IDs and exact CSV feature strings are compared. "
            "Matching feature values can occur in independent records. "
            "Disjoint IDs do not rule out other leakage or prove model quality."
        ),
    }
