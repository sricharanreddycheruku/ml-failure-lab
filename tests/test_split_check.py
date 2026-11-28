import csv
import json
import subprocess
import sys

import pytest

from ml_failure_lab.split_check import check_split


def write_csv(path, rows, header=("source_id", "feature")):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_shared_ids_are_informational_unless_new_ids_are_required(tmp_path):
    train = write_csv(tmp_path / "train.csv", [("private-A", "1"), ("private-B", "2")])
    test = write_csv(tmp_path / "test.csv", [("private-A", "1"), ("private-A", "1")])
    report = check_split(train, test, "source_id", ["feature"])
    assert report["status"] == "INFO"
    assert report["shared_ids"] == 1
    assert report["affected_test_rows"] == 2
    assert report["exact_feature_overlap"] == {
        "distinct_shared_rows": 1,
        "affected_test_rows": 2,
        "matching_train_test_pairs": 2,
    }
    assert "private-A" not in json.dumps(report)
    assert check_split(train, test, "source_id", require_new_ids=True)["status"] == "FAIL"


def test_duplicate_features_do_not_prove_shared_source_identity(tmp_path):
    train = write_csv(tmp_path / "train.csv", [("A", "1"), ("A", "1")])
    test = write_csv(tmp_path / "test.csv", [("B", "1"), ("B", "1"), ("C", "1")])
    report = check_split(train, test, "source_id", ["feature"], True)
    assert report["status"] == "PASS"
    assert report["shared_ids"] == 0
    assert report["exact_feature_overlap"]["matching_train_test_pairs"] == 6
    assert report["exact_feature_overlap"]["affected_test_rows"] == 3


def test_csv_matching_is_exact_and_supports_quoted_unicode(tmp_path):
    train = write_csv(tmp_path / "train.csv", [("α,one", "1.0")])
    test = write_csv(tmp_path / "test.csv", [("α,one", "1")])
    report = check_split(train, test, "source_id", ["feature"])
    assert report["shared_ids"] == 1
    assert report["exact_feature_overlap"]["distinct_shared_rows"] == 0


def test_valid_multiline_fields_are_supported(tmp_path):
    train = write_csv(tmp_path / "train.csv", [("A\nB", "1"), ("C", "2")])
    test = write_csv(tmp_path / "test.csv", [("A\nB", "1"), ("D", "3")])
    report = check_split(train, test, "source_id", ["feature"], require_new_ids=True)
    assert report["status"] == "FAIL"
    assert report["shared_ids"] == 1
    assert report["train_rows"] == 2
    assert report["test_rows"] == 2


@pytest.mark.parametrize(
    "contents,match",
    [
        ("", "header"),
        ("source_id,source_id\nA,B\n", "unique header"),
        ("other,feature\nA,1\n", "missing columns"),
        ("source_id,feature\n", "nonempty"),
        ("source_id,feature\n ,1\n", "blank"),
        ("source_id,feature\nA\n", "malformed"),
        ("source_id,feature\nA,1,extra\n", "malformed"),
        ('source_id,feature\n"A\nB",1\nC,2,extra\n', "malformed CSV record at line 4"),
        ('source_id,feature\n"A\nB",1\nC, \n', "blank checked value at line 4"),
    ],
)
def test_invalid_input_is_not_reported_as_clean(tmp_path, contents, match):
    train = tmp_path / "train.csv"
    train.write_text(contents)
    test = write_csv(tmp_path / "test.csv", [("B", "1")])
    with pytest.raises(ValueError, match=match):
        check_split(train, test, "source_id", ["feature"])


@pytest.mark.parametrize("features", [["feature", "feature"], ["source_id"]])
def test_feature_selection_excludes_ids_and_duplicate_columns(tmp_path, features):
    path = write_csv(tmp_path / "data.csv", [("A", "1")])
    with pytest.raises(ValueError, match="unique"):
        check_split(path, path, "source_id", features)


@pytest.mark.parametrize(
    "same_id,require_new,expected", [(True, False, 0), (True, True, 1), (False, True, 0)]
)
def test_cli_returns_policy_exit_code_and_machine_readable_report(
    tmp_path, same_id, require_new, expected
):
    train = write_csv(tmp_path / "train.csv", [("A", "1")])
    test = write_csv(tmp_path / "test.csv", [("A" if same_id else "B", "2")])
    args = [
        sys.executable,
        "-m",
        "ml_failure_lab",
        "check-split",
        str(train),
        str(test),
        "--id-column",
        "source_id",
        "--json",
    ]
    if require_new:
        args.append("--require-new-ids")
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    assert result.returncode == expected, result.stderr
    assert json.loads(result.stdout)["shared_ids"] == int(same_id)


def test_cli_invalid_csv_exits_two_without_traceback(tmp_path):
    train = write_csv(tmp_path / "train.csv", [("A", "1")])
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_failure_lab",
            "check-split",
            str(train),
            str(train),
            "--id-column",
            "missing",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "missing columns" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("corrupt_train", [True, False])
def test_cli_nonexistent_file_exits_two_without_traceback(tmp_path, corrupt_train):
    valid = write_csv(tmp_path / "valid.csv", [("A", "1")])
    missing = tmp_path / "nonexistent.csv"
    train_path = missing if corrupt_train else valid
    test_path = valid if corrupt_train else missing
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_failure_lab",
            "check-split",
            str(train_path),
            str(test_path),
            "--id-column",
            "source_id",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Invalid split input:" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("corrupt_train", [True, False])
def test_cli_invalid_utf8_exits_two_without_traceback(tmp_path, corrupt_train):
    valid = write_csv(tmp_path / "valid.csv", [("A", "1")])
    bad_utf8 = tmp_path / "bad_utf8.csv"
    bad_utf8.write_bytes(b"\x80\xff\xfe\xfa")
    train_path = bad_utf8 if corrupt_train else valid
    test_path = valid if corrupt_train else bad_utf8
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ml_failure_lab",
            "check-split",
            str(train_path),
            str(test_path),
            "--id-column",
            "source_id",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Invalid split input:" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""


def test_bom_and_crlf_csv_matches_plain_utf8(tmp_path):
    # A UTF-8 BOM before the header and Windows line endings must not change
    # how the first column or a non-ASCII quoted ID is read (issue #37).
    bom_path = tmp_path / "bom.csv"
    bom_path.write_bytes(b"\xef\xbb\xbf" + 'source_id,feature\r\n"café,α",ok\r\n'.encode())
    plain_path = tmp_path / "plain.csv"
    plain_path.write_text('source_id,feature\n"café,α",ok\n', encoding="utf-8")
    test = write_csv(tmp_path / "test.csv", [("café,α", "ok")])

    bom_report = check_split(bom_path, test, "source_id", ["feature"])
    plain_report = check_split(plain_path, test, "source_id", ["feature"])

    assert bom_report == plain_report
    assert bom_report["train_rows"] == 1
    assert bom_report["shared_ids"] == 1
    assert bom_report["affected_test_rows"] == 1
