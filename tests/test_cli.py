import json
import subprocess
import sys

import pytest

from ml_failure_lab.registry import CASES


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "ml_failure_lab", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_list_exposes_actual_runnable_cases():
    result = cli("list")
    assert result.returncode == 0
    assert result.stdout.splitlines() == list(CASES)


@pytest.mark.parametrize("name", CASES)
def test_json_runs_real_experiment_and_preserves_interpretation(name):
    result = cli("run", name, "--json")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["case"] == name
    assert payload["measurements"]
    assert payload["requirement"]
    assert payload["interpretation"]
    assert payload["limitation"]


@pytest.mark.parametrize("name", CASES)
def test_verify_exit_code_exposes_failure_then_correction(name):
    bad = cli("verify", name, "--approach", "bad")
    fixed = cli("verify", name, "--approach", "fixed")
    assert bad.returncode == 1
    assert bad.stdout.startswith("FAIL:")
    assert fixed.returncode == 0, fixed.stderr
    assert fixed.stdout.startswith("PASS:")


def test_unknown_case_reports_available_choices():
    result = cli("run", "does-not-exist")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr
    assert "group-leakage" in result.stderr
