# ML failure lab

Small Python experiments that reveal misleading ML results, explain the cause, and test the correction.

Start with a question: **does 100% test accuracy mean this model works for people it has never seen?** The first experiment gets 100% by recognizing repeated fictional people. Holding out whole people changes the question and the result.

Every case includes an incorrect approach, a correction, measured output, and a test for the declared requirement. The examples use synthetic data and run on a CPU. No dataset download, GPU, paid API, or account is needed to run them.

## Run an example

You need Python 3.11 or newer and Git. From a terminal:

```text
git clone https://github.com/sricharanreddycheruku/ml-failure-lab.git
cd ml-failure-lab
python -m venv .venv
```

Activate the environment in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

If PowerShell prevents activation, use `.venv\Scripts\python.exe` in place of `python` in the commands below. You do not need to change your system's execution policy.

Install and run:

```text
python -m pip install -e .
python -m ml_failure_lab list
python -m ml_failure_lab run group-leakage
```

An editable install makes the package use the files in this checkout. The first installation downloads NumPy and scikit-learn. After installation, the experiments themselves run offline.

## Available cases

| Case | Question | What the correction does |
|---|---|---|
| [group-leakage](docs/cases/group-leakage.md) | Are test entities really absent from training? | Places all records of an entity on one side of the split |
| [feature-selection](docs/cases/feature-selection.md) | Did choosing features use held-out labels? | Fits the feature selector on training rows only |
| [minority-recall](docs/cases/minority-recall.md) | Does the model detect the rare target class? | Reports minority recall and support alongside accuracy |

Run another case or print structured results:

```text
python -m ml_failure_lab run feature-selection
python -m ml_failure_lab run minority-recall --json
```

## See the requirement fail, then pass

```text
python -m ml_failure_lab verify group-leakage --approach bad
python -m ml_failure_lab verify group-leakage --approach fixed
```

The first command prints `FAIL` and exits with status 1. The second prints `PASS` and exits with status 0. The failure is intentional: it exposes the incorrect approach.

A test here is a check of a stated condition, such as whether train and test contain any of the same people. A higher score does not establish that the condition was met. Some corrections reduce an inflated score. The minority-recall correction changes the report while keeping the same classifier.

## Contribute a useful case

If you have encountered a misleading metric, a data split that answered the wrong question, or preprocessing that leaked held-out information, help turn it into a small runnable case.

Read [CONTRIBUTING.md](CONTRIBUTING.md), use the [case template](templates/case/README.md), and check the [starter issues](https://github.com/sricharanreddycheruku/ml-failure-lab/labels/good%20first%20issue). You can also propose a case before writing code.

A complete contribution includes the experiment, explanation, correction, and verification. Useful improvements to the runner, tests, setup instructions and accessible documentation are welcome too. Name-only entries, duplicate cases and untested advice do not help the project.

## Development checks

```text
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m build
```

The workflow runs tests on Windows and Linux with Python 3.11 through 3.14. Check the [Actions results](https://github.com/sricharanreddycheruku/ml-failure-lab/actions) for the actual status of each run. macOS is not currently in that matrix.

If you use [uv](https://docs.astral.sh/uv/), `uv.lock` also records a resolved dependency set:

```text
uv sync --locked --extra dev
uv run python -m pytest -q
```

The pip workflow above resolves versions within the package's supported dependency ranges. Use the lock when you want the recorded resolution.

## Scope and references

This is an educational collection. Its small fixtures expose particular mechanisms; their exact scores are not forecasts for real datasets. A passing case check verifies that case's requirement and does not certify an arbitrary ML project.

The topic already has useful resources. See [scikit-learn's common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [Equinor's ML pitfalls course](https://github.com/equinor/ml-pitfalls), and [Kapoor and Narayanan's research on leakage](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/). Our experiments use original synthetic fixtures. Cases should cite relevant sources and explain their own limits.

See the [roadmap](docs/roadmap.md), [code of conduct](CODE_OF_CONDUCT.md), and [MIT license](LICENSE).
