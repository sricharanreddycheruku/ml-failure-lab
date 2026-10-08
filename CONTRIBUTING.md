# Contributing

You can contribute a small ML failure case, improve an explanation, reproduce a setup problem, or fix a real bug in the runner.

A pull request is a proposed change that the maintainer reviews before adding it to the repository. You do not need to know the whole project to contribute one case.

## Start with an issue

Check [open issues](https://github.com/sricharanreddycheruku/ml-failure-lab/issues) for work that interests you. `good first issue` marks work with a small, described scope. Comment with your intended approach so we can avoid duplicate work and resolve assumptions before implementation.

The [contribution map](docs/contribution-map.md) groups tasks by topic and starting knowledge. Read an issue's comments even when it has no assignee. If a contributor already has an agreed plan, choose another task.

For a new idea, use the case-proposal issue form. Explain the actual prediction task, the misleading step, the requirement to test, and the data you will use. A failure must be more specific than "this model is bad."

## Prepare your checkout

Fork this repository on GitHub. A fork is your own copy, where you can push changes. Clone your fork, create a virtual environment and install the development dependencies as shown in the README.

Create a branch for your change:

```text
git switch -c new/your-case-name
python -m pip install -e ".[dev]"
```

## Add a case

Use [the case template](templates/case/README.md). Add the experiment under `src/ml_failure_lab/cases/`, its explanation under `docs/cases/`, and tests under `tests/`. Register the case in `src/ml_failure_lab/registry.py`.

For a new case, use a new module, explanation and `tests/test_your_case.py` file. Keep the registry and README edits small. That lets contributors work on separate files while the maintainer resolves shared-file changes during review.

Every case needs:

- A precise question and the prerequisites a learner needs.
- Small synthetic input, or data with documented redistribution rights.
- A runnable incorrect approach and a correction.
- A check that rejects the incorrect approach and accepts the correction for the stated task.
- Real output, an explanation of its meaning and the limits of the example.

Keep the code, tests and explanation in one coherent PR. A contribution should add useful behavior or understanding. Do not split one complete case into several PRs to inflate activity.

Prefer cases that run within 30 seconds on a normal CPU. That is a project budget, not a claim about every possible machine. Avoid private data, heavyweight model downloads, paid services and network calls during experiments.

## Write the tests and explanation

Test the requirement directly. For example, assert that test entity IDs are absent from training. Do not assert that the corrected model always has a higher score. A seed makes a run repeatable; it does not make a statistical effect universal.

Use tolerances for numerical calculations where needed. Run the actual example before documenting output. Explain when an apparently incorrect approach would be valid for a different task.

Define new terms before using them. Show the exact operation that causes the error. Cite useful references and keep the explanation in your own words. A complete translation needs a fluent reviewer who can check technical meaning.

AI assistance is allowed, but you are responsible for understanding and testing the contribution. Disclose substantial assistance in the PR description. We will review the code, reasoning and provenance, and may ask you to explain a result or revise an assertion.

## Verify and submit

```text
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m ml_failure_lab run your-case
python -m ml_failure_lab verify your-case --approach bad
python -m ml_failure_lab verify your-case --approach fixed
```

The `bad` verification should fail intentionally. The complete test suite should pass because it checks that this failure is detected.

Commit your change, push your branch to your fork, and open a pull request against this repository's `main` branch. Use the PR template to describe the failure and the commands you actually ran.

The maintainer aims to acknowledge contributions within one day and review them within two days when available. This is a review target, not a guarantee. If a contribution needs a scientific correction, we will explain the missing assumption or failing condition.

Contributions are distributed under this project's MIT license. Submit only work and data you have the right to share. Follow the [code of conduct](CODE_OF_CONDUCT.md).
