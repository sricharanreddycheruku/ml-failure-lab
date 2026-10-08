# Case template

Start by agreeing on the task and requirement in an issue. A case is an experiment with a specific teaching purpose, not a checklist of everything that can go wrong.

Create a module in `src/ml_failure_lab/cases/` with:

```python
def run() -> dict:
    return {
        "case": "your-case-id",
        "question": "The exact question this evaluation should answer",
        "measurements": [],  # Populate from the actual experiment.
        "requirement": "The condition your test checks",
        "interpretation": "What the measured results mean",
        "limitation": "What this example cannot establish",
    }


def verify(approach: str) -> None:
    # Run the chosen approach and check its requirement.
    # Raise ValueError with a useful explanation if it fails.
    raise NotImplementedError
```

The template deliberately cannot run until you implement the experiment. Do not submit placeholder measurements or a check that always passes.

Add documentation with the question, prerequisites, input provenance, incorrect operation, correction, run commands, tested requirement and limitations. Add tests showing that `verify("bad")` rejects the incorrect approach and `verify("fixed")` accepts the correction. Register the module and link the explanation in the README.

Use the current cases to see the complete structure. Keep the case independent of other cases and small enough for a contributor to explain every step.
