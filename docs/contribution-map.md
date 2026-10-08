# Find a contribution

Choose a topic you want to understand, then read its issue and comments. The issue contains a suggested input, the mistake to demonstrate, acceptance checks and limits. You can ask questions before writing code.

Check the [current open issues](https://github.com/sricharanreddycheruku/ml-failure-lab/issues) before starting. An agreed plan in the comments means someone is already working on that issue, even if GitHub shows no assignee. Choose another task in that case. The source-image augmentation task in [#7](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/7) already has a contributor's agreed plan; read that discussion before proposing any overlapping work.

## Smaller first cases

You should be comfortable reading a small Python function and working with lists or arrays. The issue can help you learn the ML concept while you implement it. These tasks still need a runnable experiment, explanation and tests.

If you want to start with arithmetic rather than model training, try [batch accuracy in #10](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/10) or [comparing predictions in #21](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/21). For #10, nine correct predictions in one batch and one incorrect prediction in another are nine correct out of ten overall. Giving each batch equal weight instead produces 50%. Your case will show and test the difference.

| Issue | What you will demonstrate | Useful starting knowledge |
|---|---|---|
| [#1](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/1) | An imputer must learn its missing-value replacement statistic from training rows | Averages and missing values |
| [#2](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/2) | Category columns must keep the same meaning during prediction | Dictionaries and category encoding |
| [#3](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/3) | Different target and prediction shapes can create a pairwise error matrix | NumPy array shapes |
| [#4](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/4) | Shuffling features and labels separately breaks their row identities | Indexing and permutations |
| [#9](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/9) | Refitting a scaler on test data changes the coordinates supplied to the model | Averages and scaling |
| [#10](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/10) | Unequal batches need record counts when computing total accuracy | Fractions and list operations |
| [#11](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/11) | The probability column must match the requested event label | Class labels and array indexing |
| [#12](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/12) | Numeric columns must follow the training feature order | Named columns and simple linear functions |
| [#13](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/13) | One aggregate score can hide failure in a predefined group | Fractions and grouping records |
| [#14](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/14) | Scoring fitting records measures training performance | A train/test split and a decision tree |
| [#21](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/21) | A paired comparison must evaluate candidates on the same held-out records | Squared errors and record IDs |

## Cases with more evaluation decisions

These involve timing, label access or repeated data splits. Start by explaining the prediction task and the data available at each step in the issue. The maintainer can help narrow the fixture before implementation.

| Issue | What you will demonstrate | Useful starting knowledge |
|---|---|---|
| [#6](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/6) | Restore target units before reporting error in those units | Affine transformations, MSE and RMSE |
| [#15](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/15) | A rolling feature must use observations available at prediction time | Ordered series and rolling windows |
| [#16](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/16) | A model frozen before a future period cannot fit using later labels | Timestamp comparisons and train/test splits |
| [#17](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/17) | Scaling must fit separately within each cross-validation fold | Cross-validation and fitted transformations |
| [#18](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/18) | A decision threshold must be selected without final-test labels | Probabilities, thresholds and validation data |
| [#19](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/19) | A target encoding must exclude the evaluated fold's labels | Group averages and validation data |
| [#20](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/20) | Resampling a test set changes the population weighting in its metrics | Confusion counts, precision and recall |
| [#22](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/22) | An outcome recorded later cannot be an input to an earlier prediction | Feature availability and timestamps |

## Verification and setup work

- [#5](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/5) adds a regression fixture for shared entity IDs with different features and unequal record counts. This suits someone who wants to learn the existing group-leakage case and pytest.
- [#8](https://github.com/sricharanreddycheruku/ml-failure-lab/issues/8) asks for a fresh Linux walkthrough. If you find a reproducible setup problem, propose a fix with before-and-after evidence. If the instructions work, a confirmation comment is useful; it does not require a PR.

## From an idea to a pull request

Comment on one available issue with the input you will use, the misleading operation and the requirement your check will enforce. An alternative fixture is welcome when it teaches the same requirement.

After the scope is agreed, follow [CONTRIBUTING.md](../CONTRIBUTING.md) and the [case template](../templates/case/README.md). For a new case, keep its module, explanation and tests in separate files named for that case. Add its registry entry and README link with small edits. This reduces overlap when several people contribute at once. The maintainer will reconcile shared-file changes during review.

Keep the full case in one PR. Use measured output and explain its limits. A corrected workflow does not have to score higher; its test must check the stated requirement. The tasks in this map are planned contributions. The README's available-cases table lists what you can run today.
