# Roadmap

The first release contains three seed cases. The next goal is a small collection that learners can run and contributors can review without heavyweight setup.

## Next cases

| Proposed case | Requirement to expose |
|---|---|
| Exact duplicates across a split | A test of genuinely new records cannot contain the same canonical record in training |
| Imputation fitted before splitting | The imputer's statistics must come from training rows |
| Categorical encoding mismatch | Learned category columns must mean the same thing during training and prediction |
| Target-array broadcasting | Error calculations must compare aligned samples, rather than every prediction against every target |
| Independent shuffling of features and labels | Feature rows and their labels must retain the same identity |
| Augmentations shared across splits | Derived records must follow their source's split for a test of new sources |
| Future observations in rolling features | A feature can use only information available at its prediction time |
| Threshold chosen from final test labels | Threshold selection needs development data, separate from final evaluation |
| Reusing test data to choose hyperparameters | The final test data must stay outside model selection |
| Scaled target metrics in the wrong units | Reported error units must match the target's declared units |

These are proposals. Agree on the prediction task and fixture before writing a case. A split that is inappropriate for one task can be appropriate for another.

## Contribution experience

Improve setup instructions using actual fresh-environment failures. Keep the case template short. Add useful boundary tests and keep numerical checks stable across the supported environments.

A future release may include a searchable documentation page and more framework-specific cases. We will add those when learners or contributors need them.

Keep examples on small public or synthetic data. A passing collection is not an automatic audit of somebody else's ML project.
