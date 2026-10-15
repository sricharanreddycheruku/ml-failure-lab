# Source-image leakage through augmentation

## Question

Does evaluation measure performance on previously unseen source images, or does it allow transformed versions of the same image to appear in both training and testing?

## The fixture

This case generates six small images using NumPy. Each source image has an original version and a horizontally flipped version. Both versions retain the same source ID and label.

The fixture is intentionally small and CPU-only. It downloads no images and requires no model training.

## Incorrect approach: random row split

The incorrect approach splits the augmented image records independently. This can put different versions of the same original source image into both training and testing.

The test checks that the train and test source-ID sets overlap. Such overlap makes the evaluation unsuitable for measuring generalization to previously unseen source images.

## Corrected approach: split by source ID

The corrected approach assigns each original source to one split before evaluating its derived images. All versions of a source stay together.

The test checks that the intersection of the training and test source-ID sets is empty.

## Why the distinction matters

When the evaluation question is performance on new source images, the original sources must be split first. Flipped or otherwise transformed versions must remain with their original source.

Evaluating new transformations of already-known sources can be a different, valid question. The appropriate split depends on the intended use of the model.

A corrected split does not necessarily produce a higher score. It changes what the evaluation measures, and the result depends on the data and task.

## Run the case

```powershell
python -m ml_failure_lab run source-image-split
python -m ml_failure_lab verify source-image-split --approach bad
python -m ml_failure_lab verify source-image-split --approach fixed
```

The `bad` approach should fail verification, while the `fixed` approach should pass.

## Tests

The tests reject the random row split and accept the source-based split. They also check that the incorrect split shares source IDs and that the corrected split has none in common.

## Limitations

This is a toy example demonstrating data-splitting logic, not a benchmark of model performance. It does not establish that source-based splitting always produces a higher score.