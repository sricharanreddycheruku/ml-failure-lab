# Check your own CSV split

The experiments teach a mechanism with small synthetic inputs. This command applies two specific checks to your own CSV files, locally. It does not upload data.

An ID identifies the source you want to hold out, such as a person, device or original image. Multiple records can share that ID. Choose a source ID rather than a newly assigned row number if the question is performance on new sources.

```text
python -m ml_failure_lab check-split train.csv test.csv --id-column source_id --features measurement_a measurement_b --require-new-ids --json
```

The report counts shared IDs and affected test records. The selected feature columns are also compared as exact CSV strings. It reports how many distinct feature rows occur in both files, how many test rows are affected, and how many train/test row pairs match. Counts are reported without exposing the identifier values or feature values.

## Try the supplied files

```text
python -m ml_failure_lab check-split examples/split-check/train.csv examples/split-check/test-overlap.csv --id-column source_id --features measurement_a measurement_b --require-new-ids
python -m ml_failure_lab check-split examples/split-check/train.csv examples/split-check/test-new-sources.csv --id-column source_id --features measurement_a measurement_b --require-new-ids
```

The first command exits 1. It finds one shared source ID affecting one test record, and one shared feature row. The second exits 0 because the declared source IDs are disjoint. These are original synthetic CSV files, not measurements from a deployed model.

## Choose the evaluation question

Without `--require-new-ids`, the command only reports observations and exits 0. Shared IDs can be appropriate when predicting new observations of known people or devices. With the flag, a shared ID violates the new-source requirement and exits 1. Invalid files exit 2, including empty data, missing columns, blank checked values or malformed records.

Matching feature values are informational. Two independent records can have identical measurements, so their presence is not proof of leakage. Conversely, changed or augmented versions can have different values while sharing a source. ID quality matters.

## Limits

This checks only identifiers and exact strings in the columns you select. `1` and `1.0` do not match. It does not check approximate duplicates, label availability, fitted preprocessing, model selection or whether IDs were assigned correctly. Passing the selected ID requirement does not certify the whole evaluation.

Both files are loaded into memory. Use this for manageable CSV files, not streaming or very large datasets. Blank checked features must be addressed explicitly before running the check. No new dependencies are needed.
