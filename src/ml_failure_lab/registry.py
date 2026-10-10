from ml_failure_lab.cases import (
    different_comparison_records,
    feature_selection,
    group_leakage,
    minority_recall,
    unequal_batch_metrics,
)

CASES = {
    "group-leakage": group_leakage,
    "feature-selection": feature_selection,
    "minority-recall": minority_recall,
    "different-comparison-records": different_comparison_records,
    "unequal-batch-metrics": unequal_batch_metrics,
}
