from ml_failure_lab.cases import (
    feature_selection,
    group_leakage,
    minority_recall,
    source_image_split,
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
    "source-image-split": source_image_split,
}
    "different-comparison-records": different_comparison_records,
    "unequal-batch-metrics": unequal_batch_metrics,
}
