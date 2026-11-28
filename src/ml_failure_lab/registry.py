from ml_failure_lab.cases import (
    different_comparison_records,
    feature_selection,
    group_leakage,
    minority_recall,
    numeric_feature_order,
    source_image_split,
    test_scaler_refit,
    training_only_imputation,
    unequal_batch_metrics,
)

CASES = {
    "group-leakage": group_leakage,
    "feature-selection": feature_selection,
    "minority-recall": minority_recall,
    "source-image-split": source_image_split,
    "different-comparison-records": different_comparison_records,
    "unequal-batch-metrics": unequal_batch_metrics,
    "numeric-feature-order": numeric_feature_order,
    "test-scaler-refit": test_scaler_refit,
    "training-only-imputation": training_only_imputation,
}
