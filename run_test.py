"""Run Part B segmentation/lapse modeling and Part C support triage."""

from src.customer_models import build_lapse_model, rfm_and_segments
from src.triage import compare_prompts


def main():
    segments, diagnostics, profile, explained_variance = rfm_and_segments()
    print("RFM rows:", len(segments))
    print(
        "RFM summary:\n",
        segments[["recency", "frequency", "monetary"]].agg(
            ["mean", "median", "min", "max"]
        ).to_string(),
    )
    print("K-means diagnostics:\n", diagnostics.to_string(index=False))
    print("Segment profile:\n", profile.to_string(index=False))
    print(
        "PCA explained variance ratios:",
        explained_variance,
        "total:",
        explained_variance.sum(),
    )

    report = build_lapse_model()
    print(
        "Cohort/split:",
        report["cohort_rows"],
        "customers;",
        report["lapsed"],
        "lapsed;",
        report["train_rows"],
        "train /",
        report["test_rows"],
        "test",
    )
    print("Baseline:", report["baseline"])
    for model_name, metrics in report["models"].items():
        print(f"{model_name.title()} grid-search and test metrics:", metrics)
    print(
        "Training-only scaled frequency mean:",
        report["training_frequency_mean"],
        "| full-cohort mean:",
        report["full_cohort_frequency_mean"],
    )
    print("No missing transformed features:", report["no_missing_features"])
    print(
        "Balanced logistic (train class counts "
        f"{report['balanced_training_class_counts']}):",
        report["balanced_logistic"],
    )
    print("Top random-forest features:", report["top_forest_features"])
    print("Logistic coefficients:", report["logistic_coefficients"])
    print(
        "Selected model:",
        report["selected_model"],
        "| reloaded predictions exact:",
        report["persistence_matches"],
    )
    print("Retention list:\n", report["retention_list"].to_string(index=False))
    print("True lapses in retention list:", report["retention_list_true_lapses"])
    print(
        "The retention list is illustrative and is not a leakage-free evaluation "
        "because its Premium filter uses whole-window RFM."
    )

    print("\n=== PART C: TRIAGE ===")
    triage = compare_prompts()
    print("Zero-shot:", triage["zero_shot"])
    print("Zero-shot confusion matrix:\n", triage["zero_shot_confusion"])
    print("Few-shot:", triage["few_shot"])
    print("6000-token zero-shot budget demo:", triage["budget_demo"])

if __name__ == "__main__":
    main()
