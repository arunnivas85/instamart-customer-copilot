"""Part B: RFM segmentation and a leakage-aware lapse model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs"
SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
NUMERIC_FEATURES = [
    "recency",
    "frequency",
    "monetary",
    "avg_discount",
    "cancellation_rate",
    "avg_delivery",
    "age",
    "tenure_days",
]
CATEGORICAL_FEATURES = ["favourite_category", "city", "gender"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load():
    customers = pd.read_csv(CLEAN / "customers.csv")
    orders = pd.read_csv(CLEAN / "orders.csv")
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    return customers, orders


def rfm_and_segments(k=4):
    customers, orders = load()
    completed = orders.loc[orders["status"].eq("completed")].copy()
    rfm = (
        completed.groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (SNAPSHOT - rfm["last_order"]).dt.days
    rfm = rfm.drop(columns="last_order")

    rfm_columns = ["recency", "frequency", "monetary"]
    scaled_rfm = StandardScaler().fit_transform(rfm[rfm_columns])
    diagnostics = []
    for cluster_count in range(2, 9):
        model = KMeans(
            n_clusters=cluster_count, n_init=10, random_state=42
        ).fit(scaled_rfm)
        diagnostics.append(
            {
                "k": cluster_count,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(scaled_rfm, model.labels_),
            }
        )

    if k not in {3, 4, 5}:
        raise ValueError("The business-approved segment count must be 3, 4, or 5.")
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    rfm["segment_id"] = model.fit_predict(scaled_rfm)
    profile = (
        rfm.groupby("segment_id")
        .agg(
            size=("customer_id", "size"),
            recency=("recency", "mean"),
            frequency=("frequency", "mean"),
            monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    premium_id = int(profile.loc[profile["monetary"].idxmax(), "segment_id"])
    remaining = profile.loc[profile["segment_id"].ne(premium_id)]
    frequent_id = int(remaining.loc[remaining["frequency"].idxmax(), "segment_id"])
    remaining = remaining.loc[remaining["segment_id"].ne(frequent_id)]
    dormant_id = int(remaining.loc[remaining["recency"].idxmax(), "segment_id"])
    segment_names = {
        premium_id: "Premium",
        frequent_id: "Frequent Regulars",
        dormant_id: "Dormant",
    }
    other_ids = sorted(
        remaining.loc[remaining["segment_id"].ne(dormant_id), "segment_id"].tolist()
    )
    for index, segment_id in enumerate(other_ids):
        segment_names[int(segment_id)] = (
            "Cooling Off" if len(other_ids) == 1 else f"Developing Buyers {index + 1}"
        )
    rfm["segment_name"] = rfm["segment_id"].map(segment_names)
    rfm["is_premium"] = rfm["segment_id"].eq(premium_id)

    OUT.mkdir(parents=True, exist_ok=True)
    rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "segment_id",
            "segment_name",
            "is_premium",
        ]
    ].to_csv(OUT / "customer_segments.csv", index=False)
    explained_variance = PCA(n_components=2).fit(scaled_rfm).explained_variance_ratio_
    return rfm, pd.DataFrame(diagnostics), profile, explained_variance


def _preprocessor():
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="unknown"),
            ),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def _class_metrics(y_true, prediction):
    return {
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, prediction).tolist(),
    }


def _contains_missing(matrix):
    values = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)
    return bool(pd.isna(values).any())


def build_lapse_model():
    customers, orders = load()
    products = pd.read_csv(CLEAN / "products.csv")
    cutoff = pd.Timestamp("2025-09-01 00:00:00")
    label_window_end = pd.Timestamp("2025-10-01 00:00:00")

    history = orders.loc[orders["order_ts"].lt(cutoff)].copy()
    completed = history.loc[history["status"].eq("completed")].copy()
    cohort = (
        completed.groupby("customer_id")
        .agg(
            last_order=("order_ts", "max"),
            frequency=("order_id", "nunique"),
            monetary=("order_amount", "sum"),
            avg_discount=("discount_pct", "mean"),
            avg_delivery=("delivery_minutes", "mean"),
        )
        .reset_index()
    )
    cohort["recency"] = (cutoff - cohort["last_order"]).dt.days
    cohort = cohort.drop(columns="last_order")

    final_orders = history.loc[history["status"].isin(["completed", "cancelled"])]
    status_counts = (
        final_orders.groupby("customer_id")["status"]
        .agg(
            total="count",
            cancelled=lambda status: status.eq("cancelled").sum(),
        )
        .reset_index()
    )
    cohort = cohort.merge(status_counts, on="customer_id", how="left")
    cohort["cancellation_rate"] = cohort["cancelled"] / cohort["total"]

    categorized_orders = completed.merge(
        products[["product_id", "category"]], on="product_id", how="left"
    )
    category_counts = (
        categorized_orders.groupby(["customer_id", "category"])
        .size()
        .rename("order_count")
        .reset_index()
        .sort_values(
            ["customer_id", "order_count", "category"],
            ascending=[True, False, True],
        )
    )
    favourite_category = (
        category_counts.drop_duplicates("customer_id")
        .rename(columns={"category": "favourite_category"})
        [["customer_id", "favourite_category"]]
    )
    cohort = cohort.merge(favourite_category, on="customer_id", how="left")
    cohort = cohort.merge(customers, on="customer_id", how="left")
    cohort["signup_date"] = pd.to_datetime(cohort["signup_date"])
    cohort["tenure_days"] = (cutoff - cohort["signup_date"]).dt.days

    label_orders = orders.loc[
        orders["order_ts"].ge(cutoff)
        & orders["order_ts"].lt(label_window_end)
        & orders["status"].eq("completed")
    ]
    active_customer_ids = set(label_orders["customer_id"])
    cohort["lapsed"] = (~cohort["customer_id"].isin(active_customer_ids)).astype(int)
    cohort = cohort.drop(
        columns=["signup_date", "total", "cancelled"]
    ).set_index("customer_id")

    # Features are built only from the pre-cutoff history; the September window
    # is consulted solely to create the target label.
    X = cohort[FEATURES].copy()
    y = cohort["lapsed"].copy()
    if X.isna().any().any():
        missing_columns = X.columns[X.isna().any()].tolist()
        print(
            "Features with missing raw values "
            f"(imputed in the train-fitted pipeline): {missing_columns}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    models = {}
    searches = {}
    model_grids = [
        (
            "logistic",
            LogisticRegression(max_iter=1000),
            {"model__C": [0.01, 0.1, 1, 10, 100]},
        ),
        (
            "forest",
            RandomForestClassifier(n_estimators=100, random_state=42),
            {
                "model__max_depth": [3, 5, 8],
                "model__min_samples_leaf": [5, 10, 20],
            },
        ),
    ]
    for name, estimator, grid in model_grids:
        pipeline = Pipeline([("preprocessor", _preprocessor()), ("model", estimator)])
        search = GridSearchCV(
            pipeline, grid, cv=5, scoring="f1", n_jobs=-1
        ).fit(X_train, y_train)
        prediction = search.predict(X_test)
        models[name] = {
            "best_params": search.best_params_,
            "cv_f1": float(search.best_score_),
            **_class_metrics(y_test, prediction),
        }
        searches[name] = search

    baseline_prediction = np.zeros(len(y_test), dtype=int)
    baseline = {
        "accuracy": float(accuracy_score(y_test, baseline_prediction)),
        **_class_metrics(y_test, baseline_prediction),
    }

    logistic_search = searches["logistic"]
    best_logistic = logistic_search.best_estimator_
    fitted_preprocessor = best_logistic.named_steps["preprocessor"]
    transformed_train = fitted_preprocessor.transform(X_train)
    transformed_test = fitted_preprocessor.transform(X_test)
    no_missing_features = not (
        _contains_missing(transformed_train) or _contains_missing(transformed_test)
    )
    assert no_missing_features, "Preprocessed train/test features contain missing values."
    numeric_scaler = (
        fitted_preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    )
    frequency_position = NUMERIC_FEATURES.index("frequency")
    training_frequency_mean = float(numeric_scaler.mean_[frequency_position])
    full_cohort_frequency_mean = float(X["frequency"].mean())

    train_class_counts = y_train.value_counts()
    majority_class = int(train_class_counts.idxmax())
    minority_class = 1 - majority_class
    duplicate_count = int(
        train_class_counts[majority_class] - train_class_counts[minority_class]
    )
    duplicate_X = X_train.loc[y_train.eq(minority_class)].sample(
        n=duplicate_count, replace=True, random_state=42
    )
    duplicate_y = y_train.loc[duplicate_X.index]
    X_balanced = pd.concat([X_train, duplicate_X])
    y_balanced = pd.concat([y_train, duplicate_y])
    balanced_logistic = Pipeline(
        [
            ("preprocessor", _preprocessor()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    C=logistic_search.best_params_["model__C"],
                ),
            ),
        ]
    ).fit(X_balanced, y_balanced)
    balanced_metrics = _class_metrics(
        y_test, balanced_logistic.predict(X_test)
    )

    forest = searches["forest"].best_estimator_
    forest_feature_names = forest.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    forest_importances = forest.named_steps["model"].feature_importances_
    top_forest_features = [
        {
            "feature": str(forest_feature_names[index]),
            "importance": float(forest_importances[index]),
        }
        for index in np.argsort(forest_importances)[::-1][:5]
    ]
    logistic_feature_names = best_logistic.named_steps[
        "preprocessor"
    ].get_feature_names_out()
    logistic_coefficients = best_logistic.named_steps["model"].coef_[0]
    desired_coefficients = {}
    for feature in ["recency", "frequency", "cancellation_rate"]:
        feature_name = f"numeric__{feature}"
        position = int(np.flatnonzero(logistic_feature_names == feature_name)[0])
        coefficient = float(logistic_coefficients[position])
        desired_coefficients[feature] = {
            "coefficient": coefficient,
            "sign": "positive" if coefficient > 0 else "negative",
        }

    # Select the tuned model with the best held-out lapsed-class F1; both
    # searches are already complete before this one-time test comparison.
    selected_name = max(models, key=lambda name: models[name]["f1"])
    selected_model = searches[selected_name].best_estimator_
    selected_prediction = selected_model.predict(X_test)
    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / "lapse_model.joblib"
    joblib.dump(selected_model, model_path)
    reloaded_model = joblib.load(model_path)
    persistence_matches = np.array_equal(
        selected_prediction, reloaded_model.predict(X_test)
    )
    assert persistence_matches, "Reloaded model predictions differ from original."

    segments, _, _, _ = rfm_and_segments()
    premium_customer_ids = set(
        segments.loc[segments["is_premium"], "customer_id"]
    )
    lapse_probabilities = selected_model.predict_proba(X_test)[
        :, list(selected_model.classes_).index(1)
    ]
    retention_list = pd.DataFrame(
        {
            "customer_id": X_test.index.to_numpy(),
            "lapse_probability": lapse_probabilities,
            "actually_lapsed": y_test.to_numpy(),
        }
    )
    retention_list = (
        retention_list.loc[
            retention_list["customer_id"].isin(premium_customer_ids)
        ]
        .sort_values(
            "lapse_probability", ascending=False, kind="mergesort"
        )
        .head(10)
        .reset_index(drop=True)
    )
    retention_list.to_csv(OUT / "retention_list.csv", index=False)

    return {
        "cohort_rows": int(len(cohort)),
        "lapsed": int(y.sum()),
        "lapsed_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "train_class_counts": {
            int(label): int(count) for label, count in train_class_counts.items()
        },
        "baseline": baseline,
        "models": models,
        "balanced_training_class_counts": {
            int(label): int(count)
            for label, count in y_balanced.value_counts().items()
        },
        "balanced_logistic": balanced_metrics,
        "training_frequency_mean": training_frequency_mean,
        "full_cohort_frequency_mean": full_cohort_frequency_mean,
        "no_missing_features": no_missing_features,
        "top_forest_features": top_forest_features,
        "logistic_coefficients": desired_coefficients,
        "selected_model": selected_name,
        "persistence_matches": persistence_matches,
        "retention_list": retention_list,
        "retention_list_true_lapses": int(retention_list["actually_lapsed"].sum()),
    }
