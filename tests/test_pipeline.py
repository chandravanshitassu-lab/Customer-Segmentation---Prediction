"""
test_pipeline.py
----------------
Customer Segmentation & Prediction — Week 11 Capstone
Unit and integration tests for the ML pipeline.

Run with:
    pytest tests/test_pipeline.py -v
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import (
    load_data, clean_data, handle_missing_values,
    encode_categoricals, scale_for_clustering,
    CLUSTERING_FEATURES, TARGET_COLUMN, MODEL_FEATURES
)
from clustering import (
    run_elbow_method, train_kmeans, train_agglomerative, train_dbscan
)
from segmentation import build_segment_profiles, assign_segment_names
from modeling import split_segment_data, build_rf_pipeline, build_dt_pipeline
from evaluation import evaluate_model

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "customer_churn.csv")


@pytest.fixture(scope="module")
def raw_df():
    """Load the raw dataset once for all tests."""
    return load_data(DATA_PATH)


@pytest.fixture(scope="module")
def cleaned_df(raw_df):
    """Cleaned and encoded DataFrame."""
    df = clean_data(raw_df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    return df


@pytest.fixture(scope="module")
def scaled_data_and_scaler(cleaned_df):
    return scale_for_clustering(cleaned_df, CLUSTERING_FEATURES)


@pytest.fixture(scope="module")
def scaled_data(scaled_data_and_scaler):
    return scaled_data_and_scaler[0]


@pytest.fixture(scope="module")
def clustered_df(cleaned_df, scaled_data):
    """DataFrame with KMeans cluster labels added."""
    df = cleaned_df.copy()
    _, labels = train_kmeans(scaled_data, n_clusters=3)
    df["Cluster"] = labels
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Data Loading
# ─────────────────────────────────────────────────────────────────────────────
class TestDataLoading:
    def test_file_exists(self):
        """Dataset file must exist."""
        assert os.path.exists(DATA_PATH), f"Dataset not found at {DATA_PATH}"

    def test_load_returns_dataframe(self, raw_df):
        """load_data must return a DataFrame."""
        assert isinstance(raw_df, pd.DataFrame)

    def test_dataframe_not_empty(self, raw_df):
        """DataFrame must have rows and columns."""
        assert raw_df.shape[0] > 0, "DataFrame has no rows"
        assert raw_df.shape[1] > 0, "DataFrame has no columns"

    def test_expected_columns_present(self, raw_df):
        """Core columns must be present."""
        required = ["Tenure", "MonthlyCharges", "TotalCharges", "Churn"]
        for col in required:
            assert col in raw_df.columns, f"Missing required column: {col}"

    def test_target_column_binary(self, raw_df):
        """Churn column must be binary (0/1)."""
        assert set(raw_df["Churn"].unique()).issubset({0, 1}), \
            "Churn column contains values other than 0 and 1"

    def test_minimum_rows(self, raw_df):
        """Expect at least 400 rows."""
        assert raw_df.shape[0] >= 400, f"Too few rows: {raw_df.shape[0]}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
class TestPreprocessing:
    def test_clean_data_no_duplicates(self, cleaned_df):
        """No duplicate rows after cleaning."""
        assert cleaned_df.duplicated().sum() == 0

    def test_no_missing_values_after_imputation(self, cleaned_df):
        """All missing values should be handled."""
        assert cleaned_df.isnull().sum().sum() == 0, \
            f"Missing values remain: {cleaned_df.isnull().sum()}"

    def test_encoding_creates_expected_columns(self, cleaned_df):
        """Encoding must create Contract_encoded, PaperlessBilling_encoded, PaymentMethod_encoded."""
        for col in ["Contract_encoded", "PaperlessBilling_encoded", "PaymentMethod_encoded"]:
            assert col in cleaned_df.columns, f"Missing encoded column: {col}"

    def test_contract_encoded_range(self, cleaned_df):
        """Contract_encoded must be in {0, 1, 2}."""
        assert set(cleaned_df["Contract_encoded"].unique()).issubset({0, 1, 2})

    def test_paperless_billing_encoded_binary(self, cleaned_df):
        """PaperlessBilling_encoded must be binary."""
        assert set(cleaned_df["PaperlessBilling_encoded"].unique()).issubset({0, 1})

    def test_numerical_columns_numeric(self, cleaned_df):
        """Numerical columns must be numeric dtype."""
        for col in ["Tenure", "MonthlyCharges", "TotalCharges"]:
            assert pd.api.types.is_numeric_dtype(cleaned_df[col]), \
                f"Column {col} is not numeric"

    def test_scaling_output_shape(self, scaled_data, cleaned_df):
        """Scaled data must have same number of rows as input."""
        assert scaled_data.shape[0] == len(cleaned_df)
        assert scaled_data.shape[1] == len(CLUSTERING_FEATURES)

    def test_scaling_zero_mean(self, scaled_data):
        """Each feature should have approximately zero mean after StandardScaler."""
        means = scaled_data.mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_scaling_unit_variance(self, scaled_data):
        """Each feature should have approximately unit variance."""
        stds = scaled_data.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-1)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Clustering
# ─────────────────────────────────────────────────────────────────────────────
class TestClustering:
    def test_elbow_method_returns_dict(self, scaled_data):
        """run_elbow_method must return a dict keyed by K."""
        results = run_elbow_method(scaled_data, k_range=range(2, 5))
        assert isinstance(results, dict)
        assert all(k in results for k in range(2, 5))

    def test_elbow_method_inertia_decreasing(self, scaled_data):
        """Inertia must decrease as K increases."""
        results = run_elbow_method(scaled_data, k_range=range(2, 7))
        ks = sorted(results.keys())
        inertias = [results[k]["inertia"] for k in ks]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1], \
                f"Inertia did not decrease: K={ks[i]} → K={ks[i+1]}"

    def test_kmeans_labels_count(self, scaled_data):
        """K-Means must produce exactly n_clusters unique labels."""
        for k in [2, 3, 4]:
            _, labels = train_kmeans(scaled_data, n_clusters=k)
            assert len(np.unique(labels)) == k, \
                f"Expected {k} clusters, got {len(np.unique(labels))}"

    def test_kmeans_labels_length(self, scaled_data, cleaned_df):
        """Labels array must have one label per customer."""
        _, labels = train_kmeans(scaled_data, n_clusters=3)
        assert len(labels) == len(cleaned_df)

    def test_kmeans_labels_are_integers(self, scaled_data):
        """K-Means labels must be integer arrays."""
        _, labels = train_kmeans(scaled_data, n_clusters=3)
        assert np.issubdtype(labels.dtype, np.integer), \
            f"Expected integer labels, got {labels.dtype}"

    def test_agglomerative_labels_count(self, scaled_data):
        """Agglomerative must produce exactly n_clusters labels."""
        _, labels, _ = train_agglomerative(scaled_data, n_clusters=3)
        assert len(np.unique(labels)) == 3

    def test_dbscan_noise_label(self, scaled_data):
        """DBSCAN noise label (-1) must be handled without error."""
        _, labels, n_clusters, n_noise, _ = train_dbscan(scaled_data, eps=0.5, min_samples=5)
        assert isinstance(n_noise, int)
        assert n_noise >= 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Segmentation
# ─────────────────────────────────────────────────────────────────────────────
class TestSegmentation:
    def test_profiles_shape(self, clustered_df):
        """Profile DataFrame must have one row per cluster."""
        n_clusters = clustered_df["Cluster"].nunique()
        profiles = build_segment_profiles(clustered_df, label_col="Cluster")
        assert len(profiles) == n_clusters

    def test_profiles_count_sum(self, clustered_df):
        """Profile counts must sum to total customers."""
        profiles = build_segment_profiles(clustered_df, label_col="Cluster")
        assert profiles["Count"].sum() == len(clustered_df)

    def test_profiles_percentage_sums_to_100(self, clustered_df):
        """Percentages must sum to ~100%."""
        profiles = build_segment_profiles(clustered_df, label_col="Cluster")
        total_pct = profiles["Percentage"].sum()
        assert abs(total_pct - 100.0) < 1.0, \
            f"Percentages sum to {total_pct}, expected ~100"

    def test_segment_names_all_assigned(self, clustered_df):
        """Every cluster must receive a unique name."""
        profiles = build_segment_profiles(clustered_df, label_col="Cluster")
        names = assign_segment_names(profiles)
        assert len(names) == clustered_df["Cluster"].nunique()
        # All names must be unique strings
        assert len(set(names.values())) == len(names), "Duplicate segment names assigned"

    def test_churn_rate_non_negative(self, clustered_df):
        """Churn rates must be non-negative."""
        profiles = build_segment_profiles(clustered_df, label_col="Cluster")
        assert (profiles["Churn_Rate"] >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Modeling
# ─────────────────────────────────────────────────────────────────────────────
class TestModeling:
    def test_split_feasibility_check(self, clustered_df):
        """split_segment_data must return feasible=False if segment is too small."""
        tiny_df = clustered_df.iloc[:5].copy()
        _, _, _, _, feasible, reason = split_segment_data(tiny_df)
        assert not feasible
        assert "Too few" in reason or "samples" in reason.lower()

    def test_split_feasibility_one_class(self, clustered_df):
        """split_segment_data must return feasible=False if only one class exists."""
        one_class_df = clustered_df[clustered_df["Churn"] == 0].copy().head(50)
        _, _, _, _, feasible, reason = split_segment_data(one_class_df)
        assert not feasible

    def test_rf_pipeline_builds(self):
        """RF pipeline must instantiate without error."""
        pipe = build_rf_pipeline()
        assert pipe is not None

    def test_dt_pipeline_builds(self):
        """DT pipeline must instantiate without error."""
        pipe = build_dt_pipeline()
        assert pipe is not None

    def test_rf_trains_and_predicts(self, clustered_df):
        """Random Forest must train and produce binary predictions."""
        df_seg = clustered_df[clustered_df["Cluster"] == clustered_df["Cluster"].iloc[0]].copy()
        X_train, X_test, y_train, y_test, feasible, _ = split_segment_data(df_seg)
        if not feasible:
            pytest.skip("Segment not suitable for modelling")

        available_features = [f for f in MODEL_FEATURES if f in df_seg.columns]
        X_train = X_train[available_features]
        X_test = X_test[available_features]

        pipe = build_rf_pipeline()
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        assert len(preds) == len(y_test)
        assert set(preds).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Evaluation
# ─────────────────────────────────────────────────────────────────────────────
class TestEvaluation:
    def test_evaluate_model_returns_all_keys(self, clustered_df):
        """evaluate_model must return all required metric keys."""
        df_seg = clustered_df[clustered_df["Cluster"] == clustered_df["Cluster"].value_counts().idxmax()].copy()
        X_train, X_test, y_train, y_test, feasible, _ = split_segment_data(df_seg)
        if not feasible:
            pytest.skip("Segment not suitable for modelling")

        available_features = [f for f in MODEL_FEATURES if f in df_seg.columns]
        X_train = X_train[available_features]
        X_test = X_test[available_features]

        pipe = build_rf_pipeline()
        pipe.fit(X_train, y_train)
        result = evaluate_model(pipe, X_test, y_test, "TestSeg", "RF")

        for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
            assert key in result, f"Missing key: {key}"

    def test_accuracy_in_valid_range(self, clustered_df):
        """Accuracy must be between 0 and 1."""
        df_seg = clustered_df[clustered_df["Cluster"] == clustered_df["Cluster"].value_counts().idxmax()].copy()
        X_train, X_test, y_train, y_test, feasible, _ = split_segment_data(df_seg)
        if not feasible:
            pytest.skip("Segment not suitable for modelling")

        available_features = [f for f in MODEL_FEATURES if f in df_seg.columns]
        X_train = X_train[available_features]
        X_test = X_test[available_features]

        pipe = build_rf_pipeline()
        pipe.fit(X_train, y_train)
        result = evaluate_model(pipe, X_test, y_test, "TestSeg", "RF")

        assert 0.0 <= result["accuracy"] <= 1.0
        assert 0.0 <= result["precision"] <= 1.0
        assert 0.0 <= result["recall"] <= 1.0
        assert 0.0 <= result["f1"] <= 1.0

    def test_confusion_matrix_is_2x2(self, clustered_df):
        """Confusion matrix must be 2×2."""
        df_seg = clustered_df[clustered_df["Cluster"] == clustered_df["Cluster"].value_counts().idxmax()].copy()
        X_train, X_test, y_train, y_test, feasible, _ = split_segment_data(df_seg)
        if not feasible:
            pytest.skip("Segment not suitable for modelling")

        available_features = [f for f in MODEL_FEATURES if f in df_seg.columns]
        X_train = X_train[available_features]
        X_test = X_test[available_features]

        pipe = build_rf_pipeline()
        pipe.fit(X_train, y_train)
        result = evaluate_model(pipe, X_test, y_test, "TestSeg", "RF")

        cm = result["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2


# ─────────────────────────────────────────────────────────────────────────────
# Tests: File Outputs
# ─────────────────────────────────────────────────────────────────────────────
class TestFileOutputs:
    def test_segmentation_data_exists(self):
        """segmentation_data.csv must exist after pipeline run."""
        path = os.path.join(BASE_DIR, "segmentation_data.csv")
        assert os.path.exists(path), "segmentation_data.csv not found — run run_project.py first"

    def test_segment_profiles_md_exists(self):
        """segment_profiles.md must exist."""
        path = os.path.join(BASE_DIR, "segment_profiles.md")
        assert os.path.exists(path), "segment_profiles.md not found — run run_project.py first"

    def test_model_evaluation_csv_exists(self):
        """model_evaluation_results.csv must exist."""
        path = os.path.join(BASE_DIR, "model_evaluation_results.csv")
        assert os.path.exists(path), \
            "model_evaluation_results.csv not found — run run_project.py first"

    def test_model_evaluation_csv_has_data(self):
        """Evaluation CSV must have at least one result row."""
        path = os.path.join(BASE_DIR, "model_evaluation_results.csv")
        if not os.path.exists(path):
            pytest.skip("CSV not yet generated — run run_project.py first")
        df = pd.read_csv(path)
        assert len(df) > 0, "model_evaluation_results.csv is empty"

    def test_figures_directory_has_files(self):
        """outputs/figures must contain PNG files."""
        fig_dir = os.path.join(BASE_DIR, "outputs", "figures")
        if not os.path.exists(fig_dir):
            pytest.skip("figures directory not yet created — run run_project.py first")
        png_files = [f for f in os.listdir(fig_dir) if f.endswith(".png")]
        assert len(png_files) > 0, "No PNG figures found in outputs/figures/"

    def test_requirements_txt_exists(self):
        """requirements.txt must exist."""
        path = os.path.join(BASE_DIR, "requirements.txt")
        assert os.path.exists(path)

    def test_pdf_or_txt_recommendations_exist(self):
        """Business recommendations PDF (or .txt fallback) must exist."""
        pdf = os.path.join(BASE_DIR, "business_recommendations.pdf")
        txt = os.path.join(BASE_DIR, "business_recommendations.txt")
        exists = os.path.exists(pdf) or os.path.exists(txt)
        assert exists, "Neither business_recommendations.pdf nor .txt found"
