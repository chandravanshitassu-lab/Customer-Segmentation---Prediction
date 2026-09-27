"""
modeling.py
-----------
Customer Segmentation & Prediction — Week 11 Capstone
Random Forest, Decision Tree, Logistic Regression training,
hyperparameter tuning with GridSearchCV, and feature importance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42
TARGET_COLUMN = "Churn"

MODEL_FEATURES = [
    "Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
    "Contract_encoded", "PaperlessBilling_encoded", "PaymentMethod_encoded"
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Split
# ─────────────────────────────────────────────────────────────────────────────
def split_segment_data(segment_df: pd.DataFrame, test_size: float = 0.25
                       ) -> tuple:
    """
    Stratified train/test split for a single segment's data.

    Returns (X_train, X_test, y_train, y_test, feasible: bool, reason: str).

    feasibility checks:
      - Minimum 20 samples.
      - Both classes present in the segment.
      - Both classes present in training split after stratification.
    """
    available_features = [f for f in MODEL_FEATURES if f in segment_df.columns]
    X = segment_df[available_features]
    y = segment_df[TARGET_COLUMN]

    n = len(segment_df)
    unique_classes = y.nunique()

    if n < 20:
        return None, None, None, None, False, f"Too few samples (n={n}). Minimum 20 required."
    if unique_classes < 2:
        majority = y.mode()[0]
        return None, None, None, None, False, (
            f"Only one class present (all Churn={majority}). "
            "A classifier cannot be trained on a single-class problem."
        )

    # Check if stratification is possible
    min_class_count = y.value_counts().min()
    if min_class_count < 2:
        return None, None, None, None, False, (
            f"Minority class has only {min_class_count} sample(s). "
            "Stratified split is not feasible."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    # Post-split check
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return None, None, None, None, False, (
            "After stratified split, one partition contains only one class. "
            "This segment does not support a separate classifier."
        )

    return X_train, X_test, y_train, y_test, True, "OK"


# ─────────────────────────────────────────────────────────────────────────────
# Build sklearn Pipelines
# ─────────────────────────────────────────────────────────────────────────────
def _make_pipeline(model):
    """Wrap model in a Pipeline with median imputer + StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def build_rf_pipeline(n_estimators: int = 100, max_depth=None,
                      min_samples_split: int = 2, min_samples_leaf: int = 1,
                      max_features: str = "sqrt") -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        max_features=max_features, random_state=RANDOM_STATE, class_weight="balanced"
    )
    return _make_pipeline(rf)


def build_dt_pipeline() -> Pipeline:
    dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE,
                                class_weight="balanced")
    return _make_pipeline(dt)


def build_lr_pipeline() -> Pipeline:
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                            class_weight="balanced", solver="lbfgs")
    return _make_pipeline(lr)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter Tuning
# ─────────────────────────────────────────────────────────────────────────────
def tune_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                       scoring: str = "f1") -> tuple:
    """
    GridSearchCV over Random Forest hyperparameters.
    Uses StratifiedKFold(n_splits=5) to avoid data leakage:
      the test set is NEVER used during tuning.

    Why F1 as scoring?
      Churn prediction is a class-imbalanced problem (churn ~10.6%).
      F1 balances Precision and Recall, penalising both false positives
      and false negatives.  Accuracy alone would be misleading because
      a model predicting "no churn" for everyone would score ~89%.

    Returns (best_pipeline, best_params, best_cv_score).
    """
    n_train = len(y_train)

    # Adaptive param grid based on segment size
    if n_train < 60:
        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5, None],
            "model__min_samples_split": [2, 5],
            "model__min_samples_leaf": [1, 2],
        }
    else:
        param_grid = {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [3, 5, 10, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        }

    n_splits = min(5, int(y_train.value_counts().min()))
    n_splits = max(n_splits, 2)  # at least 2 folds

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    base_pipeline = build_rf_pipeline()
    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
        error_score=0.0
    )
    grid_search.fit(X_train, y_train)

    print(f"  [tuning] Best CV {scoring}: {grid_search.best_score_:.4f}")
    print(f"  [tuning] Best params: {grid_search.best_params_}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# ─────────────────────────────────────────────────────────────────────────────
# Feature Importance
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(pipeline: Pipeline, feature_names: list,
                            segment_name: str, save_path: str = None):
    """
    Extract and plot feature importance from a trained Random Forest pipeline.
    """
    try:
        rf_model = pipeline.named_steps["model"]
        importances = rf_model.feature_importances_
    except AttributeError:
        print(f"  [feature_importance] Model does not support feature_importances_ — skipping.")
        return

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(feature_names) * 0.55)))
    colours = plt.cm.viridis(np.linspace(0.2, 0.85, len(fi_df)))
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                   color=colours, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, fi_df["Importance"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9)
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title(f"Feature Importance — {segment_name}", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [feature_importance] Figure saved -> {save_path}")
    plt.close()

    return fi_df.sort_values("Importance", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Overfitting Analysis (Train vs Validation)
# ─────────────────────────────────────────────────────────────────────────────
def overfitting_analysis(pipeline: Pipeline, X_train, y_train, X_test, y_test,
                         segment_name: str) -> dict:
    """
    Compare training accuracy vs. test accuracy to detect overfitting.
    Also runs cross-validation on training data.

    Returns dict with train_score, test_score, cv_mean, cv_std.
    """
    train_score = pipeline.score(X_train, y_train)
    test_score = pipeline.score(X_test, y_test)

    n_splits = min(5, int(y_train.value_counts().min()))
    n_splits = max(n_splits, 2)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                scoring="accuracy", error_score=0.0)

    result = {
        "segment": segment_name,
        "train_accuracy": round(float(train_score), 4),
        "test_accuracy": round(float(test_score), 4),
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "gap": round(float(train_score - test_score), 4),
    }

    status = "OK"
    if result["gap"] > 0.15:
        status = "⚠️ Possible Overfitting (train-test gap > 15%)"
    elif result["test_accuracy"] < 0.60:
        status = "⚠️ Possible Underfitting (test accuracy < 60%)"
    result["status"] = status

    print(f"  [overfit] {segment_name}: train={train_score:.4f}  "
          f"test={test_score:.4f}  gap={result['gap']:.4f}  -> {status}")
    return result
