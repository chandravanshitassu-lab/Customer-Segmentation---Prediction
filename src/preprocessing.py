"""
preprocessing.py
----------------
Customer Segmentation & Prediction — Week 11 Capstone
Data loading, inspection, cleaning, and preprocessing utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
NUMERICAL_FEATURES = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL_FEATURES = ["Contract", "PaymentMethod", "PaperlessBilling"]
TARGET_COLUMN = "Churn"
ID_COLUMN = "CustomerID"

# Features used for clustering (numerical only, to avoid ordinal bias)
CLUSTERING_FEATURES = ["Tenure", "MonthlyCharges", "TotalCharges"]

MODEL_FEATURES = [
    "Tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Contract_encoded",
    "PaymentMethod_encoded",
    "PaperlessBilling_encoded",
    "SeniorCitizen"
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    """Load the customer churn CSV and return a DataFrame."""
    df = pd.read_csv(filepath)
    print(f"[load_data] Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data Inspection
# ─────────────────────────────────────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> dict:
    """
    Return a summary dict with shape, dtypes, missing values,
    duplicates, and value counts for categorical columns.
    """
    summary = {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "categorical_value_counts": {
            col: df[col].value_counts().to_dict()
            for col in df.select_dtypes(include="object").columns
            if col != ID_COLUMN
        },
        "numerical_stats": df[NUMERICAL_FEATURES].describe().to_dict(),
        "churn_distribution": df[TARGET_COLUMN].value_counts().to_dict()
        if TARGET_COLUMN in df.columns else {},
    }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame:
      - Drop exact duplicate rows.
      - Strip whitespace from string columns.
      - Cast numerical columns to the correct dtype.

    Returns a cleaned copy of the DataFrame.
    """
    df = df.copy()

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[clean_data] Dropped {before - after} duplicate row(s).")

    # Strip whitespace from object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # Ensure numeric columns are correctly typed
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"[clean_data] Cleaned dataset: {df.shape[0]} rows × {df.shape[1]} columns.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Missing-Value Treatment
# ─────────────────────────────────────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
      - Numerical columns: median imputation.
      - Categorical columns: mode (most-frequent) imputation.
    """
    df = df.copy()
    num_missing = df.isnull().sum()
    if num_missing.sum() == 0:
        print("[handle_missing_values] No missing values detected — skipping imputation.")
        return df

    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            median = df[col].median()
            df[col].fillna(median, inplace=True)
            print(f"  [impute] '{col}' -> filled with median ({median:.2f})")

    for col in df.select_dtypes(include="object").columns:
        if df[col].isnull().any():
            mode = df[col].mode()[0]
            df[col].fillna(mode, inplace=True)
            print(f"  [impute] '{col}' -> filled with mode ('{mode}')")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Categorical Encoding
# ─────────────────────────────────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns for modelling:
      - Contract    -> ordinal (Month-to-month=0, One year=1, Two year=2)
      - PaymentMethod -> label-encoded
      - PaperlessBilling -> binary (Yes=1, No=0)

    Returns DataFrame with new encoded columns added.
    The original categorical columns are retained for profiling.
    """
    df = df.copy()

    # Contract — ordinal encoding (longer contract = higher value)
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    df["Contract_encoded"] = df["Contract"].map(contract_map)

    # PaperlessBilling — binary
    pb_map = {"No": 0, "Yes": 1}
    df["PaperlessBilling_encoded"] = df["PaperlessBilling"].map(pb_map)

    # PaymentMethod — label-encoded
    le = LabelEncoder()
    df["PaymentMethod_encoded"] = le.fit_transform(df["PaymentMethod"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature Scaling (for clustering)
# ─────────────────────────────────────────────────────────────────────────────
def scale_for_clustering(df: pd.DataFrame, features: list) -> tuple:
    """
    Scale numerical features using StandardScaler.
    Returns:
      - scaled_array  : numpy array of scaled features
      - scaler        : fitted StandardScaler (for inverse_transform if needed)

    Why scaling is important for distance-based clustering:
      K-Means and Agglomerative Clustering compute Euclidean distances.
      A feature with large magnitude (e.g., TotalCharges ~ 0–8000) would
      completely dominate a feature on a small scale (e.g., Tenure ~ 1–71)
      without standardisation.  StandardScaler centres each feature at mean=0
      and scales to unit variance so every feature contributes equally.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[features])
    return scaled_array, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Full Preprocessing Pipeline for Modelling
# ─────────────────────────────────────────────────────────────────────────────
def build_model_preprocessor() -> ColumnTransformer:
    """
    Build an sklearn ColumnTransformer that:
      - Scales numerical features (StandardScaler)
      - Passes encoded categorical features through unchanged

    This preprocessor is integrated inside each segment's Random Forest
    pipeline to prevent data leakage.
    """
    numerical_cols = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen",
                      "Contract_encoded", "PaperlessBilling_encoded", "PaymentMethod_encoded"]

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, numerical_cols),
    ])

    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
# Full Preprocessing Entry-Point
# ─────────────────────────────────────────────────────────────────────────────
def full_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load -> clean -> impute -> encode categorical columns.
    Returns a fully preprocessed DataFrame ready for clustering and modelling.
    """
    df = load_data(filepath)
    df = clean_data(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    print("[full_preprocess] Preprocessing complete.")
    return df
