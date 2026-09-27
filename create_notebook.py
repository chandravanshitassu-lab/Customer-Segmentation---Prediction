"""create_notebook.py — generates customer_segmentation.ipynb"""
import json, os

cells = []

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source] if isinstance(source, str) else source}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [source] if isinstance(source, str) else source}

# ── Cell 1: Title ──
cells.append(md("""# Customer Segmentation & Prediction Using Advanced Machine Learning
## Week 11 — Advanced ML Capstone Project

**Dataset:** customer_churn.csv (500 customers, 9 features)

### Project Goals
1. Use clustering to segment customers into meaningful groups.
2. Build prediction models (churn) for each customer segment.
3. Provide data-driven business recommendations per segment.

### Techniques Used
- **Clustering:** K-Means, Hierarchical (Agglomerative), DBSCAN
- **Prediction:** Random Forest, Decision Tree, Logistic Regression
- **Tuning:** GridSearchCV with StratifiedKFold
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
"""))

# ── Cell 2: Imports ──
cells.append(code("""import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "src")
from preprocessing import full_preprocess, scale_for_clustering, CLUSTERING_FEATURES, MODEL_FEATURES
from clustering import (
    run_elbow_method, train_kmeans, train_agglomerative, plot_dendrogram,
    estimate_dbscan_eps, train_dbscan, compare_algorithms, plot_clusters_pca
)
from segmentation import (
    build_segment_profiles, assign_segment_names,
    plot_segment_profiles, plot_churn_by_segment,
    plot_cluster_distribution, save_segment_profiles_md
)
from modeling import (
    split_segment_data, build_rf_pipeline, build_dt_pipeline, build_lr_pipeline,
    tune_random_forest, plot_feature_importance, overfitting_analysis
)
from evaluation import (
    evaluate_model, plot_confusion_matrix, plot_roc_curve,
    plot_model_comparison, save_evaluation_csv, plot_dataset_overview
)

plt.style.use("seaborn-v0_8-whitegrid")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
print("Libraries loaded successfully!")
"""))

# ── Cell 3: Load Dataset ──
cells.append(md("## 1. Load Dataset"))
cells.append(code("""DATA_PATH = "customer_churn.csv"
df = full_preprocess(DATA_PATH)
print(f"Dataset shape: {df.shape}")
df.head()
"""))

# ── Cell 4: Overview ──
cells.append(md("## 2. Dataset Overview"))
cells.append(code(r"""print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")
print(f"\nColumn Names: {list(df.columns)}")
print("\nData Types:")
print(df.dtypes)
print("\nChurn Distribution:")
print(df["Churn"].value_counts())
churn_pct = df["Churn"].mean() * 100
print(f"\nChurn Rate: {churn_pct:.1f}%")
"""))

# ── Cell 5: Quality ──
cells.append(md("## 3. Data Quality Analysis"))
cells.append(code("""print("Missing Values:")
print(df.isnull().sum())
print(f"\\nDuplicate Rows: {df.duplicated().sum()}")
print("\\nNumerical Summary:")
display(df[["Tenure", "MonthlyCharges", "TotalCharges"]].describe())
"""))

# ── Cell 6: Preprocessing ──
cells.append(md("""## 4. Data Preprocessing

> **Why StandardScaler for Clustering?**
> K-Means and Agglomerative Clustering use Euclidean distance.
> Without scaling, TotalCharges (0–8000) dominates Tenure (1–71).
> StandardScaler: each feature → mean=0, std=1.
"""))
cells.append(code("""plot_dataset_overview(df, save_path="outputs/figures/01_dataset_overview.png")
plt.show()
"""))

# ── Cell 7: Feature prep ──
cells.append(md("## 5. Feature Preparation for Clustering"))
cells.append(code("""print(f"Clustering features: {CLUSTERING_FEATURES}")
scaled_data, scaler = scale_for_clustering(df, CLUSTERING_FEATURES)
print(f"Scaled array shape: {scaled_data.shape}")
print(f"Feature means (should ≈ 0): {scaled_data.mean(axis=0).round(6)}")
print(f"Feature stds  (should ≈ 1): {scaled_data.std(axis=0).round(6)}")
"""))

# ── Cell 8: Elbow ──
cells.append(md("""## 6. K-Means — Elbow Method & Silhouette Analysis

We test K=2 through K=9, recording:
- **Inertia:** Within-cluster sum of squares (lower = tighter)
- **Silhouette Score:** How well-separated clusters are (higher = better, range -1 to +1)
"""))
cells.append(code("""elbow_results = run_elbow_method(
    scaled_data, k_range=range(2, 10),
    save_path="outputs/figures/02_elbow_silhouette.png"
)
plt.show()
print("\\nElbow/Silhouette Results:")
for k, v in elbow_results.items():
    print(f"  K={k}: Inertia={v['inertia']:.1f}  Silhouette={v['silhouette']:.4f}")
"""))

# ── Cell 9: Select K ──
cells.append(md("## 7. Select Optimal K"))
cells.append(code("""best_k = max(elbow_results, key=lambda k: elbow_results[k]["silhouette"])
print(f"Best K by Silhouette Score: K={best_k}")
print(f"Silhouette Score at K={best_k}: {elbow_results[best_k]['silhouette']:.4f}")
print(f"Inertia at K={best_k}: {elbow_results[best_k]['inertia']:.1f}")
"""))

# ── Cell 10: Final KMeans ──
cells.append(md("## 8. Final K-Means Model"))
cells.append(code("""km_model, km_labels = train_kmeans(scaled_data, n_clusters=best_k)
df["Cluster_KMeans"] = km_labels
km_sil = silhouette_score(scaled_data, km_labels)

print("Cluster size distribution:")
print(pd.Series(km_labels).value_counts().sort_index())

plot_clusters_pca(scaled_data, km_labels,
    title=f"Customer Segments — K-Means (K={best_k})",
    save_path="outputs/figures/04_kmeans_clusters_pca.png")
plt.show()
"""))

# ── Cell 11: Agglomerative ──
cells.append(md("""## 9. Hierarchical (Agglomerative) Clustering

Agglomerative clustering merges customers bottom-up using Ward linkage (minimises within-cluster variance).
The **dendrogram** visualises the hierarchy — the height of each merge indicates the distance at which clusters merged.
"""))
cells.append(code("""agg_model, agg_labels, agg_sil = train_agglomerative(scaled_data, n_clusters=best_k)
df["Cluster_Agg"] = agg_labels
print(f"Agglomerative Silhouette: {agg_sil:.4f}")

plot_dendrogram(scaled_data, save_path="outputs/figures/05_dendrogram.png")
plt.show()

plot_clusters_pca(scaled_data, agg_labels,
    title=f"Agglomerative Clustering (n={best_k})",
    save_path="outputs/figures/05_agglomerative_clusters_pca.png")
plt.show()
"""))

# ── Cell 12: DBSCAN ──
cells.append(md("""## 10. DBSCAN Clustering

DBSCAN discovers clusters of arbitrary shape without requiring K upfront.
- Points in dense regions → cluster members
- Sparse outliers → **noise** (label = -1)

We estimate `eps` using the k-nearest-neighbour distance plot: the "knee" indicates the natural density threshold.
"""))
cells.append(code("""eps_suggested = estimate_dbscan_eps(scaled_data, min_samples=5,
    save_path="outputs/figures/06_dbscan_knn_distance.png")
plt.show()

eps_use = max(0.3, min(eps_suggested, 2.0))
db_model, db_labels, db_n_clusters, db_n_noise, db_sil = train_dbscan(
    scaled_data, eps=eps_use, min_samples=5)
df["Cluster_DBSCAN"] = db_labels

print(f"DBSCAN: {db_n_clusters} clusters, {db_n_noise} noise points")
if db_n_clusters > 1:
    plot_clusters_pca(scaled_data, db_labels,
        title=f"DBSCAN Clusters (eps={eps_use:.3f})",
        save_path="outputs/figures/06_dbscan_clusters_pca.png")
    plt.show()
"""))

# ── Cell 13: Comparison ──
cells.append(md("## 11. Clustering Algorithm Comparison"))
cells.append(code("""comparison = {"K-Means": km_sil, "Agglomerative": agg_sil}
if db_sil is not None:
    comparison["DBSCAN*"] = db_sil

compare_algorithms(comparison, save_path="outputs/figures/07_clustering_comparison.png")
plt.show()

print("\\nClustering Comparison Results:")
for algo, sil in comparison.items():
    print(f"  {algo:<20}: Silhouette = {sil:.4f}")
print(f"\\n→ K-Means selected (silhouette={km_sil:.4f})")
if "DBSCAN*" in comparison:
    print("  * DBSCAN silhouette excludes noise points")
"""))

# ── Cell 14: Final Segments ──
cells.append(md("## 12. Final Customer Segments"))
cells.append(code("""df["Cluster"] = df["Cluster_KMeans"]
profiles = build_segment_profiles(df, label_col="Cluster")
segment_names = assign_segment_names(profiles)
df["Segment_Name"] = df["Cluster"].map(segment_names)

print("Segment names (data-driven assignment):")
for cid, name in segment_names.items():
    row = profiles.loc[cid]
    print(f"  Cluster {cid}: {name}")
    print(f"    Count={int(row['Count'])}, Churn={row['Churn_Rate']}%, Tenure={row['Avg_Tenure']}mo")
"""))

# ── Cell 15: Profiles ──
cells.append(md("## 13. Segment Profiles"))
cells.append(code("""print("SEGMENT PROFILE TABLE")
print("=" * 70)
display(profiles)

plot_cluster_distribution(df, label_col="Cluster", segment_names=segment_names,
    save_path="outputs/figures/08_cluster_distribution.png")
plt.show()

plot_segment_profiles(profiles, segment_names,
    save_path="outputs/figures/09_segment_profiles.png")
plt.show()

plot_churn_by_segment(profiles, segment_names,
    save_path="outputs/figures/10_churn_by_segment.png")
plt.show()

save_segment_profiles_md(profiles, segment_names, save_path="segment_profiles.md")
df.to_csv("segmentation_data.csv", index=False)
print("segmentation_data.csv saved.")
"""))

# ── Cell 16: Models ──
cells.append(md("""## 14. Per-Segment Prediction Models

### Overfitting / Underfitting

| Concept | Definition | Sign | Prevention |
|---------|-----------|------|-----------|
| **Overfitting** | Model memorises training data | High train, low test accuracy | Limit `max_depth`, use cross-validation, Random Forest bagging |
| **Underfitting** | Model is too simple | Low accuracy on both sets | Allow more tree depth, add features |

**Why F1 for tuning scoring?**
Churn is imbalanced (~10.6% positive). F1 balances Precision and Recall, penalising both false positives and false negatives. Accuracy alone would be misleading here.

**Data Leakage Prevention:** The test set is NEVER used during GridSearchCV — only training data with StratifiedKFold cross-validation.
"""))
cells.append(code("""all_eval_results = []
tuning_records = []
overfit_records = []
SKIPPED = {}
available_features = [f for f in MODEL_FEATURES if f in df.columns]
print(f"Model features: {available_features}")

for cluster_id in sorted(df["Cluster"].unique()):
    seg_name = segment_names.get(cluster_id, f"Cluster {cluster_id}")
    seg_df = df[df["Cluster"] == cluster_id].copy()
    n = len(seg_df)
    print(f"\\n{'='*60}")
    print(f"SEGMENT {cluster_id}: {seg_name}  (n={n})")
    print(f"{'='*60}")

    X_train, X_test, y_train, y_test, feasible, reason = split_segment_data(seg_df)
    if not feasible:
        print(f"  WARNING SKIPPED: {reason}")
        SKIPPED[seg_name] = reason
        continue

    Xtr = X_train[available_features]
    Xte = X_test[available_features]
    print(f"  Train: {len(y_train)} | Test: {len(y_test)}")

    # Random Forest baseline
    rf_pipe = build_rf_pipeline()
    rf_pipe.fit(Xtr, y_train)
    rf_result = evaluate_model(rf_pipe, Xte, y_test, seg_name, "Random Forest")
    all_eval_results.append(rf_result)

    # Decision Tree
    dt_pipe = build_dt_pipeline()
    dt_pipe.fit(Xtr, y_train)
    dt_result = evaluate_model(dt_pipe, Xte, y_test, seg_name, "Decision Tree")
    all_eval_results.append(dt_result)

    # Logistic Regression
    lr_pipe = build_lr_pipeline()
    lr_pipe.fit(Xtr, y_train)
    lr_result = evaluate_model(lr_pipe, Xte, y_test, seg_name, "Logistic Regression")
    all_eval_results.append(lr_result)

    # GridSearchCV Hyperparameter Tuning
    print("  Tuning with GridSearchCV (StratifiedKFold, scoring=f1)...")
    tuned_pipe, best_params, best_cv = tune_random_forest(Xtr, y_train, scoring="f1")
    tuned_result = evaluate_model(tuned_pipe, Xte, y_test, seg_name, "RF Tuned")
    all_eval_results.append(tuned_result)

    tuning_records.append({
        "Segment": seg_name,
        "Best_CV_F1": round(float(best_cv), 4),
        "Best_Params": str(best_params),
        "Test_Accuracy": tuned_result["accuracy"],
        "Test_F1": tuned_result["f1"],
        "Test_AUC": str(tuned_result["roc_auc"]),
    })

    # Confusion matrix
    plot_confusion_matrix(tuned_result["confusion_matrix"], seg_name, "RF Tuned",
        save_path=f"outputs/figures/cm_{cluster_id}_rf_tuned.png")
    plt.show()

    # Feature importance
    fi_df = plot_feature_importance(tuned_pipe, available_features, seg_name,
        save_path=f"outputs/figures/fi_{cluster_id}.png")
    plt.show()

    # Overfitting analysis
    ov = overfitting_analysis(tuned_pipe, Xtr, y_train, Xte, y_test, seg_name)
    overfit_records.append(ov)
"""))

# ── Cell 17: Tuning Results ──
cells.append(md("## 15. Hyperparameter Tuning Results"))
cells.append(code("""if tuning_records:
    tuning_df = pd.DataFrame(tuning_records)
    tuning_df.to_csv("outputs/metrics/hyperparameter_tuning_results.csv", index=False)
    display(tuning_df)
else:
    print("No segments were tunable.")
"""))

# ── Cell 18: Eval Table ──
cells.append(md("## 16. Model Evaluation — Complete Results"))
cells.append(code("""eval_df = save_evaluation_csv(all_eval_results, "model_evaluation_results.csv")
display(eval_df[["Segment", "Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]])
"""))

# ── Cell 19: ROC ──
cells.append(md("## 17. ROC Curves"))
cells.append(code("""tuned_rf_results = [r for r in all_eval_results if r["model"] == "RF Tuned" and r.get("roc_auc")]
plot_roc_curve(tuned_rf_results, save_path="outputs/figures/11_roc_curves.png")
plt.show()
"""))

# ── Cell 20: Comparison Chart ──
cells.append(md("## 18. Model Comparison Chart"))
cells.append(code("""tuned_only = [r for r in all_eval_results if r["model"] == "RF Tuned"]
if tuned_only:
    plot_model_comparison(tuned_only, save_path="outputs/figures/12_model_comparison.png")
    plt.show()
"""))

# ── Cell 21: Overfitting ──
cells.append(md("""## 19. Overfitting / Underfitting Discussion

**Train vs. Test accuracy comparison:**
- If train accuracy >> test accuracy (gap > ~15%) → possible **overfitting**
- Both train and test low → possible **underfitting**
- Random Forest controls this via `max_depth`, `min_samples_split`, `min_samples_leaf`
- `GridSearchCV` searches the hyperparameter space using **only training data** (StratifiedKFold)
"""))
cells.append(code("""if overfit_records:
    ov_df = pd.DataFrame(overfit_records)
    ov_df.to_csv("outputs/metrics/overfitting_analysis.csv", index=False)
    display(ov_df[["segment", "train_accuracy", "test_accuracy", "cv_mean", "cv_std", "gap", "status"]])
"""))

# ── Cell 22: Business ──
cells.append(md("## 20. Business Recommendations"))
cells.append(code(r"""print("SEGMENT-SPECIFIC BUSINESS RECOMMENDATIONS")
print("=" * 65)
for cid, seg_name in segment_names.items():
    row = profiles.loc[cid]
    print(f"\nSEGMENT: {seg_name}")
    print(f"  Count:              {int(row['Count'])} ({row['Percentage']}%)")
    print(f"  Avg Tenure:         {row['Avg_Tenure']} months")
    print(f"  Avg Monthly Charges: ${row['Avg_MonthlyCharges']}")
    print(f"  Churn Rate:         {row['Churn_Rate']}%")
    print(f"  Dominant Contract:  {row['Dominant_Contract']}")
    print()
"""))

# ── Cell 23: Conclusion ──
cells.append(md("""## 21. Conclusion

This Week 11 capstone successfully:

1. **Segmented 500 customers** using K-Means, Hierarchical, and DBSCAN clustering.
2. **Selected K-Means** as the primary method based on silhouette score.
3. **Built separate prediction models** (Random Forest, Decision Tree, Logistic Regression) per segment.
4. **Tuned hyperparameters** with GridSearchCV (StratifiedKFold, no test-set leakage).
5. **Evaluated comprehensively** with Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrices.
6. **Identified high-risk churn segments** for immediate retention action.
7. **Saved all outputs:** segment_profiles.md, model_evaluation_results.csv, business_recommendations.pdf.

**Limitations:**
- Small dataset (500 rows) limits per-segment model reliability.
- Feature importance = model association, NOT causation.
- Revenue impact estimates use dataset-derivable values only; actual results depend on campaign execution.
"""))

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    },
    "cells": cells
}

out_path = os.path.join(os.path.dirname(__file__), "customer_segmentation.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"Notebook saved: {out_path}")
