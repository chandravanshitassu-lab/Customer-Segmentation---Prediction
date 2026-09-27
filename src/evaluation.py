"""
evaluation.py
-------------
Customer Segmentation & Prediction — Week 11 Capstone
Comprehensive model evaluation: metrics, confusion matrices, ROC curves,
and model comparison charts.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# Core Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_model(pipeline, X_test, y_test,
                   segment_name: str, model_name: str) -> dict:
    """
    Compute all evaluation metrics for a trained pipeline.

    Returns a dict with:
      accuracy, precision, recall, f1, roc_auc (if defined),
      confusion_matrix, classification_report
    """
    y_pred = pipeline.predict(X_test)

    # ROC-AUC requires predict_proba or decision_function
    try:
        if hasattr(pipeline, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            y_prob = pipeline.decision_function(X_test)

        if y_test.nunique() < 2:
            roc_auc = None
            roc_note = "Undefined — only one class in test set"
        else:
            roc_auc = roc_auc_score(y_test, y_prob)
            roc_note = f"{roc_auc:.4f}"
    except Exception as e:
        y_prob = None
        roc_auc = None
        roc_note = f"Error: {e}"

    # Use zero_division=0 to handle undefined precision/recall gracefully
    metrics = {
        "segment": segment_name,
        "model": model_name,
        "n_test": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "roc_note": roc_note,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "y_prob": y_prob,
        "y_pred": y_pred,
        "y_test": y_test,
    }

    print(f"  [{segment_name} | {model_name}]  "
          f"Acc={metrics['accuracy']:.4f}  "
          f"P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}  "
          f"AUC={roc_note}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(cm_list: list, segment_name: str, model_name: str,
                          save_path: str = None):
    """
    Plot a confusion matrix heatmap.
    cm_list: 2-D list [[TN, FP],[FN, TP]]
    """
    cm = np.array(cm_list)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pred No Churn", "Pred Churn"],
                yticklabels=["Actual No Churn", "Actual Churn"],
                linewidths=0.5, ax=ax)
    ax.set_title(f"Confusion Matrix\n{segment_name} — {model_name}",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xlabel("Predicted", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [cm] Figure saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# ROC Curve Plot
# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curve(eval_results: list, save_path: str = None):
    """
    Plot ROC curves for multiple models on the same axes.
    eval_results: list of dicts from evaluate_model() where roc_auc is not None.
    """
    valid = [r for r in eval_results if r.get("roc_auc") is not None
             and r.get("y_prob") is not None]

    if not valid:
        print("  [roc] No valid ROC data available — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    colours = plt.cm.tab10.colors

    for i, r in enumerate(valid):
        fpr, tpr, _ = roc_curve(r["y_test"], r["y_prob"])
        roc_auc = auc(fpr, tpr)
        label = f"{r['segment']} — {r['model']} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=colours[i % len(colours)], linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Segments and Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [roc] Figure saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Model Comparison Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_model_comparison(all_results: list, save_path: str = None):
    """
    Grouped bar chart comparing Accuracy, Precision, Recall, F1, ROC-AUC
    across all segment–model combinations.
    """
    records = []
    for r in all_results:
        records.append({
            "Label": f"{r['segment']}\n{r['model']}",
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "ROC-AUC": r["roc_auc"] if r["roc_auc"] is not None else 0,
        })
    df = pd.DataFrame(records)

    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    x = np.arange(len(df))
    width = 0.15
    colours = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, ax = plt.subplots(figsize=(max(12, len(df) * 2.5), 7))
    for i, (metric, colour) in enumerate(zip(metrics, colours)):
        offset = (i - len(metrics) / 2) * width + width / 2
        bars = ax.bar(x + offset, df[metric], width,
                      label=metric, color=colour, edgecolor="black",
                      linewidth=0.5, alpha=0.85)

    ax.set_xlabel("Segment — Model", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Performance Comparison Across All Segments", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [model_comparison] Figure saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Save Evaluation Results CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_evaluation_csv(all_results: list, save_path: str):
    """
    Save a tidy CSV of all model evaluation results (excluding non-serialisable fields).
    """
    rows = []
    for r in all_results:
        rows.append({
            "Segment": r["segment"],
            "Model": r["model"],
            "N_Test": r["n_test"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1": r["f1"],
            "ROC_AUC": r["roc_auc"] if r["roc_auc"] is not None else "N/A",
            "ROC_Note": r.get("roc_note", ""),
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"[evaluation] Results saved -> {save_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Distribution / Churn Overview
# ─────────────────────────────────────────────────────────────────────────────
def plot_dataset_overview(df: pd.DataFrame, save_path: str = None):
    """
    4-panel overview: churn distribution, tenure, monthly charges, total charges.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dataset Overview", fontsize=15, fontweight="bold")

    # Churn distribution
    churn_counts = df["Churn"].value_counts()
    colours = ["#2196F3", "#F44336"]
    axes[0, 0].bar(["No Churn (0)", "Churn (1)"], churn_counts.values,
                   color=colours, edgecolor="black", linewidth=0.7)
    for bar, val in zip(axes[0, 0].patches, churn_counts.values):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 2,
                        f"{val}\n({100*val/len(df):.1f}%)",
                        ha="center", fontsize=10)
    axes[0, 0].set_title("Churn Distribution", fontsize=12)
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(axis="y", alpha=0.3)

    # Tenure distribution
    axes[0, 1].hist(df["Tenure"], bins=20, color="#4CAF50", edgecolor="black", linewidth=0.5)
    axes[0, 1].set_title("Tenure Distribution (months)", fontsize=12)
    axes[0, 1].set_xlabel("Tenure (months)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(alpha=0.3)

    # Monthly Charges by Churn
    axes[1, 0].boxplot(
        [df[df["Churn"] == 0]["MonthlyCharges"].values,
         df[df["Churn"] == 1]["MonthlyCharges"].values],
        tick_labels=["No Churn", "Churn"],
        patch_artist=True,
        boxprops=dict(facecolor="#90CAF9"),
        medianprops=dict(color="red", linewidth=2)
    )
    axes[1, 0].set_title("Monthly Charges by Churn Status", fontsize=12)
    axes[1, 0].set_ylabel("Monthly Charges ($)")
    axes[1, 0].grid(alpha=0.3)

    # Total Charges by Churn
    axes[1, 1].boxplot(
        [df[df["Churn"] == 0]["TotalCharges"].values,
         df[df["Churn"] == 1]["TotalCharges"].values],
        tick_labels=["No Churn", "Churn"],
        patch_artist=True,
        boxprops=dict(facecolor="#FFCC80"),
        medianprops=dict(color="red", linewidth=2)
    )
    axes[1, 1].set_title("Total Charges by Churn Status", fontsize=12)
    axes[1, 1].set_ylabel("Total Charges ($)")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[overview] Figure saved -> {save_path}")
    plt.close()
