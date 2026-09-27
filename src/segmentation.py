"""
segmentation.py
---------------
Customer Segmentation & Prediction — Week 11 Capstone
Segment profiling, naming, and business characterisation utilities.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


NUMERICAL_FEATURES = ["Tenure", "MonthlyCharges", "TotalCharges"]
TARGET_COLUMN = "Churn"


# ─────────────────────────────────────────────────────────────────────────────
# Segment Profiling
# ─────────────────────────────────────────────────────────────────────────────
def build_segment_profiles(df: pd.DataFrame, label_col: str = "Cluster") -> pd.DataFrame:
    """
    For each cluster, compute:
      - customer_count, percentage
      - mean/median of numerical features
      - churn_rate
      - most common Contract, PaymentMethod, PaperlessBilling

    Returns a summary DataFrame (one row per cluster).
    """
    n_total = len(df)
    rows = []

    for cluster_id in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cluster_id]
        n = len(sub)
        row = {
            "Cluster": cluster_id,
            "Count": n,
            "Percentage": round(100 * n / n_total, 1),
            "Avg_Tenure": round(sub["Tenure"].mean(), 1),
            "Avg_MonthlyCharges": round(sub["MonthlyCharges"].mean(), 1),
            "Avg_TotalCharges": round(sub["TotalCharges"].mean(), 1),
            "Churn_Rate": round(sub[TARGET_COLUMN].mean() * 100, 1)
            if TARGET_COLUMN in sub.columns else None,
            "Dominant_Contract": sub["Contract"].mode()[0]
            if "Contract" in sub.columns else None,
            "Dominant_PaymentMethod": sub["PaymentMethod"].mode()[0]
            if "PaymentMethod" in sub.columns else None,
            "Dominant_PaperlessBilling": sub["PaperlessBilling"].mode()[0]
            if "PaperlessBilling" in sub.columns else None,
            "SeniorCitizen_Rate": round(sub["SeniorCitizen"].mean() * 100, 1)
            if "SeniorCitizen" in sub.columns else None,
        }
        rows.append(row)

    profiles = pd.DataFrame(rows).set_index("Cluster")
    return profiles


# ─────────────────────────────────────────────────────────────────────────────
# Automatic Segment Naming (data-driven)
# ─────────────────────────────────────────────────────────────────────────────
def assign_segment_names(profiles: pd.DataFrame) -> dict:
    """
    Assign meaningful business names to each cluster based on actual data.

    Rules (applied in priority order):
      1. Highest churn rate -> "High-Risk Churners"
      2. Lowest churn rate + longest tenure -> "Loyal Long-Term Customers"
      3. Highest MonthlyCharges -> "Premium High-Spenders"
      4. Lowest MonthlyCharges + short tenure -> "Budget New Customers"
      5. Remaining -> "Mid-Value Engaged Customers"

    Returns dict {cluster_id: name}
    """
    df = profiles.copy().reset_index()
    names = {}
    assigned = set()

    # Sort helpers
    by_churn = df.sort_values("Churn_Rate", ascending=False)
    by_tenure = df.sort_values(["Avg_Tenure", "Churn_Rate"], ascending=[False, True])
    by_spend = df.sort_values("Avg_MonthlyCharges", ascending=False)
    by_low_spend = df.sort_values("Avg_MonthlyCharges", ascending=True)

    def pick_first_unassigned(sorted_df, name):
        for _, row in sorted_df.iterrows():
            cid = row["Cluster"]
            if cid not in assigned:
                names[cid] = name
                assigned.add(cid)
                return

    # Rule 1: highest churn -> High-Risk Churners
    pick_first_unassigned(by_churn, "High-Risk Churners")

    # Rule 2: second highest churn -> At-Risk Mid-Tenure Customers
    pick_first_unassigned(by_churn, "At-Risk Mid-Tenure Customers")

    # Rule 3: longest tenure + low churn -> Loyal Long-Term Customers
    pick_first_unassigned(by_tenure, "Loyal Long-Term Customers")

    # Rule 4: highest spend -> Premium High-Spenders
    pick_first_unassigned(by_spend, "Premium High-Spenders")

    # Rule 5: lowest spend -> Budget Conscious Customers
    pick_first_unassigned(by_low_spend, "Budget Conscious Customers")

    # Rule 6+: any remaining
    extra_count = 1
    for cid in df["Cluster"]:
        if cid not in assigned:
            names[cid] = f"Mid-Value Segment {extra_count}"
            assigned.add(cid)
            extra_count += 1

    return names


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────
def plot_segment_profiles(profiles: pd.DataFrame, segment_names: dict,
                          save_path: str = None):
    """
    Grouped bar chart comparing segments on key numerical features.
    """
    features = ["Avg_Tenure", "Avg_MonthlyCharges"]
    labels = [segment_names.get(c, f"Cluster {c}") for c in profiles.index]

    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Customer Segment Profiles", fontsize=14, fontweight="bold")

    colours = plt.cm.Set2.colors

    for ax_idx, feat in enumerate(features):
        vals = profiles[feat].values
        bars = axes[ax_idx].bar(x, vals, color=[colours[i % len(colours)] for i in range(len(x))],
                                edgecolor="black", linewidth=0.7)
        axes[ax_idx].set_xticks(x)
        axes[ax_idx].set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        axes[ax_idx].set_ylabel(feat.replace("_", " "), fontsize=11)
        axes[ax_idx].set_title(f"{feat.replace('_', ' ')} by Segment", fontsize=12)
        axes[ax_idx].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            axes[ax_idx].text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 0.5,
                              f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[segment_profiles] Figure saved -> {save_path}")
    plt.close()


def plot_churn_by_segment(profiles: pd.DataFrame, segment_names: dict,
                          save_path: str = None):
    """
    Bar chart of churn rate per segment.
    """
    labels = [segment_names.get(c, f"Cluster {c}") for c in profiles.index]
    churn_rates = profiles["Churn_Rate"].values

    colours = ["#e74c3c" if r == max(churn_rates) else "#3498db" for r in churn_rates]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, churn_rates, color=colours, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Churn Rate (%)", fontsize=12)
    ax.set_title("Churn Rate by Customer Segment", fontsize=13, fontweight="bold")
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, churn_rates):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[churn_by_segment] Figure saved -> {save_path}")
    plt.close()


def plot_cluster_distribution(df: pd.DataFrame, label_col: str = "Cluster",
                              segment_names: dict = None,
                              save_path: str = None):
    """
    Pie chart showing how customers are distributed across segments.
    """
    counts = df[label_col].value_counts().sort_index()
    if segment_names:
        labels = [f"{segment_names.get(c, f'Cluster {c}')}\n({counts[c]})" for c in counts.index]
    else:
        labels = [f"Cluster {c}\n({counts[c]})" for c in counts.index]

    colours = plt.cm.Set3.colors[:len(counts)]
    fig, ax = plt.subplots(figsize=(9, 9))
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=labels, autopct="%1.1f%%",
        colors=colours, startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
        textprops=dict(fontsize=10))
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Customer Segment Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[cluster_dist] Figure saved -> {save_path}")
    plt.close()


def save_segment_profiles_md(profiles: pd.DataFrame, segment_names: dict,
                              save_path: str):
    """
    Write segment_profiles.md with methodology, profiles, and business interpretation.
    """
    lines = []
    lines.append("# Customer Segment Profiles\n")
    lines.append("## Methodology\n")
    lines.append(
        "Customers were segmented using **K-Means clustering** (selected as the primary "
        "method after comparing with Agglomerative Clustering and DBSCAN using silhouette scores).\n\n"
        "**Feature space used for clustering:** Tenure, MonthlyCharges, TotalCharges — "
        "all standardised with `StandardScaler` to ensure equal contribution from each feature.\n\n"
        "**Number of clusters** was determined by the Elbow Method (inertia vs. K) and the "
        "Silhouette Score (higher = better-separated clusters).\n"
    )
    lines.append("\n---\n")
    lines.append("## Segment Summary Table\n")

    header = "| Cluster | Segment Name | Count | % | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Churn Rate | Dominant Contract |"
    divider = "|---------|-------------|-------|---|-----------|---------------------|-------------------|------------|-------------------|"
    lines.append(header)
    lines.append(divider)

    for cluster_id, row in profiles.iterrows():
        name = segment_names.get(cluster_id, f"Cluster {cluster_id}")
        lines.append(
            f"| {cluster_id} | {name} | {int(row['Count'])} | {row['Percentage']}% | "
            f"{row['Avg_Tenure']} | \${row['Avg_MonthlyCharges']} | "
            f"\${row['Avg_TotalCharges']} | {row['Churn_Rate']}% | "
            f"{row['Dominant_Contract']} |"
        )

    lines.append("\n---\n")
    lines.append("## Detailed Segment Profiles\n")

    biz_interpretations = {
        "High-Risk Churners": (
            "This segment has the highest churn rate. Customers here are at the greatest "
            "risk of leaving the service. Immediate retention campaigns, loyalty incentives, "
            "or personalised outreach are strongly recommended."
        ),
        "Loyal Long-Term Customers": (
            "Long tenure customers with low churn. They represent the backbone of the business. "
            "Maintaining satisfaction through premium service and reward programmes is key."
        ),
        "Premium High-Spenders": (
            "High monthly charges indicate high value to the business. Even a small churn in "
            "this group has large revenue impact. Personalised account management and exclusive "
            "offers can improve retention."
        ),
        "Budget New Customers": (
            "Lower spending and shorter tenure suggest these are newer or more cost-sensitive "
            "customers. Upselling opportunities and onboarding improvements can move them to "
            "higher-value segments."
        ),
        "Mid-Value Engaged Customers": (
            "Mid-range spending and moderate tenure. These customers present a balanced "
            "opportunity for upselling and churn prevention through targeted engagement."
        ),
    }

    for cluster_id, row in profiles.iterrows():
        name = segment_names.get(cluster_id, f"Cluster {cluster_id}")
        interp = biz_interpretations.get(name, "See business recommendations for this segment.")
        lines.append(f"### Segment {cluster_id}: {name}\n")
        lines.append(f"- **Customer Count:** {int(row['Count'])} ({row['Percentage']}% of total)")
        lines.append(f"- **Average Tenure:** {row['Avg_Tenure']} months")
        lines.append(f"- **Average Monthly Charges:** \${row['Avg_MonthlyCharges']}")
        lines.append(f"- **Average Total Charges:** \${row['Avg_TotalCharges']}")
        lines.append(f"- **Churn Rate:** {row['Churn_Rate']}%")
        lines.append(f"- **Dominant Contract Type:** {row['Dominant_Contract']}")
        lines.append(f"- **Dominant Payment Method:** {row['Dominant_PaymentMethod']}")
        lines.append(f"- **Paperless Billing:** {row['Dominant_PaperlessBilling']}")
        lines.append(f"- **Senior Citizen Rate:** {row['SeniorCitizen_Rate']}%\n")
        lines.append(f"**Business Interpretation:**\n{interp}\n")
        lines.append("---\n")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[segment_profiles] Saved -> {save_path}")
