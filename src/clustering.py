"""
clustering.py
-------------
Customer Segmentation & Prediction — Week 11 Capstone
K-Means, Hierarchical (Agglomerative), and DBSCAN clustering utilities.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA

RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# K-Means — Elbow Method
# ─────────────────────────────────────────────────────────────────────────────
def run_elbow_method(scaled_data: np.ndarray, k_range: range = range(2, 10),
                     save_path: str = None) -> dict:
    """
    Run K-Means for each k in k_range, record inertia and silhouette score.
    Plots the Elbow curve and saves it.

    Returns dict: {k: {"inertia": float, "silhouette": float}}
    """
    results = {}
    inertias = []
    silhouettes = []
    ks = list(k_range)

    for k in ks:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(scaled_data)
        inertia = km.inertia_
        sil = silhouette_score(scaled_data, labels)
        results[k] = {"inertia": inertia, "silhouette": sil}
        inertias.append(inertia)
        silhouettes.append(sil)
        print(f"  K={k:2d}  Inertia={inertia:10.2f}  Silhouette={sil:.4f}")

    # ── Elbow + Silhouette plot ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("K-Means: Elbow Method & Silhouette Scores", fontsize=14, fontweight="bold")

    # Elbow
    axes[0].plot(ks, inertias, "bo-", linewidth=2, markersize=8)
    axes[0].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=12)
    axes[0].set_title("Elbow Method", fontsize=13)
    axes[0].grid(alpha=0.3)
    axes[0].set_xticks(ks)

    # Silhouette
    best_k = ks[int(np.argmax(silhouettes))]
    colors = ["red" if k == best_k else "steelblue" for k in ks]
    bars = axes[1].bar(ks, silhouettes, color=colors, edgecolor="black", linewidth=0.7)
    axes[1].set_xlabel("Number of Clusters (K)", fontsize=12)
    axes[1].set_ylabel("Silhouette Score", fontsize=12)
    axes[1].set_title("Silhouette Score by K", fontsize=13)
    axes[1].set_xticks(ks)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].annotate(f"Best K={best_k}\n({max(silhouettes):.4f})",
                     xy=(best_k, max(silhouettes)),
                     xytext=(best_k + 0.3, max(silhouettes) - 0.01),
                     fontsize=10, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[elbow] Figure saved -> {save_path}")
    plt.close()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# K-Means — Final Model
# ─────────────────────────────────────────────────────────────────────────────
def train_kmeans(scaled_data: np.ndarray, n_clusters: int) -> tuple:
    """
    Train final K-Means model with n_clusters.
    Returns (fitted_model, labels_array).
    """
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=20,
                random_state=RANDOM_STATE, max_iter=500)
    labels = km.fit_predict(scaled_data)
    sil = silhouette_score(scaled_data, labels)
    print(f"[kmeans] K={n_clusters}  Inertia={km.inertia_:.2f}  Silhouette={sil:.4f}")
    return km, labels


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical / Agglomerative Clustering
# ─────────────────────────────────────────────────────────────────────────────
def train_agglomerative(scaled_data: np.ndarray, n_clusters: int,
                        linkage_method: str = "ward") -> tuple:
    """
    Train Agglomerative Clustering.
    Returns (fitted_model, labels_array, silhouette_score).
    """
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = agg.fit_predict(scaled_data)
    sil = silhouette_score(scaled_data, labels)
    print(f"[agglomerative] n_clusters={n_clusters}  Linkage={linkage_method}  "
          f"Silhouette={sil:.4f}")
    return agg, labels, sil


def plot_dendrogram(scaled_data: np.ndarray, save_path: str = None,
                    max_samples: int = 100):
    """
    Plot a truncated dendrogram using Ward linkage.
    Uses at most max_samples rows for readability.
    """
    # Subsample for readability if dataset is large
    if len(scaled_data) > max_samples:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(scaled_data), max_samples, replace=False)
        data_sub = scaled_data[idx]
    else:
        data_sub = scaled_data

    Z = linkage(data_sub, method="ward")

    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               leaf_rotation=45, leaf_font_size=9,
               show_contracted=True,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title("Hierarchical Clustering — Dendrogram (Truncated, Ward Linkage)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Customer Index (or cluster size)", fontsize=11)
    ax.set_ylabel("Euclidean Distance", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[dendrogram] Figure saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN
# ─────────────────────────────────────────────────────────────────────────────
def estimate_dbscan_eps(scaled_data: np.ndarray, min_samples: int = 5,
                        save_path: str = None) -> float:
    """
    Use the k-nearest-neighbour distance plot to estimate a good eps value.
    The elbow/knee of the sorted k-NN distances is a common heuristic.
    Returns the suggested eps value.
    """
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(scaled_data)
    distances, _ = nbrs.kneighbors(scaled_data)
    kth_distances = np.sort(distances[:, -1])  # distance to k-th neighbour

    # Simple knee detection: look for maximum second derivative
    d2 = np.diff(np.diff(kth_distances))
    knee_idx = np.argmax(d2) + 2  # offset for two diff operations
    suggested_eps = float(kth_distances[knee_idx])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(kth_distances, linewidth=1.5, color="steelblue")
    ax.axvline(x=knee_idx, color="red", linestyle="--", label=f"Knee ≈ idx {knee_idx}")
    ax.axhline(y=suggested_eps, color="orange", linestyle="--",
               label=f"Suggested eps ≈ {suggested_eps:.3f}")
    ax.set_title(f"DBSCAN — k-NN Distance Plot (k={min_samples})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Points sorted by distance", fontsize=11)
    ax.set_ylabel(f"Distance to {min_samples}-th Nearest Neighbour", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[dbscan eps] Figure saved -> {save_path}")
    plt.close()

    print(f"[dbscan] Suggested eps ≈ {suggested_eps:.4f}  (min_samples={min_samples})")
    return suggested_eps


def train_dbscan(scaled_data: np.ndarray, eps: float,
                 min_samples: int = 5) -> tuple:
    """
    Train DBSCAN model.
    Returns (fitted_model, labels_array).
    Noise points are labelled -1.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(scaled_data)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = 100 * n_noise / len(labels)
    print(f"[dbscan] eps={eps:.4f}  min_samples={min_samples}  "
          f"Clusters={n_clusters}  Noise={n_noise} ({noise_pct:.1f}%)")

    if n_clusters > 1:
        mask = labels != -1
        sil = silhouette_score(scaled_data[mask], labels[mask]) if mask.sum() > 1 else None
        print(f"[dbscan] Silhouette (excl. noise) = {sil:.4f}" if sil else
              "[dbscan] Not enough points for silhouette.")
    else:
        sil = None
        print("[dbscan] Only 1 cluster found — silhouette undefined.")

    return db, labels, n_clusters, n_noise, sil


# ─────────────────────────────────────────────────────────────────────────────
# Clustering Comparison
# ─────────────────────────────────────────────────────────────────────────────
def compare_algorithms(results: dict, save_path: str = None):
    """
    Bar chart comparing silhouette scores across clustering algorithms.
    results = {"K-Means": sil, "Agglomerative": sil, "DBSCAN": sil or None}
    """
    valid = {k: v for k, v in results.items() if v is not None}
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(valid.keys(), valid.values(), color=["steelblue", "coral", "mediumseagreen"],
                  edgecolor="black", linewidth=0.7, width=0.5)
    for bar, val in zip(bars, valid.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Silhouette Score (higher is better)", fontsize=12)
    ax.set_title("Clustering Algorithm Comparison — Silhouette Score", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(valid.values()) * 1.25)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[compare] Figure saved -> {save_path}")
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Cluster Visualisation (PCA 2-D)
# ─────────────────────────────────────────────────────────────────────────────
def plot_clusters_pca(scaled_data: np.ndarray, labels: np.ndarray,
                      title: str = "Customer Segments (PCA 2-D Projection)",
                      save_path: str = None):
    """
    Project scaled data to 2-D with PCA and plot colour-coded cluster membership.
    Noise points (label == -1) are shown in grey.
    """
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(scaled_data)
    var_explained = pca.explained_variance_ratio_

    unique_labels = sorted(set(labels))
    palette = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "grey" if lbl == -1 else palette[i % len(palette)]
        name = "Noise" if lbl == -1 else f"Segment {lbl}"
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], label=name, alpha=0.7, s=60, edgecolors="white", linewidth=0.3)

    ax.set_xlabel(f"PC1 ({var_explained[0]*100:.1f}% variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({var_explained[1]*100:.1f}% variance)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, markerscale=1.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[pca_plot] Figure saved -> {save_path}")
    plt.close()
