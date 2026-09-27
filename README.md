# Customer Segmentation & Prediction Using Advanced Machine Learning

**Week 11 — Advanced Machine Learning Capstone Project**

---

## 1. Project Overview

This project applies advanced machine learning techniques to segment customers and predict churn for each segment. Using a telecommunications customer dataset (500 customers), we:

- **Segment** customers into meaningful groups using K-Means, Hierarchical (Agglomerative), and DBSCAN clustering
- **Profile** each segment with data-driven business names and characteristics
- **Build** separate Random Forest churn prediction models for each segment
- **Tune** models with GridSearchCV + StratifiedKFold
- **Evaluate** comprehensively: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **Recommend** segment-specific business retention strategies

---

## 2. Dataset

**File:** `customer_churn.csv`

| Property | Value |
|----------|-------|
| Rows | 500 customers |
| Columns | 9 |
| Missing Values | None |
| Duplicate Rows | None |
| Target Variable | `Churn` (0=No, 1=Yes) |
| Overall Churn Rate | ~10.6% |

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| CustomerID | String | Unique customer identifier |
| Tenure | Integer | Months as a customer (1–71) |
| MonthlyCharges | Integer | Monthly bill amount ($20–$199) |
| TotalCharges | Integer | Total amount charged ($159–$7992) |
| Contract | String | Month-to-month / One year / Two year |
| PaymentMethod | String | Credit Card / Electronic Check / Bank Transfer |
| PaperlessBilling | String | Yes / No |
| SeniorCitizen | Integer | 1=Senior, 0=Non-senior |
| Churn | Integer | 1=Churned, 0=Retained |

---

## 3. Setup Instructions

### Prerequisites
- Python 3.8+ installed
- pip package manager

### Installation

```bash
# Step 1: Create a virtual environment
python -m venv venv

# Step 2: Activate (Windows)
venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt
```

### Run the Project

**Option A — Run the complete pipeline (generates all outputs):**
```bash
python run_project.py
```

**Option B — Open the Jupyter notebook:**
```bash
jupyter notebook customer_segmentation.ipynb
```

**Option C — Run the tests:**
```bash
pytest tests/test_pipeline.py -v
```

---

## 4. Project Structure

```
Customer-Segmentation-Prediction/
│
├── customer_segmentation.ipynb     # Main Jupyter notebook (23 sections)
├── customer_churn.csv              # Raw dataset
├── segmentation_data.csv           # Dataset with cluster labels added
├── segment_profiles.md             # Detailed segment characterisation
├── model_evaluation_results.csv    # All model metrics
├── business_recommendations.pdf    # Business report (PDF)
├── run_project.py                  # Master pipeline script
├── create_notebook.py              # Notebook generator script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── src/
│   ├── preprocessing.py            # Data loading, cleaning, encoding, scaling
│   ├── clustering.py               # K-Means, Agglomerative, DBSCAN, plots
│   ├── segmentation.py             # Segment profiling, naming, visualisation
│   ├── modeling.py                 # RF, DT, LR pipelines, GridSearchCV
│   └── evaluation.py               # Metrics, confusion matrix, ROC, charts
│
├── tests/
│   └── test_pipeline.py            # 30+ unit & integration tests
│
├── outputs/
│   ├── figures/                    # All PNG visualisations (15+ figures)
│   └── metrics/                    # Tuning & overfitting CSVs
│
└── screenshots/                    # Screenshots for documentation
```

---

## 5. Methodology

### Data Preprocessing
1. Load CSV → inspect shape, dtypes, missing values, duplicates
2. Clean: drop duplicates, strip whitespace, enforce numeric types
3. Impute: median (numerical), mode (categorical) — none needed here
4. Encode categoricals:
   - `Contract` → ordinal (0/1/2)
   - `PaperlessBilling` → binary (0/1)
   - `PaymentMethod` → label-encoded
5. Scale for clustering: `StandardScaler` on Tenure, MonthlyCharges, TotalCharges

> **Why scaling?** K-Means uses Euclidean distance. Without scaling, TotalCharges (0–8000) dominates Tenure (1–71). StandardScaler brings all features to mean=0, std=1.

### Clustering
- **K-Means:** Tested K=2 to K=9; selected K by silhouette score
- **Agglomerative:** Ward linkage, same K as K-Means; dendrogram generated
- **DBSCAN:** eps estimated via k-NN distance plot (knee heuristic)
- **Selection:** K-Means chosen (highest silhouette score)

### Prediction Models (per segment)
- **Random Forest** — primary model, bagging + feature randomisation
- **Decision Tree** — comparison model (max_depth=5)
- **Logistic Regression** — linear baseline

### Hyperparameter Tuning
- `GridSearchCV` with `StratifiedKFold(n_splits=5)`
- Tuned: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Scoring: **F1** (appropriate for imbalanced churn classification)
- Test set **never** used during tuning (no data leakage)

---

## 6. Results

### Clustering Summary

| Algorithm | Clusters | Silhouette Score |
|-----------|----------|-----------------|
| K-Means | 6 | 0.3080 |
| Agglomerative | 6 | 0.2292 |
| DBSCAN | 1* | N/A |

*DBSCAN found 1 cluster with the data's density characteristics (all points dense together)

### Customer Segments (6 segments from K-Means)

| Cluster | Segment Name | Count | % | Churn Rate |
|---------|-------------|-------|---|-----------|
| 0 | Loyal Long-Term Customers | 91 | 18.2% | 0.0% |
| 1 | Premium High-Spenders | 93 | 18.6% | 0.0% |
| 2 | High-Risk Churners | 88 | 17.6% | 26.1% |
| 3 | Budget New Customers | 81 | 16.2% | 0.0% |
| 4 | At-Risk Mid-Tenure Customers | 69 | 13.8% | 15.9% |
| 5 | Mid-Value Segment 1 | 78 | 15.6% | 24.4% |

> **Note:** Segment names are generated automatically from actual cluster characteristics (churn rate, tenure, charges). See `segment_profiles.md` for full details.

---

## 7. Customer Segments

### Cluster 2 — High-Risk Churners (26.1% churn)
- Short tenure (~15 months), high monthly charges (~$127)
- Month-to-month contracts, Electronic Check payment
- **Action:** Immediate retention outreach, contract upgrade incentives

### Cluster 5 — Mid-Value Segment 1 (24.4% churn)
- Moderate tenure (~23 months), highest charges (~$170)
- One-year contracts, Credit Card payment
- **Action:** Personalised offers, lock in with longer contracts

### Cluster 4 — At-Risk Mid-Tenure (15.9% churn)
- Short tenure (~17 months), low charges (~$64), high TotalCharges
- Month-to-month contracts
- **Action:** Monitor closely, proactive engagement

### Cluster 0 — Loyal Long-Term Customers (0% churn)
- Longest tenure (~55 months), moderate charges (~$91)
- **Action:** Reward loyalty, use as brand advocates

### Cluster 1 — Premium High-Spenders (0% churn)
- Long tenure (~53 months), highest charges (~$160)
- **Action:** Premium service, exclusive perks

### Cluster 3 — Budget New Customers (0% churn)
- Moderate tenure (~50 months), lowest charges (~$61)
- **Action:** Upsell opportunities

---

## 8. Model Evaluation

> See `model_evaluation_results.csv` for complete results.

All metrics are calculated on **held-out test data** that was never used during training or tuning.

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Of predicted churners, how many truly churned |
| Recall | Of actual churners, how many were identified |
| F1 | Harmonic mean of Precision and Recall |
| ROC-AUC | Area under the ROC curve (probability ranking quality) |

> **Note on imbalanced data:** With only ~10.6% churn overall, accuracy is misleading. F1 and ROC-AUC are the primary evaluation metrics.

---

## 9. Feature Importance

Feature importance plots are saved in `outputs/figures/fi_*.png`.

Key drivers identified by Random Forest (typical across segments):
- **Tenure** — longer-tenured customers are more loyal
- **MonthlyCharges** — higher charges correlate with churn risk
- **TotalCharges** — reflects cumulative customer value
- **Contract_encoded** — month-to-month customers churn more

> **Important:** Feature importance reflects model association, **not causation**. A feature being important does not mean it *causes* churn.

---

## 10. Business Recommendations

Full recommendations in `business_recommendations.pdf` and `segment_profiles.md`.

| Segment | Priority | Key Strategy |
|---------|----------|-------------|
| High-Risk Churners | 🔴 HIGH | Immediate outreach, contract upgrade incentives |
| Mid-Value Segment 1 | 🔴 HIGH | Personalised retention offers |
| At-Risk Mid-Tenure | 🟡 MEDIUM | Proactive engagement, monitor monthly |
| Loyal Long-Term | 🟢 LOW | Reward loyalty, referral programme |
| Premium High-Spenders | 🟢 LOW | Premium service, exclusive perks |
| Budget New Customers | 🟡 MEDIUM | Onboarding improvement, upsell path |

---

## 11. Testing

Run the test suite:
```bash
pytest tests/test_pipeline.py -v
```

The test suite covers:
- **Data Loading:** File existence, shape, required columns, binary target
- **Preprocessing:** No duplicates, no missing values after imputation, encoded columns
- **Scaling:** Zero mean, unit variance
- **Clustering:** Elbow inertia decreasing, correct cluster counts, DBSCAN noise handling
- **Segmentation:** Profile counts sum to total, names unique, churn rates non-negative
- **Modeling:** Feasibility checks (small segments, single class), pipeline builds, binary predictions
- **Evaluation:** All metric keys present, valid ranges, 2×2 confusion matrix
- **File Outputs:** CSV, MD, PDF files exist and have data

---

## 12. Limitations

1. **Small dataset (500 rows):** Segment-level models have limited training data. Segments with very few churners show unreliable recall.
2. **Missing features:** No satisfaction scores, service tickets, or competitor data — known churn drivers.
3. **Clustering subjectivity:** The number of clusters is chosen by silhouette score heuristic, not guaranteed business validity.
4. **Revenue estimates:** Impact calculations use dataset-derivable values only. Actual results depend on campaign effectiveness.
5. **Feature importance ≠ causation:** The model identifies associations; domain expertise is needed to establish causal mechanisms.
6. **DBSCAN limitation:** With this dataset's density characteristics, DBSCAN collapsed to 1 cluster — showing K-Means is more appropriate here.

---

## 13. Conclusion

This Week 11 capstone successfully demonstrates:
- **Unsupervised learning** for customer segmentation (3 algorithms compared)
- **Supervised learning** with separate per-segment classifiers
- **Hyperparameter optimisation** without data leakage
- **Comprehensive evaluation** with 5 metrics + visualisations
- **Business translation** of ML results into actionable recommendations

The analysis identifies two high-risk segments (Clusters 2 and 5) requiring immediate retention attention, while three loyal segments with 0% churn can be leveraged for growth through referrals and upselling.

---

*Generated by the Customer Segmentation & Prediction ML Pipeline — Week 11 Capstone*
