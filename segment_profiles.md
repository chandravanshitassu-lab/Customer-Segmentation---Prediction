# Customer Segment Profiles

## Methodology

Customers were segmented using **K-Means clustering** (selected as the primary method after comparing with Agglomerative Clustering and DBSCAN using silhouette scores).

**Feature space used for clustering:** Tenure, MonthlyCharges, TotalCharges — all standardised with `StandardScaler` to ensure equal contribution from each feature.

**Number of clusters** was determined by the Elbow Method (inertia vs. K) and the Silhouette Score (higher = better-separated clusters).


---

## Segment Summary Table

| Cluster | Segment Name | Count | % | Avg Tenure | Avg Monthly Charges | Avg Total Charges | Churn Rate | Dominant Contract |
|---------|-------------|-------|---|-----------|---------------------|-------------------|------------|-------------------|
| 0 | Loyal Long-Term Customers | 91 | 18.2% | 54.7 | \$91.3 | \$6607.7 | 0.0% | Month-to-month |
| 1 | Premium High-Spenders | 93 | 18.6% | 53.1 | \$159.5 | \$2756.0 | 0.0% | One year |
| 2 | High-Risk Churners | 88 | 17.6% | 15.0 | \$126.8 | \$2178.0 | 26.1% | Month-to-month |
| 3 | Budget Conscious Customers | 81 | 16.2% | 49.6 | \$60.8 | \$2437.2 | 0.0% | One year |
| 4 | Mid-Value Segment 1 | 69 | 13.8% | 17.3 | \$63.6 | \$5827.5 | 15.9% | Month-to-month |
| 5 | At-Risk Mid-Tenure Customers | 78 | 15.6% | 23.3 | \$169.5 | \$6027.7 | 24.4% | One year |

---

## Detailed Segment Profiles

### Segment 0: Loyal Long-Term Customers

- **Customer Count:** 91 (18.2% of total)
- **Average Tenure:** 54.7 months
- **Average Monthly Charges:** \$91.3
- **Average Total Charges:** \$6607.7
- **Churn Rate:** 0.0%
- **Dominant Contract Type:** Month-to-month
- **Dominant Payment Method:** Bank Transfer
- **Paperless Billing:** Yes
- **Senior Citizen Rate:** 50.5%

**Business Interpretation:**
Long tenure customers with low churn. They represent the backbone of the business. Maintaining satisfaction through premium service and reward programmes is key.

---

### Segment 1: Premium High-Spenders

- **Customer Count:** 93 (18.6% of total)
- **Average Tenure:** 53.1 months
- **Average Monthly Charges:** \$159.5
- **Average Total Charges:** \$2756.0
- **Churn Rate:** 0.0%
- **Dominant Contract Type:** One year
- **Dominant Payment Method:** Bank Transfer
- **Paperless Billing:** No
- **Senior Citizen Rate:** 38.7%

**Business Interpretation:**
High monthly charges indicate high value to the business. Even a small churn in this group has large revenue impact. Personalised account management and exclusive offers can improve retention.

---

### Segment 2: High-Risk Churners

- **Customer Count:** 88 (17.6% of total)
- **Average Tenure:** 15.0 months
- **Average Monthly Charges:** \$126.8
- **Average Total Charges:** \$2178.0
- **Churn Rate:** 26.1%
- **Dominant Contract Type:** Month-to-month
- **Dominant Payment Method:** Electronic Check
- **Paperless Billing:** No
- **Senior Citizen Rate:** 52.3%

**Business Interpretation:**
This segment has the highest churn rate. Customers here are at the greatest risk of leaving the service. Immediate retention campaigns, loyalty incentives, or personalised outreach are strongly recommended.

---

### Segment 3: Budget Conscious Customers

- **Customer Count:** 81 (16.2% of total)
- **Average Tenure:** 49.6 months
- **Average Monthly Charges:** \$60.8
- **Average Total Charges:** \$2437.2
- **Churn Rate:** 0.0%
- **Dominant Contract Type:** One year
- **Dominant Payment Method:** Credit Card
- **Paperless Billing:** Yes
- **Senior Citizen Rate:** 55.6%

**Business Interpretation:**
See business recommendations for this segment.

---

### Segment 4: Mid-Value Segment 1

- **Customer Count:** 69 (13.8% of total)
- **Average Tenure:** 17.3 months
- **Average Monthly Charges:** \$63.6
- **Average Total Charges:** \$5827.5
- **Churn Rate:** 15.9%
- **Dominant Contract Type:** Month-to-month
- **Dominant Payment Method:** Credit Card
- **Paperless Billing:** No
- **Senior Citizen Rate:** 60.9%

**Business Interpretation:**
See business recommendations for this segment.

---

### Segment 5: At-Risk Mid-Tenure Customers

- **Customer Count:** 78 (15.6% of total)
- **Average Tenure:** 23.3 months
- **Average Monthly Charges:** \$169.5
- **Average Total Charges:** \$6027.7
- **Churn Rate:** 24.4%
- **Dominant Contract Type:** One year
- **Dominant Payment Method:** Credit Card
- **Paperless Billing:** Yes
- **Senior Citizen Rate:** 43.6%

**Business Interpretation:**
See business recommendations for this segment.

---
