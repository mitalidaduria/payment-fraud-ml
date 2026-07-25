# payment-fraud-ml
End-to-end payment fraud detection pipeline for highly imbalanced FinTech data (1.72% fraud rate) using XGBoost, SHAP, and FastAPI.
# payment-fraud-ml

> **Production-grade machine learning pipeline for payment fraud detection — EDA to serving.**

A real-world FinTech ML pipeline handling extreme class imbalance (1.72% fraud rate), temporal velocity signals, and business-focused PR-AUC evaluation.

---

##  Key Domain Insight: Why Accuracy is Useless

In card-present transactions, the baseline fraud rate is approximately **1.72%**. 

A naive model predicting `is_fraud = 0` (no fraud) for every transaction achieves **98.28% accuracy**, yet catches **0% of fraud** and incurs massive financial losses. 

Therefore, this repository optimizes for **Precision-Recall Area Under Curve (PR-AUC)** and **Cost-Sensitive Thresholding** rather than accuracy.

---

##  Repository Architecture

```text
payment-fraud-ml/
├── src/
│   ├── data/          # Synthetic transaction generator & data ingestion
│   ├── features/      # Feature engineering (velocity metrics, ratios)
│   ├── models/        # Model training pipelines (XGBoost, LightGBM)
│   ├── evaluation/    # PR-AUC, confusion matrix, cost-sensitive analysis
│   └── serving/       # FastAPI REST endpoint for live inference
├── tests/             # Unit tests with pytest
├── notebooks/         # Exploratory Data Analysis (EDA) & SHAP explainability
├── scripts/           # Execution & training automation scripts
├── reports/           # Saved evaluation metrics & figures
├── requirements.txt   # Core Python dependencies
└── README.md
