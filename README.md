# Payment Fraud ML Pipeline

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
[![Python Tests](https://github.com/mitalidaduria/payment-fraud-ml/actions/workflows/python-tests.yml/badge.svg)](https://github.com/mitalidaduria/payment-fraud-ml/actions/workflows/python-tests.yml)

> **67% of payment fraud occurs between midnight and 05:00 — this repository builds the domain-driven ML system to detect it.**

## 📊 Key EDA Insights
* **Off-Hours Spike:** Fraudulent activity peaks between 00:00–05:00 when manual review windows are minimal.
* **Card Testing Patterns:** High volume of micro-transactions (< £10.00) followed immediately by high-value transactions (≥ £4,500.00).
* **Severe Class Imbalance:** Fraud accounts for ~1–2% of total transactions, making Precision-Recall evaluation critical over standard ROC-AUC metrics.
---

##  Feature Engineering Architecture (`PaymentFeatureEngineer`)

Wrapped in an `sklearn`-compatible transformer (`BaseEstimator`, `TransformerMixin`) to learn baseline statistics strictly during `fit()` on training data, preventing training-serving skew and data leakage.

###  Engineered Domain Feature Matrix (20 Features)

| Category | Feature Name | Business Rationale |
| :--- | :--- | :--- |
| **Temporal** | `is_night` | Fraud spikes between 00:00–05:00. |
| | `is_peak_business` | Standard business hours benchmark (09:00–17:00). |
| | `is_weekend` | Non-standard trading window indicator. |
| **Amount & Anomaly** | `amount_log` | Normalizes heavily right-skewed transaction amounts. |
| | `amount_zscore` | Standardized deviation calculated using training statistics. |
| | `is_micro_txn` | Detects card testing (< £10.00). |
| | `is_large_txn` | High-value target transactions (≥ £4,500.00). |
| | `is_round_amount` | Flags round numbers often seen in automated script testing. |
| **Velocity Metrics** | `is_high_velocity` | Exceeds 95th percentile transaction frequency threshold. |
| | `velocity_log` | Normalized hourly transaction count. |
| **Account Risk** | `is_new_account` | High risk window (< 30 days account age). |
| | `account_age_log` | Scaled continuous account maturity. |
| **Interaction Signals** | `night_velocity` | Combines off-hours activity with high transaction speed. |
| | `intl_large` | Cross-border high-value transaction risk flag. |
| | `new_acct_night` | High-risk new account nocturnal activity. |
| | `prev_fail_risk` | Unsettled account with recent failed transactions. |

---

##  Testing & CI Pipeline

Automated with GitHub Actions on Python 3.10:
* **Data Leakage Check:** Verifies Z-score standardization parameters derive strictly from training subsets.
* **Flag Accuracy Check:** Validates exact condition matching on card-testing thresholds.
* **Shape Verification:** Confirms matrix expansion consistency across fits.
