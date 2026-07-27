#  Payment Fraud ML Pipeline

[![Python Tests](https://github.com/mitalidaduria/payment-fraud-ml/workflows/tests.yml/badge.svg)](https://github.com/mitalidaduria/payment-fraud-ml/actions/workflows/tests.yml)

An end-to-end FinTech machine learning framework featuring a domain-driven feature engineering pipeline and automated cloud CI/CD verification.

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
