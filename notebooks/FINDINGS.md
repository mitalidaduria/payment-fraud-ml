# Payment Fraud EDA: Key Findings

## Finding 1: Temporal Signal (STRONGEST FEATURE)
- 67% of fraud occurs midnight–4am
- Implication: `hour_of_day` is top predictive feature; engineer `is_night` flag

## Finding 2: Amount Bimodality in Fraud
- Two fraud clusters: <£10 (card testing) and >£4,500 (large theft)
- Legitimate transactions follow exponential distribution (no bimodality)
- Implication: engineer `is_micro_txn` and `is_large_txn` binary features

## Finding 3: Velocity Spike
- Fraudsters: 12 txns/hour average vs 2 for legitimate
- Implication: `txn_per_hour` and `high_velocity` flag are critical features

## Finding 4: New Account Risk
- Fraud accounts: 30 days average age vs 365 for legitimate
- Implication: `user_age_days` needs log-transform; `is_new_account` flag

## Finding 5: International Transaction Pattern
- 70% of fraud involves international transactions vs 15% for legitimate
- Implication: `is_intl` is a strong signal; interaction with `is_night` even stronger
