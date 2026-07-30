import os

# Create directory structure
os.makedirs("src/evaluation", exist_ok=True)
with open("src/evaluation/__init__.py", "w") as f:
    pass

# Write evaluator.py code
code = '''# src/evaluation/evaluator.py

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score
import shap


class FraudModelEvaluator:
    """
    Evaluates fraud detection models using financial business metrics
    and precision-recall trade-offs.
    """
    def __init__(self, avg_fraud_gbp=500, false_alarm_cost_gbp=15):
        # Average fraud amount saved per True Positive
        self.fraud_val = avg_fraud_gbp 
        # Cost of blocking/friction on a legitimate transaction (False Positive)
        self.fa_cost = false_alarm_cost_gbp 

    def threshold_analysis(self, y_test, y_scores, n=12):
        """
        Prints a business impact grid evaluating financial outcomes across 'n' decision thresholds.
        """
        prec, rec, thresh = precision_recall_curve(y_test, y_scores)
        n_legit = (y_test == 0).sum()

        pr_auc = average_precision_score(y_test, y_scores)
        print(f"PR-AUC: {pr_auc:.4f}\\n")

        print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Fraud%':>8} {'FAlarm%':>9} {'Net £':>10}")
        print("-" * 68)

        step = max(1, len(thresh) // n)

        for i in range(0, len(thresh), step):
            t = thresh[i]
            p = prec[i]
            r = rec[i]
            f1 = 2 * p * r / (p + r + 1e-9)

            y_pred = (y_scores >= t).astype(int)
            fp = ((y_pred == 1) & (y_test == 0)).sum()
            tp = ((y_pred == 1) & (y_test == 1)).sum()

            fa_pct = (fp / n_legit) * 100
            net = (tp * self.fraud_val) - (fp * self.fa_cost)

            print(f"{t:>10.2f} {p:>10.3f} {r:>8.3f} {f1:>6.3f} {r * 100:>7.1f}% {fa_pct:>8.2f}% {net:>10,.0f}")

    def regulatory_explanation(self, shap_values, feature_names, threshold=0.5, score=None):
        """
        Converts top SHAP drivers into human-readable text for GDPR Article 22 compliance.
        """
        if score is not None and score < threshold:
            return "Transaction approved."

        factor_map = {
            'is_night': 'unusual transaction timing',
            'is_high_velocity': 'high recent transaction frequency',
            'is_intl': 'international transaction pattern',
            'is_new_account': 'account age',
            'is_micro_txn': 'transaction amount pattern',
            'is_large_txn': 'unusually large amount'
        }

        pairs = sorted(zip(feature_names, shap_values), key=lambda x: -x[1])
        reasons = [factor_map.get(f, f.replace('_', ' ')) for f, v in pairs[:2] if v > 0]

        if not reasons:
            reasons = ["routine risk verification protocol"]

        return f"This transaction was flagged for security review based on: {' and '.join(reasons)}. Contact support to verify your identity."


class FraudExplainer:
    """
    SHAP-based model explainer for tree-based models (e.g., XGBoost).
    """
    def __init__(self, model, feature_names):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def explain(self, x_single):
        """
        Prints top features pushing risk scores UP and DOWN for a single transaction.
        """
        sv = self.explainer.shap_values(x_single.reshape(1, -1))[0]
        pairs = list(zip(self.feature_names, sv))

        fraud_drivers = [(f, v) for f, v in sorted(pairs, key=lambda x: -x[1]) if v > 0][:3]
        legit_drivers = [(f, v) for f, v in sorted(pairs, key=lambda x: x[1]) if v < 0][:3]

        print("Factors INCREASING fraud likelihood:")
        for f, v in fraud_drivers:
            print(f"  + {f:25s}: {v:+.4f}")

        print("\\nFactors REDUCING fraud likelihood:")
        for f, v in legit_drivers:
            print(f"  - {f:25s}: {v:+.4f}")
'''

with open("src/evaluation/evaluator.py", "w") as f:
    f.write(code)

print("Files successfully created in Colab workspace!")
