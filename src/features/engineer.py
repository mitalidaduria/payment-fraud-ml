import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PaymentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer for payment fraud detection.
    Computes temporal flags, amount anomaly signals, and velocity/risk metrics.
    """

    def __init__(self, velocity_windows_hours=[1, 24]):
        self.velocity_windows_hours = velocity_windows_hours

    def fit(self, X, y=None):
        """
        Calculates baseline statistics on training data to avoid data leakage.
        """
        df = X.copy()
        if "amount" in df.columns:
            self.amount_mean_ = df["amount"].mean()
            self.amount_std_ = df["amount"].std()
        if "txn_per_hour" in df.columns:
            self.velocity_p95_ = df["txn_per_hour"].quantile(0.95)
        if "user_id" in df.columns and "amount" in df.columns:
            self.user_stats_ = (
                df.groupby("user_id")["amount"]
                .agg(user_avg_amount="mean", user_std_amount="std")
                .reset_index()
            )
        else:
            self.user_stats_ = pd.DataFrame()
        return self

    def transform(self, X):
        """
        Transforms raw transaction logs into expressive feature vectors.
        """
        df = X.copy()

        # 1. Temporal Signals
        if "hour_of_day" in df.columns:
            df["is_night"] = df["hour_of_day"].between(0, 5).astype(int)
            df["is_peak_business"] = df["hour_of_day"].between(9, 17).astype(int)
        elif "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["is_night"] = df["hour"].apply(
                lambda h: 1 if (h >= 23 or h <= 5) else 0
            )

        if "day_of_week" in df.columns:
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # 2. Amount & Anomaly Signals
        if "amount" in df.columns:
            df["amount_log"] = np.log1p(df["amount"])
            if hasattr(self, "amount_mean_") and hasattr(self, "amount_std_"):
                df["amount_zscore"] = (df["amount"] - self.amount_mean_) / (
                    self.amount_std_ + 1e-9
                )
            df["is_micro_txn"] = (df["amount"] < 10.0).astype(int)
            df["is_large_txn"] = (df["amount"] >= 4500.0).astype(int)
            df["is_round_amount"] = (df["amount"] % 100 == 0).astype(int)

        # 3. Velocity Metrics
        if "txn_per_hour" in df.columns:
            if hasattr(self, "velocity_p95_"):
                df["is_high_velocity"] = (
                    df["txn_per_hour"] > self.velocity_p95_
                ).astype(int)
            df["velocity_log"] = np.log1p(df["txn_per_hour"])

        # 4. Account Age Metrics
        if "user_age_days" in df.columns:
            df["is_new_account"] = (df["user_age_days"] < 30).astype(int)
            df["account_age_log"] = np.log1p(df["user_age_days"])

        # 5. Interaction Features
        if "is_night" in df.columns and "velocity_log" in df.columns:
            df["night_velocity"] = df["is_night"] * df["velocity_log"]
        if "is_intl" in df.columns and "is_large_txn" in df.columns:
            df["intl_large"] = df["is_intl"] * df["is_large_txn"]
        if "is_new_account" in df.columns and "is_night" in df.columns:
            df["new_acct_night"] = df["is_new_account"] * df["is_night"]
        if "prev_fail_30d" in df.columns and "is_new_account" in df.columns:
            df["prev_fail_risk"] = df["prev_fail_30d"] * df["is_new_account"]

        return df
