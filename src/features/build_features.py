import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class PaymentFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature engineering transformer for payment fraud detection.
    Computes temporal flags, amount anomaly signals, and rolling user velocity metrics.
    """
    def __init__(self, velocity_windows_hours=[1, 24]):
        self.velocity_windows_hours = velocity_windows_hours
        self.user_stats_ = pd.DataFrame()

    def fit(self, X, y=None):
        """
        Calculates baseline user statistics on training data to avoid data leakage.
        """
        df = X.copy()
        if 'user_id' in df.columns and 'amount' in df.columns:
            self.user_stats_ = df.groupby('user_id')['amount'].agg(
                user_avg_amount='mean',
                user_std_amount='std'
            ).reset_index()
        return self

    def transform(self, X):
        """
        Transforms raw transaction logs into expressive feature vectors.
        """
        df = X.copy()

        # 1. Temporal Signals
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            # Flag off-hour transactions (11 PM - 5 AM)
            df['is_night'] = df['hour'].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)

        # 2. Amount & Anomaly Signals
        if 'amount' in df.columns:
            # Log transform to reduce right-skewness: log1p(x) = log(1 + x)
            df['log_amount'] = np.log1p(df['amount'])
            
            # Card testing / micro-transaction flag
            df['is_micro_tx'] = (df['amount'] <= 1.0).astype(int)
            
            # High-value transaction flag
            df['is_high_val_tx'] = (df['amount'] >= 1000.0).astype(int)

            # Behavioral deviation ratio relative to baseline
            if not self.user_stats_.empty and 'user_id' in df.columns:
                df = df.merge(self.user_stats_, on='user_id', how='left')
                df['user_avg_amount'] = df['user_avg_amount'].fillna(df['amount'])
                df['amount_to_avg_ratio'] = df['amount'] / (df['user_avg_amount'] + 1e-5)

        # 3. Rolling Velocity Metrics
        if all(col in df.columns for col in ['user_id', 'timestamp', 'amount']):
            df = df.sort_values(by=['user_id', 'timestamp'])

            for hours in self.velocity_windows_hours:
                window_str = f'{hours}h'
                
                # Rolling transaction counts per user
                df[f'tx_count_{window_str}'] = (
                    df.groupby('user_id')
                    .rolling(window_str, on='timestamp')['amount']
                    .count()
                    .reset_index(level=0, drop=True)
                )

                # Rolling total spend per user
                df[f'tx_sum_{window_str}'] = (
                    df.groupby('user_id')
                    .rolling(window_str, on='timestamp')['amount']
                    .sum()
                    .reset_index(level=0, drop=True)
                )

                # High velocity risk flag (> 5 transactions within 1 hour)
                if hours == 1:
                    df['is_high_velocity'] = (df[f'tx_count_{window_str}'] > 5).astype(int)

        return df


if __name__ == "__main__":
    # Quick execution sanity check
    sample_data = pd.DataFrame({
        'user_id': [101, 101, 101, 102],
        'timestamp': pd.to_datetime([
            '2026-07-27 01:15:00',
            '2026-07-27 01:20:00',
            '2026-07-27 01:25:00',
            '2026-07-27 14:00:00'
        ]),
        'amount': [0.50, 1500.00, 12.00, 45.00]
    })

    engineer = PaymentFeatureEngineer()
    result = engineer.fit_transform(sample_data)
    print("Feature Pipeline Output Sample:")
    print(result[['user_id', 'timestamp', 'amount', 'is_night', 'is_micro_tx', 'is_high_velocity']])
