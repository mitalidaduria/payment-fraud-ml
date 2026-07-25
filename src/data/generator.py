from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class DataConfig:
  n_samples: int = 100_000
  fraud_rate: float = 0.0172  # Realistic 1.72% card-present fraud rate
  random_state: int = 42


def _hour_dist():
  """Generates realistic hourly distribution for legitimate activity."""
  w = np.ones(24)
  w[9:18] *= 3  # Daytime peak
  w[12:14] *= 1.5  # Lunch spike
  return w / w.sum()


def generate_payment_data(config: DataConfig = DataConfig()) -> pd.DataFrame:
  np.random.seed(config.random_state)
  n = config.n_samples
  n_fraud = int(n * config.fraud_rate)
  n_legit = n - n_fraud

  # 1. Legitimate transactions
  legit = pd.DataFrame({
      'amount': np.random.exponential(800, n_legit).clip(10, 50000),
      'hour_of_day': np.random.choice(range(24), n_legit, p=_hour_dist()),
      'day_of_week': np.random.randint(0, 7, n_legit),
      'gateway': np.random.choice(
          ['Razorpay', 'PayU', 'Stripe', 'HDFC'], n_legit
      ),
      'txn_per_hour': np.random.poisson(2, n_legit),
      'user_age_days': np.random.exponential(365, n_legit)
      .astype(int)
      .clip(1, 3650),
      'is_intl': np.random.choice([0, 1], n_legit, p=[0.85, 0.15]),
      'prev_fail_30d': np.random.choice(
          range(5), n_legit, p=[0.7, 0.15, 0.08, 0.04, 0.03]
      ),
      'merchant_cat': np.random.choice(
          ['retail', 'food', 'travel', 'utilities', 'entertainment'], n_legit
      ),
      'is_fraud': 0,
  })

  # 2. Fraudulent transactions (Pattern signals: night hours, velocity spikes, new accounts, international)
  low_amounts = np.random.uniform(1, 50, n_fraud // 2)  # Card testing
  high_amounts = np.random.uniform(
      4500, 9999, n_fraud - (n_fraud // 2)
  )  # High value theft
  fraud_amounts = np.concatenate([low_amounts, high_amounts])
  np.random.shuffle(fraud_amounts)

  fraud = pd.DataFrame({
      'amount': fraud_amounts.clip(1, 50000),
      'hour_of_day': np.random.choice(
          [0, 1, 2, 3, 23], n_fraud
      ),  # Off-hour transactions
      'day_of_week': np.random.randint(0, 7, n_fraud),
      'gateway': np.random.choice(
          ['Razorpay', 'PayU', 'Stripe', 'HDFC'], n_fraud
      ),
      'txn_per_hour': np.random.poisson(12, n_fraud),  # High velocity spike
      'user_age_days': np.random.exponential(30, n_fraud)
      .astype(int)
      .clip(1, 3650),  # Newer accounts
      'is_intl': np.random.choice([0, 1], n_fraud, p=[0.3, 0.7]),  # Higher intl
      'prev_fail_30d': np.random.choice(
          range(5), n_fraud, p=[0.3, 0.2, 0.2, 0.15, 0.15]
      ),
      'merchant_cat': np.random.choice(
          ['retail', 'food', 'travel', 'utilities', 'entertainment'], n_fraud
      ),
      'is_fraud': 1,
  })

  return (
      pd.concat([legit, fraud])
      .sample(frac=1, random_state=config.random_state)
      .reset_index(drop=True)
  )


if __name__ == '__main__':
  df = generate_payment_data()
  print(
      f'Generated: {len(df):,} records | Fraud rate:'
      f' {df.is_fraud.mean()*100:.2f}%'
  )
