# tests/test_features.py
import pytest
from src.features.engineer import PaymentFeatureEngineer
from src.data.generator import generate_payment_data, DataConfig

def test_no_data_leakage():
    df = generate_payment_data(DataConfig(n_samples=500))
    X = df.drop('is_fraud', axis=1)
    fe = PaymentFeatureEngineer()
    fe.fit(X.iloc[:400])          # fit on train only
    X_test = fe.transform(X.iloc[400:])  # transform test
    assert 'amount_zscore' in X_test.columns
    # amount_zscore uses training mean/std, not test statistics

def test_micro_txn_flag():
    df = generate_payment_data(DataConfig(n_samples=200))
    X = df.drop('is_fraud', axis=1)
    fe = PaymentFeatureEngineer().fit(X)
    Xt = fe.transform(X)
    assert ((X['amount'] < 10) == Xt['is_micro_txn'].astype(bool)).all()

def test_output_has_more_features():
    df = generate_payment_data(DataConfig(n_samples=200))
    X = df.drop('is_fraud', axis=1)
    fe = PaymentFeatureEngineer().fit(X)
    assert fe.transform(X).shape[1] > X.shape[1]
