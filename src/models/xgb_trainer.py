import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Suppress verbose Optuna logs to keep terminal output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)


class XGBFraudTrainer:

  def __init__(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50):
    self.X = X
    self.y = y
    self.n_trials = n_trials
    self.best_params = None
    self.best_model = None

  def _objective(self, trial):
    # Hyperparameter search ranges
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float(
            'learning_rate', 0.01, 0.3, log=True
        ),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 100),
        'eval_metric': 'aucpr',
        'random_state': 42,
        'n_jobs': -1,
    }

    # Stratified 5-Fold Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_auc_scores = []

    for train_idx, val_idx in skf.split(self.X, self.y):
      X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
      y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]

      model = XGBClassifier(**params)
      model.fit(X_train, y_train)

      # Predict probabilities for positive class (Fraud = 1)
      preds = model.predict_proba(X_val)[:, 1]
      score = average_precision_score(y_val, preds)
      pr_auc_scores.append(score)

    return sum(pr_auc_scores) / len(pr_auc_scores)

  def optimise(self):
    print(f"🚀 Starting Optuna Bayesian HPO across {self.n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(self._objective, n_trials=self.n_trials)

    self.best_params = study.best_params
    print(f"✅ Optimization Complete!")
    print(f"🏆 Best Validation PR-AUC: {study.best_value:.4f}")
    print("📌 Best Hyperparameters:", self.best_params)
    return self.best_params

  def train(self):
    if not self.best_params:
      raise ValueError("Please run .optimise() before calling .train()!")

    print("📊 Training final model with best hyperparameters...")
    with mlflow.start_run(run_name="XGBoost_Optuna_Fraud"):
      mlflow.log_params(self.best_params)

      self.best_model = XGBClassifier(
          **self.best_params, eval_metric="aucpr", random_state=42, n_jobs=-1
      )
      self.best_model.fit(self.X, self.y)

      final_preds = self.best_model.predict_proba(self.X)[:, 1]
      final_pr_auc = average_precision_score(self.y, final_preds)
      mlflow.log_metric("train_pr_auc", final_pr_auc)

      mlflow.xgboost.log_model(self.best_model, "xgb_fraud_model")

      print(f"🎉 Model logged to MLflow! Final Train PR-AUC: {final_pr_auc:.4f}")
      return self.best_model

  def feature_importance(self):
    if not self.best_model:
      raise ValueError("Train the model first!")

    importance = pd.Series(
        self.best_model.feature_importances_, index=self.X.columns
    ).sort_values(ascending=False)

    print("\n🔍 Top 10 Feature Importances:")
    print(importance.head(10))
    return importance


if __name__ == "__main__":
  from sklearn.datasets import make_classification

  # Generate dummy dataset matching 1.72% class imbalance
  print("Generating synthetic payment fraud data...")
  X_dummy, y_dummy = make_classification(
      n_samples=2000,
      n_features=20,
      weights=[0.9828, 0.0172],
      random_state=42,
  )
  X_df = pd.DataFrame(X_dummy, columns=[f"feat_{i}" for i in range(20)])
  y_df = pd.Series(y_dummy)

  trainer = XGBFraudTrainer(X_df, y_df, n_trials=50)
  trainer.optimise()
  trainer.train()
  trainer.feature_importance()
