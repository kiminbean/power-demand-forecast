#!/usr/bin/env python3
"""
SMP v3.9: LightGBM with Optuna Hyperparameter Optimization

Based on v3.8 success (R² 0.815), this version applies Optuna tuning to:
- num_leaves, max_depth, min_child_samples
- learning_rate, feature_fraction, bagging_fraction
- reg_alpha, reg_lambda (L1/L2 regularization)

Goal: Push R² beyond 0.815 towards 0.9+
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# LightGBM & Optuna
try:
    import lightgbm as lgb
    import optuna
    from optuna.samplers import TPESampler
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Missing dependency: {e}")
    print("Run: pip install lightgbm optuna")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Suppress Optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class SMPDataPipeline:
    """Data pipeline for LightGBM model."""

    def __init__(self):
        self.feature_names = []

    def load_data(self) -> pd.DataFrame:
        """Load EPSIS data"""
        data_path = PROJECT_ROOT / 'data' / 'smp' / 'smp_5years_epsis.csv'
        df = pd.read_csv(data_path)

        df = df[df['smp_mainland'] > 0].copy()

        def fix_hour_24(ts):
            if ' 24:00' in str(ts):
                date_part = str(ts).replace(' 24:00', '')
                return pd.to_datetime(date_part) + pd.Timedelta(days=1)
            return pd.to_datetime(ts)

        df['datetime'] = df['timestamp'].apply(fix_hour_24)
        df = df.sort_values('datetime').reset_index(drop=True)

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features with NO data leakage."""
        self.feature_names = []

        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)

        # All features use only PAST information (shift by 1 or more)
        features = {}

        # ===== Time features (known at prediction time) =====
        hour = df['hour'].values
        features['hour'] = hour
        features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        day_of_week = df['datetime'].dt.dayofweek.values
        features['dayofweek'] = day_of_week
        features['dow_sin'] = np.sin(2 * np.pi * day_of_week / 7)
        features['dow_cos'] = np.cos(2 * np.pi * day_of_week / 7)
        features['is_weekend'] = (day_of_week >= 5).astype(float)

        month = df['datetime'].dt.month.values
        features['month'] = month
        features['month_sin'] = np.sin(2 * np.pi * month / 12)
        features['month_cos'] = np.cos(2 * np.pi * month / 12)

        # Day of year for yearly seasonality
        day_of_year = df['datetime'].dt.dayofyear.values
        features['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        features['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)

        # ===== Season/Peak flags =====
        features['is_summer'] = ((month >= 6) & (month <= 8)).astype(float)
        features['is_winter'] = ((month == 12) | (month <= 2)).astype(float)
        features['peak_morning'] = ((hour >= 9) & (hour <= 12)).astype(float)
        features['peak_evening'] = ((hour >= 17) & (hour <= 21)).astype(float)
        features['off_peak'] = ((hour >= 1) & (hour <= 6)).astype(float)

        # ===== Lag features (crucial for LightGBM) =====
        for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 96, 168]:  # Added 4, 5, 168 (1 week)
            features[f'smp_lag{lag}'] = smp_series.shift(lag).values

        # ===== Rolling statistics (SHIFTED by 1 to avoid leakage) =====
        for window in [6, 12, 24, 48, 72, 168]:  # Added 168 (1 week)
            features[f'smp_ma{window}'] = smp_series.rolling(window, min_periods=1).mean().shift(1).values
            features[f'smp_std{window}'] = smp_series.rolling(window, min_periods=1).std().shift(1).fillna(0).values
            features[f'smp_min{window}'] = smp_series.rolling(window, min_periods=1).min().shift(1).values
            features[f'smp_max{window}'] = smp_series.rolling(window, min_periods=1).max().shift(1).values

        # ===== Diff features (SHIFTED by 1) =====
        features['smp_diff1'] = smp_series.diff(1).shift(1).fillna(0).values
        features['smp_diff24'] = smp_series.diff(24).shift(1).fillna(0).values
        features['smp_diff168'] = smp_series.diff(168).shift(1).fillna(0).values  # Week-over-week

        # ===== Ratio features =====
        # Current lag vs moving average (momentum-like)
        features['smp_lag1_vs_ma24'] = (smp_series.shift(1) / smp_series.rolling(24).mean().shift(1)).fillna(1).values
        features['smp_lag1_vs_ma168'] = (smp_series.shift(1) / smp_series.rolling(168).mean().shift(1)).fillna(1).values

        # ===== Volatility features =====
        features['smp_range24'] = (
            smp_series.rolling(24).max().shift(1) - smp_series.rolling(24).min().shift(1)
        ).fillna(0).values
        features['smp_range168'] = (
            smp_series.rolling(168).max().shift(1) - smp_series.rolling(168).min().shift(1)
        ).fillna(0).values

        # Create DataFrame
        feature_df = pd.DataFrame(features)
        feature_df['datetime'] = df['datetime'].values
        feature_df['target'] = df['smp_mainland'].values

        # Drop rows with NaN (from lag/rolling)
        feature_df = feature_df.dropna().reset_index(drop=True)

        self.feature_names = [c for c in feature_df.columns if c not in ['datetime', 'target']]

        logger.info(f"  Features: {len(self.feature_names)}, Records: {len(feature_df)}")

        return feature_df


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for LightGBM."""

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42,

        # Hyperparameters to tune
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0),
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
        ]
    )

    val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)

    return val_r2


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Train final model with best hyperparameters."""

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42,
        **best_params
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,  # More rounds for final model
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.log_evaluation(period=200)
        ]
    )

    return model


def main():
    if not HAS_DEPS:
        logger.error("Missing dependencies. Run: pip install lightgbm optuna")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.9: LightGBM with Optuna Optimization")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    pipeline = SMPDataPipeline()
    df = pipeline.load_data()
    feature_df = pipeline.create_features(df)

    feature_names = pipeline.feature_names
    logger.info(f"Total features: {len(feature_names)}")

    # Prepare data
    X = feature_df[feature_names].values
    y = feature_df['target'].values

    # Train/Val/Test split (70/15/15)
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # ===== Optuna Optimization =====
    logger.info("\n" + "=" * 60)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 60)

    n_trials = 100
    logger.info(f"Running {n_trials} trials...")

    study = optuna.create_study(
        direction='maximize',  # Maximize R²
        sampler=TPESampler(seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )

    logger.info(f"\nBest trial:")
    logger.info(f"  R²: {study.best_value:.4f}")
    logger.info(f"  Params: {study.best_params}")

    # ===== Train Final Model =====
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING FINAL MODEL WITH BEST PARAMS")
    logger.info("=" * 60)

    model = train_final_model(X_train, y_train, X_val, y_val, study.best_params)

    # ===== Evaluation =====
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    # Predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, train_pred)
    val_r2 = r2_score(y_val, val_pred)
    test_r2 = r2_score(y_test, test_pred)

    mask_test = y_test > 10
    test_mape = mean_absolute_percentage_error(y_test[mask_test], test_pred[mask_test]) * 100
    test_mae = mean_absolute_error(y_test, test_pred)

    logger.info(f"\nTrain R²: {train_r2:.4f}")
    logger.info(f"Val R²:   {val_r2:.4f}")
    logger.info(f"Test R²:  {test_r2:.4f}")
    logger.info(f"Test MAPE: {test_mape:.2f}%")
    logger.info(f"Test MAE:  {test_mae:.2f}")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_9_optuna_lgb"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model.save_model(str(output_dir / "lightgbm_model.txt"))

    # Save metrics and params
    metrics = {
        'train_r2': train_r2,
        'val_r2': val_r2,
        'test_r2': test_r2,
        'test_mape': test_mape,
        'test_mae': test_mae,
        'n_trials': n_trials,
        'best_params': study.best_params,
        'features': len(feature_names),
        'feature_names': feature_names,
        'records': len(feature_df),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² | Notes |")
    logger.info("|-------|------|-----|-------|")
    logger.info(f"| v3.2 (BiLSTM) | 7.42% | 0.760 | Optuna-tuned LSTM |")
    logger.info(f"| v3.8 (LightGBM) | 5.46% | 0.815 | Default params |")
    logger.info(f"| v3.9 (Optuna LGB) | {test_mape:.2f}% | {test_r2:.3f} | {n_trials} trials |")

    if test_r2 > 0.815:
        logger.info(f"\n✅ IMPROVED over v3.8: R² +{test_r2 - 0.815:.4f}")
    else:
        logger.info(f"\n⚠️ No improvement over v3.8")

    if test_r2 >= 0.9:
        logger.info("🎉 REACHED R² 0.9+ TARGET!")

    # Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("TOP 15 FEATURE IMPORTANCE")
    logger.info("=" * 60)
    importance = model.feature_importance(importance_type='gain')
    feature_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for i, (fname, imp) in enumerate(feature_imp[:15], 1):
        logger.info(f"  {i:2d}. {fname}: {imp:.0f}")


if __name__ == "__main__":
    main()
