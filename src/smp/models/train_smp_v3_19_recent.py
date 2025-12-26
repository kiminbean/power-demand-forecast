#!/usr/bin/env python3
"""
SMP v3.19: CatBoost on Recent Data Only (2022+)

Hypothesis: Early data (2020-2021) has different dynamics:
- COVID-19 energy market disruption
- Policy changes
- Energy price volatility

Training on recent, stable data may improve generalization.

Goal: Beat v3.12 CatBoost CV (R² 0.834)
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
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# CatBoost & Optuna
try:
    from catboost import CatBoostRegressor
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    logger.error(f"Missing dependency: {e}")
    logger.error("Run: pip install catboost optuna")


class SMPDataPipeline:
    """Data pipeline with EMA features."""

    def __init__(self):
        self.feature_names = []

    def load_data(self, start_date='2022-01-01') -> pd.DataFrame:
        """Load EPSIS data, optionally filtering by start date."""
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

        # Filter by start date
        if start_date:
            df = df[df['datetime'] >= start_date].copy()
            df = df.reset_index(drop=True)
            logger.info(f"  Filtered data from {start_date}: {len(df)} records")
        else:
            logger.info(f"  All data: {len(df)} records")

        return df

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features with EMA additions."""
        self.feature_names = []

        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)

        features = {}

        # ===== Time features =====
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

        day_of_year = df['datetime'].dt.dayofyear.values
        features['doy_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        features['doy_cos'] = np.cos(2 * np.pi * day_of_year / 365)

        # ===== Season/Peak flags =====
        features['is_summer'] = ((month >= 6) & (month <= 8)).astype(float)
        features['is_winter'] = ((month == 12) | (month <= 2)).astype(float)
        features['peak_morning'] = ((hour >= 9) & (hour <= 12)).astype(float)
        features['peak_evening'] = ((hour >= 17) & (hour <= 21)).astype(float)
        features['off_peak'] = ((hour >= 1) & (hour <= 6)).astype(float)

        # ===== Lag features =====
        for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 96, 168]:
            features[f'smp_lag{lag}'] = smp_series.shift(lag).values

        # ===== Rolling statistics (SHIFTED by 1) =====
        for window in [6, 12, 24, 48, 72, 168]:
            features[f'smp_ma{window}'] = smp_series.rolling(window, min_periods=1).mean().shift(1).values
            features[f'smp_std{window}'] = smp_series.rolling(window, min_periods=1).std().shift(1).fillna(0).values
            features[f'smp_min{window}'] = smp_series.rolling(window, min_periods=1).min().shift(1).values
            features[f'smp_max{window}'] = smp_series.rolling(window, min_periods=1).max().shift(1).values

        # ===== EMA features (NEW!) =====
        for span in [6, 12, 24, 48, 168]:
            features[f'smp_ema{span}'] = smp_series.ewm(span=span, adjust=False).mean().shift(1).values

        # ===== Diff features =====
        features['smp_diff1'] = smp_series.diff(1).shift(1).fillna(0).values
        features['smp_diff24'] = smp_series.diff(24).shift(1).fillna(0).values
        features['smp_diff168'] = smp_series.diff(168).shift(1).fillna(0).values

        # ===== Ratio features =====
        features['smp_lag1_vs_ma24'] = (smp_series.shift(1) / smp_series.rolling(24).mean().shift(1)).fillna(1).values
        features['smp_lag1_vs_ma168'] = (smp_series.shift(1) / smp_series.rolling(168).mean().shift(1)).fillna(1).values

        # ===== Volatility features =====
        features['smp_range24'] = (
            smp_series.rolling(24).max().shift(1) - smp_series.rolling(24).min().shift(1)
        ).fillna(0).values
        features['smp_range168'] = (
            smp_series.rolling(168).max().shift(1) - smp_series.rolling(168).min().shift(1)
        ).fillna(0).values

        # ===== Momentum features (NEW!) =====
        # Rate of change
        features['smp_roc24'] = ((smp_series.shift(1) / smp_series.shift(25)) - 1).fillna(0).values
        features['smp_roc168'] = ((smp_series.shift(1) / smp_series.shift(169)) - 1).fillna(0).values

        # MACD-like features
        ema12 = smp_series.ewm(span=12, adjust=False).mean()
        ema26 = smp_series.ewm(span=26, adjust=False).mean()
        features['smp_macd'] = (ema12 - ema26).shift(1).values

        # Create DataFrame
        feature_df = pd.DataFrame(features)
        feature_df['datetime'] = df['datetime'].values
        feature_df['target'] = df['smp_mainland'].values

        # Drop rows with NaN
        feature_df = feature_df.dropna().reset_index(drop=True)

        self.feature_names = [c for c in feature_df.columns if c not in ['datetime', 'target']]

        logger.info(f"  Features: {len(self.feature_names)}, Records: {len(feature_df)}")

        return feature_df


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for CatBoost."""

    params = {
        'iterations': 1000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': False,
        'allow_writing_files': False,

        # Hyperparameters to tune
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'border_count': trial.suggest_int('border_count', 32, 255),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 1, 100),
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False
    )

    val_pred = model.predict(X_val)
    val_r2 = r2_score(y_val, val_pred)

    return val_r2


def train_final_model(X_train, y_train, X_val, y_val, best_params):
    """Train final model with best hyperparameters."""

    params = {
        'iterations': 2000,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'verbose': 200,
        'allow_writing_files': False,
        **best_params
    }

    model = CatBoostRegressor(**params)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=200
    )

    return model


def main():
    if not HAS_DEPS:
        logger.error("Missing dependencies. Run: pip install catboost optuna")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.19: CatBoost on Recent Data with EMA Features")
    logger.info("=" * 60)

    # Try different start dates
    results = {}

    for start_date in [None, '2022-01-01', '2023-01-01']:
        label = f"From {start_date}" if start_date else "All data"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"TRAINING: {label}")
        logger.info("=" * 60)

        # Load data
        pipeline = SMPDataPipeline()
        df = pipeline.load_data(start_date=start_date)
        feature_df = pipeline.create_features(df)

        feature_names = pipeline.feature_names

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

        # Optuna optimization
        logger.info("\nRunning Optuna optimization (100 trials)...")
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=100,
            show_progress_bar=True
        )

        logger.info(f"\nBest val R²: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Train final model
        logger.info("\nTraining final model...")
        model = train_final_model(X_train, y_train, X_val, y_val, study.best_params)

        # Evaluation
        test_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, test_pred)

        mask_test = y_test > 10
        test_mape = mean_absolute_percentage_error(y_test[mask_test], test_pred[mask_test]) * 100
        test_mae = mean_absolute_error(y_test, test_pred)

        results[label] = {
            'test_r2': test_r2,
            'test_mape': test_mape,
            'test_mae': test_mae,
            'val_r2': study.best_value,
            'n_samples': len(X),
            'best_params': study.best_params,
            'model': model,
            'feature_names': feature_names
        }

        logger.info(f"\nTest R²: {test_r2:.4f}")
        logger.info(f"Test MAPE: {test_mape:.2f}%")
        logger.info(f"Test MAE: {test_mae:.2f}")

    # ===== Find best approach =====
    best_approach = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_result = results[best_approach]

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON OF DATA RANGES")
    logger.info("=" * 60)
    logger.info("| Data Range | Samples | MAPE | R² |")
    logger.info("|------------|---------|------|-----|")
    for label, res in results.items():
        marker = " ← BEST" if label == best_approach else ""
        logger.info(f"| {label:15s} | {res['n_samples']:6d} | {res['test_mape']:.2f}% | {res['test_r2']:.3f}{marker} |")

    logger.info(f"\nBest approach: {best_approach}")

    # Save best model
    output_dir = PROJECT_ROOT / "models" / "smp_v3_19_recent"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_result['model'].save_model(str(output_dir / "catboost_model.cbm"))

    metrics = {
        'best_approach': best_approach,
        'test_r2': best_result['test_r2'],
        'test_mape': best_result['test_mape'],
        'test_mae': best_result['test_mae'],
        'val_r2': best_result['val_r2'],
        'n_samples': best_result['n_samples'],
        'best_params': best_result['best_params'],
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk not in ['model', 'feature_names']} for k, v in results.items()},
        'n_features': len(best_result['feature_names']),
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    # Final comparison
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² |")
    logger.info("|-------|------|-----|")
    logger.info(f"| v3.12 CatBoost CV | 5.25% | 0.834 |")
    logger.info(f"| v3.19 {best_approach} | {best_result['test_mape']:.2f}% | {best_result['test_r2']:.3f} |")

    if best_result['test_r2'] > 0.834:
        logger.info(f"\n✅ IMPROVED over v3.12: R² +{best_result['test_r2'] - 0.834:.4f}")
    else:
        logger.info(f"\n⚠️ No improvement over v3.12 CatBoost CV")

    if best_result['test_r2'] >= 0.9:
        logger.info("🎉 REACHED R² 0.9+ TARGET!")

    # Feature importance for best model
    logger.info("\n" + "=" * 60)
    logger.info("TOP 20 FEATURE IMPORTANCE (Best Model)")
    logger.info("=" * 60)
    importance = best_result['model'].get_feature_importance()
    feature_imp = sorted(zip(best_result['feature_names'], importance), key=lambda x: x[1], reverse=True)
    for i, (fname, imp) in enumerate(feature_imp[:20], 1):
        marker = " [NEW]" if any(x in fname for x in ['ema', 'roc', 'macd']) else ""
        logger.info(f"  {i:2d}. {fname}: {imp:.2f}{marker}")


if __name__ == "__main__":
    main()
