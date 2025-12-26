#!/usr/bin/env python3
"""
SMP v3.14: CatBoost with Fuel Price Features

Hypothesis: Fuel prices directly influence SMP because:
- Natural gas price affects CCGT marginal cost
- Oil prices affect oil-fired generation cost
- Higher fuel costs → higher SMP

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


class SMPDataPipelineWithFuel:
    """Data pipeline with fuel price integration."""

    def __init__(self):
        self.feature_names = []

    def load_smp_data(self) -> pd.DataFrame:
        """Load EPSIS SMP data"""
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

        logger.info(f"  SMP data: {len(df)} records, {df['datetime'].min()} ~ {df['datetime'].max()}")

        return df

    def load_fuel_data(self) -> pd.DataFrame:
        """Load fuel price data"""
        fuel_path = PROJECT_ROOT / 'data' / 'external' / 'fuel' / 'fuel_prices.csv'
        fuel_df = pd.read_csv(fuel_path)
        fuel_df['date'] = pd.to_datetime(fuel_df['date'])
        fuel_df = fuel_df.sort_values('date').reset_index(drop=True)

        logger.info(f"  Fuel data: {len(fuel_df)} records, {fuel_df['date'].min()} ~ {fuel_df['date'].max()}")

        return fuel_df

    def merge_fuel_to_smp(self, smp_df: pd.DataFrame, fuel_df: pd.DataFrame) -> pd.DataFrame:
        """Merge daily fuel prices to hourly SMP data"""
        # Extract date from SMP datetime
        smp_df = smp_df.copy()
        smp_df['date'] = smp_df['datetime'].dt.date
        smp_df['date'] = pd.to_datetime(smp_df['date'])

        # Merge fuel prices (use previous day's price to avoid leakage)
        fuel_df = fuel_df.copy()
        fuel_df['date'] = fuel_df['date'] + pd.Timedelta(days=1)  # Shift by 1 day

        merged = smp_df.merge(fuel_df, on='date', how='left')

        # Forward fill missing fuel prices
        fuel_cols = ['wti_crude', 'brent_crude', 'natural_gas', 'heating_oil']
        for col in fuel_cols:
            merged[col] = merged[col].ffill()

        # Drop rows without fuel data
        merged = merged.dropna(subset=fuel_cols)

        logger.info(f"  Merged data: {len(merged)} records (fuel data available)")

        return merged

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features including fuel price features."""
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

        # ===== SMP Lag features =====
        for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 96, 168]:
            features[f'smp_lag{lag}'] = smp_series.shift(lag).values

        # ===== SMP Rolling statistics (SHIFTED by 1) =====
        for window in [6, 12, 24, 48, 72, 168]:
            features[f'smp_ma{window}'] = smp_series.rolling(window, min_periods=1).mean().shift(1).values
            features[f'smp_std{window}'] = smp_series.rolling(window, min_periods=1).std().shift(1).fillna(0).values
            features[f'smp_min{window}'] = smp_series.rolling(window, min_periods=1).min().shift(1).values
            features[f'smp_max{window}'] = smp_series.rolling(window, min_periods=1).max().shift(1).values

        # ===== SMP Diff features =====
        features['smp_diff1'] = smp_series.diff(1).shift(1).fillna(0).values
        features['smp_diff24'] = smp_series.diff(24).shift(1).fillna(0).values
        features['smp_diff168'] = smp_series.diff(168).shift(1).fillna(0).values

        # ===== SMP Ratio features =====
        features['smp_lag1_vs_ma24'] = (smp_series.shift(1) / smp_series.rolling(24).mean().shift(1)).fillna(1).values
        features['smp_lag1_vs_ma168'] = (smp_series.shift(1) / smp_series.rolling(168).mean().shift(1)).fillna(1).values

        # ===== SMP Volatility features =====
        features['smp_range24'] = (
            smp_series.rolling(24).max().shift(1) - smp_series.rolling(24).min().shift(1)
        ).fillna(0).values
        features['smp_range168'] = (
            smp_series.rolling(168).max().shift(1) - smp_series.rolling(168).min().shift(1)
        ).fillna(0).values

        # ===== FUEL PRICE FEATURES (NEW!) =====
        fuel_cols = ['wti_crude', 'brent_crude', 'natural_gas', 'heating_oil']

        for col in fuel_cols:
            fuel_series = pd.Series(df[col].values)

            # Current value (already shifted by 1 day in merge)
            features[col] = fuel_series.values

            # Lag features (1-7 days = 24-168 hours)
            for lag_days in [1, 2, 3, 7]:
                lag_hours = lag_days * 24
                features[f'{col}_lag{lag_days}d'] = fuel_series.shift(lag_hours).values

            # Rolling means (7, 14, 30 days)
            for window_days in [7, 14, 30]:
                window_hours = window_days * 24
                features[f'{col}_ma{window_days}d'] = fuel_series.rolling(window_hours, min_periods=24).mean().shift(24).values

            # Daily change (compared to previous day)
            features[f'{col}_diff1d'] = fuel_series.diff(24).shift(24).fillna(0).values

            # Ratio to 7-day MA
            ma7d = fuel_series.rolling(7 * 24, min_periods=24).mean().shift(24)
            features[f'{col}_vs_ma7d'] = (fuel_series / ma7d).fillna(1).values

        # ===== Cross-fuel features =====
        wti = pd.Series(df['wti_crude'].values)
        ng = pd.Series(df['natural_gas'].values)

        # Oil/Gas spread (important for fuel switching)
        features['wti_ng_ratio'] = (wti / ng).fillna(wti.mean() / ng.mean()).values

        # Create DataFrame
        feature_df = pd.DataFrame(features)
        feature_df['datetime'] = df['datetime'].values
        feature_df['target'] = df['smp_mainland'].values

        # Drop rows with NaN
        feature_df = feature_df.dropna().reset_index(drop=True)

        self.feature_names = [c for c in feature_df.columns if c not in ['datetime', 'target']]

        logger.info(f"  Total features: {len(self.feature_names)}, Records: {len(feature_df)}")

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


def train_with_cv(X, y, feature_names, n_splits=5):
    """Train with time series cross-validation."""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Run Optuna optimization (fewer trials per fold)
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42 + fold)
        )

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=50,
            show_progress_bar=False
        )

        # Train final model with best params
        params = {
            'iterations': 2000,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            **study.best_params
        }

        model = CatBoostRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=100,
            verbose=False
        )

        val_pred = model.predict(X_val)
        val_r2 = r2_score(y_val, val_pred)
        cv_scores.append(val_r2)
        models.append(model)

        logger.info(f"  Fold {fold}: R² = {val_r2:.4f}")

    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    logger.info(f"\n  CV Mean R²: {mean_cv:.4f} (+/- {std_cv:.4f})")

    # Return best model (highest validation R²)
    best_idx = np.argmax(cv_scores)
    return models[best_idx], cv_scores


def main():
    if not HAS_DEPS:
        logger.error("Missing dependencies. Run: pip install catboost optuna")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.14: CatBoost with Fuel Price Features")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    pipeline = SMPDataPipelineWithFuel()
    smp_df = pipeline.load_smp_data()
    fuel_df = pipeline.load_fuel_data()

    # Merge fuel prices to SMP
    logger.info("\nMerging fuel prices...")
    merged_df = pipeline.merge_fuel_to_smp(smp_df, fuel_df)

    # Create features
    logger.info("\nCreating features...")
    feature_df = pipeline.create_features(merged_df)

    feature_names = pipeline.feature_names
    logger.info(f"Total features: {len(feature_names)}")

    # Show fuel-related features
    fuel_features = [f for f in feature_names if any(x in f for x in ['wti', 'brent', 'natural_gas', 'heating', 'ng_ratio'])]
    logger.info(f"Fuel features added: {len(fuel_features)}")

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

    # ===== Train with CV =====
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING WITH 5-FOLD TIME SERIES CV")
    logger.info("=" * 60)

    # Combine train+val for CV
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    best_model, cv_scores = train_with_cv(X_trainval, y_trainval, feature_names)

    # ===== Final Evaluation on Test Set =====
    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)

    test_pred = best_model.predict(X_test)
    test_r2 = r2_score(y_test, test_pred)

    mask_test = y_test > 10
    test_mape = mean_absolute_percentage_error(y_test[mask_test], test_pred[mask_test]) * 100
    test_mae = mean_absolute_error(y_test, test_pred)

    logger.info(f"\nTest R²:  {test_r2:.4f}")
    logger.info(f"Test MAPE: {test_mape:.2f}%")
    logger.info(f"Test MAE:  {test_mae:.2f}")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_14_fuel"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model.save_model(str(output_dir / "catboost_model.cbm"))

    metrics = {
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'test_r2': test_r2,
        'test_mape': test_mape,
        'test_mae': test_mae,
        'n_features': len(feature_names),
        'n_fuel_features': len(fuel_features),
        'feature_names': feature_names,
        'fuel_features': fuel_features,
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
    logger.info("| Model | MAPE | R² | Features |")
    logger.info("|-------|------|-----|----------|")
    logger.info(f"| v3.12 CatBoost CV | 5.25% | 0.834 | 60 |")
    logger.info(f"| v3.14 +Fuel | {test_mape:.2f}% | {test_r2:.3f} | {len(feature_names)} |")

    if test_r2 > 0.834:
        logger.info(f"\n✅ IMPROVED over v3.12: R² +{test_r2 - 0.834:.4f}")
    else:
        logger.info(f"\n⚠️ No improvement over v3.12 CatBoost CV")

    if test_r2 >= 0.9:
        logger.info("🎉 REACHED R² 0.9+ TARGET!")

    # Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("TOP 20 FEATURE IMPORTANCE")
    logger.info("=" * 60)
    importance = best_model.get_feature_importance()
    feature_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for i, (fname, imp) in enumerate(feature_imp[:20], 1):
        marker = " [FUEL]" if any(x in fname for x in ['wti', 'brent', 'natural_gas', 'heating', 'ng_ratio']) else ""
        logger.info(f"  {i:2d}. {fname}: {imp:.2f}{marker}")


if __name__ == "__main__":
    main()
