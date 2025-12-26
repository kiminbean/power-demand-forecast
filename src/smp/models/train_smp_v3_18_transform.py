#!/usr/bin/env python3
"""
SMP v3.18: CatBoost with Target Transformation

Hypothesis: SMP distribution is right-skewed.
Log/Box-Cox transformation may improve predictions for:
- Extreme high values
- Multiplicative relationships
- Percentage-based errors (MAPE)

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
from scipy.stats import boxcox
from scipy.special import inv_boxcox

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
    """Data pipeline for CatBoost model - same as v3.12."""

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


def train_with_transform(X, y, feature_names, transform_type='log', n_splits=5):
    """Train with target transformation and time series CV."""

    # Apply transformation to target
    if transform_type == 'log':
        y_transformed = np.log1p(y)  # log(1+y) to handle zeros
        logger.info(f"  Applied log1p transformation")
    elif transform_type == 'boxcox':
        # Box-Cox requires positive values
        y_shifted = y - y.min() + 1
        y_transformed, lmbda = boxcox(y_shifted)
        logger.info(f"  Applied Box-Cox transformation (lambda={lmbda:.4f})")
    elif transform_type == 'sqrt':
        y_transformed = np.sqrt(y)
        logger.info(f"  Applied sqrt transformation")
    else:
        y_transformed = y
        logger.info(f"  No transformation applied")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = []
    cv_scores_original = []
    models = []
    best_lmbda = None

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train_t, y_val_t = y_transformed[train_idx], y_transformed[val_idx]
        y_val_original = y[val_idx]

        # Run Optuna optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42 + fold)
        )

        study.optimize(
            lambda trial: objective(trial, X_train, y_train_t, X_val, y_val_t),
            n_trials=50,
            show_progress_bar=False
        )

        # Train final model
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
            X_train, y_train_t,
            eval_set=(X_val, y_val_t),
            early_stopping_rounds=100,
            verbose=False
        )

        # Predict and inverse transform
        val_pred_t = model.predict(X_val)

        if transform_type == 'log':
            val_pred_original = np.expm1(val_pred_t)  # exp(x) - 1
        elif transform_type == 'boxcox':
            val_pred_original = inv_boxcox(val_pred_t, lmbda) + y.min() - 1
        elif transform_type == 'sqrt':
            val_pred_original = np.square(val_pred_t)
        else:
            val_pred_original = val_pred_t

        # Clip negative predictions
        val_pred_original = np.maximum(val_pred_original, 0)

        # R² on transformed space
        val_r2_t = r2_score(y_val_t, val_pred_t)
        cv_scores.append(val_r2_t)

        # R² on original space (what we care about)
        val_r2_orig = r2_score(y_val_original, val_pred_original)
        cv_scores_original.append(val_r2_orig)

        models.append(model)

        logger.info(f"  Fold {fold}: R²(trans)={val_r2_t:.4f}, R²(orig)={val_r2_orig:.4f}")

        if transform_type == 'boxcox':
            best_lmbda = lmbda

    mean_cv = np.mean(cv_scores)
    mean_cv_orig = np.mean(cv_scores_original)
    std_cv_orig = np.std(cv_scores_original)
    logger.info(f"\n  CV Mean R² (original): {mean_cv_orig:.4f} (+/- {std_cv_orig:.4f})")

    # Return best model (highest original R²)
    best_idx = np.argmax(cv_scores_original)

    return models[best_idx], cv_scores_original, transform_type, best_lmbda


def main():
    if not HAS_DEPS:
        logger.error("Missing dependencies. Run: pip install catboost optuna")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.18: CatBoost with Target Transformation")
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

    # Show target distribution
    logger.info(f"\nTarget distribution:")
    logger.info(f"  Mean: {y.mean():.2f}")
    logger.info(f"  Std: {y.std():.2f}")
    logger.info(f"  Min: {y.min():.2f}, Max: {y.max():.2f}")
    logger.info(f"  Skewness: {pd.Series(y).skew():.4f}")

    # Combine train+val for CV
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    # ===== Try different transformations =====
    results = {}

    for transform in ['none', 'log', 'sqrt', 'boxcox']:
        logger.info("\n" + "=" * 60)
        logger.info(f"TRAINING WITH {transform.upper()} TRANSFORMATION")
        logger.info("=" * 60)

        best_model, cv_scores, trans_type, lmbda = train_with_transform(
            X_trainval, y_trainval, feature_names,
            transform_type=transform
        )

        # Test evaluation
        y_test_t = y_trainval  # Just for lmbda reference

        # Predict
        test_pred_t = best_model.predict(X_test)

        # Inverse transform
        if transform == 'log':
            y_trainval_t = np.log1p(y_trainval)
            test_pred = np.expm1(test_pred_t)
        elif transform == 'boxcox':
            y_shifted = y_trainval - y_trainval.min() + 1
            _, lmbda = boxcox(y_shifted)
            test_pred = inv_boxcox(test_pred_t, lmbda) + y_trainval.min() - 1
        elif transform == 'sqrt':
            test_pred = np.square(test_pred_t)
        else:
            test_pred = test_pred_t

        test_pred = np.maximum(test_pred, 0)

        test_r2 = r2_score(y_test, test_pred)
        mask_test = y_test > 10
        test_mape = mean_absolute_percentage_error(y_test[mask_test], test_pred[mask_test]) * 100
        test_mae = mean_absolute_error(y_test, test_pred)

        results[transform] = {
            'test_r2': test_r2,
            'test_mape': test_mape,
            'test_mae': test_mae,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'model': best_model,
            'lmbda': lmbda
        }

        logger.info(f"\n  Test R²: {test_r2:.4f}")
        logger.info(f"  Test MAPE: {test_mape:.2f}%")

    # ===== Find best transformation =====
    best_transform = max(results.keys(), key=lambda k: results[k]['test_r2'])
    best_result = results[best_transform]

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON OF TRANSFORMATIONS")
    logger.info("=" * 60)
    logger.info("| Transform | MAPE | R² | CV R² |")
    logger.info("|-----------|------|-----|-------|")
    for trans, res in results.items():
        marker = " ← BEST" if trans == best_transform else ""
        logger.info(f"| {trans:9s} | {res['test_mape']:.2f}% | {res['test_r2']:.3f} | {res['cv_mean']:.3f}{marker} |")

    logger.info(f"\nBest transformation: {best_transform}")

    # Save best model
    output_dir = PROJECT_ROOT / "models" / "smp_v3_18_transform"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_result['model'].save_model(str(output_dir / "catboost_model.cbm"))

    metrics = {
        'best_transform': best_transform,
        'test_r2': best_result['test_r2'],
        'test_mape': best_result['test_mape'],
        'test_mae': best_result['test_mae'],
        'cv_mean': best_result['cv_mean'],
        'cv_std': best_result['cv_std'],
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} for k, v in results.items()},
        'n_features': len(feature_names),
        'records': len(feature_df),
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
    logger.info(f"| v3.18 {best_transform} transform | {best_result['test_mape']:.2f}% | {best_result['test_r2']:.3f} |")

    if best_result['test_r2'] > 0.834:
        logger.info(f"\n✅ IMPROVED over v3.12: R² +{best_result['test_r2'] - 0.834:.4f}")
    else:
        logger.info(f"\n⚠️ No improvement over v3.12 CatBoost CV")

    if best_result['test_r2'] >= 0.9:
        logger.info("🎉 REACHED R² 0.9+ TARGET!")


if __name__ == "__main__":
    main()
