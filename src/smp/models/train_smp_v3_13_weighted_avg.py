#!/usr/bin/env python3
"""
SMP v3.13: Optimized Weighted Average Ensemble

Based on v3.12 findings:
- Simple average (R² 0.833) was better than Ridge stacking (R² 0.811)
- CatBoost had highest weight (0.47) in meta-learner

This version optimizes weights directly using scipy minimize.

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
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

# Tree models
try:
    import lightgbm as lgb
    from catboost import CatBoostRegressor
    import xgboost as xgb
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    print(f"Missing dependency: {e}")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class SMPDataPipeline:
    """Data pipeline for ensemble."""

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
        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)
        features = {}

        # Time features
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

        # Season/Peak flags
        features['is_summer'] = ((month >= 6) & (month <= 8)).astype(float)
        features['is_winter'] = ((month == 12) | (month <= 2)).astype(float)
        features['peak_morning'] = ((hour >= 9) & (hour <= 12)).astype(float)
        features['peak_evening'] = ((hour >= 17) & (hour <= 21)).astype(float)
        features['off_peak'] = ((hour >= 1) & (hour <= 6)).astype(float)

        # Lag features
        for lag in [1, 2, 3, 4, 5, 6, 12, 24, 48, 72, 96, 168]:
            features[f'smp_lag{lag}'] = smp_series.shift(lag).values

        # Rolling statistics (SHIFTED by 1)
        for window in [6, 12, 24, 48, 72, 168]:
            features[f'smp_ma{window}'] = smp_series.rolling(window, min_periods=1).mean().shift(1).values
            features[f'smp_std{window}'] = smp_series.rolling(window, min_periods=1).std().shift(1).fillna(0).values
            features[f'smp_min{window}'] = smp_series.rolling(window, min_periods=1).min().shift(1).values
            features[f'smp_max{window}'] = smp_series.rolling(window, min_periods=1).max().shift(1).values

        # Diff features
        features['smp_diff1'] = smp_series.diff(1).shift(1).fillna(0).values
        features['smp_diff24'] = smp_series.diff(24).shift(1).fillna(0).values
        features['smp_diff168'] = smp_series.diff(168).shift(1).fillna(0).values

        # Ratio features
        features['smp_lag1_vs_ma24'] = (smp_series.shift(1) / smp_series.rolling(24).mean().shift(1)).fillna(1).values
        features['smp_lag1_vs_ma168'] = (smp_series.shift(1) / smp_series.rolling(168).mean().shift(1)).fillna(1).values

        # Volatility features
        features['smp_range24'] = (smp_series.rolling(24).max().shift(1) - smp_series.rolling(24).min().shift(1)).fillna(0).values
        features['smp_range168'] = (smp_series.rolling(168).max().shift(1) - smp_series.rolling(168).min().shift(1)).fillna(0).values

        feature_df = pd.DataFrame(features)
        feature_df['datetime'] = df['datetime'].values
        feature_df['target'] = df['smp_mainland'].values
        feature_df = feature_df.dropna().reset_index(drop=True)

        self.feature_names = [c for c in feature_df.columns if c not in ['datetime', 'target']]
        logger.info(f"  Features: {len(self.feature_names)}, Records: {len(feature_df)}")
        return feature_df


def get_lgb_model():
    """LightGBM with best params."""
    return lgb.LGBMRegressor(
        objective='regression', n_estimators=1500, num_leaves=90, max_depth=10,
        learning_rate=0.0145, feature_fraction=0.628, bagging_fraction=0.903,
        bagging_freq=1, reg_alpha=5.05, reg_lambda=3.38, verbose=-1, n_jobs=-1, random_state=42
    )


def get_catboost_model():
    """CatBoost with best params."""
    return CatBoostRegressor(
        iterations=1500, depth=9, learning_rate=0.046, l2_leaf_reg=0.74,
        bagging_temperature=0.91, random_strength=0.043, border_count=104,
        min_data_in_leaf=22, loss_function='RMSE', random_seed=42, verbose=False, allow_writing_files=False
    )


def get_xgb_model():
    """XGBoost with best params."""
    return xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1500, max_depth=7, learning_rate=0.035,
        min_child_weight=75, subsample=0.867, colsample_bytree=0.821, reg_alpha=0.0014,
        reg_lambda=5.78, gamma=0.35, tree_method='hist', random_state=42, verbosity=0
    )


def train_base_models_cv(X_train, y_train, X_test, n_folds=5):
    """Train base models with K-fold CV."""
    n_train, n_test = len(X_train), len(X_test)

    oof_lgb = np.zeros(n_train)
    oof_cat = np.zeros(n_train)
    oof_xgb = np.zeros(n_train)

    test_lgb = np.zeros(n_test)
    test_cat = np.zeros(n_test)
    test_xgb = np.zeros(n_test)

    kf = KFold(n_splits=n_folds, shuffle=False)

    logger.info(f"\nTraining base models with {n_folds}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
        logger.info(f"  Fold {fold}/{n_folds}")

        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # LightGBM
        lgb_model = get_lgb_model()
        lgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        oof_lgb[val_idx] = lgb_model.predict(X_val)
        test_lgb += lgb_model.predict(X_test) / n_folds

        # CatBoost
        cat_model = get_catboost_model()
        cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=100)
        oof_cat[val_idx] = cat_model.predict(X_val)
        test_cat += cat_model.predict(X_test) / n_folds

        # XGBoost
        xgb_model = get_xgb_model()
        xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict(X_val)
        test_xgb += xgb_model.predict(X_test) / n_folds

    logger.info("\n  OOF R² scores:")
    logger.info(f"    LightGBM: {r2_score(y_train, oof_lgb):.4f}")
    logger.info(f"    CatBoost: {r2_score(y_train, oof_cat):.4f}")
    logger.info(f"    XGBoost:  {r2_score(y_train, oof_xgb):.4f}")

    return (oof_lgb, oof_cat, oof_xgb), (test_lgb, test_cat, test_xgb)


def optimize_weights(oof_preds, y_train):
    """Find optimal weights using scipy minimize."""
    oof_lgb, oof_cat, oof_xgb = oof_preds

    def neg_r2(weights):
        w1, w2, w3 = weights
        # Normalize weights
        total = w1 + w2 + w3
        w1, w2, w3 = w1/total, w2/total, w3/total
        pred = w1 * oof_lgb + w2 * oof_cat + w3 * oof_xgb
        return -r2_score(y_train, pred)

    # Grid search for initial estimate
    logger.info("\nOptimizing weights...")
    best_r2 = -np.inf
    best_weights = (1/3, 1/3, 1/3)

    for w1 in np.arange(0.1, 0.6, 0.05):
        for w2 in np.arange(0.1, 0.6, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.1:
                continue
            pred = w1 * oof_lgb + w2 * oof_cat + w3 * oof_xgb
            r2 = r2_score(y_train, pred)
            if r2 > best_r2:
                best_r2 = r2
                best_weights = (w1, w2, w3)

    logger.info(f"  Grid search best: w_lgb={best_weights[0]:.2f}, w_cat={best_weights[1]:.2f}, w_xgb={best_weights[2]:.2f}")
    logger.info(f"  Grid search R²: {best_r2:.4f}")

    # Fine-tune with scipy
    result = minimize(
        neg_r2,
        x0=best_weights,
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )

    final_weights = result.x
    total = sum(final_weights)
    final_weights = tuple(w/total for w in final_weights)

    final_pred = final_weights[0] * oof_lgb + final_weights[1] * oof_cat + final_weights[2] * oof_xgb
    final_r2 = r2_score(y_train, final_pred)

    logger.info(f"  Optimized: w_lgb={final_weights[0]:.3f}, w_cat={final_weights[1]:.3f}, w_xgb={final_weights[2]:.3f}")
    logger.info(f"  Optimized OOF R²: {final_r2:.4f}")

    return final_weights


def main():
    if not HAS_DEPS:
        logger.error("Missing dependencies.")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.13: Optimized Weighted Average Ensemble")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    pipeline = SMPDataPipeline()
    df = pipeline.load_data()
    feature_df = pipeline.create_features(df)
    feature_names = pipeline.feature_names

    # Prepare data
    X = feature_df[feature_names].values
    y = feature_df['target'].values

    # Train/Test split (85/15)
    n = len(X)
    train_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    logger.info(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train base models
    oof_preds, test_preds = train_base_models_cv(X_train, y_train, X_test, n_folds=5)
    oof_lgb, oof_cat, oof_xgb = oof_preds
    test_lgb, test_cat, test_xgb = test_preds

    # Optimize weights
    weights = optimize_weights(oof_preds, y_train)
    w_lgb, w_cat, w_xgb = weights

    # Final predictions
    weighted_pred = w_lgb * test_lgb + w_cat * test_cat + w_xgb * test_xgb

    # Evaluation
    logger.info("\n" + "=" * 60)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 60)

    mask = y_test > 10

    # Individual models
    lgb_r2 = r2_score(y_test, test_lgb)
    lgb_mape = mean_absolute_percentage_error(y_test[mask], test_lgb[mask]) * 100

    cat_r2 = r2_score(y_test, test_cat)
    cat_mape = mean_absolute_percentage_error(y_test[mask], test_cat[mask]) * 100

    xgb_r2 = r2_score(y_test, test_xgb)
    xgb_mape = mean_absolute_percentage_error(y_test[mask], test_xgb[mask]) * 100

    # Simple average
    avg_pred = (test_lgb + test_cat + test_xgb) / 3
    avg_r2 = r2_score(y_test, avg_pred)
    avg_mape = mean_absolute_percentage_error(y_test[mask], avg_pred[mask]) * 100

    # Weighted average
    weighted_r2 = r2_score(y_test, weighted_pred)
    weighted_mape = mean_absolute_percentage_error(y_test[mask], weighted_pred[mask]) * 100
    weighted_mae = mean_absolute_error(y_test, weighted_pred)

    logger.info("\n| Model | MAPE | R² |")
    logger.info("|-------|------|-----|")
    logger.info(f"| LightGBM | {lgb_mape:.2f}% | {lgb_r2:.4f} |")
    logger.info(f"| CatBoost | {cat_mape:.2f}% | {cat_r2:.4f} |")
    logger.info(f"| XGBoost | {xgb_mape:.2f}% | {xgb_r2:.4f} |")
    logger.info(f"| Simple Avg | {avg_mape:.2f}% | {avg_r2:.4f} |")
    logger.info(f"| **Weighted Avg** | **{weighted_mape:.2f}%** | **{weighted_r2:.4f}** |")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_13_weighted_avg"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'weights': {'lgb': w_lgb, 'catboost': w_cat, 'xgb': w_xgb},
        'lgb_r2': lgb_r2, 'lgb_mape': lgb_mape,
        'cat_r2': cat_r2, 'cat_mape': cat_mape,
        'xgb_r2': xgb_r2, 'xgb_mape': xgb_mape,
        'avg_r2': avg_r2, 'avg_mape': avg_mape,
        'weighted_r2': weighted_r2, 'weighted_mape': weighted_mape, 'weighted_mae': weighted_mae,
        'features': len(feature_names), 'n_folds': 5,
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH PREVIOUS MODELS")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² | Notes |")
    logger.info("|-------|------|-----|-------|")
    logger.info(f"| v3.2 (BiLSTM) | 7.42% | 0.760 | Baseline |")
    logger.info(f"| v3.10 (CatBoost) | 5.38% | 0.826 | Single model best |")
    logger.info(f"| v3.12 (CatBoost CV) | 5.25% | 0.834 | Previous best |")
    logger.info(f"| v3.13 (Weighted Avg) | {weighted_mape:.2f}% | {weighted_r2:.3f} | Optimized weights |")

    if weighted_r2 > 0.834:
        logger.info(f"\n✅ WEIGHTED AVG IMPROVED: R² +{weighted_r2 - 0.834:.4f}")
    elif weighted_r2 > 0.833:
        logger.info(f"\n✅ Better than simple average")
    else:
        logger.info(f"\n⚠️ No improvement")

    if weighted_r2 >= 0.9:
        logger.info("🎉 REACHED R² 0.9+ TARGET!")


if __name__ == "__main__":
    main()
