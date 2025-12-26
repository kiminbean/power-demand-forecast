#!/usr/bin/env python3
"""
SMP v3.15: CatBoost with Supply Margin Features

Integrates Jeju supply/demand data:
- 공급예비력 (Supply Reserve)
- 공급능력 (Supply Capacity)
- 계통수요 (System Demand)

Note: Supply data only available from 2023-09-01
This limits training data but may improve model accuracy.

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


def load_supply_data() -> pd.DataFrame:
    """Load and merge supply/demand data from Jeju extract files."""
    data_dir = PROJECT_ROOT / 'data' / 'jeju_extract'

    files = {
        'supply_reserve': data_dir / '공급예비력.csv',  # Supply Reserve
        'supply_capacity': data_dir / '공급능력.csv',   # Supply Capacity
        'system_demand': data_dir / '계통수요.csv',     # System Demand
    }

    all_data = []

    for name, filepath in files.items():
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            continue

        # Read with proper encoding for Korean headers
        # Try multiple encodings
        for encoding in ['euc-kr', 'cp949', 'utf-8']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.warning(f"Could not decode {filepath}")
            continue

        # The first column is date, rest are hourly values (1시 ~ 24시)
        date_col = df.columns[0]

        # Melt to long format
        melted = pd.melt(
            df,
            id_vars=[date_col],
            var_name='hour_str',
            value_name=name
        )

        # Extract hour number from column names like "1시 " or " 1시 "
        melted['hour'] = melted['hour_str'].str.extract(r'(\d+)').astype(int)
        melted['date'] = pd.to_datetime(melted[date_col])

        # Create datetime
        melted['datetime'] = melted['date'] + pd.to_timedelta(melted['hour'] - 1, unit='h')

        all_data.append(melted[['datetime', name]])

    if not all_data:
        raise ValueError("No supply data files found!")

    # Merge all dataframes
    result = all_data[0]
    for df in all_data[1:]:
        result = result.merge(df, on='datetime', how='outer')

    result = result.sort_values('datetime').reset_index(drop=True)

    logger.info(f"  Supply data: {len(result)} records")
    logger.info(f"  Date range: {result['datetime'].min()} ~ {result['datetime'].max()}")

    return result


class SMPSupplyPipeline:
    """Data pipeline with supply margin features."""

    def __init__(self):
        self.feature_names = []
        self.supply_data = None

    def load_data(self) -> pd.DataFrame:
        """Load EPSIS SMP data."""
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

        logger.info(f"  SMP data: {len(df)} records")
        logger.info(f"  Date range: {df['datetime'].min()} ~ {df['datetime'].max()}")

        return df

    def load_supply(self) -> pd.DataFrame:
        """Load supply data."""
        self.supply_data = load_supply_data()
        return self.supply_data

    def create_features(self, df: pd.DataFrame, use_supply: bool = True) -> pd.DataFrame:
        """Create features with optional supply margin data."""
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

        # ===== Supply Margin Features (if available) =====
        if use_supply and self.supply_data is not None:
            logger.info("  Adding supply margin features...")

            # Merge supply data
            merged = df[['datetime']].merge(self.supply_data, on='datetime', how='left')

            # Check overlap
            n_valid = merged['supply_reserve'].notna().sum()
            logger.info(f"  Supply data overlap: {n_valid}/{len(merged)} ({100*n_valid/len(merged):.1f}%)")

            if n_valid > 0:
                # Supply Reserve
                if 'supply_reserve' in merged.columns:
                    sr = merged['supply_reserve']
                    features['supply_reserve'] = sr.values
                    features['supply_reserve_lag1'] = sr.shift(1).values
                    features['supply_reserve_lag24'] = sr.shift(24).values
                    features['supply_reserve_ma24'] = sr.rolling(24, min_periods=1).mean().shift(1).values
                    features['supply_reserve_std24'] = sr.rolling(24, min_periods=1).std().shift(1).fillna(0).values

                # Supply Capacity
                if 'supply_capacity' in merged.columns:
                    sc = merged['supply_capacity']
                    features['supply_capacity'] = sc.values
                    features['supply_capacity_lag1'] = sc.shift(1).values
                    features['supply_capacity_lag24'] = sc.shift(24).values
                    features['supply_capacity_ma24'] = sc.rolling(24, min_periods=1).mean().shift(1).values

                # System Demand
                if 'system_demand' in merged.columns:
                    sd = merged['system_demand']
                    features['system_demand'] = sd.values
                    features['system_demand_lag1'] = sd.shift(1).values
                    features['system_demand_lag24'] = sd.shift(24).values
                    features['system_demand_ma24'] = sd.rolling(24, min_periods=1).mean().shift(1).values
                    features['system_demand_std24'] = sd.rolling(24, min_periods=1).std().shift(1).fillna(0).values

                # Derived features
                if 'supply_capacity' in merged.columns and 'system_demand' in merged.columns:
                    # Capacity utilization rate
                    features['capacity_util'] = (
                        merged['system_demand'] / merged['supply_capacity'].replace(0, np.nan)
                    ).fillna(0).values

                    # Supply margin ratio
                    if 'supply_reserve' in merged.columns:
                        features['reserve_ratio'] = (
                            merged['supply_reserve'] / merged['supply_capacity'].replace(0, np.nan)
                        ).fillna(0).values

                        # Low reserve warning (below 10%)
                        features['low_reserve'] = (features['reserve_ratio'] < 0.1).astype(float)

        # Create DataFrame
        feature_df = pd.DataFrame(features)
        feature_df['datetime'] = df['datetime'].values
        feature_df['target'] = df['smp_mainland'].values

        # Drop rows with NaN in core features (not supply features)
        core_features = [c for c in feature_df.columns
                        if c not in ['datetime', 'target']
                        and not c.startswith('supply')
                        and not c.startswith('system')
                        and not c.startswith('capacity')
                        and not c.startswith('reserve')
                        and not c.startswith('low_reserve')]

        mask = feature_df[core_features].notna().all(axis=1)
        feature_df = feature_df[mask].reset_index(drop=True)

        # Fill NaN in supply features with 0 (for records before 2023-09)
        supply_cols = [c for c in feature_df.columns
                      if c.startswith('supply') or c.startswith('system')
                      or c.startswith('capacity') or c.startswith('reserve')
                      or c.startswith('low_reserve')]
        feature_df[supply_cols] = feature_df[supply_cols].fillna(0)

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


def train_cv_model(X, y, feature_names, best_params, n_splits=5):
    """Train with cross-validation."""

    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    best_model = None
    best_val_r2 = -np.inf

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        params = {
            'iterations': 2000,
            'loss_function': 'RMSE',
            'eval_metric': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            **best_params
        }

        model = CatBoostRegressor(**params)

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            early_stopping_rounds=100,
            verbose=False
        )

        val_pred = model.predict(X_val_fold)
        val_r2 = r2_score(y_val_fold, val_pred)

        mask = y_val_fold > 10
        val_mape = mean_absolute_percentage_error(y_val_fold[mask], val_pred[mask]) * 100
        val_mae = mean_absolute_error(y_val_fold, val_pred)

        fold_results.append({
            'fold': fold,
            'r2': val_r2,
            'mape': val_mape,
            'mae': val_mae
        })

        logger.info(f"  Fold {fold}: R²={val_r2:.4f}, MAPE={val_mape:.2f}%")

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_model = model

    # Average metrics
    avg_r2 = np.mean([r['r2'] for r in fold_results])
    avg_mape = np.mean([r['mape'] for r in fold_results])
    avg_mae = np.mean([r['mae'] for r in fold_results])

    logger.info(f"\n  CV Average: R²={avg_r2:.4f}, MAPE={avg_mape:.2f}%, MAE={avg_mae:.2f}")

    return best_model, fold_results, avg_r2, avg_mape, avg_mae


def main():
    if not HAS_DEPS:
        logger.error("Missing dependencies. Run: pip install catboost optuna")
        return

    logger.info("=" * 60)
    logger.info("SMP v3.15: CatBoost with Supply Margin Features")
    logger.info("=" * 60)

    # Load data
    pipeline = SMPSupplyPipeline()

    logger.info("\n[1] Loading SMP data...")
    df = pipeline.load_data()

    logger.info("\n[2] Loading supply margin data...")
    pipeline.load_supply()

    # Run experiments: with and without supply features
    results = {}

    for use_supply in [False, True]:
        label = "With Supply" if use_supply else "Baseline"
        logger.info(f"\n{'=' * 60}")
        logger.info(f"EXPERIMENT: {label}")
        logger.info("=" * 60)

        # Create features
        logger.info("\n[3] Creating features...")
        feature_df = pipeline.create_features(df, use_supply=use_supply)
        feature_names = pipeline.feature_names

        # Prepare data
        X = feature_df[feature_names].values
        y = feature_df['target'].values

        # Train/Val split for Optuna (70/30)
        n = len(X)
        train_end = int(n * 0.7)

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:]
        y_val = y[train_end:]

        logger.info(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

        # Optuna optimization
        logger.info("\n[4] Running Optuna optimization (50 trials)...")
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )

        study.optimize(
            lambda trial: objective(trial, X_train, y_train, X_val, y_val),
            n_trials=50,
            show_progress_bar=True
        )

        logger.info(f"\nBest trial R²: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        # Cross-validation
        logger.info("\n[5] Training with 5-fold CV...")
        model, fold_results, cv_r2, cv_mape, cv_mae = train_cv_model(
            X, y, feature_names, study.best_params, n_splits=5
        )

        results[label] = {
            'cv_r2': cv_r2,
            'cv_mape': cv_mape,
            'cv_mae': cv_mae,
            'fold_results': fold_results,
            'best_params': study.best_params,
            'n_features': len(feature_names),
            'model': model,
            'feature_names': feature_names
        }

    # ===== Compare results =====
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON: Baseline vs With Supply")
    logger.info("=" * 60)
    logger.info("| Model | Features | MAPE | R² |")
    logger.info("|-------|----------|------|-----|")

    for label, res in results.items():
        logger.info(f"| {label:15s} | {res['n_features']:3d} | {res['cv_mape']:.2f}% | {res['cv_r2']:.3f} |")

    # Find best approach
    best_approach = max(results.keys(), key=lambda k: results[k]['cv_r2'])
    best_result = results[best_approach]

    logger.info(f"\nBest approach: {best_approach}")

    # Compare with v3.12
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON WITH v3.12 CatBoost CV")
    logger.info("=" * 60)
    logger.info("| Model | MAPE | R² |")
    logger.info("|-------|------|-----|")
    logger.info(f"| v3.12 CatBoost CV | 5.25% | 0.834 |")
    logger.info(f"| v3.15 {best_approach} | {best_result['cv_mape']:.2f}% | {best_result['cv_r2']:.3f} |")

    if best_result['cv_r2'] > 0.834:
        logger.info(f"\n✅ IMPROVED over v3.12: R² +{best_result['cv_r2'] - 0.834:.4f}")
    else:
        logger.info(f"\n⚠️ No improvement over v3.12 CatBoost CV")

    # Save results
    output_dir = PROJECT_ROOT / "models" / "smp_v3_15_supply"
    output_dir.mkdir(parents=True, exist_ok=True)

    best_result['model'].save_model(str(output_dir / "catboost_model.cbm"))

    metrics = {
        'best_approach': best_approach,
        'cv_r2': best_result['cv_r2'],
        'cv_mape': best_result['cv_mape'],
        'cv_mae': best_result['cv_mae'],
        'n_features': best_result['n_features'],
        'best_params': best_result['best_params'],
        'all_results': {k: {kk: vv for kk, vv in v.items() if kk not in ['model', 'feature_names', 'fold_results']}
                       for k, v in results.items()},
        'timestamp': datetime.now().isoformat()
    }

    with open(output_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {output_dir}")

    # Feature importance for supply model
    if "With Supply" in results:
        logger.info("\n" + "=" * 60)
        logger.info("TOP 20 FEATURE IMPORTANCE (With Supply)")
        logger.info("=" * 60)

        supply_result = results["With Supply"]
        importance = supply_result['model'].get_feature_importance()
        feature_imp = sorted(
            zip(supply_result['feature_names'], importance),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (fname, imp) in enumerate(feature_imp[:20], 1):
            marker = " [SUPPLY]" if any(x in fname for x in ['supply', 'system', 'capacity', 'reserve']) else ""
            logger.info(f"  {i:2d}. {fname}: {imp:.2f}{marker}")


if __name__ == "__main__":
    main()
