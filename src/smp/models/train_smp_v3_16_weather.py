#!/usr/bin/env python3
"""
SMP v3.16: CatBoost with Weather Features

Hypothesis: Weather affects SMP through:
- Temperature -> Electricity demand (heating/cooling)
- Wind speed -> Wind power generation
- Humidity -> Comfort levels, AC usage
- Seasonality correlation

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


class SMPDataPipelineWithWeather:
    """Data pipeline with weather integration."""

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

    def load_weather_data(self) -> pd.DataFrame:
        """Load Jeju hourly weather data from multiple years."""
        weather_dfs = []
        raw_path = PROJECT_ROOT / 'data' / 'raw'

        for year in range(2020, 2025):
            file_path = raw_path / f'jeju_temp_hourly_{year}.csv'
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    weather_dfs.append(df)
                    logger.info(f"    Loaded {year}: {len(df)} records")
                except Exception as e:
                    logger.warning(f"    Failed to load {year}: {e}")

        if not weather_dfs:
            logger.error("No weather data files found!")
            return pd.DataFrame()

        weather_df = pd.concat(weather_dfs, ignore_index=True)

        # Parse datetime (format: "2021-01-01 00:00")
        weather_df['datetime'] = pd.to_datetime(weather_df['일시'])

        # Rename columns to English
        weather_df = weather_df.rename(columns={
            '기온': 'temperature',
            '강수량': 'precipitation',
            '풍속': 'wind_speed',
            '풍향': 'wind_direction',
            '습도': 'humidity',
            '현지기압': 'pressure',
            '해면기압': 'sea_pressure',
            '일조': 'sunshine',
            '일사': 'solar_radiation',
            '전운량': 'cloud_cover',
            '지면온도': 'ground_temp'
        })

        # Select relevant columns
        cols_to_keep = ['datetime', 'temperature', 'precipitation', 'wind_speed',
                        'humidity', 'pressure', 'cloud_cover', 'ground_temp']
        cols_available = [c for c in cols_to_keep if c in weather_df.columns]
        weather_df = weather_df[cols_available].copy()

        # Convert to numeric (handle missing values)
        for col in cols_available[1:]:
            weather_df[col] = pd.to_numeric(weather_df[col], errors='coerce')

        weather_df = weather_df.sort_values('datetime').reset_index(drop=True)

        logger.info(f"  Total weather data: {len(weather_df)} records")

        return weather_df

    def merge_weather_to_smp(self, smp_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Merge weather data to SMP data by datetime."""
        merged = smp_df.merge(weather_df, on='datetime', how='left')

        # Forward fill missing weather data
        weather_cols = [c for c in weather_df.columns if c != 'datetime']
        for col in weather_cols:
            merged[col] = merged[col].ffill().bfill()

        n_weather = merged['temperature'].notna().sum()
        logger.info(f"  Merged data: {len(merged)} records, {n_weather} with weather")

        return merged

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features including weather features."""
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

        # ===== WEATHER FEATURES (NEW!) =====

        # Temperature features (shift by 1 hour for safety)
        if 'temperature' in df.columns:
            temp = pd.Series(df['temperature'].values)

            # Current value (shifted by 1 to avoid leakage)
            features['temp'] = temp.shift(1).values

            # Lag features
            for lag in [1, 2, 3, 6, 12, 24]:
                features[f'temp_lag{lag}'] = temp.shift(lag + 1).values

            # Rolling stats
            for window in [6, 12, 24]:
                features[f'temp_ma{window}'] = temp.rolling(window, min_periods=1).mean().shift(1).values
                features[f'temp_std{window}'] = temp.rolling(window, min_periods=1).std().shift(1).fillna(0).values

            # Heating/Cooling Degree Hours (reference: 18C)
            features['heating_degree'] = np.maximum(18 - temp.shift(1), 0).values
            features['cooling_degree'] = np.maximum(temp.shift(1) - 26, 0).values

            # Temperature change
            features['temp_diff1'] = temp.diff(1).shift(1).fillna(0).values
            features['temp_diff24'] = temp.diff(24).shift(1).fillna(0).values

            # Extreme temperature flags
            features['temp_extreme_cold'] = (temp.shift(1) < 5).astype(float).values
            features['temp_extreme_hot'] = (temp.shift(1) > 30).astype(float).values

        # Humidity features
        if 'humidity' in df.columns:
            humidity = pd.Series(df['humidity'].values)
            features['humidity'] = humidity.shift(1).values
            features['humidity_ma24'] = humidity.rolling(24, min_periods=1).mean().shift(1).values

            # Discomfort index (approximate)
            if 'temperature' in df.columns:
                temp = pd.Series(df['temperature'].values)
                # THI = 0.81*T + 0.01*RH*(0.99*T - 14.3) + 46.3
                thi = 0.81 * temp + 0.01 * humidity * (0.99 * temp - 14.3) + 46.3
                features['discomfort_index'] = thi.shift(1).values

        # Wind speed features
        if 'wind_speed' in df.columns:
            wind = pd.Series(df['wind_speed'].values)
            features['wind_speed'] = wind.shift(1).values
            features['wind_ma6'] = wind.rolling(6, min_periods=1).mean().shift(1).values
            features['wind_ma24'] = wind.rolling(24, min_periods=1).mean().shift(1).values

            # High wind flag (affects renewable generation)
            features['high_wind'] = (wind.shift(1) > 10).astype(float).values

        # Cloud cover features
        if 'cloud_cover' in df.columns:
            cloud = pd.Series(df['cloud_cover'].values)
            features['cloud_cover'] = cloud.shift(1).values
            features['cloud_ma24'] = cloud.rolling(24, min_periods=1).mean().shift(1).values

        # Pressure features
        if 'pressure' in df.columns:
            pressure = pd.Series(df['pressure'].values)
            features['pressure'] = pressure.shift(1).values
            features['pressure_diff24'] = pressure.diff(24).shift(1).fillna(0).values

        # ===== Interaction features =====
        if 'temperature' in df.columns:
            temp = pd.Series(df['temperature'].values)

            # Temperature x Hour interaction
            features['temp_x_hour_sin'] = (temp.shift(1) * features['hour_sin']).values
            features['temp_x_hour_cos'] = (temp.shift(1) * features['hour_cos']).values

            # Temperature x Weekend
            features['temp_x_weekend'] = (temp.shift(1) * features['is_weekend']).values

            # Temperature x Season
            features['temp_x_summer'] = (temp.shift(1) * features['is_summer']).values
            features['temp_x_winter'] = (temp.shift(1) * features['is_winter']).values

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
    logger.info("SMP v3.16: CatBoost with Weather Features")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    pipeline = SMPDataPipelineWithWeather()
    smp_df = pipeline.load_smp_data()

    logger.info("\nLoading weather data...")
    weather_df = pipeline.load_weather_data()

    # Merge weather to SMP
    logger.info("\nMerging weather data...")
    merged_df = pipeline.merge_weather_to_smp(smp_df, weather_df)

    # Create features
    logger.info("\nCreating features...")
    feature_df = pipeline.create_features(merged_df)

    feature_names = pipeline.feature_names
    logger.info(f"Total features: {len(feature_names)}")

    # Show weather-related features
    weather_features = [f for f in feature_names if any(x in f for x in
        ['temp', 'humidity', 'wind', 'cloud', 'pressure', 'heating', 'cooling', 'discomfort', 'high_wind'])]
    logger.info(f"Weather features added: {len(weather_features)}")

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
    output_dir = PROJECT_ROOT / "models" / "smp_v3_16_weather"
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
        'n_weather_features': len(weather_features),
        'feature_names': feature_names,
        'weather_features': weather_features,
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
    logger.info(f"| v3.14 +Fuel | 12.56% | 0.534 | 101 |")
    logger.info(f"| v3.16 +Weather | {test_mape:.2f}% | {test_r2:.3f} | {len(feature_names)} |")

    if test_r2 > 0.834:
        logger.info(f"\n✅ IMPROVED over v3.12: R² +{test_r2 - 0.834:.4f}")
    else:
        logger.info(f"\n⚠️ No improvement over v3.12 CatBoost CV")

    if test_r2 >= 0.9:
        logger.info("🎉 REACHED R² 0.9+ TARGET!")

    # Feature importance
    logger.info("\n" + "=" * 60)
    logger.info("TOP 25 FEATURE IMPORTANCE")
    logger.info("=" * 60)
    importance = best_model.get_feature_importance()
    feature_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
    for i, (fname, imp) in enumerate(feature_imp[:25], 1):
        marker = " [WEATHER]" if any(x in fname for x in
            ['temp', 'humidity', 'wind', 'cloud', 'pressure', 'heating', 'cooling', 'discomfort']) else ""
        logger.info(f"  {i:2d}. {fname}: {imp:.2f}{marker}")


if __name__ == "__main__":
    main()
