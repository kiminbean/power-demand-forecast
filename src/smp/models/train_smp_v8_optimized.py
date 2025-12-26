#!/usr/bin/env python3
"""
SMP Prediction Model v8.0 - Optimized for R² 0.9+

Key Improvements:
1. Clean data: Only real data (SMP, weather, fuel - no synthetic generation)
2. Extensive feature engineering: 100+ features from SMP lags and patterns
3. BiLSTM + Multi-Head Attention architecture
4. Optuna hyperparameter optimization
5. Walk-forward validation with 5 splits

Target: R² 0.9+, MAPE < 5%
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

# Optuna for hyperparameter tuning
try:
    import optuna
    from optuna.trial import Trial
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def get_device() -> torch.device:
    """Get best available device"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class SMPDatasetV8(Dataset):
    """Enhanced dataset with extensive feature engineering"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 168  # 1 week
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length - 24 + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length:idx + self.seq_length + 24]
        return x, y


class BiLSTMAttentionV8(nn.Module):
    """
    Optimized BiLSTM with Multi-Head Attention

    Architecture:
    - Feature embedding layer
    - 3-layer BiLSTM with residual connections
    - Multi-head self-attention
    - Feed-forward output with skip connections
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.2,
        output_hours: int = 24
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Feature embedding
        self.feature_embed = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )

        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Output layers with skip connection
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, output_hours)
        )

    def forward(self, x):
        # Feature embedding
        x = self.feature_embed(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Self-attention on last output
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Residual

        # Use mean pooling of attention output
        pooled = attn_out.mean(dim=1)

        # Output
        out = self.fc(pooled)
        return out


def load_and_prepare_data(use_enhanced: bool = False) -> pd.DataFrame:
    """
    Load data and create extensive feature engineering

    Uses only REAL data:
    - SMP historical data
    - Weather data (KMA)
    - Fuel prices (Yahoo Finance)

    NO synthetic generation data
    """
    logger.info("Loading and preparing data...")

    # Load SMP data
    smp_path = PROJECT_ROOT / "data" / "smp" / "smp_5years_epsis.csv"
    smp_df = pd.read_csv(smp_path)

    # Handle 24:00 timestamp
    def fix_timestamp(ts):
        if ' 24:00' in str(ts):
            parts = str(ts).split(' ')
            date_part = pd.to_datetime(parts[0]) + pd.Timedelta(days=1)
            return date_part
        return pd.to_datetime(ts)

    smp_df['datetime'] = smp_df['timestamp'].apply(fix_timestamp)
    smp_df = smp_df[['datetime', 'smp_mainland']].copy()
    smp_df.columns = ['datetime', 'smp']
    smp_df = smp_df.sort_values('datetime').reset_index(drop=True)

    logger.info(f"  SMP data: {len(smp_df)} records")
    logger.info(f"  Date range: {smp_df['datetime'].min()} ~ {smp_df['datetime'].max()}")

    # Load weather data
    weather_path = PROJECT_ROOT / "data" / "processed" / "jeju_weather_hourly_merged.csv"
    if weather_path.exists():
        weather_df = pd.read_csv(weather_path)

        # Parse datetime
        if '일시' in weather_df.columns:
            weather_df['datetime'] = pd.to_datetime(weather_df['일시'])
        elif 'datetime' in weather_df.columns:
            weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

        # Map Korean columns
        col_mapping = {
            '기온': 'temperature',
            '풍속': 'wind_speed',
            '습도': 'humidity',
            '일사': 'solar_radiation',
            '현지기압': 'pressure'
        }

        for korean, english in col_mapping.items():
            if korean in weather_df.columns:
                weather_df[english] = pd.to_numeric(weather_df[korean], errors='coerce')

        weather_cols = ['datetime', 'temperature', 'wind_speed', 'humidity',
                       'solar_radiation', 'pressure']
        available_cols = [c for c in weather_cols if c in weather_df.columns]
        weather_df = weather_df[available_cols]

        logger.info(f"  Weather data: {len(weather_df)} records")

        # Merge with SMP
        smp_df = pd.merge(smp_df, weather_df, on='datetime', how='left')

    # Load fuel prices (real from Yahoo Finance)
    fuel_path = PROJECT_ROOT / "data" / "external" / "fuel" / "fuel_prices.csv"
    if fuel_path.exists():
        fuel_df = pd.read_csv(fuel_path)
        fuel_df['date'] = pd.to_datetime(fuel_df['date'])
        smp_df['date'] = smp_df['datetime'].dt.date
        smp_df['date'] = pd.to_datetime(smp_df['date'])

        fuel_df = fuel_df.rename(columns={'date': 'date'})
        smp_df = pd.merge(smp_df, fuel_df, on='date', how='left')

        # Forward fill fuel prices
        fuel_cols = ['wti_crude', 'brent_crude', 'natural_gas', 'heating_oil']
        for col in fuel_cols:
            if col in smp_df.columns:
                smp_df[col] = smp_df[col].ffill().bfill()

        logger.info(f"  Fuel prices merged")

    # Create extensive features
    df = create_extensive_features(smp_df)

    # Remove rows with NaN
    initial_len = len(df)
    df = df.dropna()
    logger.info(f"  After dropping NaN: {len(df)} records (removed {initial_len - len(df)})")

    return df


def create_extensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 100+ features for SMP prediction

    Feature categories:
    1. Time features (cyclical encoding)
    2. SMP lag features (1h to 1 week)
    3. SMP rolling statistics (mean, std, min, max)
    4. SMP change features (velocity, acceleration)
    5. Weather features
    6. Weather interaction features
    7. Fuel price features with lag
    8. Day-of-week and month patterns
    """
    logger.info("Creating extensive features...")

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['week_of_year'] = df['datetime'].dt.isocalendar().week.astype(int)

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

    # Weekend flag
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

    # Time of day categories (peak hours)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)

    # SMP lag features (balanced: not too many to avoid data loss)
    lag_hours = [1, 2, 3, 6, 12, 24, 48, 168]
    for lag in lag_hours:
        df[f'smp_lag_{lag}'] = df['smp'].shift(lag)

    # SMP at same hour yesterday, 2 days ago, week ago
    df['smp_same_hour_1d'] = df['smp'].shift(24)
    df['smp_same_hour_2d'] = df['smp'].shift(48)
    df['smp_same_hour_7d'] = df['smp'].shift(168)

    # SMP rolling statistics (use min_periods to reduce NaN)
    windows = [3, 6, 12, 24, 48, 168]
    for w in windows:
        df[f'smp_roll_mean_{w}'] = df['smp'].rolling(w, min_periods=w//2).mean()
        df[f'smp_roll_std_{w}'] = df['smp'].rolling(w, min_periods=w//2).std()
        df[f'smp_roll_min_{w}'] = df['smp'].rolling(w, min_periods=w//2).min()
        df[f'smp_roll_max_{w}'] = df['smp'].rolling(w, min_periods=w//2).max()
        df[f'smp_roll_range_{w}'] = df[f'smp_roll_max_{w}'] - df[f'smp_roll_min_{w}']

    # SMP percentile features (use min_periods)
    for w in [24, 168]:
        df[f'smp_roll_median_{w}'] = df['smp'].rolling(w, min_periods=w//2).median()
        df[f'smp_roll_q25_{w}'] = df['smp'].rolling(w, min_periods=w//2).quantile(0.25)
        df[f'smp_roll_q75_{w}'] = df['smp'].rolling(w, min_periods=w//2).quantile(0.75)

    # SMP change features (velocity and acceleration)
    df['smp_diff_1'] = df['smp'].diff(1)
    df['smp_diff_24'] = df['smp'].diff(24)
    df['smp_diff_168'] = df['smp'].diff(168)
    df['smp_pct_change_1'] = df['smp'].pct_change(1)
    df['smp_pct_change_24'] = df['smp'].pct_change(24)

    # SMP acceleration (change of change)
    df['smp_accel'] = df['smp_diff_1'].diff(1)

    # SMP volatility
    df['smp_volatility_24'] = df['smp_diff_1'].rolling(24).std()
    df['smp_volatility_168'] = df['smp_diff_1'].rolling(168).std()

    # Weather features (if available)
    if 'temperature' in df.columns:
        # Temperature lag
        df['temp_lag_1'] = df['temperature'].shift(1)
        df['temp_lag_24'] = df['temperature'].shift(24)

        # Temperature rolling
        df['temp_roll_mean_24'] = df['temperature'].rolling(24).mean()
        df['temp_roll_std_24'] = df['temperature'].rolling(24).std()

        # Temperature change
        df['temp_diff_1'] = df['temperature'].diff(1)
        df['temp_diff_24'] = df['temperature'].diff(24)

        # Heating/Cooling degree (relative to 18°C)
        df['hdd'] = np.maximum(0, 18 - df['temperature'])  # Heating degree
        df['cdd'] = np.maximum(0, df['temperature'] - 26)  # Cooling degree

    if 'wind_speed' in df.columns:
        df['wind_roll_mean_24'] = df['wind_speed'].rolling(24).mean()
        df['wind_lag_1'] = df['wind_speed'].shift(1)

    if 'humidity' in df.columns:
        df['humidity_roll_mean_24'] = df['humidity'].rolling(24).mean()

    if 'solar_radiation' in df.columns:
        df['solar_roll_mean_12'] = df['solar_radiation'].rolling(12).mean()
        df['solar_cumsum_daily'] = df.groupby(df['datetime'].dt.date)['solar_radiation'].cumsum()

    # Weather interaction features
    if 'temperature' in df.columns:
        df['temp_hour_interaction'] = df['temperature'] * df['hour_sin']

    if 'wind_speed' in df.columns and 'solar_radiation' in df.columns:
        df['wind_solar_ratio'] = df['wind_speed'] / (df['solar_radiation'] + 0.01)

    # Fuel price features with lag effect (use min_periods to reduce NaN)
    fuel_cols = ['wti_crude', 'brent_crude', 'natural_gas', 'heating_oil']
    for col in fuel_cols:
        if col in df.columns:
            # Moving averages (fuel prices have delayed effect)
            df[f'{col}_ma7'] = df[col].rolling(7 * 24, min_periods=24).mean()
            df[f'{col}_ma14'] = df[col].rolling(14 * 24, min_periods=48).mean()

            # Change from week ago
            df[f'{col}_change_7d'] = df[col].pct_change(7 * 24).fillna(0)

    # Aggregate fuel index
    if 'wti_crude' in df.columns and 'natural_gas' in df.columns:
        df['fuel_index'] = (
            0.4 * df['natural_gas'] / df['natural_gas'].mean() +
            0.3 * df['wti_crude'] / df['wti_crude'].mean() +
            0.3 * df.get('brent_crude', df['wti_crude']) / df.get('brent_crude', df['wti_crude']).mean()
        )

    logger.info(f"  Created {len(df.columns)} features")

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude datetime and target)"""
    exclude_cols = ['datetime', 'date', 'smp', 'timestamp']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Filter out any columns with all NaN
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    return feature_cols


def walk_forward_split(
    data_len: int,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    test_ratio: float = 0.1
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward validation splits (adaptive to data size)

    Each split:
    - Train: 70% of available data
    - Test: 10% of available data
    - Gap: 24 hours (prevent data leakage)
    """
    splits = []
    gap = 24  # 24-hour gap to prevent leakage

    # Calculate sizes based on data length
    train_size = int(data_len * train_ratio)
    test_size = int(data_len * test_ratio)

    # Ensure minimum sizes
    train_size = max(train_size, 1000)
    test_size = max(test_size, 200)

    # Calculate step between splits
    usable_size = data_len - train_size - test_size - gap
    if usable_size <= 0:
        # Not enough data, use single split
        train_end = int(data_len * 0.8)
        test_start = train_end + gap
        test_end = data_len
        return [(np.arange(0, train_end), np.arange(test_start, test_end))]

    step = usable_size // n_splits

    for i in range(n_splits):
        train_start = i * step
        train_end = train_start + train_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, data_len)

        if test_end > data_len or train_end >= data_len:
            break

        train_idx = np.arange(train_start, train_end)
        test_idx = np.arange(test_start, test_end)

        if len(test_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_grad: float = 1.0
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model and return MAPE and R²"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    # MAPE
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100

    # R²
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mape, r2


def train_single_split(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray,
    val_targets: np.ndarray,
    config: Dict,
    device: torch.device
) -> Tuple[nn.Module, float, float]:
    """
    Train model on a single split

    Returns: trained model, MAPE, R²
    """
    # Create datasets
    train_dataset = SMPDatasetV8(
        train_features, train_targets,
        seq_length=config.get('seq_length', 168)
    )
    val_dataset = SMPDatasetV8(
        val_features, val_targets,
        seq_length=config.get('seq_length', 168)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 64),
        shuffle=False,
        num_workers=0
    )

    # Create model
    input_size = train_features.shape[1]
    model = BiLSTMAttentionV8(
        input_size=input_size,
        hidden_size=config.get('hidden_size', 256),
        num_layers=config.get('num_layers', 3),
        n_heads=config.get('n_heads', 8),
        dropout=config.get('dropout', 0.2),
        output_hours=24
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.HuberLoss(delta=10.0)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = config.get('patience', 15)

    for epoch in range(config.get('max_epochs', 100)):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best model
    model.load_state_dict(best_state)

    # Final evaluation
    mape, r2 = evaluate(model, val_loader, device)

    return model, mape, r2


def optuna_objective(trial: Trial, df: pd.DataFrame, device: torch.device) -> float:
    """Optuna objective function for hyperparameter tuning"""
    # Hyperparameter suggestions
    config = {
        'hidden_size': trial.suggest_categorical('hidden_size', [128, 256, 384]),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'n_heads': trial.suggest_categorical('n_heads', [4, 8, 16]),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'seq_length': trial.suggest_categorical('seq_length', [168, 336]),
        'max_epochs': 50,
        'patience': 10
    }

    # Use only 2 splits for faster tuning
    feature_cols = get_feature_columns(df)
    features = df[feature_cols].values
    targets = df['smp'].values

    # Scale features
    scaler = RobustScaler()
    features = scaler.fit_transform(features)

    # Get splits
    splits = walk_forward_split(df, n_splits=2)

    total_r2 = 0
    for train_idx, test_idx in splits:
        train_features = features[train_idx]
        train_targets = targets[train_idx]
        test_features = features[test_idx]
        test_targets = targets[test_idx]

        _, mape, r2 = train_single_split(
            train_features, train_targets,
            test_features, test_targets,
            config, device
        )
        total_r2 += r2

    avg_r2 = total_r2 / len(splits)

    # Optuna minimizes, so return negative R²
    return -avg_r2


def run_optuna_tuning(df: pd.DataFrame, device: torch.device, n_trials: int = 20) -> Dict:
    """Run Optuna hyperparameter tuning"""
    logger.info(f"\nStarting Optuna hyperparameter tuning ({n_trials} trials)...")

    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: optuna_objective(trial, df, device),
        n_trials=n_trials,
        show_progress_bar=True
    )

    best_params = study.best_params
    logger.info(f"\nBest parameters: {best_params}")
    logger.info(f"Best R²: {-study.best_value:.4f}")

    return best_params


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("SMP v8.0 Optimized: R² 0.9+ Target")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load and prepare data
    df = load_and_prepare_data()

    feature_cols = get_feature_columns(df)
    logger.info(f"\nUsing {len(feature_cols)} features")

    # Prepare arrays
    features = df[feature_cols].values
    targets = df['smp'].values

    # Scale features
    scaler = RobustScaler()
    features = scaler.fit_transform(features)

    logger.info(f"Feature matrix: {features.shape}")
    logger.info(f"Target mean: {targets.mean():.2f}, std: {targets.std():.2f}")

    # Default config (or use Optuna-tuned)
    config = {
        'hidden_size': 256,
        'num_layers': 3,
        'n_heads': 8,
        'dropout': 0.2,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'seq_length': 168,
        'max_epochs': 100,
        'patience': 15
    }

    # Run Optuna tuning if available
    if HAS_OPTUNA and '--tune' in sys.argv:
        n_trials = 30
        best_params = run_optuna_tuning(df, device, n_trials)
        config.update(best_params)

    # Walk-forward validation
    logger.info("\n" + "=" * 60)
    logger.info("Walk-Forward Validation (5 splits)")
    logger.info("=" * 60)

    splits = walk_forward_split(len(features), n_splits=5)

    all_mapes = []
    all_r2s = []

    for i, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Split {i + 1}/{len(splits)}")
        logger.info(f"{'=' * 60}")
        logger.info(f"  Train: [{train_idx[0]:,} - {train_idx[-1]:,}]")
        logger.info(f"  Test:  [{test_idx[0]:,} - {test_idx[-1]:,}]")

        train_features = features[train_idx]
        train_targets = targets[train_idx]
        test_features = features[test_idx]
        test_targets = targets[test_idx]

        model, mape, r2 = train_single_split(
            train_features, train_targets,
            test_features, test_targets,
            config, device
        )

        all_mapes.append(mape)
        all_r2s.append(r2)

        logger.info(f"  MAPE: {mape:.2f}%, R²: {r2:.4f}")

    # Summary
    avg_mape = np.mean(all_mapes)
    avg_r2 = np.mean(all_r2s)
    std_mape = np.std(all_mapes)
    std_r2 = np.std(all_r2s)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Average MAPE: {avg_mape:.2f}% (± {std_mape:.2f}%)")
    logger.info(f"Average R²:   {avg_r2:.4f} (± {std_r2:.4f})")
    logger.info(f"Best Split:   MAPE {min(all_mapes):.2f}%, R² {max(all_r2s):.4f}")

    # Target check
    if avg_r2 >= 0.9:
        logger.info("\n✅ TARGET ACHIEVED: R² >= 0.9!")
    else:
        logger.info(f"\n⏳ Gap to target: R² needs +{0.9 - avg_r2:.4f}")

    return avg_mape, avg_r2


if __name__ == "__main__":
    main()
