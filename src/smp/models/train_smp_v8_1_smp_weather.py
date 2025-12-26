#!/usr/bin/env python3
"""
SMP Prediction Model v8.1 - SMP + Weather Only

Focuses on maximizing data usage:
1. Uses full SMP data (43k+ records)
2. Weather data (temperature, wind, humidity, solar)
3. Extensive SMP-based features
4. NO fuel prices (causes data loss)

Target: R² 0.9+
"""

import sys
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

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


class SMPDataset(Dataset):
    """Dataset for SMP prediction"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_length: int = 168
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


class BiLSTMAttention(nn.Module):
    """BiLSTM with Multi-Head Attention"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
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

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours)
        )

    def forward(self, x):
        x = self.feature_embed(x)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)
        pooled = attn_out.mean(dim=1)
        return self.fc(pooled)


def load_data() -> pd.DataFrame:
    """Load SMP and weather data"""
    logger.info("Loading data...")

    # Load SMP
    smp_path = PROJECT_ROOT / "data" / "smp" / "smp_5years_epsis.csv"
    smp_df = pd.read_csv(smp_path)

    # Fix timestamp
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

    logger.info(f"  SMP: {len(smp_df)} records ({smp_df['datetime'].min()} ~ {smp_df['datetime'].max()})")

    # Load weather
    weather_path = PROJECT_ROOT / "data" / "processed" / "jeju_weather_hourly_merged.csv"
    if weather_path.exists():
        weather_df = pd.read_csv(weather_path)

        if '일시' in weather_df.columns:
            weather_df['datetime'] = pd.to_datetime(weather_df['일시'])
        elif 'datetime' in weather_df.columns:
            weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])

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

        logger.info(f"  Weather: {len(weather_df)} records")

        # Merge
        smp_df = pd.merge(smp_df, weather_df, on='datetime', how='left')

        # Fill missing weather with interpolation
        for col in ['temperature', 'wind_speed', 'humidity', 'solar_radiation', 'pressure']:
            if col in smp_df.columns:
                smp_df[col] = smp_df[col].interpolate(method='linear').ffill().bfill()

    return smp_df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features from SMP and weather"""
    logger.info("Creating features...")

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Time flags
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_morning_peak'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 21)).astype(int)
    df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)

    # SMP lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
        df[f'smp_lag_{lag}'] = df['smp'].shift(lag)

    # SMP same hour patterns
    df['smp_same_hour_1d'] = df['smp'].shift(24)
    df['smp_same_hour_7d'] = df['smp'].shift(168)

    # SMP rolling statistics
    for w in [6, 12, 24, 48, 168]:
        df[f'smp_roll_mean_{w}'] = df['smp'].rolling(w, min_periods=1).mean()
        df[f'smp_roll_std_{w}'] = df['smp'].rolling(w, min_periods=1).std()
        df[f'smp_roll_min_{w}'] = df['smp'].rolling(w, min_periods=1).min()
        df[f'smp_roll_max_{w}'] = df['smp'].rolling(w, min_periods=1).max()

    # SMP change
    df['smp_diff_1'] = df['smp'].diff(1)
    df['smp_diff_24'] = df['smp'].diff(24)
    df['smp_pct_change_1'] = df['smp'].pct_change(1).fillna(0).clip(-1, 1)
    df['smp_pct_change_24'] = df['smp'].pct_change(24).fillna(0).clip(-1, 1)

    # SMP volatility
    df['smp_volatility_24'] = df['smp_diff_1'].rolling(24, min_periods=1).std()
    df['smp_volatility_168'] = df['smp_diff_1'].rolling(168, min_periods=1).std()

    # Weather features
    if 'temperature' in df.columns:
        df['temp_lag_1'] = df['temperature'].shift(1)
        df['temp_lag_24'] = df['temperature'].shift(24)
        df['temp_roll_mean_24'] = df['temperature'].rolling(24, min_periods=1).mean()
        df['hdd'] = np.maximum(0, 18 - df['temperature'])
        df['cdd'] = np.maximum(0, df['temperature'] - 26)
        df['temp_hour_interaction'] = df['temperature'] * df['hour_sin']

    if 'wind_speed' in df.columns:
        df['wind_lag_1'] = df['wind_speed'].shift(1)
        df['wind_roll_mean_24'] = df['wind_speed'].rolling(24, min_periods=1).mean()

    if 'solar_radiation' in df.columns:
        df['solar_roll_mean_12'] = df['solar_radiation'].rolling(12, min_periods=1).mean()

    # Drop first 168 rows (need for lag features)
    df = df.iloc[168:].reset_index(drop=True)

    # Fill any remaining NaN
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

    logger.info(f"  Features: {len(df.columns)}, Records: {len(df)}")

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get feature columns"""
    exclude = ['datetime', 'date', 'smp', 'timestamp']
    return [c for c in df.columns if c not in exclude]


def walk_forward_split(
    data_len: int,
    n_splits: int = 5,
    train_size: int = 8760,  # 1 year
    test_size: int = 720    # 1 month
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create walk-forward splits"""
    splits = []
    gap = 24

    usable = data_len - train_size - test_size - gap
    if usable <= 0:
        train_end = int(data_len * 0.8)
        return [(np.arange(0, train_end), np.arange(train_end + gap, data_len))]

    step = usable // n_splits

    for i in range(n_splits):
        train_start = i * step
        train_end = train_start + train_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, data_len)

        if test_end > data_len:
            break

        splits.append((np.arange(train_start, train_end), np.arange(test_start, test_end)))

    return splits


def train_and_evaluate(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    test_targets: np.ndarray,
    config: Dict,
    device: torch.device
) -> Tuple[float, float]:
    """Train model and return MAPE, R²"""

    train_ds = SMPDataset(train_features, train_targets, config['seq_length'])
    test_ds = SMPDataset(test_features, test_targets, config['seq_length'])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)

    model = BiLSTMAttention(
        input_size=train_features.shape[1],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        n_heads=config['n_heads'],
        dropout=config['dropout']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.HuberLoss(delta=10.0)

    best_loss = float('inf')
    patience = 0

    for epoch in range(config['max_epochs']):
        # Train
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(test_loader)

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
            if patience >= config['patience']:
                logger.info(f"  Early stopping at epoch {epoch + 1}")
                break

    # Load best and evaluate
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Metrics
    mask = targets != 0
    mape = np.mean(np.abs((targets[mask] - preds[mask]) / targets[mask])) * 100
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return mape, r2


def main():
    """Main function"""
    logger.info("=" * 60)
    logger.info("SMP v8.1: SMP + Weather Only (Full Data)")
    logger.info("=" * 60)

    device = get_device()
    logger.info(f"Device: {device}")

    # Load and prepare data
    df = load_data()
    df = create_features(df)

    feature_cols = get_feature_columns(df)
    features = df[feature_cols].values
    targets = df['smp'].values

    # Scale
    scaler = RobustScaler()
    features = scaler.fit_transform(features)

    logger.info(f"\nFeatures: {len(feature_cols)}, Records: {len(features)}")
    logger.info(f"Target mean: {targets.mean():.2f}, std: {targets.std():.2f}")

    # Config
    config = {
        'hidden_size': 256,
        'num_layers': 2,
        'n_heads': 8,
        'dropout': 0.2,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'seq_length': 168,
        'max_epochs': 100,
        'patience': 15
    }

    # Walk-forward validation
    splits = walk_forward_split(len(features))
    logger.info(f"\n{len(splits)} splits for walk-forward validation")

    all_mapes, all_r2s = [], []

    for i, (train_idx, test_idx) in enumerate(splits):
        logger.info(f"\nSplit {i + 1}/{len(splits)}")
        logger.info(f"  Train: [{train_idx[0]:,} - {train_idx[-1]:,}]")
        logger.info(f"  Test:  [{test_idx[0]:,} - {test_idx[-1]:,}]")

        mape, r2 = train_and_evaluate(
            features[train_idx], targets[train_idx],
            features[test_idx], targets[test_idx],
            config, device
        )

        all_mapes.append(mape)
        all_r2s.append(r2)
        logger.info(f"  MAPE: {mape:.2f}%, R²: {r2:.4f}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"Average MAPE: {np.mean(all_mapes):.2f}% (± {np.std(all_mapes):.2f}%)")
    logger.info(f"Average R²:   {np.mean(all_r2s):.4f} (± {np.std(all_r2s):.4f})")
    logger.info(f"Best:         MAPE {min(all_mapes):.2f}%, R² {max(all_r2s):.4f}")

    if np.mean(all_r2s) >= 0.9:
        logger.info("\n✅ TARGET ACHIEVED: R² >= 0.9!")
    else:
        logger.info(f"\n⏳ Gap: R² needs +{0.9 - np.mean(all_r2s):.4f}")

    return np.mean(all_mapes), np.mean(all_r2s)


if __name__ == "__main__":
    main()
