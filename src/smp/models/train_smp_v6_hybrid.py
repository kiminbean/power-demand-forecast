#!/usr/bin/env python3
"""
SMP v6.0 Hybrid: BiLSTM+Attention + Weather Features

Phase 3 Strategy:
- Use proven BiLSTM+Attention architecture (from v3.1)
- Add weather features (from v5.0)
- Enhanced SMP features (from v4.1)
- Walk-forward validation

Target: R² > 0.80, MAPE < 7.5%
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler

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


class BiLSTMAttention(nn.Module):
    """
    BiLSTM with Multi-Head Self-Attention
    Based on v3.1 proven architecture with improvements
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.2,
        output_hours: int = 24
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Feed-forward
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours * 3)  # q10, q50, q90
        )

        self.output_hours = output_hours

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # BiLSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual + LayerNorm
        x = self.layer_norm(lstm_out + attn_out)

        # Use last timestep
        x = x[:, -1, :]  # (batch, hidden*2)

        # Output
        out = self.fc(x)  # (batch, output_hours * 3)

        return out.view(-1, self.output_hours, 3)


class SMPDataset(Dataset):
    """Dataset for SMP prediction with sliding window"""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int = 168,
        pred_len: int = 24
    ):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.targets[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y


def load_smp_data() -> pd.DataFrame:
    """Load SMP data"""
    smp_path = PROJECT_ROOT / "data" / "smp" / "smp_5years_epsis.csv"

    df = pd.read_csv(smp_path)

    # Handle 24:00 timestamps by converting to next day 00:00
    def fix_timestamp(ts):
        if ' 24:00' in str(ts):
            parts = str(ts).split(' ')
            date_part = pd.to_datetime(parts[0]) + pd.Timedelta(days=1)
            return date_part
        return pd.to_datetime(ts)

    df['datetime'] = df['timestamp'].apply(fix_timestamp)
    df['land'] = df['smp_mainland']  # Use mainland SMP

    # Filter 2022-2024 (like v3.1)
    df = df[(df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2024-12-31')]

    # Filter zero values
    df = df[df['land'] > 0]

    return df.sort_values('datetime').reset_index(drop=True)


def load_weather_data() -> pd.DataFrame:
    """Load processed weather data"""
    weather_path = PROJECT_ROOT / "data" / "processed" / "jeju_weather_hourly_merged.csv"

    if not weather_path.exists():
        logger.warning(f"Weather file not found: {weather_path}")
        return pd.DataFrame()

    df = pd.read_csv(weather_path)

    # Parse datetime (Korean column name: 일시)
    if '일시' in df.columns:
        df['datetime'] = pd.to_datetime(df['일시'])
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
    elif 'timestamp' in df.columns:
        df['datetime'] = pd.to_datetime(df['timestamp'])

    # Map Korean column names to English
    col_mapping = {
        '기온': 'temperature',
        '풍속': 'wind_speed',
        '습도': 'humidity',
        '일사': 'solar_radiation',
        '현지기압': 'pressure'
    }

    for korean, english in col_mapping.items():
        if korean in df.columns:
            df[english] = pd.to_numeric(df[korean], errors='coerce')

    return df


def create_features(smp_df: pd.DataFrame, weather_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Create combined features: SMP + Weather + Time
    """
    df = smp_df.copy()

    # Merge weather data
    if len(weather_df) > 0:
        df = df.merge(weather_df, on='datetime', how='left')

        # Fill missing weather values
        weather_cols = ['temperature', 'wind_speed', 'humidity', 'solar_radiation', 'pressure']
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].ffill().bfill().fillna(df[col].median() if df[col].notna().any() else 0)

    # === SMP Features ===
    smp = df['land'].values

    # Lag features
    for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]:
        df[f'smp_lag_{lag}'] = df['land'].shift(lag)

    # Rolling statistics
    for window in [6, 12, 24, 48, 168]:
        df[f'smp_roll_mean_{window}'] = df['land'].rolling(window=window, min_periods=1).mean()
        df[f'smp_roll_std_{window}'] = df['land'].rolling(window=window, min_periods=1).std()

    # Daily patterns
    df['smp_daily_mean'] = df.groupby(df['datetime'].dt.date)['land'].transform('mean')
    df['smp_daily_std'] = df.groupby(df['datetime'].dt.date)['land'].transform('std')
    df['smp_diff_from_daily_mean'] = df['land'] - df['smp_daily_mean']

    # Hour-of-day average (historical pattern)
    df['hour'] = df['datetime'].dt.hour
    hourly_avg = df.groupby('hour')['land'].transform('mean')
    df['smp_hourly_avg'] = hourly_avg
    df['smp_diff_from_hourly'] = df['land'] - hourly_avg

    # === Time Features ===
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    df['month'] = df['datetime'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['is_weekend'] = (df['dayofweek'] >= 5).astype(float)
    df['is_business_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 18)).astype(float)

    # === Weather Features ===
    weather_features = []
    if 'temperature' in df.columns:
        df['temp_squared'] = df['temperature'] ** 2  # Non-linear temp effect
        weather_features.extend(['temperature', 'temp_squared'])
    if 'wind_speed' in df.columns:
        weather_features.append('wind_speed')
    if 'humidity' in df.columns:
        weather_features.append('humidity')
    if 'solar_radiation' in df.columns:
        df['solar_radiation'] = df['solar_radiation'].fillna(0)
        weather_features.append('solar_radiation')
    if 'pressure' in df.columns:
        weather_features.append('pressure')

    # === Compile Features ===
    feature_cols = (
        [f'smp_lag_{lag}' for lag in [1, 2, 3, 6, 12, 24, 48, 72, 168]] +
        [f'smp_roll_mean_{w}' for w in [6, 12, 24, 48, 168]] +
        [f'smp_roll_std_{w}' for w in [6, 12, 24, 48, 168]] +
        ['smp_daily_mean', 'smp_daily_std', 'smp_diff_from_daily_mean',
         'smp_hourly_avg', 'smp_diff_from_hourly'] +
        ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
         'is_weekend', 'is_business_hour'] +
        weather_features
    )

    feature_cols = [c for c in feature_cols if c in df.columns]

    # Handle missing values
    df = df.ffill().bfill()

    features = df[feature_cols].values.astype(np.float32)
    targets = df['land'].values.astype(np.float32)

    # Handle inf/nan
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme values
    for i in range(features.shape[1]):
        p1, p99 = np.percentile(features[:, i], [1, 99])
        features[:, i] = np.clip(features[:, i], p1, p99)

    # Store target stats for denormalization
    target_mean = targets.mean()
    target_std = targets.std()

    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Weather features: {len(weather_features)}")
    logger.info(f"  Samples: {len(features)}")

    return features, targets, target_mean, target_std


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list = [0.1, 0.5, 0.9]) -> torch.Tensor:
    """Quantile loss function
    pred: (batch, output_hours, 3) - predictions for q10, q50, q90
    target: (batch, output_hours) - actual values
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = target - pred[..., i]  # (batch, output_hours)
        losses.append(torch.max(q * errors, (q - 1) * errors))
    return torch.mean(torch.stack(losses))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """Train one epoch"""
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)  # (batch, 24, 3)
        loss = quantile_loss(pred, y)  # y is (batch, 24)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float
) -> dict:
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Use median (q50)
            pred_q50 = pred[..., 1]

            all_preds.append(pred_q50.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    targets = np.concatenate(all_targets).flatten()

    # Denormalize
    preds_denorm = preds * target_std + target_mean
    targets_denorm = targets * target_std + target_mean

    # Metrics
    mae = np.mean(np.abs(preds_denorm - targets_denorm))
    rmse = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))

    # Robust MAPE (exclude small values)
    mask = np.abs(targets_denorm) > 10
    if mask.sum() > 0:
        mape = np.mean(np.abs(preds_denorm[mask] - targets_denorm[mask]) / np.abs(targets_denorm[mask])) * 100
    else:
        mape = 0

    # R²
    ss_res = np.sum((targets_denorm - preds_denorm) ** 2)
    ss_tot = np.sum((targets_denorm - targets_denorm.mean()) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def walk_forward_validation(
    features: np.ndarray,
    targets: np.ndarray,
    target_mean: float,
    target_std: float,
    n_splits: int = 5,
    train_size: int = 8760,
    test_size: int = 720,
    seq_len: int = 168,
    pred_len: int = 24
) -> dict:
    """
    Walk-forward validation
    """
    device = get_device()
    results = []

    total_samples = len(features)
    min_required = train_size + test_size + seq_len + pred_len

    if total_samples < min_required:
        raise ValueError(f"Not enough data: {total_samples} < {min_required}")

    # Calculate step between splits
    available = total_samples - min_required
    step = available // (n_splits - 1) if n_splits > 1 else 0

    best_model = None
    best_r2 = -float('inf')

    for split_idx in range(n_splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-Forward Split {split_idx + 1}/{n_splits}")
        logger.info(f"{'='*60}")

        # Calculate indices
        start_idx = split_idx * step
        train_end = start_idx + train_size
        test_end = train_end + test_size

        logger.info(f"  Train: [{start_idx:,} - {train_end:,}]")
        logger.info(f"  Test:  [{train_end:,} - {test_end:,}]")

        # Split data
        train_features = features[start_idx:train_end]
        train_targets = targets[start_idx:train_end]
        test_features = features[train_end:test_end]
        test_targets = targets[train_end:test_end]

        # Normalize using training statistics
        scaler = RobustScaler()
        train_features_norm = scaler.fit_transform(train_features)
        test_features_norm = scaler.transform(test_features)

        # Target normalization
        train_target_mean = train_targets.mean()
        train_target_std = train_targets.std()
        train_targets_norm = (train_targets - train_target_mean) / (train_target_std + 1e-8)
        test_targets_norm = (test_targets - train_target_mean) / (train_target_std + 1e-8)

        # Create datasets
        train_dataset = SMPDataset(train_features_norm, train_targets_norm, seq_len, pred_len)
        test_dataset = SMPDataset(test_features_norm, test_targets_norm, seq_len, pred_len)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Create model
        input_size = train_features.shape[1]
        model = BiLSTMAttention(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            n_heads=4,
            dropout=0.2,
            output_hours=pred_len
        ).to(device)

        # Optimizer with cosine annealing
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        # Training
        n_epochs = 100
        patience = 15
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()

            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)  # (batch, 24, 3)
                    loss = quantile_loss(pred, y)  # y is (batch, 24)
                    val_loss += loss.item()
            val_loss /= len(test_loader)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model weights
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Load best weights
        model.load_state_dict(best_weights)

        # Evaluate
        metrics = evaluate(model, test_loader, device, train_target_mean, train_target_std)

        logger.info(f"  MAPE: {metrics['mape']:.2f}%, R²: {metrics['r2']:.4f}")

        results.append(metrics)

        # Keep track of best model
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model = model

    # Average results
    avg_results = {
        'mae': np.mean([r['mae'] for r in results]),
        'rmse': np.mean([r['rmse'] for r in results]),
        'mape': np.mean([r['mape'] for r in results]),
        'r2': np.mean([r['r2'] for r in results])
    }

    return avg_results, best_model


def main():
    logger.info("=" * 60)
    logger.info("SMP v6.0 Hybrid: BiLSTM+Attention + Weather")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    smp_df = load_smp_data()
    logger.info(f"  SMP records: {len(smp_df):,}")

    weather_df = load_weather_data()
    logger.info(f"  Weather records: {len(weather_df):,}")

    # Create features
    logger.info("\nCreating features...")
    features, targets, target_mean, target_std = create_features(smp_df, weather_df)

    logger.info(f"  Date range: {smp_df['datetime'].min()} ~ {smp_df['datetime'].max()}")

    # Walk-forward validation
    logger.info("\n" + "=" * 60)
    logger.info("Starting Walk-Forward Validation...")
    logger.info("=" * 60)

    results, best_model = walk_forward_validation(
        features=features,
        targets=targets,
        target_mean=target_mean,
        target_std=target_std,
        n_splits=5,
        train_size=8760,  # 1 year
        test_size=720     # 1 month
    )

    # Print final results
    logger.info("\n" + "=" * 60)
    logger.info("Walk-Forward Results (Average)")
    logger.info("=" * 60)
    logger.info(f"  MAE:  {results['mae']:.2f} won/kWh")
    logger.info(f"  RMSE: {results['rmse']:.2f} won/kWh")
    logger.info(f"  MAPE: {results['mape']:.2f}%")
    logger.info(f"  R²:   {results['r2']:.4f}")

    # Save model
    model_dir = PROJECT_ROOT / "models" / "smp_v6"
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_model.state_dict(), model_dir / "smp_v6_model.pt")
    logger.info(f"\n  Model saved: {model_dir / 'smp_v6_model.pt'}")

    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("Comparison with Previous Versions")
    logger.info("=" * 60)
    logger.info(f"  v3.1 (BiLSTM+Attn):     MAPE 7.83%, R² 0.736")
    logger.info(f"  v5.0 (Transformer):      MAPE 8.25%, R² 0.537")
    logger.info(f"  v6.0 (BiLSTM+Weather):  MAPE {results['mape']:.2f}%, R² {results['r2']:.3f}")

    return results


if __name__ == "__main__":
    main()
