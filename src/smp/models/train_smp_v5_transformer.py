"""
SMP Prediction Model v5.0 - Transformer + Weather Features
============================================================
Phase 2: 기상 데이터 활용 + Transformer 아키텍처

전략:
1. 기상 데이터 활용 (81% 커버리지)
2. 2022-2024 데이터 집중 (안정적 패턴)
3. Transformer 기반 아키텍처 (LSTM 대체)
4. Walk-forward 검증

Author: Claude Code
Date: 2024-12-25
"""

import os
import sys
import json
import logging
import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConfigV5:
    """Configuration for SMP v5.0 model"""
    smp_data_path: str = "data/smp/smp_5years_epsis.csv"
    weather_data_path: str = "data/processed/jeju_weather_hourly_merged.csv"
    output_dir: str = "models/smp_v5"

    # Model architecture - Transformer
    input_hours: int = 48
    output_hours: int = 24
    d_model: int = 64           # Transformer dimension
    n_heads: int = 4            # Attention heads
    n_encoder_layers: int = 3   # Transformer encoder layers
    dim_feedforward: int = 128  # FFN dimension
    dropout: float = 0.2

    # Training
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 0.0005
    patience: int = 30
    grad_clip: float = 1.0
    warmup_steps: int = 500

    # Walk-forward validation
    n_splits: int = 5
    train_window: int = 365 * 24  # 1 year
    test_window: int = 30 * 24    # 1 month

    # Data range
    train_start: str = '2022-01-01'
    train_end: str = '2024-12-31'


class WeatherEnhancedDataPipeline:
    """Data pipeline with weather features"""

    def __init__(self, config: ConfigV5):
        self.config = config
        self.feature_names: List[str] = []
        self.scaler = StandardScaler()
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def load_and_merge_data(self) -> pd.DataFrame:
        """Load SMP and weather data, merge them"""
        logger.info("Loading SMP data...")
        smp_df = pd.read_csv(self.config.smp_data_path)

        # Handle timestamps
        smp_df['timestamp'] = smp_df['timestamp'].str.replace(' 24:00', ' 00:00')
        smp_df['datetime'] = pd.to_datetime(smp_df['timestamp'], errors='coerce')
        mask = smp_df['datetime'].isna()
        if mask.any():
            smp_df.loc[mask, 'datetime'] = (
                pd.to_datetime(smp_df.loc[mask, 'date']) +
                pd.to_timedelta(smp_df.loc[mask, 'hour'].clip(upper=23), unit='h')
            )

        smp_df = smp_df.dropna(subset=['smp_mainland'])
        smp_df = smp_df[smp_df['smp_mainland'] > 0]  # Filter zero values

        # Date range filter
        smp_df = smp_df[
            (smp_df['datetime'] >= self.config.train_start) &
            (smp_df['datetime'] <= self.config.train_end)
        ]
        smp_df = smp_df.sort_values('datetime').reset_index(drop=True)
        logger.info(f"  SMP records: {len(smp_df):,}")

        # Load weather data
        logger.info("Loading weather data...")
        weather_df = pd.read_csv(self.config.weather_data_path)
        weather_df['datetime'] = pd.to_datetime(weather_df['일시'])

        weather_cols = {
            '기온': 'temperature',
            '풍속': 'wind_speed',
            '습도': 'humidity',
            '일사': 'solar_radiation',
            '강수량': 'precipitation',
            '해면기압': 'pressure'
        }

        weather_clean = weather_df[['datetime']].copy()
        for orig_col, new_col in weather_cols.items():
            if orig_col in weather_df.columns:
                weather_clean[new_col] = pd.to_numeric(weather_df[orig_col], errors='coerce')

        weather_clean = weather_clean.sort_values('datetime').reset_index(drop=True)
        logger.info(f"  Weather records: {len(weather_clean):,}")

        # Merge SMP with weather
        merged = pd.merge_asof(
            smp_df.sort_values('datetime'),
            weather_clean.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=2)
        )

        # Check coverage
        weather_coverage = merged['temperature'].notna().sum() / len(merged) * 100
        logger.info(f"  Weather coverage: {weather_coverage:.1f}%")

        # Fill missing weather with interpolation
        for col in weather_cols.values():
            if col in merged.columns:
                merged[col] = merged[col].interpolate(method='linear', limit_direction='both')
                merged[col] = merged[col].fillna(merged[col].median())

        merged = merged.sort_values('datetime').reset_index(drop=True)
        logger.info(f"  Merged records: {len(merged):,}")
        logger.info(f"  Date range: {merged['datetime'].min()} ~ {merged['datetime'].max()}")

        return merged

    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create feature set with weather"""
        logger.info("Creating features with weather...")
        self.feature_names = []
        features = []

        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)

        # ========== 1. Base SMP Features (4) ==========
        features.append(smp)
        features.append(df['smp_jeju'].values)
        features.append(df['smp_max'].values)
        features.append(df['smp_min'].values)
        self.feature_names.extend(['smp_mainland', 'smp_jeju', 'smp_max', 'smp_min'])

        # ========== 2. Time Features (7) ==========
        hour = df['hour'].values
        features.append(np.sin(2 * np.pi * hour / 24))
        features.append(np.cos(2 * np.pi * hour / 24))
        self.feature_names.extend(['hour_sin', 'hour_cos'])

        dow = df['datetime'].dt.dayofweek.values
        features.append(np.sin(2 * np.pi * dow / 7))
        features.append(np.cos(2 * np.pi * dow / 7))
        features.append((dow >= 5).astype(float))
        self.feature_names.extend(['dow_sin', 'dow_cos', 'is_weekend'])

        month = df['datetime'].dt.month.values
        features.append(np.sin(2 * np.pi * month / 12))
        features.append(np.cos(2 * np.pi * month / 12))
        self.feature_names.extend(['month_sin', 'month_cos'])

        # ========== 3. Season/Peak Features (5) ==========
        features.append(((month >= 6) & (month <= 8)).astype(float))
        features.append(((month == 12) | (month <= 2)).astype(float))
        features.append(((hour >= 9) & (hour <= 12)).astype(float))
        features.append(((hour >= 17) & (hour <= 21)).astype(float))
        features.append(((hour >= 23) | (hour <= 5)).astype(float))
        self.feature_names.extend(['is_summer', 'is_winter', 'peak_morning', 'peak_evening', 'off_peak'])

        # ========== 4. SMP Statistics (8) ==========
        for window in [6, 24, 168]:
            features.append(smp_series.rolling(window, min_periods=1).mean().values)
            self.feature_names.append(f'smp_ma{window}')

        features.append(smp_series.rolling(24, min_periods=1).std().fillna(0).values)
        self.feature_names.append('smp_std24')

        features.append(np.diff(smp, prepend=smp[0]))
        features.append(df['smp_max'].values - df['smp_min'].values)
        features.append(smp_series.pct_change().fillna(0).values)
        features.append(smp - smp_series.rolling(24, min_periods=1).mean().values)
        self.feature_names.extend(['smp_diff', 'smp_range', 'smp_pct_change', 'smp_momentum'])

        # ========== 5. Lag Features (4) ==========
        for lag in [24, 48, 168, 336]:
            features.append(smp_series.shift(lag).bfill().values)
            self.feature_names.append(f'smp_lag_{lag}h')

        # ========== 6. Weather Features (8) ==========
        if 'temperature' in df.columns:
            temp = df['temperature'].values
            features.append(temp)
            self.feature_names.append('temperature')

            # HDD/CDD
            features.append(np.maximum(18 - temp, 0))
            features.append(np.maximum(temp - 24, 0))
            self.feature_names.extend(['hdd', 'cdd'])

            # Temperature lag
            temp_series = pd.Series(temp)
            features.append(temp_series.shift(24).bfill().values)
            self.feature_names.append('temp_lag_24h')

        if 'wind_speed' in df.columns:
            features.append(df['wind_speed'].values)
            self.feature_names.append('wind_speed')

        if 'humidity' in df.columns:
            features.append(df['humidity'].values)
            self.feature_names.append('humidity')

        if 'solar_radiation' in df.columns:
            features.append(df['solar_radiation'].fillna(0).values)
            self.feature_names.append('solar_radiation')

        if 'pressure' in df.columns:
            features.append(df['pressure'].values)
            self.feature_names.append('pressure')

        feature_array = np.column_stack(features)
        target_array = smp

        logger.info(f"  Total features: {len(self.feature_names)}")
        logger.info(f"  Feature shape: {feature_array.shape}")

        return feature_array, target_array

    def normalize(self, features: np.ndarray, targets: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features and targets"""
        features = features.copy()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if fit:
            self.feature_clips = []
            for i in range(features.shape[1]):
                col = features[:, i]
                valid_col = col[np.isfinite(col)]
                if len(valid_col) > 0:
                    q1, q99 = np.percentile(valid_col, [1, 99])
                    clip_min = q1 - 3 * (q99 - q1)
                    clip_max = q99 + 3 * (q99 - q1)
                else:
                    clip_min, clip_max = -1e10, 1e10
                self.feature_clips.append((clip_min, clip_max))
                features[:, i] = np.clip(col, clip_min, clip_max)
        else:
            for i, (clip_min, clip_max) in enumerate(self.feature_clips):
                features[:, i] = np.clip(features[:, i], clip_min, clip_max)

        if fit:
            features_normalized = self.scaler.fit_transform(features)
            self.target_mean = float(np.mean(targets))
            self.target_std = float(np.std(targets))
        else:
            features_normalized = self.scaler.transform(features)

        targets_normalized = (targets - self.target_mean) / (self.target_std + 1e-8)

        return features_normalized, targets_normalized


class SMPDatasetV5(Dataset):
    """Dataset for SMP v5 model"""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 input_hours: int = 48, output_hours: int = 24):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.input_hours = input_hours
        self.output_hours = output_hours

    def __len__(self):
        return len(self.features) - self.input_hours - self.output_hours + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.input_hours]
        y = self.targets[idx + self.input_hours:idx + self.input_hours + self.output_hours]
        return x, y


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SMPTransformerV5(nn.Module):
    """Transformer-based SMP Prediction Model v5.0"""

    def __init__(self, input_size: int, d_model: int = 64, n_heads: int = 4,
                 n_encoder_layers: int = 3, dim_feedforward: int = 128,
                 dropout: float = 0.2, output_hours: int = 24):
        super().__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_hours = output_hours

        # Input projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=200, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Output heads
        self.point_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_hours)
        )

        self.q10_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_hours)
        )

        self.q50_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_hours)
        )

        self.q90_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_hours)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Use last timestep for prediction
        features = x[:, -1, :]

        result = {
            'point': self.point_head(features),
            'q10': self.q10_head(features),
            'q50': self.q50_head(features),
            'q90': self.q90_head(features)
        }

        return result


class CombinedLossV5(nn.Module):
    """Combined loss with Huber + Quantile"""

    def __init__(self, mse_weight: float = 0.5, quantile_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.quantile_weight = quantile_weight
        self.huber = nn.SmoothL1Loss()

    def quantile_loss(self, pred, target, q):
        error = target - pred
        return torch.mean(torch.max(q * error, (q - 1) * error))

    def forward(self, predictions, targets):
        point_loss = self.huber(predictions['point'], targets)

        q10_loss = self.quantile_loss(predictions['q10'], targets, 0.1)
        q50_loss = self.quantile_loss(predictions['q50'], targets, 0.5)
        q90_loss = self.quantile_loss(predictions['q90'], targets, 0.9)
        quantile_loss = (q10_loss + q50_loss + q90_loss) / 3

        total_loss = self.mse_weight * point_loss + self.quantile_weight * quantile_loss

        return total_loss, {'point_loss': point_loss.item(), 'quantile_loss': quantile_loss.item()}


class SMPTrainerV5:
    """Training engine for SMP v5 model"""

    def __init__(self, config: ConfigV5):
        self.config = config
        self.device = self._get_device()
        self.pipeline = WeatherEnhancedDataPipeline(config)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def prepare_data(self):
        """Prepare data"""
        logger.info("=" * 60)
        logger.info("Preparing data...")
        logger.info("=" * 60)

        merged_df = self.pipeline.load_and_merge_data()
        features, targets = self.pipeline.create_features(merged_df)

        return features, targets

    def create_split_loaders(self, features, targets, train_start, train_end, test_start, test_end):
        """Create dataloaders for a specific split"""
        self.pipeline.scaler = StandardScaler()

        train_features, train_targets = self.pipeline.normalize(
            features[train_start:train_end], targets[train_start:train_end], fit=True
        )
        test_features, test_targets = self.pipeline.normalize(
            features[test_start:test_end], targets[test_start:test_end], fit=False
        )

        # Validation from training
        val_size = int(len(train_features) * 0.1)
        train_dataset = SMPDatasetV5(
            train_features[:-val_size], train_targets[:-val_size],
            self.config.input_hours, self.config.output_hours
        )
        val_dataset = SMPDatasetV5(
            train_features[-val_size:], train_targets[-val_size:],
            self.config.input_hours, self.config.output_hours
        )
        test_dataset = SMPDatasetV5(
            test_features, test_targets,
            self.config.input_hours, self.config.output_hours
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train(self):
        """Train with walk-forward validation"""
        logger.info("=" * 60)
        logger.info("SMP v5.0 Transformer + Weather Training")
        logger.info("=" * 60)

        features, targets = self.prepare_data()
        n = len(features)

        train_window = self.config.train_window
        test_window = self.config.test_window
        n_splits = self.config.n_splits

        total_required = train_window + test_window * n_splits
        if n < total_required:
            n_splits = max(1, (n - train_window) // test_window)

        start_point = n - train_window - test_window * n_splits

        all_metrics = []
        best_model_state = None
        best_mape = float('inf')

        for split_idx in range(n_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"Walk-Forward Split {split_idx + 1}/{n_splits}")
            logger.info(f"{'='*60}")

            train_start = start_point + split_idx * test_window
            train_end = train_start + train_window
            test_start = train_end
            test_end = test_start + test_window

            logger.info(f"  Train: [{train_start:,} - {train_end:,}]")
            logger.info(f"  Test:  [{test_start:,} - {test_end:,}]")

            train_loader, val_loader, test_loader = self.create_split_loaders(
                features, targets, train_start, train_end, test_start, test_end
            )

            # Create model
            model = SMPTransformerV5(
                input_size=len(self.pipeline.feature_names),
                d_model=self.config.d_model,
                n_heads=self.config.n_heads,
                n_encoder_layers=self.config.n_encoder_layers,
                dim_feedforward=self.config.dim_feedforward,
                dropout=self.config.dropout,
                output_hours=self.config.output_hours
            ).to(self.device)

            if split_idx == 0:
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"  Model parameters: {n_params:,}")
                logger.info(f"  Device: {self.device}")

            # Training
            criterion = CombinedLossV5()
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

            split_best_val_loss = float('inf')
            split_best_state = None
            patience_counter = 0

            for epoch in range(self.config.epochs):
                model.train()
                train_losses = []
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss, _ = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    optimizer.step()
                    train_losses.append(loss.item())

                scheduler.step()

                model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        outputs = model(batch_x)
                        loss, _ = criterion(outputs, batch_y)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)

                if val_loss < split_best_val_loss:
                    split_best_val_loss = val_loss
                    split_best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

            if split_best_state:
                model.load_state_dict(split_best_state)

            # Evaluate
            split_metrics = self._evaluate(model, test_loader, criterion)
            all_metrics.append(split_metrics)
            logger.info(f"  Split {split_idx+1} MAPE: {split_metrics['mape']:.2f}%, R²: {split_metrics['r2']:.4f}")

            if split_metrics['mape'] < best_mape:
                best_mape = split_metrics['mape']
                best_model_state = model.state_dict().copy()

        # Average metrics
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}

        logger.info(f"\n{'='*60}")
        logger.info("Walk-Forward Results (Average)")
        logger.info(f"{'='*60}")
        logger.info(f"  MAE:  {avg_metrics['mae']:.2f} 원/kWh")
        logger.info(f"  RMSE: {avg_metrics['rmse']:.2f} 원/kWh")
        logger.info(f"  MAPE: {avg_metrics['mape']:.2f}%")
        logger.info(f"  R²:   {avg_metrics['r2']:.4f}")
        logger.info(f"  Coverage (80%): {avg_metrics['coverage_80']:.1f}%")

        model.load_state_dict(best_model_state)
        self._save_model(model, avg_metrics, all_metrics)

        return model, avg_metrics

    def _evaluate(self, model, test_loader, criterion):
        """Evaluate model"""
        model.eval()
        all_predictions = {'point': [], 'q10': [], 'q50': [], 'q90': []}
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = model(batch_x)
                for key in all_predictions:
                    all_predictions[key].append(outputs[key].cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())

        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key])
        all_targets = np.concatenate(all_targets)

        # Denormalize
        point_pred = all_predictions['point'] * self.pipeline.target_std + self.pipeline.target_mean
        q10_pred = all_predictions['q10'] * self.pipeline.target_std + self.pipeline.target_mean
        q90_pred = all_predictions['q90'] * self.pipeline.target_std + self.pipeline.target_mean
        targets = all_targets * self.pipeline.target_std + self.pipeline.target_mean

        # Metrics
        mae = np.mean(np.abs(point_pred - targets))
        rmse = np.sqrt(np.mean((point_pred - targets) ** 2))

        mask = np.abs(targets) > 10.0
        mape = np.mean(np.abs(point_pred[mask] - targets[mask]) / np.abs(targets[mask])) * 100 if mask.sum() > 0 else 0

        ss_res = np.sum((targets - point_pred) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        coverage = np.mean((targets >= q10_pred) & (targets <= q90_pred)) * 100

        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'coverage_80': float(coverage),
            'interval_width': float(np.mean(q90_pred - q10_pred))
        }

    def _save_model(self, model, avg_metrics, all_metrics):
        """Save model"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_kwargs': {
                'input_size': len(self.pipeline.feature_names),
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_encoder_layers': self.config.n_encoder_layers,
                'dim_feedforward': self.config.dim_feedforward,
                'dropout': self.config.dropout,
                'output_hours': self.config.output_hours
            },
            'metrics': avg_metrics,
            'target_mean': self.pipeline.target_mean,
            'target_std': self.pipeline.target_std,
            'feature_names': self.pipeline.feature_names,
            'version': '5.0.0'
        }
        torch.save(checkpoint, output_dir / 'smp_v5_model.pt')

        metrics_full = {
            'version': '5.0.0',
            **avg_metrics,
            'all_splits': all_metrics,
            'features': len(self.pipeline.feature_names),
            'feature_names': self.pipeline.feature_names,
            'architecture': 'Transformer',
            'improvements': [
                'Transformer encoder (replaces LSTM)',
                'Weather features integration',
                'Cosine annealing scheduler',
                'AdamW optimizer with weight decay'
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(output_dir / 'smp_v5_metrics.json', 'w') as f:
            json.dump(metrics_full, f, indent=2, ensure_ascii=False)

        logger.info(f"  Model saved: {output_dir / 'smp_v5_model.pt'}")


def main():
    logger.info("=" * 60)
    logger.info("SMP Prediction Model v5.0 - Transformer + Weather")
    logger.info("=" * 60)

    config = ConfigV5()
    trainer = SMPTrainerV5(config)
    model, metrics = trainer.train()

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info("Comparison:")
    logger.info(f"  v3.1: MAPE 7.83%, R² 0.736, Coverage 89.4%")
    logger.info(f"  v5.0: MAPE {metrics['mape']:.2f}%, R² {metrics['r2']:.3f}, Coverage {metrics['coverage_80']:.1f}%")

    return model, metrics


if __name__ == '__main__':
    main()
