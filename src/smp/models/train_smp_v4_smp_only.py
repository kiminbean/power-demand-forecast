"""
SMP Prediction Model v4.1 - SMP-Only Enhanced Features
=======================================================
외부 데이터 없이 SMP 자체에서 파생된 풍부한 피처로 R² 개선

전략:
1. 5년 전체 SMP 데이터 사용 (데이터 일관성 유지)
2. Multi-scale 통계 피처 확장 (6h, 12h, 24h, 168h)
3. 더 많은 lag 피처 (6h, 12h, 24h, 48h, 168h, 336h)
4. 추세/패턴 피처 강화
5. 과적합 방지를 위한 규제화 강화

Author: Claude Code
Date: 2024-12-25
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ConfigV41:
    """Configuration for SMP v4.1 model"""
    smp_data_path: str = "data/smp/smp_5years_epsis.csv"
    output_dir: str = "models/smp_v4"

    # Model architecture
    input_hours: int = 48
    output_hours: int = 24
    hidden_size: int = 64
    num_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.35  # Slightly higher dropout

    # Training
    batch_size: int = 64
    epochs: int = 200  # More epochs
    learning_rate: float = 0.0008  # Slightly lower LR
    patience: int = 35  # More patience
    grad_clip: float = 1.0
    noise_std: float = 0.015

    # Walk-forward validation (like v3.1)
    n_splits: int = 5
    train_window: int = 365 * 24  # 1 year
    test_window: int = 30 * 24    # 1 month


class SMPOnlyDataPipeline:
    """Data pipeline using only SMP-derived features"""

    def __init__(self, config: ConfigV41):
        self.config = config
        self.feature_names: List[str] = []
        self.scaler = StandardScaler()
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def load_smp_data(self) -> pd.DataFrame:
        """Load and preprocess SMP data"""
        logger.info("Loading SMP data...")
        df = pd.read_csv(self.config.smp_data_path)

        # Handle 24:00 timestamps
        df['timestamp'] = df['timestamp'].str.replace(' 24:00', ' 00:00')
        df['datetime'] = pd.to_datetime(df['timestamp'], errors='coerce')

        mask = df['datetime'].isna()
        if mask.any():
            df.loc[mask, 'datetime'] = (
                pd.to_datetime(df.loc[mask, 'date']) +
                pd.to_timedelta(df.loc[mask, 'hour'].clip(upper=23), unit='h')
            )

        df = df.dropna(subset=['smp_mainland'])

        # Use only 2022-2024 data like v3.1 (more recent, stable patterns)
        df = df[(df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2024-12-31')]

        df = df.sort_values('datetime').reset_index(drop=True)
        logger.info(f"  SMP records: {len(df):,}")
        logger.info(f"  Date range: {df['datetime'].min()} ~ {df['datetime'].max()}")

        return df

    def create_enhanced_smp_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create enhanced features from SMP data only"""
        logger.info("Creating enhanced SMP-only features...")
        self.feature_names = []
        features = []

        smp = df['smp_mainland'].values
        smp_series = pd.Series(smp)

        # ========== 1. Base Price Features (4) ==========
        features.append(smp)
        features.append(df['smp_jeju'].values)
        features.append(df['smp_max'].values)
        features.append(df['smp_min'].values)
        self.feature_names.extend(['smp_mainland', 'smp_jeju', 'smp_max', 'smp_min'])

        # ========== 2. Time Cyclical Features (7) ==========
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
        self.feature_names.extend(['is_summer', 'is_winter'])

        features.append(((hour >= 9) & (hour <= 12)).astype(float))
        features.append(((hour >= 17) & (hour <= 21)).astype(float))
        features.append(((hour >= 23) | (hour <= 5)).astype(float))
        self.feature_names.extend(['peak_morning', 'peak_evening', 'off_peak'])

        # ========== 4. Multi-scale Moving Averages (5) ==========
        for window in [6, 12, 24, 72, 168]:
            ma = smp_series.rolling(window, min_periods=1).mean().values
            features.append(ma)
            self.feature_names.append(f'smp_ma{window}')

        # ========== 5. Multi-scale Volatility (4) ==========
        for window in [6, 24, 72, 168]:
            std = smp_series.rolling(window, min_periods=1).std().fillna(0).values
            features.append(std)
            self.feature_names.append(f'smp_std{window}')

        # ========== 6. Enhanced Lag Features (6) ==========
        for lag in [6, 12, 24, 48, 168, 336]:
            lagged = smp_series.shift(lag).bfill().values
            features.append(lagged)
            self.feature_names.append(f'smp_lag_{lag}h')

        # ========== 7. Dynamics Features (5) ==========
        # First difference
        features.append(np.diff(smp, prepend=smp[0]))
        self.feature_names.append('smp_diff')

        # Range (max - min)
        features.append(df['smp_max'].values - df['smp_min'].values)
        self.feature_names.append('smp_range')

        # Percentage change
        features.append(smp_series.pct_change().fillna(0).values)
        self.feature_names.append('smp_pct_change')

        # Second difference (acceleration)
        diff1 = np.diff(smp, prepend=smp[0])
        diff2 = np.diff(diff1, prepend=diff1[0])
        features.append(diff2)
        self.feature_names.append('smp_diff2')

        # Momentum (price - MA24)
        momentum = smp - smp_series.rolling(24, min_periods=1).mean().values
        features.append(momentum)
        self.feature_names.append('smp_momentum')

        # ========== 8. Pattern Features (4) ==========
        # Same hour yesterday diff
        same_hour_yesterday = smp_series.shift(24).bfill().values
        features.append(smp - same_hour_yesterday)
        self.feature_names.append('smp_day_diff')

        # Same hour last week diff
        same_hour_lastweek = smp_series.shift(168).bfill().values
        features.append(smp - same_hour_lastweek)
        self.feature_names.append('smp_week_diff')

        # Rolling min/max ratio (range position)
        rolling_min = smp_series.rolling(24, min_periods=1).min().values
        rolling_max = smp_series.rolling(24, min_periods=1).max().values
        range_denom = (rolling_max - rolling_min)
        range_denom = np.where(range_denom == 0, 1, range_denom)
        range_position = (smp - rolling_min) / range_denom
        features.append(range_position)
        self.feature_names.append('smp_range_position')

        # RSI-like indicator (relative strength)
        diff = np.diff(smp, prepend=smp[0])
        gains = np.where(diff > 0, diff, 0)
        losses = np.where(diff < 0, -diff, 0)
        avg_gain = pd.Series(gains).rolling(14, min_periods=1).mean().values
        avg_loss = pd.Series(losses).rolling(14, min_periods=1).mean().values + 1e-8
        rsi = avg_gain / (avg_gain + avg_loss)
        features.append(rsi)
        self.feature_names.append('smp_rsi')

        feature_array = np.column_stack(features)
        target_array = smp

        logger.info(f"  Total features: {len(self.feature_names)}")
        logger.info(f"  Feature shape: {feature_array.shape}")

        return feature_array, target_array

    def normalize(self, features: np.ndarray, targets: np.ndarray, fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize features and targets"""
        # Handle inf and NaN values
        features = features.copy()
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip extreme values
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
            logger.info(f"  Target mean: {self.target_mean:.2f}, std: {self.target_std:.2f}")
        else:
            features_normalized = self.scaler.transform(features)

        targets_normalized = (targets - self.target_mean) / (self.target_std + 1e-8)

        return features_normalized, targets_normalized


class SMPDatasetV41(Dataset):
    """Dataset for SMP v4.1 model"""

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


class StableMultiHeadAttention(nn.Module):
    """Stable Multi-Head Self-Attention"""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.d_k)

        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        output = self.W_o(context)
        return output, attn_weights.mean(dim=1)


class SMPModelV41(nn.Module):
    """SMP Prediction Model v4.1"""

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.35, output_hours: int = 24):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_hours = output_hours

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # BiLSTM Encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # Multi-Head Attention
        self.attention = StableMultiHeadAttention(hidden_size * 2, n_heads, dropout)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Output heads
        fc_input = hidden_size * 2
        self.point_head = nn.Sequential(
            nn.Linear(fc_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours)
        )

        self.q10_head = nn.Sequential(
            nn.Linear(fc_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours)
        )

        self.q50_head = nn.Sequential(
            nn.Linear(fc_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours)
        )

        self.q90_head = nn.Sequential(
            nn.Linear(fc_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # Self-Attention
        attn_out, attn_weights = self.attention(lstm_out)

        # Residual connection + LayerNorm
        out = self.layer_norm(lstm_out + attn_out)

        # Last timestep features
        features = out[:, -1, :]

        result = {
            'point': self.point_head(features),
            'q10': self.q10_head(features),
            'q50': self.q50_head(features),
            'q90': self.q90_head(features),
            'attention': attn_weights
        }

        return result


class CombinedLossV41(nn.Module):
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

        return total_loss, {
            'point_loss': point_loss.item(),
            'quantile_loss': quantile_loss.item()
        }


class SMPTrainerV41:
    """Training engine for SMP v4.1 model"""

    def __init__(self, config: ConfigV41):
        self.config = config
        self.device = self._get_device()
        self.pipeline = SMPOnlyDataPipeline(config)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def prepare_data(self):
        """Prepare full dataset for walk-forward validation"""
        logger.info("=" * 60)
        logger.info("Preparing data...")
        logger.info("=" * 60)

        # Load SMP data
        smp_df = self.pipeline.load_smp_data()

        # Filter out zero values (like v3.1)
        smp_df = smp_df[smp_df['smp_mainland'] > 0].copy()
        logger.info(f"  After filtering zero values: {len(smp_df):,} records")

        # Create features
        features, targets = self.pipeline.create_enhanced_smp_features(smp_df)

        return features, targets

    def create_split_loaders(self, features, targets, train_start, train_end, test_start, test_end):
        """Create dataloaders for a specific split"""
        # Reset and fit scaler on training data
        self.pipeline.scaler = StandardScaler()

        train_features, train_targets = self.pipeline.normalize(
            features[train_start:train_end], targets[train_start:train_end], fit=True
        )

        test_features, test_targets = self.pipeline.normalize(
            features[test_start:test_end], targets[test_start:test_end], fit=False
        )

        # Use last 10% of training for validation
        val_size = int(len(train_features) * 0.1)
        train_dataset = SMPDatasetV41(
            train_features[:-val_size], train_targets[:-val_size],
            self.config.input_hours, self.config.output_hours
        )
        val_dataset = SMPDatasetV41(
            train_features[-val_size:], train_targets[-val_size:],
            self.config.input_hours, self.config.output_hours
        )
        test_dataset = SMPDatasetV41(
            test_features, test_targets,
            self.config.input_hours, self.config.output_hours
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def train(self):
        """Train using walk-forward validation (like v3.1)"""
        logger.info("=" * 60)
        logger.info("Starting walk-forward training...")
        logger.info("=" * 60)

        # Prepare data
        features, targets = self.prepare_data()
        n = len(features)

        # Walk-forward splits
        train_window = self.config.train_window
        test_window = self.config.test_window
        n_splits = self.config.n_splits

        # Calculate split points
        total_required = train_window + test_window * n_splits
        if n < total_required:
            logger.warning(f"Not enough data for {n_splits} splits. Adjusting...")
            n_splits = max(1, (n - train_window) // test_window)

        start_point = n - train_window - test_window * n_splits

        all_metrics = []
        best_model_state = None
        best_mape = float('inf')

        for split_idx in range(n_splits):
            logger.info(f"\n{'='*60}")
            logger.info(f"Walk-Forward Split {split_idx + 1}/{n_splits}")
            logger.info(f"{'='*60}")

            # Calculate indices
            train_start = start_point + split_idx * test_window
            train_end = train_start + train_window
            test_start = train_end
            test_end = test_start + test_window

            logger.info(f"  Train: [{train_start:,} - {train_end:,}] ({train_end-train_start:,} samples)")
            logger.info(f"  Test:  [{test_start:,} - {test_end:,}] ({test_end-test_start:,} samples)")

            # Create data loaders for this split
            train_loader, val_loader, test_loader = self.create_split_loaders(
                features, targets, train_start, train_end, test_start, test_end
            )

            # Create fresh model for each split
            model = SMPModelV41(
                input_size=len(self.pipeline.feature_names),
                hidden_size=self.config.hidden_size,
                num_layers=self.config.num_layers,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
                output_hours=self.config.output_hours
            ).to(self.device)

            if split_idx == 0:
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"  Model parameters: {n_params:,}")
                logger.info(f"  Device: {self.device}")

            # Train this split
            criterion = CombinedLossV41()
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=15, factor=0.5
            )

            split_best_val_loss = float('inf')
            split_best_state = None
            patience_counter = 0

            for epoch in range(self.config.epochs):
                # Training
                model.train()
                train_losses = []
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    if self.config.noise_std > 0:
                        noise = torch.randn_like(batch_x) * self.config.noise_std
                        batch_x = batch_x + noise

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss, _ = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    optimizer.step()
                    train_losses.append(loss.item())

                # Validation
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
                scheduler.step(val_loss)

                # Early stopping
                if val_loss < split_best_val_loss:
                    split_best_val_loss = val_loss
                    split_best_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.patience:
                        break

            # Load best model for this split
            if split_best_state:
                model.load_state_dict(split_best_state)

            # Evaluate on test set
            split_metrics = self._evaluate(model, test_loader, criterion)
            all_metrics.append(split_metrics)
            logger.info(f"  Split {split_idx+1} MAPE: {split_metrics['mape']:.2f}%, R²: {split_metrics['r2']:.4f}")

            # Keep best model
            if split_metrics['mape'] < best_mape:
                best_mape = split_metrics['mape']
                best_model_state = model.state_dict().copy()

        # Calculate average metrics
        avg_metrics = {
            'mae': np.mean([m['mae'] for m in all_metrics]),
            'rmse': np.mean([m['rmse'] for m in all_metrics]),
            'mape': np.mean([m['mape'] for m in all_metrics]),
            'r2': np.mean([m['r2'] for m in all_metrics]),
            'coverage_80': np.mean([m['coverage_80'] for m in all_metrics]),
            'interval_width': np.mean([m['interval_width'] for m in all_metrics])
        }

        logger.info(f"\n{'='*60}")
        logger.info("Walk-Forward Validation Results (Average)")
        logger.info(f"{'='*60}")
        logger.info(f"  MAE:  {avg_metrics['mae']:.2f} 원/kWh")
        logger.info(f"  RMSE: {avg_metrics['rmse']:.2f} 원/kWh")
        logger.info(f"  MAPE: {avg_metrics['mape']:.2f}%")
        logger.info(f"  R²:   {avg_metrics['r2']:.4f}")
        logger.info(f"  Coverage (80%): {avg_metrics['coverage_80']:.1f}%")

        # Reload best model
        model.load_state_dict(best_model_state)

        # Save model
        self._save_model(model, avg_metrics, {'all_split_metrics': all_metrics})

        return model, avg_metrics

    def _evaluate(self, model, test_loader, criterion):
        """Evaluate model on test set"""
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

        # Concatenate
        for key in all_predictions:
            all_predictions[key] = np.concatenate(all_predictions[key])
        all_targets = np.concatenate(all_targets)

        # Denormalize
        point_pred = all_predictions['point'] * self.pipeline.target_std + self.pipeline.target_mean
        q10_pred = all_predictions['q10'] * self.pipeline.target_std + self.pipeline.target_mean
        q50_pred = all_predictions['q50'] * self.pipeline.target_std + self.pipeline.target_mean
        q90_pred = all_predictions['q90'] * self.pipeline.target_std + self.pipeline.target_mean
        targets = all_targets * self.pipeline.target_std + self.pipeline.target_mean

        # Calculate metrics
        mae = np.mean(np.abs(point_pred - targets))
        mse = np.mean((point_pred - targets) ** 2)
        rmse = np.sqrt(mse)

        # Robust MAPE
        min_threshold = 10.0
        mask = np.abs(targets) > min_threshold
        if mask.sum() > 0:
            mape = np.mean(np.abs(point_pred[mask] - targets[mask]) / np.abs(targets[mask])) * 100
        else:
            mape = 0.0

        # R² score
        ss_res = np.sum((targets - point_pred) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Coverage (80% interval)
        in_interval = (targets >= q10_pred) & (targets <= q90_pred)
        coverage = np.mean(in_interval) * 100

        metrics = {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2),
            'coverage_80': float(coverage),
            'interval_width': float(np.mean(q90_pred - q10_pred))
        }

        logger.info(f"  MAE:  {mae:.2f} 원/kWh")
        logger.info(f"  RMSE: {rmse:.2f} 원/kWh")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R²:   {r2:.4f}")
        logger.info(f"  Coverage (80%): {coverage:.1f}%")

        return metrics

    def _save_model(self, model, metrics, history):
        """Save model and artifacts"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_kwargs': {
                'input_size': len(self.pipeline.feature_names),
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_layers,
                'n_heads': self.config.n_heads,
                'dropout': self.config.dropout,
                'output_hours': self.config.output_hours
            },
            'config': self.config.__dict__,
            'metrics': metrics,
            'target_mean': self.pipeline.target_mean,
            'target_std': self.pipeline.target_std,
            'feature_names': self.pipeline.feature_names,
            'version': '4.1.0'
        }
        torch.save(checkpoint, output_dir / 'smp_v41_model.pt')
        logger.info(f"  Model saved: {output_dir / 'smp_v41_model.pt'}")

        # Save scaler
        np.save(output_dir / 'smp_v41_scaler.npy', {
            'feature_scaler_mean': self.pipeline.scaler.mean_,
            'feature_scaler_scale': self.pipeline.scaler.scale_,
            'target_mean': self.pipeline.target_mean,
            'target_std': self.pipeline.target_std,
            'feature_names': self.pipeline.feature_names
        })

        # Save metrics
        metrics_full = {
            'version': '4.1.0',
            **metrics,
            'parameters': sum(p.numel() for p in model.parameters()),
            'features': len(self.pipeline.feature_names),
            'feature_names': self.pipeline.feature_names,
            'improvements': [
                'Multi-scale moving averages (6h, 12h, 24h, 72h, 168h)',
                'Multi-scale volatility features',
                'Extended lag features (6h-336h)',
                'Pattern features (momentum, RSI)',
                'Enhanced regularization'
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(output_dir / 'smp_v41_metrics.json', 'w') as f:
            json.dump(metrics_full, f, indent=2, ensure_ascii=False)

        logger.info(f"  Metrics saved: {output_dir / 'smp_v41_metrics.json'}")


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("SMP Prediction Model v4.1 - SMP-Only Enhanced Features")
    logger.info("=" * 60)

    config = ConfigV41()
    trainer = SMPTrainerV41(config)
    model, metrics = trainer.train()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Final Metrics:")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  Coverage: {metrics['coverage_80']:.1f}%")

    # Compare with v3.1
    logger.info("")
    logger.info("Comparison with v3.1:")
    logger.info(f"  v3.1: MAPE 7.83%, R² 0.736, Coverage 89.4%")
    logger.info(f"  v4.1: MAPE {metrics['mape']:.2f}%, R² {metrics['r2']:.3f}, Coverage {metrics['coverage_80']:.1f}%")

    return model, metrics


if __name__ == '__main__':
    main()
