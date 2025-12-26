"""
SMP Prediction Model v4.0 Enhanced
==================================
Phase 1: 수요/공급/기상 피처 추가로 R² 0.9 목표

주요 개선사항:
1. 외부 피처 추가: 전력 수요, 공급 능력, 예비율
2. 기상 피처 추가: 기온, 풍속, 일사량, 습도
3. Multi-scale 통계 피처 확장
4. 개선된 모델 아키텍처

Author: Claude Code
Date: 2024-12-25
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
class ConfigV4Enhanced:
    """Enhanced configuration for SMP v4 model"""
    # Data paths
    smp_data_path: str = "data/smp/smp_5years_epsis.csv"
    demand_data_path: str = "data/jeju_extract/계통수요.csv"
    supply_data_path: str = "data/jeju_extract/공급능력.csv"
    weather_data_path: str = "data/processed/jeju_weather_hourly_merged.csv"
    power_data_path: str = "data/raw/jeju_hourly_power_2013_2024.csv"
    output_dir: str = "models/smp_v4"

    # Model architecture (reduced complexity for smaller dataset)
    input_hours: int = 48  # Same as v3.1
    output_hours: int = 24
    hidden_size: int = 64  # Same as v3.1
    num_layers: int = 2  # Same as v3.1
    n_heads: int = 4  # Same as v3.1
    dropout: float = 0.4  # Increased for regularization

    # Training
    batch_size: int = 64
    epochs: int = 150
    learning_rate: float = 0.001
    patience: int = 25
    grad_clip: float = 1.0
    noise_std: float = 0.02

    # Data split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


class EnhancedDataPipeline:
    """Enhanced data pipeline with external features"""

    def __init__(self, config: ConfigV4Enhanced):
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

        # Fill NaT with date + hour
        mask = df['datetime'].isna()
        if mask.any():
            df.loc[mask, 'datetime'] = (
                pd.to_datetime(df.loc[mask, 'date']) +
                pd.to_timedelta(df.loc[mask, 'hour'].clip(upper=23), unit='h')
            )

        df = df.sort_values('datetime').reset_index(drop=True)
        logger.info(f"  SMP records: {len(df):,}")
        return df

    def load_demand_supply_data(self) -> pd.DataFrame:
        """Load and reshape demand/supply data"""
        logger.info("Loading demand/supply data...")

        # Load demand data
        demand_df = pd.read_csv(self.config.demand_data_path, encoding='cp949')
        supply_df = pd.read_csv(self.config.supply_data_path, encoding='cp949')

        # Reshape from wide to long format
        def reshape_hourly_data(df, value_name):
            # Get date column (first column)
            date_col = df.columns[0]
            hour_cols = df.columns[1:25]  # 1시 ~ 24시

            records = []
            for _, row in df.iterrows():
                date = pd.to_datetime(row[date_col])
                for i, col in enumerate(hour_cols, 1):
                    dt = date + pd.Timedelta(hours=i)
                    val = row[col]
                    if pd.notna(val):
                        records.append({'datetime': dt, value_name: float(val)})

            return pd.DataFrame(records)

        demand_long = reshape_hourly_data(demand_df, 'demand_mw')
        supply_long = reshape_hourly_data(supply_df, 'supply_mw')

        # Merge demand and supply
        merged = pd.merge(demand_long, supply_long, on='datetime', how='outer')
        merged = merged.sort_values('datetime').reset_index(drop=True)

        # Calculate reserve rate
        merged['reserve_mw'] = merged['supply_mw'] - merged['demand_mw']
        merged['reserve_rate'] = merged['reserve_mw'] / merged['demand_mw']

        logger.info(f"  Demand/Supply records: {len(merged):,}")
        logger.info(f"  Date range: {merged['datetime'].min()} ~ {merged['datetime'].max()}")

        return merged

    def load_weather_data(self) -> pd.DataFrame:
        """Load weather data"""
        logger.info("Loading weather data...")
        df = pd.read_csv(self.config.weather_data_path)
        df['datetime'] = pd.to_datetime(df['일시'])

        # Select relevant columns
        weather_cols = {
            '기온': 'temperature',
            '풍속': 'wind_speed',
            '습도': 'humidity',
            '일사': 'solar_radiation',
            '강수량': 'precipitation',
            '해면기압': 'pressure'
        }

        df_clean = df[['datetime']].copy()
        for orig_col, new_col in weather_cols.items():
            if orig_col in df.columns:
                df_clean[new_col] = pd.to_numeric(df[orig_col], errors='coerce')

        df_clean = df_clean.sort_values('datetime').reset_index(drop=True)
        logger.info(f"  Weather records: {len(df_clean):,}")

        return df_clean

    def load_power_data(self) -> pd.DataFrame:
        """Load power trading data"""
        logger.info("Loading power trading data...")
        df = pd.read_csv(self.config.power_data_path)

        df['datetime'] = pd.to_datetime(df['거래일자']) + pd.to_timedelta(df['시간'], unit='h')
        df = df.rename(columns={'전력거래량(MWh)': 'power_trading_mwh'})
        df = df[['datetime', 'power_trading_mwh']].copy()
        df = df.sort_values('datetime').reset_index(drop=True)

        logger.info(f"  Power trading records: {len(df):,}")
        return df

    def merge_all_data(self) -> pd.DataFrame:
        """Merge all data sources"""
        logger.info("Merging all data sources...")

        # Load all data
        smp_df = self.load_smp_data()
        demand_supply_df = self.load_demand_supply_data()
        weather_df = self.load_weather_data()
        power_df = self.load_power_data()

        # Start with SMP data
        merged = smp_df[['datetime', 'hour', 'smp_mainland', 'smp_jeju', 'smp_max', 'smp_min']].copy()

        # Merge demand/supply (use nearest match for missing data)
        merged = pd.merge_asof(
            merged.sort_values('datetime'),
            demand_supply_df.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=1)
        )

        # Merge weather
        merged = pd.merge_asof(
            merged.sort_values('datetime'),
            weather_df.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=1)
        )

        # Merge power trading
        merged = pd.merge_asof(
            merged.sort_values('datetime'),
            power_df.sort_values('datetime'),
            on='datetime',
            direction='nearest',
            tolerance=pd.Timedelta(hours=1)
        )

        # Drop rows with missing critical values
        merged = merged.dropna(subset=['smp_mainland'])

        # Add indicator for whether external data is available
        merged['has_demand_data'] = merged['demand_mw'].notna().astype(float)
        merged['has_weather_data'] = merged['temperature'].notna().astype(float)
        logger.info(f"  Records with demand data: {merged['has_demand_data'].sum():,.0f} / {len(merged):,}")
        logger.info(f"  Records with weather data: {merged['has_weather_data'].sum():,.0f} / {len(merged):,}")

        # Fill remaining NaN with 0 (indicator will tell model when data is real)
        for col in merged.columns:
            if col not in ['datetime', 'hour']:
                if merged[col].isna().any():
                    merged[col] = merged[col].fillna(0)

        merged = merged.sort_values('datetime').reset_index(drop=True)
        logger.info(f"  Merged records: {len(merged):,}")
        logger.info(f"  Columns: {list(merged.columns)}")

        return merged

    def create_enhanced_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create enhanced feature set"""
        logger.info("Creating enhanced features...")
        self.feature_names = []
        features = []

        smp = df['smp_mainland'].values

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

        # ========== 4. SMP Statistical Features (8) ==========
        smp_series = pd.Series(smp)

        # Multi-scale moving averages
        features.append(smp_series.rolling(6, min_periods=1).mean().values)
        features.append(smp_series.rolling(24, min_periods=1).mean().values)
        features.append(smp_series.rolling(168, min_periods=1).mean().values)
        self.feature_names.extend(['smp_ma6', 'smp_ma24', 'smp_ma168'])

        # Volatility
        features.append(smp_series.rolling(24, min_periods=1).std().fillna(0).values)
        features.append(smp_series.rolling(168, min_periods=1).std().fillna(0).values)
        self.feature_names.extend(['smp_std24', 'smp_std168'])

        # Dynamics
        features.append(np.diff(smp, prepend=smp[0]))
        features.append(df['smp_max'].values - df['smp_min'].values)
        features.append(smp_series.pct_change().fillna(0).values)
        self.feature_names.extend(['smp_diff', 'smp_range', 'smp_pct_change'])

        # ========== 5. Lag Features (4) ==========
        features.append(smp_series.shift(24).bfill().values)
        features.append(smp_series.shift(48).bfill().values)
        features.append(smp_series.shift(168).bfill().values)
        features.append(smp_series.shift(336).bfill().values)
        self.feature_names.extend(['smp_lag_24h', 'smp_lag_48h', 'smp_lag_168h', 'smp_lag_336h'])

        # ========== 6. NEW: Demand/Supply Features (6) ==========
        if 'demand_mw' in df.columns:
            demand = df['demand_mw'].values
            supply = df['supply_mw'].values
            reserve_rate = df['reserve_rate'].values

            features.append(demand)
            features.append(supply)
            features.append(reserve_rate)
            self.feature_names.extend(['demand_mw', 'supply_mw', 'reserve_rate'])

            # Demand statistics
            demand_series = pd.Series(demand)
            features.append(demand_series.rolling(24, min_periods=1).mean().values)
            features.append(demand_series.shift(24).bfill().values)
            features.append(np.diff(demand, prepend=demand[0]))
            self.feature_names.extend(['demand_ma24', 'demand_lag_24h', 'demand_diff'])

        # ========== 7. NEW: Weather Features (6) ==========
        weather_cols = ['temperature', 'wind_speed', 'humidity', 'solar_radiation', 'precipitation', 'pressure']
        for col in weather_cols:
            if col in df.columns:
                values = df[col].values
                features.append(values)
                self.feature_names.append(col)

        # Temperature extremes (HDD/CDD)
        if 'temperature' in df.columns:
            temp = df['temperature'].values
            hdd = np.maximum(18 - temp, 0)  # Heating Degree Days
            cdd = np.maximum(temp - 24, 0)  # Cooling Degree Days
            features.append(hdd)
            features.append(cdd)
            self.feature_names.extend(['hdd', 'cdd'])

        # ========== 8. NEW: Power Trading Features (2) ==========
        if 'power_trading_mwh' in df.columns:
            power = df['power_trading_mwh'].values
            features.append(power)
            power_series = pd.Series(power)
            features.append(power_series.rolling(24, min_periods=1).mean().values)
            self.feature_names.extend(['power_trading_mwh', 'power_ma24'])

        # ========== 9. Data Availability Indicators (2) ==========
        if 'has_demand_data' in df.columns:
            features.append(df['has_demand_data'].values)
            self.feature_names.append('has_demand_data')
        if 'has_weather_data' in df.columns:
            features.append(df['has_weather_data'].values)
            self.feature_names.append('has_weather_data')

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

        # Clip extreme values (only during fitting)
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

        # Use same normalization as features for targets (or simple z-score)
        targets_normalized = (targets - self.target_mean) / (self.target_std + 1e-8)

        return features_normalized, targets_normalized


class SMPDatasetV4(Dataset):
    """Dataset for SMP v4 model"""

    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 input_hours: int = 72, output_hours: int = 24):
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

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
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

        # Xavier initialization
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
        return output, attn_weights.mean(dim=1)  # Average attention across heads


class SMPModelV4Enhanced(nn.Module):
    """Enhanced SMP Prediction Model v4.0"""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 3,
                 n_heads: int = 8, dropout: float = 0.3, output_hours: int = 24):
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

        # Temporal Convolution
        self.conv1d = nn.Conv1d(hidden_size * 2, hidden_size * 2, kernel_size=3, padding=1)

        # Output heads
        fc_input = hidden_size * 2
        self.point_head = nn.Sequential(
            nn.Linear(fc_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_hours)
        )

        # Quantile heads
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

        # Temporal convolution
        out = out.transpose(1, 2)
        out = self.conv1d(out)
        out = out.transpose(1, 2)

        # Last timestep features
        features = out[:, -1, :]

        # Outputs
        result = {
            'point': self.point_head(features),
            'q10': self.q10_head(features),
            'q50': self.q50_head(features),
            'q90': self.q90_head(features),
            'attention': attn_weights
        }

        return result


class CombinedLossV4(nn.Module):
    """Combined loss with Huber + Quantile + Consistency"""

    def __init__(self, mse_weight: float = 0.4, quantile_weight: float = 0.4,
                 consistency_weight: float = 0.2):
        super().__init__()
        self.mse_weight = mse_weight
        self.quantile_weight = quantile_weight
        self.consistency_weight = consistency_weight
        self.huber = nn.SmoothL1Loss()

    def quantile_loss(self, pred, target, q):
        error = target - pred
        return torch.mean(torch.max(q * error, (q - 1) * error))

    def forward(self, predictions, targets):
        # Huber loss for point prediction
        point_loss = self.huber(predictions['point'], targets)

        # Quantile losses
        q10_loss = self.quantile_loss(predictions['q10'], targets, 0.1)
        q50_loss = self.quantile_loss(predictions['q50'], targets, 0.5)
        q90_loss = self.quantile_loss(predictions['q90'], targets, 0.9)
        quantile_loss = (q10_loss + q50_loss + q90_loss) / 3

        # Consistency loss (q10 < q50 < q90)
        consistency_loss = torch.mean(
            torch.relu(predictions['q10'] - predictions['q50']) +
            torch.relu(predictions['q50'] - predictions['q90'])
        )

        total_loss = (
            self.mse_weight * point_loss +
            self.quantile_weight * quantile_loss +
            self.consistency_weight * consistency_loss
        )

        return total_loss, {
            'point_loss': point_loss.item(),
            'quantile_loss': quantile_loss.item(),
            'consistency_loss': consistency_loss.item()
        }


class SMPTrainerV4:
    """Training engine for SMP v4 model"""

    def __init__(self, config: ConfigV4Enhanced):
        self.config = config
        self.device = self._get_device()
        self.pipeline = EnhancedDataPipeline(config)

    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def prepare_data(self):
        """Prepare training data"""
        logger.info("=" * 60)
        logger.info("Preparing data...")
        logger.info("=" * 60)

        # Merge all data
        merged_df = self.pipeline.merge_all_data()

        # Create features
        features, targets = self.pipeline.create_enhanced_features(merged_df)

        # Split data
        n = len(features)
        train_end = int(n * self.config.train_ratio)
        val_end = int(n * (self.config.train_ratio + self.config.val_ratio))

        # Normalize (fit only on training data)
        train_features, train_targets = self.pipeline.normalize(
            features[:train_end], targets[:train_end], fit=True
        )
        val_features, val_targets = self.pipeline.normalize(
            features[train_end:val_end], targets[train_end:val_end], fit=False
        )
        test_features, test_targets = self.pipeline.normalize(
            features[val_end:], targets[val_end:], fit=False
        )

        # Create datasets
        train_dataset = SMPDatasetV4(
            train_features, train_targets,
            self.config.input_hours, self.config.output_hours
        )
        val_dataset = SMPDatasetV4(
            val_features, val_targets,
            self.config.input_hours, self.config.output_hours
        )
        test_dataset = SMPDatasetV4(
            test_features, test_targets,
            self.config.input_hours, self.config.output_hours
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

        logger.info(f"  Train samples: {len(train_dataset):,}")
        logger.info(f"  Val samples: {len(val_dataset):,}")
        logger.info(f"  Test samples: {len(test_dataset):,}")

        return train_loader, val_loader, test_loader

    def train(self):
        """Train the model"""
        logger.info("=" * 60)
        logger.info("Starting training...")
        logger.info("=" * 60)

        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data()

        # Create model
        model = SMPModelV4Enhanced(
            input_size=len(self.pipeline.feature_names),
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
            output_hours=self.config.output_hours
        ).to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Model parameters: {n_params:,}")
        logger.info(f"  Device: {self.device}")

        # Loss and optimizer
        criterion = CombinedLossV4()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=10, factor=0.5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        history = {'train_loss': [], 'val_loss': [], 'val_mape': []}

        for epoch in range(self.config.epochs):
            # Training
            model.train()
            train_losses = []

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Add noise
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
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    outputs = model(batch_x)
                    loss, _ = criterion(outputs, batch_y)

                    val_losses.append(loss.item())
                    val_predictions.append(outputs['point'].cpu().numpy())
                    val_targets.append(batch_y.cpu().numpy())

            # Calculate metrics
            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            # Denormalize for MAPE
            val_pred_raw = np.concatenate(val_predictions)
            val_target_raw = np.concatenate(val_targets)

            # Debug: check raw values (only epoch 1)
            if epoch == 0:
                logger.info(f"  [DEBUG] Raw pred range: [{val_pred_raw.min():.2f}, {val_pred_raw.max():.2f}]")
                logger.info(f"  [DEBUG] Raw target range: [{val_target_raw.min():.2f}, {val_target_raw.max():.2f}]")
                logger.info(f"  [DEBUG] Target mean: {self.pipeline.target_mean:.2f}, std: {self.pipeline.target_std:.2f}")

            val_pred_denorm = val_pred_raw * self.pipeline.target_std + self.pipeline.target_mean
            val_target_denorm = val_target_raw * self.pipeline.target_std + self.pipeline.target_mean

            # Debug: check denormalized values (only epoch 1)
            if epoch == 0:
                logger.info(f"  [DEBUG] Denorm pred range: [{val_pred_denorm.min():.2f}, {val_pred_denorm.max():.2f}]")
                logger.info(f"  [DEBUG] Denorm target range: [{val_target_denorm.min():.2f}, {val_target_denorm.max():.2f}]")

            # Robust MAPE calculation (exclude values near zero)
            min_threshold = 10.0  # Minimum 10 won/kWh
            mask = np.abs(val_target_denorm) > min_threshold
            if mask.sum() > 0:
                val_mape = np.mean(
                    np.abs(val_pred_denorm[mask] - val_target_denorm[mask]) /
                    np.abs(val_target_denorm[mask])
                ) * 100
            else:
                val_mape = np.mean(np.abs(val_pred_denorm - val_target_denorm)) * 100  # Fallback to MAE-like

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_mape'].append(val_mape)

            scheduler.step(val_loss)

            # Logging
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                           f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                           f"MAPE: {val_mape:.2f}%")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        model.load_state_dict(best_model_state)

        # Evaluate on test set
        logger.info("=" * 60)
        logger.info("Evaluating on test set...")
        logger.info("=" * 60)

        metrics = self._evaluate(model, test_loader, criterion)

        # Save model
        self._save_model(model, metrics, history)

        return model, metrics

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

        # Robust MAPE (exclude values near zero)
        min_threshold = 10.0  # Minimum 10 won/kWh
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

        # Save model checkpoint
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
            'version': '4.0.0'
        }
        torch.save(checkpoint, output_dir / 'smp_v4_model.pt')
        logger.info(f"  Model saved: {output_dir / 'smp_v4_model.pt'}")

        # Save scaler
        np.save(output_dir / 'smp_v4_scaler.npy', {
            'feature_scaler_mean': self.pipeline.scaler.mean_,
            'feature_scaler_scale': self.pipeline.scaler.scale_,
            'target_mean': self.pipeline.target_mean,
            'target_std': self.pipeline.target_std,
            'feature_names': self.pipeline.feature_names
        })

        # Save metrics
        metrics_full = {
            'version': '4.0.0',
            **metrics,
            'parameters': sum(p.numel() for p in model.parameters()),
            'features': len(self.pipeline.feature_names),
            'feature_names': self.pipeline.feature_names,
            'improvements': [
                'External features (demand, supply, weather)',
                'Extended look-back (72h)',
                'Multi-scale statistics',
                '8-head attention',
                'Temporal convolution'
            ],
            'timestamp': datetime.now().isoformat()
        }

        with open(output_dir / 'smp_v4_metrics.json', 'w') as f:
            json.dump(metrics_full, f, indent=2, ensure_ascii=False)

        logger.info(f"  Metrics saved: {output_dir / 'smp_v4_metrics.json'}")


def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("SMP Prediction Model v4.0 Enhanced Training")
    logger.info("=" * 60)

    config = ConfigV4Enhanced()
    trainer = SMPTrainerV4(config)
    model, metrics = trainer.train()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Final Metrics:")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  Coverage: {metrics['coverage_80']:.1f}%")

    return model, metrics


if __name__ == '__main__':
    main()
