"""
Production Model Training Script
================================

최적 구성으로 Production 모델 학습

Best Configuration (EVAL-003~006 결과):
- Primary: demand_only (시간 + lag 피처)
- Secondary: weather_full (conditional_soft용)
- External Features: 제외

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Callable, Dict, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import (
    TimeSeriesScaler,
    TimeSeriesDataset,
    split_data_by_time,
    get_device
)
from models.lstm import create_model
from training.trainer import Trainer, create_scheduler
from evaluation.metrics import compute_all_metrics
from inference.feature_preprocessing import MODEL_TIMEZONE

from features import (
    add_time_features,
    add_weather_features,
    add_lag_features,
)

warnings.filterwarnings('ignore')


# ============================================================
# Feature Configurations (Based on Experiment Results)
# ============================================================

TIME_FEATURES = [
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_holiday'
]

LAG_FEATURES = [
    'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
    'demand_ma_6h', 'demand_ma_24h',
    'demand_std_24h',
    'demand_diff_1h', 'demand_diff_24h'
]

WEATHER_FEATURES = ['기온', 'THI', 'HDD', 'CDD']


# ============================================================
# Model Configuration (Optimized)
# ============================================================

MODEL_CONFIG = {
    'model_type': 'lstm',
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False
}

TRAINING_CONFIG = {
    'seq_length': 168,
    'horizon': 1,
    'batch_size': 64,
    'epochs': 100,
    'patience': 15,
    'learning_rate': 0.001,
    'scheduler': 'plateau'
}


def _validate_hourly_continuity(index: pd.DatetimeIndex, stage: str) -> None:
    """Reject non-hourly adjacent timestamps with the first offending interval."""
    intervals = index[1:] - index[:-1]
    invalid = np.flatnonzero(intervals != pd.Timedelta(hours=1))
    if len(invalid):
        position = invalid[0]
        raise ValueError(
            f"Hourly continuity violated {stage}: {index[position]} to "
            f"{index[position + 1]} (interval {intervals[position]})"
        )


def load_data(
    data_path: str,
    include_weather: bool = False
) -> Tuple[pd.DataFrame, list]:
    """데이터 로드 및 전처리"""
    df = pd.read_csv(data_path)
    try:
        timestamps = pd.to_datetime(df['datetime'], errors='coerce')
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid datetime values in training data") from exc
    if timestamps.isna().any() or not pd.api.types.is_datetime64_any_dtype(
        timestamps.dtype
    ):
        raise ValueError("Invalid or missing datetime values in training data")
    if timestamps.duplicated().any():
        raise ValueError("Duplicate datetime values in training data")
    df['datetime'] = timestamps
    df.set_index('datetime', inplace=True)
    # 시간대가 있는 학습 입력도 추론과 같은 제주 현지 달력 기준을 사용한다.
    time_index = pd.DatetimeIndex(df.index)
    if time_index.tz is not None:
        df.index = time_index.tz_convert(MODEL_TIMEZONE)
    # 지연 변수와 시퀀스가 과거에서 미래 순서를 따르도록 정렬한다.
    df.sort_index(inplace=True)
    _validate_hourly_continuity(pd.DatetimeIndex(df.index), "before lag creation")

    # 선택된 원본의 무한대가 이동평균과 파생식에서 숨겨지지 않도록 검사한다.
    raw_features = ['power_demand'] + (WEATHER_FEATURES.copy() if include_weather else [])
    if include_weather and '기온' in df.columns and 'temp_mean' in df.columns:
        # 재계산으로 덮어쓰는 기상 피처는 원본 값 대신 계산 결과를 검사한다.
        raw_features = [name for name in raw_features if name not in {'HDD', 'CDD'}]
        raw_features.append('temp_mean')
        if 'dewpoint_mean' in df.columns:
            raw_features.remove('THI')
            raw_features.append('dewpoint_mean')
    for column in raw_features:
        if column in df.columns:
            values = pd.Series(pd.to_numeric(df[column], errors='raise')).dropna()
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"Raw training feature {column} must contain finite values")

    # Feature Engineering
    df = add_time_features(df, include_holiday=True)
    df = add_lag_features(df, demand_col='power_demand')

    if include_weather and '기온' in df.columns:
        df = add_weather_features(df, include_thi=True, include_hdd_cdd=True)

    # Feature Selection
    target = 'power_demand'
    features = [target] + TIME_FEATURES + LAG_FEATURES

    if include_weather:
        for col in WEATHER_FEATURES:
            if col in df.columns:
                features.append(col)

    available = [f for f in features if f in df.columns]
    # 모델이 사용하지 않는 열의 결측치는 학습 행을 제거하지 않는다.
    df = df.dropna(subset=available)
    if not np.isfinite(df[available].to_numpy(dtype=float)).all():
        raise ValueError("Selected training features must contain finite values")
    _validate_hourly_continuity(pd.DatetimeIndex(df.index), "after dropna")
    return df[available].copy(), available


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_features: int,
    device: torch.device,
    config: Dict,
    seed: int = 42,
    verbose: int = 1,
    model_wrapper: Optional[Callable[[nn.Module], nn.Module]] = None
) -> Tuple[nn.Module, TimeSeriesScaler, Dict]:
    """모델 학습

    model_wrapper가 주어지면 생성된 기본 모델을 감싼 모듈로 학습한다
    (예: 잔차 타깃 래퍼). 기본값 None이면 기존 동작과 같다.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Scaling
    scaler = TimeSeriesScaler()
    train_scaled = scaler.fit_transform(train_df.values)
    val_scaled = scaler.transform(val_df.values)

    # Datasets
    train_ds = TimeSeriesDataset(
        train_scaled,
        seq_length=config['seq_length'],
        horizon=config['horizon']
    )
    val_ds = TimeSeriesDataset(
        val_scaled,
        seq_length=config['seq_length'],
        horizon=config['horizon']
    )

    train_loader = DataLoader(train_ds, config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, config['batch_size'])

    # Model
    model = create_model(
        model_type=MODEL_CONFIG['model_type'],
        input_size=n_features,
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_layers=MODEL_CONFIG['num_layers'],
        dropout=MODEL_CONFIG['dropout']
    ).to(device)
    if model_wrapper is not None:
        model = model_wrapper(model).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = create_scheduler(optimizer, config['scheduler'])

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        patience=config['patience'],
        verbose=verbose
    )

    return model, scaler, history


def evaluate_model(
    model: nn.Module,
    test_df: pd.DataFrame,
    scaler: TimeSeriesScaler,
    device: torch.device,
    config: Dict
) -> Dict[str, float]:
    """모델 평가"""
    test_scaled = scaler.transform(test_df.values)
    test_ds = TimeSeriesDataset(
        test_scaled,
        seq_length=config['seq_length'],
        horizon=config['horizon']
    )
    test_loader = DataLoader(test_ds, config['batch_size'])

    model.eval()
    all_preds, all_actuals = [], []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    actuals = np.concatenate(all_actuals).flatten()

    preds = scaler.inverse_transform_target(preds)
    actuals = scaler.inverse_transform_target(actuals)

    return compute_all_metrics(actuals, preds)


def save_model(
    model: nn.Module,
    scaler: TimeSeriesScaler,
    features: list,
    metrics: Dict,
    config: Dict,
    model_name: str,
    output_dir: Path
):
    """모델 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model weights
    model_path = output_dir / f"{model_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': MODEL_CONFIG,
        'training_config': config,
        'features': features,
        'n_features': len(features)
    }, model_path)

    # Scaler
    scaler_path = output_dir / f"{model_name}_scaler.pkl"
    scaler.save(str(scaler_path))

    # Metadata
    metadata = {
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'n_features': len(features),
        'model_config': MODEL_CONFIG,
        'training_config': config,
        'metrics': metrics
    }

    with open(output_dir / f"{model_name}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=float)

    print(f"  Model saved: {model_path}")
    print(f"  Scaler saved: {scaler_path}")


def train_production_models(
    data_path: str,
    output_dir: Optional[str] = None,
    seed: int = 42
) -> Dict:
    """Production 모델 학습"""
    print("=" * 70)
    print("PRODUCTION MODEL TRAINING")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {seed}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    if output_dir is None:
        output_dir = PROJECT_ROOT / "models" / "production"
    else:
        output_dir = Path(output_dir)

    results = {}

    # ========================================
    # 1. demand_only 모델 (Primary)
    # ========================================
    print("\n" + "=" * 50)
    print("1. Training demand_only Model (Primary)")
    print("=" * 50)

    df_demand, features_demand = load_data(data_path, include_weather=False)
    print(f"Features: {len(features_demand)}")
    print(f"Data shape: {df_demand.shape}")

    train_df, val_df, test_df = split_data_by_time(df_demand)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    model_demand, scaler_demand, history_demand = train_model(
        train_df, val_df,
        len(features_demand),
        device,
        TRAINING_CONFIG,
        seed=seed,
        verbose=1
    )

    metrics_demand = evaluate_model(
        model_demand, test_df, scaler_demand, device, TRAINING_CONFIG
    )

    print(f"\n[demand_only Results]")
    print(f"  MAPE: {metrics_demand['MAPE']:.2f}%")
    print(f"  RMSE: {metrics_demand['RMSE']:.2f}")
    print(f"  R²: {metrics_demand['R2']:.4f}")

    save_model(
        model_demand, scaler_demand, features_demand,
        metrics_demand, TRAINING_CONFIG,
        "demand_only", output_dir
    )

    results['demand_only'] = {
        'features': features_demand,
        'metrics': metrics_demand,
        'best_epoch': history_demand.history.get('best_epoch', 0)
    }

    # ========================================
    # 2. weather_full 모델 (for Conditional)
    # ========================================
    print("\n" + "=" * 50)
    print("2. Training weather_full Model (for Conditional)")
    print("=" * 50)

    df_weather, features_weather = load_data(data_path, include_weather=True)
    print(f"Features: {len(features_weather)}")
    print(f"Data shape: {df_weather.shape}")

    train_df_w, val_df_w, test_df_w = split_data_by_time(df_weather)

    model_weather, scaler_weather, history_weather = train_model(
        train_df_w, val_df_w,
        len(features_weather),
        device,
        TRAINING_CONFIG,
        seed=seed,
        verbose=1
    )

    metrics_weather = evaluate_model(
        model_weather, test_df_w, scaler_weather, device, TRAINING_CONFIG
    )

    print(f"\n[weather_full Results]")
    print(f"  MAPE: {metrics_weather['MAPE']:.2f}%")
    print(f"  RMSE: {metrics_weather['RMSE']:.2f}")
    print(f"  R²: {metrics_weather['R2']:.4f}")

    save_model(
        model_weather, scaler_weather, features_weather,
        metrics_weather, TRAINING_CONFIG,
        "weather_full", output_dir
    )

    results['weather_full'] = {
        'features': features_weather,
        'metrics': metrics_weather,
        'best_epoch': history_weather.history.get('best_epoch', 0)
    }

    # ========================================
    # 3. Summary
    # ========================================
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<20} {'MAPE':<12} {'RMSE':<12} {'R²':<10}")
    print("-" * 54)
    for name, res in results.items():
        m = res['metrics']
        print(f"{name:<20} {m['MAPE']:.2f}%{'':<6} {m['RMSE']:.2f}{'':<6} {m['R2']:.4f}")

    # Save combined report
    report = {
        'training_date': datetime.now().isoformat(),
        'device': str(device),
        'seed': seed,
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG,
        'results': {
            name: {
                'n_features': len(res['features']),
                'metrics': res['metrics'],
                'best_epoch': res['best_epoch']
            }
            for name, res in results.items()
        },
        'recommendation': {
            'primary_model': 'demand_only',
            'conditional_mode': 'soft',
            'winter_improvement': '+2.23%'
        }
    }

    with open(output_dir / "training_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n✓ All models saved to: {output_dir}")
    print(f"✓ Training report: {output_dir / 'training_report.json'}")

    # Usage instructions
    print("\n" + "=" * 70)
    print("USAGE INSTRUCTIONS")
    print("=" * 70)
    print("""
# Load and use production model:

import torch
from models.lstm import create_model
from data.dataset import TimeSeriesScaler

# 1. Load demand_only model (primary)
checkpoint = torch.load('models/production/demand_only.pt')
model = create_model(**checkpoint['model_config'],
                     input_size=checkpoint['n_features'])
model.load_state_dict(checkpoint['model_state_dict'])

# 2. Load scaler
scaler = TimeSeriesScaler()
scaler.load('models/production/demand_only_scaler.pkl')

# 3. For winter operation, use conditional_soft:
from models.conditional import create_conditional_predictor

predictor = create_conditional_predictor(
    demand_only_model=model_demand,
    weather_full_model=model_weather,
    train_demand=train_data['power_demand'],
    mode="soft"
)
""")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Production Models')
    parser.add_argument('--data', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv'))
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train_production_models(
        data_path=args.data,
        output_dir=args.output,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
