"""
Winter Data Test for Conditional Model
======================================

겨울철 데이터(12월, 1월, 2월)에서 conditional model 효과 검증

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.dataset import (
    TimeSeriesScaler,
    TimeSeriesDataset,
    get_device
)
from models.lstm import LSTMModel, create_model
from models.conditional import (
    Season,
    SeasonClassifier,
    InflectionDetector,
    ConditionalPredictor,
    create_conditional_predictor
)
from training.trainer import Trainer, create_scheduler
from evaluation.metrics import compute_all_metrics

from features import (
    add_time_features,
    add_weather_features,
    add_solar_features,
    add_lag_features
)

import warnings
warnings.filterwarnings('ignore')


# Feature Groups
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

WEATHER_BASIC = ['기온', '지면온도']
WEATHER_EXTENDED = ['이슬점온도', '풍속', '일사', '전운량', '강수량']
DERIVED_FEATURES = ['THI', 'HDD', 'CDD', 'wind_chill']


def load_and_prepare_data(
    data_path: str,
    include_weather: bool = True
) -> Tuple[pd.DataFrame, List[str]]:
    """데이터 로드 및 전처리"""
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)

    # 피처 엔지니어링
    df = add_time_features(df, include_holiday=True)
    df = add_lag_features(df, demand_col='power_demand')

    if include_weather and '기온' in df.columns:
        df = add_weather_features(df, include_thi=True, include_hdd_cdd=True)
        if '일사' in df.columns:
            df = add_solar_features(df)

    # 결측치 제거
    df = df.dropna()

    # 피처 선택
    target = 'power_demand'
    features = [target] + TIME_FEATURES + LAG_FEATURES

    if include_weather:
        for col in WEATHER_BASIC + WEATHER_EXTENDED + DERIVED_FEATURES:
            if col in df.columns:
                features.append(col)

    available_features = [f for f in features if f in df.columns]
    df_final = df[available_features].copy()

    return df_final, available_features


def filter_winter_data(df: pd.DataFrame) -> pd.DataFrame:
    """겨울철 데이터만 필터링 (12월, 1월, 2월)"""
    winter_mask = df.index.month.isin([12, 1, 2])
    return df[winter_mask].copy()


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    device: torch.device,
    epochs: int = 30,
    patience: int = 10,
    seed: int = 42
) -> nn.Module:
    """모델 학습"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = create_model(
        model_type='lstm',
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = create_scheduler(optimizer, 'plateau')

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler
    )

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        verbose=0
    )

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    scaler: TimeSeriesScaler,
    device: torch.device
) -> Dict[str, float]:
    """모델 평가"""
    model.eval()
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            all_preds.append(pred.cpu().numpy())
            all_actuals.append(y.cpu().numpy())

    preds = np.concatenate(all_preds).flatten()
    actuals = np.concatenate(all_actuals).flatten()

    # 역변환
    preds = scaler.inverse_transform_target(preds)
    actuals = scaler.inverse_transform_target(actuals)

    return compute_all_metrics(actuals, preds)


def evaluate_conditional(
    predictor: ConditionalPredictor,
    test_loader_demand: DataLoader,
    test_loader_weather: DataLoader,
    scaler_demand: TimeSeriesScaler,
    device: torch.device,
    timestamps: pd.DatetimeIndex
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """조건부 모델 평가"""
    predictor.eval()
    predictor.reset_stats()

    all_preds = []
    all_actuals = []

    demand_iter = iter(test_loader_demand)
    weather_iter = iter(test_loader_weather)

    batch_idx = 0
    batch_size = test_loader_demand.batch_size

    with torch.no_grad():
        for (X_demand, y_demand), (X_weather, _) in zip(demand_iter, weather_iter):
            X_demand = X_demand.to(device)
            X_weather = X_weather.to(device)

            # 현재 배치의 타임스탬프
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(timestamps))
            batch_timestamps = timestamps[start_idx:end_idx]

            # 최근 수요 데이터
            recent_demands = X_demand[:, -2:, 0].cpu().numpy()

            preds, contexts = predictor.forward(
                X_demand, X_weather,
                batch_timestamps,
                [rd for rd in recent_demands]
            )

            all_preds.append(preds.cpu().numpy())
            all_actuals.append(y_demand.cpu().numpy())

            batch_idx += 1

    preds = np.concatenate(all_preds).flatten()
    actuals = np.concatenate(all_actuals).flatten()

    # 역변환
    preds = scaler_demand.inverse_transform_target(preds)
    actuals = scaler_demand.inverse_transform_target(actuals)

    metrics = compute_all_metrics(actuals, preds)
    stats = predictor.get_stats()

    return metrics, stats


def run_winter_test(
    data_path: str,
    epochs: int = 30,
    seed: int = 42
) -> Dict:
    """겨울철 데이터 테스트 실행"""
    print("=" * 70)
    print("Winter Data Test for Conditional Model")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    # 데이터 로드
    print("\nLoading data...")
    df_demand, features_demand = load_and_prepare_data(data_path, include_weather=False)
    df_weather, features_weather = load_and_prepare_data(data_path, include_weather=True)

    print(f"  Total data: {len(df_demand)} rows")
    print(f"  demand_only features: {len(features_demand)}")
    print(f"  weather_full features: {len(features_weather)}")

    # 데이터 분할: 2013-2021 train, 2022 겨울 test
    train_end = '2021-12-31'
    test_start = '2022-12-01'
    test_end = '2023-02-28'

    train_demand = df_demand[:train_end]
    train_weather = df_weather[:train_end]

    # 겨울철 테스트 데이터 (2022년 12월 ~ 2023년 2월)
    test_demand = df_demand[test_start:test_end]
    test_weather = df_weather[test_start:test_end]

    # 검증 데이터 (2022년 1-2월)
    val_start = '2022-01-01'
    val_end = '2022-02-28'
    val_demand = df_demand[val_start:val_end]
    val_weather = df_weather[val_start:val_end]

    print(f"\n  Train period: ~{train_end} ({len(train_demand)} rows)")
    print(f"  Val period: {val_start} ~ {val_end} ({len(val_demand)} rows)")
    print(f"  Test period: {test_start} ~ {test_end} ({len(test_demand)} rows)")

    # 겨울철 테스트 데이터 필터링
    test_demand_winter = filter_winter_data(test_demand)
    test_weather_winter = filter_winter_data(test_weather)
    print(f"  Winter test data: {len(test_demand_winter)} rows")

    if len(test_demand_winter) < 168:
        print("ERROR: Not enough winter test data!")
        return {}

    # Scalers
    scaler_demand = TimeSeriesScaler()
    scaler_weather = TimeSeriesScaler()

    train_demand_scaled = scaler_demand.fit_transform(train_demand.values)
    val_demand_scaled = scaler_demand.transform(val_demand.values)
    test_demand_scaled = scaler_demand.transform(test_demand_winter.values)

    train_weather_scaled = scaler_weather.fit_transform(train_weather.values)
    val_weather_scaled = scaler_weather.transform(val_weather.values)
    test_weather_scaled = scaler_weather.transform(test_weather_winter.values)

    # Datasets
    seq_len = 168
    batch_size = 64

    train_ds_demand = TimeSeriesDataset(train_demand_scaled, seq_length=seq_len, horizon=1)
    val_ds_demand = TimeSeriesDataset(val_demand_scaled, seq_length=seq_len, horizon=1)
    test_ds_demand = TimeSeriesDataset(test_demand_scaled, seq_length=seq_len, horizon=1)

    train_ds_weather = TimeSeriesDataset(train_weather_scaled, seq_length=seq_len, horizon=1)
    val_ds_weather = TimeSeriesDataset(val_weather_scaled, seq_length=seq_len, horizon=1)
    test_ds_weather = TimeSeriesDataset(test_weather_scaled, seq_length=seq_len, horizon=1)

    train_loader_demand = DataLoader(train_ds_demand, batch_size, shuffle=True)
    val_loader_demand = DataLoader(val_ds_demand, batch_size)
    test_loader_demand = DataLoader(test_ds_demand, batch_size)

    train_loader_weather = DataLoader(train_ds_weather, batch_size, shuffle=True)
    val_loader_weather = DataLoader(val_ds_weather, batch_size)
    test_loader_weather = DataLoader(test_ds_weather, batch_size)

    # 테스트 타임스탬프
    test_timestamps = test_demand_winter.index[seq_len:]

    results = {}

    # 1. Train demand_only model
    print("\n[1] Training demand_only model...", end=" ", flush=True)
    model_demand = train_model(
        train_loader_demand, val_loader_demand,
        len(features_demand), device, epochs, seed=seed
    )
    metrics_demand = evaluate_model(
        model_demand, test_loader_demand, scaler_demand, device
    )
    results['demand_only'] = metrics_demand
    print(f"MAPE={metrics_demand['MAPE']:.2f}%")

    # 2. Train weather_full model
    print("[2] Training weather_full model...", end=" ", flush=True)
    model_weather = train_model(
        train_loader_weather, val_loader_weather,
        len(features_weather), device, epochs, seed=seed
    )
    metrics_weather = evaluate_model(
        model_weather, test_loader_weather, scaler_weather, device
    )
    results['weather_full'] = metrics_weather
    print(f"MAPE={metrics_weather['MAPE']:.2f}%")

    # 3. Conditional Hard
    print("[3] Evaluating conditional_hard...", end=" ", flush=True)
    predictor_hard = create_conditional_predictor(
        demand_only_model=model_demand,
        weather_full_model=model_weather,
        train_demand=train_demand['power_demand'],
        mode="hard"
    )
    predictor_hard.to(device)

    metrics_hard, stats_hard = evaluate_conditional(
        predictor_hard, test_loader_demand, test_loader_weather,
        scaler_demand, device, test_timestamps
    )
    results['conditional_hard'] = metrics_hard
    print(f"MAPE={metrics_hard['MAPE']:.2f}% (weather_used={stats_hard['weather_full_ratio']:.1%})")

    # 4. Conditional Soft
    print("[4] Evaluating conditional_soft...", end=" ", flush=True)
    predictor_soft = create_conditional_predictor(
        demand_only_model=model_demand,
        weather_full_model=model_weather,
        train_demand=train_demand['power_demand'],
        mode="soft"
    )
    predictor_soft.to(device)

    metrics_soft, stats_soft = evaluate_conditional(
        predictor_soft, test_loader_demand, test_loader_weather,
        scaler_demand, device, test_timestamps
    )
    results['conditional_soft'] = metrics_soft
    print(f"MAPE={metrics_soft['MAPE']:.2f}% (soft_blend={stats_soft['soft_blend_ratio']:.1%})")

    # 결과 요약
    print("\n" + "=" * 70)
    print("WINTER TEST RESULTS")
    print("=" * 70)

    for model_name, metrics in results.items():
        print(f"{model_name:20s}: MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}")

    # 개선율 계산
    baseline_mape = results['demand_only']['MAPE']
    print("\n[Improvement vs demand_only on WINTER data]")
    for model_name in ['weather_full', 'conditional_hard', 'conditional_soft']:
        improvement = (baseline_mape - results[model_name]['MAPE']) / baseline_mape * 100
        print(f"  {model_name}: {improvement:+.2f}%")

    # 통계 출력
    print(f"\n[Conditional Stats]")
    print(f"  Hard mode - weather_used: {stats_hard['weather_full_ratio']:.1%}")
    print(f"  Soft mode - soft_blend: {stats_soft['soft_blend_ratio']:.1%}")

    # 결과 저장
    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'experiment': 'Winter Test',
        'description': 'Conditional Model Test on Winter Data (Dec-Feb)',
        'timestamp': datetime.now().isoformat(),
        'test_period': f'{test_start} ~ {test_end}',
        'test_samples': len(test_demand_winter),
        'results': results,
        'conditional_stats': {
            'hard': stats_hard,
            'soft': stats_soft
        }
    }

    with open(output_dir / "winter_test_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n✓ Report saved: {output_dir / 'winter_test_report.json'}")

    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Winter Data Test')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv'))

    args = parser.parse_args()

    run_winter_test(
        data_path=args.data,
        epochs=args.epochs,
        seed=args.seed
    )
