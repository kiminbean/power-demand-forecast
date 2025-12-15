"""
EVAL-005: External Features Experiment
======================================

외부 데이터(인구, 전기차) 효과 검증 실험

비교 대상:
1. baseline: 기본 모델 (시간 + lag)
2. weather: 기상변수 포함
3. external: 외부변수 포함 (인구 + 전기차)
4. full: 모든 변수 포함

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import sys
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

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
    split_data_by_time,
    get_device
)
from models.lstm import create_model
from training.trainer import Trainer, create_scheduler
from evaluation.metrics import compute_all_metrics

from features import (
    add_time_features,
    add_weather_features,
    add_solar_features,
    add_lag_features,
    add_external_features,
    POPULATION_FEATURES,
    EV_FEATURES,
)

warnings.filterwarnings('ignore')


# ============================================================
# Feature Group 정의
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

WEATHER_FEATURES = ['기온', '지면온도', '이슬점온도', '풍속', '일사', '전운량']
DERIVED_WEATHER = ['THI', 'HDD', 'CDD', 'wind_chill']

EXTERNAL_BASIC = [
    'estimated_population',  # 실제 인구
    'ev_cumulative',         # 전기차 누적대수
]

EXTERNAL_DERIVED = [
    'tourist_ratio',         # 관광객 비율
    'ev_penetration',        # 전기차 보급률
    'ev_cumulative_log',     # 전기차 로그 변환
]


def load_and_prepare_data(
    data_path: str,
    include_weather: bool = False,
    include_external: bool = False
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

    if include_external:
        df = add_external_features(df)

    # 결측치 제거
    df = df.dropna()

    # 피처 선택
    target = 'power_demand'
    features = [target] + TIME_FEATURES + LAG_FEATURES

    if include_weather:
        for col in WEATHER_FEATURES + DERIVED_WEATHER:
            if col in df.columns:
                features.append(col)

    if include_external:
        for col in EXTERNAL_BASIC + EXTERNAL_DERIVED:
            if col in df.columns:
                features.append(col)

    available_features = [f for f in features if f in df.columns]
    df_final = df[available_features].copy()

    return df_final, available_features


def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_size: int,
    device: torch.device,
    epochs: int = 50,
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


def run_external_features_experiment(
    data_path: str,
    n_trials: int = 5,
    epochs: int = 50,
    quick: bool = False
) -> Dict:
    """외부 피처 실험 실행"""
    print("=" * 70)
    print("EVAL-005: External Features Experiment")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Trials: {n_trials}, Epochs: {epochs}")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    if quick:
        n_trials = 2
        epochs = 10
        print("Quick test mode enabled")

    # 실험 설정
    experiments = {
        'baseline': {'weather': False, 'external': False},
        'weather': {'weather': True, 'external': False},
        'external': {'weather': False, 'external': True},
        'full': {'weather': True, 'external': True}
    }

    results = {name: [] for name in experiments.keys()}

    # 데이터 로드
    print("\nLoading data for each configuration...")
    data_configs = {}
    for name, config in experiments.items():
        df, features = load_and_prepare_data(
            data_path,
            include_weather=config['weather'],
            include_external=config['external']
        )
        data_configs[name] = {'df': df, 'features': features}
        print(f"  {name}: {len(features)} features")

    for trial in range(1, n_trials + 1):
        print(f"\n--- Trial {trial}/{n_trials} (seed={41 + trial}) ---")
        seed = 41 + trial

        for name, config in experiments.items():
            df = data_configs[name]['df']
            features = data_configs[name]['features']

            # 데이터 분할
            train_df, val_df, test_df = split_data_by_time(df)

            # 스케일링
            scaler = TimeSeriesScaler()
            train_scaled = scaler.fit_transform(train_df.values)
            val_scaled = scaler.transform(val_df.values)
            test_scaled = scaler.transform(test_df.values)

            # 데이터셋 생성
            seq_len = 168
            batch_size = 64

            train_ds = TimeSeriesDataset(train_scaled, seq_length=seq_len, horizon=1)
            val_ds = TimeSeriesDataset(val_scaled, seq_length=seq_len, horizon=1)
            test_ds = TimeSeriesDataset(test_scaled, seq_length=seq_len, horizon=1)

            train_loader = DataLoader(train_ds, batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size)
            test_loader = DataLoader(test_ds, batch_size)

            # 모델 학습
            print(f"  Training {name}...", end=" ", flush=True)
            model = train_model(
                train_loader, val_loader,
                len(features), device, epochs, seed=seed
            )

            # 평가
            metrics = evaluate_model(model, test_loader, scaler, device)
            results[name].append(metrics)
            print(f"MAPE={metrics['MAPE']:.2f}%")

    # 결과 요약
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    summary = {}
    for name, model_results in results.items():
        mapes = [r['MAPE'] for r in model_results]
        r2s = [r['R2'] for r in model_results]
        summary[name] = {
            'mape_mean': np.mean(mapes),
            'mape_std': np.std(mapes),
            'r2_mean': np.mean(r2s),
            'r2_std': np.std(r2s),
            'n_features': len(data_configs[name]['features'])
        }
        print(f"{name:15s}: MAPE={np.mean(mapes):.2f}%±{np.std(mapes):.2f}, "
              f"R²={np.mean(r2s):.4f}, features={summary[name]['n_features']}")

    # 개선율 계산
    baseline_mape = summary['baseline']['mape_mean']
    print("\n[Improvement vs baseline]")
    for name in ['weather', 'external', 'full']:
        improvement = (baseline_mape - summary[name]['mape_mean']) / baseline_mape * 100
        print(f"  {name}: {improvement:+.2f}%")

    # 결과 저장
    output_dir = PROJECT_ROOT / "results" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        'experiment': 'EVAL-005',
        'description': 'External Features Experiment (Population + EV)',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': n_trials,
            'epochs': epochs,
            'sequence_length': 168,
            'external_features': EXTERNAL_BASIC + EXTERNAL_DERIVED
        },
        'summary': summary,
        'detailed_results': {k: v for k, v in results.items()}
    }

    with open(output_dir / "external_features_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=float)

    print(f"\n✓ Report saved: {output_dir / 'external_features_report.json'}")

    return report


def main():
    parser = argparse.ArgumentParser(description='EVAL-005: External Features Experiment')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--data', type=str,
                        default=str(PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv'),
                        help='Data path')

    args = parser.parse_args()

    run_external_features_experiment(
        data_path=args.data,
        n_trials=args.trials,
        epochs=args.epochs,
        quick=args.quick
    )


if __name__ == '__main__':
    main()
