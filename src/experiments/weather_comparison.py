"""
EVAL-002: 기상변수 포함/미포함 비교 실험
=========================================

기상 변수가 전력 수요 예측에 미치는 영향 분석

실험 설계:
1. Demand Only: 전력수요 + 시간 특성만 사용
2. Weather Basic: + 기온, 지중온도
3. Weather Full: + 전체 기상변수 (THI, HDD/CDD 등)

10회 반복 실험으로 통계적 유의성 검증

Reference: JPD_RNN_Weather.pdf (정현수, 길준민, 2025)

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

# Feature engineering imports
from features import (
    add_time_features,
    add_weather_features,
    add_solar_features,
    add_lag_features
)

warnings.filterwarnings('ignore')


# ============================================================
# Feature Group 정의
# ============================================================

# 기본 시간 특성 (모든 실험에 포함)
TIME_FEATURES = [
    'hour_sin', 'hour_cos',
    'dayofweek_sin', 'dayofweek_cos',
    'month_sin', 'month_cos',
    'is_weekend', 'is_holiday'
]

# Lag 특성 (전력수요 패턴) - add_lag_features가 생성하는 이름
LAG_FEATURES = [
    'demand_lag_1', 'demand_lag_24', 'demand_lag_168',
    'demand_ma_6h', 'demand_ma_24h',
    'demand_std_24h',        # 롤링 표준편차
    'demand_diff_1h', 'demand_diff_24h'  # 차분
]

# 기본 기상 변수
WEATHER_BASIC = [
    '기온',        # 기온
    '지면온도',    # 지면온도
]

# 지중온도
SOIL_TEMP = [
    'm005Te',      # 5cm 지중온도
    'm01Te',       # 10cm 지중온도
    'm02Te',       # 20cm 지중온도
    'm03Te',       # 30cm 지중온도
]

# 추가 기상 변수
WEATHER_EXTENDED = [
    '이슬점온도',  # 이슬점
    '풍속',        # 풍속
    '일사',        # 일사량
    '전운량',      # 전운량
    '강수량',      # 강수량
]

# 파생 변수 (weather_features에서 생성)
DERIVED_FEATURES = [
    'THI',           # 불쾌지수
    'HDD',           # 난방도일
    'CDD',           # 냉방도일
    'wind_chill',    # 체감온도
]

# 태양광 관련 (solar_features에서 생성)
SOLAR_FEATURES = [
    'solar_elevation',
    'is_daylight',
    'theoretical_irradiance',
    'clear_sky_index',
    'cloud_attenuation',
    'solar_estimated',
    'btm_effect',
]

# 기상 지연 변수 (lag_features에서 생성 - ma_windows=[6, 24])
WEATHER_LAG_FEATURES = [
    'temp_ma_6h', 'temp_ma_24h',
    'temp_lag_1', 'temp_lag_24',
    'temp_min_24h', 'temp_max_24h', 'temp_range_24h',
    'irradiance_ma_6h', 'irradiance_ma_24h',
    'irradiance_sum_24h',
    'humidity_ma_6h', 'humidity_ma_24h',
]

# Feature Groups 정의
FEATURE_GROUPS = {
    'demand_only': {
        'description': '전력수요 + 시간특성만',
        'features': TIME_FEATURES + LAG_FEATURES
    },
    'weather_basic': {
        'description': '+ 기온, 지중온도',
        'features': TIME_FEATURES + LAG_FEATURES + WEATHER_BASIC + SOIL_TEMP
    },
    'weather_full': {
        'description': '+ 전체 기상변수',
        'features': TIME_FEATURES + LAG_FEATURES + WEATHER_BASIC + SOIL_TEMP +
                   WEATHER_EXTENDED + DERIVED_FEATURES + SOLAR_FEATURES +
                   WEATHER_LAG_FEATURES
    }
}


# ============================================================
# 실험 설정
# ============================================================

DEFAULT_CONFIG = {
    # Data
    'data_path': PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv',
    'target_col': 'power_demand',
    'train_end': '2022-12-31 23:00:00',
    'val_end': '2023-06-30 23:00:00',

    # Model
    'sequence_length': 48,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,  # 실험용으로 줄임 (풀 실험시 100)
    'patience': 10,

    # Experiment
    'n_trials': 10,
    'horizon': 1,

    # Output
    'output_dir': PROJECT_ROOT / 'results' / 'metrics',
    'figure_dir': PROJECT_ROOT / 'results' / 'figures',
}


# ============================================================
# 데이터 준비 함수
# ============================================================

def apply_feature_engineering(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    데이터프레임에 피처 엔지니어링을 적용합니다.

    생성되는 피처:
    - 시간 특성: hour_sin/cos, dayofweek_sin/cos, month_sin/cos, is_weekend, is_holiday
    - 기상 파생: THI, HDD, CDD, wind_chill
    - 태양광: solar_elevation, is_daylight, etc.
    - 지연 변수: demand_lag_*, demand_ma_*, temp_ma_*, etc.
    """
    df = df.copy()

    # 1. 시간 특성 추가
    df = add_time_features(df)

    # 2. 기상 파생변수 추가 (THI, HDD, CDD)
    df = add_weather_features(
        df,
        temp_col='기온',
        dewpoint_col='이슬점온도',
        wind_col='풍속',
        include_thi=True,
        include_hdd_cdd=True,
        include_wind_chill=True
    )

    # 3. 태양광 특성 추가 (제주도 좌표)
    df = add_solar_features(
        df,
        irradiance_col='일사',
        cloud_col='전운량',
        include_theoretical=True,
        include_clear_sky_index=True,
        include_daylight=True,
        include_estimated_gen=True,
        include_btm=True
    )

    # 4. 지연 변수 추가
    df = add_lag_features(
        df,
        demand_col=target_col,
        temp_col='기온',
        irradiance_col='일사',
        humidity_col='습도',
        demand_lags=[1, 24, 168],  # 1시간, 24시간, 1주일 전
        ma_windows=[6, 24],        # 6시간, 24시간 이동평균
        include_demand_features=True,
        include_weather_features=True,
        include_diff=True,
        include_std=True
    )

    return df


def load_and_prepare_data(
    data_path: str,
    target_col: str,
    feature_cols: List[str],
    train_end: str,
    val_end: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TimeSeriesScaler, int]:
    """
    데이터를 로드하고 특정 피처 그룹에 맞게 준비합니다.

    Returns:
        train_data, val_data, test_data, scaler, target_idx
    """
    # 데이터 로드
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 피처 엔지니어링 적용
    df = apply_feature_engineering(df, target_col)

    # 사용 가능한 피처만 선택
    available_features = [f for f in feature_cols if f in df.columns]

    # 타겟을 첫 번째 컬럼으로
    columns = [target_col] + available_features
    columns = list(dict.fromkeys(columns))  # 중복 제거

    df_selected = df[columns].copy()

    # 결측치 처리 (lag 변수로 인한 초기 NaN 제거)
    df_selected = df_selected.dropna()

    # 분할
    train_df, val_df, test_df = split_data_by_time(df_selected, train_end, val_end)

    # 스케일링
    scaler = TimeSeriesScaler()
    train_data = scaler.fit_transform(train_df.values)
    val_data = scaler.transform(val_df.values)
    test_data = scaler.transform(test_df.values)

    target_idx = 0  # 타겟이 첫 번째

    return train_data, val_data, test_data, scaler, target_idx


def create_dataloaders(
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    target_idx: int,
    seq_length: int,
    horizon: int,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """DataLoader 생성"""
    train_dataset = TimeSeriesDataset(train_data, target_idx, seq_length, horizon)
    val_dataset = TimeSeriesDataset(val_data, target_idx, seq_length, horizon)
    test_dataset = TimeSeriesDataset(test_data, target_idx, seq_length, horizon)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================================================
# 단일 실험 함수
# ============================================================

def run_single_trial(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_features: int,
    scaler: TimeSeriesScaler,
    target_idx: int,
    config: dict,
    seed: int,
    verbose: bool = False
) -> Dict[str, float]:
    """
    단일 실험을 수행합니다.

    Returns:
        Dict: 평가 지표 (MAPE, R2, RMSE, MAE)
    """
    # 시드 설정
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    # 모델 생성
    model = create_model(
        'lstm',
        input_size=n_features,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # 학습 설정
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = create_scheduler(optimizer, 'plateau', patience=5)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        grad_clip=1.0
    )

    # 학습
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        patience=config['patience'],
        verbose=0 if not verbose else 1,
        log_interval=10
    )

    # 평가
    result = trainer.evaluate(test_loader, return_predictions=True)

    # 역스케일링
    predictions = scaler.inverse_transform_target(result['predictions'], target_idx)
    actuals = scaler.inverse_transform_target(result['actuals'], target_idx)

    # 평가 지표 계산
    metrics = compute_all_metrics(actuals, predictions)

    return {
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    }


# ============================================================
# 실험 그룹 실행 함수
# ============================================================

def run_experiment_group(
    group_name: str,
    feature_cols: List[str],
    config: dict,
    n_trials: int = 10,
    verbose: bool = True
) -> pd.DataFrame:
    """
    하나의 Feature Group에 대해 n_trials 반복 실험을 수행합니다.

    Returns:
        pd.DataFrame: 각 trial의 결과
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Feature Group: {group_name}")
        print(f"Description: {FEATURE_GROUPS[group_name]['description']}")
        print(f"Features: {len(feature_cols)}")
        print(f"Trials: {n_trials}")
        print(f"{'='*60}")

    # 데이터 준비
    train_data, val_data, test_data, scaler, target_idx = load_and_prepare_data(
        data_path=str(config['data_path']),
        target_col=config['target_col'],
        feature_cols=feature_cols,
        train_end=config['train_end'],
        val_end=config['val_end']
    )

    n_features = train_data.shape[1]
    if verbose:
        print(f"Actual features used: {n_features}")

    # DataLoader 생성
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        target_idx=target_idx,
        seq_length=config['sequence_length'],
        horizon=config['horizon'],
        batch_size=config['batch_size']
    )

    # 반복 실험
    results = []
    for trial in range(n_trials):
        seed = 42 + trial

        if verbose:
            print(f"\n  Trial {trial + 1}/{n_trials} (seed={seed})...", end=" ")

        metrics = run_single_trial(
            train_loader, val_loader, test_loader,
            n_features, scaler, target_idx,
            config, seed, verbose=False
        )

        metrics['trial'] = trial + 1
        metrics['group'] = group_name
        metrics['n_features'] = n_features
        results.append(metrics)

        if verbose:
            print(f"MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}")

    return pd.DataFrame(results)


# ============================================================
# 전체 실험 실행
# ============================================================

def run_weather_comparison_experiment(
    config: dict = None,
    n_trials: int = 10,
    groups: List[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    기상변수 비교 실험을 전체 실행합니다.

    Args:
        config: 실험 설정
        n_trials: 반복 횟수
        groups: 실험할 그룹 리스트 (기본: 전체)
        verbose: 출력 여부

    Returns:
        pd.DataFrame: 전체 결과
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if groups is None:
        groups = list(FEATURE_GROUPS.keys())

    if verbose:
        print("=" * 60)
        print("EVAL-002: Weather Variable Comparison Experiment")
        print("=" * 60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {get_device()}")
        print(f"Trials per group: {n_trials}")
        print(f"Groups: {groups}")

    all_results = []

    for group_name in groups:
        feature_cols = FEATURE_GROUPS[group_name]['features']

        results = run_experiment_group(
            group_name=group_name,
            feature_cols=feature_cols,
            config=config,
            n_trials=n_trials,
            verbose=verbose
        )

        all_results.append(results)

    # 결과 통합
    combined_results = pd.concat(all_results, ignore_index=True)

    return combined_results


# ============================================================
# 결과 분석 및 시각화
# ============================================================

def analyze_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    실험 결과를 분석합니다.

    Returns:
        pd.DataFrame: 그룹별 통계 요약
    """
    summary = results.groupby('group').agg({
        'MAPE': ['mean', 'std', 'min', 'max'],
        'R2': ['mean', 'std', 'min', 'max'],
        'RMSE': ['mean', 'std', 'min', 'max'],
        'MAE': ['mean', 'std', 'min', 'max'],
        'n_features': 'first'
    }).round(4)

    # 컬럼 이름 정리
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return summary


def print_comparison_report(results: pd.DataFrame, summary: pd.DataFrame) -> str:
    """비교 리포트를 출력합니다."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("WEATHER VARIABLE COMPARISON REPORT")
    lines.append("=" * 70)

    # 그룹별 요약
    lines.append("\n[Summary by Feature Group]")
    lines.append("-" * 70)
    lines.append(f"{'Group':<15} {'Features':>8} {'MAPE (%)':>12} {'R²':>12} {'RMSE':>12}")
    lines.append("-" * 70)

    for group in ['demand_only', 'weather_basic', 'weather_full']:
        if group in summary.index:
            row = summary.loc[group]
            n_feat = int(row['n_features_first'])
            mape_str = f"{row['MAPE_mean']:.2f}±{row['MAPE_std']:.2f}"
            r2_str = f"{row['R2_mean']:.4f}±{row['R2_std']:.4f}"
            rmse_str = f"{row['RMSE_mean']:.1f}±{row['RMSE_std']:.1f}"
            lines.append(f"{group:<15} {n_feat:>8} {mape_str:>12} {r2_str:>12} {rmse_str:>12}")

    lines.append("-" * 70)

    # 개선율 계산
    if 'demand_only' in summary.index and 'weather_full' in summary.index:
        baseline_mape = summary.loc['demand_only', 'MAPE_mean']
        full_mape = summary.loc['weather_full', 'MAPE_mean']
        mape_improvement = (baseline_mape - full_mape) / baseline_mape * 100

        baseline_r2 = summary.loc['demand_only', 'R2_mean']
        full_r2 = summary.loc['weather_full', 'R2_mean']
        r2_improvement = (full_r2 - baseline_r2) / baseline_r2 * 100

        lines.append(f"\n[Improvement: demand_only → weather_full]")
        lines.append(f"  MAPE: {baseline_mape:.2f}% → {full_mape:.2f}% ({mape_improvement:+.1f}%)")
        lines.append(f"  R²:   {baseline_r2:.4f} → {full_r2:.4f} ({r2_improvement:+.1f}%)")

    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)
    return report


def create_boxplot(
    results: pd.DataFrame,
    output_path: str,
    metric: str = 'MAPE'
) -> None:
    """
    Box plot을 생성합니다.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # MAPE Box plot
        groups = ['demand_only', 'weather_basic', 'weather_full']
        group_labels = ['Demand\nOnly', 'Weather\nBasic', 'Weather\nFull']

        data_mape = [results[results['group'] == g]['MAPE'].values for g in groups]
        data_r2 = [results[results['group'] == g]['R2'].values for g in groups]

        # MAPE
        bp1 = axes[0].boxplot(data_mape, labels=group_labels, patch_artist=True)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0].set_ylabel('MAPE (%)')
        axes[0].set_title('MAPE by Feature Group')
        axes[0].grid(True, alpha=0.3)

        # R²
        bp2 = axes[1].boxplot(data_r2, labels=group_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[1].set_ylabel('R²')
        axes[1].set_title('R² by Feature Group')
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Weather Variable Impact on Power Demand Forecasting', fontsize=12, y=1.02)
        plt.tight_layout()

        # 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Box plot saved: {output_path}")

    except ImportError:
        print("\n⚠️ matplotlib not available. Skipping box plot.")


# ============================================================
# 메인 함수
# ============================================================

def main(n_trials: int = 10, epochs: int = 50, quick_test: bool = False):
    """
    메인 실험 실행 함수

    Args:
        n_trials: 반복 횟수 (기본: 10)
        epochs: 학습 에포크 (기본: 50)
        quick_test: 빠른 테스트 모드 (3회, 10 epochs)
    """
    config = DEFAULT_CONFIG.copy()

    if quick_test:
        n_trials = 3
        config['epochs'] = 10
        print("\n⚡ Quick test mode: 3 trials, 10 epochs")
    else:
        config['epochs'] = epochs

    # 실험 실행
    results = run_weather_comparison_experiment(
        config=config,
        n_trials=n_trials,
        verbose=True
    )

    # 결과 분석
    summary = analyze_results(results)
    report = print_comparison_report(results, summary)

    # 결과 저장
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    results_path = output_dir / 'weather_comparison.csv'
    results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    # 요약 저장
    summary_path = output_dir / 'weather_comparison_summary.csv'
    summary.to_csv(summary_path)
    print(f"✓ Summary saved: {summary_path}")

    # Box plot 생성
    figure_dir = config['figure_dir']
    figure_dir.mkdir(parents=True, exist_ok=True)

    boxplot_path = figure_dir / 'weather_comparison_boxplot.png'
    create_boxplot(results, str(boxplot_path))

    # JSON 리포트 저장
    report_data = {
        'experiment': 'EVAL-002',
        'description': 'Weather Variable Comparison',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': n_trials,
            'epochs': config['epochs'],
            'sequence_length': config['sequence_length'],
            'hidden_size': config['hidden_size'],
            'horizon': config['horizon']
        },
        'summary': summary.to_dict(),
        'groups': {
            group: {
                'description': FEATURE_GROUPS[group]['description'],
                'n_features': len(FEATURE_GROUPS[group]['features'])
            }
            for group in FEATURE_GROUPS
        }
    }

    report_path = output_dir / 'weather_comparison_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Report saved: {report_path}")

    return results, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Weather Variable Comparison Experiment')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')

    args = parser.parse_args()

    main(n_trials=args.trials, epochs=args.epochs, quick_test=args.quick)
