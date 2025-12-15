"""
EVAL-003: Horizon 변화에 따른 기상변수 효과 실험
================================================

EVAL-002에서 h=1 예측에서는 기상변수의 효과가 없었음.
이는 강한 Lag 변수(demand_lag_1, corr=0.974)가 기상변수 신호를 마스킹했기 때문.

가설: 예측 horizon이 길어질수록 Lag 변수의 예측력이 감소하고,
      기상변수의 상대적 중요도가 증가할 것이다.

실험 설계:
- Horizon: h=1, 24, 48, 168 (1시간, 1일, 2일, 1주일 후 예측)
- Feature Groups: demand_only, weather_full
- 각 조합당 5회 반복 실험

Reference: EVAL-002 결과 분석 및 Gemini 토론 결과

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

# Feature engineering imports
from features import (
    add_time_features,
    add_weather_features,
    add_solar_features,
    add_lag_features
)

warnings.filterwarnings('ignore')


# ============================================================
# Feature Group 정의 (EVAL-002와 동일)
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

WEATHER_BASIC = ['기온', '지면온도']
SOIL_TEMP = ['m005Te', 'm01Te', 'm02Te', 'm03Te']
WEATHER_EXTENDED = ['이슬점온도', '풍속', '일사', '전운량', '강수량']
DERIVED_FEATURES = ['THI', 'HDD', 'CDD', 'wind_chill']
SOLAR_FEATURES = [
    'solar_elevation', 'is_daylight', 'theoretical_irradiance',
    'clear_sky_index', 'cloud_attenuation', 'solar_estimated', 'btm_effect'
]
WEATHER_LAG_FEATURES = [
    'temp_ma_6h', 'temp_ma_24h',
    'temp_lag_1', 'temp_lag_24',
    'temp_min_24h', 'temp_max_24h', 'temp_range_24h',
    'irradiance_ma_6h', 'irradiance_ma_24h',
    'irradiance_sum_24h',
    'humidity_ma_6h', 'humidity_ma_24h',
]

# 실험 대상 Feature Groups (demand_only vs weather_full만 비교)
FEATURE_GROUPS = {
    'demand_only': {
        'description': '전력수요 + 시간특성만 (Baseline)',
        'features': TIME_FEATURES + LAG_FEATURES
    },
    'weather_full': {
        'description': '+ 전체 기상변수 (Full Model)',
        'features': TIME_FEATURES + LAG_FEATURES + WEATHER_BASIC + SOIL_TEMP +
                   WEATHER_EXTENDED + DERIVED_FEATURES + SOLAR_FEATURES +
                   WEATHER_LAG_FEATURES
    }
}

# 실험할 Horizon 값들
HORIZONS = [1, 24, 48, 168]  # 1시간, 1일, 2일, 1주일


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
    'sequence_length': 168,  # 1주일치 입력 (horizon 168에 대응)
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,

    # Training
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 10,

    # Experiment
    'n_trials': 5,  # horizon당 시간이 오래 걸리므로 5회로 설정

    # Output
    'output_dir': PROJECT_ROOT / 'results' / 'metrics',
    'figure_dir': PROJECT_ROOT / 'results' / 'figures',
}


# ============================================================
# 데이터 준비 함수
# ============================================================

def apply_feature_engineering(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """데이터프레임에 피처 엔지니어링을 적용합니다."""
    df = df.copy()

    df = add_time_features(df)

    df = add_weather_features(
        df,
        temp_col='기온',
        dewpoint_col='이슬점온도',
        wind_col='풍속',
        include_thi=True,
        include_hdd_cdd=True,
        include_wind_chill=True
    )

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

    df = add_lag_features(
        df,
        demand_col=target_col,
        temp_col='기온',
        irradiance_col='일사',
        humidity_col='습도',
        demand_lags=[1, 24, 168],
        ma_windows=[6, 24],
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
    """데이터를 로드하고 특정 피처 그룹에 맞게 준비합니다."""
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    df = apply_feature_engineering(df, target_col)

    available_features = [f for f in feature_cols if f in df.columns]
    columns = [target_col] + available_features
    columns = list(dict.fromkeys(columns))

    df_selected = df[columns].copy()
    df_selected = df_selected.dropna()

    train_df, val_df, test_df = split_data_by_time(df_selected, train_end, val_end)

    scaler = TimeSeriesScaler()
    train_data = scaler.fit_transform(train_df.values)
    val_data = scaler.transform(val_df.values)
    test_data = scaler.transform(test_df.values)

    target_idx = 0

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
    """단일 실험을 수행합니다."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    model = create_model(
        'lstm',
        input_size=n_features,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

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

    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        patience=config['patience'],
        verbose=0 if not verbose else 1,
        log_interval=10
    )

    result = trainer.evaluate(test_loader, return_predictions=True)

    predictions = scaler.inverse_transform_target(result['predictions'], target_idx)
    actuals = scaler.inverse_transform_target(result['actuals'], target_idx)

    metrics = compute_all_metrics(actuals, predictions)

    return {
        'MAPE': metrics['MAPE'],
        'R2': metrics['R2'],
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE']
    }


# ============================================================
# Horizon 실험 함수
# ============================================================

def run_horizon_experiment(
    horizon: int,
    group_name: str,
    config: dict,
    n_trials: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    특정 horizon과 feature group에 대해 n_trials 반복 실험을 수행합니다.
    """
    feature_cols = FEATURE_GROUPS[group_name]['features']

    if verbose:
        print(f"\n  [h={horizon}, {group_name}] Loading data...", end=" ", flush=True)

    # 데이터 준비
    train_data, val_data, test_data, scaler, target_idx = load_and_prepare_data(
        data_path=str(config['data_path']),
        target_col=config['target_col'],
        feature_cols=feature_cols,
        train_end=config['train_end'],
        val_end=config['val_end']
    )

    n_features = train_data.shape[1]

    # DataLoader 생성 (horizon 적용)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        target_idx=target_idx,
        seq_length=config['sequence_length'],
        horizon=horizon,  # 핵심: horizon 변경
        batch_size=config['batch_size']
    )

    if verbose:
        print(f"features={n_features}, samples={len(train_loader.dataset)}")

    # 반복 실험
    results = []
    for trial in range(n_trials):
        seed = 42 + trial

        if verbose:
            print(f"    Trial {trial + 1}/{n_trials} (seed={seed})...", end=" ", flush=True)

        metrics = run_single_trial(
            train_loader, val_loader, test_loader,
            n_features, scaler, target_idx,
            config, seed, verbose=False
        )

        metrics['trial'] = trial + 1
        metrics['horizon'] = horizon
        metrics['group'] = group_name
        metrics['n_features'] = n_features
        results.append(metrics)

        if verbose:
            print(f"MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}", flush=True)

    return pd.DataFrame(results)


def run_full_horizon_comparison(
    config: dict = None,
    horizons: List[int] = None,
    groups: List[str] = None,
    n_trials: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    전체 Horizon 비교 실험을 실행합니다.
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if horizons is None:
        horizons = HORIZONS

    if groups is None:
        groups = ['demand_only', 'weather_full']

    if verbose:
        print("=" * 70)
        print("EVAL-003: Horizon Comparison Experiment")
        print("=" * 70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {get_device()}")
        print(f"Horizons: {horizons}")
        print(f"Groups: {groups}")
        print(f"Trials per combination: {n_trials}")
        print(f"Total experiments: {len(horizons) * len(groups) * n_trials}")
        print("=" * 70)

    all_results = []

    for horizon in horizons:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Horizon: h={horizon} ({horizon}시간 후 예측)")
            print(f"{'='*60}")

        for group_name in groups:
            results = run_horizon_experiment(
                horizon=horizon,
                group_name=group_name,
                config=config,
                n_trials=n_trials,
                verbose=verbose
            )
            all_results.append(results)

    combined_results = pd.concat(all_results, ignore_index=True)

    return combined_results


# ============================================================
# 결과 분석 및 시각화
# ============================================================

def analyze_horizon_results(results: pd.DataFrame) -> pd.DataFrame:
    """Horizon별, Group별 통계 요약"""
    summary = results.groupby(['horizon', 'group']).agg({
        'MAPE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'n_features': 'first'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]

    return summary


def calculate_weather_improvement(results: pd.DataFrame) -> pd.DataFrame:
    """
    각 Horizon에서 weather_full이 demand_only 대비 얼마나 개선되었는지 계산
    """
    improvements = []

    for horizon in results['horizon'].unique():
        h_data = results[results['horizon'] == horizon]

        demand_only = h_data[h_data['group'] == 'demand_only']['MAPE'].mean()
        weather_full = h_data[h_data['group'] == 'weather_full']['MAPE'].mean()

        demand_only_r2 = h_data[h_data['group'] == 'demand_only']['R2'].mean()
        weather_full_r2 = h_data[h_data['group'] == 'weather_full']['R2'].mean()

        mape_improvement = (demand_only - weather_full) / demand_only * 100
        r2_improvement = (weather_full_r2 - demand_only_r2) / demand_only_r2 * 100

        improvements.append({
            'horizon': horizon,
            'demand_only_MAPE': demand_only,
            'weather_full_MAPE': weather_full,
            'MAPE_improvement_%': mape_improvement,
            'demand_only_R2': demand_only_r2,
            'weather_full_R2': weather_full_r2,
            'R2_improvement_%': r2_improvement
        })

    return pd.DataFrame(improvements)


def print_horizon_report(results: pd.DataFrame, improvements: pd.DataFrame) -> str:
    """Horizon 비교 리포트 출력"""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("EVAL-003: HORIZON COMPARISON REPORT")
    lines.append("=" * 80)

    # Horizon별 결과
    lines.append("\n[Performance by Horizon]")
    lines.append("-" * 80)
    lines.append(f"{'Horizon':>8} {'Group':<15} {'MAPE (%)':>12} {'R²':>12} {'RMSE':>10}")
    lines.append("-" * 80)

    summary = results.groupby(['horizon', 'group']).agg({
        'MAPE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'RMSE': 'mean'
    })

    for (horizon, group), row in summary.iterrows():
        mape_str = f"{row['MAPE']['mean']:.2f}±{row['MAPE']['std']:.2f}"
        r2_str = f"{row['R2']['mean']:.4f}±{row['R2']['std']:.4f}"
        rmse_str = f"{row['RMSE']['mean']:.1f}"
        lines.append(f"{horizon:>8} {group:<15} {mape_str:>12} {r2_str:>12} {rmse_str:>10}")

    lines.append("-" * 80)

    # Weather 개선율
    lines.append("\n[Weather Variable Improvement by Horizon]")
    lines.append("-" * 80)
    lines.append(f"{'Horizon':>8} {'MAPE Δ':>12} {'R² Δ':>12} {'Weather Effect':>18}")
    lines.append("-" * 80)

    for _, row in improvements.iterrows():
        horizon = int(row['horizon'])
        mape_delta = row['MAPE_improvement_%']
        r2_delta = row['R2_improvement_%']

        # 효과 판정
        if mape_delta > 5:
            effect = "Strong Positive"
        elif mape_delta > 0:
            effect = "Weak Positive"
        elif mape_delta > -5:
            effect = "Negligible"
        else:
            effect = "Negative (Noise)"

        lines.append(f"{horizon:>8} {mape_delta:>+11.1f}% {r2_delta:>+11.1f}% {effect:>18}")

    lines.append("-" * 80)

    # 핵심 발견
    lines.append("\n[Key Findings]")

    # 가설 검증 (최소/최대 horizon 비교)
    min_horizon = improvements['horizon'].min()
    max_horizon = improvements['horizon'].max()

    min_improvement = improvements[improvements['horizon'] == min_horizon]['MAPE_improvement_%'].values[0]
    max_improvement = improvements[improvements['horizon'] == max_horizon]['MAPE_improvement_%'].values[0]

    if max_improvement > min_improvement:
        lines.append("✓ 가설 검증: Horizon이 길어질수록 기상변수의 효과가 증가함")
        lines.append(f"  - h={min_horizon}: MAPE 개선 {min_improvement:+.1f}%")
        lines.append(f"  - h={max_horizon}: MAPE 개선 {max_improvement:+.1f}%")
    else:
        lines.append("✗ 가설 기각: Horizon과 기상변수 효과 간 명확한 상관관계 없음")
        lines.append(f"  - h={min_horizon}: MAPE 개선 {min_improvement:+.1f}%")
        lines.append(f"  - h={max_horizon}: MAPE 개선 {max_improvement:+.1f}%")

    lines.append("=" * 80)

    report = "\n".join(lines)
    print(report)
    return report


def create_horizon_plots(
    results: pd.DataFrame,
    output_dir: str
) -> None:
    """Horizon 비교 시각화"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        horizons = sorted(results['horizon'].unique())
        x = np.arange(len(horizons))
        width = 0.35

        # 그룹별 평균 계산
        demand_mape = [results[(results['horizon'] == h) & (results['group'] == 'demand_only')]['MAPE'].mean() for h in horizons]
        weather_mape = [results[(results['horizon'] == h) & (results['group'] == 'weather_full')]['MAPE'].mean() for h in horizons]

        demand_r2 = [results[(results['horizon'] == h) & (results['group'] == 'demand_only')]['R2'].mean() for h in horizons]
        weather_r2 = [results[(results['horizon'] == h) & (results['group'] == 'weather_full')]['R2'].mean() for h in horizons]

        # 1. MAPE Bar Chart
        ax1 = axes[0, 0]
        bars1 = ax1.bar(x - width/2, demand_mape, width, label='Demand Only', color='#FF6B6B', alpha=0.8)
        bars2 = ax1.bar(x + width/2, weather_mape, width, label='Weather Full', color='#45B7D1', alpha=0.8)
        ax1.set_xlabel('Horizon (hours)')
        ax1.set_ylabel('MAPE (%)')
        ax1.set_title('MAPE by Horizon')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'h={h}' for h in horizons])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. R² Bar Chart
        ax2 = axes[0, 1]
        ax2.bar(x - width/2, demand_r2, width, label='Demand Only', color='#FF6B6B', alpha=0.8)
        ax2.bar(x + width/2, weather_r2, width, label='Weather Full', color='#45B7D1', alpha=0.8)
        ax2.set_xlabel('Horizon (hours)')
        ax2.set_ylabel('R²')
        ax2.set_title('R² by Horizon')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'h={h}' for h in horizons])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. MAPE Improvement Line Chart
        ax3 = axes[1, 0]
        improvements = [(d - w) / d * 100 for d, w in zip(demand_mape, weather_mape)]
        ax3.plot(horizons, improvements, 'o-', color='#4ECDC4', linewidth=2, markersize=10)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax3.fill_between(horizons, improvements, 0, alpha=0.3, color='#4ECDC4')
        ax3.set_xlabel('Horizon (hours)')
        ax3.set_ylabel('MAPE Improvement (%)')
        ax3.set_title('Weather Variable Effect by Horizon')
        ax3.grid(True, alpha=0.3)

        # 4. MAPE Trend Line
        ax4 = axes[1, 1]
        ax4.plot(horizons, demand_mape, 'o-', label='Demand Only', color='#FF6B6B', linewidth=2, markersize=8)
        ax4.plot(horizons, weather_mape, 's-', label='Weather Full', color='#45B7D1', linewidth=2, markersize=8)
        ax4.set_xlabel('Horizon (hours)')
        ax4.set_ylabel('MAPE (%)')
        ax4.set_title('MAPE Trend by Horizon')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.suptitle('EVAL-003: Horizon Comparison Results', fontsize=14, y=1.02)
        plt.tight_layout()

        # 저장
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'horizon_comparison.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Horizon comparison plot saved: {output_path}")

    except ImportError:
        print("\n⚠️ matplotlib not available. Skipping plots.")


# ============================================================
# 메인 함수
# ============================================================

def main(
    n_trials: int = 5,
    epochs: int = 50,
    horizons: List[int] = None,
    quick_test: bool = False
):
    """
    메인 실험 실행 함수

    Args:
        n_trials: 반복 횟수 (기본: 5)
        epochs: 학습 에포크 (기본: 50)
        horizons: 테스트할 horizon 리스트 (기본: [1, 24, 48, 168])
        quick_test: 빠른 테스트 모드 (2회, 10 epochs, h=[1, 24])
    """
    config = DEFAULT_CONFIG.copy()

    if quick_test:
        n_trials = 2
        config['epochs'] = 10
        horizons = [1, 24]
        print("\n⚡ Quick test mode: 2 trials, 10 epochs, h=[1, 24]")
    else:
        config['epochs'] = epochs
        if horizons is None:
            horizons = HORIZONS

    # 실험 실행
    results = run_full_horizon_comparison(
        config=config,
        horizons=horizons,
        n_trials=n_trials,
        verbose=True
    )

    # 결과 분석
    improvements = calculate_weather_improvement(results)
    report = print_horizon_report(results, improvements)

    # 결과 저장
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV 저장
    results_path = output_dir / 'horizon_comparison.csv'
    results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    # 개선율 저장
    improvements_path = output_dir / 'horizon_improvements.csv'
    improvements.to_csv(improvements_path, index=False)
    print(f"✓ Improvements saved: {improvements_path}")

    # 시각화
    figure_dir = config['figure_dir']
    create_horizon_plots(results, str(figure_dir))

    # JSON 리포트 저장
    min_horizon = improvements['horizon'].min()
    max_horizon = improvements['horizon'].max()
    min_improvement = improvements[improvements['horizon'] == min_horizon]['MAPE_improvement_%'].values[0]
    max_improvement = improvements[improvements['horizon'] == max_horizon]['MAPE_improvement_%'].values[0]

    report_data = {
        'experiment': 'EVAL-003',
        'description': 'Horizon Comparison Experiment',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': n_trials,
            'epochs': config['epochs'],
            'horizons': horizons,
            'sequence_length': config['sequence_length'],
            'hidden_size': config['hidden_size']
        },
        'improvements': improvements.to_dict(orient='records'),
        'hypothesis_validated': bool(max_improvement > min_improvement)
    }

    report_path = output_dir / 'horizon_comparison_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Report saved: {report_path}")

    return results, improvements


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Horizon Comparison Experiment')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--horizons', type=int, nargs='+', default=None, help='Horizons to test')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')

    args = parser.parse_args()

    main(
        n_trials=args.trials,
        epochs=args.epochs,
        horizons=args.horizons,
        quick_test=args.quick
    )
