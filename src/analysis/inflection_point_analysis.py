"""
Inflection Point Analysis: 변곡점(급변구간) 잔차 분석
====================================================

EVAL-002 결과에서 h=1 예측시 기상변수의 효과가 없었음.
Gemini와의 토론 결과, 기상변수는 "변곡점"에서 특히 중요할 것으로 예상.

분석 목적:
- 전력수요가 급격히 변하는 구간(상위 5%)을 식별
- 해당 구간에서 demand_only vs weather_full 모델의 잔차 비교
- 기상변수가 변곡점 예측에 도움이 되는지 검증

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

FEATURE_GROUPS = {
    'demand_only': TIME_FEATURES + LAG_FEATURES,
    'weather_full': TIME_FEATURES + LAG_FEATURES + WEATHER_BASIC + SOIL_TEMP +
                   WEATHER_EXTENDED + DERIVED_FEATURES + SOLAR_FEATURES +
                   WEATHER_LAG_FEATURES
}


# ============================================================
# 실험 설정
# ============================================================

DEFAULT_CONFIG = {
    'data_path': PROJECT_ROOT / 'data' / 'processed' / 'jeju_hourly_merged.csv',
    'target_col': 'power_demand',
    'train_end': '2022-12-31 23:00:00',
    'val_end': '2023-06-30 23:00:00',
    'sequence_length': 48,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
    'patience': 10,
    'horizon': 1,
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
        df, temp_col='기온', dewpoint_col='이슬점온도', wind_col='풍속',
        include_thi=True, include_hdd_cdd=True, include_wind_chill=True
    )
    df = add_solar_features(
        df, irradiance_col='일사', cloud_col='전운량',
        include_theoretical=True, include_clear_sky_index=True,
        include_daylight=True, include_estimated_gen=True, include_btm=True
    )
    df = add_lag_features(
        df, demand_col=target_col, temp_col='기온',
        irradiance_col='일사', humidity_col='습도',
        demand_lags=[1, 24, 168], ma_windows=[6, 24],
        include_demand_features=True, include_weather_features=True,
        include_diff=True, include_std=True
    )
    return df


def prepare_data_with_index(
    data_path: str,
    target_col: str,
    feature_cols: List[str],
    train_end: str,
    val_end: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TimeSeriesScaler, int, pd.DatetimeIndex]:
    """데이터를 준비하고 테스트 인덱스도 반환합니다."""
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

    return train_data, val_data, test_data, scaler, 0, test_df.index


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
# 모델 학습 및 예측
# ============================================================

def train_and_predict(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    n_features: int,
    scaler: TimeSeriesScaler,
    target_idx: int,
    config: dict,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    모델을 학습하고 예측값과 실제값을 반환합니다.

    Returns:
        predictions, actuals (원래 스케일)
    """
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
        verbose=0,
        log_interval=10
    )

    result = trainer.evaluate(test_loader, return_predictions=True)

    # 배열을 1D로 변환 (출력이 [N, 1] 형태일 수 있음)
    preds = np.array(result['predictions']).flatten()
    acts = np.array(result['actuals']).flatten()

    predictions = scaler.inverse_transform_target(preds, target_idx)
    actuals = scaler.inverse_transform_target(acts, target_idx)

    return predictions, actuals


# ============================================================
# 변곡점 식별 함수
# ============================================================

def identify_inflection_points(
    actuals: np.ndarray,
    percentile: float = 95
) -> np.ndarray:
    """
    전력수요 변화량이 상위 percentile에 해당하는 변곡점을 식별합니다.

    Args:
        actuals: 실제 전력수요 값
        percentile: 상위 몇 퍼센트를 변곡점으로 볼 것인지 (기본 95 = 상위 5%)

    Returns:
        변곡점 인덱스 배열 (boolean)
    """
    # 배열이 1D인지 확인하고 필요시 변환
    actuals = np.asarray(actuals).flatten()

    if len(actuals) < 2:
        return np.zeros(len(actuals), dtype=bool)

    # 전시간 대비 변화량 계산
    changes = np.abs(np.diff(actuals))

    if len(changes) == 0:
        return np.zeros(len(actuals), dtype=bool)

    # 변화량 임계값 계산
    threshold = np.percentile(changes, percentile)

    # 변곡점 식별 (첫 번째 점은 변화량 계산 불가)
    is_inflection = np.zeros(len(actuals), dtype=bool)
    is_inflection[1:] = changes >= threshold

    return is_inflection


def identify_seasonal_inflection_points(
    actuals: np.ndarray,
    index: pd.DatetimeIndex,
    percentile: float = 95
) -> Dict[str, np.ndarray]:
    """
    계절별로 변곡점을 식별합니다.

    Returns:
        Dict with keys: 'summer', 'winter', 'transition', 'all'
    """
    months = index.month

    # 계절 정의
    summer_mask = (months >= 6) & (months <= 8)
    winter_mask = (months == 12) | (months <= 2)
    transition_mask = ~summer_mask & ~winter_mask

    # 전체 변곡점
    all_inflection = identify_inflection_points(actuals, percentile)

    # 계절별 조합
    return {
        'summer': all_inflection & summer_mask[:len(all_inflection)],
        'winter': all_inflection & winter_mask[:len(all_inflection)],
        'transition': all_inflection & transition_mask[:len(all_inflection)],
        'all': all_inflection
    }


# ============================================================
# 잔차 분석 함수
# ============================================================

def analyze_residuals(
    predictions: np.ndarray,
    actuals: np.ndarray,
    inflection_mask: np.ndarray
) -> Dict[str, float]:
    """
    변곡점과 일반구간에서의 잔차를 분석합니다.

    Returns:
        Dict with metrics for inflection points and normal points
    """
    residuals = np.abs(predictions - actuals)
    percentage_errors = residuals / np.abs(actuals) * 100

    # 변곡점 구간
    inflection_residuals = residuals[inflection_mask]
    inflection_pct_errors = percentage_errors[inflection_mask]

    # 일반 구간
    normal_mask = ~inflection_mask
    normal_residuals = residuals[normal_mask]
    normal_pct_errors = percentage_errors[normal_mask]

    return {
        'inflection_mae': np.mean(inflection_residuals) if len(inflection_residuals) > 0 else np.nan,
        'inflection_mape': np.mean(inflection_pct_errors) if len(inflection_pct_errors) > 0 else np.nan,
        'inflection_rmse': np.sqrt(np.mean(inflection_residuals**2)) if len(inflection_residuals) > 0 else np.nan,
        'inflection_count': np.sum(inflection_mask),
        'normal_mae': np.mean(normal_residuals),
        'normal_mape': np.mean(normal_pct_errors),
        'normal_rmse': np.sqrt(np.mean(normal_residuals**2)),
        'normal_count': np.sum(normal_mask),
        'total_mape': np.mean(percentage_errors),
        'total_mae': np.mean(residuals)
    }


# ============================================================
# 메인 분석 함수
# ============================================================

def run_inflection_point_analysis(
    config: dict = None,
    percentile: float = 95,
    n_trials: int = 3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    변곡점 분석을 실행합니다.

    Args:
        config: 실험 설정
        percentile: 변곡점 임계값 (상위 percentile%)
        n_trials: 반복 횟수
        verbose: 출력 여부

    Returns:
        분석 결과 DataFrame
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()

    if verbose:
        print("=" * 70)
        print("Inflection Point Analysis")
        print("=" * 70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Device: {get_device()}")
        print(f"Inflection threshold: Top {100 - percentile:.0f}% changes")
        print(f"Trials: {n_trials}")
        print("=" * 70)

    all_results = []

    for group_name in ['demand_only', 'weather_full']:
        if verbose:
            print(f"\n[{group_name}] Training models...", flush=True)

        feature_cols = FEATURE_GROUPS[group_name]

        # 데이터 준비
        train_data, val_data, test_data, scaler, target_idx, test_index = \
            prepare_data_with_index(
                str(config['data_path']),
                config['target_col'],
                feature_cols,
                config['train_end'],
                config['val_end']
            )

        n_features = train_data.shape[1]

        # DataLoader 생성
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data, val_data, test_data,
            target_idx=target_idx,
            seq_length=config['sequence_length'],
            horizon=config['horizon'],
            batch_size=config['batch_size']
        )

        for trial in range(n_trials):
            seed = 42 + trial

            if verbose:
                print(f"  Trial {trial + 1}/{n_trials}...", end=" ", flush=True)

            # 학습 및 예측
            predictions, actuals = train_and_predict(
                train_loader, val_loader, test_loader,
                n_features, scaler, target_idx, config, seed
            )

            # 유효한 인덱스 조정 (sequence_length + horizon 이후부터)
            valid_start = config['sequence_length'] + config['horizon'] - 1
            valid_index = test_index[valid_start:valid_start + len(predictions)]

            # 변곡점 식별
            inflection_points = identify_seasonal_inflection_points(
                actuals, valid_index, percentile
            )

            # 각 구간별 잔차 분석
            for period, mask in inflection_points.items():
                metrics = analyze_residuals(predictions, actuals, mask)
                metrics['group'] = group_name
                metrics['trial'] = trial + 1
                metrics['period'] = period
                metrics['n_features'] = n_features
                all_results.append(metrics)

            if verbose:
                all_mask = inflection_points['all']
                inflection_mape = np.mean(np.abs(predictions[all_mask] - actuals[all_mask]) / np.abs(actuals[all_mask]) * 100)
                print(f"Inflection MAPE={inflection_mape:.2f}%", flush=True)

    return pd.DataFrame(all_results)


def analyze_and_compare_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    결과를 분석하고 demand_only vs weather_full을 비교합니다.
    """
    # 그룹별, 기간별 평균
    summary = results.groupby(['period', 'group']).agg({
        'inflection_mape': ['mean', 'std'],
        'inflection_mae': 'mean',
        'normal_mape': ['mean', 'std'],
        'normal_mae': 'mean',
        'inflection_count': 'mean'
    }).round(4)

    return summary


def print_analysis_report(results: pd.DataFrame) -> str:
    """분석 리포트를 출력합니다."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("INFLECTION POINT ANALYSIS REPORT")
    lines.append("=" * 80)

    # 기간별 비교
    for period in ['all', 'summer', 'winter', 'transition']:
        period_data = results[results['period'] == period]
        if len(period_data) == 0:
            continue

        lines.append(f"\n[{period.upper()} Period]")
        lines.append("-" * 60)

        demand_inflection = period_data[period_data['group'] == 'demand_only']['inflection_mape'].mean()
        weather_inflection = period_data[period_data['group'] == 'weather_full']['inflection_mape'].mean()

        demand_normal = period_data[period_data['group'] == 'demand_only']['normal_mape'].mean()
        weather_normal = period_data[period_data['group'] == 'weather_full']['normal_mape'].mean()

        inflection_count = period_data['inflection_count'].mean()

        lines.append(f"변곡점 수: {inflection_count:.0f}")
        lines.append("")
        lines.append(f"{'구간':<15} {'Demand Only':>12} {'Weather Full':>12} {'개선율':>10}")
        lines.append("-" * 60)

        inflection_improvement = (demand_inflection - weather_inflection) / demand_inflection * 100
        normal_improvement = (demand_normal - weather_normal) / demand_normal * 100

        lines.append(f"{'변곡점 MAPE':<15} {demand_inflection:>11.2f}% {weather_inflection:>11.2f}% {inflection_improvement:>+9.1f}%")
        lines.append(f"{'일반구간 MAPE':<15} {demand_normal:>11.2f}% {weather_normal:>11.2f}% {normal_improvement:>+9.1f}%")

        # 변곡점에서 더 큰 개선이 있는지
        if inflection_improvement > normal_improvement:
            lines.append(f"\n✓ 변곡점에서 기상변수 효과 더 큼 ({inflection_improvement:+.1f}% vs {normal_improvement:+.1f}%)")
        else:
            lines.append(f"\n✗ 변곡점에서 특별한 개선 없음")

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)
    print(report)
    return report


def create_inflection_plots(
    results: pd.DataFrame,
    output_dir: str
) -> None:
    """분석 결과 시각화"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        periods = ['all', 'summer', 'winter', 'transition']
        x = np.arange(len(periods))
        width = 0.35

        # 변곡점 MAPE
        demand_inflection = [results[(results['period'] == p) & (results['group'] == 'demand_only')]['inflection_mape'].mean() for p in periods]
        weather_inflection = [results[(results['period'] == p) & (results['group'] == 'weather_full')]['inflection_mape'].mean() for p in periods]

        ax1 = axes[0]
        ax1.bar(x - width/2, demand_inflection, width, label='Demand Only', color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width/2, weather_inflection, width, label='Weather Full', color='#45B7D1', alpha=0.8)
        ax1.set_xlabel('Period')
        ax1.set_ylabel('MAPE (%)')
        ax1.set_title('Inflection Point MAPE by Period')
        ax1.set_xticks(x)
        ax1.set_xticklabels(['All', 'Summer', 'Winter', 'Transition'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 일반구간 MAPE
        demand_normal = [results[(results['period'] == p) & (results['group'] == 'demand_only')]['normal_mape'].mean() for p in periods]
        weather_normal = [results[(results['period'] == p) & (results['group'] == 'weather_full')]['normal_mape'].mean() for p in periods]

        ax2 = axes[1]
        ax2.bar(x - width/2, demand_normal, width, label='Demand Only', color='#FF6B6B', alpha=0.8)
        ax2.bar(x + width/2, weather_normal, width, label='Weather Full', color='#45B7D1', alpha=0.8)
        ax2.set_xlabel('Period')
        ax2.set_ylabel('MAPE (%)')
        ax2.set_title('Normal Period MAPE by Period')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['All', 'Summer', 'Winter', 'Transition'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Inflection Point Analysis: Weather Variable Effect', fontsize=12, y=1.02)
        plt.tight_layout()

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        output_path = Path(output_dir) / 'inflection_point_analysis.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✓ Plot saved: {output_path}")

    except ImportError:
        print("\n⚠️ matplotlib not available. Skipping plots.")


# ============================================================
# 메인 함수
# ============================================================

def main(
    n_trials: int = 3,
    epochs: int = 50,
    percentile: float = 95,
    quick_test: bool = False
):
    """
    메인 분석 실행 함수

    Args:
        n_trials: 반복 횟수
        epochs: 학습 에포크
        percentile: 변곡점 임계값
        quick_test: 빠른 테스트 모드
    """
    config = DEFAULT_CONFIG.copy()

    if quick_test:
        n_trials = 2
        config['epochs'] = 10
        print("\n⚡ Quick test mode: 2 trials, 10 epochs")
    else:
        config['epochs'] = epochs

    # 분석 실행
    results = run_inflection_point_analysis(
        config=config,
        percentile=percentile,
        n_trials=n_trials,
        verbose=True
    )

    # 결과 분석
    summary = analyze_and_compare_results(results)
    report = print_analysis_report(results)

    # 결과 저장
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'inflection_point_analysis.csv'
    results.to_csv(results_path, index=False)
    print(f"\n✓ Results saved: {results_path}")

    # 시각화
    figure_dir = config['figure_dir']
    create_inflection_plots(results, str(figure_dir))

    # JSON 리포트
    # 전체 기간 기준 요약
    all_data = results[results['period'] == 'all']
    demand_inflection = all_data[all_data['group'] == 'demand_only']['inflection_mape'].mean()
    weather_inflection = all_data[all_data['group'] == 'weather_full']['inflection_mape'].mean()
    demand_normal = all_data[all_data['group'] == 'demand_only']['normal_mape'].mean()
    weather_normal = all_data[all_data['group'] == 'weather_full']['normal_mape'].mean()

    inflection_improvement = (demand_inflection - weather_inflection) / demand_inflection * 100
    normal_improvement = (demand_normal - weather_normal) / demand_normal * 100

    report_data = {
        'analysis': 'Inflection Point Analysis',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'n_trials': n_trials,
            'epochs': config['epochs'],
            'percentile': percentile,
            'horizon': config['horizon']
        },
        'summary': {
            'inflection_improvement_%': float(inflection_improvement),
            'normal_improvement_%': float(normal_improvement),
            'weather_helps_at_inflection': bool(inflection_improvement > normal_improvement)
        }
    }

    report_path = output_dir / 'inflection_point_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"✓ Report saved: {report_path}")

    return results, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Inflection Point Analysis')
    parser.add_argument('--trials', type=int, default=3, help='Number of trials')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--percentile', type=float, default=95, help='Inflection threshold percentile')
    parser.add_argument('--quick', action='store_true', help='Quick test mode')

    args = parser.parse_args()

    main(
        n_trials=args.trials,
        epochs=args.epochs,
        percentile=args.percentile,
        quick_test=args.quick
    )
