"""
Train/Test Data Distribution Analysis
=====================================
전력 수요 데이터의 Train/Test 분포 차이 분석

목적: R² 음수 원인 분석 (Scaling 범위 이탈 여부 확인)

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 프로젝트 설정
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'jeju_daily_dataset.csv'
OUTPUT_DIR = PROJECT_ROOT / 'results' / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 분할 비율 (학습 스크립트와 동일)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SEQUENCE_LENGTH = 14


def load_data():
    """데이터 로드"""
    df = pd.read_csv(DATA_PATH, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame) -> tuple:
    """Train/Val/Test 분할 (시퀀스 고려)"""
    n = len(df) - SEQUENCE_LENGTH
    train_end = int(n * TRAIN_RATIO) + SEQUENCE_LENGTH
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO)) + SEQUENCE_LENGTH
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df


def compute_statistics(df: pd.DataFrame, name: str) -> dict:
    """기술 통계량 계산"""
    power = df['power_sum']
    stats = {
        'name': name,
        'count': len(power),
        'mean': power.mean(),
        'std': power.std(),
        'min': power.min(),
        'max': power.max(),
        'median': power.median(),
        'date_start': df['date'].min(),
        'date_end': df['date'].max(),
    }
    return stats


def plot_timeseries(df: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """전체 시계열 + Train/Test 구간 표시"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 전체 시계열
    ax.plot(df['date'], df['power_sum'], color='gray', alpha=0.3, linewidth=0.5, label='All Data')
    
    # Train/Val/Test 구간
    ax.plot(train_df['date'], train_df['power_sum'], color='blue', alpha=0.7, linewidth=0.8, label=f'Train ({len(train_df):,})')
    ax.plot(val_df['date'], val_df['power_sum'], color='green', alpha=0.7, linewidth=0.8, label=f'Validation ({len(val_df):,})')
    ax.plot(test_df['date'], test_df['power_sum'], color='red', alpha=0.7, linewidth=0.8, label=f'Test ({len(test_df):,})')
    
    # Train Max 라인
    train_max = train_df['power_sum'].max()
    ax.axhline(y=train_max, color='blue', linestyle='--', alpha=0.5, label=f'Train Max: {train_max:,.0f}')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Power Demand (MW)')
    ax.set_title('Jeju Power Demand: Train/Val/Test Split Analysis')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'timeseries_split.png', dpi=150)
    print(f"✅ Saved: {OUTPUT_DIR / 'timeseries_split.png'}")
    plt.close()


def plot_distribution(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Train vs Test 분포 비교 히스토그램"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 히스토그램
    ax1 = axes[0]
    ax1.hist(train_df['power_sum'], bins=50, alpha=0.7, color='blue', label='Train', density=True)
    ax1.hist(test_df['power_sum'], bins=50, alpha=0.7, color='red', label='Test', density=True)
    ax1.axvline(train_df['power_sum'].max(), color='blue', linestyle='--', label=f'Train Max: {train_df["power_sum"].max():,.0f}')
    ax1.axvline(test_df['power_sum'].max(), color='red', linestyle='--', label=f'Test Max: {test_df["power_sum"].max():,.0f}')
    ax1.set_xlabel('Power Demand (MW)')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    
    # Box plot
    ax2 = axes[1]
    ax2.boxplot([train_df['power_sum'], test_df['power_sum']], labels=['Train', 'Test'])
    ax2.set_ylabel('Power Demand (MW)')
    ax2.set_title('Box Plot Comparison')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribution_comparison.png', dpi=150)
    print(f"✅ Saved: {OUTPUT_DIR / 'distribution_comparison.png'}")
    plt.close()


def plot_yearly_trend(df: pd.DataFrame):
    """연도별 평균 전력 수요 추세"""
    yearly = df.groupby(df['date'].dt.year)['power_sum'].agg(['mean', 'max', 'min']).reset_index()
    yearly.columns = ['year', 'mean', 'max', 'min']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(yearly['year'], yearly['mean'], marker='o', linewidth=2, label='Mean')
    ax.fill_between(yearly['year'], yearly['min'], yearly['max'], alpha=0.2, label='Min-Max Range')
    
    # Train/Test 구분선
    ax.axvline(x=2022.5, color='red', linestyle='--', alpha=0.7, label='Train/Test Split')
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Power Demand (MW)')
    ax.set_title('Yearly Power Demand Trend')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'yearly_trend.png', dpi=150)
    print(f"✅ Saved: {OUTPUT_DIR / 'yearly_trend.png'}")
    plt.close()


def main():
    print("="*60)
    print("📊 ANALYSIS-001: Train/Test Distribution Analysis")
    print("="*60)
    
    # 데이터 로드
    df = load_data()
    print(f"\n📂 Loaded {len(df):,} records")
    print(f"   Date range: {df['date'].min()} ~ {df['date'].max()}")
    
    # 데이터 분할
    train_df, val_df, test_df = split_data(df)
    
    # 기술 통계량 계산
    train_stats = compute_statistics(train_df, 'Train')
    val_stats = compute_statistics(val_df, 'Validation')
    test_stats = compute_statistics(test_df, 'Test')
    
    # 결과 출력
    print("\n" + "="*60)
    print("📋 STATISTICS COMPARISON")
    print("="*60)
    
    print(f"\n{'Metric':<15} {'Train':>15} {'Validation':>15} {'Test':>15}")
    print("-" * 60)
    print(f"{'Count':<15} {train_stats['count']:>15,} {val_stats['count']:>15,} {test_stats['count']:>15,}")
    print(f"{'Mean':<15} {train_stats['mean']:>15,.1f} {val_stats['mean']:>15,.1f} {test_stats['mean']:>15,.1f}")
    print(f"{'Std':<15} {train_stats['std']:>15,.1f} {val_stats['std']:>15,.1f} {test_stats['std']:>15,.1f}")
    print(f"{'Min':<15} {train_stats['min']:>15,.1f} {val_stats['min']:>15,.1f} {test_stats['min']:>15,.1f}")
    print(f"{'Max':<15} {train_stats['max']:>15,.1f} {val_stats['max']:>15,.1f} {test_stats['max']:>15,.1f}")
    print(f"{'Median':<15} {train_stats['median']:>15,.1f} {val_stats['median']:>15,.1f} {test_stats['median']:>15,.1f}")
    print("-" * 60)
    print(f"{'Date Start':<15} {str(train_stats['date_start'].date()):>15} {str(val_stats['date_start'].date()):>15} {str(test_stats['date_start'].date()):>15}")
    print(f"{'Date End':<15} {str(train_stats['date_end'].date()):>15} {str(val_stats['date_end'].date()):>15} {str(test_stats['date_end'].date()):>15}")
    
    # 스케일링 이탈 분석
    print("\n" + "="*60)
    print("⚠️ SCALING RANGE ANALYSIS")
    print("="*60)
    
    train_max = train_stats['max']
    test_max = test_stats['max']
    test_over_train = (test_df['power_sum'] > train_max).sum()
    test_over_ratio = test_over_train / len(test_df) * 100
    
    print(f"\n   Train Max: {train_max:,.1f} MW")
    print(f"   Test Max:  {test_max:,.1f} MW")
    print(f"   Test > Train Max: {test_over_train:,} records ({test_over_ratio:.1f}%)")
    
    if test_max > train_max:
        overflow = ((test_max - train_max) / train_max) * 100
        print(f"\n   ⚠️ SCALING OVERFLOW DETECTED!")
        print(f"   Test Max exceeds Train Max by {overflow:.1f}%")
        print(f"   → MinMaxScaler will produce values > 1.0 for Test data")
        print(f"   → LSTM cannot extrapolate beyond training range!")
    else:
        print(f"\n   ✅ No scaling overflow detected")
    
    # 시각화
    print("\n📈 Generating visualizations...")
    plot_timeseries(df, train_df, val_df, test_df)
    plot_distribution(train_df, test_df)
    plot_yearly_trend(df)
    
    # 결론
    print("\n" + "="*60)
    print("📝 CONCLUSION")
    print("="*60)
    
    mean_increase = ((test_stats['mean'] - train_stats['mean']) / train_stats['mean']) * 100
    print(f"\n   Mean increase (Train→Test): {mean_increase:+.1f}%")
    
    if mean_increase > 10:
        print(f"\n   🔴 CRITICAL: Significant distribution shift detected!")
        print(f"   Recommendation: Apply differencing or use expanding window validation")
    elif mean_increase > 5:
        print(f"\n   🟡 WARNING: Moderate distribution shift detected")
        print(f"   Recommendation: Consider TimeSeriesSplit cross-validation")
    else:
        print(f"\n   🟢 Distribution shift is within acceptable range")
    
    print(f"\n   Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
