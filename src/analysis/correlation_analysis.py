"""
피어슨 상관계수 히트맵 생성
DATA-001: 원본 데이터 품질 검사 및 EDA

Reference: JPD_RNN_Weather 논문에서 |r| > 0.5인 변수가 예측 성능에 영향
- 기온 (temperature)
- 지중온도 (ground temperature)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = ['AppleGothic', 'NanumGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_and_merge_data():
    """전력 데이터와 기상 데이터 병합"""
    base_path = Path(__file__).parent.parent.parent

    # 전력 데이터 로드
    power_path = base_path / "data/raw/jeju_hourly_power_2013_2024.csv"
    power_df = pd.read_csv(power_path, encoding='utf-8-sig')

    # datetime 컬럼 생성 (시간이 1-24이므로 0-23으로 변환)
    power_df['hour'] = power_df['시간'] - 1  # 1-24 -> 0-23
    power_df['datetime'] = pd.to_datetime(power_df['거래일자']) + pd.to_timedelta(power_df['hour'], unit='h')
    power_df = power_df.rename(columns={'전력거래량(MWh)': 'power_demand'})
    power_df = power_df[['datetime', 'power_demand']]

    # 기상 데이터 로드
    weather_path = base_path / "data/processed/jeju_weather_hourly_merged.csv"
    weather_df = pd.read_csv(weather_path, encoding='utf-8-sig')
    weather_df['datetime'] = pd.to_datetime(weather_df['일시'])

    # 분석에 필요한 기상 변수만 선택
    weather_cols = ['datetime', '기온', '강수량', '풍속', '풍향', '습도', '증기압',
                    '이슬점온도', '현지기압', '해면기압', '일조', '일사',
                    '지면온도', 'm005Te', 'm01Te', 'm02Te', 'm03Te']
    weather_df = weather_df[[c for c in weather_cols if c in weather_df.columns]]

    # 병합
    merged_df = pd.merge(power_df, weather_df, on='datetime', how='inner')
    print(f"[Data] Merged dataset: {len(merged_df)} rows")

    return merged_df


def create_correlation_heatmap(df, output_path):
    """피어슨 상관계수 히트맵 생성"""

    # 수치형 컬럼만 선택 (datetime 제외)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # 컬럼명 영문 매핑 (가독성)
    col_mapping = {
        'power_demand': 'Power Demand',
        '기온': 'Temperature',
        '강수량': 'Precipitation',
        '풍속': 'Wind Speed',
        '풍향': 'Wind Direction',
        '습도': 'Humidity',
        '증기압': 'Vapor Pressure',
        '이슬점온도': 'Dewpoint',
        '현지기압': 'Local Pressure',
        '해면기압': 'Sea Level Pressure',
        '일조': 'Sunshine',
        '일사': 'Solar Radiation',
        '지면온도': 'Ground Temp',
        'm005Te': 'Soil 5cm',
        'm01Te': 'Soil 10cm',
        'm02Te': 'Soil 20cm',
        'm03Te': 'Soil 30cm'
    }

    # 상관계수 계산
    corr_matrix = df[numeric_cols].corr(method='pearson')

    # 컬럼명 매핑 적용
    corr_matrix = corr_matrix.rename(index=col_mapping, columns=col_mapping)

    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(14, 12))

    # 마스크 (상삼각 행렬)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    # 히트맵 플롯
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8, 'label': 'Pearson Correlation'},
        ax=ax
    )

    ax.set_title('Pearson Correlation Heatmap: Jeju Power Demand vs Weather Variables\n(Reference: JPD_RNN_Weather, |r| > 0.5 indicates strong correlation)',
                 fontsize=12, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"[Saved] Correlation heatmap: {output_path}")

    return corr_matrix


def analyze_top_correlations(corr_matrix, target='Power Demand'):
    """전력 수요와 상관관계 높은 변수 분석"""

    if target not in corr_matrix.columns:
        print(f"[Warning] Target column '{target}' not found")
        return None

    # 전력 수요와의 상관계수
    power_corr = corr_matrix[target].drop(target).sort_values(key=abs, ascending=False)

    print("\n" + "=" * 60)
    print(f"Top Correlations with {target}")
    print("=" * 60)
    print(f"{'Variable':<25} {'Correlation':>12} {'Strength':>12}")
    print("-" * 60)

    for var, corr in power_corr.items():
        if abs(corr) >= 0.5:
            strength = "STRONG"
        elif abs(corr) >= 0.3:
            strength = "Moderate"
        else:
            strength = "Weak"

        print(f"{var:<25} {corr:>12.4f} {strength:>12}")

    print("=" * 60)

    # 논문 기준 |r| > 0.5 변수
    strong_vars = power_corr[abs(power_corr) >= 0.5].index.tolist()
    print(f"\n[Paper Reference] Variables with |r| >= 0.5: {strong_vars}")

    return power_corr


def save_correlation_csv(corr_matrix, output_path):
    """상관계수 행렬 CSV 저장"""
    corr_matrix.to_csv(output_path, encoding='utf-8-sig')
    print(f"[Saved] Correlation matrix CSV: {output_path}")


def main():
    """메인 실행"""
    base_path = Path(__file__).parent.parent.parent

    # 출력 디렉토리 생성
    figures_dir = base_path / "results/figures"
    metrics_dir = base_path / "results/metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DATA-001: Correlation Analysis")
    print("=" * 60)

    # 1. 데이터 로드 및 병합
    merged_df = load_and_merge_data()

    # 2. 상관계수 히트맵 생성
    heatmap_path = figures_dir / "correlation_heatmap.png"
    corr_matrix = create_correlation_heatmap(merged_df, heatmap_path)

    # 3. 전력 수요와의 상관관계 분석
    power_corr = analyze_top_correlations(corr_matrix)

    # 4. 상관계수 행렬 CSV 저장
    csv_path = metrics_dir / "correlation_matrix.csv"
    save_correlation_csv(corr_matrix, csv_path)

    print("\n[Complete] Correlation analysis finished!")

    return corr_matrix


if __name__ == "__main__":
    main()
