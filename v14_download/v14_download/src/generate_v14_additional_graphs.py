#!/usr/bin/env python3
"""
V14 추가 그래프 생성 스크립트
저장된 best_model.pt를 로드하여 추가 분석 그래프 생성
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

RESULTS_DIR = '/Users/ibkim/Ormi_1/power-demand-forecast/results'
DATA_DIR = '/Users/ibkim/Ormi_1/power-demand-forecast'

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# BiLSTM 모델 정의 (main 스크립트와 동일한 구조)
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.bn(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def load_and_preprocess_data():
    """데이터 로드 및 전처리 (main 스크립트와 동일)"""
    # 1. 전력 데이터 로드
    power_df = pd.read_csv(os.path.join(DATA_DIR, 'jeju_daily_power_2013_2024.csv'))
    power_df.columns = ['date', 'power_mwh']
    power_df['date'] = pd.to_datetime(power_df['date'])

    # 2. 기온 데이터 로드
    temp_df = pd.read_csv(os.path.join(DATA_DIR, 'jeju_Dtemp_2013_2025.csv'),
                          encoding='cp949', skiprows=7)
    temp_df.columns = ['date', 'station', 'avg_temp', 'min_temp', 'max_temp']
    temp_df['date'] = temp_df['date'].str.strip()
    temp_df['date'] = pd.to_datetime(temp_df['date'])
    for col in ['avg_temp', 'min_temp', 'max_temp']:
        temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    temp_df = temp_df[['date', 'avg_temp', 'min_temp', 'max_temp']]

    # 3. 일사량 데이터 로드
    sunlight1 = pd.read_csv(os.path.join(DATA_DIR, 'jeju_Dsunlight_2013_2022.csv'), encoding='cp949')
    sunlight1.columns = ['station', 'region', 'date', 'sunlight']
    sunlight2 = pd.read_csv(os.path.join(DATA_DIR, 'jeju_Dsunlight_2023_2025.csv'), encoding='cp949')
    sunlight2.columns = ['station', 'region', 'date', 'sunlight']
    sunlight_df = pd.concat([sunlight1, sunlight2], ignore_index=True)
    sunlight_df['date'] = pd.to_datetime(sunlight_df['date'])
    sunlight_df['sunlight'] = pd.to_numeric(sunlight_df['sunlight'], errors='coerce')
    sunlight_df = sunlight_df[['date', 'sunlight']]

    # 4. 이슬점 온도 데이터 로드
    dwpt1 = pd.read_csv(os.path.join(DATA_DIR, 'jeju_ DWPT_day_2013_2022.csv'), encoding='cp949')
    dwpt1.columns = ['station', 'region', 'date', 'dew_point']
    dwpt2 = pd.read_csv(os.path.join(DATA_DIR, 'jeju_ DWPT_day_2023_2025.csv'), encoding='cp949')
    dwpt2.columns = ['station', 'region', 'date', 'dew_point']
    dwpt_df = pd.concat([dwpt1, dwpt2], ignore_index=True)
    dwpt_df['date'] = pd.to_datetime(dwpt_df['date'])
    dwpt_df['dew_point'] = pd.to_numeric(dwpt_df['dew_point'], errors='coerce')
    dwpt_df = dwpt_df[['date', 'dew_point']]

    # 5. 관광객 데이터 로드
    visitors_df = pd.read_csv(os.path.join(DATA_DIR, 'jeju_daily_visitors_v10.csv'))
    visitors_df.columns = ['date', 'visitors', 'source', 'note']
    visitors_df['date'] = pd.to_datetime(visitors_df['date'])
    visitors_df['visitors'] = pd.to_numeric(visitors_df['visitors'], errors='coerce')
    visitors_df = visitors_df[['date', 'visitors']]

    # 6. 전기차 데이터 로드
    ev_df = pd.read_csv(os.path.join(DATA_DIR, 'jeju_ev_daily.csv'))
    ev_df.columns = ['date', 'ev_cumulative', 'ev_daily_new']
    ev_df['date'] = pd.to_datetime(ev_df['date'])

    # 7. 데이터 병합
    merged_df = power_df.copy()
    merged_df = merged_df.merge(temp_df, on='date', how='left')
    merged_df = merged_df.merge(sunlight_df, on='date', how='left')
    merged_df = merged_df.merge(dwpt_df, on='date', how='left')
    merged_df = merged_df.merge(visitors_df, on='date', how='left')
    merged_df = merged_df.merge(ev_df, on='date', how='left')

    # 8. 결측치 보간
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')
    merged_df = merged_df.fillna(method='bfill').fillna(method='ffill')

    # 이상치 Capping
    outlier_cols = ['power_mwh', 'visitors']
    for col in outlier_cols:
        if col in merged_df.columns:
            Q1 = merged_df[col].quantile(0.25)
            Q3 = merged_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            merged_df[col] = merged_df[col].clip(lower=lower_bound, upper=upper_bound)

    # 파생 변수 생성
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['day'] = merged_df['date'].dt.day
    merged_df['dayofweek'] = merged_df['date'].dt.dayofweek
    merged_df['is_weekend'] = (merged_df['dayofweek'] >= 5).astype(int)

    merged_df['CDD'] = merged_df['avg_temp'].apply(lambda x: max(0, x - 24))
    merged_df['HDD'] = merged_df['avg_temp'].apply(lambda x: max(0, 18 - x))

    merged_df['month_sin'] = np.sin(2 * np.pi * merged_df['month'] / 12)
    merged_df['month_cos'] = np.cos(2 * np.pi * merged_df['month'] / 12)

    # Lag 피처
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        merged_df[f'power_lag_{lag}'] = merged_df['power_mwh'].shift(lag)

    # Rolling 피처
    for window in [3, 7, 14, 21, 30]:
        merged_df[f'power_rolling_mean_{window}'] = merged_df['power_mwh'].rolling(window=window).mean()
        merged_df[f'power_rolling_std_{window}'] = merged_df['power_mwh'].rolling(window=window).std()
        merged_df[f'power_rolling_min_{window}'] = merged_df['power_mwh'].rolling(window=window).min()
        merged_df[f'power_rolling_max_{window}'] = merged_df['power_mwh'].rolling(window=window).max()

    merged_df['power_lag_365'] = merged_df['power_mwh'].shift(365)
    merged_df = merged_df.dropna().reset_index(drop=True)

    return merged_df

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def main():
    print("V14 추가 그래프 생성 시작...")

    # 데이터 로드
    df = load_and_preprocess_data()

    # 피처 선택 (V14 설정)
    feature_cols = [
        'power_rolling_mean_3', 'power_rolling_max_3', 'power_rolling_min_3',
        'power_rolling_mean_7', 'power_lag_1', 'power_rolling_max_7', 'power_rolling_min_7',
        'power_rolling_mean_14', 'power_rolling_max_14', 'power_rolling_min_14',
        'power_lag_2', 'power_lag_3', 'power_lag_7', 'year', 'power_lag_365',
        'CDD', 'HDD', 'month_sin', 'month_cos', 'is_weekend'
    ]
    feature_cols = [f for f in feature_cols if f in df.columns]

    # 데이터 분할
    train_end = '2022-12-31'
    val_end = '2023-12-31'

    train_df = df[df['date'] <= train_end].copy()
    val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
    test_df = df[df['date'] > val_end].copy()

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 스케일링
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaler_X.fit(train_df[feature_cols].values)
    scaler_y.fit(train_df['power_mwh'].values.reshape(-1, 1))

    X_test_scaled = scaler_X.transform(test_df[feature_cols].values)
    y_test_scaled = scaler_y.transform(test_df['power_mwh'].values.reshape(-1, 1)).flatten()

    # 시퀀스 생성
    seq_length = 7
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)

    # 모델 로드
    model = BiLSTMModel(
        input_dim=len(feature_cols),
        hidden_dim=96,
        num_layers=1,
        output_dim=1,
        dropout=0.1
    ).to(device)

    model.load_state_dict(torch.load(os.path.join(DATA_DIR, 'best_model.pt')))
    model.eval()

    # 예측
    X_test_t = torch.FloatTensor(X_test_seq).to(device)
    with torch.no_grad():
        predictions_scaled = model(X_test_t).cpu().numpy().flatten()

    # 역변환
    predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    # 날짜 추출
    test_dates = test_df['date'].values[seq_length:]
    test_dates = pd.to_datetime(test_dates)

    # 메트릭 계산
    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

    print(f"\n성능 지표:")
    print(f"  R²: {r2:.4f} ({r2*100:.2f}%)")
    print(f"  MAE: {mae:.2f} MWh")
    print(f"  RMSE: {rmse:.2f} MWh")
    print(f"  MAPE: {mape:.2f}%")

    # ============================================
    # 그래프 생성
    # ============================================

    # 1. 예측 vs 실제 시계열 그래프
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    ax1 = axes[0]
    ax1.plot(test_dates, actuals, 'b-', label='실제값', alpha=0.8, linewidth=1)
    ax1.plot(test_dates, predictions, 'r-', label='예측값', alpha=0.8, linewidth=1)
    ax1.fill_between(test_dates, actuals, predictions, alpha=0.3, color='gray')
    ax1.set_xlabel('날짜', fontsize=12)
    ax1.set_ylabel('전력 수요 (MWh)', fontsize=12)
    ax1.set_title(f'V14 BiLSTM: 실제값 vs 예측값 (R²={r2:.4f}, {r2*100:.2f}%)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    recent_days = 60
    ax2.plot(test_dates[-recent_days:], actuals[-recent_days:], 'b-o', label='실제값', alpha=0.8, linewidth=1.5, markersize=3)
    ax2.plot(test_dates[-recent_days:], predictions[-recent_days:], 'r-s', label='예측값', alpha=0.8, linewidth=1.5, markersize=3)
    ax2.set_xlabel('날짜', fontsize=12)
    ax2.set_ylabel('전력 수요 (MWh)', fontsize=12)
    ax2.set_title(f'V14: 최근 {recent_days}일 상세 (테스트 기간)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/v14_prediction_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  예측 vs 실제값 그래프 저장: {RESULTS_DIR}/v14_prediction_vs_actual.png")

    # 2. 산점도
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(actuals, predictions, alpha=0.5, s=20)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='이상적 예측선 (y=x)')
    ax.set_xlabel('실제값 (MWh)', fontsize=12)
    ax.set_ylabel('예측값 (MWh)', fontsize=12)
    ax.set_title(f'V14 BiLSTM: 예측 정확도 산점도\nR²={r2:.4f} ({r2*100:.2f}%)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/v14_scatter_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  산점도 저장: {RESULTS_DIR}/v14_scatter_plot.png")

    # 3. 잔차 분석
    residuals = actuals - predictions

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax1.axvline(x=residuals.mean(), color='g', linestyle='-', linewidth=2, label=f'평균: {residuals.mean():.2f}')
    ax1.set_xlabel('잔차 (MWh)', fontsize=12)
    ax1.set_ylabel('빈도', fontsize=12)
    ax1.set_title('V14: 잔차 분포 (히스토그램)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(test_dates, residuals, 'g-', alpha=0.7, linewidth=1)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.fill_between(test_dates, 0, residuals, alpha=0.3, color='green')
    ax2.set_xlabel('날짜', fontsize=12)
    ax2.set_ylabel('잔차 (MWh)', fontsize=12)
    ax2.set_title('V14: 잔차 시계열', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/v14_residual_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  잔차 분석 그래프 저장: {RESULTS_DIR}/v14_residual_analysis.png")

    # 4. 월별 성능 분석
    months = test_dates.month

    monthly_metrics = []
    for month in range(1, 13):
        mask = months == month
        if mask.sum() > 0:
            monthly_mae = mean_absolute_error(actuals[mask], predictions[mask])
            monthly_r2 = r2_score(actuals[mask], predictions[mask]) if mask.sum() > 1 else 0
            monthly_mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
            monthly_metrics.append({
                'month': month,
                'mae': monthly_mae,
                'r2': monthly_r2,
                'mape': monthly_mape,
                'count': mask.sum()
            })

    monthly_df = pd.DataFrame(monthly_metrics)

    if len(monthly_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        ax1 = axes[0]
        ax1.bar(monthly_df['month'], monthly_df['mae'], color='steelblue', edgecolor='black')
        ax1.set_xlabel('월', fontsize=12)
        ax1.set_ylabel('MAE (MWh)', fontsize=12)
        ax1.set_title('V14: 월별 MAE', fontsize=14, fontweight='bold')
        ax1.set_xticks(monthly_df['month'])
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        colors = ['green' if r2 >= 0.7 else 'orange' if r2 >= 0.5 else 'red' for r2 in monthly_df['r2']]
        ax2.bar(monthly_df['month'], monthly_df['r2'], color=colors, edgecolor='black')
        ax2.axhline(y=0.75, color='r', linestyle='--', linewidth=2, label='목표 R²=0.75')
        ax2.set_xlabel('월', fontsize=12)
        ax2.set_ylabel('R²', fontsize=12)
        ax2.set_title('V14: 월별 R²', fontsize=14, fontweight='bold')
        ax2.set_xticks(monthly_df['month'])
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        ax3 = axes[2]
        ax3.bar(monthly_df['month'], monthly_df['mape'], color='coral', edgecolor='black')
        ax3.set_xlabel('월', fontsize=12)
        ax3.set_ylabel('MAPE (%)', fontsize=12)
        ax3.set_title('V14: 월별 MAPE', fontsize=14, fontweight='bold')
        ax3.set_xticks(monthly_df['month'])
        ax3.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/v14_monthly_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  월별 성능 그래프 저장: {RESULTS_DIR}/v14_monthly_performance.png")

    # 5. 요일별 성능 분석
    dayofweek = test_dates.dayofweek

    daily_metrics = []
    day_names = ['월', '화', '수', '목', '금', '토', '일']
    for day in range(7):
        mask = dayofweek == day
        if mask.sum() > 0:
            daily_mae = mean_absolute_error(actuals[mask], predictions[mask])
            daily_r2 = r2_score(actuals[mask], predictions[mask]) if mask.sum() > 1 else 0
            daily_metrics.append({
                'day': day,
                'day_name': day_names[day],
                'mae': daily_mae,
                'r2': daily_r2,
                'count': mask.sum()
            })

    daily_df = pd.DataFrame(daily_metrics)

    if len(daily_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        colors = ['coral' if d in [5, 6] else 'steelblue' for d in daily_df['day']]
        ax1.bar(daily_df['day_name'], daily_df['mae'], color=colors, edgecolor='black')
        ax1.set_xlabel('요일', fontsize=12)
        ax1.set_ylabel('MAE (MWh)', fontsize=12)
        ax1.set_title('V14: 요일별 MAE (주황=주말)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')

        ax2 = axes[1]
        colors = ['coral' if d in [5, 6] else 'steelblue' for d in daily_df['day']]
        ax2.bar(daily_df['day_name'], daily_df['r2'], color=colors, edgecolor='black')
        ax2.axhline(y=0.75, color='r', linestyle='--', linewidth=2, label='목표 R²=0.75')
        ax2.set_xlabel('요일', fontsize=12)
        ax2.set_ylabel('R²', fontsize=12)
        ax2.set_title('V14: 요일별 R² (주황=주말)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(f'{RESULTS_DIR}/v14_daily_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  요일별 성능 그래프 저장: {RESULTS_DIR}/v14_daily_performance.png")

    print("\n" + "=" * 60)
    print("V14 추가 그래프 생성 완료!")
    print("=" * 60)
    print(f"\n생성된 파일:")
    print(f"  - {RESULTS_DIR}/v14_prediction_vs_actual.png")
    print(f"  - {RESULTS_DIR}/v14_scatter_plot.png")
    print(f"  - {RESULTS_DIR}/v14_residual_analysis.png")
    print(f"  - {RESULTS_DIR}/v14_monthly_performance.png")
    print(f"  - {RESULTS_DIR}/v14_daily_performance.png")

if __name__ == "__main__":
    main()
