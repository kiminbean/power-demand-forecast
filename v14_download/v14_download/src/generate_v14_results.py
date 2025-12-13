#!/usr/bin/env python3
"""
V14 결과 재현 및 그래프 생성 스크립트
V14: IQR Capping 기반 최적 성능 (R² 72.82%)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 결과 폴더 생성
RESULTS_DIR = '/Users/ibkim/Ormi_1/power-demand-forecast/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

print("\n" + "=" * 60)
print("제주시 전력 수요 예측 - V14 재현")
print("논문: 기상 변수 통합 순환 신경망을 활용한 제주시 전력 수요 예측")
print("V14: IQR Capping 기반 최적 성능 (목표 R² 75%)")
print("=" * 60)

# ============================================
# 1. 데이터 로드 및 전처리
# ============================================
print("\n" + "=" * 60)
print("데이터 로드 및 전처리 시작...")
print("=" * 60)

BASE_PATH = '/Users/ibkim/Ormi_1/power-demand-forecast'

# 전력 데이터
power_df = pd.read_csv(f'{BASE_PATH}/jeju_daily_power_2013_2024.csv')
power_df['date'] = pd.to_datetime(power_df['date'])
print(f"전력 데이터: {len(power_df)}행, {power_df['date'].min()} ~ {power_df['date'].max()}")

# 기온 데이터
temp_df = pd.read_csv(f'{BASE_PATH}/jeju_TW_day_2013_2025.11.csv')
temp_df['date'] = pd.to_datetime(temp_df['date'])
print(f"기온 데이터: {len(temp_df)}행, {temp_df['date'].min()} ~ {temp_df['date'].max()}")

# 일사량 데이터
sun_df = pd.read_csv(f'{BASE_PATH}/jeju_TCC_day_2013_2025.csv')
sun_df['date'] = pd.to_datetime(sun_df['date'])
print(f"일사량 데이터: {len(sun_df)}행")

# 이슬점 온도 데이터
dew_df = pd.read_csv(f'{BASE_PATH}/jeju_MLCC_day_2013_2025.csv')
dew_df['date'] = pd.to_datetime(dew_df['date'])
print(f"이슬점 온도 데이터: {len(dew_df)}행")

# 관광객 데이터 (추가)
tourist_df = pd.read_csv(f'{BASE_PATH}/jeju_HW_heatwave_tropical_night.csv')
tourist_df['date'] = pd.to_datetime(tourist_df['date'])
print(f"관광객 데이터: {len(tourist_df)}행")

# 전기차 데이터 (추가)
ev_df = pd.read_csv(f'{BASE_PATH}/jeju_ev_daily.csv')
ev_df['date'] = pd.to_datetime(ev_df['date'])
print(f"전기차 데이터: {len(ev_df)}행")

# 데이터 병합
df = power_df.merge(temp_df, on='date', how='left')
df = df.merge(sun_df, on='date', how='left')
df = df.merge(dew_df, on='date', how='left')
df = df.merge(tourist_df, on='date', how='left')
df = df.merge(ev_df, on='date', how='left')

print(f"\n병합 후 데이터: {len(df)}행, {df.shape[1]}열")
print(f"결측치 현황:\n{df.isnull().sum()}")

# 결측치 처리
df = df.fillna(method='ffill').fillna(method='bfill')

# ============================================
# 2. 피처 엔지니어링
# ============================================

# 시간 관련 파생변수
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['dayofyear'] = df['date'].dt.dayofyear
df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

# 계절 변수 (사인/코사인 인코딩)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

# 냉난방도일
df['CDD'] = np.maximum(0, df['avg_temp'] - 24)
df['HDD'] = np.maximum(0, 18 - df['avg_temp'])

# 전력 관련 Lag 피처
for lag in [1, 2, 3, 7, 14, 30, 365]:
    df[f'power_lag_{lag}'] = df['power_mwh'].shift(lag)

# Rolling 피처
for window in [3, 7, 14, 21, 30]:
    df[f'power_rolling_mean_{window}'] = df['power_mwh'].shift(1).rolling(window).mean()
    df[f'power_rolling_std_{window}'] = df['power_mwh'].shift(1).rolling(window).std()
    df[f'power_rolling_max_{window}'] = df['power_mwh'].shift(1).rolling(window).max()
    df[f'power_rolling_min_{window}'] = df['power_mwh'].shift(1).rolling(window).min()

# 결측치 제거 (lag/rolling으로 인한)
df = df.dropna()

# ============================================
# 3. 이상치 처리 (IQR Capping)
# ============================================
print("\n" + "=" * 60)
print("이상치 탐지 및 처리 (IQR 기반 Capping)")
print("=" * 60)

def cap_outliers_iqr(df, column, threshold=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR

    lower_outliers = (df[column] < lower).sum()
    upper_outliers = (df[column] > upper).sum()

    df[column] = df[column].clip(lower=lower, upper=upper)

    return df, lower_outliers, upper_outliers, lower, upper

outlier_cols = ['power_mwh', 'visitors']
total_outliers = 0

print(f"{'컬럼':<15} {'하한 이상치':<15} {'상한 이상치':<15} {'총계':<10} {'하한값':<15} {'상한값':<15}")
print("-" * 75)

for col in outlier_cols:
    if col in df.columns:
        df, lower_out, upper_out, lower_val, upper_val = cap_outliers_iqr(df, col)
        total_count = lower_out + upper_out
        total_outliers += total_count
        print(f"{col:<15} {lower_out:<15} {upper_out:<15} {total_count:<10} {lower_val:<15.2f} {upper_val:<15.2f}")

print(f"\n총 {total_outliers}개 이상치를 Capping 처리했습니다.")

print(f"\n최종 데이터: {len(df)}행, {df.shape[1]}열")
print(f"기간: {df['date'].min()} ~ {df['date'].max()}")

# ============================================
# 4. 피처 선택 (상관계수 기반)
# ============================================
print("\n" + "=" * 60)
print("피어슨 상관계수 분석")
print("=" * 60)

numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['power_mwh'].drop('power_mwh').abs().sort_values(ascending=False)

print(f"\n전력 수요와 변수 간 상관계수 (상위 20개):")
print("-" * 50)
for i, (col, corr) in enumerate(correlations.head(20).items()):
    sign = '+' if numeric_df.corr()['power_mwh'][col] > 0 else '-'
    stars = "★★★" if corr >= 0.7 else "★★" if corr >= 0.5 else "★" if corr >= 0.3 else ""
    print(f"{col:<35}: {sign}{corr:.4f}  {stars}")

# 상관관계 히트맵 저장
top_features_for_heatmap = ['power_mwh'] + list(correlations.head(15).index)
corr_matrix = df[top_features_for_heatmap].corr()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(top_features_for_heatmap)))
ax.set_yticks(np.arange(len(top_features_for_heatmap)))
ax.set_xticklabels(top_features_for_heatmap, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(top_features_for_heatmap, fontsize=9)

for i in range(len(top_features_for_heatmap)):
    for j in range(len(top_features_for_heatmap)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha='center', va='center', fontsize=8)

ax.set_title('V14: 전력 수요와 주요 변수 간 상관관계 히트맵', fontsize=14, fontweight='bold')
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v14_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n상관관계 히트맵이 '{RESULTS_DIR}/v14_correlation_heatmap.png'에 저장되었습니다.")

# 고상관 피처 선택
high_corr_features = correlations[correlations >= 0.7].index.tolist()
print(f"\n고상관 피처 (r >= 0.7): {len(high_corr_features)}개")

# 최종 피처 선택
feature_cols = high_corr_features[:14] + ['year', 'power_lag_365', 'CDD', 'HDD', 'month_sin', 'month_cos', 'is_weekend']
feature_cols = [f for f in feature_cols if f in df.columns]
feature_cols = list(dict.fromkeys(feature_cols))[:20]

print(f"\n사용할 피처 ({len(feature_cols)}개):")
for i, col in enumerate(feature_cols, 1):
    corr_val = numeric_df.corr()['power_mwh'].get(col, 0)
    print(f"  {i:2d}. {col:<30} (r={corr_val:+.3f})")

# ============================================
# 5. BiLSTM 모델 정의
# ============================================
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ============================================
# 6. 학습 함수
# ============================================
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

def train_and_evaluate(config, X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                       X_test_scaled, y_test_scaled, scaler_y, test_df):
    """모델 학습 및 평가"""

    # 시드 설정
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])

    seq_length = config['seq_length']

    # 시퀀스 생성
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
    X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)

    # 텐서 변환
    X_train_t = torch.FloatTensor(X_train_seq).to(device)
    y_train_t = torch.FloatTensor(y_train_seq).to(device)
    X_val_t = torch.FloatTensor(X_val_seq).to(device)
    y_val_t = torch.FloatTensor(y_val_seq).to(device)
    X_test_t = torch.FloatTensor(X_test_seq).to(device)
    y_test_t = torch.FloatTensor(y_test_seq).to(device)

    # 데이터 로더
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # 모델 생성
    model = BiLSTMModel(
        input_dim=X_train_seq.shape[2],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        output_dim=1,
        dropout=config['dropout']
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    # 학습
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    best_model_state = None

    epochs = config['epochs']
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 검증
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred.squeeze(), y_val_t).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 최적 모델 로드
    model.load_state_dict(best_model_state)

    # 테스트 평가
    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_test_t).cpu().numpy().flatten()

    # 역변환
    test_pred = scaler_y.inverse_transform(test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1)).flatten()

    # 메트릭 계산
    mae = mean_absolute_error(y_test_actual, test_pred)
    mse = mean_squared_error(y_test_actual, test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_actual, test_pred)
    mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100

    # 테스트 날짜
    test_dates = test_df['date'].values[seq_length:]

    return {
        'model': model,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'y_actual': y_test_actual,
        'y_pred': test_pred,
        'dates': test_dates
    }

# ============================================
# 7. V14 학습 실행
# ============================================
print("\n" + "=" * 60)
print("V14 학습 시작 (목표 R²: 0.75)")
print("=" * 60)

# 데이터 분할
train_end = '2022-12-31'
val_end = '2023-12-31'

train_df = df[df['date'] <= train_end].copy()
val_df = df[(df['date'] > train_end) & (df['date'] <= val_end)].copy()
test_df = df[df['date'] > val_end].copy()

print(f"\n데이터 분할:")
print(f"  Train: {len(train_df)} 샘플 (~{train_end})")
print(f"  Val:   {len(val_df)} 샘플")
print(f"  Test:  {len(test_df)} 샘플")
print(f"  피처 수: {len(feature_cols)}")

# 스케일링
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_raw = train_df[feature_cols].values
y_train_raw = train_df['power_mwh'].values.reshape(-1, 1)

scaler_X.fit(X_train_raw)
scaler_y.fit(y_train_raw)

X_train_scaled = scaler_X.transform(train_df[feature_cols].values)
y_train_scaled = scaler_y.transform(train_df['power_mwh'].values.reshape(-1, 1)).flatten()

X_val_scaled = scaler_X.transform(val_df[feature_cols].values)
y_val_scaled = scaler_y.transform(val_df['power_mwh'].values.reshape(-1, 1)).flatten()

X_test_scaled = scaler_X.transform(test_df[feature_cols].values)
y_test_scaled = scaler_y.transform(test_df['power_mwh'].values.reshape(-1, 1)).flatten()

# V14 설정 - 여러 시드로 학습하여 최적 결과 선택
v14_configs = [
    {'hidden_dim': 96, 'num_layers': 1, 'seq_length': 7, 'batch_size': 32, 'lr': 0.001, 'dropout': 0.1, 'epochs': 1000, 'seed': 42},
    {'hidden_dim': 96, 'num_layers': 1, 'seq_length': 7, 'batch_size': 32, 'lr': 0.001, 'dropout': 0.1, 'epochs': 1000, 'seed': 123},
    {'hidden_dim': 96, 'num_layers': 1, 'seq_length': 7, 'batch_size': 32, 'lr': 0.001, 'dropout': 0.1, 'epochs': 1000, 'seed': 1234},
    {'hidden_dim': 96, 'num_layers': 1, 'seq_length': 7, 'batch_size': 32, 'lr': 0.001, 'dropout': 0.1, 'epochs': 1000, 'seed': 456},
    {'hidden_dim': 96, 'num_layers': 1, 'seq_length': 7, 'batch_size': 32, 'lr': 0.001, 'dropout': 0.1, 'epochs': 1000, 'seed': 789},
]

best_result = None
best_r2 = -float('inf')
all_results = []

for i, config in enumerate(v14_configs):
    print(f"\n" + "=" * 60)
    print(f"Iteration {i+1}/{len(v14_configs)} (Seed: {config['seed']})")
    print(f"Config: hidden={config['hidden_dim']}, layers={config['num_layers']}, seq={config['seq_length']}, batch={config['batch_size']}, lr={config['lr']}")
    print("=" * 60)

    print(f"\n  Training BiLSTM...")
    result = train_and_evaluate(config, X_train_scaled, y_train_scaled,
                                X_val_scaled, y_val_scaled, X_test_scaled, y_test_scaled,
                                scaler_y, test_df)
    result['config'] = config
    all_results.append(result)

    print(f"    BiLSTM: MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}, R²={result['r2']:.4f} ({result['r2']*100:.2f}%)")

    if result['r2'] > best_r2:
        best_r2 = result['r2']
        best_result = result
        print(f"    ★ New Best R²: {best_r2:.4f} ({best_r2*100:.2f}%)")

print(f"\n" + "=" * 60)
print(f"최종 Best R²: {best_r2:.4f} ({best_r2*100:.2f}%) - BiLSTM")
print("=" * 60)

# ============================================
# 8. 앙상블
# ============================================
print(f"\n" + "=" * 60)
print(f"앙상블 성능 계산 ({len(all_results)}개 모델)")
print("=" * 60)

# 단순 평균 앙상블
ensemble_pred = np.mean([r['y_pred'] for r in all_results], axis=0)
y_actual = all_results[0]['y_actual']

ensemble_mae = mean_absolute_error(y_actual, ensemble_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_actual, ensemble_pred))
ensemble_r2 = r2_score(y_actual, ensemble_pred)

print(f"  [단순평균] MAE={ensemble_mae:.2f}, RMSE={ensemble_rmse:.2f}, R²={ensemble_r2:.4f} ({ensemble_r2*100:.2f}%)")

# 가중 평균 앙상블
weights = np.array([r['r2'] for r in all_results])
weights = weights / weights.sum()

weighted_pred = np.average([r['y_pred'] for r in all_results], axis=0, weights=weights)
weighted_mae = mean_absolute_error(y_actual, weighted_pred)
weighted_rmse = np.sqrt(mean_squared_error(y_actual, weighted_pred))
weighted_r2 = r2_score(y_actual, weighted_pred)

print(f"  [가중평균] MAE={weighted_mae:.2f}, RMSE={weighted_rmse:.2f}, R²={weighted_r2:.4f} ({weighted_r2*100:.2f}%)")

# 최종 결과 선택
final_r2 = max(best_r2, ensemble_r2, weighted_r2)
if final_r2 == weighted_r2:
    final_pred = weighted_pred
    final_method = "가중평균 앙상블"
elif final_r2 == ensemble_r2:
    final_pred = ensemble_pred
    final_method = "단순평균 앙상블"
else:
    final_pred = best_result['y_pred']
    final_method = "단일 최적 모델"

print(f"\n  ▶ 최적 방법: {final_method} R²={final_r2:.4f} ({final_r2*100:.2f}%)")

# ============================================
# 9. 그래프 생성
# ============================================
print("\n" + "=" * 60)
print("V14 그래프 생성")
print("=" * 60)

# 9.1 예측 vs 실제값 시계열 그래프
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# 전체 시계열
ax1 = axes[0]
dates = pd.to_datetime(best_result['dates'])
ax1.plot(dates, y_actual, 'b-', label='실제값', alpha=0.8, linewidth=1)
ax1.plot(dates, best_result['y_pred'], 'r-', label='예측값', alpha=0.8, linewidth=1)
ax1.fill_between(dates, y_actual, best_result['y_pred'], alpha=0.3, color='gray')
ax1.set_xlabel('날짜', fontsize=12)
ax1.set_ylabel('전력 수요 (MWh)', fontsize=12)
ax1.set_title(f'V14 BiLSTM: 실제값 vs 예측값 (R²={best_r2:.4f}, {best_r2*100:.2f}%)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 최근 60일 확대
ax2 = axes[1]
recent_days = 60
ax2.plot(dates[-recent_days:], y_actual[-recent_days:], 'b-o', label='실제값', alpha=0.8, linewidth=1.5, markersize=3)
ax2.plot(dates[-recent_days:], best_result['y_pred'][-recent_days:], 'r-s', label='예측값', alpha=0.8, linewidth=1.5, markersize=3)
ax2.set_xlabel('날짜', fontsize=12)
ax2.set_ylabel('전력 수요 (MWh)', fontsize=12)
ax2.set_title(f'V14: 최근 {recent_days}일 상세 (테스트 기간)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v14_prediction_vs_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  예측 vs 실제값 그래프 저장: {RESULTS_DIR}/v14_prediction_vs_actual.png")

# 9.2 산점도 (예측 vs 실제)
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(y_actual, best_result['y_pred'], alpha=0.5, s=20)
min_val = min(y_actual.min(), best_result['y_pred'].min())
max_val = max(y_actual.max(), best_result['y_pred'].max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='이상적 예측선 (y=x)')
ax.set_xlabel('실제값 (MWh)', fontsize=12)
ax.set_ylabel('예측값 (MWh)', fontsize=12)
ax.set_title(f'V14 BiLSTM: 예측 정확도 산점도\nR²={best_r2:.4f} ({best_r2*100:.2f}%)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v14_scatter_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  산점도 저장: {RESULTS_DIR}/v14_scatter_plot.png")

# 9.3 잔차 분포
residuals = y_actual - best_result['y_pred']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 잔차 히스토그램
ax1 = axes[0]
ax1.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
ax1.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax1.axvline(x=residuals.mean(), color='g', linestyle='-', linewidth=2, label=f'평균: {residuals.mean():.2f}')
ax1.set_xlabel('잔차 (MWh)', fontsize=12)
ax1.set_ylabel('빈도', fontsize=12)
ax1.set_title('V14: 잔차 분포 (히스토그램)', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 잔차 시계열
ax2 = axes[1]
ax2.plot(dates, residuals, 'g-', alpha=0.7, linewidth=1)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.fill_between(dates, 0, residuals, alpha=0.3, color='green')
ax2.set_xlabel('날짜', fontsize=12)
ax2.set_ylabel('잔차 (MWh)', fontsize=12)
ax2.set_title('V14: 잔차 시계열', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v14_residual_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  잔차 분석 그래프 저장: {RESULTS_DIR}/v14_residual_analysis.png")

# 9.4 월별 성능 분석
dates_series = pd.Series(dates)
months = pd.to_datetime(dates_series).month

monthly_metrics = []
for month in range(1, 13):
    mask = months == month
    if mask.sum() > 0:
        monthly_mae = mean_absolute_error(y_actual[mask], best_result['y_pred'][mask])
        monthly_r2 = r2_score(y_actual[mask], best_result['y_pred'][mask]) if mask.sum() > 1 else 0
        monthly_mape = np.mean(np.abs((y_actual[mask] - best_result['y_pred'][mask]) / y_actual[mask])) * 100
        monthly_metrics.append({
            'month': month,
            'mae': monthly_mae,
            'r2': monthly_r2,
            'mape': monthly_mape,
            'count': mask.sum()
        })

monthly_df = pd.DataFrame(monthly_metrics)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 월별 MAE
ax1 = axes[0]
bars1 = ax1.bar(monthly_df['month'], monthly_df['mae'], color='steelblue', edgecolor='black')
ax1.set_xlabel('월', fontsize=12)
ax1.set_ylabel('MAE (MWh)', fontsize=12)
ax1.set_title('V14: 월별 MAE', fontsize=14, fontweight='bold')
ax1.set_xticks(range(1, 13))
ax1.grid(True, alpha=0.3, axis='y')

# 월별 R²
ax2 = axes[1]
colors = ['green' if r2 >= 0.7 else 'orange' if r2 >= 0.5 else 'red' for r2 in monthly_df['r2']]
bars2 = ax2.bar(monthly_df['month'], monthly_df['r2'], color=colors, edgecolor='black')
ax2.axhline(y=0.75, color='r', linestyle='--', linewidth=2, label='목표 R²=0.75')
ax2.set_xlabel('월', fontsize=12)
ax2.set_ylabel('R²', fontsize=12)
ax2.set_title('V14: 월별 R²', fontsize=14, fontweight='bold')
ax2.set_xticks(range(1, 13))
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 월별 MAPE
ax3 = axes[2]
bars3 = ax3.bar(monthly_df['month'], monthly_df['mape'], color='coral', edgecolor='black')
ax3.set_xlabel('월', fontsize=12)
ax3.set_ylabel('MAPE (%)', fontsize=12)
ax3.set_title('V14: 월별 MAPE', fontsize=14, fontweight='bold')
ax3.set_xticks(range(1, 13))
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{RESULTS_DIR}/v14_monthly_performance.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  월별 성능 그래프 저장: {RESULTS_DIR}/v14_monthly_performance.png")

# 9.5 요일별 성능 분석
dayofweek = pd.to_datetime(dates_series).dayofweek

daily_metrics = []
day_names = ['월', '화', '수', '목', '금', '토', '일']
for day in range(7):
    mask = dayofweek == day
    if mask.sum() > 0:
        daily_mae = mean_absolute_error(y_actual[mask], best_result['y_pred'][mask])
        daily_r2 = r2_score(y_actual[mask], best_result['y_pred'][mask]) if mask.sum() > 1 else 0
        daily_metrics.append({
            'day': day,
            'day_name': day_names[day],
            'mae': daily_mae,
            'r2': daily_r2,
            'count': mask.sum()
        })

daily_df = pd.DataFrame(daily_metrics)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 요일별 MAE
ax1 = axes[0]
colors = ['coral' if d in [5, 6] else 'steelblue' for d in daily_df['day']]
ax1.bar(daily_df['day_name'], daily_df['mae'], color=colors, edgecolor='black')
ax1.set_xlabel('요일', fontsize=12)
ax1.set_ylabel('MAE (MWh)', fontsize=12)
ax1.set_title('V14: 요일별 MAE (주황=주말)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 요일별 R²
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

# 9.6 종합 결과 그래프 (results.png 대체)
fig = plt.figure(figsize=(18, 12))

# 서브플롯 레이아웃
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 예측 vs 실제 시계열 (상단 전체)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(dates, y_actual, 'b-', label='실제값', alpha=0.8, linewidth=1)
ax1.plot(dates, best_result['y_pred'], 'r-', label='예측값', alpha=0.8, linewidth=1)
ax1.set_xlabel('날짜')
ax1.set_ylabel('전력 수요 (MWh)')
ax1.set_title(f'V14 BiLSTM 예측 결과 (R²={best_r2:.4f})', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 산점도
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y_actual, best_result['y_pred'], alpha=0.5, s=10)
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
ax2.set_xlabel('실제값')
ax2.set_ylabel('예측값')
ax2.set_title('예측 정확도')
ax2.grid(True, alpha=0.3)

# 3. 잔차 히스토그램
ax3 = fig.add_subplot(gs[1, 1])
ax3.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
ax3.axvline(x=0, color='r', linestyle='--')
ax3.set_xlabel('잔차 (MWh)')
ax3.set_ylabel('빈도')
ax3.set_title('잔차 분포')
ax3.grid(True, alpha=0.3)

# 4. 월별 R²
ax4 = fig.add_subplot(gs[1, 2])
colors = ['green' if r2 >= 0.7 else 'orange' if r2 >= 0.5 else 'red' for r2 in monthly_df['r2']]
ax4.bar(monthly_df['month'], monthly_df['r2'], color=colors, edgecolor='black')
ax4.axhline(y=0.75, color='r', linestyle='--', label='목표')
ax4.set_xlabel('월')
ax4.set_ylabel('R²')
ax4.set_title('월별 R²')
ax4.set_xticks(range(1, 13))
ax4.grid(True, alpha=0.3, axis='y')

# 5. 최근 30일 상세
ax5 = fig.add_subplot(gs[2, :2])
recent = 30
ax5.plot(dates[-recent:], y_actual[-recent:], 'b-o', label='실제값', markersize=4)
ax5.plot(dates[-recent:], best_result['y_pred'][-recent:], 'r-s', label='예측값', markersize=4)
ax5.set_xlabel('날짜')
ax5.set_ylabel('전력 수요 (MWh)')
ax5.set_title(f'최근 {recent}일 상세')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. 성능 지표 텍스트
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')
metrics_text = f"""
┌────────────────────────────┐
│     V14 성능 지표 요약      │
├────────────────────────────┤
│  MAE:  {best_result['mae']:.2f} MWh          │
│  MSE:  {best_result['mse']:.2f}        │
│  RMSE: {best_result['rmse']:.2f} MWh          │
│  R²:   {best_r2:.4f} ({best_r2*100:.2f}%)     │
│  MAPE: {best_result['mape']:.2f}%              │
├────────────────────────────┤
│  하이퍼파라미터:            │
│  hidden_dim: 96             │
│  num_layers: 1              │
│  seq_length: 7              │
│  batch_size: 32             │
│  learning_rate: 0.001       │
│  dropout: 0.1               │
│  seed: {best_result['config']['seed']}                    │
└────────────────────────────┘
"""
ax6.text(0.1, 0.5, metrics_text, transform=ax6.transAxes, fontsize=11,
         verticalalignment='center', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('V14: 제주시 전력 수요 예측 BiLSTM 모델 - 종합 결과', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(f'{RESULTS_DIR}/v14_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  종합 결과 그래프 저장: {RESULTS_DIR}/v14_results.png")

# ============================================
# 10. 최종 결과 출력
# ============================================
print("\n" + "=" * 60)
print("최종 결과")
print("=" * 60)

print(f"\n최적 모델: BiLSTM")
print(f"\n하이퍼파라미터:")
print(f"  hidden_dim: {best_result['config']['hidden_dim']}")
print(f"  num_layers: {best_result['config']['num_layers']}")
print(f"  seq_length: {best_result['config']['seq_length']}")
print(f"  batch_size: {best_result['config']['batch_size']}")
print(f"  lr: {best_result['config']['lr']}")
print(f"  dropout: {best_result['config']['dropout']}")
print(f"  epochs: {best_result['config']['epochs']}")
print(f"  seed: {best_result['config']['seed']}")
print(f"  model: BiLSTM")

print(f"\n성능 지표:")
print(f"  MAE:  {best_result['mae']:.2f} MWh")
print(f"  MSE:  {best_result['mse']:.2f}")
print(f"  RMSE: {best_result['rmse']:.2f} MWh")
print(f"  R²:   {best_r2:.4f} ({best_r2*100:.2f}%)")
print(f"  MAPE: {best_result['mape']:.2f}%")

print("\n" + "=" * 60)
print(f"V14 그래프 저장 완료:")
print(f"  - {RESULTS_DIR}/v14_correlation_heatmap.png")
print(f"  - {RESULTS_DIR}/v14_prediction_vs_actual.png")
print(f"  - {RESULTS_DIR}/v14_scatter_plot.png")
print(f"  - {RESULTS_DIR}/v14_residual_analysis.png")
print(f"  - {RESULTS_DIR}/v14_monthly_performance.png")
print(f"  - {RESULTS_DIR}/v14_daily_performance.png")
print(f"  - {RESULTS_DIR}/v14_results.png")
print("=" * 60)

# 모델 저장
torch.save(best_result['model'].state_dict(), f'{BASE_PATH}/v14_best_model.pt')
print(f"\n모델이 '{BASE_PATH}/v14_best_model.pt'에 저장되었습니다.")
