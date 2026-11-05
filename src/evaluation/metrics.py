"""
EVAL-001: 평가 지표 구현
========================

전력 수요 예측 모델의 성능 평가를 위한 지표

주요 지표:
1. Primary: MAPE, R²
2. Secondary: MAE, MSE, RMSE
3. Advanced: sMAPE, MASE, MBE, CV(RMSE)

기능:
- 기본 평가 지표 계산
- 시간대별/계절별 분석
- 평가 리포트 생성
- 잔차 분석

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import math

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path


# ============================================================
# 기본 평가 지표
# ============================================================

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error (평균 제곱 오차)

    MSE = (1/n) * Σ(y_true - y_pred)²

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: MSE 값
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error (평균 제곱근 오차)

    RMSE = √MSE

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: RMSE 값
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error (평균 절대 오차)

    MAE = (1/n) * Σ|y_true - y_pred|

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: MAE 값
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error (평균 절대 백분율 오차)

    MAPE = (100/n) * Σ|y_true - y_pred| / |y_true|

    Args:
        y_true: 실제값
        y_pred: 예측값
        epsilon: 0으로 나누기 방지용 작은 값

    Returns:
        float: MAPE 값 (%)

    Note:
        - 0에 가까운 실제값이 있으면 MAPE가 매우 커질 수 있음
        - 이 경우 sMAPE 사용 권장
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # 0이 아닌 값만 사용
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return float('inf')

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted MAPE (가중 평균 절대 백분율 오차, 구간 집계용 표준)

    WMAPE = Σ|y_true - y_pred| / Σ|y_true| × 100
          = mean(|y_true - y_pred|) / mean(|y_true|) × 100

    구간(세그먼트) 집계 지표의 단일 정의다. 점별 MAPE(mape)와 다음이 다르다.
    - 오차를 실측 절대값 총합으로 가중 → 0에 가까운 실측값 하나가
      지표를 폭발시키지 않는다(0 나눗셈 폭발 처리 정책 통일).
    - 분모에 절대값을 취하므로 부호 혼합 구간에서도 정의가 일정하다.

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: WMAPE 값 (%). 실측 절대값 총합이 0이면 NaN

    Note:
        점별 MAPE가 필요하면 mape()를, 0 근처 실측이 많으면 sMAPE를 사용.
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} vs {y_pred.shape}"
        )

    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0.0 or not np.isfinite(denom):
        return float("nan")

    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R² Score (결정계수)

    R² = 1 - (SS_res / SS_tot)
    SS_res = Σ(y_true - y_pred)²
    SS_tot = Σ(y_true - mean(y_true))²

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: R² 값 (0~1, 1이면 완벽한 예측)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # 분산이 0인 경우 (모든 값이 동일)
    if ss_tot == 0:
        # 예측도 완벽하면 1, 아니면 0
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)


# ============================================================
# 고급 평가 지표
# ============================================================

def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Symmetric Mean Absolute Percentage Error (대칭 평균 절대 백분율 오차)

    sMAPE = (100/n) * Σ|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)

    Args:
        y_true: 실제값
        y_pred: 예측값
        epsilon: 0으로 나누기 방지용 작은 값

    Returns:
        float: sMAPE 값 (%, 0~200 범위)

    Note:
        - MAPE와 달리 대칭적이므로 과대/과소 예측에 동일한 패널티
        - 0에 가까운 값에 대해 더 안정적
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24
) -> float:
    """
    Mean Absolute Scaled Error (평균 절대 스케일 오차)

    MASE = MAE / MAE_naive
    MAE_naive = (1/(n-m)) * Σ|y_t - y_{t-m}| (seasonal naive forecast)

    Args:
        y_true: 테스트 실제값
        y_pred: 테스트 예측값
        y_train: 학습 데이터 (naive forecast 계산용)
        seasonality: 계절성 주기 (시간별 데이터의 경우 24)

    Returns:
        float: MASE 값 (1 미만이면 naive보다 좋음)

    Note:
        - 스케일에 독립적인 지표
        - 다른 데이터셋 간 비교에 유용
        - MASE < 1: naive보다 좋음, MASE > 1: naive보다 나쁨
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    y_train = np.asarray(y_train).flatten()

    # Naive forecast MAE (seasonal)
    mae_naive = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))

    if mae_naive == 0:
        return float('inf')

    return mae(y_true, y_pred) / mae_naive


def mbe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Bias Error (평균 편향 오차)

    MBE = (1/n) * Σ(y_pred - y_true)

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: MBE 값 (양수: 과대예측, 음수: 과소예측)

    Note:
        - 예측의 편향 방향을 나타냄
        - 양의 오차와 음의 오차가 상쇄될 수 있음
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.mean(y_pred - y_true)


def cv_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of Variation of RMSE (RMSE 변동계수)

    CV(RMSE) = RMSE / mean(y_true) * 100

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: CV(RMSE) 값 (%)

    Note:
        - 정규화된 RMSE로 스케일에 독립적
        - 다른 데이터셋 간 비교에 유용
    """
    y_true = np.asarray(y_true).flatten()
    mean_true = np.mean(y_true)

    if mean_true == 0:
        return float('inf')

    return rmse(y_true, y_pred) / mean_true * 100


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Maximum Error (최대 오차)

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: 최대 절대 오차
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.max(np.abs(y_true - y_pred))


def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Median Absolute Error (중앙값 절대 오차)

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        float: 중앙값 절대 오차

    Note:
        - MAE보다 이상치에 강건함
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    return np.median(np.abs(y_true - y_pred))


# ============================================================
# 예측 비교 검정
# ============================================================

# 1시간 앞 예측 오차는 하루 주기로 자기상관이 남으므로 24시차까지 분산을 보정한다
DM_DEFAULT_MAX_LAG = 24


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray,
                    max_lag: int = DM_DEFAULT_MAX_LAG) -> Dict[str, float]:
    """
    Diebold-Mariano 검정 (절대오차 손실, Newey-West 장기분산)

    손실 차이 d_t = |e_a,t| - |e_b,t|의 평균이 0인지 검정한다. 지표 차이가 아주 작을 때
    (예: AR을 앵커로 써서 거의 같은 예측) 우연인지 가려내기 위해 쓴다.

    Args:
        errors_a: 예측 a의 오차 (예측 - 실제)
        errors_b: 예측 b의 오차 (같은 시각)
        max_lag: Newey-West 가중치를 적용할 최대 시차

    Returns:
        Dict: mean_diff(음수면 a가 정확), stat, p_a_better(단측 p값, a가 더 정확하다는 가설)
    """
    d = np.abs(np.asarray(errors_a, dtype=float).ravel()) - np.abs(np.asarray(errors_b, dtype=float).ravel())
    if d.size <= max_lag + 1 or not np.isfinite(d).all():
        raise ValueError("diebold_mariano needs finite errors longer than max_lag + 1")
    centered = d - d.mean()
    long_run_var = centered @ centered / d.size
    for k in range(1, max_lag + 1):
        long_run_var += 2 * (1 - k / (max_lag + 1)) * (centered[k:] @ centered[:-k]) / d.size
    if long_run_var <= 0:
        # 손실 차이가 상수면 분산이 0이다. 차이도 0이면 두 예측이 같으므로 판단할 근거가 없다
        stat = 0.0 if d.mean() == 0 else float(np.sign(d.mean()) * np.inf)
    else:
        stat = float(d.mean() / np.sqrt(long_run_var / d.size))
    p_a_better = 0.5 * math.erfc(-stat / math.sqrt(2))  # 표준정규 누적분포 Φ(stat)
    return {'mean_diff': float(d.mean()), 'stat': stat, 'p_a_better': float(p_a_better)}


# ============================================================
# 통합 평가 함수
# ============================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    seasonality: int = 24
) -> Dict[str, float]:
    """
    모든 평가 지표를 계산합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        y_train: 학습 데이터 (MASE 계산용, optional)
        seasonality: 계절성 주기 (기본값: 24시간)

    Returns:
        Dict[str, float]: 모든 평가 지표

    Example:
        >>> metrics = compute_all_metrics(y_true, y_pred)
        >>> print(f"RMSE: {metrics['RMSE']:.2f}")
    """
    metrics = {
        # Primary metrics
        'MAPE': mape(y_true, y_pred),
        'R2': r2_score(y_true, y_pred),

        # Secondary metrics
        'MSE': mse(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MAE': mae(y_true, y_pred),

        # Advanced metrics
        'sMAPE': smape(y_true, y_pred),
        'MBE': mbe(y_true, y_pred),
        'CV_RMSE': cv_rmse(y_true, y_pred),
        'MaxError': max_error(y_true, y_pred),
        'MedianAE': median_absolute_error(y_true, y_pred),
    }

    # MASE (학습 데이터가 있는 경우)
    if y_train is not None:
        metrics['MASE'] = mase(y_true, y_pred, y_train, seasonality)

    return metrics


def compute_metrics_by_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: List[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    임계값별 평가 지표를 계산합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        thresholds: 임계값 리스트 (기본값: 사분위수)

    Returns:
        Dict: 각 임계값 범위별 평가 지표
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if thresholds is None:
        thresholds = [
            np.percentile(y_true, 25),
            np.percentile(y_true, 50),
            np.percentile(y_true, 75)
        ]

    results = {}

    # 각 범위별 분석
    prev_thresh = -np.inf
    for i, thresh in enumerate(thresholds + [np.inf]):
        mask = (y_true > prev_thresh) & (y_true <= thresh)

        if mask.sum() > 0:
            if thresh == np.inf:
                label = f'>{thresholds[-1]:.0f}'
            elif prev_thresh == -np.inf:
                label = f'<={thresh:.0f}'
            else:
                label = f'{prev_thresh:.0f}-{thresh:.0f}'

            results[label] = {
                'count': mask.sum(),
                'MAPE': mape(y_true[mask], y_pred[mask]),
                'RMSE': rmse(y_true[mask], y_pred[mask]),
                'MAE': mae(y_true[mask], y_pred[mask]),
            }

        prev_thresh = thresh

    return results


# ============================================================
# 시간 기반 분석
# ============================================================

def compute_metrics_by_hour(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    시간대별 평가 지표를 계산합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        timestamps: 타임스탬프

    Returns:
        pd.DataFrame: 시간대별 평가 지표
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'hour': timestamps.hour
    })

    results = []
    for hour in range(24):
        mask = df['hour'] == hour
        if mask.sum() > 0:
            results.append({
                'hour': hour,
                'count': mask.sum(),
                'MAPE': mape(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'RMSE': rmse(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'MAE': mae(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'R2': r2_score(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
            })

    return pd.DataFrame(results)


def compute_metrics_by_dayofweek(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    요일별 평가 지표를 계산합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        timestamps: 타임스탬프

    Returns:
        pd.DataFrame: 요일별 평가 지표
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'dayofweek': timestamps.dayofweek
    })

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    results = []
    for dow in range(7):
        mask = df['dayofweek'] == dow
        if mask.sum() > 0:
            results.append({
                'dayofweek': dow,
                'day_name': day_names[dow],
                'count': mask.sum(),
                'MAPE': mape(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'RMSE': rmse(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'MAE': mae(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'R2': r2_score(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
            })

    return pd.DataFrame(results)


def compute_metrics_by_month(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    월별 평가 지표를 계산합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        timestamps: 타임스탬프

    Returns:
        pd.DataFrame: 월별 평가 지표
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'month': timestamps.month
    })

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    results = []
    for month in range(1, 13):
        mask = df['month'] == month
        if mask.sum() > 0:
            results.append({
                'month': month,
                'month_name': month_names[month - 1],
                'count': mask.sum(),
                'MAPE': mape(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'RMSE': rmse(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'MAE': mae(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'R2': r2_score(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
            })

    return pd.DataFrame(results)


def compute_metrics_by_season(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    계절별 평가 지표를 계산합니다.

    계절 정의 (한국 기준):
    - 봄: 3-5월
    - 여름: 6-8월
    - 가을: 9-11월
    - 겨울: 12-2월

    Args:
        y_true: 실제값
        y_pred: 예측값
        timestamps: 타임스탬프

    Returns:
        pd.DataFrame: 계절별 평가 지표
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    def get_season(month: int) -> str:
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Fall'
        else:
            return 'Winter'

    df = pd.DataFrame({
        'actual': y_true,
        'predicted': y_pred,
        'season': [get_season(m) for m in timestamps.month]
    })

    season_order = ['Spring', 'Summer', 'Fall', 'Winter']

    results = []
    for season in season_order:
        mask = df['season'] == season
        if mask.sum() > 0:
            results.append({
                'season': season,
                'count': mask.sum(),
                'MAPE': mape(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'RMSE': rmse(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'MAE': mae(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
                'R2': r2_score(df.loc[mask, 'actual'], df.loc[mask, 'predicted']),
            })

    return pd.DataFrame(results)


# ============================================================
# 잔차 분석
# ============================================================

def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    잔차 분석을 수행합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값

    Returns:
        Dict: 잔차 통계
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    residuals = y_true - y_pred

    return {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'median': np.median(residuals),
        'q25': np.percentile(residuals, 25),
        'q75': np.percentile(residuals, 75),
        'skewness': float(pd.Series(residuals).skew()),
        'kurtosis': float(pd.Series(residuals).kurtosis()),
    }


def compute_prediction_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence_levels: List[float] = None
) -> Dict[str, float]:
    """
    예측 구간 커버리지를 계산합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        confidence_levels: 신뢰수준 리스트 (기본값: [0.5, 0.8, 0.9, 0.95])

    Returns:
        Dict: 각 신뢰수준에서의 커버리지
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.8, 0.9, 0.95]

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    residuals = y_true - y_pred
    std_residual = np.std(residuals)

    results = {}
    for level in confidence_levels:
        z = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(level, 1.96)
        lower = y_pred - z * std_residual
        upper = y_pred + z * std_residual

        coverage = np.mean((y_true >= lower) & (y_true <= upper)) * 100
        results[f'coverage_{int(level*100)}'] = coverage

    return results


# ============================================================
# 평가 리포트 생성
# ============================================================

@dataclass
class EvaluationReport:
    """평가 리포트 데이터 클래스"""

    model_name: str
    horizon: int
    timestamp: str
    overall_metrics: Dict[str, float]
    hourly_metrics: Optional[pd.DataFrame] = None
    daily_metrics: Optional[pd.DataFrame] = None
    monthly_metrics: Optional[pd.DataFrame] = None
    seasonal_metrics: Optional[pd.DataFrame] = None
    residual_analysis: Optional[Dict[str, float]] = None
    prediction_intervals: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict:
        """딕셔너리로 변환"""
        result = {
            'model_name': self.model_name,
            'horizon': self.horizon,
            'timestamp': self.timestamp,
            'overall_metrics': self.overall_metrics,
        }

        if self.residual_analysis:
            result['residual_analysis'] = self.residual_analysis

        if self.prediction_intervals:
            result['prediction_intervals'] = self.prediction_intervals

        if self.hourly_metrics is not None:
            result['hourly_metrics'] = self.hourly_metrics.to_dict('records')

        if self.seasonal_metrics is not None:
            result['seasonal_metrics'] = self.seasonal_metrics.to_dict('records')

        return result

    def save(self, filepath: str) -> None:
        """리포트를 JSON 파일로 저장"""
        def convert_to_json_serializable(obj):
            """numpy 타입을 JSON 직렬화 가능한 타입으로 변환"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            return obj

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_serializable(self.to_dict()), f, indent=2, ensure_ascii=False)

    def summary(self) -> str:
        """리포트 요약 문자열 생성"""
        lines = []
        lines.append("=" * 60)
        lines.append(f"Evaluation Report: {self.model_name}")
        lines.append(f"Horizon: {self.horizon}h | Generated: {self.timestamp}")
        lines.append("=" * 60)

        lines.append("\n[Overall Metrics]")
        lines.append("-" * 40)

        # Primary metrics
        lines.append(f"  MAPE:    {self.overall_metrics.get('MAPE', 0):.2f}%")
        lines.append(f"  R²:      {self.overall_metrics.get('R2', 0):.4f}")

        # Secondary metrics
        lines.append(f"  RMSE:    {self.overall_metrics.get('RMSE', 0):.2f}")
        lines.append(f"  MAE:     {self.overall_metrics.get('MAE', 0):.2f}")

        # Advanced metrics
        if 'sMAPE' in self.overall_metrics:
            lines.append(f"  sMAPE:   {self.overall_metrics.get('sMAPE', 0):.2f}%")
        if 'MBE' in self.overall_metrics:
            lines.append(f"  MBE:     {self.overall_metrics.get('MBE', 0):.2f}")

        # Seasonal summary
        if self.seasonal_metrics is not None:
            lines.append("\n[Seasonal Analysis]")
            lines.append("-" * 40)
            for _, row in self.seasonal_metrics.iterrows():
                lines.append(f"  {row['season']:8s}: MAPE={row['MAPE']:.2f}%, R²={row['R2']:.4f}")

        # Residual summary
        if self.residual_analysis:
            lines.append("\n[Residual Analysis]")
            lines.append("-" * 40)
            lines.append(f"  Mean:    {self.residual_analysis['mean']:.2f}")
            lines.append(f"  Std:     {self.residual_analysis['std']:.2f}")
            lines.append(f"  Skew:    {self.residual_analysis['skewness']:.3f}")

        lines.append("=" * 60)
        return "\n".join(lines)


def generate_evaluation_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: pd.DatetimeIndex = None,
    model_name: str = "LSTM",
    horizon: int = 1,
    y_train: np.ndarray = None,
    include_time_analysis: bool = True
) -> EvaluationReport:
    """
    종합 평가 리포트를 생성합니다.

    Args:
        y_true: 실제값
        y_pred: 예측값
        timestamps: 타임스탬프 (시간 기반 분석용)
        model_name: 모델 이름
        horizon: 예측 horizon
        y_train: 학습 데이터 (MASE 계산용)
        include_time_analysis: 시간 기반 분석 포함 여부

    Returns:
        EvaluationReport: 평가 리포트

    Example:
        >>> report = generate_evaluation_report(y_true, y_pred, timestamps, model_name="BiLSTM", horizon=24)
        >>> print(report.summary())
        >>> report.save("evaluation_report.json")
    """
    # Overall metrics
    overall_metrics = compute_all_metrics(y_true, y_pred, y_train)

    # Residual analysis
    residual_analysis = analyze_residuals(y_true, y_pred)

    # Prediction intervals
    prediction_intervals = compute_prediction_intervals(y_true, y_pred)

    # Time-based analysis
    hourly_metrics = None
    daily_metrics = None
    monthly_metrics = None
    seasonal_metrics = None

    if timestamps is not None and include_time_analysis:
        hourly_metrics = compute_metrics_by_hour(y_true, y_pred, timestamps)
        daily_metrics = compute_metrics_by_dayofweek(y_true, y_pred, timestamps)
        monthly_metrics = compute_metrics_by_month(y_true, y_pred, timestamps)
        seasonal_metrics = compute_metrics_by_season(y_true, y_pred, timestamps)

    return EvaluationReport(
        model_name=model_name,
        horizon=horizon,
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        overall_metrics=overall_metrics,
        hourly_metrics=hourly_metrics,
        daily_metrics=daily_metrics,
        monthly_metrics=monthly_metrics,
        seasonal_metrics=seasonal_metrics,
        residual_analysis=residual_analysis,
        prediction_intervals=prediction_intervals
    )


# ============================================================
# 모델 비교 함수
# ============================================================

def compare_models(
    results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    metric_names: List[str] = None
) -> pd.DataFrame:
    """
    여러 모델의 성능을 비교합니다.

    Args:
        results: {모델명: (y_true, y_pred)} 딕셔너리
        metric_names: 비교할 지표 이름 리스트

    Returns:
        pd.DataFrame: 모델별 성능 비교표

    Example:
        >>> results = {
        ...     'LSTM': (y_true, y_pred_lstm),
        ...     'BiLSTM': (y_true, y_pred_bilstm),
        ... }
        >>> comparison = compare_models(results)
        >>> print(comparison)
    """
    if metric_names is None:
        metric_names = ['MAPE', 'R2', 'RMSE', 'MAE']

    comparison = []

    for model_name, (y_true, y_pred) in results.items():
        metrics = compute_all_metrics(y_true, y_pred)
        row = {'Model': model_name}
        for name in metric_names:
            row[name] = metrics.get(name, np.nan)
        comparison.append(row)

    df = pd.DataFrame(comparison)

    # 랭킹 추가
    for name in metric_names:
        if name in ['R2']:  # 높을수록 좋은 지표
            df[f'{name}_rank'] = df[name].rank(ascending=False)
        else:  # 낮을수록 좋은 지표
            df[f'{name}_rank'] = df[name].rank(ascending=True)

    return df


def compare_horizons(
    y_true_dict: Dict[int, np.ndarray],
    y_pred_dict: Dict[int, np.ndarray],
    model_name: str = "Model"
) -> pd.DataFrame:
    """
    여러 예측 horizon의 성능을 비교합니다.

    Args:
        y_true_dict: {horizon: y_true} 딕셔너리
        y_pred_dict: {horizon: y_pred} 딕셔너리
        model_name: 모델 이름

    Returns:
        pd.DataFrame: horizon별 성능 비교표
    """
    comparison = []

    for horizon in sorted(y_true_dict.keys()):
        y_true = y_true_dict[horizon]
        y_pred = y_pred_dict[horizon]

        metrics = compute_all_metrics(y_true, y_pred)
        comparison.append({
            'Model': model_name,
            'Horizon': f'{horizon}h',
            'MAPE': metrics['MAPE'],
            'R2': metrics['R2'],
            'RMSE': metrics['RMSE'],
            'MAE': metrics['MAE'],
        })

    return pd.DataFrame(comparison)


# ============================================================
# 분위수·구간 예측 평가 지표 (EVAL)
# - pinball_loss: 분위수 회귀 손실 (quantile loss)
# - coverage: 명시적 하한/상한 구간의 실측 커버리지 (%)
# - interval_score: Winkler interval score (낮을수록 좋음)
# - evaluate_quantile_forecast: q10/q50/q90 통합 평가
# 기존 compute_prediction_intervals (잔차 정규분포 가정)과 달리
# 모델이 명시적으로 출력한 분위수 예측에 적용한다.
# ============================================================

def _validate_quantile_arrays(*values: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Validate nonempty, finite and equally sized 1-D metric inputs."""
    arrays = tuple(np.asarray(value, dtype=float) for value in values)
    if any(array.ndim != 1 or array.size == 0 for array in arrays):
        raise ValueError("입력은 비어 있지 않은 1차원 배열이어야 합니다")
    if any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("실측값과 예측값의 길이가 일치해야 합니다")
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("실측값과 예측값은 모두 유한해야 합니다")
    return arrays


def pinball_loss(
    y_true: np.ndarray,
    y_pred_quantile: np.ndarray,
    quantile: float,
) -> float:
    """
    분위수 τ에 대한 pinball loss (quantile loss) 계산.

    ρ_τ(u) = u*(τ - 1{u<0}), u = y - q 의 평균.
    과소예측(u>=0)은 τ배, 과대예측(u<0)은 (1-τ)배 페널티.

    Args:
        y_true: 실측값 (1차원)
        y_pred_quantile: 분위수 τ 예측값 (1차원)
        quantile: 분위수 (0 < τ < 1)

    Returns:
        평균 pinball loss

    Example:
        >>> pinball_loss(np.array([10.0]), np.array([8.0]), quantile=0.9)
        1.8
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"분위수는 0<τ<1 이어야 합니다: {quantile}")

    y_true, y_pred_quantile = _validate_quantile_arrays(y_true, y_pred_quantile)

    u = y_true - y_pred_quantile
    losses = np.where(u >= 0, u * quantile, -u * (1.0 - quantile))
    return float(np.mean(losses))


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
) -> float:
    """
    명시적 구간 [lower, upper]의 실측 커버리지 (%) 계산 (폐구간).

    Args:
        y_true: 실측값
        y_lower: 구간 하한
        y_upper: 구간 상한

    Returns:
        커버리지 (0~100 %)

    Example:
        >>> coverage(np.array([1.0, 5.0]), np.zeros(2), np.array([4.0, 4.0]))
        50.0
    """
    y_true, y_lower, y_upper = _validate_quantile_arrays(y_true, y_lower, y_upper)
    if np.any(y_lower > y_upper):
        raise ValueError("구간 하한은 상한보다 클 수 없습니다")

    inside = (y_true >= y_lower) & (y_true <= y_upper)
    return float(np.mean(inside) * 100.0)


def interval_score(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    alpha: float,
) -> float:
    """
    중앙 (1-α) 예측구간의 Winkler interval score 계산 (낮을수록 좋음).

    IS = (u - l)
         + (2/α)*(l - y)*1{y < l}
         + (2/α)*(y - u)*1{y > u}

    구간 내 실측은 폭만, 구간 밖 실측은 폭 + 누출 벌점.

    Args:
        y_true: 실측값
        y_lower: 구간 하한
        y_upper: 구간 상한
        alpha: 유의수준 (0 < α < 1, 예: 80% 구간 → 0.2)

    Returns:
        평균 interval score

    Example:
        >>> interval_score(np.array([10.0]), np.array([8.0]), np.array([12.0]), alpha=0.2)
        4.0
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha는 0<α<1 이어야 합니다: {alpha}")

    y_true, y_lower, y_upper = _validate_quantile_arrays(y_true, y_lower, y_upper)
    if np.any(y_lower > y_upper):
        raise ValueError("구간 하한은 상한보다 클 수 없습니다")

    width = y_upper - y_lower
    penalty_below = np.where(y_true < y_lower, (2.0 / alpha) * (y_lower - y_true), 0.0)
    penalty_above = np.where(y_true > y_upper, (2.0 / alpha) * (y_true - y_upper), 0.0)
    scores = width + penalty_below + penalty_above
    return float(np.mean(scores))


def diagnose_quantile_forecast(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> Dict[str, Union[int, float]]:
    """Report quantile crossing and individual losses without repairing forecasts.

    Args:
        y_true: Nonempty, finite one-dimensional observations.
        q10: Forecasts at the 10th percentile.
        q50: Forecasts at the 50th percentile.
        q90: Forecasts at the 90th percentile.

    Returns:
        Counts, percentage of rows with crossing, and individual pinball losses.
        Crossing counts overlap when both adjacent pairs cross in the same row.
    """
    y_true, q10, q50, q90 = _validate_quantile_arrays(y_true, q10, q50, q90)
    lower_crossed = q10 > q50
    upper_crossed = q50 > q90
    crossing_count = int(np.count_nonzero(lower_crossed | upper_crossed))
    return {
        "sample_count": int(y_true.size),
        "crossing_count": crossing_count,
        "crossing_rate_pct": float(100.0 * crossing_count / y_true.size),
        "q10_gt_q50_count": int(np.count_nonzero(lower_crossed)),
        "q50_gt_q90_count": int(np.count_nonzero(upper_crossed)),
        "pinball_q10": pinball_loss(y_true, q10, 0.1),
        "pinball_q50": pinball_loss(y_true, q50, 0.5),
        "pinball_q90": pinball_loss(y_true, q90, 0.9),
    }


def evaluate_quantile_forecast(
    y_true: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> Dict[str, float]:
    """
    q10/q50/q90 분위수 예측 통합 평가.

    80% 중앙 구간 [q10, q90]의 커버리지·Winkler 점수와
    각 분위수의 pinball loss를 한 번에 계산한다.

    Args:
        y_true: 실측값
        q10: 10분위 예측 (구간 하한)
        q50: 중앙값(50분위) 예측
        q90: 90분위 예측 (구간 상한)

    Returns:
        dict: pinball_q10, pinball_q50, pinball_q90,
              coverage_80, interval_score_80, mean_interval_width

    Example:
        >>> y = np.array([10.0, 20.0])
        >>> res = evaluate_quantile_forecast(y, q10=y-5, q50=y.copy(), q90=y+5)
        >>> res["coverage_80"]
        100.0
    """
    y_true, q10, q50, q90 = _validate_quantile_arrays(y_true, q10, q50, q90)
    if np.any(q10 > q50) or np.any(q50 > q90):
        raise ValueError("분위수 예측은 q10 ≤ q50 ≤ q90 이어야 합니다")
    return {
        "pinball_q10": pinball_loss(y_true, q10, quantile=0.1),
        "pinball_q50": pinball_loss(y_true, q50, quantile=0.5),
        "pinball_q90": pinball_loss(y_true, q90, quantile=0.9),
        "coverage_80": coverage(y_true, q10, q90),
        "interval_score_80": interval_score(y_true, q10, q90, alpha=0.2),
        "mean_interval_width": float(
            np.mean(np.asarray(q90, dtype=float) - np.asarray(q10, dtype=float))
        ),
    }
