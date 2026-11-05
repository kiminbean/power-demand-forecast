"""SMP v4.1 서빙용 데이터·피처 공용 모듈 (학습 스크립트와 추론기가 같은 코드를 쓴다).

v4.1 학습 원본(train_smp_v4_smp_only.py)과의 차이:
- '24:00'을 같은 날 00:00이 아니라 다음 날 00:00(거래시간 구간 끝)으로 해석한다.
  원본은 같은 날 00:00으로 바꾼 뒤 정렬해 매일 24시 행이 그날 맨 앞으로 갔다.
- 0원 행을 지우지 않는다 (제주 하루전 가격은 0원이 될 수 있고, 지우면 시간 연속성이 깨진다).
- 시간 간격이 1시간이 아닌 곳(누락 일)을 경계로 표시해 입력·목표 창이 걸치지 않게 한다.
피처 계산 자체는 원본 SMPOnlyDataPipeline.create_enhanced_smp_features를 그대로 쓴다.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.smp.models.train_smp_v4_smp_only import ConfigV41, SMPOnlyDataPipeline

REQUIRED_COLUMNS = ("date", "hour", "smp_mainland", "smp_jeju", "smp_max", "smp_min")
INPUT_HOURS = 48
OUTPUT_HOURS = 24
# 피처 중 가장 긴 과거 참조(336시간 지연)보다 길게 이력을 줘야 학습 때와 같은 피처 값이 나온다.
FEATURE_WARMUP_HOURS = 336


def load_smp_history(path: Path, end_at: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """SMP CSV를 거래시간 구간 끝 시각(datetime) 순으로 읽는다.

    Args:
        path: smp_real_epsis.csv 형식 파일.
        end_at: 이 시각(구간 끝) 이하의 행만 남긴다. None이면 전체.

    Returns:
        datetime 오름차순, 중복 제거된 DataFrame.

    Raises:
        ValueError: 필수 열이 없거나 유효한 행이 없을 때.
    """
    return prepare_smp_frame(pd.read_csv(path, encoding="utf-8-sig"), end_at, source=str(path))


def prepare_smp_frame(df: pd.DataFrame, end_at: Optional[pd.Timestamp] = None, source: str = "frame") -> pd.DataFrame:
    """SMP 표를 검증하고 거래시간 구간 끝 시각(datetime) 순으로 정리한다."""
    df = df.copy()
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns in {source}: {missing}")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["datetime"] = pd.to_datetime(df["date"], errors="coerce") + pd.to_timedelta(df["hour"], unit="h")
    for col in ("smp_mainland", "smp_jeju", "smp_max", "smp_min"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["datetime", "hour", "smp_mainland", "smp_jeju", "smp_max", "smp_min"])
    df = df[(df["hour"] >= 1) & (df["hour"] <= 24)]
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")
    if end_at is not None:
        df = df[df["datetime"] <= pd.Timestamp(end_at)]
    if df.empty:
        raise ValueError(f"no valid SMP rows in {source}")
    return df.reset_index(drop=True)


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """원본 v4.1 피처 40개와 목표값(smp_mainland)을 만든다."""
    pipeline = SMPOnlyDataPipeline(ConfigV41())
    features, targets = pipeline.create_enhanced_smp_features(df)
    return features.astype(np.float64), targets.astype(np.float64), list(pipeline.feature_names)


def contiguous_starts(datetimes: pd.Series, length: int) -> np.ndarray:
    """길이 length의 창이 1시간 간격으로 끊김 없이 이어지는 시작 인덱스."""
    hourly = (datetimes.diff() == pd.Timedelta(hours=1)).to_numpy()
    hourly[0] = False
    # breaks[i] = i번째 행 앞에서 끊긴 횟수의 누적; 창 안에서 누적값이 같으면 끊김이 없다.
    breaks = np.cumsum(~hourly)
    n = len(datetimes)
    starts = np.arange(0, max(0, n - length + 1))
    return starts[breaks[starts] == breaks[starts + length - 1]]


def fit_normalization(features: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
    """학습 구간 통계(자르기 기준·스케일러·목표값 평균/표준편차)를 계산한다 (원본 normalize와 같은 규칙)."""
    clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    q1, q99 = np.percentile(clean, [1, 99], axis=0)
    clip_min = q1 - 3 * (q99 - q1)
    clip_max = q99 + 3 * (q99 - q1)
    clipped = np.clip(clean, clip_min, clip_max)
    mean = clipped.mean(axis=0)
    scale = clipped.std(axis=0)
    scale = np.where(scale == 0, 1.0, scale)  # StandardScaler와 같은 처리
    return {
        "clip_min": clip_min, "clip_max": clip_max, "mean": mean, "scale": scale,
        "target_mean": np.float64(np.mean(targets)), "target_std": np.float64(np.std(targets)),
    }


def apply_normalization(features: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    clipped = np.clip(clean, stats["clip_min"], stats["clip_max"])
    return (clipped - stats["mean"]) / stats["scale"]


def denormalize(values: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return values * (stats["target_std"] + 1e-8) + stats["target_mean"]


def midnight_origins(datetimes: pd.Series, candidates: Sequence[int]) -> List[int]:
    """입력 창의 마지막 행이 0시(전날 24시 구간 끝)인 시작 인덱스만 남긴다 (서빙과 같은 예측 시점)."""
    values = datetimes.to_numpy()
    out = []
    for start in candidates:
        last = pd.Timestamp(values[start + INPUT_HOURS - 1])
        if last.hour == 0 and last.minute == 0:
            out.append(int(start))
    return out
