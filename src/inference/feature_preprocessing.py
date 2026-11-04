"""
추론 전처리 공유 정책 (인과적 결측 처리)
=========================================

API 추론 경로(api.service.PredictionService)와 직접 추론
경로(inference.predict.ProductionPredictor)가 동일한 결측치
전처리 계약을 따르도록 하는 단일 구현(single source of truth).

기존 차이와 문제점
------------------
- API 경로: ``bfill().ffill()`` 후 ``fillna(0)``
  → bfill이 미래 값을 과거 결측에 채워 시계열 미래 정보 누수 발생,
    잔여 NaN의 0 채움은 스케일러에 왜곡된 값 주입.
- 직접 경로: ``dropna()``
  → 누수는 없으나 내부 결측 행이 제거되면서 168시간 시퀀스의
    시간 연속성이 끊긴 채 병합됨.

통합 정책 (프로젝트 규칙: 시간순 처리, 미래 정보 누수 금지)
-----------------------------------------------------------
1. 원본 결측은 피처 생성 전에 과거 값으로만 채운다 (``ffill``) — 내부 결측 보존 및
   시간 연속성 유지. ``src/data/preprocessing.py``의
   CausalProductionPreprocessor 선례(``clip().ffill()``)와 부합.
2. ffill으로도 채울 수 없는 선행 warm-up 행(lag 피처의 구조적 결측)은
   제거한다 — 인과적으로 채울 방법이 없어 제거가 유일한 비누수 처리.
   (bfill/0 채움 금지)
3. 모델에 필요한 피처만 행 유효성에 반영한다. 피처 계산식과 체크포인트
   피처 순서는 유지한다. 선택 목록이 없으면 기존 전체 열 기준을 적용한다.

비파괴성
--------
완전한(내부 결측 없는) 입력에서는 기존 직접 경로(dropna)와
행 수·값이 정확히 동일하다. API 경로도 336행 이상의 완전한 입력에서는
예측에 사용되는 마지막 seq_length(168)행의 값이 불변이다.
단, 168~335행 입력에서는 기존 bfill(누수) 기반 예측 대신
"Insufficient data" 오류로 변경된다 — 누수 값으로 예측하는 것을
막는 의도된 계약 강화.
"""

from pathlib import Path
import sys
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from features import (
    add_lag_features, add_time_features, add_weather_features, create_weather_lag_features
)

MODEL_TIMEZONE = "Asia/Seoul"


def _validate_hourly_index(index: pd.Index) -> None:
    """Validate elapsed hourly spacing before row-based features or fills."""
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Inference data requires a DatetimeIndex")
    if index.hasnans:
        raise ValueError("Inference timestamps must not contain NaT")
    intervals = index[1:] - index[:-1]
    invalid = intervals != pd.Timedelta(hours=1)
    if invalid.any():
        position = int(invalid.argmax())
        raise ValueError(
            "Inference timestamps must be strictly increasing at hourly intervals; "
            f"positions {position} and {position + 1} are "
            f"{index[position]} and {index[position + 1]} "
            f"(interval {intervals[position]})"
        )


def _validate_computed_humidity(df: pd.DataFrame, include_thi: bool = True) -> None:
    """Reject undefined humidity or THI where both source values are present."""
    if not {'temp_mean', 'dewpoint_mean', 'humidity', 'THI'}.issubset(df.columns):
        return
    valid_inputs = df[['temp_mean', 'dewpoint_mean']].notna().all(axis=1)
    columns = ('humidity', 'THI') if include_thi else ('humidity',)
    for column in columns:
        if not np.isfinite(df.loc[valid_inputs, column].to_numpy(dtype=float)).all():
            raise ValueError(f"Computed {column} from humidity inputs must contain finite values")


def prepare_inference_features(
    df: pd.DataFrame,
    include_weather: bool = False,
    selected_features: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """추론용 피처 준비 — 양쪽 추론 경로가 공유하는 인과적 결측 처리 정책.

    원본 결측치를 과거 값으로 채운 뒤 기존 피처 계산식(시간 → 지연 → 기상)을
    적용하고, 인과적으로 채울 수 없는
    선행 warm-up 행만 제거한다. 미래 값 참조(bfill)와 0 채움은 금지.

    Args:
        df: 입력 DataFrame (datetime 인덱스 또는 'datetime' 컬럼)
        include_weather: 기상 파생 피처(THI/HDD/CDD 등) 포함 여부
        selected_features: 행 유효성을 결정하는 체크포인트 피처 목록.
            None이면 기존처럼 모든 열의 결측을 확인한다.

    Returns:
        선택된 피처에 결측이 없는 DataFrame. 선택하지 않은 열은 결측일 수 있다.

        호출자는 결과 행 수가 시퀀스 길이 이상인지 별도 확인해야 한다
        필요한 입력 길이는 선택된 피처의 warm-up과 시퀀스 길이에 따른다.
    """
    df = df.copy()

    # Ensure datetime index (기존 직접 경로와 동일한 규칙)
    if 'datetime' in df.columns:
        df.set_index('datetime', inplace=True)

    # 행 기반 lag와 과거 값 채우기는 입력이 시간순일 때만 안전하다.
    _validate_hourly_index(df.index)
    # 학습 데이터는 제주 현지 시각(시간대 없는 값)이다. 시간대가 있는
    # 입력은 같은 현지 시각으로 변환하되, 기존 naive 입력은 그대로 유지한다.
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(MODEL_TIMEZONE)

    # 이동평균은 무한대를 무시할 수 있으므로 원본 수요를 생성 전에 검사한다.
    if 'power_demand' in df.columns:
        demand = pd.to_numeric(df['power_demand'], errors='raise').dropna()
        if not np.isfinite(demand.to_numpy(dtype=float)).all():
            raise ValueError("Raw power_demand must contain finite values")

    # 원본을 먼저 채워 lag·차분·이동평균이 같은 수요 이력을 참조하게 한다.
    df = df.ffill()

    # 시간 피처 추가 (기존과 동일 — 피처 구성 불변)
    df = add_time_features(df, include_holiday=True)

    # 선택된 기상 지연 그룹만 계산해 무관한 원본을 처리하지 않는다.
    if selected_features is None:
        # 전체 지연 경로에서도 이동평균에 숨겨지는 원본 무한대를 거부한다.
        for column in ('기온', '일사', '습도'):
            if column in df.columns:
                values = pd.Series(pd.to_numeric(df[column], errors='raise')).dropna()
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"Raw {column} must contain finite values")
        df = add_lag_features(df, demand_col='power_demand')
    else:
        selected = set(selected_features)
        groups = {
            '기온': any(name.startswith(('temp_ma_', 'temp_lag_', 'temp_min_', 'temp_max_', 'temp_range_')) for name in selected),
            '일사': any(name.startswith(('irradiance_ma_', 'irradiance_sum_')) for name in selected),
            '습도': any(name.startswith('humidity_ma_') for name in selected),
        }
        for column, needed in groups.items():
            if needed and column in df.columns:
                values = pd.Series(pd.to_numeric(df[column], errors='raise')).dropna()
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"Raw {column} must contain finite values")
        df = add_lag_features(df, demand_col='power_demand', include_weather_features=False)
        weather_lags = create_weather_lag_features(
            df, include_temp=groups['기온'], include_irradiance=groups['일사'],
            include_humidity=groups['습도']
        )
        for column in weather_lags.columns:
            df[column] = weather_lags[column]

    # 체크포인트가 요구하는 기상 파생값만 계산하여 무관한 입력 오류를 격리한다.
    # 선택 목록이 없는 기존 경로는 기존 계산 조건과 동작을 유지한다.
    if include_weather and selected_features is None and '기온' in df.columns:
        for column in ('temp_mean', 'dewpoint_mean', 'wind_speed_mean'):
            if column in df.columns:
                values = pd.Series(pd.to_numeric(df[column], errors='raise')).dropna()
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"Raw {column} must contain finite values")
        df = add_weather_features(df, include_thi=True, include_hdd_cdd=True)
        if 'temp_mean' in df.columns and 'dewpoint_mean' in df.columns:
            _validate_computed_humidity(df)
    elif include_weather and selected_features is not None and 'temp_mean' in df.columns:
        selected = set(selected_features)
        # 기온이 없는 이전 경로는 이미 주어진 파생값을 변경하지 않았다.
        preserved = selected.intersection(df.columns) if '기온' not in df.columns else set()
        to_compute = selected - preserved
        thi = bool(to_compute.intersection({'humidity', 'THI'}))
        wind = 'wind_chill' in to_compute
        degrees = bool(to_compute.intersection({'HDD', 'CDD', 'HDD_cumsum', 'CDD_cumsum'}))
        if thi or wind or degrees:
            inputs = {'temp_mean'}
            if thi and 'dewpoint_mean' in df.columns:
                inputs.add('dewpoint_mean')
            if wind and 'wind_speed_mean' in df.columns:
                inputs.add('wind_speed_mean')
            for column in inputs:
                values = pd.Series(pd.to_numeric(df[column], errors='raise')).dropna()
                if not np.isfinite(values.to_numpy(dtype=float)).all():
                    raise ValueError(f"Raw {column} must contain finite values")
            saved = df[list(preserved)].copy()
            df = add_weather_features(
                df, include_thi=thi, include_wind_chill=wind, include_hdd_cdd=degrees
            )
            if thi and 'dewpoint_mean' in df.columns:
                _validate_computed_humidity(df, include_thi='THI' in to_compute)
            for column in saved.columns:
                df[column] = saved[column]

    # 결측 처리: 과거 값으로만 채움 (인과적 — 미래 정보 누수 금지).
    # 내부 결측 행을 제거하지 않아 시퀀스의 시간 연속성이 유지된다.
    df = df.ffill()

    # 선행 warm-up 행(lag/MA 구조적 결측)은 인과적으로 채울 수 없으므로 제거.
    # bfill/0 채움은 미래 누수·스케일러 왜곡을 유발하므로 사용하지 않는다.
    if selected_features is None:
        df = df.dropna()
        numeric = df.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Inference features must contain finite values")
    else:
        missing = [name for name in selected_features if name not in df.columns]
        if missing:
            raise ValueError(f"Missing configured inference features: {missing}")
        df = df.dropna(subset=list(selected_features))
        if not np.isfinite(df[list(selected_features)].to_numpy(dtype=float)).all():
            raise ValueError("Selected inference features must contain finite values")

    # 선택된 행 사이의 공백은 하나의 시퀀스로 합치면 안 된다.
    _validate_hourly_index(df.index)

    return df


__all__: List[str] = ['prepare_inference_features']
