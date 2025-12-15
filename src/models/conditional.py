"""
Conditional Model: 겨울철 변곡점 특화 조건부 예측 모델
======================================================

EVAL-003 및 변곡점 분석 결과:
- 기상변수는 대부분의 경우 예측에 도움이 되지 않음 (-4.6%)
- 단, 겨울철 변곡점(급변구간)에서만 +2.5% 개선 효과

전략:
1. 기본 모델: demand_only (기상변수 없이)
2. 겨울철 + 변곡점 감지 시: weather_full 모델로 전환
3. 두 모델의 가중 앙상블 옵션

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class Season(Enum):
    """계절 분류"""
    WINTER = "winter"       # 12, 1, 2월
    SUMMER = "summer"       # 6, 7, 8월
    TRANSITION = "transition"  # 3, 4, 5, 9, 10, 11월


@dataclass
class PredictionContext:
    """예측 컨텍스트 정보"""
    timestamp: pd.Timestamp
    season: Season
    is_inflection: bool
    recent_demand_change: float
    use_weather: bool
    confidence: float


class SeasonClassifier:
    """계절 분류기"""

    WINTER_MONTHS = {12, 1, 2}
    SUMMER_MONTHS = {6, 7, 8}

    @classmethod
    def classify(cls, timestamp: Union[pd.Timestamp, np.datetime64]) -> Season:
        """타임스탬프에서 계절 분류"""
        if isinstance(timestamp, np.datetime64):
            timestamp = pd.Timestamp(timestamp)

        month = timestamp.month

        if month in cls.WINTER_MONTHS:
            return Season.WINTER
        elif month in cls.SUMMER_MONTHS:
            return Season.SUMMER
        else:
            return Season.TRANSITION

    @classmethod
    def classify_batch(cls, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """배치 타임스탬프 계절 분류"""
        months = timestamps.month
        seasons = np.array([Season.TRANSITION] * len(timestamps))

        seasons[np.isin(months, list(cls.WINTER_MONTHS))] = Season.WINTER
        seasons[np.isin(months, list(cls.SUMMER_MONTHS))] = Season.SUMMER

        return seasons

    @classmethod
    def is_winter(cls, timestamp: Union[pd.Timestamp, np.datetime64]) -> bool:
        """겨울철 여부"""
        return cls.classify(timestamp) == Season.WINTER

    @classmethod
    def is_winter_batch(cls, timestamps: pd.DatetimeIndex) -> np.ndarray:
        """배치 겨울철 여부"""
        months = timestamps.month
        return np.isin(months, list(cls.WINTER_MONTHS))


class InflectionDetector:
    """
    변곡점(급변구간) 감지기

    전력수요의 급격한 변화를 실시간으로 감지
    상위 percentile% 변화율을 변곡점으로 정의
    """

    def __init__(
        self,
        percentile: float = 95,
        lookback_window: int = 24,
        min_change_threshold: float = None
    ):
        """
        Args:
            percentile: 변곡점으로 간주할 변화율 백분위 (default: 95)
            lookback_window: 변화율 계산을 위한 과거 윈도우 (default: 24시간)
            min_change_threshold: 최소 변화량 임계값 (옵션)
        """
        self.percentile = percentile
        self.lookback_window = lookback_window
        self.min_change_threshold = min_change_threshold

        # 학습된 임계값
        self.threshold: Optional[float] = None
        self.historical_changes: Optional[np.ndarray] = None

    def fit(self, demand_series: pd.Series) -> 'InflectionDetector':
        """
        과거 데이터로 변곡점 임계값 학습

        Args:
            demand_series: 과거 전력수요 시계열
        """
        # 절대 변화량 계산
        changes = np.abs(demand_series.diff().dropna().values)

        # 임계값 계산 (상위 percentile)
        self.threshold = np.percentile(changes, self.percentile)
        self.historical_changes = changes

        if self.min_change_threshold is not None:
            self.threshold = max(self.threshold, self.min_change_threshold)

        return self

    def detect(self, current_demand: float, previous_demand: float) -> Tuple[bool, float]:
        """
        단일 시점 변곡점 감지

        Args:
            current_demand: 현재 수요
            previous_demand: 이전 시점 수요

        Returns:
            (is_inflection, change_magnitude)
        """
        if self.threshold is None:
            raise ValueError("fit()을 먼저 호출하세요")

        change = abs(current_demand - previous_demand)
        is_inflection = change >= self.threshold

        return is_inflection, change

    def detect_batch(self, demand_series: pd.Series) -> np.ndarray:
        """
        배치 변곡점 감지

        Args:
            demand_series: 전력수요 시계열

        Returns:
            변곡점 여부 boolean 배열
        """
        if self.threshold is None:
            raise ValueError("fit()을 먼저 호출하세요")

        changes = np.abs(demand_series.diff().values)
        changes[0] = 0  # 첫 번째 값은 NaN이므로 0으로

        return changes >= self.threshold

    def get_inflection_probability(self, change: float) -> float:
        """
        변화량에 대한 변곡점 확률 계산 (소프트 결정용)

        Sigmoid 함수로 부드러운 확률 반환
        """
        if self.threshold is None:
            raise ValueError("fit()을 먼저 호출하세요")

        # Sigmoid: threshold에서 0.5, threshold*2에서 ~0.88
        scale = self.threshold * 0.5
        prob = 1 / (1 + np.exp(-(change - self.threshold) / scale))

        return float(prob)


class ConditionalPredictor(nn.Module):
    """
    조건부 예측기: 상황에 따라 다른 모델 사용

    전략:
    - 기본: demand_only 모델 사용
    - 겨울철 + 변곡점: weather_full 모델 사용

    모드:
    1. "hard": 이진 선택 (demand_only 또는 weather_full)
    2. "soft": 가중 앙상블 (확률 기반)
    """

    def __init__(
        self,
        demand_only_model: nn.Module,
        weather_full_model: nn.Module,
        inflection_detector: InflectionDetector,
        mode: str = "hard",
        weather_weight_cap: float = 0.7
    ):
        """
        Args:
            demand_only_model: 기상변수 없는 모델
            weather_full_model: 기상변수 포함 모델
            inflection_detector: 변곡점 감지기
            mode: "hard" (이진) 또는 "soft" (앙상블)
            weather_weight_cap: soft 모드에서 weather 모델 최대 가중치
        """
        super().__init__()

        self.demand_only_model = demand_only_model
        self.weather_full_model = weather_full_model
        self.inflection_detector = inflection_detector
        self.mode = mode
        self.weather_weight_cap = weather_weight_cap

        # 통계 추적
        self.prediction_stats = {
            'total': 0,
            'demand_only_used': 0,
            'weather_full_used': 0,
            'soft_blend_used': 0
        }

    def should_use_weather(
        self,
        timestamp: pd.Timestamp,
        recent_demand: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, PredictionContext]:
        """
        기상변수 모델 사용 여부 결정

        Args:
            timestamp: 예측 시점
            recent_demand: 최근 수요 데이터 (변곡점 감지용)

        Returns:
            (use_weather, weather_weight, context)
        """
        # 계절 확인
        is_winter = SeasonClassifier.is_winter(timestamp)
        season = SeasonClassifier.classify(timestamp)

        # 변곡점 확인
        is_inflection = False
        recent_change = 0.0
        inflection_prob = 0.0

        if recent_demand is not None and len(recent_demand) >= 2:
            is_inflection, recent_change = self.inflection_detector.detect(
                recent_demand[-1], recent_demand[-2]
            )
            inflection_prob = self.inflection_detector.get_inflection_probability(recent_change)

        # 결정 로직
        use_weather = is_winter and is_inflection

        # Soft 모드: 가중치 계산
        if self.mode == "soft":
            if is_winter:
                # 겨울철: 변곡점 확률에 비례한 가중치
                weather_weight = min(inflection_prob, self.weather_weight_cap)
            else:
                # 비겨울철: 기상변수 사용 안함
                weather_weight = 0.0
        else:
            # Hard 모드: 0 또는 1
            weather_weight = 1.0 if use_weather else 0.0

        # 컨텍스트 생성
        context = PredictionContext(
            timestamp=timestamp,
            season=season,
            is_inflection=is_inflection,
            recent_demand_change=recent_change,
            use_weather=use_weather,
            confidence=1.0 - abs(weather_weight - 0.5) * 2  # 중간값일수록 낮은 신뢰도
        )

        return use_weather, weather_weight, context

    def forward(
        self,
        x_demand: torch.Tensor,
        x_weather: torch.Tensor,
        timestamps: Optional[pd.DatetimeIndex] = None,
        recent_demands: Optional[np.ndarray] = None
    ) -> Tuple[torch.Tensor, List[PredictionContext]]:
        """
        조건부 예측 수행

        Args:
            x_demand: demand_only 모델 입력 (batch, seq_len, demand_features)
            x_weather: weather_full 모델 입력 (batch, seq_len, all_features)
            timestamps: 예측 시점 타임스탬프 (배치)
            recent_demands: 최근 수요 데이터 (배치)

        Returns:
            predictions, contexts
        """
        batch_size = x_demand.shape[0]
        contexts = []

        # 기본: demand_only 예측
        pred_demand = self.demand_only_model(x_demand)

        # weather_full 예측
        pred_weather = self.weather_full_model(x_weather)

        # 배치별 가중치 계산
        weights = torch.zeros(batch_size, 1, device=x_demand.device)

        if timestamps is not None:
            for i in range(batch_size):
                ts = timestamps[i] if i < len(timestamps) else pd.Timestamp.now()
                recent = recent_demands[i] if recent_demands is not None and i < len(recent_demands) else None

                use_weather, weight, context = self.should_use_weather(ts, recent)
                weights[i] = weight
                contexts.append(context)

                # 통계 업데이트
                self.prediction_stats['total'] += 1
                if weight == 0:
                    self.prediction_stats['demand_only_used'] += 1
                elif weight == 1:
                    self.prediction_stats['weather_full_used'] += 1
                else:
                    self.prediction_stats['soft_blend_used'] += 1
        else:
            # 타임스탬프 없으면 기본값 (demand_only)
            contexts = [None] * batch_size
            self.prediction_stats['total'] += batch_size
            self.prediction_stats['demand_only_used'] += batch_size

        # 가중 앙상블
        predictions = (1 - weights) * pred_demand + weights * pred_weather

        return predictions, contexts

    def predict_single(
        self,
        x_demand: torch.Tensor,
        x_weather: torch.Tensor,
        timestamp: pd.Timestamp,
        recent_demand: Optional[np.ndarray] = None
    ) -> Tuple[float, PredictionContext]:
        """단일 예측"""
        self.eval()
        with torch.no_grad():
            pred, contexts = self.forward(
                x_demand.unsqueeze(0),
                x_weather.unsqueeze(0),
                pd.DatetimeIndex([timestamp]),
                [recent_demand] if recent_demand is not None else None
            )
        return pred.item(), contexts[0]

    def get_stats(self) -> Dict[str, float]:
        """예측 통계 반환"""
        total = max(self.prediction_stats['total'], 1)
        return {
            'total_predictions': self.prediction_stats['total'],
            'demand_only_ratio': self.prediction_stats['demand_only_used'] / total,
            'weather_full_ratio': self.prediction_stats['weather_full_used'] / total,
            'soft_blend_ratio': self.prediction_stats['soft_blend_used'] / total
        }

    def reset_stats(self):
        """통계 초기화"""
        self.prediction_stats = {
            'total': 0,
            'demand_only_used': 0,
            'weather_full_used': 0,
            'soft_blend_used': 0
        }


class AdaptiveConditionalPredictor(ConditionalPredictor):
    """
    적응형 조건부 예측기

    실시간으로 각 모델의 성능을 추적하고
    가중치를 동적으로 조정
    """

    def __init__(
        self,
        demand_only_model: nn.Module,
        weather_full_model: nn.Module,
        inflection_detector: InflectionDetector,
        adaptation_rate: float = 0.1,
        performance_window: int = 100
    ):
        super().__init__(
            demand_only_model,
            weather_full_model,
            inflection_detector,
            mode="soft"
        )

        self.adaptation_rate = adaptation_rate
        self.performance_window = performance_window

        # 성능 추적 버퍼
        self.demand_errors: List[float] = []
        self.weather_errors: List[float] = []

        # 학습된 조건별 가중치
        self.learned_weights = {
            (Season.WINTER, True): 0.7,   # 겨울 + 변곡점
            (Season.WINTER, False): 0.3,  # 겨울 + 일반
            (Season.SUMMER, True): 0.1,   # 여름 + 변곡점
            (Season.SUMMER, False): 0.0,  # 여름 + 일반
            (Season.TRANSITION, True): 0.2,   # 전환기 + 변곡점
            (Season.TRANSITION, False): 0.0,  # 전환기 + 일반
        }

    def update_performance(
        self,
        actual: float,
        pred_demand: float,
        pred_weather: float,
        context: PredictionContext
    ):
        """예측 성능 업데이트"""
        demand_error = abs(actual - pred_demand)
        weather_error = abs(actual - pred_weather)

        self.demand_errors.append(demand_error)
        self.weather_errors.append(weather_error)

        # 윈도우 유지
        if len(self.demand_errors) > self.performance_window:
            self.demand_errors.pop(0)
            self.weather_errors.pop(0)

        # 가중치 업데이트
        if len(self.demand_errors) >= 10:  # 최소 샘플
            key = (context.season, context.is_inflection)

            avg_demand_error = np.mean(self.demand_errors[-10:])
            avg_weather_error = np.mean(self.weather_errors[-10:])

            # weather가 더 좋으면 가중치 증가
            if avg_weather_error < avg_demand_error:
                improvement = (avg_demand_error - avg_weather_error) / avg_demand_error
                self.learned_weights[key] = min(
                    1.0,
                    self.learned_weights[key] + self.adaptation_rate * improvement
                )
            else:
                degradation = (avg_weather_error - avg_demand_error) / avg_demand_error
                self.learned_weights[key] = max(
                    0.0,
                    self.learned_weights[key] - self.adaptation_rate * degradation
                )

    def should_use_weather(
        self,
        timestamp: pd.Timestamp,
        recent_demand: Optional[np.ndarray] = None
    ) -> Tuple[bool, float, PredictionContext]:
        """적응형 가중치로 결정"""
        season = SeasonClassifier.classify(timestamp)

        is_inflection = False
        recent_change = 0.0

        if recent_demand is not None and len(recent_demand) >= 2:
            is_inflection, recent_change = self.inflection_detector.detect(
                recent_demand[-1], recent_demand[-2]
            )

        # 학습된 가중치 사용
        key = (season, is_inflection)
        weather_weight = self.learned_weights.get(key, 0.0)

        use_weather = weather_weight > 0.5

        context = PredictionContext(
            timestamp=timestamp,
            season=season,
            is_inflection=is_inflection,
            recent_demand_change=recent_change,
            use_weather=use_weather,
            confidence=abs(weather_weight - 0.5) * 2
        )

        return use_weather, weather_weight, context


def create_conditional_predictor(
    demand_only_model: nn.Module,
    weather_full_model: nn.Module,
    train_demand: pd.Series,
    mode: str = "hard",
    adaptive: bool = False,
    inflection_percentile: float = 95
) -> Union[ConditionalPredictor, AdaptiveConditionalPredictor]:
    """
    조건부 예측기 생성 팩토리 함수

    Args:
        demand_only_model: 기상변수 없는 모델
        weather_full_model: 기상변수 포함 모델
        train_demand: 학습 데이터의 수요 시계열 (변곡점 임계값 학습용)
        mode: "hard" 또는 "soft"
        adaptive: 적응형 예측기 사용 여부
        inflection_percentile: 변곡점 백분위

    Returns:
        ConditionalPredictor 또는 AdaptiveConditionalPredictor
    """
    # 변곡점 감지기 생성 및 학습
    detector = InflectionDetector(percentile=inflection_percentile)
    detector.fit(train_demand)

    if adaptive:
        return AdaptiveConditionalPredictor(
            demand_only_model=demand_only_model,
            weather_full_model=weather_full_model,
            inflection_detector=detector
        )
    else:
        return ConditionalPredictor(
            demand_only_model=demand_only_model,
            weather_full_model=weather_full_model,
            inflection_detector=detector,
            mode=mode
        )
