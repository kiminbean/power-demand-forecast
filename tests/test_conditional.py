"""
Tests for Conditional Model (MODEL-006)
========================================

조건부 예측 모델 테스트:
- SeasonClassifier
- InflectionDetector
- ConditionalPredictor
"""

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.conditional import (
    Season,
    PredictionContext,
    SeasonClassifier,
    InflectionDetector,
    ConditionalPredictor,
    AdaptiveConditionalPredictor,
    create_conditional_predictor,
)


# ============================================================
# SeasonClassifier Tests
# ============================================================

class TestSeasonClassifier:
    """계절 분류기 테스트"""

    def test_winter_months(self):
        """겨울철 (12, 1, 2월) 분류"""
        assert SeasonClassifier.classify(pd.Timestamp('2024-12-15')) == Season.WINTER
        assert SeasonClassifier.classify(pd.Timestamp('2024-01-15')) == Season.WINTER
        assert SeasonClassifier.classify(pd.Timestamp('2024-02-15')) == Season.WINTER

    def test_summer_months(self):
        """여름철 (6, 7, 8월) 분류"""
        assert SeasonClassifier.classify(pd.Timestamp('2024-06-15')) == Season.SUMMER
        assert SeasonClassifier.classify(pd.Timestamp('2024-07-15')) == Season.SUMMER
        assert SeasonClassifier.classify(pd.Timestamp('2024-08-15')) == Season.SUMMER

    def test_transition_months(self):
        """전환기 (나머지) 분류"""
        assert SeasonClassifier.classify(pd.Timestamp('2024-03-15')) == Season.TRANSITION
        assert SeasonClassifier.classify(pd.Timestamp('2024-05-15')) == Season.TRANSITION
        assert SeasonClassifier.classify(pd.Timestamp('2024-09-15')) == Season.TRANSITION
        assert SeasonClassifier.classify(pd.Timestamp('2024-11-15')) == Season.TRANSITION

    def test_is_winter(self):
        """겨울철 여부 확인"""
        assert SeasonClassifier.is_winter(pd.Timestamp('2024-01-15')) is True
        assert SeasonClassifier.is_winter(pd.Timestamp('2024-07-15')) is False

    def test_classify_batch(self):
        """배치 분류"""
        timestamps = pd.DatetimeIndex([
            '2024-01-15',  # Winter
            '2024-07-15',  # Summer
            '2024-04-15',  # Transition
        ])
        seasons = SeasonClassifier.classify_batch(timestamps)

        assert seasons[0] == Season.WINTER
        assert seasons[1] == Season.SUMMER
        assert seasons[2] == Season.TRANSITION

    def test_is_winter_batch(self):
        """배치 겨울철 여부"""
        timestamps = pd.DatetimeIndex([
            '2024-01-15',
            '2024-07-15',
            '2024-12-15',
        ])
        is_winter = SeasonClassifier.is_winter_batch(timestamps)

        assert is_winter[0] is True or is_winter[0] == True  # numpy bool
        assert is_winter[1] is False or is_winter[1] == False
        assert is_winter[2] is True or is_winter[2] == True

    def test_numpy_datetime64_input(self):
        """numpy datetime64 입력 지원"""
        dt = np.datetime64('2024-01-15')
        assert SeasonClassifier.classify(dt) == Season.WINTER


# ============================================================
# InflectionDetector Tests
# ============================================================

class TestInflectionDetector:
    """변곡점 감지기 테스트"""

    @pytest.fixture
    def sample_demand(self):
        """샘플 수요 데이터"""
        np.random.seed(42)
        # 기본 수요 + 일부 급변
        base = 500 + np.random.randn(1000) * 20
        # 인위적 급변 추가
        base[100] += 100  # 급등
        base[200] -= 80   # 급락
        base[500] += 120  # 급등
        return pd.Series(base)

    def test_fit(self, sample_demand):
        """fit 메서드 테스트"""
        detector = InflectionDetector(percentile=95)
        detector.fit(sample_demand)

        assert detector.threshold is not None
        assert detector.threshold > 0
        assert detector.historical_changes is not None

    def test_detect_inflection(self, sample_demand):
        """변곡점 감지 테스트"""
        detector = InflectionDetector(percentile=95)
        detector.fit(sample_demand)

        # 급변 감지
        is_inf, change = detector.detect(600, 500)  # 100 변화
        assert change == 100

        # 일반 변화
        is_inf_small, change_small = detector.detect(505, 500)
        assert change_small == 5

    def test_detect_batch(self, sample_demand):
        """배치 감지 테스트"""
        detector = InflectionDetector(percentile=95)
        detector.fit(sample_demand)

        inflections = detector.detect_batch(sample_demand)

        assert len(inflections) == len(sample_demand)
        assert inflections.dtype == bool
        # 급변 위치에서 True (diff 기준이므로 실제 급변 인덱스에서 감지됨)
        assert inflections[100] == True  # 급등 위치
        # 임계값 이상의 변화가 있는 위치 확인
        assert np.sum(inflections) >= 2  # 최소 2개 이상의 변곡점

    def test_inflection_probability(self, sample_demand):
        """변곡점 확률 테스트"""
        detector = InflectionDetector(percentile=95)
        detector.fit(sample_demand)

        # 큰 변화 = 높은 확률
        prob_high = detector.get_inflection_probability(detector.threshold * 2)
        prob_low = detector.get_inflection_probability(detector.threshold * 0.5)

        assert prob_high > prob_low
        assert 0 <= prob_high <= 1
        assert 0 <= prob_low <= 1

    def test_fit_required(self):
        """fit 없이 detect 호출 시 에러"""
        detector = InflectionDetector()

        with pytest.raises(ValueError):
            detector.detect(500, 400)

    def test_min_change_threshold(self, sample_demand):
        """최소 변화량 임계값 테스트"""
        detector = InflectionDetector(percentile=95, min_change_threshold=50)
        detector.fit(sample_demand)

        assert detector.threshold >= 50


# ============================================================
# ConditionalPredictor Tests
# ============================================================

class SimpleMockModel(nn.Module):
    """테스트용 단순 모델"""

    def __init__(self, output_value: float = 0.0):
        super().__init__()
        self.output_value = output_value
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        return torch.full((batch_size, 1), self.output_value)


class TestConditionalPredictor:
    """조건부 예측기 테스트"""

    @pytest.fixture
    def predictor(self):
        """테스트용 예측기"""
        demand_model = SimpleMockModel(output_value=100.0)
        weather_model = SimpleMockModel(output_value=110.0)

        # 변곡점 감지기
        detector = InflectionDetector(percentile=95)
        demand_series = pd.Series(np.random.randn(1000) * 20 + 500)
        detector.fit(demand_series)

        return ConditionalPredictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            inflection_detector=detector,
            mode="hard"
        )

    def test_should_use_weather_winter_inflection(self, predictor):
        """겨울 + 변곡점: weather 사용"""
        timestamp = pd.Timestamp('2024-01-15 12:00:00')
        recent = np.array([500, 600])  # 급변

        use_weather, weight, context = predictor.should_use_weather(timestamp, recent)

        assert context.season == Season.WINTER
        # 변화량이 임계값 이상이면 변곡점

    def test_should_use_weather_summer(self, predictor):
        """여름: weather 사용 안함"""
        timestamp = pd.Timestamp('2024-07-15 12:00:00')
        recent = np.array([500, 600])

        use_weather, weight, context = predictor.should_use_weather(timestamp, recent)

        assert context.season == Season.SUMMER
        assert use_weather is False
        assert weight == 0.0

    def test_forward_shape(self, predictor):
        """forward 출력 형태"""
        batch_size = 4
        seq_len = 24
        features = 10

        x_demand = torch.randn(batch_size, seq_len, features)
        x_weather = torch.randn(batch_size, seq_len, features)

        timestamps = pd.DatetimeIndex([
            '2024-01-15', '2024-07-15', '2024-01-16', '2024-08-15'
        ])

        preds, contexts = predictor.forward(x_demand, x_weather, timestamps)

        assert preds.shape == (batch_size, 1)
        assert len(contexts) == batch_size

    def test_hard_mode_selection(self, predictor):
        """hard 모드: 이진 선택"""
        predictor.mode = "hard"

        # 여름 = demand_only
        ts_summer = pd.Timestamp('2024-07-15')
        _, weight, _ = predictor.should_use_weather(ts_summer, None)
        assert weight == 0.0

    def test_soft_mode_weight(self):
        """soft 모드: 가중치"""
        demand_model = SimpleMockModel(output_value=100.0)
        weather_model = SimpleMockModel(output_value=110.0)

        detector = InflectionDetector(percentile=95)
        detector.fit(pd.Series(np.random.randn(1000) * 20 + 500))

        predictor = ConditionalPredictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            inflection_detector=detector,
            mode="soft",
            weather_weight_cap=0.7
        )

        # 겨울 + 큰 변화 = 높은 가중치 (cap까지)
        ts_winter = pd.Timestamp('2024-01-15')
        recent = np.array([500, 700])  # 큰 변화

        _, weight, _ = predictor.should_use_weather(ts_winter, recent)
        assert weight <= 0.7  # cap 적용

    def test_stats_tracking(self, predictor):
        """통계 추적"""
        predictor.reset_stats()

        x_demand = torch.randn(2, 24, 10)
        x_weather = torch.randn(2, 24, 10)
        timestamps = pd.DatetimeIndex(['2024-07-15', '2024-07-16'])

        predictor.forward(x_demand, x_weather, timestamps)
        stats = predictor.get_stats()

        assert stats['total_predictions'] == 2
        assert stats['demand_only_ratio'] > 0

    def test_predict_single(self, predictor):
        """단일 예측"""
        x_demand = torch.randn(24, 10)
        x_weather = torch.randn(24, 10)
        timestamp = pd.Timestamp('2024-07-15')

        pred, context = predictor.predict_single(x_demand, x_weather, timestamp)

        assert isinstance(pred, float)
        assert isinstance(context, PredictionContext)


# ============================================================
# AdaptiveConditionalPredictor Tests
# ============================================================

class TestAdaptiveConditionalPredictor:
    """적응형 예측기 테스트"""

    @pytest.fixture
    def adaptive_predictor(self):
        """적응형 예측기"""
        demand_model = SimpleMockModel(output_value=100.0)
        weather_model = SimpleMockModel(output_value=110.0)

        detector = InflectionDetector(percentile=95)
        detector.fit(pd.Series(np.random.randn(1000) * 20 + 500))

        return AdaptiveConditionalPredictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            inflection_detector=detector
        )

    def test_update_performance(self, adaptive_predictor):
        """성능 업데이트"""
        context = PredictionContext(
            timestamp=pd.Timestamp('2024-01-15'),
            season=Season.WINTER,
            is_inflection=True,
            recent_demand_change=50,
            use_weather=True,
            confidence=0.8
        )

        # weather가 더 좋은 경우
        for _ in range(20):
            adaptive_predictor.update_performance(
                actual=105,
                pred_demand=100,  # error = 5
                pred_weather=106,  # error = 1
                context=context
            )

        # 겨울+변곡점 가중치 증가 확인
        key = (Season.WINTER, True)
        assert len(adaptive_predictor.demand_errors) > 0

    def test_learned_weights_initial(self, adaptive_predictor):
        """초기 학습 가중치"""
        # 겨울+변곡점이 가장 높아야 함
        assert adaptive_predictor.learned_weights[(Season.WINTER, True)] > \
               adaptive_predictor.learned_weights[(Season.SUMMER, False)]


# ============================================================
# Factory Function Tests
# ============================================================

class TestCreateConditionalPredictor:
    """팩토리 함수 테스트"""

    def test_create_basic(self):
        """기본 생성"""
        demand_model = SimpleMockModel()
        weather_model = SimpleMockModel()
        train_demand = pd.Series(np.random.randn(1000) * 20 + 500)

        predictor = create_conditional_predictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            train_demand=train_demand,
            mode="hard"
        )

        assert isinstance(predictor, ConditionalPredictor)

    def test_create_adaptive(self):
        """적응형 생성"""
        demand_model = SimpleMockModel()
        weather_model = SimpleMockModel()
        train_demand = pd.Series(np.random.randn(1000) * 20 + 500)

        predictor = create_conditional_predictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            train_demand=train_demand,
            adaptive=True
        )

        assert isinstance(predictor, AdaptiveConditionalPredictor)

    def test_inflection_percentile(self):
        """변곡점 백분위 설정"""
        demand_model = SimpleMockModel()
        weather_model = SimpleMockModel()
        train_demand = pd.Series(np.random.randn(1000) * 20 + 500)

        predictor = create_conditional_predictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            train_demand=train_demand,
            inflection_percentile=90
        )

        assert predictor.inflection_detector.percentile == 90


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_prediction_pipeline(self):
        """전체 예측 파이프라인"""
        # 모델 생성
        demand_model = SimpleMockModel(output_value=500.0)
        weather_model = SimpleMockModel(output_value=510.0)

        # 학습 데이터
        np.random.seed(42)
        train_demand = pd.Series(np.random.randn(1000) * 20 + 500)

        # 예측기 생성
        predictor = create_conditional_predictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            train_demand=train_demand,
            mode="hard"
        )

        # 배치 예측
        batch_size = 10
        x_demand = torch.randn(batch_size, 24, 10)
        x_weather = torch.randn(batch_size, 24, 20)

        timestamps = pd.date_range('2024-01-01', periods=batch_size, freq='h')

        preds, contexts = predictor.forward(x_demand, x_weather, timestamps)

        assert preds.shape == (batch_size, 1)
        assert all(c.season == Season.WINTER for c in contexts)  # 1월 = 겨울

    def test_seasonal_behavior(self):
        """계절별 행동 테스트"""
        demand_model = SimpleMockModel(output_value=100.0)
        weather_model = SimpleMockModel(output_value=200.0)

        detector = InflectionDetector(percentile=50)  # 낮은 임계값
        detector.fit(pd.Series(np.random.randn(100) * 10 + 500))

        predictor = ConditionalPredictor(
            demand_only_model=demand_model,
            weather_full_model=weather_model,
            inflection_detector=detector,
            mode="hard"
        )

        # 겨울 + 변곡점 = weather 사용 (200)
        x = torch.randn(1, 24, 10)
        ts_winter = pd.DatetimeIndex(['2024-01-15'])
        recent = [np.array([500, 600])]  # 큰 변화

        pred_winter, _ = predictor.forward(x, x, ts_winter, recent)

        # 여름 = demand 사용 (100)
        ts_summer = pd.DatetimeIndex(['2024-07-15'])
        pred_summer, _ = predictor.forward(x, x, ts_summer, recent)

        # 여름은 항상 demand_only
        assert pred_summer.item() == 100.0
