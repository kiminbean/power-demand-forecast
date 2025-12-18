"""
SMP 모듈 테스트
================

SMP 예측 및 입찰 시스템 통합 테스트
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
# SMP Crawler Tests
# ============================================================

class TestSMPCrawler:
    """SMP 크롤러 테스트"""

    def test_smp_crawler_import(self):
        """SMP 크롤러 임포트 테스트"""
        from src.smp.crawlers import SMPCrawler, SMPDataStore
        assert SMPCrawler is not None
        assert SMPDataStore is not None

    def test_smp_data_dataclass(self):
        """SMPData 데이터클래스 테스트"""
        from src.smp.crawlers.smp_crawler import SMPData

        data = SMPData(
            timestamp="2025-12-18 14:00",
            date="2025-12-18",
            hour=14,
            interval=1,
            smp_mainland=165.5,
            smp_jeju=158.2,
        )

        assert data.hour == 14
        assert data.smp_mainland == 165.5
        assert data.smp_jeju == 158.2


# ============================================================
# SMP Model Tests
# ============================================================

class TestSMPModels:
    """SMP 모델 테스트"""

    def test_lstm_import(self):
        """LSTM 모델 임포트 테스트"""
        from src.smp.models import SMPLSTMModel, SMPQuantileLSTM
        assert SMPLSTMModel is not None
        assert SMPQuantileLSTM is not None

    def test_tft_import(self):
        """TFT 모델 임포트 테스트"""
        from src.smp.models import SMPTFTModel
        assert SMPTFTModel is not None

    def test_lstm_forward_pass(self):
        """LSTM 순전파 테스트"""
        import torch
        from src.smp.models import SMPLSTMModel

        model = SMPLSTMModel(
            input_size=10,
            hidden_size=32,
            num_layers=1,
            dropout=0.1
        )

        batch_size = 4
        seq_len = 48
        input_size = 10

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        # Output should have 24 hours prediction
        assert output.shape[0] == batch_size
        assert output.shape[1] == 24

    def test_tft_forward_pass(self):
        """TFT 순전파 테스트"""
        import torch
        from src.smp.models import SMPTFTModel

        model = SMPTFTModel(
            input_size=10,
            hidden_size=32,
            num_heads=4,
            num_layers=1,
        )

        batch_size = 4
        seq_len = 48
        input_size = 10

        x = torch.randn(batch_size, seq_len, input_size)
        output = model(x)

        assert 'q10' in output
        assert 'q50' in output
        assert 'q90' in output
        assert output['q50'].shape[0] == batch_size


# ============================================================
# Generation Predictor Tests
# ============================================================

class TestGenerationPredictor:
    """발전량 예측기 테스트"""

    def test_solar_calculator(self):
        """태양광 계산기 테스트"""
        from src.smp.models import SolarPowerCalculator

        calc = SolarPowerCalculator(capacity_kw=1000, efficiency=0.85)

        # 정오, 맑은 날
        power = calc.calculate(
            irradiance=1000,
            temperature=25,
        )

        # 효율 반영하여 850kW 근처 예상
        assert 700 <= power <= 1000

    def test_wind_calculator(self):
        """풍력 계산기 테스트"""
        from src.smp.models import WindPowerCalculator

        calc = WindPowerCalculator(capacity_kw=1000, efficiency=0.85)

        # 정격 풍속
        power = calc.calculate(wind_speed=12)

        assert 800 <= power <= 1000

        # Cut-in 이하
        power_low = calc.calculate(wind_speed=2)

        assert power_low == 0

    def test_generation_predictor(self):
        """통합 발전량 예측기 테스트"""
        from src.smp.models import GenerationPredictor, PlantConfig

        config = PlantConfig(
            plant_type="solar",
            capacity_kw=1000,
            efficiency=0.85,
            location=(33.5, 126.5),
            name="태양광1호"
        )

        predictor = GenerationPredictor(config)

        # predict() with hours=24 returns 24 predictions
        predictions = predictor.predict(
            temperature=[25] * 24,
            irradiance=[800] * 24,
            hours=24
        )

        assert len(predictions) == 24


# ============================================================
# Bidding Strategy Tests
# ============================================================

class TestBiddingStrategy:
    """입찰 전략 테스트"""

    def test_optimizer_import(self):
        """옵티마이저 임포트 테스트"""
        from src.smp.bidding import (
            BiddingStrategyOptimizer,
            RevenueCalculator,
            RiskAnalyzer
        )
        assert BiddingStrategyOptimizer is not None
        assert RevenueCalculator is not None
        assert RiskAnalyzer is not None

    def test_optimize_with_array(self):
        """배열 입력 최적화 테스트"""
        from src.smp.bidding import BiddingStrategyOptimizer

        optimizer = BiddingStrategyOptimizer()

        smp = np.array([
            80, 85, 82, 80, 85, 95, 110, 130, 145, 155, 160, 165,
            168, 165, 158, 145, 135, 125, 115, 105, 95, 88, 82, 78
        ])

        gen = np.array([
            0, 0, 0, 0, 50, 150, 400, 600, 750, 850, 900, 920,
            900, 850, 750, 600, 400, 150, 50, 0, 0, 0, 0, 0
        ])

        strategy = optimizer.optimize(smp, gen, 1000, 0.5)

        assert len(strategy.recommended_hours) > 0
        assert strategy.total_revenue > 0
        assert strategy.average_smp > 0

    def test_optimize_with_dict(self):
        """딕셔너리 입력 최적화 테스트"""
        from src.smp.bidding import BiddingStrategyOptimizer

        optimizer = BiddingStrategyOptimizer()

        smp = {
            'q10': np.array([100] * 24) * 0.85,
            'q50': np.array([100] * 24),
            'q90': np.array([100] * 24) * 1.15,
        }

        gen = np.array([500] * 24)

        strategy = optimizer.optimize(smp, gen, 1000, 0.5)

        assert len(strategy.recommended_hours) > 0

    def test_revenue_simulation(self):
        """수익 시뮬레이션 테스트"""
        from src.smp.bidding import RevenueCalculator

        calculator = RevenueCalculator()

        smp = np.vstack([
            np.array([100] * 24) * 0.85,
            np.array([100] * 24),
            np.array([100] * 24) * 1.15,
        ])

        gen = np.array([500] * 24)

        result = calculator.simulate(smp, gen, hours=24)

        assert 'expected' in result
        assert 'best_case' in result
        assert 'worst_case' in result
        assert result['best_case'] >= result['expected']
        assert result['expected'] >= result['worst_case']

    def test_risk_levels(self):
        """리스크 수준별 테스트"""
        from src.smp.bidding import BiddingStrategyOptimizer

        optimizer = BiddingStrategyOptimizer()

        smp = np.array([100] * 24)
        gen = np.array([500] * 24)

        # Conservative
        strat_c = optimizer.optimize(smp, gen, 1000, 0.2)
        # Moderate
        strat_m = optimizer.optimize(smp, gen, 1000, 0.5)
        # Aggressive
        strat_a = optimizer.optimize(smp, gen, 1000, 0.8)

        assert strat_c.risk_level == "conservative"
        assert strat_m.risk_level == "moderate"
        assert strat_a.risk_level == "aggressive"


# ============================================================
# API Schema Tests
# ============================================================

class TestAPISchemas:
    """API 스키마 테스트"""

    def test_smp_schemas(self):
        """SMP 스키마 테스트"""
        from api.smp_schemas import (
            SMPRegion,
            SMPPredictionRequest,
            SMPDataPoint
        )

        req = SMPPredictionRequest(region=SMPRegion.JEJU, hours=24)
        assert req.region == SMPRegion.JEJU
        assert req.hours == 24

    def test_bidding_schemas(self):
        """Bidding 스키마 테스트"""
        from api.bidding_schemas import (
            EnergyType,
            RiskLevel,
            BiddingStrategyRequest
        )

        req = BiddingStrategyRequest(
            capacity_kw=1000,
            energy_type=EnergyType.SOLAR,
            risk_level=RiskLevel.MODERATE
        )

        assert req.capacity_kw == 1000
        assert req.energy_type == EnergyType.SOLAR


# ============================================================
# Integration Tests
# ============================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_pipeline(self):
        """전체 파이프라인 테스트"""
        from src.smp.bidding import BiddingStrategyOptimizer, RevenueCalculator
        from src.smp.models import GenerationPredictor, PlantConfig

        # 1. 발전량 예측
        config = PlantConfig(
            plant_type="solar",
            capacity_kw=1000,
            efficiency=0.85,
            location=(33.5, 126.5),
            name="테스트태양광"
        )
        predictor = GenerationPredictor(config)

        gen_predictions = predictor.predict(
            temperature=[25] * 24,
            irradiance=[800] * 24,
            hours=24
        )

        gen_values = np.array([p.generation_kw for p in gen_predictions])

        # 2. SMP 예측 (mock)
        smp_predictions = np.array([
            80, 85, 82, 80, 85, 95, 110, 130, 145, 155, 160, 165,
            168, 165, 158, 145, 135, 125, 115, 105, 95, 88, 82, 78
        ])

        # 3. 입찰 전략 최적화
        optimizer = BiddingStrategyOptimizer()
        strategy = optimizer.optimize(
            smp_predictions,
            gen_values,
            capacity_kw=1000,
            risk_tolerance=0.5
        )

        # 4. 수익 시뮬레이션
        calculator = RevenueCalculator()
        smp_scenarios = np.vstack([
            smp_predictions * 0.85,
            smp_predictions,
            smp_predictions * 1.15
        ])
        simulation = calculator.simulate(smp_scenarios, gen_values)

        # 검증
        assert len(strategy.recommended_hours) > 0
        assert strategy.total_revenue > 0
        assert simulation['expected'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
