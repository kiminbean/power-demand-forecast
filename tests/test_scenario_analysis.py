"""
시나리오 분석 테스트 (Task 24)
================================
Scenario analysis 모듈 테스트
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 테스트용 모델
# ============================================================================

class SimpleModel(nn.Module):
    """테스트용 간단한 모델"""

    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 24):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 3:
            # (batch, seq, features) -> (batch, seq*features)
            x = x.view(x.size(0), -1)
            # Pad or truncate to match input size
            if x.size(1) < 10:
                x = nn.functional.pad(x, (0, 10 - x.size(1)))
            else:
                x = x[:, :10]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def simple_model():
    """간단한 모델 fixture"""
    model = SimpleModel(input_size=10, hidden_size=32, output_size=24)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """샘플 입력"""
    torch.manual_seed(42)
    return torch.randn(1, 10)


@pytest.fixture
def feature_names():
    """피처 이름"""
    return ['temperature', 'humidity', 'wind_speed', 'hour', 'day',
            'month', 'demand_lag1', 'demand_lag24', 'solar', 'cloud']


@pytest.fixture
def timestamps():
    """타임스탬프"""
    base = datetime.now()
    return [base + timedelta(hours=i) for i in range(24)]


# ============================================================================
# ScenarioType 테스트
# ============================================================================

class TestScenarioType:
    """ScenarioType 테스트"""

    def test_scenario_type_values(self):
        """시나리오 타입 값"""
        from src.analysis.scenario_analysis import ScenarioType

        assert ScenarioType.BASELINE.value == "baseline"
        assert ScenarioType.HEATWAVE.value == "heatwave"
        assert ScenarioType.COLDWAVE.value == "coldwave"


# ============================================================================
# ScenarioConfig 테스트
# ============================================================================

class TestScenarioConfig:
    """ScenarioConfig 테스트"""

    def test_config_creation(self):
        """설정 생성"""
        from src.analysis.scenario_analysis import ScenarioConfig, ScenarioType

        config = ScenarioConfig(
            name="Test Heatwave",
            scenario_type=ScenarioType.HEATWAVE,
            description="Test scenario",
            temperature_delta=5.0,
            humidity_delta=-5.0,
            demand_factor=1.15
        )

        assert config.name == "Test Heatwave"
        assert config.temperature_delta == 5.0
        assert config.demand_factor == 1.15

    def test_config_to_dict(self):
        """딕셔너리 변환"""
        from src.analysis.scenario_analysis import ScenarioConfig, ScenarioType

        config = ScenarioConfig(
            name="Test",
            scenario_type=ScenarioType.CUSTOM,
            temperature_delta=3.0
        )

        result = config.to_dict()

        assert 'name' in result
        assert result['temperature_delta'] == 3.0
        assert result['scenario_type'] == 'custom'


# ============================================================================
# ScenarioResult 테스트
# ============================================================================

class TestScenarioResult:
    """ScenarioResult 테스트"""

    def test_result_creation(self, timestamps):
        """결과 생성"""
        from src.analysis.scenario_analysis import (
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        config = ScenarioConfig("Test", ScenarioType.NORMAL)
        predictions = np.random.rand(24) * 100

        result = ScenarioResult(
            scenario_config=config,
            predictions=predictions,
            timestamps=timestamps
        )

        assert result.scenario_config.name == "Test"
        assert len(result.predictions) == 24

    def test_calculate_stats(self, timestamps):
        """통계 계산"""
        from src.analysis.scenario_analysis import (
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        config = ScenarioConfig("Test", ScenarioType.NORMAL)
        predictions = np.array([50, 60, 70, 80, 100, 90, 80, 70] * 3)[:24]

        result = ScenarioResult(
            scenario_config=config,
            predictions=predictions,
            timestamps=timestamps
        )
        result.calculate_stats()

        assert result.mean_demand > 0
        assert result.max_demand >= result.mean_demand
        assert result.min_demand <= result.mean_demand
        assert 0 <= result.peak_hour < 24

    def test_change_from_baseline(self, timestamps):
        """기준 대비 변화 계산"""
        from src.analysis.scenario_analysis import (
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        config = ScenarioConfig("Test", ScenarioType.HEATWAVE)
        predictions = np.ones(24) * 110  # 10% 증가
        baseline = np.ones(24) * 100

        result = ScenarioResult(
            scenario_config=config,
            predictions=predictions,
            timestamps=timestamps,
            baseline_predictions=baseline
        )
        result.calculate_stats()

        assert abs(result.change_from_baseline - 10.0) < 0.1  # ~10% 변화

    def test_result_to_dict(self, timestamps):
        """딕셔너리 변환"""
        from src.analysis.scenario_analysis import (
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        config = ScenarioConfig("Test", ScenarioType.NORMAL)
        predictions = np.random.rand(24) * 100

        result = ScenarioResult(
            scenario_config=config,
            predictions=predictions,
            timestamps=timestamps
        )
        result.calculate_stats()

        result_dict = result.to_dict()

        assert 'scenario' in result_dict
        assert 'predictions' in result_dict
        assert 'statistics' in result_dict


# ============================================================================
# ScenarioGenerator 테스트
# ============================================================================

class TestScenarioGenerator:
    """ScenarioGenerator 테스트"""

    def test_generator_creation(self):
        """생성기 생성"""
        from src.analysis.scenario_analysis import ScenarioGenerator

        generator = ScenarioGenerator()
        assert generator is not None

    def test_get_predefined_scenario(self):
        """사전 정의 시나리오 조회"""
        from src.analysis.scenario_analysis import ScenarioGenerator, ScenarioType

        generator = ScenarioGenerator()

        heatwave = generator.get_predefined('heatwave_mild')
        assert heatwave.scenario_type == ScenarioType.HEATWAVE
        assert heatwave.temperature_delta > 0

        coldwave = generator.get_predefined('coldwave_severe')
        assert coldwave.scenario_type == ScenarioType.COLDWAVE
        assert coldwave.temperature_delta < 0

    def test_get_unknown_scenario(self):
        """알 수 없는 시나리오"""
        from src.analysis.scenario_analysis import ScenarioGenerator

        generator = ScenarioGenerator()

        with pytest.raises(ValueError):
            generator.get_predefined('unknown_scenario')

    def test_create_custom_scenario(self):
        """커스텀 시나리오 생성"""
        from src.analysis.scenario_analysis import ScenarioGenerator, ScenarioType

        generator = ScenarioGenerator()

        custom = generator.create_custom_scenario(
            name="My Custom",
            temperature_delta=4.0,
            humidity_delta=-3.0,
            demand_factor=1.12
        )

        assert custom.name == "My Custom"
        assert custom.scenario_type == ScenarioType.CUSTOM
        assert custom.temperature_delta == 4.0

    def test_create_temperature_sweep(self):
        """온도 범위 시나리오"""
        from src.analysis.scenario_analysis import ScenarioGenerator

        generator = ScenarioGenerator()

        scenarios = generator.create_temperature_sweep(
            base_temp=25.0,
            delta_range=(-5, 5),
            n_scenarios=5
        )

        assert len(scenarios) == 5
        deltas = [s.temperature_delta for s in scenarios]
        assert min(deltas) == -5.0
        assert max(deltas) == 5.0

    def test_create_peak_demand_scenario(self):
        """피크 수요 시나리오"""
        from src.analysis.scenario_analysis import ScenarioGenerator, ScenarioType

        generator = ScenarioGenerator()

        peak = generator.create_peak_demand_scenario(percentile=99)

        assert peak.scenario_type == ScenarioType.EXTREME_PEAK
        assert peak.demand_factor > 1.0


# ============================================================================
# ScenarioRunner 테스트
# ============================================================================

class TestScenarioRunner:
    """ScenarioRunner 테스트"""

    def test_runner_creation(self, simple_model, feature_names):
        """실행기 생성"""
        from src.analysis.scenario_analysis import ScenarioRunner

        runner = ScenarioRunner(simple_model, feature_names=feature_names)

        assert runner.model is not None
        assert runner.temp_feature_idx == 0  # 'temperature' 자동 탐지
        assert runner.humidity_feature_idx == 1  # 'humidity' 자동 탐지

    def test_apply_scenario(self, simple_model, feature_names, sample_input):
        """시나리오 적용"""
        from src.analysis.scenario_analysis import (
            ScenarioRunner, ScenarioConfig, ScenarioType
        )

        runner = ScenarioRunner(simple_model, feature_names=feature_names)

        scenario = ScenarioConfig(
            name="Test",
            scenario_type=ScenarioType.HEATWAVE,
            temperature_delta=5.0
        )

        original_temp = sample_input[0, 0].item()
        modified = runner.apply_scenario(sample_input, scenario)
        new_temp = modified[0, 0].item()

        assert abs(new_temp - original_temp - 5.0) < 0.01

    def test_run_scenario(self, simple_model, feature_names, sample_input, timestamps):
        """시나리오 실행"""
        from src.analysis.scenario_analysis import (
            ScenarioRunner, ScenarioConfig, ScenarioType
        )

        runner = ScenarioRunner(simple_model, feature_names=feature_names)

        scenario = ScenarioConfig(
            name="Heatwave Test",
            scenario_type=ScenarioType.HEATWAVE,
            temperature_delta=5.0,
            demand_factor=1.1
        )

        result = runner.run_scenario(scenario, sample_input, timestamps)

        assert result.scenario_config.name == "Heatwave Test"
        assert len(result.predictions) > 0
        assert result.mean_demand > 0

    def test_run_multiple_scenarios(self, simple_model, feature_names, sample_input, timestamps):
        """여러 시나리오 실행"""
        from src.analysis.scenario_analysis import (
            ScenarioRunner, ScenarioConfig, ScenarioType
        )

        runner = ScenarioRunner(simple_model, feature_names=feature_names)

        scenarios = [
            ScenarioConfig("Heat", ScenarioType.HEATWAVE, temperature_delta=5.0),
            ScenarioConfig("Cold", ScenarioType.COLDWAVE, temperature_delta=-5.0),
        ]

        results = runner.run_multiple_scenarios(
            scenarios, sample_input, timestamps, include_baseline=True
        )

        # Baseline + 2 scenarios = 3 results
        assert len(results) == 3

        # 첫 번째는 baseline
        assert results[0].scenario_config.scenario_type == ScenarioType.BASELINE


# ============================================================================
# SensitivityAnalyzer 테스트
# ============================================================================

class TestSensitivityAnalyzer:
    """SensitivityAnalyzer 테스트"""

    def test_analyzer_creation(self, simple_model, feature_names):
        """분석기 생성"""
        from src.analysis.scenario_analysis import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(simple_model, feature_names=feature_names)
        assert analyzer is not None

    def test_analyze_single_feature(self, simple_model, feature_names, sample_input):
        """단일 피처 분석"""
        from src.analysis.scenario_analysis import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(simple_model, feature_names=feature_names)

        result = analyzer.analyze_single_feature(
            sample_input,
            feature_idx=0,  # temperature
            delta_range=(-5, 5),
            n_steps=5
        )

        assert 'feature_name' in result
        assert 'deltas' in result
        assert 'predictions' in result
        assert 'sensitivity' in result
        assert len(result['deltas']) == 5

    def test_analyze_all_features(self, simple_model, feature_names, sample_input):
        """모든 피처 분석"""
        from src.analysis.scenario_analysis import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(simple_model, feature_names=feature_names)

        result = analyzer.analyze_all_features(
            sample_input,
            delta_range=(-1, 1),
            n_steps=3
        )

        assert 'feature_sensitivities' in result
        assert 'ranked_features' in result
        assert len(result['feature_sensitivities']) == len(feature_names)

    def test_compute_elasticity(self, simple_model, feature_names, sample_input):
        """탄력성 계산"""
        from src.analysis.scenario_analysis import SensitivityAnalyzer

        analyzer = SensitivityAnalyzer(simple_model, feature_names=feature_names)

        elasticity = analyzer.compute_elasticity(
            sample_input,
            feature_idx=0,
            percent_change=1.0
        )

        assert isinstance(elasticity, float)


# ============================================================================
# ScenarioComparator 테스트
# ============================================================================

class TestScenarioComparator:
    """ScenarioComparator 테스트"""

    def test_comparator_creation(self, timestamps):
        """비교기 생성"""
        from src.analysis.scenario_analysis import (
            ScenarioComparator, ScenarioResult, ScenarioConfig, ScenarioType
        )

        results = [
            ScenarioResult(
                ScenarioConfig("Baseline", ScenarioType.BASELINE),
                np.ones(24) * 100,
                timestamps
            ),
            ScenarioResult(
                ScenarioConfig("Heat", ScenarioType.HEATWAVE, temperature_delta=5.0),
                np.ones(24) * 110,
                timestamps
            ),
        ]
        for r in results:
            r.calculate_stats()

        comparator = ScenarioComparator(results)

        assert comparator.baseline is not None
        assert len(comparator.results) == 2

    def test_get_comparison_table(self, timestamps):
        """비교 테이블"""
        from src.analysis.scenario_analysis import (
            ScenarioComparator, ScenarioResult, ScenarioConfig, ScenarioType
        )

        results = [
            ScenarioResult(
                ScenarioConfig("Baseline", ScenarioType.BASELINE),
                np.ones(24) * 100,
                timestamps
            ),
            ScenarioResult(
                ScenarioConfig("Heat", ScenarioType.HEATWAVE),
                np.ones(24) * 110,
                timestamps
            ),
        ]
        for r in results:
            r.calculate_stats()

        comparator = ScenarioComparator(results)
        table = comparator.get_comparison_table()

        assert len(table) == 2
        assert 'Scenario' in table.columns
        assert 'Mean Demand' in table.columns

    def test_get_summary(self, timestamps):
        """요약"""
        from src.analysis.scenario_analysis import (
            ScenarioComparator, ScenarioResult, ScenarioConfig, ScenarioType
        )

        results = [
            ScenarioResult(
                ScenarioConfig("Baseline", ScenarioType.BASELINE),
                np.ones(24) * 100,
                timestamps
            ),
            ScenarioResult(
                ScenarioConfig("Heat", ScenarioType.HEATWAVE),
                np.ones(24) * 120,
                timestamps
            ),
            ScenarioResult(
                ScenarioConfig("Cold", ScenarioType.COLDWAVE),
                np.ones(24) * 80,
                timestamps
            ),
        ]
        for r in results:
            r.calculate_stats()

        comparator = ScenarioComparator(results)
        summary = comparator.get_summary()

        assert 'n_scenarios' in summary
        assert summary['n_scenarios'] == 3
        assert 'demand_range' in summary


# ============================================================================
# ScenarioReport 테스트
# ============================================================================

class TestScenarioReport:
    """ScenarioReport 테스트"""

    def test_report_creation(self, timestamps):
        """리포트 생성"""
        from src.analysis.scenario_analysis import (
            ScenarioReport, ScenarioComparator,
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        results = [
            ScenarioResult(
                ScenarioConfig("Baseline", ScenarioType.BASELINE),
                np.ones(24) * 100,
                timestamps
            ),
        ]
        for r in results:
            r.calculate_stats()

        comparator = ScenarioComparator(results)
        report = ScenarioReport(comparator)

        assert report is not None

    def test_report_to_dict(self, timestamps):
        """딕셔너리 변환"""
        from src.analysis.scenario_analysis import (
            ScenarioReport, ScenarioComparator,
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        results = [
            ScenarioResult(
                ScenarioConfig("Baseline", ScenarioType.BASELINE),
                np.ones(24) * 100,
                timestamps
            ),
        ]
        for r in results:
            r.calculate_stats()

        comparator = ScenarioComparator(results)
        report = ScenarioReport(comparator)

        result_dict = report.to_dict()

        assert 'generated_at' in result_dict
        assert 'summary' in result_dict
        assert 'scenarios' in result_dict

    def test_report_to_markdown(self, timestamps):
        """마크다운 변환"""
        from src.analysis.scenario_analysis import (
            ScenarioReport, ScenarioComparator,
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        results = [
            ScenarioResult(
                ScenarioConfig("Baseline", ScenarioType.BASELINE),
                np.ones(24) * 100,
                timestamps
            ),
            ScenarioResult(
                ScenarioConfig("Heat", ScenarioType.HEATWAVE),
                np.ones(24) * 110,
                timestamps
            ),
        ]
        for r in results:
            r.baseline_predictions = results[0].predictions if r != results[0] else None
            r.calculate_stats()

        comparator = ScenarioComparator(results)
        report = ScenarioReport(comparator)

        markdown = report.to_markdown()

        assert "Scenario Analysis Report" in markdown
        assert "Summary" in markdown

    def test_report_save(self, timestamps):
        """파일 저장"""
        from src.analysis.scenario_analysis import (
            ScenarioReport, ScenarioComparator,
            ScenarioResult, ScenarioConfig, ScenarioType
        )

        results = [
            ScenarioResult(
                ScenarioConfig("Baseline", ScenarioType.BASELINE),
                np.ones(24) * 100,
                timestamps
            ),
        ]
        for r in results:
            r.calculate_stats()

        comparator = ScenarioComparator(results)
        report = ScenarioReport(comparator)

        with tempfile.TemporaryDirectory() as tmpdir:
            # JSON 저장
            json_path = Path(tmpdir) / "report.json"
            report.save(str(json_path), format='json')
            assert json_path.exists()

            # Markdown 저장
            md_path = Path(tmpdir) / "report.md"
            report.save(str(md_path), format='markdown')
            assert md_path.exists()


# ============================================================================
# 헬퍼 함수 테스트
# ============================================================================

class TestHelperFunctions:
    """헬퍼 함수 테스트"""

    def test_create_standard_scenarios(self):
        """표준 시나리오 생성"""
        from src.analysis.scenario_analysis import create_standard_scenarios

        scenarios = create_standard_scenarios()

        assert len(scenarios) >= 5
        names = [s.name for s in scenarios]
        # 최소한 normal과 heatwave 포함
        assert any('Normal' in n for n in names)
        assert any('Heatwave' in n for n in names)

    def test_run_what_if_analysis(self, simple_model, sample_input, feature_names, timestamps):
        """What-if 분석 함수"""
        from src.analysis.scenario_analysis import run_what_if_analysis

        report = run_what_if_analysis(
            model=simple_model,
            input_data=sample_input,
            feature_names=feature_names,
            timestamps=timestamps
        )

        assert report is not None
        summary = report.to_dict()['summary']
        assert summary['n_scenarios'] >= 5  # Standard scenarios + baseline


# ============================================================================
# 통합 테스트
# ============================================================================

class TestIntegration:
    """통합 테스트"""

    def test_full_scenario_pipeline(self, simple_model, feature_names, sample_input, timestamps):
        """전체 시나리오 파이프라인"""
        from src.analysis.scenario_analysis import (
            ScenarioGenerator, ScenarioRunner, ScenarioComparator, ScenarioReport
        )

        # 1. 시나리오 생성
        generator = ScenarioGenerator()
        scenarios = [
            generator.get_predefined('heatwave_mild'),
            generator.get_predefined('coldwave_mild'),
        ]

        # 2. 시나리오 실행
        runner = ScenarioRunner(simple_model, feature_names=feature_names)
        results = runner.run_multiple_scenarios(
            scenarios, sample_input, timestamps, include_baseline=True
        )

        # 3. 비교
        comparator = ScenarioComparator(results)
        table = comparator.get_comparison_table()

        # 4. 리포트
        report = ScenarioReport(comparator)

        assert len(results) == 3  # baseline + 2 scenarios
        assert len(table) == 3

        # JSON 출력 확인
        report_dict = report.to_dict()
        assert 'summary' in report_dict

    def test_sensitivity_and_scenario_combined(
        self, simple_model, feature_names, sample_input, timestamps
    ):
        """민감도 분석과 시나리오 결합"""
        from src.analysis.scenario_analysis import (
            SensitivityAnalyzer, ScenarioGenerator, ScenarioRunner
        )

        # 1. 민감도 분석
        analyzer = SensitivityAnalyzer(simple_model, feature_names=feature_names)
        sensitivity = analyzer.analyze_single_feature(sample_input, 0)

        # 2. 민감도 기반 시나리오 생성
        generator = ScenarioGenerator()
        if sensitivity['avg_sensitivity'] > 0.1:
            # 민감도가 높으면 여러 시나리오 테스트
            scenarios = generator.create_temperature_sweep(
                base_temp=25, delta_range=(-5, 5), n_scenarios=5
            )

            runner = ScenarioRunner(simple_model, feature_names=feature_names)
            results = runner.run_multiple_scenarios(scenarios, sample_input, timestamps)

            assert len(results) >= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
