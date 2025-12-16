"""
시나리오 분석 모듈 (Task 24)
=============================

What-if 분석을 위한 시나리오 생성 및 평가 도구

주요 기능:
1. 기상 시나리오 생성
2. 수요 변화 시뮬레이션
3. 민감도 분석
4. 시나리오 비교

Author: Claude Code
Date: 2025-12
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# 시나리오 타입 정의
# ============================================================================

class ScenarioType(Enum):
    """시나리오 타입"""
    BASELINE = "baseline"          # 기준 시나리오
    HEATWAVE = "heatwave"          # 폭염 시나리오
    COLDWAVE = "coldwave"          # 한파 시나리오
    NORMAL = "normal"              # 평년 시나리오
    EXTREME_PEAK = "extreme_peak"  # 극단적 피크
    CUSTOM = "custom"              # 사용자 정의


@dataclass
class ScenarioConfig:
    """시나리오 설정"""
    name: str
    scenario_type: ScenarioType
    description: str = ""

    # 기상 변수 조정
    temperature_delta: float = 0.0      # 온도 변화량 (°C)
    humidity_delta: float = 0.0         # 습도 변화량 (%)
    wind_speed_delta: float = 0.0       # 풍속 변화량 (m/s)
    solar_radiation_factor: float = 1.0 # 일사량 계수

    # 수요 조정
    demand_factor: float = 1.0          # 수요 증감 계수
    peak_shift_hours: int = 0           # 피크 시프트 (시간)

    # 기간
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # 추가 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'scenario_type': self.scenario_type.value,
            'description': self.description,
            'temperature_delta': self.temperature_delta,
            'humidity_delta': self.humidity_delta,
            'wind_speed_delta': self.wind_speed_delta,
            'solar_radiation_factor': self.solar_radiation_factor,
            'demand_factor': self.demand_factor,
            'peak_shift_hours': self.peak_shift_hours,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'metadata': self.metadata
        }


@dataclass
class ScenarioResult:
    """시나리오 분석 결과"""
    scenario_config: ScenarioConfig
    predictions: np.ndarray
    timestamps: List[datetime]
    baseline_predictions: Optional[np.ndarray] = None

    # 통계
    mean_demand: float = 0.0
    max_demand: float = 0.0
    min_demand: float = 0.0
    peak_hour: int = 0
    total_energy: float = 0.0

    # 기준 대비 변화
    change_from_baseline: float = 0.0  # %

    def calculate_stats(self):
        """통계 계산"""
        self.mean_demand = float(np.mean(self.predictions))
        self.max_demand = float(np.max(self.predictions))
        self.min_demand = float(np.min(self.predictions))
        self.peak_hour = int(np.argmax(self.predictions) % 24)
        self.total_energy = float(np.sum(self.predictions))

        if self.baseline_predictions is not None and len(self.baseline_predictions) > 0:
            baseline_mean = np.mean(self.baseline_predictions)
            if baseline_mean != 0:
                self.change_from_baseline = ((self.mean_demand - baseline_mean) / baseline_mean) * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            'scenario': self.scenario_config.to_dict(),
            'predictions': self.predictions.tolist(),
            'timestamps': [t.isoformat() for t in self.timestamps],
            'statistics': {
                'mean_demand': self.mean_demand,
                'max_demand': self.max_demand,
                'min_demand': self.min_demand,
                'peak_hour': self.peak_hour,
                'total_energy': self.total_energy,
                'change_from_baseline': self.change_from_baseline
            }
        }


# ============================================================================
# 시나리오 생성기
# ============================================================================

class ScenarioGenerator:
    """
    시나리오 생성기

    다양한 기상 및 수요 시나리오를 생성합니다.

    Example:
        >>> generator = ScenarioGenerator()
        >>> heatwave = generator.create_heatwave_scenario(temperature_increase=5)
    """

    # 사전 정의 시나리오
    PREDEFINED_SCENARIOS = {
        'heatwave_mild': ScenarioConfig(
            name="Mild Heatwave",
            scenario_type=ScenarioType.HEATWAVE,
            description="약한 폭염 (기온 +3°C)",
            temperature_delta=3.0,
            humidity_delta=-5.0,
            demand_factor=1.08
        ),
        'heatwave_severe': ScenarioConfig(
            name="Severe Heatwave",
            scenario_type=ScenarioType.HEATWAVE,
            description="심한 폭염 (기온 +7°C)",
            temperature_delta=7.0,
            humidity_delta=-10.0,
            demand_factor=1.20
        ),
        'coldwave_mild': ScenarioConfig(
            name="Mild Cold Wave",
            scenario_type=ScenarioType.COLDWAVE,
            description="약한 한파 (기온 -5°C)",
            temperature_delta=-5.0,
            humidity_delta=5.0,
            demand_factor=1.10
        ),
        'coldwave_severe': ScenarioConfig(
            name="Severe Cold Wave",
            scenario_type=ScenarioType.COLDWAVE,
            description="심한 한파 (기온 -10°C)",
            temperature_delta=-10.0,
            humidity_delta=10.0,
            demand_factor=1.25
        ),
        'normal': ScenarioConfig(
            name="Normal",
            scenario_type=ScenarioType.NORMAL,
            description="평년 기상조건",
            temperature_delta=0.0,
            humidity_delta=0.0,
            demand_factor=1.0
        )
    }

    def __init__(self):
        self.scenarios: Dict[str, ScenarioConfig] = {}

    def get_predefined(self, name: str) -> ScenarioConfig:
        """사전 정의 시나리오 조회"""
        if name not in self.PREDEFINED_SCENARIOS:
            available = list(self.PREDEFINED_SCENARIOS.keys())
            raise ValueError(f"Unknown scenario: {name}. Available: {available}")
        return self.PREDEFINED_SCENARIOS[name]

    def create_custom_scenario(
        self,
        name: str,
        temperature_delta: float = 0.0,
        humidity_delta: float = 0.0,
        demand_factor: float = 1.0,
        **kwargs
    ) -> ScenarioConfig:
        """커스텀 시나리오 생성"""
        config = ScenarioConfig(
            name=name,
            scenario_type=ScenarioType.CUSTOM,
            description=kwargs.get('description', f"Custom scenario: {name}"),
            temperature_delta=temperature_delta,
            humidity_delta=humidity_delta,
            demand_factor=demand_factor,
            wind_speed_delta=kwargs.get('wind_speed_delta', 0.0),
            solar_radiation_factor=kwargs.get('solar_radiation_factor', 1.0),
            peak_shift_hours=kwargs.get('peak_shift_hours', 0),
            metadata=kwargs.get('metadata', {})
        )

        self.scenarios[name] = config
        return config

    def create_temperature_sweep(
        self,
        base_temp: float,
        delta_range: Tuple[float, float],
        n_scenarios: int
    ) -> List[ScenarioConfig]:
        """온도 변화 범위 시나리오 생성"""
        deltas = np.linspace(delta_range[0], delta_range[1], n_scenarios)
        scenarios = []

        for i, delta in enumerate(deltas):
            config = ScenarioConfig(
                name=f"Temp_Sweep_{i+1}",
                scenario_type=ScenarioType.CUSTOM,
                description=f"Temperature delta: {delta:+.1f}°C",
                temperature_delta=float(delta),
                metadata={'base_temp': base_temp, 'sweep_index': i}
            )
            scenarios.append(config)

        return scenarios

    def create_peak_demand_scenario(
        self,
        percentile: float = 99
    ) -> ScenarioConfig:
        """극단적 피크 수요 시나리오"""
        return ScenarioConfig(
            name=f"Peak_P{percentile}",
            scenario_type=ScenarioType.EXTREME_PEAK,
            description=f"{percentile}th percentile peak demand",
            temperature_delta=5.0,
            humidity_delta=-5.0,
            demand_factor=1.30,
            metadata={'percentile': percentile}
        )


# ============================================================================
# 시나리오 실행기
# ============================================================================

class ScenarioRunner:
    """
    시나리오 실행기

    모델에 시나리오를 적용하고 예측을 수행합니다.

    Args:
        model: 예측 모델 (PyTorch)
        feature_processor: 피처 전처리기

    Example:
        >>> runner = ScenarioRunner(model, processor)
        >>> result = runner.run_scenario(scenario, input_data)
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str] = None,
        temp_feature_idx: int = None,
        humidity_feature_idx: int = None
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.temp_feature_idx = temp_feature_idx
        self.humidity_feature_idx = humidity_feature_idx

        # 피처 이름에서 인덱스 자동 탐지
        self._auto_detect_feature_indices()

    def _auto_detect_feature_indices(self):
        """피처 인덱스 자동 탐지"""
        if self.temp_feature_idx is None:
            for i, name in enumerate(self.feature_names):
                if 'temp' in name.lower() or 'temperature' in name.lower():
                    self.temp_feature_idx = i
                    break

        if self.humidity_feature_idx is None:
            for i, name in enumerate(self.feature_names):
                if 'humid' in name.lower() or 'rh' in name.lower():
                    self.humidity_feature_idx = i
                    break

    def apply_scenario(
        self,
        data: torch.Tensor,
        scenario: ScenarioConfig
    ) -> torch.Tensor:
        """시나리오 적용"""
        modified_data = data.clone()

        # 온도 조정
        if scenario.temperature_delta != 0 and self.temp_feature_idx is not None:
            if modified_data.dim() == 3:
                modified_data[:, :, self.temp_feature_idx] += scenario.temperature_delta
            else:
                modified_data[:, self.temp_feature_idx] += scenario.temperature_delta

        # 습도 조정
        if scenario.humidity_delta != 0 and self.humidity_feature_idx is not None:
            if modified_data.dim() == 3:
                modified_data[:, :, self.humidity_feature_idx] += scenario.humidity_delta
            else:
                modified_data[:, self.humidity_feature_idx] += scenario.humidity_delta

        return modified_data

    def run_scenario(
        self,
        scenario: ScenarioConfig,
        input_data: torch.Tensor,
        timestamps: List[datetime] = None
    ) -> ScenarioResult:
        """시나리오 실행"""
        device = next(self.model.parameters()).device
        input_data = input_data.to(device)

        # 시나리오 적용
        modified_data = self.apply_scenario(input_data, scenario)

        # 예측
        self.model.eval()
        with torch.no_grad():
            output = self.model(modified_data)
            if isinstance(output, tuple):
                output = output[0]

            predictions = output.cpu().numpy()

            # 수요 계수 적용
            predictions = predictions * scenario.demand_factor

        # 결과 정리
        if predictions.ndim > 1:
            predictions = predictions.flatten()

        if timestamps is None:
            timestamps = [datetime.now() + timedelta(hours=i) for i in range(len(predictions))]

        result = ScenarioResult(
            scenario_config=scenario,
            predictions=predictions,
            timestamps=timestamps
        )
        result.calculate_stats()

        return result

    def run_multiple_scenarios(
        self,
        scenarios: List[ScenarioConfig],
        input_data: torch.Tensor,
        timestamps: List[datetime] = None,
        include_baseline: bool = True
    ) -> List[ScenarioResult]:
        """여러 시나리오 실행"""
        results = []
        baseline_predictions = None

        # 기준 시나리오 먼저 실행
        if include_baseline:
            baseline_scenario = ScenarioConfig(
                name="Baseline",
                scenario_type=ScenarioType.BASELINE,
                description="No modifications"
            )
            baseline_result = self.run_scenario(baseline_scenario, input_data, timestamps)
            baseline_predictions = baseline_result.predictions
            results.append(baseline_result)

        # 각 시나리오 실행
        for scenario in scenarios:
            result = self.run_scenario(scenario, input_data, timestamps)
            result.baseline_predictions = baseline_predictions
            result.calculate_stats()
            results.append(result)

        return results


# ============================================================================
# 민감도 분석
# ============================================================================

class SensitivityAnalyzer:
    """
    민감도 분석기

    피처 변화에 따른 예측 민감도를 분석합니다.

    Example:
        >>> analyzer = SensitivityAnalyzer(model)
        >>> results = analyzer.analyze_temperature_sensitivity(data, range=(-5, 5))
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: List[str] = None
    ):
        self.model = model
        self.feature_names = feature_names or []
        self.model.eval()

    def analyze_single_feature(
        self,
        input_data: torch.Tensor,
        feature_idx: int,
        delta_range: Tuple[float, float] = (-5, 5),
        n_steps: int = 11
    ) -> Dict[str, Any]:
        """단일 피처 민감도 분석"""
        device = next(self.model.parameters()).device
        input_data = input_data.to(device)

        deltas = np.linspace(delta_range[0], delta_range[1], n_steps)
        predictions = []

        with torch.no_grad():
            # 기준 예측
            base_output = self.model(input_data)
            if isinstance(base_output, tuple):
                base_output = base_output[0]
            base_pred = base_output.mean().item()

            # 각 델타에 대해 예측
            for delta in deltas:
                modified = input_data.clone()
                if modified.dim() == 3:
                    modified[:, :, feature_idx] += delta
                else:
                    modified[:, feature_idx] += delta

                output = self.model(modified)
                if isinstance(output, tuple):
                    output = output[0]
                predictions.append(output.mean().item())

        predictions = np.array(predictions)

        # 민감도 계산 (기울기)
        sensitivity = np.gradient(predictions, deltas)
        avg_sensitivity = np.mean(np.abs(sensitivity))

        return {
            'feature_name': self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f"Feature_{feature_idx}",
            'deltas': deltas.tolist(),
            'predictions': predictions.tolist(),
            'base_prediction': base_pred,
            'sensitivity': sensitivity.tolist(),
            'avg_sensitivity': float(avg_sensitivity),
            'max_change': float(np.max(predictions) - np.min(predictions))
        }

    def analyze_all_features(
        self,
        input_data: torch.Tensor,
        delta_range: Tuple[float, float] = (-1, 1),
        n_steps: int = 5
    ) -> Dict[str, Any]:
        """모든 피처 민감도 분석"""
        n_features = input_data.shape[-1]
        results = []

        for i in range(n_features):
            result = self.analyze_single_feature(
                input_data, i, delta_range, n_steps
            )
            results.append(result)

        # 민감도 순위
        sorted_by_sensitivity = sorted(
            results,
            key=lambda x: x['avg_sensitivity'],
            reverse=True
        )

        return {
            'feature_sensitivities': results,
            'ranked_features': [r['feature_name'] for r in sorted_by_sensitivity],
            'most_sensitive': sorted_by_sensitivity[0]['feature_name'] if sorted_by_sensitivity else None,
            'least_sensitive': sorted_by_sensitivity[-1]['feature_name'] if sorted_by_sensitivity else None
        }

    def compute_elasticity(
        self,
        input_data: torch.Tensor,
        feature_idx: int,
        percent_change: float = 1.0
    ) -> float:
        """
        탄력성 계산

        피처 1% 변화에 대한 예측값 % 변화
        """
        device = next(self.model.parameters()).device
        input_data = input_data.to(device)

        with torch.no_grad():
            # 기준 예측
            base_output = self.model(input_data)
            if isinstance(base_output, tuple):
                base_output = base_output[0]
            base_pred = base_output.mean().item()

            # 변화 적용
            modified = input_data.clone()
            if modified.dim() == 3:
                original_value = modified[:, :, feature_idx].mean().item()
                modified[:, :, feature_idx] *= (1 + percent_change / 100)
            else:
                original_value = modified[:, feature_idx].mean().item()
                modified[:, feature_idx] *= (1 + percent_change / 100)

            # 변화 후 예측
            new_output = self.model(modified)
            if isinstance(new_output, tuple):
                new_output = new_output[0]
            new_pred = new_output.mean().item()

        # 탄력성 = (ΔY/Y) / (ΔX/X)
        if base_pred != 0 and original_value != 0:
            pct_change_y = (new_pred - base_pred) / base_pred * 100
            elasticity = pct_change_y / percent_change
        else:
            elasticity = 0.0

        return float(elasticity)


# ============================================================================
# 시나리오 비교
# ============================================================================

class ScenarioComparator:
    """
    시나리오 비교기

    여러 시나리오 결과를 비교 분석합니다.
    """

    def __init__(self, results: List[ScenarioResult]):
        self.results = results
        self.baseline = None

        # 기준선 찾기
        for r in results:
            if r.scenario_config.scenario_type == ScenarioType.BASELINE:
                self.baseline = r
                break

    def get_comparison_table(self) -> pd.DataFrame:
        """비교 테이블 생성"""
        data = []

        for result in self.results:
            row = {
                'Scenario': result.scenario_config.name,
                'Type': result.scenario_config.scenario_type.value,
                'Mean Demand': result.mean_demand,
                'Max Demand': result.max_demand,
                'Min Demand': result.min_demand,
                'Peak Hour': result.peak_hour,
                'Total Energy': result.total_energy,
                'Change (%)': result.change_from_baseline
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_extreme_scenario(self, metric: str = 'max_demand', highest: bool = True) -> ScenarioResult:
        """극단 시나리오 조회"""
        if not self.results:
            return None

        sorted_results = sorted(
            self.results,
            key=lambda x: getattr(x, metric),
            reverse=highest
        )

        return sorted_results[0]

    def get_summary(self) -> Dict[str, Any]:
        """비교 요약"""
        if not self.results:
            return {}

        means = [r.mean_demand for r in self.results]
        maxs = [r.max_demand for r in self.results]

        return {
            'n_scenarios': len(self.results),
            'demand_range': {
                'min': float(min(means)),
                'max': float(max(means)),
                'spread': float(max(means) - min(means))
            },
            'peak_range': {
                'min': float(min(maxs)),
                'max': float(max(maxs))
            },
            'highest_demand_scenario': self.get_extreme_scenario('mean_demand', True).scenario_config.name,
            'lowest_demand_scenario': self.get_extreme_scenario('mean_demand', False).scenario_config.name
        }


# ============================================================================
# 리포트 생성
# ============================================================================

class ScenarioReport:
    """
    시나리오 분석 리포트

    시나리오 분석 결과를 리포트로 생성합니다.
    """

    def __init__(self, comparator: ScenarioComparator):
        self.comparator = comparator
        self.generated_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            'generated_at': self.generated_at.isoformat(),
            'summary': self.comparator.get_summary(),
            'scenarios': [r.to_dict() for r in self.comparator.results]
        }

    def to_markdown(self) -> str:
        """마크다운 리포트"""
        summary = self.comparator.get_summary()
        table = self.comparator.get_comparison_table()

        report = f"""# Scenario Analysis Report

Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Scenarios Analyzed**: {summary.get('n_scenarios', 0)}
- **Demand Range**: {summary.get('demand_range', {}).get('min', 0):.2f} - {summary.get('demand_range', {}).get('max', 0):.2f} MW
- **Highest Demand Scenario**: {summary.get('highest_demand_scenario', 'N/A')}
- **Lowest Demand Scenario**: {summary.get('lowest_demand_scenario', 'N/A')}

## Scenario Comparison

{table.to_markdown(index=False)}

## Key Findings

"""

        # 주요 발견 사항
        if self.comparator.baseline:
            report += "### Impact Analysis (vs Baseline)\n\n"
            for result in self.comparator.results:
                if result != self.comparator.baseline:
                    report += f"- **{result.scenario_config.name}**: {result.change_from_baseline:+.1f}% change\n"

        return report

    def save(self, path: str, format: str = 'json'):
        """파일 저장"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == 'markdown':
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.to_markdown())
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Report saved to {path}")


# ============================================================================
# 헬퍼 함수
# ============================================================================

def create_standard_scenarios() -> List[ScenarioConfig]:
    """표준 시나리오 세트 생성"""
    generator = ScenarioGenerator()

    return [
        generator.get_predefined('normal'),
        generator.get_predefined('heatwave_mild'),
        generator.get_predefined('heatwave_severe'),
        generator.get_predefined('coldwave_mild'),
        generator.get_predefined('coldwave_severe'),
    ]


def run_what_if_analysis(
    model: nn.Module,
    input_data: torch.Tensor,
    scenarios: List[ScenarioConfig] = None,
    feature_names: List[str] = None,
    timestamps: List[datetime] = None
) -> ScenarioReport:
    """
    간편한 What-if 분석 함수

    Args:
        model: 예측 모델
        input_data: 입력 데이터
        scenarios: 시나리오 리스트 (None이면 표준 시나리오 사용)
        feature_names: 피처 이름 리스트
        timestamps: 타임스탬프 리스트

    Returns:
        ScenarioReport
    """
    if scenarios is None:
        scenarios = create_standard_scenarios()

    runner = ScenarioRunner(model, feature_names=feature_names)
    results = runner.run_multiple_scenarios(
        scenarios, input_data, timestamps, include_baseline=True
    )

    comparator = ScenarioComparator(results)
    return ScenarioReport(comparator)
