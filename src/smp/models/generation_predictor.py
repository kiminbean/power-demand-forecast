"""
신재생에너지 발전량 예측기
==========================

태양광/풍력 발전량을 예측하여 민간 발전사업자의 입찰을 지원합니다.

주요 기능:
1. 태양광 발전량 예측 (일사량, 기온 기반)
2. 풍력 발전량 예측 (풍속, 풍향 기반)
3. 24시간 예측 (입찰용)
4. 설비용량 기반 스케일링
5. 불확실성 추정

발전량 계산 공식:
- 태양광: P = η × A × G × (1 - 0.005 × (T - 25))
  - η: 모듈 효율
  - A: 패널 면적
  - G: 일사량 (kW/m²)
  - T: 셀 온도

- 풍력: P = 0.5 × ρ × A × Cp × v³
  - ρ: 공기밀도 (kg/m³)
  - A: 로터 면적 (m²)
  - Cp: 출력계수 (Betz limit: 0.59)
  - v: 풍속 (m/s)

Author: Claude Code
Date: 2025-12
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime


def get_device() -> torch.device:
    """최적 디바이스 반환"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ============================================================
# 데이터 클래스
# ============================================================

@dataclass
class PlantConfig:
    """발전소 설정

    Attributes:
        plant_type: 발전소 유형 ('solar', 'wind')
        capacity_kw: 설비용량 (kW)
        efficiency: 효율 (0~1)
        location: 위치 (위도, 경도)
        name: 발전소 이름
    """
    plant_type: str  # 'solar' or 'wind'
    capacity_kw: float
    efficiency: float = 0.85  # 시스템 효율
    location: Tuple[float, float] = (33.5, 126.5)  # 제주 중심
    name: str = ""

    def __post_init__(self):
        if self.plant_type not in ('solar', 'wind'):
            raise ValueError(f"지원하지 않는 발전소 유형: {self.plant_type}")
        if self.capacity_kw <= 0:
            raise ValueError("설비용량은 양수여야 합니다")


@dataclass
class GenerationPrediction:
    """발전량 예측 결과

    Attributes:
        timestamp: 예측 시점
        hour: 예측 시간 (1-24)
        generation_kw: 예측 발전량 (kW)
        capacity_factor: 이용률 (0~1)
        uncertainty_low: 하위 예측 (10%)
        uncertainty_high: 상위 예측 (90%)
        weather_conditions: 기상 조건
    """
    timestamp: str
    hour: int
    generation_kw: float
    capacity_factor: float
    uncertainty_low: float = 0.0
    uncertainty_high: float = 0.0
    weather_conditions: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 물리 기반 발전량 계산기
# ============================================================

class SolarPowerCalculator:
    """태양광 발전량 계산기 (물리 기반)

    일사량과 기온을 기반으로 태양광 발전량을 계산합니다.

    Formula:
        P = Pmax × (G / G_stc) × (1 - γ × (Tc - 25))
        where:
            Pmax: 최대 출력 (설비용량)
            G: 현재 일사량 (W/m²)
            G_stc: 표준 조건 일사량 (1000 W/m²)
            γ: 온도 계수 (약 0.004~0.005 /°C)
            Tc: 셀 온도 (°C)
    """

    # 표준 테스트 조건 (STC)
    G_STC = 1000.0  # W/m²
    T_STC = 25.0    # °C

    # 온도 계수 (결정질 실리콘 기준)
    TEMP_COEFF = 0.004  # /°C

    # 셀 온도 = 기온 + (NOCT - 20) × G / 800
    NOCT = 45  # Nominal Operating Cell Temperature (°C)

    def __init__(self, capacity_kw: float, efficiency: float = 0.85):
        """
        Args:
            capacity_kw: 설비용량 (kW)
            efficiency: 시스템 효율 (인버터, 배선 손실 등)
        """
        self.capacity_kw = capacity_kw
        self.efficiency = efficiency

    def calculate(
        self,
        irradiance: float,
        temperature: float = 25.0
    ) -> float:
        """발전량 계산

        Args:
            irradiance: 일사량 (W/m²)
            temperature: 기온 (°C)

        Returns:
            발전량 (kW)
        """
        if irradiance <= 0:
            return 0.0

        # 셀 온도 계산
        cell_temp = temperature + (self.NOCT - 20) * irradiance / 800

        # 온도 보정 계수
        temp_factor = 1 - self.TEMP_COEFF * (cell_temp - self.T_STC)
        temp_factor = max(0.5, min(1.2, temp_factor))

        # 발전량 계산
        power = self.capacity_kw * (irradiance / self.G_STC) * temp_factor * self.efficiency

        return max(0, min(power, self.capacity_kw))

    def calculate_hourly(
        self,
        irradiance_hourly: List[float],
        temperature_hourly: List[float]
    ) -> List[float]:
        """시간별 발전량 계산

        Args:
            irradiance_hourly: 시간별 일사량 (W/m²)
            temperature_hourly: 시간별 기온 (°C)

        Returns:
            시간별 발전량 리스트 (kW)
        """
        return [
            self.calculate(irr, temp)
            for irr, temp in zip(irradiance_hourly, temperature_hourly)
        ]


class WindPowerCalculator:
    """풍력 발전량 계산기 (물리 기반)

    풍속을 기반으로 풍력 발전량을 계산합니다.

    Formula:
        P = 0.5 × ρ × A × Cp × v³ × η
        where:
            ρ: 공기밀도 (kg/m³, ~1.225 at sea level)
            A: 로터 면적 (m²)
            Cp: 출력계수 (이론적 최대: 0.593, 실제: 0.35~0.45)
            v: 풍속 (m/s)
            η: 시스템 효율
    """

    # 공기밀도 (해수면 기준)
    AIR_DENSITY = 1.225  # kg/m³

    # 출력계수 (현대 풍력터빈 기준)
    CP_TYPICAL = 0.40

    # 컷인/컷아웃 풍속
    CUT_IN_SPEED = 3.0   # m/s
    RATED_SPEED = 12.0   # m/s
    CUT_OUT_SPEED = 25.0 # m/s

    def __init__(
        self,
        capacity_kw: float,
        rotor_diameter: float = None,
        efficiency: float = 0.85
    ):
        """
        Args:
            capacity_kw: 설비용량 (kW)
            rotor_diameter: 로터 직경 (m), None이면 용량에서 추정
            efficiency: 시스템 효율
        """
        self.capacity_kw = capacity_kw
        self.efficiency = efficiency

        # 로터 직경 추정 (용량에서)
        if rotor_diameter is None:
            # 경험식: 1MW ≈ 50-60m 로터
            rotor_diameter = np.sqrt(capacity_kw / 1000) * 55

        self.rotor_diameter = rotor_diameter
        self.rotor_area = np.pi * (rotor_diameter / 2) ** 2

    def calculate(self, wind_speed: float) -> float:
        """발전량 계산

        Args:
            wind_speed: 풍속 (m/s)

        Returns:
            발전량 (kW)
        """
        # 컷인 이하
        if wind_speed < self.CUT_IN_SPEED:
            return 0.0

        # 컷아웃 이상 (안전을 위해 정지)
        if wind_speed >= self.CUT_OUT_SPEED:
            return 0.0

        # 정격 풍속 이상 (정격 출력 유지)
        if wind_speed >= self.RATED_SPEED:
            return self.capacity_kw * self.efficiency

        # Cubic power curve (컷인 ~ 정격 사이)
        # 선형 보간으로 단순화
        power_ratio = ((wind_speed - self.CUT_IN_SPEED) /
                      (self.RATED_SPEED - self.CUT_IN_SPEED)) ** 3

        power = self.capacity_kw * power_ratio * self.efficiency

        return max(0, min(power, self.capacity_kw))

    def calculate_hourly(self, wind_speed_hourly: List[float]) -> List[float]:
        """시간별 발전량 계산

        Args:
            wind_speed_hourly: 시간별 풍속 (m/s)

        Returns:
            시간별 발전량 리스트 (kW)
        """
        return [self.calculate(ws) for ws in wind_speed_hourly]


# ============================================================
# 딥러닝 기반 발전량 예측기
# ============================================================

class GenerationLSTM(nn.Module):
    """발전량 예측 LSTM 모델

    기상 데이터 시퀀스로 발전량을 예측합니다.

    Args:
        input_size: 입력 피처 수 (기상 변수 등)
        hidden_size: LSTM hidden size
        num_layers: LSTM 레이어 수
        prediction_hours: 예측 시간 수
        dropout: Dropout 비율
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        prediction_hours: int = 24,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prediction_hours = prediction_hours

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, prediction_hours)
        )

        # 출력 범위 제한 (0~1 정규화 발전량)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)

        Returns:
            output: (batch, prediction_hours) 정규화된 발전량 (0~1)
        """
        lstm_out, _ = self.lstm(x)
        context = lstm_out[:, -1, :]  # Last hidden state
        output = self.decoder(context)
        return self.sigmoid(output)  # 0~1 범위로 제한


# ============================================================
# 통합 발전량 예측기
# ============================================================

class GenerationPredictor:
    """통합 발전량 예측기

    물리 모델과 딥러닝 모델을 결합하여 발전량을 예측합니다.

    Example:
        >>> config = PlantConfig('solar', capacity_kw=1000)
        >>> predictor = GenerationPredictor(config)
        >>> predictions = predictor.predict(
        ...     irradiance=[500, 700, 900, 800, 600, 400, 200, 0] * 3,
        ...     temperature=[20, 22, 25, 28, 30, 28, 25, 22] * 3
        ... )
    """

    def __init__(
        self,
        config: PlantConfig,
        use_ml_model: bool = False,
        ml_model: Optional[nn.Module] = None
    ):
        """
        Args:
            config: 발전소 설정
            use_ml_model: ML 모델 사용 여부
            ml_model: 사전 학습된 ML 모델
        """
        self.config = config
        self.use_ml_model = use_ml_model
        self.ml_model = ml_model

        # 물리 모델 초기화
        if config.plant_type == 'solar':
            self.physics_model = SolarPowerCalculator(
                capacity_kw=config.capacity_kw,
                efficiency=config.efficiency
            )
        else:  # wind
            self.physics_model = WindPowerCalculator(
                capacity_kw=config.capacity_kw,
                efficiency=config.efficiency
            )

    def predict(
        self,
        irradiance: Optional[List[float]] = None,
        wind_speed: Optional[List[float]] = None,
        temperature: Optional[List[float]] = None,
        hours: int = 24,
        add_uncertainty: bool = True
    ) -> List[GenerationPrediction]:
        """발전량 예측

        Args:
            irradiance: 일사량 리스트 (W/m², 태양광용)
            wind_speed: 풍속 리스트 (m/s, 풍력용)
            temperature: 기온 리스트 (°C)
            hours: 예측 시간 수
            add_uncertainty: 불확실성 추가 여부

        Returns:
            GenerationPrediction 리스트
        """
        now = datetime.now()
        predictions = []

        if self.config.plant_type == 'solar':
            if irradiance is None:
                raise ValueError("태양광 예측에는 일사량 데이터가 필요합니다")
            if temperature is None:
                temperature = [25.0] * len(irradiance)

            # 물리 모델로 예측
            generation_values = self.physics_model.calculate_hourly(
                irradiance[:hours],
                temperature[:hours]
            )

        else:  # wind
            if wind_speed is None:
                raise ValueError("풍력 예측에는 풍속 데이터가 필요합니다")

            generation_values = self.physics_model.calculate_hourly(
                wind_speed[:hours]
            )

        # 예측 결과 생성
        for h, gen_kw in enumerate(generation_values):
            capacity_factor = gen_kw / self.config.capacity_kw

            # 불확실성 추정 (단순 ±20%)
            if add_uncertainty:
                uncertainty_low = gen_kw * 0.8
                uncertainty_high = gen_kw * 1.2
            else:
                uncertainty_low = gen_kw
                uncertainty_high = gen_kw

            # 기상 조건
            weather = {}
            if self.config.plant_type == 'solar':
                weather['irradiance'] = irradiance[h] if h < len(irradiance) else 0
                weather['temperature'] = temperature[h] if h < len(temperature) else 25
            else:
                weather['wind_speed'] = wind_speed[h] if h < len(wind_speed) else 0

            pred = GenerationPrediction(
                timestamp=now.strftime("%Y-%m-%d %H:%M"),
                hour=h + 1,
                generation_kw=gen_kw,
                capacity_factor=capacity_factor,
                uncertainty_low=uncertainty_low,
                uncertainty_high=uncertainty_high,
                weather_conditions=weather
            )
            predictions.append(pred)

        return predictions

    def estimate_revenue(
        self,
        predictions: List[GenerationPrediction],
        smp_prices: List[float]
    ) -> Dict[str, float]:
        """예상 수익 계산

        Args:
            predictions: 발전량 예측 리스트
            smp_prices: SMP 가격 리스트 (원/kWh)

        Returns:
            수익 정보 딕셔너리
        """
        if len(predictions) != len(smp_prices):
            min_len = min(len(predictions), len(smp_prices))
            predictions = predictions[:min_len]
            smp_prices = smp_prices[:min_len]

        # 시간별 수익 계산
        hourly_revenue = []
        total_generation = 0.0
        total_revenue = 0.0

        for pred, smp in zip(predictions, smp_prices):
            generation_kwh = pred.generation_kw  # 1시간 = kW → kWh
            revenue = generation_kwh * smp
            hourly_revenue.append(revenue)
            total_generation += generation_kwh
            total_revenue += revenue

        return {
            'total_generation_kwh': total_generation,
            'total_revenue_krw': total_revenue,
            'average_smp': sum(smp_prices) / len(smp_prices),
            'average_capacity_factor': sum(p.capacity_factor for p in predictions) / len(predictions),
            'hourly_revenue': hourly_revenue,
        }


def create_generation_predictor(
    plant_type: str,
    capacity_kw: float,
    efficiency: float = 0.85,
    **kwargs
) -> GenerationPredictor:
    """발전량 예측기 팩토리 함수

    Args:
        plant_type: 발전소 유형 ('solar', 'wind')
        capacity_kw: 설비용량 (kW)
        efficiency: 시스템 효율

    Returns:
        GenerationPredictor 인스턴스
    """
    config = PlantConfig(
        plant_type=plant_type,
        capacity_kw=capacity_kw,
        efficiency=efficiency,
        **kwargs
    )
    return GenerationPredictor(config)


if __name__ == "__main__":
    # 테스트
    print("발전량 예측기 테스트")
    print("=" * 60)

    # 태양광 예측 테스트
    print("\n1. 태양광 발전량 예측")
    solar_config = PlantConfig('solar', capacity_kw=1000)
    solar_predictor = GenerationPredictor(solar_config)

    # 샘플 기상 데이터 (24시간)
    irradiance = [0, 0, 0, 0, 50, 150, 400, 600,
                  750, 850, 900, 920, 900, 850, 750, 600,
                  400, 150, 50, 0, 0, 0, 0, 0]  # W/m²
    temperature = [18, 17, 16, 16, 17, 18, 20, 22,
                   24, 26, 28, 29, 30, 30, 29, 28,
                   26, 24, 22, 21, 20, 19, 19, 18]  # °C

    solar_predictions = solar_predictor.predict(
        irradiance=irradiance,
        temperature=temperature,
        hours=24
    )

    print(f"설비용량: {solar_config.capacity_kw} kW")
    print(f"예측 시간: 24시간")
    print("\n시간별 발전량 (kW):")
    for i, pred in enumerate(solar_predictions):
        print(f"  {i+1:2d}시: {pred.generation_kw:7.2f} kW "
              f"(이용률: {pred.capacity_factor*100:5.1f}%)")

    total_gen = sum(p.generation_kw for p in solar_predictions)
    avg_cf = sum(p.capacity_factor for p in solar_predictions) / 24
    print(f"\n총 발전량: {total_gen:.2f} kWh")
    print(f"평균 이용률: {avg_cf*100:.1f}%")

    # 풍력 예측 테스트
    print("\n" + "=" * 60)
    print("\n2. 풍력 발전량 예측")
    wind_config = PlantConfig('wind', capacity_kw=2000)
    wind_predictor = GenerationPredictor(wind_config)

    # 샘플 풍속 데이터
    wind_speed = [5, 6, 7, 8, 8, 7, 6, 5,
                  6, 8, 10, 12, 14, 15, 14, 12,
                  10, 8, 7, 6, 5, 4, 4, 5]  # m/s

    wind_predictions = wind_predictor.predict(
        wind_speed=wind_speed,
        hours=24
    )

    print(f"설비용량: {wind_config.capacity_kw} kW")
    print("\n시간별 발전량 (kW):")
    for i, pred in enumerate(wind_predictions):
        print(f"  {i+1:2d}시: {pred.generation_kw:7.2f} kW "
              f"(풍속: {wind_speed[i]:5.1f} m/s)")

    total_wind = sum(p.generation_kw for p in wind_predictions)
    print(f"\n총 발전량: {total_wind:.2f} kWh")

    # 수익 계산 테스트
    print("\n" + "=" * 60)
    print("\n3. 수익 계산")
    smp_prices = [100] * 6 + [120] * 6 + [150] * 6 + [110] * 6  # 원/kWh

    revenue = solar_predictor.estimate_revenue(solar_predictions, smp_prices)
    print(f"총 발전량: {revenue['total_generation_kwh']:.2f} kWh")
    print(f"평균 SMP: {revenue['average_smp']:.2f} 원/kWh")
    print(f"총 수익: {revenue['total_revenue_krw']:,.0f} 원")

    print("\n모든 테스트 완료!")
