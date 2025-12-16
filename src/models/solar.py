"""
태양광 발전량 추정 모델 (Task 8)
================================
기상 조건을 기반으로 태양광 발전량을 예측합니다.

주요 컴포넌트:
- SolarPositionCalculator: 태양 위치 계산
- ClearSkyModel: 맑은 날 일사량 추정
- PVSystemModel: PV 시스템 모델링
- SolarPowerPredictor: ML 기반 발전량 예측
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class PVModuleType(Enum):
    """PV 모듈 타입"""
    MONOCRYSTALLINE = 'monocrystalline'
    POLYCRYSTALLINE = 'polycrystalline'
    THIN_FILM = 'thin_film'
    BIFACIAL = 'bifacial'


class MountingType(Enum):
    """설치 타입"""
    FIXED = 'fixed'
    SINGLE_AXIS_TRACKER = 'single_axis'
    DUAL_AXIS_TRACKER = 'dual_axis'


@dataclass
class Location:
    """위치 정보"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    timezone: str = 'Asia/Seoul'
    name: str = ''


@dataclass
class PVSystemConfig:
    """PV 시스템 설정"""
    capacity_kw: float
    module_type: PVModuleType = PVModuleType.MONOCRYSTALLINE
    mounting_type: MountingType = MountingType.FIXED
    tilt_angle: float = 30.0  # 경사각 (도)
    azimuth_angle: float = 180.0  # 방위각 (남향=180도)
    efficiency: float = 0.18  # 모듈 효율
    system_losses: float = 0.14  # 시스템 손실
    temperature_coefficient: float = -0.004  # 온도 계수 (%/°C)
    reference_temperature: float = 25.0  # 기준 온도 (°C)


@dataclass
class SolarPosition:
    """태양 위치"""
    elevation: float  # 고도각 (도)
    azimuth: float  # 방위각 (도)
    zenith: float  # 천정각 (도)
    hour_angle: float  # 시간각 (도)
    declination: float  # 적위 (도)
    sunrise: Optional[datetime] = None
    sunset: Optional[datetime] = None


@dataclass
class Irradiance:
    """일사량 데이터"""
    ghi: float  # 전천일사량 (W/m²)
    dni: float  # 직달일사량 (W/m²)
    dhi: float  # 산란일사량 (W/m²)
    poa: float = 0.0  # 경사면일사량 (W/m²)
    timestamp: Optional[datetime] = None


class SolarPositionCalculator:
    """
    태양 위치 계산기

    Spencer 공식을 사용하여 태양 위치를 계산합니다.
    """

    def __init__(self, location: Location):
        """
        Args:
            location: 위치 정보
        """
        self.location = location

    def calculate(self, dt: datetime) -> SolarPosition:
        """
        태양 위치 계산

        Args:
            dt: 날짜/시간

        Returns:
            태양 위치 정보
        """
        # 날짜 정보
        day_of_year = dt.timetuple().tm_yday
        hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0

        # 라디안 변환
        lat_rad = math.radians(self.location.latitude)

        # 일년 중 각도 (라디안)
        gamma = 2 * math.pi * (day_of_year - 1) / 365

        # 적위 (declination) - Spencer 공식
        declination = (0.006918 - 0.399912 * math.cos(gamma)
                      + 0.070257 * math.sin(gamma)
                      - 0.006758 * math.cos(2 * gamma)
                      + 0.000907 * math.sin(2 * gamma)
                      - 0.002697 * math.cos(3 * gamma)
                      + 0.00148 * math.sin(3 * gamma))

        # 균시차 (equation of time) - 분
        eot = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))

        # 태양시 보정
        time_offset = eot + 4 * self.location.longitude - 60 * 9  # KST (UTC+9)
        solar_time = hour * 60 + time_offset

        # 시간각 (hour angle)
        hour_angle = (solar_time / 4) - 180  # 도
        hour_angle_rad = math.radians(hour_angle)

        # 천정각 (zenith angle)
        cos_zenith = (math.sin(lat_rad) * math.sin(declination)
                     + math.cos(lat_rad) * math.cos(declination) * math.cos(hour_angle_rad))
        cos_zenith = max(-1, min(1, cos_zenith))
        zenith = math.degrees(math.acos(cos_zenith))

        # 고도각 (elevation angle)
        elevation = 90 - zenith

        # 방위각 (azimuth angle)
        if cos_zenith > 0:
            cos_azimuth = (math.sin(declination) * math.cos(lat_rad)
                          - math.cos(declination) * math.sin(lat_rad) * math.cos(hour_angle_rad)) / math.sin(math.radians(zenith))
            cos_azimuth = max(-1, min(1, cos_azimuth))
            azimuth = math.degrees(math.acos(cos_azimuth))
            if hour_angle > 0:
                azimuth = 360 - azimuth
        else:
            azimuth = 180

        return SolarPosition(
            elevation=elevation,
            azimuth=azimuth,
            zenith=zenith,
            hour_angle=hour_angle,
            declination=math.degrees(declination)
        )

    def calculate_sunrise_sunset(self, date: datetime) -> Tuple[datetime, datetime]:
        """
        일출/일몰 시간 계산

        Args:
            date: 날짜

        Returns:
            (일출, 일몰) 시간
        """
        day_of_year = date.timetuple().tm_yday
        lat_rad = math.radians(self.location.latitude)

        # 적위 계산
        gamma = 2 * math.pi * (day_of_year - 1) / 365
        declination = (0.006918 - 0.399912 * math.cos(gamma)
                      + 0.070257 * math.sin(gamma)
                      - 0.006758 * math.cos(2 * gamma)
                      + 0.000907 * math.sin(2 * gamma))

        # 일출/일몰 시간각
        cos_omega = -math.tan(lat_rad) * math.tan(declination)
        cos_omega = max(-1, min(1, cos_omega))

        omega = math.degrees(math.acos(cos_omega))

        # 균시차
        eot = 229.18 * (0.000075 + 0.001868 * math.cos(gamma)
                       - 0.032077 * math.sin(gamma)
                       - 0.014615 * math.cos(2 * gamma)
                       - 0.040849 * math.sin(2 * gamma))

        # 태양 정오
        solar_noon = 720 - 4 * self.location.longitude - eot + 9 * 60  # KST

        # 일출/일몰
        sunrise_minutes = solar_noon - omega * 4
        sunset_minutes = solar_noon + omega * 4

        sunrise = date.replace(hour=0, minute=0, second=0) + timedelta(minutes=sunrise_minutes)
        sunset = date.replace(hour=0, minute=0, second=0) + timedelta(minutes=sunset_minutes)

        return sunrise, sunset


class ClearSkyModel:
    """
    맑은 날 일사량 모델

    Ineichen-Perez 모델을 기반으로 청천 일사량을 계산합니다.
    """

    # 대기 투과율 상수
    SOLAR_CONSTANT = 1361.0  # W/m²

    def __init__(
        self,
        location: Location,
        linke_turbidity: float = 3.0
    ):
        """
        Args:
            location: 위치 정보
            linke_turbidity: 링케 혼탁도 (기본값 3.0)
        """
        self.location = location
        self.linke_turbidity = linke_turbidity
        self.solar_calc = SolarPositionCalculator(location)

    def calculate(self, dt: datetime) -> Irradiance:
        """
        청천 일사량 계산

        Args:
            dt: 날짜/시간

        Returns:
            일사량 데이터
        """
        # 태양 위치
        solar_pos = self.solar_calc.calculate(dt)

        if solar_pos.elevation <= 0:
            return Irradiance(ghi=0.0, dni=0.0, dhi=0.0, timestamp=dt)

        # 대기질량 (air mass)
        zenith_rad = math.radians(solar_pos.zenith)
        am = 1 / (math.cos(zenith_rad) + 0.50572 * (96.07995 - solar_pos.zenith) ** (-1.6364))

        # 고도 보정
        altitude_correction = math.exp(-self.location.altitude / 8500)

        # Ineichen-Perez 모델
        fh1 = math.exp(-self.location.altitude / 8000)
        fh2 = math.exp(-self.location.altitude / 1250)

        cg1 = 5.09e-5 * self.location.altitude + 0.868
        cg2 = 3.92e-5 * self.location.altitude + 0.0387

        # 직달일사량 (DNI)
        dni = cg1 * self.SOLAR_CONSTANT * math.exp(
            -cg2 * am * (fh1 + fh2 * (self.linke_turbidity - 1))
        )
        dni = max(0, dni)

        # 산란일사량 (DHI)
        dhi = 0.0 if solar_pos.elevation <= 0 else (
            0.17 * self.SOLAR_CONSTANT * math.cos(zenith_rad) *
            (1 - math.exp(-self.linke_turbidity * am * 0.09))
        )
        dhi = max(0, dhi)

        # 전천일사량 (GHI)
        ghi = dni * math.cos(zenith_rad) + dhi
        ghi = max(0, ghi)

        return Irradiance(ghi=ghi, dni=dni, dhi=dhi, timestamp=dt)

    def calculate_daily(
        self,
        date: datetime,
        interval_minutes: int = 30
    ) -> List[Irradiance]:
        """
        일일 청천 일사량 계산

        Args:
            date: 날짜
            interval_minutes: 시간 간격 (분)

        Returns:
            일사량 데이터 리스트
        """
        results = []
        current = date.replace(hour=0, minute=0, second=0)
        end = current + timedelta(days=1)

        while current < end:
            irr = self.calculate(current)
            results.append(irr)
            current += timedelta(minutes=interval_minutes)

        return results


class POAIrradianceCalculator:
    """
    경사면 일사량 계산기

    Perez 모델을 사용하여 경사면 일사량을 계산합니다.
    """

    def __init__(
        self,
        location: Location,
        tilt: float = 30.0,
        azimuth: float = 180.0,
        albedo: float = 0.2
    ):
        """
        Args:
            location: 위치 정보
            tilt: 경사각 (도)
            azimuth: 방위각 (도, 남향=180)
            albedo: 지면 반사율
        """
        self.location = location
        self.tilt = tilt
        self.azimuth = azimuth
        self.albedo = albedo
        self.solar_calc = SolarPositionCalculator(location)

    def calculate(
        self,
        dt: datetime,
        ghi: float,
        dni: float,
        dhi: float
    ) -> float:
        """
        경사면 일사량 계산

        Args:
            dt: 날짜/시간
            ghi: 전천일사량 (W/m²)
            dni: 직달일사량 (W/m²)
            dhi: 산란일사량 (W/m²)

        Returns:
            경사면 일사량 (W/m²)
        """
        solar_pos = self.solar_calc.calculate(dt)

        if solar_pos.elevation <= 0:
            return 0.0

        # 라디안 변환
        tilt_rad = math.radians(self.tilt)
        azimuth_rad = math.radians(self.azimuth)
        solar_azimuth_rad = math.radians(solar_pos.azimuth)
        elevation_rad = math.radians(solar_pos.elevation)
        zenith_rad = math.radians(solar_pos.zenith)

        # 입사각 (Angle of Incidence)
        cos_aoi = (math.cos(zenith_rad) * math.cos(tilt_rad)
                  + math.sin(zenith_rad) * math.sin(tilt_rad)
                  * math.cos(solar_azimuth_rad - azimuth_rad))
        cos_aoi = max(0, cos_aoi)

        # 직달 성분
        poa_direct = dni * cos_aoi

        # 산란 성분 (등방성 가정)
        poa_diffuse = dhi * (1 + math.cos(tilt_rad)) / 2

        # 반사 성분
        poa_reflected = ghi * self.albedo * (1 - math.cos(tilt_rad)) / 2

        poa = poa_direct + poa_diffuse + poa_reflected
        return max(0, poa)


class PVSystemModel:
    """
    PV 시스템 모델

    태양광 발전 시스템의 출력을 계산합니다.
    """

    def __init__(
        self,
        location: Location,
        config: PVSystemConfig
    ):
        """
        Args:
            location: 위치 정보
            config: PV 시스템 설정
        """
        self.location = location
        self.config = config
        self.solar_calc = SolarPositionCalculator(location)
        self.clearsky_model = ClearSkyModel(location)
        self.poa_calc = POAIrradianceCalculator(
            location,
            config.tilt_angle,
            config.azimuth_angle
        )

    def calculate_power(
        self,
        poa: float,
        cell_temperature: float
    ) -> float:
        """
        발전량 계산

        Args:
            poa: 경사면 일사량 (W/m²)
            cell_temperature: 셀 온도 (°C)

        Returns:
            발전량 (kW)
        """
        if poa <= 0:
            return 0.0

        # 온도 보정
        temp_correction = 1 + self.config.temperature_coefficient * (
            cell_temperature - self.config.reference_temperature
        )

        # 출력 계산
        power = (
            self.config.capacity_kw
            * (poa / 1000)  # 정격 일사량 1000 W/m² 기준
            * self.config.efficiency
            * (1 - self.config.system_losses)
            * temp_correction
        )

        return max(0, power)

    def estimate_cell_temperature(
        self,
        ambient_temp: float,
        poa: float,
        wind_speed: float = 1.0
    ) -> float:
        """
        셀 온도 추정 (Sandia 모델)

        Args:
            ambient_temp: 기온 (°C)
            poa: 경사면 일사량 (W/m²)
            wind_speed: 풍속 (m/s)

        Returns:
            셀 온도 (°C)
        """
        # 모듈 타입별 계수
        if self.config.module_type == PVModuleType.MONOCRYSTALLINE:
            a, b = -3.47, -0.0594
        elif self.config.module_type == PVModuleType.POLYCRYSTALLINE:
            a, b = -3.56, -0.0750
        else:  # thin film
            a, b = -2.81, -0.0455

        # 모듈 온도
        module_temp = poa * math.exp(a + b * wind_speed) + ambient_temp

        # 셀 온도 (모듈 온도보다 약간 높음)
        delta_t = 3.0  # °C
        cell_temp = module_temp + poa / 1000 * delta_t

        return cell_temp

    def simulate(
        self,
        weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        시뮬레이션 실행

        Args:
            weather_data: 기상 데이터 (ghi, dni, dhi, temperature, wind_speed)

        Returns:
            발전량 데이터
        """
        results = []

        for idx, row in weather_data.iterrows():
            dt = idx if isinstance(idx, datetime) else datetime.now()

            # 일사량
            ghi = row.get('ghi', row.get('solar_radiation', 0)) * 1000  # MJ/m² → W/m²
            dni = row.get('dni', ghi * 0.8)
            dhi = row.get('dhi', ghi * 0.2)

            # 경사면 일사량
            poa = self.poa_calc.calculate(dt, ghi, dni, dhi)

            # 셀 온도
            ambient_temp = row.get('temperature', 25.0)
            wind_speed = row.get('wind_speed', 1.0)
            cell_temp = self.estimate_cell_temperature(ambient_temp, poa, wind_speed)

            # 발전량
            power = self.calculate_power(poa, cell_temp)

            results.append({
                'timestamp': dt,
                'poa': poa,
                'cell_temperature': cell_temp,
                'power_kw': power,
                'power_kwh': power  # 시간당
            })

        return pd.DataFrame(results)


class SolarPowerNet(nn.Module):
    """
    태양광 발전량 예측 신경망
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # 입력 처리
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 히든 레이어
        layers = []
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.hidden_layers = nn.Sequential(*layers)

        # 출력 레이어
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순전파"""
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class SolarPowerPredictor:
    """
    ML 기반 태양광 발전량 예측기
    """

    def __init__(
        self,
        pv_system: Optional[PVSystemModel] = None,
        model: Optional[nn.Module] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Args:
            pv_system: PV 시스템 모델
            model: 신경망 모델
            feature_names: 피처 이름 리스트
        """
        self.pv_system = pv_system
        self.model = model
        self.feature_names = feature_names or [
            'ghi', 'dni', 'dhi', 'temperature', 'humidity',
            'wind_speed', 'cloud_cover', 'hour', 'month'
        ]
        self._scaler = None
        self._fitted = False

    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """피처 준비"""
        features = []

        for col in self.feature_names:
            if col == 'hour':
                values = data.index.hour if hasattr(data.index, 'hour') else 12
            elif col == 'month':
                values = data.index.month if hasattr(data.index, 'month') else 6
            elif col in data.columns:
                values = data[col].values
            else:
                values = np.zeros(len(data))

            features.append(values)

        return np.column_stack(features)

    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str = 'power',
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ) -> Dict[str, List[float]]:
        """
        모델 학습

        Args:
            train_data: 학습 데이터
            target_col: 타겟 컬럼
            epochs: 에포크 수
            learning_rate: 학습률
            batch_size: 배치 크기
            device: 연산 디바이스

        Returns:
            학습 이력
        """
        if device is None:
            device = torch.device('cpu')

        # 피처 준비
        X = self.prepare_features(train_data)
        y = train_data[target_col].values

        # 정규화
        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # 텐서 변환
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        y_tensor = torch.FloatTensor(y).view(-1, 1).to(device)

        # 모델 생성
        if self.model is None:
            self.model = SolarPowerNet(X.shape[1])
        self.model = self.model.to(device)

        # 학습
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        history = {'loss': []}
        n_samples = len(X_tensor)

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            # 미니배치
            indices = torch.randperm(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                X_batch = X_tensor[batch_idx]
                y_batch = y_tensor[batch_idx]

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            history['loss'].append(epoch_loss / (n_samples / batch_size))

            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {history['loss'][-1]:.6f}")

        self._fitted = True
        return history

    def predict(
        self,
        data: pd.DataFrame,
        device: Optional[torch.device] = None
    ) -> np.ndarray:
        """
        발전량 예측

        Args:
            data: 입력 데이터
            device: 연산 디바이스

        Returns:
            예측 발전량
        """
        if device is None:
            device = torch.device('cpu')

        # 피처 준비
        X = self.prepare_features(data)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        X_tensor = torch.FloatTensor(X).to(device)

        # 예측
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor)

        return pred.cpu().numpy().flatten()

    def save(self, path: Path) -> None:
        """모델 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler': self._scaler,
            'feature_names': self.feature_names,
            'fitted': self._fitted
        }, path / 'solar_predictor.pt')

    @classmethod
    def load(cls, path: Path) -> 'SolarPowerPredictor':
        """모델 로드"""
        path = Path(path)
        checkpoint = torch.load(path / 'solar_predictor.pt', weights_only=False)

        predictor = cls(feature_names=checkpoint['feature_names'])
        predictor._scaler = checkpoint['scaler']
        predictor._fitted = checkpoint['fitted']

        # 모델 재생성
        input_size = len(checkpoint['feature_names'])
        predictor.model = SolarPowerNet(input_size)
        predictor.model.load_state_dict(checkpoint['model_state_dict'])

        return predictor


class SolarForecastEnsemble:
    """
    태양광 발전량 앙상블 예측기

    물리 모델과 ML 모델을 결합합니다.
    """

    def __init__(
        self,
        pv_system: PVSystemModel,
        ml_predictor: Optional[SolarPowerPredictor] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            pv_system: PV 시스템 모델
            ml_predictor: ML 예측기
            weights: 모델 가중치 {'physical': 0.5, 'ml': 0.5}
        """
        self.pv_system = pv_system
        self.ml_predictor = ml_predictor
        self.weights = weights or {'physical': 0.6, 'ml': 0.4}

    def predict(
        self,
        weather_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        앙상블 예측

        Args:
            weather_data: 기상 데이터

        Returns:
            예측 결과
        """
        results = []

        # 물리 모델 예측
        physical_results = self.pv_system.simulate(weather_data)

        # ML 모델 예측 (있는 경우)
        if self.ml_predictor is not None and self.ml_predictor._fitted:
            ml_predictions = self.ml_predictor.predict(weather_data)
        else:
            ml_predictions = physical_results['power_kw'].values

        # 앙상블
        ensemble_power = (
            self.weights['physical'] * physical_results['power_kw'].values
            + self.weights['ml'] * ml_predictions
        )

        results_df = physical_results.copy()
        results_df['power_physical'] = physical_results['power_kw']
        results_df['power_ml'] = ml_predictions
        results_df['power_ensemble'] = ensemble_power

        return results_df


# ============================================================================
# 팩토리 함수
# ============================================================================

def create_jeju_location() -> Location:
    """제주 위치 생성"""
    return Location(
        latitude=33.5,
        longitude=126.5,
        altitude=20.0,
        timezone='Asia/Seoul',
        name='Jeju'
    )


def create_pv_system(
    capacity_kw: float,
    location: Optional[Location] = None,
    **kwargs
) -> PVSystemModel:
    """
    PV 시스템 생성

    Args:
        capacity_kw: 설비 용량 (kW)
        location: 위치 (기본값: 제주)
        **kwargs: PVSystemConfig 추가 설정

    Returns:
        PVSystemModel 인스턴스
    """
    if location is None:
        location = create_jeju_location()

    config = PVSystemConfig(capacity_kw=capacity_kw, **kwargs)
    return PVSystemModel(location, config)


def create_solar_predictor(
    location: Optional[Location] = None,
    capacity_kw: float = 100.0,
    feature_names: Optional[List[str]] = None
) -> SolarPowerPredictor:
    """
    태양광 예측기 생성

    Args:
        location: 위치
        capacity_kw: 설비 용량
        feature_names: 피처 이름 리스트

    Returns:
        SolarPowerPredictor 인스턴스
    """
    pv_system = create_pv_system(capacity_kw, location)
    return SolarPowerPredictor(pv_system=pv_system, feature_names=feature_names)


def calculate_solar_power(
    weather_data: pd.DataFrame,
    capacity_kw: float = 100.0,
    location: Optional[Location] = None
) -> pd.DataFrame:
    """
    태양광 발전량 간편 계산

    Args:
        weather_data: 기상 데이터
        capacity_kw: 설비 용량
        location: 위치

    Returns:
        발전량 데이터
    """
    pv_system = create_pv_system(capacity_kw, location)
    return pv_system.simulate(weather_data)
