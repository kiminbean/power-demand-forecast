"""
태양광 발전량 모델 테스트
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path


class TestLocation:
    """Location 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import Location

        loc = Location(latitude=33.5, longitude=126.5, altitude=20.0)

        assert loc.latitude == 33.5
        assert loc.longitude == 126.5

    def test_default_values(self):
        """기본값"""
        from src.models.solar import Location

        loc = Location(latitude=33.5, longitude=126.5)

        assert loc.altitude == 0.0
        assert loc.timezone == 'Asia/Seoul'


class TestPVSystemConfig:
    """PVSystemConfig 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import PVSystemConfig, PVModuleType

        config = PVSystemConfig(
            capacity_kw=100.0,
            module_type=PVModuleType.MONOCRYSTALLINE
        )

        assert config.capacity_kw == 100.0
        assert config.efficiency == 0.18

    def test_default_values(self):
        """기본값"""
        from src.models.solar import PVSystemConfig

        config = PVSystemConfig(capacity_kw=50.0)

        assert config.tilt_angle == 30.0
        assert config.azimuth_angle == 180.0


class TestSolarPositionCalculator:
    """SolarPositionCalculator 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import SolarPositionCalculator, Location

        loc = Location(latitude=33.5, longitude=126.5)
        calc = SolarPositionCalculator(loc)

        assert calc.location.latitude == 33.5

    def test_calculate_noon(self):
        """정오 태양 위치"""
        from src.models.solar import SolarPositionCalculator, Location

        loc = Location(latitude=33.5, longitude=126.5)
        calc = SolarPositionCalculator(loc)

        # 정오
        dt = datetime(2025, 6, 21, 12, 0)  # 하지
        pos = calc.calculate(dt)

        assert pos.elevation > 0  # 태양이 떠 있음
        assert pos.zenith < 90

    def test_calculate_night(self):
        """야간 태양 위치"""
        from src.models.solar import SolarPositionCalculator, Location

        loc = Location(latitude=33.5, longitude=126.5)
        calc = SolarPositionCalculator(loc)

        # 자정
        dt = datetime(2025, 6, 21, 0, 0)
        pos = calc.calculate(dt)

        assert pos.elevation < 0  # 태양이 지평선 아래

    def test_calculate_sunrise_sunset(self):
        """일출/일몰 계산"""
        from src.models.solar import SolarPositionCalculator, Location

        loc = Location(latitude=33.5, longitude=126.5)
        calc = SolarPositionCalculator(loc)

        date = datetime(2025, 6, 21)
        sunrise, sunset = calc.calculate_sunrise_sunset(date)

        assert sunrise.hour < 6  # 일출은 오전
        assert sunset.hour > 18  # 일몰은 저녁
        assert sunset > sunrise


class TestClearSkyModel:
    """ClearSkyModel 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import ClearSkyModel, Location

        loc = Location(latitude=33.5, longitude=126.5)
        model = ClearSkyModel(loc)

        assert model.linke_turbidity == 3.0

    def test_calculate_noon(self):
        """정오 청천 일사량"""
        from src.models.solar import ClearSkyModel, Location

        loc = Location(latitude=33.5, longitude=126.5)
        model = ClearSkyModel(loc)

        dt = datetime(2025, 6, 21, 12, 0)
        irr = model.calculate(dt)

        assert irr.ghi > 0
        assert irr.dni > 0
        assert irr.dhi > 0

    def test_calculate_night(self):
        """야간 일사량"""
        from src.models.solar import ClearSkyModel, Location

        loc = Location(latitude=33.5, longitude=126.5)
        model = ClearSkyModel(loc)

        dt = datetime(2025, 6, 21, 0, 0)
        irr = model.calculate(dt)

        assert irr.ghi == 0
        assert irr.dni == 0
        assert irr.dhi == 0

    def test_calculate_daily(self):
        """일일 일사량"""
        from src.models.solar import ClearSkyModel, Location

        loc = Location(latitude=33.5, longitude=126.5)
        model = ClearSkyModel(loc)

        date = datetime(2025, 6, 21)
        irradiances = model.calculate_daily(date, interval_minutes=60)

        assert len(irradiances) == 24
        assert any(irr.ghi > 0 for irr in irradiances)


class TestPOAIrradianceCalculator:
    """POAIrradianceCalculator 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import POAIrradianceCalculator, Location

        loc = Location(latitude=33.5, longitude=126.5)
        calc = POAIrradianceCalculator(loc, tilt=30.0, azimuth=180.0)

        assert calc.tilt == 30.0

    def test_calculate(self):
        """경사면 일사량 계산"""
        from src.models.solar import POAIrradianceCalculator, Location

        loc = Location(latitude=33.5, longitude=126.5)
        calc = POAIrradianceCalculator(loc, tilt=30.0, azimuth=180.0)

        dt = datetime(2025, 6, 21, 12, 0)
        poa = calc.calculate(dt, ghi=800, dni=600, dhi=200)

        assert poa > 0


class TestPVSystemModel:
    """PVSystemModel 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import PVSystemModel, PVSystemConfig, Location

        loc = Location(latitude=33.5, longitude=126.5)
        config = PVSystemConfig(capacity_kw=100.0)
        model = PVSystemModel(loc, config)

        assert model.config.capacity_kw == 100.0

    def test_calculate_power(self):
        """발전량 계산"""
        from src.models.solar import PVSystemModel, PVSystemConfig, Location

        loc = Location(latitude=33.5, longitude=126.5)
        config = PVSystemConfig(capacity_kw=100.0)
        model = PVSystemModel(loc, config)

        power = model.calculate_power(poa=800, cell_temperature=35.0)

        assert power > 0
        assert power < config.capacity_kw

    def test_calculate_power_zero_poa(self):
        """일사량 0일 때"""
        from src.models.solar import PVSystemModel, PVSystemConfig, Location

        loc = Location(latitude=33.5, longitude=126.5)
        config = PVSystemConfig(capacity_kw=100.0)
        model = PVSystemModel(loc, config)

        power = model.calculate_power(poa=0, cell_temperature=25.0)

        assert power == 0

    def test_estimate_cell_temperature(self):
        """셀 온도 추정"""
        from src.models.solar import PVSystemModel, PVSystemConfig, Location

        loc = Location(latitude=33.5, longitude=126.5)
        config = PVSystemConfig(capacity_kw=100.0)
        model = PVSystemModel(loc, config)

        cell_temp = model.estimate_cell_temperature(
            ambient_temp=25.0, poa=800, wind_speed=2.0
        )

        assert cell_temp > 25.0  # 기온보다 높음

    def test_simulate(self):
        """시뮬레이션"""
        from src.models.solar import PVSystemModel, PVSystemConfig, Location

        loc = Location(latitude=33.5, longitude=126.5)
        config = PVSystemConfig(capacity_kw=100.0)
        model = PVSystemModel(loc, config)

        # 가상 기상 데이터
        weather = pd.DataFrame({
            'ghi': [0.5, 0.8, 0.6],  # MJ/m²
            'temperature': [25.0, 28.0, 26.0],
            'wind_speed': [2.0, 1.5, 2.5]
        }, index=pd.date_range('2025-06-21 10:00', periods=3, freq='h'))

        results = model.simulate(weather)

        assert 'power_kw' in results.columns
        assert len(results) == 3


class TestSolarPowerNet:
    """SolarPowerNet 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import SolarPowerNet

        model = SolarPowerNet(input_size=9, hidden_size=64)

        assert model.input_size == 9
        assert model.hidden_size == 64

    def test_forward(self):
        """순전파"""
        from src.models.solar import SolarPowerNet

        model = SolarPowerNet(input_size=9)
        x = torch.randn(16, 9)

        out = model(x)

        assert out.shape == (16, 1)


class TestSolarPowerPredictor:
    """SolarPowerPredictor 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import SolarPowerPredictor

        predictor = SolarPowerPredictor()

        assert len(predictor.feature_names) > 0

    def test_prepare_features(self):
        """피처 준비"""
        from src.models.solar import SolarPowerPredictor

        predictor = SolarPowerPredictor(feature_names=['ghi', 'temperature'])

        data = pd.DataFrame({
            'ghi': [500, 600, 700],
            'temperature': [25, 26, 27]
        })

        features = predictor.prepare_features(data)

        assert features.shape == (3, 2)

    def test_fit(self):
        """학습"""
        from src.models.solar import SolarPowerPredictor

        predictor = SolarPowerPredictor(feature_names=['ghi', 'temperature'])

        data = pd.DataFrame({
            'ghi': np.random.rand(100) * 1000,
            'temperature': np.random.rand(100) * 20 + 15,
            'power': np.random.rand(100) * 50
        })

        history = predictor.fit(data, epochs=10, batch_size=16)

        assert 'loss' in history
        assert predictor._fitted

    def test_predict(self):
        """예측"""
        from src.models.solar import SolarPowerPredictor

        predictor = SolarPowerPredictor(feature_names=['ghi', 'temperature'])

        train_data = pd.DataFrame({
            'ghi': np.random.rand(100) * 1000,
            'temperature': np.random.rand(100) * 20 + 15,
            'power': np.random.rand(100) * 50
        })
        predictor.fit(train_data, epochs=5)

        test_data = pd.DataFrame({
            'ghi': [500, 600, 700],
            'temperature': [25, 26, 27]
        })

        predictions = predictor.predict(test_data)

        assert len(predictions) == 3

    def test_save_load(self, tmp_path):
        """저장/로드"""
        from src.models.solar import SolarPowerPredictor

        predictor = SolarPowerPredictor(feature_names=['ghi', 'temperature'])

        train_data = pd.DataFrame({
            'ghi': np.random.rand(50) * 1000,
            'temperature': np.random.rand(50) * 20 + 15,
            'power': np.random.rand(50) * 50
        })
        predictor.fit(train_data, epochs=5)
        predictor.save(tmp_path)

        loaded = SolarPowerPredictor.load(tmp_path)

        assert loaded._fitted
        assert loaded.feature_names == predictor.feature_names


class TestSolarForecastEnsemble:
    """SolarForecastEnsemble 테스트"""

    def test_creation(self):
        """생성"""
        from src.models.solar import (
            SolarForecastEnsemble, PVSystemModel, PVSystemConfig, Location
        )

        loc = Location(latitude=33.5, longitude=126.5)
        config = PVSystemConfig(capacity_kw=100.0)
        pv_system = PVSystemModel(loc, config)

        ensemble = SolarForecastEnsemble(pv_system)

        assert ensemble.weights['physical'] == 0.6

    def test_predict(self):
        """앙상블 예측"""
        from src.models.solar import (
            SolarForecastEnsemble, PVSystemModel, PVSystemConfig, Location
        )

        loc = Location(latitude=33.5, longitude=126.5)
        config = PVSystemConfig(capacity_kw=100.0)
        pv_system = PVSystemModel(loc, config)

        ensemble = SolarForecastEnsemble(pv_system)

        weather = pd.DataFrame({
            'ghi': [0.5, 0.8, 0.6],
            'temperature': [25.0, 28.0, 26.0],
            'wind_speed': [2.0, 1.5, 2.5]
        }, index=pd.date_range('2025-06-21 10:00', periods=3, freq='h'))

        results = ensemble.predict(weather)

        assert 'power_physical' in results.columns
        assert 'power_ensemble' in results.columns


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_jeju_location(self):
        """제주 위치 생성"""
        from src.models.solar import create_jeju_location

        loc = create_jeju_location()

        assert loc.latitude == 33.5
        assert loc.longitude == 126.5
        assert loc.name == 'Jeju'

    def test_create_pv_system(self):
        """PV 시스템 생성"""
        from src.models.solar import create_pv_system

        system = create_pv_system(capacity_kw=100.0)

        assert system.config.capacity_kw == 100.0

    def test_create_pv_system_with_location(self):
        """위치 지정 PV 시스템"""
        from src.models.solar import create_pv_system, Location

        loc = Location(latitude=35.0, longitude=127.0)
        system = create_pv_system(capacity_kw=50.0, location=loc)

        assert system.location.latitude == 35.0

    def test_create_solar_predictor(self):
        """예측기 생성"""
        from src.models.solar import create_solar_predictor

        predictor = create_solar_predictor(capacity_kw=100.0)

        assert predictor.pv_system is not None

    def test_calculate_solar_power(self):
        """간편 발전량 계산"""
        from src.models.solar import calculate_solar_power

        weather = pd.DataFrame({
            'ghi': [0.5, 0.8, 0.6],
            'temperature': [25.0, 28.0, 26.0],
            'wind_speed': [2.0, 1.5, 2.5]
        }, index=pd.date_range('2025-06-21 10:00', periods=3, freq='h'))

        results = calculate_solar_power(weather, capacity_kw=100.0)

        assert 'power_kw' in results.columns


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self):
        """전체 워크플로우"""
        from src.models.solar import (
            create_jeju_location, create_pv_system,
            ClearSkyModel, SolarPowerPredictor
        )

        # 1. 위치 및 시스템 설정
        location = create_jeju_location()
        pv_system = create_pv_system(100.0, location)

        # 2. 청천 일사량 계산
        clearsky = ClearSkyModel(location)
        date = datetime(2025, 6, 21)
        irradiances = clearsky.calculate_daily(date, interval_minutes=60)

        # 3. 기상 데이터 준비
        weather_data = pd.DataFrame({
            'ghi': [irr.ghi / 1000 for irr in irradiances],  # W/m² → MJ/m²
            'dni': [irr.dni / 1000 for irr in irradiances],
            'dhi': [irr.dhi / 1000 for irr in irradiances],
            'temperature': [25.0] * len(irradiances),
            'wind_speed': [2.0] * len(irradiances)
        }, index=pd.date_range(date, periods=len(irradiances), freq='h'))

        # 4. 발전량 시뮬레이션
        results = pv_system.simulate(weather_data)

        assert len(results) == 24
        assert 'power_kw' in results.columns
        # 낮에는 발전량이 있어야 함
        assert results['power_kw'].max() > 0

    def test_ml_pipeline(self):
        """ML 파이프라인"""
        from src.models.solar import SolarPowerPredictor

        # 학습 데이터 생성
        n_samples = 200
        train_data = pd.DataFrame({
            'ghi': np.random.rand(n_samples) * 1000,
            'temperature': np.random.rand(n_samples) * 20 + 15,
            'humidity': np.random.rand(n_samples) * 40 + 40,
            'wind_speed': np.random.rand(n_samples) * 5,
        })
        train_data['power'] = (
            train_data['ghi'] * 0.08
            - (train_data['temperature'] - 25) * 0.5
            + np.random.randn(n_samples) * 5
        )

        # 예측기 생성 및 학습
        predictor = SolarPowerPredictor(
            feature_names=['ghi', 'temperature', 'humidity', 'wind_speed']
        )
        history = predictor.fit(train_data, epochs=20, batch_size=32)

        # 예측
        test_data = pd.DataFrame({
            'ghi': [500, 800, 300],
            'temperature': [25, 30, 20],
            'humidity': [60, 55, 70],
            'wind_speed': [2, 1, 3]
        })
        predictions = predictor.predict(test_data)

        assert len(predictions) == 3
        assert history['loss'][-1] < history['loss'][0]  # 손실 감소

