"""
제주도 전력 수요 예측 대시보드 v1.0
=====================================

EPSIS 스타일의 전문적인 전력 수요 예측 대시보드

주요 기능:
1. 실시간 수급 현황 (EPSIS 스타일 게이지)
2. 다중 시간대 예측 시각화 (1h/6h/24h)
3. 시나리오 분석 (폭염/한파)
4. 신재생에너지 발전량 예측
5. 과거 데이터 분석

Usage:
    streamlit run src/dashboard/app_v1.py

    # API 서버 실행 필요:
    uvicorn api.main:app --port 8000  # 전력 수요 예측
    uvicorn api.main:app --port 8001  # 신재생에너지 (별도 프로젝트)

Author: Power Demand Forecast Team
Version: 1.0.0
Date: 2025-12
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from pathlib import Path
import sys
import io

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# EPSIS 크롤러 import (직접 임포트로 다른 크롤러 의존성 회피)
try:
    import importlib.util
    epsis_spec = importlib.util.spec_from_file_location(
        "epsis_crawler",
        PROJECT_ROOT / "tools" / "crawlers" / "epsis_crawler.py"
    )
    epsis_module = importlib.util.module_from_spec(epsis_spec)
    epsis_spec.loader.exec_module(epsis_module)
    EPSISCrawler = epsis_module.EPSISCrawler
    JejuEstimator = epsis_module.JejuEstimator
    PowerSupplyData = epsis_module.PowerSupplyData
    EPSIS_AVAILABLE = True
except Exception as e:
    EPSIS_AVAILABLE = False
    print(f"EPSIS crawler import failed: {e}")

# 제주 전력수급현황 크롤러 import
try:
    jeju_spec = importlib.util.spec_from_file_location(
        "jeju_power_crawler",
        PROJECT_ROOT / "tools" / "crawlers" / "jeju_power_crawler.py"
    )
    jeju_module = importlib.util.module_from_spec(jeju_spec)
    jeju_spec.loader.exec_module(jeju_module)
    JejuPowerCrawler = jeju_module.JejuPowerCrawler
    JejuPowerData = jeju_module.JejuPowerData
    JEJU_CRAWLER_AVAILABLE = True
except Exception as e:
    JEJU_CRAWLER_AVAILABLE = False
    print(f"Jeju power crawler import failed: {e}")


# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="제주도 전력 수요 예측 v1.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Config 클래스
# ============================================================================

class Config:
    """대시보드 설정"""

    # API 설정
    DEMAND_API_URL = "http://localhost:8000"
    RENEWABLE_API_URL = "http://localhost:8001"

    # 데이터 경로
    DATA_PATH = PROJECT_ROOT / "data" / "processed"
    MODEL_PATH = PROJECT_ROOT / "models"

    # 제주도 전력 시스템 기준값 (MW)
    JEJU_SUPPLY_CAPACITY = 1500  # 기본 공급능력
    JEJU_PEAK_DEMAND = 1100      # 피크 수요 기준
    RESERVE_WARNING_THRESHOLD = 10  # 예비율 경고 기준 (%)
    RESERVE_CRITICAL_THRESHOLD = 5   # 예비율 위험 기준 (%)

    # 자동 갱신 간격 (초)
    REFRESH_INTERVAL = 60

    # EPSIS 스타일 + 기존 색상 통합
    COLORS = {
        # 수급 현황 (EPSIS 스타일)
        'supply': '#0054FF',      # 공급능력 - 파랑
        'demand': '#FF0000',      # 현재수요 - 빨강
        'reserve': '#00B050',     # 예비력 - 초록
        'warning': '#FFC000',     # 경고 - 노랑
        'critical': '#C00000',    # 위험 - 진한빨강

        # 예측 (기존)
        'prediction': '#3B82F6',  # 예측 - 파랑
        'actual': '#10B981',      # 실제 - 초록
        'confidence': 'rgba(59, 130, 246, 0.2)',  # 신뢰구간

        # 신재생
        'solar': '#F59E0B',       # 태양광 - 호박색
        'wind': '#3B82F6',        # 풍력 - 파랑
        'renewable_total': '#10B981',  # 합계 - 초록

        # 배경/그리드
        'grid': '#E5E7EB',
        'background': '#F9FAFB',

        # 헤더
        'primary': '#1E3A8A',
        'secondary': '#64748B',
    }

    # 시나리오 프리셋
    SCENARIOS = {
        "normal": {"name": "평년", "temp_delta": 0, "humidity_delta": 0, "demand_factor": 1.0},
        "heatwave_mild": {"name": "약한 폭염 (+3°C)", "temp_delta": 3, "humidity_delta": -5, "demand_factor": 1.08},
        "heatwave_severe": {"name": "심한 폭염 (+7°C)", "temp_delta": 7, "humidity_delta": -10, "demand_factor": 1.20},
        "coldwave_mild": {"name": "약한 한파 (-5°C)", "temp_delta": -5, "humidity_delta": 5, "demand_factor": 1.10},
        "coldwave_severe": {"name": "심한 한파 (-10°C)", "temp_delta": -10, "humidity_delta": 10, "demand_factor": 1.25},
    }

    # 시나리오별 색상
    SCENARIO_COLORS = {
        'normal': '#64748B',
        'heatwave_mild': '#F97316',
        'heatwave_severe': '#DC2626',
        'coldwave_mild': '#0EA5E9',
        'coldwave_severe': '#1D4ED8'
    }


# ============================================================================
# CSS 스타일
# ============================================================================

st.markdown("""
<style>
    /* 메인 헤더 */
    .main-header {
        font-size: 2.2rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.3rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* API 상태 배지 */
    .api-connected {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }
    .api-disconnected {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.3rem 0.8rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* 게이지 카드 */
    .gauge-card {
        background: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    .gauge-title {
        font-size: 0.9rem;
        color: #64748B;
        margin-bottom: 0.5rem;
    }
    .gauge-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
    .gauge-unit {
        font-size: 0.9rem;
        color: #94A3B8;
    }

    /* 상태 인디케이터 */
    .status-safe { color: #10B981; }
    .status-warning { color: #F59E0B; }
    .status-danger { color: #EF4444; }

    /* 섹션 구분 */
    .section-divider {
        border-top: 2px solid #E5E7EB;
        margin: 1.5rem 0;
    }

    /* 탭 스타일 개선 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API 클라이언트
# ============================================================================

class PowerDemandAPIClient:
    """전력 수요 예측 API 클라이언트"""

    def __init__(self, base_url: str = Config.DEMAND_API_URL):
        self.base_url = base_url

    def health_check(self) -> Dict[str, Any]:
        """API 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {"status": "offline", "models_loaded": False}

    def get_models(self) -> Optional[Dict]:
        """모델 정보 조회"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def predict(self, data: List[Dict], model_type: str = "conditional") -> Optional[Dict]:
        """단일 예측"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"data": data, "model_type": model_type},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"예측 API 오류: {e}")
        return None

    def predict_conditional(self, data: List[Dict], mode: str = "soft") -> Optional[Dict]:
        """조건부 예측 (겨울철 최적화)"""
        try:
            response = requests.post(
                f"{self.base_url}/predict/conditional",
                json={"data": data, "mode": mode},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"조건부 예측 오류: {e}")
        return None

    def predict_batch(self, data: List[Dict], model_type: str = "demand_only", step: int = 1) -> Optional[Dict]:
        """배치 예측 (슬라이딩 윈도우)"""
        try:
            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={"data": data, "model_type": model_type, "step": step},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"배치 예측 오류: {e}")
        return None


class RenewableAPIClient:
    """신재생에너지 발전량 예측 API 클라이언트"""

    def __init__(self, base_url: str = Config.RENEWABLE_API_URL):
        self.base_url = base_url

    def health_check(self) -> Dict[str, Any]:
        """API 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {"status": "offline", "models_loaded": {"wind": False, "solar": False}}

    def get_models(self) -> Optional[Dict]:
        """모델 정보 조회"""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return None

    def predict(self, weather: Dict, energy_type: str = "both", include_uncertainty: bool = True) -> Optional[Dict]:
        """단일 예측"""
        try:
            if isinstance(weather.get("datetime"), datetime):
                weather = weather.copy()
                weather["datetime"] = weather["datetime"].isoformat()

            response = requests.post(
                f"{self.base_url}/predict",
                json={
                    "weather": weather,
                    "energy_type": energy_type,
                    "include_uncertainty": include_uncertainty
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"신재생 예측 오류: {e}")
        return None

    def predict_batch(self, weather_data: List[Dict], energy_type: str = "both") -> Optional[Dict]:
        """배치 예측 (최대 168시간)"""
        try:
            converted_data = []
            for w in weather_data:
                w_copy = w.copy()
                if isinstance(w_copy.get("datetime"), datetime):
                    w_copy["datetime"] = w_copy["datetime"].isoformat()
                converted_data.append(w_copy)

            response = requests.post(
                f"{self.base_url}/predict/batch",
                json={"weather_data": converted_data, "energy_type": energy_type},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            st.error(f"신재생 배치 예측 오류: {e}")
        return None


# ============================================================================
# DataManager 클래스
# ============================================================================

class DataManager:
    """데이터 관리 클래스"""

    @staticmethod
    @st.cache_data(ttl=300)
    def load_historical_data() -> Optional[pd.DataFrame]:
        """과거 데이터 로드"""
        try:
            data_file = Config.DATA_PATH / "jeju_hourly_merged.csv"
            if data_file.exists():
                df = pd.read_csv(data_file)
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df.set_index('datetime', inplace=True)
                return df
        except Exception as e:
            st.error(f"데이터 로드 실패: {e}")
        return None

    @staticmethod
    def prepare_api_data(df: pd.DataFrame, n_points: int = 168) -> List[Dict]:
        """DataFrame을 API 요청 형식으로 변환"""
        recent_data = df.tail(n_points).copy()

        api_data = []
        for idx, row in recent_data.iterrows():
            record = {
                "datetime": idx.isoformat() if isinstance(idx, pd.Timestamp) else str(idx),
                "power_demand": float(row['power_demand']),
            }

            # 기상 데이터 추가
            if '기온' in row and pd.notna(row['기온']):
                record["temperature"] = float(row['기온'])
            if '습도' in row and pd.notna(row['습도']):
                record["humidity"] = float(row['습도'])
            if '풍속' in row and pd.notna(row['풍속']):
                record["wind_speed"] = float(row['풍속'])
            if '강수량' in row and pd.notna(row['강수량']):
                record["precipitation"] = float(row['강수량'])

            api_data.append(record)

        return api_data

    @staticmethod
    def apply_weather_modification(df: pd.DataFrame, temp_delta: float = 0, humidity_delta: float = 0) -> pd.DataFrame:
        """기상 조건 수정 적용"""
        modified = df.copy()

        if '기온' in modified.columns:
            modified['기온'] = modified['기온'] + temp_delta
        if '습도' in modified.columns:
            modified['습도'] = (modified['습도'] + humidity_delta).clip(0, 100)

        return modified

    @staticmethod
    def calculate_supply_status(
        current_demand: float,
        supply_capacity: float = Config.JEJU_SUPPLY_CAPACITY
    ) -> Dict[str, Any]:
        """수급 상태 계산"""
        reserve_power = supply_capacity - current_demand
        reserve_rate = (reserve_power / supply_capacity) * 100 if supply_capacity > 0 else 0
        utilization = (current_demand / supply_capacity) * 100 if supply_capacity > 0 else 0

        # 상태 판단
        if reserve_rate >= Config.RESERVE_WARNING_THRESHOLD:
            status = "safe"
            status_text = "정상"
        elif reserve_rate >= Config.RESERVE_CRITICAL_THRESHOLD:
            status = "warning"
            status_text = "주의"
        else:
            status = "danger"
            status_text = "위험"

        return {
            "supply_capacity": supply_capacity,
            "current_demand": current_demand,
            "reserve_power": reserve_power,
            "reserve_rate": reserve_rate,
            "utilization": utilization,
            "status": status,
            "status_text": status_text
        }

    @staticmethod
    @st.cache_data(ttl=60)  # 1분 캐시
    def fetch_epsis_realtime() -> Optional[Dict[str, Any]]:
        """EPSIS 실시간 전력 수급 데이터 조회"""
        if not EPSIS_AVAILABLE:
            return None

        try:
            crawler = EPSISCrawler(timeout=15, max_retries=2)
            jeju_estimator = JejuEstimator()

            # 오늘 데이터 조회
            data = crawler.fetch_realtime_data()
            crawler.close()

            if not data:
                return None

            # 최신 데이터 추출
            latest_national = data[-1]
            latest_jeju = jeju_estimator.estimate_jeju_demand(latest_national)

            # 최근 24시간 데이터 (5분 간격 = 288건 중 최근 288건)
            recent_data = data[-288:] if len(data) >= 288 else data
            jeju_data = [jeju_estimator.estimate_jeju_demand(d) for d in recent_data]

            # dataclass를 dict로 변환 (pickle 직렬화 가능하도록)
            return {
                'national': {
                    'latest': latest_national.to_dict(),
                    'history': [d.to_dict() for d in recent_data],
                },
                'jeju': {
                    'latest': latest_jeju.to_dict(),
                    'history': [d.to_dict() for d in jeju_data],
                },
                'fetched_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'data_count': len(data),
            }

        except Exception as e:
            st.warning(f"EPSIS 데이터 조회 실패: {e}")
            return None

    @staticmethod
    @st.cache_data(ttl=3600)  # 1시간 캐시
    def fetch_jeju_actual_data() -> Optional[Dict[str, Any]]:
        """제주 실측 전력수급 데이터 로드 (공공데이터포털)"""
        if not JEJU_CRAWLER_AVAILABLE:
            return None

        try:
            # ZIP 파일 경로 (data 디렉토리)
            zip_path = PROJECT_ROOT / "data" / "jeju_power_supply.zip"

            if not zip_path.exists():
                return None

            crawler = JejuPowerCrawler()
            data = crawler.load_from_zip(zip_path)
            crawler.close()

            if not data:
                return None

            # 최신 데이터 추출
            latest = data[-1]

            # 최근 7일 데이터 (168시간)
            recent_data = data[-168:] if len(data) >= 168 else data

            return {
                'latest': latest.to_dict(),
                'history': [d.to_dict() for d in recent_data],
                'total_records': len(data),
                'date_range': {
                    'start': data[0].timestamp,
                    'end': data[-1].timestamp,
                },
                'fetched_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'source': 'data.go.kr (한국전력거래소_제주 전력수급현황)',
            }

        except Exception as e:
            st.warning(f"제주 실측 데이터 로드 실패: {e}")
            return None

    @staticmethod
    def create_sample_weather(
        base_datetime: datetime,
        hours: int = 24,
        temp: float = 15.0,
        humidity: float = 60.0,
        wind_speed: float = 5.0,
        wind_direction: float = 270.0
    ) -> List[Dict]:
        """샘플 기상 데이터 생성 (신재생 API용)"""
        weather_list = []

        for h in range(hours):
            dt = base_datetime + timedelta(hours=h)
            hour = dt.hour

            # 시간대별 변동 적용
            temp_adj = temp + 5 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else temp - 3
            solar_rad = max(0, 3.5 * np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0

            weather = {
                "datetime": dt.isoformat(),
                "temperature": temp_adj,
                "humidity": humidity,
                "wind_speed": wind_speed + np.random.uniform(-1, 1),
                "wind_direction": wind_direction,
                "pressure": 1013.0 + np.random.uniform(-5, 5),
                "solar_radiation": solar_rad,
                "cloud_cover": np.random.randint(0, 5),
            }
            weather_list.append(weather)

        return weather_list


# ============================================================================
# GaugeComponents 클래스 (EPSIS 스타일)
# ============================================================================

class GaugeComponents:
    """EPSIS 스타일 게이지 컴포넌트"""

    @staticmethod
    def create_supply_gauge(supply_capacity: float, max_value: float = 1500) -> go.Figure:
        """공급능력 게이지"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=supply_capacity,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "공급능력", 'font': {'size': 14, 'color': '#64748B'}},
            number={'suffix': " MW", 'font': {'size': 28, 'color': Config.COLORS['supply']}},
            gauge={
                'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': '#CBD5E1'},
                'bar': {'color': Config.COLORS['supply']},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E5E7EB",
                'steps': [
                    {'range': [0, max_value * 0.7], 'color': '#EFF6FF'},
                    {'range': [max_value * 0.7, max_value * 0.9], 'color': '#DBEAFE'},
                    {'range': [max_value * 0.9, max_value], 'color': '#BFDBFE'}
                ],
                'threshold': {
                    'line': {'color': "#1E40AF", 'width': 2},
                    'thickness': 0.75,
                    'value': supply_capacity
                }
            }
        ))
        fig.update_layout(
            height=180,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Arial'}
        )
        return fig

    @staticmethod
    def create_demand_gauge(current_demand: float, supply_capacity: float, max_value: float = 1500) -> go.Figure:
        """현재수요 게이지 (이용률에 따른 색상 변화)"""
        utilization = (current_demand / supply_capacity) * 100 if supply_capacity > 0 else 0

        # 이용률에 따른 색상 결정
        if utilization < 70:
            bar_color = Config.COLORS['reserve']  # 안전 - 초록
        elif utilization < 85:
            bar_color = Config.COLORS['warning']  # 주의 - 노랑
        elif utilization < 95:
            bar_color = '#FF6B6B'  # 경고 - 연한 빨강
        else:
            bar_color = Config.COLORS['critical']  # 위험 - 진한 빨강

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_demand,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "현재수요", 'font': {'size': 14, 'color': '#64748B'}},
            number={'suffix': " MW", 'font': {'size': 28, 'color': bar_color}},
            gauge={
                'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': '#CBD5E1'},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E5E7EB",
                'steps': [
                    {'range': [0, max_value * 0.7], 'color': '#FEF2F2'},
                    {'range': [max_value * 0.7, max_value * 0.85], 'color': '#FEE2E2'},
                    {'range': [max_value * 0.85, max_value], 'color': '#FECACA'}
                ],
                'threshold': {
                    'line': {'color': Config.COLORS['supply'], 'width': 3},
                    'thickness': 0.75,
                    'value': supply_capacity
                }
            }
        ))
        fig.update_layout(
            height=180,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Arial'}
        )
        return fig

    @staticmethod
    def create_reserve_gauge(reserve_power: float, max_value: float = 500) -> go.Figure:
        """예비력 게이지"""
        # 예비력에 따른 색상
        if reserve_power >= 150:
            bar_color = Config.COLORS['reserve']
        elif reserve_power >= 75:
            bar_color = Config.COLORS['warning']
        else:
            bar_color = Config.COLORS['critical']

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=reserve_power,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "예비력", 'font': {'size': 14, 'color': '#64748B'}},
            number={'suffix': " MW", 'font': {'size': 28, 'color': bar_color}},
            gauge={
                'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': '#CBD5E1'},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E5E7EB",
                'steps': [
                    {'range': [0, 75], 'color': '#FEE2E2'},
                    {'range': [75, 150], 'color': '#FEF3C7'},
                    {'range': [150, max_value], 'color': '#D1FAE5'}
                ]
            }
        ))
        fig.update_layout(
            height=180,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Arial'}
        )
        return fig

    @staticmethod
    def create_reserve_rate_gauge(reserve_rate: float, max_value: float = 30) -> go.Figure:
        """공급예비율 게이지"""
        # 예비율에 따른 색상
        if reserve_rate >= Config.RESERVE_WARNING_THRESHOLD:
            bar_color = Config.COLORS['reserve']
        elif reserve_rate >= Config.RESERVE_CRITICAL_THRESHOLD:
            bar_color = Config.COLORS['warning']
        else:
            bar_color = Config.COLORS['critical']

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=reserve_rate,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "공급예비율", 'font': {'size': 14, 'color': '#64748B'}},
            number={'suffix': "%", 'font': {'size': 28, 'color': bar_color}},
            gauge={
                'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': '#CBD5E1'},
                'bar': {'color': bar_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#E5E7EB",
                'steps': [
                    {'range': [0, 5], 'color': '#FEE2E2'},
                    {'range': [5, 10], 'color': '#FEF3C7'},
                    {'range': [10, max_value], 'color': '#D1FAE5'}
                ],
                'threshold': {
                    'line': {'color': Config.COLORS['critical'], 'width': 2},
                    'thickness': 0.75,
                    'value': Config.RESERVE_WARNING_THRESHOLD
                }
            }
        ))
        fig.update_layout(
            height=180,
            margin=dict(l=20, r=20, t=40, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'family': 'Arial'}
        )
        return fig


# ============================================================================
# Charts 클래스
# ============================================================================

class Charts:
    """차트 생성 클래스"""

    # -------------------------------------------------------------------------
    # 실시간 수급 차트
    # -------------------------------------------------------------------------

    @staticmethod
    def create_supply_status_chart(
        df: pd.DataFrame,
        supply_capacity: float = Config.JEJU_SUPPLY_CAPACITY,
        hours: int = 24
    ) -> go.Figure:
        """실시간 수급 현황 차트 (EPSIS 스타일)"""
        recent = df.tail(hours).copy()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 공급능력 (수평선)
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=[supply_capacity] * len(recent),
                mode='lines',
                name='공급능력',
                line=dict(color=Config.COLORS['supply'], width=2, dash='dash')
            ),
            secondary_y=False
        )

        # 현재수요
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=recent['power_demand'],
                mode='lines',
                name='전력수요',
                line=dict(color=Config.COLORS['demand'], width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 0, 0, 0.1)'
            ),
            secondary_y=False
        )

        # 예비력 계산 및 표시
        reserve = supply_capacity - recent['power_demand']
        fig.add_trace(
            go.Scatter(
                x=recent.index,
                y=reserve,
                mode='lines',
                name='예비력',
                line=dict(color=Config.COLORS['reserve'], width=2, dash='dot')
            ),
            secondary_y=False
        )

        # 기온 (보조 Y축)
        if '기온' in recent.columns:
            fig.add_trace(
                go.Scatter(
                    x=recent.index,
                    y=recent['기온'],
                    mode='lines',
                    name='기온',
                    line=dict(color='#9CA3AF', width=1)
                ),
                secondary_y=True
            )

        fig.update_layout(
            title="실시간 전력 수급 현황",
            template="plotly_white",
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )

        fig.update_xaxes(title_text="시간")
        fig.update_yaxes(title_text="전력 (MW)", secondary_y=False)
        fig.update_yaxes(title_text="기온 (°C)", secondary_y=True)

        return fig

    # -------------------------------------------------------------------------
    # 예측 차트
    # -------------------------------------------------------------------------

    @staticmethod
    def create_prediction_chart(
        historical_df: pd.DataFrame,
        prediction_value: float,
        prediction_time: datetime,
        model_used: str
    ) -> go.Figure:
        """실시간 예측 차트"""
        fig = go.Figure()

        # 최근 48시간 실제 데이터
        recent = historical_df.tail(48)

        fig.add_trace(go.Scatter(
            x=recent.index,
            y=recent['power_demand'],
            mode='lines',
            name='실제 수요',
            line=dict(color=Config.COLORS['actual'], width=2)
        ))

        # 예측 포인트
        fig.add_trace(go.Scatter(
            x=[prediction_time],
            y=[prediction_value],
            mode='markers+text',
            name=f'예측 ({model_used})',
            marker=dict(color=Config.COLORS['demand'], size=15, symbol='star'),
            text=[f'{prediction_value:.0f} MW'],
            textposition='top center',
            textfont=dict(size=14, color=Config.COLORS['demand'])
        ))

        # 예측선 연결
        last_actual = recent['power_demand'].iloc[-1]
        last_time = recent.index[-1]

        fig.add_trace(go.Scatter(
            x=[last_time, prediction_time],
            y=[last_actual, prediction_value],
            mode='lines',
            name='예측 추이',
            line=dict(color=Config.COLORS['prediction'], width=2, dash='dash')
        ))

        fig.update_layout(
            title="전력 수요 예측",
            xaxis_title="시간",
            yaxis_title="전력 수요 (MW)",
            height=400,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_multi_horizon_chart(
        historical_df: pd.DataFrame,
        predictions: Dict[str, Dict]
    ) -> go.Figure:
        """다중 시간대 예측 차트"""
        fig = go.Figure()

        # 과거 데이터
        recent = historical_df.tail(72)
        fig.add_trace(go.Scatter(
            x=recent.index,
            y=recent['power_demand'],
            mode='lines',
            name='실제 수요',
            line=dict(color=Config.COLORS['actual'], width=2)
        ))

        # 예측 시간대별 색상
        horizon_colors = {
            '1h': '#3B82F6',
            '6h': '#8B5CF6',
            '12h': '#EC4899',
            '24h': '#F59E0B',
            '48h': '#EF4444'
        }

        for horizon, data in predictions.items():
            if data is None:
                continue

            color = horizon_colors.get(horizon, '#6B7280')
            pred_time = pd.to_datetime(data.get('timestamp'))
            pred_value = data.get('prediction')

            # 예측 포인트
            fig.add_trace(go.Scatter(
                x=[pred_time],
                y=[pred_value],
                mode='markers+text',
                name=f'{horizon} 예측',
                marker=dict(color=color, size=12, symbol='diamond'),
                text=[f'{pred_value:.0f}'],
                textposition='top center'
            ))

        fig.update_layout(
            title="다중 시간대 전력 수요 예측",
            xaxis_title="시간",
            yaxis_title="전력 수요 (MW)",
            height=450,
            template="plotly_white",
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_batch_prediction_chart(
        predictions: List[Dict],
        historical_df: pd.DataFrame
    ) -> go.Figure:
        """배치 예측 차트"""
        fig = go.Figure()

        # 최근 실제 데이터
        recent = historical_df.tail(72)
        fig.add_trace(go.Scatter(
            x=recent.index,
            y=recent['power_demand'],
            mode='lines',
            name='실제 수요',
            line=dict(color=Config.COLORS['actual'], width=2)
        ))

        # 예측 데이터
        if predictions:
            pred_times = [pd.to_datetime(p['timestamp']) for p in predictions]
            pred_values = [p['prediction'] for p in predictions]

            fig.add_trace(go.Scatter(
                x=pred_times,
                y=pred_values,
                mode='lines+markers',
                name='예측',
                line=dict(color=Config.COLORS['prediction'], width=2),
                marker=dict(size=4)
            ))

        fig.update_layout(
            title="배치 예측 결과",
            xaxis_title="시간",
            yaxis_title="전력 수요 (MW)",
            height=400,
            template="plotly_white",
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_scenario_comparison_chart(scenarios_results: Dict[str, Dict]) -> go.Figure:
        """시나리오 비교 차트"""
        fig = go.Figure()

        for scenario_name, result in scenarios_results.items():
            if result and 'predictions' in result:
                config = Config.SCENARIOS.get(scenario_name, {})
                display_name = config.get('name', scenario_name)
                color = Config.SCENARIO_COLORS.get(scenario_name, '#64748B')

                pred_times = [pd.to_datetime(p['timestamp']) for p in result['predictions']]
                pred_values = [p['prediction'] for p in result['predictions']]

                fig.add_trace(go.Scatter(
                    x=pred_times,
                    y=pred_values,
                    mode='lines',
                    name=display_name,
                    line=dict(color=color, width=2)
                ))

        fig.update_layout(
            title="시나리오별 예측 비교",
            xaxis_title="시간",
            yaxis_title="전력 수요 (MW)",
            height=450,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_scenario_heatmap(scenarios_data: Dict[str, List[Dict]]) -> go.Figure:
        """시나리오별 시간대 수요 히트맵"""
        hours = list(range(24))
        scenario_names = list(scenarios_data.keys())

        z_data = []
        y_labels = []

        for scenario in scenario_names:
            if scenarios_data[scenario] and 'predictions' in scenarios_data[scenario]:
                hourly_data = {}
                for pred in scenarios_data[scenario]['predictions']:
                    hour = pd.to_datetime(pred['timestamp']).hour
                    hourly_data[hour] = pred['prediction']

                row = [hourly_data.get(h, 0) for h in hours]
                z_data.append(row)
                y_labels.append(Config.SCENARIOS.get(scenario, {}).get('name', scenario))

        if not z_data:
            return go.Figure()

        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[f'{h:02d}:00' for h in hours],
            y=y_labels,
            colorscale='RdYlGn_r',
            colorbar=dict(title='수요 (MW)')
        ))

        fig.update_layout(
            title="시나리오별 시간대 수요 히트맵",
            xaxis_title="시간",
            yaxis_title="시나리오",
            height=300
        )

        return fig

    # -------------------------------------------------------------------------
    # 신재생에너지 차트
    # -------------------------------------------------------------------------

    @staticmethod
    def create_renewable_chart(predictions: List[Dict], energy_type: str = "both") -> go.Figure:
        """신재생에너지 발전량 예측 차트"""
        fig = go.Figure()

        if not predictions:
            return fig

        timestamps = [pd.to_datetime(p.get('datetime', p.get('timestamp'))) for p in predictions]

        # 데이터 추출
        solar_vals = []
        wind_vals = []

        for p in predictions:
            preds = p.get('predictions', {})
            if isinstance(preds, dict):
                solar_vals.append(preds.get('solar', 0) or 0)
                wind_vals.append(preds.get('wind', 0) or 0)
            else:
                solar_vals.append(0)
                wind_vals.append(0)

        # 태양광
        if energy_type in ["solar", "both"] and any(v > 0 for v in solar_vals):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=solar_vals,
                mode='lines+markers',
                name='태양광',
                line=dict(color=Config.COLORS['solar'], width=2),
                fill='tozeroy',
                fillcolor='rgba(245, 158, 11, 0.2)',
                marker=dict(size=4)
            ))

        # 풍력
        if energy_type in ["wind", "both"] and any(v > 0 for v in wind_vals):
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=wind_vals,
                mode='lines+markers',
                name='풍력',
                line=dict(color=Config.COLORS['wind'], width=2),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.2)',
                marker=dict(size=4)
            ))

        # 합계
        if energy_type == "both":
            total = [s + w for s, w in zip(solar_vals, wind_vals)]
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=total,
                mode='lines',
                name='합계',
                line=dict(color=Config.COLORS['renewable_total'], width=3, dash='dash')
            ))

        fig.update_layout(
            title="신재생에너지 발전량 예측",
            xaxis_title="시간",
            yaxis_title="발전량 (MW)",
            height=450,
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_renewable_pie_chart(solar_mw: float, wind_mw: float) -> go.Figure:
        """신재생에너지 구성 비율 파이 차트"""
        total = solar_mw + wind_mw
        if total == 0:
            total = 1

        fig = go.Figure(data=[go.Pie(
            labels=['태양광', '풍력'],
            values=[solar_mw, wind_mw],
            marker=dict(colors=[Config.COLORS['solar'], Config.COLORS['wind']]),
            hole=0.4,
            textinfo='label+percent',
            textfont_size=14
        )])

        fig.update_layout(
            title="신재생에너지 구성 비율",
            height=350,
            showlegend=True,
            annotations=[dict(
                text=f'{total:.0f}<br>MW',
                x=0.5, y=0.5,
                font_size=16,
                showarrow=False
            )]
        )

        return fig

    @staticmethod
    def create_energy_overview_chart(demand_mw: float, solar_mw: float, wind_mw: float) -> go.Figure:
        """통합 에너지 현황 차트"""
        renewable_total = solar_mw + wind_mw
        net_demand = demand_mw - renewable_total

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("에너지 수급 현황", "신재생 비율"),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )

        # 막대 그래프
        fig.add_trace(
            go.Bar(
                x=['전력 수요', '태양광', '풍력', '순수요'],
                y=[demand_mw, solar_mw, wind_mw, max(0, net_demand)],
                marker_color=[Config.COLORS['demand'], Config.COLORS['solar'],
                             Config.COLORS['wind'], '#64748B'],
                text=[f'{v:.0f}' for v in [demand_mw, solar_mw, wind_mw, max(0, net_demand)]],
                textposition='outside'
            ),
            row=1, col=1
        )

        # 파이 차트
        renewable_ratio = (renewable_total / demand_mw * 100) if demand_mw > 0 else 0
        fig.add_trace(
            go.Pie(
                labels=['신재생', '기타'],
                values=[renewable_total, max(0, net_demand)],
                marker=dict(colors=[Config.COLORS['renewable_total'], '#CBD5E1']),
                hole=0.4,
                textinfo='percent'
            ),
            row=1, col=2
        )

        fig.update_layout(
            height=400,
            template="plotly_white",
            showlegend=False
        )

        return fig

    # -------------------------------------------------------------------------
    # 과거 데이터 차트
    # -------------------------------------------------------------------------

    @staticmethod
    def create_hourly_pattern_chart(data: pd.DataFrame) -> go.Figure:
        """시간대별 패턴 차트"""
        df = data.copy()
        df['hour'] = df.index.hour

        hourly_avg = df.groupby('hour')['power_demand'].agg(['mean', 'std']).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=hourly_avg['hour'],
            y=hourly_avg['mean'],
            error_y=dict(type='data', array=hourly_avg['std'], visible=True),
            marker_color=Config.COLORS['prediction'],
            name='평균 수요'
        ))

        fig.update_layout(
            title="시간대별 평균 전력 수요",
            xaxis_title="시간 (0-23)",
            yaxis_title="전력 수요 (MW)",
            height=350,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def create_weekly_pattern_chart(data: pd.DataFrame) -> go.Figure:
        """요일별 패턴 차트"""
        df = data.copy()
        df['dayofweek'] = df.index.dayofweek

        days = ['월', '화', '수', '목', '금', '토', '일']
        daily_avg = df.groupby('dayofweek')['power_demand'].agg(['mean', 'std']).reset_index()

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=[days[i] for i in daily_avg['dayofweek']],
            y=daily_avg['mean'],
            error_y=dict(type='data', array=daily_avg['std'], visible=True),
            marker_color=Config.COLORS['prediction'],
            name='평균 수요'
        ))

        fig.update_layout(
            title="요일별 평균 전력 수요",
            xaxis_title="요일",
            yaxis_title="전력 수요 (MW)",
            height=350,
            template="plotly_white"
        )

        return fig


# ============================================================================
# 페이지 렌더링 함수
# ============================================================================

def render_supply_status_page(
    api_client: PowerDemandAPIClient,
    historical_df: pd.DataFrame
):
    """실시간 수급 현황 페이지 렌더링"""
    st.header("📊 실시간 수급 현황")

    # EPSIS 실시간 데이터 섹션
    if EPSIS_AVAILABLE:
        st.subheader("🔴 EPSIS 실시간 데이터")

        # EPSIS 데이터 가져오기
        with st.spinner("EPSIS 데이터 조회 중..."):
            epsis_data = DataManager.fetch_epsis_realtime()

        if epsis_data:
            # 데이터 소스 정보
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.caption(f"🕐 조회 시점: {epsis_data['fetched_at']}")
            with col_info2:
                st.caption(f"📊 데이터 건수: {epsis_data['data_count']}건")
            with col_info3:
                if st.button("🔄 새로고침", key="epsis_refresh"):
                    DataManager.fetch_epsis_realtime.clear()
                    st.rerun()

            # 전국 vs 제주 탭
            epsis_tab1, epsis_tab2, epsis_tab3 = st.tabs(["🇰🇷 전국 현황", "🏝️ 제주 추정", "📊 제주 실측"])

            with epsis_tab1:
                national = epsis_data['national']['latest']

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    fig = GaugeComponents.create_supply_gauge(
                        national['supply_capacity'],
                        max_value=120000
                    )
                    fig.update_layout(title={'text': "공급능력 (전국)"})
                    st.plotly_chart(fig, use_container_width=True, key="nat_supply")
                with col2:
                    fig = GaugeComponents.create_demand_gauge(
                        national['current_demand'],
                        national['supply_capacity'],
                        max_value=120000
                    )
                    fig.update_layout(title={'text': "현재수요 (전국)"})
                    st.plotly_chart(fig, use_container_width=True, key="nat_demand")
                with col3:
                    fig = GaugeComponents.create_reserve_gauge(
                        national['reserve_power'],
                        max_value=50000
                    )
                    fig.update_layout(title={'text': "예비력 (전국)"})
                    st.plotly_chart(fig, use_container_width=True, key="nat_reserve")
                with col4:
                    fig = GaugeComponents.create_reserve_rate_gauge(
                        national['reserve_rate']
                    )
                    fig.update_layout(title={'text': "예비율 (전국)"})
                    st.plotly_chart(fig, use_container_width=True, key="nat_rate")

                st.caption(f"📅 데이터 시점: {national['timestamp']}")

                # 전국 실시간 추이 차트
                st.markdown("---")
                st.subheader("📈 전국 실시간 수급 추이")

                national_history = epsis_data['national']['history']
                if national_history:
                    chart_data_nat = pd.DataFrame([
                        {
                            'timestamp': d['timestamp'],
                            '현재수요': d['current_demand'],
                            '공급능력': d['supply_capacity'],
                            '예비력': d['reserve_power'],
                            '예비율': d['reserve_rate'],
                        }
                        for d in national_history
                    ])
                    chart_data_nat['timestamp'] = pd.to_datetime(chart_data_nat['timestamp'])
                    chart_data_nat = chart_data_nat.sort_values('timestamp')

                    # 보조 Y축(예비율%)을 포함한 차트 생성
                    fig_nat = make_subplots(specs=[[{"secondary_y": True}]])

                    fig_nat.add_trace(go.Scatter(
                        x=chart_data_nat['timestamp'],
                        y=chart_data_nat['공급능력'],
                        mode='lines',
                        name='공급능력',
                        line=dict(color=Config.COLORS['supply'], width=3)
                    ), secondary_y=False)

                    fig_nat.add_trace(go.Scatter(
                        x=chart_data_nat['timestamp'],
                        y=chart_data_nat['현재수요'],
                        mode='lines',
                        name='현재수요',
                        line=dict(color=Config.COLORS['demand'], width=3)
                    ), secondary_y=False)

                    fig_nat.add_trace(go.Scatter(
                        x=chart_data_nat['timestamp'],
                        y=chart_data_nat['예비력'],
                        mode='lines',
                        name='예비력',
                        line=dict(color=Config.COLORS['reserve'], width=3)
                    ), secondary_y=False)

                    # 예비율(%) - 보조 Y축
                    fig_nat.add_trace(go.Scatter(
                        x=chart_data_nat['timestamp'],
                        y=chart_data_nat['예비율'],
                        mode='lines',
                        name='예비율(%)',
                        line=dict(color='#9C27B0', width=3, dash='dash')
                    ), secondary_y=True)

                    fig_nat.update_layout(
                        title="전국 전력 수급 추이 (EPSIS 실시간, 5분 간격)",
                        xaxis_title="시간",
                        height=450,
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
                    )
                    fig_nat.update_yaxes(title_text="전력 (MW)", secondary_y=False)
                    fig_nat.update_yaxes(title_text="예비율 (%)", secondary_y=True)

                    st.plotly_chart(fig_nat, use_container_width=True, key="epsis_national_trend")

                # 전국 상세 데이터
                with st.expander("📋 전국 시간별 데이터"):
                    if national_history:
                        df_nat = pd.DataFrame([
                            {
                                '시간': d['timestamp'],
                                '공급능력(MW)': d['supply_capacity'],
                                '현재수요(MW)': d['current_demand'],
                                '예비력(MW)': d['reserve_power'],
                                '예비율(%)': d['reserve_rate'],
                            }
                            for d in national_history[-48:]  # 최근 48건 (4시간)
                        ])
                        st.dataframe(df_nat.round(1), use_container_width=True, hide_index=True)

            with epsis_tab2:
                jeju = epsis_data['jeju']['latest']

                # 제주 추정치 4개 게이지
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    fig = GaugeComponents.create_supply_gauge(jeju['supply_capacity'])
                    st.plotly_chart(fig, width="stretch", key="jeju_supply")
                with col2:
                    fig = GaugeComponents.create_demand_gauge(
                        jeju['current_demand'],
                        jeju['supply_capacity']
                    )
                    st.plotly_chart(fig, width="stretch", key="jeju_demand")
                with col3:
                    fig = GaugeComponents.create_reserve_gauge(jeju['reserve_power'])
                    st.plotly_chart(fig, width="stretch", key="jeju_reserve")
                with col4:
                    fig = GaugeComponents.create_reserve_rate_gauge(jeju['reserve_rate'])
                    st.plotly_chart(fig, width="stretch", key="jeju_rate")

                # 상태 메시지 및 이용률 계산
                utilization_rate = (jeju['current_demand'] / jeju['supply_capacity'] * 100) if jeju['supply_capacity'] > 0 else 0
                status = "safe" if jeju['reserve_rate'] >= 10 else "warning" if jeju['reserve_rate'] >= 5 else "danger"
                status_text = "정상" if jeju['reserve_rate'] >= 10 else "주의" if jeju['reserve_rate'] >= 5 else "위험"
                status_class = f"status-{status}"

                st.markdown(f"""
                <div style="text-align: center; padding: 10px; background: #F8FAFC; border-radius: 8px; margin: 10px 0;">
                    <span style="font-size: 1.1rem;">제주 수급 상태 (추정): </span>
                    <span class="{status_class}" style="font-size: 1.3rem; font-weight: bold;">
                        {status_text}
                    </span>
                    <span style="color: #64748B; margin-left: 20px;">
                        이용률: {utilization_rate:.1f}%
                    </span>
                </div>
                """, unsafe_allow_html=True)

                st.info("⚠️ 제주 데이터는 전국 데이터 기반 **추정치**입니다. (계절별 비율 적용)")

            with epsis_tab3:
                # 제주 실측 데이터 (공공데이터포털)
                st.markdown("#### 📊 제주 실측 전력수급 현황")
                st.caption("데이터 출처: 공공데이터포털 (한국전력거래소_제주 전력수급현황)")

                jeju_actual = DataManager.fetch_jeju_actual_data()

                if jeju_actual:
                    # 데이터 정보
                    col_info1, col_info2, col_info3 = st.columns(3)
                    with col_info1:
                        st.caption(f"📊 총 데이터: {jeju_actual['total_records']:,}건")
                    with col_info2:
                        st.caption(f"📅 기간: {jeju_actual['date_range']['start'][:10]} ~ {jeju_actual['date_range']['end'][:10]}")
                    with col_info3:
                        if st.button("🔄 새로고침", key="jeju_actual_refresh"):
                            DataManager.fetch_jeju_actual_data.clear()
                            st.rerun()

                    latest_jeju = jeju_actual['latest']

                    # 4개 게이지
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        fig = GaugeComponents.create_supply_gauge(latest_jeju['supply_capacity'])
                        st.plotly_chart(fig, width="stretch", key="jeju_actual_supply")
                    with col2:
                        fig = GaugeComponents.create_demand_gauge(
                            latest_jeju['system_demand'],
                            latest_jeju['supply_capacity']
                        )
                        fig.update_layout(title={'text': "계통수요"})
                        st.plotly_chart(fig, width="stretch", key="jeju_actual_demand")
                    with col3:
                        fig = GaugeComponents.create_reserve_gauge(latest_jeju['supply_reserve'])
                        fig.update_layout(title={'text': "공급예비력"})
                        st.plotly_chart(fig, width="stretch", key="jeju_actual_reserve")
                    with col4:
                        fig = GaugeComponents.create_reserve_rate_gauge(latest_jeju['reserve_rate'])
                        st.plotly_chart(fig, width="stretch", key="jeju_actual_rate")

                    # 추가 메트릭
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("예측수요", f"{latest_jeju['forecast_demand']:.1f} MW")
                    with col2:
                        st.metric("운영예비력", f"{latest_jeju['operation_reserve']:.1f} MW")

                    st.caption(f"📅 데이터 시점: {latest_jeju['timestamp']}")

                    # 제주 실측 추이 차트
                    st.markdown("---")
                    st.subheader("📈 제주 실측 수급 추이 (최근 7일)")

                    jeju_actual_history = jeju_actual['history']
                    if jeju_actual_history:
                        chart_data_actual = pd.DataFrame([
                            {
                                'timestamp': d['timestamp'],
                                '계통수요': d['system_demand'],
                                '공급능력': d['supply_capacity'],
                                '공급예비력': d['supply_reserve'],
                                '예측수요': d['forecast_demand'],
                                '예비율': d['reserve_rate'],
                            }
                            for d in jeju_actual_history
                        ])
                        chart_data_actual['timestamp'] = pd.to_datetime(chart_data_actual['timestamp'])
                        chart_data_actual = chart_data_actual.sort_values('timestamp')

                        # 보조 Y축(예비율%)을 포함한 차트 생성
                        fig_actual = make_subplots(specs=[[{"secondary_y": True}]])

                        fig_actual.add_trace(go.Scatter(
                            x=chart_data_actual['timestamp'],
                            y=chart_data_actual['공급능력'],
                            mode='lines',
                            name='공급능력',
                            line=dict(color=Config.COLORS['supply'], width=3)
                        ), secondary_y=False)

                        fig_actual.add_trace(go.Scatter(
                            x=chart_data_actual['timestamp'],
                            y=chart_data_actual['계통수요'],
                            mode='lines',
                            name='계통수요',
                            line=dict(color=Config.COLORS['demand'], width=3)
                        ), secondary_y=False)

                        fig_actual.add_trace(go.Scatter(
                            x=chart_data_actual['timestamp'],
                            y=chart_data_actual['공급예비력'],
                            mode='lines',
                            name='공급예비력',
                            line=dict(color=Config.COLORS['reserve'], width=3)
                        ), secondary_y=False)

                        fig_actual.add_trace(go.Scatter(
                            x=chart_data_actual['timestamp'],
                            y=chart_data_actual['예측수요'],
                            mode='lines',
                            name='예측수요',
                            line=dict(color='#FF9800', width=2, dash='dot')
                        ), secondary_y=False)

                        # 예비율(%) - 보조 Y축
                        fig_actual.add_trace(go.Scatter(
                            x=chart_data_actual['timestamp'],
                            y=chart_data_actual['예비율'],
                            mode='lines',
                            name='예비율(%)',
                            line=dict(color='#9C27B0', width=3, dash='dash')
                        ), secondary_y=True)

                        fig_actual.update_layout(
                            title="제주 전력 수급 추이 (공공데이터포털 실측, 1시간 간격)",
                            xaxis_title="시간",
                            height=450,
                            template="plotly_white",
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
                        )
                        fig_actual.update_yaxes(title_text="전력 (MW)", secondary_y=False)
                        fig_actual.update_yaxes(title_text="예비율 (%)", secondary_y=True)

                        st.plotly_chart(fig_actual, use_container_width=True, key="jeju_actual_trend")

                    # 제주 실측 상세 데이터
                    with st.expander("📋 제주 실측 시간별 데이터"):
                        if jeju_actual_history:
                            df_jeju_actual = pd.DataFrame([
                                {
                                    '시간': d['timestamp'],
                                    '계통수요(MW)': d['system_demand'],
                                    '공급능력(MW)': d['supply_capacity'],
                                    '공급예비력(MW)': d['supply_reserve'],
                                    '예측수요(MW)': d['forecast_demand'],
                                    '운영예비력(MW)': d['operation_reserve'],
                                    '예비율(%)': d['reserve_rate'],
                                }
                                for d in jeju_actual_history[-48:]  # 최근 48건 (48시간)
                            ])
                            st.dataframe(df_jeju_actual.round(1), use_container_width=True, hide_index=True)

                    st.success("✅ 제주 실측 데이터 표시 완료 (공공데이터포털)")

                else:
                    st.warning("⚠️ 제주 실측 데이터를 로드할 수 없습니다.")
                    st.info("""
                    **제주 실측 데이터 사용 방법:**
                    1. 공공데이터포털에서 '한국전력거래소_제주 전력수급현황' 검색
                    2. ZIP 파일 다운로드 후 `data/jeju_power_supply.zip` 으로 저장
                    3. 대시보드 새로고침

                    [📥 공공데이터포털 바로가기](https://www.data.go.kr/data/15125113/fileData.do)
                    """)

            # EPSIS 실시간 추이 차트
            st.markdown("---")
            st.subheader("📈 EPSIS 실시간 수급 추이")

            jeju_history = epsis_data['jeju']['history']
            if jeju_history:
                # 데이터프레임 변환
                chart_data = pd.DataFrame([
                    {
                        'timestamp': d['timestamp'],
                        '현재수요': d['current_demand'],
                        '공급능력': d['supply_capacity'],
                        '예비력': d['reserve_power'],
                        '예비율': d['reserve_rate'],
                    }
                    for d in jeju_history
                ])
                chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
                chart_data = chart_data.sort_values('timestamp')

                # 보조 Y축(예비율%)을 포함한 차트 생성
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(go.Scatter(
                    x=chart_data['timestamp'],
                    y=chart_data['공급능력'],
                    mode='lines',
                    name='공급능력',
                    line=dict(color=Config.COLORS['supply'], width=3)
                ), secondary_y=False)

                fig.add_trace(go.Scatter(
                    x=chart_data['timestamp'],
                    y=chart_data['현재수요'],
                    mode='lines',
                    name='현재수요',
                    line=dict(color=Config.COLORS['demand'], width=3)
                ), secondary_y=False)

                fig.add_trace(go.Scatter(
                    x=chart_data['timestamp'],
                    y=chart_data['예비력'],
                    mode='lines',
                    name='예비력',
                    line=dict(color=Config.COLORS['reserve'], width=3)
                ), secondary_y=False)

                # 예비율(%) - 보조 Y축
                fig.add_trace(go.Scatter(
                    x=chart_data['timestamp'],
                    y=chart_data['예비율'],
                    mode='lines',
                    name='예비율(%)',
                    line=dict(color='#9C27B0', width=3, dash='dash')
                ), secondary_y=True)

                fig.update_layout(
                    title="제주 전력 수급 추이 (EPSIS 기반 추정, 5분 간격)",
                    xaxis_title="시간",
                    height=450,
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center")
                )
                fig.update_yaxes(title_text="전력 (MW)", secondary_y=False)
                fig.update_yaxes(title_text="예비율 (%)", secondary_y=True)

                st.plotly_chart(fig, use_container_width=True, key="epsis_trend")

            # EPSIS 상세 데이터
            with st.expander("📋 EPSIS 시간별 데이터 (제주 추정)"):
                if jeju_history:
                    df_epsis = pd.DataFrame([
                        {
                            '시간': d['timestamp'],
                            '공급능력(MW)': d['supply_capacity'],
                            '현재수요(MW)': d['current_demand'],
                            '예비력(MW)': d['reserve_power'],
                            '예비율(%)': d['reserve_rate'],
                        }
                        for d in jeju_history[-48:]  # 최근 48건 (4시간)
                    ])
                    st.dataframe(df_epsis.round(1), width="stretch", hide_index=True)

        else:
            st.warning("EPSIS 데이터를 불러올 수 없습니다. 과거 데이터를 표시합니다.")

    else:
        st.info("💡 EPSIS 크롤러가 비활성화되어 있습니다. 과거 데이터만 표시됩니다.")

    # 기존 과거 데이터 섹션
    st.markdown("---")
    st.subheader("📚 과거 데이터 기반 분석")

    if historical_df is not None and len(historical_df) > 0:
        current_demand = historical_df['power_demand'].iloc[-1]
        supply_status = DataManager.calculate_supply_status(current_demand)

        # 과거 데이터 요약
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("최근 수요", f"{current_demand:.0f} MW")
        with col2:
            avg_24h = historical_df['power_demand'].tail(24).mean()
            st.metric("24h 평균", f"{avg_24h:.0f} MW")
        with col3:
            max_24h = historical_df['power_demand'].tail(24).max()
            st.metric("24h 최대", f"{max_24h:.0f} MW")
        with col4:
            min_24h = historical_df['power_demand'].tail(24).min()
            st.metric("24h 최소", f"{min_24h:.0f} MW")

        # 실시간 수급 추이 차트
        st.subheader("24시간 수급 추이 (과거 데이터)")
        fig = Charts.create_supply_status_chart(historical_df, supply_status['supply_capacity'])
        st.plotly_chart(fig, width="stretch", key="supply_chart")

        # 데이터 그리드
        with st.expander("📋 시간별 상세 데이터 (과거)"):
            recent_24h = historical_df.tail(24)[['power_demand', '기온', '습도', '풍속']].copy()
            recent_24h.columns = ['전력수요(MW)', '기온(°C)', '습도(%)', '풍속(m/s)']
            st.dataframe(recent_24h.round(1), width="stretch")
    else:
        st.warning("과거 데이터를 불러올 수 없습니다.")


def render_prediction_page(
    api_client: PowerDemandAPIClient,
    historical_df: pd.DataFrame,
    model_type: str,
    temp_delta: float,
    humidity_delta: float
):
    """예측 시각화 페이지 렌더링"""
    st.header("🔮 예측 시각화")

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("예측 설정")

        # 예측 실행 버튼
        if st.button("🚀 예측 실행", type="primary", width="stretch"):
            with st.spinner("예측 중..."):
                # 기상 조건 수정 적용
                modified_data = DataManager.apply_weather_modification(
                    historical_df,
                    temp_delta=temp_delta,
                    humidity_delta=humidity_delta
                )

                # API 데이터 준비
                api_data = DataManager.prepare_api_data(modified_data, n_points=168)

                # API 호출
                if model_type == "conditional":
                    result = api_client.predict_conditional(api_data, mode="soft")
                else:
                    result = api_client.predict(api_data, model_type=model_type)

                if result:
                    st.session_state['last_prediction'] = result
                    st.success("예측 완료!")

        # 결과 표시
        if 'last_prediction' in st.session_state:
            result = st.session_state['last_prediction']

            st.markdown("---")
            st.subheader("예측 결과")

            st.metric(
                "예측 수요",
                f"{result['prediction']:.1f} MW",
                delta=f"{result['prediction'] - historical_df['power_demand'].iloc[-1]:.1f} MW"
            )

            st.caption(f"모델: {result.get('model_used', 'N/A')}")
            st.caption(f"처리시간: {result.get('processing_time_ms', 0):.1f}ms")

    with col1:
        if 'last_prediction' in st.session_state:
            result = st.session_state['last_prediction']
            pred_time = pd.to_datetime(result.get('timestamp', datetime.now()))

            fig = Charts.create_prediction_chart(
                historical_df,
                result['prediction'],
                pred_time,
                result.get('model_used', 'unknown')
            )
            st.plotly_chart(fig, width="stretch", key="pred_chart")
        else:
            st.info("오른쪽의 '예측 실행' 버튼을 클릭하여 예측을 시작하세요.")

            # 기본 차트 표시
            recent = historical_df.tail(72)
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=recent.index,
                y=recent['power_demand'],
                mode='lines',
                name='최근 수요',
                line=dict(color=Config.COLORS['actual'], width=2)
            ))
            fig.update_layout(
                title="최근 72시간 전력 수요",
                xaxis_title="시간",
                yaxis_title="전력 수요 (MW)",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, width="stretch", key="default_chart")

    # 시나리오 분석 섹션
    st.markdown("---")
    st.subheader("📊 시나리오 분석")

    col1, col2 = st.columns(2)

    with col1:
        compare_scenarios = st.multiselect(
            "비교할 시나리오 선택",
            options=list(Config.SCENARIOS.keys()),
            default=["normal", "heatwave_mild", "coldwave_mild"],
            format_func=lambda x: Config.SCENARIOS[x]["name"]
        )

    with col2:
        batch_step = st.slider("예측 간격 (시간)", 1, 6, 1)

    if st.button("📊 시나리오 분석 실행", type="primary"):
        if compare_scenarios:
            with st.spinner("시나리오 분석 중..."):
                scenarios_results = {}
                progress_bar = st.progress(0)

                for i, scenario in enumerate(compare_scenarios):
                    config = Config.SCENARIOS[scenario]

                    modified_data = DataManager.apply_weather_modification(
                        historical_df,
                        temp_delta=config["temp_delta"],
                        humidity_delta=config["humidity_delta"]
                    )

                    api_data = DataManager.prepare_api_data(modified_data, n_points=200)
                    result = api_client.predict_batch(api_data, model_type="demand_only", step=batch_step)

                    if result:
                        for pred in result['predictions']:
                            pred['prediction'] *= config["demand_factor"]
                        scenarios_results[scenario] = result

                    progress_bar.progress((i + 1) / len(compare_scenarios))

                st.session_state['scenarios_results'] = scenarios_results
                st.success("시나리오 분석 완료!")

    # 시나리오 결과 표시
    if 'scenarios_results' in st.session_state:
        results = st.session_state['scenarios_results']

        if results:
            col1, col2 = st.columns(2)

            with col1:
                fig = Charts.create_scenario_comparison_chart(results)
                st.plotly_chart(fig, width="stretch", key="scenario_chart")

            with col2:
                fig = Charts.create_scenario_heatmap(results)
                st.plotly_chart(fig, width="stretch", key="scenario_heatmap")

            # 통계 테이블
            st.subheader("시나리오 비교 통계")
            comparison_data = []
            for scenario_name, result in results.items():
                if result and 'predictions' in result:
                    config = Config.SCENARIOS[scenario_name]
                    predictions = [p['prediction'] for p in result['predictions']]

                    comparison_data.append({
                        "시나리오": config["name"],
                        "온도 변화": f"{config['temp_delta']:+d}°C",
                        "평균 수요": f"{np.mean(predictions):.1f} MW",
                        "피크 수요": f"{np.max(predictions):.1f} MW",
                        "최소 수요": f"{np.min(predictions):.1f} MW",
                    })

            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), width="stretch", hide_index=True)


def render_renewable_page(
    renewable_api: RenewableAPIClient,
    historical_df: pd.DataFrame
):
    """신재생에너지 페이지 렌더링"""
    st.header("🌱 신재생에너지 발전량 예측")

    # API 상태 확인
    health = renewable_api.health_check()
    api_online = health.get("status") == "healthy"

    if not api_online:
        st.warning("⚠️ 신재생에너지 API 서버에 연결할 수 없습니다.")
        st.code("cd ../kpx-demand-forecast && uvicorn api.main:app --port 8001", language="bash")
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("⚙️ 예측 설정")

        energy_type = st.selectbox(
            "에너지 타입",
            options=["both", "solar", "wind"],
            format_func=lambda x: {"both": "☀️💨 태양광 + 풍력", "solar": "☀️ 태양광만", "wind": "💨 풍력만"}.get(x, x)
        )

        forecast_hours = st.slider("예측 시간 (h)", 6, 168, 24, step=6)

        st.markdown("---")
        st.subheader("🌤️ 기상 조건")

        input_temp = st.number_input("기온 (°C)", value=15.0, min_value=-20.0, max_value=45.0)
        input_humidity = st.number_input("습도 (%)", value=60.0, min_value=0.0, max_value=100.0)
        input_wind_speed = st.number_input("풍속 (m/s)", value=5.0, min_value=0.0, max_value=50.0)

        if st.button("🚀 신재생 발전량 예측", type="primary", width="stretch"):
            with st.spinner("예측 중..."):
                base_dt = datetime.now().replace(minute=0, second=0, microsecond=0)
                weather_data = DataManager.create_sample_weather(
                    base_datetime=base_dt,
                    hours=forecast_hours,
                    temp=input_temp,
                    humidity=input_humidity,
                    wind_speed=input_wind_speed
                )

                result = renewable_api.predict_batch(weather_data, energy_type)

                if result and result.get("success"):
                    st.session_state['renewable_prediction'] = result
                    st.success(f"✅ 예측 완료! ({result.get('total_hours', 0)}시간)")

    with col1:
        if 'renewable_prediction' in st.session_state:
            result = st.session_state['renewable_prediction']
            stats = result.get('statistics', {})

            # 요약 메트릭
            st.subheader("📊 예측 결과 요약")
            metric_cols = st.columns(4)

            solar_stats = stats.get('solar', {})
            wind_stats = stats.get('wind', {})

            with metric_cols[0]:
                st.metric("☀️ 태양광 평균", f"{solar_stats.get('mean_mw', 0):.1f} MW")
            with metric_cols[1]:
                st.metric("☀️ 태양광 피크", f"{solar_stats.get('max_mw', 0):.1f} MW")
            with metric_cols[2]:
                st.metric("💨 풍력 평균", f"{wind_stats.get('mean_mw', 0):.1f} MW")
            with metric_cols[3]:
                st.metric("💨 풍력 피크", f"{wind_stats.get('max_mw', 0):.1f} MW")

            # 발전량 차트
            st.subheader("📈 시간별 발전량 예측")
            predictions = result.get('predictions', [])
            if predictions:
                fig = Charts.create_renewable_chart(predictions, energy_type)
                st.plotly_chart(fig, width="stretch", key="renewable_chart")

            # 구성 비율
            col_pie1, col_pie2 = st.columns(2)

            with col_pie1:
                total_solar = solar_stats.get('total_mwh', 0)
                total_wind = wind_stats.get('total_mwh', 0)
                fig = Charts.create_renewable_pie_chart(total_solar, total_wind)
                st.plotly_chart(fig, width="stretch", key="renewable_pie")

            with col_pie2:
                st.markdown("### 📋 상세 통계")
                stats_table = []
                if solar_stats:
                    stats_table.append({
                        "타입": "☀️ 태양광",
                        "평균 (MW)": f"{solar_stats.get('mean_mw', 0):.1f}",
                        "최대 (MW)": f"{solar_stats.get('max_mw', 0):.1f}",
                        "총량 (MWh)": f"{solar_stats.get('total_mwh', 0):.1f}",
                    })
                if wind_stats:
                    stats_table.append({
                        "타입": "💨 풍력",
                        "평균 (MW)": f"{wind_stats.get('mean_mw', 0):.1f}",
                        "최대 (MW)": f"{wind_stats.get('max_mw', 0):.1f}",
                        "총량 (MWh)": f"{wind_stats.get('total_mwh', 0):.1f}",
                    })
                if stats_table:
                    st.dataframe(pd.DataFrame(stats_table), width="stretch", hide_index=True)
        else:
            st.info("👈 오른쪽에서 기상 조건을 입력하고 '신재생 발전량 예측' 버튼을 클릭하세요.")


def render_historical_page(historical_df: pd.DataFrame, date_range: Tuple):
    """과거 데이터 페이지 렌더링"""
    st.header("📈 과거 데이터 분석")

    if historical_df is None:
        st.warning("데이터를 불러올 수 없습니다.")
        return

    # 날짜 필터링
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        mask = (historical_df.index.date >= start_date) & (historical_df.index.date <= end_date)
        filtered_data = historical_df[mask]
    else:
        filtered_data = historical_df.tail(168)

    if len(filtered_data) > 0:
        st.success(f"선택 기간: {len(filtered_data):,}개 레코드")

        # 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("평균 수요", f"{filtered_data['power_demand'].mean():.1f} MW")
        with col2:
            st.metric("최대 수요", f"{filtered_data['power_demand'].max():.1f} MW")
        with col3:
            st.metric("최소 수요", f"{filtered_data['power_demand'].min():.1f} MW")
        with col4:
            st.metric("표준편차", f"{filtered_data['power_demand'].std():.1f} MW")

        # 차트
        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=filtered_data.index,
                y=filtered_data['power_demand'],
                mode='lines',
                name='전력 수요',
                line=dict(color=Config.COLORS['prediction'])
            ))
            fig.update_layout(
                title="전력 수요 추이",
                xaxis_title="시간",
                yaxis_title="MW",
                height=400,
                template="plotly_white"
            )
            st.plotly_chart(fig, width="stretch", key="hist_trend")

        with col2:
            fig = Charts.create_hourly_pattern_chart(filtered_data)
            st.plotly_chart(fig, width="stretch", key="hist_hourly")

        # 요일별 패턴
        fig = Charts.create_weekly_pattern_chart(filtered_data)
        st.plotly_chart(fig, width="stretch", key="hist_weekly")

        # 데이터 다운로드
        st.markdown("---")
        st.subheader("📥 데이터 다운로드")

        col1, col2 = st.columns(2)

        with col1:
            csv = filtered_data.to_csv()
            st.download_button(
                label="📄 CSV 다운로드",
                data=csv,
                file_name=f"jeju_power_demand_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

        with col2:
            # Excel 다운로드
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                filtered_data.to_excel(writer, sheet_name='전력수요')

            st.download_button(
                label="📊 Excel 다운로드",
                data=buffer.getvalue(),
                file_name=f"jeju_power_demand_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        # 상세 데이터
        with st.expander("📋 상세 데이터 보기"):
            display_cols = ['power_demand']
            if '기온' in filtered_data.columns:
                display_cols.extend(['기온', '습도', '풍속'])
            st.dataframe(filtered_data[display_cols].round(2), width="stretch")
    else:
        st.warning("선택한 기간에 데이터가 없습니다.")


def render_system_info_page(
    demand_api: PowerDemandAPIClient,
    renewable_api: RenewableAPIClient,
    historical_df: pd.DataFrame
):
    """시스템 정보 페이지 렌더링"""
    st.header("⚙️ 시스템 정보")

    demand_health = demand_api.health_check()
    renewable_health = renewable_api.health_check()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("⚡ 전력 수요 예측 API")
        st.caption(f"URL: {Config.DEMAND_API_URL}")

        if demand_health.get("status") == "healthy":
            st.success("✅ 연결됨")
        else:
            st.error("❌ 오프라인")

        st.json(demand_health)

    with col2:
        st.subheader("🌱 신재생에너지 API")
        st.caption(f"URL: {Config.RENEWABLE_API_URL}")

        if renewable_health.get("status") == "healthy":
            st.success("✅ 연결됨")
        else:
            st.error("❌ 오프라인")

        st.json(renewable_health)

    with col3:
        st.subheader("📊 데이터 정보")

        if historical_df is not None:
            st.markdown(f"""
            - **총 레코드**: {len(historical_df):,}
            - **기간**: {historical_df.index.min().strftime('%Y-%m-%d')} ~ {historical_df.index.max().strftime('%Y-%m-%d')}
            - **컬럼 수**: {len(historical_df.columns)}
            - **수요 범위**: {historical_df['power_demand'].min():.1f} ~ {historical_df['power_demand'].max():.1f} MW
            """)

        if st.button("🔄 새로고침"):
            st.rerun()

    # 모델 정보
    st.markdown("---")
    st.subheader("🤖 모델 정보")

    model_info = demand_api.get_models()
    if model_info:
        for model in model_info.get('models', []):
            with st.expander(f"📦 {model['name']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"- **타입**: {model.get('type', 'N/A')}")
                    st.markdown(f"- **피처 수**: {model.get('n_features', 'N/A')}")
                with col2:
                    st.markdown(f"- **시퀀스 길이**: {model.get('seq_length', 'N/A')}")
                    st.markdown(f"- **Hidden Size**: {model.get('hidden_size', 'N/A')}")

    # API 엔드포인트
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📡 전력 수요 API 엔드포인트")
        endpoints = [
            {"Method": "GET", "Endpoint": "/health", "설명": "상태 확인"},
            {"Method": "GET", "Endpoint": "/models", "설명": "모델 정보"},
            {"Method": "POST", "Endpoint": "/predict", "설명": "단일 예측"},
            {"Method": "POST", "Endpoint": "/predict/conditional", "설명": "조건부 예측"},
            {"Method": "POST", "Endpoint": "/predict/batch", "설명": "배치 예측"},
        ]
        st.dataframe(pd.DataFrame(endpoints), width="stretch", hide_index=True)

    with col2:
        st.subheader("📡 신재생에너지 API 엔드포인트")
        endpoints = [
            {"Method": "GET", "Endpoint": "/health", "설명": "상태 확인"},
            {"Method": "GET", "Endpoint": "/models", "설명": "모델 정보"},
            {"Method": "POST", "Endpoint": "/predict", "설명": "단일 예측"},
            {"Method": "POST", "Endpoint": "/predict/batch", "설명": "배치 예측"},
        ]
        st.dataframe(pd.DataFrame(endpoints), width="stretch", hide_index=True)


# ============================================================================
# 메인 함수
# ============================================================================

def main():
    """메인 함수"""

    # API 클라이언트 초기화
    demand_api = PowerDemandAPIClient()
    renewable_api = RenewableAPIClient()

    # API 상태 확인
    demand_health = demand_api.health_check()
    renewable_health = renewable_api.health_check()

    demand_online = demand_health.get("status") == "healthy"
    renewable_online = renewable_health.get("status") == "healthy"

    # 헤더
    st.markdown('<p class="main-header">⚡ 제주도 전력 수요 예측 시스템 v1.0</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">EPSIS 스타일 실시간 수급 현황 | 예측 시각화 | 신재생에너지 분석</p>', unsafe_allow_html=True)

    # 상단 상태 표시
    header_col1, header_col2, header_col3, header_col4 = st.columns([1, 1, 1, 2])

    with header_col1:
        if demand_online:
            st.markdown('<span class="api-connected">🟢 수요예측 API</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="api-disconnected">🔴 수요예측 API</span>', unsafe_allow_html=True)

    with header_col2:
        if renewable_online:
            st.markdown('<span class="api-connected">🟢 신재생 API</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="api-disconnected">🔴 신재생 API</span>', unsafe_allow_html=True)

    with header_col3:
        st.markdown(f"<span style='color: #64748B;'>갱신: {datetime.now().strftime('%H:%M:%S')}</span>", unsafe_allow_html=True)

    with header_col4:
        pass  # 빈 공간

    st.markdown("---")

    # 사이드바
    with st.sidebar:
        st.title("⚙️ 설정")

        # 모델 선택
        st.subheader("🤖 모델 선택")
        model_type = st.selectbox(
            "예측 모델",
            options=["conditional", "demand_only", "weather_full"],
            format_func=lambda x: {
                "conditional": "조건부 앙상블 (권장)",
                "demand_only": "수요 전용",
                "weather_full": "기상 포함"
            }.get(x, x)
        )

        st.markdown("---")

        # 기상 조건 수정
        st.subheader("🌡️ 기상 조건 수정")

        temp_delta = st.slider("온도 변화 (°C)", -15.0, 15.0, 0.0, 0.5)
        humidity_delta = st.slider("습도 변화 (%)", -30.0, 30.0, 0.0, 1.0)

        # 시나리오 프리셋
        st.subheader("📊 시나리오 프리셋")
        scenario_options = {v["name"]: k for k, v in Config.SCENARIOS.items()}
        selected_preset = st.selectbox(
            "프리셋 선택",
            options=["직접 설정"] + list(scenario_options.keys())
        )

        if selected_preset != "직접 설정":
            preset_key = scenario_options[selected_preset]
            preset = Config.SCENARIOS[preset_key]
            temp_delta = float(preset["temp_delta"])
            humidity_delta = float(preset["humidity_delta"])
            st.info(f"온도: {temp_delta:+.0f}°C, 습도: {humidity_delta:+.0f}%")

        st.markdown("---")

        # 날짜 범위
        st.subheader("📅 데이터 범위")
        date_range = st.date_input(
            "기간 선택",
            value=(
                datetime.now().date() - timedelta(days=7),
                datetime.now().date()
            )
        )

        st.markdown("---")

        # 자동 갱신
        st.subheader("🔄 자동 갱신")
        auto_refresh = st.checkbox("자동 갱신 (60초)", value=False)

        if auto_refresh:
            st.caption("60초마다 자동으로 갱신됩니다.")

    # 데이터 로드
    historical_data = DataManager.load_historical_data()

    if historical_data is None or len(historical_data) == 0:
        st.error("과거 데이터를 로드할 수 없습니다. 데이터 파일을 확인해주세요.")
        st.code(f"예상 경로: {Config.DATA_PATH / 'jeju_hourly_merged.csv'}")
        return

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 실시간 수급 현황",
        "🔮 예측 시각화",
        "🌱 신재생에너지",
        "📈 과거 데이터",
        "⚙️ 시스템 정보"
    ])

    with tab1:
        render_supply_status_page(demand_api, historical_data)

    with tab2:
        render_prediction_page(demand_api, historical_data, model_type, temp_delta, humidity_delta)

    with tab3:
        render_renewable_page(renewable_api, historical_data)

    with tab4:
        render_historical_page(historical_data, date_range)

    with tab5:
        render_system_info_page(demand_api, renewable_api, historical_data)

    # 자동 갱신
    if auto_refresh:
        import time
        time.sleep(Config.REFRESH_INTERVAL)
        st.rerun()


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    main()
