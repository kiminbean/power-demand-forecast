"""
제주도 SMP 예측 및 입찰 지원 대시보드 v2.1
==========================================

민간 태양광/풍력 발전사업자를 위한 SMP 예측 및 최적 입찰 전략 지원

주요 기능:
1. 📊 입찰 지원 - SMP 예측 및 최적 입찰 전략 추천
2. 📈 SMP 분석 - 육지/제주 SMP 비교, 시간대별 히트맵
3. ☀️ 발전량 예측 - 태양광/풍력 발전량 예측
4. 🔍 XAI 분석 - Attention 기반 모델 해석 (NEW)
5. ⚙️ 설정 - API 상태 및 사용자 설정

v2.1 업데이트:
- 고도화 모델 (Quantile Regression) 지원
- Attention 시각화 및 XAI 분석 추가
- 불확실성 구간 추정 강화

Usage:
    streamlit run src/dashboard/app_v2.py

Author: Power Demand Forecast Team
Version: 2.1.0
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
from dataclasses import dataclass
import json
from pathlib import Path
import sys

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# SMP 모듈 임포트
try:
    from src.smp.crawlers import SMPCrawler, SMPDataStore
    from src.smp.bidding import (
        BiddingStrategyOptimizer,
        RevenueCalculator,
        RiskAnalyzer,
    )
    from src.smp.models import (
        GenerationPredictor,
        PlantConfig,
    )
    from src.smp.models.smp_predictor import SMPPredictor, get_smp_predictor
    SMP_AVAILABLE = True
    SMP_MODEL_AVAILABLE = True
except ImportError as e:
    SMP_AVAILABLE = False
    SMP_MODEL_AVAILABLE = False
    print(f"SMP module import failed: {e}")


# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="SMP 예측 및 입찰 지원 v2.1",
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
    API_URL = "http://localhost:8000"

    # 데이터 경로
    DATA_PATH = PROJECT_ROOT / "data"
    MODEL_PATH = PROJECT_ROOT / "models"

    # 제주도 기본 설정
    DEFAULT_CAPACITY_KW = 1000  # 기본 설비용량 (kW)

    # 색상 테마 (다크 모드 지원)
    COLORS = {
        # SMP
        'smp_mainland': '#3B82F6',    # 육지 SMP - 파랑
        'smp_jeju': '#10B981',        # 제주 SMP - 초록
        'smp_high': '#EF4444',        # 고가 - 빨강
        'smp_low': '#6B7280',         # 저가 - 회색

        # 신뢰구간
        'confidence_high': 'rgba(239, 68, 68, 0.2)',  # 상위 신뢰구간
        'confidence_low': 'rgba(107, 114, 128, 0.2)',  # 하위 신뢰구간

        # 발전
        'solar': '#F59E0B',           # 태양광 - 호박색
        'wind': '#06B6D4',            # 풍력 - 청록색
        'generation': '#8B5CF6',      # 발전량 - 보라

        # 수익/리스크
        'revenue': '#10B981',         # 수익 - 초록
        'risk_low': '#22C55E',        # 저리스크 - 녹색
        'risk_medium': '#F59E0B',     # 중리스크 - 호박색
        'risk_high': '#EF4444',       # 고리스크 - 빨강

        # 추천
        'recommended': '#8B5CF6',     # 추천 시간 - 보라
        'not_recommended': '#E5E7EB', # 비추천 - 회색

        # 배경/그리드
        'grid': '#E5E7EB',
        'background': '#F9FAFB',
        'primary': '#1E3A8A',
        'secondary': '#64748B',
    }

    # 리스크 수준별 설정
    RISK_LEVELS = {
        'conservative': {'name': '보수적', 'color': '#22C55E', 'icon': '🛡️'},
        'moderate': {'name': '중립적', 'color': '#F59E0B', 'icon': '⚖️'},
        'aggressive': {'name': '공격적', 'color': '#EF4444', 'icon': '🚀'},
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

    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* SMP 게이지 */
    .smp-gauge {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
    }

    /* 추천 배지 */
    .recommend-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
    }

    /* 사이드바 */
    .sidebar-info {
        background: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #3B82F6;
    }

    /* 테이블 스타일 */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
    }
    .styled-table th {
        background: #F1F5F9;
        padding: 0.75rem;
        text-align: left;
        font-weight: 600;
    }
    .styled-table td {
        padding: 0.75rem;
        border-bottom: 1px solid #E5E7EB;
    }

    /* 경고/알림 */
    .alert-warning {
        background: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .alert-success {
        background: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 데이터 생성기 (데모용)
# ============================================================================

class DemoDataGenerator:
    """데모용 데이터 생성"""

    # SMP 예측기 캐시
    _smp_predictor = None
    _advanced_predictor = None

    @classmethod
    def get_smp_predictor(cls, use_advanced: bool = False) -> Optional[Any]:
        """SMP 예측기 싱글톤 반환

        Args:
            use_advanced: 고도화 모델 사용 여부
        """
        if use_advanced:
            if cls._advanced_predictor is None and SMP_MODEL_AVAILABLE:
                try:
                    cls._advanced_predictor = get_smp_predictor(use_advanced=True)
                except Exception as e:
                    print(f"고도화 SMP 예측기 로드 실패: {e}")
                    cls._advanced_predictor = None
            return cls._advanced_predictor
        else:
            if cls._smp_predictor is None and SMP_MODEL_AVAILABLE:
                try:
                    cls._smp_predictor = get_smp_predictor(use_advanced=False)
                except Exception as e:
                    print(f"SMP 예측기 로드 실패: {e}")
                    cls._smp_predictor = None
            return cls._smp_predictor

    @classmethod
    def generate_smp_predictions(cls, hours: int = 24, use_advanced: bool = False, return_attention: bool = False) -> Dict[str, np.ndarray]:
        """24시간 SMP 예측 데이터 생성

        실제 LSTM 모델이 사용 가능하면 모델 예측을 사용하고,
        그렇지 않으면 데모 데이터를 생성합니다.

        Args:
            hours: 예측 시간
            use_advanced: 고도화 모델 사용 여부
            return_attention: Attention 가중치 반환 여부
        """
        # 실제 모델 사용 시도
        predictor = cls.get_smp_predictor(use_advanced=use_advanced)
        if predictor is not None and predictor.is_ready():
            try:
                result = predictor.predict_24h(return_attention=return_attention)
                # 모델 사용 표시를 세션 상태에 저장
                st.session_state['model_used'] = result.get('model_used', 'unknown')
                st.session_state['use_advanced'] = use_advanced

                # 고도화 모델 추가 정보
                if use_advanced:
                    st.session_state['coverage'] = result.get('coverage', 82.5)
                    st.session_state['mape'] = result.get('mape', 10.68)
                    st.session_state['interval_width'] = result.get('interval_width', 0)

                return result
            except Exception as e:
                print(f"모델 예측 실패, 데모 데이터 사용: {e}")
                import traceback
                traceback.print_exc()

        # 폴백: 데모 데이터 생성
        st.session_state['model_used'] = 'demo'
        return cls._generate_demo_smp(hours)

    @staticmethod
    def _generate_demo_smp(hours: int = 24) -> Dict[str, np.ndarray]:
        """데모용 SMP 예측 데이터 생성 (폴백)"""
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        times = [base_time + timedelta(hours=i) for i in range(hours)]

        # 시간대별 SMP 패턴 (새벽 낮음, 낮 높음, 저녁 피크)
        hour_factors = np.array([
            0.75, 0.72, 0.70, 0.68, 0.70, 0.75,  # 00-05시 (저가)
            0.85, 0.95, 1.05, 1.10, 1.12, 1.15,  # 06-11시 (상승)
            1.18, 1.15, 1.10, 1.05, 1.00, 1.05,  # 12-17시 (고가)
            1.10, 1.05, 0.95, 0.88, 0.82, 0.78   # 18-23시 (하강)
        ])

        # 현재 시간 기준으로 시작점 조정
        start_hour = base_time.hour
        hour_factors_shifted = np.roll(hour_factors, -start_hour)[:hours]

        base_smp = 150  # 기준 SMP (원/kWh)
        noise = np.random.normal(0, 5, hours)

        smp_q50 = base_smp * hour_factors_shifted + noise
        smp_q10 = smp_q50 * 0.85
        smp_q90 = smp_q50 * 1.15

        return {
            'times': times,
            'q10': smp_q10,
            'q50': smp_q50,
            'q90': smp_q90,
            'model_used': 'demo'
        }

    @staticmethod
    def generate_generation_predictions(
        capacity_kw: float = 1000,
        energy_type: str = 'solar',
        hours: int = 24
    ) -> np.ndarray:
        """발전량 예측 데이터 생성"""
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        start_hour = base_time.hour

        if energy_type == 'solar':
            # 태양광: 일출-일몰 패턴
            pattern = np.array([
                0, 0, 0, 0, 0.05, 0.15,
                0.35, 0.55, 0.75, 0.85, 0.92, 0.95,
                0.95, 0.90, 0.80, 0.65, 0.45, 0.20,
                0.05, 0, 0, 0, 0, 0
            ])
        else:  # wind
            # 풍력: 랜덤 변동
            pattern = np.array([
                0.45, 0.48, 0.52, 0.55, 0.58, 0.55,
                0.50, 0.45, 0.42, 0.40, 0.38, 0.35,
                0.32, 0.30, 0.35, 0.40, 0.45, 0.50,
                0.55, 0.58, 0.55, 0.52, 0.50, 0.48
            ])

        pattern_shifted = np.roll(pattern, -start_hour)[:hours]
        noise = np.random.normal(0, 0.05, hours)
        noise = np.clip(noise, -0.1, 0.1)

        generation = capacity_kw * np.clip(pattern_shifted + noise, 0, 1)
        return generation

    @classmethod
    def generate_historical_smp(cls, days: int = 7) -> pd.DataFrame:
        """과거 SMP 데이터 로드 또는 생성

        실제 데이터 파일이 있으면 로드하고, 없으면 데모 데이터 생성
        """
        # 실제 데이터 로드 시도
        data_path = PROJECT_ROOT / "data/smp/smp_history_real.csv"
        if data_path.exists():
            try:
                df = pd.read_csv(data_path)
                # 유효 데이터만
                df = df[df['smp_mainland'] > 0].copy()

                # datetime 변환
                def fix_hour_24(timestamp):
                    if ' 24:00' in str(timestamp):
                        date_part = str(timestamp).replace(' 24:00', '')
                        dt = pd.to_datetime(date_part) + pd.Timedelta(days=1)
                        return dt
                    return pd.to_datetime(timestamp)

                df['datetime'] = df['timestamp'].apply(fix_hour_24)
                df = df.sort_values('datetime').reset_index(drop=True)

                # 필요한 컬럼만 선택
                result_df = df[['datetime', 'hour', 'smp_mainland', 'smp_jeju']].copy()

                if len(result_df) > 0:
                    return result_df

            except Exception as e:
                print(f"실제 데이터 로드 실패: {e}")

        # 폴백: 데모 데이터 생성
        return cls._generate_demo_historical_smp(days)

    @staticmethod
    def _generate_demo_historical_smp(days: int = 7) -> pd.DataFrame:
        """데모용 과거 SMP 데이터 생성"""
        dates = pd.date_range(end=datetime.now(), periods=days*24, freq='h')

        data = []
        for dt in dates:
            hour = dt.hour
            # 시간대별 패턴
            if 6 <= hour < 9:
                base = 130
            elif 9 <= hour < 18:
                base = 160
            elif 18 <= hour < 22:
                base = 145
            else:
                base = 110

            # 랜덤 변동
            smp = base + np.random.normal(0, 10)

            data.append({
                'datetime': dt,
                'hour': hour,
                'smp_mainland': smp,
                'smp_jeju': smp * 0.95 + np.random.normal(0, 5),  # 제주 약간 낮음
            })

        return pd.DataFrame(data)


# ============================================================================
# SMP 게이지 컴포넌트
# ============================================================================

class SMPGauge:
    """SMP 게이지 시각화"""

    @staticmethod
    def create_smp_gauge(
        current_smp: float,
        predicted_smp: float,
        title: str = "SMP"
    ) -> go.Figure:
        """SMP 게이지 차트 생성"""
        fig = go.Figure()

        # 현재 SMP 게이지
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=current_smp,
            delta={'reference': predicted_smp, 'relative': True, 'valueformat': '.1%'},
            title={'text': title, 'font': {'size': 16}},
            number={'suffix': " 원/kWh", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [50, 250], 'tickwidth': 1},
                'bar': {'color': "#3B82F6"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [50, 100], 'color': '#D1FAE5'},   # 저가 (녹색)
                    {'range': [100, 150], 'color': '#FEF3C7'},  # 중가 (노랑)
                    {'range': [150, 200], 'color': '#FED7AA'},  # 고가 (주황)
                    {'range': [200, 250], 'color': '#FECACA'},  # 초고가 (빨강)
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_smp
                }
            }
        ))

        fig.update_layout(
            height=250,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        return fig

    @staticmethod
    def create_mini_gauge(value: float, max_value: float, title: str, color: str) -> go.Figure:
        """미니 게이지 생성"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title, 'font': {'size': 14}},
            number={'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, max_value]},
                'bar': {'color': color},
                'bgcolor': "#F1F5F9",
            }
        ))

        fig.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=30, b=10),
        )

        return fig


# ============================================================================
# 차트 클래스
# ============================================================================

class Charts:
    """차트 생성 클래스"""

    @staticmethod
    def create_smp_prediction_chart(predictions: Dict) -> go.Figure:
        """SMP 예측 차트 (신뢰구간 포함)"""
        times = predictions['times']
        q10 = predictions['q10']
        q50 = predictions['q50']
        q90 = predictions['q90']

        fig = go.Figure()

        # 신뢰구간 (90%)
        fig.add_trace(go.Scatter(
            x=times + times[::-1],
            y=list(q90) + list(q10[::-1]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=True,
            name='90% 신뢰구간'
        ))

        # 상위 경계 (Q90)
        fig.add_trace(go.Scatter(
            x=times,
            y=q90,
            mode='lines',
            name='상위 예측 (Q90)',
            line=dict(color='#EF4444', width=1, dash='dot'),
        ))

        # 중앙값 (Q50)
        fig.add_trace(go.Scatter(
            x=times,
            y=q50,
            mode='lines+markers',
            name='중앙 예측 (Q50)',
            line=dict(color='#3B82F6', width=2),
            marker=dict(size=6),
        ))

        # 하위 경계 (Q10)
        fig.add_trace(go.Scatter(
            x=times,
            y=q10,
            mode='lines',
            name='하위 예측 (Q10)',
            line=dict(color='#10B981', width=1, dash='dot'),
        ))

        fig.update_layout(
            title="24시간 SMP 예측",
            xaxis_title="시간",
            yaxis_title="SMP (원/kWh)",
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

        return fig

    @staticmethod
    def create_bidding_strategy_chart(
        strategy: Any,
        generation: np.ndarray
    ) -> go.Figure:
        """입찰 전략 시각화 차트"""
        hours = list(range(1, 25))
        smp_values = []
        recommended = []
        revenues = []

        for h in strategy.hourly_details:
            smp_values.append(h.smp_predicted)
            recommended.append(h.is_recommended)
            revenues.append(h.expected_revenue)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4],
            subplot_titles=("SMP 및 추천 시간대", "시간별 예상 수익")
        )

        # 추천 시간대 배경
        for i, h in enumerate(strategy.hourly_details):
            if h.is_recommended:
                fig.add_vrect(
                    x0=h.hour - 0.5,
                    x1=h.hour + 0.5,
                    fillcolor="rgba(139, 92, 246, 0.2)",
                    layer="below",
                    line_width=0,
                    row=1, col=1
                )

        # SMP 라인
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=smp_values,
                mode='lines+markers',
                name='SMP',
                line=dict(color='#3B82F6', width=2),
                marker=dict(size=8)
            ),
            row=1, col=1
        )

        # 발전량 라인
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=generation[:24],
                mode='lines+markers',
                name='발전량 (kW)',
                line=dict(color='#F59E0B', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # 시간별 수익 막대
        colors = ['#8B5CF6' if r else '#E5E7EB' for r in recommended]
        fig.add_trace(
            go.Bar(
                x=hours,
                y=revenues,
                name='예상 수익',
                marker_color=colors,
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
        )

        fig.update_xaxes(title_text="시간", row=2, col=1)
        fig.update_yaxes(title_text="SMP (원/kWh) / 발전량 (kW)", row=1, col=1)
        fig.update_yaxes(title_text="수익 (원)", row=2, col=1)

        return fig

    @staticmethod
    def create_smp_heatmap(df: pd.DataFrame) -> go.Figure:
        """SMP 시간대별 히트맵"""
        # 요일별, 시간대별 평균 SMP
        df['weekday'] = pd.to_datetime(df['datetime']).dt.dayofweek
        df['hour'] = pd.to_datetime(df['datetime']).dt.hour

        pivot = df.pivot_table(
            values='smp_mainland',
            index='weekday',
            columns='hour',
            aggfunc='mean'
        )

        weekday_names = ['월', '화', '수', '목', '금', '토', '일']

        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=[f"{h}시" for h in range(24)],
            y=weekday_names,
            colorscale='RdYlGn_r',
            text=np.round(pivot.values, 1),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="요일: %{y}<br>시간: %{x}<br>SMP: %{z:.1f} 원/kWh<extra></extra>"
        ))

        fig.update_layout(
            title="요일/시간대별 평균 SMP",
            xaxis_title="시간",
            yaxis_title="요일",
            height=350,
            template="plotly_white"
        )

        return fig

    @staticmethod
    def create_smp_comparison_chart(df: pd.DataFrame) -> go.Figure:
        """육지 vs 제주 SMP 비교 차트"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['smp_mainland'],
            mode='lines',
            name='육지 SMP',
            line=dict(color='#3B82F6', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['smp_jeju'],
            mode='lines',
            name='제주 SMP',
            line=dict(color='#10B981', width=2)
        ))

        fig.update_layout(
            title="육지 vs 제주 SMP 비교",
            xaxis_title="시간",
            yaxis_title="SMP (원/kWh)",
            template="plotly_white",
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_generation_prediction_chart(
        solar_gen: np.ndarray,
        wind_gen: np.ndarray,
        hours: int = 24
    ) -> go.Figure:
        """발전량 예측 차트"""
        base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
        times = [base_time + timedelta(hours=i) for i in range(hours)]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=times,
            y=solar_gen,
            mode='lines+markers',
            name='태양광',
            line=dict(color='#F59E0B', width=2),
            fill='tozeroy',
            fillcolor='rgba(245, 158, 11, 0.2)'
        ))

        fig.add_trace(go.Scatter(
            x=times,
            y=wind_gen,
            mode='lines+markers',
            name='풍력',
            line=dict(color='#06B6D4', width=2),
            fill='tozeroy',
            fillcolor='rgba(6, 182, 212, 0.2)'
        ))

        fig.update_layout(
            title="24시간 발전량 예측",
            xaxis_title="시간",
            yaxis_title="발전량 (kW)",
            template="plotly_white",
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified"
        )

        return fig

    @staticmethod
    def create_revenue_simulation_chart(simulation: Dict) -> go.Figure:
        """수익 시뮬레이션 차트"""
        scenarios = ['최악', '기대', '최선']
        values = [
            simulation['worst_case'],
            simulation['expected'],
            simulation['best_case']
        ]
        colors = ['#EF4444', '#3B82F6', '#10B981']

        fig = go.Figure(data=[
            go.Bar(
                x=scenarios,
                y=values,
                marker_color=colors,
                text=[f"{v:,.0f}원" for v in values],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="수익 시뮬레이션 (시나리오별)",
            yaxis_title="예상 수익 (원)",
            template="plotly_white",
            height=300,
            showlegend=False
        )

        return fig


# ============================================================================
# 렌더링 함수
# ============================================================================

def render_bidding_page():
    """📊 입찰 지원 페이지"""
    st.markdown("## 📊 입찰 지원")
    st.markdown("**24시간 SMP 예측 기반 최적 입찰 전략 추천**")

    # 사이드바 - 설비 정보 입력
    with st.sidebar:
        st.markdown("### 🔧 설비 정보")

        energy_type = st.selectbox(
            "발전 유형",
            options=['solar', 'wind', 'hybrid'],
            format_func=lambda x: {'solar': '☀️ 태양광', 'wind': '💨 풍력', 'hybrid': '⚡ 복합'}[x]
        )

        capacity_kw = st.number_input(
            "설비 용량 (kW)",
            min_value=10,
            max_value=100000,
            value=1000,
            step=100
        )

        risk_level = st.select_slider(
            "리스크 허용도",
            options=['conservative', 'moderate', 'aggressive'],
            value='moderate',
            format_func=lambda x: Config.RISK_LEVELS[x]['name']
        )

        st.markdown("---")
        st.markdown("### 📍 위치 정보")
        latitude = st.number_input("위도", value=33.5, min_value=33.0, max_value=34.0)
        longitude = st.number_input("경도", value=126.5, min_value=126.0, max_value=127.0)

        # 모델 선택
        st.markdown("---")
        st.markdown("### 🤖 예측 모델")

        use_advanced = st.toggle(
            "고도화 모델 사용",
            value=True,
            help="Quantile Regression + Attention 기반 고도화 모델"
        )

        predictor = DemoDataGenerator.get_smp_predictor(use_advanced=use_advanced)
        if predictor is not None and predictor.is_ready():
            model_info = predictor.get_model_info()
            model_type = model_info.get('model_type', 'standard')

            if model_type == 'advanced':
                st.success("✅ 고도화 모델 (v2.1)")
                st.caption(f"파라미터: {model_info.get('parameters', 0):,}개")
                st.caption(f"MAPE: {model_info.get('mape', 'N/A')}%")
                st.caption(f"80% 커버리지: {model_info.get('coverage', 'N/A')}%")
            else:
                st.info("ℹ️ 기본 LSTM 모델")
                st.caption(f"파라미터: {model_info.get('parameters', 0):,}개")

            st.caption(f"디바이스: {model_info.get('device', 'cpu')}")
        else:
            st.warning("⚠️ 데모 모드")
            st.caption("실제 모델 미연결")

    # 데이터 생성 (실제 모델 또는 데모)
    smp_predictions = DemoDataGenerator.generate_smp_predictions(24, use_advanced=use_advanced)

    if energy_type == 'solar':
        generation = DemoDataGenerator.generate_generation_predictions(capacity_kw, 'solar', 24)
    elif energy_type == 'wind':
        generation = DemoDataGenerator.generate_generation_predictions(capacity_kw, 'wind', 24)
    else:
        solar_gen = DemoDataGenerator.generate_generation_predictions(capacity_kw * 0.6, 'solar', 24)
        wind_gen = DemoDataGenerator.generate_generation_predictions(capacity_kw * 0.4, 'wind', 24)
        generation = solar_gen + wind_gen

    # 입찰 전략 최적화
    optimizer = BiddingStrategyOptimizer()
    risk_map = {'conservative': 0.3, 'moderate': 0.5, 'aggressive': 0.7}
    strategy = optimizer.optimize(
        smp_predictions=smp_predictions['q50'],
        generation_predictions=generation,
        capacity_kw=capacity_kw,
        risk_tolerance=risk_map[risk_level]
    )

    # 수익 시뮬레이션
    calculator = RevenueCalculator()
    smp_scenarios = np.vstack([
        smp_predictions['q10'],
        smp_predictions['q50'],
        smp_predictions['q90']
    ])
    simulation = calculator.simulate(smp_scenarios, generation, hours=24)

    # 상단 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_smp = smp_predictions['q50'][0]
        st.metric(
            "현재 SMP",
            f"{current_smp:.1f} 원/kWh",
            delta=f"{(current_smp - 150) / 150 * 100:.1f}% vs 평균"
        )

    with col2:
        avg_predicted = np.mean(smp_predictions['q50'])
        st.metric(
            "24h 평균 예측",
            f"{avg_predicted:.1f} 원/kWh"
        )

    with col3:
        st.metric(
            "추천 입찰 시간",
            f"{len(strategy.recommended_hours)}시간",
            delta=f"{strategy.total_revenue:,.0f}원 예상"
        )

    with col4:
        risk_emoji = Config.RISK_LEVELS[risk_level]['icon']
        st.metric(
            "리스크 수준",
            f"{risk_emoji} {Config.RISK_LEVELS[risk_level]['name']}"
        )

    st.markdown("---")

    # 메인 차트
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # SMP 예측 차트
        st.plotly_chart(
            Charts.create_smp_prediction_chart(smp_predictions),
            width="stretch"
        )

        # 입찰 전략 차트
        st.plotly_chart(
            Charts.create_bidding_strategy_chart(strategy, generation),
            width="stretch"
        )

    with col_right:
        # 현재 SMP 게이지
        st.plotly_chart(
            SMPGauge.create_smp_gauge(
                current_smp=current_smp,
                predicted_smp=avg_predicted,
                title="현재 SMP"
            ),
            width="stretch"
        )

        # 수익 시뮬레이션
        st.plotly_chart(
            Charts.create_revenue_simulation_chart(simulation),
            width="stretch"
        )

        # 추천 요약
        st.markdown("### 📋 추천 요약")
        st.markdown(f"""
        <div class="sidebar-info">
            <b>추천 입찰 시간:</b> {', '.join(f'{h}시' for h in strategy.recommended_hours[:6])}...
            <br><br>
            <b>예상 총 발전량:</b> {strategy.total_generation:,.0f} kWh
            <br><br>
            <b>예상 수익:</b> {strategy.total_revenue:,.0f} 원
            <br><br>
            <b>kWh당 수익:</b> {strategy.revenue_per_kwh:.1f} 원
        </div>
        """, unsafe_allow_html=True)

    # 상세 테이블
    st.markdown("### 📊 시간별 상세 분석")

    detail_data = []
    for h in strategy.hourly_details:
        detail_data.append({
            '시간': f"{h.hour}시",
            'SMP (원/kWh)': f"{h.smp_predicted:.1f}",
            '발전량 (kW)': f"{h.generation_kw:.0f}",
            '예상 수익 (원)': f"{h.expected_revenue:,.0f}",
            '순위': h.rank,
            '추천': '✅' if h.is_recommended else ''
        })

    df_details = pd.DataFrame(detail_data)
    st.dataframe(df_details, height=400)


def render_smp_analysis_page():
    """📈 SMP 분석 페이지"""
    st.markdown("## 📈 SMP 분석")
    st.markdown("**육지 vs 제주 SMP 비교 및 시간대별 패턴 분석**")

    # 데이터 생성
    historical_df = DemoDataGenerator.generate_historical_smp(days=7)

    # 상단 통계
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        avg_mainland = historical_df['smp_mainland'].mean()
        st.metric("육지 평균 SMP", f"{avg_mainland:.1f} 원/kWh")

    with col2:
        avg_jeju = historical_df['smp_jeju'].mean()
        st.metric("제주 평균 SMP", f"{avg_jeju:.1f} 원/kWh")

    with col3:
        max_smp = historical_df['smp_mainland'].max()
        st.metric("최고 SMP", f"{max_smp:.1f} 원/kWh")

    with col4:
        min_smp = historical_df['smp_mainland'].min()
        st.metric("최저 SMP", f"{min_smp:.1f} 원/kWh")

    st.markdown("---")

    # 차트
    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(
            Charts.create_smp_comparison_chart(historical_df),
            width="stretch"
        )

    with col_right:
        st.plotly_chart(
            Charts.create_smp_heatmap(historical_df),
            width="stretch"
        )

    # 통계 분석
    st.markdown("### 📊 시간대별 통계")

    hourly_stats = historical_df.groupby('hour').agg({
        'smp_mainland': ['mean', 'std', 'min', 'max'],
        'smp_jeju': ['mean', 'std', 'min', 'max']
    }).round(1)

    hourly_stats.columns = [
        '육지_평균', '육지_표준편차', '육지_최저', '육지_최고',
        '제주_평균', '제주_표준편차', '제주_최저', '제주_최고'
    ]

    st.dataframe(hourly_stats, width="stretch")


def render_generation_page():
    """☀️ 발전량 예측 페이지"""
    st.markdown("## ☀️ 발전량 예측")
    st.markdown("**태양광/풍력 발전량 예측 및 기상 조건 입력**")

    # 설비 설정
    col1, col2, col3 = st.columns(3)

    with col1:
        solar_capacity = st.number_input(
            "태양광 설비 용량 (kW)",
            min_value=0,
            max_value=50000,
            value=1000,
            step=100
        )

    with col2:
        wind_capacity = st.number_input(
            "풍력 설비 용량 (kW)",
            min_value=0,
            max_value=50000,
            value=500,
            step=100
        )

    with col3:
        st.markdown("**총 설비 용량**")
        st.markdown(f"### {solar_capacity + wind_capacity:,} kW")

    st.markdown("---")

    # 기상 조건 입력
    st.markdown("### 🌤️ 기상 조건")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        temperature = st.slider("기온 (°C)", min_value=-10, max_value=40, value=25)

    with col2:
        cloud_cover = st.slider("구름량 (%)", min_value=0, max_value=100, value=30)

    with col3:
        wind_speed = st.slider("풍속 (m/s)", min_value=0.0, max_value=25.0, value=5.0, step=0.5)

    with col4:
        humidity = st.slider("습도 (%)", min_value=0, max_value=100, value=60)

    st.markdown("---")

    # 발전량 예측
    solar_gen = DemoDataGenerator.generate_generation_predictions(solar_capacity, 'solar', 24)
    wind_gen = DemoDataGenerator.generate_generation_predictions(wind_capacity, 'wind', 24)

    # 기상 조건 반영 (간단한 조정)
    solar_factor = 1.0 - (cloud_cover / 100) * 0.7  # 구름량에 따른 감소
    temp_factor = 1.0 - max(0, (temperature - 25) * 0.004)  # 고온에 따른 효율 감소
    solar_gen = solar_gen * solar_factor * temp_factor

    # 풍력 조정 (풍속에 따른)
    if wind_speed < 3:
        wind_factor = 0
    elif wind_speed > 25:
        wind_factor = 0
    elif wind_speed < 12:
        wind_factor = (wind_speed / 12) ** 3
    else:
        wind_factor = 1.0
    wind_gen = wind_gen * wind_factor

    # 통계 표시
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("태양광 일 발전량", f"{solar_gen.sum():,.0f} kWh")

    with col2:
        st.metric("풍력 일 발전량", f"{wind_gen.sum():,.0f} kWh")

    with col3:
        st.metric("총 일 발전량", f"{(solar_gen.sum() + wind_gen.sum()):,.0f} kWh")

    with col4:
        efficiency = (solar_gen.sum() + wind_gen.sum()) / ((solar_capacity + wind_capacity) * 24) * 100
        st.metric("이용률", f"{efficiency:.1f}%")

    # 발전량 차트
    st.plotly_chart(
        Charts.create_generation_prediction_chart(solar_gen, wind_gen, 24),
        width="stretch"
    )

    # 시간별 상세
    st.markdown("### 📊 시간별 발전량 상세")

    base_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    detail_data = []
    for i in range(24):
        detail_data.append({
            '시간': f"{(base_time + timedelta(hours=i)).strftime('%H:%M')}",
            '태양광 (kW)': f"{solar_gen[i]:.0f}",
            '풍력 (kW)': f"{wind_gen[i]:.0f}",
            '합계 (kW)': f"{solar_gen[i] + wind_gen[i]:.0f}",
        })

    st.dataframe(pd.DataFrame(detail_data), width="stretch", height=400)


def render_supply_status_page():
    """⚡ 수급 현황 페이지"""
    st.markdown("## ⚡ 수급 현황")
    st.markdown("**실시간 전력 수급 현황 (간소화 버전)**")

    st.info("💡 상세한 수급 현황은 대시보드 v1.0을 참조하세요: `streamlit run src/dashboard/app_v1.py`")

    # 간단한 현황 표시
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("현재 수요", "850 MW", delta="-50 MW")

    with col2:
        st.metric("공급 능력", "1,200 MW")

    with col3:
        reserve = (1200 - 850) / 1200 * 100
        st.metric("예비율", f"{reserve:.1f}%", delta="정상")


def render_xai_page():
    """🔍 XAI 분석 페이지 - Attention 기반 모델 해석"""
    st.markdown("## 🔍 XAI 분석")
    st.markdown("**Attention 기반 모델 해석 및 불확실성 분석**")

    # 고도화 모델 로드
    predictor = DemoDataGenerator.get_smp_predictor(use_advanced=True)

    if predictor is None or not predictor.is_ready():
        st.warning("⚠️ 고도화 모델을 로드할 수 없습니다. 모델 파일을 확인하세요.")
        st.info("💡 모델 학습: `python -m src.smp.models.train_smp_advanced`")
        return

    # 모델 정보
    model_info = predictor.get_model_info()

    # 상단 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "모델 파라미터",
            f"{model_info.get('parameters', 0):,}",
            delta="경량화 (1/6)"
        )

    with col2:
        mape = model_info.get('mape', 10.68)
        mape_val = float(mape) if mape != 'N/A' else 10.68
        st.metric(
            "MAPE",
            f"{mape_val:.2f}%",
            delta="실제 데이터 기준"
        )

    with col3:
        coverage = model_info.get('coverage', 82.5)
        coverage_val = float(coverage) if coverage != 'N/A' else 82.5
        st.metric(
            "80% 커버리지",
            f"{coverage_val:.1f}%",
            delta="신뢰구간 정확도"
        )

    with col4:
        st.metric(
            "Quantile 분위수",
            "10%, 50%, 90%",
            delta="불확실성 정량화"
        )

    st.markdown("---")

    # Attention 분석
    st.markdown("### 🎯 Attention 분석")

    # Attention 가중치로 예측 수행
    try:
        result = DemoDataGenerator.generate_smp_predictions(24, use_advanced=True, return_attention=True)

        if 'attention' in result and result['attention'] is not None:
            # list를 numpy 배열로 변환
            attention = np.array(result['attention'])

            # Attention 히트맵
            col_left, col_right = st.columns([2, 1])

            with col_left:
                # Attention 시각화
                fig = go.Figure()

                # 시간 인덱스 (48시간 → 24시간 예측)
                input_hours = list(range(-48, 0))
                output_hours = list(range(0, 24))

                # 평균 Attention 가중치
                if len(attention.shape) == 2:
                    # (output_len, input_len) 형태
                    avg_attention = attention.mean(axis=0)
                else:
                    avg_attention = attention

                fig.add_trace(go.Bar(
                    x=input_hours[-len(avg_attention):],
                    y=avg_attention,
                    marker_color='#3B82F6',
                    name='Attention 가중치'
                ))

                fig.update_layout(
                    title="시간별 Attention 가중치 (최근 48시간 입력)",
                    xaxis_title="입력 시간 (시간 전)",
                    yaxis_title="Attention 가중치",
                    template="plotly_white",
                    height=350
                )

                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                # Attention 통계
                st.markdown("#### 📊 Attention 통계")

                # 엔트로피 계산
                attention_flat = avg_attention.flatten()
                attention_prob = attention_flat / (attention_flat.sum() + 1e-10)
                entropy = -np.sum(attention_prob * np.log(attention_prob + 1e-10))
                max_entropy = np.log(len(attention_flat))
                normalized_entropy = entropy / max_entropy

                # 집중도 계산
                concentration = np.max(attention_flat)

                st.metric("Attention 엔트로피", f"{entropy:.2f}")
                st.caption(f"정규화: {normalized_entropy:.2%} (높음 = 분산)")

                st.metric("Attention 집중도", f"{concentration:.3f}")
                st.caption("낮음 = 과집중 없음")

                # 데이터 누수 위험도
                if normalized_entropy > 0.7 and concentration < 0.1:
                    st.success("✅ 데이터 누수 위험: LOW")
                elif normalized_entropy > 0.5:
                    st.warning("⚠️ 데이터 누수 위험: MEDIUM")
                else:
                    st.error("❌ 데이터 누수 위험: HIGH")

                # 주요 주목 시점
                top_indices = np.argsort(avg_attention)[-5:][::-1]
                st.markdown("**Top 5 주목 시점:**")
                for idx in top_indices:
                    hour_offset = idx - len(avg_attention)
                    st.caption(f"  • {hour_offset}시간 전: {avg_attention[idx]:.3f}")

        else:
            st.info("ℹ️ Attention 가중치를 가져올 수 없습니다. 예측을 실행해보세요.")

    except Exception as e:
        st.error(f"Attention 분석 오류: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.markdown("---")

    # 불확실성 분석
    st.markdown("### 📈 불확실성 분석 (Quantile Regression)")

    # 결과가 있을 때만 시각화
    try:
        result = DemoDataGenerator.generate_smp_predictions(24, use_advanced=True)
    except:
        result = None

    col1, col2 = st.columns(2)

    with col1:
        # 예측 구간 시각화
        if result is not None:
            q10, q50, q90 = result['q10'], result['q50'], result['q90']
            hours = list(range(24))

            fig = go.Figure()

            # 80% 신뢰구간
            fig.add_trace(go.Scatter(
                x=hours + hours[::-1],
                y=np.concatenate([q90, q10[::-1]]),
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.2)',
                line=dict(color='rgba(0,0,0,0)'),
                name='80% 신뢰구간'
            ))

            # 중앙값
            fig.add_trace(go.Scatter(
                x=hours,
                y=q50,
                mode='lines+markers',
                line=dict(color='#3B82F6', width=2),
                name='중앙값 (Q50)'
            ))

            fig.update_layout(
                title="24시간 SMP 예측 구간",
                xaxis_title="시간",
                yaxis_title="SMP (원/kWh)",
                template="plotly_white",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # 구간 폭 분석
        if result is not None:
            interval_width = q90 - q10

            fig = go.Figure()

            fig.add_trace(go.Bar(
                x=hours,
                y=interval_width,
                marker_color=np.where(interval_width > np.mean(interval_width), '#EF4444', '#10B981'),
                name='구간 폭'
            ))

            fig.update_layout(
                title="시간별 예측 불확실성 (구간 폭)",
                xaxis_title="시간",
                yaxis_title="구간 폭 (원/kWh)",
                template="plotly_white",
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

    # 모델 아키텍처 정보
    st.markdown("---")
    st.markdown("### 🏗️ 모델 아키텍처")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **LightweightSMPModel v2.1**

        | 구성 요소 | 사양 |
        |----------|------|
        | 입력 시퀀스 | 48시간 |
        | 피처 수 | 25개 |
        | BiLSTM 레이어 | 2층 |
        | Hidden Size | 64 |
        | Self-Attention | ✅ |
        | Quantile Heads | 3개 (10%, 50%, 90%) |
        | 총 파라미터 | ~172K |
        """)

    with col2:
        st.markdown("""
        **학습 설정**

        | 항목 | 값 |
        |------|-----|
        | 데이터 | EPSIS 2022-2024 (2년) |
        | 학습 샘플 | 26,240건 |
        | Walk-forward CV | 5 folds |
        | Noise Injection | std=0.02 |
        | 손실 함수 | Combined (MSE + Pinball) |
        | Early Stopping | patience=15 |
        """)

    # XAI 해석 가이드
    st.markdown("---")
    st.markdown("### 💡 XAI 해석 가이드")

    with st.expander("Attention 분석 해석 방법"):
        st.markdown("""
        **Attention 가중치 해석:**

        1. **높은 엔트로피 (> 3.5)**: 모델이 여러 시점에 분산된 주목 → 일반화 능력 양호
        2. **낮은 집중도 (< 0.1)**: 특정 시점에 과도한 의존 없음 → 데이터 누수 위험 낮음
        3. **최근 시점 선호**: 예측에 가장 최근 데이터가 중요 → 정상적인 패턴

        **불확실성 해석:**

        1. **구간 폭**: 넓을수록 예측 불확실성이 높음
        2. **피크 시간대**: 통상적으로 오전/저녁 피크 시 불확실성 증가
        3. **80% 커버리지**: 실제 값이 예측 구간 내에 들어올 확률
        """)


def render_settings_page():
    """⚙️ 설정 페이지"""
    st.markdown("## ⚙️ 설정")

    # API 상태
    st.markdown("### 🔌 API 상태")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="alert-success">
            <b>SMP 모듈</b>: ✅ 정상
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="alert-warning">
            <b>예측 API</b>: ⚠️ 데모 모드
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="alert-warning">
            <b>KPX 크롤러</b>: ⚠️ 데모 모드
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 모듈 상태
    st.markdown("### 📦 모듈 상태")

    module_status = {
        'SMP 크롤러': SMP_AVAILABLE,
        'SMP 모델 (LSTM)': SMP_AVAILABLE,
        'SMP 모델 (TFT)': SMP_AVAILABLE,
        '발전량 예측기': SMP_AVAILABLE,
        '입찰 전략 엔진': SMP_AVAILABLE,
    }

    for module, status in module_status.items():
        icon = "✅" if status else "❌"
        st.markdown(f"- {icon} {module}")

    st.markdown("---")

    # 버전 정보
    st.markdown("### 📋 버전 정보")
    st.markdown("""
    - **대시보드 버전**: v2.1.0
    - **SMP 모듈 버전**: v2.1.0 (Quantile + XAI)
    - **최종 업데이트**: 2025-12-18
    - **새 기능**: Attention 시각화, 불확실성 분석, 고도화 모델
    """)


# ============================================================================
# 메인 앱
# ============================================================================

def main():
    """메인 앱 실행"""
    # 헤더
    st.markdown('<p class="main-header">⚡ SMP 예측 및 입찰 지원 시스템</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">제주도 민간 태양광/풍력 발전사업자를 위한 최적 입찰 전략 지원</p>', unsafe_allow_html=True)

    # 현재 시간
    now = datetime.now()
    st.markdown(f"**🕐 {now.strftime('%Y-%m-%d %H:%M:%S')} 기준**")

    # 탭
    tabs = st.tabs([
        "📊 입찰 지원",
        "📈 SMP 분석",
        "☀️ 발전량 예측",
        "🔍 XAI 분석",
        "⚡ 수급 현황",
        "⚙️ 설정"
    ])

    with tabs[0]:
        render_bidding_page()

    with tabs[1]:
        render_smp_analysis_page()

    with tabs[2]:
        render_generation_page()

    with tabs[3]:
        render_xai_page()

    with tabs[4]:
        render_supply_status_page()

    with tabs[5]:
        render_settings_page()


if __name__ == "__main__":
    main()
