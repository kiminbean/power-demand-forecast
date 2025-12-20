"""
eXeco 대시보드 - 제주 전력 지도 v4.2
=====================================

Figma 디자인 100% 반영 (eXeco_main node 3316:358)

Usage:
    streamlit run src/dashboard/app_execo.py --server.port 8507

Author: Power Demand Forecast Team
Version: 4.2.0 (eXeco Design - Figma Exact Match)
Date: 2025-12
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
import plotly.graph_objects as go

# 프로젝트 루트
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 데이터 경로
PLANTS_CSV = PROJECT_ROOT / "data" / "jeju_plants" / "jeju_power_plants.csv"

# ============================================================================
# 페이지 설정
# ============================================================================
st.set_page_config(
    page_title="eXeco | 제주 전력 지도",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# Figma 정확 CSS (node 3316:358 기반) - 100% 매칭
# ============================================================================
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');

    /* Streamlit 기본 요소 숨기기 */
    #MainMenu, footer, .stDeployButton {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .block-container {padding: 0 !important; max-width: 100% !important;}
    .stApp {background: #ffffff !important;}

    /* 전체 컨테이너 - Figma 1920x1080 */
    .main-container {
        background: #ffffff;
        width: 100%;
        min-height: 100vh;
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 헤더 - Frame 22 (3316:359) - 1920x80 */
    .execo-header {
        background: #04265e;
        height: 80px;
        padding: 20px 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 0.4px solid #d8d8d8;
    }

    .execo-logo {
        color: #ffffff;
        font-size: 32px;
        font-weight: bold;
        font-style: italic;
        letter-spacing: -1px;
    }

    .menu-icon {
        color: #ffffff;
        font-size: 24px;
        cursor: pointer;
        display: grid;
        grid-template-columns: repeat(2, 8px);
        grid-template-rows: repeat(2, 8px);
        gap: 4px;
    }

    .menu-dot {
        width: 8px;
        height: 8px;
        background: #ffffff;
        border-radius: 2px;
    }

    /* 메인 콘텐츠 영역 - Frame 97 (3316:362) */
    .content-area {
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 24px;
    }

    /* KPI 카드 행 - Frame 88 (3316:363) */
    .kpi-row {
        display: flex;
        gap: 24px;
        width: 100%;
    }

    /* 개별 KPI 카드 - 355.2x152 각각 */
    .kpi-card {
        flex: 1;
        background: #f8f8f8;
        border-radius: 8px;
        padding: 32px 24px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 14px;
        min-height: 152px;
    }

    .kpi-title {
        font-size: 20px;
        font-weight: 500;
        color: #000000;
        letter-spacing: -0.8px;
        line-height: normal;
    }

    .kpi-content {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 14px;
        padding: 0 14px;
        width: 100%;
    }

    .kpi-value-group {
        display: flex;
        align-items: flex-end;
        gap: 4px;
    }

    .kpi-value {
        font-size: 32px;
        font-weight: bold;
        color: #000000;
        line-height: 50px;
        letter-spacing: -1.28px;
    }

    .kpi-unit {
        font-size: 20px;
        font-weight: bold;
        color: #000000;
        line-height: 36px;
        letter-spacing: -0.8px;
    }

    .kpi-badge {
        background: #ffffff;
        padding: 4px 14px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: normal;
        color: #272727;
        line-height: 30px;
        letter-spacing: -0.72px;
        white-space: nowrap;
    }

    /* 색상 변형 */
    .value-blue { color: #0048ff !important; }
    .badge-blue {
        background: rgba(0,72,255,0.1) !important;
        color: #0048ff !important;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .value-red { color: #ff1d1d !important; }
    .badge-red { background: #ffeaea !important; color: #ff1d1d !important; }
    .badge-green { background: rgba(0,197,21,0.1) !important; color: #00c515 !important; }

    /* 메인 콘텐츠 행 - Frame 98 (3316:415) */
    .main-row {
        display: flex;
        gap: 24px;
        flex: 1;
        min-height: 722px;
    }

    /* 왼쪽 차트 영역 - Frame 95 (3316:416) - 1300px */
    .chart-section {
        width: 70%;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .section-title {
        font-size: 24px;
        font-weight: bold;
        color: #000000;
        letter-spacing: -0.96px;
    }

    .section-subtitle {
        font-size: 16px;
        font-weight: normal;
        color: #000000;
        letter-spacing: -0.64px;
        margin-left: 10px;
    }

    /* 차트 컨테이너 - Frame 65 (3316:425) */
    .chart-container {
        background: #f8f8f8;
        border-radius: 14px;
        padding: 24px;
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 24px;
    }

    .chart-inner {
        background: #ffffff;
        border-radius: 14px;
        padding: 24px;
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }

    /* 범례 - Frame 171 (3316:433) */
    .legend-row {
        display: flex;
        gap: 10px;
        justify-content: flex-end;
        flex-wrap: wrap;
        align-items: center;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0;
    }

    .legend-icon {
        width: 32px;
        height: 17.5px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .legend-area {
        width: 24px;
        height: 9.5px;
        border-radius: 2px;
    }

    .legend-line {
        width: 24px;
        height: 2px;
    }

    .legend-line-dashed {
        width: 24px;
        height: 2px;
        background: repeating-linear-gradient(
            to right,
            currentColor,
            currentColor 4px,
            transparent 4px,
            transparent 8px
        );
    }

    .legend-text {
        font-size: 10px;
        font-weight: 500;
        color: #000000;
        letter-spacing: -0.4px;
    }

    /* 오른쪽 영역 - Frame 120 (3316:553) - 548px */
    .right-section {
        width: 30%;
        display: flex;
        flex-direction: column;
        gap: 24px;
    }

    /* 히트맵 카드 - Frame 96 (3316:555) */
    .heatmap-card {
        background: #f8f8f8;
        border-radius: 14px;
        padding: 24px;
    }

    .heatmap-header {
        display: flex;
        align-items: center;
        height: 52px;
        margin-bottom: 24px;
    }

    .heatmap-title {
        font-size: 24px;
        font-weight: bold;
        color: #000000;
        letter-spacing: -0.96px;
    }

    .heatmap-content {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
    }

    /* 범례 박스 - Frame 105 (3316:559) */
    .legend-box {
        background: #ffffff;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.14);
    }

    .legend-box-title {
        font-size: 14px;
        font-weight: bold;
        color: #000000;
        letter-spacing: -0.56px;
        margin-bottom: 14px;
        text-align: center;
    }

    .legend-box-item {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 8px;
    }

    .legend-box-item:last-child {
        margin-bottom: 0;
    }

    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 100px;
    }

    .dot-solar { background: #ff4a4a; }
    .dot-wind { background: #4a89ff; }
    .dot-ess { background: #ffbd00; }

    .legend-box-text {
        font-size: 14px;
        font-weight: normal;
        color: #000000;
        letter-spacing: -0.56px;
    }

    /* 제주 지도 컨테이너 */
    .jeju-map-container {
        position: relative;
        width: 385px;
        height: 255px;
    }

    .jeju-map-img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        mix-blend-mode: darken;
    }

    .map-marker {
        position: absolute;
        border-radius: 100px;
    }

    /* 발전소 현황 - Frame 77 (3316:584) */
    .plant-card {
        background: #f8f8f8;
        border-radius: 8px;
        padding: 24px;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .plant-title {
        font-size: 20px;
        font-weight: bold;
        color: #000000;
        letter-spacing: -0.8px;
        margin-bottom: 14px;
    }

    .plant-list {
        background: #ffffff;
        border-radius: 14px;
        padding: 14px 24px;
    }

    .plant-row {
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 8px;
    }

    .plant-row:last-child {
        margin-bottom: 0;
    }

    .plant-type {
        width: 54px;
        font-size: 18px;
        font-weight: normal;
        color: #000000;
        letter-spacing: -0.72px;
        line-height: 28px;
    }

    .plant-info {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .plant-count {
        font-size: 18px;
        font-weight: bold;
        color: #232323;
        letter-spacing: -0.72px;
        line-height: 34px;
    }

    .plant-gen {
        font-size: 14px;
        font-weight: normal;
        color: #232323;
        letter-spacing: -0.56px;
        line-height: 30px;
    }

    /* 기상 정보 */
    .weather-section {
        margin-top: 14px;
    }

    .weather-title {
        font-size: 14px;
        font-weight: normal;
        color: #000000;
        letter-spacing: -0.56px;
        line-height: 28px;
        width: 54px;
    }

    .weather-info {
        display: flex;
        gap: 14px;
        font-size: 14px;
        font-weight: 500;
        color: #232323;
        letter-spacing: -0.56px;
        line-height: 28px;
    }

    /* 푸터 - Frame 74 (3316:615) */
    .footer {
        text-align: center;
        padding: 10px;
        font-size: 12px;
        font-weight: normal;
        color: #232323;
        letter-spacing: -0.48px;
        line-height: 18px;
    }

    /* Streamlit 컬럼 오버라이드 */
    [data-testid="column"] {
        padding: 0 !important;
    }

    .stPlotlyChart {
        margin-top: -10px;
    }

    /* 반응형 */
    @media (max-width: 1400px) {
        .main-row { flex-direction: column; }
        .chart-section, .right-section { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 데이터 함수
# ============================================================================
@st.cache_data(ttl=60)
def load_plants():
    if PLANTS_CSV.exists():
        df = pd.read_csv(PLANTS_CSV)
        return df[df['status'] == '운영중']
    return pd.DataFrame()


def get_realtime_data():
    """실시간 데이터 (시뮬레이션) - Figma 값 기준"""
    hour = datetime.now().hour

    # Figma 디자인의 정확한 값 사용
    demand = 707.4
    reserve_rate = 94.5
    smp = 114.8
    smp_change = -6.0
    smp_change_pct = -5.0
    frequency = 60.01
    temperature = 3
    wind_speed = 7.2
    solar_radiation = 0  # 야간
    humidity = 49
    cloud_cover = 67

    return {
        'demand': demand,
        'reserve_rate': reserve_rate,
        'smp': smp,
        'smp_change': smp_change,
        'smp_change_pct': smp_change_pct,
        'frequency': frequency,
        'temperature': temperature,
        'wind_speed': wind_speed,
        'solar_radiation': solar_radiation,
        'humidity': humidity,
        'cloud_cover': cloud_cover
    }


def generate_chart_data():
    """차트 데이터 생성 - Figma 디자인 매칭"""
    # 12/19 03:00 ~ 12/19 24:00 (21시간)
    base_date = datetime(2024, 12, 19, 3, 0)
    hours = 22  # 03:00 to 24:00

    data = []
    for i in range(hours):
        t = base_date + timedelta(hours=i)
        h = t.hour

        # Figma 차트와 유사한 패턴 생성
        # 풍력: 1000-1200 MW 범위 (녹색 영역)
        wind_base = 1100 + 100 * np.sin(np.pi * (h - 6) / 12)

        # 태양광: 낮에만 발전 (노란색 영역)
        if 6 <= h <= 18:
            solar_base = 150 * np.sin(np.pi * (h - 6) / 12)
        else:
            solar_base = 0

        # 전력수요: 파란색 선 (600-800 MW)
        demand_base = 700 + 50 * np.sin(np.pi * (h - 8) / 10)

        # 공급능력: 회색 선 (2000-2200 MW)
        supply_base = 2100 + 100 * np.sin(np.pi * h / 24)

        # 노이즈 추가
        wind_actual = wind_base + np.random.uniform(-30, 30)
        wind_forecast = wind_base + np.random.uniform(-50, 50)
        solar_actual = max(0, solar_base + np.random.uniform(-20, 20))
        solar_forecast = max(0, solar_base + np.random.uniform(-30, 30))
        demand_actual = demand_base + np.random.uniform(-20, 20)
        demand_forecast = demand_base + np.random.uniform(-30, 30)
        supply_actual = supply_base + np.random.uniform(-50, 50)
        supply_forecast = supply_base + np.random.uniform(-70, 70)

        data.append({
            'time': t,
            'wind_actual': wind_actual,
            'wind_forecast': wind_forecast,
            'solar_actual': solar_actual,
            'solar_forecast': solar_forecast,
            'demand_actual': demand_actual,
            'demand_forecast': demand_forecast,
            'supply_actual': supply_actual,
            'supply_forecast': supply_forecast,
        })

    return pd.DataFrame(data)


def create_power_chart(df):
    """전력수급 차트 생성 - Figma 디자인 정확 매칭"""
    fig = go.Figure()

    # Y축 범위: 0-2500
    y_max = 2500

    # 1. 풍력 예측 (연한 초록 영역) - 맨 아래 레이어
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_forecast'],
        name='풍력(예측)',
        fill='tozeroy',
        fillcolor='rgba(144, 238, 144, 0.5)',
        line=dict(color='rgba(144, 238, 144, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 2. 태양광 예측 (연한 노랑 영역)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_forecast'] + df['solar_forecast'],
        name='태양광(예측)',
        fill='tonexty',
        fillcolor='rgba(255, 235, 153, 0.5)',
        line=dict(color='rgba(255, 235, 153, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 3. 풍력 실측 (진한 초록 영역)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_actual'],
        name='풍력(실측)',
        fill='tozeroy',
        fillcolor='rgba(76, 175, 80, 0.8)',
        line=dict(color='rgba(76, 175, 80, 1)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 4. 태양광 실측 (진한 노랑/주황 영역)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind_actual'] + df['solar_actual'],
        name='태양광(실측)',
        fill='tonexty',
        fillcolor='rgba(255, 193, 7, 0.8)',
        line=dict(color='rgba(255, 193, 7, 1)', width=1),
        showlegend=False,
        hoverinfo='skip'
    ))

    # 5. 전력수요 예측 (파란 점선)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['demand_forecast'],
        name='전력수요(예측)',
        line=dict(color='#2196F3', width=2, dash='dash'),
        mode='lines',
        showlegend=False
    ))

    # 6. 전력수요 실측 (파란 실선)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['demand_actual'],
        name='전력수요(실측)',
        line=dict(color='#1565C0', width=2),
        mode='lines',
        showlegend=False
    ))

    # 7. 공급능력 예측 (회색 점선)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['supply_forecast'],
        name='공급능력(예측)',
        line=dict(color='#9E9E9E', width=2, dash='dash'),
        mode='lines',
        showlegend=False
    ))

    # 8. 공급능력 실측 (회색 실선)
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['supply_actual'],
        name='공급능력(실측)',
        line=dict(color='#424242', width=2),
        mode='lines',
        showlegend=False
    ))

    # 현재 시간 수직선 (주황색) - 15:00 위치
    current_time = datetime(2024, 12, 19, 15, 0)
    fig.add_vline(x=current_time, line_width=2, line_color='rgba(255, 152, 0, 0.9)')

    # 레이아웃
    fig.update_layout(
        height=480,
        margin=dict(l=80, r=20, t=10, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,
        xaxis=dict(
            title='',
            tickformat='%m/%d %H:%M',
            tickvals=[datetime(2024, 12, 19, 3, 0), datetime(2024, 12, 19, 6, 0),
                      datetime(2024, 12, 19, 9, 0), datetime(2024, 12, 19, 12, 0),
                      datetime(2024, 12, 19, 15, 0), datetime(2024, 12, 19, 18, 0),
                      datetime(2024, 12, 19, 21, 0), datetime(2024, 12, 20, 0, 0)],
            ticktext=['12/19 03:00', '12/19 06:00', '12/19 09:00', '12/19 12:00',
                      '12/19 15:00', '12/19 18:00', '12/19 21:00', '12/19 24:00'],
            gridcolor='#e8e8e8',
            showgrid=True,
            tickfont=dict(size=14, color='#000000', family='Noto Sans KR'),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='전력(MW)', font=dict(size=16, color='#000000', family='Noto Sans KR')),
            range=[0, y_max],
            tickvals=[0, 500, 1000, 1500, 2000, 2500],
            gridcolor='#e8e8e8',
            showgrid=True,
            tickfont=dict(size=14, color='#000000', family='Noto Sans KR'),
            zeroline=False,
        ),
        hovermode='x unified'
    )

    return fig


# ============================================================================
# 메인 앱
# ============================================================================
def main():
    # 데이터 로드
    plants_df = load_plants()
    data = get_realtime_data()
    chart_df = generate_chart_data()

    # 발전소 통계 - Figma 값 사용
    solar_count = 6
    solar_cap = 5
    solar_gen = 0.0

    wind_count = 14
    wind_cap = 220
    wind_gen = 79.9

    ess_count = 4
    ess_cap = 75
    ess_gen = 11.7

    re_ratio = 15.6

    # ========== 전체 레이아웃 시작 ==========
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # ========== 헤더 - Frame 22 (3316:359) ==========
    st.markdown('''
        <div class="execo-header">
            <span class="execo-logo">eXeco</span>
            <div class="menu-icon">
                <div class="menu-dot"></div>
                <div class="menu-dot"></div>
                <div class="menu-dot"></div>
                <div class="menu-dot"></div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('<div class="content-area">', unsafe_allow_html=True)

    # ========== KPI 카드 행 - Frame 88 (3316:363) ==========
    st.markdown(f'''
        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">현재 수요</div>
                <div class="kpi-content">
                    <div class="kpi-value-group">
                        <span class="kpi-value">{data["demand"]}</span>
                        <span class="kpi-unit">MW</span>
                    </div>
                    <span class="kpi-badge">예비율 {data["reserve_rate"]}%</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">현재 SMP (제주)</div>
                <div class="kpi-content">
                    <div class="kpi-value-group">
                        <span class="kpi-value value-blue">{data["smp"]}</span>
                        <span class="kpi-unit value-blue">원</span>
                    </div>
                    <span class="kpi-badge badge-blue">{data["smp_change"]}원({data["smp_change_pct"]}%) ▼</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">재생에너지 비율</div>
                <div class="kpi-content">
                    <div class="kpi-value-group">
                        <span class="kpi-value value-red">{re_ratio}</span>
                        <span class="kpi-unit value-red">%</span>
                    </div>
                    <span class="kpi-badge badge-red">태양광+풍력</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">계통 주파수</div>
                <div class="kpi-content">
                    <div class="kpi-value-group">
                        <span class="kpi-value">{data["frequency"]}</span>
                        <span class="kpi-unit">Hz</span>
                    </div>
                    <span class="kpi-badge badge-green">정상</span>
                </div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">기상 현황</div>
                <div class="kpi-content">
                    <div class="kpi-value-group">
                        <span class="kpi-value">{data["temperature"]}</span>
                        <span class="kpi-unit">°C</span>
                    </div>
                    <span class="kpi-badge">풍속 {data["wind_speed"]} m/s</span>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # ========== 메인 콘텐츠 행 ==========
    st.markdown('<div class="main-row">', unsafe_allow_html=True)

    col_left, col_right = st.columns([70, 30])

    with col_left:
        # 차트 섹션 - Frame 95 (3316:416)
        st.markdown('''
            <div class="chart-container">
                <div class="section-header">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span class="section-title">제주 전력수급 현황</span>
                        <span class="section-subtitle">실측vs예측(MW)</span>
                    </div>
                </div>
                <div class="chart-inner">
                    <div class="legend-row">
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-area" style="background: rgba(144, 238, 144, 0.6);"></div></div>
                            <span class="legend-text">풍력(예측)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-area" style="background: rgba(255, 235, 153, 0.6);"></div></div>
                            <span class="legend-text">태양광(예측)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-area" style="background: rgba(76, 175, 80, 0.9);"></div></div>
                            <span class="legend-text">풍력(실측)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-area" style="background: rgba(255, 193, 7, 0.9);"></div></div>
                            <span class="legend-text">태양광(실측)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-line-dashed" style="color: #2196F3;"></div></div>
                            <span class="legend-text">전력수요(예측)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-line-dashed" style="color: #9E9E9E;"></div></div>
                            <span class="legend-text">공급능력(예측)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-line" style="background: #1565C0;"></div></div>
                            <span class="legend-text">전력수요(실측)</span>
                        </div>
                        <div class="legend-item">
                            <div class="legend-icon"><div class="legend-line" style="background: #424242;"></div></div>
                            <span class="legend-text">공급능력(실측)</span>
                        </div>
                    </div>
        ''', unsafe_allow_html=True)

        # 차트
        fig = create_power_chart(chart_df)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.markdown('</div></div>', unsafe_allow_html=True)

    with col_right:
        # 히트맵 카드 - Frame 96 (3316:555)
        st.markdown(f'''
            <div class="heatmap-card">
                <div class="heatmap-header">
                    <span class="heatmap-title">발전량 히트맵 표시</span>
                </div>
                <div class="heatmap-content">
                    <div class="legend-box">
                        <div class="legend-box-title">발전소 유형</div>
                        <div class="legend-box-item">
                            <div class="legend-dot dot-solar"></div>
                            <span class="legend-box-text">태양광</span>
                        </div>
                        <div class="legend-box-item">
                            <div class="legend-dot dot-wind"></div>
                            <span class="legend-box-text">풍력</span>
                        </div>
                        <div class="legend-box-item">
                            <div class="legend-dot dot-ess"></div>
                            <span class="legend-box-text">ESS</span>
                        </div>
                    </div>
                    <div class="jeju-map-container">
                        <svg viewBox="0 0 420 280" style="width:100%;height:100%;">
                            <!-- 제주도 실루엣 -->
                            <path d="M40,150 Q60,70 150,45 Q220,25 320,40 Q400,65 420,130
                                     Q430,175 395,215 Q330,265 210,270 Q80,265 45,215 Q25,180 40,150 Z"
                                  fill="#90EE90" stroke="#228B22" stroke-width="1" opacity="0.9"/>
                            <!-- 한라산 -->
                            <ellipse cx="210" cy="140" rx="40" ry="25" fill="#2E8B57" opacity="0.3"/>

                            <!-- ESS 마커 (노란색) -->
                            <circle cx="349" cy="58" r="5" fill="#ffbd00"/>
                            <circle cx="326" cy="63" r="5" fill="#ffbd00"/>
                            <circle cx="252" cy="215" r="5" fill="#ffbd00"/>
                            <circle cx="389" cy="192" r="5" fill="#ffbd00"/>

                            <!-- 풍력 마커 (파란색) -->
                            <circle cx="236" cy="83" r="5" fill="#4a89ff"/>
                            <circle cx="194" cy="94" r="11.5" fill="#4a89ff"/>
                            <circle cx="168" cy="125" r="5" fill="#4a89ff"/>
                            <circle cx="217" cy="215" r="5" fill="#4a89ff"/>
                            <circle cx="444" cy="162" r="5" fill="#4a89ff"/>

                            <!-- 태양광 마커 (빨간색) - 큰 것 -->
                            <circle cx="269" cy="61" r="17" fill="#ff4a4a"/>
                            <circle cx="320" cy="205" r="10.5" fill="#ff4a4a"/>
                        </svg>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 발전소 현황 카드 - Frame 77 (3316:584)
        st.markdown(f'''
            <div class="plant-card">
                <div>
                    <div class="plant-title">발전소 현황</div>
                    <div class="plant-list">
                        <div class="plant-row">
                            <span class="plant-type">태양광</span>
                            <div class="plant-info">
                                <span class="plant-count">{solar_count}개소 | {solar_cap}MW</span>
                                <span class="plant-gen">(발전량 : {solar_gen}MW)</span>
                            </div>
                        </div>
                        <div class="plant-row">
                            <span class="plant-type">풍력</span>
                            <div class="plant-info">
                                <span class="plant-count">{wind_count}개소|{wind_cap}MW</span>
                                <span class="plant-gen">(발전량 : {wind_gen}MW)</span>
                            </div>
                        </div>
                        <div class="plant-row">
                            <span class="plant-type">ESS</span>
                            <div class="plant-info">
                                <span class="plant-count">{ess_count}개소 | {ess_cap}MW</span>
                                <span class="plant-gen">(충방전 : {ess_gen}MW)</span>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="weather-section">
                    <div class="weather-title">기상 정보</div>
                    <div class="weather-info">
                        <span>일사량 ({data["solar_radiation"]}w/m²)</span>
                        <span>|</span>
                        <span>풍향 SE</span>
                        <span>|</span>
                        <span>운량 {data["cloud_cover"]}%</span>
                        <span>|</span>
                        <span>습도 {data["humidity"]}%</span>
                    </div>
                </div>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # main-row 닫기

    # 푸터 - Frame 74 (3316:615)
    st.markdown('''
        <div class="footer">
            제주 전력 지도 v4.0 | Powered by AI | © 2025 Power Demand Forecast Team<br>
            데이터 출처: EPSIS, 기상청 AMOS | 모델: LSTM + Quantile Regression
        </div>
    ''', unsafe_allow_html=True)

    st.markdown('</div></div>', unsafe_allow_html=True)  # content-area, main-container 닫기


if __name__ == "__main__":
    main()
