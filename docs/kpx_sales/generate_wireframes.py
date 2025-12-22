#!/usr/bin/env python3
"""
KPX 영업용 와이어프레임 생성 스크립트
한국전력거래소(KPX) 직원 대상 프레젠테이션용 와이어프레임
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow
import numpy as np
from pathlib import Path

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = Path(__file__).parent / "wireframes"
OUTPUT_DIR.mkdir(exist_ok=True)

# 색상 팔레트
COLORS = {
    'primary': '#1E3A8A',      # 진한 파랑
    'secondary': '#3B82F6',    # 밝은 파랑
    'accent': '#10B981',       # 초록
    'warning': '#F59E0B',      # 주황
    'danger': '#EF4444',       # 빨강
    'bg_light': '#F3F4F6',     # 밝은 회색
    'bg_dark': '#1F2937',      # 어두운 회색
    'text': '#374151',         # 텍스트
    'border': '#D1D5DB',       # 테두리
}


def create_rounded_rect(ax, x, y, width, height, color, label='', fontsize=10, alpha=1.0, text_color='white'):
    """둥근 사각형 생성"""
    fancy_box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        edgecolor='none',
        alpha=alpha
    )
    ax.add_patch(fancy_box)
    if label:
        ax.text(x + width/2, y + height/2, label,
                ha='center', va='center', fontsize=fontsize,
                color=text_color, fontweight='bold')


def wireframe_01_main_dashboard():
    """와이어프레임 1: 메인 대시보드"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # 배경
    ax.add_patch(Rectangle((0, 0), 16, 10, facecolor=COLORS['bg_light']))

    # 헤더
    create_rounded_rect(ax, 0.2, 9.2, 15.6, 0.6, COLORS['primary'],
                       '⚡ 제주도 전력 수요 예측 시스템 v2.1', fontsize=14)

    # 사이드바
    create_rounded_rect(ax, 0.2, 0.2, 2.5, 8.8, COLORS['bg_dark'], '', alpha=0.9)

    sidebar_items = ['📊 대시보드', '🔮 수요 예측', '💰 SMP 예측', '🔍 XAI 분석', '📈 시나리오', '⚙️ 설정']
    for i, item in enumerate(sidebar_items):
        y_pos = 8.5 - i * 0.8
        color = COLORS['secondary'] if i == 0 else 'white'
        ax.text(0.5, y_pos, item, fontsize=10, color=color, fontweight='bold')

    # KPI 카드 영역
    kpi_data = [
        ('현재 수요', '1,245 MW', COLORS['primary']),
        ('예측 수요 (+1h)', '1,312 MW', COLORS['secondary']),
        ('예비율', '18.5%', COLORS['accent']),
        ('SMP 예측', '105.2 ₩/kWh', COLORS['warning']),
    ]

    for i, (title, value, color) in enumerate(kpi_data):
        x = 3 + i * 3.2
        create_rounded_rect(ax, x, 7.5, 3, 1.5, 'white', '', alpha=1.0, text_color=COLORS['text'])
        ax.add_patch(Rectangle((x, 7.5), 3, 1.5, fill=False, edgecolor=COLORS['border'], linewidth=1))
        ax.text(x + 1.5, 8.6, title, ha='center', fontsize=9, color=COLORS['text'])
        ax.text(x + 1.5, 8.0, value, ha='center', fontsize=14, color=color, fontweight='bold')

    # 메인 차트 영역 - 24시간 예측
    create_rounded_rect(ax, 3, 3.5, 9.5, 3.8, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((3, 3.5), 9.5, 3.8, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(3.3, 7.0, '📈 24시간 전력 수요 예측', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 예측 그래프 시뮬레이션
    x_data = np.linspace(3.5, 12, 24)
    actual = 1200 + 150 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.randn(24) * 20
    predicted = 1200 + 150 * np.sin(np.linspace(0, 2*np.pi, 24))
    upper = predicted + 50
    lower = predicted - 50

    y_scale = lambda y: 3.8 + (y - 1050) / 350 * 2.8

    ax.fill_between(x_data, y_scale(lower), y_scale(upper), alpha=0.3, color=COLORS['secondary'])
    ax.plot(x_data, y_scale(actual), 'o-', color=COLORS['primary'], markersize=3, label='실측')
    ax.plot(x_data, y_scale(predicted), '--', color=COLORS['accent'], linewidth=2, label='예측')

    ax.text(10, 4.0, '실측 ─  예측 ---  80% 신뢰구간 ▒', fontsize=8, color=COLORS['text'])

    # 우측 패널 - 알림
    create_rounded_rect(ax, 12.8, 3.5, 2.9, 3.8, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((12.8, 3.5), 2.9, 3.8, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(13.0, 7.0, '🔔 알림', fontsize=11, fontweight='bold', color=COLORS['text'])

    alerts = [
        ('⚠️', '피크 수요 예상', '15:00'),
        ('✅', '예비율 정상', '현재'),
        ('📊', 'SMP 상승 예상', '09:00'),
    ]
    for i, (icon, msg, time) in enumerate(alerts):
        y = 6.4 - i * 0.8
        ax.text(13.0, y, f'{icon} {msg}', fontsize=8, color=COLORS['text'])
        ax.text(15.4, y, time, fontsize=7, color=COLORS['text'], ha='right')

    # 하단 테이블
    create_rounded_rect(ax, 3, 0.3, 12.7, 3, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((3, 0.3), 12.7, 3, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(3.3, 3.0, '📋 시간대별 예측 상세', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 테이블 헤더
    headers = ['시간', '예측(MW)', '신뢰구간', 'MAPE', '상태']
    for i, h in enumerate(headers):
        ax.text(3.5 + i * 2.5, 2.5, h, fontsize=9, fontweight='bold', color=COLORS['primary'])

    # 테이블 데이터
    table_data = [
        ('09:00', '1,245', '1,195~1,295', '8.2%', '🟢'),
        ('12:00', '1,380', '1,320~1,440', '9.1%', '🟡'),
        ('15:00', '1,425', '1,360~1,490', '10.5%', '🟠'),
        ('18:00', '1,310', '1,250~1,370', '7.8%', '🟢'),
    ]
    for row_idx, row in enumerate(table_data):
        y = 2.0 - row_idx * 0.4
        for col_idx, cell in enumerate(row):
            ax.text(3.5 + col_idx * 2.5, y, cell, fontsize=8, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_main_dashboard.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg_light'], edgecolor='none')
    plt.close()
    print("✅ 01_main_dashboard.png 생성 완료")


def wireframe_02_smp_prediction():
    """와이어프레임 2: SMP 예측 및 입찰 지원"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # 배경
    ax.add_patch(Rectangle((0, 0), 16, 10, facecolor=COLORS['bg_light']))

    # 헤더
    create_rounded_rect(ax, 0.2, 9.2, 15.6, 0.6, COLORS['primary'],
                       '💰 SMP 예측 및 입찰 지원 시스템', fontsize=14)

    # 좌측: SMP 예측 그래프
    create_rounded_rect(ax, 0.2, 4.5, 8, 4.5, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((0.2, 4.5), 8, 4.5, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(0.5, 8.7, '📈 48시간 SMP 예측', fontsize=12, fontweight='bold', color=COLORS['text'])

    # SMP 그래프
    x_data = np.linspace(0.5, 7.8, 48)
    smp_pred = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, 48)) + np.random.randn(48) * 5
    upper = smp_pred + 15
    lower = smp_pred - 15

    y_scale = lambda y: 4.8 + (y - 70) / 80 * 3.2

    ax.fill_between(x_data, y_scale(lower), y_scale(upper), alpha=0.3, color=COLORS['secondary'])
    ax.plot(x_data, y_scale(smp_pred), '-', color=COLORS['primary'], linewidth=2)

    # Y축 레이블
    for val in [80, 100, 120, 140]:
        y = y_scale(val)
        ax.text(0.3, y, f'{val}', fontsize=7, ha='right', color=COLORS['text'])
        ax.plot([0.5, 7.8], [y, y], '--', color=COLORS['border'], linewidth=0.5, alpha=0.5)

    ax.text(4, 4.6, '예측 ─  80% 신뢰구간 ▒', fontsize=8, color=COLORS['text'], ha='center')

    # 우측 상단: 모델 선택
    create_rounded_rect(ax, 8.5, 7, 7.3, 2, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((8.5, 7), 7.3, 2, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(8.7, 8.7, '🤖 모델 선택', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 토글 스위치
    ax.text(8.9, 8.2, '기본 모델 (LSTM)', fontsize=9, color=COLORS['text'])
    create_rounded_rect(ax, 12.5, 8.1, 0.8, 0.3, COLORS['border'], '', alpha=0.5)

    ax.text(8.9, 7.6, '고도화 모델 (Quantile + Attention)', fontsize=9, color=COLORS['text'])
    create_rounded_rect(ax, 12.5, 7.5, 0.8, 0.3, COLORS['accent'], '', alpha=1.0)
    ax.add_patch(patches.Circle((13.1, 7.65), 0.12, color='white'))

    ax.text(13.5, 8.2, '❌', fontsize=10)
    ax.text(13.5, 7.6, '✅', fontsize=10)

    # 우측 중앙: 입찰 추천
    create_rounded_rect(ax, 8.5, 4.5, 7.3, 2.3, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((8.5, 4.5), 7.3, 2.3, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(8.7, 6.5, '💡 입찰 추천', fontsize=11, fontweight='bold', color=COLORS['text'])

    recommendations = [
        ('최적 입찰가', '102.5 ₩/kWh', COLORS['accent']),
        ('예상 낙찰 확률', '78.3%', COLORS['secondary']),
        ('추천 입찰량', '450 MW', COLORS['primary']),
    ]
    for i, (label, value, color) in enumerate(recommendations):
        y = 6.0 - i * 0.45
        ax.text(8.9, y, label, fontsize=9, color=COLORS['text'])
        ax.text(13.5, y, value, fontsize=10, fontweight='bold', color=color, ha='left')

    # 하단: Quantile 분포
    create_rounded_rect(ax, 0.2, 0.2, 7.8, 4.1, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((0.2, 0.2), 7.8, 4.1, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(0.5, 4.0, '📊 SMP 분포 예측 (Quantile)', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 분위수 박스플롯 시뮬레이션
    times = ['09:00', '12:00', '15:00', '18:00', '21:00']
    for i, t in enumerate(times):
        x = 1 + i * 1.4
        # 박스플롯
        q10, q50, q90 = 85 + i*5, 100 + i*8, 120 + i*10
        y_box = lambda y: 0.5 + (y - 80) / 80 * 2.8

        ax.plot([x, x], [y_box(q10), y_box(q90)], '-', color=COLORS['primary'], linewidth=2)
        ax.plot([x-0.15, x+0.15], [y_box(q10), y_box(q10)], '-', color=COLORS['primary'], linewidth=2)
        ax.plot([x-0.15, x+0.15], [y_box(q90), y_box(q90)], '-', color=COLORS['primary'], linewidth=2)
        ax.add_patch(Rectangle((x-0.2, y_box(q50)-0.1), 0.4, 0.2, facecolor=COLORS['accent']))
        ax.text(x, 0.35, t, fontsize=7, ha='center', color=COLORS['text'])

    ax.text(1, 3.5, 'Q10', fontsize=7, color=COLORS['text'])
    ax.text(1, 2.5, 'Q50', fontsize=7, color=COLORS['accent'], fontweight='bold')
    ax.text(1, 1.5, 'Q90', fontsize=7, color=COLORS['text'])

    # 하단 우측: 리스크 분석
    create_rounded_rect(ax, 8.5, 0.2, 7.3, 4.1, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((8.5, 0.2), 7.3, 4.1, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(8.7, 4.0, '⚠️ 리스크 분석', fontsize=11, fontweight='bold', color=COLORS['text'])

    risk_items = [
        ('가격 변동성', '중간', COLORS['warning'], 0.6),
        ('예측 불확실성', '낮음', COLORS['accent'], 0.3),
        ('피크 리스크', '높음', COLORS['danger'], 0.8),
        ('공급 부족 위험', '낮음', COLORS['accent'], 0.2),
    ]

    for i, (label, level, color, pct) in enumerate(risk_items):
        y = 3.4 - i * 0.7
        ax.text(8.9, y, label, fontsize=9, color=COLORS['text'])
        # 프로그레스 바
        ax.add_patch(Rectangle((11.5, y-0.1), 3.5, 0.25, facecolor=COLORS['bg_light']))
        ax.add_patch(Rectangle((11.5, y-0.1), 3.5 * pct, 0.25, facecolor=color))
        ax.text(15.2, y, level, fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_smp_prediction.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg_light'], edgecolor='none')
    plt.close()
    print("✅ 02_smp_prediction.png 생성 완료")


def wireframe_03_xai_analysis():
    """와이어프레임 3: XAI 분석 (설명 가능한 AI)"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(Rectangle((0, 0), 16, 10, facecolor=COLORS['bg_light']))

    # 헤더
    create_rounded_rect(ax, 0.2, 9.2, 15.6, 0.6, COLORS['primary'],
                       '🔍 XAI 분석 - 설명 가능한 AI', fontsize=14)

    # 좌측 상단: Attention 시각화
    create_rounded_rect(ax, 0.2, 4.8, 7.8, 4.2, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((0.2, 4.8), 7.8, 4.2, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(0.5, 8.7, '👁️ Attention 가중치 시각화', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 히트맵 시뮬레이션
    attention_data = np.random.rand(8, 24)
    attention_data[:, -5:] = attention_data[:, -5:] * 1.5  # 최근 시점 강조
    attention_data = np.clip(attention_data, 0, 1)

    for i in range(8):
        for j in range(24):
            intensity = attention_data[i, j]
            color = plt.cm.Blues(intensity)
            ax.add_patch(Rectangle((0.5 + j*0.3, 5.2 + i*0.4), 0.28, 0.38, facecolor=color))

    ax.text(0.4, 8.4, '피처', fontsize=8, color=COLORS['text'], rotation=90, va='top')
    ax.text(4, 5.0, '시간 (t-48h → t)', fontsize=8, color=COLORS['text'], ha='center')
    ax.text(6.5, 8.4, '최근 시점 주목 ↗', fontsize=8, color=COLORS['danger'], fontweight='bold')

    # 우측 상단: 피처 중요도
    create_rounded_rect(ax, 8.2, 4.8, 7.6, 4.2, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((8.2, 4.8), 7.6, 4.2, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(8.5, 8.7, '📊 피처 중요도 (SHAP 기반)', fontsize=11, fontweight='bold', color=COLORS['text'])

    features = [
        ('기온 (Temperature)', 0.85),
        ('일사량 (Solar Radiation)', 0.72),
        ('전일 동시간 수요', 0.68),
        ('시간대 (Hour)', 0.55),
        ('요일 (Day of Week)', 0.42),
        ('습도 (Humidity)', 0.35),
        ('풍속 (Wind Speed)', 0.28),
    ]

    for i, (feat, importance) in enumerate(features):
        y = 8.2 - i * 0.45
        ax.text(8.5, y, feat, fontsize=8, color=COLORS['text'])
        bar_width = importance * 4.5
        color = COLORS['primary'] if importance > 0.5 else COLORS['secondary']
        ax.add_patch(Rectangle((11.5, y-0.1), bar_width, 0.25, facecolor=color, alpha=0.7))
        ax.text(11.5 + bar_width + 0.1, y, f'{importance:.0%}', fontsize=7, color=COLORS['text'])

    # 좌측 하단: 예측 신뢰도 분석
    create_rounded_rect(ax, 0.2, 0.2, 7.8, 4.4, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((0.2, 0.2), 7.8, 4.4, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(0.5, 4.3, '🎯 예측 신뢰도 분석', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 신뢰도 게이지
    ax.text(1.5, 3.6, '현재 예측 신뢰도', fontsize=10, fontweight='bold', color=COLORS['text'])

    # 반원 게이지
    theta = np.linspace(np.pi, 0, 100)
    r = 1.2
    x_gauge = 2.5 + r * np.cos(theta)
    y_gauge = 2.5 + r * np.sin(theta)
    ax.plot(x_gauge, y_gauge, '-', color=COLORS['border'], linewidth=8)

    # 채워진 부분 (82.5%)
    theta_filled = np.linspace(np.pi, np.pi * (1 - 0.825), 50)
    x_filled = 2.5 + r * np.cos(theta_filled)
    y_filled = 2.5 + r * np.sin(theta_filled)
    ax.plot(x_filled, y_filled, '-', color=COLORS['accent'], linewidth=8)

    ax.text(2.5, 2.2, '82.5%', fontsize=18, fontweight='bold', color=COLORS['accent'], ha='center')
    ax.text(2.5, 1.7, '80% 구간 커버리지', fontsize=8, color=COLORS['text'], ha='center')

    # 신뢰도 지표
    metrics = [
        ('Attention Entropy', '3.87', '분산된 주목'),
        ('Attention Concentration', '0.026', '과집중 없음'),
        ('데이터 누수 위험', 'LOW', '안전'),
    ]

    for i, (name, value, desc) in enumerate(metrics):
        y = 3.5 - i * 0.7
        ax.text(5, y, name, fontsize=8, color=COLORS['text'])
        ax.text(7.2, y, value, fontsize=9, fontweight='bold', color=COLORS['accent'], ha='right')
        ax.text(7.5, y, desc, fontsize=7, color=COLORS['text'])

    # 우측 하단: 예측 근거
    create_rounded_rect(ax, 8.2, 0.2, 7.6, 4.4, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((8.2, 0.2), 7.6, 4.4, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(8.5, 4.3, '💬 예측 근거 (자연어 설명)', fontsize=11, fontweight='bold', color=COLORS['text'])

    explanation_text = [
        "🔹 오늘 15시 예측 수요 1,425 MW의 주요 근거:",
        "",
        "  1. 기온 28°C로 냉방 수요 증가 예상 (+12%)",
        "  2. 일사량 850 W/m²로 태양광 발전 최대",
        "     → 계통 수요 감소 효과 (-5%)",
        "  3. 전일 동시간 대비 유사 패턴 확인",
        "  4. 주중 화요일로 산업용 수요 유지",
        "",
        "⚠️ 불확실성 요인: 구름량 증가 가능성 (±50MW)"
    ]

    for i, line in enumerate(explanation_text):
        ax.text(8.5, 3.8 - i * 0.38, line, fontsize=8, color=COLORS['text'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_xai_analysis.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg_light'], edgecolor='none')
    plt.close()
    print("✅ 03_xai_analysis.png 생성 완료")


def wireframe_04_scenario_analysis():
    """와이어프레임 4: 시나리오 분석"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(Rectangle((0, 0), 16, 10, facecolor=COLORS['bg_light']))

    # 헤더
    create_rounded_rect(ax, 0.2, 9.2, 15.6, 0.6, COLORS['primary'],
                       '📈 시나리오 분석 - What-if 시뮬레이션', fontsize=14)

    # 좌측: 시나리오 설정
    create_rounded_rect(ax, 0.2, 4.5, 5, 4.5, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((0.2, 4.5), 5, 4.5, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(0.5, 8.7, '🎛️ 시나리오 설정', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 프리셋 버튼
    presets = ['기본', '폭염', '한파', '휴일', '사용자 정의']
    for i, p in enumerate(presets):
        x = 0.5 + (i % 3) * 1.5
        y = 8.2 - (i // 3) * 0.5
        color = COLORS['secondary'] if p == '폭염' else COLORS['border']
        create_rounded_rect(ax, x, y, 1.3, 0.4, color, p, fontsize=8,
                           text_color='white' if p == '폭염' else COLORS['text'])

    # 슬라이더
    params = [
        ('기온', '+5°C', 0.7),
        ('일사량', '+20%', 0.6),
        ('습도', '0%', 0.5),
        ('풍속', '-10%', 0.4),
    ]

    for i, (name, value, pos) in enumerate(params):
        y = 6.8 - i * 0.8
        ax.text(0.5, y, name, fontsize=9, color=COLORS['text'])
        ax.add_patch(Rectangle((2, y-0.1), 2.5, 0.2, facecolor=COLORS['bg_light']))
        ax.add_patch(Rectangle((2, y-0.1), 2.5 * pos, 0.2, facecolor=COLORS['secondary']))
        ax.add_patch(patches.Circle((2 + 2.5 * pos, y), 0.15, color=COLORS['primary']))
        ax.text(4.7, y, value, fontsize=8, color=COLORS['primary'], fontweight='bold')

    # 실행 버튼
    create_rounded_rect(ax, 1.5, 4.7, 2.5, 0.6, COLORS['accent'], '▶ 시뮬레이션 실행', fontsize=10)

    # 중앙: 비교 차트
    create_rounded_rect(ax, 5.4, 4.5, 7, 4.5, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((5.4, 4.5), 7, 4.5, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(5.7, 8.7, '📊 시나리오 비교', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 멀티 라인 차트
    x_data = np.linspace(5.8, 12, 24)
    base = 1200 + 100 * np.sin(np.linspace(0, 2*np.pi, 24))
    heat = base * 1.15
    cold = base * 0.9

    y_scale = lambda y: 4.8 + (y - 1000) / 600 * 3.5

    ax.plot(x_data, y_scale(base), '-', color=COLORS['border'], linewidth=2, label='기본')
    ax.plot(x_data, y_scale(heat), '-', color=COLORS['danger'], linewidth=2, label='폭염')
    ax.plot(x_data, y_scale(cold), '-', color=COLORS['secondary'], linewidth=2, label='한파')

    ax.text(6, 8.4, '── 기본  ', fontsize=8, color=COLORS['border'])
    ax.text(7.5, 8.4, '── 폭염  ', fontsize=8, color=COLORS['danger'])
    ax.text(9, 8.4, '── 한파', fontsize=8, color=COLORS['secondary'])

    # 차이 영역 표시
    ax.fill_between(x_data, y_scale(base), y_scale(heat), alpha=0.2, color=COLORS['danger'])

    # 우측: 영향 분석
    create_rounded_rect(ax, 12.6, 4.5, 3.2, 4.5, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((12.6, 4.5), 3.2, 4.5, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(12.8, 8.7, '📋 영향 분석', fontsize=11, fontweight='bold', color=COLORS['text'])

    impacts = [
        ('피크 수요', '+215 MW', COLORS['danger']),
        ('일평균 수요', '+12.3%', COLORS['warning']),
        ('예비율', '-3.2%p', COLORS['danger']),
        ('SMP 예상', '+18.5원', COLORS['warning']),
        ('발전 비용', '+2.1억원', COLORS['warning']),
    ]

    for i, (name, value, color) in enumerate(impacts):
        y = 8.1 - i * 0.65
        ax.text(12.8, y, name, fontsize=8, color=COLORS['text'])
        ax.text(15.5, y, value, fontsize=9, fontweight='bold', color=color, ha='right')

    # 하단: 상세 테이블
    create_rounded_rect(ax, 0.2, 0.2, 15.6, 4.1, 'white', '', alpha=1.0)
    ax.add_patch(Rectangle((0.2, 0.2), 15.6, 4.1, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(0.5, 4.0, '📑 시간대별 시나리오 상세 비교', fontsize=11, fontweight='bold', color=COLORS['text'])

    # 테이블
    headers = ['시간', '기본(MW)', '폭염(MW)', '차이', '증가율', '예비율', '리스크']
    for i, h in enumerate(headers):
        ax.text(0.7 + i * 2.2, 3.5, h, fontsize=9, fontweight='bold', color=COLORS['primary'])

    data = [
        ('09:00', '1,180', '1,320', '+140', '+11.9%', '22.1%', '🟢'),
        ('12:00', '1,250', '1,415', '+165', '+13.2%', '19.5%', '🟡'),
        ('15:00', '1,280', '1,495', '+215', '+16.8%', '15.2%', '🟠'),
        ('18:00', '1,310', '1,485', '+175', '+13.4%', '16.8%', '🟡'),
        ('21:00', '1,220', '1,380', '+160', '+13.1%', '20.3%', '🟢'),
    ]

    for row_idx, row in enumerate(data):
        y = 3.0 - row_idx * 0.5
        for col_idx, cell in enumerate(row):
            color = COLORS['danger'] if col_idx == 3 and int(cell.replace('+', '')) > 150 else COLORS['text']
            ax.text(0.7 + col_idx * 2.2, y, cell, fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_scenario_analysis.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg_light'], edgecolor='none')
    plt.close()
    print("✅ 04_scenario_analysis.png 생성 완료")


def wireframe_05_system_architecture():
    """와이어프레임 5: 시스템 아키텍처"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(Rectangle((0, 0), 16, 10, facecolor='white'))

    # 제목
    ax.text(8, 9.5, '🏗️ 시스템 아키텍처', fontsize=16, fontweight='bold',
            color=COLORS['primary'], ha='center')

    # 데이터 소스 계층
    ax.text(2, 8.5, '📡 데이터 소스', fontsize=11, fontweight='bold', color=COLORS['text'])
    sources = [
        ('KPX\nEPSIS', COLORS['primary']),
        ('기상청\nAMOS', COLORS['secondary']),
        ('공공데이터\n포털', COLORS['accent']),
    ]
    for i, (name, color) in enumerate(sources):
        x = 0.5 + i * 2.5
        create_rounded_rect(ax, x, 7.3, 2, 1, color, name, fontsize=8)

    # 화살표
    for i in range(3):
        ax.annotate('', xy=(1.5 + i*2.5, 6.3), xytext=(1.5 + i*2.5, 7.2),
                   arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))

    # 데이터 처리 계층
    ax.text(2, 6.2, '⚙️ 데이터 처리', fontsize=11, fontweight='bold', color=COLORS['text'])
    create_rounded_rect(ax, 0.5, 5, 7, 1, COLORS['bg_dark'],
                       '크롤러 → 전처리 → 피처 엔지니어링 → 스케일링', fontsize=9)

    ax.annotate('', xy=(4, 4), xytext=(4, 4.9),
               arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=1.5))

    # ML 모델 계층
    ax.text(2, 3.9, '🧠 ML 모델', fontsize=11, fontweight='bold', color=COLORS['text'])
    models = [
        ('LSTM\n(기본)', COLORS['primary']),
        ('BiLSTM\n(양방향)', COLORS['secondary']),
        ('TFT\n(Transformer)', COLORS['accent']),
        ('Ensemble\n(앙상블)', COLORS['warning']),
    ]
    for i, (name, color) in enumerate(models):
        x = 0.3 + i * 1.9
        create_rounded_rect(ax, x, 2.5, 1.7, 1.2, color, name, fontsize=8)

    # 우측: API 서버
    ax.text(10, 8.5, '🚀 서비스 계층', fontsize=11, fontweight='bold', color=COLORS['text'])

    create_rounded_rect(ax, 8.5, 5.5, 3, 2.5, COLORS['primary'], '', alpha=0.9)
    ax.text(10, 7.7, 'FastAPI', fontsize=11, fontweight='bold', color='white', ha='center')
    ax.text(10, 7.2, 'REST API', fontsize=9, color='white', ha='center')
    ax.text(10, 6.7, '• /predict', fontsize=8, color='white', ha='center')
    ax.text(10, 6.3, '• /scenarios', fontsize=8, color='white', ha='center')
    ax.text(10, 5.9, '• /explain', fontsize=8, color='white', ha='center')

    # 연결선
    ax.plot([7.5, 8.5], [3.1, 6], '-', color=COLORS['text'], linewidth=1.5)

    # 대시보드
    create_rounded_rect(ax, 12, 5.5, 3.5, 2.5, COLORS['accent'], '', alpha=0.9)
    ax.text(13.75, 7.7, 'Streamlit', fontsize=11, fontweight='bold', color='white', ha='center')
    ax.text(13.75, 7.2, '대시보드', fontsize=9, color='white', ha='center')
    ax.text(13.75, 6.5, '• 실시간 모니터링', fontsize=8, color='white', ha='center')
    ax.text(13.75, 6.1, '• 예측 시각화', fontsize=8, color='white', ha='center')
    ax.text(13.75, 5.7, '• XAI 분석', fontsize=8, color='white', ha='center')

    ax.plot([11.5, 12], [6.75, 6.75], '-', color=COLORS['text'], linewidth=1.5)

    # 하단: 모니터링
    ax.text(10, 4.2, '📊 모니터링', fontsize=11, fontweight='bold', color=COLORS['text'])
    monitors = [
        ('Prometheus\n메트릭', COLORS['warning']),
        ('알림\n시스템', COLORS['danger']),
        ('로깅\n시스템', COLORS['secondary']),
    ]
    for i, (name, color) in enumerate(monitors):
        x = 8.5 + i * 2.5
        create_rounded_rect(ax, x, 2.5, 2.2, 1.5, color, name, fontsize=8)

    # 연결선 (API to 모니터링)
    ax.plot([10, 10], [5.5, 4], '-', color=COLORS['text'], linewidth=1.5)
    ax.plot([8.5, 15.2], [4, 4], '-', color=COLORS['text'], linewidth=1.5)
    for x in [9.5, 12, 14.5]:
        ax.plot([x, x], [4, 4], 'o', color=COLORS['text'], markersize=4)

    # 하단: 성능 지표
    create_rounded_rect(ax, 0.5, 0.3, 15, 1.8, COLORS['bg_light'], '', alpha=1.0)
    ax.add_patch(Rectangle((0.5, 0.3), 15, 1.8, fill=False, edgecolor=COLORS['border'], linewidth=1))
    ax.text(1, 1.9, '📈 성능 지표', fontsize=10, fontweight='bold', color=COLORS['text'])

    metrics = [
        ('MAPE', '10.68%'),
        ('MAE', '11.27원/kWh'),
        ('80% Coverage', '82.5%'),
        ('API 응답', '<100ms'),
        ('테스트', '1,436건 통과'),
    ]
    for i, (name, value) in enumerate(metrics):
        x = 1 + i * 3
        ax.text(x, 1.3, name, fontsize=9, color=COLORS['text'])
        ax.text(x, 0.8, value, fontsize=11, fontweight='bold', color=COLORS['primary'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_system_architecture.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("✅ 05_system_architecture.png 생성 완료")


def wireframe_06_value_proposition():
    """와이어프레임 6: KPX 가치 제안"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')

    # 배경 그라데이션 효과
    for i in range(100):
        alpha = 0.5 - i * 0.003
        ax.add_patch(Rectangle((0, i*0.1), 16, 0.1, facecolor=COLORS['primary'], alpha=alpha))

    # 제목
    ax.text(8, 9, '💡 KPX를 위한 가치 제안', fontsize=20, fontweight='bold',
            color='white', ha='center')
    ax.text(8, 8.4, 'AI 기반 전력 수요 예측 시스템의 비즈니스 가치',
            fontsize=12, color='white', ha='center', alpha=0.9)

    # 가치 카드들
    values = [
        {
            'icon': '🎯',
            'title': '예측 정확도 향상',
            'desc': '딥러닝 기반 예측으로\nMAPE 10% 이하 달성',
            'metric': '10.68%',
            'metric_label': 'MAPE',
        },
        {
            'icon': '💰',
            'title': '비용 절감',
            'desc': '정확한 예측으로\n예비력 최적화',
            'metric': '~5%',
            'metric_label': '연료비 절감 추정',
        },
        {
            'icon': '⚡',
            'title': '운영 효율화',
            'desc': '자동화된 예측으로\n의사결정 시간 단축',
            'metric': '24h',
            'metric_label': '예측 범위',
        },
        {
            'icon': '🔍',
            'title': '투명한 AI',
            'desc': 'XAI로 예측 근거\n명확히 설명',
            'metric': '82.5%',
            'metric_label': '신뢰 구간 정확도',
        },
    ]

    for i, v in enumerate(values):
        x = 0.5 + i * 4
        # 카드 배경
        create_rounded_rect(ax, x, 4.5, 3.5, 3.3, 'white', '', alpha=0.95)
        ax.add_patch(Rectangle((x, 4.5), 3.5, 3.3, fill=False,
                               edgecolor=COLORS['secondary'], linewidth=2))

        # 아이콘
        ax.text(x + 1.75, 7.3, v['icon'], fontsize=24, ha='center')
        # 제목
        ax.text(x + 1.75, 6.6, v['title'], fontsize=11, fontweight='bold',
                color=COLORS['primary'], ha='center')
        # 설명
        ax.text(x + 1.75, 5.9, v['desc'], fontsize=9, color=COLORS['text'],
                ha='center', va='top')
        # 지표
        ax.text(x + 1.75, 5.0, v['metric'], fontsize=18, fontweight='bold',
                color=COLORS['accent'], ha='center')
        ax.text(x + 1.75, 4.7, v['metric_label'], fontsize=8,
                color=COLORS['text'], ha='center')

    # 하단: 도입 효과
    create_rounded_rect(ax, 0.5, 0.5, 15, 3.7, 'white', '', alpha=0.95)
    ax.add_patch(Rectangle((0.5, 0.5), 15, 3.7, fill=False,
                           edgecolor=COLORS['secondary'], linewidth=2))

    ax.text(8, 3.8, '📊 예상 도입 효과', fontsize=14, fontweight='bold',
            color=COLORS['primary'], ha='center')

    effects = [
        ('🔋', '예비율 관리', '불필요한 예비력 3~5% 감축\n→ 연간 수십억 원 절감 가능'),
        ('📈', 'SMP 안정화', '정확한 수요 예측으로\nSMP 변동성 완화'),
        ('⚠️', '리스크 감소', '피크 수요 사전 예측으로\n비상 발전 가동 최소화'),
        ('🤖', '업무 자동화', '수동 예측 업무 80% 자동화\n인력 효율화'),
    ]

    for i, (icon, title, desc) in enumerate(effects):
        x = 1 + i * 3.8
        ax.text(x, 3.2, icon, fontsize=16)
        ax.text(x + 0.5, 3.2, title, fontsize=10, fontweight='bold',
                color=COLORS['primary'])
        ax.text(x, 2.5, desc, fontsize=8, color=COLORS['text'], va='top')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '06_value_proposition.png', dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg_light'], edgecolor='none')
    plt.close()
    print("✅ 06_value_proposition.png 생성 완료")


def main():
    """모든 와이어프레임 생성"""
    print("\n" + "="*50)
    print("🎨 KPX 영업용 와이어프레임 생성 시작")
    print("="*50 + "\n")

    wireframe_01_main_dashboard()
    wireframe_02_smp_prediction()
    wireframe_03_xai_analysis()
    wireframe_04_scenario_analysis()
    wireframe_05_system_architecture()
    wireframe_06_value_proposition()

    print("\n" + "="*50)
    print(f"✅ 모든 와이어프레임 생성 완료!")
    print(f"📁 저장 위치: {OUTPUT_DIR}")
    print("="*50 + "\n")

    # 파일 목록 출력
    print("생성된 파일:")
    for f in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  • {f.name}")


if __name__ == "__main__":
    main()
