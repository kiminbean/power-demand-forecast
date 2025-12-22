#!/usr/bin/env python3
"""
KPX 및 중소 친환경 발전소 영업용 자료 생성
- 시스템 문제점 분석
- 영업 시나리오
- 대시보드 핵심 기능 설명
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc
from matplotlib.path import Path as MplPath
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path

# Font settings
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path(__file__).parent / "sales_materials"
OUTPUT_DIR.mkdir(exist_ok=True)

# Color Palette
C = {
    'primary': '#2563EB',
    'primary_dark': '#1E40AF',
    'primary_light': '#DBEAFE',
    'secondary': '#10B981',
    'secondary_light': '#D1FAE5',
    'warning': '#F59E0B',
    'warning_light': '#FEF3C7',
    'danger': '#EF4444',
    'danger_light': '#FEE2E2',
    'purple': '#8B5CF6',
    'purple_light': '#EDE9FE',
    'slate_50': '#F8FAFC',
    'slate_100': '#F1F5F9',
    'slate_200': '#E2E8F0',
    'slate_300': '#CBD5E1',
    'slate_400': '#94A3B8',
    'slate_500': '#64748B',
    'slate_600': '#475569',
    'slate_700': '#334155',
    'slate_800': '#1E293B',
    'slate_900': '#0F172A',
    'white': '#FFFFFF',
}


def add_shadow(ax, x, y, w, h, offset=0.08, alpha=0.08):
    shadow = FancyBboxPatch(
        (x + offset, y - offset), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor='black', edgecolor='none', alpha=alpha, zorder=0
    )
    ax.add_patch(shadow)


def create_card(ax, x, y, w, h, title='', title_color=None, shadow=True):
    if shadow:
        add_shadow(ax, x, y, w, h)
    card = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=C['white'], edgecolor=C['slate_200'], linewidth=1, zorder=1
    )
    ax.add_patch(card)
    if title:
        ax.text(x + 0.4, y + h - 0.5, title,
                fontsize=12, fontweight='bold', color=title_color or C['slate_800'], zorder=2)


def create_01_problem_analysis():
    """제주도 전력 시스템 문제점 분석"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Background gradient
    for i in range(120):
        c = 0.98 - i * 0.001
        ax.add_patch(Rectangle((0, i * 0.1), 16, 0.1, facecolor=(c, c, c + 0.01), edgecolor='none'))

    # Title
    ax.text(8, 11.3, '제주도 전력 시스템 핵심 문제점', fontsize=22, fontweight='bold',
            color=C['slate_900'], ha='center')
    ax.text(8, 10.8, 'Gemini AI Architect Analysis', fontsize=11, color=C['slate_500'], ha='center')

    # Problem 1: Duck Curve
    create_card(ax, 0.5, 6.8, 7.3, 3.7, 'Problem 1: Duck Curve 현상', C['danger'])

    # Duck curve graph
    x_duck = np.linspace(1, 7, 24)
    demand = 1000 + 300 * np.sin((x_duck - 1) / 6 * np.pi)
    solar = 400 * np.maximum(0, np.sin((x_duck - 1) / 6 * np.pi - 0.3))
    net_demand = demand - solar

    y_scale = lambda y: 7.2 + (y - 500) / 800 * 2.5

    ax.fill_between(x_duck, y_scale(np.zeros_like(demand)), y_scale(solar), alpha=0.3, color=C['warning'])
    ax.plot(x_duck, y_scale(demand), '-', color=C['slate_500'], linewidth=2, label='Total Demand')
    ax.plot(x_duck, y_scale(net_demand), '-', color=C['danger'], linewidth=2.5, label='Net Demand')

    ax.text(2, 9.5, 'Duck Curve', fontsize=9, color=C['danger'], fontweight='bold')
    ax.text(4.5, 8, 'Solar', fontsize=8, color=C['warning'])

    ax.text(0.8, 7.0, '• 낮: 태양광 과잉 -> 출력 제한\n• 저녁: 급격한 수요 증가 -> 비상 발전', fontsize=9, color=C['slate_600'])

    # Problem 2: SMP Volatility
    create_card(ax, 8.2, 6.8, 7.3, 3.7, 'Problem 2: SMP 변동성 증가', C['warning'])

    # SMP volatility chart
    x_smp = np.linspace(8.6, 15, 30)
    smp = 100 + 30 * np.sin(np.linspace(0, 6*np.pi, 30)) + np.random.randn(30) * 15
    smp = np.clip(smp, 50, 180)

    y_smp = lambda y: 7.2 + (y - 50) / 150 * 2.5

    ax.plot(x_smp, y_smp(smp), '-', color=C['warning'], linewidth=2)
    ax.fill_between(x_smp, y_smp(np.full_like(smp, 50)), y_smp(smp), alpha=0.2, color=C['warning'])

    ax.axhline(y_smp(100), xmin=0.54, xmax=0.96, linestyle='--', color=C['slate_400'], linewidth=1)
    ax.text(15.2, y_smp(100), 'Avg', fontsize=7, color=C['slate_400'])

    ax.text(8.5, 7.0, '• 재생에너지 불확실성 -> 가격 예측 난이도 증가\n• 중소 발전소 수익성 리스크 확대', fontsize=9, color=C['slate_600'])

    # Problem 3: Grid Constraints
    create_card(ax, 0.5, 3.0, 7.3, 3.5, 'Problem 3: 계통 제약 (출력 제어)', C['purple'])

    # Curtailment illustration
    ax.add_patch(FancyBboxPatch((1, 3.5), 2.5, 2.2, boxstyle="round,rounding_size=0.02",
                                facecolor=C['purple_light'], edgecolor=C['purple'], linewidth=2))
    ax.text(2.25, 5.3, 'KPX\n출력 제어\n명령', fontsize=9, ha='center', color=C['purple'], fontweight='bold')

    ax.annotate('', xy=(4.5, 4.6), xytext=(3.6, 4.6),
                arrowprops=dict(arrowstyle='->', color=C['danger'], lw=2))

    ax.add_patch(FancyBboxPatch((4.7, 3.8), 2.5, 1.8, boxstyle="round,rounding_size=0.02",
                                facecolor=C['danger_light'], edgecolor=C['danger'], linewidth=2))
    ax.text(5.95, 4.7, '발전소\n강제 정지', fontsize=9, ha='center', color=C['danger'], fontweight='bold')

    ax.text(0.8, 3.2, '• 계통 안정성 우선 -> 민간 발전소 손실\n• 예측 불가능한 수익 감소', fontsize=9, color=C['slate_600'])

    # Problem 4: Information Asymmetry
    create_card(ax, 8.2, 3.0, 7.3, 3.5, 'Problem 4: 정보 비대칭', C['primary'])

    # Information flow
    ax.add_patch(Circle((9.5, 4.5), 0.6, facecolor=C['primary_light'], edgecolor=C['primary'], linewidth=2))
    ax.text(9.5, 4.5, 'KPX', fontsize=10, ha='center', va='center', color=C['primary'], fontweight='bold')

    ax.add_patch(Circle((14, 5.2), 0.5, facecolor=C['slate_100'], edgecolor=C['slate_400'], linewidth=1.5))
    ax.text(14, 5.2, 'SME', fontsize=8, ha='center', va='center', color=C['slate_600'])

    ax.add_patch(Circle((14, 3.8), 0.5, facecolor=C['slate_100'], edgecolor=C['slate_400'], linewidth=1.5))
    ax.text(14, 3.8, 'SME', fontsize=8, ha='center', va='center', color=C['slate_600'])

    ax.annotate('', xy=(13.4, 5.0), xytext=(10.2, 4.7),
                arrowprops=dict(arrowstyle='->', color=C['slate_400'], lw=1.5, connectionstyle='arc3,rad=0.2'))
    ax.annotate('', xy=(13.4, 4.0), xytext=(10.2, 4.3),
                arrowprops=dict(arrowstyle='->', color=C['slate_400'], lw=1.5, connectionstyle='arc3,rad=-0.2'))

    ax.text(11.5, 5.3, '제한된 정보', fontsize=8, color=C['slate_500'])

    ax.text(8.5, 3.2, '• 중소 발전소: 시장 정보 부족\n• 입찰 전략 수립의 어려움', fontsize=9, color=C['slate_600'])

    # Bottom: Key Insight from Gemini
    create_card(ax, 0.5, 0.3, 15, 2.4, 'Gemini AI Architect 핵심 인사이트', C['secondary'])

    insights = [
        '"단순 수요 예측을 넘어 계통 제약(Grid Constraint)을 고려한 예측이 필수"',
        '"MAPE 10%의 평균 성능보다 30%+ 극단적 오차(Tail Risk) 관리가 더 중요"',
        '"예측만으로는 부족, 실시간 자원 최적화(Asset Optimization)까지 확장 필요"',
    ]

    for i, insight in enumerate(insights):
        ax.text(1, 2.1 - i * 0.55, f'• {insight}', fontsize=9, color=C['slate_700'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_problem_analysis.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("01_problem_analysis.png generated")


def create_02_solution_value():
    """솔루션 가치 제안"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Gradient background
    for i in range(60):
        c = 0.12 + i * 0.003
        ax.add_patch(Rectangle((0, 12 - i * 0.2), 16, 0.2, facecolor=(c, c, c + 0.05), edgecolor='none'))
    ax.add_patch(Rectangle((0, 0), 16, 6, facecolor=C['slate_50'], edgecolor='none'))

    # Title
    ax.text(8, 11.3, 'AI 입찰 보조 시스템', fontsize=24, fontweight='bold', color='white', ha='center')
    ax.text(8, 10.7, 'KPX & 중소 친환경 발전소를 위한 솔루션', fontsize=12, color=C['slate_300'], ha='center')

    # Value cards
    values = [
        {
            'title': 'KPX 가치',
            'icon': 'K',
            'color': C['primary'],
            'items': [
                '예비력 최적화 (3~5% 절감)',
                '계통 안정성 사전 예측',
                '피크 수요 24시간 사전 대비',
                '운영 의사결정 투명화',
            ]
        },
        {
            'title': '중소 발전소 가치',
            'icon': 'S',
            'color': C['secondary'],
            'items': [
                '최적 입찰가 추천',
                '낙찰 확률 예측 (78%+)',
                '출력 제어 리스크 사전 알림',
                '수익성 시나리오 분석',
            ]
        },
        {
            'title': '기술 차별점',
            'icon': 'T',
            'color': C['purple'],
            'items': [
                'Quantile Regression 불확실성',
                'XAI 기반 예측 근거 설명',
                '80% 신뢰구간 (82.5% 커버리지)',
                'Walk-forward CV 검증',
            ]
        },
    ]

    for i, v in enumerate(values):
        x = 0.8 + i * 5.1
        y = 6.5

        # Card
        add_shadow(ax, x, y, 4.8, 3.5)
        card = FancyBboxPatch((x, y), 4.8, 3.5, boxstyle="round,rounding_size=0.05",
                              facecolor=C['white'], edgecolor=C['slate_200'], linewidth=1, zorder=1)
        ax.add_patch(card)

        # Icon circle
        ax.add_patch(Circle((x + 0.7, y + 2.8), 0.4, facecolor=v['color'], edgecolor='none', zorder=2))
        ax.text(x + 0.7, y + 2.8, v['icon'], fontsize=14, color='white', ha='center', va='center',
                fontweight='bold', zorder=3)

        # Title
        ax.text(x + 1.3, y + 2.8, v['title'], fontsize=13, fontweight='bold', color=C['slate_800'],
                va='center', zorder=2)

        # Items
        for j, item in enumerate(v['items']):
            ax.text(x + 0.5, y + 2.0 - j * 0.45, f'• {item}', fontsize=9, color=C['slate_600'], zorder=2)

    # Performance metrics
    create_card(ax, 0.8, 0.5, 14.4, 5.7, '핵심 성능 지표', C['primary'])

    metrics = [
        ('MAPE', '10.68%', '실제 2년 데이터 검증', C['primary']),
        ('신뢰구간', '82.5%', '80% 구간 커버리지', C['secondary']),
        ('MAE', '11.27원', 'SMP 예측 오차', C['warning']),
        ('파라미터', '172K', '경량화 모델', C['purple']),
    ]

    for i, (name, value, desc, color) in enumerate(metrics):
        x = 1.5 + i * 3.5
        y = 3.5

        # Metric box
        ax.add_patch(FancyBboxPatch((x, y), 3, 1.8, boxstyle="round,rounding_size=0.03",
                                    facecolor=color, alpha=0.1, edgecolor=color, linewidth=2, zorder=2))

        ax.text(x + 1.5, y + 1.5, value, fontsize=20, fontweight='bold', color=color, ha='center', zorder=3)
        ax.text(x + 1.5, y + 0.9, name, fontsize=11, color=C['slate_700'], ha='center', zorder=3)
        ax.text(x + 1.5, y + 0.4, desc, fontsize=8, color=C['slate_500'], ha='center', zorder=3)

    # Bottom: ROI
    ax.text(8, 1.5, '예상 ROI: 예비력 최적화 연간 수십억 원 절감 가능', fontsize=11,
            color=C['slate_600'], ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_solution_value.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("02_solution_value.png generated")


def create_03_dashboard_features():
    """대시보드 핵심 기능"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Background
    ax.add_patch(Rectangle((0, 0), 16, 12, facecolor=C['slate_100'], edgecolor='none'))

    # Title
    ax.text(8, 11.4, '대시보드 핵심 기능', fontsize=22, fontweight='bold', color=C['slate_900'], ha='center')
    ax.text(8, 10.9, '실시간 예측 + XAI 분석 + 입찰 지원', fontsize=11, color=C['slate_500'], ha='center')

    # Feature 1: Real-time Prediction
    create_card(ax, 0.5, 6.5, 5, 4.1, '1. 실시간 수요 예측', C['primary'])

    # Mini chart
    x_chart = np.linspace(1, 5, 24)
    y_pred = 1 + 0.5 * np.sin(np.linspace(0, 2*np.pi, 24))
    y_scale = lambda y: 7 + y * 1.5

    ax.fill_between(x_chart, y_scale(y_pred - 0.2), y_scale(y_pred + 0.2), alpha=0.3, color=C['primary'])
    ax.plot(x_chart, y_scale(y_pred), '-', color=C['primary'], linewidth=2)

    features_1 = ['24시간 사전 예측', 'Quantile 신뢰구간', '실시간 업데이트 (5분)']
    for i, f in enumerate(features_1):
        ax.text(0.8, 7.2 - i * 0.35, f'✓ {f}', fontsize=9, color=C['slate_600'])

    # Feature 2: SMP Prediction
    create_card(ax, 5.7, 6.5, 5, 4.1, '2. SMP 예측 & 입찰 지원', C['secondary'])

    # Quantile bars
    times_q = ['09시', '12시', '15시', '18시']
    for i, t in enumerate(times_q):
        x_bar = 6.2 + i * 1.1
        q10, q50, q90 = 0.3 + i*0.1, 0.5 + i*0.12, 0.8 + i*0.08
        y_bar = lambda y: 8.5 + y * 1.5

        ax.plot([x_bar, x_bar], [y_bar(q10), y_bar(q90)], '-', color=C['secondary'], linewidth=4)
        ax.add_patch(Rectangle((x_bar-0.1, y_bar(q50)-0.05), 0.2, 0.1, facecolor=C['warning'], zorder=3))
        ax.text(x_bar, 8.35, t, fontsize=7, ha='center', color=C['slate_500'])

    features_2 = ['최적 입찰가 추천', '낙찰 확률 78%+', '리스크 분석']
    for i, f in enumerate(features_2):
        ax.text(6, 7.8 - i * 0.35, f'✓ {f}', fontsize=9, color=C['slate_600'])

    # Feature 3: XAI Analysis
    create_card(ax, 10.9, 6.5, 4.6, 4.1, '3. XAI 설명 가능성', C['purple'])

    # Feature importance bars
    features_xai = [('기온', 0.85), ('일사량', 0.72), ('전일수요', 0.68), ('시간대', 0.55)]
    for i, (name, imp) in enumerate(features_xai):
        y_bar = 9.6 - i * 0.5
        ax.add_patch(Rectangle((11.2, y_bar - 0.12), imp * 3, 0.25, facecolor=C['purple'], alpha=0.7))
        ax.text(11.1, y_bar, name, fontsize=8, ha='right', va='center', color=C['slate_600'])

    features_3 = ['Attention 시각화', '자연어 근거 설명']
    for i, f in enumerate(features_3):
        ax.text(11.2, 7.4 - i * 0.35, f'✓ {f}', fontsize=9, color=C['slate_600'])

    # Feature 4: Scenario Analysis
    create_card(ax, 0.5, 2.0, 5, 4.2, '4. 시나리오 분석', C['warning'])

    scenarios = [('기본', 1.0, C['slate_400']), ('폭염', 1.15, C['danger']), ('한파', 0.9, C['primary'])]
    for i, (name, mult, color) in enumerate(scenarios):
        x_s = np.linspace(1, 5, 12)
        y_s = 0.5 + 0.3 * np.sin(np.linspace(0, 2*np.pi, 12)) * mult
        y_scale_s = lambda y: 3.2 + y * 2
        ax.plot(x_s, y_scale_s(y_s), '-', color=color, linewidth=2, label=name)
        ax.text(5.1, y_scale_s(y_s[-1]), name, fontsize=7, color=color, va='center')

    features_4 = ['What-if 시뮬레이션', '민감도 분석', '피크 리스크 예측']
    for i, f in enumerate(features_4):
        ax.text(0.8, 2.7 - i * 0.35, f'✓ {f}', fontsize=9, color=C['slate_600'])

    # Feature 5: Alert System
    create_card(ax, 5.7, 2.0, 5, 4.2, '5. 실시간 알림', C['danger'])

    alerts = [
        ('피크 수요 예상', '15:00', C['warning']),
        ('출력 제어 가능성', '14:30', C['danger']),
        ('예비율 정상', '현재', C['secondary']),
    ]

    for i, (msg, time, color) in enumerate(alerts):
        y_a = 5.5 - i * 0.7
        ax.add_patch(Circle((6.1, y_a), 0.15, facecolor=color, edgecolor='none'))
        ax.text(6.4, y_a, msg, fontsize=9, va='center', color=C['slate_700'])
        ax.text(10.4, y_a, time, fontsize=8, va='center', ha='right', color=C['slate_500'])

    features_5 = ['Slack/Email 연동', '임계값 기반 트리거']
    for i, f in enumerate(features_5):
        ax.text(6, 2.7 - i * 0.35, f'✓ {f}', fontsize=9, color=C['slate_600'])

    # Feature 6: API Integration
    create_card(ax, 10.9, 2.0, 4.6, 4.2, '6. API 연동', C['slate_700'])

    ax.text(11.2, 5.5, 'FastAPI REST API', fontsize=10, fontweight='bold', color=C['slate_700'])
    ax.text(11.2, 5.1, 'POST /predict\nGET /models\nPOST /explain', fontsize=9, color=C['slate_600'],
            family='monospace')

    features_6 = ['100ms 이하 응답', '1,436 테스트 통과', 'Docker 지원']
    for i, f in enumerate(features_6):
        ax.text(11.2, 3.5 - i * 0.35, f'✓ {f}', fontsize=9, color=C['slate_600'])

    # Bottom CTA
    ax.add_patch(FancyBboxPatch((3, 0.3), 10, 1.3, boxstyle="round,rounding_size=0.03",
                                facecolor=C['primary'], edgecolor='none'))
    ax.text(8, 0.95, '3개월 Pilot 테스트로 실제 효과 검증', fontsize=14, fontweight='bold',
            color='white', ha='center')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '03_dashboard_features.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("03_dashboard_features.png generated")


def create_04_sme_bidding_flow():
    """중소 발전소 입찰 프로세스"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Background
    ax.add_patch(Rectangle((0, 0), 16, 10, facecolor=C['white'], edgecolor='none'))

    # Title
    ax.text(8, 9.5, '중소 친환경 발전소 입찰 최적화 프로세스', fontsize=20, fontweight='bold',
            color=C['slate_900'], ha='center')

    # Process flow
    steps = [
        {'title': '1. 발전 예측', 'desc': '기상 데이터 기반\n발전량 예측', 'color': C['primary']},
        {'title': '2. SMP 예측', 'desc': 'Quantile 기반\n가격 분포 예측', 'color': C['secondary']},
        {'title': '3. 리스크 분석', 'desc': '출력 제어 확률\n피크 리스크 평가', 'color': C['warning']},
        {'title': '4. 입찰 추천', 'desc': '최적 입찰가\n낙찰 확률 제시', 'color': C['purple']},
    ]

    for i, step in enumerate(steps):
        x = 1.5 + i * 3.8
        y = 6

        # Step box
        add_shadow(ax, x, y, 3.3, 2.5)
        ax.add_patch(FancyBboxPatch((x, y), 3.3, 2.5, boxstyle="round,rounding_size=0.04",
                                    facecolor=C['white'], edgecolor=step['color'], linewidth=2, zorder=1))

        # Number circle
        ax.add_patch(Circle((x + 1.65, y + 2.1), 0.35, facecolor=step['color'], edgecolor='none', zorder=2))
        ax.text(x + 1.65, y + 2.1, str(i+1), fontsize=14, color='white', ha='center', va='center',
                fontweight='bold', zorder=3)

        ax.text(x + 1.65, y + 1.5, step['title'].split('. ')[1], fontsize=11, fontweight='bold',
                color=C['slate_800'], ha='center', zorder=2)
        ax.text(x + 1.65, y + 0.7, step['desc'], fontsize=9, color=C['slate_600'], ha='center', zorder=2)

        # Arrow
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 3.6, y + 1.25), xytext=(x + 3.3, y + 1.25),
                       arrowprops=dict(arrowstyle='->', color=C['slate_400'], lw=2))

    # Result box
    create_card(ax, 2, 1.5, 12, 3.8, '입찰 결과 시뮬레이션', C['secondary'])

    # Before/After comparison
    ax.text(4, 4.5, 'Before (수동 입찰)', fontsize=11, fontweight='bold', color=C['danger'], ha='center')
    ax.text(4, 4.0, '낙찰률: 45%', fontsize=10, color=C['slate_600'], ha='center')
    ax.text(4, 3.6, '평균 수익: -5%', fontsize=10, color=C['danger'], ha='center')
    ax.text(4, 3.2, '출력 제어 손실: 높음', fontsize=10, color=C['slate_600'], ha='center')

    ax.plot([7, 7], [2, 4.8], '--', color=C['slate_300'], linewidth=1)
    ax.text(7, 4.9, 'AI 도입', fontsize=9, color=C['slate_500'], ha='center')

    ax.text(10, 4.5, 'After (AI 입찰 보조)', fontsize=11, fontweight='bold', color=C['secondary'], ha='center')
    ax.text(10, 4.0, '낙찰률: 78%', fontsize=10, color=C['secondary'], ha='center', fontweight='bold')
    ax.text(10, 3.6, '평균 수익: +12%', fontsize=10, color=C['secondary'], ha='center', fontweight='bold')
    ax.text(10, 3.2, '출력 제어 손실: 사전 대응', fontsize=10, color=C['slate_600'], ha='center')

    # Key message
    ax.text(7, 2.0, '"AI가 추천한 시간대에 입찰하여 낙찰 확률 78%, 연간 수익 12% 향상"',
            fontsize=10, color=C['slate_700'], ha='center', style='italic')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '04_sme_bidding_flow.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("04_sme_bidding_flow.png generated")


def create_05_implementation_summary():
    """구현 설명서 요약"""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Background
    ax.add_patch(Rectangle((0, 0), 16, 12, facecolor=C['slate_50'], edgecolor='none'))

    # Title
    ax.text(8, 11.4, '시스템 구현 설명서 (요약)', fontsize=22, fontweight='bold', color=C['slate_900'], ha='center')

    # Tech Stack
    create_card(ax, 0.5, 8, 5, 3.2, 'Tech Stack', C['primary'])
    stack = [
        ('Backend', 'Python 3.13, FastAPI'),
        ('ML', 'PyTorch, LSTM, Quantile'),
        ('Dashboard', 'Streamlit'),
        ('Infra', 'Docker, GitHub Actions'),
    ]
    for i, (cat, tech) in enumerate(stack):
        ax.text(0.9, 10.5 - i * 0.6, f'{cat}:', fontsize=9, color=C['slate_600'], fontweight='bold')
        ax.text(2.3, 10.5 - i * 0.6, tech, fontsize=9, color=C['slate_700'])

    # Model Architecture
    create_card(ax, 5.7, 8, 5, 3.2, 'Model Architecture', C['secondary'])
    arch = [
        ('Input', '48h sequence, 20 features'),
        ('Encoder', 'BiLSTM (64 hidden)'),
        ('Output', 'Quantile (10%, 50%, 90%)'),
        ('XAI', 'Attention + SHAP'),
    ]
    for i, (layer, desc) in enumerate(arch):
        ax.text(6.1, 10.5 - i * 0.6, f'{layer}:', fontsize=9, color=C['slate_600'], fontweight='bold')
        ax.text(7.5, 10.5 - i * 0.6, desc, fontsize=9, color=C['slate_700'])

    # Performance
    create_card(ax, 10.9, 8, 4.6, 3.2, 'Performance', C['purple'])
    perf = [
        ('MAPE', '10.68%'),
        ('Coverage', '82.5%'),
        ('Latency', '<100ms'),
        ('Tests', '1,436 passed'),
    ]
    for i, (metric, value) in enumerate(perf):
        ax.text(11.3, 10.5 - i * 0.6, f'{metric}:', fontsize=9, color=C['slate_600'], fontweight='bold')
        ax.text(12.8, 10.5 - i * 0.6, value, fontsize=9, color=C['purple'], fontweight='bold')

    # API Endpoints
    create_card(ax, 0.5, 4.3, 7.5, 3.4, 'API Endpoints', C['slate_700'])
    endpoints = [
        ('GET', '/health', 'System health check'),
        ('POST', '/predict', 'Demand prediction'),
        ('POST', '/smp/predict', 'SMP forecast'),
        ('POST', '/explain', 'XAI explanation'),
        ('GET', '/scenarios', 'Scenario analysis'),
    ]
    for i, (method, path, desc) in enumerate(endpoints):
        y = 7.0 - i * 0.55
        color = C['secondary'] if method == 'GET' else C['primary']
        ax.add_patch(FancyBboxPatch((0.8, y - 0.15), 0.6, 0.35, boxstyle="round,rounding_size=0.02",
                                    facecolor=color, edgecolor='none'))
        ax.text(1.1, y, method, fontsize=8, color='white', ha='center', va='center', fontweight='bold')
        ax.text(1.6, y, path, fontsize=9, color=C['slate_800'], va='center', family='monospace')
        ax.text(4.5, y, desc, fontsize=9, color=C['slate_500'], va='center')

    # Key Features Implementation
    create_card(ax, 8.2, 4.3, 7.3, 3.4, 'Key Features', C['warning'])
    features = [
        'Quantile Regression으로 80% 신뢰구간 제공',
        'Walk-forward CV (5-fold) 시계열 검증',
        'Attention 기반 XAI 설명 가능성',
        'Noise Injection (2%) 로버스트성 강화',
        'ARIMA 앙상블 하이브리드 접근',
    ]
    for i, f in enumerate(features):
        ax.text(8.5, 7.0 - i * 0.55, f'• {f}', fontsize=9, color=C['slate_700'])

    # Deployment Guide
    create_card(ax, 0.5, 0.5, 15, 3.5, 'Quick Start Guide', C['secondary'])

    code = '''# 1. 설치
pip install -r requirements.txt

# 2. 모델 학습
python -m src.smp.models.train_smp_advanced

# 3. 대시보드 실행
streamlit run src/dashboard/app_v2.py --server.port 8502

# 4. API 서버 실행
uvicorn api.main:app --host 0.0.0.0 --port 8000'''

    ax.add_patch(FancyBboxPatch((0.8, 0.8), 7.5, 2.8, boxstyle="round,rounding_size=0.02",
                                facecolor=C['slate_800'], edgecolor='none'))
    ax.text(1, 3.3, code, fontsize=8, color=C['slate_100'], family='monospace', va='top')

    # Contact
    ax.text(12.5, 2.5, 'Contact', fontsize=11, fontweight='bold', color=C['slate_700'])
    ax.text(12.5, 2.0, 'GitHub: kiminbean', fontsize=9, color=C['slate_600'])
    ax.text(12.5, 1.6, 'power-demand-forecast', fontsize=9, color=C['slate_600'])
    ax.text(12.5, 1.2, 'v2.1.0', fontsize=9, color=C['slate_500'])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '05_implementation_summary.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("05_implementation_summary.png generated")


def main():
    print("\n" + "=" * 60)
    print("  KPX Sales Materials Generation")
    print("=" * 60 + "\n")

    create_01_problem_analysis()
    create_02_solution_value()
    create_03_dashboard_features()
    create_04_sme_bidding_flow()
    create_05_implementation_summary()

    print("\n" + "=" * 60)
    print(f"  All materials generated!")
    print(f"  Location: {OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
