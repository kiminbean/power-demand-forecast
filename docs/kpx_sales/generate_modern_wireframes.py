#!/usr/bin/env python3
"""
KPX 영업용 모던 와이어프레임 생성 스크립트
더 세련되고 현대적인 UI 디자인
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, PathPatch
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import numpy as np
from pathlib import Path as FilePath

# 한글 폰트 설정 (macOS)
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 출력 디렉토리
OUTPUT_DIR = FilePath(__file__).parent / "wireframes"
OUTPUT_DIR.mkdir(exist_ok=True)

# 모던 색상 팔레트 (Tailwind CSS 영감)
COLORS = {
    # Primary
    'primary_50': '#EFF6FF',
    'primary_100': '#DBEAFE',
    'primary_200': '#BFDBFE',
    'primary_500': '#3B82F6',
    'primary_600': '#2563EB',
    'primary_700': '#1D4ED8',
    'primary_900': '#1E3A8A',

    # Secondary (Slate)
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

    # Accent colors
    'emerald_400': '#34D399',
    'emerald_500': '#10B981',
    'amber_400': '#FBBF24',
    'amber_500': '#F59E0B',
    'rose_400': '#FB7185',
    'rose_500': '#F43F5E',
    'violet_500': '#8B5CF6',

    # Backgrounds
    'bg_gradient_start': '#F8FAFC',
    'bg_gradient_end': '#E2E8F0',
    'card_bg': '#FFFFFF',
    'sidebar_bg': '#0F172A',
}


def add_shadow(ax, x, y, width, height, offset=0.05, alpha=0.1):
    """카드에 그림자 효과 추가"""
    shadow = FancyBboxPatch(
        (x + offset, y - offset), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor='black',
        edgecolor='none',
        alpha=alpha,
        zorder=0
    )
    ax.add_patch(shadow)


def create_card(ax, x, y, width, height, title='', shadow=True):
    """모던 카드 컴포넌트"""
    if shadow:
        add_shadow(ax, x, y, width, height, offset=0.08, alpha=0.08)

    card = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=COLORS['card_bg'],
        edgecolor=COLORS['slate_200'],
        linewidth=1,
        zorder=1
    )
    ax.add_patch(card)

    if title:
        ax.text(x + 0.3, y + height - 0.4, title,
                fontsize=11, fontweight='bold', color=COLORS['slate_700'],
                zorder=2)


def create_kpi_card(ax, x, y, title, value, subtitle='', color=None, icon=''):
    """KPI 카드 컴포넌트"""
    width, height = 3.2, 1.8

    add_shadow(ax, x, y, width, height, offset=0.06, alpha=0.06)

    # 카드 배경
    card = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['card_bg'],
        edgecolor=COLORS['slate_100'],
        linewidth=1,
        zorder=1
    )
    ax.add_patch(card)

    # 상단 액센트 바
    if color:
        accent = FancyBboxPatch(
            (x, y + height - 0.08), width, 0.08,
            boxstyle="round,pad=0,rounding_size=0.02",
            facecolor=color,
            edgecolor='none',
            zorder=2
        )
        ax.add_patch(accent)

    # 아이콘 배경
    if icon:
        icon_bg = Circle((x + 0.5, y + height - 0.6), 0.25,
                        facecolor=COLORS['primary_50'], edgecolor='none', zorder=2)
        ax.add_patch(icon_bg)
        ax.text(x + 0.5, y + height - 0.6, icon, fontsize=12, ha='center', va='center', zorder=3)

    # 텍스트
    ax.text(x + 0.9, y + height - 0.5, title,
            fontsize=9, color=COLORS['slate_500'], zorder=3)
    ax.text(x + 0.9, y + height - 1.1, value,
            fontsize=18, fontweight='bold', color=color or COLORS['slate_800'], zorder=3)
    if subtitle:
        ax.text(x + 0.9, y + height - 1.5, subtitle,
                fontsize=8, color=COLORS['slate_400'], zorder=3)


def create_sidebar_item(ax, x, y, text, active=False):
    """사이드바 메뉴 아이템"""
    width, height = 2.8, 0.6

    if active:
        bg = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=COLORS['primary_600'],
            edgecolor='none',
            alpha=0.9,
            zorder=2
        )
        ax.add_patch(bg)
        text_color = 'white'
    else:
        text_color = COLORS['slate_400']

    ax.text(x + 0.3, y + height/2, text,
            fontsize=10, color=text_color, va='center', zorder=3)


def draw_line_chart(ax, x_start, y_start, width, height, show_confidence=True):
    """세련된 라인 차트"""
    n_points = 24
    x_data = np.linspace(x_start + 0.3, x_start + width - 0.3, n_points)

    # 데이터 생성
    base = np.sin(np.linspace(0, 2*np.pi, n_points)) * 0.3 + 0.5
    actual = base + np.random.randn(n_points) * 0.05
    predicted = base

    y_scale = lambda y: y_start + 0.5 + y * (height - 1)

    # 신뢰구간
    if show_confidence:
        upper = predicted + 0.1
        lower = predicted - 0.1
        ax.fill_between(x_data, y_scale(lower), y_scale(upper),
                       alpha=0.2, color=COLORS['primary_500'], zorder=3)

    # 그리드 라인
    for i in range(5):
        y_grid = y_start + 0.5 + i * (height - 1) / 4
        ax.plot([x_start + 0.3, x_start + width - 0.3], [y_grid, y_grid],
               '-', color=COLORS['slate_200'], linewidth=0.5, zorder=2)

    # 실측 라인
    ax.plot(x_data, y_scale(actual), 'o-',
           color=COLORS['slate_600'], markersize=3, linewidth=1.5,
           label='실측', zorder=4)

    # 예측 라인
    ax.plot(x_data, y_scale(predicted), '--',
           color=COLORS['emerald_500'], linewidth=2,
           label='예측', zorder=4)


def modern_01_main_dashboard():
    """모던 메인 대시보드"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # 배경 그라데이션 효과
    for i in range(110):
        alpha = 1.0
        color_val = 0.97 - i * 0.001
        ax.add_patch(Rectangle((0, i * 0.1), 18, 0.1,
                               facecolor=(color_val, color_val, color_val + 0.01),
                               edgecolor='none', zorder=0))

    # ===== 사이드바 =====
    sidebar = FancyBboxPatch(
        (0.2, 0.2), 3.2, 10.6,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=COLORS['sidebar_bg'],
        edgecolor='none',
        zorder=1
    )
    ax.add_patch(sidebar)

    # 로고/타이틀
    ax.text(1.8, 10.2, 'JEJU POWER', fontsize=14, fontweight='bold',
            color='white', ha='center', zorder=2)
    ax.text(1.8, 9.8, 'AI Forecast System', fontsize=8,
            color=COLORS['slate_400'], ha='center', zorder=2)

    # 구분선
    ax.plot([0.6, 3.0], [9.4, 9.4], '-', color=COLORS['slate_700'], linewidth=1, zorder=2)

    # 메뉴 아이템
    menu_items = [
        ('Dashboard', True),
        ('Demand Forecast', False),
        ('SMP Prediction', False),
        ('XAI Analysis', False),
        ('Scenarios', False),
        ('Settings', False),
    ]

    for i, (item, active) in enumerate(menu_items):
        create_sidebar_item(ax, 0.4, 8.6 - i * 0.8, item, active)

    # 하단 정보
    ax.text(1.8, 1.0, 'v2.1.0', fontsize=8, color=COLORS['slate_500'],
            ha='center', zorder=2)
    ax.text(1.8, 0.6, 'Last update: 17:30', fontsize=7, color=COLORS['slate_600'],
            ha='center', zorder=2)

    # ===== 헤더 =====
    ax.text(3.8, 10.4, 'Dashboard', fontsize=20, fontweight='bold',
            color=COLORS['slate_800'], zorder=2)
    ax.text(3.8, 9.9, '24-hour power demand forecast overview',
            fontsize=10, color=COLORS['slate_500'], zorder=2)

    # 현재 시간
    time_bg = FancyBboxPatch(
        (15, 10.1), 2.6, 0.6,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['slate_100'],
        edgecolor='none',
        zorder=1
    )
    ax.add_patch(time_bg)
    ax.text(16.3, 10.4, '2024-12-18 17:30', fontsize=10,
            color=COLORS['slate_600'], ha='center', zorder=2)

    # ===== KPI 카드 =====
    kpi_data = [
        ('Current Demand', '1,245 MW', 'Real-time', COLORS['primary_600'], 'E'),
        ('Forecast (+1h)', '1,312 MW', '+5.4%', COLORS['emerald_500'], 'T'),
        ('Reserve Rate', '18.5%', 'Normal', COLORS['amber_500'], 'R'),
        ('SMP Forecast', '105.2 W/kWh', '+2.3%', COLORS['violet_500'], 'S'),
    ]

    for i, (title, value, sub, color, icon) in enumerate(kpi_data):
        create_kpi_card(ax, 3.8 + i * 3.5, 7.8, title, value, sub, color, icon)

    # ===== 메인 차트 =====
    create_card(ax, 3.8, 2.8, 10.2, 4.7, '24-Hour Demand Forecast')
    draw_line_chart(ax, 3.8, 2.8, 10.2, 4.2)

    # 범례
    ax.plot([4.5, 5.0], [3.2, 3.2], 'o-', color=COLORS['slate_600'], markersize=3, linewidth=1.5)
    ax.text(5.2, 3.2, 'Actual', fontsize=8, color=COLORS['slate_600'], va='center')

    ax.plot([7, 7.5], [3.2, 3.2], '--', color=COLORS['emerald_500'], linewidth=2)
    ax.text(7.7, 3.2, 'Predicted', fontsize=8, color=COLORS['emerald_500'], va='center')

    ax.add_patch(Rectangle((10, 3.1), 0.5, 0.2, facecolor=COLORS['primary_500'], alpha=0.2))
    ax.text(10.7, 3.2, '80% CI', fontsize=8, color=COLORS['primary_500'], va='center')

    # X축 레이블
    times = ['00:00', '06:00', '12:00', '18:00', '24:00']
    for i, t in enumerate(times):
        x = 4.1 + i * 2.3
        ax.text(x, 3.0, t, fontsize=8, color=COLORS['slate_400'], ha='center')

    # ===== 우측 패널: 알림 =====
    create_card(ax, 14.3, 2.8, 3.3, 4.7, 'Alerts')

    alerts = [
        ('Peak demand expected', '15:00', COLORS['amber_500']),
        ('Reserve rate normal', 'Now', COLORS['emerald_500']),
        ('SMP increase forecast', '09:00', COLORS['rose_500']),
    ]

    for i, (msg, time, color) in enumerate(alerts):
        y = 6.6 - i * 1.0
        # 상태 점
        ax.add_patch(Circle((14.7, y), 0.12, facecolor=color, edgecolor='none', zorder=3))
        ax.text(15.0, y, msg, fontsize=8, color=COLORS['slate_600'], va='center', zorder=3)
        ax.text(17.3, y, time, fontsize=7, color=COLORS['slate_400'], ha='right', va='center', zorder=3)

    # ===== 하단 테이블 =====
    create_card(ax, 3.8, 0.3, 13.8, 2.3, 'Hourly Forecast Details')

    # 테이블 헤더
    headers = ['Time', 'Forecast (MW)', 'Confidence', 'MAPE', 'Status']
    col_widths = [2, 2.5, 2.5, 2, 2]
    x_pos = 4.2

    for i, (h, w) in enumerate(zip(headers, col_widths)):
        ax.text(x_pos, 2.1, h, fontsize=9, fontweight='bold', color=COLORS['slate_600'])
        x_pos += w

    # 헤더 구분선
    ax.plot([4.0, 17.2], [1.95, 1.95], '-', color=COLORS['slate_200'], linewidth=1, zorder=3)

    # 테이블 데이터
    table_data = [
        ('09:00', '1,245', '1,195 ~ 1,295', '8.2%', 'Normal'),
        ('12:00', '1,380', '1,320 ~ 1,440', '9.1%', 'Caution'),
        ('15:00', '1,425', '1,360 ~ 1,490', '10.5%', 'Warning'),
    ]

    status_colors = {'Normal': COLORS['emerald_500'], 'Caution': COLORS['amber_500'], 'Warning': COLORS['rose_500']}

    for row_idx, row in enumerate(table_data):
        y = 1.65 - row_idx * 0.45
        x_pos = 4.2
        for col_idx, (cell, w) in enumerate(zip(row, col_widths)):
            if col_idx == 4:  # Status column
                color = status_colors.get(cell, COLORS['slate_600'])
                ax.add_patch(FancyBboxPatch(
                    (x_pos - 0.1, y - 0.15), 1.2, 0.35,
                    boxstyle="round,pad=0.01,rounding_size=0.02",
                    facecolor=color, alpha=0.15, edgecolor='none', zorder=3
                ))
                ax.text(x_pos + 0.5, y, cell, fontsize=8, color=color, ha='center', va='center', zorder=4)
            else:
                ax.text(x_pos, y, cell, fontsize=8, color=COLORS['slate_600'], va='center', zorder=3)
            x_pos += w

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '01_main_dashboard.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("01_main_dashboard.png generated (Modern)")


def modern_02_smp_prediction():
    """모던 SMP 예측 페이지"""
    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    ax.axis('off')

    # 배경
    for i in range(110):
        color_val = 0.97 - i * 0.001
        ax.add_patch(Rectangle((0, i * 0.1), 18, 0.1,
                               facecolor=(color_val, color_val, color_val + 0.01),
                               edgecolor='none', zorder=0))

    # ===== 사이드바 =====
    sidebar = FancyBboxPatch(
        (0.2, 0.2), 3.2, 10.6,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        facecolor=COLORS['sidebar_bg'],
        edgecolor='none',
        zorder=1
    )
    ax.add_patch(sidebar)

    ax.text(1.8, 10.2, 'JEJU POWER', fontsize=14, fontweight='bold',
            color='white', ha='center', zorder=2)
    ax.text(1.8, 9.8, 'AI Forecast System', fontsize=8,
            color=COLORS['slate_400'], ha='center', zorder=2)

    ax.plot([0.6, 3.0], [9.4, 9.4], '-', color=COLORS['slate_700'], linewidth=1, zorder=2)

    menu_items = [
        ('Dashboard', False),
        ('Demand Forecast', False),
        ('SMP Prediction', True),
        ('XAI Analysis', False),
        ('Scenarios', False),
        ('Settings', False),
    ]

    for i, (item, active) in enumerate(menu_items):
        create_sidebar_item(ax, 0.4, 8.6 - i * 0.8, item, active)

    # ===== 헤더 =====
    ax.text(3.8, 10.4, 'SMP Prediction & Bidding Support', fontsize=20, fontweight='bold',
            color=COLORS['slate_800'], zorder=2)
    ax.text(3.8, 9.9, '48-hour System Marginal Price forecast with uncertainty quantification',
            fontsize=10, color=COLORS['slate_500'], zorder=2)

    # ===== 좌측: SMP 예측 차트 =====
    create_card(ax, 3.8, 5.0, 8.5, 4.7, '48-Hour SMP Forecast')

    # SMP 차트
    n_points = 48
    x_data = np.linspace(4.2, 11.8, n_points)
    smp_base = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, n_points))
    smp_noise = np.random.randn(n_points) * 5
    smp_pred = smp_base + smp_noise
    upper = smp_pred + 15
    lower = smp_pred - 15

    y_scale = lambda y: 5.3 + (y - 70) / 80 * 3.5

    # 그리드
    for val in [80, 100, 120, 140]:
        y = y_scale(val)
        ax.plot([4.2, 11.8], [y, y], '-', color=COLORS['slate_200'], linewidth=0.5, zorder=2)
        ax.text(4.0, y, f'{val}', fontsize=7, ha='right', va='center', color=COLORS['slate_400'])

    # 신뢰구간
    ax.fill_between(x_data, y_scale(lower), y_scale(upper),
                   alpha=0.2, color=COLORS['primary_500'], zorder=3)

    # 예측 라인
    ax.plot(x_data, y_scale(smp_pred), '-',
           color=COLORS['primary_600'], linewidth=2, zorder=4)

    # 범례
    ax.text(5, 5.5, 'Prediction', fontsize=8, color=COLORS['primary_600'])
    ax.text(7, 5.5, '80% CI', fontsize=8, color=COLORS['primary_500'])

    ax.text(4.5, 5.2, 'Today', fontsize=7, color=COLORS['slate_400'])
    ax.text(8.5, 5.2, 'Tomorrow', fontsize=7, color=COLORS['slate_400'])
    ax.plot([8, 8], [5.3, 8.8], '--', color=COLORS['slate_300'], linewidth=1, zorder=3)

    # ===== 우측 상단: 모델 선택 =====
    create_card(ax, 12.6, 7.5, 5, 2.2, 'Model Selection')

    # 기본 모델
    ax.text(12.9, 9.0, 'Basic Model (LSTM)', fontsize=9, color=COLORS['slate_600'], zorder=3)
    toggle_off = FancyBboxPatch(
        (16.2, 8.85), 0.9, 0.35,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['slate_300'],
        edgecolor='none', zorder=3
    )
    ax.add_patch(toggle_off)
    ax.add_patch(Circle((16.45, 9.02), 0.12, facecolor='white', edgecolor='none', zorder=4))

    # 고도화 모델
    ax.text(12.9, 8.3, 'Advanced (Quantile + Attention)', fontsize=9, color=COLORS['slate_800'],
            fontweight='bold', zorder=3)
    toggle_on = FancyBboxPatch(
        (16.2, 8.15), 0.9, 0.35,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        facecolor=COLORS['emerald_500'],
        edgecolor='none', zorder=3
    )
    ax.add_patch(toggle_on)
    ax.add_patch(Circle((16.85, 8.32), 0.12, facecolor='white', edgecolor='none', zorder=4))

    # ===== 우측 중앙: 입찰 추천 =====
    create_card(ax, 12.6, 4.9, 5, 2.4, 'Bidding Recommendation')

    recommendations = [
        ('Optimal Bid', '102.5 W/kWh', COLORS['emerald_500']),
        ('Win Probability', '78.3%', COLORS['primary_600']),
        ('Recommended Qty', '450 MW', COLORS['violet_500']),
    ]

    for i, (label, value, color) in enumerate(recommendations):
        y = 6.7 - i * 0.55
        ax.text(12.9, y, label, fontsize=9, color=COLORS['slate_500'], zorder=3)
        ax.text(17.3, y, value, fontsize=11, fontweight='bold', color=color, ha='right', zorder=3)

    # ===== 좌측 하단: Quantile 분포 =====
    create_card(ax, 3.8, 0.3, 6, 4.5, 'SMP Quantile Distribution')

    times = ['09:00', '12:00', '15:00', '18:00', '21:00']
    for i, t in enumerate(times):
        x = 4.5 + i * 1.0

        # Quantile values (시뮬레이션)
        q10 = 85 + i * 5
        q50 = 100 + i * 8
        q90 = 120 + i * 10

        y_box = lambda y: 0.8 + (y - 80) / 80 * 2.8

        # 수직선 (Q10 ~ Q90)
        ax.plot([x, x], [y_box(q10), y_box(q90)], '-',
               color=COLORS['primary_600'], linewidth=3, zorder=3)

        # Q10, Q90 마커
        ax.plot([x - 0.1, x + 0.1], [y_box(q10), y_box(q10)], '-',
               color=COLORS['primary_600'], linewidth=3, zorder=3)
        ax.plot([x - 0.1, x + 0.1], [y_box(q90), y_box(q90)], '-',
               color=COLORS['primary_600'], linewidth=3, zorder=3)

        # Q50 (중앙값)
        ax.add_patch(Rectangle((x - 0.15, y_box(q50) - 0.1), 0.3, 0.2,
                               facecolor=COLORS['emerald_500'], edgecolor='none', zorder=4))

        # 시간 레이블
        ax.text(x, 0.55, t, fontsize=7, ha='center', color=COLORS['slate_500'])

    # 범례
    ax.text(4.5, 4.3, 'Q90', fontsize=7, color=COLORS['primary_600'])
    ax.text(4.5, 2.5, 'Q50 (Median)', fontsize=7, color=COLORS['emerald_500'])
    ax.text(4.5, 1.0, 'Q10', fontsize=7, color=COLORS['primary_600'])

    # ===== 우측 하단: 리스크 분석 =====
    create_card(ax, 10.1, 0.3, 7.5, 4.5, 'Risk Analysis')

    risk_items = [
        ('Price Volatility', 'Medium', COLORS['amber_500'], 0.6),
        ('Forecast Uncertainty', 'Low', COLORS['emerald_500'], 0.3),
        ('Peak Risk', 'High', COLORS['rose_500'], 0.8),
        ('Supply Shortage', 'Low', COLORS['emerald_500'], 0.2),
    ]

    for i, (label, level, color, pct) in enumerate(risk_items):
        y = 4.0 - i * 0.85
        ax.text(10.5, y, label, fontsize=9, color=COLORS['slate_600'], zorder=3)

        # 프로그레스 바 배경
        bar_bg = FancyBboxPatch(
            (13.5, y - 0.15), 3, 0.35,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=COLORS['slate_100'],
            edgecolor='none', zorder=3
        )
        ax.add_patch(bar_bg)

        # 프로그레스 바
        bar = FancyBboxPatch(
            (13.5, y - 0.15), 3 * pct, 0.35,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color,
            edgecolor='none', zorder=4
        )
        ax.add_patch(bar)

        # 레벨 텍스트
        ax.text(17.0, y, level, fontsize=8, color=color, fontweight='bold', ha='right', zorder=5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '02_smp_prediction.png', dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print("02_smp_prediction.png generated (Modern)")


def main():
    """모던 와이어프레임 생성"""
    print("\n" + "="*60)
    print("  Modern KPX Wireframe Generation")
    print("="*60 + "\n")

    modern_01_main_dashboard()
    modern_02_smp_prediction()

    print("\n" + "="*60)
    print(f"  All modern wireframes generated!")
    print(f"  Location: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
