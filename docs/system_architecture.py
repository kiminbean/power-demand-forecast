"""
RE-BMS 시스템 아키텍처 다이어그램 생성 (Updated 2024-12-24)
- SMP Prediction API 추가
- AI Bidding Optimizer 추가
- BiLSTM+Attention 모델 추가
- 추가 크롤러 포함
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'NanumGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def create_system_architecture():
    """전체 시스템 아키텍처 다이어그램 생성"""

    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    ax.set_aspect('equal')
    ax.axis('off')

    # 색상 정의
    colors = {
        'client': '#4FC3F7',      # 하늘색
        'gateway': '#81C784',     # 연두색
        'service': '#FFB74D',     # 주황색
        'crawler': '#BA68C8',     # 보라색
        'external': '#EF5350',    # 빨간색
        'model': '#42A5F5',       # 파란색
        'storage': '#78909C',     # 회색
        'arrow': '#37474F',       # 진회색
    }

    def draw_box(x, y, w, h, label, color, fontsize=10, sublabel=None):
        """박스 그리기"""
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color,
            edgecolor='#333333',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)

        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')
            ax.text(x + w/2, y + h/2 - 0.25, sublabel, ha='center', va='center',
                   fontsize=fontsize-2, color='white', alpha=0.9)
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')

    def draw_group_box(x, y, w, h, label, color):
        """그룹 박스 (배경) 그리기"""
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.3",
            facecolor=color,
            edgecolor='#666666',
            linewidth=1.5,
            alpha=0.15
        )
        ax.add_patch(box)
        ax.text(x + 0.2, y + h - 0.3, label, ha='left', va='top',
               fontsize=11, fontweight='bold', color='#333333')

    def draw_arrow(start, end, color='#37474F', style='->'):
        """화살표 그리기"""
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle=style, color=color, lw=2,
                                  connectionstyle='arc3,rad=0'))

    # ============================================================
    # TITLE
    # ============================================================
    ax.text(10, 14.5, 'RE-BMS 제주 전력 수요 예측 시스템', ha='center', va='center',
           fontsize=18, fontweight='bold', color='#1a237e')
    ax.text(10, 14.0, 'System Architecture Overview (Updated 2024-12-24)', ha='center', va='center',
           fontsize=12, color='#666666')

    # ============================================================
    # 1. CLIENT LAYER (맨 위)
    # ============================================================
    draw_group_box(0.5, 12, 19, 1.7, '📱 Client Layer', colors['client'])
    draw_box(2.5, 12.3, 3.5, 1, 'Mobile App', colors['client'], sublabel='Expo React Native')
    draw_box(7.5, 12.3, 3.5, 1, 'Web Dashboard', colors['client'], sublabel='React (v5-v8)')
    draw_box(12.5, 12.3, 3.5, 1, 'API Client', colors['client'], sublabel='REST/JSON')

    # ============================================================
    # 2. API GATEWAY (두 번째)
    # ============================================================
    draw_group_box(0.5, 9.8, 19, 1.7, '🚀 API Gateway', colors['gateway'])
    draw_box(2.5, 10.1, 3.5, 1, 'Vite Proxy', colors['gateway'], sublabel='Dev Server')
    draw_box(7.5, 10.1, 4, 1, 'FastAPI', colors['gateway'], sublabel='Port 8000')
    draw_box(13, 10.1, 3, 1, 'CORS', colors['gateway'], sublabel='Middleware')

    # ============================================================
    # 3. SERVICE LAYER (세 번째) - 확장됨
    # ============================================================
    draw_group_box(0.5, 6.8, 19, 2.5, '⚙️ Service Layer', colors['service'])
    draw_box(1, 7.6, 2.8, 1.3, 'v6_routes', colors['service'], sublabel='Dashboard API')
    draw_box(4.2, 7.6, 2.8, 1.3, 'smp_routes', colors['service'], sublabel='SMP Predict API')
    draw_box(7.4, 7.6, 2.8, 1.3, 'realtime_api', colors['service'], sublabel='Data Client')
    draw_box(10.6, 7.6, 2.8, 1.3, 'ai_bidding', colors['service'], sublabel='Bid Optimizer')
    draw_box(13.8, 7.6, 2.5, 1.3, 'renewable', colors['service'], sublabel='Solar/Wind')
    draw_box(16.7, 7.6, 2.5, 1.3, 'service.py', colors['service'], sublabel='Demand Pred')

    # ============================================================
    # 4. WEB CRAWLERS (왼쪽 아래)
    # ============================================================
    draw_group_box(0.5, 3.8, 7.5, 2.5, '🔍 Web Crawlers', colors['crawler'])
    draw_box(1, 4.6, 2, 1.2, 'SMP', colors['crawler'], sublabel='Crawler')
    draw_box(3.3, 4.6, 2, 1.2, 'Jeju', colors['crawler'], sublabel='Realtime')
    draw_box(5.6, 4.6, 2, 1.2, 'KMA', colors['crawler'], sublabel='Weather')
    # 추가 크롤러 (작은 박스)
    draw_box(1, 4.1, 2, 0.4, 'EPSIS', colors['crawler'], fontsize=8)
    draw_box(3.3, 4.1, 2, 0.4, 'FuelCost', colors['crawler'], fontsize=8)

    # ============================================================
    # 5. ML MODELS (오른쪽 아래) - SMP 모델 추가
    # ============================================================
    draw_group_box(8.5, 3.8, 11, 2.5, '🧠 ML Models', colors['model'])
    draw_box(9, 4.5, 2.2, 1.3, 'LightGBM', colors['model'], sublabel='Solar')
    draw_box(11.5, 4.5, 2.2, 1.3, 'Estimator', colors['model'], sublabel='Wind')
    draw_box(14, 4.5, 2.2, 1.3, 'BiLSTM', colors['model'], sublabel='Demand')
    # SMP 모델 (핵심 - 강조)
    box = FancyBboxPatch(
        (16.5, 4.5), 2.5, 1.3,
        boxstyle="round,pad=0.02,rounding_size=0.2",
        facecolor='#1565C0',  # 더 진한 파란색
        edgecolor='#FFD700',  # 금색 테두리
        linewidth=3,
        alpha=0.95
    )
    ax.add_patch(box)
    ax.text(17.75, 5.25, 'BiLSTM+Att', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')
    ax.text(17.75, 4.9, 'SMP v3.1', ha='center', va='center',
           fontsize=9, color='#FFD700', fontweight='bold')

    # ============================================================
    # 6. EXTERNAL DATA SOURCES (맨 아래)
    # ============================================================
    draw_group_box(0.5, 0.8, 19, 2.5, '📊 External Data Sources', colors['external'])
    draw_box(1.5, 1.3, 3, 1.5, 'KPX', colors['external'], sublabel='전력거래소')
    draw_box(5, 1.3, 3, 1.5, 'KMA', colors['external'], sublabel='기상청')
    draw_box(8.5, 1.3, 3, 1.5, 'EPSIS', colors['external'], sublabel='전력통계')
    draw_box(12, 1.3, 3, 1.5, 'CSV', colors['storage'], sublabel='Local Data')
    draw_box(15.5, 1.3, 3, 1.5, 'PT/PKL', colors['storage'], sublabel='Model Files')

    # ============================================================
    # ARROWS (데이터 흐름)
    # ============================================================
    # Client -> Gateway
    draw_arrow((4.25, 12.3), (4.25, 11.1))
    draw_arrow((9.25, 12.3), (9.5, 11.1))
    draw_arrow((14.25, 12.3), (14.5, 11.1))

    # Gateway -> Service
    draw_arrow((9.5, 10.1), (9.5, 9.3))

    # Service -> Crawlers
    draw_arrow((8.8, 7.6), (5.6, 6.3))

    # Service -> ML Models
    draw_arrow((12, 7.6), (14, 6.3))

    # Crawlers -> External
    draw_arrow((2, 4.1), (2, 2.8))
    draw_arrow((4.3, 4.1), (6.5, 2.8))

    # ML Models -> Storage
    draw_arrow((17.75, 4.5), (17, 2.8))

    # ============================================================
    # LEGEND
    # ============================================================
    legend_y = 0.3
    ax.text(1, legend_y, '⭐ Key: BiLSTM+Attention v3.1 = SMP 예측 핵심 모델 (MAPE 7.83%)',
           fontsize=10, color='#1565C0', fontweight='bold')

    # Save
    plt.tight_layout()
    plt.savefig('docs/system_architecture_v2.png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("✅ Saved: docs/system_architecture_v2.png")

    # Also save as PDF
    plt.savefig('docs/system_architecture_v2.pdf', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("✅ Saved: docs/system_architecture_v2.pdf")

    plt.close()

if __name__ == '__main__':
    create_system_architecture()
