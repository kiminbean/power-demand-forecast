"""
BiLSTM+Attention v3.1 Model Architecture & Data Flow Diagram
============================================================

SMP 예측 모델의 아키텍처 및 데이터 흐름 다이어그램 생성

Author: Claude Code
Date: 2024-12-24
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def create_model_architecture():
    """모델 아키텍처 다이어그램 생성"""

    fig, ax = plt.subplots(1, 1, figsize=(16, 20))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 20)
    ax.set_aspect('equal')
    ax.axis('off')

    # 색상 정의
    colors = {
        'input': '#4FC3F7',       # 하늘색
        'lstm': '#81C784',        # 연두색
        'attention': '#FFB74D',   # 주황색
        'norm': '#BA68C8',        # 보라색
        'output': '#42A5F5',      # 파란색
        'result': '#EF5350',      # 빨간색
        'arrow': '#37474F',
    }

    def draw_box(x, y, w, h, label, color, fontsize=9, sublabel=None, highlight=False):
        """박스 그리기"""
        edgecolor = '#FFD700' if highlight else '#333333'
        linewidth = 3 if highlight else 2

        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth,
            alpha=0.9
        )
        ax.add_patch(box)

        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.2, label, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')
            ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center',
                   fontsize=fontsize-1, color='white', alpha=0.9)
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')

    def draw_group_box(x, y, w, h, label, color):
        """그룹 박스"""
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color,
            edgecolor='#666666',
            linewidth=1.5,
            alpha=0.15
        )
        ax.add_patch(box)
        ax.text(x + 0.15, y + h - 0.25, label, ha='left', va='top',
               fontsize=10, fontweight='bold', color='#333333')

    def draw_arrow(start, end, color='#37474F', style='->', curved=False):
        """화살표"""
        if curved:
            connectionstyle = 'arc3,rad=0.2'
        else:
            connectionstyle = 'arc3,rad=0'

        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle=style, color=color, lw=1.5,
                                  connectionstyle=connectionstyle))

    # ==========================================================================
    # TITLE
    # ==========================================================================
    ax.text(8, 19.5, 'BiLSTM+Attention v3.1', ha='center', va='center',
           fontsize=16, fontweight='bold', color='#1a237e')
    ax.text(8, 19.0, 'SMP Prediction Model Architecture', ha='center', va='center',
           fontsize=12, color='#666666')

    # ==========================================================================
    # 1. INPUT LAYER
    # ==========================================================================
    draw_group_box(0.5, 16.5, 15, 2, '1. Input Layer', colors['input'])

    draw_box(1, 17, 3, 1, 'Raw SMP Data', colors['input'], sublabel='48h history')
    draw_box(4.5, 17, 3, 1, 'Feature Eng.', colors['input'], sublabel='22 features')
    draw_box(8, 17, 3.5, 1, 'StandardScaler', colors['input'], sublabel='Z-score norm')
    draw_box(12, 17, 3, 1, 'Input Tensor', colors['input'], sublabel='(B, 48, 22)')

    draw_arrow((4, 17.5), (4.5, 17.5))
    draw_arrow((7.5, 17.5), (8, 17.5))
    draw_arrow((11.5, 17.5), (12, 17.5))

    # ==========================================================================
    # 2. BiLSTM ENCODER
    # ==========================================================================
    draw_group_box(0.5, 13.3, 15, 2.7, '2. BiLSTM Encoder (hidden=64, layers=2)', colors['lstm'])

    draw_box(2, 14.2, 5, 1.3, 'Forward LSTM', colors['lstm'], sublabel='seq_len=48')
    draw_box(9, 14.2, 5, 1.3, 'Backward LSTM', colors['lstm'], sublabel='seq_len=48')
    draw_box(5.5, 13.6, 5, 0.8, 'Concatenate', colors['lstm'], fontsize=8, sublabel='output: (B, 48, 128)')

    draw_arrow((8, 16.5), (4.5, 15.5))
    draw_arrow((8, 16.5), (11.5, 15.5))
    draw_arrow((4.5, 14.2), (6.5, 14.4), curved=True)
    draw_arrow((11.5, 14.2), (9.5, 14.4), curved=True)

    # ==========================================================================
    # 3. MULTI-HEAD SELF-ATTENTION
    # ==========================================================================
    draw_group_box(0.5, 9.5, 15, 3.3, '3. Multi-Head Self-Attention (4 heads, d_k=32)', colors['attention'])

    # Q, K, V projections
    draw_box(1.5, 11.5, 2.5, 0.9, 'W_Q', colors['attention'], sublabel='Query')
    draw_box(4.5, 11.5, 2.5, 0.9, 'W_K', colors['attention'], sublabel='Key')
    draw_box(7.5, 11.5, 2.5, 0.9, 'W_V', colors['attention'], sublabel='Value')

    # Attention computation
    draw_box(3, 10.3, 4, 0.9, 'Scaled Dot-Product', colors['attention'], sublabel='QK^T / sqrt(d_k)')
    draw_box(8, 10.3, 3, 0.9, 'Softmax', colors['attention'], sublabel='Attention')
    draw_box(11.5, 10.3, 3, 0.9, 'Context', colors['attention'], sublabel='Attn x V')

    draw_arrow((8, 13.3), (2.75, 12.4))
    draw_arrow((8, 13.3), (5.75, 12.4))
    draw_arrow((8, 13.3), (8.75, 12.4))
    draw_arrow((2.75, 11.5), (4, 11.2), curved=True)
    draw_arrow((5.75, 11.5), (5.5, 11.2), curved=True)
    draw_arrow((7, 10.75), (8, 10.75))
    draw_arrow((8.75, 11.5), (11.5, 11))
    draw_arrow((11, 10.75), (11.5, 10.75))

    # ==========================================================================
    # 4. NORMALIZATION
    # ==========================================================================
    draw_group_box(0.5, 7, 15, 2, '4. Residual + LayerNorm', colors['norm'])

    draw_box(2, 7.3, 3.5, 1.2, 'Residual', colors['norm'], sublabel='LSTM + Attn')
    draw_box(6.5, 7.3, 3, 1.2, 'LayerNorm', colors['norm'], sublabel='Normalize')
    draw_box(10.5, 7.3, 4, 1.2, 'Last Timestep', colors['norm'], sublabel='(B, 128)')

    draw_arrow((8, 9.5), (3.75, 8.5))
    draw_arrow((13, 10.3), (3.75, 8.5), curved=True)
    draw_arrow((5.5, 7.9), (6.5, 7.9))
    draw_arrow((9.5, 7.9), (10.5, 7.9))

    # ==========================================================================
    # 5. OUTPUT HEADS
    # ==========================================================================
    draw_group_box(0.5, 3.5, 15, 3, '5. Output Heads (FC: 128 -> 64 -> 24)', colors['output'])

    draw_box(2, 4.8, 2.5, 1.2, 'Point', colors['output'], sublabel='Main pred', highlight=True)
    draw_box(5, 4.8, 2.5, 1.2, 'Q10 Head', colors['output'], sublabel='Lower 10%')
    draw_box(8, 4.8, 2.5, 1.2, 'Q50 Head', colors['output'], sublabel='Median')
    draw_box(11, 4.8, 2.5, 1.2, 'Q90 Head', colors['output'], sublabel='Upper 90%')

    draw_arrow((12.5, 7), (3.25, 6))
    draw_arrow((12.5, 7), (6.25, 6))
    draw_arrow((12.5, 7), (9.25, 6))
    draw_arrow((12.5, 7), (12.25, 6))

    # ==========================================================================
    # 6. RESULT
    # ==========================================================================
    draw_group_box(0.5, 0.5, 15, 2.5, '6. Prediction Output', colors['result'])

    draw_box(2, 1, 5, 1.5, '24h SMP Forecast', colors['result'],
            sublabel='q10, q50, q90 (won/kWh)', highlight=True)
    draw_box(8, 1, 3, 1.5, 'Attention', colors['result'], sublabel='XAI weights')
    draw_box(11.5, 1, 3, 1.5, 'Metrics', colors['result'], sublabel='MAPE: 7.83%')

    draw_arrow((3.25, 4.8), (3.5, 2.5))
    draw_arrow((6.25, 4.8), (4, 2.5))
    draw_arrow((9.25, 4.8), (4.5, 2.5))
    draw_arrow((12.25, 4.8), (5, 2.5))
    draw_arrow((11, 10.3), (9.5, 2.5), curved=True)

    # ==========================================================================
    # LEGEND
    # ==========================================================================
    ax.text(8, 0.2, 'BiLSTM+Attention v3.1 | MAPE: 7.83% | R2: 0.736 | Coverage: 89.4% | Params: 249,952',
           ha='center', fontsize=9, color='#666666')

    plt.tight_layout()
    plt.savefig('docs/bilstm_attention_v31_architecture.png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("Saved: docs/bilstm_attention_v31_architecture.png")

    plt.savefig('docs/bilstm_attention_v31_architecture.pdf', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("Saved: docs/bilstm_attention_v31_architecture.pdf")

    plt.close()


def create_data_flow_diagram():
    """데이터 플로우 다이어그램 생성"""

    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')

    colors = {
        'data': '#4FC3F7',
        'process': '#81C784',
        'model': '#FFB74D',
        'output': '#EF5350',
        'api': '#BA68C8',
    }

    def draw_box(x, y, w, h, label, color, fontsize=9, sublabel=None):
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color,
            edgecolor='#333333',
            linewidth=2,
            alpha=0.9
        )
        ax.add_patch(box)

        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')
            ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center',
                   fontsize=fontsize-1, color='white', alpha=0.9)
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')

    def draw_arrow(start, end, label=None):
        ax.annotate('', xy=end, xytext=start,
                   arrowprops=dict(arrowstyle='->', color='#37474F', lw=2))
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.15, label, ha='center', fontsize=7, color='#666')

    # Title
    ax.text(9, 11.5, 'BiLSTM+Attention v3.1 Data Flow', ha='center',
           fontsize=14, fontweight='bold', color='#1a237e')

    # Data Sources
    draw_box(0.5, 9, 3, 1.5, 'EPSIS', colors['data'], sublabel='SMP history')
    draw_box(0.5, 7, 3, 1.5, 'KPX Crawler', colors['data'], sublabel='Realtime SMP')
    draw_box(0.5, 5, 3, 1.5, 'KMA Weather', colors['data'], sublabel='Weather data')

    # Processing
    draw_box(5, 7, 3, 2, 'Data Pipeline', colors['process'], sublabel='Load & Validate')
    draw_box(5, 4, 3, 2, 'Feature Eng.', colors['process'], sublabel='22 features')

    # Model
    draw_box(9.5, 5.5, 4, 3, 'BiLSTM+Attention', colors['model'], sublabel='v3.1 Model')

    # Output
    draw_box(14.5, 8, 3, 1.5, 'Q10 Prediction', colors['output'], sublabel='Lower bound')
    draw_box(14.5, 6, 3, 1.5, 'Q50 Prediction', colors['output'], sublabel='Main forecast')
    draw_box(14.5, 4, 3, 1.5, 'Q90 Prediction', colors['output'], sublabel='Upper bound')

    # API
    draw_box(9.5, 1, 4, 2, 'SMP API', colors['api'], sublabel='/api/v1/smp-forecast')
    draw_box(14.5, 1, 3, 2, 'AI Optimizer', colors['api'], sublabel='Bid optimization')

    # Arrows
    draw_arrow((3.5, 9.75), (5, 8.5))
    draw_arrow((3.5, 7.75), (5, 7.75))
    draw_arrow((3.5, 5.75), (5, 5.5))
    draw_arrow((8, 7.5), (9.5, 7))
    draw_arrow((8, 5), (9.5, 6))
    draw_arrow((13.5, 8), (14.5, 8.75))
    draw_arrow((13.5, 7), (14.5, 6.75))
    draw_arrow((13.5, 6), (14.5, 4.75))
    draw_arrow((11.5, 5.5), (11.5, 3))
    draw_arrow((13.5, 2), (14.5, 2))

    plt.tight_layout()
    plt.savefig('docs/bilstm_attention_v31_dataflow.png', dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print("Saved: docs/bilstm_attention_v31_dataflow.png")

    plt.close()


if __name__ == '__main__':
    print("Generating BiLSTM+Attention v3.1 diagrams...")
    create_model_architecture()
    create_data_flow_diagram()
    print("Done!")
