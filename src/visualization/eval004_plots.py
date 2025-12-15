"""
EVAL-004 Results Visualization
==============================

Conditional Model 실험 결과 시각화

Author: Hybrid Agent Pipeline (Claude + Gemini)
Date: 2024-12
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 한글 폰트 설정
plt.rcParams['font.family'] = ['AppleGothic', 'NanumGothic', 'Malgun Gothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).parent.parent.parent


def load_results():
    """실험 결과 로드"""
    results_dir = PROJECT_ROOT / "results" / "metrics"

    # Full trial results
    with open(results_dir / "conditional_experiment_report.json", 'r') as f:
        full_results = json.load(f)

    # Winter test results
    with open(results_dir / "winter_test_report.json", 'r') as f:
        winter_results = json.load(f)

    return full_results, winter_results


def plot_full_trial_comparison(results: dict, ax: plt.Axes):
    """Full Trial 결과 비교 차트"""
    models = ['demand_only', 'weather_full', 'conditional_hard', 'conditional_soft']
    labels = ['Demand\nOnly', 'Weather\nFull', 'Conditional\nHard', 'Conditional\nSoft']

    summary = results['summary']

    mapes = [summary[m]['mape_mean'] for m in models]
    stds = [summary[m]['mape_std'] for m in models]

    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

    bars = ax.bar(labels, mapes, yerr=stds, capsize=5, color=colors,
                  edgecolor='black', linewidth=1.2, alpha=0.85)

    # 값 표시
    for bar, mape, std in zip(bars, mapes, stds):
        height = bar.get_height()
        ax.annotate(f'{mape:.2f}%\n±{std:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.1),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Full Trial Results (5 trials, 50 epochs)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(mapes) + max(stds) + 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Best model highlight
    best_idx = np.argmin(mapes)
    bars[best_idx].set_edgecolor('#27ae60')
    bars[best_idx].set_linewidth(3)


def plot_winter_test_comparison(results: dict, ax: plt.Axes):
    """겨울철 테스트 결과 비교 차트"""
    models = ['demand_only', 'weather_full', 'conditional_hard', 'conditional_soft']
    labels = ['Demand\nOnly', 'Weather\nFull', 'Conditional\nHard', 'Conditional\nSoft']

    mapes = [results['results'][m]['MAPE'] for m in models]

    colors = ['#3498db', '#e74c3c', '#9b59b6', '#2ecc71']

    bars = ax.bar(labels, mapes, color=colors, edgecolor='black',
                  linewidth=1.2, alpha=0.85)

    # 값 표시
    for bar, mape in zip(bars, mapes):
        height = bar.get_height()
        ax.annotate(f'{mape:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height + 0.1),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('MAPE (%)', fontsize=12, fontweight='bold')
    ax.set_title('Winter Test Results (2022.12 ~ 2023.02)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(mapes) + 1)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Best model highlight
    best_idx = np.argmin(mapes)
    bars[best_idx].set_edgecolor('#27ae60')
    bars[best_idx].set_linewidth(3)


def plot_improvement_comparison(full_results: dict, winter_results: dict, ax: plt.Axes):
    """개선율 비교 차트"""
    models = ['weather_full', 'conditional_hard', 'conditional_soft']
    labels = ['Weather Full', 'Conditional Hard', 'Conditional Soft']

    # Full trial improvements
    full_summary = full_results['summary']
    baseline_full = full_summary['demand_only']['mape_mean']
    full_improvements = [(baseline_full - full_summary[m]['mape_mean']) / baseline_full * 100
                         for m in models]

    # Winter improvements
    winter_res = winter_results['results']
    baseline_winter = winter_res['demand_only']['MAPE']
    winter_improvements = [(baseline_winter - winter_res[m]['MAPE']) / baseline_winter * 100
                           for m in models]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, full_improvements, width, label='Full Trial',
                   color='#3498db', edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x + width/2, winter_improvements, width, label='Winter Test',
                   color='#e74c3c', edgecolor='black', alpha=0.85)

    # 값 표시
    for bar, imp in zip(bars1, full_improvements):
        height = bar.get_height()
        y_pos = height + 0.3 if height >= 0 else height - 0.8
        ax.annotate(f'{imp:+.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold', color='#2980b9')

    for bar, imp in zip(bars2, winter_improvements):
        height = bar.get_height()
        y_pos = height + 0.3 if height >= 0 else height - 0.8
        ax.annotate(f'{imp:+.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold', color='#c0392b')

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Improvement vs demand_only (%)', fontsize=12, fontweight='bold')
    ax.set_title('MAPE Improvement Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 개선 영역 하이라이트
    ax.axhspan(0, max(max(full_improvements), max(winter_improvements)) + 1,
               alpha=0.1, color='green')
    ax.axhspan(min(min(full_improvements), min(winter_improvements)) - 1, 0,
               alpha=0.1, color='red')


def plot_r2_comparison(full_results: dict, winter_results: dict, ax: plt.Axes):
    """R² 비교 차트"""
    models = ['demand_only', 'weather_full', 'conditional_hard', 'conditional_soft']
    labels = ['Demand\nOnly', 'Weather\nFull', 'Cond.\nHard', 'Cond.\nSoft']

    # Full trial R²
    full_summary = full_results['summary']
    full_r2 = [full_summary[m]['r2_mean'] for m in models]

    # Winter R²
    winter_res = winter_results['results']
    winter_r2 = [winter_res[m]['R2'] for m in models]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, full_r2, width, label='Full Trial',
                   color='#9b59b6', edgecolor='black', alpha=0.85)
    bars2 = ax.bar(x + width/2, winter_r2, width, label='Winter Test',
                   color='#f39c12', edgecolor='black', alpha=0.85)

    # 값 표시
    for bar, r2 in zip(bars1, full_r2):
        ax.annotate(f'{r2:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    for bar, r2 in zip(bars2, winter_r2):
        ax.annotate(f'{r2:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.set_ylim(0.6, 0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_model_selection_diagram(ax: plt.Axes):
    """모델 선택 다이어그램"""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Conditional Model Selection Flow',
            ha='center', va='center', fontsize=14, fontweight='bold')

    # Input box
    input_box = mpatches.FancyBboxPatch((3.5, 8), 3, 1, boxstyle="round,pad=0.05",
                                         facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, 8.5, 'Input Data', ha='center', va='center', color='white',
            fontsize=11, fontweight='bold')

    # Arrow down
    ax.annotate('', xy=(5, 7), xytext=(5, 8),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Season check diamond
    diamond = mpatches.RegularPolygon((5, 6), numVertices=4, radius=0.8,
                                       orientation=np.pi/4,
                                       facecolor='#f39c12', edgecolor='black', linewidth=2)
    ax.add_patch(diamond)
    ax.text(5, 6, 'Winter?', ha='center', va='center', fontsize=10, fontweight='bold')

    # Yes arrow (down)
    ax.annotate('', xy=(5, 4.5), xytext=(5, 5.2),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(5.3, 4.85, 'Yes', fontsize=9, color='green', fontweight='bold')

    # No arrow (right)
    ax.annotate('', xy=(8, 6), xytext=(5.8, 6),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(6.9, 6.3, 'No', fontsize=9, color='red', fontweight='bold')

    # Soft blend box
    soft_box = mpatches.FancyBboxPatch((3.5, 3.5), 3, 1, boxstyle="round,pad=0.05",
                                        facecolor='#2ecc71', edgecolor='black', linewidth=2)
    ax.add_patch(soft_box)
    ax.text(5, 4, 'Soft Blend\n(weather weight)', ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')

    # Demand only box (right)
    demand_box = mpatches.FancyBboxPatch((7, 5.5), 2.5, 1, boxstyle="round,pad=0.05",
                                          facecolor='#3498db', edgecolor='black', linewidth=2)
    ax.add_patch(demand_box)
    ax.text(8.25, 6, 'Demand\nOnly', ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')

    # Results boxes
    result1_box = mpatches.FancyBboxPatch((3.5, 1.5), 3, 1.2, boxstyle="round,pad=0.05",
                                           facecolor='#27ae60', edgecolor='black', linewidth=2)
    ax.add_patch(result1_box)
    ax.text(5, 2.1, 'Winter: +2.23%', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')

    result2_box = mpatches.FancyBboxPatch((7, 3.5), 2.5, 1.2, boxstyle="round,pad=0.05",
                                           facecolor='#95a5a6', edgecolor='black', linewidth=2)
    ax.add_patch(result2_box)
    ax.text(8.25, 4.1, 'Baseline', ha='center', va='center',
            color='white', fontsize=10, fontweight='bold')

    # Arrows to results
    ax.annotate('', xy=(5, 2.7), xytext=(5, 3.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(8.25, 4.7), xytext=(8.25, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))


def create_eval004_visualization():
    """EVAL-004 시각화 생성"""
    print("Loading results...")
    full_results, winter_results = load_results()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Layout: 2x2 + 1 bottom
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Generate plots
    print("Generating plots...")
    plot_full_trial_comparison(full_results, ax1)
    plot_winter_test_comparison(winter_results, ax2)
    plot_improvement_comparison(full_results, winter_results, ax3)
    plot_r2_comparison(full_results, winter_results, ax4)

    # Main title
    fig.suptitle('EVAL-004: Conditional Model Experiment Results',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_dir = PROJECT_ROOT / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "eval004_conditional_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    # Create separate flow diagram
    fig2, ax_flow = plt.subplots(figsize=(10, 8))
    plot_model_selection_diagram(ax_flow)
    fig2.suptitle('EVAL-004: Model Selection Strategy', fontsize=16, fontweight='bold')

    flow_path = output_dir / "eval004_model_flow.png"
    plt.savefig(flow_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {flow_path}")

    plt.close('all')

    return output_path, flow_path


def create_summary_table():
    """요약 테이블 이미지 생성"""
    full_results, winter_results = load_results()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Table data
    columns = ['Model', 'Full Trial\nMAPE', 'Full Trial\nR²',
               'Winter\nMAPE', 'Winter\nR²', 'Improvement\n(Winter)']

    models = ['demand_only', 'weather_full', 'conditional_hard', 'conditional_soft']
    model_names = ['Demand Only', 'Weather Full', 'Conditional Hard', 'Conditional Soft']

    full_summary = full_results['summary']
    winter_res = winter_results['results']
    baseline_winter = winter_res['demand_only']['MAPE']

    data = []
    for m, name in zip(models, model_names):
        full_mape = f"{full_summary[m]['mape_mean']:.2f}%±{full_summary[m]['mape_std']:.2f}"
        full_r2 = f"{full_summary[m]['r2_mean']:.4f}"
        winter_mape = f"{winter_res[m]['MAPE']:.2f}%"
        winter_r2 = f"{winter_res[m]['R2']:.4f}"
        improvement = (baseline_winter - winter_res[m]['MAPE']) / baseline_winter * 100
        imp_str = f"{improvement:+.2f}%" if m != 'demand_only' else '-'

        data.append([name, full_mape, full_r2, winter_mape, winter_r2, imp_str])

    # Create table
    table = ax.table(cellText=data, colLabels=columns, loc='center',
                     cellLoc='center', colColours=['#3498db']*6)

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Highlight best results
    table[(4, 3)].set_facecolor('#d5f5e3')  # Best winter MAPE
    table[(4, 4)].set_facecolor('#d5f5e3')  # Best winter R²
    table[(4, 5)].set_facecolor('#d5f5e3')  # Best improvement

    ax.set_title('EVAL-004 Results Summary Table', fontsize=16, fontweight='bold', pad=20)

    output_dir = PROJECT_ROOT / "results" / "figures"
    output_path = output_dir / "eval004_summary_table.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    plt.close()
    return output_path


if __name__ == '__main__':
    print("=" * 60)
    print("EVAL-004 Visualization")
    print("=" * 60)

    main_path, flow_path = create_eval004_visualization()
    table_path = create_summary_table()

    print("\n" + "=" * 60)
    print("All visualizations saved!")
    print("=" * 60)
