"""
MODEL-009: TFT Attention 시각화 도구
===================================

학습된 TFT 모델의 Attention weight 시각화

주요 기능:
1. Attention Heatmap 생성
2. 시간대별 중요도 시각화
3. 피처별 기여도 시각화
4. 인터랙티브 플롯 (Plotly)
5. 결과 저장 기능

Author: Claude Code
Date: 2024-12
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

import torch
import torch.nn as nn


# ============================================================
# Attention Heatmap
# ============================================================

def plot_attention_heatmap(
    attention_weights: np.ndarray,
    encoder_length: int = 48,
    decoder_length: int = 24,
    figsize: Tuple[int, int] = (14, 10),
    cmap: str = 'viridis',
    title: str = 'Temporal Attention Heatmap',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Attention weight Heatmap 시각화

    Args:
        attention_weights: Attention 가중치 (seq_len, seq_len) 또는 (batch, seq_len, seq_len)
        encoder_length: Encoder 시퀀스 길이
        decoder_length: Decoder 시퀀스 길이
        figsize: Figure 크기
        cmap: 컬러맵
        title: 제목
        save_path: 저장 경로 (optional)

    Returns:
        matplotlib Figure
    """
    # 배치 차원 처리
    if attention_weights.ndim == 3:
        # 평균 또는 첫 번째 샘플 사용
        attention_weights = attention_weights.mean(axis=0)

    total_len = encoder_length + decoder_length

    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    im = ax.imshow(
        attention_weights,
        cmap=cmap,
        aspect='auto',
        interpolation='nearest'
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)

    # 축 레이블
    ax.set_xlabel('Key (Past + Future)', fontsize=12)
    ax.set_ylabel('Query (Past + Future)', fontsize=12)
    ax.set_title(title, fontsize=14)

    # Encoder/Decoder 구분선
    if encoder_length < total_len:
        ax.axvline(x=encoder_length - 0.5, color='red', linestyle='--', linewidth=2, label='Encoder|Decoder')
        ax.axhline(y=encoder_length - 0.5, color='red', linestyle='--', linewidth=2)

    # 틱 레이블 (간략화)
    tick_step = max(1, total_len // 10)
    ax.set_xticks(range(0, total_len, tick_step))
    ax.set_yticks(range(0, total_len, tick_step))
    ax.set_xticklabels([f't-{encoder_length-i}' if i < encoder_length else f't+{i-encoder_length+1}'
                        for i in range(0, total_len, tick_step)], rotation=45)
    ax.set_yticklabels([f't-{encoder_length-i}' if i < encoder_length else f't+{i-encoder_length+1}'
                        for i in range(0, total_len, tick_step)])

    ax.legend(loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_attention_by_horizon(
    attention_weights: np.ndarray,
    encoder_length: int = 48,
    decoder_length: int = 24,
    horizons: List[int] = None,
    figsize: Tuple[int, int] = (16, 4),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    예측 horizon별 Attention 분포

    Args:
        attention_weights: Attention 가중치 (seq_len, seq_len)
        encoder_length: Encoder 길이
        decoder_length: Decoder 길이
        horizons: 시각화할 horizon 리스트 (default: [1, 6, 12, 24])
        figsize: Figure 크기
        save_path: 저장 경로

    Returns:
        matplotlib Figure
    """
    if horizons is None:
        horizons = [1, 6, 12, min(24, decoder_length)]
        horizons = [h for h in horizons if h <= decoder_length]

    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)

    n_horizons = len(horizons)
    fig, axes = plt.subplots(1, n_horizons, figsize=figsize)

    if n_horizons == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        # Decoder의 h번째 시점에서의 attention
        decoder_idx = encoder_length + h - 1
        if decoder_idx < attention_weights.shape[0]:
            attention = attention_weights[decoder_idx, :encoder_length]

            ax.bar(range(encoder_length), attention, alpha=0.7)
            ax.set_xlabel('Past Time Steps')
            ax.set_ylabel('Attention Weight')
            ax.set_title(f'Horizon +{h}h')

            # 피크 강조
            peak_idx = np.argmax(attention)
            ax.axvline(x=peak_idx, color='red', linestyle='--', alpha=0.7)

    plt.suptitle('Attention Distribution by Prediction Horizon', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================
# Variable Importance (VSN Weights)
# ============================================================

def plot_variable_importance(
    variable_weights: np.ndarray,
    feature_names: List[str],
    title: str = 'Variable Selection Weights',
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Variable Selection Network 피처 중요도 시각화

    Args:
        variable_weights: VSN 가중치 (num_features,) 또는 (batch, seq_len, num_features)
        feature_names: 피처 이름 리스트
        title: 제목
        top_k: 상위 k개 피처만 표시
        figsize: Figure 크기
        save_path: 저장 경로

    Returns:
        matplotlib Figure
    """
    # 차원 축소
    if variable_weights.ndim > 1:
        variable_weights = variable_weights.mean(axis=tuple(range(variable_weights.ndim - 1)))

    # 상위 k개 선택
    n_features = len(feature_names)
    top_k = min(top_k, n_features)

    indices = np.argsort(variable_weights)[::-1][:top_k]
    top_weights = variable_weights[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=figsize)

    # 가로 막대 그래프
    colors = plt.cm.viridis(top_weights / top_weights.max())
    bars = ax.barh(range(top_k), top_weights, color=colors)

    ax.set_yticks(range(top_k))
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance Weight')
    ax.set_title(title, fontsize=14)

    # 값 표시
    for i, (bar, weight) in enumerate(zip(bars, top_weights)):
        ax.text(weight + 0.01, i, f'{weight:.3f}', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_temporal_variable_importance(
    variable_weights: np.ndarray,
    feature_names: List[str],
    encoder_length: int = 48,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    시간에 따른 피처 중요도 변화 시각화

    Args:
        variable_weights: (seq_len, num_features) 또는 (batch, seq_len, num_features)
        feature_names: 피처 이름
        encoder_length: Encoder 길이
        figsize: Figure 크기
        save_path: 저장 경로

    Returns:
        matplotlib Figure
    """
    # 배치 평균
    if variable_weights.ndim == 3:
        variable_weights = variable_weights.mean(axis=0)

    seq_len, n_features = variable_weights.shape

    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    im = ax.imshow(
        variable_weights.T,
        aspect='auto',
        cmap='YlOrRd',
        interpolation='nearest'
    )

    # 축 설정
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Feature')
    ax.set_title('Temporal Variable Importance', fontsize=14)

    # Y축 피처 이름
    if n_features <= 20:
        ax.set_yticks(range(n_features))
        ax.set_yticklabels(feature_names, fontsize=8)
    else:
        # 너무 많으면 일부만 표시
        step = n_features // 15
        ax.set_yticks(range(0, n_features, step))
        ax.set_yticklabels([feature_names[i] for i in range(0, n_features, step)], fontsize=8)

    # Encoder/Decoder 구분선
    if encoder_length < seq_len:
        ax.axvline(x=encoder_length - 0.5, color='white', linestyle='--', linewidth=2)

    plt.colorbar(im, ax=ax, label='Importance')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================
# Interactive Plotly Visualizations
# ============================================================

def plot_attention_heatmap_interactive(
    attention_weights: np.ndarray,
    encoder_length: int = 48,
    decoder_length: int = 24,
    title: str = 'Interactive Attention Heatmap',
    save_path: Optional[str] = None
) -> Optional[go.Figure]:
    """
    Plotly 인터랙티브 Attention Heatmap

    Args:
        attention_weights: Attention 가중치
        encoder_length: Encoder 길이
        decoder_length: Decoder 길이
        title: 제목
        save_path: HTML 저장 경로

    Returns:
        Plotly Figure (또는 None if Plotly 없음)
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(axis=0)

    total_len = encoder_length + decoder_length

    # 레이블 생성
    labels = [f't-{encoder_length-i}' if i < encoder_length else f't+{i-encoder_length+1}'
              for i in range(total_len)]

    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=labels,
        y=labels,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='Query: %{y}<br>Key: %{x}<br>Weight: %{z:.4f}<extra></extra>'
    ))

    # Encoder/Decoder 구분선
    fig.add_vline(x=encoder_length - 0.5, line_color='red', line_dash='dash')
    fig.add_hline(y=encoder_length - 0.5, line_color='red', line_dash='dash')

    fig.update_layout(
        title=title,
        xaxis_title='Key (Past → Future)',
        yaxis_title='Query (Past → Future)',
        width=900,
        height=800
    )

    if save_path:
        fig.write_html(save_path)

    return fig


def plot_prediction_with_uncertainty(
    predictions: np.ndarray,
    targets: np.ndarray,
    quantiles: List[float] = None,
    time_index: np.ndarray = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Quantile 예측 및 불확실성 시각화

    Args:
        predictions: 예측값 (n_samples, num_quantiles) 또는 (n_samples, seq_len, num_quantiles)
        targets: 실제값 (n_samples,) 또는 (n_samples, seq_len)
        quantiles: Quantile 리스트 (default: [0.1, 0.5, 0.9])
        time_index: 시간 인덱스
        figsize: Figure 크기
        save_path: 저장 경로

    Returns:
        matplotlib Figure
    """
    if quantiles is None:
        quantiles = [0.1, 0.5, 0.9]

    # Flatten if multi-dimensional
    if predictions.ndim == 3:
        n_samples, seq_len, n_q = predictions.shape
        predictions = predictions.reshape(-1, n_q)
        targets = targets.reshape(-1)

    n_points = len(targets)

    if time_index is None:
        time_index = np.arange(n_points)

    # 표시할 포인트 수 제한
    max_points = 500
    if n_points > max_points:
        step = n_points // max_points
        indices = range(0, n_points, step)
        predictions = predictions[indices]
        targets = targets[indices]
        time_index = time_index[indices]
        n_points = len(targets)

    fig, ax = plt.subplots(figsize=figsize)

    # Quantile 영역 (불확실성)
    if len(quantiles) >= 3:
        lower = predictions[:, 0]
        median = predictions[:, len(quantiles) // 2]
        upper = predictions[:, -1]

        ax.fill_between(
            time_index, lower, upper,
            alpha=0.3, color='blue', label=f'{quantiles[0]*100:.0f}%-{quantiles[-1]*100:.0f}% CI'
        )

        ax.plot(time_index, median, 'b-', linewidth=1.5, label='Median Prediction')
    else:
        ax.plot(time_index, predictions[:, 0], 'b-', linewidth=1.5, label='Prediction')

    # 실제값
    ax.plot(time_index, targets, 'k--', linewidth=1, alpha=0.7, label='Actual')

    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Predictions with Uncertainty')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


# ============================================================
# 종합 시각화
# ============================================================

def create_attention_report(
    model: nn.Module,
    data_loader,
    feature_names: Dict[str, List[str]],
    device: torch.device,
    output_dir: str,
    encoder_length: int = 48,
    decoder_length: int = 24
) -> Dict[str, str]:
    """
    종합 Attention 리포트 생성

    Args:
        model: TFT 모델
        data_loader: 데이터 로더
        feature_names: {'known': [...], 'unknown': [...]}
        device: 디바이스
        output_dir: 출력 디렉토리
        encoder_length: Encoder 길이
        decoder_length: Decoder 길이

    Returns:
        Dict: 생성된 파일 경로들
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved_files = {}

    # 샘플 데이터 추출
    with torch.no_grad():
        for batch in data_loader:
            known, unknown, targets, static = batch
            known = known.to(device)
            unknown = unknown.to(device)
            if static is not None:
                static = static.to(device)

            output = model(known, unknown, static, return_attention=True)
            break

    # 1. Attention Heatmap
    if 'attention_weights' in output:
        att_weights = output['attention_weights'].cpu().numpy()

        fig = plot_attention_heatmap(
            att_weights,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
            save_path=str(output_dir / 'attention_heatmap.png')
        )
        plt.close(fig)
        saved_files['attention_heatmap'] = str(output_dir / 'attention_heatmap.png')

        # Horizon별 분석
        fig = plot_attention_by_horizon(
            att_weights,
            encoder_length=encoder_length,
            decoder_length=decoder_length,
            save_path=str(output_dir / 'attention_by_horizon.png')
        )
        plt.close(fig)
        saved_files['attention_by_horizon'] = str(output_dir / 'attention_by_horizon.png')

        # Interactive (if Plotly available)
        if PLOTLY_AVAILABLE:
            fig_plotly = plot_attention_heatmap_interactive(
                att_weights,
                encoder_length=encoder_length,
                decoder_length=decoder_length,
                save_path=str(output_dir / 'attention_interactive.html')
            )
            saved_files['attention_interactive'] = str(output_dir / 'attention_interactive.html')

    # 2. Variable Importance
    if 'encoder_variable_weights' in output:
        encoder_weights = output['encoder_variable_weights'].cpu().numpy()
        all_features = feature_names.get('unknown', []) + feature_names.get('known', [])

        if len(all_features) == encoder_weights.shape[-1]:
            fig = plot_variable_importance(
                encoder_weights,
                all_features,
                title='Encoder Variable Importance',
                save_path=str(output_dir / 'encoder_variable_importance.png')
            )
            plt.close(fig)
            saved_files['encoder_variable_importance'] = str(output_dir / 'encoder_variable_importance.png')

    # 3. Prediction with Uncertainty
    predictions = output['predictions'].cpu().numpy()
    targets_np = targets.cpu().numpy()

    fig = plot_prediction_with_uncertainty(
        predictions[:100],  # 첫 100개 샘플
        targets_np[:100],
        save_path=str(output_dir / 'prediction_uncertainty.png')
    )
    plt.close(fig)
    saved_files['prediction_uncertainty'] = str(output_dir / 'prediction_uncertainty.png')

    print(f"Attention report saved to: {output_dir}")
    for name, path in saved_files.items():
        print(f"  - {name}: {path}")

    return saved_files


# ============================================================
# 테스트
# ============================================================

def test_visualizations():
    """시각화 함수 테스트"""
    print("Testing attention visualizations...")

    np.random.seed(42)

    encoder_length = 48
    decoder_length = 24
    total_len = encoder_length + decoder_length
    n_features = 10

    # 1. Attention Heatmap
    attention = np.random.rand(total_len, total_len)
    # Causal mask 적용 (lower triangular)
    attention = np.tril(attention)
    attention = attention / attention.sum(axis=1, keepdims=True)

    fig = plot_attention_heatmap(attention, encoder_length, decoder_length)
    plt.close(fig)
    print("  Attention heatmap: OK")

    # 2. Horizon별 Attention
    fig = plot_attention_by_horizon(attention, encoder_length, decoder_length)
    plt.close(fig)
    print("  Attention by horizon: OK")

    # 3. Variable Importance
    weights = np.random.rand(n_features)
    weights = weights / weights.sum()
    feature_names = [f'feature_{i}' for i in range(n_features)]

    fig = plot_variable_importance(weights, feature_names)
    plt.close(fig)
    print("  Variable importance: OK")

    # 4. Prediction with Uncertainty
    n_samples = 100
    predictions = np.random.randn(n_samples, 3) * 10 + 100
    predictions = np.sort(predictions, axis=1)  # quantiles 정렬
    targets = np.random.randn(n_samples) * 10 + 100

    fig = plot_prediction_with_uncertainty(predictions, targets)
    plt.close(fig)
    print("  Prediction uncertainty: OK")

    print("\nAll visualization tests passed!")


if __name__ == "__main__":
    test_visualizations()
