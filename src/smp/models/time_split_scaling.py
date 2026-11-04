"""Leak-free scaling and temporal split utilities for SMP training scripts.

시계열 학습 스크립트에서 스케일러가 훈련 구간에만 fit되도록 강제하는 헬퍼.
val/test 통계가 스케일러에 유출되는 것을 구조적으로 차단한다.
"""

from typing import Tuple

import numpy as np


def fit_transform_time_split(
    scaler,
    features: np.ndarray,
    train_end: int,
) -> np.ndarray:
    """Fit the scaler on the training slice only, then transform the full array.

    Args:
        scaler: sklearn-style scaler (e.g. StandardScaler).
        features: Full feature array in temporal order, shape (n, d).
        train_end: Exclusive end index of the training slice.

    Returns:
        Scaled copy of ``features``. The scaler statistics come from
        ``features[:train_end]`` only; val/test rows are transformed,
        never fitted.

    Raises:
        ValueError: If ``train_end`` is outside (0, len(features)].

    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> import numpy as np
        >>> X = np.arange(20.0).reshape(10, 2)
        >>> scaled = fit_transform_time_split(StandardScaler(), X, train_end=7)
        >>> scaled.shape == X.shape
        True
    """
    if not 0 < train_end <= len(features):
        raise ValueError(
            f"train_end must be in (0, {len(features)}], got {train_end}"
        )

    scaled = np.empty(features.shape, dtype=np.float64)
    # 스케일러는 훈련 구간에만 fit — val/test 통계 누출 방지
    scaled[:train_end] = scaler.fit_transform(features[:train_end])
    scaled[train_end:] = scaler.transform(features[train_end:])
    return scaled


def temporal_split_bounds(
    n: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> Tuple[int, int]:
    """Compute (train_end, val_end) indices for a chronological split.

    Args:
        n: Total number of samples in temporal order.
        train_frac: Fraction of samples in the training slice.
        val_frac: Fraction of samples in the validation slice
            (test takes the remaining tail).

    Returns:
        Tuple ``(train_end, val_end)`` such that
        ``[:train_end]`` is train, ``[train_end:val_end]`` is val,
        ``[val_end:]`` is test.

    Raises:
        ValueError: If fractions are invalid or leave no val/test samples.

    Example:
        >>> temporal_split_bounds(100, 0.8, 0.1)
        (80, 90)
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(
            f"Require 0 < train_frac < 1, 0 < val_frac < 1, "
            f"train_frac + val_frac < 1, got {train_frac}, {val_frac}"
        )
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if not 0 < train_end < val_end < n:
        raise ValueError(
            f"Split too small for n={n}: train_end={train_end}, val_end={val_end}"
        )
    return train_end, val_end


def carve_validation_from_train(n_train: int, val_frac: float = 0.1) -> int:
    """Return the validation size carved off the tail of a training slice.

    Args:
        n_train: Number of samples in the training slice.
        val_frac: Fraction of the training slice held out as validation.

    Returns:
        Validation size ``int(n_train * val_frac)``.

    Raises:
        ValueError: If ``val_frac`` is outside [0, 1).
    """
    if not 0 <= val_frac < 1:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    return int(n_train * val_frac)
