"""Sequence-consistency tests for demand lag features.

시퀀스 예측 구조(src/data/dataset.py의 t 입력 → t+1 목표 슬라이싱)에서
특성 행 t의 수요 차분은 목표(t+1)가 아니라 입력(t 이하)에서 계산된다.
따라서 레거시 공식 demand[t] - demand[t-p]를 유지하며, 아래 테스트는
"목표 시점 수요 변경이 입력 특성에 영향을 주지 않음"만을 검증한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.lag_features import create_demand_lag_features


@pytest.mark.parametrize("period", [1, 24])
def test_future_target_change_does_not_affect_input_features(period: int) -> None:
    """예측 목표 시점(t+1)의 수요를 바꿔도 특성 행 t의 값은 불변이어야 한다."""
    demand = pd.DataFrame({"power_demand": np.arange(30, dtype=float)})
    modified = demand.copy()
    modified.loc[26, "power_demand"] = 10000.0

    original_features = create_demand_lag_features(demand)
    modified_features = create_demand_lag_features(modified)
    column = f"demand_diff_{period}h"

    # 레거시 공식 검증: 행 t의 차분은 demand[t] - demand[t-period]
    assert original_features.loc[25, column] == pytest.approx(float(period))
    # 목표 시점(t=26) 수요 변경은 입력 특성 행(t=25)에 영향을 주지 않는다.
    assert modified_features.loc[25, column] == pytest.approx(
        original_features.loc[25, column]
    )
    # 변경된 시점 자체(행 26)의 특성은 달라진다 (행 26은 이후 예측의 입력).
    assert modified_features.loc[26, column] != original_features.loc[26, column]


def test_legacy_formula_matches_manual_computation() -> None:
    """레거시 차분 공식이 수동 계산과 일치하는지 검증한다."""
    values = np.array([10.0, 13.0, 7.0, 22.0, 5.0, 40.0, 18.0])
    demand = pd.DataFrame({"power_demand": values})
    features = create_demand_lag_features(demand)

    expected_diff_1h = np.full(len(values), np.nan)
    expected_diff_1h[1:] = values[1:] - values[:-1]
    actual = features["demand_diff_1h"].to_numpy(dtype=float)
    np.testing.assert_allclose(actual[1:], expected_diff_1h[1:])
