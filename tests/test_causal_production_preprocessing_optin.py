import numpy as np
import pandas as pd
import pytest

from src.data import preprocessing


def sample():
    return pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=6, freq="h"),
                         "value": [np.nan, 1., np.nan, 3., 100., np.nan]})


def test_train_only_bounds_and_causal_fill_are_reusable():
    data = sample()
    processor = preprocessing.CausalProductionPreprocessor(["value"])
    processor.fit(data, train_end=data.datetime.iloc[3])
    result = processor.transform(data)
    assert pd.isna(result.value.iloc[0])
    assert result.value.iloc[2] == 1
    assert result.value.iloc[4] == 4
    assert result.value.iloc[5] == 4
    changed = data.copy()
    changed.loc[4, "value"] = 99999
    other = preprocessing.CausalProductionPreprocessor(["value"]).fit(
        changed, train_end=data.datetime.iloc[3]
    )
    pd.testing.assert_frame_equal(result, other.transform(changed))


def test_future_values_cannot_fill_earlier_gaps():
    data = sample()
    processor = preprocessing.CausalProductionPreprocessor(["value"]).fit(
        data, train_end=data.datetime.iloc[3]
    )
    changed = data.copy()
    changed.loc[3:, "value"] = [999, 999, 999]
    pd.testing.assert_frame_equal(processor.transform(data).iloc[:3],
                                  processor.transform(changed).iloc[:3])


def test_transform_requires_fit_and_valid_unique_chronology():
    data = sample()
    processor = preprocessing.CausalProductionPreprocessor(["value"])
    with pytest.raises(ValueError, match="fit"):
        processor.transform(data)
    processor.fit(data, train_end=data.datetime.iloc[3])
    for invalid in (data.iloc[::-1], data.assign(datetime=pd.NaT),
                    data.assign(datetime=data.datetime.iloc[0])):
        with pytest.raises(ValueError):
            processor.transform(invalid)


def test_empty_training_column_is_rejected():
    data = sample()
    data["value"] = np.nan
    with pytest.raises(ValueError, match="value"):
        preprocessing.CausalProductionPreprocessor(["value"]).fit(
            data, train_end=data.datetime.iloc[3]
        )


def test_existing_interpolation_still_backfills_and_interpolates():
    df = pd.DataFrame({"value": [np.nan, 1., np.nan, 3.]})
    result, _ = preprocessing.interpolate_missing_values(df)
    assert result.value.tolist() == [1., 1., 2., 3.]
