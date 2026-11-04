"""Regression tests for temporal split boundary membership."""

import pandas as pd
import pytest

from src.data.dataset import split_data_by_time


@pytest.mark.parametrize("present_train,present_val", [(True, True), (True, False), (False, True), (False, False)])
def test_split_retains_all_rows_with_present_or_absent_boundaries(
    present_train: bool, present_val: bool
) -> None:
    """Exact boundaries belong to the preceding split, without row loss."""
    train_end = pd.Timestamp("2023-01-01 02:00:00")
    val_end = pd.Timestamp("2023-01-01 05:00:00")
    index = pd.date_range("2023-01-01", periods=8, freq="h")
    if not present_train:
        index = index[index != train_end]
    if not present_val:
        index = index[index != val_end]
    df = pd.DataFrame({"value": range(len(index))}, index=index)

    train, val, test = split_data_by_time(df, str(train_end), str(val_end))

    pd.testing.assert_frame_equal(train, df[df.index <= train_end])
    pd.testing.assert_frame_equal(val, df[(df.index > train_end) & (df.index <= val_end)])
    pd.testing.assert_frame_equal(test, df[df.index > val_end])
    pd.testing.assert_frame_equal(pd.concat([train, val, test]), df)
    assert train.index.intersection(val.index).empty
    assert train.index.intersection(test.index).empty
    assert val.index.intersection(test.index).empty


def test_duplicate_boundary_timestamps_stay_in_preceding_split() -> None:
    """Each row at an exact boundary remains in the earlier partition."""
    index = pd.to_datetime([
        "2023-01-01 01:00", "2023-01-01 02:00", "2023-01-01 02:00",
        "2023-01-01 03:00", "2023-01-01 05:00", "2023-01-01 05:00",
        "2023-01-01 06:00",
    ])
    df = pd.DataFrame({"value": range(len(index))}, index=index)
    train, val, test = split_data_by_time(df, "2023-01-01 02:00", "2023-01-01 05:00")

    assert train["value"].tolist() == [0, 1, 2]
    assert val["value"].tolist() == [3, 4, 5]
    assert test["value"].tolist() == [6]
    pd.testing.assert_frame_equal(pd.concat([train, val, test]), df)


def test_aware_index_interprets_naive_boundaries_in_index_timezone() -> None:
    index = pd.date_range("2023-01-01", periods=8, freq="h", tz="Asia/Seoul")
    df = pd.DataFrame({"value": range(8)}, index=index)
    train, val, test = split_data_by_time(df, "2023-01-01 02:00", "2023-01-01 05:00")
    assert [len(train), len(val), len(test)] == [3, 3, 2]
    pd.testing.assert_frame_equal(pd.concat([train, val, test]), df)


def test_aware_boundaries_preserve_absolute_instants() -> None:
    index = pd.date_range("2023-01-01", periods=8, freq="h", tz="Asia/Seoul")
    df = pd.DataFrame({"value": range(8)}, index=index)
    train, val, test = split_data_by_time(df, "2022-12-31 17:00+00:00", "2022-12-31 20:00+00:00")
    assert [len(train), len(val), len(test)] == [3, 3, 2]
    pd.testing.assert_frame_equal(pd.concat([train, val, test]), df)


def test_reverse_boundaries_are_rejected() -> None:
    df = pd.DataFrame({"value": range(8)}, index=pd.date_range("2023-01-01", periods=8, freq="h"))
    with pytest.raises(ValueError, match="train_end.*val_end"):
        split_data_by_time(df, "2023-01-01 05:00", "2023-01-01 02:00")


def test_aware_boundary_with_naive_index_is_rejected() -> None:
    df = pd.DataFrame({"value": range(8)}, index=pd.date_range("2023-01-01", periods=8, freq="h"))
    with pytest.raises(ValueError, match="Timezone-aware boundary"):
        split_data_by_time(df, "2023-01-01 02:00+00:00", "2023-01-01 05:00")


@pytest.mark.parametrize("which", ["train", "validation", "index"])
def test_nat_is_rejected_in_boundaries_or_index(which: str) -> None:
    index = pd.date_range("2023-01-01", periods=8, freq="h")
    if which == "index":
        index = index.insert(0, pd.NaT)
    df = pd.DataFrame({"value": range(len(index))}, index=index)
    train = "NaT" if which == "train" else "2023-01-01 02:00"
    val = "NaT" if which == "validation" else "2023-01-01 05:00"
    with pytest.raises(ValueError, match="NaT"):
        split_data_by_time(df, train, val)


@pytest.mark.parametrize("tz", [None, "Asia/Seoul"])
def test_unsorted_index_is_rejected_before_sequence_creation(tz: str | None) -> None:
    index = pd.date_range("2023-01-01", periods=8, freq="h", tz=tz)
    df = pd.DataFrame({"value": range(8)}, index=index).iloc[[3, 0, 1, 2, 4, 5, 6, 7]]
    with pytest.raises(ValueError, match="chronological"):
        split_data_by_time(df, "2023-01-01 02:00", "2023-01-01 05:00")
