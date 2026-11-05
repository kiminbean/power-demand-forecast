"""Tests for tools/restore_raw_demand.py (undo the old IQR clip of power_demand)."""

import pytest

from tools import restore_raw_demand as rrd

RAW_HEADER = "거래일자,시간,지역,전력거래량(MWh)"


def _write(path, lines):
    # 원본·처리 CSV 모두 BOM이 붙은 UTF-8이다
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


@pytest.fixture
def files(tmp_path):
    raw = tmp_path / "raw.csv"
    _write(raw, [RAW_HEADER,
                 "2024-07-01,1,제주도,500.1",
                 "2024-07-01,2,제주도,700.123456",
                 "2024-07-01,3,제주도,650.5"])
    processed = tmp_path / "processed.csv"
    # 02:00 → 상한 600.0으로 잘린 값, 나머지는 원본과 같다
    _write(processed, ["datetime,power_demand,기온",
                       "2024-07-01 00:00:00,500.1,25.0",
                       "2024-07-01 01:00:00,600.0,25.5",
                       "2024-07-01 02:00:00,600.0,26.0"])
    return raw, processed


def test_report_mode_counts_but_does_not_write(files):
    raw, processed = files
    before = processed.read_bytes()
    result = rrd.restore(processed, rrd.load_raw(raw), in_place=False)
    assert result["restored"] == 2 and result["cap"] == pytest.approx(600.0)
    assert processed.read_bytes() == before


def test_in_place_changes_only_clipped_fields(files):
    raw, processed = files
    rrd.restore(processed, rrd.load_raw(raw), in_place=True)
    text = processed.read_text(encoding="utf-8-sig").splitlines()
    # 잘린 칸만 원본 문자열로 바뀌고 다른 칸·행은 그대로여야 한다
    assert text == ["datetime,power_demand,기온",
                    "2024-07-01 00:00:00,500.1,25.0",
                    "2024-07-01 01:00:00,700.123456,25.5",
                    "2024-07-01 02:00:00,650.5,26.0"]
    assert processed.read_bytes().startswith(b"\xef\xbb\xbf")


def test_rejects_difference_that_is_not_a_clip(files):
    raw, processed = files
    # 상한이 아닌 값이 원본과 다르면 클리핑이 아닌 다른 가공이므로 멈춘다
    _write(processed, ["datetime,power_demand,기온",
                       "2024-07-01 00:00:00,480.0,25.0",
                       "2024-07-01 01:00:00,600.0,25.5",
                       "2024-07-01 02:00:00,600.0,26.0"])
    with pytest.raises(ValueError, match="not a clip"):
        rrd.restore(processed, rrd.load_raw(raw), in_place=False)


def test_rejects_timestamp_missing_from_raw(files):
    raw, processed = files
    _write(processed, ["datetime,power_demand,기온",
                       "2024-07-02 00:00:00,500.1,25.0"])
    with pytest.raises(ValueError, match="not in raw"):
        rrd.restore(processed, rrd.load_raw(raw), in_place=False)


def test_rejects_unexpected_header(files):
    raw, processed = files
    _write(processed, ["datetime,기온,power_demand", "2024-07-01 00:00:00,25.0,500.1"])
    with pytest.raises(ValueError, match="header"):
        rrd.restore(processed, rrd.load_raw(raw), in_place=False)
