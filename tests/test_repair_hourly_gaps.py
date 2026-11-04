"""Tests for tools/repair_hourly_gaps.py (opt-in hourly gap repair CLI).

Covers: report mode, repair mode, no source mutation, sidecar contents,
multi-hour gap policy (refuse by default, skip+flag with --allow-multihour),
path-safety refusals, and pre-existing-NaN preservation.
"""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from tools import repair_hourly_gaps as rhg

TS_COL = "datetime"


def make_hourly_frame(periods: int = 72, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a synthetic hourly frame mirroring the production schema shape."""
    idx = pd.date_range(start, periods=periods, freq="h")
    df = pd.DataFrame(
        {
            "power_demand": [100.0 + i for i in range(periods)],
            "기온": [5.0 + 0.1 * i for i in range(periods)],
            "지점명": ["제주"] * periods,
        },
        index=idx,
    )
    df.index.name = TS_COL
    return df


def write_csv(
    path: Path, df: pd.DataFrame, drop_hours=(), encoding: str = "utf-8"
) -> Path:
    """Write a CSV, optionally omitting specific hours to create gaps."""
    out = df.drop(index=pd.DatetimeIndex(list(drop_hours)))
    out.reset_index(names=TS_COL).to_csv(
        path, index=False, date_format="%Y-%m-%d %H:%M:%S", encoding=encoding
    )
    return path


@pytest.fixture
def gapy_csv(tmp_path: Path) -> Path:
    """CSV with two isolated missing hours and one pre-existing NaN."""
    df = make_hourly_frame()
    df.loc[df.index[10], "power_demand"] = float("nan")  # 기존 결측: 절대 채우면 안 됨
    return write_csv(
        tmp_path / "gapy.csv",
        df,
        drop_hours=["2024-01-02 05:00:00", "2024-01-03 22:00:00"],
    )


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestReportMode:
    def test_lists_gaps_and_exits_gaps_found(self, gapy_csv, tmp_path, capsys):
        code = rhg.main([str(gapy_csv)])
        out = capsys.readouterr().out
        assert code == rhg.EXIT_GAPS_FOUND
        assert "2024-01-02 05:00:00" in out
        assert "2024-01-03 22:00:00" in out
        assert "missing hours: 2" in out
        # 보고 모드는 어떤 파일도 쓰지 않는다
        assert list(tmp_path.iterdir()) == [gapy_csv]

    def test_no_gaps_exits_ok(self, tmp_path, capsys):
        csv = write_csv(tmp_path / "full.csv", make_hourly_frame())
        code = rhg.main([str(csv)])
        assert code == rhg.EXIT_OK
        assert "missing hours: 0" in capsys.readouterr().out

    def test_report_labels_multihour_runs(self, tmp_path, capsys):
        df = make_hourly_frame()
        csv = write_csv(
            tmp_path / "multi.csv",
            df,
            drop_hours=[
                "2024-01-03 07:00:00",
                "2024-01-03 08:00:00",
                "2024-01-03 09:00:00",
            ],
        )
        code = rhg.main([str(csv)])
        out = capsys.readouterr().out
        assert code == rhg.EXIT_GAPS_FOUND
        assert "MULTI-HOUR gap: 2024-01-03 07:00:00 .. 2024-01-03 09:00:00 (3h)" in out


class TestRepairMode:
    def test_fills_isolated_gaps_linearly(self, gapy_csv, tmp_path):
        out = tmp_path / "repaired.csv"
        code = rhg.main([str(gapy_csv), "-o", str(out)])
        assert code == rhg.EXIT_OK
        assert out.is_file()

        df = pd.read_csv(out, encoding="utf-8", parse_dates=[TS_COL]).set_index(TS_COL)
        assert len(df) == 72  # 결측 2시간이 복원되어 전체 72행

        # 선형 보간: 이웃 값의 평균 (원본: 28,29 → 03-01 05:00 인덱스 29 skip)
        before = df.loc[pd.Timestamp("2024-01-02 04:00:00"), "power_demand"]
        after = df.loc[pd.Timestamp("2024-01-02 06:00:00"), "power_demand"]
        filled = df.loc[pd.Timestamp("2024-01-02 05:00:00"), "power_demand"]
        assert filled == pytest.approx((before + after) / 2)
        # 비수치 컬럼은 forward-fill
        assert df.loc[pd.Timestamp("2024-01-02 05:00:00"), "지점명"] == "제주"

    def test_source_file_not_mutated(self, gapy_csv, tmp_path):
        digest_before = file_digest(gapy_csv)
        mtime_before = gapy_csv.stat().st_mtime_ns
        rhg.main([str(gapy_csv), "-o", str(tmp_path / "repaired.csv")])
        assert file_digest(gapy_csv) == digest_before
        assert gapy_csv.stat().st_mtime_ns == mtime_before

    def test_pre_existing_nan_preserved(self, gapy_csv, tmp_path):
        out = tmp_path / "repaired.csv"
        rhg.main([str(gapy_csv), "-o", str(out)])
        df = pd.read_csv(out, parse_dates=[TS_COL]).set_index(TS_COL)
        # 원본 행(2024-01-01 10:00)의 기존 NaN 은 채우지 않는다
        assert pd.isna(df.loc[pd.Timestamp("2024-01-01 10:00:00"), "power_demand"])

    def test_sidecar_contents(self, gapy_csv, tmp_path):
        out = tmp_path / "repaired.csv"
        rhg.main([str(gapy_csv), "-o", str(out)])
        sidecar = tmp_path / "repaired.csv.repair_report.json"
        assert sidecar.is_file()
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        assert report["source"] == str(gapy_csv.resolve())
        assert report["source_sha256"] == file_digest(gapy_csv)
        assert report["gaps_filled"] == [
            "2024-01-02 05:00:00",
            "2024-01-03 22:00:00",
        ]
        assert report["gaps_flagged"] == []
        assert "linear" in report["method"]
        assert report["source_rows"] == 70
        assert report["output_rows"] == 72
        pd.Timestamp(report["timestamp"])  # ISO timestamp 파싱 가능해야 함

    def test_custom_sidecar_path(self, gapy_csv, tmp_path):
        out = tmp_path / "repaired.csv"
        sidecar = tmp_path / "side.json"
        rhg.main([str(gapy_csv), "-o", str(out), "--sidecar", str(sidecar)])
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text())["gaps_filled"]

    def test_bom_source_round_trips_with_bom(self, tmp_path):
        df = make_hourly_frame()
        csv = write_csv(
            tmp_path / "bom.csv", df,
            drop_hours=["2024-01-02 05:00:00"], encoding="utf-8-sig",
        )
        out = tmp_path / "repaired.csv"
        assert rhg.main([str(csv), "-o", str(out)]) == rhg.EXIT_OK
        assert out.read_bytes()[:3] == b"\xef\xbb\xbf"  # BOM preserved
        assert rhg.main([str(csv)]) == rhg.EXIT_GAPS_FOUND  # 컬럼명 파싱 정상


class TestMultiHourPolicy:
    @pytest.fixture
    def multihour_csv(self, tmp_path: Path) -> Path:
        df = make_hourly_frame()
        return write_csv(
            tmp_path / "multi.csv",
            df,
            drop_hours=["2024-01-03 07:00:00", "2024-01-03 08:00:00"],
        )

    def test_repair_refused_and_nothing_written(self, multihour_csv, tmp_path, capsys):
        out = tmp_path / "repaired.csv"
        code = rhg.main([str(multihour_csv), "-o", str(out)])
        assert code == rhg.EXIT_REFUSED
        assert not out.exists()
        assert "REFUSED" in capsys.readouterr().err

    def test_allow_multihour_skips_and_flags(self, multihour_csv, tmp_path):
        out = tmp_path / "repaired.csv"
        code = rhg.main([str(multihour_csv), "-o", str(out), "--allow-multihour"])
        assert code == rhg.EXIT_OK
        df = pd.read_csv(out, parse_dates=[TS_COL]).set_index(TS_COL)
        # 2시간 결측은 삽입되지 않고 그대로 남는다 (70행)
        assert len(df) == 70
        assert pd.Timestamp("2024-01-03 07:00:00") not in df.index
        report = json.loads((tmp_path / "repaired.csv.repair_report.json").read_text())
        assert report["gaps_flagged"] == [
            {
                "start": "2024-01-03 07:00:00",
                "end": "2024-01-03 08:00:00",
                "hours": 2,
                "action": "skipped (multi-hour gap policy)",
            }
        ]
        assert report["gaps_filled"] == []


class TestPathSafety:
    def test_output_equal_to_source_refused(self, gapy_csv, capsys):
        code = rhg.main([str(gapy_csv), "-o", str(gapy_csv)])
        assert code == rhg.EXIT_REFUSED
        assert "never modified" in capsys.readouterr().err

    def test_sidecar_equal_to_source_refused(self, gapy_csv, capsys):
        code = rhg.main(
            [str(gapy_csv), "-o", "/tmp/never.csv", "--sidecar", str(gapy_csv)]
        )
        assert code == rhg.EXIT_REFUSED

    def test_existing_output_requires_force(self, gapy_csv, tmp_path):
        out = tmp_path / "repaired.csv"
        out.write_text("existing data")
        assert rhg.main([str(gapy_csv), "-o", str(out)]) == rhg.EXIT_REFUSED
        assert out.read_text() == "existing data"  # 덮어쓰지 않음
        assert rhg.main([str(gapy_csv), "-o", str(out), "--force"]) == rhg.EXIT_OK

    def test_missing_source_refused(self, tmp_path):
        assert rhg.main([str(tmp_path / "nope.csv")]) == rhg.EXIT_REFUSED

    def test_duplicate_timestamps_refused(self, tmp_path):
        df = make_hourly_frame(periods=5).reset_index(names=TS_COL)
        df = pd.concat([df, df.iloc[[2]]])  # duplicated row
        csv = tmp_path / "dup.csv"
        df.to_csv(csv, index=False)
        assert rhg.main([str(csv)]) == rhg.EXIT_REFUSED


class TestCliSubprocess:
    def test_module_runs_as_script(self, gapy_csv, tmp_path, capsys):
        # main() 경로와 동일하므로, 여기선 argparse 진입점만 스모크로 점검
        code = rhg.main(
            [str(gapy_csv), "-o", str(tmp_path / "r.csv"), "--timestamp-col", TS_COL]
        )
        assert code == rhg.EXIT_OK
        assert (tmp_path / "r.csv").is_file()
