#!/usr/bin/env python3
"""Opt-in CLI tool to report and repair isolated hourly gaps in a time-series CSV.

Report mode (default) prints the hourly gaps found in the source file and
writes nothing. Repair mode (-o/--output) writes a REPAIRED COPY of the data
to a user-specified new path; the source file is never modified.

Repair policy:
- Only isolated missing hours (single-hour gaps with both neighbors present)
  are filled.
- Numeric columns are filled with pandas linear interpolation, computed from
  the surrounding original rows only.
- Non-numeric columns at filled rows are forward-filled from the previous row.
- Pre-existing NaN values inside original rows are never touched.
- Multi-hour gaps (2+ consecutive missing hours) cause repair to be REFUSED
  (exit code 2) unless --allow-multihour is given; with --allow-multihour the
  multi-hour rows are skipped (left absent) and flagged in the sidecar report.

A JSON sidecar report (source, gaps filled, method, timestamp, ...) is written
next to the output file ("<output>.repair_report.json" by default).

Exit codes:
- 0: success (no gaps found, or repair completed)
- 1: report mode found gaps
- 2: refused (multi-hour gap, unsafe output path, or existing target file
  without --force) or invalid input

Example:
    python tools/repair_hourly_gaps.py data/processed/jeju_hourly_merged.csv
    python tools/repair_hourly_gaps.py data/processed/jeju_hourly_merged.csv \
        -o /tmp/jeju_hourly_merged_repaired.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

DEFAULT_TIMESTAMP_COL = "datetime"
HOURLY_FREQ = "h"
CSV_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
BOM_ENCODING = "utf-8-sig"
PLAIN_ENCODING = "utf-8"
INTERP_METHOD = "linear"
REPAIR_METHOD = (
    "pandas linear interpolation on numeric columns "
    f"(isolated missing hours only); forward-fill on non-numeric columns; "
    "pre-existing NaNs in original rows preserved"
)

EXIT_OK = 0
EXIT_GAPS_FOUND = 1
EXIT_REFUSED = 2


def detect_encoding(path: Path) -> str:
    """Return 'utf-8-sig' when the file starts with a BOM, else 'utf-8'."""
    with open(path, "rb") as fh:
        head = fh.read(3)
    return BOM_ENCODING if head == b"\xef\xbb\xbf" else PLAIN_ENCODING


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hourly_frame(path: Path, timestamp_col: str) -> pd.DataFrame:
    """Load the CSV, parse the timestamp column, and index it monotonically.

    Args:
        path: CSV file to load.
        timestamp_col: Name of the hourly timestamp column.

    Returns:
        DataFrame indexed (ascending) by the timestamp column.

    Raises:
        ValueError: If the column is missing, unparseable, or duplicated.
    """
    encoding = detect_encoding(path)
    df = pd.read_csv(path, encoding=encoding)
    if timestamp_col not in df.columns:
        raise ValueError(
            f"Timestamp column {timestamp_col!r} not found; "
            f"available columns: {list(df.columns)[:10]}..."
        )
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="raise")
    if df[timestamp_col].duplicated().any():
        dup = df.loc[df[timestamp_col].duplicated(), timestamp_col].iloc[0]
        raise ValueError(f"Duplicated timestamp {dup}; refusing to repair")
    df = df.sort_values(timestamp_col).set_index(timestamp_col)
    return df


def find_gap_runs(index: pd.DatetimeIndex) -> List[List[pd.Timestamp]]:
    """Find runs of consecutive missing hours in a DatetimeIndex.

    The expected full hourly range spans min(index)..max(index), so missing
    hours at the very edges cannot be detected (documented limitation).

    Args:
        index: Observed timestamps (unique, sorted).

    Returns:
        List of runs; each run is a list of the missing timestamps in it.
        A run of length 1 is an isolated gap; length >= 2 is a multi-hour gap.
    """
    full = pd.date_range(index.min(), index.max(), freq=HOURLY_FREQ)
    missing = full.difference(index)
    runs: List[List[pd.Timestamp]] = []
    current: List[pd.Timestamp] = []
    for ts in missing:
        if current and (ts - current[-1]) == pd.Timedelta(hours=1):
            current.append(ts)
        else:
            if current:
                runs.append(current)
            current = [ts]
    if current:
        runs.append(current)
    return runs


def split_runs(
    runs: Sequence[Sequence[pd.Timestamp]],
) -> Tuple[List[pd.Timestamp], List[List[pd.Timestamp]]]:
    """Split gap runs into (isolated single-hour gaps, multi-hour runs)."""
    isolated: List[pd.Timestamp] = []
    multi: List[List[pd.Timestamp]] = []
    for run in runs:
        if len(run) == 1:
            isolated.append(run[0])
        else:
            multi.append(list(run))
    return isolated, multi


def repair_frame(
    df: pd.DataFrame, isolated: Sequence[pd.Timestamp]
) -> pd.DataFrame:
    """Return a copy of ``df`` on the full hourly index with isolated gaps filled.

    도메인 정책: 오직 '이웃 두 시각이 모두 존재하는 1시간 결측'만 채운다.
    - 수치 컬럼: 원본 이웃 값으로부터 선형 보간 (pandas interpolate, limit_area="inside")
    - 비수치 컬럼: 직전 원본 행 값으로 forward-fill
    - 원본 행에 원래 있던 NaN 은 절대 건드리지 않는다 (보간 결과에서
      격차 행의 값만 가져온다).
    """
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=HOURLY_FREQ)
    reindexed = df.reindex(full_idx)

    numeric_cols = list(df.select_dtypes(include="number").columns)
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    # 격차 행에만 값을 넣는다: 원본 행은 보간 결과와 무관하게 유지.
    if isolated:
        interp = (
            reindexed[numeric_cols]
            .interpolate(method=INTERP_METHOD, limit_area="inside")
            if numeric_cols
            else None
        )
        ffilled = reindexed[non_numeric_cols].ffill() if non_numeric_cols else None
        for ts in isolated:
            if interp is not None:
                reindexed.loc[ts, numeric_cols] = interp.loc[ts].values
            if ffilled is not None:
                reindexed.loc[ts, non_numeric_cols] = ffilled.loc[ts].values

    # 재색인으로 삽입된 '격자 행' 중 채우지 않은 행(비격리 결측)은 제거해
    # 원본과 동일한 관측 행 집합 + 채워진 격리 결측만 남긴다.
    inserted = full_idx.difference(df.index)
    keep_inserted = pd.DatetimeIndex(list(isolated))
    to_drop = pd.DatetimeIndex(inserted.difference(keep_inserted))
    return reindexed.drop(index=to_drop)


def format_ts(ts: pd.Timestamp) -> str:
    """Format a timestamp the way the CLI displays it."""
    return ts.strftime(CSV_DATE_FORMAT)


def print_report(
    df: pd.DataFrame,
    isolated: Sequence[pd.Timestamp],
    multi: Sequence[Sequence[pd.Timestamp]],
) -> None:
    """Print a human-readable gap report."""
    full = pd.date_range(df.index.min(), df.index.max(), freq=HOURLY_FREQ)
    expected = len(full)
    print(
        f"rows: {len(df)}  "
        f"range: {format_ts(df.index.min())} .. {format_ts(df.index.max())}"
    )
    print(f"expected hourly rows: {expected}  missing hours: {expected - len(df)}")
    for ts in isolated:
        print(f"  isolated gap: {format_ts(ts)}")
    for run in multi:
        span = f"{format_ts(run[0])} .. {format_ts(run[-1])} ({len(run)}h)"
        print(f"  MULTI-HOUR gap: {span}")


def build_sidecar(
    source: Path,
    output: Path,
    isolated: Sequence[pd.Timestamp],
    multi: Sequence[Sequence[pd.Timestamp]],
    n_source_rows: int,
    n_output_rows: int,
) -> dict:
    """Build the sidecar report dictionary."""
    return {
        "source": str(source.resolve()),
        "source_sha256": sha256_of(source),
        "output": str(output.resolve()),
        "gaps_filled": [format_ts(ts) for ts in isolated],
        "gaps_flagged": [
            {
                "start": format_ts(run[0]),
                "end": format_ts(run[-1]),
                "hours": len(run),
                "action": "skipped (multi-hour gap policy)",
            }
            for run in multi
        ],
        "method": REPAIR_METHOD,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_rows": n_source_rows,
        "output_rows": n_output_rows,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Report hourly gaps in a CSV and optionally write a repaired copy.",
    )
    parser.add_argument("source", type=Path, help="Source CSV (never modified)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write a repaired COPY to this new path (enables repair mode)",
    )
    parser.add_argument(
        "--timestamp-col",
        default=DEFAULT_TIMESTAMP_COL,
        help=f"Timestamp column name (default: {DEFAULT_TIMESTAMP_COL})",
    )
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=None,
        help='Sidecar JSON path (default: "<output>.repair_report.json")',
    )
    parser.add_argument(
        "--allow-multihour",
        action="store_true",
        help="Do not refuse on multi-hour gaps; skip and flag them instead",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting an existing output/sidecar file (never the source)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns the process exit code."""
    args = parse_args(argv)
    source: Path = args.source

    if not source.is_file():
        print(f"ERROR: source file not found: {source}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        df = load_hourly_frame(source, args.timestamp_col)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    isolated, multi = split_runs(find_gap_runs(pd.DatetimeIndex(df.index)))

    if args.output is None:
        # 보고 모드: 아무 파일도 쓰지 않는다.
        print_report(df, isolated, multi)
        return EXIT_GAPS_FOUND if (isolated or multi) else EXIT_OK

    # --- repair mode -----------------------------------------------------
    output: Path = args.output
    sidecar: Path = (
        args.sidecar if args.sidecar else Path(str(output) + ".repair_report.json")
    )
    source_resolved = source.resolve()
    if output.resolve() == source_resolved or sidecar.resolve() == source_resolved:
        print(
            "ERROR: output/sidecar must not be the source file; "
            "source is never modified",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if output.exists() and not args.force:
        print(
            f"ERROR: output already exists: {output} (use --force to overwrite)",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if sidecar.exists() and not args.force:
        print(
            f"ERROR: sidecar already exists: {sidecar} (use --force to overwrite)",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    if multi and not args.allow_multihour:
        print(
            "REFUSED: multi-hour gap(s) found; only isolated single-hour gaps",
            file=sys.stderr,
        )
        print(
            "are repaired. Re-run with --allow-multihour to skip and flag them.",
            file=sys.stderr,
        )
        print_report(df, isolated, multi)
        return EXIT_REFUSED

    repaired = repair_frame(df, isolated)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoding = detect_encoding(source)
    repaired.reset_index(names=args.timestamp_col).to_csv(
        output, index=False, date_format=CSV_DATE_FORMAT, encoding=encoding
    )
    sidecar_dict = build_sidecar(
        source, output, isolated, multi, len(df), len(repaired)
    )
    sidecar.write_text(
        json.dumps(sidecar_dict, indent=2, ensure_ascii=False) + "\n",
        encoding=PLAIN_ENCODING,
    )

    print(f"repaired copy: {output} ({len(repaired)} rows)")
    print(
        f"gaps filled: {len(isolated)} isolated; "
        f"flagged: {len(multi)} multi-hour run(s)"
    )
    print(f"sidecar: {sidecar}")
    print(f"source untouched: {source}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
