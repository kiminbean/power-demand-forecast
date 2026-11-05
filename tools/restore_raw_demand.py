#!/usr/bin/env python3
"""Restore the unclipped power_demand values in processed hourly CSVs.

The old preprocess_power_data default clipped the target at Q3 + 1.5*IQR of the
full 2013-2024 series (694.658 MWh). The cap leaked test-period information and
erased 919 genuine summer peaks (2019-2024). This tool puts the raw KPX traded
volume back into the power_demand field of those rows only; every other byte of
the file stays the same, so the git diff shows exactly the restored values.

A differing row is restored only when it looks like the clip: the processed value
equals the file's maximum (the cap) and is below the raw value. Anything else
stops the tool, because it would be a different transformation.

Report mode (default) prints what would change and writes nothing.

Example:
    python tools/restore_raw_demand.py data/processed/jeju_hourly_cleaned.csv \
        data/processed/jeju_hourly_merged.csv            # report
    python tools/restore_raw_demand.py ... --in-place    # rewrite the files
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DEFAULT = PROJECT_ROOT / "data/raw/jeju_hourly_power_2013_2024.csv"
RAW_VALUE_COL = "전력거래량(MWh)"
# CSV에 6자리 소수로 저장된 값을 비교하므로 그보다 작은 차이는 같은 값으로 본다
TOLERANCE = 1e-6


def load_raw(path: Path) -> Dict[str, str]:
    """Map 'YYYY-MM-DD HH:MM:SS' (interval start) to the raw value text."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype={RAW_VALUE_COL: str})
    regions = set(df["지역"].astype(str).str.strip())
    if regions != {"제주도"}:
        raise ValueError(f"Raw file must contain only 제주도 rows, got {sorted(regions)}")
    # 시간 1~24는 (h-1):00에 시작하는 구간이다
    ts = pd.to_datetime(df["거래일자"], errors="raise") + pd.to_timedelta(df["시간"] - 1, unit="h")
    if ts.duplicated().any():
        raise ValueError("Duplicate timestamps in raw data")
    values = df[RAW_VALUE_COL].str.strip()
    if values.isna().any():
        raise ValueError("Missing values in raw data")
    return dict(zip(ts.dt.strftime("%Y-%m-%d %H:%M:%S"), values))


def restore(csv_path: Path, raw: Dict[str, str], in_place: bool) -> Dict:
    """Restore clipped power_demand fields in one processed CSV."""
    text = csv_path.read_text(encoding="utf-8-sig")
    lines = text.split("\n")
    if lines[0].split(",")[:2] != ["datetime", "power_demand"]:
        raise ValueError(f"{csv_path}: header must start with datetime,power_demand")
    rows = [line.split(",", 2) for line in lines[1:] if line]
    cap = max(float(r[1]) for r in rows)

    restored, by_year = [], Counter()
    for i, row in enumerate(rows):
        ts = row[0]
        if ts not in raw:
            raise ValueError(f"{csv_path}: {ts} not in raw data")
        value, raw_value = float(row[1]), float(raw[ts])
        if abs(value - raw_value) <= TOLERANCE:
            continue
        if abs(value - cap) > TOLERANCE or raw_value < value:
            raise ValueError(f"{csv_path}: {ts} differs from raw ({value} vs {raw_value}) "
                             "and is not a clip at the file maximum")
        rows[i][1] = raw[ts]
        restored.append(ts)
        by_year[ts[:4]] += 1

    if in_place and restored:
        new_lines = [lines[0]] + [",".join(r) for r in rows]
        tmp = csv_path.with_name(csv_path.name + ".tmp")
        # 원본과 같은 형식(BOM, LF, 마지막 줄바꿈)으로 쓰고 한 번에 교체한다
        tmp.write_text("\n".join(new_lines) + ("\n" if text.endswith("\n") else ""),
                       encoding="utf-8-sig")
        os.replace(tmp, csv_path)
    return {"file": str(csv_path), "restored": len(restored), "cap": cap,
            "by_year": dict(sorted(by_year.items())), "written": bool(in_place and restored)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv", nargs="+", type=Path, help="processed hourly CSV(s)")
    ap.add_argument("--raw", type=Path, default=RAW_DEFAULT)
    ap.add_argument("--in-place", action="store_true", help="rewrite the CSV files")
    args = ap.parse_args()
    raw = load_raw(args.raw)
    for path in args.csv:
        result = restore(path, raw, in_place=args.in_place)
        print(f"{result['file']}: restored {result['restored']} rows above cap "
              f"{result['cap']} by year {result['by_year']} (written: {result['written']})")


if __name__ == "__main__":
    main()
