#!/usr/bin/env python3
"""Download KPX hourly regional traded volume and write the Jeju rows in the raw-file format.

Source: 공공데이터포털 file dataset 15133498, 한국전력거래소_지역별 시간대별 전력거래량
(https://www.data.go.kr/data/15133498/fileData.do). It is the same series as
data/raw/jeju_hourly_power_2013_2024.csv: KPX 지역 전력거래량 for 제주도, 송전단, MWh,
거래시간 1-24. KPX publishes one year per release (the 2025 file was registered
2026-06-30), so the output year comes from the data, not from an argument.

The download follows the portal's own page flow and needs no API key. The tool writes
data/raw/jeju_hourly_power_<year>.csv (UTF-8 BOM, header 거래일자,시간,지역,전력거래량(MWh)),
keeping the source value text unchanged.

Example:
    python tools/fetch_kpx_jeju_trade.py                    # download, validate, write
    python tools/fetch_kpx_jeju_trade.py --source file.csv  # use a file downloaded by hand
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PORTAL = "https://www.data.go.kr"
DATASET_PK = "15133498"
# 파일 상세 키. KPX가 새 연도 파일로 바꿔도 그대로이고, 내려받기 번호(atchFileId)만 바뀐다
DATASET_DETAIL_PK = "uddi:cabf7071-8260-4270-bb71-bfcae23917fe"
SOURCE_COLUMNS = ["거래일", "거래시간", "지역", "전력거래량"]
REGION = "제주도"
TIMEOUT = 60


def download() -> bytes:
    """포털 화면이 쓰는 절차(세션 → 내려받기 번호 조회 → 내려받기)를 그대로 밟는다."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
                            "Referer": f"{PORTAL}/data/{DATASET_PK}/fileData.do"})
    session.get(f"{PORTAL}/data/{DATASET_PK}/fileData.do", timeout=TIMEOUT).raise_for_status()
    resp = session.post(f"{PORTAL}/tcs/dss/selectFileDataDownload.do", timeout=TIMEOUT, data={
        "publicDataPk": DATASET_PK, "publicDataDetailPk": DATASET_DETAIL_PK, "atchFileId": "",
        "fileDetailSn": "1", "publicDataTyCode": "PR0051"})
    resp.raise_for_status()
    info = resp.json()
    if not info.get("status") or not info.get("atchFileId"):
        raise RuntimeError(f"data.go.kr returned no download id: {str(info)[:300]}")
    limit = session.post(f"{PORTAL}/cmm/cmm/check-limit.json", timeout=TIMEOUT,
                         data={"atchFileId": info["atchFileId"], "fileDetailSn": info["fileDetailSn"]})
    if limit.ok and limit.json().get("needCaptcha"):
        raise RuntimeError("data.go.kr asks for a captcha; download the file by hand and pass --source")
    resp = session.get(f"{PORTAL}/cmm/cmm/fileDownload.do", timeout=TIMEOUT,
                       params={"atchFileId": info["atchFileId"], "fileDetailSn": info["fileDetailSn"]})
    resp.raise_for_status()
    return resp.content


def jeju_rows(content: bytes) -> pd.DataFrame:
    """제주도 행만 골라 원본 raw 파일 형식으로 바꾼다. 형식이 다르거나 시각이 비면 실패한다."""
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", dtype=str)
    if list(df.columns) != SOURCE_COLUMNS:
        raise ValueError(f"Unexpected columns {list(df.columns)}; expected {SOURCE_COLUMNS}")
    df = df.apply(lambda col: col.str.strip())
    jeju = df[df["지역"] == REGION]
    if jeju.empty:
        raise ValueError(f"No {REGION} rows in the source file")
    hours = pd.to_numeric(jeju["거래시간"], errors="raise").astype(int)
    values = pd.to_numeric(jeju["전력거래량"], errors="raise")
    if not hours.between(1, 24).all():
        raise ValueError("거래시간 must be within 1-24")
    if values.isna().any() or (values <= 0).any():
        raise ValueError("전력거래량 must be positive numbers")
    # 시간 h는 (h-1):00에 시작하는 구간이다 (tools/restore_raw_demand.py와 같은 규칙)
    ts = pd.to_datetime(jeju["거래일"], format="%Y-%m-%d", errors="raise") + pd.to_timedelta(hours - 1, unit="h")
    if ts.duplicated().any():
        raise ValueError("Duplicate hours in the source file")
    if ts.min().year != ts.max().year:
        raise ValueError(f"Expected one year per release, got {ts.min()} to {ts.max()}")
    missing = pd.date_range(ts.min(), ts.max(), freq="h").difference(ts)
    if len(missing):
        raise ValueError(f"{len(missing)} hours missing, first {missing[0]}")
    out = pd.DataFrame({"거래일자": jeju["거래일"].to_numpy(), "시간": hours.to_numpy(),
                        "지역": REGION, "전력거래량(MWh)": jeju["전력거래량"].to_numpy()})
    return out.iloc[ts.to_numpy().argsort(kind="stable")].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, help="CSV downloaded by hand from the dataset page")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing file that differs")
    args = ap.parse_args()

    content = args.source.read_bytes() if args.source else download()
    rows = jeju_rows(content)
    year = int(rows["거래일자"].iloc[0][:4])
    out = args.out_dir / f"jeju_hourly_power_{year}.csv"
    text = rows.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
    # 정산 수정으로 같은 해 값이 바뀔 수 있으니 다른 내용은 명시적으로 허락할 때만 덮어쓴다
    if out.exists() and out.read_bytes() != text and not args.overwrite:
        raise SystemExit(f"{out} exists with different content; pass --overwrite to replace it")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(text)
    os.replace(tmp, out)
    print(f"source sha256 {hashlib.sha256(content).hexdigest()}")
    print(f"wrote {out}: {len(rows)} hours, {rows['거래일자'].iloc[0]} 시간{rows['시간'].iloc[0]} "
          f"to {rows['거래일자'].iloc[-1]} 시간{rows['시간'].iloc[-1]}")


if __name__ == "__main__":
    main()
