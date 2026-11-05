"""제주 하루전시장 SMP 시간별 연속 이력 구축.

원천:
- 제주시범사업 API(B552115/JejuSmpLfd2) jsmp: 2024-07-02부터 하루 96행(1시간 × 15분 구간 4개).
  2026-11-05 대조에서 jsmp는 smp_5years_epsis.csv·smp_real_epsis.csv의 smp_jeju와 100% 일치했고,
  KPX 제주 하루전시장 포털 표와도 일치했다 (smp_mainland는 육지 SMP라 18~26% 시간대에서 다르다).
- API에 아직 없는 최근 거래일(보통 오늘·어제)은 smp_real_epsis.csv의 smp_jeju로 채운다.

출력 열은 기존 SMP CSV와 같다. 피처 정의(v4.1)를 바꾸지 않도록 smp_mainland·smp_jeju 모두에
제주 하루전 가격을 넣고, smp_max/smp_min은 기존 파일과 같은 규칙(±2%)으로 만든다.

실행: python -m src.smp.data.jeju_da_history --output data/smp/jeju_da_smp_hourly.csv
"""

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

JEJU_SMP_API = "https://apis.data.go.kr/B552115/JejuSmpLfd2/getJejuSmpLfd2"
ROWS_PER_PAGE = 1000
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 6
# data.go.kr는 짧은 시간 연속 호출에 429를 돌려준다 (2026-11-05 전체 이력 수집 중 67번째 페이지에서 발생).
PAGE_INTERVAL_SECONDS = 0.5
RATE_LIMIT_BACKOFF_SECONDS = 10
# 기존 SMP 파일의 smp_max/smp_min은 가격 ×1.02 / ×0.98로 만든 값이다 (실측 일중 최고·최저 아님).
MAX_FACTOR, MIN_FACTOR = 1.02, 0.98
COLUMNS = ["timestamp", "date", "hour", "smp_mainland", "smp_jeju", "smp_max", "smp_min", "source", "is_synthetic"]


def api_key() -> str:
    key = os.environ.get("DATA_GO_KR_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DATA_GO_KR_API_KEY is not set")
    return key


def fetch_page(page_no: int, key: str, rows: int = ROWS_PER_PAGE) -> Tuple[List[dict], int]:
    """API 한 페이지(최신 날짜부터 내림차순)를 가져온다. 일시 오류는 재시도한다."""
    query = urllib.parse.urlencode({"serviceKey": key, "pageNo": page_no, "numOfRows": rows, "dataType": "JSON"}, safe="%")
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(f"{JEJU_SMP_API}?{query}", timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.load(response)["response"]
            if payload["header"]["resultCode"] != "00":
                raise RuntimeError(f"Jeju SMP API error {payload['header']}")
            body = payload["body"]
            items = body["items"]["item"] if body.get("items") else []
            return (items if isinstance(items, list) else [items]), int(body["totalCount"])
        except (urllib.error.URLError, TimeoutError, KeyError, ValueError) as exc:
            last_error = exc
            rate_limited = isinstance(exc, urllib.error.HTTPError) and exc.code == 429
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS * (attempt + 1) if rate_limited else 1 + attempt)
    raise RuntimeError(f"Jeju SMP API page {page_no} failed: {last_error}")


def hourly_from_items(items: Iterable[dict]) -> Dict[Tuple[str, int], float]:
    """15분 구간 4행을 시간 하나로 모은다. 하루전 가격은 구간마다 같아야 하며, 다르면 실패한다."""
    groups: Dict[Tuple[str, int], List[float]] = defaultdict(list)
    for item in items:
        price = item.get("jsmp")
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            raise ValueError(f"invalid jsmp: {item}")
        groups[(str(item["date"]), int(item["hour"]))].append(float(price))
    hourly = {}
    for key, prices in groups.items():
        if max(prices) - min(prices) > 1e-6:
            raise ValueError(f"day-ahead price differs within an hour: {key} {prices}")
        hourly[key] = prices[0]
    return hourly


def fetch_hourly(key: str, cache_dir: Path, max_pages: Optional[int] = None) -> Dict[Tuple[str, int], float]:
    """API 전체 이력을 페이지 단위로 받는다.

    페이지는 최신순이라 새 데이터가 쌓이면 번호가 밀린다. 그래서 1페이지(최신)는 항상 새로 받고,
    나머지는 (전체 행 수, 페이지 번호)로 캐시해 한도(429)로 중단돼도 다음 실행에서 이어서 받는다.
    중단되면 그때까지 받은 페이지만으로 결과를 돌려준다.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    first, total = fetch_page(1, key)
    pages = -(-total // ROWS_PER_PAGE)
    if max_pages is not None:
        pages = min(pages, max_pages)
    items = list(first)
    fetched = 1
    for page_no in range(2, pages + 1):
        cached = cache_dir / f"total{total}_page{page_no}.json"
        if cached.exists():
            items.extend(json.loads(cached.read_text(encoding="utf-8")))
            fetched += 1
            continue
        time.sleep(PAGE_INTERVAL_SECONDS)
        try:
            page_items, _ = fetch_page(page_no, key)
        except RuntimeError as exc:
            logger.warning("stopped at page %d/%d: %s (rerun later to resume)", page_no, pages, exc)
            break
        cached.write_text(json.dumps(page_items, ensure_ascii=False), encoding="utf-8")
        items.extend(page_items)
        fetched += 1
    logger.info("API rows %d from %d/%d pages (total rows %d)", len(items), fetched, pages, total)
    return hourly_from_items(items)


def to_frame(hourly: Dict[Tuple[str, int], float], source: str) -> pd.DataFrame:
    rows = []
    for (day, hour), price in hourly.items():
        date = pd.Timestamp(day)
        rows.append({
            "timestamp": f"{date:%Y-%m-%d} {hour:02d}:00", "date": f"{date:%Y-%m-%d}", "hour": hour,
            "smp_mainland": price, "smp_jeju": price,
            "smp_max": round(price * MAX_FACTOR, 4), "smp_min": round(price * MIN_FACTOR, 4),
            "source": source, "is_synthetic": False,
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def from_local(path: Path, source: str) -> pd.DataFrame:
    """기존 SMP 파일의 smp_jeju(제주 하루전 공표값)를 같은 형식으로 읽는다."""
    if not path.exists():
        logger.warning("local SMP file missing: %s", path)
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, encoding="utf-8-sig")
    hourly = {(pd.Timestamp(d).strftime("%Y%m%d"), int(h)): float(p)
              for d, h, p in zip(df["date"], df["hour"], df["smp_jeju"])
              if pd.notna(p) and pd.notna(h) and 1 <= int(h) <= 24}
    return to_frame(hourly, source)


def merge_history(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """앞의 원천을 우선한다 (같은 거래일·시간이 여러 원천에 있으면 첫 번째 값을 쓴다)."""
    merged = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    merged = merged.drop_duplicates(["date", "hour"], keep="first")
    return merged.sort_values(["date", "hour"]).reset_index(drop=True)


def build(output: Path, local_paths: List[Path], cache_dir: Path, use_api: bool = True,
          max_pages: Optional[int] = None) -> pd.DataFrame:
    """API(공식) → 로컬 파일 순으로 우선해 시간별 연속 이력을 만든다."""
    frames = []
    if use_api:
        try:
            frames.append(to_frame(fetch_hourly(api_key(), cache_dir, max_pages), "제주시범사업 API jsmp"))
        except RuntimeError as exc:
            logger.warning("Jeju SMP API unavailable (%s); building from local files only", exc)
    # 최신 파일(KPX 포털 기반)을 5년 파일보다 우선한다.
    frames.extend(from_local(path, f"EPSIS {path.name} (smp_jeju)") for path in local_paths)
    history = merge_history(frames)
    output.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(output, index=False, encoding="utf-8-sig")
    logger.info("wrote %s: %d rows, %s ~ %s, sources %s", output, len(history), history["date"].min(),
                history["date"].max(), history["source"].value_counts().to_dict())
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hourly Jeju day-ahead SMP history")
    parser.add_argument("--output", type=Path, default=Path("data/smp/jeju_da_smp_hourly.csv"))
    parser.add_argument("--local", type=Path, nargs="*",
                        default=[Path("data/smp/smp_real_epsis.csv"), Path("data/smp/smp_5years_epsis.csv")])
    parser.add_argument("--cache-dir", type=Path, default=Path("data/smp/.jeju_api_pages"))
    parser.add_argument("--no-api", action="store_true", help="로컬 파일만 병합 (API 한도 소진 시)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    build(args.output, args.local, args.cache_dir, use_api=not args.no_api)


if __name__ == "__main__":
    main()
