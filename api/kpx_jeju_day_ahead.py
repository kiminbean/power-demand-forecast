"""Read official KPX Jeju day-ahead SMP; never a real-time-market price."""

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import re
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

SOURCE_URL = "https://www.kpx.or.kr/smpJeju.es?mid=a10606080200&device=pc"
DATA_SOURCE = "KPX 제주 하루전시장"
SEOUL = ZoneInfo("Asia/Seoul")
TIMEOUT_SECONDS = 10.0
DAYS_IN_TABLE = 7
HOURS_IN_DAY = 24
_DATE = re.compile(r"(?<!\d)(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*거래분")
_HEADING = re.compile(r"(\d{2})\.(\d{2})(?:\s*\([월화수목금토일]\))?")
_PRICE = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)")
_BLOCK = re.compile(
    r"접근\s*(?:이\s*)?(?:차단|거부)|보안\s*(?:차단|검사)|"
    r"access\s+denied|security\s+(?:check|challenge)|captcha|cloudflare",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class JejuDayAheadSmp:
    """A Jeju day-ahead price in KRW/kWh, not real-time SMP."""

    trade_day: str
    trade_hour: int
    price: float
    data_source: str = DATA_SOURCE
    source_url: str = SOURCE_URL
    market_type: str = "day_ahead"


def parse_jeju_day_ahead(html: str, now: datetime | None = None) -> JejuDayAheadSmp | None:
    """Parse a verified portal table for the current Seoul day and hour.

    Args:
        html: Official portal HTML.
        now: Timezone-aware instant; defaults to the current time in Seoul.

    Returns:
        Day-ahead price with provenance, or None if any required data is unsafe.
    """
    if now is None:
        now = datetime.now(SEOUL)
    if now.tzinfo is None or now.utcoffset() is None:
        return None
    now = now.astimezone(SEOUL)
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    text = soup.get_text(" ", strip=True)
    if (
        "제주(하루전시장)" not in title
        or "계통한계가격(SMP)" not in title
        or _BLOCK.search(text)
    ):
        return None
    # 전체 연도가 표시된 거래일을 기준으로 과거 페이지를 거부한다.
    dates = _DATE.findall(text)
    if len(dates) != 1:
        return None
    try:
        year, month, day = map(int, dates[0])
        reference = datetime(year, month, day).date()
    except ValueError:
        return None
    if reference != now.date():
        return None
    units = [p.get_text(" ", strip=True) for p in soup.select("p.info_top")]
    if len(units) != 1 or not re.fullmatch(r"단위\s*:\s*원/kWh", units[0]):
        return None
    tables = [t for t in soup.find_all("table") if t.caption and t.caption.get_text(" ", strip=True) == "제주 SMP"]
    if len(tables) != 1:
        return None
    table = tables[0]
    thead, tbody = table.find("thead"), table.find("tbody")
    if not thead or not tbody:
        return None
    head_rows = thead.find_all("tr", recursive=False)
    if len(head_rows) != 1:
        return None
    headings = head_rows[0].find_all(["th", "td"], recursive=False)
    if len(headings) != DAYS_IN_TABLE + 1 or headings[0].get_text(" ", strip=True) != "구분":
        return None
    if any(cell.has_attr("rowspan") or cell.has_attr("colspan") for cell in headings):
        return None
    # 거래일을 마지막 날짜로 삼아 연말 전후 열의 연도를 안전하게 확정한다.
    for index, cell in enumerate(headings[1:]):
        match = _HEADING.fullmatch(cell.get_text(" ", strip=True))
        expected = reference - timedelta(days=DAYS_IN_TABLE - 1 - index)
        if not match or (int(match[1]), int(match[2])) != (expected.month, expected.day):
            return None
    rows = {}
    summaries = set()
    for row in tbody.find_all("tr", recursive=False):
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) != DAYS_IN_TABLE + 1 or any(
            cell.has_attr("rowspan") or cell.has_attr("colspan") for cell in cells
        ):
            return None
        label = cells[0].get_text(" ", strip=True)
        match = re.fullmatch(r"([1-9]|1\d|2[0-4])h", label)
        if match:
            hour = int(match[1])
            if hour in rows:
                return None
            # 빈 셀 삽입으로 날짜별 가격이 밀리는 경우를 거부한다.
            if any(not _PRICE.fullmatch(cell.get_text(" ", strip=True)) for cell in cells[1:]):
                return None
            rows[hour] = cells[-1].get_text(" ", strip=True)
        elif label in {"최대", "최소", "가중평균"} and label not in summaries:
            summaries.add(label)
        else:
            return None
    if set(rows) != set(range(1, HOURS_IN_DAY + 1)):
        return None
    # 거래시간은 구간의 끝점이며 정각은 직전 구간에 속한다.
    # https://marketrule.kpx.or.kr/lmxsrv/law/twoViewForm.do?SEQ=2&SEQ_HISTORY=33&LAWGROUP=1
    at_boundary = now.minute == now.second == now.microsecond == 0
    if at_boundary and now.hour == 0:
        return None
    hour = now.hour if at_boundary else now.hour + 1
    value = rows[hour]
    if not _PRICE.fullmatch(value):
        return None
    try:
        price = float(value.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(price):
        return None
    return JejuDayAheadSmp(now.strftime("%Y%m%d"), hour, price)


async def fetch_jeju_day_ahead(now: datetime | None = None) -> JejuDayAheadSmp | None:
    """Fetch the official Jeju day-ahead SMP with a bounded timeout.

    Args:
        now: Optional timezone-aware instant for the requested Seoul hour.

    Returns:
        Verified day-ahead price, or None on network or validation failure.
    """
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, follow_redirects=False) as client:
            response = await client.get(SOURCE_URL)
            response.raise_for_status()
        return parse_jeju_day_ahead(response.text, now)
    except httpx.HTTPError:
        return None
