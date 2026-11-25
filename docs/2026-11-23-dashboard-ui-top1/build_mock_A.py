"""Direction A(지도 주인공형) 정적 목업 생성기.

- 발전소 위치·용량은 실제 등록부(data/jeju_plants/jeju_power_plants.json, 운영중·분산형 제외)와 같은 값을 하드코딩했다.
- 수급 곡선은 app_v4.create_realtime_power_chart의 일간 패턴 식을 그대로 따라 계산한다(난수는 고정 시드).
- 출력: direction_A.html (같은 폴더). 외부 JS 없음, 글꼴만 jsDelivr Pretendard CSS.

사용: .venv/bin/python -I build_mock_A.py
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from urllib.parse import quote

OUT = Path(__file__).resolve().parent / "direction_A.html"
SEED = 202611232335 // 300  # N4: 5분 버킷 시드

# ----------------------------------------------------------------- 공용 색
S = {
    "demand": "#3B82F6", "supply": "#64748B", "reserve": "#16A34A", "solar": "#B7791F",
    "wind": "#0891B2", "ess": "#8B5CF6", "thermal": "#8D6E63", "hvdc": "#7C8DB0", "smp": "#C026D3",
}
TYPE_KO = {"wind": "풍력", "solar": "태양광", "ess": "ESS", "thermal": "화력"}

# ----------------------------------------------------------------- 등록부 (운영중, 분산형 태양광 제외)
PLANTS = [
    ("한경풍력발전단지", "wind", 33.339417, 126.169222, 21.0, "제주시"),
    ("상명풍력발전단지", "wind", 33.33925, 126.289556, 21.0, "제주시"),
    ("가시리풍력발전단지", "wind", 33.3576, 126.7461, 30.0, "서귀포시"),
    ("김녕풍력발전단지", "wind", 33.556, 126.748, 30.0, "제주시"),
    ("행원풍력발전단지", "wind", 33.549, 126.799, 8.385, "제주시"),
    ("삼달풍력발전단지", "wind", 33.379, 126.863, 33.0, "서귀포시"),
    ("성산1풍력발전단지", "wind", 33.4399, 126.9229, 12.0, "서귀포시"),
    ("성산2풍력발전단지", "wind", 33.435, 126.918, 8.0, "서귀포시"),
    ("신창풍력발전단지", "wind", 33.365, 126.174, 1.7, "제주시"),
    ("월정풍력발전단지", "wind", 33.5556, 126.7925, 1.5, "제주시"),
    ("탐라해상풍력발전단지", "wind", 33.2942, 126.1631, 30.0, "제주시"),
    ("가파도풍력발전단지", "wind", 33.1637, 126.2742, 0.5, "서귀포시"),
    ("월령풍력발전단지", "wind", 33.417, 126.233, 2.0, "제주시"),
    ("제주음풍력발전단지", "wind", 33.48, 126.55, 21.0, "제주시"),
    ("동복-북촌풍력발전단지", "wind", 33.53, 126.72, 30.0, "제주시"),
    ("대정풍력발전단지", "wind", 33.22, 126.28, 45.0, "서귀포시"),
    ("구좌태양광발전소", "solar", 33.52, 126.85, 0.75, "제주시"),
    ("송당리태양광", "solar", 33.45, 126.78, 1.834, "제주시"),
    ("서화리태양광", "solar", 33.42, 126.76, 1.5, "서귀포시"),
    ("가시리태양광", "solar", 33.36, 126.75, 1.2, "서귀포시"),
    ("제주 변동성완화 ESS 1", "ess", 33.51, 126.54, 30.0, "제주시"),
    ("제주 피크저감 ESS", "ess", 33.45, 126.56, 20.0, "제주시"),
    ("한경풍력 연계 ESS", "ess", 33.34, 126.17, 10.0, "제주시"),
    ("가시리풍력 연계 ESS", "ess", 33.3576, 126.75, 15.0, "서귀포시"),
    ("남제주화력 1,2호기", "thermal", 33.267, 126.515, 200.0, "서귀포시"),
    ("남제주 LNG복합", "thermal", 33.268, 126.516, 158.8, "서귀포시"),
    ("제주 LNG복합화력", "thermal", 33.492, 126.475, 240.0, "제주시"),
]

COAST = [
    (126.162, 33.296), (126.168, 33.318), (126.172, 33.338), (126.168, 33.352), (126.176, 33.366),
    (126.196, 33.379), (126.214, 33.392), (126.224, 33.410), (126.236, 33.423), (126.252, 33.421),
    (126.266, 33.419), (126.284, 33.436), (126.306, 33.452), (126.330, 33.466), (126.358, 33.475),
    (126.384, 33.482), (126.410, 33.489), (126.444, 33.500), (126.470, 33.506), (126.500, 33.514),
    (126.528, 33.519), (126.552, 33.521), (126.580, 33.525), (126.610, 33.533), (126.636, 33.540),
    (126.668, 33.546), (126.700, 33.551), (126.728, 33.557), (126.752, 33.563), (126.780, 33.565),
    (126.800, 33.562), (126.828, 33.552), (126.852, 33.536), (126.876, 33.522), (126.898, 33.508),
    (126.918, 33.488), (126.930, 33.470), (126.944, 33.462), (126.940, 33.450), (126.928, 33.440),
    (126.916, 33.424), (126.906, 33.406), (126.892, 33.388), (126.876, 33.370), (126.858, 33.348),
    (126.840, 33.328), (126.812, 33.312), (126.780, 33.296), (126.748, 33.284), (126.716, 33.274),
    (126.684, 33.268), (126.652, 33.262), (126.618, 33.252), (126.586, 33.244), (126.560, 33.240),
    (126.530, 33.236), (126.500, 33.231), (126.470, 33.225), (126.440, 33.236), (126.410, 33.241),
    (126.380, 33.236), (126.350, 33.234), (126.326, 33.232), (126.304, 33.226), (126.296, 33.206),
    (126.280, 33.202), (126.258, 33.206), (126.236, 33.212), (126.212, 33.222), (126.190, 33.240),
    (126.176, 33.262), (126.166, 33.280),
]
UDO = [(126.940, 33.512), (126.948, 33.522), (126.962, 33.520), (126.968, 33.508), (126.962, 33.494),
       (126.950, 33.490), (126.942, 33.498)]

MAP_W, MAP_H = 1000, 625
LON0, LAT0 = 126.10, 33.605
KX, KY = 1111.1, 1330.6


def proj(lon: float, lat: float) -> tuple[float, float]:
    return ((lon - LON0) * KX, (LAT0 - lat) * KY)


def poly(points: list[tuple[float, float]]) -> str:
    pts = [proj(lo, la) for lo, la in points]
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + " Z"


def fmt(v: float, d: int = 1) -> str:
    return f"{v:,.{d}f}"


# ----------------------------------------------------------------- 아이콘 (CSS 마스크, 퍼센트 인코딩)
ICONS = {
    "map": "<path d='M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2z'/><path d='M9 4v14M15 6v14'/>",
    "bolt": "<path d='M13 2 3 14h9l-1 8 10-12h-9l1-8z'/>",
    "sun": "<circle cx='12' cy='12' r='4'/><path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4'/>",
    "wind": "<path d='M3 8h11a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h8'/>",
    "battery": "<rect x='2' y='7' width='17' height='10' rx='2'/><path d='M22 11v2M6 10v4M10 10v4'/>",
    "flame": "<path d='M12 2s5 4.5 5 10a5 5 0 0 1-10 0c0-2.5 1.5-4 1.5-4S9 11 11 11c0-3 1-6 1-9z'/>",
    "plug": "<path d='M9 2v5M15 2v5M6 7h12v4a6 6 0 0 1-12 0V7zM12 17v5'/>",
    "clock": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/>",
    "refresh": "<path d='M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5'/>",
    "info": "<circle cx='12' cy='12' r='9'/><path d='M12 11v5M12 8h.01'/>",
    "alert": "<path d='M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'/><path d='M12 9v4M12 17h.01'/>",
    "chart": "<path d='M3 3v18h18'/><path d='M7 15l4-4 3 3 5-6'/>",
    "tag": "<path d='M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z'/><circle cx='7.5' cy='7.5' r='1.5'/>",
    "spark": "<circle cx='12' cy='12' r='3'/><path d='M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1'/>",
    "thermo": "<path d='M14 14.8V3.5a2.5 2.5 0 0 0-5 0v11.3a4.5 4.5 0 1 0 5 0z'/>",
    "db": "<ellipse cx='12' cy='5' rx='8' ry='3'/><path d='M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3'/>",
    "plus": "<path d='M12 5v14M5 12h14'/>",
    "minus": "<path d='M5 12h14'/>",
    "layers": "<path d='M12 2 2 7l10 5 10-5-10-5z'/><path d='M2 17l10 5 10-5M2 12l10 5 10-5'/>",
    "gauge": "<path d='M12 14l4-4'/><path d='M3.3 19a10 10 0 1 1 17.4 0'/>",
    "chev": "<path d='M9 6l6 6-6 6'/>",
    "chev2": "<path d='M6 7l5 5-5 5M13 7l5 5-5 5'/>",
    "leaf": "<path d='M11 20A7 7 0 0 1 4 13c0-6 6-9 16-9 0 10-3 16-9 16z'/><path d='M4 20c4-6 8-9 12-11'/>",
    "list": "<path d='M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01'/>",
}


def icon_css() -> str:
    rules = []
    for name, body in ICONS.items():
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
            "stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>" + body + "</svg>"
        )
        # '<', '>', '#', '"'를 인코딩해 style 블록에 날 '<'가 남지 않게 한다(DOMPurify가 블록째 지움, 실측).
        enc = quote(svg, safe=" =':/,.-")
        rules.append(f'.pd-ico--{name}{{--ico:url("data:image/svg+xml,{enc}")}}')
    return "\n".join(rules)


def ico(name: str, extra: str = "") -> str:
    return f'<span class="pd-ico pd-ico--{name} {extra}" aria-hidden="true"></span>'


def chip(kind: str, text: str) -> str:
    return f'<span class="pd-chip pd-chip--{kind}">{text}</span>'


# ----------------------------------------------------------------- 수급 곡선 (app_v4 패턴 식 재현)
def load_pattern(h: int) -> float:
    if 6 <= h <= 9:
        return 0.9 + 0.1 * (h - 6) / 3
    if 9 <= h <= 14:
        return 1.0 - 0.15 * (h - 9) / 5
    if 14 <= h <= 20:
        return 0.85 + 0.25 * (h - 14) / 6
    if 20 <= h <= 23:
        return 1.1 - 0.2 * (h - 20) / 3
    return 0.85


def supply_series() -> dict:
    rng = random.Random(SEED)
    cur_d, cur_s = 718.0, 1233.0
    base_d, base_s = cur_d / load_pattern(23), cur_s / load_pattern(23)
    # 30분 간격, now=23:35 → 과거 24점(11:35~23:35), 미래 24점(00:05~11:35)
    past_min = [23 * 60 + 35 - 30 * i for i in range(23, -1, -1)]
    pts_d, pts_s = [], []
    for m in past_min:
        h = (m // 60) % 24
        n = rng.uniform(-0.02, 0.02)
        pts_d.append(base_d * load_pattern(h) * (1 + n))
        pts_s.append(base_s * load_pattern(h) * (1 + n * 0.5))
    pts_d[-1], pts_s[-1] = cur_d, cur_s
    fut_min = [23 * 60 + 35 + 30 * i for i in range(1, 25)]
    f_d, f_s, f_u, f_l = [], [], [], []
    for i, m in enumerate(fut_min, start=1):
        h = (m // 60) % 24
        d = base_d * load_pattern(h)
        s = base_s * load_pattern(h)
        unc = 0.02 + i * 0.003
        f_d.append(d), f_s.append(s), f_u.append(d * (1 + unc)), f_l.append(d * (1 - unc))
    return {"past_min": past_min, "fut_min": fut_min, "pd": pts_d, "ps": pts_s, "fd": f_d, "fs": f_s,
            "fu": f_u, "fl": f_l, "cur_d": cur_d, "cur_s": cur_s}


def hhmm(m: int) -> str:
    m %= 24 * 60
    return f"{m // 60:02d}:{m % 60:02d}"


def supply_chart_svg(w: int, h: int, compact: bool) -> str:
    d = supply_series()
    ml, mr, mt, mb = (44, 14, 30, 30) if compact else (56, 20, 34, 34)
    pw, ph = w - ml - mr, h - mt - mb
    t0, t1 = d["past_min"][0], d["fut_min"][-1]
    ymax = 1600

    def X(m: float) -> float:
        return ml + (m - t0) / (t1 - t0) * pw

    def Y(v: float) -> float:
        return mt + ph - v / ymax * ph

    now = 23 * 60 + 35
    out = [f'<svg class="pd-chart-svg" viewBox="0 0 {w} {h}" role="img" aria-label="수급 추이 차트: 지난 12시간과 다음 12시간의 수요·공급능력 일간 패턴 추정, 현재값 수요 718 MW와 공급능력 1,233 MW는 KPX 실측">']
    out.append(
        '<defs>'
        f'<pattern id="hatch-res-{w}" patternUnits="userSpaceOnUse" width="8" height="8" patternTransform="rotate(45)">'
        f'<rect width="8" height="8" fill="{S["reserve"]}" fill-opacity=".06"/>'
        f'<line x1="0" y1="0" x2="0" y2="8" stroke="{S["reserve"]}" stroke-width="2" stroke-opacity=".38"/></pattern>'
        '</defs>'
    )
    # 미래 구간 음영
    out.append(f'<rect x="{X(now):.1f}" y="{mt}" width="{X(t1) - X(now):.1f}" height="{ph}" class="ch-future"/>')
    # 격자와 y눈금
    for v in range(0, ymax + 1, 400):
        y = Y(v)
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" class="ch-grid"/>')
        out.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" class="ch-tick" text-anchor="end">{v:,}</text>')
    out.append(f'<text x="{ml - 8}" y="{mt - 14}" class="ch-unit" text-anchor="end">MW</text>')
    # x 눈금
    step = 6 * 60 if compact else 3 * 60
    first = ((t0 // step) + 1) * step
    for m in range(first, t1 + 1, step):
        out.append(f'<text x="{X(m):.1f}" y="{mt + ph + 20}" class="ch-tick" text-anchor="middle">{hhmm(m)}</text>')
    # 예비력 빗금 (공급-수요 사이) 과거+미래
    xs = d["past_min"] + d["fut_min"]
    sup = d["ps"] + d["fs"]
    dem = d["pd"] + d["fd"]
    top = " L".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(xs, sup))
    bot = " L".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(reversed(xs), reversed(dem)))
    out.append(f'<path d="M{top} L{bot} Z" fill="url(#hatch-res-{w})"/>')
    # 미래 가정 범위
    fx = [now] + d["fut_min"]
    fu = [d["cur_d"]] + d["fu"]
    fl = [d["cur_d"]] + d["fl"]
    up = " L".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(fx, fu))
    lo = " L".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(reversed(fx), reversed(fl)))
    out.append(f'<path d="M{up} L{lo} Z" fill="{S["demand"]}" fill-opacity=".12"/>')
    # 공급능력·수요 점선 (추정 패턴)
    for series, col, wdt in ((sup, S["supply"], 1.8), (dem, S["demand"], 2.4)):
        path = " L".join(f"{X(m):.1f},{Y(v):.1f}" for m, v in zip(xs, series))
        out.append(f'<path d="M{path}" fill="none" stroke="{col}" stroke-width="{wdt}" stroke-dasharray="2 4" stroke-linecap="round"/>')
    # 현재선 + pill
    xn = X(now)
    out.append(f'<line x1="{xn:.1f}" x2="{xn:.1f}" y1="{mt}" y2="{mt + ph}" stroke="{S["supply"]}" stroke-width="1.5"/>')
    pill_w = 92 if compact else 104
    out.append(f'<rect x="{xn - pill_w / 2:.1f}" y="{mt - 24}" width="{pill_w}" height="20" rx="10" fill="{S["supply"]}"/>')
    out.append(f'<text x="{xn:.1f}" y="{mt - 10}" class="ch-pill" text-anchor="middle">현재 23:35 KST</text>')
    # 미래 라벨
    if not compact:
        out.append(f'<text x="{X(t1) - 8:.1f}" y="{mt + 16}" class="ch-note" text-anchor="end">다음 12시간 · 일간 패턴 연장 (예측 모델 아님)</text>')
        out.append(f'<text x="{ml + 8}" y="{mt + 16}" class="ch-note">지난 12시간 · 일간 패턴 추정 (현재값만 실측)</text>')
    # KPX 실측 점
    for v, col, label in ((d["cur_s"], S["supply"], "1,233 MW"), (d["cur_d"], S["demand"], "718 MW")):
        out.append(f'<circle cx="{xn:.1f}" cy="{Y(v):.1f}" r="5.5" fill="{col}" class="ch-meas"/>')
        out.append(f'<text x="{xn - 10:.1f}" y="{Y(v) - 10:.1f}" class="ch-val" text-anchor="end">{label}</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- 시간대별 재생 출력 추정 (등록부 기준)
def hourly_profile() -> tuple[list[float], list[float]]:
    rng = random.Random(SEED + 7)
    solar_cap = sum(p[4] for p in PLANTS if p[1] == "solar")
    wind_cap = sum(p[4] for p in PLANTS if p[1] == "wind")
    solar = [solar_cap * math.sin(math.pi * (h - 6) / 12) * 0.8 if 6 <= h <= 18 else 0.0 for h in range(24)]
    wind = [wind_cap * rng.uniform(0.4, 0.7) for _ in range(24)]
    return wind, solar


def hourly_bars_svg(w: int, h: int, current: int = 23, big: bool = False) -> str:
    wind, solar = hourly_profile()
    ml, mr, mt, mb = (8, 40, 18, 24) if not big else (12, 48, 20, 28)
    pw, ph = w - ml - mr, h - mt - mb
    ymax = 250
    bw = pw / 24
    out = [f'<svg class="pd-chart-svg" viewBox="0 0 {w} {h}" role="img" aria-label="시간대별 풍력·태양광 출력 추정 막대 0시~23시, 시뮬레이션 추정치">']
    out.append(
        '<defs>'
        f'<pattern id="hw-{w}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="{S["wind"]}" fill-opacity=".22"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{S["wind"]}" stroke-width="2.2" stroke-opacity=".75"/></pattern>'
        f'<pattern id="hs-{w}" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">'
        f'<rect width="6" height="6" fill="{S["solar"]}" fill-opacity=".25"/>'
        f'<line x1="0" y1="0" x2="0" y2="6" stroke="{S["solar"]}" stroke-width="2.2" stroke-opacity=".8"/></pattern>'
        '</defs>'
    )
    for v in (0, 100, 200):
        y = mt + ph - v / ymax * ph
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" class="ch-grid"/>')
        out.append(f'<text x="{ml + pw + 6}" y="{y + 4:.1f}" class="ch-tick">{v}</text>')
    for i in range(24):
        x = ml + i * bw + bw * 0.14
        bwi = bw * 0.72
        hw = wind[i] / ymax * ph
        hs = solar[i] / ymax * ph
        y_w = mt + ph - hw
        out.append(f'<rect x="{x:.1f}" y="{y_w:.1f}" width="{bwi:.1f}" height="{hw:.1f}" fill="url(#hw-{w})" stroke="{S["wind"]}" stroke-width=".8"/>')
        if hs > 0.4:
            out.append(f'<rect x="{x:.1f}" y="{y_w - hs:.1f}" width="{bwi:.1f}" height="{hs:.1f}" fill="url(#hs-{w})" stroke="{S["solar"]}" stroke-width=".8"/>')
        if i == current:
            out.append(f'<rect x="{x - 2:.1f}" y="{mt - 2}" width="{bwi + 4:.1f}" height="{ph + 4}" rx="3" class="ch-cur"/>')
    for hh in (0, 6, 12, 18):
        x = ml + hh * bw + bw / 2
        out.append(f'<text x="{x:.1f}" y="{mt + ph + 17}" class="ch-tick" text-anchor="middle">{hh}시</text>')
    x = ml + current * bw + bw / 2
    out.append(f'<text x="{x:.1f}" y="{mt + ph + 17}" class="ch-tick ch-tick--strong" text-anchor="middle">{current}시</text>')
    out.append(f'<text x="{ml + pw + 6}" y="{mt - 6}" class="ch-unit">MW</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- SMP (예측 q50 계단 + 80% 구간)
SMP_Q50 = [182, 176, 171, 168, 170, 179, 188, 172, 141, 117, 104, 100, 101, 109, 124, 149, 176, 196, 203, 201, 194, 188, 191, 199]
SMP_W = [14, 14, 13, 13, 13, 14, 16, 18, 22, 26, 28, 30, 30, 28, 25, 22, 20, 18, 16, 16, 16, 16, 16, 17]
SMP_NOW = 208.9
SMP_STATS = {"avg": 158.2, "max": 210.4, "min": 88.2}


def smp_chart_svg(w: int, h: int, compact: bool) -> str:
    ml, mr, mt, mb = (40, 16, 30, 30) if compact else (52, 92, 34, 34)
    pw, ph = w - ml - mr, h - mt - mb
    ymin, ymax = 40, 260
    bw = pw / 24

    def X(i: float) -> float:
        return ml + i * bw

    def Y(v: float) -> float:
        return mt + ph - (v - ymin) / (ymax - ymin) * ph

    out = [f'<svg class="pd-chart-svg" viewBox="0 0 {w} {h}" role="img" aria-label="제주 하루전 SMP 24시간 예측 계단선과 80% 구간, 현재 거래시간 24h 실측 208.9원">']
    # 지난 거래시간 음영 (0..23)
    out.append(f'<rect x="{X(0):.1f}" y="{mt}" width="{X(23) - X(0):.1f}" height="{ph}" class="ch-past"/>')
    # 현재 거래시간 밴드
    out.append(f'<rect x="{X(23):.1f}" y="{mt}" width="{bw:.1f}" height="{ph}" class="ch-curband"/>')
    for v in (50, 100, 150, 200, 250):
        y = Y(v)
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" class="ch-grid"/>')
        out.append(f'<text x="{ml - 8}" y="{y + 4:.1f}" class="ch-tick" text-anchor="end">{v}</text>')
    out.append(f'<text x="{ml - 8}" y="{mt - 14}" class="ch-unit" text-anchor="end">원/kWh</text>')
    # 80% 구간 (계단)
    up, lo = [], []
    for i, (q, wd) in enumerate(zip(SMP_Q50, SMP_W)):
        up += [(X(i), Y(q + wd)), (X(i + 1), Y(q + wd))]
        lo += [(X(i), Y(q - wd)), (X(i + 1), Y(q - wd))]
    band = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in up) + " L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in reversed(lo)) + " Z"
    out.append(f'<path d="{band}" fill="{S["smp"]}" fill-opacity=".13"/>')
    # q50 계단 점선
    st = []
    for i, q in enumerate(SMP_Q50):
        st += [(X(i), Y(q)), (X(i + 1), Y(q))]
    out.append('<path d="M' + " L".join(f"{x:.1f},{y:.1f}" for x, y in st) + f'" fill="none" stroke="{S["smp"]}" stroke-width="2.2" stroke-dasharray="6 4"/>')
    # 일평균·최고·최저 기준선 (EPSIS 기준일) 오른쪽 끝 라벨
    for key, label in (("max", "일최고"), ("avg", "일평균"), ("min", "일최저")):
        y = Y(SMP_STATS[key])
        out.append(f'<line x1="{ml}" x2="{ml + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="{S["supply"]}" stroke-width="1.2" stroke-dasharray="1 4" stroke-linecap="round"/>')
        if not compact:
            out.append(f'<text x="{ml + pw + 8}" y="{y + 4:.1f}" class="ch-ref">{label} {SMP_STATS[key]}</text>')
    # 실측 현재가 점
    xc = X(23.5)
    out.append(f'<circle cx="{xc:.1f}" cy="{Y(SMP_NOW):.1f}" r="5.5" fill="{S["smp"]}" class="ch-meas"/>')
    out.append(f'<text x="{xc - 10:.1f}" y="{Y(SMP_NOW) - 10:.1f}" class="ch-val" text-anchor="end">208.9 실측</text>')
    # 현재 거래시간 pill
    pill_w = 96
    out.append(f'<rect x="{xc - pill_w + 10:.1f}" y="{mt - 24}" width="{pill_w}" height="20" rx="10" fill="{S["supply"]}"/>')
    out.append(f'<text x="{xc - pill_w / 2 + 10:.1f}" y="{mt - 10}" class="ch-pill" text-anchor="middle">거래시간 24h</text>')
    step = 6 if compact else 3
    for i in range(0, 24, step):
        out.append(f'<text x="{X(i) + bw / 2:.1f}" y="{mt + ph + 20}" class="ch-tick" text-anchor="middle">{i:02d}:00</text>')
    out.append("</svg>")
    return "".join(out)


# ----------------------------------------------------------------- 게이지·도넛
def arc(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    """위쪽 반원에서 왼쪽(비율 0)→오른쪽(비율 1)으로 그리는 호. 반원 안의 호는 항상 180° 이하라 large-arc=0."""
    t0, t1 = math.pi * (1 - a0), math.pi * (1 - a1)
    x0, y0 = cx + r * math.cos(t0), cy - r * math.sin(t0)
    x1, y1 = cx + r * math.cos(t1), cy - r * math.sin(t1)
    return f"M{x0:.1f},{y0:.1f} A{r},{r} 0 0 1 {x1:.1f},{y1:.1f}"


def gauge_svg(value: float, vmin: float, vmax: float, color: str, steps: list[tuple[float, float, str, float]],
              threshold: float, label: str, unit: str, marks: list[tuple[float, str]]) -> str:
    w, h = 300, 150
    cx, cy, r = 150, 122, 88
    out = [f'<svg class="pd-gauge-svg" viewBox="0 0 {w} {h}" role="img" aria-label="{label} 게이지 {value:,.1f}{unit}">']
    out.append(f'<path d="{arc(cx, cy, r, 0, 1)}" class="g-track" stroke-width="16" fill="none"/>')
    for a, b, col, op in steps:
        out.append(f'<path d="{arc(cx, cy, r + 13, (a - vmin) / (vmax - vmin), (b - vmin) / (vmax - vmin))}" '
                   f'stroke="{col}" stroke-opacity="{op}" stroke-width="5" fill="none"/>')
    frac = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    out.append(f'<path d="{arc(cx, cy, r, 0, frac)}" stroke="{color}" stroke-width="16" fill="none"/>')
    tf = (threshold - vmin) / (vmax - vmin)
    t = math.pi * (1 - tf)
    x0, y0 = cx + (r - 12) * math.cos(t), cy - (r - 12) * math.sin(t)
    x1, y1 = cx + (r + 18) * math.cos(t), cy - (r + 18) * math.sin(t)
    out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#DC2626" stroke-width="2.5"/>')
    # 축 양 끝 라벨은 호 아래, 기준선 라벨은 호 바깥
    out.append(f'<text x="{cx - r:.1f}" y="{cy + 18}" class="ch-tick" text-anchor="middle">{vmin:,.0f}</text>')
    out.append(f'<text x="{cx + r:.1f}" y="{cy + 18}" class="ch-tick" text-anchor="middle">{vmax:,.0f}</text>')
    for mv, mtxt in marks:
        f = (mv - vmin) / (vmax - vmin)
        t = math.pi * (1 - f)
        x, y = cx + (r + 30) * math.cos(t), cy - (r + 30) * math.sin(t)
        anchor = "end" if f < 0.45 else ("start" if f > 0.55 else "middle")
        out.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" class="ch-tick" text-anchor="{anchor}">{mtxt}</text>')
    out.append(f'<text x="{cx}" y="{cy - 6}" class="g-val" text-anchor="middle">{value:,.1f}<tspan class="g-unit"> {unit}</tspan></text>')
    out.append("</svg>")
    return "".join(out)


def donut_svg(parts: list[tuple[str, float, str]]) -> str:
    w = h = 220
    cx = cy = 110
    r, rin = 92, 60
    total = sum(v for _, v, _ in parts)
    out = [f'<svg class="pd-gauge-svg" viewBox="0 0 {w} {h}" role="img" aria-label="전력 공급 구성 도넛, 시뮬레이션 추정">']
    out.append("<defs>")
    for name, _, col in parts:
        out.append(
            f'<pattern id="dn-{name}" patternUnits="userSpaceOnUse" width="7" height="7" patternTransform="rotate(45)">'
            f'<rect width="7" height="7" fill="{col}" fill-opacity=".35"/>'
            f'<line x1="0" y1="0" x2="0" y2="7" stroke="{col}" stroke-width="3" stroke-opacity=".9"/></pattern>'
        )
    out.append("</defs>")
    a = -math.pi / 2
    for name, v, col in parts:
        if v <= 0:
            continue
        b = a + 2 * math.pi * v / total
        large = 1 if b - a > math.pi else 0
        x0, y0 = cx + r * math.cos(a), cy + r * math.sin(a)
        x1, y1 = cx + r * math.cos(b), cy + r * math.sin(b)
        xi1, yi1 = cx + rin * math.cos(b), cy + rin * math.sin(b)
        xi0, yi0 = cx + rin * math.cos(a), cy + rin * math.sin(a)
        out.append(
            f'<path d="M{x0:.1f},{y0:.1f} A{r},{r} 0 {large} 1 {x1:.1f},{y1:.1f} L{xi1:.1f},{yi1:.1f} '
            f'A{rin},{rin} 0 {large} 0 {xi0:.1f},{yi0:.1f} Z" fill="url(#dn-{name})" stroke="{col}" stroke-width="1.2"/>'
        )
        a = b
    out.append(f'<text x="{cx}" y="{cy - 2}" class="g-val" text-anchor="middle">≈760</text>')
    out.append(f'<text x="{cx}" y="{cy + 18}" class="ch-tick" text-anchor="middle">MW · 추정 합계</text>')
    out.append("</svg>")
    return "".join(out)


def cap_bar(parts: dict[str, float]) -> str:
    total = sum(parts.values())
    segs = "".join(
        f'<span style="flex:{v:.2f};background:var(--s-{k})" title="{TYPE_KO[k]} {fmt(v)} MW"></span>'
        for k, v in parts.items() if v > 0
    )
    return f'<span class="pd-capbar" role="img" aria-label="설비 구성 합계 {fmt(total)} MW">{segs}</span>'


# ----------------------------------------------------------------- 지도
def map_svg() -> str:
    out = [f'<svg class="pd-map-svg" viewBox="0 0 {MAP_W} {MAP_H}" preserveAspectRatio="xMidYMid meet" role="img" '
           'aria-label="제주 발전소 지도: 풍력 16곳, 태양광 4곳, ESS 4곳, 화력 3곳. 점 크기는 설비용량.">']
    out.append(f'<rect width="{MAP_W}" height="{MAP_H}" class="map-sea"/>')
    out.append(f'<path d="{poly(COAST)}" class="map-land"/>')
    out.append(f'<path d="{poly(UDO)}" class="map-land"/>')
    for lo, la, rx, ry in ((126.2725, 33.168, 0.008, 0.006), (126.141, 33.309, 0.004, 0.003)):
        x, y = proj(lo, la)
        out.append(f'<ellipse cx="{x:.1f}" cy="{y:.1f}" rx="{rx * KX:.1f}" ry="{ry * KY:.1f}" class="map-land"/>')
    hx, hy = proj(126.533, 33.362)
    for rx, ry, op in ((250, 128, 0.55), (165, 86, 0.75), (78, 40, 1.0)):
        out.append(f'<ellipse cx="{hx:.1f}" cy="{hy:.1f}" rx="{rx}" ry="{ry}" transform="rotate(-9 {hx:.1f} {hy:.1f})" class="map-relief" opacity="{op}"/>')
    for name, lo, la, cls in (
        ("제주시", 126.515, 33.497, "map-label map-label--city"), ("서귀포시", 126.565, 33.257, "map-label map-label--city"),
        ("한라산", 126.533, 33.366, "map-label"), ("한림", 126.272, 33.405, "map-label"), ("성산", 126.905, 33.452, "map-label"),
        ("대정", 126.248, 33.232, "map-label"), ("표선", 126.818, 33.338, "map-label"), ("애월", 126.345, 33.452, "map-label"),
        ("조천", 126.640, 33.523, "map-label"), ("남원", 126.700, 33.290, "map-label"), ("우도", 126.954, 33.528, "map-label"),
    ):
        x, y = proj(lo, la)
        out.append(f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="middle">{name}</text>')
    # 점: 큰 것 먼저 그려 작은 점이 위에 오게 한다
    for name, typ, la, lo, cap, region in sorted(PLANTS, key=lambda p: -p[4]):
        x, y = proj(lo, la)
        r = max(4.0, min(14.0, 3.2 + 0.75 * math.sqrt(cap)))
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" class="dot dot--{typ}" style="--r:{r:.1f}px"><title>{name} · {TYPE_KO[typ]} · {fmt(cap)} MW · {region}</title></circle>')
    out.append("</svg>")
    return "".join(out)


def callout_html() -> str:
    def region(name: str) -> tuple[int, float, dict]:
        rows = [p for p in PLANTS if p[5] == name]
        parts = {k: sum(p[4] for p in rows if p[1] == k) for k in ("wind", "solar", "ess", "thermal")}
        return len(rows), sum(p[4] for p in rows), parts

    html = []
    for name, x, y, est in (("제주시", 150, 30, "≈350"), ("서귀포시", 700, 468, "≈420")):
        n, cap, parts = region(name)
        html.append(
            f'<div class="pd-callout" style="left:{x / MAP_W * 100:.2f}%;top:{y / MAP_H * 100:.2f}%">'
            f'<div class="pd-callout__name">{name}<span>{n}곳</span></div>'
            f'<div class="pd-callout__row"><span>설비</span><b>{fmt(cap)} MW</b></div>'
            f'<div class="pd-callout__row"><span>추정 출력</span><b class="is-est">{est} MW</b></div>'
            f'{cap_bar(parts)}</div>'
        )
    return "".join(html)


# ----------------------------------------------------------------- HTML
def kpi(label: str, icon: str, chip_html: str, value: str, unit: str, sub: str, meta1: str, meta2: str, extra: str = "") -> str:
    return (
        f'<article class="pd-kpi {extra}">'
        f'<div class="pd-kpi__head"><span class="pd-kpi__label">{ico(icon)}{label}</span>{chip_html}</div>'
        f'<div class="pd-kpi__value"><span class="num">{value}</span><span class="pd-kpi__unit">{unit}</span></div>'
        f'<div class="pd-kpi__sub">{sub}</div>'
        f'<div class="pd-kpi__meta"><span>{meta1}</span><span>{meta2}</span></div>'
        "</article>"
    )


def build() -> str:
    wind_cap = sum(p[4] for p in PLANTS if p[1] == "wind")
    solar_cap = sum(p[4] for p in PLANTS if p[1] == "solar")
    ess_cap = sum(p[4] for p in PLANTS if p[1] == "ess")
    th_cap = sum(p[4] for p in PLANTS if p[1] == "thermal")
    total_cap = wind_cap + solar_cap + ess_cap + th_cap

    def reg(name: str) -> tuple[int, float, dict]:
        rows = [p for p in PLANTS if p[5] == name]
        return len(rows), sum(p[4] for p in rows), {k: sum(p[4] for p in rows if p[1] == k) for k in ("wind", "solar", "ess", "thermal")}

    jn, jc, jp = reg("제주시")
    sn, sc, sp = reg("서귀포시")
    allp = {"wind": wind_cap, "solar": solar_cap, "ess": ess_cap, "thermal": th_cap}

    kpis = "".join([
        kpi("현재 수요", "bolt", chip("meas", "실측"), "718", "MW",
            '<span class="pd-pill pd-pill--ok"><i></i>예비율 71.9% · 정상</span>',
            "출처 KPX 실시간 (API 서버)", "관측 2026-11-23 23:35 · 1분 경과"),
        kpi("제주 하루전 SMP", "tag", chip("meas", "실측"), "208.9", "원/kWh",
            '<span class="pd-delta">▲ 84.0원 (+67.2%)</span><span class="pd-kpi__hint">전 거래시간 대비</span>',
            "출처 EPSIS 실데이터", "거래시간 2026-11-23 24h"),
        kpi("재생에너지 비율", "leaf", chip("est", "추정 · 실측 아님"), "0.7", "%",
            '<span class="pd-kpi__hint">태양광+풍력 ÷ 섬 전체 수요</span>',
            "섬 전체 태양광+풍력 ML 추정치 (API 서버)", "기준 2026-11-23 23:35"),
        kpi("기상 현황", "thermo", chip("meas", "실측"), "16.8", "°C",
            '<span class="pd-kpi__subval">풍속 <b class="num">4.7</b> m/s · 비</span>',
            "출처 기상청 초단기실황 (API 서버)", "조회 2026-11-23 23:35"),
    ])

    insp_bars_d = hourly_bars_svg(410, 150)
    insp_bars_m = hourly_bars_svg(326, 140)

    region_table = (
        '<table class="pd-table pd-table--dense">'
        "<caption class=\"sr-only\">지역별 발전소 집계</caption>"
        "<thead><tr><th scope=\"col\">지역</th><th scope=\"col\" class=\"r\">발전소</th><th scope=\"col\" class=\"r\">설비 MW</th>"
        "<th scope=\"col\" class=\"r\">추정 출력 MW</th><th scope=\"col\" class=\"col-mix\">설비 구성</th></tr></thead><tbody>"
        f'<tr class="is-sel"><th scope="row">제주도 전체</th><td class="r num">27</td><td class="r num">{fmt(total_cap)}</td><td class="r num is-est">≈770</td><td class="col-mix">{cap_bar(allp)}</td></tr>'
        f'<tr><th scope="row" class="ind">제주시</th><td class="r num">{jn}</td><td class="r num">{fmt(jc)}</td><td class="r num is-est">≈350</td><td class="col-mix">{cap_bar(jp)}</td></tr>'
        f'<tr><th scope="row" class="ind">서귀포시</th><td class="r num">{sn}</td><td class="r num">{fmt(sc)}</td><td class="r num is-est">≈420</td><td class="col-mix">{cap_bar(sp)}</td></tr>'
        '<tr class="pd-table__group"><th scope="rowgroup" colspan="5">지도 밖 자원 (위치 없음)</th></tr>'
        '<tr class="is-muted"><th scope="row" class="ind">분산형 태양광 (합산)</th><td class="r num">1,620</td><td class="r num">557.3</td><td class="r">—</td><td class="pd-table__note col-mix">점 미표시</td></tr>'
        '<tr class="is-muted"><th scope="row" class="ind">HVDC 연계</th><td class="r num">3회선</td><td class="r num">900.0</td><td class="r">—</td><td class="pd-table__note col-mix">육지 연계</td></tr>'
        "</tbody></table>"
    )

    legend_items = "".join(
        f'<li><span class="pd-dot" style="--c:var(--s-{k})"></span>{TYPE_KO[k]} <b class="num">{n}곳</b> · <span class="num">{fmt(c)}</span> MW</li>'
        for k, n, c in (("wind", 16, wind_cap), ("solar", 4, solar_cap), ("ess", 4, ess_cap), ("thermal", 3, th_cap))
    )

    demand_gauge = gauge_svg(718, 0, 1200, S["demand"],
                             [(0, 600, S["supply"], 0.22), (600, 960, S["supply"], 0.40)], 1080,
                             "수요", "MW", [(1080, "1,080")])
    reserve_gauge = gauge_svg(71.9, 0, 79.1, S["reserve"],
                              [(0, 10, "#DC2626", 0.85), (10, 15, "#B45309", 0.85), (15, 79.1, S["reserve"], 0.5)], 10,
                              "예비율", "%", [(10, "10"), (15, "15")])

    donut = donut_svg([("thermal", 597.0, S["thermal"]), ("wind", 108.0, S["wind"]), ("ess", 51.0, S["ess"]), ("solar", 0.0, S["solar"])])

    wind_h, solar_h = hourly_profile()
    plant_rows = []
    rng = random.Random(SEED + 11)
    est = {"thermal": 609.3 / th_cap, "wind": 108.7 / wind_cap, "ess": 51.5 / ess_cap, "solar": 0.0}
    for name, typ, la, lo, cap, region in sorted(PLANTS, key=lambda p: -p[4])[:8]:
        g = cap * est[typ] * rng.uniform(0.95, 1.05)
        util = g / cap * 100 if cap else 0
        over = util > 100
        bar = min(util, 100)
        plant_rows.append(
            f'<tr><th scope="row">{name}</th><td><span class="pd-dot" style="--c:var(--s-{typ})"></span>{TYPE_KO[typ]}</td>'
            f'<td>{region}</td><td class="r num">{fmt(cap)}</td><td class="r num is-est">{fmt(g)}</td>'
            f'<td><span class="pd-prog{" is-over" if over else ""}"><span style="width:{bar:.0f}%"></span></span>'
            f'<span class="num">{util:.0f}%</span>{" <span class=\"pd-flag\">100% 초과 · 모형 한계</span>" if over else ""}</td>'
            f'<td><span class="pd-pill pd-pill--ok pd-pill--sm"><i></i>운영중</span></td></tr>'
        )
    plant_table = (
        '<table class="pd-table"><caption class="sr-only">발전소 등록부 상위 8곳</caption><thead><tr>'
        '<th scope="col">발전소</th><th scope="col">유형</th><th scope="col">지역</th><th scope="col" class="r">설비 MW</th>'
        '<th scope="col" class="r">추정 출력 MW</th><th scope="col">이용률 (추정)</th><th scope="col">상태</th></tr></thead><tbody>'
        + "".join(plant_rows) + '</tbody></table><p class="pd-card__foot">설비 큰 순 8곳 · 나머지 19곳은 표를 내려 보기 · 이용률 = 추정 출력 ÷ 설비</p>'
    )

    timeline_d = hourly_bars_svg(840, 230, big=True)
    timeline_m = hourly_bars_svg(326, 170)

    html = TEMPLATE
    repl = {
        "ICON_CSS": icon_css(),
        "KPIS": kpis,
        "MAP_SVG": map_svg(),
        "CALLOUTS": callout_html(),
        "LEGEND_ITEMS": legend_items,
        "INSP_BARS_D": insp_bars_d,
        "INSP_BARS_M": insp_bars_m,
        "REGION_TABLE": region_table,
        "TOTAL_CAP": fmt(total_cap),
        "SUPPLY_D": supply_chart_svg(1334, 380, False),
        "SUPPLY_T": supply_chart_svg(900, 340, False),
        "SUPPLY_M": supply_chart_svg(326, 280, True),
        "DEMAND_GAUGE": demand_gauge,
        "RESERVE_GAUGE": reserve_gauge,
        "SMP_D": smp_chart_svg(864, 340, False),
        "SMP_M": smp_chart_svg(326, 260, True),
        "DONUT": donut,
        "TIMELINE_D": timeline_d,
        "TIMELINE_M": timeline_m,
        "PLANT_TABLE": plant_table,
        "SCALE_W": f"{0.1076 * KX / MAP_W * 100:.2f}",
        "I_MAP": ico("map"), "I_CHART": ico("chart"), "I_GAUGE": ico("gauge"), "I_BOLT": ico("bolt"),
        "I_TAG": ico("tag"), "I_SPARK": ico("spark"), "I_LAYERS": ico("layers"), "I_CLOCK": ico("clock"),
        "I_REFRESH": ico("refresh"), "I_INFO": ico("info"), "I_DB": ico("db"), "I_PLUS": ico("plus"),
        "I_MINUS": ico("minus"), "I_CHEV": ico("chev"), "I_CHEV2": ico("chev2"), "I_PLUG": ico("plug"),
        "I_WIND": ico("wind"), "I_SUN": ico("sun"), "I_LIST": ico("list"), "I_ALERT": ico("alert"),
        "I_FLAME": ico("flame"), "I_BATTERY": ico("battery"),
        "C_MEAS": chip("meas", "실측"), "C_FC": chip("fc", "예측"), "C_EST": chip("est", "추정 · 실측 아님"),
        "C_SIM": chip("est", "시뮬레이션 · 실측 아님"), "C_EX": chip("ex", "예시 · 실제 값 아님"),
        "C_PATTERN": chip("est", "추정 패턴"),
    }
    for k, v in repl.items():
        html = html.replace("@@" + k + "@@", v)
    if "@@" in html:
        raise RuntimeError("unreplaced placeholder: " + html[html.index("@@"):html.index("@@") + 40])
    return html


TEMPLATE = (Path(__file__).resolve().parent / "mock_A_template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print("wrote", OUT, OUT.stat().st_size, "bytes")
