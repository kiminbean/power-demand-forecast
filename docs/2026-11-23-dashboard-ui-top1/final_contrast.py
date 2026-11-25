"""최종 빌드 명세(2026-11-23-dashboard-ui-top1-design.md)의 토큰 대비를 계산한다.

WCAG 2.x 상대 휘도·대비. 반투명 색은 실제로 깔리는 면 위에 sRGB 알파 합성한 뒤 잰다.
빗금 칩은 줄무늬(가장 진한 지점)와 바탕(가장 옅은 지점) 둘 다 재고 낮은 값을 쓴다.

실행: .venv/bin/python -I docs/2026-11-23-dashboard-ui-top1/final_contrast.py [--json out.json]  (저장소 루트에서)
종료 코드: 기준 미달이 하나라도 있으면 1.
"""
from __future__ import annotations

import json
import sys

AA_TEXT, AA_LARGE, UI = 4.5, 3.0, 3.0


def hex_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def over(top: tuple, alpha: float, bottom: tuple) -> tuple:
    return tuple(t * alpha + b * (1 - alpha) for t, b in zip(top, bottom))


def lum(rgb: tuple) -> float:
    def ch(v: float) -> float:
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (ch(x) for x in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a: tuple, b: tuple) -> float:
    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def c(x) -> tuple:
    if isinstance(x, tuple):
        return x
    return hex_rgb(x)


def to_hex(rgb: tuple) -> str:
    return "#" + "".join(f"{round(v):02X}" for v in rgb)


# ---- 표면 ------------------------------------------------------------------
SURF = {
    "light": {"canvas": "#F6F8FB", "card": "#FFFFFF", "well": "#F1F5F9", "input": "#EEF2F7"},
    "dark": {"canvas": "#0B1220", "card": "#121B31", "well": "#1A2439", "input": "#1A2439"},
}

# ---- 글자·UI 토큰 -----------------------------------------------------------
TEXT = {
    "text-1": ("#0F172A", "#F1F5F9", AA_TEXT),
    "text-2": ("#334155", "#CBD5E1", AA_TEXT),
    "text-3": ("#56657A", "#94A3B8", AA_TEXT),
    "link": ("#1D4ED8", "#60A5FA", AA_TEXT),
    "primary (활성 탭 글자)": ("#2563EB", "#60A5FA", AA_TEXT),
    "focus (2px 링)": ("#2563EB", "#60A5FA", UI),
    "border-strong (컨트롤·점선 테두리)": ("#7B8798", "#6B7A93", UI),
}

rows_text = []
fails = []
for name, (lv, dv, req) in TEXT.items():
    for theme, val in (("light", lv), ("dark", dv)):
        vals = {s: ratio(c(val), c(bg)) for s, bg in SURF[theme].items()}
        mn = min(vals.values())
        rows_text.append((name, theme, val, vals, mn, req))
        if mn < req:
            fails.append(f"text {name} {theme} {mn:.2f}")

# ---- 상태 pill (글자 on 칩 바탕, 바탕은 카드 위 합성) ------------------------
STATUS = {
    "정상 status-online": (("#166534", "#DCFCE7", 1.0), ("#4ADE80", "#4ADE80", 0.14)),
    "관심 status-warning": (("#92400E", "#FEF3C7", 1.0), ("#FBBF24", "#FBBF24", 0.14)),
    "주의 status-danger": (("#9A3412", "#FFEDD5", 1.0), ("#FB923C", "#FB923C", 0.14)),
    "위험 status-critical": (("#B91C1C", "#FEE2E2", 1.0), ("#F87171", "#F87171", 0.16)),
    "정보 info": (("#1D4ED8", "#DBEAFE", 1.0), ("#93C5FD", "#60A5FA", 0.14)),
}
rows_status = []
for name, (lt, dk) in STATUS.items():
    for theme, (fg, bg, a) in (("light", lt), ("dark", dk)):
        card = c(SURF[theme]["card"])
        bg_c = over(c(bg), a, card)
        on_chip = ratio(c(fg), bg_c)
        on_card = ratio(c(fg), card)
        on_canvas = ratio(c(fg), c(SURF[theme]["canvas"]))
        mn = min(on_chip, on_card, on_canvas)
        rows_status.append((name, theme, fg, to_hex(bg_c), on_chip, on_card, on_canvas))
        if mn < AA_TEXT:
            fails.append(f"status {name} {theme} {mn:.2f}")

# ---- 출처 칩 ----------------------------------------------------------------
# (글자, 바탕, 바탕 알파, 줄무늬색, 줄무늬 알파 or None)
CHIPS = {
    "실측·확정": (("#334155", "#FFFFFF", 1.0, None, None), ("#CBD5E1", "#121B31", 1.0, None, None)),
    "예측": (("#1D4ED8", "#EFF6FF", 1.0, None, None), ("#93C5FD", "#60A5FA", 0.12, None, None)),
    "추정·시뮬레이션 (빗금)": (("#334155", "#F1F5F9", 1.0, "#64748B", 0.22),
                          ("#CBD5E1", "#1A2439", 1.0, "#94A3B8", 0.22)),
    "예시 (리본·칩)": (("#92400E", "#FFFBEB", 1.0, None, None), ("#FCD34D", "#F59E0B", 0.12, None, None)),
}
rows_chip = []
for name, (lt, dk) in CHIPS.items():
    for theme, (fg, bg, a, stripe, sa) in (("light", lt), ("dark", dk)):
        card = c(SURF[theme]["card"])
        base = over(c(bg), a, card)
        worst = ratio(c(fg), base)
        stripe_hex = None
        if stripe:
            st = over(c(stripe), sa, base)
            stripe_hex = to_hex(st)
            worst = min(worst, ratio(c(fg), st))
        rows_chip.append((name, theme, fg, to_hex(base), stripe_hex, worst))
        if worst < AA_TEXT:
            fails.append(f"chip {name} {theme} {worst:.2f}")

# ---- 흰 글자 pill (테마 공용) ------------------------------------------------
PILLS = {"현재 시각 pill": "#475569", "관심 경계 pill": "#B45309", "위험 경계 pill": "#B91C1C",
         "로고 바탕(primary L)": "#2563EB"}
rows_pill = []
for name, bg in PILLS.items():
    r = ratio(c("#FFFFFF"), c(bg))
    rows_pill.append((name, bg, r))
    if r < AA_TEXT:
        fails.append(f"pill {name} {r:.2f}")

# ---- 계열 색 (테마 공용, 4개 바탕 3:1) ---------------------------------------
SERIES = {
    "demand 수요": "#3B82F6", "supply 공급능력": "#64748B", "reserve 예비력": "#16A34A",
    "solar 태양광": "#B7791F", "wind 풍력": "#0891B2", "ess ESS": "#8B5CF6",
    "thermal 화력": "#8D6E63", "hvdc HVDC": "#7C8DB0", "smp SMP": "#C026D3", "other 기타": "#4D7C0F",
    "boundary 관심 경계선": "#B45309", "alert 위험·임계선": "#DC2626",
}
rows_series = []
for name, col in SERIES.items():
    vals = {
        "L card": ratio(c(col), c("#FFFFFF")), "L canvas": ratio(c(col), c("#F6F8FB")),
        "D card": ratio(c(col), c("#121B31")), "D canvas": ratio(c(col), c("#0B1220")),
    }
    mn = min(vals.values())
    rows_series.append((name, col, vals, mn))
    if mn < UI and not name.startswith("now"):
        fails.append(f"series {name} {mn:.2f}")

# 현재선 #475569는 다크 바탕에서 3:1 미달 가능 → 다크에서는 pill(흰 글자)·선 폭으로 식별. 별도 표기.

# ---- 예비력 빗금 위 점선 (선이 빗금 면 경계에 놓임) -----------------------------
HATCH = {"fill": ("#16A34A", 0.04), "stripe": ("#16A34A", 0.08)}
rows_hatch = []
for theme in ("light", "dark"):
    card = c(SURF[theme]["card"])
    fill = over(c(HATCH["fill"][0]), HATCH["fill"][1], card)
    stripe = over(c(HATCH["stripe"][0]), HATCH["stripe"][1], fill)
    for name, col in (("수요 #3B82F6", "#3B82F6"), ("공급능력 #64748B", "#64748B")):
        r = min(ratio(c(col), fill), ratio(c(col), stripe))
        rows_hatch.append((theme, name, to_hex(fill), to_hex(stripe), r))
        if r < UI:
            fails.append(f"hatch {theme} {name} {r:.2f}")

# ---- Plotly 글자 (Streamlit 1.52.1 테마 함수 bF 실값) ---------------------------
PLOTLY = {
    "기본 눈금·주석 getGray70 (L #808495 / D #E6EAF1)": ("#808495", "#E6EAF1"),
    "현재선 자리표시자 '#000036' = getGray70 (그래픽 3:1 기준)": ("#808495", "#E6EAF1"),
    "자리표시자 '#000037' getGray90 (L #262730 / D #FAFAFA)": ("#262730", "#FAFAFA"),
}
rows_plotly = []
for name, (lv, dv) in PLOTLY.items():
    l_ = min(ratio(c(lv), c("#FFFFFF")), ratio(c(lv), c("#F6F8FB")))
    d_ = min(ratio(c(dv), c("#121B31")), ratio(c(dv), c("#0B1220")))
    rows_plotly.append((name, l_, d_))

# ---- 지도 기호 vs 타일 (근사 타일색, 구현 때 재측정) -----------------------------
TILES = {"positron 육지 ≈#FAFAF8": "#FAFAF8", "positron 바다 ≈#D4DADC": "#D4DADC",
         "dark_matter 육지 ≈#262626": "#262626", "dark_matter 바다 ≈#0E0E0E": "#0E0E0E"}
rows_map = []
for name in ("wind 풍력", "solar 태양광", "ess ESS", "thermal 화력"):
    col = SERIES[name]
    vals = {t: ratio(c(col), c(v)) for t, v in TILES.items()}
    rows_map.append((name, col, vals))
outline = {"L 외곽선 var(--text-color) #0F172A": ("#0F172A", ("#FAFAF8", "#D4DADC")),
           "D 외곽선 var(--text-color) #F1F5F9": ("#F1F5F9", ("#262626", "#0E0E0E"))}
rows_outline = []
for name, (col, tiles) in outline.items():
    rows_outline.append((name, [ratio(c(col), c(t)) for t in tiles]))


def fmt(v: float) -> str:
    return f"{v:.2f}"


def main() -> int:
    out = []
    out.append("### A. 글자·UI 토큰 (바탕 4종: canvas · card · well · input)")
    out.append("| 토큰 | 테마 | 값 | canvas | card | well | input | 최저 | 기준 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for name, theme, val, vals, mn, req in rows_text:
        out.append(f"| {name} | {theme} | `{val}` | {fmt(vals['canvas'])} | {fmt(vals['card'])} | "
                   f"{fmt(vals['well'])} | {fmt(vals['input'])} | **{fmt(mn)}** | {req} |")
    out.append("")
    out.append("### B. 상태 pill (글자 / 카드 위 합성 바탕)")
    out.append("| 상태 · 클래스 | 테마 | 글자 | 바탕(합성) | 글자/바탕 | 글자/card | 글자/canvas |")
    out.append("|---|---|---|---|---|---|---|")
    for name, theme, fg, bg, a, b, cc in rows_status:
        out.append(f"| {name} | {theme} | `{fg}` | `{bg}` | {fmt(a)} | {fmt(b)} | {fmt(cc)} |")
    out.append("")
    out.append("### C. 출처 칩 (빗금은 줄무늬 지점까지 포함한 최저값)")
    out.append("| 칩 | 테마 | 글자 | 바탕(합성) | 줄무늬(합성) | 최저 대비 |")
    out.append("|---|---|---|---|---|---|")
    for name, theme, fg, base, stripe, worst in rows_chip:
        out.append(f"| {name} | {theme} | `{fg}` | `{base}` | {('`' + stripe + '`') if stripe else '—'} | {fmt(worst)} |")
    out.append("")
    out.append("### D. 흰 글자 pill (두 테마 공용)")
    out.append("| pill | 바탕 | 흰 글자 대비 |")
    out.append("|---|---|---|")
    for name, bg, r in rows_pill:
        out.append(f"| {name} | `{bg}` | {fmt(r)} |")
    out.append("")
    out.append("### E. 계열 색 (두 테마 공용 단일값, 그래픽 3:1)")
    out.append("| 계열 | 색 | L card #FFFFFF | L canvas #F6F8FB | D card #121B31 | D canvas #0B1220 | 최저 |")
    out.append("|---|---|---|---|---|---|---|")
    for name, col, vals, mn in rows_series:
        out.append(f"| {name} | `{col}` | {fmt(vals['L card'])} | {fmt(vals['L canvas'])} | "
                   f"{fmt(vals['D card'])} | {fmt(vals['D canvas'])} | **{fmt(mn)}** |")
    out.append("")
    out.append("### F. 예비력 빗금 면 위 점선 (fill 4% + 줄무늬 8%)")
    out.append("| 테마 | 선 | 면(합성) | 줄무늬(합성) | 최저 대비 |")
    out.append("|---|---|---|---|---|")
    for theme, name, fill, stripe, r in rows_hatch:
        out.append(f"| {theme} | {name} | `{fill}` | `{stripe}` | {fmt(r)} |")
    out.append("")
    out.append("### G. Plotly 글자색 (Streamlit 1.52.1 번들 bF·SF 실값, 카드·캔버스 중 낮은 값)")
    out.append("| 글자 | 라이트 | 다크 |")
    out.append("|---|---|---|")
    for name, l_, d_ in rows_plotly:
        out.append(f"| {name} | {fmt(l_)} | {fmt(d_)} |")
    out.append("")
    out.append("### H. 지도 기호 채움 vs 타일(근사), 외곽선")
    out.append("| 기호 | 색 | " + " | ".join(TILES.keys()) + " |")
    out.append("|---|---|" + "---|" * len(TILES))
    for name, col, vals in rows_map:
        out.append(f"| {name} | `{col}` | " + " | ".join(fmt(v) for v in vals.values()) + " |")
    for name, vals in rows_outline:
        out.append(f"| {name} | — | " + " | ".join(fmt(v) for v in vals) + " | — | — |")
    out.append("")
    out.append(f"미달: {len(fails)}건" + (": " + "; ".join(fails) if fails else ""))
    print("\n".join(out))
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fails": fails}, f, ensure_ascii=False, indent=2)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
