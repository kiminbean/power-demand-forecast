"""390px 터치 에뮬레이션에서 메인 Plotly 차트와 folium 지도 위 세로 스와이프가 페이지를 스크롤하는지 측정. 읽기 전용."""
import json, sys, time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8514/"
SCROLL_JS = """() => { const c = [document.scrollingElement, document.querySelector('[data-testid=stMain]'),
  document.querySelector('[data-testid=stAppViewContainer]'), document.querySelector('.stApp')];
  return c.map(e => e ? Math.round(e.scrollTop) : null); }"""

def swipe(cdp, x, y, dy=-220, steps=8):
    cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]})
    for i in range(1, steps + 1):
        cdp.send("Input.dispatchTouchEvent", {"type": "touchMove", "touchPoints": [{"x": x, "y": y + dy * i / steps}]})
        time.sleep(0.03)
    cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
    time.sleep(1.2)

res = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True, color_scheme="dark")
    pg = ctx.new_page(); pg.goto(URL, wait_until="networkidle", timeout=90000); time.sleep(6)
    cdp = ctx.new_cdp_session(pg)
    for name, sel in [("main_plotly_chart", ".js-plotly-plot"), ("folium_map", "iframe[title*='folium'], [data-testid=stCustomComponentV1]")]:
        el = pg.query_selector(sel)
        if not el:
            res[name] = "not found"; continue
        el.scroll_into_view_if_needed(); pg.evaluate("(e)=>e.scrollIntoView({block:'center'})", el); time.sleep(1.5)
        bb = el.bounding_box()
        before = pg.evaluate(SCROLL_JS)
        xr0 = pg.evaluate("(g)=> g._fullLayout ? JSON.stringify(g._fullLayout.xaxis.range) : null", el) if name.startswith("main") else None
        swipe(cdp, bb["x"] + bb["width"] / 2, bb["y"] + bb["height"] * 0.7)
        after = pg.evaluate(SCROLL_JS)
        xr1 = pg.evaluate("(g)=> g._fullLayout ? JSON.stringify(g._fullLayout.xaxis.range) : null", el) if name.startswith("main") else None
        # control: same swipe on a non-chart area (KPI row / page header)
        res[name] = {"box": {k: round(v) for k, v in bb.items()}, "scroll_before": before, "scroll_after": after,
                     "xrange_changed": (xr0 != xr1) if xr0 else None}
    # control swipe on plain text area near top
    pg.evaluate("()=>{const m=document.querySelector('[data-testid=stMain]'); if(m) m.scrollTop=0; window.scrollTo(0,0);}"); time.sleep(1)
    before = pg.evaluate(SCROLL_JS); swipe(cdp, 195, 300); after = pg.evaluate(SCROLL_JS)
    res["control_text_area"] = {"scroll_before": before, "scroll_after": after}
    ctx.close(); b.close()
json.dump(res, sys.stdout, ensure_ascii=False, indent=1)
