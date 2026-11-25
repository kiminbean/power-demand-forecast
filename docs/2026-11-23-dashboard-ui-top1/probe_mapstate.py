"""(8차 동등성, 읽기 전용) MAP-STATE 다시 재기: 숨은 탭에서 지도 iframe이 다시 마운트되는 조건을 만들고 실시간 탭으로 돌아와 시점을 잰다.

빌더 f8_mapstate.py(= 7차 probe_mapzoom4 조작 + 캐시 넘김 62초 + iframe 표지)를 바탕으로, 다시 실행을 일으키는 자리를 바꿔 볼 수 있게 했다.
  trigger: plant  = 발전 현황 탭 '추정 출력 보기' 토글(수용 절차와 같음)
           smp    = SMP 예측 탭 '차트/표' 중 '표'
           type   = 발전 현황 탭 발전 유형 '풍력'
           inspsrc= 실시간 탭 인스펙터 '출처' 펼침을 먼저 열고(7차 실패 조건) plant
사용: probe_mapstate.py <light|dark> <width> <trials> <out.json> <trigger> [wait_s=62]
(F10에서 $T 탐침 묶음에 더함: 지도·CSS를 고치면 plant·inspsrc·smp·type 네 설정을 다시 돌린다. 원본 eval/round8/parity/r8_mapstate.py)
내 브라우저 세션의 위젯만 바꾼다. 서버·파일은 바꾸지 않는다.
"""
import json
import sys
import time
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8514/"
READY = '[data-testid="stApp"][data-test-script-state="notRunning"]'
SCHEME, W, N, OUT, TRIG = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
WAIT = float(sys.argv[6]) if len(sys.argv) > 6 else 62.0
H = 900 if W > 500 else 844
MAPSTATE = r"""() => { const m = window.map; if (!m) return null; const c = m.getCenter(); const s = m.getSize();
  const r = m.getContainer().getBoundingClientRect();
  return {lat: +c.lat.toFixed(3), lng: +c.lng.toFixed(3), zoom: m.getZoom(), size: [s.x, s.y], box: [Math.round(r.width), Math.round(r.height)],
          mark: window.__probeMark === 1}; }"""


def frame(pg):
    for f in pg.frames:
        if "streamlit_folium" in (f.url or ""):
            return f
    return None


def settle(pg, t=2.0):
    time.sleep(0.6)
    pg.wait_for_selector(READY, timeout=90000)
    time.sleep(t)


def tab(pg, i):
    pg.query_selector_all('[data-testid="stTabs"] button[role="tab"]')[i].click()
    time.sleep(2.0)


def ms(pg):
    fr = frame(pg)
    return fr.evaluate(MAPSTATE) if fr else None


def trigger(pg):
    if TRIG in ("plant", "inspsrc"):
        tab(pg, 2)
        pg.locator('.st-key-plant_est label').first.click()
    elif TRIG == "smp":
        tab(pg, 1)
        pg.locator('.st-key-smp_view button').filter(has_text="표").first.click()
    elif TRIG == "type":
        tab(pg, 2)
        pg.locator('.st-key-plant_type button').filter(has_text="풍력").first.click()
    settle(pg)


res = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for k in range(N):
        ctx = b.new_context(viewport={"width": W, "height": H}, color_scheme=SCHEME, locale="ko-KR")
        pg = ctx.new_page()
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)[:160]))
        pg.goto(URL, wait_until="domcontentloaded", timeout=90000)
        pg.wait_for_selector(READY, timeout=90000)
        time.sleep(5)
        t0 = time.time()
        if TRIG == "inspsrc":
            s = pg.locator('.pd-insp summary').filter(has_text="출처").first
            s.scroll_into_view_if_needed(); s.click(); time.sleep(0.8)
        before = ms(pg)
        fr = frame(pg)
        if fr:
            fr.evaluate("() => { window.__probeMark = 1; }")
        time.sleep(max(0.0, WAIT - (time.time() - t0)))
        trigger(pg)
        tab(pg, 0)
        time.sleep(1.5)
        after = ms(pg)
        ok = bool(after) and abs(after["lat"] - 33.375) <= 0.01 and abs(after["lng"] - 126.549) <= 0.01 and after["zoom"] == 10
        row = {"trial": k, "scheme": SCHEME, "width": W, "trigger": TRIG, "before": before, "after": after,
               "remounted": bool(after) and not after.get("mark"), "home_ok": ok, "pageerrors": errs[:3]}
        if k == 0 or not ok:
            ifr = pg.query_selector('.st-key-map_stage')
            if ifr:
                ifr.scroll_into_view_if_needed(); time.sleep(0.3)
                ifr.screenshot(path=OUT.replace(".json", f"_t{k}{'' if ok else '_FAIL'}.png"))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        res.append(row)
        ctx.close()
    b.close()
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("SUMMARY", SCHEME, W, TRIG, "home_ok", sum(r["home_ok"] for r in res), "/", len(res),
      "remounted", sum(r["remounted"] for r in res))
