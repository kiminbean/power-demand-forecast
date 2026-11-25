"""dev 8514 첫 로드 성능 기준선: 전송 바이트(호스트별), WebSocket 수신량, 준비 시각. 4G 근사 조건 1회 + 무제한 1회. 읽기 전용."""
import json, sys, time
from collections import defaultdict
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8514/"
READY_JS = """() => document.querySelectorAll('.js-plotly-plot').length >= 3 &&
   !document.querySelector('[data-testid=stStatusWidget]') && !!document.querySelector('[data-testid=stCustomComponentV1]')"""

def run(throttle):
    with sync_playwright() as p:
        b = p.chromium.launch()
        ctx = b.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
        pg = ctx.new_page(); cdp = ctx.new_cdp_session(pg)
        cdp.send("Network.enable")
        if throttle:  # 4G 근사: 9 Mbps down / 1.5 Mbps up / RTT 150 ms
            cdp.send("Network.emulateNetworkConditions", {"offline": False, "latency": 150,
                     "downloadThroughput": 9e6 / 8, "uploadThroughput": 1.5e6 / 8})
        reqs, by_host = {}, defaultdict(int)
        cdp.on("Network.requestWillBeSent", lambda e: reqs.__setitem__(e["requestId"], e["request"]["url"]))
        cdp.on("Network.loadingFinished", lambda e: by_host.__setitem__(urlparse(reqs.get(e["requestId"], "")).netloc or "other",
                                                                       by_host[urlparse(reqs.get(e["requestId"], "")).netloc or "other"] + e.get("encodedDataLength", 0)))
        ws = {"frames": 0, "bytes": 0}
        def on_ws(w):
            w.on("framereceived", lambda payload: (ws.__setitem__("frames", ws["frames"] + 1),
                                                   ws.__setitem__("bytes", ws["bytes"] + (len(payload) if payload else 0))))
        pg.on("websocket", on_ws)
        t0 = time.time(); pg.goto(URL, wait_until="domcontentloaded", timeout=120000)
        ready = None
        for _ in range(240):
            if pg.evaluate(READY_JS): ready = round(time.time() - t0, 1); break
            time.sleep(0.5)
        time.sleep(4)  # 지도 iframe 자원까지 마저 받기
        res = {"throttled_4g": throttle, "ready_s": ready, "requests": len(reqs),
               "transfer_kb_by_host": {h: round(v / 1024) for h, v in sorted(by_host.items(), key=lambda x: -x[1])},
               "transfer_kb_total": round(sum(by_host.values()) / 1024), "ws_frames": ws["frames"], "ws_kb": round(ws["bytes"] / 1024)}
        ctx.close(); b.close(); return res

json.dump([run(False), run(True)], sys.stdout, ensure_ascii=False, indent=1)
