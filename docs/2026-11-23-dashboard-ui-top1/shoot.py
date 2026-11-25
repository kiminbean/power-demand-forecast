"""Streamlit 대시보드 스크린샷·레이아웃 검사 도구 (headless Chromium).

사용 예:
    .venv/bin/python shoot.py --url http://127.0.0.1:8514/ --out /path/out --label after \
        --widths 1440,390 --scheme dark --tabs

하는 일:
- 각 폭마다 새 브라우저 컨텍스트(색 구성표 에뮬레이션)로 페이지를 연다.
- Streamlit 스크립트 실행이 끝날 때까지 기다린 뒤 차트·지도 렌더링 여유를 둔다.
- 가로 넘침(문서·주요 컨테이너)과 뷰포트 밖으로 삐져나온 요소를 찾는다.
- 화면 텍스트에서 undefined/NaN/Traceback 같은 깨진 값 표시를 찾는다.
- --tabs면 각 탭을 눌러 전체 높이 스크린샷을 저장한다.
- 콘솔 오류와 페이지 예외를 모은다.
- 결과 JSON을 stdout과 <out>/summary_<label>.json에 쓴다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

READY_SELECTOR = '[data-testid="stApp"][data-test-script-state="notRunning"]'
DEFAULT_HEIGHT = 900
MAX_SHOT_HEIGHT = 9000
RENDER_SETTLE_SEC = 4.0
TAB_SETTLE_SEC = 2.0
BROKEN_TEXT_PATTERNS = [
    r"\bundefined\b",
    r"\bNaN\b",
    r"\bnan\b",
    r"Traceback",
    r"\bNone\b",
    r"Error:",
    r"KeyError",
    r"\[object Object\]",
]

OVERFLOW_JS = """
() => {
  const vw = window.innerWidth;
  const doc = document.documentElement;
  const containers = ['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]', '.main', 'section.stMain']
    .map(sel => document.querySelector(sel)).filter(Boolean);
  const containerOverflow = containers.map(el => ({
    sel: el.getAttribute('data-testid') || el.className.toString().slice(0, 40),
    scrollWidth: el.scrollWidth, clientWidth: el.clientWidth,
    overflow: el.scrollWidth - el.clientWidth,
  }));
  const offenders = [];
  const all = document.querySelectorAll('body *');
  for (const el of all) {
    // Plotly 글자 측정용 숨은 SVG(#js-plotly-tester, left -10000px)는 앱 내용이 아니고 스크롤도 만들지 않는다(1차 평가)
    if (el.closest('#js-plotly-tester')) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none') continue;
    if (r.right > vw + 1 || r.left < -1) {
      // 스크롤 가능한 조상 안에 있는 요소(예: 가로 스크롤 탭, 표)는 넘침으로 치지 않는다.
      let p = el.parentElement, clipped = false;
      while (p && p !== document.body) {
        const ps = getComputedStyle(p);
        if (['auto', 'scroll', 'hidden', 'clip'].includes(ps.overflowX)) {
          const pr = p.getBoundingClientRect();
          if (pr.right <= vw + 1 && pr.left >= -1) { clipped = true; break; }
        }
        p = p.parentElement;
      }
      if (clipped) continue;
      offenders.push({
        tag: el.tagName.toLowerCase(),
        testid: el.getAttribute('data-testid') || '',
        cls: (el.className && el.className.toString ? el.className.toString() : '').slice(0, 60),
        left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width),
        text: (el.innerText || '').trim().slice(0, 50),
      });
    }
  }
  return {
    viewport: vw,
    docScrollWidth: doc.scrollWidth,
    docOverflow: doc.scrollWidth - doc.clientWidth,
    containerOverflow,
    offenders: offenders.slice(0, 25),
    offenderCount: offenders.length,
  };
}
"""

CONTENT_HEIGHT_JS = """
() => {
  const cands = [document.scrollingElement,
    document.querySelector('[data-testid="stMain"]'),
    document.querySelector('[data-testid="stAppViewContainer"]'),
    document.querySelector('.main')].filter(Boolean);
  return Math.max(...cands.map(el => el.scrollHeight));
}
"""


def wait_ready(page: Page, timeout_ms: int = 90000, static: bool = False) -> None:
    """Streamlit 첫 실행이 끝날 때까지 기다린다(정적 HTML이면 로드만 기다린다)."""
    if static:
        page.wait_for_load_state("load", timeout=timeout_ms)
        time.sleep(1.5)
        return
    page.wait_for_selector('[data-testid="stApp"]', timeout=timeout_ms)
    deadline = time.time() + timeout_ms / 1000
    # 스크립트가 한 번이라도 running 이었다가 notRunning 으로 돌아오는 것을 확인한다.
    while time.time() < deadline:
        if page.query_selector(READY_SELECTOR) and page.query_selector('[data-testid="stMainBlockContainer"], .block-container'):
            break
        time.sleep(0.5)
    time.sleep(RENDER_SETTLE_SEC)


def broken_text(page: Page) -> list[str]:
    """화면에 보이는 텍스트에서 깨진 값 표시를 찾는다."""
    try:
        text = page.inner_text('[data-testid="stApp"]')
        # Plotly 제목·축 등 SVG <text>는 innerText에 안 잡히므로 따로 모은다.
        text += "\n" + page.evaluate(
            "() => [...document.querySelectorAll('svg text')].map(t => t.textContent).join(' | ')"
        )
    except Exception:  # noqa: BLE001 - 진단 도구라 실패해도 계속한다
        return ["<inner_text failed>"]
    hits = []
    for pat in BROKEN_TEXT_PATTERNS:
        for m in re.finditer(pat, text):
            start = max(0, m.start() - 30)
            hits.append(f"{pat} :: ...{text[start:m.end() + 30]!r}...")
    return hits[:20]


def shoot_full(page: Page, path: Path, width: int) -> int:
    """콘텐츠 높이만큼 뷰포트를 늘려 전체 화면을 찍는다."""
    height = int(page.evaluate(CONTENT_HEIGHT_JS) or DEFAULT_HEIGHT)
    height = max(DEFAULT_HEIGHT, min(height + 40, MAX_SHOT_HEIGHT))
    page.set_viewport_size({"width": width, "height": height})
    time.sleep(1.2)
    page.screenshot(path=str(path), full_page=True)
    page.set_viewport_size({"width": width, "height": DEFAULT_HEIGHT})
    return height


def run(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    widths = [int(w) for w in args.widths.split(",") if w.strip()]
    result: dict = {"url": args.url, "label": args.label, "scheme": args.scheme, "runs": []}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for width in widths:
            ctx = browser.new_context(
                viewport={"width": width, "height": DEFAULT_HEIGHT},
                color_scheme=args.scheme,
                device_scale_factor=args.scale,
                locale="ko-KR",
            )
            page = ctx.new_page()
            console_errors: list[str] = []
            page.on("console", lambda m: console_errors.append(m.text[:200]) if m.type == "error" else None)
            page.on("pageerror", lambda e: console_errors.append("pageerror: " + str(e)[:200]))
            run_info: dict = {"width": width, "shots": []}
            try:
                page.goto(args.url, wait_until="domcontentloaded", timeout=90000)
                wait_ready(page, static=args.static)
                run_info["overflow"] = page.evaluate(OVERFLOW_JS)
                run_info["brokenText"] = broken_text(page)
                tabs = page.query_selector_all('[data-testid="stTabs"] button[role="tab"]') if args.tabs else []
                if not tabs:
                    shot = out / f"{args.label}_{args.scheme}_{width}_main.png"
                    h = shoot_full(page, shot, width)
                    run_info["shots"].append({"file": str(shot), "height": h})
                for i in range(len(tabs)):
                    tabs = page.query_selector_all('[data-testid="stTabs"] button[role="tab"]')
                    name = (tabs[i].inner_text() or f"tab{i}").strip().replace(" ", "")
                    tabs[i].click()
                    time.sleep(TAB_SETTLE_SEC)
                    ov = page.evaluate(OVERFLOW_JS)
                    shot = out / f"{args.label}_{args.scheme}_{width}_tab{i}.png"
                    h = shoot_full(page, shot, width)
                    # 명세 G4: 탭마다 깨진 글자를 본다(숨은 탭 패널의 SVG 글자도 잡히므로 탭별 결과를 따로 남긴다)
                    run_info["shots"].append({
                        "file": str(shot), "tab": name, "height": h,
                        "docOverflow": ov["docOverflow"], "offenderCount": ov["offenderCount"],
                        "offenders": ov["offenders"][:8],
                        "brokenText": broken_text(page),
                    })
                run_info["brokenTextAfterTabs"] = broken_text(page)
            except Exception as exc:  # noqa: BLE001
                run_info["error"] = f"{type(exc).__name__}: {exc}"[:400]
            run_info["consoleErrors"] = console_errors[:20]
            result["runs"].append(run_info)
            ctx.close()
        browser.close()
    summary = out / f"summary_{args.label}_{args.scheme}.json"
    summary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="shot")
    ap.add_argument("--widths", default="1440")
    ap.add_argument("--scheme", default="light", choices=["light", "dark", "no-preference"])
    ap.add_argument("--tabs", action="store_true", help="각 탭을 눌러 전체 스크린샷")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--static", action="store_true", help="Streamlit이 아닌 정적 HTML(목업)일 때")
    args = ap.parse_args()
    if not args.url.startswith(("http://127.0.0.1", "http://localhost", "file://")):
        print("refusing non-local URL", file=sys.stderr)
        return 2
    res = run(args)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
