"""렌더링된 Streamlit 페이지의 글자 대비(WCAG 2.x)를 실측한다.

사용 예:
    .venv/bin/python contrast.py --url http://127.0.0.1:8514/ --scheme dark --width 1440 --tabs

방법:
- 보이는 텍스트 요소(HTML 텍스트 노드의 부모, SVG <text>)마다 글자색과 실제 배경색을 구한다.
- 배경은 조상 요소를 거슬러 올라가며 반투명 배경을 알파 합성한다. 그라데이션은 첫 색 정지점으로 근사한다.
- 일반 글자 4.5:1, 큰 글자(24px 이상 또는 18.66px 이상 굵게) 3:1 기준으로 불합격을 모은다.
- 결과 JSON을 stdout에 쓴다. 탭이 있으면 탭마다 검사한다.

한계: 배경 이미지·지도 타일·캔버스(st.dataframe) 위 글자는 근사치이거나 제외된다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from playwright.sync_api import sync_playwright

READY = '[data-testid="stApp"][data-test-script-state="notRunning"]'

AUDIT_JS = r"""
() => {
  const parse = (c) => {
    if (!c) return null;
    const m = c.match(/rgba?\(([^)]+)\)/);
    if (!m) return null;
    const p = m[1].split(/[ ,\/]+/).filter(Boolean).map(Number);
    return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1};
  };
  const firstGradientColor = (bgImage) => {
    if (!bgImage || bgImage === 'none') return null;
    const m = bgImage.match(/rgba?\([^)]+\)/);
    return m ? parse(m[0]) : null;
  };
  const over = (top, bottom) => {
    const a = top.a + bottom.a * (1 - top.a);
    if (a === 0) return {r: 0, g: 0, b: 0, a: 0};
    return {
      r: (top.r * top.a + bottom.r * bottom.a * (1 - top.a)) / a,
      g: (top.g * top.a + bottom.g * bottom.a * (1 - top.a)) / a,
      b: (top.b * top.a + bottom.b * bottom.a * (1 - top.a)) / a,
      a,
    };
  };
  const effectiveBg = (el) => {
    const layers = [];
    let node = el;
    while (node && node.nodeType === 1) {
      const st = getComputedStyle(node);
      const grad = firstGradientColor(st.backgroundImage);
      const bg = parse(st.backgroundColor);
      if (grad && grad.a > 0) layers.push(grad);
      else if (bg && bg.a > 0) layers.push(bg);
      if (layers.length && layers[layers.length - 1].a >= 0.999) break;
      node = node.parentElement;
    }
    let acc = {r: 255, g: 255, b: 255, a: 1};
    for (let i = layers.length - 1; i >= 0; i--) acc = over(layers[i], acc);
    return acc;
  };
  const lum = (c) => {
    const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  };
  const ratio = (a, b) => { const l1 = lum(a), l2 = lum(b); return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05); };
  const visible = (el) => {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const st = getComputedStyle(el);
    if (st.visibility === 'hidden' || st.display === 'none' || Number(st.opacity) === 0) return false;
    // 비활성 탭 패널 안 요소 제외
    if (el.closest('[role="tabpanel"][hidden]')) return false;
    return true;
  };
  const results = [];
  const seen = new Set();
  const walker = document.createTreeWalker(document.querySelector('[data-testid="stApp"]') || document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const t = walker.currentNode;
    const text = t.textContent.trim();
    if (!text || text.length < 1) continue;
    const el = t.parentElement;
    if (!el || seen.has(el)) continue;
    if (el.closest('svg')) continue;
    seen.add(el);
    if (!visible(el)) continue;
    const st = getComputedStyle(el);
    let fg = parse(st.color);
    if (!fg) continue;
    const bg = effectiveBg(el);
    const op = Number(st.opacity);
    // 글자 자체의 알파와 요소 불투명도를 배경 위에 합성
    fg = over({...fg, a: fg.a * (isNaN(op) ? 1 : op)}, bg);
    const size = parseFloat(st.fontSize);
    const weight = parseInt(st.fontWeight, 10) || 400;
    const large = size >= 24 || (size >= 18.66 && weight >= 700);
    const r = ratio(fg, bg);
    results.push({text: text.slice(0, 40), ratio: Math.round(r * 100) / 100, size, weight, large,
      need: large ? 3 : 4.5, pass: r >= (large ? 3 : 4.5),
      testid: (el.closest('[data-testid]') || {}).getAttribute ? el.closest('[data-testid]').getAttribute('data-testid') : '',
      cls: (el.className && el.className.toString ? el.className.toString() : '').slice(0, 40)});
  }
  // SVG 텍스트 (Plotly): fill 색과 페이지 배경으로 근사
  for (const t of document.querySelectorAll('svg text')) {
    const text = (t.textContent || '').trim();
    if (!text || !visible(t)) continue;
    const st = getComputedStyle(t);
    const fill = parse(st.fill);
    if (!fill) continue;
    const bg = effectiveBg(t.closest('div') || document.body);
    const fg = over(fill, bg);
    const size = parseFloat(st.fontSize) || 12;
    const r = ratio(fg, bg);
    results.push({text: text.slice(0, 40), ratio: Math.round(r * 100) / 100, size, weight: 400, large: size >= 24,
      need: size >= 24 ? 3 : 4.5, pass: r >= (size >= 24 ? 3 : 4.5), testid: 'svg-text', cls: ''});
  }
  const fails = results.filter(r => !r.pass).sort((a, b) => a.ratio - b.ratio);
  return {checked: results.length, failCount: fails.length, worst: fails.slice(0, 40)};
}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True)
    ap.add_argument("--scheme", default="light", choices=["light", "dark", "no-preference"])
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--tabs", action="store_true")
    args = ap.parse_args()
    if not args.url.startswith(("http://127.0.0.1", "http://localhost")):
        print("refusing non-local URL", file=sys.stderr)
        return 2
    out: dict = {"url": args.url, "scheme": args.scheme, "width": args.width, "pages": []}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": args.width, "height": 1000}, color_scheme=args.scheme)
        page = ctx.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector(READY, timeout=90000)
        time.sleep(4)
        tabs = page.query_selector_all('[data-testid="stTabs"] button[role="tab"]') if args.tabs else []
        if not tabs:
            out["pages"].append({"tab": "main", **page.evaluate(AUDIT_JS)})
        for i in range(len(tabs)):
            tabs = page.query_selector_all('[data-testid="stTabs"] button[role="tab"]')
            name = (tabs[i].inner_text() or f"tab{i}").strip()
            tabs[i].click()
            time.sleep(1.5)
            out["pages"].append({"tab": name, **page.evaluate(AUDIT_JS)})
        browser.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
