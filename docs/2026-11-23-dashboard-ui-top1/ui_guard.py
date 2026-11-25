"""app_v4.py UI 개편의 정적 계약 검사 (빌드 명세 9장 G-S1~G-S8).

사용:
    .venv/bin/python -I docs/2026-11-23-dashboard-ui-top1/ui_guard.py --app src/dashboard/app_v4.py  (저장소 루트에서)

검사 (모두 AST 기반, 앱을 실행하지 않는다)
- G-S1 main() 서브트리의 .markdown( 호출은 정확히 11개이고, 각각 의도한 카드다
        (회귀 테스트의 탐색 규칙을 그대로 재현해, 테스트가 '의도한 카드'를 잡는지 확인 = 의미 없는 통과 방지)
- G-S2 카드 f-string이 테스트 문맥의 허용 이름만 쓴다
- G-S3 @st.fragment로 꾸민 함수 안에 경보 부작용 호출이 없다 (add_alert, send_alert, send_critical_alert, toast)
- G-S4 모듈 최상위에 <style> markdown이 없고, main 밖 모듈 최상위 함수·클래스 안에 .markdown( 호출이 없다
        (T5는 모듈 전체를 너비 우선으로 훑으므로 그런 호출이 main() 안 카드보다 먼저 검사된다. 명세 5.1-7)
- G-S5 st.html로 넘기는 CSS 문자열 함수(_css_dashboard)의 본문에 '<'가 <style>/</style> 말고는 없다
- G-S6 Plotly 레이아웃에 title=None·xaxis_title=None·template= 인자가 없다
- G-S7 reserve_color·reserve_gauge·demand_gauge 할당이 main() 서브트리에 이름당 정확히 1개다
- G-S8 화면 계층에 이모지가 없다 (명세 G15). 범위: main, logic_guard가 표현 계층으로 보는 모듈 최상위 이름,
        st.set_page_config 인자, 모듈 docstring. EmailNotifier·SlackNotifier의 알림 본문은 화면 밖이라 제외한다.
종료 코드: 실패가 있으면 1.

2026-11-24 감사 반영 때 G-S4 확장과 G-S8을 더했다(원본: 세션 임시 폴더 tools/ui_guard.py).
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

# logic_guard.py의 표현 계층 판정과 같은 규칙(명세 5.1-5)
PRESENTATION_PREFIXES = ("create_", "render_", "_render", "_css", "_theme", "_chart", "_fmt", "_html")
PRESENTATION_NAMES = {"main", "SERIES", "BOUNDARY_COLOR", "ALERT_RED", "PD_ICONS", "_MapThemeSync"}
# 그림 문자 범위: 보조 그림 문자·기호, 기타 기호·딩뱃, 별·원, 시계·기술 기호, 이모지 변형 선택자
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐⭕⌚-⏿️]")


def is_presentation(name: str) -> bool:
    return name in PRESENTATION_NAMES or name.startswith(PRESENTATION_PREFIXES) or (name.isupper() and "CSS" in name)


def top_level_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return None

CARD_TITLES = {
    # 제목: (기대 클래스 조각, 허용 이름 집합 또는 None=자리표시자 금지)
    "현재 수요": ("pd-kpi", {"power_status", "reserve_class", "reserve_rate", "reserve_text",
                         "demand_observed_text", "demand_elapsed_text"}),
    "제주 하루전 SMP": ("pd-kpi", {"smp_data", "smp_change_class", "smp_change_html", "str"}),
    "재생에너지 비율": ("pd-kpi", {"power_status"}),
    "기상 현황": ("pd-kpi", {"weather", "str"}),
    "계통 주파수": ("pd-card-sim", {"power_status"}),
    "입찰 추천": ("pd-example", {"optimal_bid"}),
    "XAI 분석 요약": ("pd-example", None),
    "리스크 알림": ("pd-example", None),
    "모델 성능": ("pd-model", None),
    "일간 요약": ("pd-example", None),
    "데이터 출처": ("pd-footer", None),
}
SIDE_EFFECTS = {"add_alert", "send_alert", "send_critical_alert", "toast"}
BUILTINS = {"str", "len", "round", "abs", "min", "max", "int", "float"}


def texts_of(arg: ast.AST) -> list[str]:
    if isinstance(arg, ast.JoinedStr):
        return [str(p.value) for p in arg.values if isinstance(p, ast.Constant)]
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return [arg.value]
    return []


def markdown_calls(root: ast.AST) -> list[ast.Call]:
    return [n for n in ast.walk(root) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "markdown" and n.args]


def is_fragment(fn: ast.FunctionDef) -> bool:
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        if isinstance(target, ast.Attribute) and target.attr == "fragment":
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True)
    args = ap.parse_args()
    src = Path(args.app).read_text(encoding="utf-8")
    tree = ast.parse(src)
    main_fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    results: list[tuple[str, bool, str]] = []

    # G-S1: 테스트와 같은 탐색 순서(BFS)로 카드별 첫 일치 호출을 찾고, 의도한 카드인지 본다
    calls = markdown_calls(main_fn)
    results.append(("G-S1 main() .markdown( 호출 수 = 11", len(calls) == 11, f"{len(calls)}개"))
    module_calls = markdown_calls(tree)
    for title, (cls, allowed) in CARD_TITLES.items():
        pool = module_calls if title in ("모델 성능", "일간 요약") else calls  # T5는 모듈 전체 BFS
        hit = None
        for c in pool:
            arg = c.args[0]
            if title in ("모델 성능", "일간 요약") and isinstance(arg, ast.JoinedStr) and not all(
                    isinstance(v, ast.Constant) for v in arg.values):
                continue
            if title in ("XAI 분석 요약", "리스크 알림") and not isinstance(arg, ast.Constant):
                continue
            if any(title in t for t in texts_of(arg)):
                hit = c
                break
        ok = hit is not None and any(cls in t for t in texts_of(hit.args[0]))
        results.append((f"G-S1 '{title}' → 의도한 카드({cls})", ok, f"line {getattr(hit, 'lineno', '-')}"))
        # G-S2: 허용 이름
        if hit is not None:
            used = {n.id for n in ast.walk(hit.args[0]) if isinstance(n, ast.Name)}
            if allowed is None:
                ok2 = not used and (isinstance(hit.args[0], ast.Constant) or all(
                    isinstance(v, ast.Constant) for v in getattr(hit.args[0], "values", [])))
                results.append((f"G-S2 '{title}' 자리표시자 없음", ok2, ",".join(sorted(used)) or "-"))
            else:
                ok2 = used <= (allowed | BUILTINS)
                results.append((f"G-S2 '{title}' 허용 이름만", ok2, ",".join(sorted(used - allowed - BUILTINS)) or "-"))

    # G-S3: fragment 안 부작용 금지
    bad = []
    for fn in [n for n in ast.walk(main_fn) if isinstance(n, ast.FunctionDef) and is_fragment(n)]:
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in SIDE_EFFECTS:
                bad.append(f"{fn.name}:{n.lineno}:{n.func.attr}")
    frag_count = sum(1 for n in ast.walk(main_fn) if isinstance(n, ast.FunctionDef) and is_fragment(n))
    results.append(("G-S3 fragment 안 경보 부작용 0", not bad, (", ".join(bad) or "-") + f" (fragment {frag_count}개)"))

    # G-S4: 모듈 최상위 <style> markdown 금지 + main 밖 최상위 함수·클래스 안 .markdown( 금지
    top_style = [n.lineno for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "markdown"
                 and any("<style" in t for t in texts_of(n.value.args[0]) if n.value.args)]
    outside_main = [f"{n.name}:{c.lineno}" for n in tree.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name != "main"
                    for c in markdown_calls(n)]
    results.append(("G-S4 모듈 최상위 <style> markdown 0 · main 밖 함수 .markdown( 0",
                    not top_style and not outside_main,
                    f"style={top_style or '-'} outside_main={', '.join(outside_main) or '-'}"))

    # G-S5: CSS 본문에 '<' 금지 (_css_dashboard가 있으면 문자열 상수 검사)
    css_fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_css_dashboard"), None)
    if css_fn is None:
        results.append(("G-S5 _css_dashboard 존재", False, "없음"))
    else:
        consts = [n.value for n in ast.walk(css_fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        body = "".join(consts).replace("<style>", "").replace("</style>", "")
        results.append(("G-S5 CSS 본문에 '<' 없음", "<" not in body, f"{body.count('<')}개"))

    # G-S6: Plotly 금지 인자
    bad6 = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg in ("title", "xaxis_title", "yaxis_title") and isinstance(kw.value, ast.Constant) \
                        and kw.value.value is None:
                    bad6.append(f"{n.lineno}:{kw.arg}=None")
                if kw.arg == "template" and isinstance(n.func, ast.Attribute) and n.func.attr == "update_layout":
                    bad6.append(f"{n.lineno}:template")
    results.append(("G-S6 title=None·template 0", not bad6, ", ".join(bad6) or "-"))

    # G-S7: 게이지 할당 이름당 1개
    counts = {k: 0 for k in ("reserve_color", "reserve_gauge", "demand_gauge")}
    for n in ast.walk(main_fn):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) \
                and n.targets[0].id in counts:
            counts[n.targets[0].id] += 1
    results.append(("G-S7 게이지 할당 이름당 1개", all(v == 1 for v in counts.values()), str(counts)))

    # G-S8: 화면 계층 이모지 0 (알림 본문 클래스는 범위 밖)
    emoji_hits: dict[str, int] = {}
    doc = ast.get_docstring(tree, clean=False) or ""
    if EMOJI_RE.search(doc):
        emoji_hits["(module docstring)"] = len(EMOJI_RE.findall(doc))
    for node in tree.body:
        name = top_level_name(node)
        seg = ast.get_source_segment(src, node) or ""
        if name is not None and is_presentation(name):
            hits = len(EMOJI_RE.findall(seg))
            if hits:
                emoji_hits[name] = hits
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "set_page_config":
            hits = len(EMOJI_RE.findall(seg))
            if hits:
                emoji_hits["st.set_page_config"] = hits
    results.append(("G-S8 화면 계층 이모지 0", not emoji_hits,
                    ", ".join(f"{k}={v}" for k, v in emoji_hits.items()) or "-"))

    fails = 0
    for name, ok, detail in results:
        print(("PASS " if ok else "FAIL ") + name + f"  [{detail}]")
        fails += 0 if ok else 1
    print(f"\n{len(results) - fails}/{len(results)} PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
