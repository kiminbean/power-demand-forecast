"""빌드 명세 문서(docs/2026-11-23-dashboard-ui-top1-design.md)의 코드 블록만으로 main()·create_gauge_chart를
조립해, 수정 없는 tests/test_v4_*.py를 돌린다. 문서에 적힌 코드가 그대로 테스트를 통과하는지 확인한다.

실행: .venv/bin/python -I docs/2026-11-23-dashboard-ui-top1/doc_dryrun.py  (저장소 루트에서)
"""
from __future__ import annotations

import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WT = Path(__file__).resolve().parents[2]  # docs/2026-11-23-dashboard-ui-top1/ 두 단계 위 = 저장소(워크트리) 루트
DOC = WT / "docs/2026-11-23-dashboard-ui-top1-design.md"
HERE = Path(__file__).resolve().parent
# 워크트리 docs 폴더에 합성 앱 사본이 쌓이지 않게 시스템 임시 폴더에 만든다(2026-11-24 감사 반영 때 바꿈)
OUT = Path(tempfile.gettempdir()) / "pd_doc_dryrun_out"
PY = sys.executable


def blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.S)


def section(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def main() -> int:
    doc = DOC.read_text(encoding="utf-8")
    skel = blocks(section(doc, "### 5.3 `main()` 골격", "골격 규칙 [계약]"))[0]
    kpi = blocks(section(doc, "### 5.5 KPI 타일 4장", "상태 변화"))[0]
    s56 = blocks(section(doc, "### 5.6 테스트 대상 카드 6장", "예시 강등 규칙"))
    freq, bid, model, xai, risk, daily = s56
    footer = blocks(section(doc, "### 5.14 푸터", "- 금지어"))[0]
    gauge = blocks(section(doc, "### 6.4 게이지", "- 바꾸면 안 되는 것(T7)"))[0]

    lines = skel.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == "with col1:":
            # col1~col4 자리표시자 묶음을 5.5절 블록으로 바꾼다
            j = i
            while not (lines[j].strip().startswith("...") and "기상 현황" in lines[j]):
                j += 1
            out.append(kpi)
            i = j + 1
            continue
        if ln.strip().startswith("...") and "계통 주파수" in ln:
            out.append(freq); i += 1; continue
        if ln.strip().startswith("optimal_bid = ") and i + 1 < len(lines) and "입찰 추천" in lines[i + 1]:
            out.append(bid); i += 2; continue
        if ln.strip().startswith("...") and "모델 성능" in ln:
            out.append(model); i += 1; continue
        if ln.strip().startswith("...") and "XAI 분석 요약" in ln:
            out.append(xai); i += 1; continue
        if ln.strip().startswith("...") and "리스크 알림" in ln:
            out.append(risk); i += 1; continue
        if ln.strip().startswith("...") and "일간 요약" in ln:
            out.append(daily); i += 1; continue
        if ln.strip().startswith("...") and "푸터" in ln:
            out.append(footer); i += 1; continue
        out.append(ln)
        i += 1
    main_src = "".join(out)
    leftovers = [l for l in main_src.splitlines() if l.strip().startswith("...")]
    assert not leftovers, leftovers
    ast.parse(main_src)

    src = (WT / "src/dashboard/app_v4.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    src_lines = src.splitlines(keepends=True)
    edits = []
    for n in tree.body:
        start = min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
        if isinstance(n, ast.FunctionDef) and n.name == "main":
            edits.append((start, n.end_lineno, main_src))
        elif isinstance(n, ast.FunctionDef) and n.name == "create_gauge_chart":
            edits.append((start, n.end_lineno, gauge))
        elif (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "markdown"
              and n.value.args and isinstance(n.value.args[0], ast.Constant)
              and "<style>" in str(n.value.args[0].value)):
            edits.append((start, n.end_lineno, "# (명세) CSS는 main() 첫 줄 st.html로\n"))
    for a, b, text in sorted(edits, reverse=True):
        src_lines[a - 1:b] = [text if text.endswith("\n") else text + "\n"]
    out_src = "".join(src_lines)
    ast.parse(out_src)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src/dashboard").mkdir(parents=True)
    (OUT / "tests").mkdir()
    (OUT / "data/jeju_plants").mkdir(parents=True)
    (OUT / "src/dashboard/app_v4.py").write_text(out_src, encoding="utf-8")
    for f in sorted((WT / "tests").glob("test_v4_*.py")):
        shutil.copy2(f, OUT / "tests" / f.name)
    for f in ("__init__.py", "conftest.py"):
        shutil.copy2(WT / "tests" / f, OUT / "tests" / f)
    shutil.copy2(WT / "data/jeju_plants/jeju_power_plants.json", OUT / "data/jeju_plants/jeju_power_plants.json")
    print(f"edits={len(edits)} main_lines={main_src.count(chr(10))}")
    r = subprocess.run([PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=OUT,
                       capture_output=True, text=True)
    print(r.stdout[-1500:])
    print(r.stderr[-1500:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
