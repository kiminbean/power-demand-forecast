"""빌드 명세 골격(final_skeleton.py)을 실제 app_v4.py에 끼워 넣은 합성 모듈을 만들고,
수정하지 않은 tests/test_v4_*.py 35개를 그 모듈에 대해 실행한다.

- 워크트리 소스는 읽기만 한다. 결과물은 scratchpad/design/dryrun/ 아래에만 쓴다.
- 합성 모듈 = 원본 app_v4.py
    - 모듈 최상위 <style> st.markdown 제거 (명세: CSS는 main() 첫 줄 st.html)
    - create_gauge_chart, main 을 골격 버전으로 교체
- 실행: .venv/bin/python -I docs/2026-11-23-dashboard-ui-top1/dryrun_build.py  (저장소 루트에서)
"""
from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

WT = Path(__file__).resolve().parents[2]  # 저장소(워크트리) 루트
HERE = Path(__file__).resolve().parent
OUT = HERE / "dryrun"
PY = sys.executable


def node_span(src_lines: list[str], node: ast.AST) -> tuple[int, int]:
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return start, node.end_lineno  # 1-based inclusive


def main() -> int:
    src = (WT / "src/dashboard/app_v4.py").read_text(encoding="utf-8")
    skel = (HERE / "final_skeleton.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    skel_tree = ast.parse(skel)
    skel_lines = skel.splitlines(keepends=True)
    repl = {}
    for n in skel_tree.body:
        if isinstance(n, ast.FunctionDef):
            a, b = node_span(skel_lines, n)
            repl[n.name] = "".join(skel_lines[a - 1:b])
    lines = src.splitlines(keepends=True)
    edits = []  # (start, end, text)
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in repl:
            a, b = node_span(lines, n)
            edits.append((a, b, repl[n.name]))
        elif (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "markdown"
              and n.value.args and isinstance(n.value.args[0], ast.Constant)
              and "<style>" in str(n.value.args[0].value)):
            a, b = node_span(lines, n)
            edits.append((a, b, "# (명세) 전역 CSS는 main() 첫 줄 st.html(_css_dashboard())로 옮겼다\n"))
    assert {e[2][:30] for e in edits}, "no edits"
    for a, b, text in sorted(edits, reverse=True):
        lines[a - 1:b] = [text if text.endswith("\n") else text + "\n"]
    out_src = "".join(lines)
    ast.parse(out_src)  # 문법 확인

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "src/dashboard").mkdir(parents=True)
    (OUT / "tests").mkdir(parents=True)
    (OUT / "data/jeju_plants").mkdir(parents=True)
    (OUT / "src/dashboard/app_v4.py").write_text(out_src, encoding="utf-8")
    for f in sorted((WT / "tests").glob("test_v4_*.py")):
        shutil.copy2(f, OUT / "tests" / f.name)
    for f in ("__init__.py", "conftest.py"):
        shutil.copy2(WT / "tests" / f, OUT / "tests" / f)
    shutil.copy2(WT / "data/jeju_plants/jeju_power_plants.json", OUT / "data/jeju_plants/jeju_power_plants.json")
    print(f"edits: {len(edits)} · synthetic module lines: {out_src.count(chr(10))}")
    r = subprocess.run([PY, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=OUT,
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    print(r.stderr[-2000:])
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
