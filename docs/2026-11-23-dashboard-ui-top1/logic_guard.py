"""UI 개편이 데이터·알림 로직을 바꾸지 않았는지 AST로 비교한다.

사용 예:
    .venv/bin/python logic_guard.py --before <main>/src/dashboard/app_v4.py --after <wt>/src/dashboard/app_v4.py

규칙:
- 최상위 함수·클래스·상수 할당을 이름별로 모아 ast.dump(주석·docstring 제외)로 비교한다.
- 표현 계층(PRESENTATION)은 바뀌어도 '허용'으로, 나머지는 바뀌면 '검토 필요'로 보고한다.
- 사라진 이름과 새로 생긴 이름도 보고한다.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# 바뀌어도 되는 표현 계층 이름(차트·지도·화면 구성)
PRESENTATION_PREFIXES = ("create_", "render_", "_render", "_css", "_theme", "_chart", "_fmt", "_html")
# 빌드 명세 5.1-5의 예외 이름 5개(접두어 규칙을 따르지 않는 표현 계층 상수·클래스). 2026-11-24 감사 반영 때 추가.
PRESENTATION_NAMES = {"main", "SERIES", "BOUNDARY_COLOR", "ALERT_RED", "PD_ICONS", "_MapThemeSync"}


def _strip_docstrings(node: ast.AST) -> ast.AST:
    for sub in ast.walk(node):
        body = getattr(sub, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) \
                and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            sub.body = body[1:] or [ast.Pass()]
    return node


def collect(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for node in tree.body:
        name = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        if name is None:
            continue
        out[name] = ast.dump(_strip_docstrings(node), include_attributes=False)
    return out


def is_presentation(name: str) -> bool:
    return name in PRESENTATION_NAMES or name.startswith(PRESENTATION_PREFIXES) or name.isupper() and "CSS" in name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()
    before, after = collect(Path(args.before)), collect(Path(args.after))
    changed = sorted(n for n in before.keys() & after.keys() if before[n] != after[n])
    report = {
        "changed_presentation": [n for n in changed if is_presentation(n)],
        "changed_logic_REVIEW": [n for n in changed if not is_presentation(n)],
        "removed": sorted(before.keys() - after.keys()),
        "added": sorted(after.keys() - before.keys()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["changed_logic_REVIEW"] or report["removed"] else 0


if __name__ == "__main__":
    sys.exit(main())
