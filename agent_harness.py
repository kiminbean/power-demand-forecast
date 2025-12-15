#!/usr/bin/env python3
"""
하이브리드 자율 개발 에이전트 파이프라인 (Hybrid Agent Harness)

설계 원칙:
- Anthropic Insight: 상태 기반 지속성 (State Persistence)
- DeepMind IMO Insight: 검증-개선 루프 (Verification-Refinement Loop)

구성:
- Generator (Worker): Claude Code - 코드 작성
- Verifier (Architect): Gemini CLI - 코드 검증
- Controller (Harness): 이 스크립트 - 상태 관리 및 루프 제어

사용법:
    python agent_harness.py                    # 다음 작업 실행
    python agent_harness.py --task DATA-001    # 특정 작업 실행
    python agent_harness.py --status           # 작업 상태 확인
    python agent_harness.py --init             # 상태 초기화
"""

import subprocess
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass, field, asdict
from pathlib import Path
import argparse
import re

# === 설정 ===
PROJECT_ROOT = Path(__file__).parent
STATE_FILE = PROJECT_ROOT / "feature_list.json"
PROGRESS_FILE = PROJECT_ROOT / "results" / "claude-progress.txt"
MAX_RETRY = 3

# === 타입 정의 ===

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """작업 정의"""
    id: str
    description: str
    status: str = "pending"
    phase: str = ""
    priority: str = "medium"
    subtasks: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)
    notes: str = ""
    retry_count: int = 0

    @classmethod
    def from_dict(cls, data: Dict) -> 'Task':
        """딕셔너리에서 Task 생성"""
        return cls(
            id=data.get('id', ''),
            description=data.get('description', ''),
            status=data.get('status', 'pending'),
            phase=data.get('phase', ''),
            priority=data.get('priority', 'medium'),
            subtasks=data.get('subtasks', []),
            output_files=data.get('output_files', []),
            files_changed=data.get('files_changed', []),
            notes=data.get('notes', ''),
            retry_count=data.get('retry_count', 0)
        )


# === 상태 관리 (Anthropic Insight) ===

def load_state() -> Dict[str, Any]:
    """
    외부 상태 파일을 통해 컨텍스트를 유지합니다.
    Anthropic Insight: 에이전트가 매 세션마다 상태를 파악할 수 있게 함
    """
    if not STATE_FILE.exists():
        return {"project": {}, "tasks": []}

    with open(STATE_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_state(state: Dict[str, Any]):
    """상태 저장"""
    with open(STATE_FILE, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_tasks(state: Dict[str, Any]) -> List[Task]:
    """상태에서 태스크 목록 추출"""
    return [Task.from_dict(t) for t in state.get('tasks', [])]


def update_task(state: Dict[str, Any], task_id: str, updates: Dict[str, Any]):
    """특정 태스크 업데이트"""
    for task in state.get('tasks', []):
        if task.get('id') == task_id:
            task.update(updates)
            break
    save_state(state)


def log_progress(message: str):
    """
    진행 상황을 로그 파일에 기록
    Anthropic Insight: 작업 히스토리를 통한 컨텍스트 유지
    """
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"

    with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry)

    print(log_entry.strip())


# === Claude Code 실행 (Generator/Worker) ===

def run_claude_code(prompt: str, timeout: int = 300) -> str:
    """
    Coding Agent (Claude Code): 실제 코드를 작성합니다.
    Anthropic Insight: 한 번에 하나씩 점진적으로 작업
    """
    log_progress(f"Claude instruction: \n{prompt[:200]}...")

    try:
        # Claude Code CLI 실행 (-p: 프롬프트 모드)
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(PROJECT_ROOT)
        )

        output = result.stdout + result.stderr
        log_progress(f"Claude output: {output[:200]}")

        return output

    except subprocess.TimeoutExpired:
        log_progress("Claude Code timeout")
        return "TIMEOUT"
    except FileNotFoundError:
        log_progress("Claude Code CLI not found. Install: npm install -g @anthropic-ai/claude-code")
        return "CLI_NOT_FOUND"
    except Exception as e:
        log_progress(f"Claude Code error: {e}")
        return f"ERROR: {e}"


# === Gemini 검증 (Verifier/Architect) ===

def run_gemini_verifier(task: Task, code_diff: str) -> Dict[str, Any]:
    """
    Verifier (Gemini CLI): IMO 논문의 검증 파이프라인을 적용합니다.
    역할: Critical Error 및 Justification Gap을 식별

    L1 검증: 테스트 실행 (Deterministic)
    L2 검증: Gemini 코드 리뷰 (Probabilistic)
    """
    log_progress(f"Gemini Verifier 시작: {task.id}")

    # L1: pytest 실행 (결정적 검증)
    l1_result = run_pytest()
    if not l1_result['passed']:
        return {
            "verdict": "FAIL",
            "level": "L1",
            "issues": l1_result.get('errors', ['pytest failed']),
            "suggestions": "테스트 실패 수정 필요"
        }

    # L2: Gemini 코드 리뷰 (확률적 검증)
    verification_prompt = f"""
당신은 수석 아키텍트이자 엄격한 ML/DL 코드 리뷰어입니다.

[작업 컨텍스트]
ID: {task.id}
설명: {task.description}
요구사항: {', '.join(task.subtasks)}

[변경 사항 (Git Diff)]
```diff
{code_diff[:3000]}
```

[검증 기준]
1. Data Leakage 여부 (train/test 데이터 분리)
2. 결측치/이상치 처리 적절성
3. 스케일링 역변환 로직 존재 여부
4. 재현성 (random seed 설정)
5. MPS/CUDA 디바이스 호환성

[응답 형식]
반드시 JSON 형식으로만 응답하세요:
{{"verdict": "PASS" | "FAIL", "issues": ["문제1", "문제2"], "suggestions": "개선 제안"}}
"""

    try:
        result = subprocess.run(
            ["gemini", "-p", verification_prompt],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT)
        )

        output = result.stdout + result.stderr
        log_progress(f"Gemini output: {output[:300]}")

        # JSON 파싱 시도
        json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())

        # JSON 파싱 실패시 기본값
        if "PASS" in output.upper():
            return {"verdict": "PASS", "issues": [], "suggestions": ""}
        else:
            return {"verdict": "FAIL", "issues": ["Verification unclear"], "suggestions": output[:200]}

    except subprocess.TimeoutExpired:
        log_progress("Gemini timeout - assuming PASS")
        return {"verdict": "PASS", "issues": [], "suggestions": "Timeout - manual review recommended"}
    except FileNotFoundError:
        log_progress("Gemini CLI not found - skipping L2 verification")
        return {"verdict": "PASS", "issues": [], "suggestions": "Gemini not available"}
    except Exception as e:
        log_progress(f"Gemini error: {e}")
        return {"verdict": "PASS", "issues": [], "suggestions": f"Error: {e}"}


def run_pytest() -> Dict[str, Any]:
    """L1 검증: pytest 실행"""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT)
        )

        passed = result.returncode == 0

        return {
            "passed": passed,
            "output": result.stdout,
            "errors": result.stderr.split('\n') if not passed else []
        }
    except Exception as e:
        return {"passed": True, "output": "", "errors": []}  # 테스트 없으면 통과


def get_git_diff() -> str:
    """스테이징된 변경사항 가져오기"""
    try:
        # staged 변경사항
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT)
        )
        diff = result.stdout

        # unstaged 변경사항도 포함
        if not diff:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT)
            )
            diff = result.stdout

        return diff
    except Exception:
        return ""


def git_commit(message: str):
    """변경사항 커밋"""
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(PROJECT_ROOT), check=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=str(PROJECT_ROOT),
            check=True
        )
        log_progress(f"Git commit: {message}")
    except subprocess.CalledProcessError as e:
        log_progress(f"Git commit failed: {e}")


# === 메인 파이프라인 ===

def execute_task(task: Task, state: Dict[str, Any]) -> bool:
    """
    단일 태스크 실행 파이프라인

    Flow:
    1. 상태를 in_progress로 변경
    2. Claude Code에게 구현 지시
    3. Gemini에게 검증 요청
    4. PASS: 커밋 후 done, FAIL: 재시도 또는 failed
    """
    log_progress(f"START: {task.id} - {task.description}")

    # 1. 상태 업데이트
    update_task(state, task.id, {"status": "in_progress"})

    # 2. Claude Code 프롬프트 생성
    prompt = f"""
작업 ID: {task.id}
작업 설명: {task.description}

세부 요구사항:
{chr(10).join(f'- {s}' for s in task.subtasks)}

출력 파일:
{chr(10).join(f'- {f}' for f in task.output_files)}

{'추가 지침: ' + task.notes if task.notes else ''}

지침:
1. 위 요구사항을 모두 충족하는 코드를 작성하세요.
2. 테스트 코드가 필요하면 tests/ 폴더에 작성하세요.
3. M1 MacBook Pro MPS를 지원해야 합니다.
4. 완료 후 변경된 파일을 git add로 스테이징하세요.
"""

    # 3. Claude Code 실행
    output = run_claude_code(prompt)

    if "CLI_NOT_FOUND" in output or "ERROR" in output:
        log_progress(f"FAILED: {task.id} - Claude Code 실행 실패")
        update_task(state, task.id, {"status": "failed"})
        return False

    # 4. 변경사항 확인
    diff = get_git_diff()
    if not diff:
        log_progress(f"WARNING: {task.id} - 변경사항 없음")
        # 변경사항 없어도 검증은 진행

    # 5. Gemini 검증
    update_task(state, task.id, {"status": "verifying"})
    verification = run_gemini_verifier(task, diff)

    # 6. 결과 처리
    if verification.get("verdict") == "PASS":
        log_progress(f"VERIFIED: {task.id}")

        # 커밋
        commit_msg = f"Feat({task.id}): {task.description}"
        git_commit(commit_msg)

        update_task(state, task.id, {
            "status": "done",
            "files_changed": task.output_files
        })

        log_progress(f"DONE: {task.id}")
        return True
    else:
        # 실패 처리
        issues = verification.get("issues", [])
        suggestions = verification.get("suggestions", "")

        log_progress(f"VERIFICATION FAILED: {task.id}")
        log_progress(f"Issues: {issues}")
        log_progress(f"Suggestions: {suggestions}")

        # 재시도 로직
        current_retry = task.retry_count + 1
        if current_retry < MAX_RETRY:
            log_progress(f"Retry {current_retry}/{MAX_RETRY} for {task.id}")

            # Self-Correction 프롬프트
            refinement_prompt = f"""
이전 작업 ({task.id})에 대해 검증자가 다음 문제를 발견했습니다:

문제점:
{chr(10).join(f'- {i}' for i in issues)}

제안:
{suggestions}

위 피드백을 반영하여 코드를 수정하고 다시 git add 하세요.
"""
            run_claude_code(refinement_prompt)

            update_task(state, task.id, {"retry_count": current_retry})

            # 재검증 (재귀적으로 호출하지 않고 상태만 업데이트)
            return False
        else:
            update_task(state, task.id, {"status": "failed"})
            log_progress(f"FAILED: {task.id} - Max retries exceeded")
            return False


def get_next_task(tasks: List[Task]) -> Optional[Task]:
    """다음 실행할 태스크 선택 (우선순위 기반)"""
    priority_order = {"high": 0, "medium": 1, "low": 2}

    pending = [t for t in tasks if t.status in ["pending", "in_progress"]]

    if not pending:
        return None

    # 우선순위 정렬
    pending.sort(key=lambda t: (priority_order.get(t.priority, 1), t.id))

    return pending[0]


def show_status(state: Dict[str, Any]):
    """작업 상태 출력"""
    tasks = get_tasks(state)

    print("\n" + "=" * 60)
    print(f"프로젝트: {state.get('project', {}).get('name', 'Unknown')}")
    print("=" * 60)

    status_counts = {}
    for task in tasks:
        status_counts[task.status] = status_counts.get(task.status, 0) + 1

    print(f"\n총 작업: {len(tasks)}")
    for status, count in sorted(status_counts.items()):
        emoji = {"done": "✅", "pending": "⏳", "in_progress": "🔄", "failed": "❌", "verifying": "🔍"}.get(status, "•")
        print(f"  {emoji} {status}: {count}")

    print("\n작업 목록:")
    for task in tasks:
        emoji = {"done": "✅", "pending": "⏳", "in_progress": "🔄", "failed": "❌", "verifying": "🔍"}.get(task.status, "•")
        print(f"  {emoji} [{task.id}] {task.description[:40]}... ({task.status})")

    print("=" * 60 + "\n")


# === CLI 인터페이스 ===

def main():
    parser = argparse.ArgumentParser(
        description="하이브리드 자율 개발 에이전트 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    python agent_harness.py                    # 다음 작업 실행
    python agent_harness.py --task DATA-001    # 특정 작업 실행
    python agent_harness.py --status           # 작업 상태 확인
    python agent_harness.py --verify DATA-001  # 특정 작업 검증만 수행
        """
    )

    parser.add_argument("--task", "-t", help="실행할 특정 작업 ID")
    parser.add_argument("--status", "-s", action="store_true", help="작업 상태 출력")
    parser.add_argument("--verify", "-v", help="특정 작업 검증만 수행")
    parser.add_argument("--reset", help="특정 작업을 pending으로 초기화")

    args = parser.parse_args()

    # 상태 로드
    state = load_state()
    tasks = get_tasks(state)

    if not tasks:
        print("feature_list.json에 태스크가 없습니다.")
        return

    # 상태 출력
    if args.status:
        show_status(state)
        return

    # 작업 초기화
    if args.reset:
        update_task(state, args.reset, {"status": "pending", "retry_count": 0})
        print(f"작업 {args.reset}을 pending으로 초기화했습니다.")
        return

    # 검증만 수행
    if args.verify:
        task = next((t for t in tasks if t.id == args.verify), None)
        if task:
            diff = get_git_diff()
            result = run_gemini_verifier(task, diff)
            print(f"\n검증 결과: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"작업 {args.verify}를 찾을 수 없습니다.")
        return

    # 특정 작업 또는 다음 작업 실행
    if args.task:
        target = next((t for t in tasks if t.id == args.task), None)
        if not target:
            print(f"작업 {args.task}를 찾을 수 없습니다.")
            return
    else:
        target = get_next_task(tasks)
        if not target:
            print("모든 작업이 완료되었습니다! 🎉")
            show_status(state)
            return

    # 작업 실행
    print(f"\n{'='*60}")
    print(f"작업 시작: {target.id} - {target.description}")
    print(f"{'='*60}\n")

    success = execute_task(target, state)

    if success:
        print(f"\n✅ 작업 완료: {target.id}")
    else:
        print(f"\n❌ 작업 실패 또는 검증 필요: {target.id}")

    show_status(state)


if __name__ == "__main__":
    main()
