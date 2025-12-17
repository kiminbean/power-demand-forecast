#!/usr/bin/env python3
"""
Claude Code 대화 기록에서 사용자 프롬프트 추출

Usage:
    python scripts/extract_prompts.py
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Claude 대화 기록 디렉토리
CLAUDE_DIR = Path.home() / ".claude"
PROJECT_DIR = CLAUDE_DIR / "projects" / "-Users-ibkim-Ormi-1-power-demand-forecast"
OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "CONVERSATION_PROMPTS.md"


def extract_prompts_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """JSONL 파일에서 사용자 프롬프트 추출"""
    prompts = []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)

                    # Claude Code JSONL 형식: type이 "user"이고 message.role이 "user"인 경우
                    if data.get('type') == 'user':
                        message = data.get('message', {})
                        if isinstance(message, dict) and message.get('role') == 'user':
                            content = message.get('content', '')

                            # content가 리스트인 경우 텍스트 추출
                            if isinstance(content, list):
                                text_parts = []
                                for item in content:
                                    if isinstance(item, dict) and item.get('type') == 'text':
                                        text_parts.append(item.get('text', ''))
                                    elif isinstance(item, str):
                                        text_parts.append(item)
                                content = '\n'.join(text_parts)

                            if content and len(content) > 10:  # 짧은 메시지 제외
                                timestamp = data.get('timestamp', '')
                                session_id = data.get('sessionId', '')
                                prompts.append({
                                    'timestamp': timestamp,
                                    'content': content[:3000],  # 내용 제한
                                    'file': file_path.name,
                                    'session_id': session_id
                                })

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return prompts


def format_prompt_for_markdown(prompt: Dict[str, Any], index: int) -> str:
    """프롬프트를 Markdown 형식으로 변환"""
    timestamp = prompt.get('timestamp', 'Unknown')
    content = prompt.get('content', '')

    # 타임스탬프 포맷팅
    if timestamp and isinstance(timestamp, str):
        try:
            if 'T' in timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                timestamp = dt.strftime('%Y-%m-%d %H:%M')
        except:
            pass

    # 코드 블록 내용은 그대로 유지
    if '```' in content:
        formatted_content = content
    else:
        # 일반 텍스트는 인용 블록으로
        formatted_content = content

    return f"""
### Prompt #{index}
> **시간**: {timestamp}

```
{formatted_content}
```

---
"""


def main():
    print("=" * 60)
    print("Claude Code 대화 기록 프롬프트 추출")
    print("=" * 60)

    all_prompts = []

    # 프로젝트 대화 기록 파일들
    if PROJECT_DIR.exists():
        print(f"\n프로젝트 디렉토리: {PROJECT_DIR}")
        jsonl_files = sorted(PROJECT_DIR.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True)

        for file_path in jsonl_files:
            if file_path.stat().st_size > 0:
                print(f"  처리 중: {file_path.name} ({file_path.stat().st_size / 1024:.1f} KB)")
                prompts = extract_prompts_from_jsonl(file_path)
                all_prompts.extend(prompts)
                print(f"    -> {len(prompts)}개 프롬프트 추출")

    # 전역 히스토리 파일 (선택적)
    history_file = CLAUDE_DIR / "history.jsonl"
    if history_file.exists():
        print(f"\n전역 히스토리: {history_file}")
        prompts = extract_prompts_from_jsonl(history_file)
        # power-demand-forecast 관련 프롬프트만 필터링 (키워드 기반)
        keywords = ['전력', '수요', '예측', 'power', 'demand', 'forecast', 'lstm', 'tft',
                   'streamlit', 'fastapi', 'api', '태양광', '풍력', 'kpx', '제주', 'jeju']
        filtered = [p for p in prompts if any(kw in p.get('content', '').lower() for kw in keywords)]
        all_prompts.extend(filtered)
        print(f"  -> {len(filtered)}개 관련 프롬프트 추출")

    # 중복 제거 (내용 기준)
    seen_contents = set()
    unique_prompts = []
    for p in all_prompts:
        content_hash = hash(p.get('content', '')[:500])
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_prompts.append(p)

    # 시간순 정렬
    unique_prompts.sort(key=lambda x: x.get('timestamp', ''), reverse=False)

    print(f"\n총 {len(unique_prompts)}개 고유 프롬프트 추출됨")

    # Markdown 파일 생성
    print(f"\n출력 파일: {OUTPUT_FILE}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"""# 전력 수요 예측 프로젝트 - 실제 대화 프롬프트

> **생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
> **총 프롬프트 수**: {len(unique_prompts)}개
> **프로젝트**: 제주도 전력 수요 예측 시스템

---

## 목차

이 문서는 Claude Code와의 실제 대화에서 사용된 프롬프트를 시간순으로 정리한 것입니다.

---

## 프롬프트 목록

""")

        for i, prompt in enumerate(unique_prompts, 1):
            f.write(format_prompt_for_markdown(prompt, i))

        f.write(f"""
---

## 통계

- **총 프롬프트**: {len(unique_prompts)}개
- **추출 소스**: Claude Code 대화 기록 (.jsonl)
- **프로젝트 경로**: `~/.claude/projects/-Users-ibkim-Ormi-1-power-demand-forecast/`

---

> 이 문서는 `scripts/extract_prompts.py` 스크립트로 자동 생성되었습니다.
""")

    print(f"\n✅ 완료: {OUTPUT_FILE}")
    print(f"   파일 크기: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
