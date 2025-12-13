#!/bin/bash
# 수동 자동 커밋 실행 래퍼 스크립트
# 사용법: ./commit.sh

echo "🤖 Running auto-commit..."
.git/hooks/auto-commit.sh
echo "✅ Done!"
