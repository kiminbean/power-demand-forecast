#!/bin/bash
# 자동 커밋 시스템 설치/제거 스크립트

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLIST_SOURCE="$PROJECT_ROOT/.launchd/com.powerdemand.autocommit.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.powerdemand.autocommit.plist"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    echo "Usage: $0 {install|uninstall|status}"
    echo ""
    echo "Commands:"
    echo "  install    - 자동 커밋 시스템 설치 및 활성화 (30분마다 실행)"
    echo "  uninstall  - 자동 커밋 시스템 제거"
    echo "  status     - 자동 커밋 시스템 상태 확인"
    echo ""
    echo "수동 실행: ./commit.sh"
}

install_autocommit() {
    echo -e "${YELLOW}[Setup] Installing auto-commit system...${NC}"

    # LaunchAgents 디렉토리 생성
    mkdir -p "$HOME/Library/LaunchAgents"

    # plist 파일 복사
    cp "$PLIST_SOURCE" "$PLIST_DEST"
    echo -e "${GREEN}✓ Copied plist to LaunchAgents${NC}"

    # launchd에 등록
    launchctl load "$PLIST_DEST" 2>/dev/null || true
    echo -e "${GREEN}✓ Loaded auto-commit service${NC}"

    # 서비스 시작
    launchctl start com.powerdemand.autocommit 2>/dev/null || true
    echo -e "${GREEN}✓ Started auto-commit service${NC}"

    echo ""
    echo -e "${GREEN}[Setup] Auto-commit system installed successfully!${NC}"
    echo ""
    echo "설정된 동작:"
    echo "  • 30분마다 자동으로 변경사항 커밋"
    echo "  • 로그: logs/autocommit.log"
    echo "  • 에러: logs/autocommit.error.log"
    echo ""
    echo "수동 실행: ./commit.sh"
    echo "상태 확인: ./setup-autocommit.sh status"
    echo "제거: ./setup-autocommit.sh uninstall"
}

uninstall_autocommit() {
    echo -e "${YELLOW}[Setup] Uninstalling auto-commit system...${NC}"

    # 서비스 중지
    launchctl stop com.powerdemand.autocommit 2>/dev/null || true
    echo -e "${GREEN}✓ Stopped auto-commit service${NC}"

    # launchd에서 제거
    launchctl unload "$PLIST_DEST" 2>/dev/null || true
    echo -e "${GREEN}✓ Unloaded auto-commit service${NC}"

    # plist 파일 삭제
    rm -f "$PLIST_DEST"
    echo -e "${GREEN}✓ Removed plist file${NC}"

    echo ""
    echo -e "${GREEN}[Setup] Auto-commit system uninstalled successfully!${NC}"
    echo ""
    echo "수동 실행은 여전히 가능: ./commit.sh"
}

check_status() {
    echo -e "${YELLOW}[Status] Checking auto-commit system status...${NC}"
    echo ""

    if [ -f "$PLIST_DEST" ]; then
        echo -e "${GREEN}✓ Auto-commit service is installed${NC}"

        # launchctl로 상태 확인
        if launchctl list | grep -q com.powerdemand.autocommit; then
            echo -e "${GREEN}✓ Service is loaded and running${NC}"

            # 마지막 실행 시간 확인
            if [ -f "$PROJECT_ROOT/logs/autocommit.log" ]; then
                echo ""
                echo "Last execution log:"
                tail -5 "$PROJECT_ROOT/logs/autocommit.log"
            fi
        else
            echo -e "${RED}✗ Service is not running${NC}"
            echo "Run: ./setup-autocommit.sh install"
        fi
    else
        echo -e "${RED}✗ Auto-commit service is not installed${NC}"
        echo "Run: ./setup-autocommit.sh install"
    fi

    echo ""
    echo "Configuration:"
    echo "  • Interval: Every 30 minutes"
    echo "  • Log: logs/autocommit.log"
    echo "  • Error: logs/autocommit.error.log"
}

# Main
case "${1:-}" in
    install)
        install_autocommit
        ;;
    uninstall)
        uninstall_autocommit
        ;;
    status)
        check_status
        ;;
    *)
        show_help
        exit 1
        ;;
esac
