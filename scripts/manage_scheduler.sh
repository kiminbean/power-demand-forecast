#!/bin/bash
#
# SMP Scheduler Management Script
# ===============================
#
# Usage:
#   ./scripts/manage_scheduler.sh start    # Start scheduler
#   ./scripts/manage_scheduler.sh stop     # Stop scheduler
#   ./scripts/manage_scheduler.sh restart  # Restart scheduler
#   ./scripts/manage_scheduler.sh status   # Check status
#   ./scripts/manage_scheduler.sh install  # Install as launchd service
#   ./scripts/manage_scheduler.sh uninstall # Remove launchd service
#   ./scripts/manage_scheduler.sh run-now  # Run crawler immediately
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PLIST_NAME="com.rebms.smp-scheduler"
PLIST_SRC="$SCRIPT_DIR/smp_scheduler.plist"
PLIST_DST="$HOME/Library/LaunchAgents/$PLIST_NAME.plist"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
LOG_DIR="$PROJECT_ROOT/logs"

# Ensure log directory exists
mkdir -p "$LOG_DIR"

print_header() {
    echo ""
    echo "=================================================="
    echo " SMP Scheduler Manager"
    echo "=================================================="
}

check_python() {
    if [ ! -f "$PYTHON" ]; then
        echo "Error: Python not found at $PYTHON"
        echo "Please activate virtual environment first."
        exit 1
    fi
}

start() {
    check_python
    echo "Starting SMP scheduler..."

    if launchctl list | grep -q "$PLIST_NAME"; then
        echo "Scheduler is already running (via launchd)"
        return
    fi

    cd "$PROJECT_ROOT"
    nohup "$PYTHON" -m src.smp.crawlers.scheduler --foreground \
        > "$LOG_DIR/smp_scheduler.out" 2> "$LOG_DIR/smp_scheduler.err" &

    echo "Scheduler started with PID: $!"
    echo "Logs: $LOG_DIR/smp_scheduler.log"
}

stop() {
    echo "Stopping SMP scheduler..."

    # Stop launchd service if running
    if launchctl list | grep -q "$PLIST_NAME"; then
        launchctl unload "$PLIST_DST" 2>/dev/null || true
    fi

    # Kill any running scheduler processes
    pkill -f "src.smp.crawlers.scheduler" 2>/dev/null || true

    echo "Scheduler stopped"
}

restart() {
    stop
    sleep 2
    start
}

status() {
    print_header

    echo ""
    echo "Launchd Service:"
    if launchctl list | grep -q "$PLIST_NAME"; then
        echo "  Status: RUNNING (launchd)"
        launchctl list | grep "$PLIST_NAME"
    else
        echo "  Status: Not loaded"
    fi

    echo ""
    echo "Process:"
    if pgrep -f "src.smp.crawlers.scheduler" > /dev/null; then
        echo "  Status: RUNNING"
        ps aux | grep "src.smp.crawlers.scheduler" | grep -v grep
    else
        echo "  Status: Not running"
    fi

    echo ""
    echo "Data Status:"
    check_python
    cd "$PROJECT_ROOT"
    "$PYTHON" -m src.smp.crawlers.scheduler --status
}

install() {
    check_python
    echo "Installing SMP scheduler as launchd service..."

    # Create log directory
    mkdir -p "$LOG_DIR"

    # Copy plist to LaunchAgents
    mkdir -p "$HOME/Library/LaunchAgents"
    cp "$PLIST_SRC" "$PLIST_DST"

    # Load the service
    launchctl load "$PLIST_DST"

    echo ""
    echo "Service installed and started!"
    echo "  Plist: $PLIST_DST"
    echo "  Logs:  $LOG_DIR/smp_scheduler.log"
    echo ""
    echo "The scheduler will:"
    echo "  - Update SMP data every hour (at :15)"
    echo "  - Do full sync daily at 06:00"
    echo "  - Auto-start on login"
}

uninstall() {
    echo "Uninstalling SMP scheduler service..."

    # Stop and unload
    launchctl unload "$PLIST_DST" 2>/dev/null || true

    # Remove plist
    rm -f "$PLIST_DST"

    echo "Service uninstalled"
}

run_now() {
    check_python
    echo "Running SMP crawler now..."
    cd "$PROJECT_ROOT"
    "$PYTHON" -m src.smp.crawlers.scheduler --run-now --hours "${2:-48}"
}

logs() {
    echo "Showing scheduler logs..."
    tail -f "$LOG_DIR/smp_scheduler.log"
}

case "$1" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    status)
        status
        ;;
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    run-now)
        run_now "$@"
        ;;
    logs)
        logs
        ;;
    *)
        print_header
        echo ""
        echo "Usage: $0 {start|stop|restart|status|install|uninstall|run-now|logs}"
        echo ""
        echo "Commands:"
        echo "  start     - Start scheduler in background"
        echo "  stop      - Stop scheduler"
        echo "  restart   - Restart scheduler"
        echo "  status    - Show scheduler status"
        echo "  install   - Install as launchd service (auto-start)"
        echo "  uninstall - Remove launchd service"
        echo "  run-now   - Run crawler immediately"
        echo "  logs      - Follow log output"
        echo ""
        exit 1
        ;;
esac
