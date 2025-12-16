#!/usr/bin/env python3
"""
Project Status Backup Script
============================

Saves current project status to backup file.
Run this periodically or after significant changes.

Usage:
    python scripts/backup_status.py
    python scripts/backup_status.py --message "Completed frontend setup"
"""

import subprocess
import argparse
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BACKUP_DIR = PROJECT_ROOT / ".claude" / "backups"
STATUS_FILE = BACKUP_DIR / "PROJECT_STATUS.md"
HISTORY_DIR = BACKUP_DIR / "history"


def get_git_info():
    """Get recent git commits"""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-10"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT
        )
        return result.stdout.strip()
    except Exception:
        return "Unable to get git info"


def get_test_status():
    """Get test count from last run"""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30
        )
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if 'test' in line.lower():
                return line
        return "Tests available"
    except Exception:
        return "Unable to get test status"


def backup_status(message: str = None):
    """Create a timestamped backup"""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Read current status
    if STATUS_FILE.exists():
        current_status = STATUS_FILE.read_text()

        # Save to history
        history_file = HISTORY_DIR / f"status_{timestamp}.md"
        history_file.write_text(current_status)
        print(f"✓ Backup saved: {history_file.name}")

    # Update timestamp in status file
    if STATUS_FILE.exists():
        content = STATUS_FILE.read_text()

        # Update last updated timestamp
        import re
        new_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M KST")
        content = re.sub(
            r'> Last Updated: .*',
            f'> Last Updated: {new_timestamp}',
            content
        )

        # Add message if provided
        if message:
            # Find Notes section and add message
            if "## Notes" in content:
                content = content.replace(
                    "## Notes",
                    f"## Notes\n- [{new_timestamp}] {message}"
                )

        STATUS_FILE.write_text(content)
        print(f"✓ Status updated: {new_timestamp}")

    # Show recent git commits
    print(f"\nRecent commits:")
    print(get_git_info())


def main():
    parser = argparse.ArgumentParser(description="Backup project status")
    parser.add_argument(
        "--message", "-m",
        type=str,
        help="Status message to add"
    )
    args = parser.parse_args()

    backup_status(args.message)
    print("\n✓ Backup complete!")


if __name__ == "__main__":
    main()
