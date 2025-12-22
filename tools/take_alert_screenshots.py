#!/usr/bin/env python3
"""Take screenshots of each alert level by patching get_current_power_status."""
import subprocess
import time
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.sync_api import sync_playwright

DASHBOARD_PATH = PROJECT_ROOT / "src" / "dashboard" / "app_v4.py"
SCREENSHOT_DIR = PROJECT_ROOT / "docs" / "screenshots"

# Alert levels to capture: (reserve_rate, filename, description)
ALERT_LEVELS = [
    (12.0, "07_alert_caution.png", "Caution (10-15%)"),
    (7.0, "08_alert_warning.png", "Warning (5-10%)"),
    (3.0, "09_alert_critical.png", "Critical (<5%)"),
]


def patch_dashboard(reserve_rate: float) -> str:
    """Patch get_current_power_status to return fixed reserve rate."""
    patch_code = f'''
# AUTO-GENERATED PATCH - DO NOT EDIT
_ORIGINAL_GET_CURRENT_POWER_STATUS = get_current_power_status

@st.cache_data(ttl=5)
def get_current_power_status():
    """Patched function for screenshot."""
    result = _ORIGINAL_GET_CURRENT_POWER_STATUS.__wrapped__()
    result = dict(result)
    result['reserve_rate'] = {reserve_rate}
    result['data_source'] = '테스트 모드 ({reserve_rate}%)'
    return result
# END PATCH
'''

    content = DASHBOARD_PATH.read_text()

    # Find the position before the main() function
    marker = "def main():"
    if marker in content:
        idx = content.index(marker)
        patched = content[:idx] + patch_code + "\n" + content[idx:]
        return patched
    else:
        raise ValueError("Could not find insertion point: def main():")


def take_screenshot(port: int, output_path: Path, wait_time: int = 8):
    """Take screenshot using Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(f"http://localhost:{port}", wait_until="networkidle")
        time.sleep(wait_time)  # Wait for Streamlit to fully render
        page.screenshot(path=str(output_path), full_page=False)
        browser.close()
    print(f"Screenshot saved: {output_path}")


def main():
    # Read original content
    original_content = DASHBOARD_PATH.read_text()
    port = 8505  # Use different port

    for reserve_rate, filename, description in ALERT_LEVELS:
        print(f"\n{'='*50}")
        print(f"Taking screenshot: {description}")
        print(f"Reserve rate: {reserve_rate}%")
        print(f"{'='*50}")

        process = None
        try:
            # Patch the code
            patched_content = patch_dashboard(reserve_rate)
            DASHBOARD_PATH.write_text(patched_content)
            print("Code patched")

            # Start Streamlit
            env = os.environ.copy()
            env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

            process = subprocess.Popen(
                [
                    sys.executable, "-m", "streamlit", "run",
                    str(DASHBOARD_PATH),
                    "--server.port", str(port),
                    "--server.headless", "true",
                    "--browser.gatherUsageStats", "false",
                ],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )

            print("Waiting for server to start...")
            time.sleep(12)  # Wait for server

            # Take screenshot
            output_path = SCREENSHOT_DIR / filename
            take_screenshot(port, output_path, wait_time=6)

        except Exception as e:
            print(f"Error: {e}")
            raise

        finally:
            # Kill the process
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                print("Server stopped")

            # Restore original
            DASHBOARD_PATH.write_text(original_content)
            print("Code restored")

        # Wait before next
        time.sleep(2)

    print("\n" + "="*50)
    print("All screenshots complete!")
    print("="*50)


if __name__ == "__main__":
    main()
