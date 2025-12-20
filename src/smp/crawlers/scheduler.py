"""
SMP 크롤러 스케줄러
==================

EPSIS SMP 데이터를 주기적으로 수집하는 스케줄러.

Usage:
    # 백그라운드 실행
    python -m src.smp.crawlers.scheduler &

    # 포그라운드 실행 (디버깅)
    python -m src.smp.crawlers.scheduler --foreground

    # 즉시 실행 (테스트)
    python -m src.smp.crawlers.scheduler --run-now

Author: Claude Code
Date: 2025-12
"""

import argparse
import logging
import signal
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# 프로젝트 루트를 sys.path에 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.smp.crawlers.epsis_crawler import EPSISCrawler, save_to_csv, print_statistics

# 로깅 설정
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "smp_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 데이터 저장 경로
DATA_PATH = PROJECT_ROOT / "data" / "smp" / "smp_real_epsis.csv"

# 스케줄 설정
SCHEDULE_CONFIG = {
    'hourly_update': {
        'enabled': True,
        'hour': '*',           # 매시간
        'minute': 15,          # 15분에 실행 (KPX 데이터 갱신 후)
    },
    'daily_full_sync': {
        'enabled': True,
        'hour': 6,             # 오전 6시
        'minute': 0,
        'days_back': 7,        # 7일치 재수집 (누락 방지)
    },
}

# 글로벌 종료 플래그
shutdown_flag = False


def signal_handler(signum, frame):
    """시그널 핸들러 (Graceful shutdown)"""
    global shutdown_flag
    logger.info(f"Signal {signum} received, shutting down...")
    shutdown_flag = True


def fetch_latest_data(hours_back: int = 48) -> bool:
    """최근 데이터 수집

    Args:
        hours_back: 수집할 과거 시간 수

    Returns:
        성공 여부
    """
    try:
        logger.info(f"Starting data fetch (last {hours_back} hours)")

        # 기간 계산
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(hours=hours_back)

        start_date = start_dt.strftime("%Y%m%d")
        end_date = end_dt.strftime("%Y%m%d")

        with EPSISCrawler() as crawler:
            df_new = crawler.fetch_range(start_date, end_date)

        if df_new.empty:
            logger.warning("No new data fetched")
            return False

        logger.info(f"Fetched {len(df_new)} new records")

        # 기존 데이터와 병합
        if DATA_PATH.exists():
            import pandas as pd
            df_existing = pd.read_csv(DATA_PATH)

            # 새 데이터 추가 및 중복 제거
            df_merged = pd.concat([df_existing, df_new], ignore_index=True)
            df_merged = df_merged.drop_duplicates(subset=['timestamp'], keep='last')
            df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

            save_to_csv(df_merged, DATA_PATH)
            logger.info(f"Merged data saved: {len(df_merged)} total records")
        else:
            save_to_csv(df_new, DATA_PATH)
            logger.info(f"New data saved: {len(df_new)} records")

        return True

    except Exception as e:
        logger.error(f"Data fetch failed: {e}", exc_info=True)
        return False


def fetch_today_data() -> bool:
    """오늘 데이터만 수집 (시간별 업데이트용)"""
    try:
        logger.info("Fetching today's data")

        today = datetime.now().strftime("%Y%m%d")

        with EPSISCrawler() as crawler:
            df_new = crawler.fetch_range(today, today)

        if df_new.empty:
            logger.warning("No data for today yet")
            return False

        logger.info(f"Fetched {len(df_new)} records for today")

        # 기존 데이터와 병합
        if DATA_PATH.exists():
            import pandas as pd
            df_existing = pd.read_csv(DATA_PATH)

            # 오늘 날짜 데이터 제거 후 새 데이터 추가
            today_str = datetime.now().strftime("%Y-%m-%d")
            df_existing = df_existing[df_existing['date'] != today_str]

            df_merged = pd.concat([df_existing, df_new], ignore_index=True)
            df_merged = df_merged.sort_values('timestamp').reset_index(drop=True)

            save_to_csv(df_merged, DATA_PATH)
            logger.info(f"Today's data updated: {len(df_new)} records")
        else:
            save_to_csv(df_new, DATA_PATH)

        return True

    except Exception as e:
        logger.error(f"Today's data fetch failed: {e}", exc_info=True)
        return False


def run_scheduler():
    """스케줄러 실행 (APScheduler 사용)"""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        logger.error("APScheduler not installed. Run: pip install apscheduler")
        logger.info("Falling back to simple loop scheduler...")
        run_simple_scheduler()
        return

    scheduler = BlockingScheduler()

    # 시간별 업데이트 (매시 15분)
    if SCHEDULE_CONFIG['hourly_update']['enabled']:
        scheduler.add_job(
            fetch_today_data,
            CronTrigger(minute=SCHEDULE_CONFIG['hourly_update']['minute']),
            id='hourly_update',
            name='Hourly SMP Update',
            misfire_grace_time=300,
        )
        logger.info("Scheduled: Hourly update at minute 15")

    # 일간 전체 동기화 (오전 6시)
    if SCHEDULE_CONFIG['daily_full_sync']['enabled']:
        days_back = SCHEDULE_CONFIG['daily_full_sync']['days_back']
        scheduler.add_job(
            lambda: fetch_latest_data(hours_back=days_back * 24),
            CronTrigger(
                hour=SCHEDULE_CONFIG['daily_full_sync']['hour'],
                minute=SCHEDULE_CONFIG['daily_full_sync']['minute']
            ),
            id='daily_sync',
            name='Daily Full Sync',
            misfire_grace_time=3600,
        )
        logger.info(f"Scheduled: Daily sync at 06:00 ({days_back} days)")

    # 시그널 핸들러 등록
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("=" * 50)
    logger.info("SMP Crawler Scheduler Started")
    logger.info(f"Data path: {DATA_PATH}")
    logger.info("=" * 50)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


def run_simple_scheduler():
    """간단한 루프 기반 스케줄러 (APScheduler 없을 때 대체)"""
    import time

    logger.info("=" * 50)
    logger.info("Simple Loop Scheduler Started")
    logger.info("=" * 50)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    last_hourly = None
    last_daily = None

    while not shutdown_flag:
        now = datetime.now()

        # 시간별 업데이트 (매시 15분)
        if now.minute == 15 and last_hourly != now.hour:
            logger.info("Running hourly update...")
            fetch_today_data()
            last_hourly = now.hour

        # 일간 동기화 (오전 6시)
        if now.hour == 6 and now.minute == 0 and last_daily != now.date():
            logger.info("Running daily sync...")
            fetch_latest_data(hours_back=7 * 24)
            last_daily = now.date()

        # 1분 대기
        time.sleep(60)

    logger.info("Scheduler stopped")


def get_status() -> dict:
    """스케줄러 상태 확인"""
    import pandas as pd

    status = {
        'scheduler': 'unknown',
        'data_file': str(DATA_PATH),
        'data_exists': DATA_PATH.exists(),
        'last_record': None,
        'total_records': 0,
    }

    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            status['total_records'] = len(df)
            status['last_record'] = df['timestamp'].max()
            status['date_range'] = f"{df['date'].min()} ~ {df['date'].max()}"
        except Exception as e:
            status['error'] = str(e)

    return status


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='SMP Crawler Scheduler')
    parser.add_argument('--foreground', '-f', action='store_true',
                        help='Run in foreground (default: background)')
    parser.add_argument('--run-now', '-r', action='store_true',
                        help='Run fetch immediately and exit')
    parser.add_argument('--status', '-s', action='store_true',
                        help='Show scheduler status')
    parser.add_argument('--hours', '-H', type=int, default=48,
                        help='Hours to fetch (for --run-now)')
    args = parser.parse_args()

    if args.status:
        # 상태 확인
        status = get_status()
        print("\n" + "=" * 50)
        print("SMP Scheduler Status")
        print("=" * 50)
        for key, value in status.items():
            print(f"  {key}: {value}")
        print("=" * 50)
        return

    if args.run_now:
        # 즉시 실행
        print(f"\nFetching last {args.hours} hours of data...")
        success = fetch_latest_data(hours_back=args.hours)
        if success:
            status = get_status()
            print(f"\nSuccess! Total records: {status['total_records']}")
            print(f"Last record: {status['last_record']}")
        else:
            print("\nFetch failed. Check logs for details.")
        return

    # 스케줄러 실행
    if not args.foreground:
        # 데몬 모드 (백그라운드)
        print("Starting scheduler in background...")
        print(f"Log file: {LOG_DIR / 'smp_scheduler.log'}")

        # Unix 계열에서 데몬화
        if os.name != 'nt':
            pid = os.fork()
            if pid > 0:
                print(f"Scheduler started with PID: {pid}")
                sys.exit(0)

            os.setsid()
            os.chdir("/")

            # stdin/stdout/stderr 리다이렉트
            sys.stdin = open(os.devnull, 'r')
            sys.stdout = open(LOG_DIR / 'smp_scheduler.out', 'a')
            sys.stderr = open(LOG_DIR / 'smp_scheduler.err', 'a')

    run_scheduler()


if __name__ == "__main__":
    main()
