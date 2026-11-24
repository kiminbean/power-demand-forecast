"""
제주도 전력 지도 대시보드 v4.0
================================

60hz.io 스타일의 인터랙티브 지도 기반 전력 수요 예측 대시보드
실제 EPSIS 데이터 연동

주요 기능:
1. 제주도 지도 - 발전소 위치 및 실시간 발전량
2. SMP 예측 - EPSIS 실제 데이터 + AI 예측
3. 실시간 현황 - 제주 계통 수요/공급 실데이터
4. 기상 연동 - 기상 데이터 오버레이
5. XAI 분석 - 예측 근거 설명

Usage:
    PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python streamlit run src/dashboard/app_v4.py --server.port 8504

Author: Power Demand Forecast Team
Version: 4.0.1 (EPSIS Integration)
Date: 2025-12
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote
from html import escape as html_escape
import plotly.graph_objects as go
import folium
from folium import plugins
from folium.utilities import JsCode
from branca.element import MacroElement
from jinja2 import Template
from streamlit_folium import st_folium
import json
import re
from pathlib import Path
import sys
import random
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 데이터 경로
DATA_PATH = PROJECT_ROOT / "data"
SMP_DATA_PATH = DATA_PATH / "smp"
JEJU_DATA_PATH = DATA_PATH / "jeju_extract"
PLANTS_DATA_PATH = DATA_PATH / "jeju_plants"

# SMP 모듈 임포트
try:
    from src.smp.models.smp_predictor import SMPPredictor, get_smp_predictor
    SMP_MODEL_AVAILABLE = True
except ImportError as e:
    SMP_MODEL_AVAILABLE = False
    print(f"SMP module import failed: {e}")

# EPSIS 크롤러 임포트
try:
    from src.smp.crawlers.epsis_crawler import EPSISCrawler
    EPSIS_CRAWLER_AVAILABLE = True
except ImportError as e:
    EPSIS_CRAWLER_AVAILABLE = False
    print(f"EPSIS crawler import failed: {e}")

# 제주 실시간 크롤러 임포트
try:
    from tools.crawlers.jeju_realtime_crawler import JejuRealtimeCrawler, JejuRealtimeData
    JEJU_REALTIME_AVAILABLE = True
except ImportError as e:
    JEJU_REALTIME_AVAILABLE = False
    print(f"Jeju realtime crawler import failed: {e}")


# ============================================================================
# Alert History System (v4.0.2)
# ============================================================================

ALERT_HISTORY_PATH = PROJECT_ROOT / "data" / "alerts" / "alert_history.json"


class AlertHistory:
    """예비율 경보 이력 관리 클래스"""

    MAX_HISTORY = 100  # 최대 저장 이력 수

    def __init__(self, file_path: Path = ALERT_HISTORY_PATH):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._history: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        """파일에서 이력 로드"""
        if self.file_path.exists():
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save(self):
        """파일에 이력 저장"""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Alert history save failed: {e}")

    def add_alert(self, reserve_rate: float, status: str, title: str, message: str):
        """새 경보 이력 추가 (중복 방지: 같은 status가 연속되면 추가 안함)"""
        now = datetime.now()

        # 최근 경보와 같은 status면 스킵 (1분 이내)
        if self._history:
            last = self._history[0]
            last_time = datetime.fromisoformat(last['timestamp'])
            if last['status'] == status and (now - last_time).seconds < 60:
                return

        alert = {
            'timestamp': now.isoformat(),
            'reserve_rate': round(reserve_rate, 2),
            'status': status,
            'title': title,
            'message': message
        }

        self._history.insert(0, alert)

        # 최대 개수 유지
        if len(self._history) > self.MAX_HISTORY:
            self._history = self._history[:self.MAX_HISTORY]

        self._save()

    def get_recent(self, count: int = 10) -> List[Dict]:
        """최근 경보 이력 조회"""
        return self._history[:count]

    def get_stats(self) -> Dict:
        """경보 통계"""
        if not self._history:
            return {'total': 0, 'critical': 0, 'danger': 0, 'warning': 0}

        stats = {'total': len(self._history), 'critical': 0, 'danger': 0, 'warning': 0}
        for alert in self._history:
            status = alert.get('status', '')
            if status in stats:
                stats[status] += 1
        return stats

    def clear(self):
        """이력 초기화"""
        self._history = []
        self._save()


# 전역 AlertHistory 인스턴스
@st.cache_resource
def get_alert_history() -> AlertHistory:
    """AlertHistory 싱글톤 인스턴스 반환"""
    return AlertHistory()


# ============================================================================
# Email Notification System (v4.0.3)
# ============================================================================

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(PROJECT_ROOT / ".env")

EMAIL_LOG_PATH = PROJECT_ROOT / "data" / "alerts" / "email_log.json"


class EmailNotifier:
    """이메일 알림 발송 클래스 (위험 경보용)"""

    # Rate limiting: 같은 상태의 이메일은 5분 내 재발송 방지
    RATE_LIMIT_MINUTES = 5

    def __init__(self):
        # SMTP 설정 로드
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER", "")
        self.smtp_password = os.getenv("SMTP_PASSWORD", "")
        self.sender_email = os.getenv("ALERT_SENDER_EMAIL", self.smtp_user)
        self.recipient_emails = self._parse_recipients(os.getenv("ALERT_RECIPIENT_EMAILS", ""))
        self.enabled = os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true"

        # 이메일 발송 로그 (rate limiting용)
        self.log_path = EMAIL_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._email_log: List[Dict] = self._load_log()

    def _parse_recipients(self, recipients_str: str) -> List[str]:
        """콤마로 구분된 이메일 주소 파싱"""
        if not recipients_str:
            return []
        return [email.strip() for email in recipients_str.split(",") if email.strip()]

    def _load_log(self) -> List[Dict]:
        """이메일 발송 로그 로드"""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_log(self):
        """이메일 발송 로그 저장"""
        try:
            # 최근 100개만 유지
            self._email_log = self._email_log[-100:]
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self._email_log, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Email log save failed: {e}")

    def _can_send(self, alert_status: str) -> bool:
        """Rate limiting 체크: 같은 status의 이메일이 최근 N분 내 발송되었는지 확인"""
        if not self._email_log:
            return True

        now = datetime.now()
        cutoff = now - timedelta(minutes=self.RATE_LIMIT_MINUTES)

        for log_entry in reversed(self._email_log):
            log_time = datetime.fromisoformat(log_entry['timestamp'])
            if log_time < cutoff:
                break
            if log_entry['status'] == alert_status:
                return False
        return True

    def _log_email(self, status: str, recipients: List[str], success: bool, error: str = None):
        """이메일 발송 로그 기록"""
        self._email_log.append({
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'recipients': recipients,
            'success': success,
            'error': error
        })
        self._save_log()

    def is_configured(self) -> bool:
        """이메일 설정이 완료되었는지 확인"""
        return bool(
            self.enabled and
            self.smtp_user and
            self.smtp_password and
            self.recipient_emails
        )

    def send_critical_alert(
        self,
        reserve_rate: float,
        status: str,
        title: str,
        message: str,
        power_data: Dict = None
    ) -> Tuple[bool, str]:
        """
        위험 경보 이메일 발송

        Args:
            reserve_rate: 현재 예비율 (%)
            status: 경보 상태 (critical, danger, warning)
            title: 경보 제목
            message: 경보 메시지
            power_data: 추가 전력 데이터 (선택)

        Returns:
            (성공 여부, 메시지)
        """
        # 설정 확인
        if not self.is_configured():
            return False, "Email notification not configured"

        # Critical 경보만 이메일 발송 (옵션으로 danger도 포함 가능)
        if status not in ["critical"]:
            return False, f"Email only sent for critical alerts (current: {status})"

        # Rate limiting 체크
        if not self._can_send(status):
            return False, f"Rate limited: {status} email sent within last {self.RATE_LIMIT_MINUTES} minutes"

        # 이메일 내용 구성
        subject = f"🚨 [제주 전력] {title} - 예비율 {reserve_rate:.1f}%"

        # HTML 이메일 본문
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Pretendard Variable', Pretendard, -apple-system, BlinkMacSystemFont, 'Apple SD Gothic Neo', 'Segoe UI', 'Malgun Gothic', sans-serif; }}
                .alert-box {{
                    background-color: #ff4444;
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 10px 0;
                }}
                .info-table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                .info-table th, .info-table td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                .info-table th {{ background-color: #333; color: white; }}
                .critical {{ color: #ff4444; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="alert-box">
                <h1>🚨 {title}</h1>
                <p style="font-size: 18px;">{message}</p>
                <p style="font-size: 24px; font-weight: bold;">예비율: {reserve_rate:.1f}%</p>
            </div>

            <h2>전력 수급 현황</h2>
            <table class="info-table">
                <tr>
                    <th>항목</th>
                    <th>값</th>
                </tr>
                <tr>
                    <td>예비율</td>
                    <td class="critical">{reserve_rate:.1f}%</td>
                </tr>
        """

        if power_data:
            html_body += f"""
                <tr>
                    <td>현재 수요</td>
                    <td>{power_data.get('demand', 'N/A')} MW</td>
                </tr>
                <tr>
                    <td>공급 용량</td>
                    <td>{power_data.get('supply', 'N/A')} MW</td>
                </tr>
                <tr>
                    <td>예비력</td>
                    <td>{power_data.get('reserve', 'N/A')} MW</td>
                </tr>
            """

        html_body += f"""
            </table>

            <p style="margin-top: 20px; color: #666;">
                발송 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                제주 전력 수급 모니터링 시스템
            </p>
        </body>
        </html>
        """

        # 이메일 발송
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(self.recipient_emails)

            # HTML 본문 추가
            msg.attach(MIMEText(html_body, 'html'))

            # SMTP 연결 및 발송
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_password)
                server.sendmail(
                    self.sender_email,
                    self.recipient_emails,
                    msg.as_string()
                )

            # 성공 로그
            self._log_email(status, self.recipient_emails, True)
            return True, f"Email sent to {len(self.recipient_emails)} recipients"

        except smtplib.SMTPAuthenticationError as e:
            error_msg = f"SMTP authentication failed: {e}"
            self._log_email(status, self.recipient_emails, False, error_msg)
            return False, error_msg
        except smtplib.SMTPException as e:
            error_msg = f"SMTP error: {e}"
            self._log_email(status, self.recipient_emails, False, error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Email send failed: {e}"
            self._log_email(status, self.recipient_emails, False, error_msg)
            return False, error_msg

    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        """최근 이메일 발송 로그 조회"""
        return self._email_log[-count:]


# 전역 EmailNotifier 인스턴스
@st.cache_resource
def get_email_notifier() -> EmailNotifier:
    """EmailNotifier 싱글톤 인스턴스 반환"""
    return EmailNotifier()


# ============================================================================
# Slack Notification System (v4.0.4)
# ============================================================================

import urllib.request
import urllib.error

SLACK_LOG_PATH = PROJECT_ROOT / "data" / "alerts" / "slack_log.json"


class SlackNotifier:
    """Slack 웹훅 알림 발송 클래스"""

    # Rate limiting: 같은 상태의 알림은 5분 내 재발송 방지
    RATE_LIMIT_MINUTES = 5

    def __init__(self):
        self.webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
        self.channel = os.getenv("SLACK_CHANNEL", "#alerts")
        self.enabled = os.getenv("SLACK_ALERTS_ENABLED", "false").lower() == "true"

        # 발송 로그 (rate limiting용)
        self.log_path = SLACK_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._slack_log: List[Dict] = self._load_log()

    def _load_log(self) -> List[Dict]:
        """파일에서 로그 로드"""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []

    def _save_log(self):
        """파일에 로그 저장"""
        try:
            self._slack_log = self._slack_log[-100:]
            with open(self.log_path, 'w', encoding='utf-8') as f:
                json.dump(self._slack_log, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"Slack log save failed: {e}")

    def _can_send(self, alert_status: str) -> bool:
        """Rate limiting 체크"""
        if not self._slack_log:
            return True

        now = datetime.now()
        cutoff = now - timedelta(minutes=self.RATE_LIMIT_MINUTES)

        for log_entry in reversed(self._slack_log):
            log_time = datetime.fromisoformat(log_entry['timestamp'])
            if log_time < cutoff:
                break
            if log_entry['status'] == alert_status:
                return False
        return True

    def _log_message(self, status: str, success: bool, error: str = None):
        """Slack 발송 로그 기록"""
        self._slack_log.append({
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'success': success,
            'error': error
        })
        self._save_log()

    def is_configured(self) -> bool:
        """Slack 설정이 완료되었는지 확인"""
        return bool(self.enabled and self.webhook_url)

    def send_alert(
        self,
        reserve_rate: float,
        status: str,
        title: str,
        message: str,
        power_data: Dict = None
    ) -> Tuple[bool, str]:
        """
        Slack 알림 발송

        Args:
            reserve_rate: 현재 예비율 (%)
            status: 경보 상태 (critical, danger, warning)
            title: 경보 제목
            message: 경보 메시지
            power_data: 추가 전력 데이터 (선택)

        Returns:
            (성공 여부, 메시지)
        """
        # 설정 확인
        if not self.is_configured():
            return False, "Slack notification not configured"

        # Rate limiting 체크
        if not self._can_send(status):
            return False, f"Rate limited: {status} alert sent within last {self.RATE_LIMIT_MINUTES} minutes"

        # 상태별 이모지 및 색상
        status_config = {
            "critical": {"emoji": "🚨", "color": "#ff0000"},
            "danger": {"emoji": "⚠️", "color": "#ff8800"},
            "warning": {"emoji": "📢", "color": "#ffcc00"},
        }
        config = status_config.get(status, {"emoji": "ℹ️", "color": "#0088ff"})

        # Slack Block Kit 메시지 구성
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{config['emoji']} {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{message}*"
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*예비율:*\n{reserve_rate:.1f}%"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*상태:*\n{status.upper()}"
                    }
                ]
            }
        ]

        # 전력 데이터 추가
        if power_data:
            blocks.append({
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*현재 수요:*\n{power_data.get('demand', 'N/A')} MW"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*공급 용량:*\n{power_data.get('supply', 'N/A')} MW"
                    }
                ]
            })

        # 타임스탬프 추가
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 제주 전력 수급 모니터링"
                }
            ]
        })

        # Slack 페이로드
        payload = {
            "channel": self.channel,
            "username": "제주 전력 알림",
            "icon_emoji": ":zap:",
            "attachments": [
                {
                    "color": config['color'],
                    "blocks": blocks
                }
            ]
        }

        # 웹훅 전송
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    self._log_message(status, True)
                    return True, "Slack message sent successfully"
                else:
                    error_msg = f"Slack API returned status {response.status}"
                    self._log_message(status, False, error_msg)
                    return False, error_msg

        except urllib.error.HTTPError as e:
            error_msg = f"Slack HTTP error: {e.code} {e.reason}"
            self._log_message(status, False, error_msg)
            return False, error_msg
        except urllib.error.URLError as e:
            error_msg = f"Slack URL error: {e.reason}"
            self._log_message(status, False, error_msg)
            return False, error_msg
        except Exception as e:
            error_msg = f"Slack send failed: {e}"
            self._log_message(status, False, error_msg)
            return False, error_msg

    def get_recent_logs(self, count: int = 10) -> List[Dict]:
        """최근 Slack 발송 로그 조회"""
        return self._slack_log[-count:]


# 전역 SlackNotifier 인스턴스
@st.cache_resource
def get_slack_notifier() -> SlackNotifier:
    """SlackNotifier 싱글톤 인스턴스 반환"""
    return SlackNotifier()


# ============================================================================
# 페이지 설정
# ============================================================================

# 페이지 아이콘은 Material 아이콘이다(이모지는 OS 글꼴마다 모양이 달라 화면 계층에서 쓰지 않는다, 명세 3.8절).
# 화면 CSS는 모듈 최상위 st.markdown이 아니라 main() 첫 줄의 st.html(_css_dashboard())로 넣는다(명세 5.1·5.15절).
st.set_page_config(
    page_title="제주 전력 지도",
    page_icon=":material/map:",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================================
# EPSIS 실제 데이터 로딩 함수
# ============================================================================

def _fix_timestamp_24h(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 24:00 timestamp format (convert to 00:00 of next day)"""
    if 'timestamp' in df.columns:
        ts = df['timestamp'].astype(str)
        is_24h = ts.str.contains(' 24:00')
        # 24:00 → 같은 날 00:00이 아니라 다음 날 00:00으로 변환
        ts = ts.str.replace(' 24:00', ' 00:00')
        df['timestamp'] = pd.to_datetime(ts, errors='coerce')
        df.loc[is_24h, 'timestamp'] = df.loc[is_24h, 'timestamp'] + pd.Timedelta(days=1)
        df = df.dropna(subset=['timestamp'])
    return df


@st.cache_data(ttl=60)  # 1분 캐시 (KPX 5분 업데이트)
def fetch_jeju_realtime() -> Optional[Dict]:
    """KPX 제주 실시간 전력수급 데이터 조회"""
    if not JEJU_REALTIME_AVAILABLE:
        return None

    try:
        with JejuRealtimeCrawler(timeout=10) as crawler:
            data = crawler.fetch_realtime()
            if data:
                return data.to_dict()
    except Exception as e:
        print(f"KPX realtime fetch failed: {e}")

    return None


@st.cache_data(ttl=3600)
def load_smp_history() -> pd.DataFrame:
    """EPSIS SMP 히스토리 데이터 로드"""
    try:
        # 실제 EPSIS 데이터
        smp_file = SMP_DATA_PATH / "smp_real_epsis.csv"
        if smp_file.exists():
            df = pd.read_csv(smp_file, encoding='utf-8-sig')
            df = _fix_timestamp_24h(df)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df

        # 대체 파일
        smp_file = SMP_DATA_PATH / "smp_5years_epsis.csv"
        if smp_file.exists():
            df = pd.read_csv(smp_file, encoding='utf-8-sig')
            df = _fix_timestamp_24h(df)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            return df

    except Exception as e:
        st.warning(f"SMP 데이터 로드 실패: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_jeju_demand_data() -> pd.DataFrame:
    """제주 계통 수요 데이터 로드"""
    try:
        demand_file = JEJU_DATA_PATH / "계통수요.csv"
        if demand_file.exists():
            df = pd.read_csv(demand_file, encoding='cp949')
            # 컬럼명 정리 (날짜 + 24시간)
            df.columns = ['date'] + [f'h{i}' for i in range(1, 25)]
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        st.warning(f"수요 데이터 로드 실패: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_jeju_supply_data() -> pd.DataFrame:
    """제주 공급능력 데이터 로드"""
    try:
        supply_file = JEJU_DATA_PATH / "공급능력.csv"
        if supply_file.exists():
            df = pd.read_csv(supply_file, encoding='cp949')
            df.columns = ['date'] + [f'h{i}' for i in range(1, 25)]
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        st.warning(f"공급 데이터 로드 실패: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_jeju_reserve_data() -> pd.DataFrame:
    """제주 공급예비력 데이터 로드"""
    try:
        reserve_file = JEJU_DATA_PATH / "공급예비력.csv"
        if reserve_file.exists():
            df = pd.read_csv(reserve_file, encoding='cp949')
            df.columns = ['date'] + [f'h{i}' for i in range(1, 25)]
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        st.warning(f"예비력 데이터 로드 실패: {e}")

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_jeju_forecast_data() -> pd.DataFrame:
    """제주 예측수요 데이터 로드"""
    try:
        forecast_file = JEJU_DATA_PATH / "예측수요.csv"
        if forecast_file.exists():
            df = pd.read_csv(forecast_file, encoding='cp949')
            df.columns = ['date'] + [f'h{i}' for i in range(1, 25)]
            df['date'] = pd.to_datetime(df['date'])
            return df
    except Exception as e:
        st.warning(f"예측 데이터 로드 실패: {e}")

    return pd.DataFrame()


# ============================================================================
# 데이터 처리 함수
# ============================================================================

@st.cache_data(ttl=3600)
def load_power_plants_data() -> Dict:
    """제주도 발전소 실제 데이터 로드 (공공데이터포털 + thewindpower.net)"""
    try:
        json_file = PLANTS_DATA_PATH / "jeju_power_plants.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"발전소 데이터 로드 실패: {e}")
    return {}


@st.cache_data(ttl=60)
def get_jeju_power_plants() -> pd.DataFrame:
    """제주도 발전소 데이터 (공공데이터포털 + thewindpower.net 실제 데이터)

    데이터 출처:
    - 공공데이터포털: 제주특별자치도_풍력발전현황, 제주에너지공사_발전시설 현황
    - 한국에너지공단: 풍력기 위치정보 (WGS84 좌표)
    - The Wind Power (thewindpower.net): 풍력발전소 상세 좌표
    """

    # 실제 데이터 로드
    plants_data = load_power_plants_data()

    plants = []

    # 풍력 발전소 (실제 좌표 데이터)
    if 'wind_farms' in plants_data:
        for wf in plants_data['wind_farms']:
            if wf.get('status') == '운영중':  # 운영 중인 발전소만
                plants.append({
                    "id": wf.get('id'),
                    "name": wf.get('name'),
                    "name_en": wf.get('name_en'),
                    "type": "wind",
                    "subtype": wf.get('subtype', 'onshore'),
                    "lat": wf.get('latitude'),
                    "lng": wf.get('longitude'),
                    "capacity": wf.get('capacity_mw', 0),
                    "operator": wf.get('operator'),
                    "status": wf.get('status'),
                    "address": wf.get('address'),
                    "source": wf.get('source')
                })

    # 태양광 발전소 — 다른 앱(API JEJU_PLANTS·v7 지도)과 같은 기준: 운영중, 분산형 합산값(1,620개소 합계) 제외
    if 'solar_farms' in plants_data:
        for sf in plants_data['solar_farms']:
            if sf.get('status') != '운영중' or sf.get('subtype') == 'distributed':
                continue
            plants.append({
                "id": sf.get('id'),
                "name": sf.get('name'),
                "name_en": sf.get('name_en'),
                "type": "solar",
                "subtype": sf.get('subtype', 'ground-mounted'),
                "lat": sf.get('latitude'),
                "lng": sf.get('longitude'),
                "capacity": sf.get('capacity_mw', 0),
                "operator": sf.get('operator'),
                "status": sf.get('status'),
                "address": sf.get('address'),
                "source": sf.get('source')
            })

    # ESS 설비 — 운영중만 (건설중 제외, 다른 앱과 같은 기준)
    if 'ess_facilities' in plants_data:
        for ess in plants_data['ess_facilities']:
            if ess.get('status') != '운영중':
                continue
            plants.append({
                "id": ess.get('id'),
                "name": ess.get('name'),
                "name_en": ess.get('name_en'),
                "type": "ess",
                "subtype": ess.get('subtype', 'utility-scale'),
                "lat": ess.get('latitude'),
                "lng": ess.get('longitude'),
                "capacity": ess.get('capacity_mw', 0),
                "capacity_mwh": ess.get('capacity_mwh', 0),
                "operator": ess.get('operator'),
                "status": ess.get('status'),
                "address": ess.get('address'),
                "purpose": ess.get('purpose'),
                "source": ess.get('source')
            })

    # 화력 발전소 (이 앱에만 표시; 운영중만)
    if 'thermal_plants' in plants_data:
        for tp in plants_data['thermal_plants']:
            if tp.get('status') != '운영중':
                continue
            plants.append({
                "id": tp.get('id'),
                "name": tp.get('name'),
                "name_en": tp.get('name_en'),
                "type": "thermal",
                "subtype": tp.get('subtype', 'combined-cycle'),
                "lat": tp.get('latitude'),
                "lng": tp.get('longitude'),
                "capacity": tp.get('capacity_mw', 0),
                "operator": tp.get('operator'),
                "status": tp.get('status'),
                "address": tp.get('address'),
                "fuel_type": tp.get('fuel_type'),
                "source": tp.get('source')
            })

    # 데이터가 없는 경우 fallback
    if not plants:
        plants = [
            {"id": "fallback_1", "name": "한경풍력발전단지", "type": "wind", "lat": 33.339417, "lng": 126.169222, "capacity": 21.0, "status": "운영중"},
            {"id": "fallback_2", "name": "가시리풍력발전단지", "type": "wind", "lat": 33.3576, "lng": 126.7461, "capacity": 30.0, "status": "운영중"},
            {"id": "fallback_3", "name": "제주ESS", "type": "ess", "lat": 33.5100, "lng": 126.5400, "capacity": 30.0, "status": "운영중"},
        ]

    df = pd.DataFrame(plants)

    # 현재 발전량 계산 (KPX 실시간 데이터 기반)
    hour = datetime.now().hour

    # KPX 실시간 데이터로 총 발전량 가져오기
    realtime_data = fetch_jeju_realtime()
    if realtime_data:
        # 실제 수요 기반 발전량 분배
        total_demand = realtime_data.get('current_demand', 800)

        # 유형별 설비용량 합계
        type_capacities = df.groupby('type')['capacity'].sum().to_dict()
        total_wind_cap = type_capacities.get('wind', 0)
        total_solar_cap = type_capacities.get('solar', 0)
        total_thermal_cap = type_capacities.get('thermal', 0)
        total_ess_cap = type_capacities.get('ess', 0)

        # 시간대별 재생에너지 출력 추정 (실제 KPX 데이터 기반 스케일링)
        solar_ratio = np.sin(np.pi * max(0, hour - 6) / 12) if 6 <= hour <= 18 else 0
        wind_ratio = 0.5 + 0.2 * np.sin(np.pi * hour / 24)

        # 발전량 분배 (총 수요 기준)
        total_solar_gen = min(total_solar_cap * solar_ratio * 0.85, total_demand * 0.25)
        total_wind_gen = min(total_wind_cap * wind_ratio * 0.7, total_demand * 0.35)
        total_thermal_gen = max(0, total_demand - total_solar_gen - total_wind_gen)
        total_ess_gen = (realtime_data.get('supply_capacity', total_demand) - total_demand) * 0.1

        # 각 발전소에 비례 배분
        def distribute_generation(row):
            capacity = row.get('capacity', 0)
            plant_type = row.get('type', '')
            type_total_cap = type_capacities.get(plant_type, 1)

            if capacity <= 0 or type_total_cap <= 0:
                return 0

            ratio = capacity / type_total_cap

            if plant_type == 'solar':
                return total_solar_gen * ratio * random.uniform(0.9, 1.1)
            elif plant_type == 'wind':
                return total_wind_gen * ratio * random.uniform(0.85, 1.15)
            elif plant_type == 'thermal':
                return total_thermal_gen * ratio * random.uniform(0.95, 1.05)
            else:  # ESS
                return total_ess_gen * ratio * random.uniform(0.8, 1.2)

        df['generation'] = df.apply(distribute_generation, axis=1)
        df['data_source'] = 'KPX 실시간'
    else:
        # 폴백: 기존 시뮬레이션 방식
        def calculate_generation(row):
            capacity = row.get('capacity', 0)
            if capacity <= 0:
                return 0

            plant_type = row.get('type', '')

            if plant_type == 'solar':
                if 6 <= hour <= 18:
                    solar_factor = np.sin(np.pi * (hour - 6) / 12)
                    return capacity * solar_factor * random.uniform(0.7, 1.0)
                return 0
            elif plant_type == 'wind':
                base_factor = 0.5 + 0.2 * np.sin(np.pi * hour / 12)
                return capacity * base_factor * random.uniform(0.6, 1.0)
            elif plant_type == 'thermal':
                return capacity * random.uniform(0.7, 0.95)
            else:  # ESS
                if 10 <= hour <= 15:
                    return -capacity * random.uniform(0.3, 0.7)
                elif 18 <= hour <= 21:
                    return capacity * random.uniform(0.5, 0.9)
                else:
                    return capacity * random.uniform(-0.2, 0.3)

        df['generation'] = df.apply(calculate_generation, axis=1)
        df['data_source'] = '시뮬레이션'

    # 풍력·태양광·ESS 발전량은 v6·v7·모바일과 같은 API 서버 시뮬레이션 값을 ID로 맞춰 쓴다 (화력만 자체 추정).
    api_resources = fetch_backend_json("/api/v1/resources")
    if isinstance(api_resources, list):
        api_output = {
            r.get("id"): float(r["current_output"])
            for r in api_resources
            if isinstance(r, dict) and _finite_number(r.get("current_output"))
        }
        matched = df['id'].isin(api_output.keys())
        df.loc[matched, 'generation'] = df.loc[matched, 'id'].map(api_output)
        df.loc[matched, 'data_source'] = 'API 서버 시뮬레이션'

    df['utilization'] = df.apply(
        lambda row: min(max(abs(row['generation']) / row['capacity'] * 100, 0), 100) if row['capacity'] > 0 else 0,
        axis=1
    )

    return df


# 다른 공개 앱(v5·v6·v7·모바일)과 같은 값을 보여 주도록 재생에너지 비율·계통 주파수·기상·SMP 예측은
# FastAPI 서버 값을 쓴다. 서버에 연결하지 못하면 기존 추정·시뮬레이션 값을 쓰고 그렇게 표시한다.
import http.client  # noqa: E402  (fetch_backend_json 예외 처리용)

BACKEND_API_URL = os.getenv("POWERDEMAND_API_URL", "http://127.0.0.1:8001").strip().rstrip("/")
BACKEND_TIMEOUT_SECONDS = 5


@st.cache_data(ttl=60)
def fetch_backend_json(path: str) -> Optional[Dict]:
    """API 서버 JSON 조회. 실패하면 None (호출 측이 출처를 시뮬레이션으로 표시)."""
    try:
        with urllib.request.urlopen(f"{BACKEND_API_URL}{path}", timeout=BACKEND_TIMEOUT_SECONDS) as response:
            if response.status != 200:
                return None
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError):
        # IncompleteRead·BadStatusLine은 OSError가 아니므로 따로 잡아야 페이지 전체가 깨지지 않는다.
        return None


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def _backend_kpi_overrides() -> Dict:
    """API 서버의 수요·예비율·재생에너지 비율(섬 전체 ML 추정)·계통 주파수(시뮬레이션)를 그대로 쓴다.

    같은 KPX 크롤러라도 이 앱과 API 서버가 서로 다른 시점에 조회하면 값이 어긋나므로
    (2026-11-05 16:57: 713.0 MW·122.3% vs 712.0 MW·124.0%) API 서버 값을 단일 원천으로 쓴다.
    """
    kpis = fetch_backend_json("/api/v1/dashboard/kpis")
    if not isinstance(kpis, dict):
        return {}
    overrides = {}
    if _finite_number(kpis.get("current_demand_mw")) and _finite_number(kpis.get("supply_reserve_rate")):
        overrides["demand"] = round(float(kpis["current_demand_mw"]), 1)
        overrides["reserve_rate"] = round(float(kpis["supply_reserve_rate"]), 1)
        # 파일·시뮬레이션 대체 경로에서도 출처 라벨이 실제 값의 출처와 맞도록 함께 바꾼다.
        overrides["data_source"] = "KPX 실시간 (API 서버)"
        overrides["data_date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    if _finite_number(kpis.get("renewable_ratio_pct")):
        overrides["renewable_ratio"] = round(float(kpis["renewable_ratio_pct"]), 1)
        # 같은 화면의 공급 구성(로컬 추정)과 계산 근거가 다르므로 출처를 밝힌다.
        overrides["renewable_source"] = "섬 전체 태양광+풍력 ML 추정치 (API 서버)"
    if _finite_number(kpis.get("grid_frequency")):
        overrides["frequency"] = round(float(kpis["grid_frequency"]), 2)
    return overrides


@st.cache_data(ttl=60)
def get_current_power_status() -> Dict:
    """현재 전력 수급 현황 (KPX 실시간 데이터 우선)"""
    hour = datetime.now().hour
    today = datetime.now().date()

    # 1순위: KPX 실시간 데이터
    realtime_data = fetch_jeju_realtime()
    if realtime_data:
        demand = realtime_data.get('current_demand', 800)
        total_supply = realtime_data.get('supply_capacity', demand * 1.15)
        # reserve_rate 사용 (supply_reserve는 MW 값이므로 사용하지 않음)
        reserve_rate = realtime_data.get('reserve_rate', 15.0)
        operation_reserve = realtime_data.get('operation_reserve', 0)

        # 재생에너지 비율 추정 (실시간 데이터에서 가져오거나 추정)
        # KPX에서 제공하는 경우 사용, 아니면 시간대 기반 추정
        solar = 150 * np.sin(np.pi * max(0, hour - 6) / 12) if 6 <= hour <= 18 else 0
        solar = min(solar * 1.5, 300)  # 최대 태양광 출력 제한
        wind = 200 * (0.5 + 0.3 * np.sin(np.pi * hour / 24))  # 풍력 출력 추정
        thermal = max(0, demand - solar - wind - 30)
        ess = (total_supply - demand) * 0.1 if total_supply > demand else -30

        renewable_ratio = ((solar + wind) / demand * 100) if demand > 0 else 0

        data_source = "KPX 실시간"
        data_date = datetime.now().strftime("%Y-%m-%d %H:%M")

        return {
            "demand": round(demand, 1),
            "supply": {
                "solar": round(max(0, solar), 1),
                "wind": round(wind, 1),
                "thermal": round(thermal, 1),
                "ess": round(ess, 1),
            },
            "total_supply": round(total_supply, 1),
            "reserve_rate": round(reserve_rate, 1),
            "operation_reserve": round(operation_reserve, 1),
            "frequency": round(60 + random.uniform(-0.01, 0.01), 3),
            "renewable_ratio": round(renewable_ratio, 1),
            "data_source": data_source,
            "data_date": data_date,
            **_backend_kpi_overrides(),
        }

    # 2순위: EPSIS 파일 데이터
    demand_df = load_jeju_demand_data()
    supply_df = load_jeju_supply_data()
    reserve_df = load_jeju_reserve_data()

    if not demand_df.empty:
        latest_row = demand_df.iloc[-1]
        hour_col = f'h{hour if hour > 0 else 24}'

        if hour_col in latest_row:
            demand = float(latest_row[hour_col])
        else:
            demand = float(latest_row[[c for c in latest_row.index if c.startswith('h')]].mean())

        # 공급능력
        if not supply_df.empty:
            supply_row = supply_df.iloc[-1]
            if hour_col in supply_row:
                total_supply = float(supply_row[hour_col])
            else:
                total_supply = demand * 1.15
        else:
            total_supply = demand * 1.15

        # 예비력
        if not reserve_df.empty:
            reserve_row = reserve_df.iloc[-1]
            if hour_col in reserve_row:
                reserve = float(reserve_row[hour_col])
                reserve_rate = (reserve / demand) * 100 if demand > 0 else 0
            else:
                reserve_rate = 15.0
        else:
            reserve_rate = ((total_supply - demand) / demand) * 100 if demand > 0 else 0

        data_source = "EPSIS 파일"
        data_date = str(latest_row['date'])[:10] if 'date' in latest_row else "최신"

    else:
        # 3순위: 시뮬레이션 폴백
        base_demand = {
            0: 680, 1: 650, 2: 620, 3: 600, 4: 595, 5: 610,
            6: 650, 7: 720, 8: 800, 9: 860, 10: 900, 11: 920,
            12: 910, 13: 915, 14: 930, 15: 920, 16: 900, 17: 890,
            18: 920, 19: 950, 20: 920, 21: 870, 22: 800, 23: 730
        }
        demand = base_demand[hour] * random.uniform(0.95, 1.05)
        total_supply = demand * 1.15
        reserve_rate = 15.0
        data_source = "시뮬레이션"
        data_date = str(today)

    # 재생에너지 발전량 추정
    solar = 150 * np.sin(np.pi * max(0, hour - 6) / 12) if 6 <= hour <= 18 else 0
    solar *= random.uniform(0.7, 1.0)
    wind = 200 * random.uniform(0.3, 0.8)
    thermal = max(0, demand - solar - wind - 50)
    ess = 50 * random.uniform(-0.5, 0.5)

    renewable_ratio = ((solar + wind) / demand * 100) if demand > 0 else 0

    return {
        "demand": round(demand, 1),
        "supply": {
            "solar": round(max(0, solar), 1),
            "wind": round(wind, 1),
            "thermal": round(thermal, 1),
            "ess": round(ess, 1),
        },
        "total_supply": round(total_supply, 1),
        "reserve_rate": round(reserve_rate, 1),
        "operation_reserve": 0,
        "frequency": round(60 + random.uniform(-0.02, 0.02), 3),
        "renewable_ratio": round(renewable_ratio, 1),
        "data_source": data_source,
        "data_date": data_date,
        **_backend_kpi_overrides(),
    }


def _format_observation_age(data_date: object, now: Optional[datetime] = None) -> Tuple[str, str]:
    """수요 데이터의 관측 시각과 경과 시간 텍스트를 계산한다.

    EPSIS 파일 데이터는 관측 시점이 며칠 과거일 수 있어,
    '현재 수요' 카드에 관측 시각과 경과 시간을 함께 표시해야
    데이터 신선도가 사용자에게 드러난다.

    Args:
        data_date: power_status['data_date'] 값 (문자열 또는 날짜).
        now: 경과 시간 계산 기준 시각 (기본: 현재 시각, 테스트 주입용).

    Returns:
        (관측 시각 텍스트, 경과 시간 텍스트). 파싱 불가 시 경과는 'N/A'.
    """
    if now is None:
        now = datetime.now()
    observed_text = str(data_date)
    try:
        observed = pd.to_datetime(str(data_date))
    except (ValueError, TypeError):
        return observed_text, "N/A"
    if pd.isna(observed):
        return observed_text, "N/A"
    elapsed = now - observed.to_pydatetime()
    if elapsed.total_seconds() < 0:
        elapsed = timedelta(0)
    total_hours = int(elapsed.total_seconds() // 3600)
    days, hours = divmod(total_hours, 24)
    # 날짜 단위 파일 데이터는 'N일 N시간 경과'로, 당일은 'N시간/N분 경과'로 표시
    if days > 0:
        elapsed_text = f"{days}일 {hours}시간 경과"
    elif hours > 0:
        elapsed_text = f"{hours}시간 경과"
    else:
        minutes = max(1, int(elapsed.total_seconds() // 60))
        elapsed_text = f"{minutes}분 경과"
    return observed_text, elapsed_text


@st.cache_data(ttl=60)
def get_smp_data() -> Dict:
    """SMP 데이터 (EPSIS 실제 데이터 기반)"""
    current_hour = datetime.now().hour

    # 실제 SMP 데이터 로드
    smp_df = load_smp_history()

    if not smp_df.empty:
        # smp_jeju가 0이면 smp_mainland로 대체 (데이터 품질 이슈 대응)
        if 'smp_mainland' in smp_df.columns:
            smp_df['smp_jeju'] = smp_df.apply(
                lambda row: row['smp_mainland'] if row['smp_jeju'] == 0 or pd.isna(row['smp_jeju']) else row['smp_jeju'],
                axis=1
            )

        # 최근 데이터 사용
        recent_df = smp_df.tail(48)  # 최근 48시간

        # 거래시간 H는 (H-1)시~H시 구간이다. 지금 적용 중인 구간은 now.hour+1이며,
        # 파일에는 오늘 남은 시간의 하루전 가격도 이미 들어 있으므로 구간 끝이 지금 구간 끝 이하인 행만 쓴다.
        # (이전에는 now.hour 행을 써서 이미 끝난 구간 가격을 보여 줬다.)
        now = datetime.now()
        current_end = pd.Timestamp(now.date()) + pd.Timedelta(hours=current_hour + 1)
        interval_end = pd.to_datetime(smp_df['date']).dt.normalize() + pd.to_timedelta(smp_df['hour'], unit='h')
        published = smp_df.assign(interval_end=interval_end)
        published = published[(published['interval_end'] <= current_end) & (published['smp_jeju'] > 0)]
        published = published.sort_values('interval_end')
        if not published.empty:
            latest = published.iloc[-1]
            current_smp = float(latest['smp_jeju'])
            trade_day = latest['interval_end'] - pd.Timedelta(hours=1)
            trade_label = f"{trade_day:%Y-%m-%d} {int(latest['hour']):02d}h"
            # 바로 직전 구간 값이 있을 때만 변화율을 만든다
            previous = published[published['interval_end'] == latest['interval_end'] - pd.Timedelta(hours=1)]
            prev_smp = float(previous['smp_jeju'].iloc[-1]) if not previous.empty else None
        else:
            current_smp = float(recent_df[recent_df['smp_jeju'] > 0]['smp_jeju'].mean())
            trade_label = None
            prev_smp = None

        # 시간대별 평균 SMP (0이 아닌 값만)
        valid_smp = smp_df[smp_df['smp_jeju'] > 0]
        hourly_avg = valid_smp.groupby('hour')['smp_jeju'].mean().to_dict()

        # 24시간 예측 (실제 패턴 + 변동)
        predictions = []
        for h in range(24):
            future_hour = (current_hour + h) % 24
            if future_hour == 0:
                future_hour = 24

            base_pred = hourly_avg.get(future_hour, 100)

            # 최근 트렌드 반영
            if h < 6:  # 가까운 시간은 더 정확
                noise = random.uniform(0.95, 1.05)
            else:
                noise = random.uniform(0.85, 1.15)

            pred = base_pred * noise
            q10 = pred * 0.85
            q90 = pred * 1.15

            predictions.append({
                "hour": future_hour,
                "time": (datetime.now() + timedelta(hours=h)).strftime("%H:00"),
                "smp": round(pred, 1),
                "q10": round(q10, 1),
                "q90": round(q90, 1),
            })

        # 통계 (마지막 날짜 기준 1일 통계 - '일평균/일최고/일최저' 라벨과 일치)
        latest_date = smp_df['date'].max()
        latest_day = smp_df[smp_df['date'] == latest_date]
        latest_valid = latest_day[latest_day['smp_jeju'] > 0]
        stats_df = latest_valid if not latest_valid.empty else valid_smp
        daily_avg = float(stats_df['smp_jeju'].mean())
        daily_max = float(stats_df['smp_jeju'].max())
        daily_min = float(stats_df['smp_jeju'].min())

        data_source = "EPSIS 실데이터"
        data_range = f"{smp_df['date'].min().strftime('%Y-%m-%d')} ~ {smp_df['date'].max().strftime('%Y-%m-%d')}"
        record_count = len(smp_df)

    else:
        # 폴백: 시뮬레이션 데이터
        base_smp = {
            0: 85, 1: 80, 2: 78, 3: 75, 4: 76, 5: 80,
            6: 95, 7: 110, 8: 125, 9: 135, 10: 140, 11: 145,
            12: 138, 13: 140, 14: 150, 15: 145, 16: 140, 17: 135,
            18: 145, 19: 160, 20: 155, 21: 140, 22: 120, 23: 100
        }

        current_smp = base_smp[current_hour] * random.uniform(0.9, 1.1)
        prev_smp = base_smp[(current_hour - 1) % 24] * random.uniform(0.9, 1.1)
        trade_label = None

        predictions = []
        for h in range(24):
            future_hour = (current_hour + h) % 24
            pred = base_smp[future_hour] * random.uniform(0.85, 1.15)
            q10 = pred * 0.8
            q90 = pred * 1.2
            predictions.append({
                "hour": future_hour,
                "time": (datetime.now() + timedelta(hours=h)).strftime("%H:00"),
                "smp": round(pred, 1),
                "q10": round(q10, 1),
                "q90": round(q90, 1),
            })

        daily_avg = sum(base_smp.values()) / 24
        daily_max = max(base_smp.values()) * 1.1
        daily_min = min(base_smp.values()) * 0.9
        data_source = "시뮬레이션"
        data_range = "N/A"
        record_count = 0

    # SMP 변화 — 이전 실측값이 없으면 가짜 변화율을 만들지 않는다
    if prev_smp is not None and prev_smp > 0:
        change = current_smp - prev_smp
        change_pct = change / prev_smp * 100
    else:
        change = None
        change_pct = None

    return {
        "current": round(current_smp, 1),
        "change": round(change, 1) if change is not None else None,
        "change_pct": round(change_pct, 1) if change_pct is not None else None,
        "prev_available": prev_smp is not None and prev_smp > 0,
        "trade_label": trade_label,
        "predictions": predictions,
        "daily_avg": round(daily_avg, 1),
        "daily_max": round(daily_max, 1),
        "daily_min": round(daily_min, 1),
        "data_source": data_source,
        "data_range": data_range,
        "record_count": record_count,
    }


@st.cache_data(ttl=300)
def get_weather_data() -> Dict:
    """기상 데이터 — API 서버(기상청 초단기실황)가 주는 기온·풍속·습도를 우선 쓴다."""
    hour = datetime.now().hour

    weather = {
        "temperature": round(10 + 8 * np.sin(np.pi * (hour - 6) / 12) + random.uniform(-2, 2), 1),
        "humidity": round(60 + random.uniform(-15, 15), 0),
        "wind_speed": round(5 + random.uniform(0, 10), 1),
        "wind_direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        "cloud_cover": round(random.uniform(0, 80), 0),
        "solar_radiation": round(800 * np.sin(np.pi * max(0, hour - 6) / 12) if 6 <= hour <= 18 else 0, 0),
        "precipitation": round(random.uniform(0, 2), 1) if random.random() > 0.7 else 0,
        "source": "simulation",
    }
    kpis = fetch_backend_json("/api/v1/dashboard/kpis")
    api_weather = kpis.get("weather") if isinstance(kpis, dict) else None
    if isinstance(api_weather, dict) and all(
        _finite_number(api_weather.get(key)) for key in ("temperature", "wind_speed", "humidity")
    ):
        weather.update({
            "temperature": round(float(api_weather["temperature"]), 1),
            "wind_speed": round(float(api_weather["wind_speed"]), 1),
            "humidity": round(float(api_weather["humidity"]), 0),
            "condition": api_weather.get("condition"),
            "source": "kma",
        })
    return weather


@st.cache_data(ttl=60)
def get_backend_smp_predictions() -> Optional[Dict]:
    """다른 앱과 같은 /smp-forecast 예측(오늘 거래시간 1~24)을 차트 형식으로 변환한다."""
    forecast = fetch_backend_json("/api/v1/smp-forecast")
    if not isinstance(forecast, dict):
        return None
    series = [forecast.get(key) for key in ("q10", "q50", "q90")]
    if not all(isinstance(s, list) and len(s) == 24 and all(_finite_number(v) for v in s) for s in series):
        return None
    q10, q50, q90 = series
    # 인덱스 i는 i시~i+1시 구간(거래시간 i+1)이며 다른 앱과 같이 시작 시각으로 표기한다.
    predictions = [
        {"hour": i + 1, "time": f"{i:02d}:00", "smp": round(q50[i], 1), "q10": round(q10[i], 1), "q90": round(q90[i], 1)}
        for i in range(24)
    ]
    return {
        "predictions": predictions,
        "current_index": datetime.now().hour,
        "forecast_date": forecast.get("forecast_date"),
        "forecast_model": forecast.get("model_used"),
    }


# ============================================================================
# 지도 생성 함수
# ============================================================================

def create_jeju_map(plants_df: pd.DataFrame, show_heatmap: bool = False) -> folium.Map:
    """제주 발전소 지도(명세 6.7절). 등록부의 위치·설비만 기호로 보이고, 시뮬레이션 값은 팝업에서 '추정'으로만 보인다.

    - 타일: OSM 표준 타일 하나. CARTO 무키 타일은 "API KEY REQUIRED" 워터마크만 준다(J13, 2026-11-24 확인).
      테마는 타일 칸에 CSS 필터로 입히고, 라이트·다크 판단은 iframe 안 _MapThemeSync가 한다(J14).
      st.context.theme로 고르면 지도가 fragment 밖이라 메뉴로 테마를 바꾼 뒤 다음 전체 실행까지 어긋나고,
      테마마다 지도 JS가 달라져 iframe이 다시 마운트된다.
    - 기호: 모양(풍력 원·태양광 사각·ESS 마름모·화력 삼각)과 색으로 유형을, 크기로 설비용량을 보인다(J8).
    - 터치 기기에서는 끌기를 꺼서 한 손가락 스와이프가 페이지를 스크롤한다(J15·N17). 두 손가락은 확대·이동.
    - 범례는 지도 밖 _html_map_legend가 맡는다(옛 지도 안 고정 범례는 화력을 빠뜨렸고 지도를 가렸다).
    """
    m = folium.Map(
        location=[33.38, 126.55], zoom_start=10, tiles=None, control_scale=True,
        font_size="0.75rem",                     # D7 근본 원인: 기본 1rem이 출처 표기를 16px로 만든다
        zoom_control="topleft", zoom_snap=0.25,
        scroll_wheel_zoom=False,                 # 데스크톱 휠은 페이지 스크롤
        dragging=JsCode("!window.matchMedia('(pointer: coarse)').matches"),  # 터치 기기만 끌기 끔 (J15)
        max_bounds=True, min_lat=32.9, max_lat=33.9, min_lon=125.8, max_lon=127.3,
        # folium 0.20은 tiles=None이면 min_zoom·max_zoom 인자를 기본 타일에만 넘기고 버린다(0단계 E3) → 지도 옵션으로 직접 준다
        **{"minZoom": 8, "maxZoom": 14},
    )
    # folium 기본 자원 중 이 지도가 쓰는 것만 남긴다(N16·G11 첫 로드). 기호는 DivIcon SVG, 툴팁·팝업 틀은 Leaflet 기본이라
    # Bootstrap·Font Awesome·Glyphicons·awesome-markers는 쓰지 않는다(첫 로드 약 67 KB, S6 실측). jQuery는 남긴다:
    # folium 0.20 Popup 템플릿이 팝업 HTML을 $(`…`)[0]로 만들어서 빼면 첫 팝업에서 스크립트가 멈춘다.
    # Bootstrap이 하던 body 여백 0은 PD_CSS_MAP이 맡는다. 대신 iframe 안 글꼴을 바깥 화면과 같은 Pretendard로 맞춘다(F8).
    # st_folium은 머리의 link·script를 지우고 각 요소의 default_css·default_js만 불러오므로 지도 객체의 목록을 바꾼다
    # (folium.Map 클래스 목록은 그대로 두고 이 인스턴스에만 준다)
    m.default_js = [item for item in folium.Map.default_js if item[0] in ("leaflet", "jquery")]
    m.default_css = ([item for item in folium.Map.default_css if item[0] == "leaflet_css"]
                     + [("pretendard_css", PD_CSS_FONT_URL)])
    folium.TileLayer(tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                     attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
                     name="OpenStreetMap", control=False, max_zoom=19).add_to(m)
    _MapThemeSync().add_to(m)
    # folium.Element 문자열은 Jinja 템플릿이라 '{#'가 주석 시작이 된다. 지도 높이 규칙이 조용히 사라지면
    # Leaflet이 높이 0에서 fitBounds를 계산해 세계 지도(zoom 0)가 되므로(0단계 E3) 바로 실패시킨다
    if any(token in PD_CSS_MAP for token in ("{#", "{{", "{%")):
        raise ValueError("PD_CSS_MAP에 Jinja 구문 문자('{#'·'{{'·'{%')가 있어 iframe CSS 일부가 사라진다")
    m.get_root().header.add_child(folium.Element("<style>" + PD_CSS_MAP + "</style>"))

    type_labels = {"solar": "태양광", "wind": "풍력", "ess": "ESS", "thermal": "화력"}
    subtype_labels = {
        'onshore': '육상', 'offshore': '해상', 'island': '도서',
        'ground-mounted': '지상설치', 'rooftop': '옥상설치', 'community': '시민참여',
        'utility-scale': '대규모', 'renewable-coupled': '재생연계',
        'combined-cycle': '복합화력', 'steam': '기력',
    }
    registry_ref = _fmt_registry_ref(load_power_plants_data())

    def cell(value: object) -> str:
        """등록부 문자열을 팝업·툴팁 HTML에 넣는다. 비었으면 '—'.

        folium은 팝업·툴팁을 JS 템플릿 문자열(백틱)에 그대로 넣으므로 HTML 이스케이프에 더해 백틱과 '$'도 바꾼다.
        """
        text = value.strip() if isinstance(value, str) else ""
        if not text:
            return "—"
        return html_escape(text).replace("`", "&#96;").replace("$", "&#36;")

    # 큰 기호부터 추가해 작은 기호가 위에 놓이게 한다(작은 발전소가 큰 기호에 가려 눌리지 않게)
    ordered = plants_df.assign(_cap=pd.to_numeric(plants_df['capacity'], errors='coerce').fillna(0.0))
    for _, plant in ordered.sort_values('_cap', ascending=False, kind='stable').iterrows():
        kind = plant['type']
        type_label = type_labels.get(kind, cell(kind))
        cap = float(plant['_cap'])
        name = cell(plant.get('name'))
        subtype_raw = plant.get('subtype')
        subtype = subtype_labels.get(subtype_raw, subtype_raw) if isinstance(subtype_raw, str) and subtype_raw else ""
        type_text = f"{type_label} ({cell(subtype)})" if subtype else type_label
        generation, utilization = plant.get('generation'), plant.get('utilization')
        generation_text = f"≈{generation:.1f} MW" if _finite_number(generation) else "—"
        utilization_text = f"{utilization:.1f}%" if _finite_number(utilization) else "—"
        source_note = ("출처 예비 목록 (등록부를 읽지 못함)" if str(plant.get('id', '')).startswith('fallback_')
                       else f"출처 발전소 {cell(registry_ref)}")
        # 팝업 글자색은 iframe 테마 변수를 따른다(PD_CSS_MAP). 옛 9px #999 고지(2.85:1)를 12px로 바꿨다
        popup_html = (
            f'<div class="pd-pop"><p class="pd-pop__title">{name}</p><table>'
            f'<tr><th>유형</th><td>{type_text}</td></tr>'
            f'<tr><th>설비용량</th><td>{cap:,.1f} MW</td></tr>'
            f'<tr><th>현재 발전량 (추정)</th><td>{generation_text}</td></tr>'
            f'<tr><th>이용률 (추정)</th><td>{utilization_text}</td></tr>'
            f'<tr><th>운영사</th><td>{cell(plant.get("operator"))}</td></tr>'
            f'<tr><th>위치</th><td>{cell(plant.get("address"))}</td></tr>'
            f'<tr><th>상태</th><td>{cell(plant.get("status"))}</td></tr></table>'
            f'<p class="pd-pop__note">발전량·이용률은 시뮬레이션 추정치 (실측 아님)<br>{source_note}</p></div>'
        )
        # 툴팁·기호 이름은 등록부 사실만(이름·유형·설비). 시뮬레이션 출력은 넣지 않는다(J4)
        label = f"{name} · {type_label} · 설비 {cap:,.1f} MW"
        icon_html, box = _html_marker(kind, cap, label)
        folium.Marker(
            location=[plant['lat'], plant['lng']],
            icon=folium.DivIcon(html=icon_html, icon_size=(box, box), icon_anchor=(box // 2, box // 2),
                                class_name="pd-mk"),
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=label,
        ).add_to(m)

    # 지역 callout: 등록부 사실(곳 수·설비·유형 구성)만 바다 위에 둔다. 좁은 iframe(480px 이하)에서는 CSS로 숨긴다
    summary = _fmt_registry_summary(plants_df, None, "전체")
    # 위치는 아래 fit_bounds(줌 10)에서 바다 위·지도 칸 안에 오도록 정했다: 1024px(높이 440)에서도 위아래로 잘리지 않고,
    # 서귀포시 상자가 오른쪽 아래 출처 표기와 겹치지 않는다(명세 6.7 기본값 [33.62, 126.33]·[33.14, 126.80]에서 조정)
    for region, location in (("제주시", [33.585, 126.30]), ("서귀포시", [33.18, 126.62])):
        block = summary["regions"][region]
        if block["count"] == 0 or block["mw"] <= 0:
            continue
        segments = "".join(
            f'<span style="width:{block["by_type"][kind]["mw"] / block["mw"] * 100:.1f}%;background:{SERIES[kind]}"></span>'
            for kind in ("wind", "solar", "ess", "thermal") if block["by_type"][kind]["mw"] > 0
        )
        callout = (f'<div class="pd-callout"><b>{region}</b> · {block["count"]}곳 · 설비 {block["mw"]:,.1f} MW'
                   f'<span class="pd-callout__bar" aria-hidden="true">{segments}</span></div>')
        folium.Marker(location, icon=folium.DivIcon(html=callout, icon_size=(220, 44), icon_anchor=(0, 22),
                                                   class_name="pd-callout-mk"),
                      interactive=False, keyboard=False).add_to(m)

    # 열지도(토글 켤 때만): 시뮬레이션 출력이라 지도 바로 아래에 추정 칩을 붙인다(main). 무지개 대신 풍력 청록 한 색 농도
    if show_heatmap:
        heat_data = [[row['lat'], row['lng'], row['generation']]
                     for _, row in plants_df.iterrows() if _finite_number(row['generation']) and row['generation'] > 0]
        if heat_data:
            plugins.HeatMap(
                heat_data, min_opacity=0.25, radius=28, blur=22,
                gradient={0.2: "#A5F3FC", 0.5: "#22D3EE", 0.8: "#0891B2", 1.0: "#0E7490"},
            ).add_to(m)

    # 명세 6.7 기본값([[33.10, 126.10], [33.66, 127.00]])은 위도 폭 0.56°가 880×480 칸에 줌 10으로 들어가지 않아(최대 0.55°)
    # 줌 9.75가 되고 섬이 칸 폭의 56%만 채웠다(1차 평가). 위도 폭 0.49°면 1440·1366·1280·1024·768px에서 줌 10(정수 줌이라
    # 타일 이음선도 생기지 않는다), 390·430px에서 줌 9다. 등록부 기호 27곳(위도 33.164~33.556, 경도 126.163~126.923)이 모두 들어간다
    m.fit_bounds([[33.13, 126.12], [33.62, 126.98]])
    return m


# ============================================================================
# 차트 생성 함수
# ============================================================================

def create_smp_chart(smp_data: Dict, *, now: Optional[datetime] = None) -> go.Figure:
    """제주 SMP 24시간 계단 차트(명세 6.3절, D3·J3·N11).

    하루전 시장 가격은 거래시간마다 값이 하나라 계단선(hv)으로 그린다. 모델 예측일 때만 대시 계단과 80% 구간 띠
    (q10–q90)를 그린다. API 대체값·앱 로컬 대체는 점선 계단만 그린다: 로컬의 ±15%는 고정 가정값이고 API 대체값의
    분위수도 모델 구간이 아니어서, 띠를 그리면 통계 구간처럼 읽힌다(J3). 확정 가격 점은 EPSIS 거래시간이 예측일과
    같은 날일 때만 찍는다(최근 48시간 평균·시뮬레이션 값은 확정 가격이 아니다). 입력 지연은 그림 안 빨간 주석이 아니라
    카드 머리 칩이 밝힌다.

    툴팁은 계단 가운데에 둔 보이지 않는 점이 맡는다. 계단 시작점에 툴팁을 두면 계단 오른쪽 절반에서 다음 거래시간 값이
    잡히고, 계단을 닫는 마지막 점(24:00)에서 같은 값이 한 번 더 나온다.

    Args:
        smp_data: get_smp_data + API 예측을 합친 dict. _fmt_smp_display_keys 키가 없으면 여기서 붙인다.
        now: 화면 기준 시각(기본: 지금). fragment가 머리·표와 같은 시각을 넘겨 현재 거래시간이 어긋나지 않게 한다.

    Returns:
        _chart_base(높이 360)를 적용한 그림. 예측 값이 없으면 안내 문장만 있는 빈 그림.
    """
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    keys = _fmt_smp_with_keys(smp_data, now)
    axis = _fmt_smp_slots(smp_data, now)
    slots = axis["slots"]
    fig = go.Figure()
    if not slots:
        fig.add_annotation(text="SMP 예측 값이 없어 그리지 않습니다", x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False)
        _chart_base(fig, height=360)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    is_model = keys["pd_fc_kind"] == "model"
    # 마지막 값을 한 번 더 넣어 마지막 거래시간 계단을 닫는다(x는 칸 시작 24개 + 마지막 칸 끝)
    xs = [s["start"] for s in slots] + [slots[-1]["end"]]
    q50 = [s["q50"] for s in slots]
    if is_model:
        # 띠 면(13%)은 다크 카드 위에서 바탕과 거의 구분되지 않았다(1차 평가). 면은 명세 값 그대로 두고 같은 색 45% 1px 가장자리로
        # 구간의 위·아래 끝을 두 테마에서 읽히게 한다(중앙값 2.2px 대시와 굵기·선 모양이 달라 섞이지 않는다)
        band_edge = dict(width=1, shape="hv", color="rgba(192,38,211,0.45)")
        fig.add_trace(go.Scatter(x=xs, y=[s["q90"] for s in slots] + [slots[-1]["q90"]], mode="lines",
                                 name="80% 구간 위 (q90)", line=band_edge, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xs, y=[s["q10"] for s in slots] + [slots[-1]["q10"]], mode="lines",
                                 name="80% 구간 (q10–q90)", line=band_edge, fill="tonexty",
                                 fillcolor="rgba(192,38,211,0.13)", hoverinfo="skip"))
    if is_model:
        median_name = "예측 중앙값"
    elif keys["pd_fc_kind"] == "api_fallback":
        median_name = "API 대체값 (추정 · 모델 예측 아님)"
    else:
        median_name = "시간대 평균 패턴 (추정)"
    fig.add_trace(go.Scatter(x=xs, y=q50 + [q50[-1]], mode="lines", name=median_name, hoverinfo="skip",
                             line=dict(color=SERIES["smp"], width=2.2, dash="dash" if is_model else "dot", shape="hv")))

    # 지난 거래시간(옅게)·현재 거래시간(진하게) 음영. '#000036'(getGray70)은 두 테마의 카드 위에서 그래픽 3:1을 넘는다.
    # 지난 구간은 예측일 축에서만 뜻이 있다(지금부터 그린 축은 첫 칸이 지금이다)
    if axis["past_until"] is not None and axis["past_until"] > xs[0]:
        fig.add_vrect(x0=xs[0], x1=axis["past_until"], fillcolor="#000036", opacity=0.04, line_width=0, layer="below")
    cur = axis["cur"]
    if cur is not None:
        slot = slots[cur]
        fig.add_vrect(x0=slot["start"], x1=slot["end"], fillcolor="#000036", opacity=0.12, line_width=0, layer="below")
        # 흰 글자 pill(#475569 위 7.58:1). 양 끝 세 칸에서는 pill을 칸 바깥쪽 끝에 맞춰 그림 밖으로 잘리지 않게 한다
        if cur <= 2:
            pill_x, pill_anchor = slot["start"], "left"
        elif cur >= len(slots) - 3:
            pill_x, pill_anchor = slot["end"], "right"
        else:
            pill_x, pill_anchor = slot["start"] + timedelta(minutes=30), "center"
        fig.add_annotation(x=pill_x, y=1, yref="paper", xanchor=pill_anchor, yanchor="bottom", showarrow=False,
                           text=f"거래시간 {slot['trade_hour']:02d}h", bgcolor="#475569",
                           font=dict(color="#FFFFFF", size=12), borderpad=4)

    # 기준선: 중앙값 일평균. 라벨은 플롯 안 오른쪽 위라 오른쪽 여백을 쓰지 않아 390px에서도 잘리지 않는다
    mean = sum(q50) / len(q50)
    fig.add_hline(y=mean, line=dict(dash="dot", color="#000036", width=1))
    fig.add_annotation(x=1, xref="paper", y=mean, xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"{'예측' if is_model else '패턴'} 평균 {mean:,.1f}")

    # 툴팁 전용 점(보이지 않음)은 계단 가운데에 둬서 커서가 놓인 계단의 값이 잡힌다. 통합 툴팁 머리는 칸 시작 시각.
    # 통합 툴팁은 trace를 거꾸로 나열하므로 확정 가격 점 → 80% 구간 → 중앙값 순서로 넣어 중앙값이 머리 바로 아래 온다
    hover_x = [s["start"] + timedelta(minutes=30) for s in slots]
    hover_fmt = "%H:00부터 1시간"
    custom = [[f"{s['trade_hour']:02d}h", s["q10"], s["q90"]] for s in slots]

    # 확정 가격 점: 채운 점 + 테두리 '#000038'(카드 면)이라 계단선 위에 떠 보인다(뒤의 툴팁 점은 보이지 않는다)
    confirmed = axis["confirmed"]
    y_values = [s["q10"] if is_model else s["q50"] for s in slots] + [s["q90"] if is_model else s["q50"] for s in slots]
    if keys["pd_confirmed_point"] and confirmed is not None and _finite_number(smp_data.get("current")):
        price = float(smp_data["current"])
        y_values.append(price)
        point_x = slots[confirmed]["start"] + timedelta(minutes=30)
        fig.add_trace(go.Scatter(x=[point_x], y=[price], mode="markers", name="확정 가격",
                                 marker=dict(size=11, color=SERIES["smp"], line=dict(color="#000038", width=2)),
                                 xhoverformat=hover_fmt,
                                 hovertemplate="%{y:,.1f} 원/kWh<extra>확정 가격 · EPSIS 하루전</extra>"))
        right_half = confirmed >= len(slots) - 6
        fig.add_annotation(x=point_x, y=price, text=f"확정 {price:,.1f}", showarrow=False, yanchor="bottom", yshift=6,
                           xanchor="right" if right_half else "left", xshift=-6 if right_half else 6)

    if is_model:
        fig.add_trace(go.Scatter(x=hover_x, y=q50, mode="markers", name="80% 구간 (q10–q90)", customdata=custom,
                                 marker=dict(size=10, color=SERIES["smp"], opacity=0), xhoverformat=hover_fmt,
                                 hovertemplate="%{customdata[1]:,.1f}–%{customdata[2]:,.1f} 원/kWh"
                                               "<extra>80% 구간 (q10–q90)</extra>"))
    median_tip = ("%{y:,.1f} 원/kWh<extra>예측 중앙값 · 거래시간 %{customdata[0]}</extra>" if is_model
                  else "%{y:,.1f} 원/kWh<extra>추정 패턴 · 예측 모델 아님</extra>")
    fig.add_trace(go.Scatter(x=hover_x, y=q50, mode="markers", name=median_name, customdata=custom,
                             marker=dict(size=10, color=SERIES["smp"], opacity=0), xhoverformat=hover_fmt,
                             hovertemplate=median_tip))

    # 높이 440(명세 6.3 기본값 360에서 조정): 1181px 이상에서 차트 카드(8/12)가 오른쪽 열(통계·시간대별 요약·모델·예시)보다
    # 약 170px 먼저 끝나 빈 칸이 생겼다(1차 평가). 계단 사이 간격도 넓어져 거래시간별 값을 읽기 쉽다
    _chart_base(fig, height=440)
    # 위를 비워 현재 pill 아래의 '확정'·'평균' 글자가 위 테두리에 붙지 않게 한다. 축 제목은 두지 않는다(단위는 카드 머리)
    low, high = min(y_values), max(y_values)
    span = max(high - low, 10.0)
    bottom = low - span * 0.12 if low < 0 else max(0.0, low - span * 0.12)
    # x 범위 끝을 1초 당겨 24:00 눈금이 '00:00'으로 찍히지 않게 한다. 눈금 수는 축 길이에 맡긴다(명세 6.1절)
    fig.update_layout(
        yaxis=dict(tickformat=",", ticksuffix="", range=[bottom, high + span * 0.22]),
        xaxis=dict(tickformat="%H:%M", range=[xs[0], xs[-1] - timedelta(seconds=1)]),
    )
    return fig


def create_realtime_power_chart(demand_now: float, supply_now: float, demand_measured: bool,
                                supply_measured: bool, *, now: Optional[datetime] = None,
                                query_hhmm: str = "") -> go.Figure:
    """수급 추이: 지난 12시간 · 다음 12시간 (명세 6.2절, N3·N4·J3·J5).

    곡선은 앱 내장 일간 패턴(get_load_pattern, 손으로 정한 구간 선형식)에 현재값을 맞춘 '추정 패턴'이다.
    시간별 실측 이력을 주는 원천이 아직 없으므로(명세 10장 P2) 선은 모두 점선으로 그리고 이름에 '추정 패턴'을
    붙인다. 실측일 수 있는 것은 현재 점 두 개뿐이고, 지금 잰 값이 아니면 빈 원과 '실측 아님' 툴팁으로 그린다.
    EPSIS 자료가 아니므로 'EPSIS 패턴'이라고 쓰지 않는다. 고정 가정값 범위 띠는 통계 구간이 아니라서 그리지 않는다(J3).

    Args:
        demand_now: 지금 수요(MW). 수요 KPI와 같은 power_status['demand']를 받아 두 숫자가 어긋나지 않게 한다(N3).
        supply_now: 지금 공급능력(MW). power_status['total_supply'].
        demand_measured: 수요 현재값이 지금 잰 KPX 값인지(_fmt_provenance ③⑦).
        supply_measured: 공급능력 현재값이 KPX 조회값인지.
        now: 화면 기준 시각(기본: 지금). 초 아래를 버려 같은 분·같은 입력이면 같은 그림이 된다.
        query_hhmm: 실측 현재 점 툴팁에 쓰는 조회 시각(HH:MM). 관측 시각이 아니다(명세 7.3절).

    Returns:
        _chart_base를 적용한 그림. 현재값이 유한수가 아니면 안내 문장만 있는 빈 그림.
    """
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    # 5분 버킷 시드의 지역 난수: 전역 난수 상태를 바꾸지 않고, 위젯을 눌러 다시 그려도 5분 안에서는 곡선이 같다(N4).
    # 버킷 경계는 KST 정각·5분 단위와 맞으므로 한 버킷 안에서는 30분 격자의 시(hour)도 바뀌지 않는다
    rng = np.random.default_rng(int(now.timestamp() // 300))

    # 시간 범위는 옛 차트와 같다: 과거 24점·미래 25점(30분 간격, 둘 다 지금 점을 포함)
    past_hours = pd.date_range(end=now, periods=24, freq='30min')
    future_hours = pd.date_range(start=now, periods=25, freq='30min')

    # ===== 일간 패턴 함수 (단일 진실 공급원) =====
    def get_load_pattern(hour: int) -> float:
        """제주 전력 수요 일간 패턴 - 단일 함수로 통일 (손으로 정한 구간 선형식, 실측·EPSIS 자료 아님)"""
        if 6 <= hour <= 9:
            return 0.9 + 0.1 * (hour - 6) / 3  # 아침 증가
        elif 9 <= hour <= 14:
            return 1.0 - 0.15 * (hour - 9) / 5  # 낮 감소 (태양광)
        elif 14 <= hour <= 20:
            return 0.85 + 0.25 * (hour - 14) / 6  # 저녁 피크
        elif 20 <= hour <= 23:
            return 1.1 - 0.2 * (hour - 20) / 3  # 야간 감소
        else:
            return 0.85  # 심야

    fig = go.Figure()
    if not (_finite_number(demand_now) and _finite_number(supply_now)):
        # 현재값이 없으면 패턴의 기준을 정할 수 없다. 지어낸 기본값으로 곡선을 그리지 않는다
        fig.add_annotation(text="수요·공급능력 현재값이 없어 곡선을 그리지 않습니다", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        _chart_base(fig, height=360)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    # 현재값을 지금 시각의 패턴 계수로 나눠 기준값을 만들고, 시각마다 패턴을 곱한다
    current_pattern = get_load_pattern(now.hour)
    base_demand = demand_now / current_pattern if current_pattern > 0 else demand_now
    base_supply = supply_now / current_pattern if current_pattern > 0 else supply_now

    demand, supply = [], []
    for h in past_hours:
        pattern = get_load_pattern(h.hour)
        noise = rng.uniform(-0.02, 0.02)  # 약간의 변동성 (5분 버킷 시드)
        demand.append(base_demand * pattern * (1 + noise))
        supply.append(base_supply * pattern * (1 + noise * 0.5))
    # 지금 점은 현재값 그대로(과거 곡선의 끝 = 미래 곡선의 시작). 미래는 같은 패턴을 잇기만 한다(예측 모델 아님)
    demand[-1], supply[-1] = demand_now, supply_now
    for h in future_hours[1:]:
        pattern = get_load_pattern(h.hour)
        demand.append(base_demand * pattern)
        supply.append(base_supply * pattern)

    # 과거와 미래를 한 trace로 잇는다(지금 점 한 번). 소수 1자리로 줄여 60초 fragment 전송량을 줄인다
    xs = [t.to_pydatetime() for t in past_hours] + [t.to_pydatetime() for t in future_hours[1:]]
    demand = [round(float(v), 1) for v in demand]
    supply = [round(float(v), 1) for v in supply]
    boundary = [round(v / 1.15, 1) for v in supply]  # 관심 경계: 예비율 15% = 공급능력 ÷ 1.15
    end = xs[-1]

    # 공급능력 → 수요 순서여야 수요의 fill="tonexty"가 두 선 사이(예비력)를 칠한다.
    # 예비력 면은 4% + 빗금 8%로 연하게 둬서 점선이 빗금 위에서도 3:1을 넘는다(J5, 명세 3.5절 F표)
    fig.add_trace(go.Scatter(
        x=xs, y=supply, mode="lines", name="공급능력 (추정 패턴)",
        line=dict(color=SERIES["supply"], width=1.8, dash="dot"),
        hovertemplate="%{y:,.0f} MW<extra>공급능력 · 추정 패턴</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=demand, mode="lines", name="수요 (추정 패턴)",
        line=dict(color=SERIES["demand"], width=2.4, dash="dot"),
        fill="tonexty", fillcolor="rgba(22,163,74,0.04)",
        fillpattern=dict(shape="/", fgcolor=SERIES["reserve"], fgopacity=0.08, size=10, solidity=0.12),
        hovertemplate="%{y:,.0f} MW<extra>수요 · 추정 패턴</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=boundary, mode="lines", name="관심 경계 (예비율 15%)",
        line=dict(color=BOUNDARY_COLOR, width=1.2, dash="dash"),
        hovertemplate="%{y:,.0f} MW<extra>관심 경계 · 공급능력 ÷ 1.15</extra>",
    ))

    # 현재 점: 지금 잰 값이면 채운 원(테두리 '#000038' = 카드 면이라 선 위에 떠 보인다), 아니면 빈 원(색 외 단서).
    # 통합 툴팁에서 두 점이 함께 보이므로 <extra>에 어느 값인지와 출처를 함께 쓴다
    for value, color, name, measured, below in (
        (demand_now, SERIES["demand"], "수요 현재값", demand_measured, True),
        (supply_now, SERIES["supply"], "공급능력 현재값", supply_measured, False),
    ):
        if measured:
            source = f"KPX 실측 · 조회 {query_hhmm} KST" if query_hhmm else "KPX 실측"
        else:
            source = "추정 · 실측 아님"
        fig.add_trace(go.Scatter(
            x=[now], y=[value], mode="markers", name=name,
            marker=dict(size=11, color=color, symbol="circle" if measured else "circle-open",
                        line=dict(color="#000038" if measured else color, width=2)),
            hovertemplate=f"%{{y:,.0f}} MW<extra>{name} · {source}</extra>",
        ))
        # 값 글자는 색을 주지 않아 '#000037'(두 테마 AA)을 따른다. 수요는 점 아래, 공급능력은 점 위에 두어
        # 두 값이 가까워져도 글자끼리 멀어진다. 현재선 왼쪽에 두는 이유: 오른쪽 끝의 관심 경계 pill과
        # 360~390px 폭에서 몇 px 간격까지 붙었다(S4 실측). 선이 글자를 지나가도 CSS 테두리(11절)로 읽힌다
        fig.add_annotation(x=now, y=value, text=f"{value:,.0f} MW", showarrow=False, xanchor="right", xshift=-8,
                           yanchor="top" if below else "bottom", yshift=-6 if below else 6)

    # 현재선 '#000036'(getGray70)은 그래픽 3:1을 두 테마에서 넘는다. datetime 축에서
    # add_vline(annotation_text=…)은 plotly 6.5 + pandas 2.3에서 TypeError라 pill은 주석으로 따로 단다(명세 6.1절)
    fig.add_vline(x=now, line=dict(color="#000036", width=1.5))
    fig.add_annotation(x=now, y=1, yref="paper", yanchor="bottom", showarrow=False,
                       text=f"현재 {now:%H:%M} KST", bgcolor="#475569",
                       font=dict(color="#FFFFFF", size=12), borderpad=4)    # 흰 글자 7.58:1
    fig.add_vrect(x0=now, x1=end, fillcolor="#000036", opacity=0.06, line_width=0, layer="below")  # 미래 음영
    # 관심 경계 pill: #B45309 위 흰 글자 5.02:1(명세 3.3절). 플롯 오른쪽 끝 안쪽에 붙여 390px에서도 안 잘린다
    fig.add_annotation(x=end, y=boundary[-1], text="관심 경계 15%", showarrow=False, xanchor="right",
                       yanchor="bottom", yshift=2, bgcolor=BOUNDARY_COLOR,
                       font=dict(color="#FFFFFF", size=11), borderpad=3)

    # 단위는 축 제목 대신 y 눈금 위에 둔다(와이어프레임 4.2절 'MW 1,600 ┤'). 축 제목은 회전 글자라 좁은 폭에서 자리를 먹는다
    fig.add_annotation(x=0, xref="paper", y=1, yref="paper", xanchor="right", yanchor="bottom", xshift=-4,
                       text="MW", showarrow=False)
    # 구간 설명은 플롯 안 글자 대신 카드 아래 시간 띠가 맡는다(명세 5.7절). Plotly 범례 대신 범례 겸 표(D5)
    _chart_base(fig, height=360)
    # 위를 15% 비워 공급능력 값 글자가 위 여백의 현재 pill과 겹치지 않게 한다. 0부터 그린다(rangemode tozero 기준)
    y_top = max(max(supply), max(demand)) * 1.15
    fig.update_layout(
        yaxis=dict(tickformat=",", rangemode="tozero", range=[0, y_top]),
        xaxis=dict(tickformat="%H:%M", hoverformat="%m-%d %H:%M"),  # 눈금 수는 축 길이에 맡긴다(명세 6.1절)
    )
    return fig


def create_gauge_chart(value: float, title: str, max_val: float = 100,
                       color: str = '#3B82F6', suffix: str = '',
                       reserve: bool = False) -> go.Figure:
    """게이지. 예비율은 낮을수록 위험한 구간색을 쓴다.

    T7(test_v4_reserve_gauge)이 이 함수를 go만 있는 이름공간에서 실행하므로
    리터럴과 builtins만 쓴다. 글자색 '#000037'은 Streamlit 1.52.1 Plotly 테마의
    자리표시자(getGray90: 라이트 #262730 / 다크 #FAFAFA)라 두 테마에서 4.5:1을 넘는다.
    """
    if reserve:
        # 예비율은 낮을수록 위험하며 표시 범위는 실제 값을 포함한다.
        low = min(0, value)
        max_val = max(max_val, value * 1.1)
        steps = [
            {'range': [low, 10], 'color': 'rgba(220, 38, 38, 0.22)'},
            {'range': [10, 15], 'color': 'rgba(180, 83, 9, 0.22)'},
            {'range': [15, max_val], 'color': 'rgba(22, 163, 74, 0.18)'},
        ]
        threshold = 10
    else:
        low = 0
        steps = [
            {'range': [0, max_val * 0.5], 'color': 'rgba(100, 116, 139, 0.10)'},
            {'range': [max_val * 0.5, max_val * 0.8], 'color': 'rgba(100, 116, 139, 0.18)'},
        ]
        threshold = max_val * 0.9
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={'suffix': suffix, 'valueformat': '.1f' if reserve else ',.0f',
                'font': {'size': 28, 'color': '#000037'}},
        title={'text': title, 'font': {'size': 13, 'color': '#000037'}},
        gauge={
            'axis': {'range': [low, max_val], 'nticks': 6,
                     'tickfont': {'size': 12, 'color': '#000037'}},
            'bar': {'color': color, 'thickness': 0.6},
            'bgcolor': 'rgba(100, 116, 139, 0.10)',
            'borderwidth': 0,
            'steps': steps,
            'threshold': {'line': {'color': '#DC2626', 'width': 2.5}, 'thickness': 0.85, 'value': threshold},
        },
    ))
    fig.update_layout(margin=dict(l=28, r=28, t=16, b=8), height=190, dragmode=False)
    return fig


def create_supply_donut(power_status: Dict) -> go.Figure:
    """전력 공급 구성 도넛(명세 6.5절, T8·N9·N10). 값은 앱 로컬 추정 power_status['supply'] 그대로다.

    조각 글자를 없애고 값은 카드의 범례 겸 표(_html_mix_table)가 나른다: 바깥 글자는 좁은 폭에서 잘렸고(N9),
    조각 안 12px 흰 글자는 대비가 3.64~4.62로 AA에 못 미친다. 색 외 단서로 조각마다 빗금 방향을 다르게 하고
    (태양광 / · 풍력 \\ · 화력 x · ESS 점) 표 견본도 같은 방향이라 색을 못 구분해도 조각과 행을 맞출 수 있다.
    모든 조각이 빗금이라 '추정·시뮬레이션 = 빗금' 문법(명세 7.1절)도 그대로다. 가운데 숫자는 수요가 아니라 조각 합이다.
    """
    supply = power_status['supply']

    labels = ['태양광', '풍력', '화력', 'ESS']
    values = [supply['solar'], supply['wind'], supply['thermal'], abs(supply['ess'])]
    colors = [SERIES["solar"], SERIES["wind"], SERIES["thermal"], SERIES["ess"]]

    # 조각 경계 '#000038'은 테마 배경(카드 면)이라 두 테마에서 조각 사이가 갈라져 보인다.
    # 무늬 바탕색(bgcolor)을 조각 색으로 직접 준다: 기본 fillmode "replace"는 조각 색을 무늬로 바꾸고, 바탕색을 주지 않으면
    # Streamlit 테마 템플릿의 #FFFFFF(그림을 처음 그린 테마 배경)가 들어가 흰 바탕에 흰 빗금만 남는다(조각이 안 보임, 6단계 실측)
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        sort=False,
        direction="clockwise",
        textinfo="none",
        marker=dict(colors=colors, line=dict(color="#000038", width=2),
                    pattern=dict(shape=["/", "\\", "x", "."], bgcolor=colors, fgcolor="rgba(255,255,255,0.45)",
                                 size=6, solidity=0.25)),
        hovertemplate="%{label} %{value:,.0f} MW (%{percent})<extra>시뮬레이션 추정</extra>",
    )])

    # 가운데: 조각 합(추정 숫자는 ≈ + 10 MW 단위 반올림, 명세 3.6절). 글자색은 _chart_base의 '#000037'
    total = sum(v for v in values if _finite_number(v))
    fig.add_annotation(text=f"<b>≈{round(total, -1):,.0f} MW</b><br><span style='font-size:12px'>추정 합계</span>",
                       x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=16))

    return _chart_base(fig, height=260, margin=dict(l=8, r=8, t=8, b=8))


def create_generation_timeline(plants_df: pd.DataFrame) -> go.Figure:
    """시간대별 발전량 (시뮬레이션) 누적 막대(명세 6.6절, T8·N4·N9).

    생성 로직은 그대로다: 태양광 = 등록부 태양광 설비 × sin × 0.8(6~18시), 풍력 = 등록부 풍력 설비 × U(0.4, 0.7).
    N4만 적용했다: 난수는 5분 버킷 시드의 지역 객체라 전역 난수 상태를 바꾸지 않고, 위젯을 눌러 다시 그려도 5분 안에서는
    그림이 같다. 모두 시뮬레이션 값이라 막대를 빗금으로 칠하고(추정·시뮬레이션 = 빗금, 명세 7.1절) 고지는 카드 머리 칩이
    맡는다(그림 안 고지 주석은 좁은 폭에서 잘렸다, N9). 빗금 방향은 도넛·구성 표와 같다(태양광 / · 풍력 \\).
    """
    # 시간대별 발전량 시뮬레이션
    now = datetime.now()
    hours = list(range(24))
    current_hour = now.hour
    # 5분 버킷 시드(KST 정각·5분 단위와 경계가 같아 한 버킷 안에서는 현재 시(hour)도 바뀌지 않는다)
    rng = random.Random(int(now.timestamp() // 300))

    solar_gen = []
    wind_gen = []

    for h in hours:
        # 태양광
        if 6 <= h <= 18:
            solar = plants_df[plants_df['type'] == 'solar']['capacity'].sum() * np.sin(np.pi * (h - 6) / 12) * 0.8
        else:
            solar = 0
        solar_gen.append(solar)

        # 풍력
        wind = plants_df[plants_df['type'] == 'wind']['capacity'].sum() * rng.uniform(0.4, 0.7)
        wind_gen.append(wind)

    # 소수 1자리로 줄여 표로 보기와 같은 값을 보이고 fragment 전송량을 줄인다
    solar_gen = [round(float(v), 1) for v in solar_gen]
    wind_gen = [round(float(v), 1) for v in wind_gen]
    tip = "%{y:,.1f} MW<extra>%{fullData.name} · 시뮬레이션, 실측 아님</extra>"

    # 무늬 바탕색을 막대 채움색(계열 35%)으로 직접 준다. 주지 않으면 기본 fillmode "replace"가 채움색을 버리고 테마 템플릿의
    # 배경색(처음 그린 테마 값)을 깔아, 테마를 바꾼 뒤 막대 안이 이전 테마 바탕색으로 남는다(도넛과 같은 원인)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hours, y=wind_gen, name="풍력", hovertemplate=tip,
        marker=dict(color="rgba(8,145,178,0.35)", line=dict(color=SERIES["wind"], width=1),
                    pattern=dict(shape="\\", bgcolor="rgba(8,145,178,0.35)", fgcolor=SERIES["wind"], fgopacity=0.6,
                                 size=6, solidity=0.3)),
    ))
    fig.add_trace(go.Bar(
        x=hours, y=solar_gen, name="태양광", hovertemplate=tip,
        marker=dict(color="rgba(183,121,31,0.35)", line=dict(color=SERIES["solar"], width=1),
                    pattern=dict(shape="/", bgcolor="rgba(183,121,31,0.35)", fgcolor=SERIES["solar"], fgopacity=0.6,
                                 size=6, solidity=0.3)),
    ))

    # 현재 시각 막대: 점선 상자('#000037' = getGray90, 두 테마에서 4.5:1 이상) + 흰 글자 pill(#475569 위 7.58:1)
    total_now = wind_gen[current_hour] + solar_gen[current_hour]
    fig.add_shape(type="rect", x0=current_hour - 0.42, x1=current_hour + 0.42, y0=0, y1=total_now,
                  line=dict(color="#000037", width=1.2, dash="dot"))
    if current_hour <= 1:
        pill_anchor = "left"
    elif current_hour >= 22:
        pill_anchor = "right"
    else:
        pill_anchor = "center"
    fig.add_annotation(x=current_hour, y=1, yref="paper", xanchor=pill_anchor, yanchor="bottom", showarrow=False,
                       text=f"현재 {current_hour:02d}시", bgcolor="#475569", font=dict(color="#FFFFFF", size=12),
                       borderpad=4)
    # 단위는 축 제목 대신 y 눈금 위에 둔다(수급 차트와 같은 자리). 축 제목은 회전 글자라 좁은 폭에서 자리를 먹는다
    fig.add_annotation(x=0, xref="paper", y=1, yref="paper", xanchor="right", yanchor="bottom", xshift=-4,
                       text="MW", showarrow=False)

    _chart_base(fig, height=250)
    fig.update_layout(barmode="stack", bargap=0.28)
    # 눈금 글자는 ticktext, 통합 툴팁 머리는 ticksuffix로 '7시'처럼 읽힌다(숫자 축이라 날짜 축 규칙과 무관)
    fig.update_xaxes(tickmode="array", tickvals=list(range(0, 24, 3)), ticktext=[f"{h}시" for h in range(0, 24, 3)],
                     ticksuffix="시", range=[-0.6, 23.6])
    fig.update_yaxes(tickformat=",", rangemode="tozero")

    return fig


# ============================================================================
# 계열 색 · 차트 공통 (빌드 명세 3.4절 · 6.1절)
# ============================================================================

# app_v4.py 모듈 상수. 지도·차트·범례·칩·표가 모두 여기서 읽는다 (F5)
# Streamlit 1.52.1은 테마별 차트 색을 나눌 수 없어(1.62부터) 두 테마의 카드·캔버스 바탕 모두에서 3:1을 넘는 단일값을 쓴다.
SERIES = {
    "demand": "#3B82F6",   # 수요
    "supply": "#64748B",   # 공급능력
    "reserve": "#16A34A",  # 예비력 (면은 4% + 빗금 8%)
    "solar": "#B7791F",    # 태양광 (앰버 #F59E0B는 라이트 2.02로 미달 → 황토)
    "wind": "#0891B2",     # 풍력 (수요 파랑과 분리, web-v7 풍력 청록 계열)
    "ess": "#8B5CF6",      # ESS
    "thermal": "#8D6E63",  # 화력 LNG·유류 (지도 빨강·도넛 회색 불일치 해소)
    "hvdc": "#7C8DB0",     # HVDC 연계
    "smp": "#C026D3",      # SMP
    "other": "#4D7C0F",    # 기타 (10색 채움용)
}
BOUNDARY_COLOR = "#B45309"  # 관심 경계선 (예비율 15%)
ALERT_RED = "#DC2626"       # 경보·임계선 전용. 데이터 계열에 쓰지 않는다


def _chart_base(fig: go.Figure, *, height: int, margin: Optional[Dict] = None) -> go.Figure:
    """테마 색은 Streamlit에 맡기고, 글자색은 1.52.1 자리표시자 '#000037'(getGray90)로 AA를 맞춘다.

    Streamlit 1.52.1은 figure JSON 문자열의 '#000032'~'#000040'을 활성 테마 색으로 바꿔 그린다
    ('#000036' = getGray70, '#000037' = getGray90, '#000038' = 테마 배경 = 카드 면). 그래서 글자색을 하드코딩하지 않는다.
    기본 눈금색 getGray70(라이트 #808495)은 흰 카드 위 3.49:1로 AA 미달이라 글자에는 '#000037'만 쓴다.
    단, 이 치환은 그림을 처음 그릴 때만 일어난다: ⋮ → Settings로 테마를 바꾸면 이미 색이 들어간 그림에 다시 적용되므로
    글자가 이전 테마 색으로 남는다(S3 실측). 테마 전환은 PD_CSS_COMPONENTS 10절 CSS가 글자를 페이지 토큰으로 덮어 맞춘다.
    이 치환은 1.52.1 번들 전제다(버전을 올리면 명세 9.4절 E4와 테마 전환 검사를 다시 돌린다).

    끌기·확대를 끄고(dragmode=False, fixedrange) 모드바를 숨기면(호출부 config) 손가락 스와이프가 차트에 갇히지 않고
    페이지를 스크롤한다(N17). 제목·범례는 그림에 두지 않는다: title=None은 '<b>undefined</b>'가 되고(D3),
    범례는 카드의 HTML 범례 겸 표가 맡는다.

    Args:
        fig: 꾸밀 그림. 제자리에서 바꾸고 그대로 돌려준다.
        height: 그림 높이(px). Plotly 높이는 CSS로 바꿀 수 없으므로 모든 폭에서 같다.
        margin: 여백 dict. 없으면 l=8, r=8, t=32, b=8.

    Returns:
        같은 fig.
    """
    fig.update_layout(
        height=height,
        margin=margin or dict(l=8, r=8, t=32, b=8),
        dragmode=False, hovermode="x unified", showlegend=False,
        font=dict(color="#000037", size=12),
        hoverlabel=dict(font=dict(color="#000037")),
    )
    fig.update_xaxes(fixedrange=True, tickfont=dict(color="#000037", size=12))
    fig.update_yaxes(fixedrange=True, tickfont=dict(color="#000037", size=12), zeroline=False)
    return fig


# ============================================================================
# 화면 CSS · 아이콘 (빌드 명세 3장 토큰 · 3.8절 아이콘 · 5.15절 CSS)
# ============================================================================

# 아이콘 path 원본은 docs/2026-11-23-dashboard-ui-top1/icons.json(시안 A ICONS 24개)이고 wave는 3.8절에서 더했다.
# 이모지 대신 CSS 마스크로 그리므로 글꼴·OS에 따라 모양이 바뀌지 않고 글자색(currentColor)을 따른다.
PD_ICONS: Dict[str, str] = {
    "map": "<path d='M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2z'/><path d='M9 4v14M15 6v14'/>",
    "bolt": "<path d='M13 2 3 14h9l-1 8 10-12h-9l1-8z'/>",
    "sun": "<circle cx='12' cy='12' r='4'/><path d='M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4'/>",
    "wind": "<path d='M3 8h11a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h8'/>",
    "battery": "<rect x='2' y='7' width='17' height='10' rx='2'/><path d='M22 11v2M6 10v4M10 10v4'/>",
    "flame": "<path d='M12 2s5 4.5 5 10a5 5 0 0 1-10 0c0-2.5 1.5-4 1.5-4S9 11 11 11c0-3 1-6 1-9z'/>",
    "plug": "<path d='M9 2v5M15 2v5M6 7h12v4a6 6 0 0 1-12 0V7zM12 17v5'/>",
    "clock": "<circle cx='12' cy='12' r='9'/><path d='M12 7v5l3 2'/>",
    "refresh": "<path d='M21 12a9 9 0 1 1-2.6-6.4M21 4v5h-5'/>",
    "info": "<circle cx='12' cy='12' r='9'/><path d='M12 11v5M12 8h.01'/>",
    "alert": "<path d='M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z'/><path d='M12 9v4M12 17h.01'/>",
    "chart": "<path d='M3 3v18h18'/><path d='M7 15l4-4 3 3 5-6'/>",
    "tag": "<path d='M20.6 13.4 13.4 20.6a2 2 0 0 1-2.8 0L3 13V3h10l7.6 7.6a2 2 0 0 1 0 2.8z'/><circle cx='7.5' cy='7.5' r='1.5'/>",
    "spark": "<circle cx='12' cy='12' r='3'/><path d='M12 3v3M12 18v3M3 12h3M18 12h3M5.6 5.6l2.1 2.1M16.3 16.3l2.1 2.1M5.6 18.4l2.1-2.1M16.3 7.7l2.1-2.1'/>",
    "thermo": "<path d='M14 14.8V3.5a2.5 2.5 0 0 0-5 0v11.3a4.5 4.5 0 1 0 5 0z'/>",
    "db": "<ellipse cx='12' cy='5' rx='8' ry='3'/><path d='M4 5v14c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3'/>",
    "plus": "<path d='M12 5v14M5 12h14'/>",
    "minus": "<path d='M5 12h14'/>",
    "layers": "<path d='M12 2 2 7l10 5 10-5-10-5z'/><path d='M2 17l10 5 10-5M2 12l10 5 10-5'/>",
    "gauge": "<path d='M12 14l4-4'/><path d='M3.3 19a10 10 0 1 1 17.4 0'/>",
    "chev": "<path d='M9 6l6 6-6 6'/>",
    "chev2": "<path d='M6 7l5 5-5 5M13 7l5 5-5 5'/>",
    "leaf": "<path d='M11 20A7 7 0 0 1 4 13c0-6 6-9 16-9 0 10-3 16-9 16z'/><path d='M4 20c4-6 8-9 12-11'/>",
    "list": "<path d='M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01'/>",
    "wave": "<path d='M2 12h3l3-7 4 14 3-7h7'/>",
}

# 토큰은 light-dark()로 두 테마 값을 함께 둔다. Streamlit 1.52.1이 stApp·stSidebar에 활성 테마의
# color-scheme을 넣으므로 ⋮ → Settings에서 테마를 바꾸면 rerun 없이 바로 따라간다(명세 F1).
# 세 상수 모두 여는 꺾쇠를 쓰지 않는다: st.html의 style 블록에 꺾쇠가 섞이면 DOMPurify가 블록 전체를 지운다.
PD_CSS_TOKENS = """
/* ---- 1. 토큰: light-dark() 미지원 브라우저는 라이트 값으로 떨어진다 ---- */
:root{
  --pd-canvas:#F6F8FB;--pd-surface:#FFFFFF;--pd-well:#F1F5F9;--pd-raised:#FFFFFF;
  --pd-border:#E2E8F0;--pd-border-strong:#7B8798;
  --pd-text-1:#0F172A;--pd-text-2:#334155;--pd-text-3:#56657A;--pd-link:#1D4ED8;
  --pd-primary:#2563EB;--pd-focus:#2563EB;--pd-shadow-c:rgba(15,23,42,.06);
  --pd-ok-fg:#166534;--pd-ok-bg:#DCFCE7;--pd-warn-fg:#92400E;--pd-warn-bg:#FEF3C7;
  --pd-danger-fg:#9A3412;--pd-danger-bg:#FFEDD5;--pd-crit-fg:#B91C1C;--pd-crit-bg:#FEE2E2;
  --pd-info-fg:#1D4ED8;--pd-info-bg:#DBEAFE;--pd-meas-border:#94A3B8;
  --pd-fc-fg:#1D4ED8;--pd-fc-bg:#EFF6FF;--pd-fc-border:#3B82F6;
  --pd-est-fg:#334155;--pd-est-bg:#F1F5F9;--pd-est-stripe:rgba(100,116,139,.22);
  --pd-ex-fg:#92400E;--pd-ex-bg:#FFFBEB;--pd-ex-border:rgba(217,119,6,.55);
  --pd-s-demand:#3B82F6;--pd-s-supply:#64748B;--pd-s-reserve:#16A34A;--pd-s-solar:#B7791F;
  --pd-s-wind:#0891B2;--pd-s-ess:#8B5CF6;--pd-s-thermal:#8D6E63;--pd-s-hvdc:#7C8DB0;--pd-s-smp:#C026D3;
}
@supports (color: light-dark(#000, #fff)){
  :root{
    --pd-canvas:light-dark(#F6F8FB,#0B1220);--pd-surface:light-dark(#FFFFFF,#121B31);
    --pd-well:light-dark(#F1F5F9,#1A2439);--pd-raised:light-dark(#FFFFFF,#1A2439);
    --pd-border:light-dark(#E2E8F0,#24324D);--pd-border-strong:light-dark(#7B8798,#6B7A93);
    --pd-text-1:light-dark(#0F172A,#F1F5F9);--pd-text-2:light-dark(#334155,#CBD5E1);
    --pd-text-3:light-dark(#56657A,#94A3B8);--pd-link:light-dark(#1D4ED8,#60A5FA);
    --pd-primary:light-dark(#2563EB,#60A5FA);--pd-focus:light-dark(#2563EB,#60A5FA);
    --pd-shadow-c:light-dark(rgba(15,23,42,.06),transparent);
    --pd-ok-fg:light-dark(#166534,#4ADE80);--pd-ok-bg:light-dark(#DCFCE7,rgba(74,222,128,.14));
    --pd-warn-fg:light-dark(#92400E,#FBBF24);--pd-warn-bg:light-dark(#FEF3C7,rgba(251,191,36,.14));
    --pd-danger-fg:light-dark(#9A3412,#FB923C);--pd-danger-bg:light-dark(#FFEDD5,rgba(251,146,60,.14));
    --pd-crit-fg:light-dark(#B91C1C,#F87171);--pd-crit-bg:light-dark(#FEE2E2,rgba(248,113,113,.16));
    --pd-info-fg:light-dark(#1D4ED8,#93C5FD);--pd-info-bg:light-dark(#DBEAFE,rgba(96,165,250,.14));
    --pd-meas-border:light-dark(#94A3B8,#64748B);
    --pd-fc-fg:light-dark(#1D4ED8,#93C5FD);--pd-fc-bg:light-dark(#EFF6FF,rgba(96,165,250,.12));--pd-fc-border:light-dark(#3B82F6,#60A5FA);
    --pd-est-fg:light-dark(#334155,#CBD5E1);--pd-est-bg:light-dark(#F1F5F9,#1A2439);
    --pd-est-stripe:light-dark(rgba(100,116,139,.22),rgba(148,163,184,.22));
    --pd-ex-fg:light-dark(#92400E,#FCD34D);--pd-ex-bg:light-dark(#FFFBEB,rgba(245,158,11,.12));
    --pd-ex-border:light-dark(rgba(217,119,6,.55),rgba(245,158,11,.45));
  }
}
"""

# Streamlit 1.52.1 DOM(stHeader 60px, stMainBlockContainer)에 기대는 규칙이다. 버전을 올리면 명세 9.4절 E1·E7을 다시 돌린다.
PD_CSS_BASE = """
/* ---- 2. 바탕·헤더·글꼴 ---- */
[data-testid="stApp"]{background:var(--pd-canvas);word-break:keep-all;overflow-wrap:anywhere}
[data-testid="stHeader"]{background:transparent;pointer-events:none}
[data-testid="stHeader"] button{pointer-events:auto;border-radius:8px;background:color-mix(in srgb,var(--pd-canvas) 88%,transparent)}
[data-testid="stMainBlockContainer"]{max-width:1440px;padding:.5rem 2rem 3rem}
@media (max-width:640px){[data-testid="stMainBlockContainer"]{padding:.5rem 1rem 2.5rem}}
.pd-num,[data-testid="stPlotlyChart"] text{font-variant-numeric:tabular-nums}
:focus-visible{outline:2px solid var(--pd-focus);outline-offset:2px}
/* 탭·본문 스크롤 칸은 Streamlit 규칙 '.st-…:focus{outline:none}'(0,2,0)이 위 전역 규칙(0,1,0)을 이겨 키보드 초점이
   보이지 않았다(1차 평가: 탭 outline none, WCAG 2.4.7). 탭 줄은 가로 스크롤 칸이라 바깥 테두리가 잘리므로 안쪽(-2px)에 그린다 */
[data-testid="stTab"]:focus-visible,[data-testid="stMain"]:focus-visible{outline:2px solid var(--pd-focus)!important;outline-offset:-2px}
[data-testid="stTab"]:focus-visible{border-radius:8px}
/* 헤더 »·⋮ 버튼의 기본 초점 고리(rgba .2)는 캔버스 대비 1.5:1(다크 1.04:1)이라 안 보였다. 이 둘이 사이드바·테마 메뉴의 유일한 입구다 */
[data-testid="stHeader"] button:focus-visible{box-shadow:0 0 0 2px var(--pd-focus)!important}
/* 토글은 input이 0×0이라 전역 테두리가 보이지 않는다 → 보이는 label에 테두리를 준다 */
[data-testid="stCheckbox"] label:has(input:focus-visible){outline:2px solid var(--pd-focus);outline-offset:2px;border-radius:6px}
/* 표 도구 막대(열 보기·CSV·검색·전체 화면) 버튼에 초점이 있는 동안 막대가 투명하게 남지 않게 하고, 22px 버튼이 간격 없이
   붙어 있어 누르기 대상 최소 24px(WCAG 2.5.8)에 못 미쳤으므로 24px로 키운다 */
[data-testid="stElementContainer"]:focus-within [data-testid="stElementToolbar"]{opacity:1!important}
[data-testid="stBaseButton-elementToolbar"]{min-width:24px;min-height:24px}
/* 접힌 사이드바는 화면 밖(-256px)에 있어도 자동 갱신 토글이 첫 Tab 자리를 차지해 초점이 사라졌고, 그 자리에서 Space를 누르면
   자동 갱신이 조용히 꺼졌다(1차 평가 실측). 접힌 동안만 탭 순서에서 뺀다(열면 aria-expanded="true"라 바로 보인다) */
[data-testid="stSidebar"][aria-expanded="false"]{visibility:hidden}
/* st.markdown 문단·목록은 Streamlit 규칙 '.st-emotion-cache-… p{word-break:break-word}'가 stApp의 keep-all을 덮어
   '학/습한'·'오른/쪽'처럼 낱말 가운데서 끊겼다(1차 평가 계산 스타일 확인). 긴 토막은 overflow-wrap이 계속 끊어 준다 */
[data-testid="stMarkdownContainer"] :is(p,li){word-break:keep-all!important}
/* expander 제목·토글 이름이 Streamlit 기본 16px 보통 굵기라 카드 제목(15px 굵게)보다 크게 보여 위계가 뒤집혔다 → 본문 단계 */
[data-testid="stExpander"] summary p{font-size:14px;font-weight:600;color:var(--pd-text-2)}
[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p{font-size:14px}
@media (prefers-reduced-motion:reduce){.pd-fresh-dot{animation:none!important}}
"""

# 카드·KPI·칩·반응형 행·탭. 높이 연쇄·:has() 행 선택자·data-baseweb 탭 선택자는 1.52.1 DOM 전제다(명세 10장 업그레이드 체크리스트).
# 탭 라벨의 :material: 아이콘은 1.52.1에서 stIconMaterial이 아니라 인라인 display를 가진 span[role="img"]라
# 명세의 stIconMaterial 선택자로는 숨겨지지 않는다(0단계 E7 실측). 그래서 span[role="img"]를 !important로 숨긴다.
PD_CSS_COMPONENTS = """
/* ---- 3. 카드 ---- */
.st-key-map_stage,[class*="st-key-card_"]{background:var(--pd-surface);border-color:var(--pd-border)!important;border-radius:12px;box-shadow:0 1px 2px var(--pd-shadow-c);padding:16px 20px!important}
.st-key-map_stage{border-radius:14px;padding:16px!important}
@media (max-width:640px){.st-key-map_stage,[class*="st-key-card_"]{padding:12px!important}}
/* 카드·expander 안 st.markdown(주파수·예시·모델 카드)은 마크다운 컨테이너의 아래 여백 -16px가 카드 안쪽 여백을 지워 글자·칸이
   테두리에 닿았다(1차 평가: 390 주파수 카드 0.6px 넘침, 모델 칸 1px, 입찰 예시 점선 상자). 4절 KPI 줄과 같은 처리 */
[data-testid="stMarkdownContainer"]:has(> .pd-card-sim, > .pd-example, > .pd-model){margin-bottom:0}
/* ---- 4. KPI: 같은 높이 연쇄 (시안 A 실험 연쇄. code_audit D6 안은 stMarkdown 안쪽 div 때문에 맞지 않았다) ---- */
.st-key-kpi_row [data-testid="stHorizontalBlock"]{align-items:stretch;gap:16px}
.st-key-kpi_row [data-testid="stColumn"]>[data-testid="stVerticalBlock"],
.st-key-kpi_row [data-testid="stElementContainer"],
.st-key-kpi_row [data-testid="stMarkdown"],
.st-key-kpi_row [data-testid="stMarkdown"]>div,
.st-key-kpi_row [data-testid="stMarkdownContainer"]{height:100%}
.st-key-kpi_row [data-testid="stMarkdownContainer"]{margin-bottom:0}
@media (max-width:1180px){
  .st-key-kpi_row [data-testid="stHorizontalBlock"]{display:grid!important;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
  .st-key-kpi_row [data-testid="stColumn"]{width:auto!important;min-width:0!important;flex:none!important}
}
@media (max-width:640px){.st-key-kpi_row [data-testid="stHorizontalBlock"]{gap:8px}}
.pd-kpi{position:relative;height:100%;display:flex;flex-direction:column;gap:6px;padding:14px 16px;background:var(--pd-surface);border:1px solid var(--pd-border);border-radius:12px;box-shadow:0 1px 2px var(--pd-shadow-c);color:var(--pd-text-1)}
.pd-kpi__head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:6px}
.pd-kpi__label{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:600;color:var(--pd-text-2)}
.pd-kpi__value{display:flex;align-items:baseline;gap:5px;white-space:nowrap}
.pd-kpi__value .pd-num{font-size:30px;font-weight:700;line-height:1.15}
.pd-kpi__unit{font-size:14px;font-weight:500;color:var(--pd-text-3)}
.pd-kpi__sub{font-size:13px;color:var(--pd-text-2)}
.pd-kpi__meta{margin-top:auto;padding-top:8px;border-top:1px solid var(--pd-border);display:flex;flex-direction:column;gap:2px;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
.pd-kpi.is-est .pd-kpi__value .pd-num,.pd-sim-value .pd-num{color:var(--pd-text-2)}
.pd-kpi.is-stale .pd-kpi__meta{color:var(--pd-warn-fg)}
.pd-kpi.is-est::before,.pd-card-sim::before{content:"";position:absolute;left:0;top:12px;bottom:12px;width:3px;border-radius:2px;background:repeating-linear-gradient(180deg,var(--pd-border-strong) 0 3px,transparent 3px 6px)}
.pd-card-sim{position:relative;padding-left:10px}
/* 640px 이하 2×2에서는 칩을 늘 라벨 아래 줄로 내린다(명세 4.1절 390). 라벨이 긴 카드만 칩이 접히면 같은 행의 값 높이가
   27px 어긋났다(1차 평가: 현재 수요 244 · SMP 271) */
@media (max-width:640px){.pd-kpi{padding:12px}.pd-kpi__value .pd-num{font-size:24px}.pd-kpi__head{flex-direction:column;align-items:flex-start;gap:4px}}
/* ---- 5. 칩·pill·글리프 ---- */
.pd-chip{display:inline-flex;align-items:center;gap:5px;min-height:22px;padding:0 8px;border-radius:999px;border:1px solid var(--pd-meas-border);font-size:12px;font-weight:600;line-height:1.2;color:var(--pd-text-2);white-space:nowrap}
.pd-chip--plain{border-color:var(--pd-border);color:var(--pd-text-2)}
.pd-chip__glyph,.pd-glyph{flex:none;display:inline-block}
.pd-chip--measured .pd-chip__glyph,.pd-glyph--measured{width:7px;height:7px;border-radius:50%;background:currentColor}
.pd-chip--forecast{color:var(--pd-fc-fg);background:var(--pd-fc-bg);border:1px dashed var(--pd-fc-border)}
.pd-chip--forecast .pd-chip__glyph,.pd-glyph--forecast{width:14px;border-top:2px dashed currentColor}
.pd-chip--est{color:var(--pd-est-fg);border:1.5px dotted var(--pd-border-strong);background:repeating-linear-gradient(135deg,var(--pd-est-stripe) 0 2px,transparent 2px 6px),var(--pd-est-bg)}
.pd-chip--est .pd-chip__glyph,.pd-glyph--est{width:12px;height:8px;border:1.5px dotted currentColor;border-radius:2px}
.pd-chip--example{color:var(--pd-ex-fg);background:var(--pd-ex-bg);border:1px dashed var(--pd-ex-border)}
.pd-chip--example .pd-chip__glyph,.pd-glyph--example{width:12px;height:8px;border:1.5px dashed currentColor;border-radius:2px}
.pd-chip--stale{color:var(--pd-warn-fg);background:var(--pd-warn-bg);border-color:transparent}
.pd-pill{display:inline-flex;align-items:center;gap:6px;min-height:24px;padding:0 10px;border-radius:999px;font-size:12px;font-weight:700;white-space:nowrap}
.pd-pill__dot{width:7px;height:7px;border-radius:50%;background:currentColor}
.status-online{color:var(--pd-ok-fg);background:var(--pd-ok-bg)}
.status-warning{color:var(--pd-warn-fg);background:var(--pd-warn-bg)}
.status-danger{color:var(--pd-danger-fg);background:var(--pd-danger-bg)}
.status-critical{color:var(--pd-crit-fg);background:var(--pd-crit-bg)}
.pd-pill.is-est{color:var(--pd-est-fg);background:repeating-linear-gradient(135deg,var(--pd-est-stripe) 0 2px,transparent 2px 6px),var(--pd-est-bg)}
.pd-delta{font-size:13px;font-weight:600;color:var(--pd-text-2)} /* 오름·내림에 빨강·초록을 쓰지 않는다 */
.pd-ico{display:inline-block;width:16px;height:16px;flex:none;background-color:currentColor;-webkit-mask:var(--ico) center/contain no-repeat;mask:var(--ico) center/contain no-repeat}
/* ---- 6. 반응형 행 (컬럼을 품은 행은 :has로 정확히 고른다. 하위 st.columns에는 걸리지 않는다) ---- */
@media (max-width:1180px){
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-map_col),
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_smp),
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_mix){flex-wrap:wrap}
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-map_col) > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_smp) > [data-testid="stColumn"],
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_mix) > [data-testid="stColumn"]{width:100%!important;flex:1 1 100%!important;min-width:0!important}
  .st-key-map_col iframe{height:440px!important}
}
/* 상태 줄(수급 게이지 · 계통 상태 · 주파수)은 모든 폭에서 격자다. st.columns([5, 4, 3]) 그대로 3열이면 세 카드가 게이지
   카드 높이(336px)로 늘어나 계통 상태 164px·주파수 220px가 비었고(1차 평가 1440 실측), 1320px 미만에서는 게이지 폭이
   220px 아래로 내려갔다(S4 실측 1280px 214px). 그래서 1181px 이상은 게이지 7 | 오른쪽 5에 계통 상태와 주파수를 위아래로
   쌓는다(게이지 카드가 두 행에 걸쳐 두 카드가 남는 높이를 나눠 가진다, 1181px에서 게이지 약 296px). 1180px 이하는
   게이지 한 행 + 두 카드 한 행, 640px 이하는 1열 */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_grid){display:grid!important;grid-template-columns:minmax(0,7fr) minmax(0,5fr);gap:16px}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_grid) > [data-testid="stColumn"]{width:auto!important;min-width:0!important;flex:none!important}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_grid) > [data-testid="stColumn"]:first-child{grid-row:span 2}
@media (max-width:1180px){
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_grid){grid-template-columns:1fr 1fr}
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_grid) > [data-testid="stColumn"]:first-child{grid-row:auto;grid-column:1 / -1}
}
@media (max-width:1023px){
  .st-key-ex_zone [data-testid="stHorizontalBlock"]{flex-wrap:wrap}
  .st-key-ex_zone [data-testid="stColumn"]{width:100%!important;flex:1 1 100%!important}
}
@media (max-width:640px){
  [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_grid){grid-template-columns:1fr}
  .st-key-map_col iframe{height:340px!important}
}
/* ---- 7. 탭 ---- */
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:4px;border-bottom:1px solid var(--pd-border)}
[data-testid="stTab"]{padding:10px 12px}
[data-testid="stTab"] p{font-size:15px;font-weight:600;color:var(--pd-text-3)}
[data-testid="stTab"][aria-selected="true"] p{color:var(--pd-primary)}
@media (max-width:640px){[data-testid="stTab"] p{font-size:14px}[data-testid="stTab"] span[role="img"]{display:none!important}}
/* ---- 8. 머리: 앱 바·경보 배너·상태 문장·읽는 법 띠·카드 머리 (2단계. 3장 토큰·타입 스케일만 쓴다) ---- */
/* 앱 바는 투명 헤더 줄(60px)과 겹친다. 오른쪽 124px는 fragment 실행 표시(stStatusWidget)와 메뉴 자리다(0단계 E1 실측) */
.pd-appbar{display:flex;align-items:center;justify-content:space-between;gap:12px;min-height:48px;padding:0 124px 0 24px;color:var(--pd-text-1)}
.pd-appbar p{margin:0}
.pd-appbar__brand{display:flex;align-items:center;gap:10px;flex:none}
/* 로고 바탕은 3.3절 흰 글자 쌍(#2563EB 위 흰 글자 5.17:1)이라 두 테마가 같은 값을 쓴다 */
.pd-appbar__logo{display:inline-flex;align-items:center;justify-content:center;flex:none;width:32px;height:32px;border-radius:8px;background:#2563EB;color:#FFFFFF}
.pd-appbar__logo .pd-ico{width:18px;height:18px}
.pd-appbar__title{font-size:17px;font-weight:700;line-height:1.2;white-space:nowrap;color:var(--pd-text-1)}
.pd-appbar__sub{font-size:12px;line-height:1.45;white-space:nowrap;color:var(--pd-text-3)}
.pd-appbar__meta{display:flex;flex-wrap:wrap;align-items:center;justify-content:flex-end;gap:6px;min-width:0}
.pd-appbar__short{display:none}
@media (max-width:1180px){.pd-appbar__sub,.pd-appbar__long{display:none}.pd-appbar__short{display:inline}}
@media (max-width:640px){
  .pd-appbar{flex-direction:column;align-items:stretch;gap:6px;padding:0}
  .pd-appbar__brand{min-height:44px;padding:0 40px}
  .pd-appbar__meta{justify-content:flex-start}
  .pd-appbar__auto{display:none}
}
/* 칩·pill은 기본이 한 줄이다. 좁은 칸에서 긴 출처 문구가 칸 밖으로 넘치지 않게 이 자리들만 줄바꿈을 허용한다.
   모서리 12px는 한 줄(높이 22~24px)일 때 999px와 같은 모양이고, 여러 줄로 접히면 글자가 둥근 끝 밖으로 나가지 않는다 */
.pd-appbar__meta .pd-chip,.pd-kpi__head .pd-chip,.pd-status .pd-chip,.pd-kpi__sub .pd-pill{white-space:normal;padding-top:2px;padding-bottom:2px;line-height:1.3;border-radius:12px}
.pd-alert{display:flex;align-items:center;gap:12px;padding:12px 16px;border-radius:12px;border-left:4px solid currentColor}
.pd-alert p{margin:0}
.pd-alert--warning{color:var(--pd-warn-fg);background:var(--pd-warn-bg)}
.pd-alert--danger{color:var(--pd-danger-fg);background:var(--pd-danger-bg)}
.pd-alert--critical{color:var(--pd-crit-fg);background:var(--pd-crit-bg)}
.pd-alert__icon{width:20px;height:20px}
.pd-alert__body{flex:1 1 auto;min-width:0}
.pd-alert__title{font-size:15px;font-weight:700;line-height:1.35}
.pd-alert__msg{font-size:13px;line-height:1.55;color:var(--pd-text-1)}
.pd-alert__value{flex:none;margin-left:auto}
.pd-alert__value:empty{display:none}
/* 좁은 폭: 아이콘과 문장은 한 줄에 두고(본문 기준폭 0) 칩만 문장 아래로 내린다 */
@media (max-width:640px){.pd-alert{flex-wrap:wrap;align-items:flex-start;padding:12px}.pd-alert__body{flex:1 1 0}.pd-alert__value{flex:1 1 100%;margin-left:32px}}
.pd-status{font-size:15px;font-weight:500;line-height:1.55;color:var(--pd-text-1)}
.pd-status>.pd-pill,.pd-status>.pd-chip{margin-right:8px;vertical-align:1px}
/* Streamlit 전역 규칙 p{font-size:1rem;font-weight:400}이 p에 직접 걸리므로 상속값 대신 p에 직접 준다 */
.pd-status__text{display:inline;margin:0;font-size:15px;font-weight:500}
.pd-status__text b{font-weight:600}
.pd-status__src{display:inline-block;margin-left:12px;font-size:12px;font-weight:500;white-space:nowrap;color:var(--pd-text-3)}
@media (max-width:1180px){.pd-status__src{display:block;margin:2px 0 0;white-space:normal}}
.pd-legend{display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
.pd-legend__title{font-weight:700;color:var(--pd-text-2)}
.pd-legend__item{display:inline-flex;align-items:center;gap:6px}
.pd-legend__item b{font-weight:600;color:var(--pd-text-2)}
.pd-legend .pd-glyph{color:var(--pd-text-2)}
.pd-legend .pd-glyph--forecast{color:var(--pd-fc-fg)}
.pd-legend .pd-glyph--example{color:var(--pd-ex-fg)}
@media (max-width:640px){.pd-legend{gap:4px 12px}.pd-legend__desc,.pd-legend__long{display:none}}
.pd-card__head{display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:6px 10px}
.pd-card__title{display:inline-flex;align-items:center;gap:6px;font-size:15px;font-weight:700;line-height:1.35;color:var(--pd-text-1)}
/* st.markdown 안에서는 Streamlit의 .st-emotion-cache-… p{font-size:inherit}(0,1,1)가 이기므로 컨테이너로 한 단계 높인다 */
.pd-card__note,[data-testid="stMarkdownContainer"] .pd-card__note{margin:0;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
.pd-sim-value{display:flex;align-items:baseline;gap:5px;margin:10px 0 6px;white-space:nowrap}
.pd-sim-value .pd-num{font-size:24px;font-weight:700;line-height:1.15}
/* 3칸 카드 머리(제목·부제·칩, 명세 5.7절). 칩 묶음의 자동 여백이 남는 공간을 먼저 가져가므로 제목·부제는 붙어 있고,
   2칸 머리(제목|칩)는 space-between 그대로 양 끝에 놓인다 */
.pd-card__sub{font-size:13px;font-weight:500;line-height:1.35;color:var(--pd-text-3)}
.pd-card__head:has(> .pd-card__sub){justify-content:flex-start}
.pd-card__chips{display:inline-flex;flex-wrap:wrap;align-items:center;gap:6px;margin-left:auto}
.pd-card__meta{display:flex;align-items:center;gap:6px;margin-top:4px;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
.pd-card__meta .pd-ico{width:14px;height:14px}
/* 선택된 세그먼트는 primary 글자가 primary 10% 바탕 위에 놓여 라이트에서 4.49:1로 AA 미달이다(S3 실측). 링크 토큰
   (라이트 #1D4ED8 약 5.8:1, 다크는 primary와 같은 #60A5FA)으로 바꾼다. 지역·SMP 보기·발전 유형 세그먼트 모두에 걸린다 */
[data-testid="stBaseButton-segmented_controlActive"]{color:var(--pd-link)!important}
/* ---- 9. 지도 무대 · 인스펙터 (4단계. iframe 안쪽 CSS는 PD_CSS_MAP) ---- */
/* iframe이 인라인 요소면 기준선 여백 8px가 더해져 지도 아래 간격이 26px가 된다(0단계 E3) */
.st-key-map_col iframe{display:block;border-radius:10px}
@media (min-width:641px){.st-key-map_head [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"]{align-items:flex-end}}
.pd-maplegend{display:flex;flex-wrap:wrap;align-items:center;gap:4px 16px;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
.pd-maplegend__item{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.pd-maplegend__item b{font-weight:600;color:var(--pd-text-2)}
.pd-maplegend__note{flex-basis:100%}
/* 지도 건너뛰기 링크: 평소에는 보이지 않고(화면 읽기 프로그램은 읽는다) 키보드 초점을 받으면 머리 아래에 링크 칩으로 나타난다 */
.pd-skip{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
.pd-skip:focus-visible{position:static;display:inline-flex;width:auto;height:auto;clip-path:none;margin-top:6px;padding:2px 10px;border-radius:999px;background:var(--pd-surface);font-size:12px;font-weight:600;color:var(--pd-link)}
/* 지도 기호와 같은 모양(원·사각·마름모·삼각) + 인라인 --c 계열 색. 카드 면 위 채움 대비는 3.4절 card 열(최저 3.64) */
.pd-mkey{flex:none;display:inline-block;width:11px;height:11px;background:var(--c)}
.pd-mkey--wind{border-radius:50%}
.pd-mkey--solar{border-radius:2px}
.pd-mkey--ess{width:9px;height:9px;margin:0 1px;transform:rotate(45deg)}
.pd-mkey--thermal{width:12px;clip-path:polygon(50% 0,100% 100%,0 100%)}
.pd-insp,.pd-insp__side,.pd-insp__main{display:flex;flex-direction:column;gap:12px;min-width:0;color:var(--pd-text-2)}
.pd-insp__crumb{display:flex;flex-wrap:wrap;align-items:center;gap:2px 6px;font-size:13px;font-weight:600;line-height:1.45;color:var(--pd-text-2)}
.pd-insp__crumb .pd-ico{width:14px;height:14px;color:var(--pd-text-3)}
.pd-insp__crumb b{font-weight:700;color:var(--pd-text-1)}
.pd-insp__ref{margin-left:auto;font-size:12px;font-weight:500;color:var(--pd-text-3)}
.pd-insp__notice{margin:0;padding:8px 10px;border-radius:10px;font-size:12px;font-weight:600;line-height:1.45;color:var(--pd-warn-fg);background:var(--pd-warn-bg)}
.pd-insp__stats{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.pd-stat{display:flex;flex-direction:column;gap:2px;min-width:0;padding:10px 12px;border-radius:10px;background:var(--pd-well)}
.pd-stat__label{font-size:12px;font-weight:600;line-height:1.35;color:var(--pd-text-2)}
.pd-stat__value{font-size:20px;font-weight:700;line-height:1.2;white-space:nowrap;color:var(--pd-text-1)}
.pd-stat__value small{margin-left:3px;font-size:13px;font-weight:500;color:var(--pd-text-3)}
.pd-stat__value--text{font-size:13px;font-weight:600;line-height:1.45;white-space:normal}
.pd-stat__src{font-size:12px;line-height:1.4;color:var(--pd-text-3)}
.pd-insp__table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.4;font-variant-numeric:tabular-nums;color:var(--pd-text-2)}
.pd-insp__table th{padding:6px 8px;font-size:12px;font-weight:600;text-align:left;white-space:nowrap;color:var(--pd-text-3);background:var(--pd-well)}
.pd-insp__table th:first-child{border-radius:8px 0 0 8px}
.pd-insp__table th:last-child{border-radius:0 8px 8px 0}
.pd-insp__table td{padding:6px 8px;border-bottom:1px solid var(--pd-border);vertical-align:middle}
.pd-insp__table .pd-num{text-align:right;white-space:nowrap}
.pd-insp__table tr[aria-current="true"] td{font-weight:700;color:var(--pd-primary)}
.pd-insp__child td:first-child{padding-left:12px;white-space:nowrap}
/* 트리 기호는 화면 읽기 프로그램이 읽지 않게 대체 글자를 비운다(미지원 브라우저는 선언을 버리고 들여쓰기만 남는다) */
.pd-insp__child td:first-child::before{content:"└ " / "";color:var(--pd-text-3)}
.pd-insp__sect td{padding:12px 8px 4px;font-size:12px;font-weight:700;color:var(--pd-text-3)}
.pd-insp__note{font-size:12px;color:var(--pd-text-3)}
/* 지도 밖 자원 행의 덧말(위치 없음 · 송전량 실측 미연동)은 640px 이하에서 4번째 칸 대신 이름 아래 둘째 줄로 보인다 */
.pd-insp__rowsub{display:none}
.pd-bar{display:flex;width:100%;min-width:56px;height:8px;border-radius:4px;overflow:hidden;background:var(--pd-well)}
.pd-bar>span{display:block;height:100%;background:var(--c)}
.pd-insp__foot{margin:0;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
/* 무대가 세로로 쌓이는 폭(641~1180px)에서는 인스펙터를 2단(왼쪽 경로·통계, 오른쪽 지역 표)으로 */
@media (max-width:1180px) and (min-width:641px){.pd-insp{display:grid;grid-template-columns:1fr 1.3fr;align-items:start;gap:16px}}
/* 통계는 1180px 이하 모든 폭에서 2칸 + 가로 전체 1칸이다: 2단 인스펙터의 왼쪽 단이 좁아 3칸이면 칸이 71~89px가 되어
   '974.2 MW'가 옆 칸으로 넘쳤다(1차 평가: 768px 7px, 641px 25px) */
@media (max-width:1180px){
  .pd-insp__stats{grid-template-columns:repeat(2,minmax(0,1fr))}
  .pd-insp__stats .pd-stat:last-child{grid-column:1 / -1}
}
@media (max-width:640px){
  .pd-stat__value{font-size:17px}
  /* 구성 열과 덧말 칸을 함께 숨겨 모든 행이 3칸이 되게 한다(머리 띠·행 선이 같은 폭에서 끝난다) */
  .pd-insp__table .pd-insp__mix,.pd-insp__table td.pd-insp__note{display:none}
  .pd-insp__table th:nth-last-child(2){border-radius:0 8px 8px 0}
  .pd-insp__rowsub{display:block;padding-left:14px;font-size:12px;font-weight:400;line-height:1.35;white-space:normal;color:var(--pd-text-3)}
}
/* ---- 10. Plotly 글자: 테마 전환을 따라가게 페이지 토큰으로 칠한다 (명세 9.4절 E4 대체안) ----
   1.52.1은 '#000037' 같은 자리표시자를 그림을 처음 그릴 때만 테마 색으로 바꾼다. ⋮ → Settings로 테마를 바꾸면 이미 색이
   들어간 그림에 다시 적용하므로 글자가 이전 테마 색으로 남는다(라이트→다크에서 어두운 바탕 위 #262730, S3 실측).
   그림 바탕은 템플릿이 다시 칠해 따라오므로 글자만 토큰으로 맞춘다. 바탕을 칠한 주석(흰 글자 pill·카드색 상자)과
   도넛 조각 글자는 건드리지 않는다(조각 안 흰 글자 대체안을 덮지 않게) */
[data-testid="stPlotlyChart"] :is(.xtick,.ytick,.g-xtitle,.g-ytitle,.xangularaxistick) text,
[data-testid="stPlotlyChart"] .indicatorlayer text.title{fill:var(--pd-text-2)!important}
[data-testid="stPlotlyChart"] .hoverlayer text,
[data-testid="stPlotlyChart"] .indicatorlayer text.number,
[data-testid="stPlotlyChart"] .annotation-text-g:has(rect.bg[style*="fill-opacity: 0;"]) text.annotation-text{fill:var(--pd-text-1)!important}
/* ---- 11. 수급 추이 카드 · 상태 줄 (5단계, 명세 5.7·5.9절. 3장 토큰과 인라인 --c 계열 색만 쓴다) ---- */
/* 범례 겸 표: 칸마다 견본 · 이름 · 읽는 법 | 현재값 · 값 출처. 1180px 이하 2칸, 640px 이하 1칸 */
.pd-legend-table{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin:0;padding:0;list-style:none}
.pd-legend-table__cell{display:flex;align-items:center;gap:10px;min-width:0;margin:0;padding:8px 10px;border-radius:10px;background:var(--pd-well);font-size:13px}
.pd-legend-table__txt{display:flex;flex-direction:column;gap:1px;min-width:0;line-height:1.35}
.pd-legend-table__name{font-size:13px;font-weight:600;color:var(--pd-text-1)}
.pd-legend-table__desc{font-size:12px;color:var(--pd-text-3)}
.pd-legend-table__val{display:flex;flex-direction:column;align-items:flex-end;margin-left:auto;font-size:14px;font-weight:700;line-height:1.35;text-align:right;white-space:nowrap;color:var(--pd-text-1)}
.pd-legend-table__val small{font-size:12px;font-weight:500;color:var(--pd-text-3)}
@media (max-width:1180px){.pd-legend-table{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:640px){.pd-legend-table{grid-template-columns:1fr}}
/* 견본 26px: 차트와 같은 선 모양(점선·대시)과 현재 점 모양(채운 원 = 지금 잰 값, 빈 원 = 실측 아님) */
.pd-swatch{position:relative;flex:none;display:inline-block;width:26px;height:14px;border-radius:3px}
.pd-swatch--line::before,.pd-swatch--dash::before{content:"";position:absolute;left:0;right:0;top:6px;border-top:2px dotted var(--c)}
.pd-swatch--dash::before{border-top-style:dashed}
.pd-swatch--line::after{content:"";position:absolute;left:8px;top:2px;width:10px;height:10px;box-sizing:border-box;border-radius:50%;background:var(--c);box-shadow:0 0 0 2px var(--pd-well)}
.pd-swatch--line.is-open::after{background:var(--pd-well);border:2px solid var(--c)}
.pd-swatch--hatch{background:repeating-linear-gradient(135deg,color-mix(in srgb,var(--c) 55%,transparent) 0 2px,transparent 2px 5px);border:1px solid color-mix(in srgb,var(--c) 60%,transparent)}
/* 시간 띠: 차트 아래에서 지난·지금·다음 구간의 뜻을 글로 밝힌다(C·EM 스크러버 이식). 640px 이하는 3줄 */
.pd-timestrip{display:grid;grid-template-columns:minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:8px;font-size:12px;line-height:1.4}
.pd-timestrip__seg{display:flex;align-items:center;justify-content:center;gap:6px;min-height:26px;padding:3px 10px;border-radius:8px;border:1.5px dotted var(--pd-border-strong);background:repeating-linear-gradient(135deg,var(--pd-est-stripe) 0 2px,transparent 2px 6px),var(--pd-est-bg);color:var(--pd-est-fg);text-align:center}
.pd-timestrip__now{display:inline-flex;align-items:center;justify-content:center;gap:6px;font-weight:700;white-space:nowrap;color:var(--pd-text-1)}
.pd-timestrip__dot{flex:none;width:9px;height:9px;box-sizing:border-box;border-radius:50%;background:currentColor}
.pd-timestrip__dot.is-open{background:transparent;border:1.5px solid currentColor}
@media (max-width:640px){.pd-timestrip{grid-template-columns:1fr}}
.pd-card__foot{margin:0;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
/* 수급 차트의 바탕 없는 주석(현재값·단위)에 카드 면 색 테두리를 둘러 점선이 지나가도 읽히게 한다.
   바탕 상자('#000038')는 테마 전환 뒤 이전 테마 색으로 남으므로(S3 실측) 쓰지 않고 토큰으로 칠한다 */
.st-key-card_supply [data-testid="stPlotlyChart"] .annotation-text-g:has(rect.bg[style*="fill-opacity: 0;"]) text.annotation-text{paint-order:stroke;stroke:var(--pd-surface);stroke-width:4px;stroke-linejoin:round}
/* 상태 줄 카드 3장은 같은 행에서 높이를 맞춘다(열 → 세로 블록 → 레이아웃 래퍼 → 카드, 1.52.1 DOM 실측).
   1180px 이하 격자(6절)에서도 같은 행의 두 카드가 같은 높이가 된다 */
.st-key-status_row [data-testid="stColumn"]>[data-testid="stVerticalBlock"]>[data-testid="stLayoutWrapper"]{flex:1 1 auto;display:flex;flex-direction:column}
.st-key-status_row [data-testid="stLayoutWrapper"]>[class*="st-key-card_"]{flex:1 1 auto}
/* 게이지 그림 폭 상한 340px(가운데 정렬): 높이가 190px로 고정이라(명세 6.4절) 폭이 약 350px를 넘으면 호가 높이에
   맞춰 커져 꼭대기 눈금 글자가 그림 위로 잘린다(S4 실측: 1024px에서 451px 폭일 때 '60'이 svg 밖). 폭으로 크기가 정해지게 둔다 */
.st-key-card_gauges [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]){width:100%;max-width:340px;margin-inline:auto}
/* 게이지 이름표: 그림 안 제목 대신 이름과 읽는 법을 HTML로 둔다(명세 6.4절 title='') */
.pd-gauge-label{display:flex;flex-direction:column;align-items:center;gap:2px;text-align:center}
.pd-gauge-label__name{font-size:13px;font-weight:600;line-height:1.35;color:var(--pd-text-2)}
.pd-gauge-label__how{font-size:12px;line-height:1.4;color:var(--pd-text-3)}
/* 계통 상태: 예비율 단계 사다리(순서 척도라 4칸 같은 폭) + 다음 단계까지 여유 MW */
.pd-grid-state{display:flex;flex-direction:column;gap:12px;min-width:0;color:var(--pd-text-2)}
.pd-grid-state p{margin:0}
.pd-ladder{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:4px;margin:0;padding:0;list-style:none}
.pd-ladder__step{display:flex;flex-direction:column;align-items:center;justify-content:center;gap:2px;min-width:0;min-height:52px;margin:0;padding:6px 4px;border:1px solid transparent;border-radius:8px;font-size:13px;font-weight:600;line-height:1.3;text-align:center}
.pd-ladder__step small{font-size:12px;font-weight:500;white-space:nowrap}
/* 현재 칸만 단계 바탕·글자 + 테두리. 나머지는 well 바탕 text-3(상태 클래스보다 앞서게 선택자를 한 단계 높인다) */
.pd-ladder .pd-ladder__step:not(.is-current){background:var(--pd-well);color:var(--pd-text-3)}
.pd-ladder__step.is-current{border-color:currentColor;font-weight:700}
.pd-grid-state__value{font-size:14px;line-height:1.5;color:var(--pd-text-2)}
.pd-grid-state__value b{font-size:20px;font-weight:700;color:var(--pd-text-1)}
@media (max-width:640px){.pd-grid-state__value b{font-size:17px}}
/* ---- 12. SMP 예측 · 발전 현황 · 분석 탭 (6단계, 명세 4.3~4.5·5.10·5.11절. 3장 토큰과 인라인 --c만 쓴다) ---- */
/* 카드 머리 칩도 좁은 칸에서 긴 출처 문구가 칸 밖으로 넘치지 않게 줄바꿈을 허용한다(8절 앱 바·KPI와 같은 처리) */
.pd-card__chips .pd-chip{white-space:normal;padding-top:2px;padding-bottom:2px;line-height:1.3;border-radius:12px}
.pd-smp-explain{margin-top:6px!important}
/* 바탕 없는 차트 주석('평균'·'확정'·단위)에 카드 면 색 테두리를 둘러 계단선·막대가 지나가도 읽히게 한다(11절과 같은 처리) */
:is(.st-key-card_smp,.st-key-card_timeline) [data-testid="stPlotlyChart"] .annotation-text-g:has(rect.bg[style*="fill-opacity: 0;"]) text.annotation-text{paint-order:stroke;stroke:var(--pd-surface);stroke-width:4px;stroke-linejoin:round}
/* 테마 전환: 1.52.1은 '#000036'·'#000037'·'#000038' 자리표시자를 그림을 처음 그릴 때만 바꾼다(S3 실측). SMP 그림은 한 시간
   안에서, 도넛·타임라인은 5분 안에서 그림이 같아 다시 그려지지 않으므로 음영·평균선·현재 상자·점 테두리·조각 경계를
   페이지 토큰으로 칠해 전환을 바로 따라가게 한다(text-3·text-1은 두 테마 카드 위 그래픽 3:1 이상) */
.st-key-card_smp [data-testid="stPlotlyChart"] .shapelayer path{fill:var(--pd-text-3)!important;stroke:var(--pd-text-3)!important}
.st-key-card_smp [data-testid="stPlotlyChart"] .scatterlayer .point{stroke:var(--pd-surface)!important}
.st-key-card_timeline [data-testid="stPlotlyChart"] .shapelayer path{stroke:var(--pd-text-1)!important}
.st-key-card_mix [data-testid="stPlotlyChart"] .slice path.surface{stroke:var(--pd-surface)!important}
/* SMP 통계: 이름 | 값 + 단위 */
.pd-statlist{display:flex;flex-direction:column;margin:8px 0 6px;padding:0}
.pd-statlist__row{display:flex;align-items:baseline;justify-content:space-between;gap:12px;padding:7px 0;border-bottom:1px solid var(--pd-border)}
.pd-statlist__row:last-child{border-bottom:0}
.pd-statlist dt{display:flex;flex-wrap:wrap;align-items:center;gap:4px 6px;margin:0;font-size:13px;font-weight:500;line-height:1.4;color:var(--pd-text-2)}
.pd-statlist dd{margin:0;white-space:nowrap;text-align:right}
.pd-statlist dd b{font-size:15px;font-weight:700;color:var(--pd-text-1)}
.pd-statlist dd small,.pd-mini-table small{margin-left:4px;font-size:12px;font-weight:500;color:var(--pd-text-3)}
/* 작은 표(시간대 블록·공급 구성): 머리 well 바탕, 숫자 열 오른쪽 정렬 */
.pd-mini-table{width:100%;margin:8px 0 6px;border-collapse:collapse;font-size:13px;line-height:1.4;font-variant-numeric:tabular-nums;color:var(--pd-text-2)}
.pd-mini-table th{padding:6px 8px;font-size:12px;font-weight:600;text-align:right;white-space:nowrap;color:var(--pd-text-3);background:var(--pd-well)}
.pd-mini-table th:first-child{text-align:left;border-radius:8px 0 0 8px}
.pd-mini-table th:last-child{border-radius:0 8px 8px 0}
.pd-mini-table td{padding:7px 8px;border-bottom:1px solid var(--pd-border);text-align:right;white-space:nowrap}
.pd-mini-table td:first-child{text-align:left}
.pd-mini-table tr:last-child td{border-bottom:0}
.pd-mini-table b{font-weight:600;color:var(--pd-text-1)}
.pd-smp-model__name{margin:8px 0 4px;font-size:13px;font-weight:600;line-height:1.5;color:var(--pd-text-1)}
/* 공급 구성 표 견본: 도넛 조각과 같은 색 + 같은 방향 빗금(흰 45%는 도넛 pattern.fgcolor와 같은 값이다) */
.pd-mix-name{display:inline-flex;align-items:center;gap:8px;font-weight:600;color:var(--pd-text-1)}
.pd-swatch[class*="pd-swatch--hatch-"]{width:18px;height:14px;background-color:var(--c)}
.pd-swatch--hatch-slash{background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.45) 0 1.5px,transparent 1.5px 5px)}
.pd-swatch--hatch-backslash{background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.45) 0 1.5px,transparent 1.5px 5px)}
.pd-swatch--hatch-cross{background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.45) 0 1.5px,transparent 1.5px 5px),repeating-linear-gradient(135deg,rgba(255,255,255,.45) 0 1.5px,transparent 1.5px 5px)}
.pd-swatch--hatch-dot{background-image:radial-gradient(rgba(255,255,255,.6) 1px,transparent 1.4px);background-size:5px 5px}
/* 시간대별 발전량 범례 견본: 막대와 같은 모양(계열 35% 면 + 같은 색 빗금 + 1px 테두리)과 현재 시각 점선 상자 */
.pd-swatch--bar{width:18px;height:14px;border:1px solid var(--c);background-color:color-mix(in srgb,var(--c) 35%,transparent)}
.pd-swatch--bar-slash{background-image:repeating-linear-gradient(135deg,color-mix(in srgb,var(--c) 70%,transparent) 0 1.5px,transparent 1.5px 5px)}
.pd-swatch--bar-backslash{background-image:repeating-linear-gradient(45deg,color-mix(in srgb,var(--c) 70%,transparent) 0 1.5px,transparent 1.5px 5px)}
.pd-swatch--now{width:14px;height:14px;border:1.2px dotted var(--pd-text-1);border-radius:2px}
/* 발전 현황 첫 행(도넛 5 · 타임라인 7)은 같은 행에서 카드 높이를 맞춘다(11절 상태 줄과 같은 연쇄). 타임라인의 '표로 보기'를
   펼치면(표 24행 878px) 도넛 카드까지 늘어나 빈 카드가 되므로, 행 안에 펼친 expander가 있으면 맞추지 않는다 */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_mix):not(:has(details[open])) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]{flex:1 1 auto;display:flex;flex-direction:column}
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-card_mix):not(:has(details[open])) [data-testid="stLayoutWrapper"] > [class*="st-key-card_"]{flex:1 1 auto}
/* 도넛 카드: 641px 이상에서는 도넛 | 구성 표를 나란히 둔다(머리는 두 칸에 걸침). 세로로 쌓으면 카드가 타임라인 카드보다
   약 180px 길어져 같은 행 높이 맞춤이 타임라인 카드 아래에 빈 칸을 만들었다(1440 실측 583px 대 약 400px) */
@media (min-width:641px){
  .st-key-card_mix{display:grid!important;grid-template-columns:minmax(0,1fr) minmax(0,1fr);column-gap:16px;row-gap:12px;align-items:center;align-content:start}
  .st-key-card_mix>[data-testid="stElementContainer"]:first-child{grid-column:1 / -1;align-self:start}
}
.pd-mix-table td.pd-num{white-space:normal}
/* 등록부 표 위 요약 줄(전역 p{font-size:1rem}이 p에 직접 걸리므로 p에 크기를 준다) */
.pd-plant-summary{display:flex;flex-direction:column;gap:4px;line-height:1.5;color:var(--pd-text-2)}
.pd-plant-summary p{margin:0;font-size:13px}
.pd-plant-summary__warn{display:flex;align-items:flex-start;gap:6px}
.pd-plant-summary__warn .pd-ico{margin-top:2px;color:var(--pd-text-3)}
/* 등록부 표 유형 필터: 좁은 폭에서 버튼이 칸 밖으로 넘치지 않게 줄바꿈한다(명세 4.4절 390) */
.st-key-plant_type [data-baseweb="button-group"],.st-key-plant_type [role="radiogroup"]{flex-wrap:wrap}
/* 분석 탭 섹션 머리 · 예시 리본 */
.pd-section-head{display:flex;flex-direction:column;gap:2px}
.pd-section-head p{margin:0}
.pd-section-head__title{font-size:17px;font-weight:700;line-height:1.3;color:var(--pd-text-1)}
.pd-section-head__sub{font-size:13px;line-height:1.45;color:var(--pd-text-3)}
.pd-example-ribbon{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;padding:8px 12px;border-radius:10px;border:1px dashed var(--pd-ex-border);background:var(--pd-ex-bg);font-size:13px;line-height:1.45;color:var(--pd-ex-fg)}
.pd-example-ribbon b{font-weight:700}
/* 분석 탭 모델 카드(st.markdown): 칸 제목은 h4가 아니라 p라 Streamlit 제목 앵커가 붙지 않는다. st.markdown 안의
   .st-emotion-cache-… p{font-size:inherit}(0,1,1)를 이기도록 마크다운 컨테이너로 한 단계 높인다(2단계 메모) */
.pd-model{display:flex;flex-direction:column;gap:10px;color:var(--pd-text-2)}
.pd-model__grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.pd-model__cell{display:flex;flex-direction:column;gap:8px;min-width:0;margin:0;padding:12px 14px;border-radius:10px;background:var(--pd-well)}
[data-testid="stMarkdownContainer"] .pd-model__name,.pd-model__name{display:flex;align-items:flex-start;gap:6px;margin:0;font-size:13px;font-weight:700;line-height:1.4;color:var(--pd-text-1)}
.pd-model__name .pd-ico{margin-top:1px}
[data-testid="stMarkdownContainer"] .pd-model dl,.pd-model dl{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:baseline;gap:4px 12px;margin:0;font-size:13px;line-height:1.4}
[data-testid="stMarkdownContainer"] .pd-model dt,.pd-model dt{margin:0;font-weight:500;color:var(--pd-text-2)}
[data-testid="stMarkdownContainer"] .pd-model dd,.pd-model dd{margin:0;font-size:15px;font-weight:700;text-align:right;color:var(--pd-text-1)}
[data-testid="stMarkdownContainer"] .pd-model__cell .pd-card__note{margin-top:auto}
@media (max-width:640px){.pd-model__grid{grid-template-columns:1fr}}
/* 예시 카드: 점선 테두리 · well 바탕 · 회색 작은 숫자 · 상태색 없음(명세 5.6절 예시 강등 규칙) */
.pd-example{position:relative;display:flex;flex-direction:column;gap:8px;height:100%;padding:14px 16px;border:1.5px dashed var(--pd-border-strong);border-radius:12px;background:var(--pd-well);color:var(--pd-text-2)}
.pd-example .pd-card__title{color:var(--pd-text-2)}
.pd-example__ribbon{align-self:flex-start;padding:1px 8px;border-radius:999px;border:1px dashed var(--pd-ex-border);background:var(--pd-ex-bg);font-size:12px;font-weight:700;line-height:1.5;color:var(--pd-ex-fg)}
.pd-example__value .pd-num{font-size:18px;font-weight:600;line-height:1.3;color:var(--pd-text-2)}
[data-testid="stMarkdownContainer"] .pd-example__note,.pd-example__note{margin:0;font-size:12px;line-height:1.45;color:var(--pd-text-3)}
.pd-example__num{margin-left:6px;font-size:15px;font-weight:600;color:var(--pd-text-2)}
[data-testid="stMarkdownContainer"] .pd-example__lead,.pd-example__lead{margin:0;font-size:13px;font-weight:600;line-height:1.45;color:var(--pd-text-2)}
[data-testid="stMarkdownContainer"] .pd-example__list,.pd-example__list{margin:0;padding-left:18px;font-size:13px;line-height:1.55;color:var(--pd-text-2)}
[data-testid="stMarkdownContainer"] .pd-example__list li,.pd-example__list li{margin:0;padding:0;font-size:13px}
.pd-example__item{display:flex;flex-direction:column;gap:2px;padding:4px 0 4px 10px;border-left:2px dotted var(--pd-border-strong)}
.pd-example__item b{font-size:13px;font-weight:600;color:var(--pd-text-2)}
.pd-example__item span{font-size:12px;line-height:1.45;color:var(--pd-text-3)}
.pd-example__grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.pd-example__grid>div{display:flex;flex-direction:column;gap:2px;min-width:0;padding:8px 10px;border-radius:8px;background:var(--pd-surface)}
.pd-example__grid span{font-size:12px;color:var(--pd-text-3)}
.pd-example__grid b{font-size:15px;font-weight:600;color:var(--pd-text-2)}
/* 예시 3장은 같은 행에서 높이를 맞춘다(4절 KPI 연쇄와 같은 방식). 1023px 이하에서는 6절 규칙으로 1열이 된다 */
.st-key-ex_zone [data-testid="stHorizontalBlock"]{align-items:stretch}
.st-key-ex_zone [data-testid="stColumn"]>[data-testid="stVerticalBlock"],
.st-key-ex_zone [data-testid="stElementContainer"],
.st-key-ex_zone [data-testid="stMarkdown"],
.st-key-ex_zone [data-testid="stMarkdown"]>div,
.st-key-ex_zone [data-testid="stMarkdownContainer"]{height:100%}
.st-key-ex_zone [data-testid="stMarkdownContainer"]{margin-bottom:0}
/* ---- 13. 사이드바 · 출처와 기준 시각 · 푸터 (7단계, 명세 4.6·5.12~5.14절. 3장 토큰만 쓴다) ---- */
/* 사이드바도 Streamlit이 color-scheme을 넣으므로 토큰이 사이드바 테마(라이트 #FFFFFF · 다크 #101A2E)를 따른다.
   전역 p·ol{font-size:1rem}이 요소에 직접 걸리므로 글자 크기는 클래스로 요소에 직접 준다(2단계 메모) */
.pd-side-head{margin:4px 0 0;font-size:13px;font-weight:700;line-height:1.35;color:var(--pd-text-2)}
.pd-side-hint{margin:0;font-size:12px;line-height:1.5;color:var(--pd-text-3)}
.pd-side-alerts{display:flex;flex-direction:column;gap:8px;color:var(--pd-text-2)}
/* 건수 세 칸은 기본 사이드바 폭(안쪽 약 216px)에서 한 줄에 들어가게 좁게 둔다 */
.pd-side-alerts__counts{display:flex;flex-wrap:wrap;gap:4px}
.pd-side-count{display:inline-flex;align-items:center;gap:5px;min-height:26px;padding:0 8px;border-radius:999px;background:var(--pd-well);font-size:12px;font-weight:600;line-height:1.2;color:var(--pd-text-2)}
.pd-side-count b{font-size:13px;font-weight:700;color:var(--pd-text-1)}
/* 단계 점은 상태 글자 토큰(well 위 그래픽 3:1 이상). 글자가 단계 이름을 함께 쓴다 */
.pd-side-count__dot{flex:none;width:7px;height:7px;border-radius:50%;background:var(--pd-text-3)}
.pd-side-count--critical .pd-side-count__dot{background:var(--pd-crit-fg)}
.pd-side-count--danger .pd-side-count__dot{background:var(--pd-danger-fg)}
.pd-side-count--warning .pd-side-count__dot{background:var(--pd-warn-fg)}
.pd-side-alerts__list{display:flex;flex-direction:column;gap:4px;margin:0;padding:0;list-style:none}
.pd-side-alerts__list li{margin:0;padding:2px 0 2px 8px;border-left:2px solid var(--pd-border);font-size:12px;line-height:1.5;color:var(--pd-text-2)}
.pd-side-level{font-weight:700}
.pd-side-level--critical{color:var(--pd-crit-fg)}
.pd-side-level--danger{color:var(--pd-danger-fg)}
.pd-side-level--warning{color:var(--pd-warn-fg)}
.pd-side-alerts__empty{margin:0;font-size:13px;line-height:1.5;color:var(--pd-text-2)}
/* 출처와 기준 시각: st.html 한 개라 카드 면·테두리를 직접 칠한다(3절 카드와 같은 값) */
.pd-sources{display:flex;flex-direction:column;gap:10px;padding:16px 20px;border:1px solid var(--pd-border);border-radius:12px;background:var(--pd-surface);box-shadow:0 1px 2px var(--pd-shadow-c);color:var(--pd-text-2)}
.pd-sources__table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.45;color:var(--pd-text-2)}
.pd-sources__table thead th{padding:6px 10px;font-size:12px;font-weight:600;text-align:left;white-space:nowrap;color:var(--pd-text-3);background:var(--pd-well)}
.pd-sources__table thead th:first-child{border-radius:8px 0 0 8px}
.pd-sources__table thead th:last-child{border-radius:0 8px 8px 0}
.pd-sources__table tbody th{padding:8px 10px;font-weight:600;text-align:left;white-space:nowrap;vertical-align:top;color:var(--pd-text-1)}
.pd-sources__table td{padding:8px 10px;vertical-align:top}
.pd-sources__table tbody tr+tr>*{border-top:1px solid var(--pd-border)}
.pd-sources__basis{white-space:nowrap;font-variant-numeric:tabular-nums;color:var(--pd-text-2)}
/* 상태 칩은 한 줄로 둔다(칩 글자가 두 줄로 접히면 칩 모양이 깨진다, 1024 실측). 칩이 둘이면 칸 안에서 다음 줄로 간다 */
.pd-sources__chips .pd-chip{margin:0 4px 2px 0}
.pd-sources__m{display:none}
.pd-nowrap{white-space:nowrap}
@media (max-width:1180px){.pd-sources__basis{white-space:normal}}
@media (max-width:640px){.pd-sources{padding:12px}}
/* 좁은 폭: 표를 2줄 목록으로(1줄 항목 + 칩, 2줄 출처 · 기준). 머리 줄은 화면 읽기 프로그램용으로만 남긴다.
   기준폭은 명세 기본값 640px 대신 720px: 641~720px에서는 4칸 표의 출처 칸이 약 80px로 좁아 한 행이 205px까지 늘어났다(641 실측) */
@media (max-width:720px){
  .pd-sources__table,.pd-sources__table tbody{display:block}
  .pd-sources__table thead{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
  .pd-sources__table tbody tr{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:start;gap:4px 8px;padding:8px 0}
  .pd-sources__table tbody tr+tr{border-top:1px solid var(--pd-border)}
  .pd-sources__table tbody tr+tr>*{border-top:0}
  .pd-sources__table tbody th,.pd-sources__table td{padding:0;white-space:normal}
  .pd-sources__table tbody th{grid-column:1;grid-row:1}
  .pd-sources__chips{grid-column:2;grid-row:1;text-align:right}
  .pd-sources__src{grid-column:1 / -1;grid-row:2;font-size:12px}
  .pd-sources__basis{display:none}
  .pd-sources__m{display:inline;color:var(--pd-text-3)}
}
/* 대화상자는 포털이라 stApp의 한글 줄바꿈 규칙을 물려받지 않는다(1440 실측: '지웁니|다'로 끊김). 글자 규칙만 준다 */
[role="dialog"]{word-break:keep-all;overflow-wrap:anywhere}
/* 푸터(st.markdown): 마크다운 컨테이너의 p 규칙(0,1,1)을 이기도록 컨테이너로 한 단계 높인다 */
.pd-footer{padding:14px 0;border-top:1px solid var(--pd-border)}
[data-testid="stMarkdownContainer"] .pd-footer p,.pd-footer p{margin:0;font-size:12px;line-height:1.6;color:var(--pd-text-3)}
/* 좁은 폭에서 두 문단(출처 줄·모델 줄)이 여러 줄로 접히면 경계가 안 보여 문단 사이만 조금 띄운다 */
[data-testid="stMarkdownContainer"] .pd-footer p+p{margin-top:4px}
.pd-footer b{font-weight:700;color:var(--pd-text-2)}
"""

# 지도 iframe 머리에 넣는 CSS(명세 6.7절). 바깥 페이지 토큰은 iframe에 닿지 않으므로 Streamlit 컴포넌트가 iframe :root에
# 넣는 테마 변수(--background-color·--text-color·--primary-color·--secondary-background-color)만 쓴다.
# folium.Element는 이 문자열을 Jinja 템플릿으로 읽는다: 여는 중괄호 바로 뒤에 '#'이 오면 주석 시작이 되므로
# '@media (…){ #map_div…'처럼 중괄호 뒤에 공백을 둔다(0단계 E3, create_jeju_map이 검사한다).
# map_div가 iframe 높이(100vh)를 따르는 조건을 240px 이상으로 둔다: 1180px 초과 폭에서는 Leaflet이 fitBounds를 계산하는
# 순간 iframe이 아직 과도기 높이(실측 18px)라 1px 조건이면 지도가 18px로 잡혀 minZoom 8로 떨어진다(S3 실측).
# 과도기에는 st_folium 인라인 480px를 쓰고, 실제 높이(480·440·340px)가 되면 iframe 높이를 따른다.
# 지도 iframe은 바깥 테마 글꼴(.streamlit/config.toml [theme] font)을 물려받지 않으므로 같은 CSS를 iframe에도 불러온다.
# 주소가 같아 바깥 화면이 이미 받은 CSS·글꼴 조각은 브라우저 캐시로 다시 쓴다(config.toml과 같은 판을 유지할 것)
PD_CSS_FONT_URL = ("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/"
                   "pretendardvariable-dynamic-subset.min.css")
PD_CSS_MAP = """
html,body{margin:0;padding:0}
html,body,.leaflet-container{background:var(--background-color)}
@media (min-height:240px){ #map_div{height:100vh!important}}
.leaflet-container{color:var(--text-color);font-family:"Pretendard Variable",var(--font),-apple-system,"Apple SD Gothic Neo",sans-serif}
.leaflet-tile-pane{filter:grayscale(.85) saturate(.6) brightness(1.04) contrast(.9)}
/* 소수 줌(641px 9.75 · 640px 9.5 · 360px 8.75)에서 타일 사이 1px 밝은 이음선이 지도를 가로질렀다(1차 평가). OSM 타일(256px)을
   0.5px씩 겹쳐 틈을 덮는다(정수 줌에서는 0.2% 확대라 보이지 않는다, 1차 수정 때 360·640 다크 스크린샷으로 확인) */
.leaflet-tile-container img.leaflet-tile{width:256.5px!important;height:256.5px!important}
html[data-pd-theme="dark"] .leaflet-tile-pane{filter:invert(1) hue-rotate(180deg) brightness(.9) contrast(.88) grayscale(.5)}
.leaflet-control-attribution{font-size:12px;line-height:1.4;color:var(--text-color);background:color-mix(in srgb,var(--background-color) 85%,transparent)!important}
.leaflet-control-attribution a{color:var(--primary-color)}
.leaflet-control-scale-line{color:var(--text-color);border-color:var(--text-color);text-shadow:none;background:color-mix(in srgb,var(--background-color) 80%,transparent)}
/* 축척은 미터 막대만 둔다(국내 화면에서 마일 막대는 잡음, 1차 평가) */
.leaflet-control-scale-line+.leaflet-control-scale-line{display:none}
/* 지도 칸은 키보드 초점 자리(화살표로 이동)인데 기본 테두리가 iframe 가장자리 밖이라 잘려 안 보였다. 타일·기호 위에 안쪽 테두리를 겹친다 */
.leaflet-container:focus-visible::after{content:"";position:absolute;inset:0;border:2px solid var(--primary-color);border-radius:10px;pointer-events:none;z-index:1000}
.leaflet-bar a{background:var(--background-color);color:var(--text-color);border-color:var(--secondary-background-color)}
.leaflet-bar a.leaflet-disabled{background:var(--secondary-background-color);color:color-mix(in srgb,var(--text-color) 45%,transparent)}
.leaflet-popup-content-wrapper,.leaflet-popup-tip{background:var(--background-color);color:var(--text-color)}
.leaflet-container a.leaflet-popup-close-button{color:var(--text-color)}
.leaflet-tooltip{background:var(--background-color);color:var(--text-color);border-color:var(--secondary-background-color);font-size:12px}
.leaflet-tooltip-top:before{border-top-color:var(--background-color)}
.leaflet-tooltip-bottom:before{border-bottom-color:var(--background-color)}
.leaflet-tooltip-left:before{border-left-color:var(--background-color)}
.leaflet-tooltip-right:before{border-right-color:var(--background-color)}
.leaflet-marker-icon.pd-mk{display:flex;align-items:center;justify-content:center;background:none;border:0;border-radius:50%}
.leaflet-marker-icon.pd-mk:focus-visible{outline:2px solid var(--primary-color);outline-offset:2px}
.pd-callout-mk{background:none;border:0}
.pd-callout{display:inline-block;padding:6px 10px;border-radius:10px;background:var(--background-color);color:var(--text-color);border:1px solid var(--secondary-background-color);font-size:12px;line-height:1.35;white-space:nowrap}
.pd-callout b{font-weight:700}
.pd-callout__bar{display:flex;height:4px;margin-top:4px;border-radius:2px;overflow:hidden}
.pd-callout__bar span{display:block;height:100%}
.pd-pop{font-size:13px;line-height:1.4}
.pd-pop__title{margin:0 0 6px;font-size:14px;font-weight:700;line-height:1.3}
.pd-pop table{width:100%;border-collapse:collapse}
.pd-pop th{padding:2px 12px 2px 0;font-weight:500;text-align:left;white-space:nowrap;opacity:.8}
.pd-pop td{padding:2px 0;font-weight:600;text-align:right}
.pd-pop__note{margin:6px 0 0;font-size:12px;opacity:.8;border-top:1px solid var(--secondary-background-color);padding-top:4px}
@media (max-width:480px){.pd-callout,.leaflet-control-scale{display:none}}
"""


def _css_icons() -> str:
    """아이콘마다 `.pd-ico--<이름>{--ico:url(data URI)}` 규칙을 만든다.

    SVG를 퍼센트 인코딩해야 style 블록에 날 꺾쇠가 남지 않는다(명세 3.8절 [계약]).
    인코딩하지 않은 SVG가 하나라도 섞이면 DOMPurify가 style 블록 전체를 지워 화면 CSS가 모두 사라진다.
    """
    rules = []
    for name, body in PD_ICONS.items():
        svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='black' "
            "stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'>" + body + "</svg>"
        )
        enc = quote(svg, safe=" =':/,.-")
        rules.append(f'.pd-ico--{name}{{--ico:url("data:image/svg+xml,{enc}")}}')
    return "\n".join(rules)


def _css_dashboard() -> str:
    """화면 CSS 전체(토큰 + 기반 + 컴포넌트 + 아이콘)를 style 블록 하나로 돌려준다.

    main() 첫 줄의 st.html로 넣는다. style만 담은 st.html은 레이아웃 칸을 차지하지 않는다.
    본문에 여는 꺾쇠가 있으면 DOMPurify가 블록 전체를 지우므로, 조용히 깨지지 않게 바로 실패시킨다.
    """
    body = PD_CSS_TOKENS + PD_CSS_BASE + PD_CSS_COMPONENTS + _css_icons()
    if chr(60) in body:
        raise ValueError("화면 CSS 본문에 여는 꺾쇠(문자 코드 60)가 있다: DOMPurify가 style 블록 전체를 지운다")
    return "<style>" + body + "</style>"


# ============================================================================
# 표시 계층: 경보 단계 · 출처 판정 · 머리 HTML (명세 5.4·5.5·7장)
# 값과 경보 판정은 바꾸지 않는다(F7). "이 숫자를 어떻게 읽어야 하는지"만 정해 화면에 붙인다.
# ============================================================================

def _html_chip(kind: str, text: str) -> str:
    """출처 칩 하나를 만든다(명세 3.3·7.1절).

    Args:
        kind: 글리프가 붙는 4상태 'measured'·'forecast'·'est'·'example', 글리프 없는 'plain'·'stale'.
        text: 칩 글자. 이 함수는 이스케이프하지 않으므로 외부에서 온 문자열은 호출부가 html_escape로 넘긴다.

    Returns:
        span 한 개의 HTML.

    Raises:
        ValueError: kind가 위 여섯 가지가 아닐 때. 오타로 칩 모양(= 데이터 상태 표기)이 조용히 바뀌지 않게 한다.
    """
    if kind in ("plain", "stale"):
        return f'<span class="pd-chip pd-chip--{kind}">{text}</span>'
    if kind not in ("measured", "forecast", "est", "example"):
        raise ValueError(f"알 수 없는 칩 종류: {kind!r}")
    return (f'<span class="pd-chip pd-chip--{kind}"><span class="pd-chip__glyph" aria-hidden="true"></span>'
            f'{text}</span>')


def _html_meta_chip(icon: str, text_html: str, extra_class: str = "") -> str:
    """앱 바의 아이콘 칩(중립 외곽선)을 만든다.

    글자를 span 하나로 감싼다. 칩은 inline-flex라 글자 조각이 따로 놓이면 조각 사이마다 gap이 끼어
    "SMP 거래시간 24h" 같은 한 문장이 벌어진다.

    Raises:
        ValueError: PD_ICONS에 없는 아이콘 이름. 빈 마스크(보이지 않는 아이콘)로 조용히 그려지지 않게 한다.
    """
    if icon not in PD_ICONS:
        raise ValueError(f"PD_ICONS에 없는 아이콘: {icon!r}")
    cls = f"pd-chip pd-chip--plain {extra_class}".strip()
    return (f'<span class="{cls}"><span class="pd-ico pd-ico--{icon}" aria-hidden="true"></span>'
            f'<span>{text_html}</span></span>')


def _fmt_reserve_level(reserve_rate: float) -> Dict:
    """예비율 단계표(명세 7.4절, KPX 기준 5·10·15%).

    옛 main()의 if 문 값과 문구를 그대로 옮겼다. alert_title·alert_msg는 Slack·메일 본문에 그대로 들어가므로
    글자를 바꾸면 운영 알림 문구가 바뀐다. 화면용 필드(banner_msg·verdict·다음 경계)만 더했다.
    banner_msg에 수치를 넣지 않는 이유: 배너는 role="alert"라 문자열이 매분 바뀌면 화면 읽기 프로그램이 매분 다시 읽는다.

    Args:
        reserve_rate: 예비율(%).

    Returns:
        status·class·text·show_alert·alert_title·alert_msg·banner_msg·verdict·next_name·next_ratio.
        next_ratio는 "공급능력 ÷ 비율 = 다음 단계 경계의 수요"에 쓰는 비율이고, 위험 단계는 다음 경계가 없어 None이다.
    """
    if reserve_rate < 5:
        return {"status": "critical", "class": "status-critical", "text": "위험", "show_alert": True,
                "alert_title": "전력 수급 위험 경보",
                "alert_msg": f"예비율 {reserve_rate:.1f}% - 즉각적인 부하 감축 필요",
                "banner_msg": "예비율이 5% 아래입니다. 즉각적인 부하 감축이 필요한 단계입니다",
                "verdict": "공급이 빠듯합니다", "next_name": None, "next_ratio": None}
    if reserve_rate < 10:
        return {"status": "danger", "class": "status-danger", "text": "주의", "show_alert": True,
                "alert_title": "전력 수급 주의 경보",
                "alert_msg": f"예비율 {reserve_rate:.1f}% - 전력 수급 상황 주시 필요",
                "banner_msg": "예비율이 10% 아래입니다. 수급 상황을 주시하는 단계입니다",
                "verdict": "여유가 부족합니다", "next_name": "위험", "next_ratio": 1.05}
    if reserve_rate < 15:
        return {"status": "warning", "class": "status-warning", "text": "관심", "show_alert": True,
                "alert_title": "전력 수급 관심 단계",
                "alert_msg": f"예비율 {reserve_rate:.1f}% - 전력 사용 절감 협조 요청",
                "banner_msg": "예비율이 15% 아래입니다. 전력 사용 절감 협조를 요청하는 단계입니다",
                "verdict": "여유가 줄고 있습니다", "next_name": "주의", "next_ratio": 1.10}
    return {"status": "normal", "class": "status-online", "text": "정상", "show_alert": False,
            "alert_title": None, "alert_msg": None, "banner_msg": None,
            "verdict": "넉넉합니다", "next_name": "관심", "next_ratio": 1.15}


def _fmt_provenance(power_status: Dict, api_kpis: Optional[Dict], kpx_local: Optional[Dict],
                    now: datetime) -> Dict:
    """수요·예비율·공급능력의 출처 상태와 자리별 문구를 한 곳에서 정한다(명세 7.2·7.2b절, J1).

    _backend_kpi_overrides가 API 값을 쓰면 KPX·EPSIS 파일·시뮬레이션 세 경로 모두 data_source가
    "KPX 실시간 (API 서버)"로 덮이고 data_date에는 관측 시각이 아니라 앱 조회 시각이 들어간다.
    그래서 실측 여부는 API 응답의 data_source(출처 항목을 " + "로 이은 문자열)에 수요 출처 항목이
    들어 있는지로 판정한다(위에서부터 첫 일치 ①~⑨). 표시 전용이라 값과 경보 판정은 건드리지 않는다(F7).

    Args:
        power_status: get_current_power_status() 결과(또는 테스트 모드 사본).
        api_kpis: fetch_backend_json("/api/v1/dashboard/kpis") 결과. 실패면 None.
        kpx_local: fetch_jeju_realtime() 결과. 실패면 None.
        now: 화면 기준 시각.

    Returns:
        상태 키(demand_state·reserve_state·demand_measured·reserve_measured·reserve_known·supply_measured·
        all_measured·time_kind·src_label·reserve_note·reserve_why·query_hhmm), KPI 표시 키 묶음 display_keys
        (pd_chip·pd_src_label·pd_state_class·pd_renew_time·pd_time), 자리별 문구(fresh_chip·status_src·
        supply_chips·supply_meta·supply_foot·legend_demand_note·legend_supply_note·strip_now·demand_chips·
        demand_meta·grid_chip).
    """
    ds = str(power_status.get("data_source", ""))
    api_src = api_kpis.get("data_source") if isinstance(api_kpis, dict) else None
    has_api_src = isinstance(api_src, str) and bool(api_src.strip())
    kpx_is_dict = isinstance(kpx_local, dict)

    reserve_note, reserve_why = "", ""
    if ds == "테스트 모드":                                    # ① 관리자 테스트 값
        demand_state, reserve_state, time_kind = "test", "test", "test"
        src_label, reserve_note = "관리자 테스트 값", " · 테스트 값"
    elif ds == "KPX 실시간 (API 서버)" and not has_api_src:   # ② 캐시 시차 등으로 API 출처를 못 읽음
        demand_state, reserve_state, time_kind = "est", "est", "query"
        src_label, reserve_note = "API 서버 값 (출처 확인 불가)", " · 실측 아님"
    elif ds == "KPX 실시간 (API 서버)" and "Jeju Crawler" in api_src:  # ③ 제주 KPX 크롤러
        demand_state, reserve_state, time_kind = "measured", "measured", "query"
        src_label = "KPX 제주 실시간 (API 서버)"
    elif ds == "KPX 실시간 (API 서버)" and "EPSIS Demand" in api_src:  # ④ 과거 파일 30일 평균, 예비율은 API 기본값
        demand_state, reserve_state, time_kind = "est", "default", "query"
        src_label, reserve_note = "EPSIS 30일 평균 (API 서버 대체값)", " · API 기본값 (실측 아님)"
        reserve_why = "API 서버가 실측 대신 기본 예비율(25%)을 보냈습니다"
    elif ds == "KPX 실시간 (API 서버)" and "KPX" in api_src:  # ⑤ 전국 공공 API: 수요 × 0.018, 예비율은 전국 값
        demand_state, reserve_state, time_kind = "est", "national", "query"
        src_label, reserve_note = "전국 값 × 0.018 비례 추정 (API 서버)", " · 전국 예비율 (제주 값 아님)"
        reserve_why = "API 서버가 제주 대신 전국 예비율을 보냈습니다"
    elif ds == "KPX 실시간 (API 서버)":                        # ⑥ 수요 출처 항목 없음: API 기본값 650 MW·25%
        demand_state, reserve_state, time_kind = "sim", "default", "query"
        src_label, reserve_note = "API 서버 기본값 650 MW·25% (실측 아님)", " · API 기본값 (실측 아님)"
        reserve_why = "API 서버가 실측 대신 기본값(수요 650 MW·예비율 25%)을 보냈습니다"
    elif ds == "KPX 실시간":                                   # ⑦ API 실패, 앱이 KPX를 직접 조회
        # 응답에 키가 없으면 앱이 기본값(수요 800 MW·예비율 15%)을 쓴다. 지금 응답 자체가 없으면(두 캐시의
        # 만료 시차) 앱이 KPX 응답을 받아 "KPX 실시간"을 붙인 판정을 따른다.
        demand_ok = not kpx_is_dict or _finite_number(kpx_local.get("current_demand"))
        reserve_ok = not kpx_is_dict or _finite_number(kpx_local.get("reserve_rate"))
        demand_state = "measured" if demand_ok else "sim"
        reserve_state = "measured" if reserve_ok else "default"
        time_kind = "query"
        src_label = "KPX 제주 실시간 (앱 직접 조회)" if demand_ok else "앱 기본값 800 MW (KPX 수요 미수신, 실측 아님)"
        if not reserve_ok:
            reserve_note, reserve_why = " · 앱 기본값 (실측 아님)", "앱이 기본 예비율(15%)을 쓰고 있습니다"
    elif ds == "EPSIS 파일":                                   # ⑧ 과거 파일 값: 잰 값이지만 지연
        demand_state, reserve_state, time_kind = "stale", "stale", "date"
        src_label, reserve_note = "EPSIS 파일 (과거 값)", " · 지연"
    else:                                                      # ⑨ 앱 시뮬레이션(예비율 15% 고정)
        demand_state, reserve_state, time_kind = "sim", "default", "none"
        src_label, reserve_note = "시뮬레이션 (실측 아님)", " · 앱 기본값 (실측 아님)"
        reserve_why = "앱이 기본 예비율(15%)을 쓰고 있습니다"

    demand_measured = demand_state == "measured"
    reserve_measured = reserve_state == "measured"
    reserve_known = reserve_state not in ("national", "default")
    # 앱 KPX 조회에 공급능력이 없으면 앱은 수요 × 1.15나 EPSIS 파일 값을 쓰는데, API 값으로 출처가 덮인 뒤에는
    # 어느 쪽인지 알 수 없으므로 보수적으로 추정으로 본다
    supply_measured = bool(kpx_is_dict and _finite_number(kpx_local.get("supply_capacity")))
    all_measured = demand_measured and reserve_measured and supply_measured

    data_date = power_status.get("data_date", "N/A")
    try:
        queried = datetime.strptime(str(data_date).strip(), "%Y-%m-%d %H:%M")
    except ValueError:
        queried = None
    # "조회"는 관측 시각이 아니라 앱이 값을 받아 온 시각이다(관측 시각 표시는 승인 P1 뒤, 7.3절)
    query_hhmm = f"{queried:%H:%M}" if queried else f"{now:%H:%M}"
    _, elapsed = _format_observation_age(data_date, now)
    if elapsed == "N/A":
        elapsed = "경과 시간 확인 불가"
    elapsed_short = elapsed.split("일 ")[0] + "일 경과" if "일 " in elapsed else elapsed  # 앱 바 칩은 일 단위까지만
    obs_date = html_escape(str(data_date)[:10])
    screen = f"{now:%H:%M}"
    curve_tail = " · 곡선은 앱 내장 일간 패턴 × 현재값 (가정값, 예측 모델 아님)"

    # 7.2b 표: 지금 잰 값이 아니면 실측 칩(채운 점)과 "KPX 조회"라는 말을 쓰지 않는다
    if demand_state == "measured":
        fresh_chip = _html_meta_chip("clock", f"KPX 조회 {query_hhmm} KST · {elapsed}")
        status_src = f"KPX 조회 {query_hhmm} KST"
        pd_chip = _html_chip("measured", "실측")
        supply_meta = f"출처 KPX 제주 실시간 · 조회 {query_hhmm} KST · {elapsed}{curve_tail}"
        foot_demand = "KPX 제주 실시간(현재값)"
        legend_demand_note = "점선 = 추정 패턴 · 점 = KPX 조회값"
        strip_now = f"● 지금 {screen} · KPX 조회"
        demand_meta = f"출처 {src_label} · 조회 {query_hhmm} KST · {elapsed}"
    elif demand_state == "stale":
        fresh_chip = (_html_meta_chip("clock", f"EPSIS 파일 {obs_date} · {elapsed_short}")
                      + _html_chip("stale", "지연"))
        status_src = f"EPSIS 파일 · {obs_date}"
        pd_chip = _html_chip("measured", "실측") + _html_chip("stale", "지연")
        supply_meta = f"출처 EPSIS 파일 {obs_date} · {elapsed}{curve_tail}"
        foot_demand = f"EPSIS 파일 {obs_date}(현재값, 지연)"
        legend_demand_note = "점선 = 추정 패턴 · 점 = EPSIS 파일 값 (지연)"
        strip_now = f"○ 지금 {screen} · EPSIS 파일 값 (지연)"
        demand_meta = f"출처 EPSIS 파일 {obs_date} · {elapsed}"
    elif demand_state == "est":
        fresh_chip = _html_chip("est", f"대체값 · {src_label}")
        status_src = f"{src_label} · 조회 {query_hhmm} KST"
        pd_chip = _html_chip("est", "추정 · 실측 아님")
        supply_meta = f"출처 {src_label} · 조회 {query_hhmm} KST · {elapsed}{curve_tail}"
        foot_demand = f"{src_label}(현재값, 실측 아님)"
        legend_demand_note = "점선 = 추정 패턴 · 점 = 현재값 (추정)"
        strip_now = f"○ 지금 {screen} · 추정값"
        demand_meta = f"출처 {src_label} · 조회 {query_hhmm} KST"
    elif demand_state == "sim":
        fresh_chip = _html_chip("est", "시뮬레이션 모드 · 실측 아님")
        status_src = src_label
        pd_chip = _html_chip("est", "시뮬레이션 · 실측 아님")
        supply_meta = f"출처 {src_label} · 화면 {screen} KST{curve_tail}"
        foot_demand = f"{src_label}(현재값)"
        legend_demand_note = "점선 = 추정 패턴 · 점 = 현재값 (시뮬레이션)"
        strip_now = f"○ 지금 {screen} · 시뮬레이션 값"
        demand_meta = f"출처 {src_label} · 화면 {screen} KST"
    else:  # test
        fresh_chip = _html_chip("est", "테스트 모드")
        status_src = "관리자 테스트 값"
        pd_chip = _html_chip("est", "테스트 값")
        supply_meta = f"관리자 테스트 값 · 화면 {screen} KST"
        foot_demand = "관리자 테스트 값"
        legend_demand_note = "점 = 테스트 값"
        strip_now = f"○ 지금 {screen} · 테스트 값"
        demand_meta = f"관리자 테스트 값 · 화면 {screen} KST"

    # 공급능력은 따로 판정하므로 한 카드에 두 상태가 섞일 수 있다(실측 칩은 실제로 잰 현재값에만 붙인다)
    if demand_state == "test":
        supply_chips = _html_chip("est", "테스트 모드")
    elif demand_state == "measured":
        supply_chips = _html_chip("est", "추정 패턴") + _html_chip(
            "measured", "현재값만 실측" if supply_measured else "수요 현재값만 실측")
    elif demand_state == "stale":
        supply_chips = _html_chip("est", "추정 패턴") + _html_chip("measured", "현재값 = EPSIS 파일 · 지연")
    elif supply_measured:
        word = "추정" if demand_state == "est" else "시뮬레이션"
        supply_chips = (_html_chip("est", f"추정 패턴 · 수요 현재값도 {word}")
                        + _html_chip("measured", "공급능력 현재값만 실측"))
    else:
        word = "추정" if demand_state == "est" else "시뮬레이션"
        supply_chips = _html_chip("est", f"추정 패턴 · 현재값도 {word}")

    if demand_state == "test" or (demand_measured and supply_measured):
        foot_first = foot_demand
    else:
        supply_part = "KPX 제주 실시간(현재값)" if supply_measured else "추정(KPX 공급능력 미수신)"
        foot_first = f"수요 {foot_demand} · 공급능력 {supply_part}"
    supply_foot = f"출처: {foot_first} · 곡선은 앱 내장 일간 패턴(가정값, 실측·EPSIS 아님) · 화면 갱신 {screen} KST"
    legend_supply_note = "점선 = 현재 공급능력 × 수요 패턴(가정) · " + (
        "점 = KPX 조회값" if supply_measured else "점 = 현재값 (추정 · KPX 공급능력 미수신)")

    if demand_state == "test":
        grid_chip = _html_chip("est", "테스트 값")
    elif all_measured:
        grid_chip = _html_chip("measured", "실측 기반 계산")
    else:
        grid_chip = _html_chip("est", "추정값 기준")

    display_keys = {
        "pd_chip": pd_chip,
        "pd_src_label": src_label,
        "pd_state_class": {"measured": "", "stale": "is-stale"}.get(demand_state, "is-est"),
        # 재생 비율은 API 값(renewable_source 키가 있음)이면 조회 시각, 앱 추정이면 화면 시각
        "pd_renew_time": (f"조회 {query_hhmm} KST" if "renewable_source" in power_status
                          else f"화면 {screen} KST (앱 추정)"),
        "pd_time": (f"조회 {query_hhmm} KST" if isinstance(api_kpis, dict)
                    else f"화면 {screen} KST (앱 시뮬레이션)"),
    }
    return {
        "demand_state": demand_state, "reserve_state": reserve_state,
        "demand_measured": demand_measured, "reserve_measured": reserve_measured,
        "reserve_known": reserve_known, "supply_measured": supply_measured, "all_measured": all_measured,
        "time_kind": time_kind, "src_label": src_label, "reserve_note": reserve_note, "reserve_why": reserve_why,
        "query_hhmm": query_hhmm, "display_keys": display_keys,
        "fresh_chip": fresh_chip, "status_src": status_src,
        "supply_chips": supply_chips, "supply_meta": supply_meta, "supply_foot": supply_foot,
        "legend_demand_note": legend_demand_note, "legend_supply_note": legend_supply_note,
        "strip_now": strip_now, "demand_chips": pd_chip, "demand_meta": demand_meta, "grid_chip": grid_chip,
    }


def _fmt_reserve_display(level: Dict, prov: Dict) -> Tuple[str, str]:
    """KPI 예비율 pill의 (클래스, 글자)를 정한다(명세 7.2b절).

    기본값(API 25%·앱 15%)이나 전국 예비율로 "정상"을 말하지 않도록 그 경우에는 단계 대신 "판정 안 함"을 쓴다.
    추정·테스트 값은 단계 글자는 두되 빗금(is-est)과 덧말로 실측이 아님을 함께 보인다.
    """
    if not prov["reserve_known"]:
        return "is-est", "판정 안 함" + prov["reserve_note"]
    if prov["reserve_state"] in ("measured", "stale"):
        return level["class"], level["text"] + prov["reserve_note"]
    return f"{level['class']} is-est", level["text"] + prov["reserve_note"]


def _fmt_observation_display(prov: Dict, observed_text: str, elapsed_text: str) -> Tuple[str, str]:
    """'현재 수요' 카드 관측 줄의 두 칸(명세 7.3절).

    카드 f-string의 "관측" 글자(T8 대상)는 그대로 두고 뒤의 값만 정직하게 바꾼다. API 경로의 data_date는
    관측 시각이 아니라 앱 조회 시각이므로 "시각 미제공 · 조회 HH:MM KST"로 쓴다(관측 시각은 승인 P1 뒤).
    """
    kind = prov["time_kind"]
    if kind == "test":
        return "없음 (테스트 값)", "실측 아님"
    if kind == "none":
        return "없음 (시뮬레이션 값)", "실측 아님"
    elapsed = elapsed_text if elapsed_text != "N/A" else "경과 시간 확인 불가"
    if kind == "date":
        return html_escape(str(observed_text)), elapsed
    return f"시각 미제공 · 조회 {prov['query_hhmm']} KST", elapsed


def _fmt_weather_provenance(api_kpis: Optional[Dict]) -> Optional[str]:
    """기상 값의 출처 종류를 API 출처 문자열로 정한다(명세 7.2절 끝).

    API는 실시간 기상과 파일이 모두 실패하면 5.0 °C·3.0 m/s 기본값을 출처 항목 없이 보내고, 앱은 값이
    유한하기만 하면 source='kma'로 둔다. 그래서 source만 보면 API 기본값을 실측처럼 보이게 된다.

    Returns:
        'file'(과거 자료 24시간 평균) · 'live'(실시간) · 'default'(API 기본값) · None(API 응답 없음 = 앱 시뮬레이션 경로).
    """
    if not isinstance(api_kpis, dict):
        return None
    src = api_kpis.get("data_source")
    src = src if isinstance(src, str) else ""
    if "KMA Weather (파일)" in src:
        return "file"
    if any(token in src for token in ("KMA Weather Crawler", "KMA Real-time API", "기상청 (실시간)")):
        return "live"
    return "default"


def _fmt_smp_display_keys(smp_data: Dict, now: datetime) -> Dict:
    """SMP 출처 판정 표시 키(명세 5.10절). KPI·SMP 탭 카드·출처 표가 같은 판정을 쓰게 한 곳에서 만든다.

    Returns:
        pd_src_label(현재가 출처 문구), pd_stale_chip(오늘 가격이 아니면 지연 칩, 아니면 빈 문자열),
        pd_fc_kind('model'·'api_fallback'·'local'), pd_fc_short(칩용 짧은 모델 이름), pd_confirmed_point(bool).
    """
    is_epsis = str(smp_data.get("data_source", "")).startswith("EPSIS")
    trade_label = smp_data.get("trade_label")
    if is_epsis and trade_label:
        src_label = "EPSIS 하루전 (파일)"
    elif is_epsis:
        # 거래시간이 없으면 get_smp_data가 current에 최근 48시간 평균을 넣는다(확정 가격이 아니다)
        src_label = "최근 48시간 평균 (확정 가격 아님)"
    else:
        src_label = "시뮬레이션 값 (실측 아님)"

    trade_date = None
    if trade_label:
        try:
            trade_date = datetime.strptime(str(trade_label)[:10], "%Y-%m-%d").date()
        except ValueError:
            trade_date = None
    stale_chip = ""
    if is_epsis and trade_date is not None and (now.date() - trade_date).days > 0:
        stale_chip = _html_chip("stale", f"지연 · {(now.date() - trade_date).days}일 전 가격")

    model = smp_data.get("forecast_model")
    if "forecast_model" not in smp_data:
        fc_kind = "local"  # API 예측을 못 받아 앱이 시간대 평균 패턴으로 채웠다
    elif isinstance(model, str) and any(token in model for token in ("대체값", "Fallback", "fallback")):
        fc_kind = "api_fallback"  # API가 모델 대신 패턴·EPSIS 값을 보냈다
    else:
        fc_kind = "model"
    name = model.strip() if isinstance(model, str) else ""
    # 칩은 한 줄이라 긴 모델 이름을 넣으면 390px에서 넘친다. 운영 이름 세 가지는 짧은 이름으로 바꾼다
    if "기준선" in name:
        fc_short = "기준선"
    elif "v4.1" in name:
        fc_short = "v4.1"
    elif "v3.2" in name:
        fc_short = "v3.2 · 재검증 전"
    elif not name:
        fc_short = "모델 이름 미제공"
    else:
        fc_short = html_escape(name if len(name) <= 12 else name[:12] + "…")
    confirmed_point = bool(is_epsis and trade_date is not None
                           and str(smp_data.get("forecast_date") or "") == f"{trade_date:%Y-%m-%d}")
    return {"pd_src_label": src_label, "pd_stale_chip": stale_chip, "pd_fc_kind": fc_kind,
            "pd_fc_short": fc_short, "pd_confirmed_point": confirmed_point}


def _html_app_bar(power_status: Dict, smp_data: Dict, level: Dict, prov: Dict, auto_refresh: bool) -> str:
    """앱 바(명세 5.4절): 로고·제목·부제 | 수신 칩 · SMP 거래시간 칩 · 자동 갱신 칩.

    Streamlit 헤더 줄(투명 60px)과 겹쳐 한 줄로 놓인다. 제목은 h1이 아니라 role="heading"인 p다
    (Streamlit 전역 제목 스타일·앵커 링크를 피한다). 1180px 이하에서는 부제를 숨기고 칩 글자를 줄이며,
    640px 이하에서는 2행(수신 칩과 SMP 칩만)이 된다. power_status·level은 명세 시그니처대로 받는다
    (경보 단계는 배너가 맡으므로 앱 바에는 쓰지 않는다).
    """
    label = str(smp_data.get("trade_label") or "").strip()
    hour = label.split(" ")[-1] if label else ""
    if hour.endswith("h") and hour[:-1].isdigit():
        smp_text = f'SMP<span class="pd-appbar__long"> 거래시간</span> {hour}'
    else:
        smp_text = 'SMP<span class="pd-appbar__long"> 거래시간</span> 확인 불가'
    if auto_refresh:
        auto_text = '<span class="pd-appbar__long">60초마다 자동 갱신</span><span class="pd-appbar__short">60초</span>'
    else:
        auto_text = '<span class="pd-appbar__long">자동 갱신 꺼짐</span><span class="pd-appbar__short">갱신 꺼짐</span>'
    return (
        '<div class="pd-appbar">'
        '<div class="pd-appbar__brand">'
        '<span class="pd-appbar__logo"><span class="pd-ico pd-ico--map" aria-hidden="true"></span></span>'
        '<div><p class="pd-appbar__title" role="heading" aria-level="1">제주 전력 지도</p>'
        '<p class="pd-appbar__sub">제주 전력 수급·SMP·재생에너지</p></div>'
        '</div>'
        f'<div class="pd-appbar__meta">{prov["fresh_chip"]}{_html_meta_chip("tag", smp_text)}'
        f'{_html_meta_chip("refresh", auto_text, "pd-appbar__auto")}</div>'
        '</div>'
    )


def _html_alert_banner(level: Dict, prov: Dict) -> str:
    """경보 배너(명세 5.4절, 예비율 15% 미만일 때만).

    매분 바뀌는 수치를 넣지 않는다: st.html은 문자열이 바뀌면 안쪽 DOM을 통째로 바꾸므로 role="alert"가
    60초 fragment마다 다시 읽힌다. 고정 문장이면 나타날 때와 단계가 바뀔 때만 읽힌다.
    예비율이 제주 실측이 아닌데(전국 예비율 등) 경보 단계가 나오면, 판정은 그대로 두고(F7) 그 사실을 밝힌다.
    """
    if prov["reserve_known"]:
        msg = level["banner_msg"]
    else:
        msg = f"{prov['reserve_why']} · 경보는 이 값으로 판정됐습니다"
    chip = "" if prov["all_measured"] else _html_chip("est", "추정값 기준")
    return (
        f'<div class="pd-alert pd-alert--{level["status"]}" role="alert">'
        '<span class="pd-ico pd-ico--alert pd-alert__icon" aria-hidden="true"></span>'
        f'<div class="pd-alert__body"><p class="pd-alert__title">{level["alert_title"]}</p>'
        f'<p class="pd-alert__msg">{msg} · 기준 관심 15% · 주의 10% · 위험 5%</p></div>'
        f'<div class="pd-alert__value">{chip}</div>'
        '</div>'
    )


def _html_status_line(power_status: Dict, level: Dict, prov: Dict) -> str:
    """상태 문장(명세 5.4절): 단계 pill + 쉬운 판정 문장 + 예비율과 출처.

    문장 안에서 퍼센트를 다시 계산하지 않는다. API 예비율과 앱이 조회한 공급능력은 조회 시점이 달라
    1~2%p 어긋날 수 있어, 재계산 값과 KPX 값을 나란히 두면 비전문 독자가 혼동한다. 그래서 MW 차이는
    화면 KPI와 같은 반올림 값으로 계산하고, 예비율은 출처를 붙여 따로 보인다.
    예비율이 기본값·전국 값이면 "넉넉합니다" 같은 판정을 쓰지 않는다. 60초마다 바뀌므로 aria-live를 쓰지 않는다.
    """
    reserve_class = _fmt_reserve_display(level, prov)[0]
    pill_text = level["text"] if prov["reserve_known"] else "확인 불가"
    pill = (f'<span class="pd-pill status-badge {reserve_class}"><span class="pd-pill__dot" aria-hidden="true">'
            f'</span>{pill_text}</span>')
    chip = "" if prov["all_measured"] else _html_chip("est", "추정값 기준")

    demand, supply = power_status.get("demand"), power_status.get("total_supply")
    if not prov["reserve_known"]:
        sentence = f'<b>지금 제주 예비율은 확인할 수 없습니다.</b> {prov["reserve_why"]}.'
    elif _finite_number(demand) and _finite_number(supply):
        # KPI 타일의 {:,.0f}와 같은 반올림으로 정수를 만든 뒤 뺀다(화면에 보인 두 수와 차이가 맞게)
        d_mw, s_mw = int(f"{demand:.0f}"), int(f"{supply:.0f}")
        head = f'<b>지금 제주 전력은 {level["verdict"]}.</b> '
        if s_mw > d_mw:
            sentence = (f'{head}쓰는 전력(수요) {d_mw:,} MW보다 낼 수 있는 전력(공급능력) {s_mw:,} MW가 '
                        f'{s_mw - d_mw:,} MW 더 많습니다.')
        elif s_mw < d_mw:
            sentence = (f'{head}쓰는 전력(수요) {d_mw:,} MW보다 낼 수 있는 전력(공급능력) {s_mw:,} MW가 '
                        f'{d_mw - s_mw:,} MW 적습니다 (두 값의 조회 시점이 다를 수 있음).')
        else:
            sentence = f'{head}쓰는 전력(수요)과 낼 수 있는 전력(공급능력)이 {d_mw:,} MW로 같습니다.'
    else:
        sentence = f'<b>지금 제주 전력은 {level["verdict"]}.</b> 수요나 공급능력 값이 없어 둘의 차이는 보이지 않습니다.'
    reserve_rate = power_status.get("reserve_rate")
    reserve_txt = f"{reserve_rate:.1f}%" if _finite_number(reserve_rate) else "—"
    return (
        f'<div class="pd-status">{pill}{chip}'
        f'<p class="pd-status__text">{sentence}</p>'
        f'<span class="pd-status__src">예비율 {reserve_txt} · {prov["status_src"]}</span>'
        '</div>'
    )


def _html_legend_strip() -> str:
    """데이터 상태 읽는 법 띠(명세 5.4절, 정적). 칩·차트가 쓰는 4상태 글리프를 한 줄로 풀어 준다.

    640px 이하에서는 제목을 "읽는 법"으로 줄이고 설명을 숨긴다(와이어프레임 4.1절 390). 숨길 부분만 span으로 감쌌다.
    """
    items = (
        ("measured", "실측·확정", "잰 값·확정 가격 (실선·채운 점)"),
        ("forecast", "예측", "모델 출력 (대시·80% 구간 띠)"),
        ("est", "추정·시뮬레이션", "계산·가정·모의 값 (점선·빗금)"),
        ("example", "예시", "설명용 고정 값 (점선 테두리·호박색 띠)"),
    )
    cells = "".join(
        f'<span class="pd-legend__item"><span class="pd-glyph pd-glyph--{kind}" aria-hidden="true"></span>'
        f'<b>{name}</b><span class="pd-legend__desc">{desc}</span></span>'
        for kind, name, desc in items
    )
    return (
        '<div class="pd-legend" role="group" aria-label="데이터 상태 읽는 법">'
        '<span class="pd-legend__title"><span class="pd-legend__long">데이터 상태 </span>읽는 법</span>'
        f'{cells}</div>'
    )


def _html_card_head(title: str, subtitle: str, chips_html: str, meta_html: str) -> str:
    """카드·무대 머리(명세 5.7절): 제목 · 부제 · 칩 한 줄 + 메타 줄(출처 · HH:MM KST · 경과 | 해석).

    차트 제목은 여기에만 둔다(Plotly title 키를 쓰지 않는다). 부제·칩·메타가 비면 그 칸을 만들지 않는다:
    빈 칸에도 flex gap이 끼어 제목 옆이 벌어지고, 메타가 비면 시계 아이콘만 남는다.
    네 인자 모두 호출부가 만든 HTML이다(API·파일에서 온 문자열은 호출부가 html_escape로 넘긴다).
    """
    sub = f'<span class="pd-card__sub">{subtitle}</span>' if subtitle else ""
    chips = f'<span class="pd-card__chips">{chips_html}</span>' if chips_html else ""
    head = (f'<div class="pd-card__head"><span class="pd-card__title" role="heading" aria-level="2">{title}</span>'
            f'{sub}{chips}</div>')
    if meta_html:
        head += (f'<div class="pd-card__meta"><span class="pd-ico pd-ico--clock" aria-hidden="true"></span>'
                 f'<span>{meta_html}</span></div>')
    return head


# ============================================================================
# 표시 계층: 지도 무대 · 인스펙터 (명세 5.8·6.7절)
# 숫자는 등록부 사실(곳 수·설비·지역)만 센다. 시뮬레이션 출력은 지도 무대에 넣지 않는다(J4: 추정 출력 합계가
# 실측 수요보다 커서 첫 화면 신뢰를 깎는다). 출력 추정치는 지도 팝업과 발전 현황 탭에서만 '추정'으로 보인다.
# ============================================================================

class _MapThemeSync(MacroElement):
    """컴포넌트가 iframe :root에 넣는 --background-color 밝기로 <html data-pd-theme>를 정한다(1초 간격).

    테마 판단을 iframe 안에서 하므로 지도 JS가 테마에 따라 바뀌지 않아 iframe이 다시 마운트되지 않고,
    ⋮ → Settings로 테마를 바꾸면 다음 전체 실행을 기다리지 않고 타일 필터가 따라간다(명세 J14).
    지도 변수를 쓰지 않으므로 st_folium이 지도 변수 이름을 map_div로 바꿔도 영향이 없다.
    """
    _template = Template(r"""
{% macro script(this, kwargs) %}
(function () {
  function mode() {
    var v = getComputedStyle(document.documentElement).getPropertyValue("--background-color").trim();
    var m = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})/i.exec(v), r, g, b;
    if (m) { r = parseInt(m[1], 16); g = parseInt(m[2], 16); b = parseInt(m[3], 16); }
    else { m = /(\d+)\D+(\d+)\D+(\d+)/.exec(v); if (!m) { return "light"; } r = +m[1]; g = +m[2]; b = +m[3]; }
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 128 ? "dark" : "light";
  }
  function apply() { var t = mode(); if (document.documentElement.dataset.pdTheme !== t) { document.documentElement.dataset.pdTheme = t; } }
  apply(); setInterval(apply, 1000);
})();
{% endmacro %}
""")


def _html_marker(kind: str, capacity_mw: float, label: str = "") -> Tuple[str, int]:
    """지도 기호 SVG와 DivIcon 칸 크기를 돌려준다(명세 6.7절 J8).

    모양(풍력 원·태양광 사각·ESS 마름모·화력 삼각)으로도 유형을 구분해 색만으로 유형을 전하지 않는다.
    크기는 등록부 설비용량이다: 시뮬레이션 출력으로 정하면 60초마다 기호 크기가 흔들려 추정값이 사실처럼 보인다.
    채움은 SERIES 계열 색, 외곽선은 iframe 테마 글자색이다. 라이트 바다 위 채움은 3:1에 못 미치므로 외곽선이
    경계 대비를 맡는다(명세 3.5절 H). iframe 안 Leaflet DOM이라 st.html과 달리 인라인 SVG가 지워지지 않는다.

    Args:
        kind: 'wind'·'solar'·'ess'·'thermal'.
        capacity_mw: 설비용량(MW). 유한수가 아니거나 음수면 가장 작은 기호가 된다.
        label: 화면 읽기 프로그램이 읽을 이름(이스케이프한 HTML). 비우면 SVG를 숨긴다.

    Returns:
        (SVG HTML, 칸 크기 px). 칸은 터치 대상이라 최소 24px이다.

    Raises:
        ValueError: 알 수 없는 kind. 모양 없는 기호가 조용히 그려지지 않게 한다.
    """
    shapes = {
        "wind": '<circle cx="12" cy="12" r="10.5"/>',
        "solar": '<rect x="2" y="2" width="20" height="20" rx="2.5"/>',
        "ess": '<path d="M12 1.5 22.5 12 12 22.5 1.5 12z"/>',
        "thermal": '<path d="M12 2.5 22.5 21h-21z"/>',
    }
    if kind not in shapes:
        raise ValueError(f"알 수 없는 발전 유형: {kind!r}")
    capacity = float(capacity_mw) if _finite_number(capacity_mw) else 0.0
    size = int(round(min(28, max(10, 6.4 + 1.5 * np.sqrt(max(capacity, 0.0))))))
    box = max(24, size)
    # CSS 변수는 SVG 표현 속성에 쓸 수 없고 style에서만 풀리므로 채움·외곽선을 style로 준다
    style = (f"fill:{SERIES[kind]};stroke:var(--text-color);stroke-width:1.25;stroke-opacity:.8;"
             "vector-effect:non-scaling-stroke")
    a11y = f'role="img" aria-label="{label}"' if label else 'aria-hidden="true"'
    svg = (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" {a11y} style="display:block;overflow:visible">'
           + shapes[kind].replace("/>", f' style="{style}"/>') + "</svg>")
    return svg, box


def _fmt_region(address: object) -> str:
    """주소 앞부분으로 행정시를 정한다(명세 5.8절). 추측해 넣지 않으므로 그 밖은 '미상'이다."""
    text = address.strip() if isinstance(address, str) else ""
    if text.startswith("제주시"):
        return "제주시"
    if text.startswith("서귀포시"):
        return "서귀포시"
    return "미상"


def _fmt_registry_summary(plants_df: pd.DataFrame, registry: Optional[Dict], region: str = "전체") -> Dict:
    """인스펙터·지도 머리·범례·callout이 쓰는 등록부 사실 숫자를 한 곳에서 센다(명세 5.8절, G13).

    plants_df는 get_jeju_power_plants 결과(운영중만, 분산형 합산값 제외)이고, 지역은 _fmt_region(address)로 나눈다.
    registry(load_power_plants_data 결과)에서는 지도에서 빠진 항목(운영중이 아닌 곳)과 지도 밖 자원
    (분산형 태양광 합산값, HVDC 연계)만 센다. 출력·이용률 같은 시뮬레이션 값은 세지 않는다(J4).

    Args:
        plants_df: 발전소 표. type·capacity·address·id 열을 쓴다(address가 없으면 모두 '미상').
        registry: 등록부 dict. dict가 아니면 등록부 항목 숫자는 None·0·빈 값으로 둔다.
        region: '전체'·'제주시'·'서귀포시'.

    Returns:
        selected·total·regions(제주시·서귀포시·미상) 묶음(각 count·mw·by_type{유형: count·mw}),
        excluded(상태별 count·mw)와 excluded_note("건설중 2곳·165 MW"), distributed_mw·distributed_sites,
        hvdc_count·hvdc_mw, fallback(등록부를 못 읽어 예비 목록인지).

    Raises:
        ValueError: region이 위 세 값이 아닐 때(선택지 오타가 '전체' 숫자로 조용히 바뀌지 않게).
    """
    if region not in ("전체", "제주시", "서귀포시"):
        raise ValueError(f"알 수 없는 지역: {region!r}")
    kinds = ("wind", "solar", "ess", "thermal")
    df = plants_df if isinstance(plants_df, pd.DataFrame) else pd.DataFrame()
    types = df["type"] if "type" in df.columns else pd.Series("", index=df.index)
    caps = (pd.to_numeric(df["capacity"], errors="coerce").fillna(0.0) if "capacity" in df.columns
            else pd.Series(0.0, index=df.index))
    regions = df["address"].map(_fmt_region) if "address" in df.columns else pd.Series("미상", index=df.index)

    def block(mask: pd.Series) -> Dict:
        return {
            "count": int(mask.sum()),
            "mw": float(caps[mask].sum()),
            "by_type": {kind: {"count": int((mask & (types == kind)).sum()),
                               "mw": float(caps[mask & (types == kind)].sum())} for kind in kinds},
        }

    total = block(pd.Series(True, index=df.index))
    by_region = {name: block(regions == name) for name in ("제주시", "서귀포시", "미상")}

    reg = registry if isinstance(registry, dict) else {}
    excluded: Dict[str, Dict] = {}
    for key in ("wind_farms", "solar_farms", "ess_facilities", "thermal_plants"):
        items = reg.get(key)
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("subtype") == "distributed":
                continue
            status = str(item.get("status") or "상태 미상").strip()
            if status == "운영중":
                continue
            entry = excluded.setdefault(status, {"count": 0, "mw": 0.0})
            entry["count"] += 1
            entry["mw"] += float(item["capacity_mw"]) if _finite_number(item.get("capacity_mw")) else 0.0
    # 165.0 MW처럼 정수면 "165 MW", 아니면 소수 1자리(명세 4.2절 "건설중 2곳·165 MW 제외")
    excluded_note = " · ".join(
        f'{status} {e["count"]}곳·' + (f'{e["mw"]:,.0f}' if abs(e["mw"] - round(e["mw"])) < 0.05 else f'{e["mw"]:,.1f}')
        + " MW" for status, e in excluded.items())

    solar_items = reg.get("solar_farms") if isinstance(reg.get("solar_farms"), list) else []
    distributed = next((s for s in solar_items if isinstance(s, dict) and s.get("subtype") == "distributed"), None)
    distributed_mw = (float(distributed["capacity_mw"])
                      if distributed is not None and _finite_number(distributed.get("capacity_mw")) else None)
    # 개소 수는 등록부에 숫자 칸이 없고 운영사 문구 "다수 (1,620개소)"에만 있다
    sites = re.search(r"([\d,]+)\s*개소", str(distributed.get("operator") or "")) if distributed is not None else None
    hvdc_items = reg.get("hvdc_links") if isinstance(reg.get("hvdc_links"), list) else []
    hvdc = [h for h in hvdc_items if isinstance(h, dict) and h.get("status") == "운영중"]
    ids = df["id"].astype(str) if "id" in df.columns else pd.Series("", index=df.index, dtype=str)
    return {
        "selected": total if region == "전체" else by_region[region],
        "total": total,
        "regions": by_region,
        "excluded": excluded,
        "excluded_note": excluded_note,
        "distributed_mw": distributed_mw,
        "distributed_sites": int(sites.group(1).replace(",", "")) if sites else None,
        "hvdc_count": len(hvdc),
        "hvdc_mw": float(sum(float(h["capacity_mw"]) for h in hvdc if _finite_number(h.get("capacity_mw")))),
        "fallback": bool(ids.str.startswith("fallback_").any()),
    }


def _fmt_registry_ref(registry: Optional[Dict]) -> str:
    """등록부 기준일 문구 "등록부 2024년 기준 · 2025-12-18 갱신"(metadata.data_reference·last_updated)."""
    meta = registry.get("metadata") if isinstance(registry, dict) else None
    meta = meta if isinstance(meta, dict) else {}
    reference = str(meta.get("data_reference") or "").strip()
    basis = re.search(r"\d{4}년\s*기준", reference)
    updated = str(meta.get("last_updated") or "").strip()
    parts = [p for p in ((basis.group(0) if basis else reference), (f"{updated} 갱신" if updated else "")) if p]
    return "등록부 " + " · ".join(parts) if parts else "등록부 기준일 확인 불가"


def _fmt_registry_sources(registry: Optional[Dict]) -> str:
    """등록부 metadata.sources를 " · "로 그대로 잇는다(없으면 빈 문자열).

    함수 docstring에만 있는 출처(한국에너지공단·thewindpower.net)는 등록부 JSON에 없으므로 쓰지 않는다(명세 5.8절).
    """
    meta = registry.get("metadata") if isinstance(registry, dict) else None
    sources = meta.get("sources") if isinstance(meta, dict) else None
    if not isinstance(sources, list):
        return ""
    return " · ".join(str(s).strip() for s in sources if str(s).strip())


def _html_map_head(plants_df: pd.DataFrame, registry: Optional[Dict]) -> str:
    """지도 무대 머리(명세 5.8절): '발전소 지도' + 운영중 곳 수·설비(건설중 제외 표기) + 등록부 칩."""
    summary = _fmt_registry_summary(plants_df, registry, "전체")
    total = summary["total"]
    title = '<span class="pd-ico pd-ico--map" aria-hidden="true"></span>발전소 지도'
    # 지도 iframe은 기호·확대 버튼·출처 링크로 Tab 자리 약 30개를 차지한다. 키보드 사용자가 지도 아래(범례)로 건너뛸 수 있게
    # 초점을 받을 때만 보이는 링크를 둔다(1차 평가). 목표는 _html_map_legend의 id
    skip = '<a class="pd-skip" href="#pd-map-end">지도 건너뛰기</a>'
    if summary["fallback"]:
        return _html_card_head(title, f'예비 목록 {total["count"]}곳 · 등록부를 읽지 못함',
                               _html_chip("est", "예비 목록 · 등록부 아님"), "") + skip
    note = f' ({summary["excluded_note"]} 제외)' if summary["excluded_note"] else ""
    return _html_card_head(title, f'운영중 {total["count"]}곳 · 설비 {total["mw"]:,.1f} MW{note}',
                           _html_chip("measured", "위치·설비 등록부"), "") + skip


def _html_map_legend(plants_df: pd.DataFrame) -> str:
    """지도 밖 범례(명세 5.8절): 유형별 기호·곳 수·설비 + 기호 크기 읽는 법.

    옛 지도 안 고정 범례는 iframe을 줄이면 지도를 가렸고 화력이 빠져 있었다(D7). 기호는 지도와 같은 모양의
    CSS 도형이고 색은 SERIES 인라인 --c라 지도·범례·표가 같은 색이다(F5).
    """
    by_type = _fmt_registry_summary(plants_df, None, "전체")["total"]["by_type"]
    items = (("wind", "풍력"), ("solar", "태양광"), ("ess", "ESS"), ("thermal", "화력"))
    cells = "".join(
        f'<span class="pd-maplegend__item"><span class="pd-mkey pd-mkey--{kind}" style="--c:{SERIES[kind]}" '
        f'aria-hidden="true"></span><b>{name}</b>{by_type[kind]["count"]}곳 · {by_type[kind]["mw"]:,.1f} MW</span>'
        for kind, name in items
    )
    return ('<div class="pd-maplegend" id="pd-map-end" tabindex="-1" role="group" aria-label="발전소 지도 범례">' + cells
            + '<span class="pd-maplegend__note">기호 크기 = 설비용량 (예: 1 · 30 · 240 MW) · 목록은 발전 현황 탭</span>'
            + '</div>')


def _html_inspector(plants_df: pd.DataFrame, registry: Optional[Dict], region: str) -> str:
    """지역 인스펙터(명세 5.8절): 경로 · 통계 3칸 · 지역 표 · 지도 밖 자원 · 출처.

    등록부 사실만 보인다(J4: 시뮬레이션 출력·이용률·0–24시 출력 막대를 넣지 않는다). 지역을 고르면 통계가
    그 지역 값이 되고 표의 그 행에 aria-current와 강조색을 준다. 지도 시점은 바꾸지 않는다(지도를 다시
    마운트하지 않게, 지역 확대는 10장 후속 L1).
    """
    summary = _fmt_registry_summary(plants_df, registry, region)
    selected, total = summary["selected"], summary["total"]
    kinds = (("wind", "풍력"), ("solar", "태양광"), ("ess", "ESS"), ("thermal", "화력"))
    src = "예비 목록" if summary["fallback"] else "등록부"

    def bar(block: Dict) -> str:
        """유형별 설비 비중 100% 막대. 색만으로 읽지 않게 비중 글자를 aria-label·title로 함께 둔다."""
        parts = [(kind, name, block["by_type"][kind]["mw"] / block["mw"] * 100) for kind, name in kinds
                 if block["mw"] > 0 and block["by_type"][kind]["mw"] > 0]
        text = " · ".join(f"{name} {pct:.0f}%" for _, name, pct in parts) or "설비 없음"
        spans = "".join(f'<span style="width:{pct:.1f}%;--c:{SERIES[kind]}"></span>' for kind, _, pct in parts)
        return f'<span class="pd-bar" role="img" aria-label="설비 구성 {text}" title="{text}">{spans}</span>'

    def row(name: str, block: Dict, current: bool, child: bool) -> str:
        cls = ' class="pd-insp__child"' if child else ""
        cur = ' aria-current="true"' if current else ""
        return (f'<tr{cls}{cur}><td>{name}</td><td class="pd-num">{block["count"]}</td>'
                f'<td class="pd-num">{block["mw"]:,.1f}</td><td class="pd-insp__mix">{bar(block)}</td></tr>')

    rows = [row("제주도 전체", total, region == "전체", False)]
    for name in ("제주시", "서귀포시", "미상"):
        block = summary["regions"][name]
        if name == "미상" and block["count"] == 0:
            continue  # 주소로 지역을 못 정한 곳이 있을 때만 보인다(추측해 넣지 않는다)
        rows.append(row(name, block, region == name, True))
    dist_mw, dist_sites = summary["distributed_mw"], summary["distributed_sites"]
    dist_mw_text = f"{dist_mw:,.1f}" if dist_mw is not None else "—"
    dist_sites_cell = f"{dist_sites:,}개소" if dist_sites is not None else "—"
    hvdc_count, hvdc_mw = summary["hvdc_count"], summary["hvdc_mw"]
    rows.append('<tr class="pd-insp__sect"><td colspan="4">지도 밖 자원</td></tr>')
    # 덧말은 넓은 폭에서는 4번째 칸, 640px 이하(구성 열을 숨기는 폭)에서는 이름 아래 둘째 줄(rowsub)로 보인다.
    # 좁은 폭에서 이 두 행만 4칸이라 머리 띠·행 선이 다른 행보다 길게 끝났다(1차 평가)
    for res_name, sites_cell, mw_cell, res_note in (
        ("분산형 태양광", dist_sites_cell, dist_mw_text, "위치 없음"),
        ("HVDC 연계", f"{hvdc_count}회선" if hvdc_count else "—", f"{hvdc_mw:,.1f}" if hvdc_count else "—",
         "육지 연계 · 송전량 실측 미연동"),
    ):
        rows.append(f'<tr class="pd-insp__child"><td>{res_name}<small class="pd-insp__rowsub">{res_note}</small></td>'
                    f'<td class="pd-num">{sites_cell}</td><td class="pd-num">{mw_cell}</td>'
                    f'<td class="pd-insp__note">{res_note}</td></tr>')
    table = ('<table class="pd-insp__table" aria-label="지역별 발전소 등록부 요약"><thead><tr>'
             '<th scope="col">지역</th><th scope="col" class="pd-num">발전소</th>'
             '<th scope="col" class="pd-num">설비 MW</th><th scope="col" class="pd-insp__mix">설비 구성</th>'
             '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>')

    if region == "전체":
        unit = "<small>MW</small>" if dist_mw is not None else ""
        third = (f'<div class="pd-stat"><span class="pd-stat__label">지도 밖 분산형 태양광</span>'
                 f'<span class="pd-stat__value pd-num">{dist_mw_text}{unit}</span>'
                 f'<span class="pd-stat__src">{dist_sites_cell if dist_sites is not None else "개소 수 미제공"} · 위치 없음'
                 '</span></div>')
    else:
        mix = " · ".join(f'{name} {selected["by_type"][kind]["count"]}' for kind, name in kinds)
        third = (f'<div class="pd-stat"><span class="pd-stat__label">유형 구성 (곳)</span>'
                 f'<span class="pd-stat__value pd-stat__value--text">{mix}</span>'
                 f'<span class="pd-stat__src">{src}</span></div>')
    stats = (f'<div class="pd-stat"><span class="pd-stat__label">발전소</span>'
             f'<span class="pd-stat__value pd-num">{selected["count"]}<small>곳</small></span>'
             f'<span class="pd-stat__src">{src}</span></div>'
             f'<div class="pd-stat"><span class="pd-stat__label">설비용량</span>'
             f'<span class="pd-stat__value pd-num">{selected["mw"]:,.1f}<small>MW</small></span>'
             f'<span class="pd-stat__src">{src}</span></div>' + third)

    crumb = (f'<div class="pd-insp__crumb"><span>제주도</span><span class="pd-ico pd-ico--chev" aria-hidden="true">'
             f'</span><b>{region}</b><span class="pd-insp__ref">{html_escape(_fmt_registry_ref(registry))}</span></div>')
    notice = (f'<p class="pd-insp__notice">등록부를 읽지 못해 예비 목록 {total["count"]}곳만 표시합니다</p>'
              if summary["fallback"] else "")
    sources = html_escape(_fmt_registry_sources(registry))
    foot = (f'<p class="pd-insp__foot">출처 발전소 등록부{f"({sources})" if sources else ""} · '
            '출력 추정치는 발전 현황 탭(시뮬레이션 · 실측 아님)</p>')
    return (f'<div class="pd-insp"><div class="pd-insp__side">{crumb}{notice}<div class="pd-insp__stats">{stats}'
            f'</div></div><div class="pd-insp__main">{table}{foot}</div></div>')


# ============================================================================
# 표시 계층: 수급 추이 카드 · 상태 줄 (명세 5.7·5.9·6.2절)
# 곡선은 추정 패턴이고 실측일 수 있는 것은 현재값뿐이다(N3). 자리마다 문구는 _fmt_provenance가 한 곳에서 정하고
# 여기서는 그대로 쓴다. 숫자는 화면 KPI와 같은 반올림 정수로 계산해 보인 숫자끼리 맞게 한다.
# ============================================================================

def _html_card_foot(text: str) -> str:
    """카드 각주(명세 5.7절): 원천 목록 + 화면 갱신 시각. 12px text-3.

    text는 호출부가 만든 HTML이다(_fmt_provenance 문구는 외부 문자열을 이미 이스케이프했다).
    """
    return f'<p class="pd-card__foot">{text}</p>'


def _html_supply_legend(power_status: Dict, prov: Dict) -> str:
    """수급 차트의 범례 겸 표(명세 5.7절, Open Electricity 문법): 견본 · 이름 · 읽는 법 | 현재값 · 값 출처.

    Plotly 범례 대신 이 표가 계열을 설명한다(D5). 견본은 차트와 같은 선 모양이고, 현재 점 모양도 같다
    (채운 원 = 지금 잰 값, 빈 원 = 실측 아님). 지금 잰 값이 아니면 'KPX 조회'라는 말을 쓰지 않는다(7.2b절).
    MW 차이·경계 여유는 화면에 보인 반올림 정수끼리 계산한다(보인 두 수의 차이와 맞게).
    """
    demand, supply = power_status.get("demand"), power_status.get("total_supply")
    reserve_rate = power_status.get("reserve_rate")
    d_mw = int(f"{demand:.0f}") if _finite_number(demand) else None
    s_mw = int(f"{supply:.0f}") if _finite_number(supply) else None
    b_mw = int(f"{supply / 1.15:.0f}") if s_mw is not None else None

    def mw(value: Optional[int]) -> str:
        return "—" if value is None else f"{value:,} MW".replace("-", "−")

    query = f"조회 {prov['query_hhmm']} KST"
    demand_src = {"measured": query, "stale": "EPSIS 파일 · 지연", "est": "추정 · 실측 아님",
                  "sim": "시뮬레이션 · 실측 아님", "test": "테스트 값"}.get(prov["demand_state"], "확인 불가")
    supply_src = query if prov["supply_measured"] else "추정 · 실측 아님"
    if prov["reserve_state"] == "measured":
        reserve_how = "KPX 값"
    elif prov["reserve_state"] == "stale":
        reserve_how = "EPSIS 파일 값 (지연)"
    else:
        reserve_how = prov["reserve_note"].lstrip(" ·") or "실측 아님"
    reserve_val = mw(s_mw - d_mw if s_mw is not None and d_mw is not None else None)
    if _finite_number(reserve_rate):
        reserve_val += f" · {reserve_rate:.1f}%"
    if b_mw is not None and d_mw is not None:
        margin = b_mw - d_mw
        margin_txt = f"여유 +{margin:,} MW" if margin >= 0 else f"경계보다 수요가 {-margin:,} MW 많음"
    else:
        margin_txt = "여유 계산 불가"

    def cell(swatch: str, color: str, name: str, desc: str, value: str, src: str) -> str:
        return (f'<li class="pd-legend-table__cell"><span class="pd-swatch {swatch}" style="--c:{color}" '
                f'aria-hidden="true"></span><span class="pd-legend-table__txt">'
                f'<span class="pd-legend-table__name">{name}</span><span class="pd-legend-table__desc">{desc}</span>'
                f'</span><span class="pd-legend-table__val pd-num">{value}<small>{src}</small></span></li>')

    cells = (
        cell("pd-swatch--line" + ("" if prov["demand_measured"] else " is-open"), SERIES["demand"], "수요",
             prov["legend_demand_note"], mw(d_mw), demand_src),
        cell("pd-swatch--line" + ("" if prov["supply_measured"] else " is-open"), SERIES["supply"], "공급능력",
             prov["legend_supply_note"], mw(s_mw), supply_src),
        cell("pd-swatch--hatch", SERIES["reserve"], "예비력", f"공급능력 − 수요 (MW) · 예비율은 {reserve_how}",
             reserve_val, "계산값"),
        cell("pd-swatch--dash", BOUNDARY_COLOR, "관심 경계", f"공급능력 ÷ 1.15 · {margin_txt}", mw(b_mw), "계산값"),
    )
    return f'<ul class="pd-legend-table" aria-label="수급 추이 범례와 현재값">{"".join(cells)}</ul>'


def _html_time_strip(prov: Dict) -> str:
    """차트 아래 시간 띠(명세 5.7절, C·EM 스크러버 이식): 지난 12시간 · 지금 · 다음 12시간이 무엇인지 글로 밝힌다.

    플롯 안에 구간 글자를 넣으면 좁은 폭에서 선·눈금과 겹치므로 여기서 말한다. 같은 뜻이 범례 겸 표·각주에도
    있어 화면 읽기 프로그램에는 숨긴다. 가운데 칸은 prov['strip_now']의 첫 글자(● 실측 · ○ 그 밖)를 CSS 점으로 바꿔
    칩 글리프와 같은 모양으로 그린다. 실측 이력·운영 예측이 연결되면(10장 P2·P3) 같은 자리의 문구만 바뀐다.
    """
    now_text = prov["strip_now"]
    dot = ""
    if now_text[:2] in ("● ", "○ "):
        dot = ('<span class="pd-timestrip__dot" aria-hidden="true"></span>' if now_text[0] == "●"
               else '<span class="pd-timestrip__dot is-open" aria-hidden="true"></span>')
        now_text = now_text[2:]
    seg = '<span class="pd-timestrip__seg"><span class="pd-glyph pd-glyph--est" aria-hidden="true"></span>{}</span>'
    return ('<div class="pd-timestrip" aria-hidden="true">'
            + seg.format("지난 12시간 · 일간 패턴 추정")
            + f'<span class="pd-timestrip__now">{dot}<span>{now_text}</span></span>'
            + seg.format("다음 12시간 · 패턴 연장 · 예측 모델 아님")
            + '</div>')


def _chart_table_supply(fig: go.Figure, now_label: str = "지금") -> pd.DataFrame:
    """수급 차트의 '표로 보기'(명세 6.2절, J10): 30분 점을 하나씩 건너뛴 1시간 간격 표.

    차트에 실제로 그린 값을 옮긴다(차트와 표가 어긋나지 않게). 지금 점을 기준으로 앞뒤 1시간마다 한 행씩 고르므로
    과거 24점(지금 − 11.5시간부터)에서는 지금 − 11시간이 첫 행이다. 지금 행의 구분 칸에만 출처 상태 문구
    (prov['strip_now'])를 쓴다.

    Args:
        fig: create_realtime_power_chart가 만든 그림.
        now_label: 지금 행의 구분 칸 문구.

    Returns:
        열 시각 · 수요 (MW) · 공급능력 (MW) · 구분. 그림에 곡선이 없으면(현재값 없음) 빈 표.
    """
    columns = ["시각", "수요 (MW)", "공급능력 (MW)", "구분"]
    traces = {t.name: t for t in fig.data}
    demand, supply, point = (traces.get("수요 (추정 패턴)"), traces.get("공급능력 (추정 패턴)"),
                             traces.get("수요 현재값"))
    if demand is None or supply is None or point is None or not len(demand.x):
        return pd.DataFrame(columns=columns)
    xs = pd.to_datetime(list(demand.x))
    now_x = pd.to_datetime(point.x[0])
    now_i = int(np.argmin(np.abs((xs - now_x).total_seconds())))
    rows = []
    for i in range(now_i % 2, len(xs), 2):
        kind = now_label if i == now_i else ("지난 추정 패턴" if i < now_i else "다음 추정 패턴")
        rows.append([f"{xs[i]:%m-%d %H:%M}", int(f"{demand.y[i]:.0f}"), int(f"{supply.y[i]:.0f}"), kind])
    return pd.DataFrame(rows, columns=columns)


def _fmt_supply_columns() -> Dict:
    """수급 '표로 보기' 열 설정: MW 열에 천 단위 구분을 넣는다(F12).

    st.dataframe의 정수 기본 형식은 구분 기호가 없다(S4 실측 '1651'). 글자로 바꾸면 오른쪽 정렬이 사라지므로
    숫자 열 그대로 두고 브라우저 지역 형식('localized')을 쓴다.
    """
    return {
        "수요 (MW)": st.column_config.NumberColumn(format="localized"),
        "공급능력 (MW)": st.column_config.NumberColumn(format="localized"),
    }


def _html_gauge_label(name: str, how_to_read: str) -> str:
    """게이지 위 이름표(명세 5.9절). 게이지 그림에는 제목을 넣지 않고(title='') 이름과 읽는 법을 HTML로 둔다."""
    return (f'<div class="pd-gauge-label"><span class="pd-gauge-label__name">{name}</span>'
            f'<span class="pd-gauge-label__how">{how_to_read}</span></div>')


def _html_grid_state(power_status: Dict, level: Dict, prov: Dict) -> str:
    """계통 상태 카드(명세 5.9절, B 이식, J6): 예비율 단계 사다리 + 다음 단계까지 수요 여유 MW.

    같은 숫자를 네 번 되풀이하던 수급 표 대신 운영 판단에 쓰는 숫자 하나를 보인다.
    여유 MW = 공급능력 ÷ 다음 경계비 − 수요(정상 → 관심 1.15, 관심 → 주의 1.10, 주의 → 위험 1.05).
    경계 비율은 tests/test_dashboard.py의 5/10/15% 기준과 같다. 화면에 보인 반올림 정수끼리 계산한다.
    예비율이 기본값·전국 값이면(reserve_known 거짓) 현재 칸을 강조하지 않고 단계를 판정하지 않는다고 쓴다
    ('정상'·'넉넉'을 말하지 않는다, 7.2절). 사다리 4칸은 순서 척도라 같은 폭이다.
    """
    known = prov["reserve_known"]
    steps = (("critical", "status-critical", "위험", "5% 미만"), ("danger", "status-danger", "주의", "5–10%"),
             ("warning", "status-warning", "관심", "10–15%"), ("normal", "status-online", "정상", "15% 이상"))
    items = []
    for status, cls, name, band in steps:
        if known and level["status"] == status:
            items.append(f'<li class="pd-ladder__step {cls} is-current" aria-current="step">{name}<small>{band}</small></li>')
        else:
            items.append(f'<li class="pd-ladder__step {cls}">{name}<small>{band}</small></li>')

    demand, supply = power_status.get("demand"), power_status.get("total_supply")
    reserve_rate = power_status.get("reserve_rate")
    rate_txt = f"{reserve_rate:.1f}%" if _finite_number(reserve_rate) else "—"
    have_mw = _finite_number(demand) and _finite_number(supply)
    if not known:
        value = ("예비율 기본값이라 단계를 판정하지 않습니다" if prov["reserve_state"] == "default"
                 else "전국 예비율이라 제주 단계를 판정하지 않습니다")
    elif level["next_ratio"] is None:
        value = f'<b class="pd-num">{rate_txt}</b> 예비율 · 위험 단계 · 예비율 5% 미만'
    elif have_mw:
        margin = int(f"{supply / level['next_ratio']:.0f}") - int(f"{demand:.0f}")
        margin_txt = f"+{margin:,}" if margin >= 0 else f"−{-margin:,}"
        value = (f'<b class="pd-num">{rate_txt}</b> 예비율 · <b class="pd-num">{margin_txt} MW</b> '
                 f'{level["next_name"]} 단계까지 수요 여유')
        if margin < 0:
            value += " (두 값의 조회 시점이 다를 수 있음)"
    else:
        value = f'<b class="pd-num">{rate_txt}</b> 예비율 · 수요·공급능력 값이 없어 여유를 계산하지 않습니다'

    # 각주는 값 줄이 쓴 경계를 풀어 쓴다. 위험 단계는 다음 경계가 없어 지나온 위험 경계를, 판정 안 함이면 관심 경계를 보인다
    if known and level["next_ratio"] is not None:
        ratio, bound = level["next_ratio"], level["next_name"]
    elif known:
        ratio, bound = 1.05, "위험"
    else:
        ratio, bound = 1.15, "관심"
    if have_mw:
        note = (f'{bound} 경계 {int(f"{supply / ratio:.0f}"):,} MW = 공급능력 {int(f"{supply:.0f}"):,} MW ÷ {ratio:.2f}'
                f' · {prov["status_src"]}')
    else:
        note = prov["status_src"]
    return (f'<div class="pd-grid-state">{_html_card_head("계통 상태", "", prov["grid_chip"], "")}'
            f'<ol class="pd-ladder" aria-label="예비율 단계">{"".join(items)}</ol>'
            f'<p class="pd-grid-state__value">{value}</p>'
            f'<p class="pd-card__note">{note}</p></div>')


# ============================================================================
# 표시 계층: SMP 예측 · 발전 현황 · 분석 탭 (6단계, 명세 5.10·5.11·6.3·6.5·6.6절)
# SMP 탭은 live_smp, 도넛·타임라인 행은 live_generation fragment(300초)가 그린다. 값은 바꾸지 않고(F7) 칩·문구·표로
# 이 숫자가 무엇인지를 밝힌다: 확정 가격(EPSIS 하루전) · 예측(API 모델) · 추정(대체값·패턴·시뮬레이션) · 예시.
# ============================================================================

def _fmt_screen_meta(kind: str, now: datetime, cache_note: str) -> str:
    """카드 메타 "시뮬레이션 · 화면 23:36 KST (60초 캐시)"(명세 5.2·5.11절).

    관측·조회 시각을 주는 원천이 없는 값은 화면을 그린 시각과 캐시 길이만 밝힌다(관측 시각처럼 보이지 않게).
    """
    return f"{kind} · 화면 {now:%H:%M} KST ({cache_note})"


def _fmt_smp_with_keys(smp_data: Dict, now: datetime) -> Dict:
    """SMP 출처 판정 표시 키(pd_*)가 붙은 dict. fragment가 이미 붙여 넘겼으면 그대로 돌려준다."""
    if "pd_fc_kind" in smp_data:
        return smp_data
    return {**smp_data, **_fmt_smp_display_keys(smp_data, now)}


def _fmt_smp_slots(smp_data: Dict, now: datetime) -> Dict:
    """SMP 예측 24칸을 시각 축에 놓는다(명세 6.3절 x축 규칙). 차트·표·시간대 블록·머리가 같은 칸을 쓴다.

    - 예측일이 있고 API 예측이면(current_index 있음): 예측일 0시부터 24칸. 인덱스 i = i시~i+1시 = 거래시간 i+1.
    - 예측일이 없으면(앱 로컬 대체, API '패턴 기반 대체값'·'EPSIS Fallback'): 지금 시각부터 24칸, 첫 칸이 현재 거래시간.
      API 값은 거래시간 순서라 지금 칸부터 돌려 놓고, 로컬 값은 get_smp_data가 이미 지금부터 만들었다.
    거래시간 H는 (H-1)시~H시 구간이다(get_smp_data와 같은 정의).

    Returns:
        slots(칸마다 start·end·trade_hour·q10·q50·q90), date_axis(예측일 축인지), cur(지금 칸, 축 밖이면 None),
        past_until(지난 구간 음영의 끝 시각, 없으면 None), confirmed(확정 가격 칸, 없으면 None),
        forecast_date(예측일 'YYYY-MM-DD' 또는 None). 예측 목록이 비었거나 깨졌으면 slots는 빈 목록이다.
    """
    empty = {"slots": [], "date_axis": False, "cur": None, "past_until": None, "confirmed": None,
             "forecast_date": None}
    preds = smp_data.get("predictions")
    if not isinstance(preds, list) or not preds or not all(
            isinstance(p, dict) and _finite_number(p.get("smp")) for p in preds):
        return empty

    def value(p: Dict, key: str) -> float:
        # 분위수가 없거나 깨졌으면 중앙값으로 둔다(그 칸에서 띠 폭이 0이 된다)
        return round(float(p[key]), 1) if _finite_number(p.get(key)) else round(float(p["smp"]), 1)

    fdate = None
    if isinstance(smp_data.get("forecast_date"), str):
        try:
            fdate = datetime.strptime(smp_data["forecast_date"].strip()[:10], "%Y-%m-%d")
        except ValueError:
            fdate = None
    api_order = "current_index" in smp_data and len(preds) == 24
    date_axis = fdate is not None and api_order
    hour0 = now.replace(minute=0, second=0, microsecond=0)
    slots = []
    for h in range(len(preds)):
        if date_axis:
            p, start, trade_hour = preds[h], fdate + timedelta(hours=h), h + 1
        else:
            # 날짜 없는 API 값은 지금 거래시간(인덱스 now.hour)부터 돌린다. 로컬 값은 h번째가 지금 + h시간이다
            p = preds[(now.hour + h) % len(preds)] if api_order else preds[h]
            start = hour0 + timedelta(hours=h)
            trade_hour = start.hour + 1
        slots.append({"start": start, "end": start + timedelta(hours=1), "trade_hour": trade_hour,
                      "q50": round(float(p["smp"]), 1), "q10": value(p, "q10"), "q90": value(p, "q90")})

    cur = past_until = confirmed = None
    if date_axis:
        k = int((now - fdate).total_seconds() // 3600)
        if 0 <= k < len(slots):
            cur, past_until = k, slots[k]["start"]
        elif k >= len(slots):
            past_until = slots[-1]["end"]  # 예측일이 이미 지났다: 축 전체가 지난 구간
        # 확정 가격 칸: _fmt_smp_display_keys의 pd_confirmed_point와 같은 조건(EPSIS + 거래시간 날짜 = 예측일)
        found = re.match(r"(\d{4}-\d{2}-\d{2}) (\d{2})h$", str(smp_data.get("trade_label") or "").strip())
        if (found and str(smp_data.get("data_source", "")).startswith("EPSIS")
                and found.group(1) == f"{fdate:%Y-%m-%d}" and 1 <= int(found.group(2)) <= len(slots)):
            confirmed = int(found.group(2)) - 1
    else:
        cur = 0
    return {"slots": slots, "date_axis": date_axis, "cur": cur, "past_until": past_until, "confirmed": confirmed,
            "forecast_date": f"{fdate:%Y-%m-%d}" if fdate is not None else None}


def _html_smp_kind_chip(keys: Dict) -> str:
    """SMP 예측 종류 칩(명세 5.10절): 모델이면 예측 칩, API 대체값·앱 로컬 대체면 추정 칩."""
    if keys["pd_fc_kind"] == "model":
        return _html_chip("forecast", f"예측 · {keys['pd_fc_short']}")
    if keys["pd_fc_kind"] == "api_fallback":
        return _html_chip("est", "추정 · API 대체값 (모델 예측 아님)")
    return _html_chip("est", "추정 · 시간대 평균 패턴 (예측 모델 아님)")


def _html_smp_head(smp_data: Dict, now: datetime, *, refresh_note: str = "5분마다 갱신") -> str:
    """SMP 차트 카드 머리(명세 5.10절): 제목 · 축 범위 · 예측/추정 칩 · 확정 가격 칩 · 지연 칩 + 메타 + 쉬운 설명.

    입력 지연은 예전처럼 그림 안 빨간 주석이 아니라 '예측일 지연' 칩으로 밝힌다. 예측일이 내일이면 지연이 아니므로
    칩을 달지 않는다(부제의 예측일이 그 사실을 보인다). 단위는 축 제목 대신 부제에 둔다.
    """
    keys = _fmt_smp_with_keys(smp_data, now)
    axis = _fmt_smp_slots(smp_data, now)
    fdate = axis["forecast_date"]
    subtitle = (f"예측일 {fdate} · 거래시간 1–24 · 원/kWh" if axis["date_axis"] else "지금부터 24시간 · 원/kWh")
    chips = _html_smp_kind_chip(keys)
    point = bool(keys["pd_confirmed_point"] and axis["confirmed"] is not None)
    if point:
        chips += _html_chip("measured", "확정 가격 1점")
    if fdate is not None and fdate < f"{now:%Y-%m-%d}":
        chips += _html_chip("stale", "예측일 지연")
    meta = f"예측일 {fdate or '미제공'} · 화면 {now:%H:%M} KST ({refresh_note})"
    if keys["pd_fc_kind"] == "model":
        explain = "SMP(계통한계가격)는 시간대별 도매 전력 가격입니다. 대시 계단 = 예측 중앙값, 띠 = 80% 범위"
    else:
        explain = "SMP(계통한계가격)는 시간대별 도매 전력 가격입니다. 점선 계단 = 추정 패턴(예측 모델 아님)"
    # 같은 날이어도 오른쪽 통계(그날 확정 가격)와 계단(예측값)의 최고·최저가 달라 모순처럼 읽혔고, '거래시간 10h'가 몇 시인지
    # 풀어 쓴 곳이 없었다(1차 평가 멘토 5초 판독) → 점이 1개뿐인 이유와 거래시간 읽는 법을 한 번 밝힌다
    explain += (", 점 = 지금 거래시간 확정 가격(하루 확정 가격 통계는 SMP 통계 카드)." if point else ".")
    # 예시는 차트 pill과 같은 거래시간으로 든다(pill '거래시간 11h'가 바로 풀린다). 지금 칸이 축 밖이면 일반 규칙만 쓴다
    if axis["cur"] is not None:
        hour_now = axis["slots"][axis["cur"]]["trade_hour"]
        explain += f" 거래시간 {hour_now:02d}h = {hour_now - 1:02d}–{hour_now:02d}시."
    else:
        explain += " 거래시간 Nh = (N-1)시–N시."
    return (_html_card_head("제주 SMP 24시간", subtitle, chips, meta)
            + f'<p class="pd-card__note pd-smp-explain">{explain}</p>')


def _chart_table_smp(smp_data: Dict, *, now: Optional[datetime] = None) -> pd.DataFrame:
    """SMP '표' 보기(명세 5.10절, J10): 차트와 같은 24칸을 거래시간 · 구간 · 중앙값 · (q10 · q90) · 확정 가격으로.

    q10·q90 열은 모델 예측일 때만 넣는다: 로컬 대체의 ±15%는 가정값이고 API 대체값의 분위수도 모델 구간이 아니다.
    확정 가격 열은 숫자가 한 칸뿐이라 빈 칸이 'None'으로 보이지 않게 글자 열('—')로 둔다.
    """
    now = (now or datetime.now()).replace(second=0, microsecond=0)
    keys = _fmt_smp_with_keys(smp_data, now)
    axis = _fmt_smp_slots(smp_data, now)
    is_model = keys["pd_fc_kind"] == "model"
    median_col = "예측 중앙값 (원/kWh)" if is_model else "추정 패턴 (원/kWh)"
    point = axis["confirmed"] if keys["pd_confirmed_point"] and _finite_number(smp_data.get("current")) else None
    rows = []
    for i, slot in enumerate(axis["slots"]):
        end = "24:00" if slot["end"].hour == 0 else f"{slot['end']:%H:%M}"
        row = {"거래시간": f"{slot['trade_hour']:02d}h" + (" · 지금" if i == axis["cur"] else ""),
               "구간": f"{slot['start']:%H:%M}–{end}", median_col: slot["q50"]}
        if is_model:
            row["q10 (원/kWh)"] = slot["q10"]
            row["q90 (원/kWh)"] = slot["q90"]
        row["확정 가격 (원/kWh)"] = f"{float(smp_data['current']):,.1f}" if i == point else "—"
        rows.append(row)
    columns = ["거래시간", "구간", median_col] + (["q10 (원/kWh)", "q90 (원/kWh)"] if is_model else []) + ["확정 가격 (원/kWh)"]
    return pd.DataFrame(rows, columns=columns)


def _fmt_number_columns(names: List[str], fmt: str = "%.1f") -> Dict:
    """st.dataframe 숫자 열 형식(소수 자릿수를 열마다 맞춘다). 표에 없는 이름은 Streamlit이 무시한다."""
    return {name: st.column_config.NumberColumn(format=fmt) for name in names}


def _fmt_smp_foot(smp_data: Dict, now: datetime) -> str:
    """SMP 카드 출처 줄(명세 5.10절): 예측 원천 · 확정 가격 원천 · 화면 갱신 시각."""
    keys = _fmt_smp_with_keys(smp_data, now)
    if keys["pd_fc_kind"] == "model":
        first = f"출처: 예측 = API 서버 SMP 예측 ({keys['pd_fc_short']})"
    elif keys["pd_fc_kind"] == "api_fallback":
        first = "출처: 예측 = API 서버 대체값 (모델 예측 아님)"
    else:
        first = "출처: 예측 = 앱 로컬 대체 (API 예측 미수신 · 시간대 평균 패턴, 예측 모델 아님)"
    parts = [first]
    if keys["pd_confirmed_point"]:
        parts.append(f"확정 가격 = {keys['pd_src_label']} · 거래시간 {html_escape(str(smp_data.get('trade_label')))}")
    parts.append(f"화면 갱신 {now:%H:%M} KST")
    return " · ".join(parts)


def _html_smp_stats(smp_data: Dict) -> str:
    """SMP 통계 카드(명세 5.10절, N11): 현재가 · 일평균 · 일최고 · 일최저(원/kWh)와 기준일.

    일 통계는 EPSIS 파일의 마지막 날 하루전 확정 가격이다. 최고·최저에 빨강·초록을 쓰지 않는다(상태가 아니다).
    거래시간이 없으면 현재가는 최근 48시간 평균이라 그 줄에만 추정 칩을 단다. 시뮬레이션이면 '확정'을 쓰지 않는다.
    """
    is_epsis = str(smp_data.get("data_source", "")).startswith("EPSIS")
    label = str(smp_data.get("trade_label") or "").strip()

    def num(v: object) -> str:
        return f"{float(v):,.1f}" if _finite_number(v) else "—"

    if is_epsis and label:
        current_name, current_chip = f"현재가 (거래시간 {html_escape(label.split(' ')[-1])})", ""
    elif is_epsis:
        current_name, current_chip = "현재가 (최근 48시간 평균)", _html_chip("est", "추정 · 최근 48시간 평균")
    else:
        current_name, current_chip = "현재가 (시뮬레이션)", ""
    rows = ((current_name, smp_data.get("current"), current_chip), ("일평균", smp_data.get("daily_avg"), ""),
            ("일최고", smp_data.get("daily_max"), ""), ("일최저", smp_data.get("daily_min"), ""))
    items = "".join(
        f'<div class="pd-statlist__row"><dt>{name}{chip}</dt>'
        f'<dd><b class="pd-num">{num(v)}</b><small>원/kWh</small></dd></div>'
        for name, v, chip in rows)
    data_range = str(smp_data.get("data_range") or "")
    end = data_range.split("~")[-1].strip() if "~" in data_range else ""
    if is_epsis:
        chips = _html_chip("measured", "확정 가격") + str(smp_data.get("pd_stale_chip", ""))
        meta = f"기준일 {html_escape(end) or '—'} · EPSIS 하루전 (확정)"
        foot = "일 통계는 EPSIS 파일의 마지막 날 기준이라 예측 차트 날짜와 다를 수 있습니다"
    else:
        chips = _html_chip("est", "시뮬레이션 · 실측 아님")
        meta = "기준일 — · SMP 파일을 읽지 못한 시간대 기본값"
        foot = "SMP 파일이 없어 앱이 시간대 기본값으로 만든 값입니다 (확정 가격 아님)"
    return (_html_card_head("SMP 통계", "", chips, meta)
            + f'<dl class="pd-statlist" aria-label="SMP 통계">{items}</dl><p class="pd-card__note">{foot}</p>')


def _fmt_smp_blocks(smp_data: Dict, *, now: Optional[datetime] = None) -> List[Dict]:
    """시간대 블록 3행(명세 5.10절): 심야 00–08 · 주간 08–17 · 저녁 17–24의 중앙값 평균·최고·최저(소수 1자리).

    칸은 시작 시각의 시(hour)로 나눈다. 예측일 축이면 인덱스 0–7 · 8–16 · 17–23과 같고, 지금부터 그린 축(대체값)도
    같은 시간대끼리 묶인다.
    """
    axis = _fmt_smp_slots(smp_data, (now or datetime.now()).replace(second=0, microsecond=0))
    out = []
    for name, hours, span in (("심야", "00–08시", range(0, 8)), ("주간", "08–17시", range(8, 17)),
                              ("저녁", "17–24시", range(17, 24))):
        vals = [s["q50"] for s in axis["slots"] if s["start"].hour in span]
        out.append({"name": name, "hours": hours, "count": len(vals),
                    "avg": round(sum(vals) / len(vals), 1) if vals else None,
                    "max": round(max(vals), 1) if vals else None, "min": round(min(vals), 1) if vals else None})
    return out


def _html_smp_blocks(smp_data: Dict, now: datetime) -> str:
    """시간대 블록 카드(명세 5.10절, B 이식). 칩은 차트 머리와 같은 예측/추정 칩이다."""
    keys = _fmt_smp_with_keys(smp_data, now)
    axis = _fmt_smp_slots(smp_data, now)

    def num(v: Optional[float]) -> str:
        return "—" if v is None else f"{v:,.1f}"

    body = "".join(
        f'<tr><td><b>{b["name"]}</b><small>{b["hours"]}</small></td><td class="pd-num">{num(b["avg"])}</td>'
        f'<td class="pd-num">{num(b["max"])}</td><td class="pd-num">{num(b["min"])}</td></tr>'
        for b in _fmt_smp_blocks(smp_data, now=now))
    scope = f"예측일 {axis['forecast_date']}" if axis["date_axis"] else "지금부터 24시간"
    basis = "예측 중앙값" if keys["pd_fc_kind"] == "model" else "추정 패턴"
    return (_html_card_head("시간대별 요약", "", _html_smp_kind_chip(keys), f"{scope} · 화면 {now:%H:%M} KST")
            + '<table class="pd-mini-table" aria-label="시간대별 SMP 요약"><thead><tr><th scope="col">시간대</th>'
            + '<th scope="col">평균</th><th scope="col">최고</th><th scope="col">최저</th></tr></thead>'
            + f'<tbody>{body}</tbody></table><p class="pd-card__note">단위 원/kWh · {basis} 기준</p>')


def _html_smp_model(smp_data: Dict) -> str:
    """예측 모델 카드(명세 5.10절). API 모델 이름은 그대로(검증 MAPE 포함) 보인다.

    이름이 기준선일 때만 README walk-forward 성적을 붙인다. 고정 문구 'naive · MAPE 11.75%'를 모델 이름처럼 붙이지
    않는다(다른 모델이 운영되면 틀린 말이 된다). 대체값·로컬 대체면 '예측 모델 없음'이라고 쓴다.
    """
    keys = _fmt_smp_with_keys(smp_data, datetime.now())
    model = smp_data.get("forecast_model")
    raw = model.strip() if isinstance(model, str) else ""
    name = html_escape(raw) if raw else "이름 미제공"
    note = ""
    if keys["pd_fc_kind"] == "model":
        # 카드 제목이 '예측 모델'이라 줄에는 API 이름만 그대로 쓴다('예측 모델 기준선 예측 …'처럼 겹치지 않게)
        # 칩은 경로(/smp-forecast) 대신 쉬운 말로 쓴다(1차 평가 멘토). 경로는 출처와 기준 시각 표가 그대로 보인다
        chip, line = _html_chip("plain", "API 서버 예측"), name
        if any(token in raw for token in ("기준선", "전날 같은 시간", "naive", "어제 같은 시각", "baseline")):
            note = "walk-forward 2025-07-22~12-18 · MAPE 11.75% · R² 0.203 · 기준선 자체(학습한 모델 아님)"
        if "MAPE" in raw:
            # API 이름 끝의 '검증 MAPE'는 API 서버가 실행 중에 계산한 값이고 평가 기간을 주지 않는다. 바로 아래 walk-forward
            # 값과 숫자가 달라 어느 쪽이 기준인지 헷갈렸다(1차 평가) → 두 숫자의 출처·기간을 밝힌다(값은 바꾸지 않는다)
            basis = ("이름의 '검증 MAPE'는 API 서버 계산값(평가 기간 미제공)"
                     + (" · walk-forward 값은 2026-11-06 벤치마크라 숫자가 다릅니다" if note else ""))
            note = f"{note}<br>{basis}" if note else basis
    elif keys["pd_fc_kind"] == "api_fallback":
        chip, line = _html_chip("est", "API 대체값"), f"예측 모델 없음 · API 대체값 ({name})"
    else:
        chip, line = _html_chip("est", "앱 로컬 대체"), "예측 모델 없음 · 시간대 평균 패턴(추정)"
    return (_html_card_head("예측 모델", "", chip, "")
            + f'<p class="pd-smp-model__name">{line}</p>'
            + (f'<p class="pd-card__note">{note}</p>' if note else ""))


def _html_mix_table(power_status: Dict) -> str:
    """전력 공급 구성 범례 겸 표(명세 5.11절): 견본 · 원 · 추정 MW(≈, 정수) · 비중(%).

    견본은 도넛 조각과 같은 색·같은 빗금 방향이라 색을 못 구분해도 행과 조각을 맞출 수 있다. 비중은 도넛과 같은
    기준(절댓값 합)이다. 충전 중인 ESS(음수)는 도넛이 크기만 보이므로 표에 '(충전)'을 밝힌다.
    """
    supply = power_status.get("supply") if isinstance(power_status.get("supply"), dict) else {}
    hour = datetime.now().hour
    rows = []
    for key, name, hatch in (("solar", "태양광", "slash"), ("wind", "풍력", "backslash"),
                             ("thermal", "화력", "cross"), ("ess", "ESS", "dot")):
        raw = supply.get(key)
        rows.append((key, name, hatch, abs(float(raw)) if _finite_number(raw) else None,
                     _finite_number(raw) and raw < 0))
    total = sum(r[3] for r in rows if r[3] is not None)
    # 큰 값부터(값 없음은 끝). 도넛은 조각 순서가 고정이라 행 순서 대신 견본 빗금으로 맞춘다
    rows.sort(key=lambda r: -1.0 if r[3] is None else r[3], reverse=True)
    cells = []
    for key, name, hatch, val, charging in rows:
        # 숫자와 단위 사이는 줄바꿈 금지(&nbsp;), 좁은 칸에서는 덧말 '(야간)'·'(충전)'만 다음 줄로 내린다
        if val is None:
            mw = "—"
        elif val == 0:
            mw = "0&nbsp;MW" + (" <small>(야간)</small>" if key == "solar" and not 6 <= hour <= 18 else "")
        else:
            mw = f"≈{val:,.0f}&nbsp;MW" + (" <small>(충전)</small>" if charging else "")
        share = f"{val / total * 100:.0f}%" if val is not None and total > 0 else "—"
        cells.append(f'<tr><td><span class="pd-mix-name"><span class="pd-swatch pd-swatch--hatch-{hatch}" '
                     f'style="--c:{SERIES[key]}" aria-hidden="true"></span>{name}</span></td>'
                     f'<td class="pd-num">{mw}</td><td class="pd-num">{share}</td></tr>')
    # 명세 8.4: 추정 합계가 수요와 다른 것은 모형의 한계라고 이 탭에서 밝힌다. 합계는 도넛 가운데와 같은 값·같은 반올림이다
    demand = power_status.get("demand")
    limit_note = ""
    if total > 0 and _finite_number(demand) and abs(total - float(demand)) >= 10:
        limit_note = (f'<p class="pd-card__note pd-mix-limit">추정 합계 ≈{round(total, -1):,.0f} MW는 지금 수요 '
                      f'{float(demand):,.0f} MW와 맞지 않습니다 — 발전량을 수요에 맞추지 않고 HVDC 연계·ESS 충방전도 넣지 않은 '
                      '시뮬레이션 모형의 한계입니다 (실측 아님)</p>')
    return ('<table class="pd-mini-table pd-mix-table" aria-label="전력 공급 구성 (시뮬레이션 추정)"><thead><tr>'
            '<th scope="col">발전원</th><th scope="col">추정 출력</th><th scope="col">비중</th></tr></thead>'
            f'<tbody>{"".join(cells)}</tbody></table>{limit_note}')


def _fmt_timeline_sub(plants_df: pd.DataFrame, registry: Optional[Dict]) -> str:
    """타임라인 카드 부제(명세 5.11절): 그림이 쓰는 설비(운영중 풍력·태양광)와 빠진 분산형 태양광.

    그림은 풍력·태양광 설비만 쓰므로 운영중 전체 곳 수 대신 두 유형의 곳 수를 쓴다(화력·ESS는 그림에 없다).
    """
    summary = _fmt_registry_summary(plants_df, registry, "전체")
    by_type = summary["total"]["by_type"]
    if summary["fallback"]:
        return f"예비 목록 {summary['total']['count']}곳 기준 (등록부를 읽지 못함)"
    parts = [f"운영중 풍력 {by_type['wind']['count']}곳 · 태양광 {by_type['solar']['count']}곳 설비 기준"]
    if summary["distributed_mw"] is not None:
        parts.append(f"분산형 태양광 {summary['distributed_mw']:,.1f} MW 제외")
    return " · ".join(parts)


def _html_timeline_legend() -> str:
    """시간대별 발전량 범례(명세 6.1절: Plotly 범례 대신 HTML). 막대와 같은 색·같은 빗금 방향 견본과 현재 시각 상자.

    그림 안 범례를 지운 뒤 이 카드만 계열 이름이 툴팁에만 있어 막대 색을 읽을 수 없었다(1차 평가 멘토).
    """
    def item(kind: str, hatch: str, name: str) -> str:
        return (f'<span class="pd-maplegend__item" role="listitem"><span class="pd-swatch pd-swatch--bar '
                f'pd-swatch--bar-{hatch}" style="--c:{SERIES[kind]}" aria-hidden="true"></span><b>{name}</b></span>')
    return ('<div class="pd-maplegend" role="list" aria-label="시간대별 발전량 범례">'
            + item("wind", "backslash", "풍력") + item("solar", "slash", "태양광")
            + '<span class="pd-maplegend__item" role="listitem"><span class="pd-swatch pd-swatch--now" aria-hidden="true">'
              '</span>점선 상자 = 지금 시각</span>'
            + '<span class="pd-maplegend__item" role="listitem">빗금 막대 = 시뮬레이션 추정 · 실측 아님</span></div>')


def _chart_table_timeline(fig: go.Figure) -> pd.DataFrame:
    """타임라인 '표로 보기'(명세 5.3절, J10): 그림에 실제로 그린 막대 값 24행(차트와 표가 어긋나지 않게).

    지금 시(hour)는 그림의 현재 시각 상자 위치에서 읽는다. 모든 값이 시뮬레이션이라 열 이름에 '추정'을 붙인다.
    """
    columns = ["시각", "풍력 (MW · 추정)", "태양광 (MW · 추정)", "합계 (MW · 추정)", "구분"]
    traces = {t.name: t for t in fig.data}
    wind, solar = traces.get("풍력"), traces.get("태양광")
    if wind is None or solar is None or wind.x is None or not len(wind.x):
        return pd.DataFrame(columns=columns)
    current = None
    for shape in fig.layout.shapes or ():
        if shape.type == "rect" and _finite_number(shape.x0):
            current = int(round(float(shape.x0) + 0.42))
    rows = []
    for h, w, s in zip(wind.x, wind.y, solar.y, strict=True):  # 두 막대는 같은 24칸으로 만든다(어긋나면 바로 실패)
        hour = int(h)
        rows.append([f"{hour:02d}:00–{hour + 1:02d}:00", round(float(w), 1), round(float(s), 1),
                     round(float(w) + float(s), 1), "지금" if hour == current else ""])
    return pd.DataFrame(rows, columns=columns)


def _fmt_plant_table(plants_df: pd.DataFrame, type_filter: str) -> pd.DataFrame:
    """발전소 등록부 표(명세 5.11절, N8): 발전소 · 유형 · 지역 · 설비 · 추정 출력 · 이용률(추정) · 상태.

    상태 필터는 두지 않는다: get_jeju_power_plants가 운영중만 담으므로 선택지가 '운영중' 하나뿐이다.
    유형 이름에 이모지를 쓰지 않고 화력을 넣는다(옛 매핑에는 화력이 없어 화력 행 유형이 빈 칸이었다).

    Raises:
        ValueError: type_filter가 '전체'·'풍력'·'태양광'·'ESS'·'화력'이 아닐 때(오타가 전체 표로 조용히 바뀌지 않게).
    """
    names = {"wind": "풍력", "solar": "태양광", "ess": "ESS", "thermal": "화력"}
    if type_filter != "전체" and type_filter not in names.values():
        raise ValueError(f"알 수 없는 발전 유형: {type_filter!r}")
    columns = ["발전소", "유형", "지역", "설비 (MW)", "추정 출력 (MW)", "이용률 (추정)", "상태"]
    df = plants_df if isinstance(plants_df, pd.DataFrame) else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=columns)

    def col(name: str, default: object) -> pd.Series:
        return df[name] if name in df.columns else pd.Series(default, index=df.index)

    out = pd.DataFrame({
        "발전소": col("name", "").fillna("").astype(str),
        "유형": col("type", "").map(names).fillna("기타"),
        "지역": col("address", None).map(_fmt_region),
        "설비 (MW)": pd.to_numeric(col("capacity", np.nan), errors="coerce").round(1),
        "추정 출력 (MW)": pd.to_numeric(col("generation", np.nan), errors="coerce").round(1),
        "이용률 (추정)": pd.to_numeric(col("utilization", np.nan), errors="coerce").round(1),
        "상태": col("status", "").fillna("").astype(str),
    }, columns=columns)
    if type_filter != "전체":
        out = out[out["유형"] == type_filter]
    return out.reset_index(drop=True)


def _fmt_plant_columns() -> Dict:
    """등록부 표 열 설정(명세 5.11절). 추정 열 머리 도움말에 '실측 아님'을 둔다(표 캔버스에는 칩을 그릴 수 없다).

    이용률 막대는 회색이다: 기본 primary 파랑 채움 막대는 이 앱 문법에서 '실측'처럼 읽혀 시뮬레이션 값과 맞지 않았다
    (1차 평가 멘토). 1.52.1 ProgressColumn의 color는 테마 기본 팔레트 이름(gray)을 받는다.
    """
    return {
        "설비 (MW)": st.column_config.NumberColumn(format="%.1f"),
        "추정 출력 (MW)": st.column_config.NumberColumn(format="%.1f", help="시뮬레이션 추정치 · 실측 아님"),
        "이용률 (추정)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%", color="gray",
                                                      help="추정 출력 ÷ 설비용량 · 실측 아님"),
    }


def _html_plant_summary(display_df: pd.DataFrame, registry: Optional[Dict], type_filter: str = "전체") -> str:
    """등록부 표 위 요약 줄(명세 5.11절): 표시 곳 수 · 설비 · 운영중만 표시(제외 항목) · 출력은 추정.

    제외 항목은 고른 유형의 등록부 항목만 센다(풍력을 고르면 건설중 풍력만). 추정 출력이 설비용량을 넘는 행은
    표 캔버스에 경고를 그릴 수 없으므로 표 밖 문장으로 밝힌다(HVDC 수입을 넣지 않는 시뮬레이션 모형의 한계).
    비교는 표에 보인 소수 1자리 값끼리 한다(보인 두 수가 같은데 '넘는다'고 쓰지 않게).
    """
    reg = registry if isinstance(registry, dict) else {}
    if not reg:
        lines = [f"등록부를 읽지 못해 예비 목록 {len(display_df)}곳만 표시합니다 · 출력은 시뮬레이션 추정"]
    else:
        keys = {"풍력": "wind_farms", "태양광": "solar_farms", "ESS": "ess_facilities", "화력": "thermal_plants"}
        picked = list(keys.values()) if type_filter == "전체" else [keys[type_filter]]
        excluded = _fmt_registry_summary(pd.DataFrame(), {k: reg.get(k) for k in picked})["excluded_note"]
        capacity = pd.to_numeric(display_df.get("설비 (MW)"), errors="coerce").sum() if len(display_df) else 0.0
        lines = [f"표시 {len(display_df)}곳 · 설비 {capacity:,.1f} MW · 운영중만 표시"
                 + (f" ({excluded} 제외)" if excluded else "") + " · 출력은 시뮬레이션 추정"]
    if len(display_df):
        over = display_df[pd.to_numeric(display_df["추정 출력 (MW)"], errors="coerce")
                          > pd.to_numeric(display_df["설비 (MW)"], errors="coerce")]
        if len(over):
            names = [html_escape(str(n)) for n in over["발전소"]]
            listed = ", ".join(names[:3]) + (f" 외 {len(names) - 3}곳" if len(names) > 3 else "")
            lines.append(f'<span class="pd-ico pd-ico--info" aria-hidden="true"></span>'
                         f'<span>추정 출력이 설비용량을 넘는 발전소 {len(over)}곳: {listed} — 모형 한계, 실측 아님</span>')
    first = f'<p>{lines[0]}</p>'
    rest = "".join(f'<p class="pd-plant-summary__warn">{line}</p>' for line in lines[1:])
    return f'<div class="pd-plant-summary">{first}{rest}</div>'


def _html_section_head(title: str, subtitle: str = "") -> str:
    """탭 안 섹션 머리(명세 4.5절). 제목은 h2가 아니라 role="heading"인 p다(Streamlit 제목 앵커 링크를 피한다)."""
    sub = f'<p class="pd-section-head__sub">{subtitle}</p>' if subtitle else ""
    return (f'<div class="pd-section-head"><p class="pd-section-head__title" role="heading" aria-level="2">{title}</p>'
            f'{sub}</div>')


def _html_example_ribbon() -> str:
    """예시 expander 맨 위 리본(명세 4.5절): 아래 카드 숫자가 설명용 고정 값임을 먼저 밝힌다."""
    return ('<div class="pd-example-ribbon" role="note"><span class="pd-glyph pd-glyph--example" aria-hidden="true">'
            '</span><b>예시 화면 · 실제 값 아님</b><span>기능을 설명하려고 넣은 고정 문구와 숫자입니다</span></div>')


# ============================================================================
# 표시 계층: 사이드바 · 출처와 기준 시각 (명세 4.6·5.12·5.13절, 7단계)
# 경보 이력은 화면에 보일 것만 고른다(공유 파일 data/alerts/alert_history.json을 다시 쓰지 않는다).
# 출처 표는 KPI·카드가 쓰는 판정 함수를 다시 불러 같은 상태를 한 표에 모은다(판정 규칙을 두 군데에 두지 않는다).
# ============================================================================

def _html_sidebar_section(title: str) -> str:
    """사이드바 섹션 머리(명세 5.12절, 13/700 text-2). st.markdown 제목이 아니라 role="heading"인 p다(제목 앵커를 피한다)."""
    return f'<p class="pd-side-head" role="heading" aria-level="2">{title}</p>'


def _html_theme_hint() -> str:
    """화면 테마를 바꾸는 법(명세 4.6·5.12절). 1.52.1은 고른 테마를 저장하지 않아 새로 고치면 시스템 설정으로 돌아간다."""
    return ('<p class="pd-side-hint">화면 테마: 오른쪽 위 ⋮ → Settings → Choose app theme (Light · Dark · Use system setting). '
            '새로 고치면 시스템 설정으로 돌아갑니다 (현재 Streamlit 버전 제약)</p>')


def _fmt_alert_history(entries: object, now: datetime) -> Dict:
    """사이드바 경보 이력의 표시값(명세 5.12절, D10). 공유 파일은 다시 쓰지 않고 화면에 보일 것만 고른다.

    1) 시드 제외를 먼저 한다: 저장소에 커밋된 예시 기록(f85c4f45, 2025-12-19 정각 15분 간격 5건)은 앱이 쓴 기록이
       아니다. 앱은 status critical·danger·warning과 제목 "전력 수급 … 경보/단계"만 쓴다(_fmt_reserve_level).
       그래서 status가 'caution'이거나 제목이 "관심 (CAUTION)" 꼴인 항목은 건수·목록·마지막 기록에서 모두 뺀다.
    2) 남은 앱 기록의 단계는 제목의 위험·주의·관심으로, 제목으로 못 정하면 status로 정한다(앱 어휘에서 warning = 관심).
    3) 최근 7일만 세고 최신순 최대 5행을 보인다. 7일 안에 없을 때만 그보다 오래된 앱 기록의 마지막 시각을 붙인다
       (시드 시각이 실제 경보처럼 보이지 않게 시드는 여기에도 넣지 않는다).

    Args:
        entries: AlertHistory.get_recent() 결과(dict 목록). 목록이 아니면 빈 이력으로 본다.
        now: 7일 창의 기준 시각(앱 기록과 같은 naive 로컬 시각).

    Returns:
        counts(위험·주의·관심 7일 건수), rows(최대 5행: time·level·cls·reserve), recent_total(7일 건수 합),
        last_old(7일 안 기록이 없을 때만 그보다 오래된 앱 기록의 마지막 시각, 없으면 None),
        seed_count(뺀 시드 수), invalid_count(시각·단계를 읽지 못해 뺀 기록 수. 조용히 버리지 않고 화면에 밝힌다).
    """
    seed_title = re.compile(r"^(관심|주의|위험) \((CAUTION|WARNING|CRITICAL)\)$")
    status_level = {"critical": "위험", "danger": "주의", "warning": "관심"}
    level_class = {"위험": "critical", "주의": "danger", "관심": "warning"}
    records: List[Dict] = []
    seed_count = invalid_count = 0
    for item in entries if isinstance(entries, list) else []:
        if not isinstance(item, dict):
            invalid_count += 1
            continue
        status = str(item.get("status") or "").strip()
        title = str(item.get("title") or "").strip()
        if status == "caution" or seed_title.match(title):
            seed_count += 1
            continue
        level = next((name for name in ("위험", "주의", "관심") if name in title), None) or status_level.get(status)
        try:
            stamp = datetime.fromisoformat(str(item.get("timestamp")).strip())
        except ValueError:
            stamp = None
        if level is None or stamp is None:
            invalid_count += 1
            continue
        if stamp.tzinfo is not None:  # 앱은 naive 로컬 시각을 쓴다. 다른 도구가 넣은 시간대 붙은 값은 로컬로 맞춰 비교한다
            stamp = stamp.astimezone().replace(tzinfo=None)
        rate = item.get("reserve_rate")
        records.append({"stamp": stamp, "level": level,
                        "reserve": f"{float(rate):.1f}%" if _finite_number(rate) else "—"})
    records.sort(key=lambda r: r["stamp"], reverse=True)
    window_start = now - timedelta(days=7)
    recent = [r for r in records if r["stamp"] >= window_start]
    older = [r for r in records if r["stamp"] < window_start]
    return {
        "counts": {name: sum(1 for r in recent if r["level"] == name) for name in ("위험", "주의", "관심")},
        "rows": [{"time": f"{r['stamp']:%Y-%m-%d %H:%M}", "level": r["level"], "cls": level_class[r["level"]],
                  "reserve": r["reserve"]} for r in recent[:5]],
        "recent_total": len(recent),
        "last_old": f"{older[0]['stamp']:%Y-%m-%d %H:%M}" if (not recent and older) else None,
        "seed_count": seed_count,
        "invalid_count": invalid_count,
    }


def _html_alert_history(hist: Dict) -> str:
    """사이드바 경보 이력(명세 4.6·5.12절): 위험·주의·관심 7일 건수 + 최근 기록 최대 5행.

    st.metric·이모지를 쓰지 않는다(D12). 단계 글자가 함께 있어 색만으로 단계를 전하지 않는다.
    시드·읽지 못한 기록을 뺐으면 그 사실을 작게 밝힌다(파일과 화면 건수가 왜 다른지 알 수 있게).
    """
    counts = "".join(
        f'<span class="pd-side-count pd-side-count--{cls}"><span class="pd-side-count__dot" aria-hidden="true"></span>'
        f'{name} <b class="pd-num">{hist["counts"][name]}</b></span>'
        for name, cls in (("위험", "critical"), ("주의", "danger"), ("관심", "warning")))
    if hist["rows"]:
        body = '<ol class="pd-side-alerts__list">' + "".join(
            f'<li><span class="pd-num">{r["time"]}</span> · <b class="pd-side-level pd-side-level--{r["cls"]}">'
            f'{r["level"]}</b> · 예비율 <span class="pd-num">{r["reserve"]}</span></li>' for r in hist["rows"]) + "</ol>"
    else:
        tail = f' · 마지막 기록 {hist["last_old"]}' if hist["last_old"] else ""
        body = f'<p class="pd-side-alerts__empty">최근 7일 경보 없음{tail}</p>'
    notes = []
    if hist["seed_count"]:
        notes.append(f'예시 기록 {hist["seed_count"]}건 제외 (앱 경보 아님)')
    if hist["invalid_count"]:
        notes.append(f'읽지 못한 기록 {hist["invalid_count"]}건 제외')
    note = f'<p class="pd-side-hint">{" · ".join(notes)}</p>' if notes else ""
    return ('<div class="pd-side-alerts"><div class="pd-side-alerts__counts" role="group" aria-label="최근 7일 경보 건수">'
            f'{counts}</div>{body}{note}</div>')


@st.dialog("경보 이력 초기화")
def _render_clear_history_dialog(alert_history: AlertHistory) -> None:
    """경보 이력 초기화 확인 대화상자(명세 5.2·5.12절, N6). 관리자 플래그로 띄운 서버에서만 열 수 있다.

    대화상자는 포털로 그려져 페이지 color-scheme을 물려받지 않으므로 pd- HTML 없이 Streamlit 요소만 쓴다(5.1-4).
    안내문은 st.write로 쓴다: main 밖 함수 안 .markdown( 호출은 회귀 테스트의 너비 우선 탐색을 가로챈다(5.1-7).
    확인 버튼은 secondary다: 1.52.1 primary 버튼은 다크 #60A5FA 위 흰 글자 2.54:1로 AA 미달이다(8.4절).
    """
    st.write("공유 파일 data/alerts/alert_history.json의 경보 이력을 모두 지웁니다. 되돌릴 수 없습니다.")
    confirm_col, cancel_col = st.columns(2)
    if confirm_col.button("모두 지우기", type="secondary", icon=":material/delete:", key="clear_history_confirm",
                          width="stretch"):
        alert_history.clear()
        st.rerun()
    if cancel_col.button("취소", key="clear_history_cancel", width="stretch"):
        st.rerun()


def _fmt_data_status(power_status: Dict, smp_data: Dict, backend_predictions: Optional[Dict], weather: Dict,
                     api_kpis: Optional[Dict], kpx_local: Optional[Dict], registry: Optional[Dict], now: datetime,
                     *, plants_df: Optional[pd.DataFrame] = None) -> List[Dict]:
    """출처와 기준 시각 표의 8행(명세 5.13절). 행마다 name·source·basis·chips(HTML)를 돌려준다.

    판정은 KPI·카드와 같은 함수(_fmt_provenance·_fmt_smp_display_keys·_fmt_weather_provenance·_fmt_registry_summary)를
    다시 불러 정한다. 같은 입력이면 KPI 칩과 이 표의 칩이 어긋나지 않는다. 행 이름은 카드 제목과 겹치지 않게
    쓴다(재생 비중·기상·발전소 출력·주파수, 명세 5.3절 골격 규칙). API에서 온 문자열(모델 이름·날짜)은 이스케이프한다.

    Args:
        power_status·smp_data·backend_predictions·weather·api_kpis·kpx_local·registry: 같은 이름의 캐시 함수 결과
            (power_status는 테스트 모드면 사본).
        now: 화면 기준 시각.
        plants_df: get_jeju_power_plants() 결과. 등록부 곳 수·출력 원천을 센다(없으면 그 부분을 비운다).
    """
    rows: List[Dict] = []
    prov = _fmt_provenance(power_status, api_kpis, kpx_local, now)
    data_date = str(power_status.get("data_date", "")).strip()
    try:
        queried: Optional[datetime] = datetime.strptime(data_date, "%Y-%m-%d %H:%M")
    except ValueError:
        queried = None
    _, elapsed = _format_observation_age(data_date or "N/A", now)
    elapsed = elapsed if elapsed != "N/A" else "경과 시간 확인 불가"
    screen = f"화면 {now:%H:%M}"
    if prov["time_kind"] == "query":
        demand_basis = f"조회 {prov['query_hhmm']} · {elapsed}"
    elif prov["time_kind"] == "date":
        demand_basis = f"{html_escape(data_date[:10])} · {elapsed}"
    elif prov["time_kind"] == "test":
        demand_basis = f"테스트 값 · {screen}"
    else:
        demand_basis = f"시각 없음 · {screen}"
    reserve_tail = {"default": " · 예비율 기본값", "national": " · 예비율 전국 값 (제주 아님)"}.get(prov["reserve_state"], "")
    rows.append({"name": "수요·예비율", "source": prov["src_label"] + reserve_tail, "basis": demand_basis,
                 "chips": prov["display_keys"]["pd_chip"]})

    # 공급능력은 따로 판정한다(7.2절 끝). 테스트 모드는 예비율만 바꾸므로 공급능력 시각은 원래 조회 시각을 쓴다
    if queried is not None:
        supply_basis = f"조회 {queried:%H:%M} · {elapsed}"
    elif prov["time_kind"] == "date":
        supply_basis = demand_basis
    else:
        supply_basis = f"시각 없음 · {screen}"
    if prov["supply_measured"]:
        supply = ("KPX 제주 실시간 (앱 직접 조회)", _html_chip("measured", "실측"))
    else:
        supply = ("추정 · KPX 공급능력 미수신", _html_chip("est", "추정 · 실측 아님"))
    rows.append({"name": "공급능력", "source": supply[0], "basis": supply_basis, "chips": supply[1]})

    # SMP는 live_header와 같이 API 예측을 합친 dict로 판정해야 KPI 칩과 같은 상태가 나온다
    smp = {**smp_data, **backend_predictions} if isinstance(backend_predictions, dict) else dict(smp_data)
    keys = _fmt_smp_display_keys(smp, now)
    is_epsis = str(smp.get("data_source", "")).startswith("EPSIS")
    trade_label = smp.get("trade_label")
    if is_epsis and trade_label:
        smp_chip = _html_chip("measured", "확정 가격") + keys["pd_stale_chip"]
    elif is_epsis:
        smp_chip = _html_chip("est", "추정 · 최근 48시간 평균")
    else:
        smp_chip = _html_chip("est", "시뮬레이션 · 실측 아님")
    rows.append({"name": "제주 하루전 SMP", "source": keys["pd_src_label"],
                 "basis": f"거래시간 {html_escape(str(trade_label))}" if trade_label else "거래시간 확인 불가",
                 "chips": smp_chip})

    model = smp.get("forecast_model")
    model_name = html_escape(model.strip()) if isinstance(model, str) and model.strip() else "모델 이름 미제공"
    if keys["pd_fc_kind"] == "model":
        forecast = (f"API 서버 /smp-forecast · {model_name}", _html_chip("forecast", "예측"))
    elif keys["pd_fc_kind"] == "api_fallback":
        forecast = (f"API 대체값 (모델 예측 아님) · {model_name}", _html_chip("est", "추정 · 모델 예측 아님"))
    else:
        forecast = ("앱 로컬 대체 (시간대 평균 패턴)", _html_chip("est", "추정 · 시간대 평균 패턴"))
    forecast_date = str(smp.get("forecast_date") or "").strip()
    forecast_chips = forecast[1]
    if keys["pd_fc_kind"] == "local":
        forecast_basis = "지금부터 24시간 (예측일 없음)"
    elif forecast_date:
        forecast_basis = f"예측일 {html_escape(forecast_date)}"
        try:
            if datetime.strptime(forecast_date[:10], "%Y-%m-%d").date() < now.date():
                forecast_chips += _html_chip("stale", "예측일 지연")
        except ValueError:
            pass  # 날짜 꼴이 아니면 지연 여부를 판정하지 않는다(문자열은 이미 그대로 보인다)
    else:
        forecast_basis = "예측일 미제공"
    rows.append({"name": "SMP 예측", "source": forecast[0], "basis": forecast_basis, "chips": forecast_chips})

    # 기상: KPI 카드와 같은 규칙(값이 앱 시뮬레이션이면 API 출처 판정을 붙이지 않는다)
    weather_src = _fmt_weather_provenance(api_kpis) if weather.get("source") == "kma" else None
    if weather.get("source") != "kma":
        weather_row = ("시뮬레이션 (실측 아님)", _html_chip("est", "시뮬레이션 · 실측 아님"))
    elif not isinstance(api_kpis, dict):
        # 기상 캐시(5분)에 API 값이 남았는데 KPI 응답(60초 캐시)이 끊긴 사이: 실측인지 확인할 수 없으므로 실측 칩을 쓰지 않는다
        weather_row = ("API 서버 값 · 출처 확인 불가 (API 응답 없음, 최대 5분 전 값)", _html_chip("est", "추정 · 출처 확인 불가"))
    elif weather_src == "file":
        weather_row = ("기상청 과거 자료 24시간 평균 (API 서버)", _html_chip("est", "추정 · 과거 자료 평균"))
    elif weather_src == "default":
        weather_row = ("API 서버 기본값 (실측 아님)", _html_chip("est", "추정 · API 기본값 (실측 아님)"))
    else:
        weather_row = ("기상청 초단기실황 (API 서버)", _html_chip("measured", "실측"))
    # get_weather_data는 ttl=300이고 관측 시각을 돌려주지 않는다
    rows.append({"name": "기상", "source": weather_row[0], "basis": "앱 5분 캐시 (시각 미제공)", "chips": weather_row[1]})

    if "renewable_source" in power_status:
        renew = ("섬 전체 태양광+풍력 ML 추정 (API 서버)", f"조회 {prov['query_hhmm']}")
    else:
        renew = ("앱 추정 (시간대 패턴 · 실측 아님)", screen)
    rows.append({"name": "재생 비중", "source": renew[0], "basis": renew[1], "chips": _html_chip("est", "추정 · 실측 아님")})

    df = plants_df if isinstance(plants_df, pd.DataFrame) else None
    if df is not None:
        summary = _fmt_registry_summary(df, registry, "전체")
        if summary["fallback"]:
            plants_row = ("앱 내장 예비 목록 (등록부를 읽지 못함)", _html_chip("est", "예비 목록 · 등록부 아님"))
        else:
            excluded = ", ".join(f"{status} {e['count']}곳" for status, e in summary["excluded"].items())
            plants_row = (f"발전소 등록부 (운영중 {summary['total']['count']}곳"
                          + (f", {excluded} 제외)" if excluded else ")"), _html_chip("measured", "확정"))
    else:
        plants_row = ("발전소 등록부", _html_chip("measured", "확정"))
    reference = _fmt_registry_ref(registry)
    rows.append({"name": "발전소 위치·설비", "source": plants_row[0],
                 "basis": reference[len("등록부 "):] if reference.startswith("등록부 ") else reference,
                 "chips": plants_row[1]})

    # 발전소 출력은 API 서버 시뮬레이션 값(ID로 맞춘 행)이고 화력 등 나머지는 앱 추정이다. 주파수도 API 시뮬레이션이다
    origins = (df["data_source"] if df is not None and "data_source" in df.columns
               else pd.Series(dtype=object))
    n_api = int((origins == "API 서버 시뮬레이션").sum())
    if n_api and n_api == len(origins):
        output_src = "API 서버 시뮬레이션"
    elif n_api:
        output_src = f"API 서버 시뮬레이션 · {len(origins) - n_api}곳은 앱 추정"
    else:
        output_src = "앱 시뮬레이션 (API 서버 출력 미수신)"
    output_basis = (f"조회 {prov['query_hhmm']} · 60초 캐시" if isinstance(api_kpis, dict)
                    else f"{screen} · 60초 캐시")
    rows.append({"name": "발전소 출력·주파수", "source": output_src, "basis": output_basis,
                 "chips": _html_chip("est", "시뮬레이션 · 실측 아님")})
    return rows


def _html_data_status(power_status: Dict, smp_data: Dict, backend_predictions: Optional[Dict], weather: Dict,
                      api_kpis: Optional[Dict], kpx_local: Optional[Dict], registry: Optional[Dict], now: datetime,
                      *, plants_df: Optional[pd.DataFrame] = None, refresh_note: str = "60초마다 갱신") -> str:
    """출처와 기준 시각 카드(명세 5.13절): 항목 · 출처 · 기준 (KST) · 상태 표 + 각주.

    640px 이하에서는 표를 2줄 목록(1줄 항목 + 칩, 2줄 출처 · 기준)으로 바꾼다. 둘째 줄은 같은 글자를 한 칸에 둔
    좁은 폭 전용 span이 맡는다(표 칸 둘은 한 줄로 합칠 수 없어서). 넓은 폭에서는 그 span을 숨긴다.
    """
    rows = _fmt_data_status(power_status, smp_data, backend_predictions, weather, api_kpis, kpx_local, registry, now,
                            plants_df=plants_df)

    def parts(text: str) -> str:
        # 기준 칸은 " · " 사이에서만 줄을 바꾼다: 좁은 표(641~1180px)에서 "2026-11-|24" 같은 날짜·시각 토막이 끊겼다(768 실측)
        return " · ".join(f'<span class="pd-nowrap">{part}</span>' for part in text.split(" · "))

    body = "".join(
        f'<tr><th scope="row">{r["name"]}</th>'
        f'<td class="pd-sources__src">{r["source"]}<span class="pd-sources__m"> · {parts(r["basis"])}</span></td>'
        f'<td class="pd-sources__basis">{parts(r["basis"])}</td><td class="pd-sources__chips">{r["chips"]}</td></tr>'
        for r in rows)
    head = _html_card_head("출처와 기준 시각", "항목마다 값이 어디서 왔고 언제 기준인지", "",
                           f"화면 {now:%H:%M} KST ({refresh_note})")
    return ('<section class="pd-sources" aria-label="출처와 기준 시각">' + head
            + '<table class="pd-sources__table"><thead><tr><th scope="col">항목</th><th scope="col">출처</th>'
            + '<th scope="col">기준 (KST)</th><th scope="col">상태</th></tr></thead>'
            + f'<tbody>{body}</tbody></table>'
            + '<p class="pd-card__foot">실제로 잰 시간별 이력은 아직 연결되지 않아 수급 곡선은 일간 패턴으로 그린 추정입니다 · '
            + '화면 테마: 오른쪽 위 ⋮ → Settings → Choose app theme (새로 고치면 시스템 설정으로 돌아갑니다)</p></section>')


# ============================================================================
# 메인 대시보드
# ============================================================================

def main():
    """메인 함수"""
    st.html(_css_dashboard())  # <style>만 담아 레이아웃 공간 0 (st.html은 event 컨테이너로 간다)

    # 전체 실행 경로에서 쓰는 데이터만 읽는다. fragment는 각자 같은 캐시 함수를 다시 부른다(캐시 재사용).
    # 한 칸(st.empty) 안에서 읽는 이유: 캐시가 만료된 실행에서는 st.cache_data가 진행 표시 자리를 지금 위치에 끼워 넣어
    # 뒤 요소(머리·탭)의 위치가 한 칸씩 밀린다. 그러면 직전 실행과 st.tabs 위치가 달라져 탭이 다시 마운트되고 선택 탭이
    # 첫 탭으로 돌아갔다(6단계 실측: 발전 유형 필터를 누르면 '실시간 현황'으로 튐). fragment 안의 같은 처리는 S4 참고
    with st.empty():
        plants_df = get_jeju_power_plants()
        power_status = get_current_power_status()
        registry = load_power_plants_data()                   # 캐시 재사용: 지도 밖 자원·등록부 기준일

    # 관리자 도구(알림 테스트·이력 초기화)는 실행 명령에 이 플래그를 준 서버에서만 보인다(N6). 공유 .env에는 넣지 않는다:
    # 워크트리 .env는 운영 .env의 심볼릭 링크라 거기에 넣으면 공개 운영 화면에 파괴적 컨트롤이 나타난다(명세 9.0절)
    admin_mode = os.environ.get("POWERDEMAND_DASHBOARD_ADMIN", "").strip() == "1"
    test_alert = False
    test_reserve = None

    # ========== 사이드바: 보기 · 경보 이력 · 관리자 도구 (명세 4.6·5.12절, 기본 접힘) ==========
    # 옛 '데이터 출처' 안내(st.info 2개)는 본문 '출처와 기준 시각' 표로 옮겼다(T1 기준점은 푸터 하나만 남긴다)
    with st.sidebar:
        st.html(_html_sidebar_section("보기"))
        # 60초 fragment(머리·KPI·수급·출처 표)와 300초 fragment(SMP·발전 탭)를 다시 그린다. 끄면 위젯을 만질 때만 갱신된다
        auto_refresh = st.toggle("60초마다 자동 갱신", value=True, key="auto_refresh")
        st.html(_html_theme_hint())
        st.html(_html_sidebar_section("경보 이력 · 최근 7일"))
        alert_history = get_alert_history()
        # 이력은 아래 경보 판정·기록 블록 뒤에 채운다: 여기서 바로 그리면 이번 실행이 남긴 경보가 빠져 배너가 떠 있는 동안
        # '최근 7일 경보 없음'이라고 말했다(1차 평가 G10: 190초 동안 어긋남). 자리만 지금 잡아 사이드바 순서는 그대로 둔다
        alert_history_slot = st.empty()
        if admin_mode:
            st.html(_html_sidebar_section("관리자 도구"))
            test_alert = st.toggle("알림 테스트 모드", value=False, key="admin_test_alert")
            if test_alert:
                test_reserve = st.slider("테스트 예비율 (%)", min_value=0.0, max_value=30.0, value=12.0, step=1.0,
                                         help="화면 표시만 바꿉니다. 테스트 모드에서는 경보를 기록하거나 보내지 않습니다")
                # 테스트 값은 사본에만 넣는다. data_source를 바꿔야 출처 판정이 '테스트'가 되어 실측 칩이 붙지 않는다
                power_status = dict(power_status)
                power_status['reserve_rate'] = test_reserve
                power_status['data_source'] = '테스트 모드'
            # 공유 이력 파일을 지우는 동작이라 확인 대화상자를 거친다(버튼 한 번으로 지우지 않는다)
            if st.button("경보 이력 초기화", key="clear_history", icon=":material/delete:"):
                _render_clear_history_dialog(alert_history)

    # 경보 판정·기록·발송: 전체 실행 경로에만 둔다 (자동 갱신 fragment 안 금지, J2)
    # fragment는 열린 세션마다 60초마다 돌기 때문에, 거기서 기록·발송하면 세션 수만큼 이력·알림이 겹친다
    reserve_rate = power_status['reserve_rate']
    level = _fmt_reserve_level(reserve_rate)
    if level['show_alert'] and not test_alert:
        alert_history = get_alert_history()
        alert_history.add_alert(reserve_rate=reserve_rate, status=level['status'],
                                title=level['alert_title'], message=level['alert_msg'])
        if level['status'] == "critical":
            email_notifier = get_email_notifier()
            if email_notifier.is_configured():
                success, email_msg = email_notifier.send_critical_alert(
                    reserve_rate=reserve_rate, status=level['status'], title=level['alert_title'],
                    message=level['alert_msg'], power_data=power_status)
                if success:
                    st.toast("이메일 발송 완료", icon=":material/mail:")
        slack_notifier = get_slack_notifier()
        if slack_notifier.is_configured():
            success, slack_msg = slack_notifier.send_alert(
                reserve_rate=reserve_rate, status=level['status'], title=level['alert_title'],
                message=level['alert_msg'], power_data=power_status)
            if success:
                st.toast("Slack 알림 발송 완료", icon=":material/chat:")
    # 전체 실행 경로에서만 채운다(fragment는 사이드바를 다시 그리지 않고 부작용도 두지 않는다, J2). 읽기만 한다
    alert_history_slot.html(_html_alert_history(_fmt_alert_history(alert_history.get_recent(100), datetime.now())))

    run_every = "60s" if auto_refresh else None
    run_every_slow = "300s" if auto_refresh else None  # 탭 2·3: 정시 전환을 5분 안에 따라잡는다(E8·G11 예산에 포함)

    # 앱 바·경보 배너·상태 문장·KPI를 60초마다 다시 그린다. main() 안에 중첩 정의해야 회귀 테스트(ast.walk(main))가
    # KPI 카드를 찾는다. 표시 키(pd_*)는 이 fragment의 사본 dict에만 넣는다(경보 발송에 넘긴 dict에는 넣지 않는다)
    @st.fragment(run_every=run_every)
    def live_header():
        now = datetime.now()
        # 캐시가 만료된 실행에서는 st.cache_data가 진행 표시 자리를 지금 위치에 끼워 넣어 뒤 요소가 그만큼 밀린다.
        # 60초 fragment가 둘 이상이면 이 fragment 실행은 '재실행으로 조기 종료'(script_finished 2)로 끝나고 1.52.1 화면은
        # 그때 지난 실행의 요소를 지우지 않으므로, 밀린 실행 다음의 적중 실행에서 앱 바·상태 문장·KPI가 두 번 보였다(S4 실측).
        # 데이터 읽기를 st.empty() 한 칸 안에서 해 진행 표시가 그 칸만 쓰게 하고, 화면 요소의 위치를 실행마다 같게 한다
        with st.empty():
            power_status = dict(get_current_power_status())
            if test_alert:
                power_status['reserve_rate'] = test_reserve
                power_status['data_source'] = '테스트 모드'
            smp_data = get_smp_data()
            backend_predictions = get_backend_smp_predictions()
            if backend_predictions:
                smp_data = {**smp_data, **backend_predictions}
            weather = dict(get_weather_data())
            api_kpis = fetch_backend_json("/api/v1/dashboard/kpis")
            prov = _fmt_provenance(power_status, api_kpis, fetch_jeju_realtime(), now)
        power_status.update(prov['display_keys'])      # pd_chip, pd_src_label, pd_state_class, pd_renew_time, pd_time
        # 기상 값이 앱 시뮬레이션이면(캐시 5분 동안 API 실패 시점의 값) API 출처 판정을 붙이지 않는다.
        # 붙이면 앱이 만든 값에 "API 기본값" 칩이 붙는다
        weather['pd_src'] = _fmt_weather_provenance(api_kpis) if weather.get('source') == 'kma' else None
        # 관측 시각을 주는 원천이 없다. API 값이면 앱 캐시 길이만 밝히고, 앱 시뮬레이션이면 화면 시각을 쓴다
        weather['pd_time'] = ('관측 시각 미제공 · 최대 5분 캐시' if weather.get('source') == 'kma'
                              else f'화면 {now:%H:%M} KST (앱 시뮬레이션)')
        if weather.get('source') == 'kma' and not isinstance(api_kpis, dict):
            # 기상 캐시(5분)는 API 값인데 KPI 응답(60초 캐시)이 끊긴 사이: 그 값이 실측인지 API 기본값인지 확인할 수 없다.
            # 카드 문법(명세 5.5)에 '확인 불가' 칸이 없어 실측 칩을 피해 가장 가까운 '실측 아님' 칸을 쓰고 사유를 시각 줄에 적는다
            weather['pd_src'] = 'default'
            weather['pd_time'] = 'API 응답 없음 · 최대 5분 전 값이라 출처 확인 불가'
        smp_data = {**smp_data, **_fmt_smp_display_keys(smp_data, now)}
        reserve_rate = power_status['reserve_rate']
        level = _fmt_reserve_level(reserve_rate)
        # 실측이 아니면 빗금 클래스와 덧말을 붙이고, 기본값(API 25%·앱 15%)이면 단계를 판정하지 않는다(7.2b)
        reserve_class, reserve_text = _fmt_reserve_display(level, prov)
        demand_observed_text, demand_elapsed_text = _format_observation_age(
            power_status.get('data_date', 'N/A')
        )
        demand_observed_text, demand_elapsed_text = _fmt_observation_display(
            prov, demand_observed_text, demand_elapsed_text
        )

        st.html(_html_app_bar(power_status, smp_data, level, prov, auto_refresh))
        # 경보 배너 자리도 st.empty()로 늘 만들어 요소 개수를 고정한다(위와 같은 이유: 배너가 조건부 요소면 배너가
        # 사라지는 순간 뒤 요소가 한 칸씩 당겨지고 지난 KPI 줄이 남는다). 빈 자리는 공간을 차지하지 않고, 배너는 따로 된
        # 요소라 글자가 같으면 DOM이 유지돼 role="alert"가 매분 다시 읽히지 않는다
        alert_slot = st.empty()
        if level['show_alert']:
            alert_slot.html(_html_alert_banner(level, prov))
        st.html(_html_status_line(power_status, level, prov))

        with st.container(key="kpi_row"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""<div class="pd-kpi {power_status.get('pd_state_class', '')}">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--bolt" aria-hidden="true"></span>현재 수요</span>{power_status.get('pd_chip', '')}</div>
<div class="pd-kpi__value"><span class="pd-num">{power_status['demand']:,.0f}</span><span class="pd-kpi__unit">MW</span></div>
<div class="pd-kpi__sub"><span class="pd-pill status-badge {reserve_class}"><span class="pd-pill__dot" aria-hidden="true"></span>예비율 {reserve_rate:.1f}% · {reserve_text}</span></div>
<div class="pd-kpi__meta"><span>출처 {power_status.get('pd_src_label', 'KPX 실시간')}</span><span>관측 {demand_observed_text} · {demand_elapsed_text}</span></div>
</div>""", unsafe_allow_html=True)
            with col2:
                # 이전 시간 실측값이 없으면 화살표/변화율 대신 안내 문구를 표시한다
                if smp_data.get('prev_available', False) and smp_data['change'] is not None:
                    # 소수 1자리로 0.0이 되는 변화는 오름(▲ +0.0%)으로 보이지 않게 '변동 없음'으로 쓴다(1차 평가, 명세 5.5 글자에서 바꿈)
                    smp_flat = abs(smp_data['change']) < 0.05
                    smp_change_class = "" if smp_flat else ("is-up" if smp_data['change'] > 0 else "is-down")
                    smp_arrow = "–" if smp_flat else ("▲" if smp_data['change'] > 0 else "▼")
                    smp_change_html = (
                        f'{smp_arrow} 0.0 원/kWh (변동 없음) · 전 거래시간 대비' if smp_flat else
                        f'{smp_arrow} {abs(smp_data["change"]):.1f} 원/kWh ({smp_data["change_pct"]:+.1f}%) · 전 거래시간 대비'
                    )
                else:
                    smp_change_class = ""
                    smp_change_html = "이전 시간 데이터 없음"
                st.markdown(f"""<div class="pd-kpi{'' if str(smp_data.get('data_source', '')).startswith('EPSIS') and smp_data.get('trade_label') else ' is-est'}">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--tag" aria-hidden="true"></span>제주 하루전 SMP</span>{'<span class="pd-chip pd-chip--measured"><span class="pd-chip__glyph" aria-hidden="true"></span>확정 가격</span>' if str(smp_data.get('data_source', '')).startswith('EPSIS') and smp_data.get('trade_label') else ('<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>추정 · 최근 48시간 평균</span>' if str(smp_data.get('data_source', '')).startswith('EPSIS') else '<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>시뮬레이션 · 실측 아님</span>')}{smp_data.get('pd_stale_chip', '')}</div>
<div class="pd-kpi__value"><span class="pd-num">{smp_data['current']:,.1f}</span><span class="pd-kpi__unit">원/kWh</span></div>
<div class="pd-kpi__sub"><span class="pd-delta {smp_change_class}">{smp_change_html}</span></div>
<div class="pd-kpi__meta"><span>출처 {smp_data.get('pd_src_label', '확인 불가')}</span><span>거래시간 {smp_data.get('trade_label') or '확인 불가'}</span></div>
</div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""<div class="pd-kpi is-est">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--leaf" aria-hidden="true"></span>재생에너지 비율</span><span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>추정 · 실측 아님</span></div>
<div class="pd-kpi__value"><span class="pd-num">{power_status['renewable_ratio']:.1f}</span><span class="pd-kpi__unit">%</span></div>
<div class="pd-kpi__sub">태양광·풍력이 지금 수요를 채우는 비율</div>
<div class="pd-kpi__meta"><span>{power_status.get('renewable_source', '태양광 + 풍력 (추정치)')}</span><span>{power_status.get('pd_renew_time', '조회 시각 미제공')} · 발전 현황 탭 구성비와 산출 근거 다름</span></div>
</div>""", unsafe_allow_html=True)
            with col4:
                st.markdown(f"""<div class="pd-kpi{'' if weather.get('source') == 'kma' and weather.get('pd_src') not in ('file', 'default') else ' is-est'}">
<div class="pd-kpi__head"><span class="pd-kpi__label"><span class="pd-ico pd-ico--thermo" aria-hidden="true"></span>기상 현황</span>{'<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>추정 · 과거 자료 평균</span>' if weather.get('pd_src') == 'file' else ('<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>추정 · API 기본값 (실측 아님)</span>' if weather.get('pd_src') == 'default' else ('<span class="pd-chip pd-chip--measured"><span class="pd-chip__glyph" aria-hidden="true"></span>실측</span>' if weather.get('source') == 'kma' else '<span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>시뮬레이션 · 실측 아님</span>'))}</div>
<div class="pd-kpi__value"><span class="pd-num">{weather['temperature']:.1f}</span><span class="pd-kpi__unit">°C</span></div>
<div class="pd-kpi__sub">풍속 {weather['wind_speed']:.1f} m/s{(' · ' + str(weather.get('condition'))) if weather.get('condition') else ''}</div>
<div class="pd-kpi__meta"><span>{'기상청 과거 자료 24시간 평균 (API 서버) · 실시간 아님' if weather.get('pd_src') == 'file' else ('API 서버 기본값 (실측 아님)' if weather.get('pd_src') == 'default' else ('기상청 초단기실황 (API 서버)' if weather.get('source') == 'kma' else '시뮬레이션 값 (실측 아님)'))}</span><span>{weather.get('pd_time', '관측 시각 미제공 · 최대 5분 캐시')}</span></div>
</div>""", unsafe_allow_html=True)

    live_header()
    # 문서 언어를 한국어로 밝힌다(N13: 화면 읽기 프로그램 발음·:lang 규칙). Streamlit index.html이 lang="en"이라 정적 요소인
    # 읽는 법 띠에 스크립트 한 줄을 붙여 마운트 때 한 번 바꾼다. 스크립트만 담은 st.html은 빈 칸과 16px 간격을 만든다(G16)
    st.html(_html_legend_strip() + '<script>document.documentElement.lang = "ko";</script>',
            unsafe_allow_javascript=True)

    tab1, tab2, tab3, tab4 = st.tabs([":material/map: 실시간 현황", ":material/sell: SMP 예측",
                                      ":material/bolt: 발전 현황", ":material/insights: 분석"])

    with tab1:
        # 지도 무대(지도 8/12 + 인스펙터 4/12, 명세 5.8절). 지도(st_folium)는 fragment 밖에 둔다:
        # 안에 두면 60초마다 지도 JS가 바뀌어 iframe이 다시 마운트된다(N14)
        with st.container(border=True, key="map_stage"):
            with st.container(key="map_head"):
                head_l, head_r = st.columns([3, 1], vertical_alignment="center")
                with head_l:
                    st.html(_html_map_head(plants_df, registry))
                with head_r:
                    show_heatmap = st.toggle("추정 출력 열지도", value=False, key="map_heatmap")
            with st.container(key="map_body"):
                map_col, insp_col = st.columns([8, 4], gap="medium")
                with map_col:
                    with st.container(key="map_col"):
                        jeju_map = create_jeju_map(plants_df, show_heatmap)
                        st_folium(jeju_map, height=480, use_container_width=True, returned_objects=[], key="jeju_map")
                        st.html(_html_map_legend(plants_df))
                        if show_heatmap:
                            st.html(_html_chip("est", "열지도 = 시뮬레이션 추정 출력 · 실측 아님"))
                with insp_col:
                    map_region = st.segmented_control("지역", ["전체", "제주시", "서귀포시"], default="전체",
                                                      key="map_region", label_visibility="collapsed") or "전체"
                    st.html(_html_inspector(plants_df, registry, map_region))

        # 수급 추이 카드와 상태 줄(게이지 · 계통 상태 · 주파수)을 60초마다 다시 그린다(N12). 지도는 이 fragment 밖이다(N14).
        # main() 안에 중첩 정의해야 회귀 테스트(ast.walk(main))가 게이지 할당과 주파수 카드를 찾는다
        @st.fragment(run_every=run_every)
        def live_supply():
            now = datetime.now()
            # 데이터 읽기를 한 칸 안에서 해 캐시 진행 표시가 화면 요소 위치를 밀지 않게 한다(live_header 주석과 같은 이유.
            # 60초 fragment가 늘면 이 fragment 실행도 조기 종료로 끝날 수 있다)
            with st.empty():
                power_status = dict(get_current_power_status())
                if test_alert:
                    power_status['reserve_rate'] = test_reserve
                    power_status['data_source'] = '테스트 모드'  # 빠지면 테스트 값에 '실측' 칩이 붙는다
                prov = _fmt_provenance(power_status, fetch_backend_json("/api/v1/dashboard/kpis"), fetch_jeju_realtime(), now)
            # 주파수 카드의 조회 시각(pd_time) 등 표시 키는 이 화면 전용 사본에만 넣는다(경보 발송 dict와 별개)
            power_status.update(prov['display_keys'])
            with st.container(border=True, key="card_supply"):
                st.html(_html_card_head("수급 추이", "지난 12시간 · 다음 12시간", prov['supply_chips'], prov['supply_meta']))
                st.html(_html_supply_legend(power_status, prov))
                # 현재 점은 KPI와 같은 값으로 그린다(N3). now·조회 시각을 넘겨 pill·시간 띠·툴팁의 시각이 같게 한다
                fig = create_realtime_power_chart(power_status['demand'], power_status['total_supply'],
                                                  prov['demand_measured'], prov['supply_measured'],
                                                  now=now, query_hhmm=prov['query_hhmm'])
                st.plotly_chart(fig, theme="streamlit", width="stretch", config={"displayModeBar": False}, key="chart_supply")
                st.html(_html_time_strip(prov))
                with st.expander("표로 보기", icon=":material/table_view:"):
                    supply_table = _chart_table_supply(fig, prov['strip_now'])  # 1시간 간격(명세 6.2절)
                    st.dataframe(supply_table, hide_index=True, width="stretch",
                                 height=(len(supply_table) + 1) * 35 + 3,  # 표 안 세로 스크롤 없음(N17)
                                 column_config=_fmt_supply_columns())
                st.html(_html_card_foot(prov['supply_foot']))
            with st.container(key="status_row"):
                c_gauge, c_grid, c_freq = st.columns([5, 4, 3], gap="medium")
                with c_gauge:
                    with st.container(border=True, key="card_gauges"):
                        st.html(_html_card_head("수급 게이지", "", prov['demand_chips'], prov['demand_meta']))
                        g1, g2 = st.columns(2, gap="small")
                        with g1:
                            st.html(_html_gauge_label("수요", "축 0–1,200 MW · 빨간 눈금 1,080 MW"))
                            demand_gauge = create_gauge_chart(
                                value=power_status['demand'], title='', max_val=1200, color='#3B82F6', suffix=' MW'
                            )
                            # 표시 형식만 덧입힌다(게이지 함수와 할당 3개는 명세 6.4·T7 계약 그대로): 눈금에 천 단위 쉼표
                            # (이름표·캡션의 '1,080'과 같은 표기, F12), 숫자는 KPI 숫자처럼 굵게
                            demand_gauge.update_traces(gauge_axis_tickformat=",", number_font_weight=600)
                            st.plotly_chart(demand_gauge, theme="streamlit", width="stretch",
                                            config={"displayModeBar": False}, key="gauge_demand")
                        with g2:
                            # 예비율이 기본값·전국 값이면(④⑤⑥⑨) 색 구간이 제주 단계처럼 읽히지 않게 이름표에 판정 안 함을 밝힌다
                            # (KPI pill·계통 상태 카드와 같은 덧말, 명세 7.2b reserve_note)
                            st.html(_html_gauge_label("예비율", "빨강 10% 미만 · 호박 10–15% · 초록 15% 이상" if prov['reserve_known']
                                                      else f"판정 안 함{prov['reserve_note']} · 색 구간은 참고용"))
                            reserve_color = '#16A34A' if power_status['reserve_rate'] >= 15 else '#B45309' if power_status['reserve_rate'] >= 10 else '#DC2626'
                            reserve_gauge = create_gauge_chart(
                                value=power_status['reserve_rate'], title='', max_val=30, color=reserve_color,
                                suffix='%', reserve=True
                            )
                            reserve_gauge.update_traces(number_font_weight=600)
                            st.plotly_chart(reserve_gauge, theme="streamlit", width="stretch",
                                            config={"displayModeBar": False}, key="gauge_reserve")
                with c_grid:
                    with st.container(border=True, key="card_grid"):
                        st.html(_html_grid_state(power_status, _fmt_reserve_level(power_status['reserve_rate']), prov))
                with c_freq:
                    # 시뮬레이션 값이라 KPI 띠에서 내려 상태 줄 맨 끝에 둔다(명세 2.2·5.9절)
                    with st.container(border=True, key="card_freq"):
                        st.markdown(f"""<div class="pd-card-sim">
<div class="pd-card__head"><span class="pd-card__title"><span class="pd-ico pd-ico--wave" aria-hidden="true"></span>계통 주파수</span><span class="pd-chip pd-chip--est"><span class="pd-chip__glyph" aria-hidden="true"></span>시뮬레이션 · 실측 아님</span></div>
<div class="pd-sim-value"><span class="pd-num">{power_status['frequency']:.2f}</span><span class="pd-kpi__unit">Hz</span></div>
<p class="pd-card__note">API 서버 시뮬레이션 값 (실측 아님) · 계통 실측 연동 전 참고용 · 정격 60 Hz · {power_status.get('pd_time', '조회 시각 미제공')}</p>
</div>""", unsafe_allow_html=True)

        live_supply()

    # SMP 탭 전체를 fragment로: 현재 거래시간 음영·pill·현재가·확정 가격 점이 앱 바·KPI와 어긋나지 않게(명세 5.3절).
    # main() 안에 중첩 정의해야 회귀 테스트(ast.walk(main))가 입찰 예시 카드를 찾는다. 300초 주기라 정시 직후 최대 5분은
    # 앱 바보다 늦을 수 있으므로 카드 메타에 화면 시각을 적는다
    @st.fragment(run_every=run_every_slow)
    def live_smp():
        now = datetime.now()
        # 300초 주기도 5분마다 60초 fragment와 같은 때 돈다. 그때 먼저 도는 실행은 조기 종료(script_finished 2)로 끝나
        # 낡은 요소가 지워지지 않으므로(S4 실측) 캐시 진행 표시가 요소 위치를 밀지 않게 데이터 읽기를 한 칸 안에서 한다
        with st.empty():
            smp_data = get_smp_data()
            backend_predictions = get_backend_smp_predictions()
        if backend_predictions:
            smp_data = {**smp_data, **backend_predictions}
        smp_data = {**smp_data, **_fmt_smp_display_keys(smp_data, now)}
        smp_l, smp_r = st.columns([8, 4], gap="medium")
        with smp_l:
            with st.container(border=True, key="card_smp"):
                st.html(_html_smp_head(smp_data, now,
                                       refresh_note="5분마다 갱신" if auto_refresh else "자동 갱신 꺼짐"))
                # 차트의 비시각 대체 수단(J10). 차트·표는 같은 한 자리를 번갈아 쓰므로 요소 개수가 바뀌지 않는다
                smp_view = st.segmented_control("보기", ["차트", "표"], default="차트", key="smp_view",
                                                label_visibility="collapsed") or "차트"
                if smp_view == "차트":
                    st.plotly_chart(create_smp_chart(smp_data, now=now), theme="streamlit", width="stretch",
                                    config={"displayModeBar": False}, key="chart_smp")
                else:
                    smp_table = _chart_table_smp(smp_data, now=now)
                    st.dataframe(smp_table, hide_index=True, width="stretch",
                                 height=(len(smp_table) + 1) * 35 + 3,  # 표 안 세로 스크롤 없음(N17)
                                 column_config=_fmt_number_columns(list(smp_table.columns[2:-1])))
                st.html(_html_card_foot(_fmt_smp_foot(smp_data, now)))
        with smp_r:
            with st.container(border=True, key="card_smp_stats"):
                st.html(_html_smp_stats(smp_data))
            with st.container(border=True, key="card_smp_blocks"):
                st.html(_html_smp_blocks(smp_data, now))
            with st.container(border=True, key="card_smp_model"):
                st.html(_html_smp_model(smp_data))
            # 지어낸 숫자는 기본 화면에 두지 않는다(J11): 입찰 예시는 접힌 expander 안에서만 보인다
            with st.expander("예시 화면 · 입찰 예시 (실데이터 아님)", icon=":material/science:"):
                optimal_bid = smp_data['current'] * 0.95
                st.markdown(f"""<div class="pd-example">
<div class="pd-example__ribbon">예시 · 실제 값 아님</div>
<div class="pd-card__head"><span class="pd-card__title">입찰 추천</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<div class="pd-example__value"><span class="pd-num">{optimal_bid:.1f} 원/kWh</span></div>
<p class="pd-example__note">예시: 현재가에 고정 할인율 5% 적용 (최적화 결과 아님)</p>
<p class="pd-example__note">낙찰 확률 예시 (고정값, 실제 낙찰 확률 아님) <span class="pd-example__num">85%</span></p>
</div>""", unsafe_allow_html=True)

    with tab2:
        live_smp()

    # 도넛·타임라인 행만 fragment로: 타임라인의 현재 시각 상자가 정시를 따라간다. 등록부 표(필터 위젯)는 밖에 둔다
    @st.fragment(run_every=run_every_slow)
    def live_generation():
        now = datetime.now()
        with st.empty():  # live_smp와 같은 이유로 데이터 읽기를 한 칸 안에서 한다
            power_status = get_current_power_status()
            plants_df = get_jeju_power_plants()
        mix_col, tl_col = st.columns([5, 7], gap="medium")
        with mix_col:
            with st.container(border=True, key="card_mix"):
                st.html(_html_card_head("전력 공급 구성 (시뮬레이션 추정)", "로컬 추정 · 상단 재생에너지 비율(섬 전체 ML 추정)과 산출 근거 다름",
                                        _html_chip("est", "시뮬레이션 · 실측 아님"), _fmt_screen_meta("시뮬레이션", now, "60초 캐시")))
                fig = create_supply_donut(power_status)
                st.plotly_chart(fig, theme="streamlit", width="stretch", config={"displayModeBar": False}, key="chart_mix")
                st.html(_html_mix_table(power_status))
        with tl_col:
            with st.container(border=True, key="card_timeline"):
                st.html(_html_card_head("시간대별 발전량 (시뮬레이션)", _fmt_timeline_sub(plants_df, registry),
                                        _html_chip("est", "시뮬레이션 · 실측 아님"), _fmt_screen_meta("시뮬레이션", now, "60초 캐시")))
                fig = create_generation_timeline(plants_df)
                st.plotly_chart(fig, theme="streamlit", width="stretch", config={"displayModeBar": False}, key="chart_timeline")
                st.html(_html_timeline_legend())
                with st.expander("표로 보기", icon=":material/table_view:"):
                    timeline_table = _chart_table_timeline(fig)
                    st.dataframe(timeline_table, hide_index=True, width="stretch",
                                 height=(len(timeline_table) + 1) * 35 + 3,  # 표 안 세로 스크롤 없음(N17)
                                 column_config=_fmt_number_columns(list(timeline_table.columns[1:4])))

    with tab3:
        live_generation()
        with st.container(border=True, key="card_plants"):
            st.html(_html_card_head("발전소 등록부", "위치·설비는 등록부, 출력·이용률은 시뮬레이션 추정", "",
                                    _fmt_registry_ref(registry)))
            type_filter = st.segmented_control("발전 유형", ["전체", "풍력", "태양광", "ESS", "화력"],
                                               default="전체", key="plant_type") or "전체"
            display_df = _fmt_plant_table(plants_df, type_filter)
            st.html(_html_plant_summary(display_df, registry, type_filter))  # "운영중만 표시 (건설중 2곳·165 MW 제외)"
            st.dataframe(display_df, hide_index=True, width="stretch", height=(len(display_df) + 1) * 35 + 3,
                         column_config=_fmt_plant_columns())

    with tab4:
        # 모델 카드 글자는 계약(명세 5.6)이라 바꾸지 않고, 비전문가가 MAPE·R²와 '화면에 쓰였는지'를 읽을 수 있게 머리에 풀어 쓴다
        st.html(_html_section_head("모델 정확도와 예시 화면",
                                   "MAPE = 평균 오차율(낮을수록 정확) · R² = 설명력(1에 가까울수록 정확). 아래 수치는 벤치마크로 "
                                   "검증한 고정 값이고, 수요 예측 모델의 결과는 아직 이 화면의 차트에 연결되지 않았습니다. "
                                   "예시 화면은 기능 설명용 고정 문구입니다"))
        with st.container(border=True, key="card_model"):
            st.markdown("""<div class="pd-model">
<div class="pd-card__head"><span class="pd-card__title"><span class="pd-ico pd-ico--chart" aria-hidden="true"></span>모델 성능</span><span class="pd-chip pd-chip--plain">벤치마크 검증값</span></div>
<p class="pd-card__note">README 운영 모델 성능 표 기준 (2026-11-06 벤치마크 검증값)</p>
<div class="pd-model__grid">
<section class="pd-model__cell"><p class="pd-model__name"><span class="pd-ico pd-ico--plug" aria-hidden="true"></span>수요 예측 — 1시간 앞 전력수요 (KPX 제주 전력거래량)</p><dl><dt>AR + GBM 잔차 보정 · MAPE</dt><dd class="pd-num">4.67%</dd><dt>R² Score</dt><dd class="pd-num">0.929</dd></dl><p class="pd-card__note">2025년 실측 8,760시간 (학습에 안 쓴 기간)</p></section>
<section class="pd-model__cell"><p class="pd-model__name"><span class="pd-ico pd-ico--sun" aria-hidden="true"></span>공급 추정 — 태양광 발전량 (GradientBoosting)</p><dl><dt>KPX 오차율</dt><dd class="pd-num">17.0%</dd><dt>R² Score</dt><dd class="pd-num">0.41</dd></dl><p class="pd-card__note">holdout 2022-12~2024-05 · 기준선(월×시간 기후값 12.5%)보다 나쁨 · API 응답의 "신뢰도 93%"는 모델 성능이 아님</p></section>
<section class="pd-model__cell"><p class="pd-model__name"><span class="pd-ico pd-ico--tag" aria-hidden="true"></span>SMP 예측 — 하루전 시장가격 (어제 같은 시각 가격)</p><dl><dt>MAPE</dt><dd class="pd-num">11.75%</dd><dt>R² Score</dt><dd class="pd-num">0.203</dd></dl><p class="pd-card__note">walk-forward 2025-07-22~12-18 · 기준선(naive) 자체, 학습한 모델 아님</p></section>
</div>
</div>""", unsafe_allow_html=True)
        # 예시 카드 3장은 접힌 expander 안에만 둔다(J11). 카드 문자열은 회귀 테스트가 찾는 일반 문자열 상수다
        with st.expander("예시 화면 3종 · 실데이터 아님", icon=":material/science:"):
            st.html(_html_example_ribbon())
            with st.container(key="ex_zone"):
                ex1, ex2, ex3 = st.columns(3, gap="medium")
                with ex1:
                    st.markdown("""<div class="pd-example">
<div class="pd-card__head"><span class="pd-card__title">XAI 분석 요약</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<p class="pd-example__lead">예시 (실제 모델 분석 결과 아님)</p>
<ul class="pd-example__list">
<li>기온 상승 (+2°C) → 수요 증가 영향 15%</li>
<li>일사량 감소 → 태양광 발전 감소 예상</li>
<li>풍속 증가 → 풍력 발전 증가 기대</li>
<li>과거 동일 시간대 패턴 반영 40%</li>
</ul>
</div>""", unsafe_allow_html=True)
                with ex2:
                    st.markdown("""<div class="pd-example">
<div class="pd-card__head"><span class="pd-card__title">리스크 알림</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<p class="pd-example__lead">예시 (실제 운영 경보 또는 입찰 권고 아님)</p>
<div class="pd-example__item"><b>덕커브 주의</b><span>14:00-16:00 태양광 발전 급증으로 SMP 하락 예상</span></div>
<div class="pd-example__item"><b>풍력 발전 증가</b><span>풍속 증가로 풍력 발전량 20% 상승 예상</span></div>
<div class="pd-example__item"><b>입찰 기회</b><span>18:00-20:00 피크 시간대 고가 입찰 권장</span></div>
</div>""", unsafe_allow_html=True)
                with ex3:
                    st.markdown("""<div class="pd-example">
<div class="pd-card__head"><span class="pd-card__title">일간 요약</span><span class="pd-chip pd-chip--example"><span class="pd-chip__glyph" aria-hidden="true"></span>예시</span></div>
<p class="pd-example__lead">고정 예시 (실제 일간 집계 아님)</p>
<div class="pd-example__grid">
<div><span>총 발전량</span><b class="pd-num">12,450 MWh</b></div>
<div><span>CO₂ 절감</span><b class="pd-num">5,890 톤</b></div>
<div><span>평균 이용률</span><b class="pd-num">42.3%</b></div>
<div><span>예상 수익</span><b class="pd-num">₩1.2B</b></div>
</div>
</div>""", unsafe_allow_html=True)

    # 출처와 기준 시각 표(명세 5.13절)를 60초마다 다시 그린다(N12). 머리·KPI·카드가 쓰는 판정 함수를 다시 불러
    # 같은 상태를 한 표에 모은다. main() 안에 중첩 정의한다(골격 규칙). 모든 탭 본문 아래, 푸터 위에 놓인다
    @st.fragment(run_every=run_every)
    def live_sources():
        now = datetime.now()
        # 세 번째 60초 fragment라 다른 60초 fragment와 같은 때 돈다. 그때 먼저 도는 실행은 조기 종료로 끝나 낡은 요소가
        # 지워지지 않으므로(S4 실측) 캐시 진행 표시가 요소 위치를 밀지 않게 데이터 읽기를 한 칸 안에서 한다
        with st.empty():
            power_status = dict(get_current_power_status())
            if test_alert:
                power_status['reserve_rate'] = test_reserve
                power_status['data_source'] = '테스트 모드'  # 빠지면 테스트 값의 행에 '실측' 칩이 붙는다
            smp_data = get_smp_data()
            backend_predictions = get_backend_smp_predictions()
            weather = get_weather_data()
            api_kpis = fetch_backend_json("/api/v1/dashboard/kpis")
            kpx_local = fetch_jeju_realtime()
            sources_plants = get_jeju_power_plants()
            sources_registry = load_power_plants_data()
        st.html(_html_data_status(power_status, smp_data, backend_predictions, weather, api_kpis, kpx_local,
                                  sources_registry, now, plants_df=sources_plants,
                                  refresh_note="60초마다 갱신" if auto_refresh else "자동 갱신 꺼짐"))

    live_sources()

    # ========== 푸터 (main() 최상위 상수: T1 '데이터 출처' 기준점) ==========
    st.markdown("""<div class="pd-footer">
<p><b>제주 전력 지도 v4.1</b> · 데이터 출처: KPX(수요·공급능력, 5분), EPSIS(SMP 하루전 가격), 초단기실황(API 서버), 발전소 등록부</p>
<p>모델: 수요 AR+GBM · 태양광 GradientBoosting · SMP 어제 같은 시각(naive) · 추정·시뮬레이션 값은 화면에 그렇게 표시합니다 · 화면 테마: 오른쪽 위 ⋮ → Settings · Power Demand Forecast Team</p>
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
