"""
제주도 전력 지도 대시보드 v4.0
================================

60hz.io 스타일의 인터랙티브 지도 기반 전력 수요 예측 대시보드
실제 EPSIS 데이터 연동

주요 기능:
1. 🗺️ 제주도 지도 - 발전소 위치 및 실시간 발전량
2. ⚡ SMP 예측 - EPSIS 실제 데이터 + AI 예측
3. 📊 실시간 현황 - 제주 계통 수요/공급 실데이터
4. 🌤️ 기상 연동 - 기상 데이터 오버레이
5. 🔍 XAI 분석 - 예측 근거 설명

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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from streamlit_folium import st_folium, folium_static
import json
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
                body {{ font-family: Arial, sans-serif; }}
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

st.set_page_config(
    page_title="제주 전력 지도 v4.0",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================================
# CSS 스타일링 (60hz.io 스타일)
# ============================================================================

st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }

    /* 사이드바 숨기기 (옵션) */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* 메인 헤더 */
    .main-header {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        padding: 1rem 2rem;
        border-radius: 1rem;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .main-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    .main-subtitle {
        color: rgba(255,255,255,0.8);
        font-size: 0.9rem;
    }

    /* 정보 카드 */
    .info-card {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }

    .info-card-title {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .info-card-value {
        color: white;
        font-size: 2rem;
        font-weight: 700;
    }

    .info-card-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .positive { color: #10b981; }
    .negative { color: #ef4444; }

    /* 상태 표시 */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .status-online {
        background: rgba(16, 185, 129, 0.2);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }

    .status-danger {
        background: rgba(239, 68, 68, 0.2);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .status-critical {
        background: rgba(239, 68, 68, 0.3);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.5);
        animation: pulse-danger 1.5s ease-in-out infinite;
    }

    @keyframes pulse-danger {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    /* 알림 배너 */
    .alert-banner {
        padding: 1rem 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .alert-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
    }

    .alert-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.3) 0%, rgba(180, 83, 9, 0.3) 100%);
        border: 1px solid rgba(245, 158, 11, 0.5);
        color: #fcd34d;
    }

    .alert-icon {
        font-size: 2rem;
    }

    .alert-content {
        flex: 1;
    }

    .alert-title {
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.25rem;
    }

    .alert-message {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* 지도 컨테이너 */
    .map-container {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 1rem;
        padding: 0.5rem;
        overflow: hidden;
    }

    /* 범례 */
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.25rem 0;
        color: white;
        font-size: 0.85rem;
    }

    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 0.75rem;
        padding: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
    }

    /* 메트릭 카드 그리드 */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1rem;
    }

    /* 차트 컨테이너 */
    .chart-container {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 1rem;
        padding: 1rem;
    }

    /* 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Metric 스타일 오버라이드 */
    [data-testid="stMetricValue"] {
        color: white;
        font-size: 1.8rem;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricDelta"] svg {
        display: none;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# EPSIS 실제 데이터 로딩 함수
# ============================================================================

def _fix_timestamp_24h(df: pd.DataFrame) -> pd.DataFrame:
    """Fix 24:00 timestamp format (convert to 00:00 of next day)"""
    if 'timestamp' in df.columns:
        # Replace 24:00 with 00:00 (pandas will handle as string first)
        df['timestamp'] = df['timestamp'].astype(str).str.replace(' 24:00', ' 00:00')
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
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

    # 태양광 발전소
    if 'solar_farms' in plants_data:
        for sf in plants_data['solar_farms']:
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

    # ESS 설비
    if 'ess_facilities' in plants_data:
        for ess in plants_data['ess_facilities']:
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

    # 화력 발전소
    if 'thermal_plants' in plants_data:
        for tp in plants_data['thermal_plants']:
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

    df['utilization'] = df.apply(
        lambda row: min(max(abs(row['generation']) / row['capacity'] * 100, 0), 100) if row['capacity'] > 0 else 0,
        axis=1
    )

    return df


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
        reserve_rate = realtime_data.get('supply_reserve', 15.0)
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
    }


@st.cache_data(ttl=60)
def get_smp_data() -> Dict:
    """SMP 데이터 (EPSIS 실제 데이터 기반)"""
    current_hour = datetime.now().hour

    # 실제 SMP 데이터 로드
    smp_df = load_smp_history()

    if not smp_df.empty:
        # 최근 데이터 사용
        recent_df = smp_df.tail(48)  # 최근 48시간

        # 현재 시간대 SMP (가장 최근 동일 시간)
        hour_data = smp_df[smp_df['hour'] == (current_hour if current_hour > 0 else 24)]
        if not hour_data.empty:
            current_smp = float(hour_data['smp_jeju'].iloc[-1])
        else:
            current_smp = float(recent_df['smp_jeju'].mean())

        # 이전 시간 SMP
        prev_hour_data = smp_df[smp_df['hour'] == ((current_hour - 1) if current_hour > 1 else 24)]
        if not prev_hour_data.empty:
            prev_smp = float(prev_hour_data['smp_jeju'].iloc[-1])
        else:
            prev_smp = current_smp * 0.98

        # 시간대별 평균 SMP (실제 데이터 기반)
        hourly_avg = smp_df.groupby('hour')['smp_jeju'].mean().to_dict()

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

        # 통계
        daily_avg = float(smp_df['smp_jeju'].mean())
        daily_max = float(smp_df['smp_jeju'].max())
        daily_min = float(smp_df['smp_jeju'].min())

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

    change = current_smp - prev_smp
    change_pct = (change / prev_smp * 100) if prev_smp > 0 else 0

    return {
        "current": round(current_smp, 1),
        "change": round(change, 1),
        "change_pct": round(change_pct, 1),
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
    """기상 데이터"""
    hour = datetime.now().hour

    return {
        "temperature": round(10 + 8 * np.sin(np.pi * (hour - 6) / 12) + random.uniform(-2, 2), 1),
        "humidity": round(60 + random.uniform(-15, 15), 0),
        "wind_speed": round(5 + random.uniform(0, 10), 1),
        "wind_direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        "cloud_cover": round(random.uniform(0, 80), 0),
        "solar_radiation": round(800 * np.sin(np.pi * max(0, hour - 6) / 12) if 6 <= hour <= 18 else 0, 0),
        "precipitation": round(random.uniform(0, 2), 1) if random.random() > 0.7 else 0,
    }


# ============================================================================
# 지도 생성 함수
# ============================================================================

def create_jeju_map(plants_df: pd.DataFrame, show_heatmap: bool = False) -> folium.Map:
    """제주도 지도 생성"""

    # 제주도 중심 좌표
    jeju_center = [33.3846, 126.5535]

    # 지도 생성 (다크 테마)
    m = folium.Map(
        location=jeju_center,
        zoom_start=10,
        tiles=None,
        control_scale=True,
    )

    # 다크 테마 타일 추가
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='CartoDB Dark',
        name='Dark Mode',
        control=False,
    ).add_to(m)

    # 발전소 타입별 색상 및 아이콘
    type_config = {
        'solar': {'color': '#fbbf24', 'icon': 'sun', 'prefix': 'fa', 'label': '태양광'},
        'wind': {'color': '#3b82f6', 'icon': 'wind', 'prefix': 'fa', 'label': '풍력'},
        'ess': {'color': '#8b5cf6', 'icon': 'battery-half', 'prefix': 'fa', 'label': 'ESS'},
        'thermal': {'color': '#ef4444', 'icon': 'fire', 'prefix': 'fa', 'label': '화력'},
    }

    # 발전소 마커 추가
    for _, plant in plants_df.iterrows():
        config = type_config.get(plant['type'], {'color': 'gray', 'icon': 'bolt', 'prefix': 'fa', 'label': '기타'})

        # 추가 정보 (operator, address, source)
        operator = plant.get('operator', '-') if pd.notna(plant.get('operator')) else '-'
        address = plant.get('address', '-') if pd.notna(plant.get('address')) else '-'
        source = plant.get('source', '-') if pd.notna(plant.get('source')) else '-'
        subtype = plant.get('subtype', '') if pd.notna(plant.get('subtype')) else ''

        # 서브타입 한글 변환
        subtype_labels = {
            'onshore': '육상', 'offshore': '해상', 'island': '도서',
            'ground-mounted': '지상설치', 'rooftop': '옥상설치', 'community': '시민참여',
            'utility-scale': '대규모', 'renewable-coupled': '재생연계',
            'combined-cycle': '복합화력'
        }
        subtype_label = subtype_labels.get(subtype, subtype)

        # 팝업 내용 (실제 데이터 정보 포함)
        popup_html = f"""
        <div style="font-family: 'Malgun Gothic', sans-serif; width: 240px;">
            <h4 style="margin: 0 0 10px 0; color: {config['color']};">
                {plant['name']}
            </h4>
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td style="color: #666;">유형</td>
                    <td style="text-align: right; font-weight: bold;">
                        {config['label']} {f"({subtype_label})" if subtype_label else ""}
                    </td>
                </tr>
                <tr>
                    <td style="color: #666;">설비용량</td>
                    <td style="text-align: right; font-weight: bold;">{plant['capacity']:.1f} MW</td>
                </tr>
                <tr>
                    <td style="color: #666;">현재 발전량</td>
                    <td style="text-align: right; font-weight: bold; color: #10b981;">
                        {plant['generation']:.1f} MW
                    </td>
                </tr>
                <tr>
                    <td style="color: #666;">이용률</td>
                    <td style="text-align: right; font-weight: bold;">{plant['utilization']:.1f}%</td>
                </tr>
                <tr>
                    <td style="color: #666;">운영사</td>
                    <td style="text-align: right; font-size: 10px;">{operator}</td>
                </tr>
                <tr>
                    <td style="color: #666;">위치</td>
                    <td style="text-align: right; font-size: 10px;">{address}</td>
                </tr>
                <tr>
                    <td style="color: #666;">상태</td>
                    <td style="text-align: right;">
                        <span style="background: {'#10b981' if plant['status'] == '운영중' else '#f59e0b' if plant['status'] == '점검중' else '#6b7280'};
                                     color: white; padding: 2px 8px; border-radius: 10px; font-size: 11px;">
                            {plant['status']}
                        </span>
                    </td>
                </tr>
            </table>
            <div style="margin-top: 8px; font-size: 9px; color: #999; border-top: 1px solid #eee; padding-top: 5px;">
                데이터 출처: {source}
            </div>
        </div>
        """

        # 마커 크기 (발전량 기반, 화력은 더 크게)
        base_radius = 10 if plant['type'] == 'thermal' else 5
        radius = max(8, min(30, abs(plant['generation']) / 10 + base_radius))

        # 원형 마커
        folium.CircleMarker(
            location=[plant['lat'], plant['lng']],
            radius=radius,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{plant['name']}: {plant['generation']:.1f} MW",
            color=config['color'],
            fill=True,
            fillColor=config['color'],
            fillOpacity=0.7,
            weight=2,
        ).add_to(m)

    # 히트맵 (옵션)
    if show_heatmap:
        heat_data = [[row['lat'], row['lng'], row['generation']]
                     for _, row in plants_df.iterrows() if row['generation'] > 0]
        if heat_data:
            plugins.HeatMap(
                heat_data,
                min_opacity=0.3,
                max_zoom=13,
                radius=30,
                blur=20,
                gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
            ).add_to(m)

    # 범례 추가
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
                background: rgba(15, 23, 42, 0.9); padding: 15px; border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.1);">
        <h4 style="margin: 0 0 10px 0; color: white; font-size: 14px;">발전소 유형</h4>
        <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0; color: white; font-size: 12px;">
            <span style="width: 14px; height: 14px; background: #fbbf24; border-radius: 50%;"></span>
            태양광
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0; color: white; font-size: 12px;">
            <span style="width: 14px; height: 14px; background: #3b82f6; border-radius: 50%;"></span>
            풍력
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin: 5px 0; color: white; font-size: 12px;">
            <span style="width: 14px; height: 14px; background: #8b5cf6; border-radius: 50%;"></span>
            ESS
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ============================================================================
# 차트 생성 함수
# ============================================================================

def create_smp_chart(smp_data: Dict) -> go.Figure:
    """SMP 예측 차트"""
    predictions = smp_data['predictions']

    fig = go.Figure()

    # 신뢰구간
    fig.add_trace(go.Scatter(
        x=[p['time'] for p in predictions],
        y=[p['q90'] for p in predictions],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip',
    ))

    fig.add_trace(go.Scatter(
        x=[p['time'] for p in predictions],
        y=[p['q10'] for p in predictions],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(59, 130, 246, 0.2)',
        name='80% 신뢰구간',
    ))

    # 예측선
    fig.add_trace(go.Scatter(
        x=[p['time'] for p in predictions],
        y=[p['smp'] for p in predictions],
        mode='lines+markers',
        name='SMP 예측',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=6),
    ))

    # 현재 시점 표시 (첫 번째 데이터 포인트에 수직선)
    if predictions:
        fig.add_shape(
            type="line",
            x0=predictions[0]['time'],
            x1=predictions[0]['time'],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#10b981", width=2, dash="dash"),
        )
        # 현재 시점 주석
        fig.add_annotation(
            x=predictions[0]['time'],
            y=1,
            yref="paper",
            text="현재",
            showarrow=False,
            font=dict(color="#10b981", size=12),
            yanchor="bottom",
        )

    fig.update_layout(
        title=None,
        xaxis_title=None,
        yaxis_title="SMP (원/kWh)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=300,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    )

    return fig


def create_supply_donut(power_status: Dict) -> go.Figure:
    """전력 공급 구성 도넛 차트"""
    supply = power_status['supply']

    labels = ['태양광', '풍력', '화력', 'ESS']
    values = [supply['solar'], supply['wind'], supply['thermal'], abs(supply['ess'])]
    colors = ['#fbbf24', '#3b82f6', '#6b7280', '#8b5cf6']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo='percent',
        textposition='outside',
        textfont=dict(color='white', size=12),
    )])

    # 중앙 텍스트
    fig.add_annotation(
        text=f"<b>{power_status['demand']}</b><br>MW",
        x=0.5, y=0.5,
        font=dict(size=20, color='white'),
        showarrow=False,
    )

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(color='white'),
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=20, b=40),
        height=280,
    )

    return fig


def create_generation_timeline(plants_df: pd.DataFrame) -> go.Figure:
    """발전량 타임라인"""
    # 시간대별 발전량 시뮬레이션
    hours = list(range(24))
    current_hour = datetime.now().hour

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
        wind = plants_df[plants_df['type'] == 'wind']['capacity'].sum() * random.uniform(0.4, 0.7)
        wind_gen.append(wind)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hours,
        y=solar_gen,
        mode='lines',
        name='태양광',
        fill='tozeroy',
        line=dict(color='#fbbf24'),
        fillcolor='rgba(251, 191, 36, 0.3)',
    ))

    fig.add_trace(go.Scatter(
        x=hours,
        y=wind_gen,
        mode='lines',
        name='풍력',
        fill='tozeroy',
        line=dict(color='#3b82f6'),
        fillcolor='rgba(59, 130, 246, 0.3)',
    ))

    # 현재 시점
    fig.add_vline(
        x=current_hour,
        line_dash="dash",
        line_color="#10b981",
    )

    fig.update_layout(
        xaxis_title="시간",
        yaxis_title="발전량 (MW)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=0),
        height=250,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(0, 24, 3)),
            ticktext=[f"{h}시" for h in range(0, 24, 3)],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
        ),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
    )

    return fig


# ============================================================================
# 메인 대시보드
# ============================================================================

def main():
    """메인 함수"""

    # 데이터 로드
    plants_df = get_jeju_power_plants()
    power_status = get_current_power_status()
    smp_data = get_smp_data()
    weather = get_weather_data()

    # ========== 사이드바: 테스트 모드 ==========
    with st.sidebar:
        st.markdown("### ⚙️ 설정")

        # 알림 테스트 모드
        test_alert = st.checkbox("🧪 알림 테스트 모드", value=False)
        if test_alert:
            test_reserve = st.slider(
                "테스트 예비율 (%)",
                min_value=0.0,
                max_value=30.0,
                value=12.0,
                step=1.0,
                help="예비율을 낮춰서 알림 시스템을 테스트합니다"
            )
            # 테스트용 예비율 적용
            power_status = dict(power_status)  # 복사본 생성
            power_status['reserve_rate'] = test_reserve
            power_status['data_source'] = '테스트 모드'

            st.warning(f"⚠️ 테스트 모드: 예비율 {test_reserve:.1f}%")

        st.markdown("---")
        st.markdown("### 📊 데이터 출처")
        st.info(f"전력: {power_status.get('data_source', 'N/A')}")
        st.info(f"SMP: {smp_data.get('data_source', 'N/A')}")

        # ========== 경보 이력 ==========
        st.markdown("---")
        st.markdown("### 📜 경보 이력")

        alert_history = get_alert_history()
        recent_alerts = alert_history.get_recent(10)
        stats = alert_history.get_stats()

        # 통계 표시
        if stats['total'] > 0:
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("🚨 위험", stats['critical'])
            with stat_cols[1]:
                st.metric("⚠️ 주의", stats['danger'])
            with stat_cols[2]:
                st.metric("📢 관심", stats['warning'])

        # 최근 경보 목록
        if recent_alerts:
            for alert in recent_alerts[:5]:
                timestamp = datetime.fromisoformat(alert['timestamp'])
                time_str = timestamp.strftime("%m/%d %H:%M")
                status = alert['status']

                # 상태별 아이콘
                if status == 'critical':
                    icon = "🚨"
                    color = "#ef4444"
                elif status == 'danger':
                    icon = "⚠️"
                    color = "#f97316"
                else:
                    icon = "📢"
                    color = "#eab308"

                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.5); padding: 8px; border-radius: 8px;
                            margin-bottom: 5px; border-left: 3px solid {color};">
                    <div style="font-size: 0.75rem; color: #94a3b8;">{time_str}</div>
                    <div style="font-size: 0.85rem; color: white;">
                        {icon} {alert['reserve_rate']}% - {alert['title']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 이력 초기화 버튼
            if st.button("🗑️ 이력 초기화", key="clear_history"):
                alert_history.clear()
                st.rerun()
        else:
            st.caption("경보 이력이 없습니다")

    # ========== 헤더 ==========
    # 데이터 출처 확인
    smp_source = smp_data.get('data_source', 'N/A')
    power_source = power_status.get('data_source', 'N/A')
    is_kpx_realtime = 'KPX' in power_source
    is_real_data = is_kpx_realtime or 'EPSIS' in smp_source or 'EPSIS' in power_source

    # 데이터 상태 표시
    if is_kpx_realtime:
        data_status_text = '🔴 KPX 실시간 연동'
        data_status_class = 'status-online'
    elif is_real_data:
        data_status_text = '📊 EPSIS 데이터 연동'
        data_status_class = 'status-online'
    else:
        data_status_text = '⚠️ 시뮬레이션 모드'
        data_status_class = 'status-warning'

    st.markdown(f"""
    <div class="main-header">
        <div>
            <h1 class="main-title">🗺️ 제주 전력 지도</h1>
            <p class="main-subtitle">실시간 재생에너지 모니터링 및 SMP 예측</p>
            <div style="margin-top: 5px;">
                <span class="status-badge {data_status_class}">
                    {data_status_text}
                </span>
            </div>
        </div>
        <div style="text-align: right; color: white;">
            <div style="font-size: 0.9rem; opacity: 0.8;">마지막 업데이트</div>
            <div style="font-size: 1.2rem; font-weight: bold;">
                {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            <div style="font-size: 0.75rem; opacity: 0.6; margin-top: 3px;">
                SMP: {smp_data.get('record_count', 0):,}건 | 수요: {power_source}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== 예비율 알림 시스템 ==========
    reserve_rate = power_status['reserve_rate']

    # 예비율 상태 판단 (KPX 기준)
    # - 정상: >= 15%
    # - 관심: >= 10%, < 15%
    # - 주의: >= 5%, < 10%
    # - 위험: < 5%
    if reserve_rate < 5:
        reserve_status = "critical"
        reserve_class = "status-critical"
        reserve_text = "위험"
        alert_class = "alert-danger"
        alert_icon = "🚨"
        alert_title = "전력 수급 위험 경보"
        alert_msg = f"예비율 {reserve_rate:.1f}% - 즉각적인 부하 감축 필요"
        show_alert = True
    elif reserve_rate < 10:
        reserve_status = "danger"
        reserve_class = "status-danger"
        reserve_text = "주의"
        alert_class = "alert-danger"
        alert_icon = "⚠️"
        alert_title = "전력 수급 주의 경보"
        alert_msg = f"예비율 {reserve_rate:.1f}% - 전력 수급 상황 주시 필요"
        show_alert = True
    elif reserve_rate < 15:
        reserve_status = "warning"
        reserve_class = "status-warning"
        reserve_text = "관심"
        alert_class = "alert-warning"
        alert_icon = "📢"
        alert_title = "전력 수급 관심 단계"
        alert_msg = f"예비율 {reserve_rate:.1f}% - 전력 사용 절감 협조 요청"
        show_alert = True
    else:
        reserve_status = "normal"
        reserve_class = "status-online"
        reserve_text = "정상"
        alert_title = None
        alert_msg = None
        show_alert = False

    # 경보 이력 저장 (테스트 모드가 아닐 때만)
    if show_alert and not test_alert:
        alert_history = get_alert_history()
        alert_history.add_alert(
            reserve_rate=reserve_rate,
            status=reserve_status,
            title=alert_title,
            message=alert_msg
        )

        # 위험(critical) 경보일 때 이메일 발송
        if reserve_status == "critical":
            email_notifier = get_email_notifier()
            if email_notifier.is_configured():
                success, email_msg = email_notifier.send_critical_alert(
                    reserve_rate=reserve_rate,
                    status=reserve_status,
                    title=alert_title,
                    message=alert_msg,
                    power_data=power_status
                )
                if success:
                    st.toast(f"📧 이메일 발송 완료", icon="✅")

        # Slack 알림 발송 (모든 경보 레벨)
        slack_notifier = get_slack_notifier()
        if slack_notifier.is_configured():
            success, slack_msg = slack_notifier.send_alert(
                reserve_rate=reserve_rate,
                status=reserve_status,
                title=alert_title,
                message=alert_msg,
                power_data=power_status
            )
            if success:
                st.toast(f"💬 Slack 알림 발송 완료", icon="✅")

    # 알림 배너 표시
    if show_alert:
        st.markdown(f"""
        <div class="alert-banner {alert_class}">
            <div class="alert-icon">{alert_icon}</div>
            <div class="alert-content">
                <div class="alert-title">{alert_title}</div>
                <div class="alert-message">{alert_msg}</div>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 2rem; font-weight: bold;">{reserve_rate:.1f}%</div>
                <div style="font-size: 0.8rem;">예비율</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ========== 상단 메트릭 카드 ==========
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">현재 수요</div>
            <div class="info-card-value">{power_status['demand']} <span style="font-size: 1rem;">MW</span></div>
            <div class="info-card-change">
                <span class="status-badge {reserve_class}">예비율 {reserve_rate:.1f}% ({reserve_text})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        smp_change_class = "positive" if smp_data['change'] >= 0 else "negative"
        smp_arrow = "↑" if smp_data['change'] >= 0 else "↓"
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">현재 SMP (제주)</div>
            <div class="info-card-value">{smp_data['current']} <span style="font-size: 1rem;">원</span></div>
            <div class="info-card-change {smp_change_class}">{smp_arrow} {abs(smp_data['change']):.1f}원 ({smp_data['change_pct']:+.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">재생에너지 비율</div>
            <div class="info-card-value">{power_status['renewable_ratio']:.1f} <span style="font-size: 1rem;">%</span></div>
            <div class="info-card-change">태양광 + 풍력</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">계통 주파수</div>
            <div class="info-card-value">{power_status['frequency']:.2f} <span style="font-size: 1rem;">Hz</span></div>
            <div class="info-card-change"><span class="status-badge status-online">정상</span></div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="info-card">
            <div class="info-card-title">기상 현황</div>
            <div class="info-card-value">{weather['temperature']:.0f} <span style="font-size: 1rem;">°C</span></div>
            <div class="info-card-change">풍속 {weather['wind_speed']:.1f} m/s</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== 메인 컨텐츠 ==========
    st.markdown("<br>", unsafe_allow_html=True)

    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ 지도", "📊 SMP 예측", "⚡ 발전 현황", "🔍 분석"])

    with tab1:
        # 지도 탭
        col_map, col_info = st.columns([3, 1])

        with col_map:
            st.markdown('<div class="map-container">', unsafe_allow_html=True)

            # 지도 옵션
            show_heatmap = st.checkbox("발전량 히트맵 표시", value=False)

            # 지도 생성 및 표시
            jeju_map = create_jeju_map(plants_df, show_heatmap)
            st_folium(jeju_map, width=None, height=500, returned_objects=[])

            st.markdown('</div>', unsafe_allow_html=True)

        with col_info:
            # 발전소 통계
            st.markdown("""
            <div class="info-card">
                <div class="info-card-title">발전소 현황</div>
            </div>
            """, unsafe_allow_html=True)

            solar_plants = plants_df[plants_df['type'] == 'solar']
            wind_plants = plants_df[plants_df['type'] == 'wind']
            ess_plants = plants_df[plants_df['type'] == 'ess']

            st.markdown(f"""
            <div class="info-card" style="padding: 1rem;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                    <span style="width: 30px; height: 30px; background: #fbbf24; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center;">☀️</span>
                    <div>
                        <div style="color: #94a3b8; font-size: 0.8rem;">태양광</div>
                        <div style="color: white; font-weight: bold;">{len(solar_plants)}개소 | {solar_plants['capacity'].sum():.0f} MW</div>
                        <div style="color: #10b981; font-size: 0.85rem;">발전량: {solar_plants['generation'].sum():.1f} MW</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
                    <span style="width: 30px; height: 30px; background: #3b82f6; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center;">💨</span>
                    <div>
                        <div style="color: #94a3b8; font-size: 0.8rem;">풍력</div>
                        <div style="color: white; font-weight: bold;">{len(wind_plants)}개소 | {wind_plants['capacity'].sum():.0f} MW</div>
                        <div style="color: #10b981; font-size: 0.85rem;">발전량: {wind_plants['generation'].sum():.1f} MW</div>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="width: 30px; height: 30px; background: #8b5cf6; border-radius: 50%;
                                display: flex; align-items: center; justify-content: center;">🔋</span>
                    <div>
                        <div style="color: #94a3b8; font-size: 0.8rem;">ESS</div>
                        <div style="color: white; font-weight: bold;">{len(ess_plants)}개소 | {ess_plants['capacity'].sum():.0f} MW</div>
                        <div style="color: #10b981; font-size: 0.85rem;">충방전: {ess_plants['generation'].sum():.1f} MW</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 기상 정보
            st.markdown(f"""
            <div class="info-card" style="padding: 1rem;">
                <div class="info-card-title">기상 정보</div>
                <div style="margin-top: 10px;">
                    <div style="display: flex; justify-content: space-between; margin: 8px 0; color: white;">
                        <span style="color: #94a3b8;">일사량</span>
                        <span>{weather['solar_radiation']:.0f} W/m²</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0; color: white;">
                        <span style="color: #94a3b8;">풍향</span>
                        <span>{weather['wind_direction']}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0; color: white;">
                        <span style="color: #94a3b8;">운량</span>
                        <span>{weather['cloud_cover']:.0f}%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 8px 0; color: white;">
                        <span style="color: #94a3b8;">습도</span>
                        <span>{weather['humidity']:.0f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        # SMP 예측 탭
        col_chart, col_summary = st.columns([2, 1])

        with col_chart:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 📈 24시간 SMP 예측")
            fig = create_smp_chart(smp_data)
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_summary:
            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">SMP 통계</div>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin: 12px 0; color: white;">
                        <span style="color: #94a3b8;">현재가</span>
                        <span style="font-weight: bold; color: #3b82f6;">{smp_data['current']} 원</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 12px 0; color: white;">
                        <span style="color: #94a3b8;">일평균</span>
                        <span>{smp_data['daily_avg']} 원</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 12px 0; color: white;">
                        <span style="color: #94a3b8;">일최고</span>
                        <span style="color: #ef4444;">{smp_data['daily_max']} 원</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 12px 0; color: white;">
                        <span style="color: #94a3b8;">일최저</span>
                        <span style="color: #10b981;">{smp_data['daily_min']} 원</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 입찰 추천
            optimal_bid = smp_data['current'] * 0.95
            st.markdown(f"""
            <div class="info-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(139, 92, 246, 0.3));">
                <div class="info-card-title">💡 입찰 추천</div>
                <div style="margin-top: 15px; color: white;">
                    <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">
                        {optimal_bid:.1f} 원/kWh
                    </div>
                    <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 5px;">
                        현재가 대비 5% 할인
                    </div>
                    <div style="margin-top: 10px; padding: 10px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                        <div style="font-size: 0.8rem; color: #94a3b8;">예상 낙찰 확률</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #fbbf24;">85%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        # 발전 현황 탭
        col_donut, col_timeline = st.columns([1, 2])

        with col_donut:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### ⚡ 전력 공급 구성")
            fig = create_supply_donut(power_status)
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_timeline:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown("### 📊 시간대별 발전량")
            fig = create_generation_timeline(plants_df)
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

        # 발전소 상세 테이블
        st.markdown("### 📋 발전소 상세 현황")

        # 필터
        col_filter1, col_filter2, _ = st.columns([1, 1, 2])
        with col_filter1:
            type_filter = st.selectbox("발전 유형", ["전체", "태양광", "풍력", "ESS"])
        with col_filter2:
            status_filter = st.selectbox("상태", ["전체", "운영중", "점검중", "건설중"])

        # 필터 적용
        filtered_df = plants_df.copy()
        type_map = {"태양광": "solar", "풍력": "wind", "ESS": "ess"}
        if type_filter != "전체":
            filtered_df = filtered_df[filtered_df['type'] == type_map[type_filter]]
        if status_filter != "전체":
            filtered_df = filtered_df[filtered_df['status'] == status_filter]

        # 테이블 표시
        display_df = filtered_df[['name', 'type', 'capacity', 'generation', 'utilization', 'status']].copy()
        display_df.columns = ['발전소명', '유형', '설비용량(MW)', '발전량(MW)', '이용률(%)', '상태']
        display_df['유형'] = display_df['유형'].map({'solar': '☀️ 태양광', 'wind': '💨 풍력', 'ess': '🔋 ESS'})
        display_df['발전량(MW)'] = display_df['발전량(MW)'].round(1)
        display_df['이용률(%)'] = display_df['이용률(%)'].round(1)

        st.dataframe(display_df, width="stretch", hide_index=True)

    with tab4:
        # 분석 탭
        st.markdown("### 🔍 AI 분석 및 인사이트")

        col_a1, col_a2 = st.columns(2)

        with col_a1:
            st.markdown("""
            <div class="info-card">
                <div class="info-card-title">🧠 XAI 분석 요약</div>
                <div style="margin-top: 15px; color: white;">
                    <p><strong>모델 예측 근거:</strong></p>
                    <ul style="color: #94a3b8; margin: 10px 0;">
                        <li>기온 상승 (+2°C) → 수요 증가 영향 15%</li>
                        <li>일사량 감소 → 태양광 발전 감소 예상</li>
                        <li>풍속 증가 → 풍력 발전 증가 기대</li>
                        <li>과거 동일 시간대 패턴 반영 40%</li>
                    </ul>
                    <p style="font-size: 0.85rem; color: #94a3b8;">
                        * Attention 가중치 기반 분석 결과
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-card">
                <div class="info-card-title">📊 모델 성능</div>
                <div style="margin-top: 15px;">
                    <div style="display: flex; justify-content: space-between; margin: 10px 0; color: white;">
                        <span style="color: #94a3b8;">MAPE</span>
                        <span style="color: #10b981; font-weight: bold;">10.68%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0; color: white;">
                        <span style="color: #94a3b8;">MAE</span>
                        <span>11.27 원/kWh</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0; color: white;">
                        <span style="color: #94a3b8;">80% Coverage</span>
                        <span style="color: #10b981;">82.5%</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; margin: 10px 0; color: white;">
                        <span style="color: #94a3b8;">R² Score</span>
                        <span>0.59</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_a2:
            st.markdown("""
            <div class="info-card">
                <div class="info-card-title">⚠️ 리스크 알림</div>
                <div style="margin-top: 15px;">
                    <div style="background: rgba(245, 158, 11, 0.2); border-left: 3px solid #f59e0b;
                                padding: 10px; margin: 10px 0; border-radius: 0 8px 8px 0;">
                        <div style="color: #f59e0b; font-weight: bold;">덕커브 주의</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">
                            14:00-16:00 태양광 발전 급증으로 SMP 하락 예상
                        </div>
                    </div>
                    <div style="background: rgba(59, 130, 246, 0.2); border-left: 3px solid #3b82f6;
                                padding: 10px; margin: 10px 0; border-radius: 0 8px 8px 0;">
                        <div style="color: #3b82f6; font-weight: bold;">풍력 발전 증가</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">
                            풍속 증가로 풍력 발전량 20% 상승 예상
                        </div>
                    </div>
                    <div style="background: rgba(16, 185, 129, 0.2); border-left: 3px solid #10b981;
                                padding: 10px; margin: 10px 0; border-radius: 0 8px 8px 0;">
                        <div style="color: #10b981; font-weight: bold;">입찰 기회</div>
                        <div style="color: #94a3b8; font-size: 0.85rem;">
                            18:00-20:00 피크 시간대 고가 입찰 권장
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="info-card">
                <div class="info-card-title">📈 일간 요약</div>
                <div style="margin-top: 15px; color: white;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                        <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                            <div style="color: #94a3b8; font-size: 0.8rem;">총 발전량</div>
                            <div style="font-size: 1.3rem; font-weight: bold; color: #10b981;">12,450 MWh</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                            <div style="color: #94a3b8; font-size: 0.8rem;">CO₂ 절감</div>
                            <div style="font-size: 1.3rem; font-weight: bold; color: #3b82f6;">5,890 톤</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                            <div style="color: #94a3b8; font-size: 0.8rem;">평균 이용률</div>
                            <div style="font-size: 1.3rem; font-weight: bold; color: #fbbf24;">42.3%</div>
                        </div>
                        <div style="text-align: center; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 8px;">
                            <div style="color: #94a3b8; font-size: 0.8rem;">예상 수익</div>
                            <div style="font-size: 1.3rem; font-weight: bold; color: #8b5cf6;">₩1.2B</div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ========== 푸터 ==========
    st.markdown("""
    <div style="text-align: center; padding: 2rem; color: #64748b; font-size: 0.85rem;">
        <p>제주 전력 지도 v4.0 | Powered by AI | © 2025 Power Demand Forecast Team</p>
        <p style="font-size: 0.75rem;">
            데이터 출처: EPSIS, 기상청 AMOS | 모델: LSTM + Quantile Regression
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
