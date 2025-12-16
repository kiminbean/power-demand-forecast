"""
부하 테스트 (Task 21)
=====================

Locust 기반 API 부하 테스트

사용법:
    locust -f tests/load_testing.py --host=http://localhost:8000
    locust -f tests/load_testing.py --host=http://localhost:8000 --headless -u 100 -r 10 -t 1m

설정:
    -u: 총 사용자 수
    -r: 초당 사용자 생성률
    -t: 테스트 시간

Author: Claude Code
Date: 2025-12
"""

import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

try:
    from locust import HttpUser, task, between, events
    from locust.runners import MasterRunner
    LOCUST_AVAILABLE = True
except ImportError:
    LOCUST_AVAILABLE = False
    HttpUser = object


# ============================================================================
# 테스트 데이터 생성
# ============================================================================

def generate_prediction_request() -> Dict[str, Any]:
    """예측 요청 생성"""
    horizons_options = [
        ["1h"],
        ["1h", "6h"],
        ["1h", "6h", "24h"],
        ["1h", "6h", "12h", "24h", "48h"]
    ]

    return {
        "location": "jeju",
        "horizons": random.choice(horizons_options),
        "model_type": random.choice(["lstm", "tft", "ensemble"]),
        "include_confidence": random.choice([True, False]),
        "features": {
            "temperature": round(random.uniform(10, 35), 1),
            "humidity": round(random.uniform(40, 90), 1)
        } if random.random() > 0.5 else None
    }


def generate_historical_request() -> Dict[str, Any]:
    """과거 데이터 요청 생성"""
    end_date = datetime.now() - timedelta(days=random.randint(1, 30))
    start_date = end_date - timedelta(days=random.randint(1, 7))

    return {
        "location": "jeju",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "resolution": random.choice(["hourly", "daily"])
    }


def generate_forecast_request() -> Dict[str, Any]:
    """예보 요청 생성"""
    return {
        "location": "jeju",
        "hours_ahead": random.choice([6, 12, 24, 48]),
        "include_weather": True
    }


# ============================================================================
# 부하 테스트 사용자 클래스
# ============================================================================

if LOCUST_AVAILABLE:
    class PowerDemandAPIUser(HttpUser):
        """
        전력 수요 예측 API 사용자 시뮬레이션

        다양한 엔드포인트에 대한 현실적인 부하를 생성합니다.
        """
        # 요청 간 대기 시간 (초)
        wait_time = between(1, 3)

        def on_start(self):
            """사용자 시작 시 호출"""
            # 헬스 체크로 시작
            self.client.get("/health")

        @task(10)
        def predict(self):
            """
            예측 요청 (가장 빈번한 요청)

            가중치 10 - 다른 태스크보다 10배 자주 실행
            """
            request_data = generate_prediction_request()

            with self.client.post(
                "/predict",
                json=request_data,
                catch_response=True,
                name="/predict"
            ) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if "predictions" in data:
                            response.success()
                        else:
                            response.failure("Invalid response structure")
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON response")
                elif response.status_code == 422:
                    # Validation error - expected for some inputs
                    response.success()
                else:
                    response.failure(f"Status {response.status_code}")

        @task(3)
        def get_historical_data(self):
            """
            과거 데이터 조회

            가중치 3
            """
            request_data = generate_historical_request()

            with self.client.post(
                "/data/historical",
                json=request_data,
                catch_response=True,
                name="/data/historical"
            ) as response:
                if response.status_code in [200, 404]:
                    response.success()
                else:
                    response.failure(f"Status {response.status_code}")

        @task(2)
        def get_forecast(self):
            """
            예보 조회

            가중치 2
            """
            request_data = generate_forecast_request()

            with self.client.post(
                "/forecast",
                json=request_data,
                catch_response=True,
                name="/forecast"
            ) as response:
                if response.status_code in [200, 404]:
                    response.success()
                else:
                    response.failure(f"Status {response.status_code}")

        @task(5)
        def health_check(self):
            """
            상태 확인

            가중치 5 - 모니터링 시스템에서 주기적으로 호출
            """
            with self.client.get("/health", catch_response=True) as response:
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data.get("status") == "healthy":
                            response.success()
                        else:
                            response.failure(f"Unhealthy status: {data.get('status')}")
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON")
                else:
                    response.failure(f"Status {response.status_code}")

        @task(1)
        def list_models(self):
            """
            모델 목록 조회

            가중치 1 - 드물게 호출
            """
            with self.client.get("/models", catch_response=True, name="/models") as response:
                if response.status_code in [200, 404]:
                    response.success()
                else:
                    response.failure(f"Status {response.status_code}")

        @task(1)
        def root_endpoint(self):
            """
            루트 엔드포인트

            가중치 1
            """
            self.client.get("/")


    class HeavyUser(HttpUser):
        """
        고부하 사용자 시뮬레이션

        빈번한 예측 요청을 생성하는 사용자
        """
        wait_time = between(0.1, 0.5)

        @task
        def rapid_predictions(self):
            """빠른 연속 예측 요청"""
            request_data = generate_prediction_request()
            self.client.post("/predict", json=request_data)


    class LightUser(HttpUser):
        """
        저부하 사용자 시뮬레이션

        간헐적으로 요청하는 사용자
        """
        wait_time = between(5, 15)

        @task(3)
        def occasional_prediction(self):
            """간헐적 예측 요청"""
            request_data = generate_prediction_request()
            self.client.post("/predict", json=request_data)

        @task(1)
        def check_health(self):
            """상태 확인"""
            self.client.get("/health")


    # ============================================================================
    # 이벤트 핸들러
    # ============================================================================

    @events.test_start.add_listener
    def on_test_start(environment, **kwargs):
        """테스트 시작 이벤트"""
        print("\n" + "=" * 60)
        print("Power Demand Forecast API Load Test Started")
        print(f"Target Host: {environment.host}")
        print("=" * 60 + "\n")


    @events.test_stop.add_listener
    def on_test_stop(environment, **kwargs):
        """테스트 종료 이벤트"""
        print("\n" + "=" * 60)
        print("Load Test Completed")
        print("=" * 60 + "\n")


# ============================================================================
# 부하 테스트 결과 분석
# ============================================================================

class LoadTestAnalyzer:
    """
    부하 테스트 결과 분석기

    Locust 결과 CSV를 분석합니다.
    """

    def __init__(self, stats_file: str = None, history_file: str = None):
        self.stats_file = stats_file
        self.history_file = history_file

    def analyze_stats(self) -> Dict[str, Any]:
        """통계 분석"""
        import pandas as pd

        if not self.stats_file:
            return {}

        df = pd.read_csv(self.stats_file)

        analysis = {
            "summary": {
                "total_requests": int(df["Request Count"].sum()),
                "total_failures": int(df["Failure Count"].sum()),
                "failure_rate": float(df["Failure Count"].sum() / df["Request Count"].sum() * 100)
            },
            "response_times": {
                "avg_response_time_ms": float(df["Average Response Time"].mean()),
                "max_response_time_ms": float(df["Max Response Time"].max()),
                "min_response_time_ms": float(df["Min Response Time"].min()),
                "p50_ms": float(df["50%"].mean()),
                "p90_ms": float(df["90%"].mean()),
                "p99_ms": float(df["99%"].mean()) if "99%" in df.columns else None
            },
            "throughput": {
                "requests_per_second": float(df["Requests/s"].sum())
            },
            "by_endpoint": {}
        }

        # 엔드포인트별 분석
        for _, row in df.iterrows():
            if row["Name"] != "Aggregated":
                analysis["by_endpoint"][row["Name"]] = {
                    "requests": int(row["Request Count"]),
                    "failures": int(row["Failure Count"]),
                    "avg_response_time_ms": float(row["Average Response Time"]),
                    "p90_ms": float(row["90%"])
                }

        return analysis

    def generate_report(self) -> str:
        """리포트 생성"""
        analysis = self.analyze_stats()

        if not analysis:
            return "No data available"

        report = """
# Load Test Report

## Summary
- Total Requests: {total_requests:,}
- Total Failures: {total_failures:,}
- Failure Rate: {failure_rate:.2f}%

## Response Times
- Average: {avg_response_time_ms:.2f} ms
- Max: {max_response_time_ms:.2f} ms
- P50: {p50_ms:.2f} ms
- P90: {p90_ms:.2f} ms

## Throughput
- Requests/second: {requests_per_second:.2f}

## By Endpoint
""".format(
            **analysis["summary"],
            **analysis["response_times"],
            **analysis["throughput"]
        )

        for endpoint, stats in analysis.get("by_endpoint", {}).items():
            report += f"""
### {endpoint}
- Requests: {stats['requests']:,}
- Failures: {stats['failures']:,}
- Avg Response Time: {stats['avg_response_time_ms']:.2f} ms
- P90: {stats['p90_ms']:.2f} ms
"""

        return report


# ============================================================================
# 성능 기준 검증
# ============================================================================

class PerformanceCriteria:
    """성능 기준 검증"""

    # 기본 성능 기준
    MAX_RESPONSE_TIME_MS = 500  # 최대 응답 시간
    MAX_P90_MS = 200            # P90 응답 시간
    MAX_FAILURE_RATE = 1.0      # 최대 실패율 (%)
    MIN_RPS = 50                # 최소 초당 요청 수

    @classmethod
    def validate(cls, analysis: Dict[str, Any]) -> Dict[str, bool]:
        """성능 기준 검증"""
        results = {}

        # 응답 시간 검증
        avg_response = analysis.get("response_times", {}).get("avg_response_time_ms", 0)
        results["avg_response_time"] = avg_response < cls.MAX_RESPONSE_TIME_MS

        p90 = analysis.get("response_times", {}).get("p90_ms", 0)
        results["p90_response_time"] = p90 < cls.MAX_P90_MS

        # 실패율 검증
        failure_rate = analysis.get("summary", {}).get("failure_rate", 100)
        results["failure_rate"] = failure_rate < cls.MAX_FAILURE_RATE

        # 처리량 검증
        rps = analysis.get("throughput", {}).get("requests_per_second", 0)
        results["throughput"] = rps >= cls.MIN_RPS

        return results

    @classmethod
    def get_summary(cls, validation_results: Dict[str, bool]) -> str:
        """검증 요약"""
        passed = sum(validation_results.values())
        total = len(validation_results)

        summary = f"\nPerformance Validation: {passed}/{total} criteria passed\n"

        for criterion, passed in validation_results.items():
            status = "✓" if passed else "✗"
            summary += f"  {status} {criterion}\n"

        return summary


# ============================================================================
# 유틸리티
# ============================================================================

def run_quick_test(host: str = "http://localhost:8000", users: int = 10, duration: str = "30s"):
    """빠른 부하 테스트 실행"""
    import subprocess

    cmd = [
        "locust",
        "-f", __file__,
        f"--host={host}",
        "--headless",
        f"-u", str(users),
        "-r", str(min(users, 5)),
        "-t", duration,
        "--csv=load_test_results"
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    if LOCUST_AVAILABLE:
        print("Run with: locust -f tests/load_testing.py --host=http://localhost:8000")
    else:
        print("Locust not installed. Install with: pip install locust")
