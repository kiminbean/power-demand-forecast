#!/usr/bin/env python
"""
API Client Example
==================

Power Demand Forecast API 사용 예시

Usage:
------
# 먼저 API 서버 시작
python run_api.py

# 다른 터미널에서 예시 실행
python examples/api_client_example.py
"""

import json
from datetime import datetime, timedelta
from typing import List, Dict

import requests


# API 서버 URL
API_BASE_URL = "http://localhost:8000"


def generate_sample_data(hours: int = 168) -> List[Dict]:
    """샘플 시계열 데이터 생성"""
    base_time = datetime.now() - timedelta(hours=hours)

    data = []
    for i in range(hours):
        hour = (base_time + timedelta(hours=i)).hour
        # 시간대별 전력 수요 패턴 시뮬레이션
        base_demand = 700 + 100 * (1 if 9 <= hour <= 18 else 0)
        demand = base_demand + (i % 24) * 5

        data.append({
            "datetime": (base_time + timedelta(hours=i)).isoformat(),
            "power_demand": demand,
            "기온": 5.0 + (hour - 12) * 0.5,  # 기온 패턴
            "습도": 60.0 + hour % 10,
            "풍속": 3.0,
            "강수량": 0.0
        })

    return data


def check_health():
    """헬스체크"""
    print("=" * 50)
    print("Health Check")
    print("=" * 50)

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def get_models():
    """모델 정보 조회"""
    print("=" * 50)
    print("Model Information")
    print("=" * 50)

    response = requests.get(f"{API_BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print()


def predict_single():
    """단일 예측"""
    print("=" * 50)
    print("Single Prediction (demand_only)")
    print("=" * 50)

    data = generate_sample_data(168)

    response = requests.post(
        f"{API_BASE_URL}/predict",
        json={
            "data": data,
            "model_type": "demand_only"
        }
    )

    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("success"):
        print(f"\n>>> 예측 전력 수요: {result['prediction']:.2f} MW")
    print()


def predict_conditional():
    """조건부 예측"""
    print("=" * 50)
    print("Conditional Prediction (soft mode)")
    print("=" * 50)

    data = generate_sample_data(168)

    response = requests.post(
        f"{API_BASE_URL}/predict/conditional",
        json={
            "data": data,
            "mode": "soft"
        }
    )

    print(f"Status Code: {response.status_code}")
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("success"):
        print(f"\n>>> 예측 전력 수요: {result['prediction']:.2f} MW")
        print(f">>> 사용 모델: {result['model_used']}")
        print(f">>> 겨울철 여부: {result['context']['is_winter']}")
    print()


def predict_batch():
    """배치 예측"""
    print("=" * 50)
    print("Batch Prediction")
    print("=" * 50)

    # 더 많은 데이터로 배치 예측
    data = generate_sample_data(200)

    response = requests.post(
        f"{API_BASE_URL}/predict/batch",
        json={
            "data": data,
            "model_type": "demand_only",
            "step": 6  # 6시간 간격
        }
    )

    print(f"Status Code: {response.status_code}")
    result = response.json()

    if result.get("success"):
        print(f"\n총 예측 수: {result['total_predictions']}")
        print(f"통계:")
        print(f"  - 평균: {result['statistics']['mean']:.2f} MW")
        print(f"  - 표준편차: {result['statistics']['std']:.2f} MW")
        print(f"  - 최소: {result['statistics']['min']:.2f} MW")
        print(f"  - 최대: {result['statistics']['max']:.2f} MW")
        print(f"\n처리 시간: {result['processing_time_ms']:.2f} ms")
        print(f"\n첫 5개 예측:")
        for pred in result['predictions'][:5]:
            print(f"  {pred['timestamp']}: {pred['prediction']:.2f} MW")
    print()


def main():
    """메인 함수"""
    print("\n" + "=" * 50)
    print("Power Demand Forecast API Client Example")
    print("=" * 50 + "\n")

    try:
        # 1. 헬스체크
        check_health()

        # 2. 모델 정보
        get_models()

        # 3. 단일 예측
        predict_single()

        # 4. 조건부 예측
        predict_conditional()

        # 5. 배치 예측
        predict_batch()

        print("=" * 50)
        print("All examples completed successfully!")
        print("=" * 50)

    except requests.ConnectionError:
        print("ERROR: Cannot connect to API server.")
        print("Please start the API server first:")
        print("  python run_api.py")
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
