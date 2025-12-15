# Power Demand Forecast API

제주도 전력 수요 예측 REST API 서비스

## 시작하기

### 로컬 실행

```bash
# 개발 모드 (hot reload)
python run_api.py --dev

# 프로덕션 모드
python run_api.py

# 커스텀 포트
python run_api.py --port 8080
```

### Docker 실행

```bash
# 프로덕션 빌드 및 실행
docker-compose up -d api

# 개발 모드 (hot reload)
docker-compose --profile dev up api-dev

# 로그 확인
docker-compose logs -f api
```

## API 엔드포인트

### 상태 확인

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API 정보 |
| GET | `/health` | 서비스 상태 확인 |
| GET | `/models` | 로드된 모델 정보 |
| GET | `/docs` | Swagger UI 문서 |
| GET | `/redoc` | ReDoc 문서 |

### 예측

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | 단일 예측 |
| POST | `/predict/conditional` | 조건부 예측 (겨울철 최적화) |
| POST | `/predict/batch` | 배치 예측 |

## 사용 예시

### 헬스체크

```bash
curl http://localhost:8000/health
```

```json
{
    "status": "healthy",
    "version": "1.0.0",
    "models_loaded": true,
    "device": "mps",
    "uptime_seconds": 120.5
}
```

### 단일 예측

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [...],  // 168개 이상의 시계열 데이터
    "model_type": "demand_only"
  }'
```

```json
{
    "success": true,
    "prediction": 875.32,
    "model_used": "demand_only",
    "timestamp": "2024-01-15T15:00:00",
    "processing_time_ms": 45.2
}
```

### 조건부 예측

```bash
curl -X POST http://localhost:8000/predict/conditional \
  -H "Content-Type: application/json" \
  -d '{
    "data": [...],
    "mode": "soft"
  }'
```

```json
{
    "success": true,
    "prediction": 892.15,
    "model_used": "conditional_soft (w=0.3)",
    "timestamp": "2024-01-15T15:00:00",
    "context": {
        "is_winter": true,
        "month": 1,
        "weather_weight": 0.3,
        "pred_demand_only": 885.0
    },
    "processing_time_ms": 78.5
}
```

### 배치 예측

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "data": [...],
    "model_type": "demand_only",
    "step": 1
  }'
```

```json
{
    "success": true,
    "predictions": [
        {"timestamp": "2024-01-15T00:00:00", "prediction": 720.5},
        {"timestamp": "2024-01-15T01:00:00", "prediction": 715.2}
    ],
    "model_used": "demand_only",
    "total_predictions": 24,
    "statistics": {
        "mean": 825.5,
        "std": 45.2,
        "min": 710.0,
        "max": 920.0
    },
    "processing_time_ms": 1250.0
}
```

## 요청 데이터 형식

```json
{
    "datetime": "2024-01-15T14:00:00",
    "power_demand": 850.5,
    "기온": 5.2,
    "습도": 65.0,
    "풍속": 3.5,
    "강수량": 0.0
}
```

- `datetime`: ISO 8601 형식
- `power_demand`: 전력 수요 (MW) - 필수
- `기온`, `습도`, `풍속`, `강수량`: 기상 데이터 - 선택

## 모델 타입

| Type | Description |
|------|-------------|
| `demand_only` | 전력 수요 데이터만 사용 |
| `weather_full` | 기상 데이터 포함 |
| `conditional` | 겨울철 자동 최적화 (권장) |

## 환경 설정

`.env.example`을 `.env`로 복사하고 필요에 따라 수정:

```bash
cp .env.example .env
```

주요 설정:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | 0.0.0.0 | 서버 호스트 |
| `PORT` | 8000 | 서버 포트 |
| `DEBUG` | false | 디버그 모드 |
| `LOG_LEVEL` | INFO | 로그 레벨 |
| `MODEL_DIR` | ./models/production | 모델 디렉토리 |
| `CORS_ORIGINS` | * | CORS 허용 오리진 |

## 테스트

```bash
# API 테스트 실행
python -m pytest tests/test_api.py -v

# 전체 테스트
python -m pytest tests/ -v
```

## 모니터링

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health
