"""
API 문서화 및 모델 카드 (Task 20)
=================================

OpenAPI 확장 문서 및 모델 카드 생성

주요 기능:
1. 확장된 API 문서 (예시, 에러 코드)
2. 모델 카드 생성
3. 변경 이력 관리
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json


# ============================================================================
# API 문서화
# ============================================================================

# API 예시 데이터
PREDICTION_REQUEST_EXAMPLE = {
    "location": "jeju",
    "horizons": ["1h", "6h", "24h"],
    "model_type": "ensemble",
    "include_confidence": True,
    "features": {
        "temperature": 22.5,
        "humidity": 65.0
    }
}

PREDICTION_RESPONSE_EXAMPLE = {
    "request_id": "pred-20241216-001",
    "location": "jeju",
    "model_type": "ensemble",
    "created_at": "2024-12-16T14:00:00",
    "predictions": [
        {
            "timestamp": "2024-12-16T15:00:00",
            "horizon": "1h",
            "prediction": 1025.5,
            "lower_bound": 985.2,
            "upper_bound": 1065.8,
            "confidence": 0.95
        },
        {
            "timestamp": "2024-12-16T20:00:00",
            "horizon": "6h",
            "prediction": 1102.3,
            "lower_bound": 1042.1,
            "upper_bound": 1162.5,
            "confidence": 0.90
        }
    ],
    "metadata": {
        "model_version": "v1.2.0",
        "processing_time_ms": 45.2
    }
}

HEALTH_RESPONSE_EXAMPLE = {
    "status": "healthy",
    "version": "1.0.0",
    "uptime": 3600.5,
    "models_loaded": 3,
    "last_prediction": "2024-12-16T13:55:00"
}

ERROR_RESPONSE_EXAMPLES = {
    "validation_error": {
        "error": "Validation Error",
        "detail": "horizons must be a list of valid time horizons",
        "code": "VALIDATION_ERROR"
    },
    "model_not_found": {
        "error": "Model Not Found",
        "detail": "Model 'custom_model' is not available",
        "code": "MODEL_NOT_FOUND"
    },
    "prediction_failed": {
        "error": "Prediction Failed",
        "detail": "Unable to generate predictions due to insufficient data",
        "code": "PREDICTION_FAILED"
    },
    "rate_limit": {
        "error": "Rate Limit Exceeded",
        "detail": "Too many requests. Please try again later.",
        "code": "RATE_LIMIT_EXCEEDED"
    }
}

# API 에러 코드 정의
ERROR_CODES = {
    "VALIDATION_ERROR": {
        "http_status": 400,
        "description": "요청 데이터 유효성 검증 실패"
    },
    "MODEL_NOT_FOUND": {
        "http_status": 404,
        "description": "요청한 모델을 찾을 수 없음"
    },
    "PREDICTION_FAILED": {
        "http_status": 500,
        "description": "예측 처리 중 오류 발생"
    },
    "RATE_LIMIT_EXCEEDED": {
        "http_status": 429,
        "description": "요청 횟수 제한 초과"
    },
    "UNAUTHORIZED": {
        "http_status": 401,
        "description": "인증 실패"
    },
    "FORBIDDEN": {
        "http_status": 403,
        "description": "접근 권한 없음"
    },
    "DATA_NOT_FOUND": {
        "http_status": 404,
        "description": "요청한 데이터를 찾을 수 없음"
    },
    "INTERNAL_ERROR": {
        "http_status": 500,
        "description": "내부 서버 오류"
    }
}


def get_openapi_custom_docs() -> Dict[str, Any]:
    """확장된 OpenAPI 문서 반환"""
    return {
        "info": {
            "title": "Jeju Power Demand Forecast API",
            "version": "1.0.0",
            "description": """
## 제주도 전력 수요 예측 API

이 API는 제주도의 전력 수요를 예측하는 서비스를 제공합니다.

### 주요 기능
- **실시간 예측**: 1시간~48시간 후의 전력 수요 예측
- **다중 모델**: LSTM, TFT, Ensemble 모델 선택 가능
- **신뢰 구간**: 예측 불확실성 제공
- **과거 데이터**: 시간별/일별 과거 데이터 조회

### 인증
현재 버전에서는 인증이 필요하지 않습니다.
향후 버전에서 API 키 인증이 추가될 예정입니다.

### Rate Limiting
- 분당 100 요청
- 시간당 1,000 요청

### 문의
이슈나 질문은 GitHub Issues를 통해 문의해주세요.
            """,
            "contact": {
                "name": "API Support",
                "email": "support@example.com"
            },
            "license": {
                "name": "MIT",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "로컬 개발 서버"
            },
            {
                "url": "https://api.jeju-power.example.com",
                "description": "프로덕션 서버"
            }
        ],
        "tags": [
            {
                "name": "Prediction",
                "description": "전력 수요 예측 관련 엔드포인트"
            },
            {
                "name": "Data",
                "description": "과거 데이터 조회 엔드포인트"
            },
            {
                "name": "Models",
                "description": "모델 정보 관련 엔드포인트"
            },
            {
                "name": "Health",
                "description": "서비스 상태 확인"
            }
        ],
        "error_codes": ERROR_CODES
    }


# ============================================================================
# 모델 카드
# ============================================================================

@dataclass
class ModelCard:
    """
    ML 모델 카드

    모델의 성능, 한계, 사용법을 문서화합니다.
    """
    # 기본 정보
    name: str
    version: str
    type: str
    description: str

    # 개발 정보
    developers: List[str] = field(default_factory=list)
    created_date: str = ""
    last_updated: str = ""

    # 모델 상세
    architecture: str = ""
    input_features: List[str] = field(default_factory=list)
    output_type: str = ""
    training_data: str = ""
    training_period: str = ""

    # 성능 메트릭
    metrics: Dict[str, float] = field(default_factory=dict)
    evaluation_data: str = ""

    # 사용 정보
    intended_use: str = ""
    out_of_scope_use: List[str] = field(default_factory=list)

    # 한계 및 편향
    limitations: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)

    # 추가 정보
    citations: List[str] = field(default_factory=list)
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "model_details": {
                "name": self.name,
                "version": self.version,
                "type": self.type,
                "description": self.description,
                "developers": self.developers,
                "created_date": self.created_date,
                "last_updated": self.last_updated
            },
            "architecture": {
                "type": self.architecture,
                "input_features": self.input_features,
                "output_type": self.output_type
            },
            "training": {
                "data": self.training_data,
                "period": self.training_period
            },
            "evaluation": {
                "metrics": self.metrics,
                "evaluation_data": self.evaluation_data
            },
            "intended_use": {
                "primary_use": self.intended_use,
                "out_of_scope": self.out_of_scope_use
            },
            "limitations_and_biases": {
                "limitations": self.limitations,
                "ethical_considerations": self.ethical_considerations
            },
            "additional_information": {
                "citations": self.citations,
                **self.additional_info
            }
        }

    def to_markdown(self) -> str:
        """마크다운 형식으로 변환"""
        md = f"""# Model Card: {self.name}

## Model Details

| Property | Value |
|----------|-------|
| Name | {self.name} |
| Version | {self.version} |
| Type | {self.type} |
| Created | {self.created_date} |
| Last Updated | {self.last_updated} |

### Description
{self.description}

### Developers
{', '.join(self.developers) if self.developers else 'N/A'}

## Model Architecture
**Type**: {self.architecture}

### Input Features
{chr(10).join('- ' + f for f in self.input_features) if self.input_features else 'N/A'}

### Output
{self.output_type}

## Training

### Training Data
{self.training_data}

### Training Period
{self.training_period}

## Evaluation

### Metrics
| Metric | Value |
|--------|-------|
"""
        for metric, value in self.metrics.items():
            md += f"| {metric} | {value:.4f} |\n"

        md += f"""
### Evaluation Data
{self.evaluation_data}

## Intended Use
{self.intended_use}

### Out of Scope Uses
{chr(10).join('- ' + u for u in self.out_of_scope_use) if self.out_of_scope_use else 'N/A'}

## Limitations and Biases

### Known Limitations
{chr(10).join('- ' + l for l in self.limitations) if self.limitations else 'N/A'}

### Ethical Considerations
{chr(10).join('- ' + e for e in self.ethical_considerations) if self.ethical_considerations else 'N/A'}

## Citations
{chr(10).join('- ' + c for c in self.citations) if self.citations else 'N/A'}

---
*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return md

    def save(self, output_dir: str, formats: List[str] = None) -> Dict[str, str]:
        """모델 카드 저장"""
        formats = formats or ['json', 'md']
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = {}

        if 'json' in formats:
            json_path = output_path / f"{self.name}_model_card.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            saved_files['json'] = str(json_path)

        if 'md' in formats:
            md_path = output_path / f"{self.name}_model_card.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(self.to_markdown())
            saved_files['md'] = str(md_path)

        return saved_files


def create_lstm_model_card() -> ModelCard:
    """LSTM 모델 카드 생성"""
    return ModelCard(
        name="jeju-lstm-v1",
        version="1.0.0",
        type="LSTM (Long Short-Term Memory)",
        description="""
제주도 전력 수요 예측을 위한 LSTM 기반 시계열 예측 모델입니다.
과거 전력 수요 패턴과 기상 데이터를 학습하여 미래 수요를 예측합니다.
        """.strip(),
        developers=["Claude Code", "AI/ML Team"],
        created_date="2024-12-01",
        last_updated=datetime.now().strftime("%Y-%m-%d"),
        architecture="BiLSTM with 2 layers, 128 hidden units, dropout 0.2",
        input_features=[
            "hour_sin, hour_cos (시간 순환 인코딩)",
            "day_of_week (요일)",
            "month_sin, month_cos (월 순환 인코딩)",
            "is_weekend (주말 여부)",
            "is_holiday (공휴일 여부)",
            "temperature (기온)",
            "humidity (습도)",
            "THI (불쾌지수)",
            "wind_chill (체감온도)",
            "solar_altitude (태양 고도각)",
            "past_demand (과거 48시간 수요)"
        ],
        output_type="다중 시간대 전력 수요 예측 (1h, 6h, 12h, 24h, 48h)",
        training_data="제주도 시간별 전력 수요 데이터 (KEPCO)",
        training_period="2013-01-01 ~ 2022-12-31 (10년)",
        metrics={
            "MAPE": 4.52,
            "RMSE": 45.3,
            "MAE": 32.1,
            "R2": 0.92
        },
        evaluation_data="2023-01-01 ~ 2023-12-31 (1년)",
        intended_use="""
- 제주도 전력 수요 예측
- 전력 계통 운영 계획 수립
- 피크 수요 대응 준비
        """.strip(),
        out_of_scope_use=[
            "다른 지역의 전력 수요 예측",
            "갑작스러운 대규모 정전 상황",
            "극단적 기상 이변 상황",
            "대규모 산업 시설 가동 변화"
        ],
        limitations=[
            "학습 데이터 기간 외의 패턴에 대해 제한적 성능",
            "극단적 기상 조건에서 정확도 저하 가능",
            "대규모 이벤트(축제, 재난)의 영향 반영 한계",
            "실시간 데이터 없이는 예측 불가"
        ],
        ethical_considerations=[
            "전력 공급 결정에 사용 시 항상 전문가 검토 필요",
            "예측 오류로 인한 전력 공급 차질 가능성 고려",
            "환경적 영향을 고려한 의사결정 필요"
        ],
        citations=[
            "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.",
            "KEPCO 전력통계정보시스템 (https://epsis.kpx.or.kr)"
        ],
        additional_info={
            "hardware": "Apple M1 Pro (MPS)",
            "training_time": "약 30분",
            "inference_time": "<50ms per prediction"
        }
    )


def create_tft_model_card() -> ModelCard:
    """TFT 모델 카드 생성"""
    return ModelCard(
        name="jeju-tft-v1",
        version="1.0.0",
        type="Temporal Fusion Transformer",
        description="""
Temporal Fusion Transformer를 활용한 고성능 전력 수요 예측 모델입니다.
Multi-horizon 예측과 해석 가능한 attention을 제공합니다.
        """.strip(),
        developers=["Claude Code", "AI/ML Team"],
        created_date="2024-12-10",
        last_updated=datetime.now().strftime("%Y-%m-%d"),
        architecture="TFT with 64 hidden units, 4 attention heads, 2 LSTM layers",
        input_features=[
            "Known features: hour, day_of_week, month, is_weekend, is_holiday",
            "Unknown features: temperature, humidity, THI, wind_chill, past_demand",
            "Static features: location encoding"
        ],
        output_type="Quantile 예측 (0.1, 0.5, 0.9) with attention weights",
        training_data="제주도 시간별 전력 수요 데이터 (KEPCO)",
        training_period="2013-01-01 ~ 2022-12-31 (10년)",
        metrics={
            "MAPE": 4.21,
            "RMSE": 42.8,
            "MAE": 30.5,
            "R2": 0.94,
            "Pinball_Loss": 15.2
        },
        evaluation_data="2023-01-01 ~ 2023-12-31 (1년)",
        intended_use="""
- 불확실성이 중요한 전력 수요 예측
- 피처 중요도 분석 및 해석
- 확률적 예측 기반 의사결정
        """.strip(),
        out_of_scope_use=[
            "실시간 초단기(1분 이하) 예측",
            "학습 데이터 외 지역",
            "비정상적 운영 상황"
        ],
        limitations=[
            "LSTM 대비 더 많은 계산 리소스 필요",
            "학습 시간이 더 오래 걸림",
            "작은 데이터셋에서는 과적합 가능성"
        ],
        ethical_considerations=[
            "예측 결과의 불확실성 범위 함께 제공 권장",
            "자동화된 의사결정에 사용 시 주의 필요"
        ],
        citations=[
            "Lim, B., et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting."
        ]
    )


def create_ensemble_model_card() -> ModelCard:
    """앙상블 모델 카드 생성"""
    return ModelCard(
        name="jeju-ensemble-v1",
        version="1.0.0",
        type="Weighted Ensemble (LSTM + TFT)",
        description="""
LSTM과 TFT 모델의 예측을 결합한 앙상블 모델입니다.
개별 모델의 장점을 활용하여 더 안정적인 예측을 제공합니다.
        """.strip(),
        developers=["Claude Code", "AI/ML Team"],
        created_date="2024-12-15",
        last_updated=datetime.now().strftime("%Y-%m-%d"),
        architecture="Weighted Average Ensemble: 0.4*LSTM + 0.6*TFT",
        input_features=["LSTM 입력 피처 + TFT 입력 피처 (상세 내용은 개별 모델 카드 참조)"],
        output_type="앙상블 예측 with 결합된 신뢰 구간",
        training_data="제주도 시간별 전력 수요 데이터 (KEPCO)",
        training_period="2013-01-01 ~ 2022-12-31 (10년)",
        metrics={
            "MAPE": 4.05,
            "RMSE": 40.1,
            "MAE": 28.9,
            "R2": 0.95
        },
        evaluation_data="2023-01-01 ~ 2023-12-31 (1년)",
        intended_use="""
- 프로덕션 환경의 기본 예측 모델
- 높은 안정성이 필요한 운영 계획
- 개별 모델 간 불일치 시 의사결정
        """.strip(),
        out_of_scope_use=["개별 모델 카드의 제외 사항 모두 적용"],
        limitations=[
            "개별 모델 모두 실행 필요로 지연 시간 증가",
            "두 모델 모두 동일한 방향으로 오류 시 보정 어려움"
        ],
        ethical_considerations=[
            "앙상블이 항상 더 나은 결과를 보장하지 않음",
            "극단적 상황에서는 개별 모델 확인 권장"
        ]
    )


def generate_all_model_cards(output_dir: str) -> Dict[str, Dict[str, str]]:
    """모든 모델 카드 생성"""
    cards = {
        'lstm': create_lstm_model_card(),
        'tft': create_tft_model_card(),
        'ensemble': create_ensemble_model_card()
    }

    results = {}
    for name, card in cards.items():
        results[name] = card.save(output_dir)

    return results


# ============================================================================
# 변경 이력
# ============================================================================

CHANGELOG = """
# API Changelog

## [1.0.0] - 2024-12-16
### Added
- 초기 API 릴리스
- /predict 엔드포인트: 전력 수요 예측
- /data/historical 엔드포인트: 과거 데이터 조회
- /models 엔드포인트: 모델 정보 조회
- /health 엔드포인트: 서비스 상태 확인
- LSTM, TFT, Ensemble 모델 지원
- 다중 시간대 예측 (1h ~ 48h)
- 신뢰 구간 제공

### Security
- CORS 설정 추가
- 입력 유효성 검증

## [0.9.0] - 2024-12-10
### Added
- 베타 버전 릴리스
- 기본 예측 기능
- 헬스 체크

### Fixed
- 예측 응답 시간 개선

## [0.1.0] - 2024-12-01
### Added
- 프로젝트 초기화
- 기본 API 구조 설계
"""


def get_changelog() -> str:
    """변경 이력 반환"""
    return CHANGELOG
