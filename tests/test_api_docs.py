"""
API 문서화 테스트 (Task 20)
===========================
모델 카드 및 API 문서 테스트
"""

import pytest
import tempfile
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# API 문서화 테스트
# ============================================================================

class TestOpenAPIDocs:
    """OpenAPI 문서 테스트"""

    def test_get_openapi_docs(self):
        """OpenAPI 문서 생성"""
        from src.api.docs import get_openapi_custom_docs

        docs = get_openapi_custom_docs()

        assert 'info' in docs
        assert docs['info']['title'] == 'Jeju Power Demand Forecast API'
        assert 'description' in docs['info']

    def test_docs_has_servers(self):
        """서버 정보 포함"""
        from src.api.docs import get_openapi_custom_docs

        docs = get_openapi_custom_docs()

        assert 'servers' in docs
        assert len(docs['servers']) >= 1

    def test_docs_has_tags(self):
        """태그 정보 포함"""
        from src.api.docs import get_openapi_custom_docs

        docs = get_openapi_custom_docs()

        assert 'tags' in docs
        tag_names = [t['name'] for t in docs['tags']]
        assert 'Prediction' in tag_names
        assert 'Health' in tag_names

    def test_docs_has_error_codes(self):
        """에러 코드 정보 포함"""
        from src.api.docs import get_openapi_custom_docs

        docs = get_openapi_custom_docs()

        assert 'error_codes' in docs
        assert 'VALIDATION_ERROR' in docs['error_codes']
        assert 'MODEL_NOT_FOUND' in docs['error_codes']


class TestExamples:
    """API 예시 테스트"""

    def test_prediction_request_example(self):
        """예측 요청 예시"""
        from src.api.docs import PREDICTION_REQUEST_EXAMPLE

        assert 'location' in PREDICTION_REQUEST_EXAMPLE
        assert 'horizons' in PREDICTION_REQUEST_EXAMPLE
        assert isinstance(PREDICTION_REQUEST_EXAMPLE['horizons'], list)

    def test_prediction_response_example(self):
        """예측 응답 예시"""
        from src.api.docs import PREDICTION_RESPONSE_EXAMPLE

        assert 'request_id' in PREDICTION_RESPONSE_EXAMPLE
        assert 'predictions' in PREDICTION_RESPONSE_EXAMPLE
        assert len(PREDICTION_RESPONSE_EXAMPLE['predictions']) > 0

    def test_error_response_examples(self):
        """에러 응답 예시"""
        from src.api.docs import ERROR_RESPONSE_EXAMPLES

        assert 'validation_error' in ERROR_RESPONSE_EXAMPLES
        assert 'model_not_found' in ERROR_RESPONSE_EXAMPLES

        for name, example in ERROR_RESPONSE_EXAMPLES.items():
            assert 'error' in example
            assert 'code' in example


class TestErrorCodes:
    """에러 코드 테스트"""

    def test_error_codes_structure(self):
        """에러 코드 구조"""
        from src.api.docs import ERROR_CODES

        for code, info in ERROR_CODES.items():
            assert 'http_status' in info
            assert 'description' in info
            assert isinstance(info['http_status'], int)

    def test_common_error_codes_exist(self):
        """일반적인 에러 코드 존재"""
        from src.api.docs import ERROR_CODES

        required_codes = [
            'VALIDATION_ERROR',
            'MODEL_NOT_FOUND',
            'PREDICTION_FAILED',
            'INTERNAL_ERROR'
        ]

        for code in required_codes:
            assert code in ERROR_CODES


# ============================================================================
# 모델 카드 테스트
# ============================================================================

class TestModelCard:
    """모델 카드 테스트"""

    def test_model_card_creation(self):
        """모델 카드 생성"""
        from src.api.docs import ModelCard

        card = ModelCard(
            name="test-model",
            version="1.0.0",
            type="LSTM",
            description="Test model"
        )

        assert card.name == "test-model"
        assert card.version == "1.0.0"
        assert card.type == "LSTM"

    def test_model_card_to_dict(self):
        """모델 카드 딕셔너리 변환"""
        from src.api.docs import ModelCard

        card = ModelCard(
            name="test-model",
            version="1.0.0",
            type="LSTM",
            description="Test model",
            metrics={'MAPE': 4.5, 'RMSE': 40.0}
        )

        result = card.to_dict()

        assert 'model_details' in result
        assert result['model_details']['name'] == 'test-model'
        assert 'evaluation' in result
        assert result['evaluation']['metrics']['MAPE'] == 4.5

    def test_model_card_to_markdown(self):
        """모델 카드 마크다운 변환"""
        from src.api.docs import ModelCard

        card = ModelCard(
            name="test-model",
            version="1.0.0",
            type="LSTM",
            description="Test model description",
            developers=["Developer 1"],
            metrics={'MAPE': 4.5}
        )

        md = card.to_markdown()

        assert "# Model Card: test-model" in md
        assert "Version | 1.0.0" in md
        assert "Test model description" in md
        assert "MAPE | 4.5" in md

    def test_model_card_save(self):
        """모델 카드 저장"""
        from src.api.docs import ModelCard

        card = ModelCard(
            name="test-model",
            version="1.0.0",
            type="LSTM",
            description="Test model"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            saved = card.save(tmpdir)

            assert 'json' in saved
            assert 'md' in saved

            # JSON 파일 확인
            json_path = Path(saved['json'])
            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)
                assert data['model_details']['name'] == 'test-model'

            # Markdown 파일 확인
            md_path = Path(saved['md'])
            assert md_path.exists()


class TestPrebuiltModelCards:
    """사전 정의된 모델 카드 테스트"""

    def test_lstm_model_card(self):
        """LSTM 모델 카드"""
        from src.api.docs import create_lstm_model_card

        card = create_lstm_model_card()

        assert 'lstm' in card.name.lower()
        assert len(card.input_features) > 0
        assert len(card.metrics) > 0
        assert 'MAPE' in card.metrics

    def test_tft_model_card(self):
        """TFT 모델 카드"""
        from src.api.docs import create_tft_model_card

        card = create_tft_model_card()

        assert 'tft' in card.name.lower()
        assert 'Transformer' in card.type
        assert len(card.metrics) > 0

    def test_ensemble_model_card(self):
        """앙상블 모델 카드"""
        from src.api.docs import create_ensemble_model_card

        card = create_ensemble_model_card()

        assert 'ensemble' in card.name.lower()
        assert 'Ensemble' in card.type

    def test_generate_all_model_cards(self):
        """모든 모델 카드 생성"""
        from src.api.docs import generate_all_model_cards

        with tempfile.TemporaryDirectory() as tmpdir:
            results = generate_all_model_cards(tmpdir)

            assert 'lstm' in results
            assert 'tft' in results
            assert 'ensemble' in results

            # 파일 생성 확인
            for model_name, files in results.items():
                assert Path(files['json']).exists()
                assert Path(files['md']).exists()


class TestModelCardContent:
    """모델 카드 내용 검증 테스트"""

    def test_lstm_has_required_sections(self):
        """LSTM 모델 카드 필수 섹션"""
        from src.api.docs import create_lstm_model_card

        card = create_lstm_model_card()
        data = card.to_dict()

        required_sections = [
            'model_details',
            'architecture',
            'training',
            'evaluation',
            'intended_use',
            'limitations_and_biases'
        ]

        for section in required_sections:
            assert section in data, f"Missing section: {section}"

    def test_metrics_are_reasonable(self):
        """메트릭 값이 합리적인지"""
        from src.api.docs import create_lstm_model_card, create_tft_model_card

        for create_fn in [create_lstm_model_card, create_tft_model_card]:
            card = create_fn()

            # MAPE는 0-100 사이여야 함
            if 'MAPE' in card.metrics:
                assert 0 <= card.metrics['MAPE'] <= 100

            # R2는 0-1 사이여야 함
            if 'R2' in card.metrics:
                assert 0 <= card.metrics['R2'] <= 1


# ============================================================================
# Changelog 테스트
# ============================================================================

class TestChangelog:
    """변경 이력 테스트"""

    def test_get_changelog(self):
        """변경 이력 가져오기"""
        from src.api.docs import get_changelog

        changelog = get_changelog()

        assert '# API Changelog' in changelog
        assert '[1.0.0]' in changelog

    def test_changelog_has_sections(self):
        """변경 이력 섹션 확인"""
        from src.api.docs import get_changelog

        changelog = get_changelog()

        assert '### Added' in changelog
        assert '### Security' in changelog or '### Fixed' in changelog


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
