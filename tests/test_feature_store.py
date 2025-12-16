"""
피처 스토어 테스트
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


class TestFeatureMetadata:
    """FeatureMetadata 테스트"""

    def test_creation(self):
        """메타데이터 생성"""
        from src.features.feature_store import FeatureMetadata

        meta = FeatureMetadata(
            name='temperature',
            dtype='float32',
            description='기온 데이터',
            source='raw'
        )

        assert meta.name == 'temperature'
        assert meta.dtype == 'float32'

    def test_to_dict(self):
        """딕셔너리 변환"""
        from src.features.feature_store import FeatureMetadata

        meta = FeatureMetadata(
            name='temp',
            dtype='float32',
            description='test'
        )

        d = meta.to_dict()

        assert isinstance(d, dict)
        assert d['name'] == 'temp'

    def test_from_dict(self):
        """딕셔너리에서 생성"""
        from src.features.feature_store import FeatureMetadata

        data = {
            'name': 'temp',
            'dtype': 'float32',
            'description': 'test',
            'source': 'raw',
            'transformation': '',
            'dependencies': [],
            'created_at': '2025-01-01',
            'version': '1.0.0',
            'tags': []
        }

        meta = FeatureMetadata.from_dict(data)

        assert meta.name == 'temp'


class TestFeatureRegistry:
    """FeatureRegistry 테스트"""

    def test_creation(self, tmp_path):
        """레지스트리 생성"""
        from src.features.feature_store import FeatureRegistry

        registry = FeatureRegistry(tmp_path / 'store')

        assert registry.store_path.exists()

    def test_register_feature(self, tmp_path):
        """피처 등록"""
        from src.features.feature_store import FeatureRegistry

        registry = FeatureRegistry(tmp_path / 'store')

        meta = registry.register_feature(
            name='temperature',
            dtype='float32',
            description='기온 데이터',
            source='raw',
            tags=['weather']
        )

        assert meta.name == 'temperature'
        assert 'weather' in meta.tags

    def test_get_feature_metadata(self, tmp_path):
        """메타데이터 조회"""
        from src.features.feature_store import FeatureRegistry

        registry = FeatureRegistry(tmp_path / 'store')
        registry.register_feature(name='temp', dtype='float32')

        meta = registry.get_feature_metadata('temp')

        assert meta is not None
        assert meta.name == 'temp'

    def test_list_features(self, tmp_path):
        """피처 목록"""
        from src.features.feature_store import FeatureRegistry

        registry = FeatureRegistry(tmp_path / 'store')
        registry.register_feature(name='temp', dtype='float32', source='raw')
        registry.register_feature(name='humidity', dtype='float32', source='raw')
        registry.register_feature(name='thi', dtype='float32', source='derived')

        all_features = registry.list_features()
        raw_features = registry.list_features(source='raw')

        assert len(all_features) == 3
        assert len(raw_features) == 2

    def test_create_feature_group(self, tmp_path):
        """피처 그룹 생성"""
        from src.features.feature_store import FeatureRegistry

        registry = FeatureRegistry(tmp_path / 'store')

        group = registry.create_feature_group(
            name='weather_features',
            features=['temp', 'humidity', 'wind'],
            description='기상 피처'
        )

        assert group.name == 'weather_features'
        assert len(group.features) == 3

    def test_add_lineage(self, tmp_path):
        """계보 추가"""
        from src.features.feature_store import FeatureRegistry

        registry = FeatureRegistry(tmp_path / 'store')

        lineage = registry.add_lineage(
            feature_name='thi',
            source_features=['temp', 'humidity'],
            transformation_code='0.8*temp + 0.01*humidity*(0.99*temp - 14.3) + 46.3',
            transformation_params={'formula': 'NOAA THI'}
        )

        assert lineage.feature_name == 'thi'
        assert 'temp' in lineage.source_features

    def test_get_lineage(self, tmp_path):
        """계보 조회"""
        from src.features.feature_store import FeatureRegistry

        registry = FeatureRegistry(tmp_path / 'store')
        registry.add_lineage(
            feature_name='thi',
            source_features=['temp', 'humidity'],
            transformation_code='formula'
        )

        lineage = registry.get_lineage('thi')

        assert lineage is not None
        assert lineage.feature_name == 'thi'


class TestFeatureStore:
    """FeatureStore 테스트"""

    def test_creation(self, tmp_path):
        """스토어 생성"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        assert store.store_path.exists()

    def test_save_features(self, tmp_path):
        """피처 저장"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        df = pd.DataFrame({
            'temp': [20.0, 21.0, 22.0],
            'humidity': [60.0, 65.0, 70.0]
        })

        meta = store.save_features(df, 'training_features', version='v1.0.0')

        assert meta['name'] == 'training_features'
        assert meta['version'] == 'v1.0.0'
        assert meta['shape'] == [3, 2]

    def test_load_features(self, tmp_path):
        """피처 로드"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        df = pd.DataFrame({
            'temp': [20.0, 21.0, 22.0],
            'humidity': [60.0, 65.0, 70.0]
        })
        store.save_features(df, 'test_features', version='v1')

        loaded = store.load_features('test_features', version='v1')

        pd.testing.assert_frame_equal(df, loaded)

    def test_load_latest_version(self, tmp_path):
        """최신 버전 로드"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [4, 5, 6]})

        store.save_features(df1, 'data', version='v1')
        store.save_features(df2, 'data', version='v2')

        loaded = store.load_features('data')  # 최신 버전

        pd.testing.assert_frame_equal(df2, loaded)

    def test_list_versions(self, tmp_path):
        """버전 목록"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        store.save_features(pd.DataFrame({'a': [1]}), 'data', version='v1')
        store.save_features(pd.DataFrame({'a': [2]}), 'data', version='v2')

        versions = store.list_versions('data')

        assert 'v1' in versions
        assert 'v2' in versions

    def test_get_feature_metadata(self, tmp_path):
        """메타데이터 조회"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        df = pd.DataFrame({'temp': [20.0, 21.0]})
        store.save_features(df, 'test', version='v1')

        meta = store.get_feature_metadata('test', 'v1')

        assert meta['columns'] == ['temp']

    def test_compare_versions(self, tmp_path):
        """버전 비교"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        df1 = pd.DataFrame({'a': [1], 'b': [2]})
        df2 = pd.DataFrame({'a': [1], 'c': [3]})

        store.save_features(df1, 'data', version='v1')
        store.save_features(df2, 'data', version='v2')

        comparison = store.compare_versions('data', 'v1', 'v2')

        assert 'c' in comparison['columns_added']
        assert 'b' in comparison['columns_removed']


class TestFeaturePipeline:
    """FeaturePipeline 테스트"""

    def test_creation(self):
        """파이프라인 생성"""
        from src.features.feature_store import FeaturePipeline

        pipeline = FeaturePipeline('test_pipeline')

        assert pipeline.name == 'test_pipeline'
        assert len(pipeline.steps) == 0

    def test_save_load(self, tmp_path):
        """저장 및 로드"""
        from src.features.feature_store import FeaturePipeline

        pipeline = FeaturePipeline('test')
        pipeline._fitted = True

        pipeline.save(tmp_path / 'pipeline')

        loaded = FeaturePipeline.load(tmp_path / 'pipeline')

        assert loaded.name == 'test'
        assert loaded._fitted


class TestFeatureValidator:
    """FeatureValidator 테스트"""

    def test_from_training(self):
        """학습 데이터에서 생성"""
        from src.features.feature_store import FeatureValidator

        df = pd.DataFrame({
            'temp': [20.0, 25.0, 30.0],
            'humidity': [60, 70, 80]
        })

        validator = FeatureValidator.from_training(df)

        assert 'temp' in validator.expected_columns
        assert 'humidity' in validator.expected_columns

    def test_validate_success(self):
        """검증 성공"""
        from src.features.feature_store import FeatureValidator

        train_df = pd.DataFrame({
            'temp': [20.0, 25.0],
            'humidity': [60.0, 70.0]
        })

        test_df = pd.DataFrame({
            'temp': [22.0, 27.0],
            'humidity': [65.0, 75.0]
        })

        validator = FeatureValidator.from_training(train_df)
        is_valid, errors = validator.validate(test_df)

        assert is_valid
        assert len(errors) == 0

    def test_validate_missing_column(self):
        """컬럼 누락 검증"""
        from src.features.feature_store import FeatureValidator

        train_df = pd.DataFrame({
            'temp': [20.0],
            'humidity': [60.0]
        })

        test_df = pd.DataFrame({
            'temp': [22.0]
            # humidity 누락
        })

        validator = FeatureValidator.from_training(train_df)
        is_valid, errors = validator.validate(test_df)

        assert not is_valid
        assert any('Missing' in e for e in errors)

    def test_save_load(self, tmp_path):
        """저장 및 로드"""
        from src.features.feature_store import FeatureValidator

        df = pd.DataFrame({'temp': [20.0, 25.0]})
        validator = FeatureValidator.from_training(df)

        validator.save(tmp_path / 'validator')
        loaded = FeatureValidator.load(tmp_path / 'validator')

        assert loaded.expected_columns == validator.expected_columns


class TestCreateFeatureStore:
    """create_feature_store 테스트"""

    def test_factory(self, tmp_path):
        """팩토리 함수"""
        from src.features.feature_store import create_feature_store

        store = create_feature_store(tmp_path / 'store')

        assert store is not None
        assert store.store_path.exists()


class TestIntegration:
    """통합 테스트"""

    def test_full_workflow(self, tmp_path):
        """전체 워크플로우"""
        from src.features.feature_store import (
            FeatureStore, FeatureValidator
        )

        # 1. 스토어 생성
        store = FeatureStore(tmp_path / 'store')

        # 2. 피처 등록
        store.registry.register_feature(
            name='temperature',
            dtype='float32',
            source='raw',
            description='기온'
        )
        store.registry.register_feature(
            name='thi',
            dtype='float32',
            source='derived',
            dependencies=['temperature', 'humidity']
        )

        # 3. 피처 그룹 생성
        store.registry.create_feature_group(
            name='weather',
            features=['temperature', 'humidity', 'thi']
        )

        # 4. 데이터 저장
        train_df = pd.DataFrame({
            'temperature': [20.0, 25.0, 30.0],
            'humidity': [60.0, 70.0, 80.0],
            'thi': [65.0, 70.0, 75.0]
        })

        store.save_features(train_df, 'weather_features', version='v1.0.0')

        # 5. 검증기 생성
        validator = FeatureValidator.from_training(train_df)
        validator.save(tmp_path / 'validator')

        # 6. 추론용 데이터 검증
        inference_df = pd.DataFrame({
            'temperature': [22.0, 27.0],
            'humidity': [65.0, 75.0],
            'thi': [67.0, 72.0]
        })

        loaded_validator = FeatureValidator.load(tmp_path / 'validator')
        is_valid, errors = loaded_validator.validate(inference_df)

        assert is_valid

        # 7. 데이터 로드
        loaded_features = store.load_features('weather_features')

        pd.testing.assert_frame_equal(train_df, loaded_features)

    def test_version_management(self, tmp_path):
        """버전 관리"""
        from src.features.feature_store import FeatureStore

        store = FeatureStore(tmp_path / 'store')

        # 여러 버전 저장
        for i in range(3):
            df = pd.DataFrame({'value': [i * 10]})
            store.save_features(df, 'data', version=f'v{i+1}')

        # 버전 확인
        versions = store.list_versions('data')
        assert len(versions) == 3

        # 특정 버전 로드
        v1_data = store.load_features('data', version='v1')
        v3_data = store.load_features('data', version='v3')

        assert v1_data['value'].iloc[0] == 0
        assert v3_data['value'].iloc[0] == 20
