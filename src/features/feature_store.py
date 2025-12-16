"""
FEAT-011: 피처 스토어 구축
==========================

재사용 가능한 피처 저장소 - 버전 관리 및 일관성 보장

주요 기능:
1. 피처 정의 및 등록
2. 피처 버전 관리
3. 학습/추론 시 동일 피처 보장
4. 피처 계보(Lineage) 추적
5. 피처 그룹 관리

Author: Claude Code
Date: 2025-12
"""

import json
import hashlib
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import warnings
from abc import ABC, abstractmethod


@dataclass
class FeatureMetadata:
    """피처 메타데이터"""
    name: str
    dtype: str
    description: str = ""
    source: str = ""  # raw, derived, external
    transformation: str = ""  # 변환 방법
    dependencies: List[str] = field(default_factory=list)  # 의존 피처
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureMetadata':
        return cls(**data)


@dataclass
class FeatureGroup:
    """피처 그룹 (관련 피처들의 집합)"""
    name: str
    description: str
    features: List[str]
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FeatureLineage:
    """피처 계보 추적"""
    feature_name: str
    source_features: List[str]
    transformation_code: str
    transformation_params: Dict[str, Any]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


class FeatureRegistry:
    """
    피처 레지스트리

    피처 정의, 버전 관리, 계보 추적을 담당합니다.

    Example:
        >>> registry = FeatureRegistry('./feature_store')
        >>> registry.register_feature('temp_scaled', dtype='float32', source='derived')
        >>> registry.get_feature_metadata('temp_scaled')
    """

    def __init__(self, store_path: Union[str, Path]):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self._metadata_path = self.store_path / 'metadata'
        self._data_path = self.store_path / 'data'
        self._lineage_path = self.store_path / 'lineage'

        for path in [self._metadata_path, self._data_path, self._lineage_path]:
            path.mkdir(parents=True, exist_ok=True)

        self._features: Dict[str, FeatureMetadata] = {}
        self._groups: Dict[str, FeatureGroup] = {}
        self._lineages: Dict[str, FeatureLineage] = {}

        self._load_registry()

    def _load_registry(self):
        """레지스트리 로드"""
        registry_file = self._metadata_path / 'registry.json'
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                data = json.load(f)

            self._features = {
                k: FeatureMetadata.from_dict(v)
                for k, v in data.get('features', {}).items()
            }
            self._groups = {
                k: FeatureGroup(**v)
                for k, v in data.get('groups', {}).items()
            }

    def _save_registry(self):
        """레지스트리 저장"""
        data = {
            'features': {k: v.to_dict() for k, v in self._features.items()},
            'groups': {k: v.to_dict() for k, v in self._groups.items()}
        }

        with open(self._metadata_path / 'registry.json', 'w') as f:
            json.dump(data, f, indent=2)

    def register_feature(
        self,
        name: str,
        dtype: str = 'float32',
        description: str = "",
        source: str = 'derived',
        transformation: str = "",
        dependencies: List[str] = None,
        version: str = "1.0.0",
        tags: List[str] = None
    ) -> FeatureMetadata:
        """
        피처 등록

        Args:
            name: 피처 이름
            dtype: 데이터 타입
            description: 설명
            source: 소스 유형 (raw, derived, external)
            transformation: 변환 방법
            dependencies: 의존 피처 리스트
            version: 버전
            tags: 태그 리스트

        Returns:
            FeatureMetadata
        """
        metadata = FeatureMetadata(
            name=name,
            dtype=dtype,
            description=description,
            source=source,
            transformation=transformation,
            dependencies=dependencies or [],
            version=version,
            tags=tags or []
        )

        self._features[name] = metadata
        self._save_registry()

        return metadata

    def get_feature_metadata(self, name: str) -> Optional[FeatureMetadata]:
        """피처 메타데이터 조회"""
        return self._features.get(name)

    def list_features(self, source: str = None, tags: List[str] = None) -> List[str]:
        """피처 목록 조회"""
        features = list(self._features.keys())

        if source:
            features = [
                f for f in features
                if self._features[f].source == source
            ]

        if tags:
            features = [
                f for f in features
                if any(t in self._features[f].tags for t in tags)
            ]

        return features

    def create_feature_group(
        self,
        name: str,
        features: List[str],
        description: str = ""
    ) -> FeatureGroup:
        """
        피처 그룹 생성

        Args:
            name: 그룹 이름
            features: 피처 리스트
            description: 설명

        Returns:
            FeatureGroup
        """
        group = FeatureGroup(
            name=name,
            features=features,
            description=description
        )

        self._groups[name] = group
        self._save_registry()

        return group

    def get_feature_group(self, name: str) -> Optional[FeatureGroup]:
        """피처 그룹 조회"""
        return self._groups.get(name)

    def add_lineage(
        self,
        feature_name: str,
        source_features: List[str],
        transformation_code: str,
        transformation_params: Dict[str, Any] = None
    ) -> FeatureLineage:
        """
        피처 계보 추가

        Args:
            feature_name: 피처 이름
            source_features: 소스 피처 리스트
            transformation_code: 변환 코드
            transformation_params: 변환 파라미터

        Returns:
            FeatureLineage
        """
        lineage = FeatureLineage(
            feature_name=feature_name,
            source_features=source_features,
            transformation_code=transformation_code,
            transformation_params=transformation_params or {}
        )

        self._lineages[feature_name] = lineage

        # 계보 저장
        lineage_file = self._lineage_path / f'{feature_name}.json'
        with open(lineage_file, 'w') as f:
            json.dump(asdict(lineage), f, indent=2)

        return lineage

    def get_lineage(self, feature_name: str) -> Optional[FeatureLineage]:
        """피처 계보 조회"""
        if feature_name in self._lineages:
            return self._lineages[feature_name]

        lineage_file = self._lineage_path / f'{feature_name}.json'
        if lineage_file.exists():
            with open(lineage_file, 'r') as f:
                data = json.load(f)
            return FeatureLineage(**data)

        return None


class FeatureStore:
    """
    피처 스토어

    피처 데이터 저장, 버전 관리, 조회를 담당합니다.

    Example:
        >>> store = FeatureStore('./feature_store')
        >>> store.save_features(df, 'training_features', version='1.0.0')
        >>> features = store.load_features('training_features', version='1.0.0')
    """

    def __init__(self, store_path: Union[str, Path]):
        self.store_path = Path(store_path)
        self.registry = FeatureRegistry(store_path)

        self._data_path = self.store_path / 'data'
        self._data_path.mkdir(parents=True, exist_ok=True)

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """DataFrame 해시 계산"""
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:8]

    def save_features(
        self,
        df: pd.DataFrame,
        name: str,
        version: str = None,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        피처 저장

        Args:
            df: 피처 DataFrame
            name: 피처셋 이름
            version: 버전 (None이면 자동 생성)
            metadata: 추가 메타데이터

        Returns:
            저장 정보 딕셔너리
        """
        # 버전 생성
        if version is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            data_hash = self._compute_hash(df)
            version = f"{timestamp}_{data_hash}"

        # 저장 경로
        feature_path = self._data_path / name / version
        feature_path.mkdir(parents=True, exist_ok=True)

        # 데이터 저장
        df.to_parquet(feature_path / 'features.parquet', index=True)

        # 메타데이터 저장
        meta = {
            'name': name,
            'version': version,
            'shape': list(df.shape),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'created_at': datetime.now().isoformat(),
            'hash': self._compute_hash(df),
            **(metadata or {})
        }

        with open(feature_path / 'metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

        # 버전 목록 업데이트
        self._update_version_list(name, version)

        return meta

    def _update_version_list(self, name: str, version: str):
        """버전 목록 업데이트"""
        version_file = self._data_path / name / 'versions.json'

        if version_file.exists():
            with open(version_file, 'r') as f:
                versions = json.load(f)
        else:
            versions = {'versions': [], 'latest': None}

        if version not in versions['versions']:
            versions['versions'].append(version)
            versions['latest'] = version

        with open(version_file, 'w') as f:
            json.dump(versions, f, indent=2)

    def load_features(
        self,
        name: str,
        version: str = None,
        columns: List[str] = None
    ) -> pd.DataFrame:
        """
        피처 로드

        Args:
            name: 피처셋 이름
            version: 버전 (None이면 최신)
            columns: 로드할 컬럼 리스트

        Returns:
            피처 DataFrame
        """
        if version is None:
            version = self.get_latest_version(name)
            if version is None:
                raise ValueError(f"No versions found for '{name}'")

        feature_path = self._data_path / name / version / 'features.parquet'

        if not feature_path.exists():
            raise FileNotFoundError(f"Features not found: {name}/{version}")

        df = pd.read_parquet(feature_path, columns=columns)
        return df

    def get_latest_version(self, name: str) -> Optional[str]:
        """최신 버전 조회"""
        version_file = self._data_path / name / 'versions.json'

        if not version_file.exists():
            return None

        with open(version_file, 'r') as f:
            versions = json.load(f)

        return versions.get('latest')

    def list_versions(self, name: str) -> List[str]:
        """버전 목록 조회"""
        version_file = self._data_path / name / 'versions.json'

        if not version_file.exists():
            return []

        with open(version_file, 'r') as f:
            versions = json.load(f)

        return versions.get('versions', [])

    def get_feature_metadata(self, name: str, version: str = None) -> Optional[Dict]:
        """피처 메타데이터 조회"""
        if version is None:
            version = self.get_latest_version(name)

        if version is None:
            return None

        meta_path = self._data_path / name / version / 'metadata.json'

        if not meta_path.exists():
            return None

        with open(meta_path, 'r') as f:
            return json.load(f)

    def compare_versions(self, name: str, v1: str, v2: str) -> Dict[str, Any]:
        """
        버전 비교

        Args:
            name: 피처셋 이름
            v1: 버전 1
            v2: 버전 2

        Returns:
            비교 결과
        """
        meta1 = self.get_feature_metadata(name, v1)
        meta2 = self.get_feature_metadata(name, v2)

        if meta1 is None or meta2 is None:
            raise ValueError("Version not found")

        comparison = {
            'v1': v1,
            'v2': v2,
            'shape_change': {
                'v1': meta1['shape'],
                'v2': meta2['shape']
            },
            'columns_added': [
                c for c in meta2['columns'] if c not in meta1['columns']
            ],
            'columns_removed': [
                c for c in meta1['columns'] if c not in meta2['columns']
            ],
            'dtype_changes': {}
        }

        # dtype 변경 확인
        for col in set(meta1['columns']) & set(meta2['columns']):
            if meta1['dtypes'][col] != meta2['dtypes'][col]:
                comparison['dtype_changes'][col] = {
                    'v1': meta1['dtypes'][col],
                    'v2': meta2['dtypes'][col]
                }

        return comparison


class FeatureTransformer(ABC):
    """피처 변환기 기본 클래스"""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'FeatureTransformer':
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


class FeaturePipeline:
    """
    피처 파이프라인

    변환 단계를 정의하고 일관되게 적용합니다.

    Example:
        >>> pipeline = FeaturePipeline()
        >>> pipeline.add_step('normalize', StandardScaler())
        >>> pipeline.add_step('encode', OneHotEncoder())
        >>> result = pipeline.fit_transform(df)
    """

    def __init__(self, name: str = 'default'):
        self.name = name
        self.steps: List[tuple] = []
        self._fitted = False

    def add_step(self, name: str, transformer: FeatureTransformer) -> 'FeaturePipeline':
        """변환 단계 추가"""
        self.steps.append((name, transformer))
        return self

    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """파이프라인 학습"""
        current_df = df.copy()

        for name, transformer in self.steps:
            transformer.fit(current_df)
            current_df = transformer.transform(current_df)

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """파이프라인 적용"""
        if not self._fitted:
            raise ValueError("Pipeline not fitted. Call fit() first.")

        current_df = df.copy()

        for name, transformer in self.steps:
            current_df = transformer.transform(current_df)

        return current_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """학습 및 변환"""
        return self.fit(df).transform(df)

    def save(self, path: Union[str, Path]):
        """파이프라인 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'pipeline.pkl', 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeaturePipeline':
        """파이프라인 로드"""
        path = Path(path)

        with open(path / 'pipeline.pkl', 'rb') as f:
            return pickle.load(f)


class FeatureValidator:
    """
    피처 검증기

    학습/추론 시 피처 일관성을 검증합니다.

    Example:
        >>> validator = FeatureValidator.from_training(train_df)
        >>> is_valid, errors = validator.validate(inference_df)
    """

    def __init__(
        self,
        expected_columns: List[str],
        expected_dtypes: Dict[str, str],
        value_ranges: Dict[str, tuple] = None
    ):
        self.expected_columns = expected_columns
        self.expected_dtypes = expected_dtypes
        self.value_ranges = value_ranges or {}

    @classmethod
    def from_training(cls, df: pd.DataFrame) -> 'FeatureValidator':
        """학습 데이터에서 검증기 생성"""
        expected_columns = list(df.columns)
        expected_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        # 수치형 컬럼의 범위 계산
        value_ranges = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            value_ranges[col] = (
                float(df[col].min()),
                float(df[col].max())
            )

        return cls(expected_columns, expected_dtypes, value_ranges)

    def validate(self, df: pd.DataFrame) -> tuple:
        """
        피처 검증

        Returns:
            (is_valid, errors): 유효성 및 에러 목록
        """
        errors = []

        # 컬럼 확인
        missing_cols = set(self.expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(self.expected_columns)

        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            warnings.warn(f"Extra columns (ignored): {extra_cols}")

        # dtype 확인
        for col in set(self.expected_columns) & set(df.columns):
            actual_dtype = str(df[col].dtype)
            expected_dtype = self.expected_dtypes.get(col)

            # 호환 가능한 타입 확인
            if not self._dtype_compatible(actual_dtype, expected_dtype):
                errors.append(f"Column '{col}': expected {expected_dtype}, got {actual_dtype}")

        # 값 범위 확인 (경고만)
        for col, (min_val, max_val) in self.value_ranges.items():
            if col in df.columns:
                actual_min = df[col].min()
                actual_max = df[col].max()

                if actual_min < min_val * 0.5 or actual_max > max_val * 2:
                    warnings.warn(
                        f"Column '{col}' range [{actual_min:.2f}, {actual_max:.2f}] "
                        f"differs from training [{min_val:.2f}, {max_val:.2f}]"
                    )

        is_valid = len(errors) == 0
        return is_valid, errors

    def _dtype_compatible(self, actual: str, expected: str) -> bool:
        """dtype 호환성 확인"""
        # 정수/실수 호환
        numeric_types = {'int32', 'int64', 'float32', 'float64'}
        if actual in numeric_types and expected in numeric_types:
            return True
        return actual == expected

    def save(self, path: Union[str, Path]):
        """검증기 저장"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            'expected_columns': self.expected_columns,
            'expected_dtypes': self.expected_dtypes,
            'value_ranges': self.value_ranges
        }

        with open(path / 'validator.json', 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FeatureValidator':
        """검증기 로드"""
        path = Path(path)

        with open(path / 'validator.json', 'r') as f:
            data = json.load(f)

        return cls(
            expected_columns=data['expected_columns'],
            expected_dtypes=data['expected_dtypes'],
            value_ranges={k: tuple(v) for k, v in data.get('value_ranges', {}).items()}
        )


def create_feature_store(path: Union[str, Path] = './feature_store') -> FeatureStore:
    """
    피처 스토어 생성 팩토리 함수

    Args:
        path: 저장 경로

    Returns:
        FeatureStore 인스턴스
    """
    return FeatureStore(path)
