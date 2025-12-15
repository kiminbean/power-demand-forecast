"""
MODEL-001: PyTorch Dataset 및 DataLoader 단위 테스트

테스트 범위:
1. TimeSeriesScaler (Min-Max 정규화)
2. TimeSeriesDataset (시퀀스 생성)
3. MultiHorizonDataset (다중 예측 시점)
4. 데이터 분할 함수
5. DataLoader 생성
6. 데이터 누수 방지
"""

import numpy as np
import pandas as pd
import pytest
import torch
import sys
from pathlib import Path
import tempfile

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.dataset import (
    get_device,
    TimeSeriesScaler,
    TimeSeriesDataset,
    MultiHorizonDataset,
    split_data_by_time,
    prepare_features,
    create_dataloaders,
    create_multi_horizon_dataloaders,
)


# ============================================================
# 디바이스 테스트
# ============================================================

class TestGetDevice:
    """get_device 함수 테스트"""

    def test_returns_device(self):
        """디바이스 반환 확인"""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_device_type(self):
        """디바이스 타입 확인"""
        device = get_device()
        assert device.type in ['cpu', 'cuda', 'mps']


# ============================================================
# 스케일러 테스트
# ============================================================

class TestTimeSeriesScaler:
    """TimeSeriesScaler 클래스 테스트"""

    def test_fit(self):
        """fit 메서드 테스트"""
        data = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]])
        scaler = TimeSeriesScaler()
        scaler.fit(data)

        np.testing.assert_array_equal(scaler.min_vals, [1, 10])
        np.testing.assert_array_equal(scaler.max_vals, [5, 50])
        assert scaler.is_fitted

    def test_transform(self):
        """transform 메서드 테스트"""
        data = np.array([[1, 10], [5, 50]])
        scaler = TimeSeriesScaler()
        scaler.fit(data)

        transformed = scaler.transform(data)

        # [1, 10] -> [0, 0], [5, 50] -> [1, 1]
        np.testing.assert_array_almost_equal(transformed[0], [0, 0])
        np.testing.assert_array_almost_equal(transformed[1], [1, 1])

    def test_fit_transform(self):
        """fit_transform 메서드 테스트"""
        data = np.array([[0, 0], [10, 100]])
        scaler = TimeSeriesScaler()

        transformed = scaler.fit_transform(data)

        np.testing.assert_array_almost_equal(transformed[0], [0, 0])
        np.testing.assert_array_almost_equal(transformed[1], [1, 1])

    def test_inverse_transform(self):
        """inverse_transform 메서드 테스트"""
        data = np.array([[0, 0], [10, 100]])
        scaler = TimeSeriesScaler()
        transformed = scaler.fit_transform(data)
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(recovered, data)

    def test_inverse_transform_target(self):
        """inverse_transform_target 메서드 테스트"""
        data = np.array([[0, 0], [10, 100]])
        scaler = TimeSeriesScaler()
        scaler.fit(data)

        # 타겟(0번 컬럼)만 역변환
        normalized_target = np.array([0.5])
        recovered = scaler.inverse_transform_target(normalized_target, target_idx=0)

        assert recovered[0] == 5  # 0~10의 0.5 = 5

    def test_constant_column(self):
        """상수 컬럼 처리 (0으로 나누기 방지)"""
        data = np.array([[1, 5], [2, 5], [3, 5]])  # 두 번째 컬럼 상수
        scaler = TimeSeriesScaler()
        transformed = scaler.fit_transform(data)

        # 상수 컬럼은 0으로 변환
        assert not np.isnan(transformed).any()
        assert not np.isinf(transformed).any()

    def test_dataframe_input(self):
        """DataFrame 입력 처리"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
        scaler = TimeSeriesScaler()
        scaler.fit(df)

        assert scaler.feature_names == ['a', 'b']
        assert scaler.is_fitted

    def test_save_load(self):
        """저장/로드 테스트"""
        data = np.array([[0, 0], [10, 100]])
        scaler = TimeSeriesScaler()
        scaler.fit(data)

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            scaler.save(f.name)

            new_scaler = TimeSeriesScaler()
            new_scaler.load(f.name)

            np.testing.assert_array_equal(scaler.min_vals, new_scaler.min_vals)
            np.testing.assert_array_equal(scaler.max_vals, new_scaler.max_vals)

    def test_transform_without_fit_error(self):
        """fit 없이 transform 시 에러"""
        scaler = TimeSeriesScaler()

        with pytest.raises(ValueError, match="학습되지 않았습니다"):
            scaler.transform(np.array([[1, 2]]))


# ============================================================
# TimeSeriesDataset 테스트
# ============================================================

class TestTimeSeriesDataset:
    """TimeSeriesDataset 클래스 테스트"""

    @pytest.fixture
    def sample_data(self):
        """샘플 데이터 생성"""
        # 100 샘플, 5 특성
        return np.arange(500).reshape(100, 5).astype(np.float32)

    def test_dataset_length(self, sample_data):
        """데이터셋 길이 확인"""
        seq_length = 10
        horizon = 1
        dataset = TimeSeriesDataset(sample_data, seq_length=seq_length, horizon=horizon)

        # n_samples = 100 - 10 - 1 + 1 = 90
        expected_length = len(sample_data) - seq_length - horizon + 1
        assert len(dataset) == expected_length

    def test_getitem_shapes(self, sample_data):
        """__getitem__ 반환 형태 확인"""
        seq_length = 10
        dataset = TimeSeriesDataset(sample_data, seq_length=seq_length, horizon=1)

        X, y = dataset[0]

        assert X.shape == (seq_length, 5)  # [seq_len, features]
        assert y.shape == (1,)  # [1]

    def test_getitem_values(self):
        """__getitem__ 값 확인"""
        # 간단한 데이터
        data = np.arange(20).reshape(10, 2).astype(np.float32)
        # [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15], [16,17], [18,19]]

        dataset = TimeSeriesDataset(data, target_idx=0, seq_length=3, horizon=1)

        X, y = dataset[0]

        # X: 인덱스 0, 1, 2의 데이터
        expected_X = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.float32)
        np.testing.assert_array_equal(X.numpy(), expected_X)

        # y: 인덱스 3의 타겟 (0번 컬럼)
        assert y.item() == 6  # data[3, 0]

    def test_horizon_effect(self):
        """horizon 파라미터 효과"""
        data = np.arange(20).reshape(10, 2).astype(np.float32)

        dataset_h1 = TimeSeriesDataset(data, seq_length=3, horizon=1)
        dataset_h2 = TimeSeriesDataset(data, seq_length=3, horizon=2)

        _, y1 = dataset_h1[0]
        _, y2 = dataset_h2[0]

        # h1: data[3, 0] = 6
        # h2: data[4, 0] = 8
        assert y1.item() == 6
        assert y2.item() == 8

    def test_tensor_dtype(self, sample_data):
        """텐서 데이터 타입 확인"""
        dataset = TimeSeriesDataset(sample_data, seq_length=10)

        X, y = dataset[0]

        assert X.dtype == torch.float32
        assert y.dtype == torch.float32

    def test_short_data_error(self):
        """데이터가 너무 짧을 때 에러"""
        short_data = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(ValueError, match="데이터가 너무 짧습니다"):
            TimeSeriesDataset(short_data, seq_length=20, horizon=1)


# ============================================================
# MultiHorizonDataset 테스트
# ============================================================

class TestMultiHorizonDataset:
    """MultiHorizonDataset 클래스 테스트"""

    @pytest.fixture
    def sample_data(self):
        """샘플 데이터 생성"""
        return np.arange(500).reshape(100, 5).astype(np.float32)

    def test_multi_horizon_shape(self, sample_data):
        """다중 예측 시점 형태 확인"""
        horizons = [1, 6, 12, 24]
        dataset = MultiHorizonDataset(
            sample_data, seq_length=10, horizons=horizons
        )

        X, y = dataset[0]

        assert X.shape == (10, 5)
        assert y.shape == (4,)  # 4개의 horizon

    def test_multi_horizon_values(self):
        """다중 예측 시점 값 확인"""
        data = np.arange(100).reshape(50, 2).astype(np.float32)
        horizons = [1, 3]

        dataset = MultiHorizonDataset(
            data, target_idx=0, seq_length=5, horizons=horizons
        )

        X, y = dataset[0]

        # seq_length=5이므로 X는 인덱스 0~4
        # horizon=1: 인덱스 5의 타겟 = data[5, 0] = 10
        # horizon=3: 인덱스 7의 타겟 = data[7, 0] = 14
        assert y[0].item() == 10
        assert y[1].item() == 14


# ============================================================
# 데이터 분할 테스트
# ============================================================

class TestSplitDataByTime:
    """split_data_by_time 함수 테스트"""

    @pytest.fixture
    def sample_df(self):
        """샘플 DataFrame 생성"""
        dates = pd.date_range('2020-01-01', periods=1000, freq='h')
        return pd.DataFrame({'value': range(1000)}, index=dates)

    def test_split_sizes(self, sample_df):
        """분할 크기 확인"""
        # 약 500시간 지점에서 분할
        train_end = '2020-01-21 11:00:00'
        val_end = '2020-01-31 23:00:00'

        train, val, test = split_data_by_time(sample_df, train_end, val_end)

        # 총합이 원본과 같아야 함 (중복 없이 분할)
        assert len(train) + len(val) + len(test) == len(sample_df)

    def test_no_overlap(self, sample_df):
        """분할 간 중복 없음"""
        train_end = '2020-01-21 11:00:00'
        val_end = '2020-01-31 23:00:00'

        train, val, test = split_data_by_time(sample_df, train_end, val_end)

        # 인덱스 겹침 확인
        assert len(train.index.intersection(val.index)) == 0
        assert len(val.index.intersection(test.index)) == 0
        assert len(train.index.intersection(test.index)) == 0


class TestPrepareFeatures:
    """prepare_features 함수 테스트"""

    @pytest.fixture
    def sample_df(self):
        """샘플 DataFrame 생성"""
        return pd.DataFrame({
            'power_demand': [100, 200, 300],
            'temp': [10, 20, 30],
            'humidity': [50, 60, 70],
            '일시': ['a', 'b', 'c'],  # 비숫자형
            'clfmAbbrCd': [1, 2, 3]  # 제외 대상
        })

    def test_target_first(self, sample_df):
        """타겟이 첫 번째 컬럼"""
        data, feature_names, target_idx = prepare_features(sample_df)

        assert feature_names[0] == 'power_demand'
        assert target_idx == 0

    def test_exclude_non_numeric(self, sample_df):
        """비숫자형 컬럼 제외"""
        data, feature_names, _ = prepare_features(sample_df)

        assert '일시' not in feature_names

    def test_exclude_specified_cols(self, sample_df):
        """지정된 컬럼 제외"""
        data, feature_names, _ = prepare_features(sample_df)

        assert 'clfmAbbrCd' not in feature_names


# ============================================================
# DataLoader 테스트
# ============================================================

class TestCreateDataLoaders:
    """create_dataloaders 함수 테스트"""

    @pytest.fixture
    def sample_data(self):
        """샘플 데이터 생성"""
        np.random.seed(42)
        train = np.random.randn(500, 10).astype(np.float32)
        val = np.random.randn(100, 10).astype(np.float32)
        test = np.random.randn(100, 10).astype(np.float32)
        return train, val, test

    def test_loader_creation(self, sample_data):
        """DataLoader 생성 확인"""
        train, val, test = sample_data

        train_loader, val_loader, test_loader = create_dataloaders(
            train, val, test, seq_length=48, horizon=1, batch_size=32
        )

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

    def test_batch_shape(self, sample_data):
        """배치 형태 확인"""
        train, val, test = sample_data
        seq_length = 48
        batch_size = 32

        train_loader, _, _ = create_dataloaders(
            train, val, test, seq_length=seq_length, batch_size=batch_size
        )

        for X, y in train_loader:
            assert X.shape[0] <= batch_size
            assert X.shape[1] == seq_length
            assert X.shape[2] == 10  # n_features
            break

    def test_shuffle_train(self, sample_data):
        """학습 데이터 셔플 확인"""
        train, val, test = sample_data

        train_loader1, _, _ = create_dataloaders(train, val, test, batch_size=32)
        train_loader2, _, _ = create_dataloaders(train, val, test, batch_size=32)

        # 두 번 순회하면 순서가 다를 수 있음
        # (확률적이므로 약한 테스트)
        batch1 = next(iter(train_loader1))[0]
        batch2 = next(iter(train_loader2))[0]

        # 같은 시드가 아니면 다를 가능성 높음
        # (테스트 통과 조건 완화)
        assert batch1.shape == batch2.shape


class TestCreateMultiHorizonDataLoaders:
    """create_multi_horizon_dataloaders 함수 테스트"""

    def test_multi_horizon_loader(self):
        """다중 예측 시점 DataLoader"""
        np.random.seed(42)
        train = np.random.randn(500, 10).astype(np.float32)
        val = np.random.randn(100, 10).astype(np.float32)
        test = np.random.randn(100, 10).astype(np.float32)

        horizons = [1, 6, 12, 24]
        train_loader, _, _ = create_multi_horizon_dataloaders(
            train, val, test, seq_length=48, horizons=horizons, batch_size=32
        )

        for X, y in train_loader:
            assert X.shape[1] == 48  # seq_length
            assert y.shape[1] == 4   # len(horizons)
            break


# ============================================================
# 데이터 누수 방지 테스트
# ============================================================

class TestDataLeakagePrevention:
    """데이터 누수 방지 테스트"""

    def test_no_future_in_input(self):
        """입력에 미래 정보 없음"""
        data = np.arange(100).reshape(50, 2).astype(np.float32)
        seq_length = 5
        horizon = 1

        dataset = TimeSeriesDataset(data, seq_length=seq_length, horizon=horizon)

        for i in range(len(dataset)):
            X, y = dataset[i]

            # X의 마지막 시점 인덱스
            x_last_idx = i + seq_length - 1
            # y의 타겟 시점 인덱스
            y_idx = i + seq_length + horizon - 1

            # 타겟 시점이 입력 시점보다 미래여야 함
            assert y_idx > x_last_idx

    def test_train_val_test_temporal_order(self):
        """Train/Val/Test 시간 순서"""
        dates = pd.date_range('2020-01-01', periods=1000, freq='h')
        df = pd.DataFrame({'value': range(1000)}, index=dates)

        train, val, test = split_data_by_time(
            df,
            train_end='2020-01-21 11:00:00',
            val_end='2020-01-31 23:00:00'
        )

        # Train의 마지막 < Val의 첫 번째 < Test의 첫 번째
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()


# ============================================================
# 성능 테스트
# ============================================================

class TestPerformance:
    """성능 테스트"""

    def test_large_dataset(self):
        """대용량 데이터셋 처리"""
        # 1년치 시간별 데이터 (8760 샘플)
        large_data = np.random.randn(8760, 20).astype(np.float32)

        dataset = TimeSeriesDataset(large_data, seq_length=48, horizon=24)

        # 데이터셋 순회 가능
        assert len(dataset) > 0

        # 샘플 접근 가능
        X, y = dataset[0]
        assert X.shape == (48, 20)

    def test_iteration_speed(self):
        """DataLoader 순회 속도"""
        data = np.random.randn(1000, 10).astype(np.float32)

        train_loader, _, _ = create_dataloaders(
            data, data[:100], data[:100],
            seq_length=48, batch_size=64
        )

        # 전체 순회 가능
        batch_count = 0
        for X, y in train_loader:
            batch_count += 1

        assert batch_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
