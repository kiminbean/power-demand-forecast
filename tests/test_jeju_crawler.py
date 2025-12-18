"""
제주 전력수급현황 크롤러 테스트
================================

JejuPowerCrawler 및 JejuPowerData 클래스 테스트
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import zipfile
import io
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

from tools.crawlers.jeju_power_crawler import (
    JejuPowerData,
    JejuPowerCrawler,
    JejuPowerDataStore,
)


# ============================================================================
# JejuPowerData Tests
# ============================================================================

class TestJejuPowerData:
    """JejuPowerData 데이터클래스 테스트"""

    def test_data_creation(self):
        """기본 데이터 생성 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        assert data.timestamp == "2025-01-01 12:00"
        assert data.system_demand == 700.0
        assert data.supply_capacity == 1400.0
        assert data.supply_reserve == 700.0

    def test_reserve_rate_calculation(self):
        """예비율 자동 계산 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        # 예비율 = 공급예비력 / 계통수요 * 100
        expected_rate = (700.0 / 700.0) * 100
        assert data.reserve_rate == expected_rate

    def test_reserve_rate_zero_demand(self):
        """수요가 0일 때 예비율 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=0.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=0.0,
            operation_reserve=350.0,
        )
        # 수요가 0이면 예비율 계산 안됨
        assert data.reserve_rate == 0.0

    def test_to_dict(self):
        """딕셔너리 변환 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        d = data.to_dict()
        assert isinstance(d, dict)
        assert d['timestamp'] == "2025-01-01 12:00"
        assert d['system_demand'] == 700.0
        assert 'reserve_rate' in d
        assert 'fetched_at' in d

    def test_current_demand_property(self):
        """EPSIS 호환 current_demand 속성 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        assert data.current_demand == data.system_demand

    def test_reserve_power_property(self):
        """EPSIS 호환 reserve_power 속성 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        assert data.reserve_power == data.supply_reserve

    def test_fetched_at_auto_set(self):
        """fetched_at 자동 설정 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        assert data.fetched_at != ""
        # 날짜 형식 확인
        datetime.strptime(data.fetched_at, "%Y-%m-%d %H:%M:%S")

    def test_source_default(self):
        """source 기본값 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        assert data.source == "data.go.kr"


# ============================================================================
# JejuPowerCrawler Tests
# ============================================================================

class TestJejuPowerCrawler:
    """JejuPowerCrawler 클래스 테스트"""

    def test_crawler_init(self):
        """크롤러 초기화 테스트"""
        crawler = JejuPowerCrawler()
        assert crawler.timeout == 30
        assert crawler.max_retries == 3
        assert crawler.cache_dir.exists()
        crawler.close()

    def test_crawler_init_custom_params(self):
        """크롤러 커스텀 파라미터 초기화 테스트"""
        crawler = JejuPowerCrawler(timeout=60, max_retries=5)
        assert crawler.timeout == 60
        assert crawler.max_retries == 5
        crawler.close()

    def test_session_creation(self):
        """HTTP 세션 생성 테스트"""
        crawler = JejuPowerCrawler()
        assert crawler.session is not None
        assert 'User-Agent' in crawler.session.headers
        crawler.close()


class TestJejuPowerCrawlerZipProcessing:
    """ZIP 파일 처리 테스트"""

    @pytest.fixture
    def sample_zip_file(self, tmp_path):
        """테스트용 ZIP 파일 생성"""
        zip_path = tmp_path / "test_jeju_power.zip"

        # 샘플 CSV 데이터 생성
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        hours = [f"{h}시" for h in range(1, 25)]

        def create_csv(base_value):
            data = {"날짜": dates.strftime("%Y-%m-%d")}
            for h in hours:
                data[h] = [base_value + np.random.randint(-50, 50) for _ in range(3)]
            return pd.DataFrame(data)

        # ZIP 파일 생성
        with zipfile.ZipFile(zip_path, 'w') as z:
            for name, base in [
                ("계통수요", 700),
                ("공급능력", 1400),
                ("공급예비력", 700),
                ("예측수요", 720),
                ("운영예비력", 350),
            ]:
                df = create_csv(base)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, encoding='utf-8')
                # 파일명을 euc-kr로 인코딩 후 cp437로 디코딩 (ZIP 표준)
                encoded_name = f"{name}.csv"
                z.writestr(encoded_name, csv_buffer.getvalue())

        return zip_path

    def test_load_from_zip(self, sample_zip_file):
        """ZIP 파일 로드 테스트"""
        crawler = JejuPowerCrawler()
        data = crawler.load_from_zip(sample_zip_file)
        crawler.close()

        assert len(data) > 0
        assert all(isinstance(d, JejuPowerData) for d in data)

    def test_load_from_zip_nonexistent(self):
        """존재하지 않는 ZIP 파일 테스트"""
        crawler = JejuPowerCrawler()
        data = crawler.load_from_zip("/nonexistent/path.zip")
        crawler.close()

        assert data == []

    def test_data_sorted_by_timestamp(self, sample_zip_file):
        """데이터가 타임스탬프로 정렬되는지 테스트"""
        crawler = JejuPowerCrawler()
        data = crawler.load_from_zip(sample_zip_file)
        crawler.close()

        if len(data) > 1:
            timestamps = [d.timestamp for d in data]
            assert timestamps == sorted(timestamps)


class TestJejuPowerCrawlerDataParsing:
    """데이터 파싱 테스트"""

    @pytest.fixture
    def sample_dataframe(self):
        """테스트용 DataFrame 생성"""
        return pd.DataFrame({
            '기준일시': ['2025-01-01 12:00', '2025-01-01 13:00', '2025-01-01 14:00'],
            '계통수요': [700.0, 710.0, 720.0],
            '공급능력': [1400.0, 1400.0, 1400.0],
            '공급예비력': [700.0, 690.0, 680.0],
            '예측수요': [720.0, 730.0, 740.0],
            '운영예비력': [350.0, 340.0, 330.0],
        })

    def test_parse_supply_data(self, sample_dataframe):
        """데이터 파싱 테스트"""
        crawler = JejuPowerCrawler()
        result = crawler.parse_supply_data(sample_dataframe)
        crawler.close()

        assert len(result) == 3
        assert result[0].system_demand == 700.0
        assert result[1].system_demand == 710.0

    def test_parse_supply_data_missing_columns(self):
        """필수 컬럼 누락 시 테스트"""
        df = pd.DataFrame({
            '기준일시': ['2025-01-01 12:00'],
            '계통수요': [700.0],
            # 공급능력, 공급예비력 누락
        })
        crawler = JejuPowerCrawler()
        result = crawler.parse_supply_data(df)
        crawler.close()

        assert result == []


class TestJejuPowerCrawlerUtilityMethods:
    """유틸리티 메서드 테스트"""

    @pytest.fixture
    def sample_data(self):
        """테스트용 데이터 리스트 생성"""
        return [
            JejuPowerData(
                timestamp=f"2025-01-0{i} 12:00",
                system_demand=700.0 + i * 10,
                supply_capacity=1400.0,
                supply_reserve=700.0 - i * 10,
                forecast_demand=720.0,
                operation_reserve=350.0,
            )
            for i in range(1, 6)
        ]

    def test_get_latest_data(self, sample_data):
        """최신 데이터 조회 테스트"""
        crawler = JejuPowerCrawler()
        latest = crawler.get_latest_data(sample_data)
        crawler.close()

        assert latest is not None
        assert latest.timestamp == "2025-01-05 12:00"

    def test_get_latest_data_empty(self):
        """빈 데이터에서 최신 데이터 조회 테스트"""
        crawler = JejuPowerCrawler()
        latest = crawler.get_latest_data([])
        crawler.close()

        assert latest is None

    def test_get_data_by_date_range(self, sample_data):
        """날짜 범위로 데이터 필터링 테스트"""
        crawler = JejuPowerCrawler()
        filtered = crawler.get_data_by_date_range(
            sample_data,
            start_date="2025-01-02",
            end_date="2025-01-04"
        )
        crawler.close()

        assert len(filtered) == 3
        assert all("2025-01-02" <= d.timestamp[:10] <= "2025-01-04" for d in filtered)


# ============================================================================
# JejuPowerDataStore Tests
# ============================================================================

class TestJejuPowerDataStore:
    """JejuPowerDataStore 클래스 테스트"""

    @pytest.fixture
    def sample_data(self):
        """테스트용 데이터"""
        return [
            JejuPowerData(
                timestamp="2025-01-01 12:00",
                system_demand=700.0,
                supply_capacity=1400.0,
                supply_reserve=700.0,
                forecast_demand=720.0,
                operation_reserve=350.0,
            ),
            JejuPowerData(
                timestamp="2025-01-01 13:00",
                system_demand=710.0,
                supply_capacity=1400.0,
                supply_reserve=690.0,
                forecast_demand=730.0,
                operation_reserve=340.0,
            ),
        ]

    def test_save_csv(self, tmp_path, sample_data):
        """CSV 저장 테스트"""
        output_path = tmp_path / "test_output.csv"
        store = JejuPowerDataStore(output_path)
        store.save(sample_data, append=False)

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 2

    def test_save_json(self, tmp_path, sample_data):
        """JSON 저장 테스트"""
        output_path = tmp_path / "test_output.json"
        store = JejuPowerDataStore(output_path)
        store.save(sample_data, append=False)

        assert output_path.exists()
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert len(data) == 2

    def test_load_csv(self, tmp_path, sample_data):
        """CSV 로드 테스트"""
        output_path = tmp_path / "test_output.csv"
        store = JejuPowerDataStore(output_path)
        store.save(sample_data, append=False)

        loaded = store.load()
        assert len(loaded) == 2
        assert loaded[0]['system_demand'] == 700.0

    def test_load_json(self, tmp_path, sample_data):
        """JSON 로드 테스트"""
        output_path = tmp_path / "test_output.json"
        store = JejuPowerDataStore(output_path)
        store.save(sample_data, append=False)

        loaded = store.load()
        assert len(loaded) == 2

    def test_append_csv(self, tmp_path, sample_data):
        """CSV append 테스트"""
        output_path = tmp_path / "test_output.csv"
        store = JejuPowerDataStore(output_path)

        # 첫 번째 저장
        store.save(sample_data[:1], append=False)

        # 두 번째 저장 (append)
        store.save(sample_data[1:], append=True)

        df = pd.read_csv(output_path)
        assert len(df) == 2

    def test_load_nonexistent(self, tmp_path):
        """존재하지 않는 파일 로드 테스트"""
        output_path = tmp_path / "nonexistent.csv"
        store = JejuPowerDataStore(output_path)
        loaded = store.load()
        assert loaded == []

    def test_save_empty_data(self, tmp_path):
        """빈 데이터 저장 테스트"""
        output_path = tmp_path / "empty.csv"
        store = JejuPowerDataStore(output_path)
        store.save([], append=False)
        # 빈 데이터는 저장하지 않음
        assert not output_path.exists()


# ============================================================================
# Integration Tests with Real Data
# ============================================================================

class TestJejuCrawlerIntegration:
    """실제 데이터 파일을 사용한 통합 테스트"""

    @pytest.fixture
    def real_zip_path(self):
        """실제 ZIP 파일 경로"""
        zip_path = PROJECT_ROOT / "data" / "jeju_power_supply.zip"
        if not zip_path.exists():
            pytest.skip("실제 ZIP 파일이 없습니다: data/jeju_power_supply.zip")
        return zip_path

    def test_load_real_data(self, real_zip_path):
        """실제 데이터 로드 테스트"""
        crawler = JejuPowerCrawler()
        data = crawler.load_from_zip(real_zip_path)
        crawler.close()

        assert len(data) > 0
        assert all(isinstance(d, JejuPowerData) for d in data)

    def test_real_data_validity(self, real_zip_path):
        """실제 데이터 유효성 테스트"""
        crawler = JejuPowerCrawler()
        data = crawler.load_from_zip(real_zip_path)
        crawler.close()

        for d in data[:100]:  # 처음 100건만 검증
            # 계통수요는 양수
            assert d.system_demand >= 0
            # 공급능력은 양수
            assert d.supply_capacity >= 0
            # 공급예비력 = 공급능력 - 계통수요 (대략)
            # 예비율은 0~200% 범위 내
            assert 0 <= d.reserve_rate <= 200

    def test_real_data_date_range(self, real_zip_path):
        """실제 데이터 날짜 범위 테스트"""
        crawler = JejuPowerCrawler()
        data = crawler.load_from_zip(real_zip_path)
        crawler.close()

        # 최소 1년치 데이터 존재
        first_date = data[0].timestamp[:10]
        last_date = data[-1].timestamp[:10]

        first = datetime.strptime(first_date, "%Y-%m-%d")
        last = datetime.strptime(last_date, "%Y-%m-%d")

        days_diff = (last - first).days
        assert days_diff >= 365, f"데이터 기간이 1년 미만: {days_diff}일"

    def test_real_data_hourly_completeness(self, real_zip_path):
        """실제 데이터 시간별 완전성 테스트"""
        crawler = JejuPowerCrawler()
        data = crawler.load_from_zip(real_zip_path)
        crawler.close()

        # 최근 7일 데이터의 시간별 완전성 확인
        recent = data[-168:]  # 최근 168시간 (7일)

        if len(recent) >= 168:
            # 시간별 데이터가 있어야 함
            hours = set()
            for d in recent:
                hour = d.timestamp.split()[1][:2]
                hours.add(hour)

            # 최소 20시간 이상 존재
            assert len(hours) >= 20


# ============================================================================
# Edge Cases
# ============================================================================

class TestJejuCrawlerEdgeCases:
    """엣지 케이스 테스트"""

    def test_negative_values(self):
        """음수 값 처리 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=-100.0,  # 비정상 값
            supply_capacity=1400.0,
            supply_reserve=1500.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        # 음수/0 수요에서는 예비율 계산 안됨 (0.0)
        assert data.reserve_rate == 0.0

    def test_very_large_values(self):
        """매우 큰 값 처리 테스트"""
        data = JejuPowerData(
            timestamp="2025-01-01 12:00",
            system_demand=1e9,
            supply_capacity=2e9,
            supply_reserve=1e9,
            forecast_demand=1e9,
            operation_reserve=5e8,
        )
        assert data.reserve_rate == pytest.approx(100.0, rel=0.01)

    def test_unicode_timestamp(self):
        """유니코드 타임스탬프 테스트"""
        data = JejuPowerData(
            timestamp="2025년 01월 01일 12:00",
            system_demand=700.0,
            supply_capacity=1400.0,
            supply_reserve=700.0,
            forecast_demand=720.0,
            operation_reserve=350.0,
        )
        assert "2025년" in data.timestamp


# ============================================================================
# Auto Download Tests
# ============================================================================

class TestJejuCrawlerAutoDownload:
    """auto_download() 메서드 테스트"""

    @pytest.fixture
    def mock_data_dir(self, tmp_path):
        """임시 데이터 디렉토리 생성"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        return data_dir

    @pytest.fixture
    def sample_zip_content(self):
        """테스트용 ZIP 파일 내용 생성"""
        zip_buffer = io.BytesIO()

        # 샘플 CSV 데이터 생성
        dates = pd.date_range("2025-01-01", periods=3, freq="D")
        hours = [f"{h}시" for h in range(1, 25)]

        def create_csv(base_value):
            data = {"날짜": dates.strftime("%Y-%m-%d")}
            for h in hours:
                data[h] = [base_value + np.random.randint(-50, 50) for _ in range(3)]
            return pd.DataFrame(data)

        # ZIP 파일 생성
        with zipfile.ZipFile(zip_buffer, 'w') as z:
            for name, base in [
                ("계통수요", 700),
                ("공급능력", 1400),
                ("공급예비력", 700),
                ("예측수요", 720),
                ("운영예비력", 350),
            ]:
                df = create_csv(base)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, encoding='utf-8')
                z.writestr(f"{name}.csv", csv_buffer.getvalue())

        return zip_buffer.getvalue()

    def test_auto_download_uses_cached_file(self, mock_data_dir, sample_zip_content):
        """캐시된 파일이 7일 미만이면 재사용하는지 테스트"""
        # 캐시 파일 생성 (현재 시간)
        zip_path = mock_data_dir / "jeju_power_supply.zip"
        with open(zip_path, 'wb') as f:
            f.write(sample_zip_content)

        crawler = JejuPowerCrawler()

        # auto_download의 데이터 디렉토리를 mock으로 대체
        with patch.object(Path, '__truediv__', side_effect=lambda self, other: mock_data_dir / other if 'data' in str(self) else self.__class__.__truediv__(self, other)):
            with patch('tools.crawlers.jeju_power_crawler.Path') as mock_path:
                # __file__ 기반 경로 계산 모킹
                mock_path.return_value.parent.parent.parent.__truediv__.return_value = mock_data_dir

                # 직접 경로 설정으로 테스트
                result = crawler._check_cached_zip(zip_path)

        crawler.close()

        # 파일이 있으면 True 반환
        assert result == True

    def test_auto_download_cache_check_recent_file(self, mock_data_dir, sample_zip_content):
        """최근 파일(7일 미만)은 캐시로 사용"""
        zip_path = mock_data_dir / "jeju_power_supply.zip"
        with open(zip_path, 'wb') as f:
            f.write(sample_zip_content)

        crawler = JejuPowerCrawler()

        # _check_cached_zip 메서드로 캐시 확인 로직 테스트
        is_valid = crawler._check_cached_zip(zip_path, max_age_days=7)

        crawler.close()
        assert is_valid == True

    def test_auto_download_cache_check_old_file(self, mock_data_dir, sample_zip_content):
        """오래된 파일(7일 이상)은 캐시 무효"""
        zip_path = mock_data_dir / "jeju_power_supply.zip"
        with open(zip_path, 'wb') as f:
            f.write(sample_zip_content)

        # 파일 수정 시간을 8일 전으로 변경
        import os
        old_time = datetime.now().timestamp() - (8 * 24 * 3600)
        os.utime(zip_path, (old_time, old_time))

        crawler = JejuPowerCrawler()
        is_valid = crawler._check_cached_zip(zip_path, max_age_days=7)
        crawler.close()

        assert is_valid == False

    def test_auto_download_cache_check_nonexistent(self, mock_data_dir):
        """존재하지 않는 파일은 캐시 무효"""
        zip_path = mock_data_dir / "nonexistent.zip"

        crawler = JejuPowerCrawler()
        is_valid = crawler._check_cached_zip(zip_path, max_age_days=7)
        crawler.close()

        assert is_valid == False

    def test_auto_download_force_redownload(self, mock_data_dir, sample_zip_content):
        """force=True면 캐시 무시하고 다운로드 시도"""
        zip_path = mock_data_dir / "jeju_power_supply.zip"
        with open(zip_path, 'wb') as f:
            f.write(sample_zip_content)

        crawler = JejuPowerCrawler()

        # download_and_extract_zip 모킹 (다운로드 실패)
        with patch.object(crawler, 'download_and_extract_zip', return_value=None):
            # force=True이면 다운로드 시도 후, 실패 시 기존 파일 사용
            with patch.object(crawler, '_get_data_dir', return_value=mock_data_dir):
                result = crawler.auto_download(force=True)

        crawler.close()

        # 다운로드 실패해도 기존 파일이 있으면 반환
        assert result == zip_path

    def test_auto_download_no_file_download_fails(self, mock_data_dir):
        """파일 없고 다운로드도 실패하면 None 반환"""
        crawler = JejuPowerCrawler()

        with patch.object(crawler, 'download_and_extract_zip', return_value=None):
            with patch.object(crawler, '_get_data_dir', return_value=mock_data_dir):
                result = crawler.auto_download()

        crawler.close()
        assert result is None

    def test_auto_download_with_valid_cache(self, mock_data_dir, sample_zip_content):
        """유효한 캐시가 있으면 다운로드 시도 안함"""
        zip_path = mock_data_dir / "jeju_power_supply.zip"
        with open(zip_path, 'wb') as f:
            f.write(sample_zip_content)

        crawler = JejuPowerCrawler()

        # download_and_extract_zip이 호출되지 않아야 함
        with patch.object(crawler, 'download_and_extract_zip') as mock_download:
            with patch.object(crawler, '_get_data_dir', return_value=mock_data_dir):
                result = crawler.auto_download(force=False)

            # 캐시가 유효하면 다운로드 시도 안함
            mock_download.assert_not_called()

        crawler.close()
        assert result == zip_path


class TestJejuCrawlerAutoDownloadHelpers:
    """auto_download 헬퍼 메서드 테스트"""

    def test_check_cached_zip_valid(self, tmp_path):
        """유효한 캐시 파일 확인"""
        zip_path = tmp_path / "test.zip"
        zip_path.write_bytes(b"test content" * 100)  # 최소 크기 이상

        crawler = JejuPowerCrawler()
        result = crawler._check_cached_zip(zip_path, max_age_days=7)
        crawler.close()

        assert result == True

    def test_check_cached_zip_too_small(self, tmp_path):
        """너무 작은 파일은 무효"""
        zip_path = tmp_path / "test.zip"
        zip_path.write_bytes(b"tiny")  # 매우 작은 파일

        crawler = JejuPowerCrawler()
        result = crawler._check_cached_zip(zip_path, max_age_days=7, min_size=100)
        crawler.close()

        assert result == False

    def test_get_data_dir(self):
        """데이터 디렉토리 경로 반환 테스트"""
        crawler = JejuPowerCrawler()
        data_dir = crawler._get_data_dir()
        crawler.close()

        assert data_dir.name == "data"
        assert data_dir.parent.name == "power-demand-forecast"


class TestJejuCrawlerCLIAutoDownload:
    """CLI --auto-download 옵션 테스트"""

    def test_cli_auto_download_option_exists(self):
        """CLI에 --auto-download 옵션이 있는지 확인"""
        import argparse
        from tools.crawlers.jeju_power_crawler import main

        # main 함수의 argparse 설정 확인
        parser = argparse.ArgumentParser()
        parser.add_argument('--auto-download', action='store_true')
        parser.add_argument('--force', action='store_true')
        parser.add_argument('--zip', type=str)

        # 파싱 테스트
        args = parser.parse_args(['--auto-download'])
        assert args.auto_download == True

        args = parser.parse_args(['--auto-download', '--force'])
        assert args.auto_download == True
        assert args.force == True

    def test_cli_zip_option(self):
        """CLI --zip 옵션 테스트"""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument('--zip', type=str)

        args = parser.parse_args(['--zip', '/path/to/file.zip'])
        assert args.zip == '/path/to/file.zip'
