#!/usr/bin/env python3
"""
제주 전력수급현황 크롤러
========================

공공데이터포털 및 전력거래소에서 제주도 전력 수급 데이터를 수집합니다.

데이터 소스:
1. 공공데이터포털 - 한국전력거래소_제주 전력수급현황
   https://www.data.go.kr/data/15125113/fileData.do
   - 제주 계통수요, 공급능력, 공급예비력, 예측수요, 운영예비력 (MW)
   - 시간별 데이터

2. 공공데이터포털 - 한국전력거래소_시간별 제주전력수요
   https://www.data.go.kr/data/15065239/fileData.do
   - 일별 1~24시 전력수요량 (MWh)

3. 전력거래소 실시간 전력수급현황 페이지 (제주 데이터 포함 시)
   https://new.kpx.or.kr/powerinfoSubmain.es

Usage:
    # 파일 데이터 다운로드 및 파싱
    python jeju_power_crawler.py --download

    # 최신 데이터 조회
    python jeju_power_crawler.py --latest

    # 특정 기간 데이터 조회
    python jeju_power_crawler.py --start 2024-01-01 --end 2024-12-31

Author: Claude Code
Created: 2025-12-17
"""

import requests
import pandas as pd
import json
import re
import time
import logging
import argparse
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any, Union, Tuple
from dataclasses import dataclass, asdict
import io

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class JejuPowerData:
    """제주 전력 수급 데이터 구조"""
    timestamp: str              # 데이터 시점 (YYYY-MM-DD HH:MM)
    system_demand: float        # 제주 계통수요 (MW)
    supply_capacity: float      # 제주 공급능력 (MW)
    supply_reserve: float       # 제주 공급예비력 (MW)
    forecast_demand: float      # 제주 예측수요 (MW)
    operation_reserve: float    # 제주 운영예비력 (MW)
    reserve_rate: float = 0.0   # 예비율 (%) - 계산값
    fetched_at: str = ""        # 크롤링 시점
    source: str = "data.go.kr"  # 데이터 소스

    def __post_init__(self):
        if not self.fetched_at:
            self.fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 예비율 계산 (공급예비력 / 계통수요 * 100)
        if self.system_demand > 0 and self.reserve_rate == 0.0:
            self.reserve_rate = (self.supply_reserve / self.system_demand) * 100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def current_demand(self) -> float:
        """EPSIS 호환용 - 현재수요"""
        return self.system_demand

    @property
    def reserve_power(self) -> float:
        """EPSIS 호환용 - 예비력"""
        return self.supply_reserve


class JejuPowerCrawler:
    """제주 전력수급현황 크롤러"""

    # 공공데이터포털 파일 데이터 URL
    DATA_PORTAL_BASE = "https://www.data.go.kr"

    # 제주 전력수급현황 데이터셋 ID
    JEJU_SUPPLY_DATASET_ID = "15125113"  # 제주 전력수급현황
    JEJU_DEMAND_DATASET_ID = "15065239"  # 시간별 제주전력수요

    # 전력거래소 실시간 페이지
    KPX_REALTIME_URL = "https://new.kpx.or.kr/powerinfoSubmain.es"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        cache_dir: Optional[Path] = None
    ):
        """
        Args:
            api_key: 공공데이터포털 API 키 (선택)
            timeout: 요청 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            cache_dir: 캐시 디렉토리
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """HTTP 세션 생성"""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
        })
        return session

    def load_from_zip(self, zip_path: Union[str, Path]) -> List[JejuPowerData]:
        """
        공공데이터포털에서 다운로드한 ZIP 파일에서 데이터 로드

        ZIP 파일 구조:
        - 계통수요.csv: 날짜 + 1~24시 (MW)
        - 공급능력.csv: 날짜 + 1~24시 (MW)
        - 공급예비력.csv: 날짜 + 1~24시 (MW)
        - 예측수요.csv: 거래일 + 1~24시 (MW)
        - 운영예비력.csv: 날짜 + 1~24시 (MW)

        Args:
            zip_path: ZIP 파일 경로

        Returns:
            JejuPowerData 리스트 (시간별 데이터)
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            logger.error(f"ZIP 파일 없음: {zip_path}")
            return []

        # ZIP 파일에서 각 CSV 추출
        dataframes = {}
        file_mapping = {
            '계통수요': 'system_demand',
            '공급능력': 'supply_capacity',
            '공급예비력': 'supply_reserve',
            '예측수요': 'forecast_demand',
            '운영예비력': 'operation_reserve',
        }

        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                for info in z.infolist():
                    # 파일명 인코딩 수정 (한글)
                    try:
                        filename = info.filename.encode('cp437').decode('euc-kr')
                    except:
                        filename = info.filename

                    base_name = filename.replace('.csv', '')

                    if base_name not in file_mapping:
                        continue

                    data = z.read(info.filename)

                    # CSV 파싱
                    for enc in ['utf-8', 'euc-kr', 'cp949']:
                        try:
                            df = pd.read_csv(io.BytesIO(data), encoding=enc)
                            dataframes[file_mapping[base_name]] = df
                            logger.debug(f"로드: {filename} ({len(df)}행)")
                            break
                        except:
                            continue

            if len(dataframes) < 3:
                logger.error(f"필수 파일 부족: {list(dataframes.keys())}")
                return []

            logger.info(f"ZIP에서 {len(dataframes)}개 파일 로드 완료")
            return self._merge_hourly_data(dataframes)

        except Exception as e:
            logger.error(f"ZIP 처리 실패: {e}")
            return []

    def _merge_hourly_data(self, dataframes: Dict[str, pd.DataFrame]) -> List[JejuPowerData]:
        """
        5개 CSV 파일을 시간별 데이터로 병합

        각 파일은 날짜 + 1~24시 컬럼 구조
        """
        result = []

        # 기준 데이터프레임 (계통수요)
        base_df = dataframes.get('system_demand')
        if base_df is None:
            logger.error("계통수요 데이터 없음")
            return result

        # 날짜 컬럼 찾기
        date_col = None
        for col in ['날짜', '거래일', 'date']:
            if col in base_df.columns:
                date_col = col
                break

        if date_col is None:
            logger.error(f"날짜 컬럼 없음: {list(base_df.columns)}")
            return result

        # 시간 컬럼 매핑 (1시~24시)
        hour_cols = {}
        for col in base_df.columns:
            col_clean = col.strip()
            for h in range(1, 25):
                if col_clean in [f'{h}시', f' {h}시 ', f'{h}시 ']:
                    hour_cols[h] = col
                    break

        logger.info(f"시간 컬럼 매핑: {len(hour_cols)}개")

        # 각 날짜, 각 시간별로 데이터 생성
        for idx, row in base_df.iterrows():
            date_str = str(row[date_col])

            for hour in range(1, 25):
                if hour not in hour_cols:
                    continue

                hour_col = hour_cols[hour]

                # 타임스탬프 생성 (24시 -> 다음날 00시로 처리)
                if hour == 24:
                    ts = f"{date_str} 00:00"
                else:
                    ts = f"{date_str} {hour:02d}:00"

                try:
                    # 각 데이터프레임에서 해당 시간 값 추출
                    system_demand = self._get_value(dataframes, 'system_demand', idx, hour_col)
                    supply_capacity = self._get_value(dataframes, 'supply_capacity', idx, hour_col)
                    supply_reserve = self._get_value(dataframes, 'supply_reserve', idx, hour_col)
                    forecast_demand = self._get_value(dataframes, 'forecast_demand', idx, hour_col)
                    operation_reserve = self._get_value(dataframes, 'operation_reserve', idx, hour_col)

                    if system_demand > 0:
                        data = JejuPowerData(
                            timestamp=ts,
                            system_demand=system_demand,
                            supply_capacity=supply_capacity,
                            supply_reserve=supply_reserve,
                            forecast_demand=forecast_demand,
                            operation_reserve=operation_reserve,
                            source="data.go.kr"
                        )
                        result.append(data)

                except Exception as e:
                    logger.debug(f"행 처리 실패: {e}")
                    continue

        # 타임스탬프 기준 정렬
        result.sort(key=lambda x: x.timestamp)
        logger.info(f"총 {len(result)}건 시간별 데이터 생성")
        return result

    def _get_value(
        self,
        dataframes: Dict[str, pd.DataFrame],
        key: str,
        idx: int,
        hour_col: str
    ) -> float:
        """데이터프레임에서 값 추출"""
        df = dataframes.get(key)
        if df is None:
            return 0.0

        # 컬럼명 공백 처리
        for col in df.columns:
            if col.strip() == hour_col.strip():
                try:
                    return float(df.iloc[idx][col])
                except:
                    return 0.0

        return 0.0

    def download_and_extract_zip(self, save_dir: Optional[Path] = None) -> Optional[Path]:
        """
        공공데이터포털에서 ZIP 파일 다운로드

        Returns:
            저장된 ZIP 파일 경로 또는 None
        """
        save_dir = save_dir or self.cache_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 데이터 페이지에서 다운로드 링크 추출
            file_url = f"{self.DATA_PORTAL_BASE}/data/{self.JEJU_SUPPLY_DATASET_ID}/fileData.do"
            response = self.session.get(file_url, timeout=self.timeout)
            response.raise_for_status()

            # 다운로드 링크 추출
            pattern = r'fileDownload\.do\?[^"\']+atchFileId=[^"\'&]+'
            matches = re.findall(pattern, response.text)

            if not matches:
                logger.error("다운로드 링크를 찾을 수 없습니다.")
                return None

            download_url = f"{self.DATA_PORTAL_BASE}/cmm/cmm/{matches[0]}"
            logger.info(f"다운로드 URL: {download_url}")

            # 파일 다운로드
            self.session.headers['Referer'] = file_url
            resp = self.session.get(download_url, timeout=60)
            resp.raise_for_status()

            # ZIP 파일 저장
            zip_path = save_dir / "jeju_power_supply.zip"
            with open(zip_path, 'wb') as f:
                f.write(resp.content)

            logger.info(f"ZIP 파일 저장: {zip_path} ({len(resp.content)} bytes)")
            return zip_path

        except Exception as e:
            logger.error(f"ZIP 다운로드 실패: {e}")
            return None

    def fetch_realtime_data(self) -> List[JejuPowerData]:
        """
        실시간 데이터 조회 (캐시된 데이터 또는 다운로드)

        Returns:
            JejuPowerData 리스트
        """
        # 먼저 캐시된 ZIP 파일 확인
        cached_zip = self.cache_dir / "jeju_power_supply.zip"

        if cached_zip.exists():
            # 캐시 파일이 1일 이내인 경우 사용
            file_age = datetime.now().timestamp() - cached_zip.stat().st_mtime
            if file_age < 86400:  # 24시간
                logger.info("캐시된 ZIP 파일 사용")
                return self.load_from_zip(cached_zip)

        # 새로 다운로드
        zip_path = self.download_and_extract_zip()
        if zip_path:
            return self.load_from_zip(zip_path)

        return []

    def download_csv_data(
        self,
        dataset_id: str = None,
        save_path: Optional[Path] = None
    ) -> Optional[pd.DataFrame]:
        """
        공공데이터포털에서 CSV 파일 다운로드

        Args:
            dataset_id: 데이터셋 ID (기본: 제주 전력수급현황)
            save_path: 저장 경로

        Returns:
            DataFrame 또는 None
        """
        dataset_id = dataset_id or self.JEJU_SUPPLY_DATASET_ID

        # 파일 데이터 페이지에서 다운로드 링크 추출
        try:
            # 공공데이터포털 파일 다운로드 URL 패턴
            # 실제 다운로드는 인증이 필요할 수 있음
            file_url = f"{self.DATA_PORTAL_BASE}/data/{dataset_id}/fileData.do"

            logger.info(f"데이터 페이지 접근: {file_url}")

            response = self.session.get(file_url, timeout=self.timeout)
            response.raise_for_status()

            # 페이지에서 실제 다운로드 링크 추출
            download_link = self._extract_download_link(response.text)

            if download_link:
                logger.info(f"다운로드 링크 발견: {download_link}")
                return self._download_and_parse_csv(download_link, save_path)
            else:
                logger.warning("다운로드 링크를 찾을 수 없습니다.")
                return None

        except Exception as e:
            logger.error(f"CSV 다운로드 실패: {e}")
            return None

    def _extract_download_link(self, html: str) -> Optional[str]:
        """HTML에서 다운로드 링크 추출"""
        # 공공데이터포털 다운로드 링크 패턴
        patterns = [
            r'href=["\']([^"\']*fileDownload[^"\']*)["\']',
            r'href=["\']([^"\']*\.csv[^"\']*)["\']',
            r'data-url=["\']([^"\']*)["\']',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                link = matches[0]
                if not link.startswith('http'):
                    link = f"{self.DATA_PORTAL_BASE}{link}"
                return link

        return None

    def _download_and_parse_csv(
        self,
        url: str,
        save_path: Optional[Path] = None
    ) -> Optional[pd.DataFrame]:
        """CSV 다운로드 및 파싱"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            # 인코딩 감지
            encoding = response.encoding or 'utf-8'
            if 'euc-kr' in response.headers.get('Content-Type', '').lower():
                encoding = 'euc-kr'

            # CSV 파싱
            df = pd.read_csv(
                io.StringIO(response.text),
                encoding=encoding
            )

            if save_path:
                df.to_csv(save_path, index=False, encoding='utf-8')
                logger.info(f"CSV 저장: {save_path}")

            return df

        except Exception as e:
            logger.error(f"CSV 파싱 실패: {e}")
            return None

    def load_cached_data(self, filename: str = "jeju_power_supply.csv") -> Optional[pd.DataFrame]:
        """캐시된 데이터 로드"""
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path)
                logger.info(f"캐시 데이터 로드: {cache_path}")
                return df
            except Exception as e:
                logger.error(f"캐시 로드 실패: {e}")
        return None

    def parse_supply_data(self, df: pd.DataFrame) -> List[JejuPowerData]:
        """
        제주 전력수급현황 DataFrame을 JejuPowerData 리스트로 변환

        예상 컬럼: 기준일시, 계통수요, 공급능력, 공급예비력, 예측수요, 운영예비력
        """
        result = []

        # 컬럼명 정규화
        column_mapping = {
            '기준일시': 'timestamp',
            '기준시간': 'timestamp',
            '일시': 'timestamp',
            '계통수요': 'system_demand',
            '제주 계통수요': 'system_demand',
            '공급능력': 'supply_capacity',
            '제주 공급능력': 'supply_capacity',
            '공급예비력': 'supply_reserve',
            '제주 공급예비력': 'supply_reserve',
            '예측수요': 'forecast_demand',
            '제주 예측수요': 'forecast_demand',
            '운영예비력': 'operation_reserve',
            '제주 운영예비력': 'operation_reserve',
        }

        # 컬럼명 변경
        df_renamed = df.rename(columns=column_mapping)

        required_cols = ['timestamp', 'system_demand', 'supply_capacity', 'supply_reserve']
        missing_cols = [col for col in required_cols if col not in df_renamed.columns]

        if missing_cols:
            logger.warning(f"필수 컬럼 누락: {missing_cols}")
            logger.info(f"사용 가능한 컬럼: {list(df_renamed.columns)}")
            return result

        for _, row in df_renamed.iterrows():
            try:
                data = JejuPowerData(
                    timestamp=str(row.get('timestamp', '')),
                    system_demand=float(row.get('system_demand', 0)),
                    supply_capacity=float(row.get('supply_capacity', 0)),
                    supply_reserve=float(row.get('supply_reserve', 0)),
                    forecast_demand=float(row.get('forecast_demand', 0)),
                    operation_reserve=float(row.get('operation_reserve', 0)),
                )
                result.append(data)
            except Exception as e:
                logger.debug(f"행 파싱 실패: {e}")
                continue

        logger.info(f"{len(result)}건 데이터 파싱 완료")
        return result

    def fetch_from_local_file(self, file_path: Union[str, Path]) -> List[JejuPowerData]:
        """
        로컬 CSV/Excel 파일에서 데이터 로드

        공공데이터포털에서 수동 다운로드한 파일 사용
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"파일 없음: {file_path}")
            return []

        try:
            if file_path.suffix.lower() == '.csv':
                # 다양한 인코딩 시도
                for encoding in ['utf-8', 'euc-kr', 'cp949']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                logger.error(f"지원하지 않는 파일 형식: {file_path.suffix}")
                return []

            logger.info(f"파일 로드 완료: {file_path} ({len(df)}행)")
            return self.parse_supply_data(df)

        except Exception as e:
            logger.error(f"파일 로드 실패: {e}")
            return []

    def fetch_realtime_from_kpx(self) -> Optional[JejuPowerData]:
        """
        전력거래소 실시간 페이지에서 제주 데이터 추출 (있는 경우)

        Note: 전력거래소 실시간 페이지는 주로 전국 데이터만 제공
              제주 데이터가 별도로 표시되는 경우에만 추출 가능
        """
        try:
            response = self.session.get(self.KPX_REALTIME_URL, timeout=self.timeout)
            response.raise_for_status()

            # 제주 관련 데이터 패턴 검색
            jeju_patterns = [
                r'제주.*?(\d+(?:,\d+)?(?:\.\d+)?)\s*MW',
                r'JEJU.*?(\d+(?:,\d+)?(?:\.\d+)?)',
            ]

            for pattern in jeju_patterns:
                matches = re.findall(pattern, response.text, re.IGNORECASE)
                if matches:
                    logger.info(f"제주 데이터 발견: {matches}")
                    # 추가 파싱 필요
                    break

            logger.info("전력거래소 실시간 페이지에서 제주 전용 데이터 없음")
            return None

        except Exception as e:
            logger.error(f"전력거래소 실시간 조회 실패: {e}")
            return None

    def get_latest_data(self, data: List[JejuPowerData]) -> Optional[JejuPowerData]:
        """최신 데이터 반환"""
        if not data:
            return None
        return sorted(data, key=lambda x: x.timestamp, reverse=True)[0]

    def get_data_by_date_range(
        self,
        data: List[JejuPowerData],
        start_date: str,
        end_date: str
    ) -> List[JejuPowerData]:
        """날짜 범위로 데이터 필터링"""
        filtered = []
        for item in data:
            try:
                ts = item.timestamp[:10]  # YYYY-MM-DD
                if start_date <= ts <= end_date:
                    filtered.append(item)
            except:
                continue
        return filtered

    def close(self):
        """세션 종료"""
        self.session.close()


class JejuPowerDataStore:
    """제주 전력 데이터 저장소"""

    def __init__(self, output_path: Union[str, Path]):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, data: List[JejuPowerData], append: bool = True):
        """데이터 저장"""
        if not data:
            logger.warning("저장할 데이터 없음")
            return

        df = pd.DataFrame([d.to_dict() for d in data])

        if self.output_path.suffix.lower() == '.csv':
            if append and self.output_path.exists():
                existing = pd.read_csv(self.output_path)
                df = pd.concat([existing, df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
            df.to_csv(self.output_path, index=False, encoding='utf-8')
        else:
            if append and self.output_path.exists():
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                existing.extend([d.to_dict() for d in data])
                # 중복 제거
                seen = set()
                unique = []
                for item in existing:
                    if item['timestamp'] not in seen:
                        seen.add(item['timestamp'])
                        unique.append(item)
                data_to_save = unique
            else:
                data_to_save = [d.to_dict() for d in data]

            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        logger.info(f"데이터 저장 완료: {self.output_path}")

    def load(self) -> List[Dict[str, Any]]:
        """데이터 로드"""
        if not self.output_path.exists():
            return []

        try:
            if self.output_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.output_path)
                return df.to_dict('records')
            else:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return []


def main():
    """CLI 메인 함수"""
    parser = argparse.ArgumentParser(description='제주 전력수급현황 크롤러')
    parser.add_argument('--download', action='store_true', help='공공데이터포털에서 다운로드')
    parser.add_argument('--file', type=str, help='로컬 파일에서 로드')
    parser.add_argument('--latest', action='store_true', help='최신 데이터 출력')
    parser.add_argument('--start', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, help='출력 파일 경로')
    parser.add_argument('--realtime', action='store_true', help='전력거래소 실시간 조회')

    args = parser.parse_args()

    crawler = JejuPowerCrawler()

    try:
        data = []

        if args.file:
            # 로컬 파일에서 로드
            data = crawler.fetch_from_local_file(args.file)
        elif args.download:
            # 공공데이터포털에서 다운로드
            df = crawler.download_csv_data()
            if df is not None:
                data = crawler.parse_supply_data(df)
        elif args.realtime:
            # 전력거래소 실시간
            result = crawler.fetch_realtime_from_kpx()
            if result:
                data = [result]

        if not data:
            print("데이터가 없습니다.")
            print("\n사용 예시:")
            print("  # 로컬 CSV 파일 로드")
            print("  python jeju_power_crawler.py --file data/jeju_power_supply.csv")
            print("\n  # 공공데이터포털에서 다운로드 (수동 다운로드 권장)")
            print("  # https://www.data.go.kr/data/15125113/fileData.do")
            return

        # 날짜 필터링
        if args.start and args.end:
            data = crawler.get_data_by_date_range(data, args.start, args.end)
            print(f"\n{args.start} ~ {args.end} 기간 데이터: {len(data)}건")

        # 최신 데이터 출력
        if args.latest or not (args.start and args.end):
            latest = crawler.get_latest_data(data)
            if latest:
                print("\n=== 제주 전력수급현황 (최신) ===")
                print(f"시점: {latest.timestamp}")
                print(f"계통수요: {latest.system_demand:,.1f} MW")
                print(f"공급능력: {latest.supply_capacity:,.1f} MW")
                print(f"공급예비력: {latest.supply_reserve:,.1f} MW")
                print(f"예비율: {latest.reserve_rate:.1f}%")
                print(f"예측수요: {latest.forecast_demand:,.1f} MW")
                print(f"운영예비력: {latest.operation_reserve:,.1f} MW")

        # 저장
        if args.output:
            store = JejuPowerDataStore(args.output)
            store.save(data, append=True)

        print(f"\n총 {len(data)}건 데이터 처리 완료")

    finally:
        crawler.close()


if __name__ == "__main__":
    main()
