"""
공공데이터포털 기상청 ASOS API 클라이언트
https://www.data.go.kr/data/15057210/openapi.do

Selenium 크롤러보다 더 안정적이고 빠릅니다.
단, API 키가 필요합니다 (공공데이터포털에서 무료 발급).

사용법:
    api = KMAAPI(api_key="your_api_key")
    df = api.get_daily_data(
        station_id="184",  # 제주
        start_date="2024-01-01",
        end_date="2024-12-31"
    )
    df.to_csv("data/raw/jeju_2024.csv", index=False)
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from io import StringIO

from .config import STATION_CODES, get_station_code


class KMAAPI:
    """공공데이터포털 기상청 ASOS API 클라이언트"""

    # API 엔드포인트
    BASE_URL = "http://apis.data.go.kr/1360000"
    ENDPOINTS = {
        "hourly": "/AsosHourlyInfoService/getWthrDataList",
        "daily": "/AsosDalyInfoService/getWthrDataList",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: 공공데이터포털 API 키. None이면 환경변수 KMA_API_KEY 사용
        """
        self.api_key = api_key or os.environ.get("KMA_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API 키가 필요합니다.\n"
                "1. 공공데이터포털(data.go.kr)에서 API 키 발급\n"
                "2. 환경변수 KMA_API_KEY 설정 또는 api_key 인자로 전달"
            )

    def _request(
        self,
        endpoint: str,
        params: Dict[str, Any],
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """API 요청"""
        url = self.BASE_URL + endpoint

        params["serviceKey"] = self.api_key
        params["dataType"] = "JSON"

        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                result_code = data.get("response", {}).get("header", {}).get("resultCode")
                if result_code != "00":
                    result_msg = data.get("response", {}).get("header", {}).get("resultMsg")
                    raise ValueError(f"API 오류: [{result_code}] {result_msg}")

                return data

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"API 요청 실패: {e}")

    def get_hourly_data(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
        start_hour: str = "00",
        end_hour: str = "23",
    ) -> pd.DataFrame:
        """시간별 기상 데이터 조회"""
        start_dt = start_date.replace("-", "")
        end_dt = end_date.replace("-", "")

        all_data = []
        page_no = 1
        num_of_rows = 999

        while True:
            params = {
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "dataCd": "ASOS",
                "dateCd": "HR",
                "startDt": start_dt,
                "startHh": start_hour,
                "endDt": end_dt,
                "endHh": end_hour,
                "stnIds": station_id,
            }

            data = self._request(self.ENDPOINTS["hourly"], params)

            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            if not items:
                break

            if isinstance(items, dict):
                items = [items]

            all_data.extend(items)

            total_count = data.get("response", {}).get("body", {}).get("totalCount", 0)
            if len(all_data) >= total_count:
                break

            page_no += 1
            time.sleep(0.2)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        return self._process_dataframe(df)

    def get_daily_data(
        self,
        station_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """일별 기상 데이터 조회"""
        start_dt = start_date.replace("-", "")
        end_dt = end_date.replace("-", "")

        all_data = []
        page_no = 1
        num_of_rows = 999

        while True:
            params = {
                "pageNo": page_no,
                "numOfRows": num_of_rows,
                "dataCd": "ASOS",
                "dateCd": "DAY",
                "startDt": start_dt,
                "endDt": end_dt,
                "stnIds": station_id,
            }

            data = self._request(self.ENDPOINTS["daily"], params)

            items = data.get("response", {}).get("body", {}).get("items", {}).get("item", [])
            if not items:
                break

            if isinstance(items, dict):
                items = [items]

            all_data.extend(items)

            total_count = data.get("response", {}).get("body", {}).get("totalCount", 0)
            if len(all_data) >= total_count:
                break

            page_no += 1
            time.sleep(0.2)

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data)
        return self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrame 후처리"""
        if df.empty:
            return df

        column_mapping = {
            "tm": "일시",
            "stnId": "지점",
            "stnNm": "지점명",
            "avgTa": "평균기온",
            "minTa": "최저기온",
            "maxTa": "최고기온",
            "avgRhm": "평균습도",
            "minRhm": "최저습도",
            "sumRn": "일강수량",
            "avgWs": "평균풍속",
            "maxWs": "최대풍속",
            "maxWd": "최대풍속풍향",
            "avgPs": "평균해면기압",
            "avgPa": "평균현지기압",
            "sumSs": "합계일조시간",
            "sumGsr": "합계일사량",
            "avgTd": "평균이슬점온도",
            "avgTs": "평균지면온도",
            "minTg": "최저초상온도",
            "avgCa": "평균전운량",
            "sumDpthFhsc": "합계적설",
            "maxDc": "최심적설",
            "avgLmac": "평균중하층운량",
            "sumFogDur": "안개지속시간",
            "ta": "기온",
            "rn": "강수량",
            "ws": "풍속",
            "wd": "풍향",
            "hm": "습도",
            "pa": "현지기압",
            "ps": "해면기압",
            "td": "이슬점온도",
            "pv": "증기압",
            "vs": "시정",
            "ss": "일조",
            "ts": "지면온도",
            "icsr": "일사",
            "dc10Tca": "전운량",
            "dc10LmcsCa": "중하층운량",
        }

        df = df.rename(columns=column_mapping)

        numeric_columns = [
            "평균기온", "최저기온", "최고기온", "평균습도", "최저습도",
            "일강수량", "평균풍속", "최대풍속", "평균해면기압", "평균현지기압",
            "합계일조시간", "합계일사량", "평균이슬점온도", "평균지면온도",
            "최저초상온도", "평균전운량", "합계적설", "최심적설",
            "기온", "강수량", "풍속", "습도", "현지기압", "해면기압",
            "이슬점온도", "증기압", "시정", "일조", "지면온도", "일사",
        ]

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "일시" in df.columns:
            df = df.sort_values("일시").reset_index(drop=True)

        return df

    def download(
        self,
        station: str,
        start_date: str,
        end_date: str,
        data_type: str = "일",
        output_dir: str = "data/raw",
        output_filename: Optional[str] = None,
    ) -> Optional[str]:
        """기상 데이터 다운로드 및 CSV 저장"""
        if station.isdigit():
            station_id = station
            station_name = f"stn{station}"
        else:
            station_id = get_station_code(station)
            station_name = station

        print(f"[KMAAPI] 데이터 다운로드 시작")
        print(f"  지점: {station_name} ({station_id})")
        print(f"  기간: {start_date} ~ {end_date}")
        print(f"  자료형태: {data_type}")

        try:
            if data_type == "시간":
                df = self.get_hourly_data(station_id, start_date, end_date)
            else:
                df = self.get_daily_data(station_id, start_date, end_date)

            if df.empty:
                print("[KMAAPI] 데이터가 없습니다.")
                return None

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if not output_filename:
                output_filename = f"{station_name}_{start_date}_{end_date}_{data_type}.csv"
                output_filename = output_filename.replace("-", "")

            filepath = output_path / output_filename
            df.to_csv(filepath, index=False, encoding="utf-8-sig")

            print(f"[KMAAPI] 다운로드 완료: {filepath}")
            print(f"  총 {len(df)} 행")

            return str(filepath)

        except Exception as e:
            print(f"[KMAAPI] 에러 발생: {e}")
            return None

    def download_batch(
        self,
        station: str,
        start_year: int,
        end_year: int,
        data_type: str = "일",
        output_dir: str = "data/raw",
    ) -> List[str]:
        """연도별 일괄 다운로드"""
        downloaded_files = []

        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            if year == datetime.now().year:
                yesterday = datetime.now() - timedelta(days=1)
                end_date = yesterday.strftime("%Y-%m-%d")

            print(f"\n[Batch] {year}년 다운로드...")

            filepath = self.download(
                station=station,
                start_date=start_date,
                end_date=end_date,
                data_type=data_type,
                output_dir=output_dir,
            )

            if filepath:
                downloaded_files.append(filepath)

            time.sleep(0.5)

        return downloaded_files


def main():
    """테스트"""
    api_key = os.environ.get("KMA_API_KEY")
    if not api_key:
        print("KMA_API_KEY 환경변수를 설정해주세요.")
        print("예: export KMA_API_KEY='your_api_key'")
        return

    api = KMAAPI(api_key)

    result = api.download(
        station="제주",
        start_date="2024-01-01",
        end_date="2024-12-31",
        data_type="일",
    )

    if result:
        print(f"\n성공: {result}")


if __name__ == "__main__":
    main()
