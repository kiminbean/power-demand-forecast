"""
기상청 기상자료개방포털 크롤러
ASOS(종관기상관측) 데이터를 Selenium으로 크롤링합니다.

사용법:
    crawler = KMACrawler()
    crawler.download(
        station="제주",
        start_date="2013-01-01",
        end_date="2024-12-31",
        elements=["기온", "습도"],
        output_dir="data/raw"
    )
"""

import os
import time
import glob
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
)

try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False

from .config import (
    STATION_CODES,
    ELEMENT_CODES,
    DATA_TYPE_CODES,
    DEFAULT_CONFIG,
    get_station_code,
    get_element_codes,
)


class KMACrawler:
    """기상청 기상자료개방포털 크롤러"""

    def __init__(
        self,
        headless: bool = True,
        download_dir: Optional[str] = None,
        timeout: int = 30,
    ):
        """
        Args:
            headless: 헤드리스 모드 사용 여부
            download_dir: 다운로드 디렉토리 (기본: data/raw)
            timeout: 대기 시간 (초)
        """
        self.headless = headless
        self.timeout = timeout
        self.driver = None

        # 다운로드 디렉토리 설정
        if download_dir:
            self.download_dir = Path(download_dir).resolve()
        else:
            self.download_dir = Path.cwd() / "data" / "raw"
        self.download_dir.mkdir(parents=True, exist_ok=True)

        # 임시 다운로드 디렉토리 (Chrome 다운로드용)
        self.temp_download_dir = Path.cwd() / ".temp_downloads"
        self.temp_download_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = DEFAULT_CONFIG["base_url"]

    def _init_driver(self):
        """Selenium WebDriver 초기화"""
        chrome_options = Options()

        if self.headless:
            chrome_options.add_argument("--headless=new")

        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--lang=ko-KR")

        # 다운로드 설정
        prefs = {
            "download.default_directory": str(self.temp_download_dir),
            "download.prompt_for_download": False,
            "download.directory_upgrade": True,
            "safebrowsing.enabled": True,
        }
        chrome_options.add_experimental_option("prefs", prefs)

        # Chrome 경고 메시지 숨기기
        chrome_options.add_experimental_option("excludeSwitches", ["enable-logging"])

        try:
            if USE_WEBDRIVER_MANAGER:
                service = Service(ChromeDriverManager().install())
            else:
                # 시스템에 설치된 chromedriver 사용
                service = Service()

            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.implicitly_wait(10)
            print("[KMACrawler] WebDriver 초기화 완료")

        except Exception as e:
            raise RuntimeError(f"WebDriver 초기화 실패: {e}")

    def _close_driver(self):
        """WebDriver 종료"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("[KMACrawler] WebDriver 종료")

    def _wait_for_element(self, by: By, value: str, timeout: int = None) -> bool:
        """요소가 나타날 때까지 대기"""
        timeout = timeout or self.timeout
        try:
            WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((by, value))
            )
            return True
        except TimeoutException:
            return False

    def _wait_for_clickable(self, by: By, value: str, timeout: int = None):
        """클릭 가능해질 때까지 대기"""
        timeout = timeout or self.timeout
        return WebDriverWait(self.driver, timeout).until(
            EC.element_to_be_clickable((by, value))
        )

    def _click_element(self, element, retry: int = 3):
        """요소 클릭 (재시도 포함)"""
        for i in range(retry):
            try:
                element.click()
                return True
            except ElementClickInterceptedException:
                time.sleep(0.5)
                try:
                    # JavaScript로 클릭 시도
                    self.driver.execute_script("arguments[0].click();", element)
                    return True
                except Exception:
                    continue
        return False

    def _close_popup(self):
        """팝업 닫기"""
        try:
            # 일반적인 닫기 버튼들
            close_selectors = [
                "button.close",
                ".modal .close",
                ".popup-close",
                "button[onclick*='close']",
                ".btn-close",
            ]
            for selector in close_selectors:
                try:
                    close_btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if close_btn.is_displayed():
                        close_btn.click()
                        time.sleep(0.3)
                except NoSuchElementException:
                    continue
        except Exception:
            pass

    def _select_station(self, station_code: str):
        """지점 선택"""
        print(f"  - 지점 선택: {station_code}")

        # 지점 선택 버튼 클릭
        try:
            stn_btn = self._wait_for_clickable(By.ID, "stnNm")
            self._click_element(stn_btn)
            time.sleep(1)

            # 지점 검색 팝업에서 지점코드 입력
            # 팝업이 열릴 때까지 대기
            self._wait_for_element(By.ID, "stnIds")

            # 직접 JavaScript로 지점 ID 설정
            self.driver.execute_script(f"""
                document.getElementById('stnIds').value = '{station_code}';
                document.getElementById('txtStnNm').value = '{station_code}';
            """)

            # 또는 지점 선택 팝업에서 검색
            try:
                search_input = self.driver.find_element(By.ID, "searchStnNm")
                search_input.clear()
                search_input.send_keys(station_code)
                time.sleep(0.5)

                # 검색 버튼 클릭
                search_btn = self.driver.find_element(
                    By.CSS_SELECTOR, "button[onclick*='stnSearch']"
                )
                self._click_element(search_btn)
                time.sleep(1)

                # 결과에서 첫 번째 항목 선택
                result_item = self.driver.find_element(
                    By.CSS_SELECTOR, ".stn-list input[type='checkbox']"
                )
                if not result_item.is_selected():
                    self._click_element(result_item)

                # 확인 버튼 클릭
                confirm_btn = self.driver.find_element(
                    By.CSS_SELECTOR, "button[onclick*='fnStnConfirm']"
                )
                self._click_element(confirm_btn)
                time.sleep(0.5)

            except NoSuchElementException:
                # 팝업이 없으면 직접 값 설정
                pass

        except Exception as e:
            print(f"    경고: 지점 선택 UI 접근 실패, JavaScript로 직접 설정: {e}")
            self.driver.execute_script(f"""
                document.getElementById('stnIds').value = '{station_code}';
            """)

    def _select_elements(self, element_codes: List[str]):
        """요소 선택"""
        print(f"  - 요소 선택: {element_codes}")

        try:
            # 요소 선택 버튼 클릭
            elem_btn = self._wait_for_clickable(By.ID, "elementNm")
            self._click_element(elem_btn)
            time.sleep(1)

            # 요소 코드 직접 설정
            codes_str = ",".join(element_codes)
            self.driver.execute_script(f"""
                document.getElementById('elementCds').value = '{codes_str}';
            """)

            # 팝업에서 요소 선택 시도
            try:
                for code in element_codes:
                    checkbox = self.driver.find_element(
                        By.CSS_SELECTOR, f"input[value='{code}']"
                    )
                    if not checkbox.is_selected():
                        self._click_element(checkbox)

                # 확인 버튼
                confirm_btn = self.driver.find_element(
                    By.CSS_SELECTOR, "button[onclick*='fnElementConfirm']"
                )
                self._click_element(confirm_btn)
                time.sleep(0.5)

            except NoSuchElementException:
                pass

        except Exception as e:
            print(f"    경고: 요소 선택 UI 접근 실패, JavaScript로 직접 설정: {e}")
            codes_str = ",".join(element_codes)
            self.driver.execute_script(f"""
                document.getElementById('elementCds').value = '{codes_str}';
            """)

    def _set_date_range(self, start_date: str, end_date: str):
        """기간 설정"""
        print(f"  - 기간 설정: {start_date} ~ {end_date}")

        # 날짜 형식 변환 (YYYY-MM-DD → YYYYMMDD)
        start_fmt = start_date.replace("-", "")
        end_fmt = end_date.replace("-", "")

        try:
            # JavaScript로 날짜 입력
            self.driver.execute_script(f"""
                document.getElementById('startDt_d').value = '{start_fmt}';
                document.getElementById('endDt_d').value = '{end_fmt}';
            """)
        except Exception as e:
            print(f"    경고: 날짜 설정 실패: {e}")

    def _set_data_type(self, data_type: str = "일"):
        """자료형태 설정"""
        print(f"  - 자료형태: {data_type}")

        type_code = DATA_TYPE_CODES.get(data_type, "F00501")

        try:
            self.driver.execute_script(f"""
                document.getElementById('dataFormCd').value = '{type_code}';
            """)
        except Exception:
            pass

    def _trigger_search(self):
        """조회 실행"""
        print("  - 조회 실행 중...")

        try:
            # 조회 버튼 클릭
            search_btn = self._wait_for_clickable(By.ID, "searchBtn")
            self._click_element(search_btn)
            time.sleep(3)

            # 결과 테이블이 로드될 때까지 대기
            self._wait_for_element(By.CSS_SELECTOR, ".data-table, .result-table, #dataList")
            print("  - 조회 완료")
            return True

        except Exception as e:
            print(f"    경고: 조회 버튼 클릭 실패: {e}")
            # JavaScript로 직접 호출
            try:
                self.driver.execute_script("goSearch();")
                time.sleep(3)
                return True
            except Exception:
                return False

    def _download_csv(self, filename: str) -> Optional[str]:
        """CSV 다운로드"""
        print("  - CSV 다운로드 중...")

        try:
            # 기존 파일 정리
            for f in self.temp_download_dir.glob("*.csv"):
                f.unlink()

            # CSV 다운로드 버튼 클릭
            csv_btn = self.driver.find_element(
                By.CSS_SELECTOR, "a[onclick*='fnCVSDownload'], button[onclick*='fnCVSDownload']"
            )
            self._click_element(csv_btn)

            # 다운로드 완료 대기
            downloaded_file = self._wait_for_download()

            if downloaded_file:
                # 파일을 목적지로 이동
                dest_path = self.download_dir / filename
                shutil.move(str(downloaded_file), str(dest_path))
                print(f"  - 다운로드 완료: {dest_path}")
                return str(dest_path)
            else:
                print("    경고: 다운로드된 파일을 찾을 수 없습니다")
                return None

        except NoSuchElementException:
            print("    경고: CSV 다운로드 버튼을 찾을 수 없습니다")
            # JavaScript로 직접 호출
            try:
                self.driver.execute_script("fnCVSDownload('CSV');")
                time.sleep(3)
                downloaded_file = self._wait_for_download()
                if downloaded_file:
                    dest_path = self.download_dir / filename
                    shutil.move(str(downloaded_file), str(dest_path))
                    return str(dest_path)
            except Exception:
                pass
            return None

        except Exception as e:
            print(f"    에러: CSV 다운로드 실패: {e}")
            return None

    def _wait_for_download(self, timeout: int = 60) -> Optional[Path]:
        """다운로드 완료 대기"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            # .crdownload 파일이 없고 .csv 파일이 있는지 확인
            crdownload_files = list(self.temp_download_dir.glob("*.crdownload"))
            csv_files = list(self.temp_download_dir.glob("*.csv"))

            if not crdownload_files and csv_files:
                # 가장 최근 파일 반환
                return max(csv_files, key=lambda f: f.stat().st_mtime)

            time.sleep(0.5)

        return None

    def download(
        self,
        station: str,
        start_date: str,
        end_date: str,
        elements: Optional[List[str]] = None,
        data_type: str = "일",
        output_filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        기상 데이터 다운로드

        Args:
            station: 지점명 (예: "제주", "서울") 또는 지점코드 (예: "184")
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            elements: 요소 목록 (예: ["기온", "습도"]). None이면 전체
            data_type: 자료형태 ("일", "시간", "분")
            output_filename: 출력 파일명. None이면 자동 생성

        Returns:
            다운로드된 파일 경로 또는 None
        """
        print(f"\n{'='*60}")
        print(f"[KMACrawler] 기상 데이터 다운로드 시작")
        print(f"  지점: {station}")
        print(f"  기간: {start_date} ~ {end_date}")
        print(f"  요소: {elements or '전체'}")
        print(f"  자료형태: {data_type}")
        print(f"{'='*60}\n")

        try:
            # 지점코드 확인
            if station.isdigit():
                station_code = station
            else:
                station_code = get_station_code(station)

            # 요소코드 확인
            element_codes = []
            if elements:
                for elem in elements:
                    codes = get_element_codes(elem)
                    element_codes.extend(codes)
                element_codes = list(set(element_codes))  # 중복 제거

            # 파일명 생성
            if not output_filename:
                station_name = station if not station.isdigit() else f"stn{station}"
                elem_str = "_".join(elements[:2]) if elements else "all"
                output_filename = f"{station_name}_{elem_str}_{start_date}_{end_date}.csv"
                output_filename = output_filename.replace("-", "")

            # WebDriver 초기화
            self._init_driver()

            # 페이지 로드
            print("[KMACrawler] 페이지 로드 중...")
            self.driver.get(self.base_url)
            time.sleep(2)

            # 팝업 닫기
            self._close_popup()

            # 자료형태 설정
            self._set_data_type(data_type)

            # 기간 설정
            self._set_date_range(start_date, end_date)

            # 지점 선택
            self._select_station(station_code)

            # 요소 선택
            if element_codes:
                self._select_elements(element_codes)

            # 조회 실행
            if not self._trigger_search():
                print("[KMACrawler] 조회 실패")
                return None

            # CSV 다운로드
            downloaded_path = self._download_csv(output_filename)

            return downloaded_path

        except Exception as e:
            print(f"[KMACrawler] 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            self._close_driver()

    def download_batch(
        self,
        station: str,
        start_year: int,
        end_year: int,
        elements: Optional[List[str]] = None,
        data_type: str = "일",
    ) -> List[str]:
        """
        연도별로 데이터 일괄 다운로드

        기상청 포털은 한 번에 조회할 수 있는 기간이 제한되어 있어
        연도별로 나누어 다운로드합니다.

        Args:
            station: 지점명 또는 지점코드
            start_year: 시작 연도
            end_year: 종료 연도
            elements: 요소 목록
            data_type: 자료형태

        Returns:
            다운로드된 파일 경로 목록
        """
        downloaded_files = []

        for year in range(start_year, end_year + 1):
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            # 현재 연도인 경우 어제 날짜까지만
            if year == datetime.now().year:
                yesterday = datetime.now() - timedelta(days=1)
                end_date = yesterday.strftime("%Y-%m-%d")

            print(f"\n[Batch] {year}년 데이터 다운로드...")

            filepath = self.download(
                station=station,
                start_date=start_date,
                end_date=end_date,
                elements=elements,
                data_type=data_type,
            )

            if filepath:
                downloaded_files.append(filepath)
                print(f"[Batch] {year}년 완료: {filepath}")
            else:
                print(f"[Batch] {year}년 실패")

            # 서버 부하 방지를 위한 대기
            time.sleep(2)

        return downloaded_files


def main():
    """테스트용 메인 함수"""
    crawler = KMACrawler(headless=False)  # 디버깅용으로 헤드리스 끔

    # 제주 2024년 기온 데이터 다운로드
    result = crawler.download(
        station="제주",
        start_date="2024-01-01",
        end_date="2024-12-31",
        elements=["기온"],
        data_type="일",
    )

    if result:
        print(f"\n성공! 다운로드 파일: {result}")
    else:
        print("\n실패!")


if __name__ == "__main__":
    main()
