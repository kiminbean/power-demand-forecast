#!/usr/bin/env python3
"""
공공데이터포털 직접 다운로드 스크립트

Downloads power-related datasets from data.go.kr
"""

import logging
import time
import re
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class PublicDataDownloader:
    """Downloads files from data.go.kr"""

    BASE_URL = "https://www.data.go.kr"

    # Known dataset IDs and their info
    DATASETS = {
        "hourly_demand_2024": {
            "id": "15065266",
            "name": "한국전력거래소_시간별 전국 전력수요량_20241231",
            "url": "https://www.data.go.kr/data/15065266/fileData.do",
        },
        "hourly_generation_2021": {
            "id": "15065387",
            "name": "한국전력거래소_시간별 발전량_20211231",
            "url": "https://www.data.go.kr/data/15065387/fileData.do",
        },
        "regional_solar_wind": {
            "id": "15065269",
            "name": "한국전력거래소_지역별 시간별 태양광 및 풍력 발전량",
            "url": "https://www.data.go.kr/data/15065269/fileData.do",
        },
    }

    def __init__(self):
        self.output_dir = PROJECT_ROOT / "data" / "external" / "public_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9',
            'Referer': 'https://www.data.go.kr/',
        })

    def download_dataset(self, dataset_key: str) -> Path:
        """Download a dataset by key"""

        if dataset_key not in self.DATASETS:
            logger.error(f"Unknown dataset: {dataset_key}")
            return None

        dataset = self.DATASETS[dataset_key]
        logger.info(f"Downloading: {dataset['name']}")

        # Get the page to find download link
        try:
            response = self.session.get(dataset['url'], timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the fileSeq (required for download)
            file_seq = None

            # Method 1: Look for data attributes
            download_btn = soup.find('button', {'class': re.compile('download')})
            if download_btn:
                file_seq = download_btn.get('data-seq') or download_btn.get('data-fileseq')

            # Method 2: Look for hidden inputs
            if not file_seq:
                hidden = soup.find('input', {'name': 'fileSeq'})
                if hidden:
                    file_seq = hidden.get('value')

            # Method 3: Look in script tags
            if not file_seq:
                scripts = soup.find_all('script')
                for script in scripts:
                    if script.string and 'fileSeq' in str(script.string):
                        match = re.search(r"fileSeq['\"]?\s*[:=]\s*['\"]?(\d+)", str(script.string))
                        if match:
                            file_seq = match.group(1)
                            break

            if file_seq:
                # Try to download using the file sequence
                download_url = f"{self.BASE_URL}/tcs/dss/selectFileDataDownload.do"
                params = {
                    'publicDataPk': dataset['id'],
                    'fileSeq': file_seq,
                }

                logger.info(f"Attempting download with fileSeq={file_seq}")

                dl_response = self.session.get(download_url, params=params, stream=True, timeout=60)

                if dl_response.status_code == 200 and 'text/csv' in dl_response.headers.get('Content-Type', ''):
                    filename = f"{dataset['name']}.csv"
                    output_path = self.output_dir / filename

                    with open(output_path, 'wb') as f:
                        for chunk in dl_response.iter_content(chunk_size=8192):
                            f.write(chunk)

                    logger.info(f"Downloaded to: {output_path}")
                    return output_path

            # If direct download fails, show manual instructions
            logger.warning("Automated download failed. Manual download required:")
            logger.warning(f"1. Visit: {dataset['url']}")
            logger.warning("2. Click the download button")
            logger.warning(f"3. Save to: {self.output_dir}")

        except Exception as e:
            logger.error(f"Download error: {e}")

        return None

    def check_local_files(self) -> dict:
        """Check what files we already have locally"""

        raw_dir = PROJECT_ROOT / "data" / "raw"
        external_dir = PROJECT_ROOT / "data" / "external"

        files = {}

        # Check raw directory
        if raw_dir.exists():
            for f in raw_dir.glob("*.csv"):
                files[f.name] = {
                    'path': f,
                    'size': f.stat().st_size,
                    'modified': datetime.fromtimestamp(f.stat().st_mtime)
                }

        # Check external directory
        for subdir in ['public_data', 'epsis']:
            check_dir = external_dir / subdir
            if check_dir.exists():
                for f in check_dir.glob("*.csv"):
                    files[f.name] = {
                        'path': f,
                        'size': f.stat().st_size,
                        'modified': datetime.fromtimestamp(f.stat().st_mtime)
                    }

        return files

    def show_available_data(self):
        """Show what data is available locally and online"""

        print("\n" + "=" * 70)
        print("데이터 현황")
        print("=" * 70)

        # Local files
        print("\n[로컬 파일]")
        files = self.check_local_files()
        for name, info in sorted(files.items()):
            size_kb = info['size'] / 1024
            print(f"  - {name} ({size_kb:.1f} KB)")

        # Online datasets
        print("\n[공공데이터포털 데이터셋]")
        for key, ds in self.DATASETS.items():
            print(f"  - {ds['name']}")
            print(f"    URL: {ds['url']}")

        print("\n" + "=" * 70)


def main():
    downloader = PublicDataDownloader()

    # Show current status
    downloader.show_available_data()

    # Try to download new data
    print("\n다운로드 시도 중...")

    for key in ['hourly_demand_2024', 'regional_solar_wind']:
        result = downloader.download_dataset(key)
        if result:
            print(f"성공: {result}")
        time.sleep(2)


if __name__ == "__main__":
    main()
