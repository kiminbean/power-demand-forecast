#!/usr/bin/env python3
"""
공공데이터포털 (data.go.kr) Crawler

Searches and downloads power generation data from:
- https://www.data.go.kr

Features:
- Search for datasets by keyword
- Download CSV files directly (no API key required for file downloads)
- Support for various power/energy datasets
"""

import logging
import time
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict
from urllib.parse import urljoin, quote

import requests
from bs4 import BeautifulSoup
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class PublicDataCrawler:
    """
    공공데이터포털 Crawler

    Searches for power generation datasets and downloads them.
    """

    BASE_URL = "https://www.data.go.kr"
    SEARCH_URL = "https://www.data.go.kr/tcs/dss/selectDataSetList.do"

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROJECT_ROOT / "data" / "external" / "public_data"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
        })

    def search_datasets(
        self,
        keyword: str = "전력 발전량",
        max_results: int = 20
    ) -> List[Dict]:
        """
        Search for datasets on 공공데이터포털

        Args:
            keyword: Search keyword (e.g., "전력 발전량", "시간별 발전량")
            max_results: Maximum number of results to return

        Returns:
            List of dataset info dictionaries
        """
        logger.info(f"Searching for: {keyword}")

        datasets = []

        # Search parameters
        params = {
            'dType': 'FILE',  # File type datasets
            'keyword': keyword,
            'currentPage': 1,
            'perPage': 20,
        }

        try:
            response = self.session.get(self.SEARCH_URL, params=params, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find dataset items
            items = soup.select('.result-list li, .data-list li, .dataset-item')

            if not items:
                # Try alternative selectors
                items = soup.find_all('div', class_=re.compile(r'data|result|item'))

            logger.info(f"Found {len(items)} potential datasets")

            for item in items[:max_results]:
                try:
                    dataset_info = self._parse_dataset_item(item)
                    if dataset_info:
                        datasets.append(dataset_info)
                except Exception as e:
                    logger.debug(f"Error parsing item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Search failed: {e}")

        return datasets

    def _parse_dataset_item(self, item) -> Optional[Dict]:
        """Parse a dataset item from search results"""

        # Try to find title
        title_elem = item.find(['a', 'h3', 'h4', 'span'], class_=re.compile(r'title|name'))
        if not title_elem:
            title_elem = item.find('a')

        if not title_elem:
            return None

        title = title_elem.get_text(strip=True)

        # Try to find link
        link = None
        if title_elem.name == 'a':
            link = title_elem.get('href', '')
        else:
            link_elem = item.find('a')
            if link_elem:
                link = link_elem.get('href', '')

        if link and not link.startswith('http'):
            link = urljoin(self.BASE_URL, link)

        # Try to find description
        desc_elem = item.find(['p', 'span', 'div'], class_=re.compile(r'desc|summary|content'))
        description = desc_elem.get_text(strip=True) if desc_elem else ""

        # Try to find organization
        org_elem = item.find(['span', 'div'], class_=re.compile(r'org|agency|provider'))
        organization = org_elem.get_text(strip=True) if org_elem else ""

        return {
            'title': title,
            'link': link,
            'description': description,
            'organization': organization,
        }

    def search_power_generation_data(self) -> List[Dict]:
        """
        Search specifically for power generation datasets

        Searches multiple keywords related to power generation
        """
        keywords = [
            "시간별 발전량",
            "전력 발전량",
            "한국전력거래소 발전량",
            "발전기별 발전량",
            "제주 전력",
        ]

        all_datasets = []
        seen_titles = set()

        for keyword in keywords:
            datasets = self.search_datasets(keyword, max_results=10)
            for ds in datasets:
                if ds['title'] not in seen_titles:
                    seen_titles.add(ds['title'])
                    all_datasets.append(ds)
            time.sleep(1)  # Rate limiting

        logger.info(f"Total unique datasets found: {len(all_datasets)}")
        return all_datasets

    def download_dataset(
        self,
        dataset_url: str,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """
        Download a dataset from a detail page

        Args:
            dataset_url: URL to the dataset detail page
            filename: Optional filename for saving

        Returns:
            Path to downloaded file, or None if failed
        """
        logger.info(f"Accessing dataset page: {dataset_url}")

        try:
            response = self.session.get(dataset_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find download links (CSV, Excel, etc.)
            download_links = []

            # Try various selectors for download buttons/links
            for selector in [
                'a[href*=".csv"]',
                'a[href*=".xlsx"]',
                'a[href*=".xls"]',
                'a[href*="download"]',
                'a.btn-download',
                'button[data-url]',
                '.file-download a',
            ]:
                links = soup.select(selector)
                download_links.extend(links)

            if not download_links:
                logger.warning("No download links found on page")
                return None

            # Try to download the first CSV/Excel file
            for link in download_links:
                href = link.get('href') or link.get('data-url')
                if not href:
                    continue

                if not href.startswith('http'):
                    href = urljoin(self.BASE_URL, href)

                # Download file
                file_path = self._download_file(href, filename)
                if file_path:
                    return file_path

        except Exception as e:
            logger.error(f"Download failed: {e}")

        return None

    def _download_file(
        self,
        url: str,
        filename: Optional[str] = None
    ) -> Optional[Path]:
        """Download a file from URL"""

        logger.info(f"Downloading: {url}")

        try:
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()

            # Determine filename
            if not filename:
                # Try to get from Content-Disposition header
                cd = response.headers.get('Content-Disposition', '')
                if 'filename=' in cd:
                    filename = re.findall(r'filename[^;=\n]*=(([\'"]).*?\2|[^;\n]*)', cd)
                    if filename:
                        filename = filename[0][0].strip('"\'')

                # Fallback to URL
                if not filename:
                    filename = url.split('/')[-1].split('?')[0]

                if not filename:
                    filename = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            # Clean filename
            filename = re.sub(r'[<>:"/\\|?*]', '_', filename)

            output_path = self.output_dir / filename

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"File download failed: {e}")
            return None

    def download_from_direct_url(
        self,
        url: str,
        filename: str
    ) -> Optional[Path]:
        """
        Download directly from a known URL

        Useful when you know the exact download URL
        """
        return self._download_file(url, filename)

    def list_available_data(self) -> None:
        """
        Print information about available power generation data on 공공데이터포털
        """
        print("\n" + "=" * 60)
        print("공공데이터포털 전력 발전량 데이터 안내")
        print("=" * 60)
        print("""
주요 데이터셋:

1. 한국전력거래소_시간별 발전량
   - 기간: 2017-01-01 ~ 2021-12-31
   - 형식: CSV (일별 24시간 컬럼)
   - URL: https://www.data.go.kr/data/15065277/fileData.do

2. 한국전력거래소_발전기별 발전량
   - 기간: 다양
   - 형식: CSV
   - URL: https://www.data.go.kr/data/15001107/fileData.do

3. 제주 시간대별 발전량 (유류/LNG)
   - 기간: 2023-05-01 ~ 2024-03-31
   - 형식: CSV
   - 직접 다운로드 필요

검색 방법:
1. https://www.data.go.kr 접속
2. "전력 발전량" 또는 "시간별 발전량" 검색
3. 파일데이터 탭 선택
4. 다운로드 버튼 클릭 (API 키 불필요)
        """)
        print("=" * 60)


def main():
    """Test the public data crawler"""
    crawler = PublicDataCrawler()

    # Show available data info
    crawler.list_available_data()

    # Search for datasets
    print("\n검색 중...")
    datasets = crawler.search_power_generation_data()

    if datasets:
        print(f"\n발견된 데이터셋 ({len(datasets)}개):")
        for i, ds in enumerate(datasets[:10], 1):
            print(f"\n{i}. {ds['title']}")
            if ds['organization']:
                print(f"   제공: {ds['organization']}")
            if ds['link']:
                print(f"   링크: {ds['link']}")
    else:
        print("\n검색 결과 없음 - 직접 공공데이터포털 접속 필요")


if __name__ == "__main__":
    main()
