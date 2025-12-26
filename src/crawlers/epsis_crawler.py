#!/usr/bin/env python3
"""
EPSIS (전력통계정보시스템) Crawler

Collects power generation data from:
- https://epsis.kpx.or.kr (Korea Power Exchange Statistics)

Data includes:
- Hourly/daily power generation by source (solar, wind, LNG, coal, nuclear)
- Jeju regional power statistics
- HVDC transmission data
"""

import logging
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import requests
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class EPSISCrawler:
    """
    EPSIS Power Statistics Crawler

    Collects:
    1. Regional power generation (Jeju focus)
    2. Renewable energy generation (solar, wind)
    3. HVDC transmission data
    """

    BASE_URL = "https://epsis.kpx.or.kr"

    # API endpoints (based on EPSIS structure)
    ENDPOINTS = {
        "hourly_generation": "/epsisnew/selectEkgeEpGenpubGrid.do",
        "regional_stats": "/epsisnew/selectEkgeEpRegionGrid.do",
        "renewable": "/epsisnew/selectEkgeEpRenewGrid.do",
    }

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROJECT_ROOT / "data" / "external" / "epsis"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Referer': 'https://epsis.kpx.or.kr/',
        })

    def crawl_jeju_hourly_generation(
        self,
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crawl Jeju hourly power generation data

        Note: EPSIS may require login or have anti-bot measures.
        This implementation uses public data portal API as fallback.
        """
        logger.info("Crawling Jeju hourly generation data...")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Try data.go.kr API first (more reliable)
        df = self._crawl_from_data_go_kr(start_date, end_date)

        if df.empty:
            logger.warning("data.go.kr API failed, trying EPSIS direct...")
            df = self._crawl_epsis_direct(start_date, end_date)

        if not df.empty:
            output_path = self.output_dir / "jeju_hourly_generation.csv"
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved to {output_path}")

        return df

    def _crawl_from_data_go_kr(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Crawl from 공공데이터포털 API

        Uses: 한국전력거래소_제주 시간별 전력 수급 현황
        API Key required - check data.go.kr for registration
        """
        logger.info("Attempting data.go.kr API...")

        # Public data portal requires API key registration
        # For demo, we'll create synthetic structure
        # In production, replace with actual API key

        api_key = self._get_api_key("DATA_GO_KR_API_KEY")

        if not api_key:
            logger.warning("No API key found for data.go.kr")
            return pd.DataFrame()

        base_url = "https://apis.data.go.kr/B553530/PowerDemand/GetJejuPowerDemand"

        all_data = []
        current = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        while current <= end:
            params = {
                "serviceKey": api_key,
                "pageNo": 1,
                "numOfRows": 100,
                "baseDt": current.strftime("%Y%m%d"),
                "dataType": "JSON"
            }

            try:
                response = self.session.get(base_url, params=params, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("response", {}).get("body", {}).get("items", [])
                    all_data.extend(items)
            except Exception as e:
                logger.warning(f"API request failed for {current}: {e}")

            current += timedelta(days=1)
            time.sleep(0.5)  # Rate limiting

        if all_data:
            return pd.DataFrame(all_data)
        return pd.DataFrame()

    def _crawl_epsis_direct(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Direct EPSIS crawling (web scraping)

        Note: May be blocked by anti-bot measures
        """
        logger.info("Attempting direct EPSIS crawl...")

        # EPSIS uses complex JS rendering, may need Selenium
        # For now, return empty and suggest manual download

        logger.warning(
            "Direct EPSIS crawling requires browser automation. "
            "Please download data manually from https://epsis.kpx.or.kr "
            "or use the data.go.kr API with proper authentication."
        )

        return pd.DataFrame()

    def crawl_renewable_generation(
        self,
        region: str = "jeju",
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crawl renewable energy generation (solar, wind)

        Focuses on Jeju region where renewable penetration is high
        """
        logger.info(f"Crawling {region} renewable generation...")

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        # Alternative: Use KPETC (한국에너지기술평가원) open data
        return self._crawl_renewable_opendata(region, start_date, end_date)

    def _crawl_renewable_opendata(
        self,
        region: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Crawl renewable data from open sources
        """
        # Try multiple sources
        sources = [
            self._try_kpetc_api,
            self._try_renewable_portal,
        ]

        for source_func in sources:
            try:
                df = source_func(region, start_date, end_date)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Source failed: {e}")

        return pd.DataFrame()

    def _try_kpetc_api(
        self,
        region: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Try KPETC renewable energy API"""
        # Placeholder - requires API registration
        return pd.DataFrame()

    def _try_renewable_portal(
        self,
        region: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Try renewable energy portal"""
        # Placeholder - requires specific portal access
        return pd.DataFrame()

    def _get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment or config file"""
        import os

        # Try environment variable
        api_key = os.environ.get(key_name)
        if api_key:
            return api_key

        # Try config file
        config_path = PROJECT_ROOT / ".env"
        if config_path.exists():
            with open(config_path) as f:
                for line in f:
                    if line.startswith(key_name):
                        return line.split("=")[1].strip()

        return None

    def create_sample_data(self) -> pd.DataFrame:
        """
        Create sample data structure for testing

        This shows the expected schema for Jeju power generation data
        """
        logger.info("Creating sample data structure...")

        dates = pd.date_range("2024-01-01", periods=24*30, freq="h")

        np = __import__('numpy')
        np.random.seed(42)

        sample_data = pd.DataFrame({
            'datetime': dates,
            'total_demand_mw': 800 + 200 * np.random.randn(len(dates)),
            'solar_generation_mw': np.random.uniform(0, 300, len(dates)),
            'wind_generation_mw': np.random.uniform(50, 400, len(dates)),
            'thermal_generation_mw': np.random.uniform(200, 500, len(dates)),
            'hvdc_import_mw': np.random.uniform(-100, 200, len(dates)),  # Negative = export
            'reserve_margin_pct': np.random.uniform(5, 25, len(dates)),
        })

        # Simulate daily patterns
        hours = sample_data['datetime'].dt.hour
        sample_data['solar_generation_mw'] *= (
            0.1 + 0.9 * np.maximum(0, 1 - ((hours - 12) / 6) ** 2)
        )

        output_path = self.output_dir / "jeju_generation_sample.csv"
        sample_data.to_csv(output_path, index=False)
        logger.info(f"Sample data saved to {output_path}")

        return sample_data


def main():
    """Test the EPSIS crawler"""
    crawler = EPSISCrawler()

    # Create sample data for testing
    sample_df = crawler.create_sample_data()
    print(f"Sample data shape: {sample_df.shape}")
    print(sample_df.head())

    # Try actual crawling (will likely need API key)
    # df = crawler.crawl_jeju_hourly_generation("2024-01-01", "2024-12-01")


if __name__ == "__main__":
    main()
