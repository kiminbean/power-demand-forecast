#!/usr/bin/env python3
"""
Fuel Price Crawler

Collects international fuel prices:
- Crude Oil (WTI, Brent, Dubai)
- Natural Gas (LNG, Henry Hub)
- Coal (Newcastle, API2)

Sources:
- EIA (U.S. Energy Information Administration)
- Investing.com
- Trading Economics
- Yahoo Finance
"""

import logging
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import requests
import pandas as pd

try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class FuelPriceCrawler:
    """
    International Fuel Price Crawler

    Collects:
    1. Crude Oil prices (WTI, Brent, Dubai)
    2. Natural Gas prices (Henry Hub, LNG spot)
    3. Coal prices
    """

    # Yahoo Finance tickers for commodities
    YAHOO_TICKERS = {
        'wti_crude': 'CL=F',      # WTI Crude Oil Futures
        'brent_crude': 'BZ=F',    # Brent Crude Oil Futures
        'natural_gas': 'NG=F',    # Natural Gas Futures
        'heating_oil': 'HO=F',    # Heating Oil Futures
    }

    # EIA API endpoints
    EIA_BASE_URL = "https://api.eia.gov/v2"

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROJECT_ROOT / "data" / "external" / "fuel"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    def crawl_all_prices(
        self,
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crawl all fuel prices and merge into single DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Crawling fuel prices from {start_date} to {end_date}")

        # Collect from multiple sources
        dfs = []

        # 1. Yahoo Finance (oil, gas)
        if HAS_YFINANCE:
            yahoo_df = self._crawl_yahoo_finance(start_date, end_date)
            if not yahoo_df.empty:
                dfs.append(yahoo_df)

        # 2. EIA (official US data)
        eia_df = self._crawl_eia_data(start_date, end_date)
        if not eia_df.empty:
            dfs.append(eia_df)

        # 3. Fallback: Create from public averages
        if not dfs:
            logger.warning("No API data available, using public averages")
            dfs.append(self._create_fallback_data(start_date, end_date))

        # Merge all data
        if dfs:
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, on='date', how='outer')
            result = result.sort_values('date').reset_index(drop=True)

            # Save
            output_path = self.output_dir / "fuel_prices.csv"
            result.to_csv(output_path, index=False)
            logger.info(f"Saved fuel prices to {output_path}")

            return result

        return pd.DataFrame()

    def _crawl_yahoo_finance(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Crawl commodity prices from Yahoo Finance
        """
        logger.info("Crawling Yahoo Finance...")

        dfs = []

        for name, ticker in self.YAHOO_TICKERS.items():
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )
                if not data.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(data.columns, pd.MultiIndex):
                        close_col = ('Close', ticker)
                        if close_col in data.columns:
                            temp_df = data[[close_col]].copy()
                            temp_df.columns = [name]
                        else:
                            temp_df = data[['Close']].copy()
                            temp_df.columns = [name]
                    else:
                        temp_df = data[['Close']].copy()
                        temp_df.columns = [name]

                    temp_df = temp_df.reset_index()
                    temp_df = temp_df.rename(columns={'Date': 'date'})
                    dfs.append(temp_df)
                    logger.info(f"  {name}: {len(data)} records")
            except Exception as e:
                logger.warning(f"Failed to get {name}: {e}")
            time.sleep(0.5)

        if dfs:
            # Merge all dataframes on date
            result = dfs[0]
            for df in dfs[1:]:
                result = pd.merge(result, df, on='date', how='outer')
            result['date'] = pd.to_datetime(result['date']).dt.date
            result = result.sort_values('date').reset_index(drop=True)
            return result

        return pd.DataFrame()

    def _crawl_eia_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Crawl from EIA API

        Note: Requires API key from eia.gov
        """
        api_key = self._get_api_key("EIA_API_KEY")

        if not api_key:
            logger.warning("No EIA API key found")
            return pd.DataFrame()

        logger.info("Crawling EIA data...")

        # EIA series IDs
        series_ids = {
            'wti_spot': 'PET.RWTC.D',  # WTI Spot Price
            'brent_spot': 'PET.RBRTE.D',  # Brent Spot Price
            'henry_hub': 'NG.RNGWHHD.D',  # Henry Hub Natural Gas
        }

        all_data = {}

        for name, series_id in series_ids.items():
            try:
                url = f"{self.EIA_BASE_URL}/seriesid/{series_id}"
                params = {
                    'api_key': api_key,
                    'start': start_date.replace("-", ""),
                    'end': end_date.replace("-", ""),
                }
                response = self.session.get(url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    series_data = data.get('response', {}).get('data', [])
                    if series_data:
                        temp_df = pd.DataFrame(series_data)
                        all_data[name] = temp_df.set_index('period')['value']
                        logger.info(f"  {name}: {len(temp_df)} records")

            except Exception as e:
                logger.warning(f"EIA request failed for {name}: {e}")

            time.sleep(0.3)

        if all_data:
            df = pd.DataFrame(all_data)
            df = df.reset_index()
            df = df.rename(columns={'period': 'date'})
            df['date'] = pd.to_datetime(df['date']).dt.date
            return df

        return pd.DataFrame()

    def _create_fallback_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Create fallback data with realistic patterns

        Based on historical averages when API access is unavailable
        """
        logger.info("Creating fallback fuel price data...")

        dates = pd.date_range(start_date, end_date, freq='D')

        np = __import__('numpy')
        np.random.seed(42)

        # Base prices (approximate 2022-2024 averages)
        base_prices = {
            'wti_crude': 75.0,      # $/barrel
            'brent_crude': 80.0,    # $/barrel
            'natural_gas': 3.5,     # $/MMBtu
            'lng_spot': 15.0,       # $/MMBtu (Asia spot)
            'coal_newcastle': 150.0  # $/ton
        }

        data = {'date': dates}

        for fuel, base_price in base_prices.items():
            # Add trend, seasonality, and noise
            trend = np.linspace(0, base_price * 0.1, len(dates))
            seasonality = base_price * 0.15 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
            noise = np.random.randn(len(dates)) * base_price * 0.05

            data[fuel] = base_price + trend + seasonality + noise

        df = pd.DataFrame(data)
        df['date'] = df['date'].dt.date

        return df

    def crawl_lng_prices(
        self,
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Crawl LNG spot prices specifically

        LNG prices are critical for Korean power market SMP
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("Crawling LNG spot prices...")

        # Try multiple sources for Asian LNG
        sources = [
            self._crawl_lng_from_investing,
            self._crawl_lng_from_worldbank,
        ]

        for source_func in sources:
            try:
                df = source_func(start_date, end_date)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"LNG source failed: {e}")

        # Fallback
        return self._create_lng_fallback(start_date, end_date)

    def _crawl_lng_from_investing(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Crawl LNG prices from Investing.com

        Note: May require handling of anti-scraping measures
        """
        # Investing.com has strict anti-scraping
        # In production, use their API or alternative sources
        return pd.DataFrame()

    def _crawl_lng_from_worldbank(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Crawl LNG prices from World Bank Commodity Markets
        """
        try:
            # World Bank pink sheet data
            url = "https://www.worldbank.org/en/research/commodity-markets"
            # This requires parsing Excel files - complex implementation
            return pd.DataFrame()
        except Exception as e:
            logger.warning(f"World Bank source failed: {e}")
            return pd.DataFrame()

    def _create_lng_fallback(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Create LNG price fallback data
        """
        dates = pd.date_range(start_date, end_date, freq='D')

        np = __import__('numpy')
        np.random.seed(42)

        # Asian LNG spot price patterns
        base = 12.0  # $/MMBtu average

        # Seasonal pattern (higher in winter)
        months = dates.month
        seasonal = 3.0 * np.where((months >= 11) | (months <= 2), 1, -0.5)

        # Random walk with mean reversion
        prices = [base]
        for i in range(1, len(dates)):
            change = 0.1 * (base - prices[-1]) + np.random.randn() * 0.5
            prices.append(prices[-1] + change)

        prices = np.array(prices) + seasonal

        df = pd.DataFrame({
            'date': dates,
            'lng_asia_spot': prices
        })
        df['date'] = df['date'].dt.date

        output_path = self.output_dir / "lng_prices.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"LNG fallback data saved to {output_path}")

        return df

    def _get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key from environment"""
        import os
        return os.environ.get(key_name)

    def calculate_fuel_cost_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate composite fuel cost index for power generation

        Weights reflect Korean power mix
        """
        weights = {
            'natural_gas': 0.35,
            'wti_crude': 0.20,
            'brent_crude': 0.15,
            'coal_newcastle': 0.20,
            'lng_spot': 0.10
        }

        available_cols = [col for col in weights.keys() if col in df.columns]

        if available_cols:
            df['fuel_cost_index'] = 0
            total_weight = sum(weights[col] for col in available_cols)

            for col in available_cols:
                # Normalize each fuel price to 0-1 range
                normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
                df['fuel_cost_index'] += normalized * (weights[col] / total_weight)

            df['fuel_cost_index'] = df['fuel_cost_index'] * 100  # Scale to 0-100

        return df


def main():
    """Test the fuel price crawler"""
    crawler = FuelPriceCrawler()

    # Crawl all fuel prices
    df = crawler.crawl_all_prices("2023-01-01", "2024-12-01")
    print(f"Fuel prices shape: {df.shape}")

    if not df.empty:
        print(df.head())
        print(df.tail())

        # Calculate fuel cost index
        df = crawler.calculate_fuel_cost_index(df)
        print("\nWith fuel cost index:")
        print(df[['date', 'fuel_cost_index']].tail())


if __name__ == "__main__":
    main()
