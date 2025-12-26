#!/usr/bin/env python3
"""
Master Data Collector for SMP Prediction

Orchestrates all crawlers and merges data into unified dataset:
1. EPSIS - Power generation data
2. Fuel prices - Oil, LNG, Coal
3. Carbon prices - K-ETS, EU-ETS
4. Weather data - Already collected
5. SMP data - Already collected

Output: Unified dataset for improved model training
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.crawlers.epsis_crawler import EPSISCrawler
from src.crawlers.fuel_crawler import FuelPriceCrawler
from src.crawlers.carbon_crawler import CarbonPriceCrawler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent


class DataCollector:
    """
    Master data collector for SMP prediction model

    Collects and merges:
    - SMP data (existing)
    - Weather data (existing)
    - Power generation data (new)
    - Fuel prices (new)
    - Carbon prices (new)
    """

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or PROJECT_ROOT / "data" / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize crawlers
        self.epsis_crawler = EPSISCrawler()
        self.fuel_crawler = FuelPriceCrawler()
        self.carbon_crawler = CarbonPriceCrawler()

    def collect_all_data(
        self,
        start_date: str = "2022-01-01",
        end_date: Optional[str] = None,
        use_fallback: bool = True
    ) -> pd.DataFrame:
        """
        Collect all data sources and merge

        Args:
            start_date: Start date for data collection
            end_date: End date (default: today)
            use_fallback: Use fallback data if API fails

        Returns:
            Merged DataFrame with all features
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info("=" * 60)
        logger.info("Starting comprehensive data collection")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info("=" * 60)

        collected_data = {}

        # 1. Load existing SMP data
        logger.info("\n[1/5] Loading SMP data...")
        smp_df = self._load_smp_data()
        if not smp_df.empty:
            collected_data['smp'] = smp_df
            logger.info(f"  SMP: {len(smp_df)} records")

        # 2. Load existing weather data
        logger.info("\n[2/5] Loading weather data...")
        weather_df = self._load_weather_data()
        if not weather_df.empty:
            collected_data['weather'] = weather_df
            logger.info(f"  Weather: {len(weather_df)} records")

        # 3. Collect power generation data
        logger.info("\n[3/5] Collecting power generation data...")
        if use_fallback:
            gen_df = self.epsis_crawler.create_sample_data()
        else:
            gen_df = self.epsis_crawler.crawl_jeju_hourly_generation(start_date, end_date)
        if not gen_df.empty:
            collected_data['generation'] = gen_df
            logger.info(f"  Generation: {len(gen_df)} records")

        # 4. Collect fuel prices
        logger.info("\n[4/5] Collecting fuel prices...")
        fuel_df = self.fuel_crawler.crawl_all_prices(start_date, end_date)
        if not fuel_df.empty:
            collected_data['fuel'] = fuel_df
            logger.info(f"  Fuel: {len(fuel_df)} records")

        # 5. Collect carbon prices
        logger.info("\n[5/5] Collecting carbon prices...")
        carbon_df = self.carbon_crawler.crawl_all_carbon_prices(start_date, end_date)
        if not carbon_df.empty:
            collected_data['carbon'] = carbon_df
            logger.info(f"  Carbon: {len(carbon_df)} records")

        # Merge all data
        logger.info("\n" + "=" * 60)
        logger.info("Merging datasets...")
        logger.info("=" * 60)

        merged_df = self._merge_all_data(collected_data, start_date, end_date)

        if not merged_df.empty:
            # Save merged data
            output_path = self.output_dir / "smp_enhanced_dataset.csv"
            merged_df.to_csv(output_path, index=False)
            logger.info(f"\nSaved enhanced dataset to {output_path}")
            logger.info(f"Total records: {len(merged_df)}")
            logger.info(f"Total features: {len(merged_df.columns)}")

        return merged_df

    def _load_smp_data(self) -> pd.DataFrame:
        """Load existing SMP data"""
        smp_path = PROJECT_ROOT / "data" / "smp" / "smp_5years_epsis.csv"

        if not smp_path.exists():
            logger.warning(f"SMP file not found: {smp_path}")
            return pd.DataFrame()

        df = pd.read_csv(smp_path)

        # Handle 24:00 timestamps
        def fix_timestamp(ts):
            if ' 24:00' in str(ts):
                parts = str(ts).split(' ')
                date_part = pd.to_datetime(parts[0]) + pd.Timedelta(days=1)
                return date_part
            return pd.to_datetime(ts)

        df['datetime'] = df['timestamp'].apply(fix_timestamp)
        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour
        df['smp'] = df['smp_mainland']

        return df[['datetime', 'date', 'hour', 'smp']]

    def _load_weather_data(self) -> pd.DataFrame:
        """Load existing weather data"""
        weather_path = PROJECT_ROOT / "data" / "processed" / "jeju_weather_hourly_merged.csv"

        if not weather_path.exists():
            logger.warning(f"Weather file not found: {weather_path}")
            return pd.DataFrame()

        df = pd.read_csv(weather_path)

        # Parse datetime
        if '일시' in df.columns:
            df['datetime'] = pd.to_datetime(df['일시'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        df['date'] = df['datetime'].dt.date
        df['hour'] = df['datetime'].dt.hour

        # Map Korean column names
        col_mapping = {
            '기온': 'temperature',
            '풍속': 'wind_speed',
            '습도': 'humidity',
            '일사': 'solar_radiation',
            '현지기압': 'pressure'
        }

        for korean, english in col_mapping.items():
            if korean in df.columns:
                df[english] = pd.to_numeric(df[korean], errors='coerce')

        weather_cols = ['datetime', 'date', 'hour', 'temperature', 'wind_speed',
                        'humidity', 'solar_radiation', 'pressure']
        available_cols = [c for c in weather_cols if c in df.columns]

        return df[available_cols]

    def _merge_all_data(
        self,
        collected_data: Dict[str, pd.DataFrame],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Merge all collected data into unified dataset
        """
        if 'smp' not in collected_data or collected_data['smp'].empty:
            logger.error("SMP data is required for merging")
            return pd.DataFrame()

        # Start with SMP as base
        result = collected_data['smp'].copy()
        result['date'] = pd.to_datetime(result['date'])

        # Merge weather (hourly)
        if 'weather' in collected_data and not collected_data['weather'].empty:
            weather = collected_data['weather'].copy()
            weather['datetime'] = pd.to_datetime(weather['datetime'])
            result = pd.merge(
                result,
                weather.drop(columns=['date', 'hour'], errors='ignore'),
                on='datetime',
                how='left'
            )
            logger.info("  Merged weather data")

        # Merge fuel prices (daily -> forward fill to hourly)
        if 'fuel' in collected_data and not collected_data['fuel'].empty:
            fuel = collected_data['fuel'].copy()
            fuel['date'] = pd.to_datetime(fuel['date'])
            result = pd.merge(
                result,
                fuel,
                on='date',
                how='left'
            )
            # Forward fill daily values
            fuel_cols = [c for c in fuel.columns if c != 'date']
            result[fuel_cols] = result[fuel_cols].ffill()
            logger.info("  Merged fuel prices")

        # Merge carbon prices (daily -> forward fill to hourly)
        if 'carbon' in collected_data and not collected_data['carbon'].empty:
            carbon = collected_data['carbon'].copy()
            carbon['date'] = pd.to_datetime(carbon['date'])
            result = pd.merge(
                result,
                carbon,
                on='date',
                how='left'
            )
            carbon_cols = [c for c in carbon.columns if c != 'date']
            result[carbon_cols] = result[carbon_cols].ffill()
            logger.info("  Merged carbon prices")

        # Merge generation data (hourly)
        if 'generation' in collected_data and not collected_data['generation'].empty:
            gen = collected_data['generation'].copy()
            gen['datetime'] = pd.to_datetime(gen['datetime'])
            result = pd.merge(
                result,
                gen,
                on='datetime',
                how='left'
            )
            logger.info("  Merged generation data")

        # Filter date range
        result = result[
            (result['date'] >= pd.to_datetime(start_date)) &
            (result['date'] <= pd.to_datetime(end_date))
        ]

        # Sort by datetime
        result = result.sort_values('datetime').reset_index(drop=True)

        # Fill remaining NaN values
        result = result.ffill().bfill()

        # Create derived features
        result = self._create_derived_features(result)

        return result

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for model training
        """
        logger.info("Creating derived features...")

        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if 'datetime' in df.columns:
            df['dayofweek'] = pd.to_datetime(df['datetime']).dt.dayofweek
            df['month'] = pd.to_datetime(df['datetime']).dt.month
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

            df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # SMP lag features
        if 'smp' in df.columns:
            for lag in [1, 2, 3, 6, 12, 24, 48, 168]:
                df[f'smp_lag_{lag}'] = df['smp'].shift(lag)

            # Rolling statistics
            for window in [6, 12, 24, 48, 168]:
                df[f'smp_roll_mean_{window}'] = df['smp'].rolling(window).mean()
                df[f'smp_roll_std_{window}'] = df['smp'].rolling(window).std()

        # Fuel price lags (for time-lagged effect)
        fuel_cols = ['wti_crude', 'brent_crude', 'natural_gas', 'lng_spot']
        for col in fuel_cols:
            if col in df.columns:
                # 1-week moving average
                df[f'{col}_ma7'] = df[col].rolling(7 * 24).mean()
                # 1-month moving average
                df[f'{col}_ma30'] = df[col].rolling(30 * 24).mean()

        # Net load proxy (if generation data available)
        if all(c in df.columns for c in ['solar_generation_mw', 'wind_generation_mw']):
            df['renewable_total'] = df['solar_generation_mw'] + df['wind_generation_mw']
            if 'total_demand_mw' in df.columns:
                df['net_load'] = df['total_demand_mw'] - df['renewable_total']

        # Handle NaN
        df = df.ffill().bfill()

        return df

    def get_feature_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of collected features
        """
        summary = {
            'total_records': len(df),
            'date_range': {
                'start': df['datetime'].min() if 'datetime' in df.columns else None,
                'end': df['datetime'].max() if 'datetime' in df.columns else None
            },
            'features': {
                'smp': [c for c in df.columns if 'smp' in c.lower()],
                'weather': [c for c in df.columns if c in ['temperature', 'wind_speed', 'humidity', 'solar_radiation', 'pressure']],
                'fuel': [c for c in df.columns if any(f in c for f in ['crude', 'gas', 'lng', 'coal'])],
                'carbon': [c for c in df.columns if any(f in c for f in ['kets', 'ets', 'carbon'])],
                'time': [c for c in df.columns if any(f in c for f in ['hour', 'day', 'month', 'sin', 'cos', 'weekend'])],
            },
            'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict()
        }

        return summary


def main():
    """Run data collection"""
    collector = DataCollector()

    # Collect all data for 2022-2024
    df = collector.collect_all_data(
        start_date="2022-01-01",
        end_date="2024-12-31",
        use_fallback=True
    )

    if not df.empty:
        print("\n" + "=" * 60)
        print("Data Collection Complete!")
        print("=" * 60)
        print(f"Records: {len(df)}")
        print(f"Features: {len(df.columns)}")
        print(f"\nColumns: {list(df.columns)}")

        # Feature summary
        summary = collector.get_feature_summary(df)
        print(f"\nFeature Categories:")
        for category, features in summary['features'].items():
            print(f"  {category}: {len(features)} features")


if __name__ == "__main__":
    main()
