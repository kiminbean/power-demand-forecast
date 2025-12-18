"""
SMP 데이터 저장소
=================

SMP 데이터의 저장, 로드, 관리를 담당합니다.

Features:
- CSV/JSON/Parquet 형식 지원
- 자동 중복 제거
- 날짜 범위 조회
- 데이터 검증 및 정합성 체크
- 모델 학습용 데이터 추출
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

import pandas as pd
import numpy as np

from .smp_crawler import SMPData

logger = logging.getLogger(__name__)


class SMPDataStore:
    """SMP 데이터 저장소

    SMP 데이터의 영속화 및 조회를 관리합니다.

    Example:
        >>> store = SMPDataStore("data/smp/smp_history.csv")
        >>> store.save(smp_data_list)
        >>> df = store.load_as_dataframe()
        >>> train_data = store.get_training_data(days=365)
    """

    def __init__(
        self,
        output_path: Union[str, Path],
        auto_create: bool = True
    ):
        """초기화

        Args:
            output_path: 저장 파일 경로 (.csv, .json, .parquet)
            auto_create: 디렉토리 자동 생성 여부
        """
        self.output_path = Path(output_path)
        self.format = self.output_path.suffix.lower().lstrip('.')

        if auto_create:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        data: List[SMPData],
        append: bool = True,
        validate: bool = True
    ) -> int:
        """데이터 저장

        Args:
            data: SMPData 리스트
            append: 기존 데이터에 추가 여부
            validate: 데이터 검증 수행 여부

        Returns:
            저장된 레코드 수
        """
        if not data:
            logger.warning("저장할 데이터 없음")
            return 0

        # 데이터프레임 변환
        df = pd.DataFrame([d.to_dict() for d in data])

        # 데이터 검증
        if validate:
            df = self._validate_data(df)

        # 기존 데이터와 병합
        if append and self.output_path.exists():
            existing_df = self.load_as_dataframe()
            if existing_df is not None and not existing_df.empty:
                df = pd.concat([existing_df, df], ignore_index=True)

        # 중복 제거 (timestamp 기준, 최신 데이터 유지)
        df = df.drop_duplicates(subset=['timestamp'], keep='last')

        # 시간순 정렬
        df = df.sort_values('timestamp').reset_index(drop=True)

        # 형식별 저장
        self._save_by_format(df)

        logger.info(f"SMP 데이터 저장 완료: {len(df)}건 → {self.output_path}")
        return len(df)

    def _save_by_format(self, df: pd.DataFrame):
        """형식별 저장 처리"""
        if self.format == 'csv':
            df.to_csv(self.output_path, index=False, encoding='utf-8-sig')
        elif self.format == 'json':
            records = df.to_dict('records')
            with open(self.output_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
        elif self.format in ('parquet', 'pq'):
            df.to_parquet(self.output_path, index=False)
        else:
            # 기본값: CSV
            df.to_csv(self.output_path, index=False, encoding='utf-8-sig')

    def load(self) -> List[Dict[str, Any]]:
        """데이터 로드 (딕셔너리 리스트)

        Returns:
            SMP 데이터 딕셔너리 리스트
        """
        df = self.load_as_dataframe()
        if df is None or df.empty:
            return []
        return df.to_dict('records')

    def load_as_dataframe(self) -> Optional[pd.DataFrame]:
        """데이터 로드 (DataFrame)

        Returns:
            SMP 데이터 DataFrame 또는 None
        """
        if not self.output_path.exists():
            return None

        try:
            if self.format == 'csv':
                return pd.read_csv(self.output_path)
            elif self.format == 'json':
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    return pd.DataFrame(json.load(f))
            elif self.format in ('parquet', 'pq'):
                return pd.read_parquet(self.output_path)
            else:
                return pd.read_csv(self.output_path)
        except Exception as e:
            logger.error(f"데이터 로드 실패: {e}")
            return None

    def load_as_smp_data(self) -> List[SMPData]:
        """데이터 로드 (SMPData 객체 리스트)

        Returns:
            SMPData 객체 리스트
        """
        records = self.load()
        result = []

        for rec in records:
            try:
                smp = SMPData(
                    timestamp=rec['timestamp'],
                    date=rec.get('date', rec['timestamp'][:10]),
                    hour=int(rec.get('hour', 0)),
                    interval=int(rec.get('interval', 1)),
                    smp_mainland=float(rec.get('smp_mainland', 0)),
                    smp_jeju=float(rec.get('smp_jeju', 0)),
                    smp_max=float(rec.get('smp_max', 0)),
                    smp_min=float(rec.get('smp_min', 0)),
                    smp_weighted_avg=float(rec.get('smp_weighted_avg', 0)),
                    is_finalized=bool(rec.get('is_finalized', False)),
                    fetched_at=rec.get('fetched_at', ''),
                    source=rec.get('source', 'unknown'),
                )
                result.append(smp)
            except Exception as e:
                logger.warning(f"레코드 변환 실패: {e}")

        return result

    def get_date_range(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """날짜 범위로 데이터 조회

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            필터링된 DataFrame
        """
        df = self.load_as_dataframe()
        if df is None or df.empty:
            return pd.DataFrame()

        # 날짜 컬럼 추출 (timestamp에서)
        if 'date' not in df.columns:
            df['date'] = df['timestamp'].str[:10]

        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        return df[mask].copy()

    def get_training_data(
        self,
        days: int = 365,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """모델 학습용 데이터 추출

        Args:
            days: 조회할 일수 (기본: 365일)
            end_date: 종료 날짜 (기본: 오늘)
            columns: 추출할 컬럼 목록

        Returns:
            학습용 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=days)
        ).strftime("%Y-%m-%d")

        df = self.get_date_range(start_date, end_date)

        if columns:
            available = [c for c in columns if c in df.columns]
            df = df[available]

        return df

    def get_latest(self, n: int = 1) -> List[Dict[str, Any]]:
        """최근 데이터 조회

        Args:
            n: 조회할 개수

        Returns:
            최근 SMP 데이터 리스트
        """
        df = self.load_as_dataframe()
        if df is None or df.empty:
            return []

        df = df.sort_values('timestamp', ascending=False)
        return df.head(n).to_dict('records')

    def get_statistics(self) -> Dict[str, Any]:
        """데이터 통계 조회

        Returns:
            통계 정보 딕셔너리
        """
        df = self.load_as_dataframe()
        if df is None or df.empty:
            return {
                'status': 'empty',
                'count': 0,
            }

        # 날짜 범위
        if 'date' not in df.columns:
            df['date'] = df['timestamp'].str[:10]

        # 통계 계산
        stats = {
            'status': 'ok',
            'count': len(df),
            'date_range': {
                'start': df['date'].min(),
                'end': df['date'].max(),
            },
            'unique_days': df['date'].nunique(),
            'smp_mainland': {
                'mean': float(df['smp_mainland'].mean()),
                'std': float(df['smp_mainland'].std()),
                'min': float(df['smp_mainland'].min()),
                'max': float(df['smp_mainland'].max()),
            },
            'smp_jeju': {
                'mean': float(df['smp_jeju'].mean()),
                'std': float(df['smp_jeju'].std()),
                'min': float(df['smp_jeju'].min()),
                'max': float(df['smp_jeju'].max()),
            },
            'file_path': str(self.output_path),
            'file_size_mb': round(self.output_path.stat().st_size / 1024 / 1024, 2) if self.output_path.exists() else 0,
        }

        return stats

    def _validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 검증 및 정제

        Args:
            df: 원본 DataFrame

        Returns:
            검증된 DataFrame
        """
        # 필수 컬럼 확인
        required = ['timestamp', 'smp_mainland']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"필수 컬럼 누락: {missing}")

        # SMP 값 범위 검증 (0 ~ 1000 원/kWh)
        if 'smp_mainland' in df.columns:
            invalid_mask = (df['smp_mainland'] < 0) | (df['smp_mainland'] > 1000)
            if invalid_mask.any():
                logger.warning(f"비정상 SMP 값 {invalid_mask.sum()}건 발견")
                df = df[~invalid_mask]

        # 중복 timestamp 로깅
        dup_count = df['timestamp'].duplicated().sum()
        if dup_count > 0:
            logger.info(f"중복 timestamp {dup_count}건 발견 (마지막 값 유지)")

        return df

    def export_for_training(
        self,
        output_path: Union[str, Path],
        features: Optional[List[str]] = None,
        target: str = 'smp_mainland',
        normalize: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """모델 학습용 데이터 내보내기

        Args:
            output_path: 내보내기 파일 경로
            features: 피처 컬럼 목록 (None이면 기본값)
            target: 타겟 컬럼
            normalize: 정규화 여부

        Returns:
            (DataFrame, 메타데이터)
        """
        df = self.load_as_dataframe()
        if df is None or df.empty:
            raise ValueError("데이터가 없습니다")

        # 기본 피처
        if features is None:
            features = ['hour', 'smp_mainland', 'smp_jeju', 'smp_weighted_avg']

        # 사용 가능한 피처만 선택
        available_features = [f for f in features if f in df.columns]
        df_export = df[available_features].copy()

        # 정규화
        meta = {'normalized': normalize, 'features': available_features}
        if normalize:
            for col in available_features:
                if df_export[col].dtype in [np.float64, np.int64, float, int]:
                    mean = df_export[col].mean()
                    std = df_export[col].std()
                    df_export[col] = (df_export[col] - mean) / std
                    meta[f'{col}_mean'] = float(mean)
                    meta[f'{col}_std'] = float(std)

        # 저장
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == '.parquet':
            df_export.to_parquet(output_path, index=False)
        else:
            df_export.to_csv(output_path, index=False)

        # 메타데이터 저장
        meta_path = output_path.with_suffix('.meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logger.info(f"학습 데이터 내보내기 완료: {output_path}")
        return df_export, meta

    def clear(self):
        """데이터 삭제"""
        if self.output_path.exists():
            self.output_path.unlink()
            logger.info(f"데이터 파일 삭제: {self.output_path}")


def main():
    """테스트 실행"""
    import argparse

    parser = argparse.ArgumentParser(description='SMP 데이터 저장소')
    parser.add_argument('--path', '-p', default='data/smp/smp_history.csv',
                        help='저장소 경로')
    parser.add_argument('--stats', '-s', action='store_true', help='통계 조회')
    parser.add_argument('--latest', '-l', type=int, default=0, help='최근 N건 조회')
    parser.add_argument('--export', '-e', type=str, help='학습 데이터 내보내기 경로')
    args = parser.parse_args()

    store = SMPDataStore(args.path)

    if args.stats:
        stats = store.get_statistics()
        print("\n=== SMP 데이터 통계 ===")
        if stats['status'] == 'empty':
            print("데이터 없음")
        else:
            print(f"총 레코드: {stats['count']:,}건")
            print(f"기간: {stats['date_range']['start']} ~ {stats['date_range']['end']}")
            print(f"일수: {stats['unique_days']}일")
            print(f"\n육지 SMP:")
            print(f"  평균: {stats['smp_mainland']['mean']:.2f} 원/kWh")
            print(f"  표준편차: {stats['smp_mainland']['std']:.2f}")
            print(f"  범위: {stats['smp_mainland']['min']:.2f} ~ {stats['smp_mainland']['max']:.2f}")
            print(f"\n제주 SMP:")
            print(f"  평균: {stats['smp_jeju']['mean']:.2f} 원/kWh")
            print(f"  표준편차: {stats['smp_jeju']['std']:.2f}")
            print(f"  범위: {stats['smp_jeju']['min']:.2f} ~ {stats['smp_jeju']['max']:.2f}")

    if args.latest > 0:
        latest = store.get_latest(args.latest)
        print(f"\n=== 최근 {len(latest)}건 ===")
        for item in latest:
            print(f"[{item['timestamp']}] 육지: {item['smp_mainland']:.2f}, 제주: {item['smp_jeju']:.2f}")

    if args.export:
        try:
            df, meta = store.export_for_training(args.export)
            print(f"\n학습 데이터 내보내기 완료: {args.export}")
            print(f"레코드 수: {len(df)}")
            print(f"피처: {meta['features']}")
        except ValueError as e:
            print(f"내보내기 실패: {e}")


if __name__ == "__main__":
    main()
