#!/usr/bin/env python3
"""
제주관광협회 일별 입도객 데이터 크롤러
====================================
제주관광협회(visitjeju.or.kr) 공식 데이터 기반
M1 GPU (MPS) 가속 EasyOCR 버전

원본: jeju_tourism_crawler_v10.py
프로젝트 통합 버전: 특정 날짜/범위 크롤링 및 CSV 업데이트 지원

Usage:
    # Python
    from tools.crawlers import JejuVisitorCrawler
    crawler = JejuVisitorCrawler()
    
    # 특정 날짜 크롤링
    data = crawler.crawl_date("2025-12-14")
    
    # 기간 크롤링
    df = crawler.crawl_range("2025-12-01", "2025-12-14")
    
    # CSV 업데이트
    crawler.update_csv("2025-12-14")
    
    # CLI
    python jeju_visitor_crawler.py --date 2025-12-14
    python jeju_visitor_crawler.py --start 2025-12-01 --end 2025-12-14
    python jeju_visitor_crawler.py --update  # 최신 데이터로 업데이트
"""

import httpx
import re
import csv
import sys
import io
import time
import warnings
import os
import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from PIL import Image, ImageEnhance
import torch

# MPS 설정
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DailyVisitors:
    """일별 입도객 데이터"""
    date: str                    # YYYY-MM-DD
    daily_visitors: Optional[int]  # 일별 입도객 수
    source: str                  # 데이터 소스 (HTML, EASYOCR_EMBED, etc.)
    notice_num: Optional[int] = None  # 게시물 번호
    
    def to_dict(self) -> dict:
        return asdict(self)


class JejuVisitorCrawler:
    """
    제주관광협회 일별 입도객 크롤러
    
    데이터 소스: http://www.visitjeju.or.kr (제주관광협회 공식)
    수집 방식: HTML 파싱 + EasyOCR (이미지 테이블)
    """
    
    BASE_URL = "http://www.visitjeju.or.kr"
    DETAIL_URL = f"{BASE_URL}/web/bbs/bbsDtl.do"
    LIST_URL = f"{BASE_URL}/web/bbs/bbsList.do"
    FILE_DOWN_URL = f"{BASE_URL}/web/bbs/bbsFileDown.do"
    
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    def __init__(
        self,
        csv_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Args:
            csv_path: 기존 CSV 파일 경로 (업데이트용)
            use_gpu: GPU 가속 사용 여부
        """
        self.use_gpu = use_gpu
        self._easyocr_reader = None
        
        # 프로젝트 기본 경로
        project_root = Path(__file__).parent.parent.parent
        
        if csv_path:
            self.csv_path = Path(csv_path)
        else:
            self.csv_path = project_root / "data" / "processed" / "jeju_daily_visitors_v10.csv"
        
        # 날짜 → 게시물번호 캐시
        self._date_to_notice: Dict[str, int] = {}
        
        logger.info(f"JejuVisitorCrawler 초기화 (CSV: {self.csv_path})")
    
    # =========================================================================
    # Device & OCR
    # =========================================================================
    
    def _get_device(self) -> str:
        """최적의 디바이스 선택"""
        if self.use_gpu:
            if torch.backends.mps.is_available():
                return 'mps'
            elif torch.cuda.is_available():
                return 'cuda'
        return 'cpu'
    
    def _get_easyocr_reader(self):
        """GPU 가속 EasyOCR 리더 초기화 (지연 로딩)"""
        if self._easyocr_reader is None:
            import easyocr
            
            device = self._get_device()
            logger.info(f"EasyOCR 로딩 중 (디바이스: {device.upper()})...")
            
            use_gpu = device in ['mps', 'cuda']
            self._easyocr_reader = easyocr.Reader(
                ['ko', 'en'],
                gpu=use_gpu,
                verbose=False
            )
            logger.info("EasyOCR 로딩 완료")
        
        return self._easyocr_reader
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    def crawl_date(self, date: str) -> Optional[DailyVisitors]:
        """
        특정 날짜의 입도객 데이터 크롤링
        
        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            
        Returns:
            DailyVisitors 또는 None
        """
        logger.info(f"크롤링: {date}")
        
        # 1. 게시물 번호 찾기
        notice_num = self._find_notice_by_date(date)
        if not notice_num:
            logger.warning(f"게시물을 찾을 수 없음: {date}")
            return None
        
        # 2. 게시물 데이터 추출
        data = self._get_post_info(notice_num)
        if not data:
            return None
        
        # 3. OCR 필요시 처리
        if data.get('needs_ocr') and not data.get('daily_visitors'):
            with httpx.Client(headers=self.HEADERS, timeout=60.0, follow_redirects=True) as client:
                value, source = self._process_ocr_item(data, client)
                if value:
                    data['daily_visitors'] = value
                    data['source'] = source
        
        if data.get('daily_visitors'):
            return DailyVisitors(
                date=data['date'],
                daily_visitors=int(data['daily_visitors']),
                source=data['source'],
                notice_num=notice_num,
            )
        
        return None
    
    def crawl_range(
        self,
        start_date: str,
        end_date: str,
        progress_callback: Optional[callable] = None,
    ) -> pd.DataFrame:
        """
        기간별 입도객 데이터 크롤링
        
        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            progress_callback: 진행 상황 콜백
            
        Returns:
            DataFrame
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        results = []
        current = start
        total_days = (end - start).days + 1
        
        logger.info(f"기간 크롤링: {start_date} ~ {end_date} ({total_days}일)")
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            
            try:
                data = self.crawl_date(date_str)
                if data:
                    results.append(data.to_dict())
            except Exception as e:
                logger.error(f"크롤링 실패 ({date_str}): {e}")
            
            if progress_callback:
                progress = (current - start).days + 1
                progress_callback(progress, total_days, date_str)
            
            current += timedelta(days=1)
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def update_csv(
        self,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[int, int]:
        """
        기존 CSV 파일 업데이트
        
        Args:
            date: 특정 날짜 업데이트
            start_date, end_date: 기간 업데이트
            
        Returns:
            (추가된 행 수, 업데이트된 행 수)
        """
        # 기존 데이터 로드
        if self.csv_path.exists():
            existing_df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
            existing_df['날짜'] = pd.to_datetime(existing_df['날짜']).dt.strftime('%Y-%m-%d')
            existing_dates = set(existing_df['날짜'].tolist())
        else:
            existing_df = pd.DataFrame(columns=['날짜', '일별_입도객수', '데이터소스', '비고'])
            existing_dates = set()
        
        # 크롤링
        if date:
            new_data = self.crawl_date(date)
            new_results = [new_data] if new_data else []
        elif start_date and end_date:
            df = self.crawl_range(start_date, end_date)
            new_results = []
            for _, row in df.iterrows():
                new_results.append(DailyVisitors(
                    date=row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else row['date'],
                    daily_visitors=int(row['daily_visitors']) if pd.notna(row['daily_visitors']) else None,
                    source=row['source'],
                    notice_num=row.get('notice_num'),
                ))
        else:
            logger.error("날짜를 지정해주세요")
            return 0, 0
        
        # 병합
        added = 0
        updated = 0
        
        for item in new_results:
            if not item or not item.daily_visitors:
                continue
            
            row_data = {
                '날짜': item.date,
                '일별_입도객수': item.daily_visitors,
                '데이터소스': item.source,
                '비고': '',
            }
            
            if item.date in existing_dates:
                # 기존 데이터 업데이트
                idx = existing_df[existing_df['날짜'] == item.date].index
                if len(idx) > 0:
                    existing_df.loc[idx[0]] = row_data
                    updated += 1
            else:
                # 새 데이터 추가
                existing_df = pd.concat([existing_df, pd.DataFrame([row_data])], ignore_index=True)
                added += 1
        
        # 정렬 후 저장
        existing_df['날짜'] = pd.to_datetime(existing_df['날짜'])
        existing_df = existing_df.sort_values('날짜').reset_index(drop=True)
        existing_df['날짜'] = existing_df['날짜'].dt.strftime('%Y-%m-%d')
        
        existing_df.to_csv(self.csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"CSV 업데이트 완료: 추가 {added}개, 업데이트 {updated}개")
        return added, updated
    
    def get_latest_date(self) -> Optional[str]:
        """CSV의 마지막 날짜 조회"""
        if not self.csv_path.exists():
            return None
        
        df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        if df.empty:
            return None
        
        df['날짜'] = pd.to_datetime(df['날짜'])
        return df['날짜'].max().strftime('%Y-%m-%d')
    
    def update_to_latest(self) -> Tuple[int, int]:
        """
        CSV를 최신 데이터로 업데이트 (마지막 날짜 이후)
        """
        latest = self.get_latest_date()
        if not latest:
            logger.warning("기존 데이터 없음")
            return 0, 0
        
        # 다음 날부터 오늘까지
        start = (datetime.strptime(latest, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        
        if start > end:
            logger.info("이미 최신 상태입니다")
            return 0, 0
        
        return self.update_csv(start_date=start, end_date=end)
    
    # =========================================================================
    # Internal Methods
    # =========================================================================
    
    def _find_notice_by_date(self, target_date: str) -> Optional[int]:
        """날짜에 해당하는 게시물 번호 찾기"""
        # 캐시 확인
        if target_date in self._date_to_notice:
            return self._date_to_notice[target_date]
        
        # 날짜 파싱
        try:
            dt = datetime.strptime(target_date, "%Y-%m-%d")
        except:
            return None
        
        # 검색 키워드 생성 (예: "2025년 12월 14일")
        search_keyword = f"{dt.year}년 {dt.month}월 {dt.day}일"
        
        # 목록 페이지에서 검색
        with httpx.Client(headers=self.HEADERS, timeout=30.0) as client:
            for page in range(1, 20):  # 최대 20페이지 검색
                try:
                    url = f"{self.LIST_URL}?bbsId=TOURSTATD&pageIndex={page}"
                    response = client.get(url)
                    
                    if response.status_code != 200:
                        continue
                    
                    # 제목에서 날짜 매칭
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for item in soup.find_all('a', href=True):
                        onclick = item.get('onclick', '')
                        title = item.get_text().strip()
                        
                        # 제목에 날짜 포함 확인
                        if f"{dt.month}월 {dt.day}일" in title or f"{dt.month}월{dt.day}일" in title:
                            match = re.search(r'fn_dtl\((\d+)\)', onclick)
                            if match:
                                notice_num = int(match.group(1))
                                self._date_to_notice[target_date] = notice_num
                                return notice_num
                    
                    # fn_dtl 패턴으로도 검색
                    matches = re.findall(r'fn_dtl\((\d+)\)', response.text)
                    for notice_num_str in matches:
                        notice_num = int(notice_num_str)
                        # 해당 게시물 날짜 확인
                        post_date = self._get_post_date(notice_num)
                        if post_date == target_date:
                            self._date_to_notice[target_date] = notice_num
                            return notice_num
                    
                except Exception as e:
                    logger.debug(f"검색 오류 (페이지 {page}): {e}")
                    continue
        
        return None
    
    def _get_post_date(self, notice_num: int) -> Optional[str]:
        """게시물 날짜 추출"""
        try:
            with httpx.Client(headers=self.HEADERS, timeout=30.0) as client:
                url = f"{self.DETAIL_URL}?bbsId=TOURSTATD&noticeNum={notice_num}"
                response = client.get(url)
                
                if response.status_code == 200:
                    return self._extract_date_from_title(response.text)
        except:
            pass
        return None
    
    def _get_post_info(self, notice_num: int) -> Optional[Dict]:
        """게시물 정보 가져오기"""
        try:
            with httpx.Client(headers=self.HEADERS, timeout=30.0, follow_redirects=True) as client:
                url = f"{self.DETAIL_URL}?bbsId=TOURSTATD&noticeNum={notice_num}"
                response = client.get(url)
                
                if response.status_code != 200:
                    return None
                
                html = response.text
                date = self._extract_date_from_title(html)
                
                if not date:
                    return None
                
                # 1. HTML에서 직접 추출 시도
                daily_visitors = self._extract_daily_visitors_from_html(html)
                
                if daily_visitors:
                    return {
                        'notice_num': notice_num,
                        'date': date,
                        'daily_visitors': daily_visitors,
                        'source': 'HTML',
                        'needs_ocr': False,
                        'img_urls': [],
                        'attachment_ids': [],
                        'base64_images': []
                    }
                
                # 2. 이미지 URL 확인
                img_urls = self._extract_embedded_image_urls(html)
                attachment_ids = self._extract_attachment_file_ids(html)
                base64_images = self._extract_base64_images(html)
                
                if img_urls or attachment_ids or base64_images:
                    return {
                        'notice_num': notice_num,
                        'date': date,
                        'daily_visitors': None,
                        'source': 'NEEDS_OCR',
                        'needs_ocr': True,
                        'img_urls': img_urls,
                        'attachment_ids': attachment_ids,
                        'base64_images': base64_images
                    }
                
                return {
                    'notice_num': notice_num,
                    'date': date,
                    'daily_visitors': None,
                    'source': 'NO_DATA',
                    'needs_ocr': False,
                    'img_urls': [],
                    'attachment_ids': [],
                    'base64_images': []
                }
        
        except Exception as e:
            logger.debug(f"게시물 정보 오류: {e}")
            return None
    
    # =========================================================================
    # Extraction Methods
    # =========================================================================
    
    def _extract_date_from_title(self, html_content: str) -> Optional[str]:
        """제목에서 날짜 추출"""
        match = re.search(r'(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일', html_content)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return f"{year}-{month:02d}-{day:02d}"
        return None
    
    def _is_valid_daily_visitors(self, value_str) -> bool:
        """유효한 입도객 수인지 확인 (5,000 ~ 150,000)"""
        try:
            value = int(str(value_str).replace(',', ''))
            return 5000 <= value <= 150000
        except:
            return False
    
    def _extract_daily_visitors_from_text(self, text: str) -> Optional[str]:
        """텍스트에서 일계 값 추출"""
        normalized = re.sub(r'[\t\s]+', ' ', text)
        
        # 패턴 1: "20XX년 (명) 일 계 숫자"
        pattern1 = re.search(r'20\d{2}년\s*\(명\)\s*일\s*계\s+([\d,]+)', normalized)
        if pattern1:
            value = pattern1.group(1).replace(',', '')
            if self._is_valid_daily_visitors(value):
                return value
        
        # 패턴 2: "일 계 숫자"
        pattern2 = re.search(r'일\s*계\s+([\d,]+)', normalized)
        if pattern2:
            value = pattern2.group(1).replace(',', '')
            if self._is_valid_daily_visitors(value):
                return value
        
        # 패턴 3: 줄별 검색
        lines = text.split('\n')
        for line in lines:
            if '일계' in line or '일 계' in line:
                numbers = re.findall(r'([\d,]+)', line)
                for num in numbers:
                    clean_num = num.replace(',', '')
                    if clean_num.isdigit() and self._is_valid_daily_visitors(clean_num):
                        return clean_num
        
        return None
    
    def _extract_daily_visitors_from_html(self, html_content: str) -> Optional[str]:
        """HTML에서 일계 값 추출"""
        soup = BeautifulSoup(html_content, 'html.parser')
        content_div = soup.find('div', class_='view-content')
        if not content_div:
            return None
        
        # 테이블에서 추출
        for table in content_div.find_all('table'):
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 2:
                    continue
                
                cell_texts = [c.get_text().strip() for c in cells]
                
                # "일계"가 첫 번째 셀에 있는 경우
                if '일계' in cell_texts[0] or '일 계' in cell_texts[0]:
                    if len(cell_texts) > 1:
                        numbers = re.findall(r'[\d,]+', cell_texts[1])
                        if numbers:
                            value = numbers[0].replace(',', '')
                            if self._is_valid_daily_visitors(value):
                                return value
                
                # "일계"가 두 번째 셀에 있는 경우
                if len(cell_texts) > 2 and ('일계' in cell_texts[1] or '일 계' in cell_texts[1]):
                    numbers = re.findall(r'[\d,]+', cell_texts[2])
                    if numbers:
                        value = numbers[0].replace(',', '')
                        if self._is_valid_daily_visitors(value):
                            return value
                
                # 행 전체에서 검색
                row_text = ' '.join(cell_texts)
                if '일계' in row_text or '일 계' in row_text:
                    result = self._extract_daily_visitors_from_text(row_text)
                    if result:
                        return result
        
        # 전체 텍스트에서 추출
        full_text = content_div.get_text()
        return self._extract_daily_visitors_from_text(full_text)
    
    def _extract_embedded_image_urls(self, html_content: str) -> List[str]:
        """임베디드 이미지 URL 추출"""
        urls = []
        
        # 패턴 1: getImage.do
        pattern1 = re.findall(r'src="(/editor/getImage\.do\?editorFile=[^"]+)"', html_content)
        urls.extend([f"{self.BASE_URL}{path}" for path in pattern1])
        
        # 패턴 2: 다른 이미지 경로
        pattern2 = re.findall(r'src="(/[^"]*\.(jpg|jpeg|png|gif))"', html_content, re.IGNORECASE)
        for path, ext in pattern2:
            if '/editor/' in path or '/upload/' in path:
                urls.append(f"{self.BASE_URL}{path}")
        
        return urls
    
    def _extract_attachment_file_ids(self, html_content: str) -> List[str]:
        """첨부파일 ID 추출 (이미지 파일만)"""
        file_ids = []
        pattern = re.findall(r"fn_DownloadFile\('(\d+)'\)[^>]*title=\"([^\"]+)\"", html_content)
        for file_id, filename in pattern:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                file_ids.append(file_id)
        return file_ids
    
    def _extract_base64_images(self, html_content: str) -> List[bytes]:
        """Base64 인코딩 이미지 추출"""
        base64_images = []
        
        pattern = re.findall(r'src="(data:image/[^;]+;base64,[^"]+)"', html_content)
        for data_url in pattern:
            try:
                header, b64_data = data_url.split(',', 1)
                image_bytes = base64.b64decode(b64_data)
                base64_images.append(image_bytes)
            except:
                pass
        
        return base64_images
    
    # =========================================================================
    # OCR Methods
    # =========================================================================
    
    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """이미지 전처리"""
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        if width < 1500:
            scale = 1500 / width
            img = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
        
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        
        return img
    
    def _extract_with_easyocr(self, image_bytes: bytes) -> Optional[str]:
        """GPU 가속 EasyOCR로 일계 값 추출"""
        try:
            reader = self._get_easyocr_reader()
            
            img = Image.open(io.BytesIO(image_bytes))
            img = self._preprocess_image(img)
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_bytes = img_byte_arr.getvalue()
            
            results = reader.readtext(img_bytes)
            
            if not results:
                return None
            
            # "일계" 찾고 같은 행의 숫자 추출
            for i, (bbox, text, conf) in enumerate(results):
                if '일계' in text or '일 계' in text:
                    ilgye_y = (bbox[0][1] + bbox[2][1]) / 2
                    
                    for j, (bbox2, text2, conf2) in enumerate(results):
                        if j == i:
                            continue
                        text2_y = (bbox2[0][1] + bbox2[2][1]) / 2
                        if abs(ilgye_y - text2_y) < 30:
                            numbers = re.findall(r'[\d,]+', text2)
                            for num in numbers:
                                clean_num = num.replace(',', '')
                                if clean_num.isdigit() and self._is_valid_daily_visitors(clean_num):
                                    return clean_num
            
            # 전체 텍스트 패턴
            all_texts = ' '.join([t[1] for t in results])
            pattern = re.search(r'일\s*계\s*([\d,]+)', all_texts)
            if pattern:
                value = pattern.group(1).replace(',', '')
                if self._is_valid_daily_visitors(value):
                    return value
            
            # 첫 번째 유효한 숫자
            for bbox, text, conf in results:
                if conf > 0.5:
                    numbers = re.findall(r'[\d,]+', text)
                    for num in numbers:
                        clean_num = num.replace(',', '')
                        if clean_num.isdigit() and self._is_valid_daily_visitors(clean_num):
                            return clean_num
            
            return None
        except Exception as e:
            logger.debug(f"OCR 오류: {e}")
            return None
    
    def _process_ocr_item(self, item: Dict, client: httpx.Client) -> Tuple[Optional[str], str]:
        """OCR 처리 (임베디드 이미지 + 첨부파일 + Base64)"""
        notice_num = item['notice_num']
        
        # 1. 임베디드 이미지 처리
        for img_url in item.get('img_urls', []):
            try:
                img_response = client.get(img_url)
                if img_response.status_code == 200:
                    value = self._extract_with_easyocr(img_response.content)
                    if value:
                        return value, 'EASYOCR_EMBED'
            except:
                pass
        
        # 2. 첨부파일 이미지 처리
        for file_id in item.get('attachment_ids', []):
            try:
                file_url = f"{self.FILE_DOWN_URL}?bbsId=TOURSTATD&noticeNum={notice_num}&fileId={file_id}"
                file_response = client.get(file_url)
                if file_response.status_code == 200:
                    content_type = file_response.headers.get('Content-Type', '')
                    if 'image' in content_type or len(file_response.content) > 10000:
                        value = self._extract_with_easyocr(file_response.content)
                        if value:
                            return value, 'EASYOCR_ATTACH'
            except:
                pass
        
        # 3. Base64 인코딩 이미지 처리
        for image_bytes in item.get('base64_images', []):
            try:
                value = self._extract_with_easyocr(image_bytes)
                if value:
                    return value, 'EASYOCR_BASE64'
            except:
                pass
        
        return None, 'OCR_FAILED'


# =============================================================================
# CLI
# =============================================================================

def main():
    """CLI 메인"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="제주관광협회 일별 입도객 크롤러",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 특정 날짜 크롤링
  python jeju_visitor_crawler.py --date 2025-12-14
  
  # 기간 크롤링
  python jeju_visitor_crawler.py --start 2025-12-01 --end 2025-12-14
  
  # 최신 데이터로 업데이트
  python jeju_visitor_crawler.py --update
  
  # CSV 파일 지정
  python jeju_visitor_crawler.py --date 2025-12-14 --csv /path/to/data.csv
        """
    )
    
    parser.add_argument('--date', type=str, help='특정 날짜 (YYYY-MM-DD)')
    parser.add_argument('--start', type=str, help='시작 날짜 (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='종료 날짜 (YYYY-MM-DD)')
    parser.add_argument('--update', action='store_true', help='최신 데이터로 업데이트')
    parser.add_argument('--csv', type=str, help='CSV 파일 경로')
    parser.add_argument('--no-gpu', action='store_true', help='GPU 비활성화')
    
    args = parser.parse_args()
    
    # 크롤러 초기화
    crawler = JejuVisitorCrawler(
        csv_path=args.csv,
        use_gpu=not args.no_gpu,
    )
    
    if args.update:
        # 최신 업데이트
        print(f"CSV 최신 업데이트: {crawler.csv_path}")
        added, updated = crawler.update_to_latest()
        print(f"완료: 추가 {added}개, 업데이트 {updated}개")
    
    elif args.date:
        # 특정 날짜
        print(f"크롤링: {args.date}")
        data = crawler.crawl_date(args.date)
        if data:
            print(f"\n=== {args.date} 제주 입도객 ===")
            print(f"입도객: {data.daily_visitors:,}명")
            print(f"소스: {data.source}")
            
            # CSV 업데이트
            added, updated = crawler.update_csv(date=args.date)
            print(f"\nCSV 업데이트: 추가 {added}개, 업데이트 {updated}개")
        else:
            print(f"데이터를 찾을 수 없습니다: {args.date}")
    
    elif args.start and args.end:
        # 기간 크롤링
        def progress(current, total, date):
            pct = current / total * 100
            print(f"\r진행: {current}/{total} ({pct:.1f}%) - {date}", end='', flush=True)
        
        df = crawler.crawl_range(args.start, args.end, progress_callback=progress)
        print()  # 줄바꿈
        
        if not df.empty:
            print(f"\n=== 수집 결과 ===")
            print(f"기간: {args.start} ~ {args.end}")
            print(f"수집: {len(df)}일")
            print(f"총 입도객: {df['daily_visitors'].sum():,}명")
            
            # CSV 업데이트
            added, updated = crawler.update_csv(start_date=args.start, end_date=args.end)
            print(f"\nCSV 업데이트: 추가 {added}개, 업데이트 {updated}개")
        else:
            print("수집된 데이터가 없습니다.")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
