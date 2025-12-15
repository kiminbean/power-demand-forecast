# 기상청 기상데이터 크롤러 사용 가이드

기상청 기상자료개방포털에서 ASOS(종관기상관측) 데이터를 자동으로 다운로드하는 도구입니다.

## 빠른 시작

### 자연어 명령으로 다운로드

```bash
# 제주도 2013~2024년 기온 데이터
python tools/crawlers/download_weather.py "제주도 2013~2024년 기온 데이터를 다운로드해줘"

# 서울 최근 5년간 습도와 강수량
python tools/crawlers/download_weather.py "서울 지역의 최근 5년간 습도와 강수량 데이터"

# 부산 2024년 일사량
python tools/crawlers/download_weather.py "부산 2024년 일사량 자료"
```

### 명령줄 인자로 다운로드

```bash
# 기본 사용법
python tools/crawlers/download_weather.py \
    --station 제주 \
    --start 2013-01-01 \
    --end 2024-12-31 \
    --elements 기온 습도

# 연도별 일괄 다운로드
python tools/crawlers/download_weather.py \
    --station 제주 \
    --start-year 2013 \
    --end-year 2024 \
    --elements 기온 \
    --batch
```

## 두 가지 다운로드 방식

### 1. Selenium 크롤러 (기본)

기상자료개방포털 웹사이트를 직접 크롤링합니다.

**장점:**
- API 키 불필요
- 모든 데이터 접근 가능

**단점:**
- Chrome 브라우저 필요
- 상대적으로 느림
- 사이트 구조 변경에 취약

```python
from tools.crawlers import KMACrawler

crawler = KMACrawler()
result = crawler.download(
    station="제주",
    start_date="2024-01-01",
    end_date="2024-12-31",
    elements=["기온", "습도"],
)
```

### 2. 공공데이터포털 API (권장)

공공데이터포털(data.go.kr) API를 사용합니다.

**장점:**
- 빠르고 안정적
- JSON/XML 응답

**단점:**
- API 키 필요 (무료 발급)
- 일부 데이터 제한 가능

```python
from tools.crawlers import KMAAPI

# 환경변수로 API 키 설정: export KMA_API_KEY='your_key'
api = KMAAPI()
result = api.download(
    station="제주",
    start_date="2024-01-01",
    end_date="2024-12-31",
    data_type="일",
)
```

## API 키 발급 방법

1. [공공데이터포털](https://www.data.go.kr) 회원가입
2. [ASOS 시간자료 API](https://www.data.go.kr/data/15057210/openapi.do) 페이지 접속
3. "활용신청" 버튼 클릭
4. 신청 후 발급된 API 키 복사
5. 환경변수 설정:

```bash
# .bashrc 또는 .zshrc에 추가
export KMA_API_KEY='your_api_key_here'
```

## 지원하는 지점

```bash
# 지점 목록 확인
python tools/crawlers/download_weather.py --list-stations
```

주요 지점:
- **서울/경기**: 서울(108), 인천(112), 수원(119)
- **강원**: 춘천(101), 강릉(105), 원주(114)
- **충청**: 대전(133), 청주(131), 서산(129)
- **전라**: 광주(156), 전주(146), 목포(165)
- **경북**: 대구(143), 포항(138), 안동(136)
- **경남**: 부산(159), 울산(152), 창원(155)
- **제주**: 제주(184), 서귀포(189), 고산(185)

## 지원하는 요소

```bash
# 요소 목록 확인
python tools/crawlers/download_weather.py --list-elements
```

주요 요소:
- **기온**: 기온(TA), 최고기온(TA_MAX), 최저기온(TA_MIN)
- **습도**: 평균습도(HM), 최소습도(HM_MIN)
- **강수**: 일강수량(RN), 강수유무(RN_YN)
- **바람**: 평균풍속(WS), 최대풍속(WS_MAX), 풍향(WD)
- **일사/일조**: 일사량(SI), 일조시간(SS)
- **기압**: 현지기압(PA), 해면기압(PS)
- **지면**: 지면온도(TS), 지중온도(TE005, TE010 등)
- **기타**: 이슬점온도(TD), 운량(CA_TOT), 적설(SD_NEW)

## 자연어 명령 예시

다음과 같은 자연어 명령을 지원합니다:

```
"제주도 2013~2024년 기온 데이터를 csv파일로 다운로드해줘"
"서울 2020년 1월부터 2023년 12월까지 습도와 강수량 데이터"
"부산 지역의 최근 5년간 일사량 자료"
"고산 2024년 풍속 데이터"
"대전 시간별 기온 데이터 2024년"
```

파싱되는 정보:
- **지점**: 지역명 자동 인식
- **기간**: 연도 범위, 단일 연도, "최근 N년" 지원
- **요소**: 기온, 습도, 강수, 바람 등 키워드 인식
- **자료형태**: "시간별", "일별" 자동 인식

## 출력 파일

다운로드된 파일은 `data/raw/` 디렉토리에 저장됩니다.

파일명 형식:
```
{지점명}_{요소}_{시작일}_{종료일}.csv
예: 제주_기온_20130101_20241231.csv
```

## Claude Code에서 사용하기

Claude Code에서 다음과 같이 요청하면 자동으로 크롤러가 실행됩니다:

```
"제주도 2013~2024년 기온 데이터를 data/raw 폴더에 다운로드해줘"
```

Claude Code는 이 요청을 파싱하여 다음 명령을 실행합니다:

```bash
python tools/crawlers/download_weather.py "제주도 2013~2024년 기온 데이터를 다운로드해줘"
```

## 문제 해결

### Chrome/ChromeDriver 관련 오류

```bash
# ChromeDriver 설치 확인
which chromedriver

# Homebrew로 설치
brew install chromedriver

# 또는 webdriver-manager 자동 설치
pip install webdriver-manager
```

### API 키 오류

```bash
# API 키 환경변수 확인
echo $KMA_API_KEY

# 환경변수 설정
export KMA_API_KEY='your_key'
```

### 타임아웃 오류

기상청 서버가 느릴 때 발생합니다. 재시도하거나 API 방식을 사용하세요.

```bash
# 브라우저 창 표시하여 디버깅
python tools/crawlers/download_weather.py --no-headless --station 제주 --start 2024-01-01 --end 2024-12-31
```

## 프로젝트 구조

```
power-demand-forecast/
├── tools/
│   └── crawlers/
│       ├── __init__.py
│       ├── config.py          # 지점코드, 요소코드 정의
│       ├── kma_crawler.py     # Selenium 크롤러
│       ├── kma_api.py         # 공공데이터포털 API 클라이언트
│       └── download_weather.py # CLI 스크립트
├── data/
│   └── raw/                   # 다운로드된 파일 저장
└── WEATHER_CRAWLER_README.md
```

## 참고 자료

- [기상자료개방포털](https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36)
- [공공데이터포털 ASOS API](https://www.data.go.kr/data/15057210/openapi.do)
- [기상청 관측지점 정보](https://data.kma.go.kr/tmeta/stn/selectStnList.do)

---

**설정 완료일**: 2025-12-14
**작성자**: Claude Code
