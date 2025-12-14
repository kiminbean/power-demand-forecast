"""
기상청 기상자료개방포털 크롤러 설정
ASOS(종관기상관측) 지점코드 및 요소코드 정의
"""

# ASOS 종관기상관측 지점코드
# 참조: https://data.kma.go.kr/tmeta/stn/selectStnList.do
STATION_CODES = {
    # 서울/경기
    "서울": "108",
    "인천": "112",
    "수원": "119",
    "동두천": "98",
    "파주": "99",
    "이천": "203",
    "양평": "202",

    # 강원
    "춘천": "101",
    "원주": "114",
    "강릉": "105",
    "동해": "106",
    "속초": "90",
    "대관령": "100",
    "태백": "216",
    "철원": "95",
    "홍천": "212",
    "영월": "121",

    # 충청
    "대전": "133",
    "청주": "131",
    "충주": "127",
    "서산": "129",
    "천안": "232",
    "보령": "135",
    "부여": "236",
    "금산": "238",

    # 전라
    "광주": "156",
    "전주": "146",
    "군산": "140",
    "목포": "165",
    "여수": "168",
    "순천": "174",
    "남원": "247",
    "장흥": "260",
    "해남": "261",
    "고흥": "262",
    "완도": "170",
    "흑산도": "169",

    # 경북
    "대구": "143",
    "포항": "138",
    "안동": "136",
    "울릉도": "115",
    "영덕": "130",
    "문경": "273",
    "영주": "271",
    "봉화": "272",
    "영천": "281",
    "구미": "279",
    "경주": "137",

    # 경남
    "부산": "159",
    "울산": "152",
    "창원": "155",
    "진주": "192",
    "통영": "162",
    "밀양": "288",
    "거제": "294",
    "거창": "284",
    "합천": "285",
    "산청": "289",
    "남해": "295",

    # 제주
    "제주": "184",
    "서귀포": "189",
    "성산": "188",
    "고산": "185",
}

# 지점코드 역매핑 (코드 → 지역명)
STATION_NAMES = {v: k for k, v in STATION_CODES.items()}

# 기상 요소코드 (ASOS 일자료 기준)
# 참조: 기상자료개방포털 요소 선택 팝업
ELEMENT_CODES = {
    # 기온 관련
    "기온": "TA",           # 기온 (°C)
    "평균기온": "TA",
    "최고기온": "TA_MAX",    # 최고기온 (°C)
    "최저기온": "TA_MIN",    # 최저기온 (°C)
    "이슬점온도": "TD",      # 이슬점온도 (°C)

    # 습도 관련
    "습도": "HM",           # 평균상대습도 (%)
    "상대습도": "HM",
    "평균습도": "HM",
    "최소습도": "HM_MIN",   # 최소상대습도 (%)

    # 강수 관련
    "강수량": "RN",         # 일강수량 (mm)
    "일강수량": "RN",
    "강수유무": "RN_YN",    # 강수유무 (0/1)

    # 바람 관련
    "풍속": "WS",           # 평균풍속 (m/s)
    "평균풍속": "WS",
    "최대풍속": "WS_MAX",   # 최대풍속 (m/s)
    "최대순간풍속": "WS_GST", # 최대순간풍속 (m/s)
    "풍향": "WD",           # 풍향 (degree)

    # 기압 관련
    "기압": "PA",           # 현지기압 (hPa)
    "해면기압": "PS",       # 해면기압 (hPa)
    "증기압": "PV",         # 증기압 (hPa)

    # 일사/일조 관련
    "일사량": "SI",         # 일사량 (MJ/m²)
    "일조시간": "SS",       # 일조시간 (hr)
    "일조율": "SS_RATE",    # 일조율 (%)

    # 지면/토양 관련
    "지면온도": "TS",       # 지면온도 (°C)
    "초상온도": "TG_MIN",   # 초상최저온도 (°C)
    "지중온도5cm": "TE005", # 5cm 지중온도 (°C)
    "지중온도10cm": "TE010",
    "지중온도20cm": "TE020",
    "지중온도30cm": "TE030",

    # 구름/시정 관련
    "운량": "CA_TOT",       # 전운량 (1/10)
    "중하층운량": "CA_MID", # 중하층운량 (1/10)
    "시정": "VS",           # 시정 (10m)

    # 적설/증발 관련
    "적설": "SD_NEW",       # 신적설 (cm)
    "적설량": "SD_NEW",
    "최심적설": "SD_MAX",   # 최심적설 (cm)
    "증발량": "EV",         # 증발량 (mm)

    # 기타
    "전운량": "CA_TOT",
    "최저운고": "CH_MIN",   # 최저운고 (100m)
}

# 요소코드 역매핑
ELEMENT_NAMES = {v: k for k, v in ELEMENT_CODES.items()}

# 자료형태 코드
DATA_TYPE_CODES = {
    "일": "F00501",      # 일자료
    "시간": "F00502",    # 시간자료
    "분": "F00503",      # 분자료 (1분)
}

# 기본 설정
DEFAULT_CONFIG = {
    "base_url": "https://data.kma.go.kr/data/grnd/selectAsosRltmList.do?pgmNo=36",
    "download_url": "https://data.kma.go.kr/data/common/downloadDataCVS.do",
    "timeout": 30,
    "wait_time": 3,
    "max_retries": 3,
}

# 요소 키워드 매핑 (자연어 → 요소코드)
ELEMENT_KEYWORDS = {
    "온도": ["TA", "TA_MAX", "TA_MIN"],
    "기온": ["TA", "TA_MAX", "TA_MIN"],
    "temperature": ["TA", "TA_MAX", "TA_MIN"],
    "temp": ["TA", "TA_MAX", "TA_MIN"],

    "습도": ["HM", "HM_MIN"],
    "humidity": ["HM", "HM_MIN"],

    "강수": ["RN", "RN_YN"],
    "비": ["RN", "RN_YN"],
    "rain": ["RN", "RN_YN"],
    "precipitation": ["RN", "RN_YN"],

    "바람": ["WS", "WS_MAX", "WS_GST", "WD"],
    "풍속": ["WS", "WS_MAX", "WS_GST"],
    "wind": ["WS", "WS_MAX", "WS_GST", "WD"],

    "일사": ["SI", "SS"],
    "일조": ["SS", "SS_RATE"],
    "sunshine": ["SI", "SS"],

    "기압": ["PA", "PS"],
    "pressure": ["PA", "PS"],

    "지면": ["TS", "TG_MIN"],
    "지중": ["TE005", "TE010", "TE020", "TE030"],
    "soil": ["TS", "TE005", "TE010", "TE020", "TE030"],

    "구름": ["CA_TOT", "CA_MID"],
    "cloud": ["CA_TOT", "CA_MID"],

    "눈": ["SD_NEW", "SD_MAX"],
    "적설": ["SD_NEW", "SD_MAX"],
    "snow": ["SD_NEW", "SD_MAX"],

    "이슬점": ["TD"],
    "dewpoint": ["TD"],
}


def get_station_code(station_name: str) -> str:
    """지역명으로 지점코드 조회"""
    # 정확한 매칭
    if station_name in STATION_CODES:
        return STATION_CODES[station_name]

    # 부분 매칭
    for name, code in STATION_CODES.items():
        if station_name in name or name in station_name:
            return code

    raise ValueError(f"알 수 없는 지점: {station_name}. 사용 가능한 지점: {list(STATION_CODES.keys())}")


def get_element_codes(element_name: str) -> list:
    """요소명으로 요소코드 목록 조회"""
    element_lower = element_name.lower()

    # 정확한 매칭
    if element_name in ELEMENT_CODES:
        return [ELEMENT_CODES[element_name]]

    # 키워드 매칭
    for keyword, codes in ELEMENT_KEYWORDS.items():
        if keyword in element_lower or element_lower in keyword:
            return codes

    # 코드 직접 입력
    if element_name.upper() in ELEMENT_NAMES:
        return [element_name.upper()]

    raise ValueError(f"알 수 없는 요소: {element_name}. 사용 가능한 요소: {list(ELEMENT_CODES.keys())}")


def list_stations():
    """사용 가능한 지점 목록 출력"""
    print("=== ASOS 종관기상관측 지점 목록 ===")
    for name, code in sorted(STATION_CODES.items(), key=lambda x: x[1]):
        print(f"  {name}: {code}")


def list_elements():
    """사용 가능한 요소 목록 출력"""
    print("=== 기상 요소 목록 ===")
    unique_codes = {}
    for name, code in ELEMENT_CODES.items():
        if code not in unique_codes:
            unique_codes[code] = name
    for code, name in sorted(unique_codes.items()):
        print(f"  {name}: {code}")
