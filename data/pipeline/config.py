from pathlib import Path

# TourAPI 4.0 (KorService2)
BASE_URL = "https://apis.data.go.kr/B551011/KorService2"

# 부산 법정동 시도코드 (areaCode=6 은 미사용/삭제 예정)
BUSAN_LDONG_REGN_CD = "26"

# 수집 대상 콘텐츠 타입 (RAG 코스 구성에 필요한 핵심만)
CONTENT_TYPES = {
    "12": "관광지",
    "14": "문화시설",
    "15": "축제/공연/행사",
    "28": "레포츠",
    "32": "숙박",
    "38": "쇼핑",
    "39": "음식점",
}

# 부산 구/군 이름 → 권역 (동선 필터링용)
REGION_ZONES = {
    "해운대구": "동부산권",
    "기장군": "동부산권",
    "수영구": "동부산권",
    "금정구": "동부산권",
    "동래구": "동부산권",
    "연제구": "중부산권",
    "부산진구": "중부산권",
    "동구": "원도심권",
    "중구": "원도심권",
    "서구": "원도심권",
    "영도구": "원도심권",
    "남구": "중부산권",
    "사상구": "서부산권",
    "사하구": "서부산권",
    "북구": "서부산권",
    "강서구": "서부산권",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DETAIL_CACHE_PATH = RAW_DIR / "detail_common_cache.json"
FAISS_DIR = PROCESSED_DIR / "faiss_index"
DEFAULT_EMBEDDING_MODEL = "jhgan/ko-sroberta-multitask"
