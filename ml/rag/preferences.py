from __future__ import annotations

from typing import Any

PURPOSE_CATEGORIES: dict[str, list[str]] = {
    "힐링": ["관광지", "음식점", "문화시설"],
    "맛집": ["음식점"],
    "문화관광": ["관광지", "문화시설"],
    "쇼핑": ["쇼핑", "음식점"],
    "자연/액티비티": ["관광지", "레포츠"],
    "종합": ["관광지", "음식점", "문화시설", "레포츠"],
}

PURPOSE_QUERIES: dict[str, str] = {
    "힐링": "힐링 산책 카페 바다 휴식",
    "맛집": "맛집 음식 로컬 식당",
    "문화관광": "문화 역사 박물관 관광지",
    "쇼핑": "쇼핑 시장 상점",
    "자연/액티비티": "자연 산 바다 트레킹 액티비티",
    "종합": "관광 맛집 문화",
}

RADIUS_KM: dict[int, float] = {1: 3.0, 2: 6.0, 3: 10.0, 4: 15.0, 5: 25.0}

INTENSITY_K: dict[int, int] = {1: 8, 2: 10, 3: 12, 4: 14, 5: 16}
INTENSITY_PICK: dict[int, int] = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}

INTENSITY_GUIDES: dict[int, str] = {
    1: "움직임 최소화, 카페·실내 위주, 하루 2~3곳",
    2: "가벼운 산책 가능, 하루 3~4곳",
    3: "보통 도보 관광, 하루 4~5곳",
    4: "다소 긴 이동·언덕 괜찮음, 하루 5~6곳",
    5: "산·숲·장거리 걷기 가능, 하루 5~7곳",
}

HIGH_INTENSITY_KEYWORDS = (
    "등산",
    "트레킹",
    "암벽",
    "해파랑",
    "숲길",
    "산림",
    "산행",
    "둘레길",
    "금정산",
    "장산",
    "백양산",
    "황령산",
    "봉우리",
)
LOW_INTENSITY_KEYWORDS = ("카페", "온천", "전시", "박물관", "갤러리", "스파")

LODGING_FIELDS = ("id", "name", "district", "address", "latitude", "longitude")


def radius_km(radius: int) -> float:
    return RADIUS_KM.get(radius, 10.0)


def categories_for_purpose(purpose: str) -> list[str]:
    return PURPOSE_CATEGORIES.get(purpose, PURPOSE_CATEGORIES["종합"])


def purpose_query(purpose: str, district: str = "", companion: str = "") -> str:
    theme = PURPOSE_QUERIES.get(purpose, PURPOSE_QUERIES["종합"])
    location = f"{district} " if district else ""
    who = f"{companion} " if companion and companion != "기타" else ""
    return f"부산 {location}{who}{theme} 여행".strip()


def intensity_label(hit: dict[str, Any]) -> str:
    text = f"{hit.get('name') or ''} {hit.get('overview') or ''} {hit.get('page_content') or ''}"
    if any(keyword in text for keyword in HIGH_INTENSITY_KEYWORDS):
        return "high"
    if any(keyword in text for keyword in LOW_INTENSITY_KEYWORDS):
        return "low"
    return "mid"


def apply_intensity(hits: list[dict[str, Any]], intensity: int) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for hit in hits:
        label = intensity_label(hit)
        if intensity <= 2 and label == "high":
            continue
        bonus = 0.0
        if intensity >= 4 and label == "high":
            bonus = 0.15
        elif intensity <= 2 and label == "low":
            bonus = 0.1
        updated = {**hit, "intensity_label": label, "score": float(hit.get("score") or 0) + bonus}
        filtered.append(updated)
    filtered.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return filtered
