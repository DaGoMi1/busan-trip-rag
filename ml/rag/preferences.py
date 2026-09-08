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

# 목적별 카테고리 검색 쿼터 (하드 필터 대신 가중 multi-recall)
PURPOSE_CATEGORY_QUOTAS: dict[str, dict[str, int]] = {
    "자연/액티비티": {
        "관광지": 8,
        "레포츠": 4,
        "음식점": 5,
        "문화시설": 2,
        "쇼핑": 1,
    },
    "맛집": {
        "음식점": 10,
        "관광지": 3,
        "문화시설": 2,
        "쇼핑": 1,
    },
    "문화관광": {
        "문화시설": 6,
        "관광지": 6,
        "음식점": 5,
        "쇼핑": 1,
    },
    "쇼핑": {
        "쇼핑": 8,
        "음식점": 5,
        "관광지": 3,
        "문화시설": 2,
    },
    "힐링": {
        "관광지": 6,
        "음식점": 5,
        "문화시설": 4,
        "쇼핑": 1,
    },
    "종합": {
        "관광지": 5,
        "음식점": 5,
        "문화시설": 4,
        "레포츠": 3,
        "쇼핑": 1,
    },
}

CATEGORY_QUERIES: dict[str, str] = {
    "관광지": "관광지 명소 산책 바다 풍경",
    "음식점": "맛집 음식 로컬 식당",
    "문화시설": "문화 박물관 전시 갤러리",
    "쇼핑": "쇼핑 시장 상점",
    "레포츠": "레포츠 액티비티 체험 스포츠",
    "축제": "축제 행사 공연",
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


def quotas_for_purpose(purpose: str, days: int = 1) -> dict[str, int]:
    base = PURPOSE_CATEGORY_QUOTAS.get(purpose, PURPOSE_CATEGORY_QUOTAS["종합"])
    scale = max(1, days)
    return {category: count * scale for category, count in base.items()}


def primary_categories_for_purpose(purpose: str) -> set[str]:
    return set(categories_for_purpose(purpose))


def purpose_query(purpose: str, district: str = "", companion: str = "") -> str:
    theme = PURPOSE_QUERIES.get(purpose, PURPOSE_QUERIES["종합"])
    location = f"{district} " if district else ""
    who = f"{companion} " if companion and companion != "기타" else ""
    return f"부산 {location}{who}{theme} 여행".strip()


def category_query(category: str, district: str = "", companion: str = "") -> str:
    theme = CATEGORY_QUERIES.get(category, category)
    location = f"{district} " if district else ""
    who = f"{companion} " if companion and companion != "기타" else ""
    return f"부산 {location}{who}{theme}".strip()


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


def demote_shopping(hits: list[dict[str, Any]], purpose: str) -> list[dict[str, Any]]:
    """쇼핑 목적이 아니면 쇼핑 장소를 뒤로 밀거나 제외 우선순위를 낮춥니다."""
    if purpose == "쇼핑":
        return hits
    non_shopping = [hit for hit in hits if hit.get("category") != "쇼핑"]
    shopping = [hit for hit in hits if hit.get("category") == "쇼핑"]
    if not non_shopping:
        return shopping
    return non_shopping + shopping[: max(0, 2)]


def ensure_daily_mix(
    hits: list[dict[str, Any]],
    pick_n: int,
    purpose: str,
) -> list[dict[str, Any]]:
    """하루 후보에 가능하면 주요 목적 장소 + 음식점 최소 1곳을 넣습니다."""
    if not hits or pick_n <= 0:
        return []

    hits = demote_shopping(hits, purpose)
    primary = primary_categories_for_purpose(purpose)
    food = [hit for hit in hits if hit.get("category") == "음식점"]
    primary_hits = [hit for hit in hits if hit.get("category") in primary]
    other = [
        hit
        for hit in hits
        if hit.get("category") not in primary and hit.get("category") != "음식점"
    ]

    if purpose == "맛집":
        return (food or hits)[:pick_n]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(place: dict[str, Any]) -> None:
        key = str(place.get("id") or place.get("name"))
        if key in seen:
            return
        seen.add(key)
        selected.append(place)

    food_slot = 1 if food and "음식점" not in primary else 0
    primary_slots = max(1, pick_n - food_slot)

    for place in primary_hits:
        if len([p for p in selected if p.get("category") in primary]) >= primary_slots:
            break
        _add(place)

    if food_slot:
        _add(food[0])

    for place in primary_hits + food + other + hits:
        if len(selected) >= pick_n:
            break
        _add(place)

    return selected[:pick_n]
