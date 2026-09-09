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

# 목적별 카테고리 검색 쿼터 (주 카테고리 비중을 더 높게)
PURPOSE_CATEGORY_QUOTAS: dict[str, dict[str, int]] = {
    "자연/액티비티": {
        "관광지": 12,
        "레포츠": 8,
        "음식점": 4,
        "문화시설": 1,
        "쇼핑": 0,
    },
    "맛집": {
        "음식점": 16,
        "관광지": 2,
        "문화시설": 1,
        "쇼핑": 0,
    },
    "문화관광": {
        "문화시설": 10,
        "관광지": 10,
        "음식점": 4,
        "쇼핑": 0,
    },
    "쇼핑": {
        "쇼핑": 12,
        "음식점": 2,
        "관광지": 2,
        "문화시설": 1,
    },
    "힐링": {
        "관광지": 10,
        "음식점": 6,
        "문화시설": 6,
        "쇼핑": 0,
    },
    "종합": {
        "관광지": 8,
        "음식점": 7,
        "문화시설": 5,
        "레포츠": 4,
        "쇼핑": 1,
    },
}
PURPOSE_PRIMARY_SCORE_BONUS = 0.28

CATEGORY_QUERIES: dict[str, str] = {
    "관광지": "관광지 명소 산책 바다 풍경",
    "음식점": "맛집 음식 로컬 식당",
    "문화시설": "문화 박물관 전시 갤러리",
    "쇼핑": "쇼핑 시장 상점 백화점",
    "레포츠": "레포츠 액티비티 체험 스포츠",
    "축제": "축제 행사 공연",
}

# 숙소 앵커 반경: 기본 + stay_index, 동일 숙소 연속 투숙 시 확대
ANCHOR_RADIUS_BASE_KM = 2.5
ANCHOR_RADIUS_CAP_KM = 8.0
MAX_HOP_KM_BY_INTENSITY: dict[int, float] = {
    1: 2.5,
    2: 3.0,
    3: 3.5,
    4: 4.0,
    5: 4.5,
}

INTENSITY_PICK: dict[int, int] = {1: 3, 2: 4, 3: 5, 4: 6, 5: 7}

INTENSITY_GUIDES: dict[int, str] = {
    1: "활동 강도 낮음 · 카페·실내 위주, 하루 2~3곳",
    2: "가벼운 산책 · 하루 3~4곳",
    3: "보통 도보 관광 · 하루 4~5곳",
    4: "활발 · 언덕·긴 일정 OK, 하루 5~6곳",
    5: "높음 · 산·숲 가능, 하루 5~7곳",
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


def anchor_radius_km(stay_index: int = 0) -> float:
    """동일 숙소 연속일(stay_index 0-based)에 따라 반경 확대.

    stay 0: 2.5km, stay 1: 4.0km, stay 2: 5.5km … (cap까지).
    1일차가 근거리 후보를 써도 이후 일자에 여유가 남도록 +1.5km씩 넓힙니다.
    """
    radius = ANCHOR_RADIUS_BASE_KM + max(0, stay_index) * 1.5
    return float(min(radius, ANCHOR_RADIUS_CAP_KM))


def max_hop_km(intensity: int, radius_km: float | None = None) -> float:
    hop = MAX_HOP_KM_BY_INTENSITY.get(intensity, 3.0)
    if radius_km is not None:
        hop = min(hop, max(1.2, radius_km))
    return hop


def stay_index_for_day(lodgings_per_day: list[dict[str, Any]], day_index: int) -> int:
    """day_index 시점에 같은 숙소로 연속 묵은 며칠째인지(0-based)."""
    if day_index < 0 or day_index >= len(lodgings_per_day):
        return 0
    current_id = str(lodgings_per_day[day_index].get("id") or "")
    stay = 0
    for offset in range(day_index - 1, -1, -1):
        prev_id = str(lodgings_per_day[offset].get("id") or "")
        if prev_id != current_id:
            break
        stay += 1
    return stay


def quotas_for_purpose(purpose: str) -> dict[str, int]:
    return dict(
        PURPOSE_CATEGORY_QUOTAS.get(purpose, PURPOSE_CATEGORY_QUOTAS["종합"])
    )


def primary_categories_for_purpose(purpose: str) -> set[str]:
    return set(PURPOSE_CATEGORIES.get(purpose, PURPOSE_CATEGORIES["종합"]))


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


def apply_purpose_bonus(hits: list[dict[str, Any]], purpose: str) -> list[dict[str, Any]]:
    """목적 주 카테고리 점수 가산."""
    primary = primary_categories_for_purpose(purpose)
    updated: list[dict[str, Any]] = []
    for hit in hits:
        bonus = PURPOSE_PRIMARY_SCORE_BONUS if hit.get("category") in primary else 0.0
        updated.append({**hit, "score": float(hit.get("score") or 0.0) + bonus})
    updated.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return updated


def ensure_daily_mix(
    hits: list[dict[str, Any]],
    pick_n: int,
    purpose: str,
) -> list[dict[str, Any]]:
    """하루 후보에 목적 주 카테고리 약 75% + 가능하면 음식 1곳을 넣습니다."""
    if not hits or pick_n <= 0:
        return []

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
    primary_slots = max(1, int((pick_n * 3 + 3) // 4))  # ~75%
    if food_slot:
        primary_slots = min(primary_slots, pick_n - food_slot)

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
