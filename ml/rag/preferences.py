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
    "맛집": {
        "음식점": 16,
        "관광지": 6,
        "문화시설": 4,
        "쇼핑": 2,
    },
    "쇼핑": {
        "쇼핑": 12,
        "음식점": 8,
        "관광지": 2,
        "문화시설": 1,
    },
    "힐링": {
        "관광지": 10,
        "음식점": 8,
        "문화시설": 6,
        "쇼핑": 0,
    },
    "종합": {
        "관광지": 8,
        "음식점": 8,
        "문화시설": 5,
        "레포츠": 4,
        "쇼핑": 1,
    },
    "문화관광": {
        "문화시설": 10,
        "관광지": 10,
        "음식점": 8,
        "쇼핑": 0,
    },
    "자연/액티비티": {
        "관광지": 12,
        "레포츠": 8,
        "음식점": 8,
        "문화시설": 1,
        "쇼핑": 0,
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
# 하루 장소 수는 강도와 무관하게 고정 (오전/오후/저녁 × 2)
DAILY_PLACE_COUNT = 6
DEFAULT_MAX_HOP_KM = 3.5

INTENSITY_GUIDES: dict[int, str] = {
    1: "낮음 · 카페·실내·완만한 산책",
    2: "가벼운 산책 · 골목·전망·여유",
    3: "보통 · 일반적인 도보 관광",
    4: "활발 · 언덕·긴 도보 OK",
    5: "높음 · 산·숲·트레킹 가능",
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


def max_hop_km(intensity: int | None = None, radius_km: float | None = None) -> float:
    """연속 이동 상한. 강도는 장소 성격만 담당하므로 hop은 반경 기준 고정."""
    _ = intensity
    hop = DEFAULT_MAX_HOP_KM
    if radius_km is not None:
        hop = min(hop, max(1.2, float(radius_km)))
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
    """강도는 장소 성격 점수만 조정(완전 제외하지 않음)."""
    scored: list[dict[str, Any]] = []
    for hit in hits:
        label = intensity_label(hit)
        bonus = 0.0
        if intensity <= 2:
            if label == "low":
                bonus = 0.18
            elif label == "mid":
                bonus = 0.05
            else:
                bonus = -0.25
        elif intensity >= 4:
            if label == "high":
                bonus = 0.18
            elif label == "mid":
                bonus = 0.05
            else:
                bonus = -0.12
        else:
            if label == "mid":
                bonus = 0.08
        scored.append(
            {
                **hit,
                "intensity_label": label,
                "score": float(hit.get("score") or 0) + bonus,
            }
        )
    scored.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return scored


def apply_purpose_bonus(hits: list[dict[str, Any]], purpose: str) -> list[dict[str, Any]]:
    """목적 주 카테고리 점수 가산."""
    primary = primary_categories_for_purpose(purpose)
    updated: list[dict[str, Any]] = []
    for hit in hits:
        bonus = PURPOSE_PRIMARY_SCORE_BONUS if hit.get("category") in primary else 0.0
        updated.append({**hit, "score": float(hit.get("score") or 0.0) + bonus})
    updated.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return updated


def daily_slot_template(purpose: str) -> list[list[str]]:
    """오전/오후/저녁 슬롯별 역할. place=비음식, food=음식점."""
    if purpose == "맛집":
        return [
            ["food", "food"],
            ["food", "place"],
            ["food", "food"],
        ]
    return [
        ["place", "food"],
        ["place", "food"],
        ["place", "food"],
    ]


def food_count_for_purpose(purpose: str) -> int:
    return sum(1 for slot in daily_slot_template(purpose) for role in slot if role == "food")


def place_count_for_purpose(purpose: str) -> int:
    return sum(1 for slot in daily_slot_template(purpose) for role in slot if role == "place")


def ensure_daily_mix(
    hits: list[dict[str, Any]],
    pick_n: int,
    purpose: str,
) -> list[dict[str, Any]]:
    """목적별 슬롯 템플릿에 맞게 음식점·비음식 장소를 채웁니다."""
    if not hits or pick_n <= 0:
        return []

    target_n = min(pick_n, DAILY_PLACE_COUNT)
    primary = primary_categories_for_purpose(purpose)
    food = [hit for hit in hits if hit.get("category") == "음식점"]
    non_food = [hit for hit in hits if hit.get("category") != "음식점"]
    primary_places = [hit for hit in non_food if hit.get("category") in primary]
    other_places = [hit for hit in non_food if hit.get("category") not in primary]
    place_pool = primary_places + other_places

    need_food = food_count_for_purpose(purpose)
    need_place = place_count_for_purpose(purpose)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(place: dict[str, Any]) -> bool:
        key = str(place.get("id") or place.get("name"))
        if key in seen:
            return False
        seen.add(key)
        selected.append(place)
        return True

    food_added = 0
    for hit in food:
        if food_added >= need_food or len(selected) >= target_n:
            break
        if _add(hit):
            food_added += 1

    place_added = 0
    for hit in place_pool:
        if place_added >= need_place or len(selected) >= target_n:
            break
        if _add(hit):
            place_added += 1

    for hit in hits:
        if len(selected) >= target_n:
            break
        _add(hit)

    return selected[:target_n]
