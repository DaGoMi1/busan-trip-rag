from __future__ import annotations

from typing import Any

from ml.vectorstore.faiss_store import haversine_km

# 이동일: 다음 후보 고를 때 도착 숙소 방향 가중 (hop + λ * 도착까지 거리)
_DESTINATION_PULL = 0.55


def place_coord(place: dict[str, Any]) -> tuple[float, float] | None:
    try:
        return float(place["latitude"]), float(place["longitude"])
    except (KeyError, TypeError, ValueError):
        return None


def distance_between(a: dict[str, Any], b: dict[str, Any]) -> float | None:
    ca, cb = place_coord(a), place_coord(b)
    if ca is None or cb is None:
        return None
    return haversine_km(ca[0], ca[1], cb[0], cb[1])


def order_nearest_neighbor(
    origin: dict[str, Any],
    places: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """숙소에서 시작해 미방문 최근접을 반복합니다."""
    return order_day_route(origin, places, destination=None)


def order_day_route(
    start: dict[str, Any],
    places: list[dict[str, Any]],
    destination: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """하루 동선 정렬.

    - 동일 숙소(또는 destination 없음): start 기준 nearest-neighbor
    - 숙소 이동일: start(전날 숙소) → destination(당일 숙소) 방향으로
      짧은 hop을 유지하면서 점점 도착 숙소 쪽으로 진행
    """
    if not places:
        return []

    moving = (
        destination is not None
        and str(destination.get("id") or "") != str(start.get("id") or "")
    )
    remaining = list(places)
    ordered: list[dict[str, Any]] = []
    current = start

    while remaining:
        best_index = 0
        best_score = float("inf")
        for index, candidate in enumerate(remaining):
            hop = distance_between(current, candidate)
            if hop is None:
                hop = float(candidate.get("distance_km") or 999.0)
            score = hop
            if moving and destination is not None:
                to_dest = distance_between(candidate, destination)
                if to_dest is not None:
                    score = hop + _DESTINATION_PULL * to_dest
            if score < best_score:
                best_score = score
                best_index = index

        next_place = remaining.pop(best_index)
        hop = distance_between(current, next_place)
        updated = {**next_place}
        if hop is not None:
            updated["hop_km"] = round(hop, 2)
        ordered.append(updated)
        current = next_place

    return ordered


def route_start_and_end(
    lodging: dict[str, Any],
    anchors: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """동선 출발·도착 숙소. 앵커가 둘이면 전날→당일, 아니면 당일만."""
    today = lodging
    if anchors and len(anchors) >= 2:
        return anchors[0], today
    return today, today
