from __future__ import annotations

from typing import Any

from ml.vectorstore.faiss_store import haversine_km


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


def mean_distance_to_set(
    place: dict[str, Any],
    others: list[dict[str, Any]],
) -> float:
    if not others:
        return 0.0
    distances = [
        distance_between(place, other)
        for other in others
        if distance_between(place, other) is not None
    ]
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def order_nearest_neighbor(
    origin: dict[str, Any],
    places: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """숙소(또는 시드)에서 시작해 미방문 최근접을 반복합니다."""
    remaining = list(places)
    ordered: list[dict[str, Any]] = []
    current = origin
    while remaining:
        best_index = 0
        best_distance = float("inf")
        for index, candidate in enumerate(remaining):
            distance = distance_between(current, candidate)
            if distance is None:
                distance = float(candidate.get("distance_km") or 999.0)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        next_place = remaining.pop(best_index)
        hop = distance_between(current, next_place)
        updated = {**next_place}
        if hop is not None:
            updated["hop_km"] = round(hop, 2)
        ordered.append(updated)
        current = next_place
    return ordered


def build_day_buckets(
    hits: list[dict[str, Any]],
    days: int,
    pick_n: int,
    lodging: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    """같은 숙소 다중일: 시드(이전 일차와 먼 곳) + 인근 채우기 + NN 정렬."""
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(days)]
    if days <= 0 or not hits:
        return buckets

    unused = list(hits)
    used_ids: set[str] = set()
    previous_day_places: list[dict[str, Any]] = []

    for day_index in range(days):
        pool = [
            hit
            for hit in unused
            if str(hit.get("id") or hit.get("name")) not in used_ids
        ]
        if not pool:
            break

        seed = _pick_seed(pool, previous_day_places)
        day_hits = [seed]
        seed_id = str(seed.get("id") or seed.get("name"))
        used_ids.add(seed_id)

        while len(day_hits) < pick_n:
            candidates = [
                hit
                for hit in pool
                if str(hit.get("id") or hit.get("name")) not in used_ids
            ]
            if not candidates:
                break
            anchor = day_hits[-1]
            def _near_key(hit: dict[str, Any]) -> float:
                distance = distance_between(anchor, hit)
                if distance is not None:
                    return distance
                return float(hit.get("distance_km") or 999.0)

            candidates.sort(key=_near_key)
            chosen = candidates[0]
            day_hits.append(chosen)
            used_ids.add(str(chosen.get("id") or chosen.get("name")))

        ordered = order_nearest_neighbor(lodging, day_hits)
        buckets[day_index] = ordered
        previous_day_places = ordered
        unused = [
            hit
            for hit in unused
            if str(hit.get("id") or hit.get("name")) not in used_ids
        ]

    return buckets


def _pick_seed(
    pool: list[dict[str, Any]],
    previous_day_places: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked = sorted(
        pool,
        key=lambda hit: float(hit.get("score") or 0.0),
        reverse=True,
    )
    top = ranked[: max(5, min(12, len(ranked)))]
    if not previous_day_places:
        return top[0]
    return max(top, key=lambda hit: mean_distance_to_set(hit, previous_day_places))
