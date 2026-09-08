from __future__ import annotations

import re
from typing import Any

from ml.rag.preferences import primary_categories_for_purpose, radius_km
from ml.rag.routing import distance_between, place_coord
from ml.vectorstore.faiss_store import haversine_km


def radius_compliance(day_plans: list[dict[str, Any]], radius: int) -> float:
    limit = radius_km(radius)
    total = 0
    ok = 0
    for plan in day_plans:
        for hit in plan.get("hits") or []:
            total += 1
            distance = hit.get("distance_km")
            if distance is None:
                lodging = plan.get("lodging") or {}
                distance = distance_between(lodging, hit)
            if distance is not None and distance <= limit + 1e-6:
                ok += 1
    return ok / total if total else 1.0


def day_compactness(day_plans: list[dict[str, Any]]) -> float:
    day_scores: list[float] = []
    for plan in day_plans:
        hits = plan.get("hits") or []
        coords = [place_coord(hit) for hit in hits]
        coords = [c for c in coords if c is not None]
        if len(coords) < 2:
            day_scores.append(0.0)
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lng = sum(c[1] for c in coords) / len(coords)
        distances = [haversine_km(lat, lng, c[0], c[1]) for c in coords]
        day_scores.append(sum(distances) / len(distances))
    return sum(day_scores) / len(day_scores) if day_scores else 0.0


def cross_day_overlap(day_plans: list[dict[str, Any]]) -> float:
    seen: set[str] = set()
    overlap = 0
    total = 0
    for plan in day_plans:
        for hit in plan.get("hits") or []:
            key = str(hit.get("id") or hit.get("name"))
            total += 1
            if key in seen:
                overlap += 1
            else:
                seen.add(key)
    return overlap / total if total else 0.0


def food_mix_rate(day_plans: list[dict[str, Any]]) -> float:
    if not day_plans:
        return 0.0
    with_food = 0
    for plan in day_plans:
        hits = plan.get("hits") or []
        if any(hit.get("category") == "음식점" for hit in hits):
            with_food += 1
    return with_food / len(day_plans)


def primary_share(day_plans: list[dict[str, Any]], purpose: str) -> float:
    """하루 일정 중 목적 주 카테고리 비율."""
    primary = primary_categories_for_purpose(purpose)
    total = 0
    matched = 0
    for plan in day_plans:
        for hit in plan.get("hits") or []:
            total += 1
            if hit.get("category") in primary:
                matched += 1
    return matched / total if total else 0.0


def faithfulness(text: str, day_plans: list[dict[str, Any]]) -> float:
    allowed_names = []
    for plan in day_plans:
        lodging = plan.get("lodging") or {}
        if lodging.get("name"):
            allowed_names.append(str(lodging["name"]))
        for hit in plan.get("hits") or []:
            if hit.get("name"):
                allowed_names.append(str(hit["name"]))
    allowed_names = sorted(set(allowed_names), key=len, reverse=True)
    mentioned = re.findall(r"^[\-\*]\s+\*?\*?([^*\n(]+)", text, flags=re.MULTILINE)
    mentioned = [name.strip() for name in mentioned if name.strip()]
    if not mentioned:
        return 1.0
    ok = 0
    for name in mentioned:
        if any(allowed in name or name in allowed for allowed in allowed_names):
            ok += 1
    return ok / len(mentioned)


def summarize_metrics(
    day_plans: list[dict[str, Any]],
    *,
    purpose: str,
    radius: int,
    llm_text: str | None = None,
) -> dict[str, float]:
    metrics = {
        "radius_compliance": round(radius_compliance(day_plans, radius), 4),
        "cross_day_overlap": round(cross_day_overlap(day_plans), 4),
        "food_mix_rate": round(food_mix_rate(day_plans), 4),
        "day_compactness": round(day_compactness(day_plans), 4),
        "primary_share": round(primary_share(day_plans, purpose), 4),
    }
    if llm_text is not None:
        metrics["faithfulness"] = round(faithfulness(llm_text, day_plans), 4)
    return metrics
