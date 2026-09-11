from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from backend.schemas.request import RecommendRequest
from ml.rag.preferences import (
    ANCHOR_RADIUS_BASE_KM,
    ANCHOR_RADIUS_CAP_KM,
    DAILY_PLACE_COUNT,
    INTENSITY_GUIDES,
    anchor_radius_km,
    apply_purpose_bonus,
    category_query,
    daily_slot_template,
    ensure_daily_mix,
    food_count_for_purpose,
    max_hop_km,
    place_count_for_purpose,
    quotas_for_purpose,
    stay_index_for_day,
)
from ml.rag.prompt_templates import course_prompt
from ml.rag.retriever import PlaceRetriever
from ml.rag.routing import (
    distance_between,
    order_day_route,
    route_start_and_end,
)

WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")
CATEGORY_ORDER = ("관광지", "레포츠", "문화시설", "음식점", "쇼핑", "축제")


class MissingLodgingError(ValueError):
    """요청한 숙소 id가 숙박 목록에 없을 때."""


class CourseRag:
    """목적 가중 multi-recall 후 LLM 조합(또는 extractive) → NN 동선."""

    def __init__(self, retriever: PlaceRetriever) -> None:
        self.retriever = retriever
        self._rag_chain = None

    @classmethod
    def from_disk(cls) -> CourseRag:
        return cls(PlaceRetriever.from_disk())

    def recommend(self, request: RecommendRequest) -> str:
        lodgings = self.resolve_lodgings(request.lodging_ids)
        candidate_plans = self._retrieve_by_day(request, lodgings, for_llm=True)
        if not any(plan["hits"] for plan in candidate_plans):
            return "조건에 맞는 장소를 찾지 못했습니다. 활동 강도나 목적을 바꿔 다시 시도해 주세요."

        if os.getenv("OPENAI_API_KEY"):
            return self._generate_with_llm(request, candidate_plans)
        final_plans = self._narrow_plans(request, candidate_plans)
        return self._generate_extractive(request, final_plans)

    def resolve_lodgings(self, lodging_ids: list[str]) -> list[dict[str, Any]]:
        resolved: list[dict[str, Any]] = []
        missing: list[str] = []
        for lodging_id in lodging_ids:
            place = self.retriever.get_by_id(lodging_id)
            if not place or place.get("category") != "숙박":
                missing.append(lodging_id)
            else:
                resolved.append(place)
        if missing:
            raise MissingLodgingError(
                "선택한 숙소를 찾을 수 없습니다: " + ", ".join(missing)
            )
        return resolved

    def _retrieve_by_day(
        self,
        request: RecommendRequest,
        lodgings: list[dict[str, Any]],
        *,
        for_llm: bool,
    ) -> list[dict[str, Any]]:
        dates = _trip_dates(request)
        pick_n = DAILY_PLACE_COUNT
        # LLM용 후보는 슬롯 채울 정도만 — 문맥·토큰 절약
        bucket_size = pick_n * 2 if for_llm else pick_n * 3
        day_lodgings = _expand_lodgings(lodgings, request.days)
        plans: list[dict[str, Any]] = []

        for day_index, trip_date in enumerate(dates):
            today = day_lodgings[day_index]
            yesterday = day_lodgings[day_index - 1] if day_index > 0 else None
            anchors = _day_anchors(yesterday, today)
            stay = stay_index_for_day(day_lodgings, day_index)
            radius_km = anchor_radius_km(stay)

            next_lodging = None
            if day_index + 1 < len(day_lodgings):
                nxt = day_lodgings[day_index + 1]
                if str(nxt.get("id")) != str(today.get("id")):
                    next_lodging = nxt

            exclude = set()
            for anchor in anchors:
                exclude.add(str(anchor.get("id")))

            hits = self._search_weighted_hits(
                request,
                anchors,
                radius_km=radius_km,
                exclude_ids=exclude,
                next_lodging=next_lodging,
            )
            capped = _cap_hits_preserving_template(
                hits,
                max(bucket_size + 4, bucket_size),
                request.purpose,
            )
            plans.append(
                {
                    "date": trip_date,
                    "lodging": today,
                    "anchors": anchors,
                    "hits": capped,
                    "radius_km": radius_km,
                }
            )
        return plans

    def _top_up_pool_for_template(
        self,
        request: RecommendRequest,
        pool: list[dict[str, Any]],
        *,
        lodging: dict[str, Any],
        radius_km: float,
        exclude_ids: set[str],
    ) -> list[dict[str, Any]]:
        """일차 간 중복 제외 후 음식점·비음식이 부족하면 근처에서 보충합니다."""
        need_food = food_count_for_purpose(request.purpose)
        need_place = place_count_for_purpose(request.purpose)
        have_food = sum(1 for hit in pool if hit.get("category") == "음식점")
        have_place = sum(1 for hit in pool if hit.get("category") != "음식점")
        if have_food >= need_food and have_place >= need_place:
            return pool

        origin_lat = _coord(lodging, "latitude")
        origin_lng = _coord(lodging, "longitude")
        district = str(lodging.get("district") or "")
        extra: list[dict[str, Any]] = []
        soft_radius = min(max(radius_km + 2.0, 4.0), ANCHOR_RADIUS_CAP_KM)

        if have_food < need_food:
            query = category_query("음식점", district, request.companion)
            vecs = self.retriever.encode_queries([query])
            extra.extend(
                self.retriever.search(
                    query=query,
                    k=max(8, need_food * 3),
                    category="음식점",
                    origin_lat=origin_lat,
                    origin_lng=origin_lng,
                    max_distance_km=soft_radius,
                    exclude_ids=exclude_ids,
                    exclude_categories=["숙박"],
                    intensity=request.intensity,
                    query_embedding=vecs.get(query),
                )
            )
        if have_place < need_place:
            primary = next(
                (
                    cat
                    for cat, count in quotas_for_purpose(request.purpose).items()
                    if count > 0 and cat != "음식점"
                ),
                "관광지",
            )
            query = category_query(primary, district, request.companion)
            vecs = self.retriever.encode_queries([query])
            extra.extend(
                self.retriever.search(
                    query=query,
                    k=max(8, need_place * 3),
                    category=primary,
                    origin_lat=origin_lat,
                    origin_lng=origin_lng,
                    max_distance_km=soft_radius,
                    exclude_ids=exclude_ids,
                    exclude_categories=["숙박"],
                    intensity=request.intensity,
                    query_embedding=vecs.get(query),
                )
            )

        if not extra:
            return pool
        merged = _merge_anchor_hits(pool + extra, [lodging])
        return apply_purpose_bonus(merged, request.purpose)

    def _narrow_plans(
        self,
        request: RecommendRequest,
        candidate_plans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pick_n = DAILY_PLACE_COUNT
        narrowed: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for plan in candidate_plans:
            lodging = plan["lodging"]
            start, end = route_start_and_end(lodging, plan.get("anchors"))
            radius = float(plan.get("radius_km") or ANCHOR_RADIUS_BASE_KM)
            hop_limit = max_hop_km(request.intensity, radius)
            pool = [
                hit
                for hit in plan["hits"]
                if str(hit.get("id") or hit.get("name")) not in used_ids
            ]
            pool = self._top_up_pool_for_template(
                request,
                pool,
                lodging=lodging,
                radius_km=radius,
                exclude_ids=used_ids | {str(lodging.get("id"))},
            )
            mixed = ensure_daily_mix(pool, pick_n, request.purpose)
            ordered = order_day_route(start, mixed, destination=end) if mixed else []
            ordered = enforce_max_hop(
                start,
                ordered,
                hop_limit,
                pool=pool,
                purpose=request.purpose,
                pick_n=pick_n,
                destination=end,
            )
            ordered = ensure_meals_in_day_plan(
                start,
                ordered,
                pool=pool,
                purpose=request.purpose,
                pick_n=pick_n,
            )
            narrowed.append({**plan, "hits": ordered})
            for hit in ordered:
                used_ids.add(str(hit.get("id") or hit.get("name")))
        return narrowed

    def _search_weighted_hits(
        self,
        request: RecommendRequest,
        anchors: list[dict[str, Any]],
        *,
        radius_km: float,
        exclude_ids: set[str],
        next_lodging: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        quotas = quotas_for_purpose(request.purpose)
        jobs: list[dict[str, Any]] = []

        for lodging in anchors:
            origin_lat = _coord(lodging, "latitude")
            origin_lng = _coord(lodging, "longitude")
            district = str(lodging.get("district") or "")
            for category, count in quotas.items():
                if count <= 0:
                    continue
                jobs.append(
                    {
                        "query": category_query(category, district, request.companion),
                        "k": count,
                        "category": category,
                        "origin_lat": origin_lat,
                        "origin_lng": origin_lng,
                        "max_distance_km": radius_km,
                    }
                )

        query_vecs = self.retriever.encode_queries(
            [str(job["query"]) for job in jobs]
        )
        pooled: list[dict[str, Any]] = []
        for job in jobs:
            hits = self.retriever.search(
                query=str(job["query"]),
                k=int(job["k"]),
                category=job["category"],
                origin_lat=job["origin_lat"],
                origin_lng=job["origin_lng"],
                max_distance_km=job["max_distance_km"],
                exclude_ids=exclude_ids,
                exclude_categories=["숙박"],
                intensity=request.intensity,
                query_embedding=query_vecs.get(str(job["query"])),
            )
            pooled.extend(hits)

        if len(pooled) < max(3, DAILY_PLACE_COUNT - 1):
            for lodging in anchors:
                origin_lat = _coord(lodging, "latitude")
                origin_lng = _coord(lodging, "longitude")
                district = str(lodging.get("district") or "")
                query = category_query("관광지", district, request.companion)
                vecs = self.retriever.encode_queries([query])
                fallback = self.retriever.search(
                    query=query,
                    k=DAILY_PLACE_COUNT * 2,
                    origin_lat=origin_lat,
                    origin_lng=origin_lng,
                    max_distance_km=radius_km,
                    exclude_ids=exclude_ids,
                    exclude_categories=["숙박"],
                    intensity=request.intensity,
                    query_embedding=vecs.get(query),
                )
                pooled.extend(fallback)

        pooled = _merge_anchor_hits(pooled, anchors)
        if len(pooled) < 2:
            soft_radius = min(radius_km + 1.0, ANCHOR_RADIUS_CAP_KM)
            soft_jobs: list[dict[str, Any]] = []
            for lodging in anchors:
                origin_lat = _coord(lodging, "latitude")
                origin_lng = _coord(lodging, "longitude")
                district = str(lodging.get("district") or "")
                query = (
                    category_query("음식점", district, request.companion)
                    if request.purpose == "맛집"
                    else category_query("관광지", district, request.companion)
                )
                soft_jobs.append(
                    {
                        "query": query,
                        "origin_lat": origin_lat,
                        "origin_lng": origin_lng,
                    }
                )
            soft_vecs = self.retriever.encode_queries(
                [str(job["query"]) for job in soft_jobs]
            )
            for job in soft_jobs:
                soft = self.retriever.search(
                    query=str(job["query"]),
                    k=DAILY_PLACE_COUNT * 3,
                    origin_lat=job["origin_lat"],
                    origin_lng=job["origin_lng"],
                    max_distance_km=soft_radius,
                    exclude_ids=exclude_ids,
                    exclude_categories=["숙박"],
                    intensity=request.intensity,
                    query_embedding=soft_vecs.get(str(job["query"])),
                )
                pooled.extend(soft)
            pooled = _merge_anchor_hits(pooled, anchors)

        pooled = apply_purpose_bonus(pooled, request.purpose)
        if next_lodging is not None:
            pooled = apply_next_lodging_bonus(pooled, next_lodging)
        return pooled

    def _generate_with_llm(
        self,
        request: RecommendRequest,
        candidate_plans: list[dict[str, Any]],
    ) -> str:
        pick_n = DAILY_PLACE_COUNT
        radii = [float(plan.get("radius_km") or 2.0) for plan in candidate_plans]
        if radii:
            band_text = (
                f"전날·당일 숙소 각각 {min(radii):.0f}~{max(radii):.0f}km 이내 "
                f"(OR, 동일 숙소면 2.5→4→5.5km 확대)"
            )
        else:
            band_text = "전날·당일 숙소 각 2km 이내(OR)"
        payload = {
            "days": request.days,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "companion": request.companion,
            "purpose": request.purpose,
            "radius_text": band_text,
            "intensity": request.intensity,
            "intensity_guide": INTENSITY_GUIDES[request.intensity],
            "pick_n": pick_n,
            "context": format_day_plans_by_category(candidate_plans),
        }
        raw = self._get_rag_chain().invoke(payload)
        final_plans = apply_llm_selections(
            candidate_plans,
            raw,
            pick_n,
            purpose=request.purpose,
            intensity=request.intensity,
            pool_enricher=lambda pool, plan, used: self._top_up_pool_for_template(
                request,
                pool,
                lodging=plan["lodging"],
                radius_km=float(plan.get("radius_km") or ANCHOR_RADIUS_BASE_KM),
                exclude_ids=used | {str(plan["lodging"].get("id"))},
            ),
        )
        if not any(plan["hits"] for plan in final_plans):
            final_plans = self._narrow_plans(request, candidate_plans)
            body = self._generate_extractive(request, final_plans)
            return body + "\n\n_(장소 id를 읽지 못해, 검색 결과로 일정을 다시 붙였어요.)_\n"
        return render_composed_course(request, final_plans, raw)

    def _get_rag_chain(self):
        if self._rag_chain is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=700)
            self._rag_chain = create_course_chain(llm)
        return self._rag_chain

    def _generate_extractive(
        self,
        request: RecommendRequest,
        day_plans: list[dict[str, Any]],
    ) -> str:
        return render_course_markdown(
            request,
            day_plans,
            reasons={},
            draft_note="검색 결과로 먼저 붙여 본 초안이에요. (API 키가 없어 AI 문장 조합은 생략)",
        )


def create_course_chain(llm) -> Any:
    return RunnablePassthrough() | course_prompt() | llm | StrOutputParser()


def _cap_hits_preserving_template(
    hits: list[dict[str, Any]],
    limit: int,
    purpose: str,
) -> list[dict[str, Any]]:
    """점수 상위 절단 시에도 슬롯용 음식점·비음식 후보를 남깁니다."""
    if limit <= 0 or not hits:
        return []
    if len(hits) <= limit:
        return hits

    need_food = food_count_for_purpose(purpose) * 3
    need_place = place_count_for_purpose(purpose) * 3
    foods = [hit for hit in hits if hit.get("category") == "음식점"]
    places = [hit for hit in hits if hit.get("category") != "음식점"]

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _add(place: dict[str, Any]) -> None:
        key = str(place.get("id") or place.get("name"))
        if key in seen or len(selected) >= limit:
            return
        seen.add(key)
        selected.append(place)

    for hit in foods[:need_food]:
        _add(hit)
    for hit in places[:need_place]:
        _add(hit)
    for hit in hits:
        if len(selected) >= limit:
            break
        _add(hit)
    return selected


def format_day_plans_by_category(day_plans: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for day_index, plan in enumerate(day_plans, start=1):
        lodging = plan["lodging"]
        radius = plan.get("radius_km")
        anchors = plan.get("anchors") or [lodging]
        start, end = route_start_and_end(lodging, anchors)
        anchor_names = " · ".join(
            f"{a.get('name')}({a.get('district')})" for a in anchors
        )
        radius_text = f", 각 앵커 {radius:.0f}km 이내(OR)" if radius is not None else ""
        if str(start.get("id")) != str(end.get("id")):
            route_text = (
                f"동선: {start.get('name')} → {end.get('name')} "
                f"(전날 숙소에서 당일 숙소 방향으로)\n"
            )
        else:
            route_text = f"동선: {lodging.get('name')} 기준\n"
        header = (
            f"### {day_index}일차 ({_format_date(plan['date'])}) "
            f"숙소: {lodging.get('name')} ({lodging.get('district')})\n"
            f"{route_text}"
            f"탐색 앵커: {anchor_names}{radius_text}\n"
            f"숙소 주소: {lodging.get('address') or '주소 없음'}\n"
            f"(아래 후보 중 선택 · id만 사용 · 목적 주 카테고리 우선)"
        )
        hits = plan.get("hits") or []
        if not hits:
            blocks.append(header + "\n후보 없음")
            continue
        # 점수 상위만 LLM에 넘겨 문맥을 줄입니다
        ranked = sorted(
            hits,
            key=lambda item: float(item.get("score") or 0.0),
            reverse=True,
        )[:14]
        by_category: dict[str, list[dict[str, Any]]] = {}
        for hit in ranked:
            by_category.setdefault(str(hit.get("category") or "기타"), []).append(hit)
        sections: list[str] = []
        ordered_cats = [c for c in CATEGORY_ORDER if c in by_category]
        ordered_cats += sorted(c for c in by_category if c not in CATEGORY_ORDER)
        for category in ordered_cats:
            lines = [f"#### {category}"]
            for hit in by_category[category]:
                place_id = hit.get("id") or hit.get("name")
                distance = hit.get("distance_km")
                distance_text = f" | 숙소 {distance}km" if distance is not None else ""
                overview = (
                    (hit.get("page_content") or hit.get("overview") or "").strip()
                    or "설명 없음"
                )
                lines.append(
                    f"[id={place_id}] {hit.get('name')} | {hit.get('district')} | "
                    f"{hit.get('category')}{distance_text}\n"
                    f"설명: {overview[:120]}"
                )
            sections.append("\n".join(lines))
        blocks.append(header + "\n\n" + "\n\n".join(sections))
    return "\n\n".join(blocks)


def apply_llm_selections(
    candidate_plans: list[dict[str, Any]],
    llm_text: str,
    pick_n: int,
    *,
    purpose: str,
    intensity: int = 3,
    pool_enricher: Any | None = None,
) -> list[dict[str, Any]]:
    selected_by_day = parse_selected_ids(llm_text)
    final_plans: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for day_index, plan in enumerate(candidate_plans, start=1):
        lodging = plan["lodging"]
        start, end = route_start_and_end(lodging, plan.get("anchors"))
        radius = float(plan.get("radius_km") or ANCHOR_RADIUS_BASE_KM)
        day_hop = max_hop_km(intensity, radius)
        pool = [
            hit
            for hit in plan.get("hits") or []
            if str(hit.get("id") or hit.get("name")) not in used_ids
        ]
        if pool_enricher is not None:
            pool = pool_enricher(pool, plan, used_ids)
        id_map = {
            str(hit.get("id") or hit.get("name")): hit for hit in pool
        }
        name_map = {
            str(hit.get("name")): hit for hit in pool if hit.get("name")
        }
        chosen_ids = selected_by_day.get(day_index) or []
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_id in chosen_ids:
            hit = id_map.get(raw_id) or name_map.get(raw_id)
            if not hit:
                continue
            key = str(hit.get("id") or hit.get("name"))
            if key in seen:
                continue
            seen.add(key)
            selected.append(hit)
            if len(selected) >= pick_n:
                break
        if not selected:
            selected = _match_names_from_text(llm_text, pool, pick_n)
        selected = _pad_selection_to_pick_n(
            selected,
            pool,
            pick_n,
            purpose,
            near=start,
        )
        ordered = order_day_route(start, selected, destination=end) if selected else []
        ordered = enforce_max_hop(
            start,
            ordered,
            day_hop,
            pool=pool,
            purpose=purpose,
            pick_n=pick_n,
            destination=end,
        )
        ordered = ensure_meals_in_day_plan(
            start,
            ordered,
            pool=pool,
            purpose=purpose,
            pick_n=pick_n,
        )
        final_plans.append({**plan, "hits": ordered})
        for hit in ordered:
            used_ids.add(str(hit.get("id") or hit.get("name")))
    return final_plans


def _pad_selection_to_pick_n(
    selected: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    pick_n: int,
    purpose: str,
    *,
    near: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """LLM 선택이 pick_n보다 적으면 가까운 후보 위주로 채웁니다."""
    if pick_n <= 0:
        return []
    if not selected:
        return ensure_daily_mix(pool, pick_n, purpose)
    if len(selected) >= pick_n or not pool:
        return selected[:pick_n]

    seen = {str(hit.get("id") or hit.get("name")) for hit in selected}
    padded = list(selected)
    remaining = [
        hit for hit in pool if str(hit.get("id") or hit.get("name")) not in seen
    ]

    def _near_key(hit: dict[str, Any]) -> float:
        if near is not None:
            dist = distance_between(near, hit)
            if dist is not None:
                return dist
        return float(hit.get("distance_km") or 999.0)

    # 목적 믹스 우선(식사 슬롯 포함), 그다음 숙소에서 가까운 순
    preferred = ensure_daily_mix(remaining, pick_n, purpose)
    preferred_ids = {str(h.get("id") or h.get("name")) for h in preferred}
    leftovers = [h for h in remaining if str(h.get("id") or h.get("name")) not in preferred_ids]
    leftovers.sort(key=_near_key)
    preferred_sorted = sorted(preferred, key=_near_key)

    for hit in preferred_sorted + leftovers:
        key = str(hit.get("id") or hit.get("name"))
        if key in seen:
            continue
        seen.add(key)
        padded.append(hit)
        if len(padded) >= pick_n:
            break
    return padded[:pick_n]


def _slot_sizes(n: int) -> list[int]:
    """하루 6곳이면 [2,2,2], 그 미만은 균등 분배."""
    if n <= 0:
        return []
    if n >= DAILY_PLACE_COUNT:
        return [2, 2, 2]
    if n == 1:
        return [1]
    if n == 2:
        return [1, 1]
    base, rem = divmod(n, 3)
    return [base + (1 if i < rem else 0) for i in range(3)]


def _recompute_hops(
    start: dict[str, Any],
    path: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rebuilt: list[dict[str, Any]] = []
    current = start
    for place in path:
        hop = distance_between(current, place)
        item = {**place}
        if hop is not None:
            item["hop_km"] = round(hop, 2)
        else:
            item.pop("hop_km", None)
        rebuilt.append(item)
        current = place
    return rebuilt


def arrange_meals_in_slots(
    hits: list[dict[str, Any]],
    purpose: str,
) -> list[dict[str, Any]]:
    """목적별 슬롯 템플릿(place/food) 순서로 하루 일정을 재배치합니다."""
    if not hits:
        return hits

    template = daily_slot_template(purpose)
    foods = [hit for hit in hits if hit.get("category") == "음식점"]
    places = [hit for hit in hits if hit.get("category") != "음식점"]
    food_i = 0
    place_i = 0
    result: list[dict[str, Any]] = []

    for slot in template:
        for role in slot:
            chosen = None
            if role == "food" and food_i < len(foods):
                chosen = foods[food_i]
                food_i += 1
            elif role == "place" and place_i < len(places):
                chosen = places[place_i]
                place_i += 1
            if chosen is not None:
                result.append(chosen)

    # 템플릿에 못 채운 자리는 남는 후보로만 뒤에 이어 붙임(역할 뒤섞지 않음)
    leftovers = foods[food_i:] + places[place_i:]
    for hit in leftovers:
        if len(result) >= DAILY_PLACE_COUNT:
            break
        key = str(hit.get("id") or hit.get("name"))
        if any(str(r.get("id") or r.get("name")) == key for r in result):
            continue
        result.append(hit)
    return result[:DAILY_PLACE_COUNT]


def ensure_meals_in_day_plan(
    start: dict[str, Any],
    ordered: list[dict[str, Any]],
    *,
    pool: list[dict[str, Any]],
    purpose: str,
    pick_n: int,
) -> list[dict[str, Any]]:
    """템플릿에 맞게 음식점·비음식 장소를 풀에서 다시 맞춰 슬롯 순서로 배치합니다."""
    target_n = min(pick_n, DAILY_PLACE_COUNT) if pick_n > 0 else DAILY_PLACE_COUNT
    if target_n <= 0:
        return []

    # 동선으로 고른 결과를 앞에 두고, 부족분은 풀에서 템플릿 수량으로 채움
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in list(ordered) + list(pool):
        key = str(hit.get("id") or hit.get("name"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(hit)

    mixed = ensure_daily_mix(merged, target_n, purpose)
    arranged = arrange_meals_in_slots(mixed, purpose)
    return _recompute_hops(start, arranged)


def parse_selected_ids(llm_text: str) -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    pattern = re.compile(
        r"SELECTED_IDS_DAY\s*(\d+)\s*:\s*([^\n]+)",
        flags=re.IGNORECASE,
    )
    for match in pattern.finditer(llm_text):
        day = int(match.group(1))
        raw = match.group(2).strip()
        ids = [part.strip().strip("`\"'") for part in re.split(r"[,，\s]+", raw) if part.strip()]
        # drop labels like id=
        cleaned = []
        for item in ids:
            item = re.sub(r"^id\s*=\s*", "", item, flags=re.IGNORECASE)
            if item:
                cleaned.append(item)
        found[day] = cleaned
    return found


def _match_names_from_text(
    llm_text: str,
    hits: list[dict[str, Any]],
    pick_n: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in sorted(hits, key=lambda h: len(str(h.get("name") or "")), reverse=True):
        name = str(hit.get("name") or "")
        if not name or name not in llm_text:
            continue
        key = str(hit.get("id") or name)
        if key in seen:
            continue
        seen.add(key)
        selected.append(hit)
        if len(selected) >= pick_n:
            break
    return selected


def render_composed_course(
    request: RecommendRequest,
    day_plans: list[dict[str, Any]],
    llm_text: str,
) -> str:
    return render_course_markdown(
        request,
        day_plans,
        reasons=_extract_reasons(llm_text),
        draft_note=None,
    )


def render_course_markdown(
    request: RecommendRequest,
    day_plans: list[dict[str, Any]],
    *,
    reasons: dict[str, str],
    draft_note: str | None,
) -> str:
    purpose = request.purpose
    title = f"## 부산 {request.days}일 · {purpose} 코스"
    if draft_note:
        title += " (초안)"

    intensity_soft = INTENSITY_GUIDES.get(request.intensity, "")

    lines = [
        title,
        "",
        (
            f"- **기간:** {request.start_date.month}/{request.start_date.day}"
            f" – {request.end_date.month}/{request.end_date.day}"
        ),
        f"- **동행:** {request.companion}",
        f"- **목적:** {purpose}",
    ]
    if intensity_soft:
        lines.append(f"- **강도:** {intensity_soft}")
    if draft_note:
        lines.append(f"- _{draft_note}_")
    lines.append("")

    for day_index, plan in enumerate(day_plans, start=1):
        lines.extend(_format_day_section(day_index, plan, reasons))
        lines.append("")
        lines.append("---")
        lines.append("")
    while lines and lines[-1] in ("", "---"):
        lines.pop()
    return "\n".join(lines).rstrip() + "\n"


def _split_day_slots(
    hits: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    """확정 장소 순서를 유지한 채 오전/오후/저녁으로 나눕니다."""
    n = len(hits)
    if n == 0:
        return []
    sizes = _slot_sizes(n)
    labels = ("오전", "오후", "저녁")[: len(sizes)]
    slots: list[tuple[str, list[dict[str, Any]]]] = []
    offset = 0
    for label, size in zip(labels, sizes, strict=True):
        if size <= 0:
            continue
        slots.append((label, hits[offset : offset + size]))
        offset += size
    return slots


_SLOT_HEADINGS = {
    "오전": "#### 오전",
    "오후": "#### 오후",
    "저녁": "#### 저녁",
}


def _human_distance(km: float) -> str:
    if km < 1.0:
        meters = int(round(km * 1000 / 10.0) * 10)
        return f"약 {meters}m"
    return f"약 {km:.1f}km"


def _place_blurb(place: dict[str, Any], reasons: dict[str, str]) -> str:
    """한 줄 소개. LLM 이유 우선, 없으면 짧은 기본 멘트."""
    name = str(place.get("name") or "")
    if name and reasons.get(name):
        return reasons[name].strip()

    category = str(place.get("category") or "")
    variants = {
        "음식점": (
            "가볍게 쉬어가기 좋은 곳이에요.",
            "한숨 돌리며 맛보기 좋아요.",
            "분위기 보며 쉬어가기 좋아요.",
        ),
        "문화시설": (
            "천천히 둘러보기 좋아요.",
            "조용히 구경하기 좋아요.",
            "공간 분위기 느껴보기 좋아요.",
        ),
        "관광지": (
            "여유롭게 산책하기 좋아요.",
            "바람 쐬며 걷기 좋아요.",
            "가볍게 둘러보기 좋아요.",
        ),
        "쇼핑": (
            "둘러보며 구경하기 좋아요.",
            "천천히 살펴보기 좋아요.",
        ),
        "레포츠": (
            "몸을 살짝 움직여 보기 좋아요.",
            "가볍게 체험해 보기 좋아요.",
        ),
    }
    overview = (place.get("overview") or place.get("page_content") or "").strip()
    if overview and len(overview) <= 48 and not overview.endswith(("이다.", "한다.", "있다.")):
        return re.sub(r"\s+", " ", overview)

    options = variants.get(category)
    if not options:
        return "일정에 맞춰 들르기 좋아요."
    return options[sum(ord(ch) for ch in name) % len(options)]


def _format_day_section(
    day_index: int,
    plan: dict[str, Any],
    reasons: dict[str, str],
) -> list[str]:
    lodging = plan["lodging"]
    start, end = route_start_and_end(lodging, plan.get("anchors"))
    moving = str(start.get("id")) != str(end.get("id"))
    trip_date = plan["date"]
    date_label = (
        f"{trip_date.month}월 {trip_date.day}일 "
        f"({WEEKDAYS[trip_date.weekday()]})"
    )

    # ### 일차 / #### 시간대 — 리스트로 줄바꿈을 강제해 Streamlit 가독성 확보
    lines = [f"### {day_index}일차 · {date_label}", ""]

    if moving:
        lines.append("🧳 **이동일**")
        lines.append("")
        lines.append(
            f"- **동선:** {start.get('name')} → {end.get('name')}"
        )
        lines.append("- 체크아웃 후, 다음 숙소 쪽으로 가볍게 이어가는 하루예요.")
    else:
        district = lodging.get("district") or "구 미상"
        lines.append(f"🏠 **{lodging.get('name')}**")
        lines.append("")
        lines.append(f"- **구:** {district}")
        address = lodging.get("address")
        if address:
            lines.append(f"- **주소:** {address}")
    lines.append("")

    hits = plan.get("hits") or []
    if not hits:
        lines.append("조건에 맞는 장소를 찾지 못했어요. 강도나 숙소를 바꿔 보시면 좋아요.")
        return lines

    stop_index = 0
    for slot_label, slot_hits in _split_day_slots(hits):
        lines.append(_SLOT_HEADINGS.get(slot_label, f"#### {slot_label}"))
        lines.append("")
        for place in slot_hits:
            stop_index += 1
            name = str(place.get("name") or "이름 없음")
            district = place.get("district") or "구 미상"
            category = place.get("category") or "기타"

            lines.append(f"**{stop_index}. {name}**")
            lines.append("")
            lines.append(f"- **구:** {district}")
            lines.append(f"- **카테고리:** {category}")

            blurb = _place_blurb(place, reasons)
            if blurb:
                lines.append(f"- **소개:** {blurb}")

            hop = place.get("hop_km")
            distance = place.get("distance_km")
            if hop is not None:
                label = "숙소에서" if stop_index == 1 else "앞에서"
                lines.append(
                    f"- **이동:** {label} {_human_distance(float(hop))}"
                )
            elif distance is not None:
                lines.append(
                    f"- **이동:** 숙소에서 {_human_distance(float(distance))}"
                )
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return lines


def _extract_reasons(llm_text: str) -> dict[str, str]:
    reasons: dict[str, str] = {}
    pattern = re.compile(
        r"^[\-\*]\s+\*?\*?([^*\n(]+?)\*?\*?\s*\([^)]*\)\s*[—\-–]\s*(.+)$",
        flags=re.MULTILINE,
    )
    for match in pattern.finditer(llm_text):
        name = match.group(1).strip()
        reason = match.group(2).strip()
        if name and reason:
            reasons[name] = reason
    # 번호 목록 형식도 허용: 1. **이름** ... — 이유
    numbered = re.compile(
        r"^\d+\.\s+\*?\*?([^*\n·|(]+?)\*?\*?[^\n]*?[—\-–]\s*(.+)$",
        flags=re.MULTILINE,
    )
    for match in numbered.finditer(llm_text):
        name = match.group(1).strip()
        reason = match.group(2).strip()
        if name and reason and name not in reasons:
            reasons[name] = reason
    return reasons


def _day_route_header(day_index: int, plan: dict[str, Any]) -> str:
    """하위 호환용 한 줄 헤더 (포맷 헬퍼에서 주로 _format_day_section 사용)."""
    lodging = plan["lodging"]
    start, end = route_start_and_end(lodging, plan.get("anchors"))
    date_text = _format_date(plan["date"])
    if str(start.get("id")) != str(end.get("id")):
        return (
            f"### {day_index}일차 ({date_text}) "
            f"이동: {start.get('name')}({start.get('district')}) "
            f"→ {end.get('name')}({end.get('district')})"
        )
    return (
        f"### {day_index}일차 ({date_text}) "
        f"시작: {lodging.get('name')} ({lodging.get('district')})"
    )


def _expand_lodgings(
    lodgings: list[dict[str, Any]],
    days: int,
) -> list[dict[str, Any]]:
    if len(lodgings) == 1:
        return [lodgings[0] for _ in range(days)]
    if len(lodgings) != days:
        raise MissingLodgingError("숙소는 1곳이거나 일차 수와 같아야 합니다.")
    return list(lodgings)


def _day_anchors(
    yesterday: dict[str, Any] | None,
    today: dict[str, Any],
) -> list[dict[str, Any]]:
    """전날 숙소 + 당일 숙소(다르면 둘 다, 같으면 하나)."""
    if yesterday is None:
        return [today]
    if str(yesterday.get("id")) == str(today.get("id")):
        return [today]
    return [yesterday, today]


def _merge_anchor_hits(
    hits: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """이중 앵커 검색 결과를 합치고, 거리는 가장 가까운 앵커 기준으로 둔다."""
    merged: dict[str, dict[str, Any]] = {}
    for hit in hits:
        key = str(hit.get("id") or hit.get("name"))
        dists = []
        for anchor in anchors:
            distance = distance_between(anchor, hit)
            if distance is not None:
                dists.append(distance)
        distance_km = round(min(dists), 2) if dists else hit.get("distance_km")
        score = float(hit.get("score") or 0.0)
        if key not in merged:
            merged[key] = {**hit, "distance_km": distance_km, "score": score}
            continue
        prev = merged[key]
        prev_score = float(prev.get("score") or 0.0)
        prev_dist = prev.get("distance_km")
        best_dist = distance_km
        if prev_dist is not None and (best_dist is None or prev_dist < best_dist):
            best_dist = prev_dist
        merged[key] = {
            **prev,
            **hit,
            "score": max(prev_score, score),
            "distance_km": best_dist,
        }
    ordered = sorted(merged.values(), key=lambda item: item.get("score") or 0, reverse=True)
    return ordered


def apply_next_lodging_bonus(
    hits: list[dict[str, Any]],
    next_lodging: dict[str, Any],
) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for hit in hits:
        distance = distance_between(hit, next_lodging)
        bonus = 0.0
        if distance is not None:
            bonus = max(0.0, 0.25 * (1.0 - min(distance, 20.0) / 20.0))
        updated.append(
            {
                **hit,
                "score": float(hit.get("score") or 0.0) + bonus,
                "next_lodging_km": round(distance, 2) if distance is not None else None,
            }
        )
    updated.sort(key=lambda item: item.get("score") or 0, reverse=True)
    return updated


def enforce_max_hop(
    start: dict[str, Any],
    ordered: list[dict[str, Any]],
    hop_limit: float,
    *,
    pool: list[dict[str, Any]],
    purpose: str,
    pick_n: int,
    destination: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """hop 상한을 지키며 pick_n까지 채웁니다.

    먼 지점에서 막히면 숙소 기준 순수 탐욕 경로와 비교해
    더 많이 채운 쪽을 고릅니다(동일 숙소 2일차 등).
    """
    if pick_n <= 0:
        return []

    def _within_limit(path: list[dict[str, Any]]) -> bool:
        current = start
        for place in path:
            hop = distance_between(current, place)
            if hop is not None and hop > hop_limit:
                return False
            current = place
        return True

    def _path_score(path: list[dict[str, Any]]) -> tuple[int, float]:
        """장소 수 우선, 동선 합이 짧을수록 가산."""
        total = 0.0
        current = start
        for place in path:
            hop = distance_between(current, place)
            if hop is not None:
                total += hop
            current = place
        return (len(path), -total)

    candidates: list[list[dict[str, Any]]] = []

    if ordered and _within_limit(ordered):
        kept = list(ordered)[:pick_n]
        if len(kept) < pick_n:
            kept = _greedy_fill_hops(
                start,
                kept,
                hop_limit,
                pool=pool,
                pick_n=pick_n,
                destination=destination,
            )
        candidates.append(kept)

    mixed = ensure_daily_mix(pool, max(pick_n, 3), purpose)
    rebuilt: list[dict[str, Any]] = []
    current = start
    for place in order_day_route(start, mixed, destination=destination):
        hop = distance_between(current, place)
        if hop is not None and hop > hop_limit:
            continue
        item = {**place}
        if hop is not None:
            item["hop_km"] = round(hop, 2)
        rebuilt.append(item)
        current = place
        if len(rebuilt) >= pick_n:
            break
    if len(rebuilt) < pick_n:
        rebuilt = _greedy_fill_hops(
            start,
            rebuilt,
            hop_limit,
            pool=pool,
            pick_n=pick_n,
            destination=destination,
        )
    if rebuilt:
        candidates.append(rebuilt)

    pure = _greedy_fill_hops(
        start,
        [],
        hop_limit,
        pool=pool,
        pick_n=pick_n,
        destination=destination,
    )
    if pure:
        candidates.append(pure)

    if not candidates:
        return list(ordered)[:pick_n]

    best = max(candidates, key=_path_score)
    return best[:pick_n]


def _greedy_fill_hops(
    start: dict[str, Any],
    path: list[dict[str, Any]],
    hop_limit: float,
    *,
    pool: list[dict[str, Any]],
    pick_n: int,
    destination: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """현재 경로 끝에서 hop 상한 안 후보를 탐욕적으로 이어 붙입니다.

    숙소에 머무는 날은 숙소 근처 밀집 구간을 우선해, 먼 곳으로 빠져
    pick_n을 못 채우는 경우를 줄입니다.
    """
    filled = list(path)
    seen = {str(hit.get("id") or hit.get("name")) for hit in filled}
    current = filled[-1] if filled else start
    moving = (
        destination is not None
        and str(destination.get("id") or "") != str(start.get("id") or "")
    )

    while len(filled) < pick_n:
        best = None
        best_score = float("inf")
        for hit in pool:
            key = str(hit.get("id") or hit.get("name"))
            if key in seen:
                continue
            hop = distance_between(current, hit)
            if hop is None:
                hop = float(hit.get("distance_km") or 999.0)
            if hop > hop_limit:
                continue
            if moving and destination is not None:
                to_dest = distance_between(hit, destination)
                score = hop + (0.55 * to_dest if to_dest is not None else 0.0)
            else:
                to_start = distance_between(start, hit)
                # 숙소 주변 클러스터를 유지 (먼 전망대 한 곳으로 고립 방지)
                score = hop + (0.9 * to_start if to_start is not None else 0.0)
            if score < best_score:
                best = hit
                best_score = score
        if best is None:
            break
        key = str(best.get("id") or best.get("name"))
        seen.add(key)
        hop = distance_between(current, best)
        item = {**best}
        if hop is not None:
            item["hop_km"] = round(hop, 2)
        filled.append(item)
        current = best
    return filled


def _trip_dates(request: RecommendRequest) -> list[date]:
    return [
        request.start_date + timedelta(days=offset)
        for offset in range(request.days)
    ]


def _format_date(value: date) -> str:
    return f"{value.isoformat()} {WEEKDAYS[value.weekday()]}"


def _coord(place: dict[str, Any], field: str) -> float | None:
    try:
        return float(place.get(field))
    except (TypeError, ValueError):
        return None
