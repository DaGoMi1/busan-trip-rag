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
    INTENSITY_GUIDES,
    INTENSITY_PICK,
    anchor_radius_km,
    apply_purpose_bonus,
    category_query,
    ensure_daily_mix,
    max_hop_km,
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
        pick_n = INTENSITY_PICK[request.intensity]
        bucket_size = pick_n * 2 if for_llm else pick_n
        day_lodgings = _expand_lodgings(lodgings, request.days)
        plans: list[dict[str, Any]] = []
        used_ids: set[str] = set()

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

            exclude = set(used_ids)
            for anchor in anchors:
                exclude.add(str(anchor.get("id")))

            hits = self._search_weighted_hits(
                request,
                anchors,
                radius_km=radius_km,
                exclude_ids=exclude,
                next_lodging=next_lodging,
            )
            capped = hits[: max(bucket_size + 4, bucket_size)]
            plans.append(
                {
                    "date": trip_date,
                    "lodging": today,
                    "anchors": anchors,
                    "hits": capped,
                    "radius_km": radius_km,
                }
            )
            for hit in capped:
                used_ids.add(str(hit.get("id") or hit.get("name")))
        return plans

    def _narrow_plans(
        self,
        request: RecommendRequest,
        candidate_plans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        pick_n = INTENSITY_PICK[request.intensity]
        narrowed: list[dict[str, Any]] = []
        for plan in candidate_plans:
            lodging = plan["lodging"]
            start, end = route_start_and_end(lodging, plan.get("anchors"))
            radius = float(plan.get("radius_km") or ANCHOR_RADIUS_BASE_KM)
            hop_limit = max_hop_km(request.intensity, radius)
            mixed = ensure_daily_mix(plan["hits"], pick_n, request.purpose)
            ordered = order_day_route(start, mixed, destination=end) if mixed else []
            ordered = enforce_max_hop(
                start,
                ordered,
                hop_limit,
                pool=plan["hits"],
                purpose=request.purpose,
                pick_n=pick_n,
                destination=end,
            )
            narrowed.append({**plan, "hits": ordered})
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
        pooled: list[dict[str, Any]] = []

        for lodging in anchors:
            origin_lat = _coord(lodging, "latitude")
            origin_lng = _coord(lodging, "longitude")
            district = str(lodging.get("district") or "")
            for category, count in quotas.items():
                if count <= 0:
                    continue
                hits = self.retriever.search(
                    category_query(category, district, request.companion),
                    k=count,
                    category=category,
                    origin_lat=origin_lat,
                    origin_lng=origin_lng,
                    max_distance_km=radius_km,
                    exclude_ids=exclude_ids,
                    exclude_categories=["숙박"],
                    intensity=request.intensity,
                )
                pooled.extend(hits)

            if len(pooled) < max(3, INTENSITY_PICK[request.intensity] - 1):
                fallback = self.retriever.search(
                    category_query("관광지", district, request.companion),
                    k=INTENSITY_PICK[request.intensity] * 2,
                    origin_lat=origin_lat,
                    origin_lng=origin_lng,
                    max_distance_km=radius_km,
                    exclude_ids=exclude_ids,
                    exclude_categories=["숙박"],
                    intensity=request.intensity,
                )
                pooled.extend(fallback)

        pooled = _merge_anchor_hits(pooled, anchors)
        if len(pooled) < 2:
            # 후보가 거의 없을 때만 반경 +1km 완화
            soft_radius = min(radius_km + 1.0, ANCHOR_RADIUS_CAP_KM)
            for lodging in anchors:
                origin_lat = _coord(lodging, "latitude")
                origin_lng = _coord(lodging, "longitude")
                district = str(lodging.get("district") or "")
                query = (
                    category_query("음식점", district, request.companion)
                    if request.purpose == "맛집"
                    else category_query("관광지", district, request.companion)
                )
                soft = self.retriever.search(
                    query,
                    k=INTENSITY_PICK[request.intensity] * 3,
                    origin_lat=origin_lat,
                    origin_lng=origin_lng,
                    max_distance_km=soft_radius,
                    exclude_ids=exclude_ids,
                    exclude_categories=["숙박"],
                    intensity=request.intensity,
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
        pick_n = INTENSITY_PICK[request.intensity]
        radii = [float(plan.get("radius_km") or 2.0) for plan in candidate_plans]
        if radii:
            band_text = (
                f"전날·당일 숙소 각각 {min(radii):.0f}~{max(radii):.0f}km 이내 "
                f"(OR, 동일 숙소면 2→3→4km 확대)"
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
        )
        if not any(plan["hits"] for plan in final_plans):
            final_plans = self._narrow_plans(request, candidate_plans)
            body = self._generate_extractive(request, final_plans)
            return body + "\n\n_(LLM 선택 id 파싱에 실패해 검색 기반 일정으로 대체했습니다.)_\n"
        return render_composed_course(request, final_plans, raw)

    def _get_rag_chain(self):
        if self._rag_chain is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            self._rag_chain = create_course_chain(llm)
        return self._rag_chain

    def _generate_extractive(
        self,
        request: RecommendRequest,
        day_plans: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"## 부산 {request.days}일 여행 코스 (검색 기반 초안)",
            "",
            f"- 기간: {request.start_date.isoformat()} ~ {request.end_date.isoformat()}",
            f"- 동행: {request.companion} / 목적: {request.purpose} / 활동 강도: {request.intensity}/5",
            "- OPENAI_API_KEY가 없어 LLM 조합 없이, 이중 숙소 앵커·목적 가중 검색·NN 동선 결과를 반환합니다.",
            "",
        ]
        for day_index, plan in enumerate(day_plans, start=1):
            lines.append(_day_route_header(day_index, plan))
            picked = plan["hits"]
            if not picked:
                lines.append("- 탐색 밴드 안에서 후보를 찾지 못했습니다.")
                lines.append("")
                continue
            for place in picked:
                distance = place.get("distance_km")
                hop = place.get("hop_km")
                distance_text = f", 숙소 {distance}km" if distance is not None else ""
                hop_text = f", 이전→{hop}km" if hop is not None else ""
                lines.append(
                    f"- **{place.get('name')}** "
                    f"({place.get('category')}, {place.get('district')}{distance_text}{hop_text})"
                )
                overview = (place.get("overview") or place.get("page_content") or "").strip()
                if overview:
                    lines.append(
                        f"  - {overview[:120]}{'...' if len(overview) > 120 else ''}"
                    )
            lines.append("")
        return "\n".join(lines)


def create_course_chain(llm) -> Any:
    return RunnablePassthrough() | course_prompt() | llm | StrOutputParser()


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
        by_category: dict[str, list[dict[str, Any]]] = {}
        for hit in hits:
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
                    f"{hit.get('region_zone')}{distance_text}\n"
                    f"주소: {hit.get('address')}\n"
                    f"설명: {overview[:300]}"
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
) -> list[dict[str, Any]]:
    selected_by_day = parse_selected_ids(llm_text)
    final_plans: list[dict[str, Any]] = []
    for day_index, plan in enumerate(candidate_plans, start=1):
        lodging = plan["lodging"]
        start, end = route_start_and_end(lodging, plan.get("anchors"))
        radius = float(plan.get("radius_km") or ANCHOR_RADIUS_BASE_KM)
        day_hop = max_hop_km(intensity, radius)
        id_map = {
            str(hit.get("id") or hit.get("name")): hit for hit in plan.get("hits") or []
        }
        name_map = {
            str(hit.get("name")): hit
            for hit in plan.get("hits") or []
            if hit.get("name")
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
            selected = _match_names_from_text(llm_text, plan.get("hits") or [], pick_n)
        if not selected:
            selected = ensure_daily_mix(plan.get("hits") or [], pick_n, purpose)
        ordered = order_day_route(start, selected, destination=end) if selected else []
        ordered = enforce_max_hop(
            start,
            ordered,
            day_hop,
            pool=plan.get("hits") or [],
            purpose=purpose,
            pick_n=pick_n,
            destination=end,
        )
        final_plans.append({**plan, "hits": ordered})
    return final_plans


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
    reasons = _extract_reasons(llm_text)
    lines = [
        f"## 부산 {request.days}일 여행 코스",
        "",
        f"- 기간: {request.start_date.isoformat()} ~ {request.end_date.isoformat()}",
        f"- 동행: {request.companion} / 목적: {request.purpose} / 활동 강도: {request.intensity}/5",
        "- 숙소 이중 앵커(전날·당일) · 목적 가중 multi-recall 후, 동선은 전날→당일 숙소 방향으로 정렬했습니다.",
        "",
    ]
    for day_index, plan in enumerate(day_plans, start=1):
        lines.append(_day_route_header(day_index, plan))
        if not plan["hits"]:
            lines.append("- 후보를 확정하지 못했습니다.")
            lines.append("")
            continue
        for place in plan["hits"]:
            distance = place.get("distance_km")
            hop = place.get("hop_km")
            meta = f"{place.get('district')}, {place.get('category')}"
            if distance is not None:
                meta += f", 숙소 {distance}km"
            if hop is not None:
                meta += f", 이전→{hop}km"
            name = str(place.get("name") or "")
            reason = reasons.get(name)
            if not reason:
                overview = (place.get("overview") or place.get("page_content") or "").strip()
                reason = (overview[:80] + "…") if len(overview) > 80 else overview
            if reason:
                lines.append(f"- **{name}** ({meta}) — {reason}")
            else:
                lines.append(f"- **{name}** ({meta})")
        lines.append("")
    return "\n".join(lines)


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
    return reasons


def _day_route_header(day_index: int, plan: dict[str, Any]) -> str:
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
    """연속 구간이 hop_limit을 넘으면 같은 날 풀에서 다시 고릅니다."""
    if not ordered:
        return []
    ok = True
    current = start
    for place in ordered:
        hop = distance_between(current, place)
        if hop is not None and hop > hop_limit:
            ok = False
            break
        current = place
    if ok:
        return ordered

    mixed = ensure_daily_mix(pool, pick_n, purpose)
    rebuilt = order_day_route(start, mixed, destination=destination)
    # hop 상한만 다시 검사하며 잘라 넣기
    trimmed: list[dict[str, Any]] = []
    current = start
    for place in rebuilt:
        hop = distance_between(current, place)
        if hop is not None and hop > hop_limit:
            continue
        item = {**place}
        if hop is not None:
            item["hop_km"] = round(hop, 2)
        trimmed.append(item)
        current = place
        if len(trimmed) >= pick_n:
            break
    if len(trimmed) >= min(2, pick_n):
        return trimmed

    # 최후: hop만 보고 탐욕
    seen: set[str] = set()
    current = start
    candidates = list(mixed) + [h for h in pool if h not in mixed]
    while len(trimmed) < pick_n and candidates:
        best = None
        best_score = float("inf")
        for hit in candidates:
            key = str(hit.get("id") or hit.get("name"))
            if key in seen:
                continue
            hop = distance_between(current, hit)
            if hop is None:
                hop = float(hit.get("distance_km") or 999.0)
            if hop > hop_limit:
                continue
            score = hop
            if (
                destination is not None
                and str(destination.get("id") or "") != str(start.get("id") or "")
            ):
                to_dest = distance_between(hit, destination)
                if to_dest is not None:
                    score = hop + 0.55 * to_dest
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
        trimmed.append(item)
        current = best
        candidates = [h for h in candidates if str(h.get("id") or h.get("name")) != key]
    return trimmed if trimmed else ordered


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
