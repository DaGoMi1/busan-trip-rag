from __future__ import annotations

import os
import re
from datetime import date, timedelta
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from backend.schemas.request import RecommendRequest
from ml.rag.preferences import (
    INTENSITY_GUIDES,
    INTENSITY_PICK,
    category_query,
    demote_shopping,
    ensure_daily_mix,
    quotas_for_purpose,
    radius_km,
)
from ml.rag.prompt_templates import course_prompt
from ml.rag.retriever import PlaceRetriever
from ml.rag.routing import build_day_buckets, order_nearest_neighbor

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
            return "조건에 맞는 장소를 찾지 못했습니다. 반경이나 목적을 바꿔 다시 시도해 주세요."

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

        if len(lodgings) == 1:
            lodging = lodgings[0]
            hits = self._search_weighted_hits(request, lodging, days=request.days)
            buckets = build_day_buckets(hits, request.days, bucket_size, lodging)
            plans = []
            for trip_date, chunk in zip(dates, buckets, strict=True):
                plans.append(
                    {
                        "date": trip_date,
                        "lodging": lodging,
                        "hits": list(chunk),
                    }
                )
            return plans

        plans: list[dict[str, Any]] = []
        for day_index, trip_date in enumerate(dates):
            lodging = lodgings[day_index]
            hits = self._search_weighted_hits(request, lodging, days=1)
            # LLM이면 상위 bucket_size만 후보로, extractive는 나중에 narrow
            capped = hits[: max(bucket_size + 4, bucket_size)]
            plans.append({"date": trip_date, "lodging": lodging, "hits": capped})
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
            mixed = ensure_daily_mix(plan["hits"], pick_n, request.purpose)
            ordered = order_nearest_neighbor(lodging, mixed) if mixed else []
            narrowed.append({**plan, "hits": ordered})
        return narrowed

    def _search_weighted_hits(
        self,
        request: RecommendRequest,
        lodging: dict[str, Any],
        *,
        days: int,
    ) -> list[dict[str, Any]]:
        max_km = radius_km(request.radius)
        origin_lat = _coord(lodging, "latitude")
        origin_lng = _coord(lodging, "longitude")
        exclude_ids = {str(lodging.get("id"))}
        district = str(lodging.get("district") or "")
        quotas = quotas_for_purpose(request.purpose, days=days)

        pooled: list[dict[str, Any]] = []
        for category, count in quotas.items():
            if count <= 0:
                continue
            hits = self.retriever.search(
                category_query(category, district, request.companion),
                k=count,
                category=category,
                origin_lat=origin_lat,
                origin_lng=origin_lng,
                max_distance_km=max_km,
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
                max_distance_km=max_km,
                exclude_ids=exclude_ids,
                exclude_categories=["숙박"],
                intensity=request.intensity,
            )
            pooled = _dedupe_hits(pooled + fallback)

        return demote_shopping(_dedupe_hits(pooled), request.purpose)

    def _generate_with_llm(
        self,
        request: RecommendRequest,
        candidate_plans: list[dict[str, Any]],
    ) -> str:
        pick_n = INTENSITY_PICK[request.intensity]
        payload = {
            "days": request.days,
            "start_date": request.start_date.isoformat(),
            "end_date": request.end_date.isoformat(),
            "companion": request.companion,
            "purpose": request.purpose,
            "radius": request.radius,
            "radius_km": radius_km(request.radius),
            "intensity": request.intensity,
            "intensity_guide": INTENSITY_GUIDES[request.intensity],
            "pick_n": pick_n,
            "context": format_day_plans_by_category(candidate_plans),
        }
        raw = self._get_rag_chain().invoke(payload)
        final_plans = apply_llm_selections(candidate_plans, raw, pick_n)
        # 선택이 비면 extractive로 폴백
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
            f"- 동행: {request.companion} / 목적: {request.purpose} / 반경: 약 {radius_km(request.radius)}km / 강도: {request.intensity}/5",
            "- OPENAI_API_KEY가 없어 LLM 조합 없이, 목적 가중 검색·믹스·NN 동선 결과를 반환합니다.",
            "",
        ]
        for day_index, plan in enumerate(day_plans, start=1):
            lodging = plan["lodging"]
            lines.append(
                f"### {day_index}일차 ({_format_date(plan['date'])}) "
                f"시작: {lodging.get('name')} ({lodging.get('district')})"
            )
            picked = plan["hits"]
            if not picked:
                lines.append("- 반경 안에서 후보를 찾지 못했습니다.")
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
        header = (
            f"### {day_index}일차 ({_format_date(plan['date'])}) "
            f"시작: {lodging.get('name')} ({lodging.get('district')})\n"
            f"숙소 주소: {lodging.get('address') or '주소 없음'}\n"
            f"(아래 후보 중 선택 · id만 사용)"
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
) -> list[dict[str, Any]]:
    selected_by_day = parse_selected_ids(llm_text)
    final_plans: list[dict[str, Any]] = []
    for day_index, plan in enumerate(candidate_plans, start=1):
        lodging = plan["lodging"]
        id_map = {
            str(hit.get("id") or hit.get("name")): hit for hit in plan.get("hits") or []
        }
        # name fallback map
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
            # try match place names appearing in that day's section of llm text
            selected = _match_names_from_text(llm_text, plan.get("hits") or [], pick_n)
        ordered = order_nearest_neighbor(lodging, selected) if selected else []
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
        f"- 동행: {request.companion} / 목적: {request.purpose} / 반경: 약 {radius_km(request.radius)}km / 강도: {request.intensity}/5",
        "- 목적 가중 multi-recall 후보에서 LLM이 조합하고, 동선은 nearest-neighbor로 확정했습니다.",
        "",
    ]
    for day_index, plan in enumerate(day_plans, start=1):
        lodging = plan["lodging"]
        lines.append(
            f"### {day_index}일차 ({_format_date(plan['date'])}) "
            f"시작: {lodging.get('name')} ({lodging.get('district')})"
        )
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


def _dedupe_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for hit in hits:
        key = str(hit.get("id") or hit.get("name"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)
    return unique
