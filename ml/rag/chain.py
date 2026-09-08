from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from backend.schemas.request import RecommendRequest
from ml.rag.preferences import (
    INTENSITY_GUIDES,
    INTENSITY_K,
    INTENSITY_PICK,
    categories_for_purpose,
    purpose_query,
    radius_km,
)
from ml.rag.prompt_templates import course_prompt
from ml.rag.retriever import PlaceRetriever

WEEKDAYS = ("월", "화", "수", "목", "금", "토", "일")


class MissingLodgingError(ValueError):
    """요청한 숙소 id가 숙박 목록에 없을 때."""


class CourseRag:
    """일차별 숙소 좌표 검색 후 Prompt -> LLM."""

    def __init__(self, retriever: PlaceRetriever) -> None:
        self.retriever = retriever
        self._rag_chain = None

    @classmethod
    def from_disk(cls) -> CourseRag:
        return cls(PlaceRetriever.from_disk())

    def recommend(self, request: RecommendRequest) -> str:
        lodgings = self.resolve_lodgings(request.lodging_ids)
        day_plans = self._retrieve_by_day(request, lodgings)
        if not any(plan["hits"] for plan in day_plans):
            return "조건에 맞는 장소를 찾지 못했습니다. 반경이나 목적을 바꿔 다시 시도해 주세요."

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
            "context": format_day_plans(day_plans),
        }
        if os.getenv("OPENAI_API_KEY"):
            return self._get_rag_chain().invoke(payload)
        return self._generate_extractive(request, day_plans)

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
    ) -> list[dict[str, Any]]:
        dates = _trip_dates(request)
        if len(lodgings) == 1:
            hits = self._search_day_hits(
                request,
                lodgings[0],
                k=min(30, INTENSITY_K[request.intensity] * request.days),
            )
            chunks = _split_hits(hits, request.days)
            return [
                {"date": trip_date, "lodging": lodgings[0], "hits": chunk}
                for trip_date, chunk in zip(dates, chunks, strict=True)
            ]

        plans: list[dict[str, Any]] = []
        for day_index, trip_date in enumerate(dates):
            lodging = lodgings[day_index]
            hits = self._search_day_hits(
                request, lodging, k=INTENSITY_K[request.intensity]
            )
            plans.append({"date": trip_date, "lodging": lodging, "hits": hits})
        return plans

    def _search_day_hits(
        self,
        request: RecommendRequest,
        lodging: dict[str, Any],
        k: int,
    ) -> list[dict[str, Any]]:
        max_km = radius_km(request.radius)
        categories = categories_for_purpose(request.purpose)
        min_docs = max(3, INTENSITY_PICK[request.intensity] - 1)
        origin_lat = _coord(lodging, "latitude")
        origin_lng = _coord(lodging, "longitude")
        query = purpose_query(
            request.purpose,
            str(lodging.get("district") or ""),
            request.companion,
        )
        exclude_ids = {str(lodging.get("id"))}
        hits = self.retriever.search(
            query,
            k=k,
            category=categories,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            max_distance_km=max_km,
            exclude_ids=exclude_ids,
            exclude_categories=["숙박"],
            intensity=request.intensity,
        )
        if len(hits) < min_docs:
            fallback = self.retriever.search(
                query,
                k=k,
                origin_lat=origin_lat,
                origin_lng=origin_lng,
                max_distance_km=max_km,
                exclude_ids=exclude_ids,
                exclude_categories=["숙박"],
                intensity=request.intensity,
            )
            hits = _dedupe_hits(hits + fallback)[:k]
        hits.sort(key=_distance_sort_key)
        return hits

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
        pick_n = INTENSITY_PICK[request.intensity]
        lines = [
            f"## 부산 {request.days}일 여행 코스 (검색 기반 초안)",
            "",
            f"- 기간: {request.start_date.isoformat()} ~ {request.end_date.isoformat()}",
            f"- 동행: {request.companion} / 목적: {request.purpose} / 반경: 약 {radius_km(request.radius)}km / 강도: {request.intensity}/5",
            "- OPENAI_API_KEY가 없어 LLM 문장 생성 없이 숙소 반경 검색 결과를 거리순으로 나눴습니다.",
            "",
        ]
        for day_index, plan in enumerate(day_plans, start=1):
            lodging = plan["lodging"]
            lines.append(
                f"### {day_index}일차 ({_format_date(plan['date'])}) "
                f"시작: {lodging.get('name')} ({lodging.get('district')})"
            )
            picked = plan["hits"][:pick_n]
            if not picked:
                lines.append("- 반경 안에서 후보를 찾지 못했습니다.")
                lines.append("")
                continue
            for place in picked:
                distance = place.get("distance_km")
                distance_text = f", {distance}km" if distance is not None else ""
                lines.append(
                    f"- **{place.get('name')}** "
                    f"({place.get('category')}, {place.get('district')}{distance_text})"
                )
                overview = (place.get("overview") or place.get("page_content") or "").strip()
                if overview:
                    lines.append(
                        f"  - {overview[:120]}{'...' if len(overview) > 120 else ''}"
                    )
            lines.append("")
        return "\n".join(lines)


def create_course_chain(llm) -> Any:
    """검색 context가 채워진 입력을 코스 문장으로 바꾸는 LangChain LCEL 체인."""
    return RunnablePassthrough() | course_prompt() | llm | StrOutputParser()


def format_day_plans(day_plans: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for day_index, plan in enumerate(day_plans, start=1):
        lodging = plan["lodging"]
        header = (
            f"### {day_index}일차 ({_format_date(plan['date'])}) "
            f"시작: {lodging.get('name')} ({lodging.get('district')})\n"
            f"숙소 주소: {lodging.get('address') or '주소 없음'}"
        )
        if not plan["hits"]:
            blocks.append(header + "\n후보 없음")
            continue
        places = []
        for i, hit in enumerate(plan["hits"], start=1):
            distance = hit.get("distance_km")
            distance_text = f" | {distance}km" if distance is not None else ""
            overview = (hit.get("page_content") or hit.get("overview") or "").strip() or "설명 없음"
            places.append(
                f"[{i}] {hit.get('name')} | {hit.get('category')} | "
                f"{hit.get('district')} | {hit.get('region_zone')}{distance_text}\n"
                f"주소: {hit.get('address')}\n"
                f"설명: {overview[:400]}"
            )
        blocks.append(header + "\n\n" + "\n\n".join(places))
    return "\n\n".join(blocks)


def _trip_dates(request: RecommendRequest) -> list[date]:
    return [
        request.start_date + timedelta(days=offset)
        for offset in range(request.days)
    ]


def _format_date(value: date) -> str:
    return f"{value.isoformat()} {WEEKDAYS[value.weekday()]}"


def _distance_sort_key(item: dict[str, Any]) -> float:
    distance = item.get("distance_km")
    return distance if distance is not None else 999.0


def _split_hits(hits: list[dict[str, Any]], days: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(days)]
    if not hits:
        return chunks
    for index, hit in enumerate(hits):
        chunks[index % days].append(hit)
    for chunk in chunks:
        chunk.sort(key=_distance_sort_key)
    return chunks


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
