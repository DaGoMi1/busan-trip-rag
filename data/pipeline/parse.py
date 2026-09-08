import re
from typing import Any

from data.pipeline.config import CONTENT_TYPES, REGION_ZONES


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def parse_list_item(raw: dict[str, Any]) -> dict[str, Any]:
    """areaBasedList2 항목을 내부 스키마로 정규화합니다."""
    content_type_id = _text(raw.get("contenttypeid"))
    address = " ".join(
        part for part in [_text(raw.get("addr1")), _text(raw.get("addr2"))] if part
    )
    district_code = _text(
        raw.get("ldongsigngucd") or raw.get("lDongSignguCd") or raw.get("sigungucode")
    )
    return {
        "content_id": _text(raw.get("contentid")),
        "content_type_id": content_type_id,
        "category": CONTENT_TYPES.get(content_type_id, "기타"),
        "name": _text(raw.get("title")),
        "address": address,
        "district": district_code,
        "district_name": "",
        "latitude": _to_float(raw.get("mapy")),
        "longitude": _to_float(raw.get("mapx")),
        "image_url": _text(raw.get("firstimage")),
        "tel": _text(raw.get("tel")),
        "cat1": _text(raw.get("cat1")),
        "cat2": _text(raw.get("cat2")),
        "cat3": _text(raw.get("cat3")),
        "modified_time": _text(raw.get("modifiedtime")),
    }


def parse_detail_common(raw: dict[str, Any]) -> dict[str, Any]:
    """detailCommon2 응답에서 RAG에 쓸 설명/홈페이지만 추출합니다."""
    return {
        "overview": _strip_html(_text(raw.get("overview"))),
        "homepage": _strip_html(_text(raw.get("homepage"))),
    }


def merge_district_names(
    records: list[dict[str, Any]],
    district_map: dict[str, str],
) -> list[dict[str, Any]]:
    for record in records:
        code = record.get("district", "")
        name = district_map.get(code, "")
        if not name:
            name = _district_from_address(record.get("address", ""))
        record["district_name"] = name
    return records


def merge_sigungu_names(
    records: list[dict[str, Any]],
    sigungu_map: dict[str, str],
) -> list[dict[str, Any]]:
    """하위 호환용 별칭."""
    return merge_district_names(records, sigungu_map)


def _district_from_address(address: str) -> str:
    match = re.search(r"부산광역시\s*([가-힣]+구|[가-힣]+군)", address)
    if not match:
        return ""
    name = match.group(1)
    return name if name in REGION_ZONES else name


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _strip_html(text: str) -> str:
    cleaned = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    return re.sub(r"\s+\n", "\n", cleaned).strip()
