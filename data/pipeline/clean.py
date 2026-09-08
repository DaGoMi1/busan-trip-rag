from data.pipeline.config import REGION_ZONES


def to_rag_document(record: dict) -> dict | None:
    """임베딩/검색에 넣을 단일 장소 문서를 만듭니다.

    이름·주소·좌표가 없는 항목은 RAG 품질을 떨어뜨리므로 제외합니다.
    """
    name = record.get("name") or ""
    if not name:
        return None
    if record.get("latitude") is None or record.get("longitude") is None:
        return None

    district_name = record.get("district_name") or ""
    region_zone = REGION_ZONES.get(district_name, "기타")
    overview = record.get("overview") or ""

    page_content = "\n".join(
        [
            f"[장소명] {name}",
            f"[카테고리] {record.get('category', '')}",
            f"[위치] {district_name} ({region_zone})",
            f"[주소] {record.get('address', '')}",
            f"[설명] {overview}" if overview else "[설명] 상세 설명 없음",
        ]
    )

    return {
        "id": record["content_id"],
        "name": name,
        "category": record.get("category"),
        "district": district_name,
        "region_zone": region_zone,
        "address": record.get("address"),
        "latitude": record.get("latitude"),
        "longitude": record.get("longitude"),
        "tel": record.get("tel"),
        "image_url": record.get("image_url"),
        "homepage": record.get("homepage"),
        "overview": overview,
        "page_content": page_content,
        "metadata": {
            "content_id": record.get("content_id"),
            "content_type_id": record.get("content_type_id"),
            "region_zone": region_zone,
            "district": district_name,
            "category": record.get("category"),
        },
    }
