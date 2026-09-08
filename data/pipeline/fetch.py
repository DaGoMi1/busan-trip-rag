import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from data.pipeline.client import DailyQuotaError, RateLimitError, TourAPIClient, normalize_items
from data.pipeline.config import (
    BUSAN_LDONG_REGN_CD,
    CONTENT_TYPES,
    DETAIL_CACHE_PATH,
    RAW_DIR,
)
from data.pipeline.parse import parse_detail_common, parse_list_item


def _list_path(content_type_id: str) -> Path:
    return RAW_DIR / f"area_based_list_{content_type_id}.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_district_map(client: TourAPIClient, refresh: bool = False) -> dict[str, str]:
    """부산 법정동 시군구 코드 → 구/군 이름."""
    cached = RAW_DIR / "busan_districts.json"
    if cached.exists() and not refresh:
        print("[collect] 구/군 코드 캐시 사용")
        return load_json(cached)

    items = client.fetch_all_pages(
        "ldongCode2",
        lDongRegnCd=BUSAN_LDONG_REGN_CD,
    )
    mapping: dict[str, str] = {}
    for item in items:
        code = str(
            item.get("code")
            or item.get("lDongSignguCd")
            or item.get("ldongsigngucd")
            or ""
        ).strip()
        name = str(
            item.get("name")
            or item.get("lDongSignguNm")
            or item.get("ldongsigngunm")
            or ""
        ).strip()
        if code and name:
            mapping[code] = name
    return mapping


def fetch_busan_list(
    client: TourAPIClient,
    content_type_id: str,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    path = _list_path(content_type_id)
    if path.exists() and not refresh:
        items = load_json(path)
        print(f"[collect] type {content_type_id} 목록 캐시 {len(items)}건")
        return items

    items = client.fetch_all_pages(
        "areaBasedList2",
        lDongRegnCd=BUSAN_LDONG_REGN_CD,
        contentTypeId=content_type_id,
        arrange="A",
    )
    save_raw(items, path.name)
    print(f"[collect] type {content_type_id} 목록 {len(items)}건")
    return items


def load_detail_cache() -> dict[str, Any]:
    if not DETAIL_CACHE_PATH.exists():
        return {}
    return json.loads(DETAIL_CACHE_PATH.read_text(encoding="utf-8"))


def save_detail_cache(cache: dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DETAIL_CACHE_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(cache, ensure_ascii=False),
        encoding="utf-8",
    )
    tmp_path.replace(DETAIL_CACHE_PATH)


def fetch_detail(
    client: TourAPIClient,
    content_id: str,
    cache: dict[str, Any],
) -> dict[str, Any]:
    if content_id in cache:
        return cache[content_id]

    body = client.get("detailCommon2", contentId=content_id)
    items = normalize_items(body.get("items", {}))
    detail = items[0] if items else {}
    cache[content_id] = detail
    return detail


def save_raw(payload: Any, filename: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / filename
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def collect_raw(
    client: TourAPIClient,
    fetch_details: bool = True,
    content_type_ids: list[str] | None = None,
    refresh_list: bool = False,
) -> dict[str, Any]:
    """부산 목록(+선택적 상세)을 수집하고 data/raw 에 저장합니다."""
    target_types = content_type_ids or list(CONTENT_TYPES.keys())
    collected_at = datetime.now(timezone.utc).isoformat()

    district_map = fetch_district_map(client, refresh=refresh_list)
    save_raw(district_map, "busan_districts.json")

    detail_cache = load_detail_cache() if fetch_details else {}
    fetched = 0
    cached = 0
    rate_limited = False

    parsed_records: list[dict[str, Any]] = []
    for type_id in target_types:
        raw_items = fetch_busan_list(client, type_id, refresh=refresh_list)

        for item in raw_items:
            record = parse_list_item(item)
            if not record["content_id"]:
                continue
            if fetch_details and not rate_limited:
                was_cached = record["content_id"] in detail_cache
                try:
                    detail = fetch_detail(client, record["content_id"], detail_cache)
                except DailyQuotaError:
                    save_detail_cache(detail_cache)
                    rate_limited = True
                    print(
                        "[collect] 일일 호출 한도를 초과했습니다. "
                        "내일(자정 이후) 같은 명령을 다시 실행하면 상세를 이어서 받습니다."
                    )
                    parsed_records.append(record)
                    continue
                except RateLimitError:
                    save_detail_cache(detail_cache)
                    rate_limited = True
                    print(
                        "[collect] 속도 제한으로 상세 수집을 중단했습니다. "
                        "10~30분 뒤 같은 명령을 다시 실행하면 이어서 받습니다."
                    )
                    parsed_records.append(record)
                    continue
                record.update(parse_detail_common(detail))
                if was_cached:
                    cached += 1
                else:
                    fetched += 1
                    if fetched % 10 == 0:
                        save_detail_cache(detail_cache)
                        print(f"[collect] 상세 신규 {fetched}건, 캐시 {cached}건")
            elif fetch_details and record["content_id"] in detail_cache:
                record.update(parse_detail_common(detail_cache[record["content_id"]]))
                cached += 1
            parsed_records.append(record)

    if fetch_details:
        save_detail_cache(detail_cache)
        print(f"[collect] 상세 완료 신규 {fetched}건, 캐시 {cached}건")

    snapshot = {
        "collected_at": collected_at,
        "lDongRegnCd": BUSAN_LDONG_REGN_CD,
        "district_map": district_map,
        "records": parsed_records,
        "rate_limited": rate_limited,
    }
    save_raw(snapshot, "busan_places_raw.json")
    return snapshot
