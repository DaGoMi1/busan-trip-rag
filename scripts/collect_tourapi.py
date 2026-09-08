"""TourAPI 부산 데이터 수집 → 파싱 → RAG 문서 정제.

사용 전:
  1. 공공데이터포털에서 '한국관광공사_국문 관광정보 서비스_GW' 활용 신청
  2. .env 에 TOUR_API_KEY 설정 (Encoding/Decoding 키 모두 가능)

실행 예:
  python -m scripts.collect_tourapi
  python -m scripts.collect_tourapi --skip-details --types 12 39
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline.clean import to_rag_document
from data.pipeline.client import TourAPIClient
from data.pipeline.config import CONTENT_TYPES, PROCESSED_DIR
from data.pipeline.fetch import collect_raw
from data.pipeline.parse import merge_district_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="부산 TourAPI 수집/정제")
    parser.add_argument(
        "--skip-details",
        action="store_true",
        help="detailCommon2 호출을 생략하고 목록 메타데이터만 수집합니다.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=list(CONTENT_TYPES.keys()),
        help="수집할 contentTypeId (기본: 12 14 15 28 32 38 39)",
    )
    parser.add_argument(
        "--refresh-list",
        action="store_true",
        help="저장된 목록 JSON을 무시하고 areaBasedList2를 다시 호출합니다.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    api_key = os.getenv("TOUR_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "TOUR_API_KEY 가 없습니다. 공공데이터포털에서 발급 후 .env 에 넣어 주세요."
        )

    client = TourAPIClient(api_key, sleep_sec=1.5)
    snapshot = collect_raw(
        client,
        fetch_details=not args.skip_details,
        content_type_ids=args.types,
        refresh_list=args.refresh_list,
    )

    records = merge_district_names(snapshot["records"], snapshot["district_map"])
    documents = [doc for record in records if (doc := to_rag_document(record))]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "busan_places.jsonl"
    with out_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps(document, ensure_ascii=False) + "\n")

    print(f"raw records : {len(records)}")
    print(f"rag docs    : {len(documents)}")
    print(f"saved       : {out_path}")
    if snapshot.get("rate_limited"):
        print("상세는 일부만 채워졌습니다. 잠시 후 다시 실행하세요.")


if __name__ == "__main__":
    main()
