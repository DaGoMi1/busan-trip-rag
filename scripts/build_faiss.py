"""busan_places.jsonl 을 임베딩해 FAISS 인덱스를 만듭니다.

실행 예:
  python -m scripts.build_faiss
  python -m scripts.build_faiss --exclude-categories 쇼핑
  python -m scripts.build_faiss --query "해운대 일몰 데이트"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.pipeline.config import (
    DEFAULT_EMBEDDING_MODEL,
    FAISS_DIR,
    PROCESSED_DIR,
)
from ml.embeddings.embedding_model import EmbeddingModel
from ml.rag.retriever import PlaceRetriever
from ml.vectorstore.faiss_store import FaissStore

JSONL_PATH = PROCESSED_DIR / "busan_places.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="부산 장소 FAISS 인덱스 구축")
    parser.add_argument(
        "--input",
        default=str(JSONL_PATH),
        help="RAG 문서 JSONL 경로",
    )
    parser.add_argument(
        "--exclude-categories",
        nargs="*",
        default=[],
        help="인덱스에서 제외할 카테고리 (예: 쇼핑)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformers 모델 이름",
    )
    parser.add_argument(
        "--query",
        default="",
        help="구축 후 검색 스모크 테스트 쿼리",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="인덱스를 다시 만들지 않고 --query 만 실행합니다.",
    )
    return parser.parse_args()


def load_documents(path: Path, exclude_categories: list[str]) -> list[dict]:
    documents: list[dict] = []
    excluded = set(exclude_categories)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("category") in excluded:
            continue
        page_content = (row.get("page_content") or "").strip()
        if not page_content:
            continue
        documents.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "category": row.get("category"),
                "district": row.get("district"),
                "region_zone": row.get("region_zone"),
                "address": row.get("address"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "overview": row.get("overview"),
                "page_content": page_content,
                "metadata": row.get("metadata") or {},
            }
        )
    return documents


def build_index(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path}")

    documents = load_documents(input_path, args.exclude_categories)
    if not documents:
        raise SystemExit("임베딩할 문서가 없습니다.")

    print(f"[build] documents : {len(documents)}")
    print(f"[build] model     : {args.model}")
    embedder = EmbeddingModel(args.model)
    embeddings = embedder.encode([doc["page_content"] for doc in documents])
    store = FaissStore.build(embeddings, documents, args.model)
    saved = store.save()
    print(f"[build] saved     : {saved}")


def run_query(query: str) -> None:
    retriever = PlaceRetriever.from_disk()
    hits = retriever.search(query, k=5)
    print(f"\n[query] {query}")
    if not hits:
        print("  (결과 없음)")
        return
    for i, hit in enumerate(hits, start=1):
        print(
            f"  {i}. {hit.get('name')} "
            f"({hit.get('category')}, {hit.get('district')}) "
            f"score={hit['score']:.3f}"
        )


def main() -> None:
    args = parse_args()
    if not args.skip_build:
        build_index(args)
    elif not FAISS_DIR.exists():
        raise SystemExit("인덱스가 없습니다. 먼저 python -m scripts.build_faiss 를 실행하세요.")

    if args.query:
        run_query(args.query)
    elif not args.skip_build:
        run_query("해운대 일몰 데이트")


if __name__ == "__main__":
    main()
