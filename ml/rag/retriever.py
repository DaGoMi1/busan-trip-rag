from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from ml.embeddings.embedding_model import EmbeddingModel
from ml.rag.preferences import LODGING_FIELDS, apply_intensity
from ml.vectorstore.faiss_store import FaissStore


class PlaceRetriever:
    """질문 텍스트로 부산 장소를 검색합니다."""

    def __init__(self, store: FaissStore, embedder: EmbeddingModel) -> None:
        self.store = store
        self.embedder = embedder

    @classmethod
    def from_disk(cls) -> PlaceRetriever:
        store = FaissStore.load()
        embedder = (
            EmbeddingModel(store.model_name) if store.model_name else EmbeddingModel()
        )
        return cls(store=store, embedder=embedder)

    def get_by_id(self, place_id: str) -> dict[str, Any] | None:
        target = str(place_id)
        for document in self.store.documents:
            if str(document.get("id")) == target:
                return document
        return None

    def list_lodgings(self, query: str = "") -> list[dict[str, Any]]:
        needle = query.strip().lower()
        lodgings: list[dict[str, Any]] = []
        for document in self.store.documents:
            if document.get("category") != "숙박":
                continue
            if needle:
                haystack = " ".join(
                    str(document.get(field) or "")
                    for field in ("name", "district", "address")
                ).lower()
                if needle not in haystack:
                    continue
            lodgings.append({field: document.get(field) for field in LODGING_FIELDS})
        return lodgings

    def search(
        self,
        query: str,
        k: int = 5,
        category: str | list[str] | None = None,
        region_zone: str | list[str] | None = None,
        origin_lat: float | None = None,
        origin_lng: float | None = None,
        max_distance_km: float | None = None,
        exclude_ids: set[str] | list[str] | None = None,
        exclude_categories: set[str] | list[str] | None = None,
        intensity: int | None = None,
    ) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode([query], show_progress=False)
        filters = {
            "category": category,
            "region_zone": region_zone,
        }
        fetch_k = k * 4 if (exclude_ids or exclude_categories or intensity) else k
        hits = self.store.search(
            query_vec,
            k=fetch_k,
            filters=filters,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            max_distance_km=max_distance_km,
        )
        blocked_ids = {str(item) for item in (exclude_ids or [])}
        blocked_categories = set(exclude_categories or [])
        if blocked_ids:
            hits = [hit for hit in hits if str(hit.get("id")) not in blocked_ids]
        if blocked_categories:
            hits = [hit for hit in hits if hit.get("category") not in blocked_categories]
        if intensity is not None:
            hits = apply_intensity(hits, intensity)
        return hits[:k]


def hit_to_document(hit: dict[str, Any]) -> Document:
    metadata = {
        "id": hit.get("id"),
        "name": hit.get("name"),
        "category": hit.get("category"),
        "district": hit.get("district"),
        "region_zone": hit.get("region_zone"),
        "address": hit.get("address"),
        "latitude": hit.get("latitude"),
        "longitude": hit.get("longitude"),
        "distance_km": hit.get("distance_km"),
        "score": hit.get("score"),
        **(hit.get("metadata") or {}),
    }
    return Document(
        page_content=hit.get("page_content") or hit.get("overview") or "",
        metadata=metadata,
    )


class LangChainPlaceRetriever(BaseRetriever):
    """FAISS PlaceRetriever를 LangChain Retriever 인터페이스로 감쌉니다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    place_retriever: PlaceRetriever
    k: int = 8
    category: str | list[str] | None = None
    region_zone: str | list[str] | None = None
    origin_lat: float | None = None
    origin_lng: float | None = None
    max_distance_km: float | None = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        hits = self.place_retriever.search(
            query,
            k=self.k,
            category=self.category,
            region_zone=self.region_zone,
            origin_lat=self.origin_lat,
            origin_lng=self.origin_lng,
            max_distance_km=self.max_distance_km,
        )
        return [hit_to_document(hit) for hit in hits]
