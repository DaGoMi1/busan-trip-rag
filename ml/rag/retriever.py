from __future__ import annotations

from typing import Any

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
        origin_lat: float | None = None,                        
        origin_lng: float | None = None,
        max_distance_km: float | None = None,
        exclude_ids: set[str] | list[str] | None = None,
        exclude_categories: set[str] | list[str] | None = None,
        intensity: int | None = None,
    ) -> list[dict[str, Any]]:
        query_vec = self.embedder.encode([query], show_progress=False)
        filters = {"category": category}
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
