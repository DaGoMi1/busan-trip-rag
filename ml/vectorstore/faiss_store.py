from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from data.pipeline.config import FAISS_DIR

EARTH_RADIUS_KM = 6371.0


class FaissStore:
    """정규화된 임베딩용 FAISS Inner Product 인덱스."""

    def __init__(
        self,
        index: faiss.Index | None = None,
        documents: list[dict[str, Any]] | None = None,
        model_name: str = "",
    ) -> None:
        self.index = index
        self.documents = documents or []
        self.model_name = model_name

    @classmethod
    def build(
        cls,
        embeddings: np.ndarray,
        documents: list[dict[str, Any]],
        model_name: str,
    ) -> FaissStore:
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2D array")
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(index=index, documents=documents, model_name=model_name)

    def save(self, directory: Path | None = None) -> Path:
        if self.index is None:
            raise RuntimeError("저장할 FAISS 인덱스가 없습니다.")
        path = directory or FAISS_DIR
        path.mkdir(parents=True, exist_ok=True)
        # faiss.write_index 는 Windows 한글 경로에서 실패할 수 있어 바이트로 저장합니다.
        serialized = faiss.serialize_index(self.index)
        (path / "index.faiss").write_bytes(np.asarray(serialized).tobytes())
        (path / "documents.jsonl").write_text(
            "\n".join(json.dumps(doc, ensure_ascii=False) for doc in self.documents)
            + "\n",
            encoding="utf-8",
        )
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "count": len(self.documents),
                    "dim": int(self.index.d),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, directory: Path | None = None) -> FaissStore:
        path = directory or FAISS_DIR
        raw = np.frombuffer((path / "index.faiss").read_bytes(), dtype="uint8")
        index = faiss.deserialize_index(raw)
        documents = [
            json.loads(line)
            for line in (path / "documents.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        model_name = ""
        meta_path = path / "meta.json"
        if meta_path.exists():
            model_name = json.loads(meta_path.read_text(encoding="utf-8")).get(
                "model_name", ""
            )
        return cls(index=index, documents=documents, model_name=model_name)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filters: dict[str, Any] | None = None,
        oversample: int = 8,
        origin_lat: float | None = None,
        origin_lng: float | None = None,
        max_distance_km: float | None = None,
    ) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("로드된 FAISS 인덱스가 없습니다.")

        vector = np.asarray(query_embedding, dtype="float32")
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        has_geo = (
            origin_lat is not None
            and origin_lng is not None
            and max_distance_km is not None
        )
        has_filters = bool(
            filters and any(value not in (None, "", [], ()) for value in filters.values())
        )
        if has_geo:
            fetch_k = len(self.documents)
        elif has_filters:
            fetch_k = min(len(self.documents), max(k * oversample, k))
        else:
            fetch_k = min(len(self.documents), k)

        scores, indices = self.index.search(vector, fetch_k)

        hits: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:
                continue
            document = self.documents[int(idx)]
            metadata = document.get("metadata") or {}
            if filters and not _match_filters(metadata, document, filters):
                continue
            hit: dict[str, Any] = {"score": float(score), **document}
            if has_geo:
                distance = _document_distance_km(document, origin_lat, origin_lng)
                if distance is None or distance > max_distance_km:
                    continue
                hit["distance_km"] = round(distance, 2)
            hits.append(hit)
            if len(hits) >= k:
                break
        return hits


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _document_distance_km(
    document: dict[str, Any],
    origin_lat: float,
    origin_lng: float,
) -> float | None:
    try:
        lat = float(document.get("latitude"))
        lng = float(document.get("longitude"))
    except (TypeError, ValueError):
        return None
    return haversine_km(origin_lat, origin_lng, lat, lng)


def _match_filters(
    metadata: dict[str, Any],
    document: dict[str, Any],
    filters: dict[str, Any],
) -> bool:
    for key, expected in filters.items():
        if expected in (None, "", [], ()):
            continue
        actual = metadata.get(key, document.get(key))
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True
