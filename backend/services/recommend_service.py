from fastapi import HTTPException

from backend.schemas.request import RecommendRequest
from ml.rag.chain import CourseRag, MissingLodgingError

_rag: CourseRag | None = None


def init_rag() -> CourseRag:
    global _rag
    _rag = CourseRag.from_disk()
    return _rag


def get_rag() -> CourseRag:
    if _rag is None:
        return init_rag()
    return _rag


def list_lodgings(query: str = "") -> list[dict]:
    return get_rag().retriever.list_lodgings(query)


async def get_recommendation(request: RecommendRequest) -> str:
    try:
        return get_rag().recommend(request)
    except MissingLodgingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
