from fastapi import APIRouter, Query

from backend.schemas.request import RecommendRequest
from backend.services.recommend_service import get_recommendation, list_lodgings

router = APIRouter()


@router.get("/lodgings")
async def lodgings(q: str = Query("", description="숙소 이름·구·주소 검색어")):
    return {"lodgings": list_lodgings(q)}


@router.post("/recommend")
async def recommend(request: RecommendRequest):
    recommendation = await get_recommendation(request)
    return {"recommendation": recommendation}
