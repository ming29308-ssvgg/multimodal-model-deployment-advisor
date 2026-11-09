import asyncio

from fastapi import APIRouter

from backend.agents.model_selector.graph import run_recommendation
from backend.schemas import RecommendationRequest, RecommendationResponse


router = APIRouter(prefix="/api/v1/model-selector", tags=["model-selector"])


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_models_for_task(
    request: RecommendationRequest,
) -> RecommendationResponse:
    result = await asyncio.to_thread(
        run_recommendation,
        request.task,
        request.preferences.model_dump(),
        (
            request.hardware_override.model_dump(exclude_none=True)
            if request.hardware_override
            else None
        ),
    )
    return RecommendationResponse.model_validate(result)

