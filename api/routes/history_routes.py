"""
ForenSight AI — History Routes

GET    /history                   — List user's analysis history
GET    /history/{analysis_id}     — Full result for one analysis
DELETE /history/{analysis_id}     — Delete an analysis record
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List

from api.auth import get_current_user
from api.database import delete_analysis, get_analysis_by_id, get_user_history
from api.schemas import AnalysisOut, HistoryItem

router = APIRouter(prefix="/history", tags=["History"])


@router.get("", response_model=List[HistoryItem], summary="Get analysis history")
async def history(
    limit: int = 50,
    user: dict = Depends(get_current_user),
):
    """
    Returns the current user's analysis history, newest first.
    Excludes the heatmap field (use GET /history/{id} to get full details).
    """
    rows = await get_user_history(user["user_id"], limit=limit)
    return [
        HistoryItem(
            **{
                **r,
                "face_detected": bool(r.get("face_detected")),
            }
        )
        for r in rows
    ]


@router.get("/{analysis_id}", response_model=AnalysisOut, summary="Get full analysis result")
async def get_analysis(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Returns the full analysis result including Grad-CAM heatmap for a specific analysis.
    Only the owner can access their own analyses.
    """
    row = await get_analysis_by_id(analysis_id, user["user_id"])
    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found: {analysis_id}",
        )
    return AnalysisOut(
        **{**row, "face_detected": bool(row.get("face_detected"))}
    )


@router.delete("/{analysis_id}", summary="Delete an analysis")
async def delete_history_item(
    analysis_id: str,
    user: dict = Depends(get_current_user),
):
    """
    Deletes a specific analysis from history.
    Only the owner can delete their own analyses.
    """
    deleted = await delete_analysis(analysis_id, user["user_id"])
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found: {analysis_id}",
        )
    return {"status": "deleted", "analysis_id": analysis_id}
