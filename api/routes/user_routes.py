"""
ForenSight AI — User Routes
GET /user/me  — Return current user's profile
"""

from fastapi import APIRouter, Depends
from api.auth import get_current_user
from api.schemas import UserOut

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/me", response_model=UserOut, summary="Get current user profile")
async def get_me(user: dict = Depends(get_current_user)):
    """Returns the profile of the currently authenticated user."""
    return UserOut(**user)
