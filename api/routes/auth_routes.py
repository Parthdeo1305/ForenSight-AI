"""
ForenSight AI — Auth Routes
POST /auth/login  — Verify Firebase ID token, return user profile
"""

from fastapi import APIRouter, HTTPException

from api.auth import verify_firebase_token
from api.database import get_or_create_user
from api.schemas import LoginRequest, UserOut

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=UserOut, summary="Sign in with Google / Firebase token")
async def login(body: LoginRequest):
    """
    Exchange a Firebase ID token (from Google Sign-In) for a ForenSight user profile.
    Creates the user if they're new.
    """
    decoded = await verify_firebase_token(body.id_token)

    user = await get_or_create_user(
        user_id=decoded.get("uid"),
        name=decoded.get("name", "ForenSight User"),
        email=decoded.get("email", ""),
        photo_url=decoded.get("picture"),
    )
    return UserOut(**user)
