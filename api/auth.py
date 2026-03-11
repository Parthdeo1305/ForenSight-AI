"""
ForenSight AI — Authentication Layer
Firebase JWT verification + FastAPI dependency injection.

On startup Firebase Admin SDK is initialized once from either:
  1. FIREBASE_SERVICE_ACCOUNT_PATH (path to service account JSON)
  2. FIREBASE_SERVICE_ACCOUNT_JSON  (raw JSON string, useful for cloud env)

For local development without Firebase, set DISABLE_AUTH=true to bypass
verification (returns a mock user).

Author: ForenSight AI Team
"""

import json
import os
import warnings
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# ─── Firebase Admin Init ──────────────────────────────────────────────────────

_firebase_initialized = False

def _init_firebase():
    global _firebase_initialized
    if _firebase_initialized:
        return

    if os.environ.get("DISABLE_AUTH", "false").lower() == "true":
        print("[Auth] DISABLE_AUTH=true — running in dev mode, all requests authenticated.")
        _firebase_initialized = True
        return

    try:
        import firebase_admin
        from firebase_admin import credentials

        sa_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "firebase-service-account.json")
        sa_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")

        if sa_json:
            print("[Auth] Loading Firebase credentials from FIREBASE_SERVICE_ACCOUNT_JSON env var.")
            cred = credentials.Certificate(json.loads(sa_json))
        elif os.path.exists(sa_path):
            print(f"[Auth] Loading Firebase credentials from file: {sa_path}")
            cred = credentials.Certificate(sa_path)
        else:
            warnings.warn(
                f"[Auth] No Firebase credentials found. "
                f"Set FIREBASE_SERVICE_ACCOUNT_JSON env var or place a key file at '{sa_path}'. "
                "Auth endpoints will return 401. Set DISABLE_AUTH=true for dev mode."
            )
            _firebase_initialized = True  # Allow app to start; auth endpoints will return 401
            return

        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)

        _firebase_initialized = True
        print("[Auth] Firebase Admin SDK initialized successfully.")

    except ImportError:
        warnings.warn("[Auth] firebase-admin not installed — auth disabled.")
        _firebase_initialized = True
    except Exception as e:
        warnings.warn(
            f"[Auth] Firebase init failed: {e}. "
            "App will start but auth endpoints will return 500. "
            "Check your FIREBASE_SERVICE_ACCOUNT_JSON env var."
        )
        _firebase_initialized = True  # Don't crash the app — let it start


_init_firebase()

# ─── Token Verification ───────────────────────────────────────────────────────

security = HTTPBearer(auto_error=False)

DISABLE_AUTH = os.environ.get("DISABLE_AUTH", "false").lower() == "true"

# Dev mock user for when auth is disabled
MOCK_USER = {
    "user_id": "dev-user-001",
    "name": "Dev User",
    "email": "dev@forensight.local",
    "photo_url": None,
}


async def verify_firebase_token(token: str) -> dict:
    """
    Verify a Firebase ID token and return the decoded claims.
    Raises HTTPException 401 on invalid/expired token.
    """
    if DISABLE_AUTH:
        return {
            "uid": MOCK_USER["user_id"],
            "name": MOCK_USER["name"],
            "email": MOCK_USER["email"],
            "picture": None,
        }

    try:
        from firebase_admin import auth as firebase_auth
        decoded = firebase_auth.verify_id_token(token, check_revoked=False)
        return decoded
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """
    FastAPI dependency: extract Bearer token from Authorization header,
    verify with Firebase, upsert user in DB, return user dict.

    Usage:
        @app.get("/protected")
        async def route(user = Depends(get_current_user)):
            ...
    """
    if DISABLE_AUTH:
        return MOCK_USER

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    decoded = await verify_firebase_token(token)

    # Upsert user in database
    from api.database import get_or_create_user
    user = await get_or_create_user(
        user_id=decoded.get("uid"),
        name=decoded.get("name", "ForenSight User"),
        email=decoded.get("email", ""),
        photo_url=decoded.get("picture"),
    )
    return user
