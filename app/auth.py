"""Session-based authentication helpers."""
import os
from datetime import timedelta
from typing import Optional

import bcrypt
from fastapi import Cookie, Depends, HTTPException, status
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from . import models
from .database import SessionLocal

SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-production-please")
SESSION_MAX_AGE = int(os.environ.get("SESSION_MAX_AGE", 60 * 60 * 24 * 7))  # 7 days
COOKIE_NAME = "tl_session"

_signer = URLSafeTimedSerializer(SECRET_KEY)


# ── Password helpers ──────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── Session cookie helpers ────────────────────────────────────────────────

def create_session_cookie(user_id: int) -> str:
    return _signer.dumps(user_id, salt="session")


def decode_session_cookie(token: str) -> Optional[int]:
    try:
        return _signer.loads(token, salt="session", max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None


# ── FastAPI dependency ────────────────────────────────────────────────────

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    tl_session: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> models.User:
    if tl_session:
        user_id = decode_session_cookie(tl_session)
        if user_id:
            user = db.query(models.User).get(user_id)
            if user:
                return user
    raise HTTPException(
        status_code=status.HTTP_303_SEE_OTHER,
        headers={"Location": "/login"},
    )


def get_current_user_optional(
    tl_session: Optional[str] = Cookie(default=None),
    db: Session = Depends(get_db),
) -> Optional[models.User]:
    """Returns user or None (for login page redirect)."""
    if tl_session:
        user_id = decode_session_cookie(tl_session)
        if user_id:
            return db.query(models.User).get(user_id)
    return None
