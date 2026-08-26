"""Authentication via Emergent-managed Google Auth."""
import uuid
from datetime import timedelta

import httpx
from fastapi import Header, HTTPException

from db import db, utcnow, NO_ID

EMERGENT_SESSION_URL = "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data"

# Guard against processing the same one-time session_id twice.
_used_session_ids: set[str] = set()


async def exchange_session_id(session_id: str) -> dict:
    """Exchange a one-time session_id with Emergent, upsert the user, mint a session."""
    if session_id in _used_session_ids:
        raise HTTPException(status_code=401, detail="Session already used")
    _used_session_ids.add(session_id)

    async with httpx.AsyncClient(timeout=20) as http:
        resp = await http.get(EMERGENT_SESSION_URL, headers={"X-Session-ID": session_id})

    if resp.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    data = resp.json()
    email = data.get("email")
    name = data.get("name")
    picture = data.get("picture")
    session_token = data.get("session_token")

    if not email or not session_token:
        raise HTTPException(status_code=401, detail="Incomplete session data")

    existing = await db.users.find_one({"email": email}, NO_ID)
    if existing:
        user_id = existing["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": name or existing.get("name"), "picture": picture or existing.get("picture")}},
        )
        user = {**existing, "name": name or existing.get("name"), "picture": picture or existing.get("picture")}
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        user = {
            "user_id": user_id,
            "email": email,
            "name": name or email.split("@")[0],
            "picture": picture,
            "created_at": utcnow().isoformat(),
        }
        await db.users.insert_one(dict(user))
        # Seed default settings.
        await db.settings.insert_one(
            {
                "user_id": user_id,
                "theme": "dark",
                "accent": "purple",
                "interface_mode": "liquid_glass",
                "performance_mode": False,
                "memory_enabled": True,
                "language": "auto",
            }
        )

    await db.user_sessions.insert_one(
        {
            "session_token": session_token,
            "user_id": user_id,
            "created_at": utcnow(),
            "expires_at": utcnow() + timedelta(days=7),
        }
    )

    user.pop("_id", None)
    return {"session_token": session_token, "user": user}


async def get_current_user(authorization: str | None = Header(default=None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ", 1)[1].strip()

    session = await db.user_sessions.find_one({"session_token": token}, NO_ID)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    expires_at = session["expires_at"]
    if expires_at.tzinfo is None:
        from datetime import timezone as _tz

        expires_at = expires_at.replace(tzinfo=_tz.utc)
    if expires_at < utcnow():
        raise HTTPException(status_code=401, detail="Session expired")

    user = await db.users.find_one({"user_id": session["user_id"]}, NO_ID)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def revoke_session(token: str):
    await db.user_sessions.delete_one({"session_token": token})
