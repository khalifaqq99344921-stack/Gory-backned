"""Gory backend API — auth, chat, vision, image generation, subscription,
feedback/reports, settings, usage and feature flags."""
import logging
import os
import uuid
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Header, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from db import db, utcnow, iso_now, NO_ID, ensure_indexes
from auth import exchange_session_id, get_current_user, revoke_session
import ai

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("gory")

GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR.mkdir(exist_ok=True)
VOICE_DIR = Path(__file__).parent / "voice_audio"
VOICE_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Gory API")
api = APIRouter(prefix="/api")

# --- Config: feature flags + plan limits -----------------------------------
FEATURE_FLAGS = {
    "ENABLE_VISION": True,
    "ENABLE_IMAGE_GENERATION": True,
    "ENABLE_IMAGE_EDITING": True,
    "ENABLE_LIVE_VOICE": True,
    "ENABLE_CAMERA_VISION": True,
    "ENABLE_GOLD_MODE": True,
    "ENABLE_PREMIUM": True,
    "ENABLE_BETA_FEATURES": False,
    "ENABLE_ADVANCED_ANIMATIONS": True,
    "ENABLE_IOS_STYLE_DOCK": True,
    "ENABLE_LIQUID_GLASS": True,
}

PLAN_LIMITS = {
    "free": {"chat": 30, "vision": 5, "image_gen": 3, "voice": 10, "voice_minutes": 0},
    "pro": {"chat": 2000, "vision": 300, "image_gen": 100, "voice": 500, "voice_minutes": 120},
}

IMAGE_COST = {"standard": 0.02, "high": 0.05}
PRICE_LABEL = "3 OMR/month"


# ---------------------------------------------------------------------------
# Subscription helpers
# ---------------------------------------------------------------------------
async def get_subscription(user_id: str) -> dict:
    sub = await db.subscriptions.find_one({"user_id": user_id}, NO_ID)
    if not sub:
        return {"user_id": user_id, "status": "free", "plan": "free"}
    # Expiry check for trial/active.
    end = sub.get("current_period_end")
    if sub.get("status") in ("trial", "active") and end:
        from datetime import datetime, timezone

        try:
            end_dt = datetime.fromisoformat(end)
            if end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            if end_dt < utcnow():
                await db.subscriptions.update_one(
                    {"user_id": user_id}, {"$set": {"status": "expired"}}
                )
                sub["status"] = "expired"
        except ValueError:
            pass
    return sub


def is_pro(sub: dict) -> bool:
    return sub.get("status") in ("trial", "active", "grace_period")


def plan_key(sub: dict) -> str:
    return "pro" if is_pro(sub) else "free"


# ---------------------------------------------------------------------------
# Usage helpers
# ---------------------------------------------------------------------------
async def check_and_increment_usage(user_id: str, kind: str, plan: str) -> None:
    date = utcnow().strftime("%Y-%m-%d")
    limit = PLAN_LIMITS[plan][kind]
    doc = await db.usage_daily.find_one({"user_id": user_id, "date": date}, NO_ID)
    current = (doc or {}).get(kind, 0)
    if current >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"You've reached your daily {kind} limit. Upgrade to Gory Pro for more.",
        )
    await db.usage_daily.update_one(
        {"user_id": user_id, "date": date},
        {"$inc": {kind: 1}, "$setOnInsert": {"user_id": user_id, "date": date}},
        upsert=True,
    )


async def get_usage_today(user_id: str) -> dict:
    date = utcnow().strftime("%Y-%m-%d")
    doc = await db.usage_daily.find_one({"user_id": user_id, "date": date}, NO_ID) or {}
    return {k: doc.get(k, 0) for k in ("chat", "vision", "image_gen", "voice", "voice_minutes")}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class SessionIn(BaseModel):
    session_id: str


class ChatIn(BaseModel):
    conversation_id: str | None = None
    text: str
    images: list[str] = Field(default_factory=list)  # base64, no data-uri prefix


class ImageGenIn(BaseModel):
    prompt: str
    aspect_ratio: str = "1:1"
    quality: str = "standard"
    style: str = "automatic"
    reference_image: str | None = None


class SettingsIn(BaseModel):
    theme: str | None = None
    accent: str | None = None
    interface_mode: str | None = None
    performance_mode: bool | None = None
    memory_enabled: bool | None = None
    language: str | None = None


class FeedbackIn(BaseModel):
    type: str  # bug | suggestion | complaint | feature_request
    message: str
    screenshot: str | None = None


class ReportIn(BaseModel):
    target_type: str  # message | image
    target_id: str
    category: str  # offensive | unsafe | incorrect | spam | privacy | other
    note: str | None = None


class ConversationPatch(BaseModel):
    title: str | None = None
    pinned: bool | None = None
    archived: bool | None = None


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@api.get("/")
async def root():
    return {"app": "Gory", "status": "ok"}


@api.post("/auth/session")
async def auth_session(body: SessionIn):
    return await exchange_session_id(body.session_id)


@api.get("/auth/me")
async def auth_me(user: dict = Depends(get_current_user)):
    sub = await get_subscription(user["user_id"])
    return {"user": user, "subscription": sub, "is_pro": is_pro(sub)}


@api.post("/auth/logout")
async def auth_logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.startswith("Bearer "):
        await revoke_session(authorization.split(" ", 1)[1].strip())
    return {"ok": True}


@api.delete("/auth/account")
async def delete_account(user: dict = Depends(get_current_user)):
    uid = user["user_id"]
    for coll in (db.conversations, db.messages, db.creations, db.settings,
                 db.subscriptions, db.usage_daily, db.user_sessions, db.feedback, db.reports):
        await coll.delete_many({"user_id": uid})
    await db.users.delete_one({"user_id": uid})
    return {"ok": True}


# ---------------------------------------------------------------------------
# Config / feature flags
# ---------------------------------------------------------------------------
@api.get("/config")
async def get_config():
    return {
        "feature_flags": FEATURE_FLAGS,
        "plan_limits": PLAN_LIMITS,
        "price_label": PRICE_LABEL,
        "trial_days": 7,
    }


# ---------------------------------------------------------------------------
# Chat routes
# ---------------------------------------------------------------------------
@api.get("/chat/conversations")
async def list_conversations(user: dict = Depends(get_current_user)):
    convos = await db.conversations.find({"user_id": user["user_id"]}, NO_ID).sort("updated_at", -1).to_list(200)
    return {"conversations": convos}


@api.get("/chat/conversations/{cid}/messages")
async def get_messages(cid: str, user: dict = Depends(get_current_user)):
    convo = await db.conversations.find_one({"id": cid, "user_id": user["user_id"]}, NO_ID)
    if not convo:
        raise HTTPException(status_code=404, detail="Conversation not found")
    msgs = await db.messages.find({"conversation_id": cid}, NO_ID).sort("created_at", 1).to_list(1000)
    return {"conversation": convo, "messages": msgs}


@api.patch("/chat/conversations/{cid}")
async def patch_conversation(cid: str, body: ConversationPatch, user: dict = Depends(get_current_user)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    if not update:
        return {"ok": True}
    res = await db.conversations.update_one({"id": cid, "user_id": user["user_id"]}, {"$set": update})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@api.delete("/chat/conversations/{cid}")
async def delete_conversation(cid: str, user: dict = Depends(get_current_user)):
    await db.conversations.delete_one({"id": cid, "user_id": user["user_id"]})
    await db.messages.delete_many({"conversation_id": cid})
    return {"ok": True}


@api.post("/chat/message")
async def chat_message(body: ChatIn, user: dict = Depends(get_current_user)):
    uid = user["user_id"]
    text = (body.text or "").strip()
    if not text and not body.images:
        raise HTTPException(status_code=400, detail="Empty message")

    allowed, reason = ai.moderate_input(text)
    if not allowed:
        raise HTTPException(status_code=400, detail=reason)

    sub = await get_subscription(uid)
    plan = plan_key(sub)

    has_images = bool(body.images)
    if has_images:
        if not FEATURE_FLAGS["ENABLE_VISION"]:
            raise HTTPException(status_code=400, detail="Vision is currently unavailable.")
        await check_and_increment_usage(uid, "vision", plan)
    await check_and_increment_usage(uid, "chat", plan)

    # Resolve / create conversation.
    cid = body.conversation_id
    if cid:
        convo = await db.conversations.find_one({"id": cid, "user_id": uid}, NO_ID)
        if not convo:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        cid = f"conv_{uuid.uuid4().hex[:12]}"
        title = (text[:40] + "…") if len(text) > 40 else (text or "Image chat")
        convo = {
            "id": cid, "user_id": uid, "title": title,
            "pinned": False, "archived": False,
            "created_at": iso_now(), "updated_at": iso_now(),
        }
        await db.conversations.insert_one(dict(convo))

    settings = await db.settings.find_one({"user_id": uid}, NO_ID) or {}
    memory_enabled = settings.get("memory_enabled", True)

    history = await db.messages.find({"conversation_id": cid}, NO_ID).sort("created_at", 1).to_list(1000)

    user_msg = {
        "id": f"msg_{uuid.uuid4().hex[:12]}", "conversation_id": cid, "role": "user",
        "content": text, "has_images": has_images, "image_count": len(body.images),
        "created_at": iso_now(),
    }
    await db.messages.insert_one(dict(user_msg))

    try:
        reply = await ai.generate_chat_reply(
            session_id=cid, prompt=text or "Describe the attached image(s).",
            history=history, image_base64_list=body.images or None,
            memory_enabled=memory_enabled,
        )
    except Exception as e:  # noqa
        logger.exception("chat generation failed")
        raise HTTPException(status_code=502, detail="Gory is having trouble connecting right now. Try again.")

    ai_msg = {
        "id": f"msg_{uuid.uuid4().hex[:12]}", "conversation_id": cid, "role": "assistant",
        "content": reply, "has_images": False, "image_count": 0, "created_at": iso_now(),
    }
    await db.messages.insert_one(dict(ai_msg))
    await db.conversations.update_one({"id": cid}, {"$set": {"updated_at": iso_now()}})

    return {"conversation_id": cid, "message": ai_msg}


# ---------------------------------------------------------------------------
# Image generation routes
# ---------------------------------------------------------------------------
@api.post("/images/generate")
async def images_generate(body: ImageGenIn, user: dict = Depends(get_current_user)):
    uid = user["user_id"]
    if not FEATURE_FLAGS["ENABLE_IMAGE_GENERATION"]:
        raise HTTPException(status_code=400, detail="Image generation is coming soon.")

    prompt = (body.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Please describe what to create.")

    allowed, reason = ai.moderate_input(prompt)
    if not allowed:
        raise HTTPException(status_code=400, detail=reason)

    sub = await get_subscription(uid)
    plan = plan_key(sub)
    await check_and_increment_usage(uid, "image_gen", plan)

    quality = body.quality if plan == "pro" else "standard"

    try:
        img_bytes = await ai.generate_image(
            session_id=f"imggen_{uuid.uuid4().hex[:8]}", prompt=prompt,
            aspect_ratio=body.aspect_ratio, quality=quality, style=body.style,
            reference_base64=body.reference_image,
        )
    except Exception:
        logger.exception("image generation failed")
        raise HTTPException(status_code=502, detail="Gory couldn't create that image. Try again.")

    if not img_bytes:
        raise HTTPException(status_code=502, detail="No image was generated. Try a different prompt.")

    creation_id = f"img_{uuid.uuid4().hex[:12]}"
    path = GENERATED_DIR / f"{creation_id}.png"
    path.write_bytes(img_bytes)

    cost = IMAGE_COST.get(quality, 0.02)
    creation = {
        "id": creation_id, "user_id": uid, "prompt": prompt,
        "aspect_ratio": body.aspect_ratio, "quality": quality, "style": body.style,
        "url": f"/api/images/file/{creation_id}",
        "is_edit": bool(body.reference_image),
        "cost_estimate": cost, "created_at": iso_now(),
    }
    await db.creations.insert_one(dict(creation))
    await db.usage_daily.update_one(
        {"user_id": uid, "date": utcnow().strftime("%Y-%m-%d")},
        {"$inc": {"image_generation_cost_estimate": cost}}, upsert=True,
    )
    return {"creation": creation}


@api.get("/images/creations")
async def list_creations(user: dict = Depends(get_current_user), skip: int = 0, limit: int = 30):
    items = (
        await db.creations.find({"user_id": user["user_id"]}, NO_ID)
        .sort("created_at", -1).skip(skip).limit(min(limit, 60)).to_list(60)
    )
    total = await db.creations.count_documents({"user_id": user["user_id"]})
    return {"creations": items, "total": total}


@api.delete("/images/creations/{cid}")
async def delete_creation(cid: str, user: dict = Depends(get_current_user)):
    await db.creations.delete_one({"id": cid, "user_id": user["user_id"]})
    p = GENERATED_DIR / f"{cid}.png"
    if p.exists():
        p.unlink()
    return {"ok": True}


@api.get("/images/file/{cid}")
async def serve_image(cid: str):
    p = GENERATED_DIR / f"{cid}.png"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(p), media_type="image/png")


# ---------------------------------------------------------------------------
# Live Voice routes
# ---------------------------------------------------------------------------
_VOICE_EXTS = {".m4a", ".mp3", ".wav", ".webm", ".mp4", ".mpeg", ".mpga"}


@api.post("/voice/turn")
async def voice_turn(
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    voice: str = Form(default="alloy"),
    user: dict = Depends(get_current_user),
):
    if not FEATURE_FLAGS["ENABLE_LIVE_VOICE"]:
        raise HTTPException(status_code=400, detail="Live Voice is coming soon.")
    uid = user["user_id"]
    sub = await get_subscription(uid)
    plan = plan_key(sub)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty audio.")

    ext = Path(file.filename or "audio.m4a").suffix.lower()
    if ext not in _VOICE_EXTS:
        ext = ".m4a"
    tmp = VOICE_DIR / f"in_{uuid.uuid4().hex[:12]}{ext}"
    tmp.write_bytes(data)
    try:
        transcript = await ai.transcribe_audio(str(tmp))
    except Exception:
        logger.exception("transcription failed")
        raise HTTPException(status_code=502, detail="Gory couldn't hear that. Try again.")
    finally:
        tmp.unlink(missing_ok=True)

    if not transcript:
        raise HTTPException(status_code=422, detail="Didn't catch that — try speaking again.")

    allowed, reason = ai.moderate_input(transcript)
    if not allowed:
        raise HTTPException(status_code=400, detail=reason)

    await check_and_increment_usage(uid, "voice", plan)

    cid = conversation_id
    if cid:
        convo = await db.conversations.find_one({"id": cid, "user_id": uid}, NO_ID)
        if not convo:
            cid = None
    if not cid:
        cid = f"conv_{uuid.uuid4().hex[:12]}"
        title = (transcript[:40] + "…") if len(transcript) > 40 else transcript
        await db.conversations.insert_one({
            "id": cid, "user_id": uid, "title": title, "pinned": False, "archived": False,
            "created_at": iso_now(), "updated_at": iso_now(),
        })

    settings = await db.settings.find_one({"user_id": uid}, NO_ID) or {}
    history = await db.messages.find({"conversation_id": cid}, NO_ID).sort("created_at", 1).to_list(1000)

    await db.messages.insert_one({
        "id": f"msg_{uuid.uuid4().hex[:12]}", "conversation_id": cid, "role": "user",
        "content": transcript, "has_images": False, "image_count": 0, "created_at": iso_now(),
    })

    try:
        reply = await ai.generate_chat_reply(
            session_id=cid, prompt=transcript, history=history,
            memory_enabled=settings.get("memory_enabled", True),
        )
    except Exception:
        logger.exception("voice reply failed")
        raise HTTPException(status_code=502, detail="Gory is having trouble right now. Try again.")

    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    await db.messages.insert_one({
        "id": msg_id, "conversation_id": cid, "role": "assistant",
        "content": reply, "has_images": False, "image_count": 0, "created_at": iso_now(),
    })
    await db.conversations.update_one({"id": cid}, {"$set": {"updated_at": iso_now()}})

    try:
        audio_bytes = await ai.synthesize_speech(reply, voice=voice if voice in
            ("alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer") else "alloy")
    except Exception:
        logger.exception("tts failed")
        audio_bytes = None

    audio_url = None
    if audio_bytes:
        aid = f"voice_{uuid.uuid4().hex[:12]}"
        (VOICE_DIR / f"{aid}.mp3").write_bytes(audio_bytes)
        audio_url = f"/api/voice/audio/{aid}"

    return {
        "conversation_id": cid,
        "transcript": transcript,
        "reply_text": reply,
        "message_id": msg_id,
        "audio_url": audio_url,
    }


@api.get("/voice/audio/{aid}")
async def voice_audio(aid: str):
    p = VOICE_DIR / f"{aid}.mp3"
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(p), media_type="audio/mpeg")


# ---------------------------------------------------------------------------
# Subscription routes
# ---------------------------------------------------------------------------
@api.get("/subscription")
async def subscription_status(user: dict = Depends(get_current_user)):
    sub = await get_subscription(user["user_id"])
    return {"subscription": sub, "is_pro": is_pro(sub), "price_label": PRICE_LABEL}


@api.post("/subscription/start-trial")
async def start_trial(user: dict = Depends(get_current_user)):
    """Activate the 7-day free trial. In production this is confirmed via
    Google Play Billing verification; here it activates for testing."""
    uid = user["user_id"]
    existing = await db.subscriptions.find_one({"user_id": uid}, NO_ID)
    if existing and existing.get("trial_used"):
        # Already used a trial → activate as active subscription instead.
        status, end = "active", (utcnow() + timedelta(days=30)).isoformat()
    else:
        status, end = "trial", (utcnow() + timedelta(days=7)).isoformat()

    sub = {
        "user_id": uid, "status": status, "plan": "pro",
        "price_label": PRICE_LABEL, "trial_used": True,
        "current_period_end": end, "platform": "google_play",
        "updated_at": iso_now(),
    }
    await db.subscriptions.update_one({"user_id": uid}, {"$set": sub}, upsert=True)
    return {"subscription": sub, "is_pro": True}


@api.post("/subscription/cancel")
async def cancel_subscription(user: dict = Depends(get_current_user)):
    uid = user["user_id"]
    await db.subscriptions.update_one(
        {"user_id": uid}, {"$set": {"status": "cancelled", "updated_at": iso_now()}}
    )
    sub = await get_subscription(uid)
    return {"subscription": sub, "is_pro": is_pro(sub)}


@api.post("/subscription/verify")
async def verify_purchase(user: dict = Depends(get_current_user)):
    """Placeholder for Google Play purchase-token verification. Wire the real
    Play Developer API check here at release."""
    return {"verified": False, "detail": "Google Play verification wired at release."}


# ---------------------------------------------------------------------------
# Settings / usage / feedback / reports
# ---------------------------------------------------------------------------
@api.get("/settings")
async def get_settings(user: dict = Depends(get_current_user)):
    s = await db.settings.find_one({"user_id": user["user_id"]}, NO_ID)
    if not s:
        s = {
            "user_id": user["user_id"], "theme": "dark", "accent": "purple",
            "interface_mode": "liquid_glass", "performance_mode": False,
            "memory_enabled": True, "language": "auto",
        }
        await db.settings.insert_one(dict(s))
    return {"settings": s}


@api.put("/settings")
async def update_settings(body: SettingsIn, user: dict = Depends(get_current_user)):
    update = {k: v for k, v in body.dict().items() if v is not None}
    await db.settings.update_one(
        {"user_id": user["user_id"]}, {"$set": update}, upsert=True
    )
    s = await db.settings.find_one({"user_id": user["user_id"]}, NO_ID)
    return {"settings": s}


@api.get("/usage")
async def usage(user: dict = Depends(get_current_user)):
    sub = await get_subscription(user["user_id"])
    plan = plan_key(sub)
    return {
        "usage": await get_usage_today(user["user_id"]),
        "limits": PLAN_LIMITS[plan],
        "plan": plan,
    }


@api.post("/feedback")
async def send_feedback(body: FeedbackIn, user: dict = Depends(get_current_user)):
    doc = {
        "id": f"fb_{uuid.uuid4().hex[:12]}", "user_id": user["user_id"],
        "type": body.type, "message": body.message,
        "has_screenshot": bool(body.screenshot), "created_at": iso_now(),
    }
    await db.feedback.insert_one(dict(doc))
    return {"ok": True, "id": doc["id"]}


@api.post("/reports")
async def report_content(body: ReportIn, user: dict = Depends(get_current_user)):
    doc = {
        "id": f"rep_{uuid.uuid4().hex[:12]}", "user_id": user["user_id"],
        "target_type": body.target_type, "target_id": body.target_id,
        "category": body.category, "note": body.note, "status": "open",
        "created_at": iso_now(),
    }
    await db.reports.insert_one(dict(doc))
    return {"ok": True, "id": doc["id"]}


# ---------------------------------------------------------------------------
app.include_router(api)
app.add_middleware(
    CORSMiddleware, allow_credentials=True, allow_origins=["*"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    await ensure_indexes()
    logger.info("Gory backend started")


@app.on_event("shutdown")
async def _shutdown():
    pass
