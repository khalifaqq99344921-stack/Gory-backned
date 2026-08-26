"""Provider-agnostic AI layer: chat, vision, image generation, moderation.

All provider credentials live on the backend (EMERGENT_LLM_KEY). The mobile app
never talks to a provider directly.
"""
import os
import re
import asyncio

from emergentintegrations.llm.chat import (
    LlmChat,
    UserMessage,
    ImageContent,
    TextDelta,
    StreamDone,
)
from emergentintegrations.llm.openai import OpenAITextToSpeech, OpenAISpeechToText

EMERGENT_LLM_KEY = os.getenv("EMERGENT_LLM_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")

_eleven_client = None
if ELEVENLABS_API_KEY:
    try:
        from elevenlabs.client import ElevenLabs

        _eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    except Exception:
        _eleven_client = None

# --- Model registry (backend decides which provider/model) -----------------
CHAT_PROVIDER = ("openai", "gpt-5.4")
IMAGE_MODEL = ("gemini", "gemini-3.1-flash-image-preview")

GORY_PERSONA = (
    "You are Gory, a premium, highly capable multimodal AI assistant. "
    "Your tagline is 'See. Speak. Create. Understand.' "
    "You reason carefully about complex questions, coding, math, writing, "
    "translation, studying and image understanding. "
    "You are fully bilingual: respond in Arabic when the user writes Arabic, "
    "in English when they write English, and handle mixed Arabic/English naturally. "
    "Preserve conversation context and ask for clarification when a request is ambiguous. "
    "If you are uncertain, say so instead of inventing facts. "
    "Adapt your response length to the request; be concise by default. "
    "Never claim to be GPT, Gemini, Claude or any specific model — you are simply Gory. "
    "Format answers with clean markdown (headings, lists, code blocks) when helpful."
)

# --- Moderation -------------------------------------------------------------
_BLOCKED_PATTERNS = [
    r"child\s+(sexual|porn|abuse)",
    r"\bcsam\b",
    r"non[-\s]?consensual",
    r"make\s+a\s+bomb",
    r"build\s+a\s+bomb",
    r"how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|explosive|nerve\s+agent)",
]


def moderate_input(text: str) -> tuple[bool, str]:
    """Return (allowed, reason). Lightweight input safety gate."""
    low = (text or "").lower()
    for pat in _BLOCKED_PATTERNS:
        if re.search(pat, low):
            return False, "This request violates our safety policy and can't be processed."
    return True, ""


def _build_chat(session_id: str, system_message: str) -> LlmChat:
    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message=system_message,
    ).with_model(*CHAT_PROVIDER)
    return chat


async def generate_chat_reply(
    session_id: str,
    prompt: str,
    history: list[dict],
    image_base64_list: list[str] | None = None,
    memory_enabled: bool = True,
) -> str:
    """Generate a full assistant reply (accumulated from the stream)."""
    system = GORY_PERSONA
    if memory_enabled and history:
        recent = history[-10:]
        convo = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Gory'}: {m['content']}" for m in recent
        )
        system = f"{GORY_PERSONA}\n\nConversation so far:\n{convo}"

    chat = _build_chat(session_id, system)

    file_contents = None
    if image_base64_list:
        file_contents = [ImageContent(b64) for b64 in image_base64_list]

    msg = UserMessage(text=prompt, file_contents=file_contents) if file_contents else UserMessage(text=prompt)

    result = ""
    async for ev in chat.stream_message(msg):
        if isinstance(ev, TextDelta):
            result += ev.content
        elif isinstance(ev, StreamDone):
            break
    return result.strip() or "…"


# --- Image generation -------------------------------------------------------
_STYLE_HINTS = {
    "automatic": "",
    "photorealistic": "photorealistic, ultra-detailed, natural lighting",
    "cinematic": "cinematic lighting, dramatic composition, film still, depth of field",
    "illustration": "clean digital illustration, vibrant colors",
    "anime": "anime style, cel shading, expressive",
    "3d": "3D render, octane, soft studio lighting",
    "artistic": "artistic painterly style, expressive brush strokes",
}

_ASPECT_HINTS = {
    "1:1": "square 1:1 composition",
    "16:9": "wide 16:9 landscape composition",
    "9:16": "tall 9:16 vertical composition",
    "4:3": "4:3 composition",
    "3:4": "3:4 portrait composition",
}


async def generate_image(
    session_id: str,
    prompt: str,
    aspect_ratio: str = "1:1",
    quality: str = "standard",
    style: str = "automatic",
    reference_base64: str | None = None,
) -> bytes | None:
    """Generate (or edit) an image. Returns raw image bytes or None."""
    import base64

    parts = [prompt]
    if style and _STYLE_HINTS.get(style):
        parts.append(_STYLE_HINTS[style])
    if _ASPECT_HINTS.get(aspect_ratio):
        parts.append(_ASPECT_HINTS[aspect_ratio])
    if quality == "high":
        parts.append("high quality, highly detailed")
    full_prompt = ", ".join(parts)

    chat = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id=session_id,
        system_message="You are Gory Create, an expert image generator.",
    ).with_model(*IMAGE_MODEL).with_params(modalities=["image", "text"])

    if reference_base64:
        msg = UserMessage(text=full_prompt, file_contents=[ImageContent(reference_base64)])
    else:
        msg = UserMessage(text=full_prompt)

    _text, images = await chat.send_message_multimodal_response(msg)
    if not images:
        return None
    return base64.b64decode(images[0]["data"])


# --- Voice: speech-to-text & text-to-speech --------------------------------
def _clean_for_tts(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    text = re.sub(r"[*_#>~|]", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def transcribe_audio(file_path: str, language: str | None = None) -> str:
    stt = OpenAISpeechToText(api_key=EMERGENT_LLM_KEY)
    with open(file_path, "rb") as f:
        resp = await stt.transcribe(file=f, response_format="json", language=language)
    text = getattr(resp, "text", None)
    if text is None and isinstance(resp, dict):
        text = resp.get("text")
    return (text or "").strip()


def _has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _elevenlabs_tts_sync(text: str) -> bytes:
    audio = _eleven_client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    data = b""
    for chunk in audio:
        if chunk:
            data += chunk
    return data


async def synthesize_speech(text: str, voice: str = "alloy") -> bytes:
    clean = _clean_for_tts(text)[:2500] or "…"
    # Prefer ElevenLabs multilingual for Arabic (native accent) when configured.
    if _eleven_client and _has_arabic(clean):
        try:
            audio = await asyncio.to_thread(_elevenlabs_tts_sync, clean)
            if audio:
                return audio
        except Exception:
            pass  # fall back to OpenAI TTS
    tts = OpenAITextToSpeech(api_key=EMERGENT_LLM_KEY)
    return await tts.generate_speech(text=clean[:4096], model="tts-1", voice=voice)
