"""Google Gemini REST API client (`generateContent`) over raw HTTPS using `requests`."""

import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

import requests

from .memories import PlaceMemory

logger = logging.getLogger("memoryaid")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash"
FALLBACK_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
)
MAX_OUTPUT_TOKENS = 200
REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAY_SECONDS = 2.0
_DISCOVERED_MODEL: str = ""

SYSTEM_PROMPT = """You are a gentle memory companion carried by a person living with dementia.
You will see reference photos of places that matter to them, each with a caregiver's note,
then a NEW photo taken by their camera right now.

Rules:
- If the new photo clearly matches a known place, tell them where they are and their personal
  connection to it, using the caregiver's note. Use names from the note.
- Only claim a match if you are confident. If unsure, do not guess a known place; describe
  what you see instead.
- If there is no match, calmly describe the setting in simple, concrete terms.
- Speak directly to them ("You're at..."). 2 to 3 short sentences. Simple everyday words.
- Be warm and reassuring. Never alarming, never clinical.
- Never quiz them, never say "you should remember", "don't you remember", or "as you know".
- Do not mention photos, cameras, notes, AI, or matching.
- Output only the words to be spoken aloud: no markdown, no lists, no emoji, no quotation marks."""

TRANSCRIPTION_PROMPT = (
    "Transcribe exactly what the speaker says in this audio recording about the room or place shown. "
    "Output only the spoken words as plain text with normal punctuation. "
    "Do not add commentary, labels, or quotation marks. If no speech is audible, output NOTHING."
)


class VisionAPIError(Exception):
    """Raised when the Gemini API fails after any applicable retry."""


def clean_spoken_text(text: str) -> str:
    """Strip stray markdown characters (*, #, _, backticks), outer quotes, and extra whitespace."""
    cleaned = re.sub(r"[*#_`]+", "", text)
    cleaned = cleaned.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'", "\u201c", "\u201d"):
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def get_gemini_endpoint(model: str = DEFAULT_MODEL) -> str:
    """Return the REST generateContent URL for the configured Gemini model."""
    clean_model = _normalize_gemini_model_name(model)
    return f"{GEMINI_API_BASE}/{clean_model}:generateContent"


def _normalize_gemini_model_name(model: str) -> str:
    """Normalize common model names (e.g. legacy Claude names in old .env files) to a valid Gemini model."""
    clean = (model or "").strip()
    if clean.startswith("models/"):
        clean = clean[len("models/") :]
    if not clean or clean.lower().startswith("claude"):
        logger.warning(
            "Configured MODEL=%r is not a Gemini model; automatically using %r instead.",
            model,
            DEFAULT_MODEL,
        )
        return DEFAULT_MODEL
    return clean


def build_messages_payload(
    places: List[PlaceMemory],
    captured_b64: str,
    model: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Construct the JSON payload for Gemini's `generateContent` REST endpoint."""
    parts: List[Dict[str, Any]] = []

    for idx, place in enumerate(places, start=1):
        parts.append(
            {
                "text": f"Known place {idx}: {place.name}\nCaregiver note: {place.note}",
            }
        )
        for img_b64 in place.images_b64:
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_b64,
                    }
                }
            )

    parts.append(
        {
            "text": "NEW photo from the camera right now:",
        }
    )
    parts.append(
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": captured_b64,
            }
        }
    )
    parts.append(
        {
            "text": "Where am I?",
        }
    )

    return {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}],
        },
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }


def build_transcription_payload(
    wav_bytes: bytes,
    captured_b64: str = "",
) -> Dict[str, Any]:
    """Construct a Gemini `generateContent` payload to transcribe recorded WAV audio."""
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    parts: List[Dict[str, Any]] = []
    if captured_b64:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": captured_b64,
                }
            }
        )
    parts.append(
        {
            "inline_data": {
                "mime_type": "audio/wav",
                "data": audio_b64,
            }
        }
    )
    parts.append({"text": TRANSCRIPTION_PROMPT})

    return {
        "contents": [
            {
                "role": "user",
                "parts": parts,
            }
        ],
        "generationConfig": {
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
    }


def transcribe_audio_with_gemini(
    wav_bytes: bytes,
    api_key: str,
    model: str = DEFAULT_MODEL,
    captured_b64: str = "",
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> str:
    """Send recorded WAV audio to Gemini and return the plain-text transcription."""
    payload = build_transcription_payload(wav_bytes=wav_bytes, captured_b64=captured_b64)
    transcript, _size, _latency = query_vision_api(
        payload=payload,
        api_key=api_key,
        model=model,
        timeout=timeout,
        retry_delay=retry_delay,
    )
    if transcript.strip().upper() == "NOTHING":
        raise VisionAPIError("No speech detected in audio recording.")
    return transcript


def summarize_payload(
    payload: Dict[str, Any], places: List[PlaceMemory], model: str = DEFAULT_MODEL
) -> Tuple[int, str]:
    """Serialize payload to JSON, compute byte size, and build a human-readable dry-run summary."""
    serialized = json.dumps(payload)
    size_bytes = len(serialized.encode("utf-8"))
    size_kb = size_bytes / 1024.0

    user_parts = payload["contents"][0]["parts"]
    num_images = sum(1 for part in user_parts if "inline_data" in part)
    text_blocks = [part["text"] for part in user_parts if "text" in part]
    max_tokens = payload.get("generationConfig", {}).get("maxOutputTokens", MAX_OUTPUT_TOKENS)

    lines = [
        "=== Memory Aid Dry-Run Summary (Gemini API) ===",
        f"Model:            {model}",
        f"Endpoint:         {get_gemini_endpoint(model)}",
        f"Max tokens:       {max_tokens}",
        f"Known places:     {len(places)}",
        f"Total images:     {num_images} ({num_images - 1} reference + 1 new capture)",
        f"Payload size:     {size_kb:.2f} KB ({size_bytes} bytes)",
        "",
        "--- Text Blocks in User Message ---",
    ]
    for i, text_block in enumerate(text_blocks, start=1):
        lines.append(f"[{i}] {text_block}")
    lines.append("===============================================")
    return size_bytes, "\n".join(lines)


def _discover_available_gemini_model(api_key: str) -> str:
    """Query `GET /v1beta/models` for this API key and return the best model supporting `generateContent`."""
    global _DISCOVERED_MODEL
    if _DISCOVERED_MODEL:
        return _DISCOVERED_MODEL
    try:
        resp = requests.get(
            GEMINI_API_BASE,
            params={"key": api_key.strip()},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("ListModels returned HTTP %d: %s", resp.status_code, (resp.text or "")[:160])
            return ""
        models = resp.json().get("models") or []
        supported: List[str] = []
        for m in models:
            methods = m.get("supportedGenerationMethods") or []
            name = m.get("name", "")
            if "generateContent" in methods and name:
                short = name[len("models/") :] if name.startswith("models/") else name
                if "gemini" in short.lower() and "tts" not in short.lower() and "embedding" not in short.lower():
                    supported.append(short)
        logger.info("Available Gemini models for this API key: %s", ", ".join(supported[:10]))
        for pref in ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "flash"):
            for s in supported:
                if pref in s:
                    _DISCOVERED_MODEL = s
                    return s
        if supported:
            _DISCOVERED_MODEL = supported[0]
            return supported[0]
    except Exception as exc:
        logger.warning("Failed to auto-discover Gemini models: %s", exc)
    return ""


def query_vision_api(
    payload: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> Tuple[str, int, float]:
    """Send the vision/audio request to the Gemini API with model fallback on 404 and 1 retry on network error, 429, or 5xx.

    Returns (cleaned_answer_text, payload_size_bytes, api_latency_seconds).
    """
    global _DISCOVERED_MODEL
    if not api_key or not api_key.strip():
        raise VisionAPIError("Missing GEMINI_API_KEY.")

    primary_model = _DISCOVERED_MODEL or _normalize_gemini_model_name(model)
    candidate_models = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    body_bytes = json.dumps(payload).encode("utf-8")
    payload_size_bytes = len(body_bytes)
    logger.info(
        "Prepared Gemini API payload: %.1f KB (%d bytes), model=%s",
        payload_size_bytes / 1024.0,
        payload_size_bytes,
        primary_model,
    )

    clean_key = api_key.strip()
    headers = {
        "x-goog-api-key": clean_key,
        "content-type": "application/json",
    }

    last_error: Exception = VisionAPIError("Unknown API failure")
    t_start = time.monotonic()

    idx = 0
    while idx < len(candidate_models):
        current_model = candidate_models[idx]
        idx += 1
        endpoint = get_gemini_endpoint(current_model)
        for attempt in (1, 2):
            try:
                response = requests.post(
                    endpoint,
                    headers=headers,
                    params={"key": clean_key},
                    data=body_bytes,
                    timeout=timeout,
                )
                status = response.status_code

                if status == 200:
                    latency = time.monotonic() - t_start
                    data = response.json()
                    candidates = data.get("candidates") or []
                    if not candidates:
                        raise VisionAPIError("Gemini API returned no candidates.")
                    parts = candidates[0].get("content", {}).get("parts") or []
                    raw_text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
                    cleaned = clean_spoken_text(raw_text)
                    if not cleaned:
                        raise VisionAPIError("Gemini API returned empty text content.")
                    _DISCOVERED_MODEL = current_model
                    return cleaned, payload_size_bytes, latency

                if status == 404:
                    err_detail = (response.text or "")[:160].strip()
                    logger.warning(
                        "Gemini model %r returned HTTP 404 (%s); discovering available models...",
                        current_model,
                        err_detail,
                    )
                    discovered = _discover_available_gemini_model(clean_key)
                    if discovered and discovered not in candidate_models:
                        candidate_models.insert(idx, discovered)
                    last_error = VisionAPIError(f"HTTP 404 model not found ({current_model}): {err_detail}")
                    break  # Try next candidate model immediately

                if status == 429 or status >= 500:
                    last_error = VisionAPIError(f"HTTP {status} from Gemini API")
                    if attempt == 1:
                        logger.warning(
                            "Gemini API returned HTTP %d on attempt 1; retrying in %.1fs...",
                            status,
                            retry_delay,
                        )
                        time.sleep(retry_delay)
                        continue
                    raise last_error

                # 4xx client error (other than 404/429) - do not retry
                err_detail = (response.text or "")[:200].strip()
                raise VisionAPIError(f"HTTP {status} client error from Gemini API: {err_detail}")

            except requests.RequestException as exc:
                last_error = VisionAPIError(f"Network error contacting Gemini API: {type(exc).__name__}")
                if attempt == 1:
                    logger.warning(
                        "Network error on attempt 1 (%s); retrying in %.1fs...",
                        type(exc).__name__,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    continue
                raise last_error from exc
            except (KeyError, IndexError, ValueError, TypeError) as exc:
                raise VisionAPIError(f"Malformed response from Gemini API: {exc}") from exc

    raise last_error
