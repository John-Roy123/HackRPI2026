"""Google Gemini REST API client (`generateContent`) over raw HTTPS using `requests`."""

import base64
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from .memories import PlaceMemory, build_contact_sheet_b64

logger = logging.getLogger("memoryaid")

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3-flash-preview"
FALLBACK_MODELS = (
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-2.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-3.1-pro-preview",
    "gemini-pro-latest",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
)
MAX_OUTPUT_TOKENS = 200
REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAY_SECONDS = 2.0
_DISCOVERED_MODEL: str = ""
_FAILED_404_MODELS: set = set()

SYSTEM_PROMPT = """You are a gentle memory companion carried by a person living with dementia.
You will be given a set of known places from their memory book (each with reference photos and a caregiver's note/description), followed by a NEW photo taken by their onboard Raspberry Pi camera right now.

Your ONLY task is to compare the NEW photo against the provided reference photos and caregiver notes, and from EXCLUSIVELY that provided information determine which known place the user is in and their personal relationship/connection to that place.

Strict Rules:
- The NEW photo comes from a small wearable Raspberry Pi camera, which may have a different camera angle, field of view, zoom, lighting, exposure, or color balance compared to the reference photos (which may have been taken on a phone or from another angle in the same room).
- Look carefully for shared landmarks, furniture, walls, cabinets, tables, windows, appliances, flooring, or room structure that indicate the user is in the SAME room or place as one of the reference tiles (#1, #2, ...), even from a different angle or distance.
- Whenever the room or area reasonably corresponds to one of the known places, select that matching place!
- Do NOT describe the visual scenery, objects, or room layout on your own. Your job is strictly to identify the matching known place and tell the user where they are and their relationship to it using ONLY the facts, names, and relationships in that place's caregiver note.
- Speak directly to the user ("You're at..." or "You're in..."), stating the place and their personal relationship/routine there based exclusively on the matching caregiver note.
- Only if the NEW photo is clearly a completely unrelated environment from all provided known places (or if no known places are provided), say: "I don't recognize this place from your memory book yet." Do NOT describe the room or surroundings.
- Use 1 to 3 short, warm, reassuring sentences in simple everyday words.
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
    contact_sheet_b64: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct the token-efficient JSON payload for Gemini's `generateContent` REST endpoint.

    Uses a single composite Contact-Sheet image containing all reference photos labeled `#1`, `#2`, etc.
    so at most 2 images total (1 reference contact sheet + 1 new camera photo) are sent per button press.
    """
    parts: List[Dict[str, Any]] = []

    if places:
        notes_lines = ["Known places in memory book:"]
        for idx, place in enumerate(places, start=1):
            notes_lines.append(f"#{idx} ({place.name}): {place.note}")
        parts.append({"text": "\n".join(notes_lines)})

        sheet_b64 = (
            contact_sheet_b64
            if contact_sheet_b64 is not None
            else build_contact_sheet_b64(places)
        )
        if sheet_b64:
            parts.append(
                {
                    "text": "Reference Contact Sheet of known places (each tile is labeled #N in its top-left corner):",
                }
            )
            parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": sheet_b64,
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
            "text": (
                "Compare the NEW photo above against the Reference Contact Sheet tiles (#1, #2, ...) and "
                "caregiver notes provided above (accounting for differences in camera angle, field of view, "
                "and lighting between the onboard Pi camera and reference photos). Do NOT describe the room. "
                "Using exclusively the matching caregiver note, tell me where I am and my relationship to this place."
            ),
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


def _discover_available_gemini_MODELS(api_key: str) -> List[str]:
    """Query `GET /v1beta/models` for this API key and return all non-failed models supporting `generateContent`."""
    try:
        resp = requests.get(
            GEMINI_API_BASE,
            params={"key": api_key.strip()},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning("ListModels returned HTTP %d: %s", resp.status_code, (resp.text or "")[:160])
            return []
        models = resp.json().get("models") or []
        supported: List[str] = []
        for m in models:
            methods = m.get("supportedGenerationMethods") or []
            name = m.get("name", "")
            if "generateContent" in methods and name:
                short = name[len("models/") :] if name.startswith("models/") else name
                low = short.lower()
                if (
                    "gemini" in low
                    and "tts" not in low
                    and "embedding" not in low
                    and "image" not in low
                    and "customtools" not in low
                    and short not in _FAILED_404_MODELS
                ):
                    supported.append(short)
        logger.info("Available Gemini models for this API key: %s", ", ".join(supported[:10]))

        # Rank newest / fastest Flash models first
        priority = (
            "gemini-3-flash-preview",
            "gemini-flash-latest",
            "gemini-2.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.1-pro-preview",
            "gemini-pro-latest",
            "gemini-2.5-pro",
        )
        ordered: List[str] = []
        for pref in priority:
            if pref in supported and pref not in ordered:
                ordered.append(pref)
        for s in supported:
            if s not in ordered:
                ordered.append(s)
        return ordered
    except Exception as exc:
        logger.warning("Failed to auto-discover Gemini models: %s", exc)
    return []


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
    candidate_models = [primary_model] + [
        m for m in FALLBACK_MODELS if m != primary_model and m not in _FAILED_404_MODELS
    ]

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
    discovered_once = False

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
                    logger.info(
                        "Gemini API succeeded with model=%s in %.2fs -> %r",
                        current_model,
                        latency,
                        cleaned,
                    )
                    return cleaned, payload_size_bytes, latency

                if status == 404:
                    _FAILED_404_MODELS.add(current_model)
                    if _DISCOVERED_MODEL == current_model:
                        _DISCOVERED_MODEL = ""
                    err_detail = (response.text or "")[:200].strip()
                    logger.warning(
                        "Gemini model %r returned HTTP 404 (%s); trying next available Gemini model...",
                        current_model,
                        err_detail,
                    )
                    if not discovered_once:
                        discovered_once = True
                        for dm in reversed(_discover_available_gemini_MODELS(clean_key)):
                            if dm not in candidate_models and dm not in _FAILED_404_MODELS:
                                candidate_models.insert(idx, dm)
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
