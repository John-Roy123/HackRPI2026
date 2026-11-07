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
MAX_OUTPUT_TOKENS = 200
REQUEST_TIMEOUT_SECONDS = 30
RETRY_DELAY_SECONDS = 2.0

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
    clean_model = model.strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model[len("models/") :]
    return f"{GEMINI_API_BASE}/{clean_model}:generateContent"


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


def query_vision_api(
    payload: Dict[str, Any],
    api_key: str,
    model: str = DEFAULT_MODEL,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> Tuple[str, int, float]:
    """Send the vision/audio request to the Gemini API with 1 retry on network error, 429, or 5xx.

    Returns (cleaned_answer_text, payload_size_bytes, api_latency_seconds).
    """
    if not api_key or not api_key.strip():
        raise VisionAPIError("Missing GEMINI_API_KEY.")

    endpoint = get_gemini_endpoint(model)
    body_bytes = json.dumps(payload).encode("utf-8")
    payload_size_bytes = len(body_bytes)
    logger.info(
        "Prepared Gemini API payload: %.1f KB (%d bytes), model=%s",
        payload_size_bytes / 1024.0,
        payload_size_bytes,
        model,
    )

    headers = {
        "x-goog-api-key": api_key.strip(),
        "content-type": "application/json",
    }

    last_error: Exception = VisionAPIError("Unknown API failure")
    t_start = time.monotonic()

    for attempt in (1, 2):
        try:
            response = requests.post(
                endpoint,
                headers=headers,
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
                return cleaned, payload_size_bytes, latency

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

            # 4xx client error (other than 429) - do not retry
            raise VisionAPIError(f"HTTP {status} client error from Gemini API")

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
