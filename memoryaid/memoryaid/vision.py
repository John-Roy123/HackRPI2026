"""Claude Messages API client over raw HTTPS using `requests`."""

import json
import logging
import re
import time
from typing import Any, Dict, List, Tuple

import requests

from .memories import PlaceMemory

logger = logging.getLogger("memoryaid")

ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
MAX_TOKENS = 200
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


class VisionAPIError(Exception):
    """Raised when the Anthropic Messages API fails after any applicable retry."""


def clean_spoken_text(text: str) -> str:
    """Strip stray markdown characters (*, #, _, backticks), outer quotes, and extra whitespace."""
    cleaned = re.sub(r"[*#_`]+", "", text)
    cleaned = cleaned.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ('"', "'", "\u201c", "\u201d"):
        cleaned = cleaned[1:-1].strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def build_messages_payload(
    places: List[PlaceMemory],
    captured_b64: str,
    model: str = "claude-haiku-5-5",
) -> Dict[str, Any]:
    """Construct the exact JSON payload for POST https://api.anthropic.com/v1/messages."""
    content: List[Dict[str, Any]] = []

    for idx, place in enumerate(places, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Known place {idx}: {place.name}\nCaregiver note: {place.note}",
            }
        )
        for img_b64 in place.images_b64:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64,
                    },
                }
            )

    content.append(
        {
            "type": "text",
            "text": "NEW photo from the camera right now:",
        }
    )
    content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": captured_b64,
            },
        }
    )
    content.append(
        {
            "type": "text",
            "text": "Where am I?",
        }
    )

    return {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
    }


def summarize_payload(payload: Dict[str, Any], places: List[PlaceMemory]) -> Tuple[int, str]:
    """Serialize payload to JSON, compute byte size, and build a human-readable dry-run summary."""
    serialized = json.dumps(payload)
    size_bytes = len(serialized.encode("utf-8"))
    size_kb = size_bytes / 1024.0

    user_content = payload["messages"][0]["content"]
    num_images = sum(1 for block in user_content if block.get("type") == "image")
    text_blocks = [block.get("text", "") for block in user_content if block.get("type") == "text"]

    lines = [
        "=== Memory Aid Dry-Run Summary ===",
        f"Model:            {payload.get('model')}",
        f"Max tokens:       {payload.get('max_tokens')}",
        f"Known places:     {len(places)}",
        f"Total images:     {num_images} ({num_images - 1} reference + 1 new capture)",
        f"Payload size:     {size_kb:.2f} KB ({size_bytes} bytes)",
        "",
        "--- Text Blocks in User Message ---",
    ]
    for i, text_block in enumerate(text_blocks, start=1):
        lines.append(f"[{i}] {text_block}")
    lines.append("==================================")
    return size_bytes, "\n".join(lines)


def query_vision_api(
    payload: Dict[str, Any],
    api_key: str,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    retry_delay: float = RETRY_DELAY_SECONDS,
) -> Tuple[str, int, float]:
    """Send the vision request to Claude Messages API with 1 retry on network error, 429, or 5xx.

    Returns (cleaned_answer_text, payload_size_bytes, api_latency_seconds).
    """
    if not api_key or not api_key.strip():
        raise VisionAPIError("Missing ANTHROPIC_API_KEY.")

    body_bytes = json.dumps(payload).encode("utf-8")
    payload_size_bytes = len(body_bytes)
    logger.info(
        "Prepared Claude API payload: %.1f KB (%d bytes), model=%s",
        payload_size_bytes / 1024.0,
        payload_size_bytes,
        payload.get("model"),
    )

    headers = {
        "x-api-key": api_key.strip(),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    last_error: Exception = VisionAPIError("Unknown API failure")
    t_start = time.monotonic()

    for attempt in (1, 2):
        try:
            response = requests.post(
                ANTHROPIC_ENDPOINT,
                headers=headers,
                data=body_bytes,
                timeout=timeout,
            )
            status = response.status_code

            if status == 200:
                latency = time.monotonic() - t_start
                data = response.json()
                raw_text = data["content"][0]["text"]
                cleaned = clean_spoken_text(raw_text)
                if not cleaned:
                    raise VisionAPIError("API returned empty text content.")
                return cleaned, payload_size_bytes, latency

            if status == 429 or status >= 500:
                last_error = VisionAPIError(f"HTTP {status} from Anthropic API")
                if attempt == 1:
                    logger.warning(
                        "Anthropic API returned HTTP %d on attempt 1; retrying in %.1fs...",
                        status,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    continue
                raise last_error

            # 4xx client error (other than 429) - do not retry
            raise VisionAPIError(f"HTTP {status} client error from Anthropic API")

        except requests.RequestException as exc:
            last_error = VisionAPIError(f"Network error contacting Anthropic API: {type(exc).__name__}")
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
            raise VisionAPIError(f"Malformed response from Anthropic API: {exc}") from exc

    raise last_error
