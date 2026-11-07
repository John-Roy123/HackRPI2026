"""Main runtime orchestration and error-handling loop for Memory Aid."""

import logging
import threading
import time
from typing import List, Optional, Protocol

from .camera import CameraError
from .config import Config
from .memories import PlaceMemory
from .speech import Speaker
from .vision import (
    VisionAPIError,
    build_messages_payload,
    query_vision_api,
    summarize_payload,
)

logger = logging.getLogger("memoryaid")

MSG_STARTUP_READY = "Hello. Press the button any time and I'll tell you where you are."
MSG_NOT_SET_UP = "I'm not set up yet. Please ask your helper to finish setting me up."
MSG_LET_ME_LOOK = "Let me take a look."
MSG_CAMERA_ERROR = "I can't see clearly just now. Please try again in a moment."
MSG_NETWORK_ERROR = "I'm having trouble seeing right now. Please try again in a moment."


class CameraProtocol(Protocol):
    def start(self) -> None: ...
    def capture_base64(self) -> str: ...
    def close(self) -> None: ...


class ButtonProtocol(Protocol):
    def start(self) -> None: ...
    def run_forever(self) -> None: ...
    def close(self) -> None: ...


def map_error_to_spoken_message(exc: Exception) -> str:
    """Map any runtime exception to a gentle, non-technical spoken reassurance."""
    if isinstance(exc, CameraError):
        return MSG_CAMERA_ERROR
    if isinstance(exc, VisionAPIError):
        return MSG_NETWORK_ERROR
    return MSG_NETWORK_ERROR


class MemoryAidApp:
    """Coordinates the button, camera, memory book, vision API, and speaker."""

    def __init__(
        self,
        config: Config,
        places: List[PlaceMemory],
        camera: CameraProtocol,
        speaker: Speaker,
        button: Optional[ButtonProtocol] = None,
    ) -> None:
        self.config = config
        self.places = places
        self.camera = camera
        self.speaker = speaker
        self.button = button
        self._busy_lock = threading.Lock()

    def start(self) -> None:
        """Initialise hardware and play the startup greeting (or setup reminder)."""
        try:
            self.camera.start()
        except Exception as exc:
            logger.error("Camera initialisation failed at startup: %s", exc)

        if self.button is not None:
            self.button.start()

        if not self.config.has_api_key:
            logger.error(
                "ANTHROPIC_API_KEY is not set! Edit /etc/memoryaid.env (or memoryaid.env) and restart."
            )
            self.speaker.speak(MSG_NOT_SET_UP)
        else:
            self.speaker.speak(MSG_STARTUP_READY)

    def handle_button_press(self) -> None:
        """Handle a single button press. Ignores presses while already busy."""
        if not self._busy_lock.acquire(blocking=False):
            logger.info("Ignoring button press (already busy processing a press).")
            return

        try:
            self._process_press()
        except Exception as exc:
            logger.exception("Unexpected error during button press handling: %s", exc)
            try:
                self.speaker.wait()
                self.speaker.speak(MSG_NETWORK_ERROR)
            except Exception:
                pass
        finally:
            self._busy_lock.release()

    def _process_press(self) -> None:
        t_press = time.monotonic()
        logger.info("Button pressed.")

        if not self.config.has_api_key:
            logger.error("Cannot process press: ANTHROPIC_API_KEY is missing.")
            self.speaker.speak(MSG_NOT_SET_UP)
            return

        # Speak "Let me take a look." asynchronously while capturing and calling API
        self.speaker.speak_async(MSG_LET_ME_LOOK)

        # Step 1: Capture and encode photo
        t_cap_start = time.monotonic()
        try:
            captured_b64 = self.camera.capture_base64()
        except Exception as exc:
            capture_err = exc if isinstance(exc, CameraError) else CameraError(str(exc))
            logger.error("Camera capture error: %s", capture_err)
            self.speaker.wait()
            self.speaker.speak(map_error_to_spoken_message(capture_err))
            return
        capture_sec = time.monotonic() - t_cap_start
        logger.info("Step [capture]: completed in %.2fs", capture_sec)

        # Step 2: Build payload and send to Claude Messages API
        t_prep_start = time.monotonic()
        payload = build_messages_payload(
            places=self.places,
            captured_b64=captured_b64,
            model=self.config.model,
        )
        prep_sec = time.monotonic() - t_prep_start

        try:
            answer_text, payload_bytes, api_sec = query_vision_api(
                payload=payload,
                api_key=self.config.anthropic_api_key,
            )
        except Exception as exc:
            vision_err = exc if isinstance(exc, VisionAPIError) else VisionAPIError(str(exc))
            logger.error("Vision API error: %s", vision_err)
            self.speaker.wait()
            self.speaker.speak(map_error_to_spoken_message(vision_err))
            return

        logger.info(
            "Step [upload+api]: payload=%d bytes (%.1f KB), prep=%.2fs, api_latency=%.2fs",
            payload_bytes,
            payload_bytes / 1024.0,
            prep_sec,
            api_sec,
        )

        # Step 3: Wait for "Let me take a look." to finish, then speak the answer
        self.speaker.wait()
        time_to_first_word = time.monotonic() - t_press
        logger.info("Press-to-first-word latency: %.2fs", time_to_first_word)

        speech_sec = self.speaker.speak(answer_text)
        total_sec = time.monotonic() - t_press
        logger.info(
            "Step [speech]: spoke answer in %.2fs (total cycle=%.2fs)",
            speech_sec,
            total_sec,
        )

    def run_dry_run(self) -> int:
        """Capture one image, build the full request payload, print the summary, and return payload size."""
        self.camera.start()
        captured_b64 = self.camera.capture_base64()
        payload = build_messages_payload(
            places=self.places,
            captured_b64=captured_b64,
            model=self.config.model,
        )
        size_bytes, summary = summarize_payload(payload, self.places)
        print(summary, flush=True)
        return size_bytes

    def close(self) -> None:
        if self.button is not None:
            self.button.close()
        self.camera.close()
