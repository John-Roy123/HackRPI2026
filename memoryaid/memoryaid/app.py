"""Main runtime orchestration for Memory Aid (Identify button + Hold-to-Record button)."""

import logging
import threading
import time
from typing import List, Optional, Protocol

from .camera import CameraError
from .config import Config
from .memories import PlaceMemory, load_memories, save_recorded_memory
from .speech import AudioRecordError, AudioRecorder, Speaker
from .vision import (
    VisionAPIError,
    build_messages_payload,
    query_vision_api,
    summarize_payload,
    transcribe_audio_with_gemini,
)

logger = logging.getLogger("memoryaid")

MSG_STARTUP_READY = "Hello. Press the button any time and I'll tell you where you are."
MSG_NOT_SET_UP = "I'm not set up yet. Please ask your helper to finish setting me up."
MSG_LET_ME_LOOK = "Let me take a look."
MSG_RECORD_SAVED = "I've saved that memory for you."
MSG_RECORD_ERROR = "I couldn't hear that clearly. Please hold the button and try again."
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
    if isinstance(exc, AudioRecordError):
        return MSG_RECORD_ERROR
    if isinstance(exc, VisionAPIError):
        return MSG_NETWORK_ERROR
    return MSG_NETWORK_ERROR


class MemoryAidApp:
    """Coordinates both buttons (Identify + Hold-to-Record), camera, electret microphone, memory book, Gemini API, and speaker."""

    def __init__(
        self,
        config: Config,
        places: List[PlaceMemory],
        camera: CameraProtocol,
        speaker: Speaker,
        recorder: Optional[AudioRecorder] = None,
        button: Optional[ButtonProtocol] = None,
    ) -> None:
        self.config = config
        self.places = places
        self.camera = camera
        self.speaker = speaker
        self.recorder = recorder or AudioRecorder(record_device=config.record_device)
        self.button = button
        self._busy_lock = threading.Lock()
        self._recording_active = False
        self._recorded_photo_b64: Optional[str] = None

    def start(self) -> None:
        """Initialise hardware, connect Bluetooth speaker, and play the startup greeting."""
        try:
            self.speaker.connect_bluetooth()
        except Exception as exc:
            logger.warning("Bluetooth speaker connection check failed at startup: %s", exc)

        try:
            self.camera.start()
        except Exception as exc:
            logger.error("Camera initialisation failed at startup: %s", exc)

        if self.button is not None:
            self.button.start()

        if not self.config.has_api_key:
            logger.error(
                "GEMINI_API_KEY is not set! Edit /etc/memoryaid.env (or memoryaid.env) and restart."
            )
            self.speaker.speak(MSG_NOT_SET_UP)
        else:
            self.speaker.speak(MSG_STARTUP_READY)

    # ------------------------------------------------------------------
    # Button 2: "Where am I?" (single press -> take photo -> tell user)
    # ------------------------------------------------------------------
    def handle_button_press(self) -> None:
        """Handle a press of the 'Where am I?' button. Ignores presses while busy."""
        if not self._busy_lock.acquire(blocking=False):
            logger.info("Ignoring 'Where am I?' button press (already busy).")
            return

        try:
            self._process_identify_press()
        except Exception as exc:
            logger.exception("Unexpected error during 'Where am I?' button handling: %s", exc)
            try:
                self.speaker.wait()
                self.speaker.speak(MSG_NETWORK_ERROR)
            except Exception:
                pass
        finally:
            self._busy_lock.release()

    def _process_identify_press(self) -> None:
        t_press = time.monotonic()
        logger.info("Identify ('Where am I?') button pressed.")

        if not self.config.has_api_key:
            logger.error("Cannot process press: GEMINI_API_KEY is missing.")
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

        # Step 2: Build payload and send to Gemini REST API
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
                api_key=self.config.gemini_api_key,
                model=self.config.model,
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

    # ------------------------------------------------------------------
    # Button 1: "Record Memory" (press and hold -> take photo & listen;
    #           release -> transcribe speech & write photo + text to file)
    # ------------------------------------------------------------------
    def handle_record_press(self) -> None:
        """Called when the user presses and holds the Record Memory button."""
        if not self._busy_lock.acquire(blocking=False):
            logger.info("Ignoring Record button press (device is already busy).")
            return

        logger.info("Record button pressed and held: taking photo and starting electret microphone capture...")
        if not self.config.has_api_key:
            logger.error("Cannot record memory: GEMINI_API_KEY is missing.")
            self._busy_lock.release()
            self.speaker.speak(MSG_NOT_SET_UP)
            return

        try:
            # Start recording audio immediately while the user holds the button
            self.recorder.start_recording()
            self._recording_active = True
            # Take a picture of the room while listening
            self._recorded_photo_b64 = self.camera.capture_base64()
            logger.info("Captured room photo during record hold; listening until button release...")
        except Exception as exc:
            logger.error("Failed to start memory recording: %s", exc)
            self._recording_active = False
            self._recorded_photo_b64 = None
            try:
                self.recorder.stop_recording()
            except Exception:
                pass
            self._busy_lock.release()
            spoken_err = MSG_CAMERA_ERROR if isinstance(exc, CameraError) else MSG_RECORD_ERROR
            self.speaker.speak(spoken_err)

    def handle_record_release(self) -> None:
        """Called when the user releases the Record Memory button."""
        if not self._recording_active:
            return

        self._recording_active = False
        photo_b64 = self._recorded_photo_b64
        self._recorded_photo_b64 = None

        try:
            logger.info("Record button released: stopping electret microphone and transcribing speech...")
            wav_bytes = self.recorder.stop_recording()
            if not photo_b64:
                raise CameraError("No photo was captured during button hold.")

            transcript = transcribe_audio_with_gemini(
                wav_bytes=wav_bytes,
                api_key=self.config.gemini_api_key,
                model=self.config.model,
                captured_b64=photo_b64,
            )
            logger.info("Transcribed spoken memory: %r", transcript)

            self._save_memory_and_reload(photo_b64=photo_b64, note_text=transcript)
            self.speaker.speak(MSG_RECORD_SAVED)
        except Exception as exc:
            logger.error("Failed to finish recording memory: %s", exc)
            self.speaker.speak(map_error_to_spoken_message(exc))
        finally:
            if self._busy_lock.locked():
                self._busy_lock.release()

    def handle_mock_record(self, spoken_text: str) -> None:
        """Simulate a press-and-hold memory recording in laptop mock mode."""
        if not self._busy_lock.acquire(blocking=False):
            logger.info("Ignoring mock record (busy).")
            return
        try:
            if not spoken_text.strip():
                logger.warning("Empty spoken text in mock record.")
                self.speaker.speak(MSG_RECORD_ERROR)
                return
            photo_b64 = self.camera.capture_base64()
            self._save_memory_and_reload(photo_b64=photo_b64, note_text=spoken_text)
            self.speaker.speak(MSG_RECORD_SAVED)
        except Exception as exc:
            logger.error("Mock record failed: %s", exc)
            self.speaker.speak(map_error_to_spoken_message(exc))
        finally:
            self._busy_lock.release()

    def _save_memory_and_reload(self, photo_b64: str, note_text: str) -> PlaceMemory:
        saved_place = save_recorded_memory(
            memories_dir=self.config.memories_dir,
            captured_b64=photo_b64,
            note_text=note_text,
        )
        # Reload places from disk so the newly recorded place is immediately active
        self.places = load_memories(self.config.memories_dir)
        return saved_place

    def run_dry_run(self) -> int:
        """Capture one image, build the full request payload, print the summary, and return payload size."""
        self.camera.start()
        captured_b64 = self.camera.capture_base64()
        payload = build_messages_payload(
            places=self.places,
            captured_b64=captured_b64,
            model=self.config.model,
        )
        size_bytes, summary = summarize_payload(payload, self.places, model=self.config.model)
        print(summary, flush=True)
        return size_bytes

    def close(self) -> None:
        if self.button is not None:
            self.button.close()
        self.camera.close()
