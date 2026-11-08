"""Main runtime orchestration for Memory Aid (Identify button + Hold-to-Record button)."""

import base64
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional, Protocol

from .camera import CameraError
from .config import Config
from .memories import (
    PlaceMemory,
    build_contact_sheet_b64,
    load_memories,
    save_recorded_memory,
    sync_remote_website_memories,
    upload_memory_to_website,
)
from .speech import AudioRecordError, AudioRecorder, Speaker
from .vision import (
    VisionAPIError,
    build_messages_payload,
    query_vision_api,
    resolve_exact_place_description,
    summarize_payload,
    transcribe_audio_with_gemini,
)

logger = logging.getLogger("memoryaid")

MSG_STARTUP_READY = "Hello. Press the button any time and I'll tell you where you are."
MSG_NOT_SET_UP = "I'm not set up yet. Please ask your helper to finish setting me up."
MSG_LET_ME_LOOK = "Let me take a look."
MSG_RECORD_ACTIVE = "Recording is active. Please describe this place."
MSG_RECORD_SAVED = "I've saved that memory for you."
MSG_RECORD_UPLOADING = "Uploading your memory to the website."
MSG_RECORD_ERROR = "I couldn't hear that clearly. Please press the record button and try again."
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
        self.contact_sheet_b64: Optional[str] = build_contact_sheet_b64(
            self.places,
            cache_dir=self.config.memories_dir / ".cache",
            captures_dir=self.config.captures_dir,
        )

    def start(self) -> None:
        """Initialise hardware, sync caregiver website memories, connect Bluetooth speaker, and play startup greeting."""
        try:
            self.sync_website_memories()
        except Exception as exc:
            logger.warning("Initial website memory sync skipped: %s", exc)

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
        """Handle a press of the 'Where am I?' button. If a recording is active, saves it first."""
        if self._recording_active:
            logger.info("Finishing and saving active recording before processing 'Where am I?' button press...")
            self.handle_record_release()

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

    def sync_website_memories(self) -> bool:
        """Check the external HackRPI2026 Caregiver Website for any new/updated photos and descriptions.

        If new or updated memories are found, downloads them into `memories/` and rebuilds the Contact Sheet.
        """
        t0 = time.monotonic()
        changed = sync_remote_website_memories(
            memories_dir=self.config.memories_dir,
            website_url=self.config.website_url,
        )
        if changed or not self.places:
            self.places = load_memories(self.config.memories_dir)
            self.contact_sheet_b64 = build_contact_sheet_b64(
                self.places,
                cache_dir=self.config.memories_dir / ".cache",
                captures_dir=self.config.captures_dir,
            )
            logger.info(
                "Step [website-sync]: loaded new/updated website memories in %.2fs (%d total places active).",
                time.monotonic() - t0,
                len(self.places),
            )
        else:
            logger.info("Step [website-sync]: up to date (checked in %.2fs).", time.monotonic() - t0)
        return changed

    def _process_identify_press(self) -> None:
        t_press = time.monotonic()
        logger.info("Identify ('Where am I?') button pressed.")

        if not self.config.has_api_key:
            logger.error("Cannot process press: GEMINI_API_KEY is missing.")
            self.speaker.speak(MSG_NOT_SET_UP)
            return

        # Speak "Let me take a look." asynchronously while syncing website, capturing, and calling API
        self.speaker.speak_async(MSG_LET_ME_LOOK)

        # Step 0: Check the external caregiver website for any new/updated photos and load them onto the device
        try:
            self.sync_website_memories()
        except Exception as exc:
            logger.warning("Website memory sync check failed (continuing with existing memories): %s", exc)

        # Step 1: Capture and encode photo
        t_cap_start = time.monotonic()
        try:
            captured_b64 = self.camera.capture_base64()
            self._persist_capture_locally(captured_b64)
        except Exception as exc:
            capture_err = exc if isinstance(exc, CameraError) else CameraError(str(exc))
            logger.error("Camera capture error: %s", capture_err)
            self.speaker.wait()
            self.speaker.speak(map_error_to_spoken_message(capture_err))
            return
        capture_sec = time.monotonic() - t_cap_start
        logger.info("Step [capture]: completed in %.2fs", capture_sec)

        # Step 2: Build payload and send to Gemini REST API
        places_summary = ", ".join(
            f"#{i} {p.name} ({len(p.images_b64)} photo(s))" for i, p in enumerate(self.places, start=1)
        )
        logger.info(
            "Comparing onboard camera capture against %d known place(s): [%s]",
            len(self.places),
            places_summary or "none",
        )
        t_prep_start = time.monotonic()
        payload = build_messages_payload(
            places=self.places,
            captured_b64=captured_b64,
            model=self.config.model,
            contact_sheet_b64=self.contact_sheet_b64,
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

        spoken_text = resolve_exact_place_description(answer_text, self.places)
        logger.info(
            "Step [upload+api]: payload=%d bytes (%.1f KB), prep=%.2fs, api_latency=%.2fs, spoken=%r",
            payload_bytes,
            payload_bytes / 1024.0,
            prep_sec,
            api_sec,
            spoken_text,
        )

        # Step 3: Wait for "Let me take a look." to finish, then speak the exact stored description
        self.speaker.wait()
        time_to_first_word = time.monotonic() - t_press
        logger.info("Press-to-first-word latency: %.2fs", time_to_first_word)

        speech_sec = self.speaker.speak(spoken_text)
        total_sec = time.monotonic() - t_press
        logger.info(
            "Step [speech]: spoke answer in %.2fs (total cycle=%.2fs)",
            speech_sec,
            total_sec,
        )

    # ------------------------------------------------------------------
    # Button 1: "Record Memory" (1st press -> take photo & start recording;
    #           2nd press -> stop recording, transcribe speech & save file)
    # ------------------------------------------------------------------
    def handle_record_press(self) -> None:
        """Toggle recording on press: 1st press starts recording, 2nd press stops and saves."""
        if self._recording_active:
            self.handle_record_release()
            return

        if not self._busy_lock.acquire(blocking=False):
            logger.info("Ignoring Record button press (device is already busy).")
            return

        logger.info("Record button pressed (1st press): starting microphone capture and taking photo...")
        if not self.config.has_api_key:
            logger.error("Cannot record memory: GEMINI_API_KEY is missing.")
            self._busy_lock.release()
            self.speaker.speak(MSG_NOT_SET_UP)
            return

        try:
            self._recording_active = True
            self._recorded_photo_b64 = self.camera.capture_base64()
            self._persist_capture_locally(self._recorded_photo_b64)
            print(
                "\n[SYSTEM] Recording is active — please describe this place. Press the Record button again when finished.",
                flush=True,
            )
            self.speaker.speak(MSG_RECORD_ACTIVE)
            self.recorder.start_recording()
            logger.info("Captured room photo and recording audio; press Record button again to stop and save.")
        except Exception as exc:
            logger.error("Failed to start memory recording: %s", exc)
            self._recording_active = False
            self._recorded_photo_b64 = None
            try:
                self.recorder.stop_recording()
            except Exception:
                pass
            if self._busy_lock.locked():
                try:
                    self._busy_lock.release()
                except RuntimeError:
                    pass
            spoken_err = MSG_CAMERA_ERROR if isinstance(exc, CameraError) else MSG_RECORD_ERROR
            self.speaker.speak(spoken_err)

    def handle_record_release(self) -> None:
        """Stop an active recording (called on 2nd press of Record button), transcribe speech, save locally, and upload."""
        if not self._recording_active:
            return

        self._recording_active = False
        photo_b64 = self._recorded_photo_b64
        self._recorded_photo_b64 = None

        try:
            logger.info("Record button pressed (2nd press): stopping microphone and transcribing speech...")
            wav_bytes = self.recorder.stop_recording()
            if not photo_b64:
                raise CameraError("No photo was captured when recording started.")

            transcript = transcribe_audio_with_gemini(
                wav_bytes=wav_bytes,
                api_key=self.config.gemini_api_key,
                model=self.config.model,
                captured_b64=photo_b64,
            )
            logger.info("Transcribed spoken memory: %r", transcript)

            saved_place = self._save_memory_and_reload(photo_b64=photo_b64, note_text=transcript)
            print(
                f"\n[SYSTEM] Recording saved locally ({saved_place.name}): {transcript!r}",
                flush=True,
            )
            self.speaker.speak(MSG_RECORD_SAVED)

            print(
                f"\n[SYSTEM] Uploading photo and description for '{saved_place.name}' to the website...",
                flush=True,
            )
            self.speaker.speak(MSG_RECORD_UPLOADING)
            self._upload_saved_memory(saved_place=saved_place, photo_b64=photo_b64)
        except Exception as exc:
            logger.error("Failed to finish recording memory: %s", exc)
            self.speaker.speak(map_error_to_spoken_message(exc))
        finally:
            if self._busy_lock.locked():
                try:
                    self._busy_lock.release()
                except RuntimeError:
                    pass

    def handle_mock_record(self, spoken_text: str) -> None:
        """Record a memory with an explicit text description (or finish an active CLI recording with text)."""
        was_recording = self._recording_active
        if was_recording:
            self._recording_active = False
            try:
                self.recorder.stop_recording()
            except Exception:
                pass
        elif not self._busy_lock.acquire(blocking=False):
            logger.info("Ignoring mock record (busy).")
            return

        try:
            if not spoken_text.strip():
                logger.warning("Empty spoken text in mock record.")
                self.speaker.speak(MSG_RECORD_ERROR)
                return
            photo_b64 = self._recorded_photo_b64
            self._recorded_photo_b64 = None
            if not photo_b64:
                print("\n[SYSTEM] Recording is active — capturing photo...", flush=True)
                self.speaker.speak(MSG_RECORD_ACTIVE)
                photo_b64 = self.camera.capture_base64()
                self._persist_capture_locally(photo_b64)
            saved_place = self._save_memory_and_reload(photo_b64=photo_b64, note_text=spoken_text)
            print(
                f"\n[SYSTEM] Recording saved locally ({saved_place.name}): {spoken_text!r}",
                flush=True,
            )
            self.speaker.speak(MSG_RECORD_SAVED)
            print(
                f"\n[SYSTEM] Uploading photo and description for '{saved_place.name}' to the website...",
                flush=True,
            )
            self.speaker.speak(MSG_RECORD_UPLOADING)
            self._upload_saved_memory(saved_place=saved_place, photo_b64=photo_b64)
        except Exception as exc:
            logger.error("Mock record failed: %s", exc)
            self.speaker.speak(map_error_to_spoken_message(exc))
        finally:
            if self._busy_lock.locked():
                try:
                    self._busy_lock.release()
                except RuntimeError:
                    pass

    def run_cli_record(self, description_arg: str = "") -> int:
        """Command-line Record & Upload toggle for proof-of-concept (`python3 -m memoryaid --record`).

        - If `description_arg` is provided on the CLI (`--record "This is the kitchen"`):
          captures a photo, speaks the active/saved/uploading prompts, saves locally, and uploads to the website.
        - Otherwise (interactive toggle mode, `--record` with no text):
          1st step starts recording + captures photo + speaks active prompt, then waits for Enter
          (or a typed description + Enter) to stop recording, save locally, and upload to the website.
        """
        import sys

        try:
            self.speaker.connect_bluetooth()
        except Exception as exc:
            logger.warning("Bluetooth speaker connection check failed: %s", exc)

        self.camera.start()

        # Mode A: Description text passed directly via `--record "This is the kitchen"`
        if description_arg and description_arg.strip():
            self.handle_mock_record(description_arg.strip())
            return 0

        # Mode B: Interactive two-step toggle via Enter in the terminal
        print("\n[SYSTEM] Starting Record & Upload mode (1st toggle: Record ACTIVE)...", flush=True)
        photo_b64 = self.camera.capture_base64()
        self._persist_capture_locally(photo_b64)
        print(
            "\n[SYSTEM] Recording is active — please describe this place.",
            flush=True,
        )
        self.speaker.speak(MSG_RECORD_ACTIVE)

        mic_started = False
        try:
            self.recorder.start_recording()
            mic_started = True
        except Exception as exc:
            logger.warning("Microphone capture unavailable (%s); you can type the description below.", exc)

        sys.stdout.write(
            "\n>>> Recording is ACTIVE! Speak into the microphone and press ENTER to stop, save & upload\n"
            "    (or type a description and press ENTER): "
        )
        sys.stdout.flush()
        try:
            user_line = sys.stdin.readline().strip()
        except KeyboardInterrupt:
            user_line = ""

        wav_bytes = b""
        if mic_started:
            try:
                wav_bytes = self.recorder.stop_recording()
            except Exception as exc:
                logger.warning("Microphone stop warning: %s", exc)

        transcript = user_line
        if not transcript and wav_bytes:
            try:
                transcript = transcribe_audio_with_gemini(
                    wav_bytes=wav_bytes,
                    api_key=self.config.gemini_api_key,
                    model=self.config.model,
                    captured_b64=photo_b64,
                )
            except Exception as exc:
                logger.warning("Audio transcription failed (%s).", exc)

        if not transcript:
            sys.stdout.write("\n>>> Enter a description for the captured photo: ")
            sys.stdout.flush()
            transcript = sys.stdin.readline().strip() or "Recorded place memory"

        saved_place = self._save_memory_and_reload(photo_b64=photo_b64, note_text=transcript)
        print(
            f"\n[SYSTEM] Recording saved locally ({saved_place.name}): {transcript!r}",
            flush=True,
        )
        self.speaker.speak(MSG_RECORD_SAVED)

        print(
            f"\n[SYSTEM] Uploading photo and description for '{saved_place.name}' to the website...",
            flush=True,
        )
        self.speaker.speak(MSG_RECORD_UPLOADING)
        self._upload_saved_memory(saved_place=saved_place, photo_b64=photo_b64)
        print(
            f"\n[SYSTEM] Done! '{saved_place.name}' saved and uploaded to the Caregiver Website.",
            flush=True,
        )
        return 0

    def _persist_capture_locally(self, captured_b64: str) -> None:
        """Unconditionally save every captured camera frame to disk inside HackRPI2026/memoryaid/."""
        if not captured_b64 or not self.config.has_api_key:
            return
        try:
            raw_bytes = base64.b64decode(captured_b64, validate=True)
            if len(raw_bytes) < 32:
                return
        except Exception:
            return

        project_root = self.config.memories_dir.resolve().parent
        ts = time.strftime("%Y%m%d_%H%M%S")
        targets = [
            project_root / "last_capture.jpg",
            project_root / "captured_photos" / "latest.jpg",
            project_root / "captured_photos" / f"capture_{ts}.jpg",
            project_root / "images" / "last_capture.jpg",
            project_root / "images" / f"capture_{ts}.jpg",
        ]
        saved: List[str] = []
        for target in targets:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw_bytes)
                saved.append(str(target))
            except OSError:
                pass
        if saved:
            logger.info("Saved camera photo locally -> %s", saved[0])

    def _save_memory_and_reload(self, photo_b64: str, note_text: str) -> PlaceMemory:
        saved_place = save_recorded_memory(
            memories_dir=self.config.memories_dir,
            captured_b64=photo_b64,
            note_text=note_text,
            website_url=self.config.website_url,
            upload_to_website=False,
        )
        # Reload places and rebuild the composite contact sheet so the new place is immediately active
        self.places = load_memories(self.config.memories_dir)
        self.contact_sheet_b64 = build_contact_sheet_b64(
            self.places,
            cache_dir=self.config.memories_dir / ".cache",
            captures_dir=self.config.captures_dir,
        )
        return saved_place

    def _upload_saved_memory(self, saved_place: PlaceMemory, photo_b64: str) -> None:
        """Upload a newly saved local memory to the Caregiver Website."""
        try:
            raw_jpeg = base64.b64decode(photo_b64)
            upload_memory_to_website(
                memories_dir=self.config.memories_dir,
                folder_name=saved_place.name,
                jpeg_bytes=raw_jpeg,
                description=saved_place.note,
                website_url=self.config.website_url,
            )
        except Exception as exc:
            logger.warning("Website upload after recording skipped: %s", exc)

    def run_dry_run(self) -> int:
        """Sync website memories, capture one image, build the full request payload, print the summary, and return payload size."""
        try:
            self.sync_website_memories()
        except Exception as exc:
            logger.warning("Dry-run website sync skipped: %s", exc)
        self.camera.start()
        captured_b64 = self.camera.capture_base64()
        payload = build_messages_payload(
            places=self.places,
            captured_b64=captured_b64,
            model=self.config.model,
            contact_sheet_b64=self.contact_sheet_b64,
        )
        size_bytes, summary = summarize_payload(payload, self.places, model=self.config.model)
        print(summary, flush=True)
        return size_bytes

    def close(self) -> None:
        if self.button is not None:
            self.button.close()
        self.camera.close()
