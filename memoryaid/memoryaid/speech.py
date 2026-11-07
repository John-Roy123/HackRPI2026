"""Speech synthesis output (espeak-ng -> aplay) and microphone recording (arecord)."""

import logging
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("memoryaid")

MAX_RECORD_SECONDS = 60


class AudioRecordError(Exception):
    """Raised when microphone recording fails."""


class AudioRecorder:
    """Records 16 kHz mono WAV audio via `arecord` while the record button is held."""

    def __init__(self, record_device: str = "default", mock: bool = False) -> None:
        self.record_device = record_device
        self.mock = mock
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._temp_path: Optional[Path] = None
        self._has_arecord = shutil.which("arecord") is not None

    def start_recording(self) -> None:
        """Start background `arecord` capture to a temporary WAV file."""
        with self._lock:
            self._cleanup_Unlocked()
            if self.mock or not self._has_arecord:
                logger.info("AudioRecorder (mock): started listening.")
                return

            fd, tmp_name = tempfile.mkstemp(prefix="memoryaid_rec_", suffix=".wav")
            import os

            os.close(fd)
            self._temp_path = Path(tmp_name)

            cmd = [
                "arecord",
                "-q",
                "-D",
                self.record_device,
                "-f",
                "S16_LE",
                "-r",
                "16000",
                "-c",
                "1",
                "-t",
                "wav",
                "-d",
                str(MAX_RECORD_SECONDS),
                str(self._temp_path),
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                logger.info(
                    "Started recording audio from ALSA device %r to %s",
                    self.record_device,
                    self._temp_path,
                )
            except Exception as exc:
                self._cleanup_Unlocked()
                raise AudioRecordError(f"Failed to start arecord: {exc}") from exc

    def stop_recording(self) -> bytes:
        """Stop `arecord`, read the recorded WAV bytes, and delete the temporary file."""
        with self._lock:
            if self.mock or not self._has_arecord:
                logger.info("AudioRecorder (mock): stopped listening.")
                return b""

            proc = self._proc
            temp_path = self._temp_path
            self._proc = None
            self._temp_path = None

            if proc is None or temp_path is None:
                raise AudioRecordError("Recording was not active.")

            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)

                stderr_bytes = proc.stderr.read() if proc.stderr else b""
                if proc.stderr:
                    proc.stderr.close()

                if not temp_path.is_file():
                    err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
                    raise AudioRecordError(f"arecord produced no file ({err_msg})")

                wav_bytes = temp_path.read_bytes()
                if len(wav_bytes) <= 44:
                    err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
                    raise AudioRecordError(f"Recorded audio is empty ({len(wav_bytes)} bytes). {err_msg}")

                logger.info("Recorded %d bytes (%.1f KB) of WAV audio.", len(wav_bytes), len(wav_bytes) / 1024.0)
                return wav_bytes
            finally:
                try:
                    if temp_path.exists():
                        temp_path.unlink()
                except OSError:
                    pass

    def _cleanup_Unlocked(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
            except Exception:
                pass
            self._proc = None
        if self._temp_path is not None:
            try:
                if self._temp_path.exists():
                    self._temp_path.unlink()
            except OSError:
                pass
            self._temp_path = None


class Speaker:
    """Thread-safe speech output supporting synchronous and background playback."""

    def __init__(
        self,
        audio_device: str = "default",
        voice: str = "en-us",
        speech_rate: int = 140,
        mock: bool = False,
    ) -> None:
        self.audio_device = audio_device
        self.voice = voice
        self.speech_rate = str(speech_rate)
        self.mock = mock
        self._lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None

        self._has_espeak = shutil.which("espeak-ng") is not None
        self._has_aplay = shutil.which("aplay") is not None
        self._has_say = shutil.which("say") is not None

    def speak(self, text: str) -> float:
        """Wait for any in-progress speech to finish, then speak `text` synchronously.

        Returns the elapsed speech duration in seconds.
        """
        self.wait()
        return self._speak_blocking(text)

    def speak_async(self, text: str) -> None:
        """Start speaking `text` in a background thread so capture/upload can proceed in parallel."""
        self.wait()
        thread = threading.Thread(
            target=self._speak_blocking,
            args=(text,),
            daemon=True,
            name="memoryaid-speech-bg",
        )
        self._bg_thread = thread
        thread.start()

    def wait(self) -> None:
        """Wait for any background speech started via `speak_async()` to complete."""
        thread = self._bg_thread
        if thread is not None and thread.is_alive():
            thread.join()
        self._bg_thread = None

    def _speak_blocking(self, text: str) -> float:
        clean_text = text.strip()
        if not clean_text:
            return 0.0

        with self._lock:
            t0 = time.monotonic()
            try:
                if not self.mock or (self._has_espeak and self._has_aplay):
                    if self._has_espeak and self._has_aplay:
                        self._run_espeak_aplay(clean_text)
                    elif self.mock and self._has_espeak:
                        self._run_espeak_direct(clean_text)
                    elif self.mock and self._has_say:
                        self._run_macos_say(clean_text)
                    else:
                        print(f"SPEAK: {clean_text}", flush=True)
                elif self._has_espeak:
                    self._run_espeak_direct(clean_text)
                elif self._has_say:
                    self._run_macos_say(clean_text)
                else:
                    print(f"SPEAK: {clean_text}", flush=True)
            except Exception as exc:
                logger.error("Speech output failed: %s", exc)
                print(f"SPEAK: {clean_text}", flush=True)

            elapsed = time.monotonic() - t0
            logger.info("Spoke (%d chars in %.2fs): %r", len(clean_text), elapsed, clean_text)
            return elapsed

    def _run_espeak_aplay(self, text: str) -> None:
        """Pipe `espeak-ng --stdout` into `aplay` using argument lists (no shell interpolation)."""
        espeak_cmd = [
            "espeak-ng",
            "-v",
            self.voice,
            "-s",
            self.speech_rate,
            "--stdout",
            text,
        ]
        aplay_cmd = [
            "aplay",
            "-q",
            "-D",
            self.audio_device,
        ]

        espeak_proc = subprocess.Popen(
            espeak_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert espeak_proc.stdout is not None
        aplay_proc = subprocess.Popen(
            aplay_cmd,
            stdin=espeak_proc.stdout,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Allow espeak_proc to receive SIGPIPE if aplay exits early
        espeak_proc.stdout.close()

        _, aplay_err = aplay_proc.communicate()
        espeak_err = espeak_proc.stderr.read() if espeak_proc.stderr else b""
        if espeak_proc.stderr:
            espeak_proc.stderr.close()
        espeak_proc.wait()

        if espeak_proc.returncode != 0:
            logger.warning(
                "espeak-ng exited with code %d: %s",
                espeak_proc.returncode,
                espeak_err.decode("utf-8", errors="replace").strip(),
            )
        if aplay_proc.returncode != 0:
            logger.warning(
                "aplay exited with code %d: %s",
                aplay_proc.returncode,
                aplay_err.decode("utf-8", errors="replace").strip(),
            )

    def _run_espeak_direct(self, text: str) -> None:
        print(f"SPEAK: {text}", flush=True)
        subprocess.run(
            ["espeak-ng", "-v", self.voice, "-s", self.speech_rate, text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _run_macos_say(self, text: str) -> None:
        print(f"SPEAK: {text}", flush=True)
        subprocess.run(
            ["say", "-r", self.speech_rate, text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
