"""Bluetooth / ALSA speaker playback (espeak-ng -> paplay/pw-play/aplay) and microphone recording (arecord)."""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("memoryaid")

MAX_RECORD_SECONDS = 60
MAC_REGEX = re.compile(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})")


class AudioRecordError(Exception):
    """Raised when microphone recording fails."""


def _speech_env() -> dict:
    """Build environment variables so systemd services can reach the user's PipeWire/PulseAudio Bluetooth session."""
    env = os.environ.copy()
    uid = os.getuid()
    runtime_dir = f"/run/user/{uid}"
    if "XDG_RUNTIME_DIR" not in env and os.path.isdir(runtime_dir):
        env["XDG_RUNTIME_DIR"] = runtime_dir
    if "DBUS_SESSION_BUS_ADDRESS" not in env and os.path.exists(f"{runtime_dir}/bus"):
        env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime_dir}/bus"
    return env


def ensure_bluetooth_connected(bluetooth_mac: str = "") -> Optional[str]:
    """Ensure the Bluetooth controller is powered on and the target (or first paired) speaker is connected.

    Returns the MAC address of the connected/targeted Bluetooth device if found, otherwise None.
    """
    if shutil.which("bluetoothctl") is None:
        return bluetooth_mac or None

    try:
        subprocess.run(
            ["bluetoothctl", "power", "on"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        pass

    target_mac = bluetooth_mac.strip().upper()
    if not target_mac:
        for subcmd in (["bluetoothctl", "devices", "Paired"], ["bluetoothctl", "devices"]):
            try:
                res = subprocess.run(
                    subcmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                match = MAC_REGEX.search(res.stdout or "")
                if match:
                    target_mac = match.group(1).upper()
                    break
            except Exception:
                continue

    if not target_mac:
        return None

    try:
        info_res = subprocess.run(
            ["bluetoothctl", "info", target_mac],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "Connected: yes" not in (info_res.stdout or ""):
            logger.info("Connecting to Bluetooth speaker %s...", target_mac)
            subprocess.run(
                ["bluetoothctl", "connect", target_mac],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Bluetooth connect check for %s failed: %s", target_mac, exc)

    return target_mac


class AudioRecorder:
    """Records 16 kHz mono WAV audio via `arecord` while the record button is held."""

    def __init__(
        self,
        record_device: str = "default",
        bluetooth_mac: str = "",
        mock: bool = False,
    ) -> None:
        self.record_device = record_device
        self.bluetooth_mac = bluetooth_mac
        self.mock = mock
        self._lock = threading.Lock()
        self._proc: Optional[subprocess.Popen] = None
        self._temp_path: Optional[Path] = None
        self._has_arecord = shutil.which("arecord") is not None

    def start_recording(self) -> None:
        """Start background `arecord` capture to a temporary WAV file."""
        with self._lock:
            self._cleanup_unlocked()
            if self.mock or not self._has_arecord:
                logger.info("AudioRecorder (mock): started listening.")
                return

            fd, tmp_name = tempfile.mkstemp(prefix="memoryaid_rec_", suffix=".wav")
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
                    env=_speech_env(),
                )
                logger.info(
                    "Started recording audio from ALSA device %r to %s",
                    self.record_device,
                    self._temp_path,
                )
            except Exception as exc:
                self._cleanup_unlocked()
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

    def _cleanup_unlocked(self) -> None:
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
    """Thread-safe speaker output supporting Bluetooth (PipeWire/PulseAudio/BlueALSA) and ALSA playback."""

    def __init__(
        self,
        audio_device: str = "default",
        bluetooth_mac: str = "",
        voice: str = "en-us",
        speech_rate: int = 140,
        mock: bool = False,
    ) -> None:
        self.audio_device = audio_device
        self.bluetooth_mac = bluetooth_mac.strip().upper()
        self.voice = voice
        self.speech_rate = str(speech_rate)
        self.mock = mock
        self._lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None

        self._has_espeak = shutil.which("espeak-ng") is not None
        self._has_aplay = shutil.which("aplay") is not None
        self._has_paplay = shutil.which("paplay") is not None
        self._has_pw_play = shutil.which("pw-play") is not None
        self._has_say = shutil.which("say") is not None

    def connect_bluetooth(self) -> None:
        """Ensure the Bluetooth speaker is connected at startup (no-op in mock mode)."""
        if self.mock:
            return
        detected_mac = ensure_bluetooth_connected(self.bluetooth_mac)
        if detected_mac and not self.bluetooth_mac:
            self.bluetooth_mac = detected_mac
            logger.info("Detected paired Bluetooth speaker: %s", detected_mac)

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
            # Always print SPEAK line so terminal users and journalctl logs see what is being spoken
            print(f"SPEAK: {clean_text}", flush=True)
            try:
                if self._has_espeak:
                    self._speak_with_espeak_fallbacks(clean_text)
                elif self._has_say:
                    self._run_macos_say(clean_text)
                else:
                    logger.warning(
                        "espeak-ng is not installed! Run: sudo apt install -y espeak-ng alsa-utils"
                    )
            except Exception as exc:
                logger.error("Speech output failed: %s", exc)

            elapsed = time.monotonic() - t0
            logger.info("Spoke (%d chars in %.2fs): %r", len(clean_text), elapsed, clean_text)
            return elapsed

    def _speak_with_espeak_fallbacks(self, text: str) -> None:
        """Synthesize speech to WAV with espeak-ng, then play it through the first working audio output."""
        wav_bytes = self._synthesize_wav(text)
        if not wav_bytes:
            # Fallback to direct espeak-ng invocation if --stdout failed
            self._run_espeak_direct(text)
            return

        # Build ordered list of playback commands to try so Bluetooth works on:
        # 1. PipeWire / PulseAudio (Bookworm default for Bluetooth speakers)
        # 2. Configured ALSA device (`default` or `bluealsa`)
        # 3. Explicit `bluealsa:DEV=<MAC>,PROFILE=a2dp` or `bluealsa`
        # 4. Direct `espeak-ng` audio output
        players: List[List[str]] = []

        if self.audio_device and self.audio_device != "default" and self._has_aplay:
            players.append(["aplay", "-q", "-D", self.audio_device])

        if self._has_paplay:
            players.append(["paplay"])
        if self._has_pw_play:
            players.append(["pw-play", "-"])

        if self._has_aplay:
            players.append(["aplay", "-q", "-D", "default"])
            players.append(["aplay", "-q"])
            if self.bluetooth_mac:
                players.append(["aplay", "-q", "-D", f"bluealsa:DEV={self.bluetooth_mac},PROFILE=a2dp"])
            players.append(["aplay", "-q", "-D", "bluealsa"])

        env = _speech_env()
        last_err = ""
        for cmd in players:
            try:
                res = subprocess.run(
                    cmd,
                    input=wav_bytes,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                    timeout=30,
                    check=False,
                )
                if res.returncode == 0:
                    return
                last_err = (res.stderr or b"").decode("utf-8", errors="replace").strip()
            except Exception as exc:
                last_err = str(exc)

        logger.warning(
            "WAV players returned non-zero (last error: %s); falling back to direct espeak-ng.",
            last_err,
        )
        self._run_espeak_direct(text)

    def _synthesize_wav(self, text: str) -> bytes:
        """Run `espeak-ng --stdout` and return the synthesized WAV audio bytes."""
        espeak_cmd = [
            "espeak-ng",
            "-v",
            self.voice,
            "-s",
            self.speech_rate,
            "--stdout",
            text,
        ]
        res = subprocess.run(
            espeak_cmd,
            capture_output=True,
            timeout=15,
            check=False,
        )
        if res.returncode != 0 or len(res.stdout) <= 44:
            err = (res.stderr or b"").decode("utf-8", errors="replace").strip()
            logger.warning("espeak-ng --stdout failed (code %d): %s", res.returncode, err)
            return b""
        return res.stdout

    def _run_espeak_direct(self, text: str) -> None:
        subprocess.run(
            ["espeak-ng", "-v", self.voice, "-s", self.speech_rate, text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=_speech_env(),
        )

    def _run_macos_say(self, text: str) -> None:
        subprocess.run(
            ["say", "-r", self.speech_rate, text],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
