"""ElevenLabs Text-to-Speech (with local fallback) and ALSA microphone recording (`arecord`)."""

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Dict, List, Optional

import requests

from .config import DEFAULT_ELEVENLABS_MODEL_ID, DEFAULT_ELEVENLABS_VOICE_ID

logger = logging.getLogger("memoryaid")

MAX_RECORD_SECONDS = 60
MAC_REGEX = re.compile(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})")
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1/text-to-speech"
ELEVENLABS_PCM_SAMPLE_RATE = 22050
ELEVENLABS_TIMEOUT_SECONDS = 20


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


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = ELEVENLABS_PCM_SAMPLE_RATE) -> bytes:
    """Wrap raw 16-bit little-endian mono PCM bytes into a standard WAV container in memory."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit (2 bytes)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def synthesize_elevenlabs_wav(
    text: str,
    api_key: str,
    voice_id: str = DEFAULT_ELEVENLABS_VOICE_ID,
    model_id: str = DEFAULT_ELEVENLABS_MODEL_ID,
    timeout: float = ELEVENLABS_TIMEOUT_SECONDS,
) -> bytes:
    """Call ElevenLabs Text-to-Speech HTTP API (`output_format=pcm_22050`) and return WAV bytes."""
    url = f"{ELEVENLABS_API_BASE}/{voice_id.strip()}"
    headers = {
        "xi-api-key": api_key.strip(),
        "Content-Type": "application/json",
    }
    params = {
        "output_format": "pcm_22050",
    }
    payload = {
        "text": text,
        "model_id": model_id.strip() or DEFAULT_ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.65,
            "similarity_boost": 0.75,
        },
    }

    response = requests.post(
        url,
        headers=headers,
        params=params,
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        err_preview = response.text[:200] if response.text else ""
        raise RuntimeError(f"ElevenLabs HTTP {response.status_code}: {err_preview}")

    pcm_data = response.content
    if not pcm_data:
        raise RuntimeError("ElevenLabs returned empty audio data.")

    return pcm16_to_wav_bytes(pcm_data, sample_rate=ELEVENLABS_PCM_SAMPLE_RATE)


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
    """Thread-safe ElevenLabs TTS speaker output supporting Bluetooth and ALSA/macOS playback."""

    def __init__(
        self,
        audio_device: str = "default",
        bluetooth_mac: str = "",
        elevenlabs_api_key: str = "",
        elevenlabs_voice_id: str = DEFAULT_ELEVENLABS_VOICE_ID,
        elevenlabs_model_id: str = DEFAULT_ELEVENLABS_MODEL_ID,
        voice: str = "en-us",
        speech_rate: int = 140,
        mock: bool = False,
    ) -> None:
        self.audio_device = audio_device
        self.bluetooth_mac = bluetooth_mac.strip().upper()
        self.elevenlabs_api_key = elevenlabs_api_key.strip()
        self.elevenlabs_voice_id = elevenlabs_voice_id.strip() or DEFAULT_ELEVENLABS_VOICE_ID
        self.elevenlabs_model_id = elevenlabs_model_id.strip() or DEFAULT_ELEVENLABS_MODEL_ID
        self.voice = voice
        self.speech_rate = str(speech_rate)
        self.mock = mock
        self._lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None
        # In-memory WAV cache for short fixed phrases (e.g. "Let me take a look.") so repeated presses have zero TTS latency
        self._wav_cache: Dict[str, bytes] = {}

        self._has_espeak = shutil.which("espeak-ng") is not None
        self._has_aplay = shutil.which("aplay") is not None
        self._has_paplay = shutil.which("paplay") is not None
        self._has_pw_play = shutil.which("pw-play") is not None
        self._has_afplay = shutil.which("afplay") is not None
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
            print(f"SPEAK: {clean_text}", flush=True)
            try:
                played = False
                if self.elevenlabs_api_key:
                    played = self._speak_with_elevenlabs(clean_text)
                else:
                    logger.warning(
                        "ELEVENLABS_API_KEY is not set; falling back to local TTS (add ELEVENLABS_API_KEY to /etc/memoryaid.env)."
                    )

                if not played:
                    if self._has_espeak:
                        self._speak_with_local_espeak(clean_text)
                    elif self._has_say:
                        self._run_macos_say(clean_text)
                    else:
                        logger.warning(
                            "No TTS engine available! Set ELEVENLABS_API_KEY or install espeak-ng."
                        )
            except Exception as exc:
                logger.error("Speech output failed: %s", exc)

            elapsed = time.monotonic() - t0
            logger.info("Spoke (%d chars in %.2fs): %r", len(clean_text), elapsed, clean_text)
            return elapsed

    def _speak_with_elevenlabs(self, text: str) -> bool:
        """Synthesize speech with ElevenLabs (using cache for short common prompts) and play the WAV audio."""
        try:
            if len(text) <= 80 and text in self._wav_cache:
                wav_bytes = self._wav_cache[text]
            else:
                wav_bytes = synthesize_elevenlabs_wav(
                    text=text,
                    api_key=self.elevenlabs_api_key,
                    voice_id=self.elevenlabs_voice_id,
                    model_id=self.elevenlabs_model_id,
                )
                if len(text) <= 80 and len(self._wav_cache) < 16:
                    self._wav_cache[text] = wav_bytes

            return self._play_wav_bytes(wav_bytes)
        except Exception as exc:
            logger.error("ElevenLabs TTS failed (%s); falling back to local TTS.", exc)
            return False

    def _play_wav_bytes(self, wav_bytes: bytes) -> bool:
        """Play in-memory WAV bytes through the first working audio output on Pi (Bluetooth/ALSA) or macOS."""
        # macOS laptop playback via afplay
        if self._has_afplay and not self._has_aplay and not self._has_paplay and not self._has_pw_play:
            fd, tmp_wav = tempfile.mkstemp(prefix="memoryaid_tts_", suffix=".wav")
            os.close(fd)
            try:
                Path(tmp_wav).write_bytes(wav_bytes)
                res = subprocess.run(
                    ["afplay", tmp_wav],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
                return res.returncode == 0
            finally:
                try:
                    os.unlink(tmp_wav)
                except OSError:
                    pass

        # Raspberry Pi / Linux playback order:
        # 1. Explicit non-default AUDIO_DEVICE via aplay
        # 2. paplay / pw-play (PipeWire / PulseAudio Bluetooth speaker on Raspberry Pi OS Bookworm)
        # 3. aplay default / bare aplay / bluealsa
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
                    return True
                last_err = (res.stderr or b"").decode("utf-8", errors="replace").strip()
            except Exception as exc:
                last_err = str(exc)

        logger.warning("WAV audio players returned non-zero (last error: %s)", last_err)
        return False

    def _speak_with_local_espeak(self, text: str) -> None:
        """Fallback local TTS using espeak-ng if ElevenLabs is unconfigured or unreachable."""
        wav_bytes = self._synthesize_espeak_wav(text)
        if wav_bytes and self._play_wav_bytes(wav_bytes):
            return
        self._run_espeak_direct(text)

    def _synthesize_espeak_wav(self, text: str) -> bytes:
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
