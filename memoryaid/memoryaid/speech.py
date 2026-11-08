"""Fish Audio Text-to-Speech (`https://api.fish.audio/v1/tts`) and Bluetooth / ALSA microphone recording."""

import array
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
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("memoryaid")

MAX_RECORD_SECONDS = 60
MAC_REGEX = re.compile(r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5})")
ARECORD_CARD_REGEX = re.compile(r"^card\s+(\d+):\s*([^\[]+)\[([^\]]+)\],\s*device\s+(\d+):", re.MULTILINE)
FISH_TTS_ENDPOINT = "https://api.fish.audio/v1/tts"
FISH_SAMPLE_RATE = 24000
FISH_TIMEOUT_SECONDS = 20


class AudioRecordError(Exception):
    """Raised when microphone recording fails."""


def _speech_env() -> dict:
    """Build environment variables so even when run via `sudo` or `systemd`, commands reach the non-root user's PipeWire/PulseAudio Bluetooth session."""
    env = os.environ.copy()
    uid = os.getuid()
    sudo_uid = os.environ.get("SUDO_UID")
    if uid == 0 and sudo_uid:
        target_uid = sudo_uid
    elif uid == 0 and os.path.isdir("/run/user/1000"):
        target_uid = "1000"
    else:
        target_uid = str(uid)

    runtime_dir = f"/run/user/{target_uid}"
    if os.path.isdir(runtime_dir):
        env["XDG_RUNTIME_DIR"] = runtime_dir
        if os.path.exists(f"{runtime_dir}/bus"):
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={runtime_dir}/bus"
        if os.path.exists(f"{runtime_dir}/pulse/native"):
            env["PULSE_SERVER"] = f"unix:{runtime_dir}/pulse/native"
    return env


def get_connected_bluetooth_mac(preferred_mac: str = "") -> Optional[str]:
    """Return the MAC address of the currently connected (or first paired) Bluetooth audio device."""
    if preferred_mac and preferred_mac.strip():
        return preferred_mac.strip().upper()
    if shutil.which("bluetoothctl") is None:
        return None

    # 1. Check currently connected Bluetooth devices first (e.g. Beats Solo 3)
    for subcmd in (
        ["bluetoothctl", "devices", "Connected"],
        ["bluetoothctl", "devices", "Paired"],
        ["bluetoothctl", "devices"],
    ):
        try:
            res = subprocess.run(
                subcmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            out = res.stdout or ""
            # Prefer a line mentioning Beats or headphones if multiple devices are paired
            for line in out.splitlines():
                if any(k in line.lower() for k in ("beats", "solo", "headphone", "airpod", "jbl", "speaker")):
                    m = MAC_REGEX.search(line)
                    if m:
                        return m.group(1).upper()
            m = MAC_REGEX.search(out)
            if m:
                return m.group(1).upper()
        except Exception:
            continue
    return None


def ensure_bluetooth_connected(bluetooth_mac: str = "") -> Optional[str]:
    """Ensure the Bluetooth controller is powered on, connected, and routed as the default audio sink."""
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

    target_mac = get_connected_bluetooth_mac(bluetooth_mac)
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
            logger.info("Connecting to Bluetooth headphones/speaker %s...", target_mac)
            subprocess.run(
                ["bluetoothctl", "connect", target_mac],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
    except Exception as exc:
        logger.warning("Bluetooth connect check for %s failed: %s", target_mac, exc)

    # If PipeWire / PulseAudio (pactl) is present, set the bluez sink as the default sink and unmute at 90%
    _route_pulse_bluetooth_sink()
    return target_mac


def _route_pulse_bluetooth_sink() -> Optional[str]:
    """Find any `bluez_output.*` or `bluez_sink.*` in PipeWire/PulseAudio, set it as default sink, and return its name."""
    if shutil.which("pactl") is None:
        return None
    env = _speech_env()
    try:
        res = subprocess.run(
            ["pactl", "list", "short", "sinks"],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
            check=False,
        )
        for line in (res.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and ("bluez" in parts[1].lower() or "bluetooth" in parts[1].lower()):
                sink_name = parts[1]
                subprocess.run(
                    ["pactl", "set-default-sink", sink_name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    timeout=3,
                    check=False,
                )
                subprocess.run(
                    ["pactl", "set-sink-mute", sink_name, "0"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    timeout=3,
                    check=False,
                )
                subprocess.run(
                    ["pactl", "set-sink-volume", sink_name, "90%"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    env=env,
                    timeout=3,
                    check=False,
                )
                return sink_name
    except Exception:
        pass
    return None


def _switch_bluetooth_card_profile(profile_keywords: Tuple[str, ...]) -> Optional[str]:
    """Switch the Bluetooth card profile in PipeWire/PulseAudio (e.g., to `headset-head-unit` for mic or `a2dp-sink` for speaker)."""
    if shutil.which("pactl") is None:
        return None
    env = _speech_env()
    try:
        res = subprocess.run(
            ["pactl", "list", "cards"],
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
            check=False,
        )
        text = res.stdout or ""
        card_blocks = text.split("Card #")
        for block in card_blocks[1:]:
            name_match = re.search(r"Name:\s*(bluez_card\.[^\s]+)", block)
            if not name_match:
                continue
            card_name = name_match.group(1)
            # Find available profiles in this card block
            for kw in profile_keywords:
                prof_match = re.search(rf"^\s*({re.escape(kw)}[^\s:]*):", block, re.MULTILINE)
                if prof_match:
                    prof_name = prof_match.group(1)
                    subprocess.run(
                        ["pactl", "set-card-profile", card_name, prof_name],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        env=env,
                        timeout=5,
                        check=False,
                    )
                    logger.info("Switched Bluetooth card %s to profile %s", card_name, prof_name)
                    return card_name
    except Exception as exc:
        logger.debug("Could not switch Bluetooth card profile: %s", exc)
    return None


def detect_and_configure_electret_mic() -> Optional[str]:
    """Auto-detect a microphone capture card via `arecord -l` or BlueALSA (`bluealsa-aplay -L`)."""
    if shutil.which("arecord") is None:
        return None

    try:
        res = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = res.stdout or ""
    except Exception as exc:
        logger.debug("arecord -l failed: %s", exc)
        return None

    matches = ARECORD_CARD_REGEX.findall(output)
    if not matches:
        return None

    chosen: Tuple[str, str, str, str] = matches[0]
    for m in matches:
        card_id, short_name, long_name, dev_id = m
        combined = f"{short_name} {long_name}".lower()
        if any(k in combined for k in ("usb", "mic", "i2s", "seeed", "wm8960", "adc", "audio")):
            chosen = m
            break

    card_num, short_name, long_name, dev_num = chosen
    plughw = f"plughw:{card_num},{dev_num}"
    logger.info(
        "Detected ALSA microphone capture device: %s (%s - %s)",
        plughw,
        short_name.strip(),
        long_name.strip(),
    )

    if shutil.which("amixer") is not None:
        for ctrl in ("Mic", "Capture", "ADC", "PGA", "Input", "Boost"):
            subprocess.run(
                ["amixer", "-c", card_num, "sset", ctrl, "95%", "unmute", "cap"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            subprocess.run(
                ["amixer", "-c", card_num, "sset", ctrl, "95%", "unmute"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    return plughw


def normalize_electret_wav_bytes(wav_bytes: bytes) -> bytes:
    """Remove DC offset and normalize quiet speech on 16-bit mono WAV recordings."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            raw_frames = wf.readframes(nframes)

        if sampwidth != 2 or len(raw_frames) < 4:
            return wav_bytes

        samples = array.array("h")
        samples.frombytes(raw_frames)
        if not samples:
            return wav_bytes

        dc_offset = int(sum(samples) // len(samples))
        if dc_offset != 0:
            for i in range(len(samples)):
                val = samples[i] - dc_offset
                if val > 32767:
                    val = 32767
                elif val < -32768:
                    val = -32768
                samples[i] = val

        peak = max(abs(s) for s in samples) or 1
        target_peak = 24000
        if 150 < peak < target_peak:
            gain = min(12.0, float(target_peak) / float(peak))
            if gain > 1.25:
                for i in range(len(samples)):
                    val = int(samples[i] * gain)
                    if val > 32767:
                        val = 32767
                    elif val < -32768:
                        val = -32768
                    samples[i] = val
                logger.info(
                    "Applied %.1fx software gain + DC offset removal to mic recording (peak=%d).",
                    gain,
                    peak,
                )

        out_buf = io.BytesIO()
        with wave.open(out_buf, "wb") as out_wf:
            out_wf.setnchannels(nchannels)
            out_wf.setsampwidth(sampwidth)
            out_wf.setframerate(framerate)
            out_wf.writeframes(samples.tobytes())
        return out_buf.getvalue()
    except Exception as exc:
        logger.debug("Audio normalization skipped: %s", exc)
        return wav_bytes


def synthesize_fish_audio_wav(
    text: str,
    api_key: str,
    reference_id: str = "",
    fish_model: str = "",
    timeout: float = FISH_TIMEOUT_SECONDS,
) -> bytes:
    """Call Fish Audio TTS REST API (`https://api.fish.audio/v1/tts`) with `format="wav"` and return WAV bytes."""
    headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    if fish_model and fish_model.strip():
        headers["model"] = fish_model.strip()

    payload: Dict[str, Any] = {
        "text": text,
        "format": "wav",
        "sample_rate": FISH_SAMPLE_RATE,
        "latency": "balanced",
    }
    if reference_id and reference_id.strip():
        payload["reference_id"] = reference_id.strip()

    response = requests.post(
        FISH_TTS_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        err_preview = response.text[:200] if response.text else ""
        raise RuntimeError(f"Fish Audio HTTP {response.status_code}: {err_preview}")

    wav_bytes = response.content
    if not wav_bytes or len(wav_bytes) <= 44:
        raise RuntimeError("Fish Audio returned empty WAV audio data.")

    return wav_bytes


class AudioRecorder:
    """Records 16 kHz mono WAV audio from the Beats Solo 3 Bluetooth mic or ALSA microphone while Button 1 is held."""

    def __init__(
        self,
        record_device: str = "auto",
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
        self._has_parec = shutil.which("parec") is not None
        self._has_pw_record = shutil.which("pw-record") is not None
        self._resolved_device: Optional[str] = None
        self._switched_bt_profile = False

    def _build_record_command(self, target_wav: Path) -> List[str]:
        """Choose the best recording command for Bluetooth headset mic (Beats Solo 3) or ALSA mic."""
        if self.record_device and self.record_device not in ("auto", "default"):
            return [
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
                str(target_wav),
            ]

        # 1. If a hardware ALSA mic is plugged in, use it directly
        detected_alsa = detect_and_configure_electret_mic()
        if detected_alsa:
            return [
                "arecord",
                "-q",
                "-D",
                detected_alsa,
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
                str(target_wav),
            ]

        # 2. Switch Bluetooth headphones (e.g., Beats Solo 3) into HFP/HSP headset mic profile if PipeWire/PulseAudio is active
        card = _switch_bluetooth_card_profile(("headset-head-unit", "handsfree_head_unit", "hfp_hf", "hsp_hs"))
        if card:
            self._switched_bt_profile = True
            time.sleep(0.3)

        if self._has_parec:
            return [
                "parec",
                "--file-format=wav",
                "--format=s16le",
                "--rate=16000",
                "--channels=1",
                str(target_wav),
            ]
        if self._has_pw_record:
            return [
                "pw-record",
                "--format=s16",
                "--rate=16000",
                "--channels=1",
                str(target_wav),
            ]

        # 3. BlueALSA SCO Bluetooth headset microphone fallback
        bt_mac = get_connected_bluetooth_mac(self.bluetooth_mac)
        if bt_mac:
            return [
                "arecord",
                "-q",
                "-D",
                f"bluealsa:DEV={bt_mac},PROFILE=sco",
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
                str(target_wav),
            ]

        return [
            "arecord",
            "-q",
            "-D",
            "default",
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
            str(target_wav),
        ]

    def start_recording(self) -> None:
        """Start background microphone capture to a temporary WAV file."""
        with self._lock:
            self._cleanup_unlocked()
            if self.mock or (not self._has_arecord and not self._has_parec and not self._has_pw_record):
                logger.info("AudioRecorder (mock): started listening.")
                return

            fd, tmp_name = tempfile.mkstemp(prefix="memoryaid_rec_", suffix=".wav")
            os.close(fd)
            self._temp_path = Path(tmp_name)

            cmd = self._build_record_command(self._temp_path)
            try:
                self._proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=_speech_env(),
                )
                logger.info("Started recording microphone via %s", " ".join(cmd[:4]))
            except Exception as exc:
                self._cleanup_unlocked()
                raise AudioRecordError(f"Failed to start recording: {exc}") from exc

    def stop_recording(self) -> bytes:
        """Stop recording, restore Bluetooth A2DP stereo playback profile if switched, and return WAV bytes."""
        with self._lock:
            if self.mock or (not self._has_arecord and not self._has_parec and not self._has_pw_record):
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

                if self._switched_bt_profile:
                    self._switched_bt_profile = False
                    _switch_bluetooth_card_profile(("a2dp-sink", "a2dp_sink"))
                    _route_pulse_bluetooth_sink()

                if not temp_path.is_file():
                    err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
                    raise AudioRecordError(f"Microphone produced no file ({err_msg})")

                wav_bytes = temp_path.read_bytes()
                if len(wav_bytes) <= 44:
                    err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
                    raise AudioRecordError(f"Recorded audio is empty ({len(wav_bytes)} bytes). {err_msg}")

                normalized_wav = normalize_electret_wav_bytes(wav_bytes)
                logger.info(
                    "Recorded %d bytes (%.1f KB) of WAV audio.",
                    len(normalized_wav),
                    len(normalized_wav) / 1024.0,
                )
                return normalized_wav
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
    """Thread-safe Fish Audio TTS speaker output supporting Bluetooth headphones (Beats Solo 3) and ALSA/macOS playback."""

    def __init__(
        self,
        audio_device: str = "default",
        bluetooth_mac: str = "",
        fish_api_key: str = "",
        fish_reference_id: str = "",
        fish_model: str = "",
        voice: str = "en-us",
        speech_rate: int = 140,
        mock: bool = False,
    ) -> None:
        self.audio_device = audio_device
        self.bluetooth_mac = bluetooth_mac.strip().upper()
        self.fish_api_key = fish_api_key.strip()
        self.fish_reference_id = fish_reference_id.strip()
        self.fish_model = fish_model.strip()
        self.voice = voice
        self.speech_rate = str(speech_rate)
        self.mock = mock
        self._lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None
        self._wav_cache: Dict[str, bytes] = {}

        self._has_espeak = shutil.which("espeak-ng") is not None
        self._has_aplay = shutil.which("aplay") is not None
        self._has_paplay = shutil.which("paplay") is not None
        self._has_pw_play = shutil.which("pw-play") is not None
        self._has_afplay = shutil.which("afplay") is not None
        self._has_say = shutil.which("say") is not None

    def connect_bluetooth(self) -> None:
        """Ensure the Bluetooth headphones/speaker are connected at startup (no-op in mock mode)."""
        if self.mock:
            return
        detected_mac = ensure_bluetooth_connected(self.bluetooth_mac)
        if detected_mac:
            self.bluetooth_mac = detected_mac
            logger.info("Active Bluetooth audio device MAC: %s", detected_mac)

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
                if self.fish_api_key:
                    played = self._speak_with_fish_audio(clean_text)
                else:
                    logger.warning(
                        "FISH_API_KEY is not set; falling back to local TTS (add FISH_API_KEY to /etc/memoryaid.env)."
                    )

                if not played:
                    if self._has_espeak:
                        self._speak_with_local_espeak(clean_text)
                    elif self._has_say:
                        self._run_macos_say(clean_text)
                    else:
                        logger.warning(
                            "No TTS engine available! Set FISH_API_KEY or install espeak-ng."
                        )
            except Exception as exc:
                logger.error("Speech output failed: %s", exc)

            elapsed = time.monotonic() - t0
            logger.info("Spoke (%d chars in %.2fs): %r", len(clean_text), elapsed, clean_text)
            return elapsed

    def _speak_with_fish_audio(self, text: str) -> bool:
        """Synthesize speech with Fish Audio (using cache for short common prompts) and play the WAV audio."""
        try:
            if len(text) <= 80 and text in self._wav_cache:
                wav_bytes = self._wav_cache[text]
            else:
                wav_bytes = synthesize_fish_audio_wav(
                    text=text,
                    api_key=self.fish_api_key,
                    reference_id=self.fish_reference_id,
                    fish_model=self.fish_model,
                )
                if len(text) <= 80 and len(self._wav_cache) < 16:
                    self._wav_cache[text] = wav_bytes

            return self._play_wav_bytes(wav_bytes)
        except Exception as exc:
            logger.error("Fish Audio TTS failed (%s); falling back to local TTS.", exc)
            return False

    def _play_wav_bytes(self, wav_bytes: bytes) -> bool:
        """Play in-memory WAV bytes specifically targeting Bluetooth headphones first, then standard audio sinks."""
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

        # Write to a temporary WAV file because `pw-play`, `paplay`, and `bluealsa` handle seekable files
        # much more reliably than stdin pipes (and prevents HDMI from swallowing stdin).
        fd, tmp_wav = tempfile.mkstemp(prefix="memoryaid_tts_", suffix=".wav")
        os.close(fd)
        try:
            Path(tmp_wav).write_bytes(wav_bytes)
            env = _speech_env()

            # Ensure Bluetooth MAC is known and Pulse/PipeWire routes to the Bluetooth sink if active
            bt_mac = self.bluetooth_mac or (get_connected_bluetooth_mac() or "")
            if bt_mac and not self.bluetooth_mac:
                self.bluetooth_mac = bt_mac
            bt_sink = _route_pulse_bluetooth_sink()

            # Build prioritized list of commands that explicitly target Bluetooth headphones BEFORE generic ALSA default (which is often HDMI!)
            players: List[List[str]] = []

            # 1. Explicit non-default AUDIO_DEVICE if user configured one
            if self.audio_device and self.audio_device != "default" and self._has_aplay:
                players.append(["aplay", "-q", "-D", self.audio_device, tmp_wav])

            # 2. PipeWire / PulseAudio targeting the discovered `bluez_output.*` sink directly
            if bt_sink and self._has_paplay:
                players.append(["paplay", f"--device={bt_sink}", tmp_wav])
            if bt_sink and self._has_pw_play:
                players.append(["pw-play", f"--target={bt_sink}", tmp_wav])

            # 3. BlueALSA targeting the Bluetooth headphones' MAC address directly (A2DP then SCO)
            if self._has_aplay and bt_mac:
                players.append(["aplay", "-q", "-D", f"bluealsa:DEV={bt_mac},PROFILE=a2dp", tmp_wav])
                players.append(["aplay", "-q", "-D", f"bluealsa:DEV={bt_mac},PROFILE=sco", tmp_wav])
            if self._has_aplay:
                players.append(["aplay", "-q", "-D", "bluealsa", tmp_wav])

            # 4. General PipeWire / PulseAudio / ALSA default fallbacks
            if self._has_paplay:
                players.append(["paplay", tmp_wav])
            if self._has_pw_play:
                players.append(["pw-play", tmp_wav])
            if self._has_aplay:
                players.append(["aplay", "-q", "-D", "pulse", tmp_wav])
                players.append(["aplay", "-q", "-D", "default", tmp_wav])

            last_err = ""
            for cmd in players:
                try:
                    res = subprocess.run(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        env=env,
                        timeout=30,
                        check=False,
                    )
                    if res.returncode == 0:
                        logger.info("Played audio via: %s", " ".join(cmd[:-1]))
                        return True
                    last_err = (res.stderr or b"").decode("utf-8", errors="replace").strip()
                except Exception as exc:
                    last_err = str(exc)

            logger.warning("All Bluetooth/ALSA WAV players failed (last error: %s)", last_err)
            return False
        finally:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass

    def _speak_with_local_espeak(self, text: str) -> None:
        """Fallback local TTS using espeak-ng if Fish Audio is unconfigured or unreachable."""
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
