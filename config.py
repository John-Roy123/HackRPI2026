"""Configuration loader and validator for Memory Aid."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("memoryaid")


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a simple KEY=VALUE environment file without third-party dependencies."""
    values: Dict[str, str] = {}
    if not path.is_file():
        return values

    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read env file %s: %s", path, exc)
        return values

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        values[key] = val
    return values


def _parse_bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes", "on")


def _parse_int(val: str, default: int, name: str) -> int:
    try:
        return int(val.strip())
    except (ValueError, TypeError):
        logger.warning("Invalid integer for %s (%r), using default %d", name, val, default)
        return default


@dataclass
class Config:
    """Runtime configuration for Memory Aid."""

    anthropic_api_key: str = ""
    model: str = "claude-haiku-5-5"
    audio_device: str = "default"
    voice: str = "en-us"
    speech_rate: int = 140
    button_pin: int = 17
    memories_dir: Path = Path("memories")
    save_last_capture: bool = False
    last_capture_path: Path = Path("last_capture.jpg")

    @property
    def has_api_key(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.strip())


def load_config(env_file: Optional[Path] = None, project_root: Optional[Path] = None) -> Config:
    """Load configuration from env files and process environment variables.

    Precedence (highest to lowest):
      1. Process environment variables (`os.environ`)
      2. Explicit `env_file` (if provided) or `./memoryaid.env`
      3. `/etc/memoryaid.env`
      4. Built-in defaults
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    merged: Dict[str, str] = {}

    etc_env = Path("/etc/memoryaid.env")
    if etc_env.is_file():
        merged.update(_parse_env_file(etc_env))

    local_env = env_file if env_file is not None else (project_root / "memoryaid.env")
    if local_env.is_file():
        merged.update(_parse_env_file(local_env))

    for key in (
        "ANTHROPIC_API_KEY",
        "MODEL",
        "AUDIO_DEVICE",
        "VOICE",
        "SPEECH_RATE",
        "BUTTON_PIN",
        "MEMORIES_DIR",
        "SAVE_LAST_CAPTURE",
    ):
        if key in os.environ:
            merged[key] = os.environ[key]

    api_key = merged.get("ANTHROPIC_API_KEY", "").strip()
    model = merged.get("MODEL", "").strip() or "claude-haiku-5-5"
    audio_device = merged.get("AUDIO_DEVICE", "").strip() or "default"
    voice = merged.get("VOICE", "").strip() or "en-us"
    speech_rate = _parse_int(merged.get("SPEECH_RATE", "140"), 140, "SPEECH_RATE")
    button_pin = _parse_int(merged.get("BUTTON_PIN", "17"), 17, "BUTTON_PIN")

    raw_memories_dir = merged.get("MEMORIES_DIR", "").strip()
    if raw_memories_dir:
        memories_dir = Path(os.path.expanduser(raw_memories_dir))
        if not memories_dir.is_absolute():
            memories_dir = (project_root / memories_dir).resolve()
    else:
        memories_dir = (project_root / "memories").resolve()

    save_last_capture = _parse_bool(merged.get("SAVE_LAST_CAPTURE", "false"))
    last_capture_path = (project_root / "last_capture.jpg").resolve()

    return Config(
        anthropic_api_key=api_key,
        model=model,
        audio_device=audio_device,
        voice=voice,
        speech_rate=speech_rate,
        button_pin=button_pin,
        memories_dir=memories_dir,
        save_last_capture=save_last_capture,
        last_capture_path=last_capture_path,
    )
