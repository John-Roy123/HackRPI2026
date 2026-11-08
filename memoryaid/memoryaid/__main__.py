"""CLI entry point for `python3 -m memoryaid`."""

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .app import MemoryAidApp
from .button import GpioButtons, KeyboardButton
from .camera import MockCamera, PiCamera
from .config import load_config
from .memories import load_memories, sync_remote_website_memories
from .speech import AudioRecorder, Speaker


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="memoryaid",
        description="Memory Aid: A screenless two-button camera device for people living with dementia.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in laptop mock mode (no picamera2 or gpiozero imports; Enter = Where am I, 'r' = Record Memory).",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a test photo or folder of photos (required with --mock).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the full API request, print a payload summary, and exit without calling the API.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a custom .env configuration file.",
    )
    parser.add_argument(
        "--website-url",
        type=str,
        default=None,
        help="Override WEBSITE_URL for the external HackRPI2026 Caregiver Website (e.g. http://192.168.1.50:5173).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    logger = logging.getLogger("memoryaid")
    args = parse_args(argv)

    if args.mock and args.image is None:
        logger.error("--image PATH is required when running with --mock.")
        return 2

    config = load_config(env_file=args.env_file)
    if args.website_url is not None:
        config.website_url = args.website_url.strip().rstrip("/")
    logger.info(
        "Loaded config: model=%s, website=%s, audio=%s, record=%s, identify_pin=%d, record_pin=%d, memories_dir=%s, mock=%s",
        config.model,
        config.website_url,
        config.audio_device,
        config.record_device,
        config.button_pin,
        config.record_button_pin,
        config.memories_dir,
        args.mock,
    )

    try:
        config.captures_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass

    try:
        sync_remote_website_memories(
            memories_dir=config.memories_dir,
            website_url=config.website_url,
        )
    except Exception as exc:
        logger.warning("Startup website sync skipped: %s", exc)

    places = load_memories(config.memories_dir)

    if args.mock:
        camera = MockCamera(
            image_path=args.image,
            save_last_capture=config.save_last_capture,
            last_capture_path=config.last_capture_path,
            captures_dir=config.captures_dir,
        )
    else:
        camera = PiCamera(
            save_last_capture=config.save_last_capture,
            last_capture_path=config.last_capture_path,
            captures_dir=config.captures_dir,
        )

    speaker = Speaker(
        audio_device=config.audio_device,
        bluetooth_mac=config.bluetooth_mac,
        fish_api_key=config.fish_api_key,
        fish_reference_id=config.fish_reference_id,
        fish_model=config.fish_model,
        voice=config.voice,
        speech_rate=config.speech_rate,
        mock=args.mock,
    )
    recorder = AudioRecorder(
        record_device=config.record_device,
        bluetooth_mac=config.bluetooth_mac,
        mock=args.mock,
    )

    if args.dry_run:
        app = MemoryAidApp(
            config=config,
            places=places,
            camera=camera,
            speaker=speaker,
            recorder=recorder,
            button=None,
        )
        try:
            size_bytes = app.run_dry_run()
            if size_bytes > int(1.5 * 1024 * 1024):
                logger.warning("Payload size (%d bytes) exceeds 1.5 MB target!", size_bytes)
            return 0
        finally:
            app.close()

    app = MemoryAidApp(
        config=config,
        places=places,
        camera=camera,
        speaker=speaker,
        recorder=recorder,
        button=None,
    )

    if args.mock:
        button = KeyboardButton(
            on_identify_press=app.handle_button_press,
            on_mock_record=app.handle_mock_record,
        )
    else:
        button = GpioButtons(
            identify_pin=config.button_pin,
            record_pin=config.record_button_pin,
            bounce_time=0.1,
            on_identify_press=app.handle_button_press,
            on_record_press=app.handle_record_press,
            on_record_release=app.handle_record_release,
        )
    app.button = button

    def _sigterm_handler(_signum: int, _frame: object) -> None:
        logger.info("Received termination signal; shutting down...")
        app.close()

    signal.signal(signal.SIGTERM, _sigterm_handler)

    try:
        app.start()
        button.run_forever()
    finally:
        app.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
