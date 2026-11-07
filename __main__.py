"""CLI entry point for `python3 -m memoryaid`."""

import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

from .app import MemoryAidApp
from .button import GpioButton, KeyboardButton
from .camera import MockCamera, PiCamera
from .config import load_config
from .memories import load_memories
from .speech import Speaker


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
        description="Memory Aid: A screenless camera device for people living with dementia.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in laptop mock mode (no picamera2 or gpiozero imports; Enter key acts as button).",
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
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    _configure_logging()
    logger = logging.getLogger("memoryaid")
    args = parse_args(argv)

    if args.mock and args.image is None:
        logger.error("--image PATH is required when running with --mock.")
        return 2

    config = load_config(env_file=args.env_file)
    logger.info(
        "Loaded config: model=%s, audio_device=%s, voice=%s, rate=%d, memories_dir=%s, mock=%s",
        config.model,
        config.audio_device,
        config.voice,
        config.speech_rate,
        config.memories_dir,
        args.mock,
    )

    places = load_memories(config.memories_dir)

    if args.mock:
        camera = MockCamera(
            image_path=args.image,
            save_last_capture=config.save_last_capture,
            last_capture_path=config.last_capture_path,
        )
    else:
        camera = PiCamera(
            save_last_capture=config.save_last_capture,
            last_capture_path=config.last_capture_path,
        )

    speaker = Speaker(
        audio_device=config.audio_device,
        voice=config.voice,
        speech_rate=config.speech_rate,
        mock=args.mock,
    )

    if args.dry_run:
        app = MemoryAidApp(
            config=config,
            places=places,
            camera=camera,
            speaker=speaker,
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
        button=None,
    )

    if args.mock:
        button = KeyboardButton(on_press=app.handle_button_press)
    else:
        button = GpioButton(
            pin=config.button_pin,
            bounce_time=0.1,
            on_press=app.handle_button_press,
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
