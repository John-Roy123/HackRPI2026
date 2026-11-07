"""Button interfaces for Raspberry Pi GPIO (gpiozero) and laptop terminal mock mode."""

import logging
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger("memoryaid")


class GpioButton:
    """Hardware momentary push button on GPIO (default GPIO17, active-low with internal pull-up)."""

    def __init__(
        self,
        pin: int = 17,
        bounce_time: float = 0.1,
        on_press: Optional[Callable[[], None]] = None,
    ) -> None:
        self._pin = pin
        self._bounce_time = bounce_time
        self._on_press = on_press
        self._button = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Lazily import gpiozero and bind the button press handler."""
        from gpiozero import Button  # type: ignore[import-not-found]

        self._button = Button(self._pin, pull_up=True, bounce_time=self._bounce_time)
        if self._on_press is not None:
            self._button.when_pressed = self._on_press
        logger.info(
            "GPIO button initialised on GPIO%d (pull_up=True, bounce_time=%.2fs).",
            self._pin,
            self._bounce_time,
        )

    def run_forever(self) -> None:
        """Block the main thread until stopped or interrupted."""
        try:
            while not self._stop_event.is_set():
                self._stop_event.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("Interrupted; shutting down GPIO button loop.")

    def close(self) -> None:
        self._stop_event.set()
        if self._button is not None:
            try:
                self._button.close()
            except Exception:
                pass
            self._button = None


class KeyboardButton:
    """Mock button for laptop testing: press Enter to trigger, type 'q' + Enter to quit."""

    def __init__(self, on_press: Optional[Callable[[], None]] = None) -> None:
        self._on_press = on_press
        self._stop_event = threading.Event()

    def start(self) -> None:
        logger.info("Mock button ready (press Enter to trigger, type 'q' and Enter to quit).")

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                sys.stdout.write("\n[Mock Mode] Press Enter to take a photo (or 'q' + Enter to quit): ")
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    # EOF on stdin
                    logger.info("End of input reached; exiting mock loop.")
                    break
                if line.strip().lower() == "q":
                    logger.info("Quit requested ('q'). Exiting.")
                    break
                if self._on_press is not None:
                    self._on_press()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received; exiting.")
                break

    def close(self) -> None:
        self._stop_event.set()
