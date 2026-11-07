"""Button interfaces for Raspberry Pi GPIO (two buttons: identify + hold-to-record) and laptop mock mode."""

import logging
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger("memoryaid")


class GpioButtons:
    """Manages the two GPIO buttons on the Raspberry Pi:
    1. Identify button (default GPIO17, Pin 11): single press tells the user what room/place they are in.
    2. Record button (default GPIO27, Pin 13): press-and-hold takes a picture and listens; release saves what they said to a file.
    """

    def __init__(
        self,
        identify_pin: int = 17,
        record_pin: int = 27,
        bounce_time: float = 0.1,
        on_identify_press: Optional[Callable[[], None]] = None,
        on_record_press: Optional[Callable[[], None]] = None,
        on_record_release: Optional[Callable[[], None]] = None,
    ) -> None:
        self._identify_pin = identify_pin
        self._record_pin = record_pin
        self._bounce_time = bounce_time
        self._on_identify_press = on_identify_press
        self._on_record_press = on_record_press
        self._on_record_release = on_record_release
        self._identify_button = None
        self._record_button = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Lazily import gpiozero and bind both buttons."""
        from gpiozero import Button  # type: ignore[import-not-found]

        self._identify_button = Button(
            self._identify_pin, pull_up=True, bounce_time=self._bounce_time
        )
        if self._on_identify_press is not None:
            self._identify_button.when_pressed = self._on_identify_press

        self._record_button = Button(
            self._record_pin, pull_up=True, bounce_time=self._bounce_time
        )
        if self._on_record_press is not None:
            self._record_button.when_pressed = self._on_record_press
        if self._on_record_release is not None:
            self._record_button.when_released = self._on_record_release

        logger.info(
            "GPIO buttons initialised: Identify=GPIO%d, Record(hold)=GPIO%d (pull_up=True, bounce=%.2fs).",
            self._identify_pin,
            self._record_pin,
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
        for btn in (self._identify_button, self._record_button):
            if btn is not None:
                try:
                    btn.close()
                except Exception:
                    pass
        self._identify_button = None
        self._record_button = None


# Alias for backward compatibility
GpioButton = GpioButtons


class KeyboardButton:
    """Mock dual-button interface for laptop testing:
    - Press Enter: trigger Button 2 ("Where am I?")
    - Type 'r' + Enter: simulate pressing & holding Button 1 (takes photo & records/accepts spoken note)
    - Type 'q' + Enter: quit
    """

    def __init__(
        self,
        on_identify_press: Optional[Callable[[], None]] = None,
        on_mock_record: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._on_identify_press = on_identify_press
        self._on_mock_record = on_mock_record
        self._stop_event = threading.Event()

    def start(self) -> None:
        logger.info(
            "Mock buttons ready: Enter = 'Where am I?' button | 'r' + Enter = 'Hold-to-Record' button | 'q' + Enter = quit."
        )

    def run_forever(self) -> None:
        while not self._stop_event.is_set():
            try:
                sys.stdout.write(
                    "\n[Mock Mode] Press Enter ('Where am I?'), 'r' + Enter (Record Memory), or 'q' + Enter (Quit): "
                )
                sys.stdout.flush()
                line = sys.stdin.readline()
                if not line:
                    logger.info("End of input reached; exiting mock loop.")
                    break
                cmd = line.strip()
                if cmd.lower() == "q":
                    logger.info("Quit requested ('q'). Exiting.")
                    break
                if cmd.lower().startswith("r"):
                    # Allow inline text like `r This is the kitchen` or prompt for what was spoken
                    spoken = cmd[1:].strip()
                    if not spoken:
                        sys.stdout.write(
                            "[Mock Record] Type what you are saying while holding the button: "
                        )
                        sys.stdout.flush()
                        spoken = sys.stdin.readline().strip()
                    if self._on_mock_record is not None:
                        self._on_mock_record(spoken)
                else:
                    if self._on_identify_press is not None:
                        self._on_identify_press()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received; exiting.")
                break

    def close(self) -> None:
        self._stop_event.set()
