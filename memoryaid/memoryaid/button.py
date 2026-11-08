"""Button interfaces for Raspberry Pi GPIO (two buttons: identify + record toggle) and CLI/laptop mode."""

import logging
import select
import sys
import threading
from typing import Callable, Optional

logger = logging.getLogger("memoryaid")


class GpioButtons:
    """Manages the two GPIO buttons on the Raspberry Pi AND interactive CLI commands:
    1. Identify button (default GPIO17, Pin 11): single press tells the user what room/place they are in.
    2. Record button (default GPIO27, Pin 13): press once to start recording + take photo; press a second time to stop, save, and upload.
    3. Terminal CLI controls (when running in an interactive terminal):
       - Type 'r' + Enter: toggle Record & Upload (1st time starts recording + captures photo; 2nd time stops, saves, and uploads)
       - Type 'r <description>' + Enter: capture photo, save with <description>, and upload immediately
       - Press Enter (or 'i' + Enter): trigger 'Where am I?'
       - Type 'q' + Enter: quit
    """

    def __init__(
        self,
        identify_pin: int = 17,
        record_pin: int = 27,
        bounce_time: float = 0.1,
        on_identify_press: Optional[Callable[[], None]] = None,
        on_record_press: Optional[Callable[[], None]] = None,
        on_record_release: Optional[Callable[[], None]] = None,
        on_text_record: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._identify_pin = identify_pin
        self._record_pin = record_pin
        self._bounce_time = bounce_time
        self._on_identify_press = on_identify_press
        self._on_record_press = on_record_press
        self._on_record_release = on_record_release
        self._on_text_record = on_text_record
        self._identify_button = None
        self._record_button = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Lazily import gpiozero and bind both buttons."""
        try:
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

            logger.info(
                "GPIO buttons initialised: Identify=GPIO%d, Record(toggle)=GPIO%d (pull_up=True, bounce=%.2fs).",
                self._identify_pin,
                self._record_pin,
                self._bounce_time,
            )
        except Exception as exc:
            logger.warning(
                "GPIO button init warning (%s); CLI keyboard controls remain active.", exc
            )

    def _print_cli_prompt(self) -> None:
        sys.stdout.write(
            "\n[CLI Controls] Type 'r' + Enter to toggle Record & Upload (or 'r <description>' + Enter), "
            "Enter for 'Where am I?', or 'q' + Enter to quit: "
        )
        sys.stdout.flush()

    def run_forever(self) -> None:
        """Block the main thread until stopped or interrupted, accepting CLI commands if stdin is a TTY."""
        interactive_stdin = False
        try:
            interactive_stdin = sys.stdin is not None and sys.stdin.isatty()
        except Exception:
            interactive_stdin = False

        if interactive_stdin:
            self._print_cli_prompt()

        try:
            while not self._stop_event.is_set():
                if not interactive_stdin:
                    self._stop_event.wait(timeout=1.0)
                    continue

                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 1.0)
                except Exception:
                    self._stop_event.wait(timeout=1.0)
                    continue

                if not ready:
                    continue

                line = sys.stdin.readline()
                if not line:
                    interactive_stdin = False
                    continue

                cmd = line.strip()
                if cmd.lower() == "q":
                    logger.info("Quit requested ('q'). Shutting down...")
                    break
                if cmd.lower() == "r":
                    if self._on_record_press is not None:
                        self._on_record_press()
                elif cmd.lower().startswith("r "):
                    spoken = cmd[2:].strip()
                    if spoken and self._on_text_record is not None:
                        self._on_text_record(spoken)
                    elif self._on_record_press is not None:
                        self._on_record_press()
                else:
                    if self._on_identify_press is not None:
                        self._on_identify_press()

                if not self._stop_event.is_set():
                    self._print_cli_prompt()
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
