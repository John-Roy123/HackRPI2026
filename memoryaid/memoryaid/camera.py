"""Camera interfaces for Raspberry Pi (picamera2) and laptop mock mode."""

import base64
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageStat

from .memories import ALLOWED_EXTENSIONS, preprocess_image_bytes

logger = logging.getLogger("memoryaid")

CAPTURE_MAX_EDGE = 640
CAPTURE_JPEG_QUALITY = 85
MAX_SAVED_CAPTURES = 50


class CameraError(Exception):
    """Raised when the camera fails to initialise or capture a frame."""


def _save_capture_files(
    jpeg_bytes: bytes,
    last_capture_path: Optional[Path] = None,
    captures_dir: Optional[Path] = None,
    mean_brightness: Optional[float] = None,
) -> None:
    """Save the captured JPEG to `last_capture.jpg` and `captured_photos/capture_<timestamp>.jpg` on board."""
    saved_paths: List[str] = []

    if last_capture_path is not None:
        try:
            last_capture_path.parent.mkdir(parents=True, exist_ok=True)
            last_capture_path.write_bytes(jpeg_bytes)
            saved_paths.append(str(last_capture_path))
        except OSError as exc:
            logger.warning("Could not write capture to %s: %s", last_capture_path, exc)

    if captures_dir is None and last_capture_path is not None:
        captures_dir = last_capture_path.parent / "captured_photos"

    if captures_dir is not None:
        try:
            captures_dir.mkdir(parents=True, exist_ok=True)
            latest_file = captures_dir / "latest.jpg"
            latest_file.write_bytes(jpeg_bytes)

            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int((time.time() % 1) * 1000)
            stamped_file = captures_dir / f"capture_{ts}_{ms:03d}.jpg"
            stamped_file.write_bytes(jpeg_bytes)
            saved_paths.append(str(stamped_file))

            # Prune old captures beyond MAX_SAVED_CAPTURES so SD card never fills up
            history = sorted(captures_dir.glob("capture_*.jpg"), key=lambda p: p.name)
            if len(history) > MAX_SAVED_CAPTURES:
                for old_file in history[: len(history) - MAX_SAVED_CAPTURES]:
                    try:
                        old_file.unlink()
                    except OSError:
                        pass
        except OSError as exc:
            logger.warning("Could not save timestamped capture in %s: %s", captures_dir, exc)

    if saved_paths:
        brightness_info = (
            f", avg_brightness={mean_brightness:.1f}/255" if mean_brightness is not None else ""
        )
        logger.info(
            "Saved onboard camera photo (%d bytes%s) -> %s",
            len(jpeg_bytes),
            brightness_info,
            " and ".join(saved_paths),
        )
        if mean_brightness is not None and mean_brightness < 20.0:
            logger.warning(
                "Captured camera image is very dark (avg_brightness=%.1f/255)! Check room lighting or camera lens.",
                mean_brightness,
            )


def encode_pil_image_to_jpeg(
    img: Image.Image,
    max_edge: int = CAPTURE_MAX_EDGE,
    quality: int = CAPTURE_JPEG_QUALITY,
    save_last_capture: bool = False,
    last_capture_path: Optional[Path] = None,
    captures_dir: Optional[Path] = None,
) -> bytes:
    """Resize a captured PIL Image, optionally save on board for inspection, and encode as JPEG bytes."""
    mean_brightness: Optional[float] = None
    try:
        stat = ImageStat.Stat(img.convert("L"))
        mean_brightness = float(stat.mean[0])
    except Exception:
        pass

    jpeg_bytes = preprocess_image_bytes(img, max_edge=max_edge, quality=quality)
    if save_last_capture and (last_capture_path is not None or captures_dir is not None):
        _save_capture_files(
            jpeg_bytes=jpeg_bytes,
            last_capture_path=last_capture_path,
            captures_dir=captures_dir,
            mean_brightness=mean_brightness,
        )
    return jpeg_bytes


def encode_pil_image_for_capture(
    img: Image.Image,
    save_last_capture: bool = False,
    last_capture_path: Optional[Path] = None,
    captures_dir: Optional[Path] = None,
) -> str:
    """Resize a captured PIL Image to max 640px long edge, JPEG quality 85, and base64-encode in memory."""
    jpeg_bytes = encode_pil_image_to_jpeg(
        img,
        max_edge=CAPTURE_MAX_EDGE,
        quality=CAPTURE_JPEG_QUALITY,
        save_last_capture=save_last_capture,
        last_capture_path=last_capture_path,
        captures_dir=captures_dir,
    )
    return base64.b64encode(jpeg_bytes).decode("ascii")


class PiCamera:
    """Hardware camera using picamera2 on Raspberry Pi OS."""

    def __init__(
        self,
        save_last_capture: bool = True,
        last_capture_path: Optional[Path] = None,
        captures_dir: Optional[Path] = None,
        warmup_seconds: float = 1.5,
    ) -> None:
        self._save_last_capture = save_last_capture
        self._last_capture_path = last_capture_path
        self._captures_dir = captures_dir
        self._warmup_seconds = warmup_seconds
        self._picam2 = None
        self._lock = threading.Lock()

    def start(self) -> None:
        """Lazily import picamera2, configure 1280x960 still capture, and wait for auto-exposure."""
        try:
            logger.info("Importing picamera2 (this takes 20-30s on Raspberry Pi Zero W)...")
            from picamera2 import Picamera2  # type: ignore[import-not-found]

            self._picam2 = Picamera2()
            still_config = self._picam2.create_still_configuration(
                main={"size": (1280, 960)}
            )
            self._picam2.configure(still_config)
            self._picam2.start()
            if self._warmup_seconds > 0:
                time.sleep(self._warmup_seconds)
            logger.info("Camera initialised (1280x960 still configuration).")
        except Exception as exc:
            raise CameraError(f"Failed to initialize PiCamera: {exc}") from exc

    def capture_base64(self) -> str:
        """Capture a single still image, save on board if enabled, resize to 640px max edge, and return base64."""
        if self._picam2 is None:
            raise CameraError("PiCamera is not started.")
        with self._lock:
            try:
                pil_image = self._picam2.capture_image("main")
            except Exception as exc:
                raise CameraError(f"Failed to capture image from PiCamera: {exc}") from exc

        return encode_pil_image_for_capture(
            pil_image,
            save_last_capture=self._save_last_capture,
            last_capture_path=self._last_capture_path,
            captures_dir=self._captures_dir,
        )

    def close(self) -> None:
        with self._lock:
            if self._picam2 is not None:
                try:
                    self._picam2.stop()
                    self._picam2.close()
                except Exception:
                    pass
                self._picam2 = None


class MockCamera:
    """Laptop mock camera that loads a local image file or cycles through a directory of images."""

    def __init__(
        self,
        image_path: Path,
        save_last_capture: bool = False,
        last_capture_path: Optional[Path] = None,
        captures_dir: Optional[Path] = None,
    ) -> None:
        self._image_path = image_path
        self._save_last_capture = save_last_capture
        self._last_capture_path = last_capture_path
        self._captures_dir = captures_dir
        self._files: List[Path] = []
        self._index = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self._image_path.exists():
            raise CameraError(f"Mock image path does not exist: {self._image_path}")

        if self._image_path.is_dir():
            self._files = sorted(
                [
                    p
                    for p in self._image_path.iterdir()
                    if p.is_file()
                    and not p.name.startswith(".")
                    and p.suffix.lower() in ALLOWED_EXTENSIONS
                ],
                key=lambda p: p.name.lower(),
            )
            if not self._files:
                raise CameraError(
                    f"No valid image files (.jpg/.jpeg/.png) found in directory: {self._image_path}"
                )
            logger.info(
                "MockCamera ready with %d image(s) from folder %s.",
                len(self._files),
                self._image_path,
            )
        else:
            self._files = [self._image_path]
            logger.info("MockCamera ready with image %s.", self._image_path)

    def capture_base64(self) -> str:
        if not self._files:
            raise CameraError("MockCamera has no images loaded.")

        with self._lock:
            chosen = self._files[self._index % len(self._files)]
            self._index += 1
            logger.info("MockCamera capturing from %s", chosen)

        try:
            with Image.open(chosen) as img:
                return encode_pil_image_for_capture(
                    img,
                    save_last_capture=self._save_last_capture,
                    last_capture_path=self._last_capture_path,
                    captures_dir=self._captures_dir,
                )
        except Exception as exc:
            raise CameraError(f"MockCamera failed to read {chosen}: {exc}") from exc

    def close(self) -> None:
        pass
