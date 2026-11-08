"""Live MJPEG HTTP camera stream server (`http://localhost:8080`) and automatic popup viewer."""

import logging
import os
import shutil
import subprocess
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any, Callable, Optional

logger = logging.getLogger("memoryaid")

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Memory Aid — Live Camera Stream</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0f172a;
      color: #f8fafc;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 20px;
    }
    .card {
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.45);
      max-width: 720px;
      width: 100%;
      text-align: center;
    }
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }
    h1 {
      font-size: 1.25rem;
      font-weight: 600;
      color: #e2e8f0;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #10b981;
      animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.35; }
    }
    .viewport {
      width: 100%;
      background: #020617;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid #334155;
      aspect-ratio: 4 / 3;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .viewport img {
      width: 100%;
      height: 100%;
      object-fit: contain;
      display: block;
    }
    .controls {
      margin-top: 14px;
      display: flex;
      gap: 10px;
      justify-content: center;
      flex-wrap: wrap;
    }
    button {
      background: #3b82f6;
      color: white;
      border: none;
      border-radius: 8px;
      padding: 10px 16px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.15s;
    }
    button:hover { background: #2563eb; }
    .hint {
      margin-top: 12px;
      font-size: 0.82rem;
      color: #94a3b8;
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="header">
      <h1>Memory Aid — Live Camera Input</h1>
      <span class="badge"><span class="dot"></span> LIVE</span>
    </div>
    <div class="viewport">
      <img id="stream" src="/stream.mjpg" alt="Live Camera Stream" />
    </div>
    <div class="controls">
      <button onclick="triggerIdentify()">Where am I? (Button 2)</button>
    </div>
    <p class="hint">GPIO17 (Pin 11) = Where am I? &nbsp;|&nbsp; GPIO27 (Pin 13, Hold) = Record Room Memory</p>
  </div>
  <script>
    function triggerIdentify() {
      fetch('/trigger/identify', { method: 'POST' }).catch(() => {});
    }
  </script>
</body>
</html>
"""


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class CameraStreamServer:
    """Serves a live MJPEG video stream on `http://0.0.0.0:<port>` (`localhost:8080`) and pops up a viewer."""

    def __init__(
        self,
        camera: Any,
        port: int = 8080,
        fps: float = 5.0,
        on_identify_press: Optional[Callable[[], None]] = None,
    ) -> None:
        self.camera = camera
        self.port = port
        self.frame_interval = 1.0 / max(1.0, fps)
        self.on_identify_press = on_identify_press
        self._server: Optional[_ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()

    def get_frame(self) -> Optional[bytes]:
        """Fetch a fresh JPEG frame from the camera for streaming."""
        try:
            if hasattr(self.camera, "capture_jpeg_bytes"):
                jpeg = self.camera.capture_jpeg_bytes(for_stream=True)
            else:
                import base64

                b64 = self.camera.capture_base64()
                jpeg = base64.b64decode(b64)
            with self._frame_lock:
                self._latest_jpeg = jpeg
            return jpeg
        except Exception as exc:
            logger.debug("Stream frame capture skipped: %s", exc)
            with self._frame_lock:
                return self._latest_jpeg

    def start(self, auto_popup: bool = True) -> str:
        """Start the background HTTP server on `0.0.0.0:<port>` and optionally pop up the live stream window."""
        parent = self

        class StreamHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                # Suppress per-frame HTTP access logs
                pass

            def do_GET(self) -> None:
                if self.path in ("/", "/index.html"):
                    body = HTML_PAGE.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path.startswith("/snapshot.jpg"):
                    frame = parent.get_frame()
                    if not frame:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "Camera not ready")
                        return
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(frame)
                    return

                if self.path.startswith("/stream.mjpg"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header(
                        "Content-Type", "multipart/x-mixed-replace; boundary=FRAME"
                    )
                    self.end_headers()
                    try:
                        while not parent._stop_event.is_set():
                            frame = parent.get_frame()
                            if frame:
                                self.wfile.write(b"--FRAME\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(frame)))
                                self.end_headers()
                                self.wfile.write(frame)
                                self.wfile.write(b"\r\n")
                            time.sleep(parent.frame_interval)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    except Exception as exc:
                        logger.debug("MJPEG client disconnected: %s", exc)
                    return

                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:
                if self.path == "/trigger/identify":
                    if parent.on_identify_press is not None:
                        threading.Thread(
                            target=parent.on_identify_press, daemon=True
                        ).start()
                    self.send_response(HTTPStatus.NO_CONTENT)
                    self.end_headers()
                    return
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        url = f"http://localhost:{self.port}"
        try:
            self._server = _ThreadingHTTPServer(("0.0.0.0", self.port), StreamHandler)
        except OSError as exc:
            logger.warning(
                "Could not bind live camera stream server on port %d (%s).",
                self.port,
                exc,
            )
            return url

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="memoryaid-mjpeg-server",
        )
        self._thread.start()
        logger.info("Live camera stream available at %s (and http://memoryaid.local:%d)", url, self.port)

        if auto_popup:
            self._popup_viewer(url)

        return url

    def _popup_viewer(self, url: str) -> None:
        """Automatically pop up a window showing the live camera stream if a desktop/browser is available."""
        def _open() -> None:
            time.sleep(0.4)
            # On macOS, open default browser
            if shutil.which("open") is not None and os.uname().sysname == "Darwin":
                try:
                    subprocess.Popen(
                        ["open", url],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    logger.info("Opened live camera stream popup in browser: %s", url)
                    return
                except Exception:
                    pass

            # On Linux / Raspberry Pi with a connected display
            if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
                for browser_cmd in (
                    ["chromium-browser", "--app=" + url],
                    ["chromium", "--app=" + url],
                    ["xdg-open", url],
                ):
                    if shutil.which(browser_cmd[0]) is not None:
                        try:
                            subprocess.Popen(
                                browser_cmd,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            logger.info("Popped up live camera stream window via %s", browser_cmd[0])
                            return
                        except Exception:
                            continue
                try:
                    webbrowser.open(url, new=1)
                except Exception:
                    pass

        threading.Thread(target=_open, daemon=True, name="memoryaid-stream-popup").start()

    def close(self) -> None:
        self._stop_event.set()
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
