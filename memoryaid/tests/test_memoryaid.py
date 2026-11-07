"""Comprehensive unit tests for Memory Aid (no network calls, no Pi hardware required)."""

import base64
import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests
from PIL import Image

from memoryaid.app import (
    MSG_CAMERA_ERROR,
    MSG_LET_ME_LOOK,
    MSG_NETWORK_ERROR,
    MSG_NOT_SET_UP,
    MemoryAidApp,
    map_error_to_spoken_message,
)
from memoryaid.camera import CameraError, MockCamera
from memoryaid.config import Config, load_config
from memoryaid.memories import PlaceMemory, load_memories
from memoryaid.vision import (
    SYSTEM_PROMPT,
    VisionAPIError,
    build_messages_payload,
    clean_spoken_text,
    query_vision_api,
)


def _create_dummy_image(path: Path, size: tuple = (800, 600), color: tuple = (120, 160, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    img.save(path, format=fmt)


class TestMemoriesAndCache(unittest.TestCase):
    """Tests for memory book loading and reference image caching."""

    def test_empty_or_missing_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing_dir = Path(tmp) / "does_not_exist"
            self.assertEqual(load_memories(missing_dir), [])

            empty_dir = Path(tmp) / "empty"
            empty_dir.mkdir()
            self.assertEqual(load_memories(empty_dir), [])

    def test_missing_note_uses_folder_name_without_underscores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            place_dir = root / "grandmas_garden"
            _create_dummy_image(place_dir / "1.jpg")

            places = load_memories(root)
            self.assertEqual(len(places), 1)
            self.assertEqual(places[0].name, "grandmas_garden")
            self.assertEqual(places[0].note, "grandmas garden")
            self.assertEqual(len(places[0].images_b64), 1)

    def test_too_many_photos_caps_at_two_sorted_and_case_insensitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            place_dir = root / "joes_cafe"
            (place_dir).mkdir(parents=True)
            (place_dir / "note.txt").write_text("Joe's Cafe note.", encoding="utf-8")

            _create_dummy_image(place_dir / "c_third.JPG", color=(255, 0, 0))
            _create_dummy_image(place_dir / "a_first.jpeg", color=(0, 255, 0))
            _create_dummy_image(place_dir / "b_second.PNG", color=(0, 0, 255))

            places = load_memories(root)
            self.assertEqual(len(places), 1)
            self.assertEqual(len(places[0].images_b64), 2)

            # Verify image dimensions were resized to max 512px on the long edge
            raw_bytes = base64.b64decode(places[0].images_b64[0])
            with Image.open(io.BytesIO(raw_bytes)) as decoded_img:
                self.assertLessEqual(max(decoded_img.size), 512)

    def test_more_than_ten_places_uses_first_ten_alphabetically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(12):
                folder = root / f"place_{i:02d}"
                folder.mkdir()
                (folder / "note.txt").write_text(f"Note {i}", encoding="utf-8")
                _create_dummy_image(folder / "1.jpg")

            places = load_memories(root)
            self.assertEqual(len(places), 10)
            self.assertEqual([p.name for p in places], [f"place_{i:02d}" for i in range(10)])

    def test_cache_invalidation_on_file_modification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            place_dir = root / "kitchen"
            photo_path = place_dir / "1.jpg"
            _create_dummy_image(photo_path, size=(600, 400), color=(10, 20, 30))

            places_first = load_memories(root)
            first_b64 = places_first[0].images_b64[0]

            cache_files = list((root / ".cache").glob("*.json"))
            self.assertEqual(len(cache_files), 1)

            # Modify the image with a different color and bump mtime
            _create_dummy_image(photo_path, size=(600, 400), color=(240, 200, 50))
            future_time = time.time() + 10.0
            os.utime(photo_path, (future_time, future_time))

            places_second = load_memories(root)
            second_b64 = places_second[0].images_b64[0]
            self.assertNotEqual(first_b64, second_b64)


class TestVisionPayloadAndCleaning(unittest.TestCase):
    """Tests for Claude Messages API payload structure, markdown stripping, and HTTP retry logic."""

    def test_request_payload_structure_order(self) -> None:
        places = [
            PlaceMemory(name="joes_cafe", note="Coffee with Maria.", images_b64=["REF_IMG_1", "REF_IMG_2"]),
            PlaceMemory(name="davids_bedroom", note="David is your son.", images_b64=["REF_IMG_3"]),
        ]
        payload = build_messages_payload(places=places, captured_b64="NEW_CAM_B64", model="claude-haiku-5-5")

        self.assertEqual(payload["model"], "claude-haiku-5-5")
        self.assertEqual(payload["max_tokens"], 200)
        self.assertEqual(payload["system"], SYSTEM_PROMPT)
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["role"], "user")

        content = payload["messages"][0]["content"]
        # Expected sequence:
        # 0: text (Known place 1)
        # 1: image (REF_IMG_1)
        # 2: image (REF_IMG_2)
        # 3: text (Known place 2)
        # 4: image (REF_IMG_3)
        # 5: text ("NEW photo from the camera right now:")
        # 6: image (NEW_CAM_B64)
        # 7: text ("Where am I?")
        self.assertEqual(len(content), 8)
        self.assertEqual(
            content[0],
            {"type": "text", "text": "Known place 1: joes_cafe\nCaregiver note: Coffee with Maria."},
        )
        self.assertEqual(content[1]["type"], "image")
        self.assertEqual(content[1]["source"]["data"], "REF_IMG_1")
        self.assertEqual(content[2]["source"]["data"], "REF_IMG_2")
        self.assertEqual(
            content[3],
            {"type": "text", "text": "Known place 2: davids_bedroom\nCaregiver note: David is your son."},
        )
        self.assertEqual(content[4]["source"]["data"], "REF_IMG_3")
        self.assertEqual(content[5], {"type": "text", "text": "NEW photo from the camera right now:"})
        self.assertEqual(content[6]["type"], "image")
        self.assertEqual(content[6]["source"]["data"], "NEW_CAM_B64")
        self.assertEqual(content[7], {"type": "text", "text": "Where am I?"})

    def test_markdown_stripping(self) -> None:
        raw = '  "**You\'re at Joe\'s Cafe.** #cafe _Maria_ is `here`."  '
        cleaned = clean_spoken_text(raw)
        self.assertEqual(cleaned, "You're at Joe's Cafe. cafe Maria is here.")

    @patch("memoryaid.vision.requests.post")
    def test_query_vision_api_retries_once_on_503_then_succeeds(self, mock_post: MagicMock) -> None:
        resp_503 = MagicMock()
        resp_503.status_code = 503

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "content": [{"type": "text", "text": "*You're at Joe's Cafe.*"}]
        }

        mock_post.side_effect = [resp_503, resp_200]
        payload = build_messages_payload([], "NEW_B64")

        text, size_bytes, _latency = query_vision_api(
            payload, api_key="test-key", timeout=5, retry_delay=0.01
        )
        self.assertEqual(text, "You're at Joe's Cafe.")
        self.assertGreater(size_bytes, 100)
        self.assertEqual(mock_post.call_count, 2)

    @patch("memoryaid.vision.requests.post")
    def test_query_vision_api_fails_after_retry_on_network_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        payload = build_messages_payload([], "NEW_B64")

        with self.assertRaises(VisionAPIError):
            query_vision_api(payload, api_key="test-key", timeout=5, retry_delay=0.01)
        self.assertEqual(mock_post.call_count, 2)


class TestErrorMappingAndAppLoop(unittest.TestCase):
    """Tests for error-to-speech mapping and runtime app orchestration."""

    def test_error_to_message_mapping(self) -> None:
        self.assertEqual(map_error_to_spoken_message(CameraError("ribbon cable loose")), MSG_CAMERA_ERROR)
        self.assertEqual(map_error_to_spoken_message(VisionAPIError("HTTP 500")), MSG_NETWORK_ERROR)
        self.assertEqual(map_error_to_spoken_message(RuntimeError("unexpected")), MSG_NETWORK_ERROR)

    def test_missing_api_key_speaks_setup_reminder(self) -> None:
        cfg = Config(anthropic_api_key="")
        mock_cam = MagicMock()
        mock_speaker = MagicMock()
        app = MemoryAidApp(config=cfg, places=[], camera=mock_cam, speaker=mock_speaker)

        app.start()
        mock_speaker.speak.assert_called_with(MSG_NOT_SET_UP)

        mock_speaker.reset_mock()
        app.handle_button_press()
        mock_speaker.speak.assert_called_once_with(MSG_NOT_SET_UP)

    @patch("memoryaid.app.query_vision_api")
    def test_successful_button_press_flow(self, mock_query: MagicMock) -> None:
        mock_query.return_value = ("You're in a sunny kitchen.", 1234, 0.25)
        cfg = Config(anthropic_api_key="sk-ant-test")
        mock_cam = MagicMock()
        mock_cam.capture_base64.return_value = "FAKE_B64"
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = 1.5

        app = MemoryAidApp(config=cfg, places=[], camera=mock_cam, speaker=mock_speaker)
        app.handle_button_press()

        mock_speaker.speak_async.assert_called_once_with(MSG_LET_ME_LOOK)
        mock_speaker.wait.assert_called_once()
        mock_speaker.speak.assert_called_once_with("You're in a sunny kitchen.")

    def test_config_loading_from_custom_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "custom.env"
            env_path.write_text(
                "ANTHROPIC_API_KEY=sk-test-123\n"
                "MODEL=claude-sonnet-5-5\n"
                "SPEECH_RATE=155\n"
                "SAVE_LAST_CAPTURE=true\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_config(env_file=env_path, project_root=Path(tmp))
                self.assertEqual(cfg.anthropic_api_key, "sk-test-123")
                self.assertEqual(cfg.model, "claude-sonnet-5-5")
                self.assertEqual(cfg.speech_rate, 155)
                self.assertTrue(cfg.save_last_capture)

    def test_mock_camera_loads_and_resizes_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.jpg"
            _create_dummy_image(img_path, size=(1280, 960), color=(50, 100, 150))
            cam = MockCamera(image_path=img_path)
            cam.start()
            b64 = cam.capture_base64()
            raw = base64.b64decode(b64)
            with Image.open(io.BytesIO(raw)) as img:
                self.assertEqual(img.size, (640, 480))


if __name__ == "__main__":
    unittest.main()
