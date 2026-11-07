"""Comprehensive unit tests for Memory Aid (no network calls, no Pi hardware required)."""

import base64
import io
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
    MSG_RECORD_SAVED,
    MemoryAidApp,
    map_error_to_spoken_message,
)
from memoryaid.camera import CameraError, MockCamera
from memoryaid.config import Config, load_config
from memoryaid.memories import PlaceMemory, load_memories, save_recorded_memory
from memoryaid.speech import pcm16_to_wav_bytes, synthesize_elevenlabs_wav
from memoryaid.vision import (
    SYSTEM_PROMPT,
    VisionAPIError,
    build_messages_payload,
    build_transcription_payload,
    clean_spoken_text,
    get_gemini_endpoint,
    query_vision_api,
)


def _create_dummy_image(path: Path, size: tuple = (800, 600), color: tuple = (120, 160, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    fmt = "PNG" if path.suffix.lower() == ".png" else "JPEG"
    img.save(path, format=fmt)


def _create_dummy_jpeg_b64(size: tuple = (640, 480), color: tuple = (100, 150, 200)) -> str:
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestMemoriesAndCache(unittest.TestCase):
    """Tests for memory book loading, saving recorded memories, and reference image caching."""

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

    def test_save_recorded_memory_writes_photo_and_note_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b64_img = _create_dummy_jpeg_b64()
            spoken = "This is Sarah's kitchen where we bake cookies every Sunday."

            place = save_recorded_memory(root, b64_img, spoken)
            self.assertEqual(place.name, "sarahs_kitchen_where_bake")
            self.assertTrue((root / place.name / "note.txt").is_file())
            self.assertTrue((root / place.name / "1.jpg").is_file())
            self.assertTrue((root / "recorded_notes.txt").is_file())
            self.assertEqual(
                (root / place.name / "note.txt").read_text(encoding="utf-8").strip(),
                spoken,
            )


class TestVisionPayloadAndCleaning(unittest.TestCase):
    """Tests for Gemini API payload structure, transcription payload, markdown stripping, and HTTP retry logic."""

    def test_request_payload_structure_order(self) -> None:
        places = [
            PlaceMemory(name="joes_cafe", note="Coffee with Maria.", images_b64=["REF_IMG_1", "REF_IMG_2"]),
            PlaceMemory(name="davids_bedroom", note="David is your son.", images_b64=["REF_IMG_3"]),
        ]
        payload = build_messages_payload(places=places, captured_b64="NEW_CAM_B64", model="gemini-2.5-flash")

        self.assertEqual(
            get_gemini_endpoint("gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        )
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 200)
        self.assertEqual(payload["system_instruction"]["parts"][0]["text"], SYSTEM_PROMPT)
        self.assertEqual(len(payload["contents"]), 1)
        self.assertEqual(payload["contents"][0]["role"], "user")

        parts = payload["contents"][0]["parts"]
        self.assertEqual(len(parts), 8)
        self.assertEqual(
            parts[0],
            {"text": "Known place 1: joes_cafe\nCaregiver note: Coffee with Maria."},
        )
        self.assertEqual(parts[1]["inline_data"]["data"], "REF_IMG_1")
        self.assertEqual(parts[2]["inline_data"]["data"], "REF_IMG_2")
        self.assertEqual(
            parts[3],
            {"text": "Known place 2: davids_bedroom\nCaregiver note: David is your son."},
        )
        self.assertEqual(parts[4]["inline_data"]["data"], "REF_IMG_3")
        self.assertEqual(parts[5], {"text": "NEW photo from the camera right now:"})
        self.assertEqual(parts[6]["inline_data"]["data"], "NEW_CAM_B64")
        self.assertEqual(parts[7], {"text": "Where am I?"})

    def test_transcription_payload_structure(self) -> None:
        payload = build_transcription_payload(wav_bytes=b"RIFF_FAKE_WAV", captured_b64="ROOM_B64")
        parts = payload["contents"][0]["parts"]
        self.assertEqual(len(parts), 3)
        self.assertEqual(parts[0]["inline_data"]["mime_type"], "image/jpeg")
        self.assertEqual(parts[0]["inline_data"]["data"], "ROOM_B64")
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "audio/wav")
        self.assertIn("text", parts[2])

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
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "*You're at Joe's Cafe.*"}]
                    }
                }
            ]
        }

        mock_post.side_effect = [resp_503, resp_200]
        payload = build_messages_payload([], "NEW_B64")

        text, size_bytes, _latency = query_vision_api(
            payload, api_key="test-key", model="gemini-2.5-flash", timeout=5, retry_delay=0.01
        )
        self.assertEqual(text, "You're at Joe's Cafe.")
        self.assertGreater(size_bytes, 100)
        self.assertEqual(mock_post.call_count, 2)

    @patch("memoryaid.vision.requests.post")
    def test_query_vision_api_fails_after_retry_on_network_error(self, mock_post: MagicMock) -> None:
        mock_post.side_effect = requests.ConnectionError("Connection refused")
        payload = build_messages_payload([], "NEW_B64")

        with self.assertRaises(VisionAPIError):
            query_vision_api(payload, api_key="test-key", model="gemini-2.5-flash", timeout=5, retry_delay=0.01)
        self.assertEqual(mock_post.call_count, 2)


class TestErrorMappingAndAppLoop(unittest.TestCase):
    """Tests for error-to-speech mapping and runtime app orchestration (both buttons)."""

    def test_error_to_message_mapping(self) -> None:
        self.assertEqual(map_error_to_spoken_message(CameraError("ribbon cable loose")), MSG_CAMERA_ERROR)
        self.assertEqual(map_error_to_spoken_message(VisionAPIError("HTTP 500")), MSG_NETWORK_ERROR)
        self.assertEqual(map_error_to_spoken_message(RuntimeError("unexpected")), MSG_NETWORK_ERROR)

    def test_missing_api_key_speaks_setup_reminder(self) -> None:
        cfg = Config(gemini_api_key="")
        mock_cam = MagicMock()
        mock_speaker = MagicMock()
        app = MemoryAidApp(config=cfg, places=[], camera=mock_cam, speaker=mock_speaker)

        app.start()
        mock_speaker.speak.assert_called_with(MSG_NOT_SET_UP)

        mock_speaker.reset_mock()
        app.handle_button_press()
        mock_speaker.speak.assert_called_once_with(MSG_NOT_SET_UP)

    @patch("memoryaid.app.query_vision_api")
    def test_successful_identify_button_press_flow(self, mock_query: MagicMock) -> None:
        mock_query.return_value = ("You're in a sunny kitchen.", 1234, 0.25)
        cfg = Config(gemini_api_key="AIzaSyTestKey")
        mock_cam = MagicMock()
        mock_cam.capture_base64.return_value = "FAKE_B64"
        mock_speaker = MagicMock()
        mock_speaker.speak.return_value = 1.5

        app = MemoryAidApp(config=cfg, places=[], camera=mock_cam, speaker=mock_speaker)
        app.handle_button_press()

        mock_speaker.speak_async.assert_called_once_with(MSG_LET_ME_LOOK)
        mock_speaker.wait.assert_called_once()
        mock_speaker.speak.assert_called_once_with("You're in a sunny kitchen.")

    @patch("memoryaid.app.transcribe_audio_with_gemini")
    def test_successful_hold_to_record_button_flow(self, mock_transcribe: MagicMock) -> None:
        mock_transcribe.return_value = "This is the living room where we watch movies."
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config(gemini_api_key="AIzaSyTestKey", memories_dir=Path(tmp))
            mock_cam = MagicMock()
            mock_cam.capture_base64.return_value = _create_dummy_jpeg_b64()
            mock_speaker = MagicMock()
            mock_recorder = MagicMock()
            mock_recorder.stop_recording.return_value = b"RIFF_WAV_DATA"

            app = MemoryAidApp(
                config=cfg,
                places=[],
                camera=mock_cam,
                speaker=mock_speaker,
                recorder=mock_recorder,
            )

            # Press and hold Button 1
            app.handle_record_press()
            mock_recorder.start_recording.assert_called_once()
            mock_cam.capture_base64.assert_called_once()

            # Release Button 1
            app.handle_record_release()
            mock_recorder.stop_recording.assert_called_once()
            mock_transcribe.assert_called_once()
            mock_speaker.speak.assert_called_once_with(MSG_RECORD_SAVED)

            # Verify the new memory was written to disk and loaded into app.places
            self.assertEqual(len(app.places), 1)
            self.assertEqual(app.places[0].note, "This is the living room where we watch movies.")
            self.assertTrue((Path(tmp) / "recorded_notes.txt").is_file())

    def test_config_loading_from_custom_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "custom.env"
            env_path.write_text(
                "GEMINI_API_KEY=AIzaSyTest123\n"
                "ELEVENLABS_API_KEY=el_test_key_456\n"
                "ELEVENLABS_VOICE_ID=custom_voice_id\n"
                "MODEL=gemini-2.5-pro\n"
                "BLUETOOTH_MAC=aa:bb:cc:dd:ee:ff\n"
                "BUTTON_PIN=17\n"
                "RECORD_BUTTON_PIN=27\n"
                "SPEECH_RATE=155\n"
                "SAVE_LAST_CAPTURE=true\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_config(env_file=env_path, project_root=Path(tmp))
                self.assertEqual(cfg.gemini_api_key, "AIzaSyTest123")
                self.assertEqual(cfg.elevenlabs_api_key, "el_test_key_456")
                self.assertEqual(cfg.elevenlabs_voice_id, "custom_voice_id")
                self.assertEqual(cfg.model, "gemini-2.5-pro")
                self.assertEqual(cfg.bluetooth_mac, "AA:BB:CC:DD:EE:FF")
                self.assertEqual(cfg.audio_device, "bluealsa:DEV=AA:BB:CC:DD:EE:FF,PROFILE=a2dp")
                self.assertEqual(cfg.button_pin, 17)
                self.assertEqual(cfg.record_button_pin, 27)
                self.assertEqual(cfg.speech_rate, 155)
                self.assertTrue(cfg.save_last_capture)

    @patch("memoryaid.speech.requests.post")
    def test_elevenlabs_tts_synthesizes_valid_wav(self, mock_post: MagicMock) -> None:
        fake_pcm = b"\x00\x01" * 1000
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_pcm
        mock_post.return_value = mock_resp

        wav_bytes = synthesize_elevenlabs_wav(
            text="You're at Joe's Cafe.",
            api_key="el_test_key",
            voice_id="21m00Tcm4TlvDq8ikWAM",
        )
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_bytes[:16])
        self.assertGreater(len(wav_bytes), len(fake_pcm))

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
