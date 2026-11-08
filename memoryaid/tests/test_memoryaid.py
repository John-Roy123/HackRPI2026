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
    MSG_RECORD_ACTIVE,
    MSG_RECORD_SAVED,
    MSG_RECORD_UPLOADING,
    MemoryAidApp,
    map_error_to_spoken_message,
)
from memoryaid.camera import CameraError, MockCamera
from memoryaid.config import Config, load_config
from memoryaid.memories import (
    PlaceMemory,
    load_memories,
    save_recorded_memory,
    sync_remote_website_memories,
)
from memoryaid.speech import normalize_electret_wav_bytes, synthesize_fish_audio_wav
from memoryaid.vision import (
    MSG_UNRECOGNIZED_PLACE,
    SYSTEM_PROMPT,
    VisionAPIError,
    build_messages_payload,
    build_transcription_payload,
    clean_spoken_text,
    get_gemini_endpoint,
    query_vision_api,
    resolve_exact_place_description,
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

    @patch("memoryaid.memories.requests.post")
    def test_save_recorded_memory_writes_photo_and_note_files(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.json.return_value = {
            "id": "pi-upload-id-1234",
            "filename": "sarahs_kitchen_where_bake-pi-uploa.jpg",
            "description": "This is Sarah's kitchen where we bake cookies every Sunday.",
            "uploadedAt": "2026-11-08T12:00:00.000Z",
        }
        mock_post.return_value = mock_resp

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b64_img = _create_dummy_jpeg_b64()
            spoken = "This is Sarah's kitchen where we bake cookies every Sunday."

            place = save_recorded_memory(
                root, b64_img, spoken, website_url="http://localhost:5173"
            )
            self.assertEqual(place.name, "sarahs_kitchen_where_bake")
            self.assertTrue((root / place.name / "note.txt").is_file())
            self.assertTrue((root / place.name / "1.jpg").is_file())
            self.assertTrue((root / "recorded_notes.txt").is_file())
            self.assertEqual(
                (root / place.name / "note.txt").read_text(encoding="utf-8").strip(),
                spoken,
            )
            # Verify the photo and description were uploaded to the Caregiver Website
            mock_post.assert_called_once()
            upload_url = mock_post.call_args[0][0]
            self.assertEqual(upload_url, "http://localhost:5173/api/photos")
            self.assertEqual(
                mock_post.call_args[1]["params"],
                {
                    "name": "sarahs_kitchen_where_bake.jpg",
                    "description": spoken,
                },
            )
            self.assertEqual(
                mock_post.call_args[1]["headers"]["Content-Type"], "image/jpeg"
            )
            self.assertEqual(
                mock_post.call_args[1]["data"], base64.b64decode(b64_img)
            )

    @patch("memoryaid.memories.requests.get")
    def test_sync_remote_website_memories_downloads_new_and_updates_changed(
        self, mock_get: MagicMock
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "memories"
            raw_jpeg = base64.b64decode(_create_dummy_jpeg_b64((640, 480), (180, 120, 60)))

            resp_list = MagicMock()
            resp_list.status_code = 200
            resp_list.json.return_value = [
                {
                    "id": "f912baba-27d0-4231-8209-23bce84d00e8",
                    "filename": "pxl-20261108-f912baba.jpg",
                    "description": "This is the living room, where you watch football and drink beer!",
                    "uploadedAt": "2026-11-08T01:25:46.366Z",
                }
            ]

            resp_img = MagicMock()
            resp_img.status_code = 200
            resp_img.content = raw_jpeg

            mock_get.side_effect = [resp_list, resp_img]

            # 1st sync: should download the new photo and return True
            changed = sync_remote_website_memories(root, website_url="http://localhost:5173")
            self.assertTrue(changed)

            places = load_memories(root)
            self.assertEqual(len(places), 1)
            self.assertTrue(places[0].name.startswith("web_"))
            self.assertEqual(
                places[0].note,
                "This is the living room, where you watch football and drink beer!",
            )
            self.assertEqual(len(places[0].images_b64), 1)

            # 2nd sync with identical data: should NOT re-download the image and should return False
            mock_get.reset_mock()
            mock_get.side_effect = [resp_list]
            changed_second = sync_remote_website_memories(root, website_url="http://localhost:5173")
            self.assertFalse(changed_second)
            self.assertEqual(mock_get.call_count, 1)


class TestVisionPayloadAndCleaning(unittest.TestCase):
    """Tests for Gemini API payload structure, transcription payload, markdown stripping, and HTTP retry logic."""

    def test_request_payload_structure_order(self) -> None:
        ref1 = _create_dummy_jpeg_b64((320, 240), (200, 50, 50))
        ref2 = _create_dummy_jpeg_b64((320, 240), (50, 200, 50))
        ref3 = _create_dummy_jpeg_b64((320, 240), (50, 50, 200))
        places = [
            PlaceMemory(name="joes_cafe", note="Coffee with Maria.", images_b64=[ref1, ref2]),
            PlaceMemory(name="davids_bedroom", note="David is your son.", images_b64=[ref3]),
        ]
        payload = build_messages_payload(places=places, captured_b64="NEW_CAM_B64", model="gemini-2.5-flash")

        self.assertEqual(
            get_gemini_endpoint("gemini-2.5-flash"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
        )
        self.assertEqual(
            get_gemini_endpoint("claude-haiku-5-5"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent",
        )
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 2048)
        self.assertEqual(payload["system_instruction"]["parts"][0]["text"], SYSTEM_PROMPT)
        self.assertEqual(len(payload["contents"]), 1)
        self.assertEqual(payload["contents"][0]["role"], "user")

        parts = payload["contents"][0]["parts"]
        # Expected sequence with Option 2 Contact Sheet:
        # 0: text (Known places list)
        # 1: text (Reference Contact Sheet header)
        # 2: inline_data (Single stitched Contact Sheet image containing all 3 reference photos)
        # 3: text ("NEW photo from the camera right now:")
        # 4: inline_data (NEW_CAM_B64)
        # 5: text (Comparison instruction)
        self.assertEqual(len(parts), 6)
        self.assertIn("#1 (joes_cafe): Coffee with Maria.", parts[0]["text"])
        self.assertIn("#2 (davids_bedroom): David is your son.", parts[0]["text"])
        self.assertIn("Reference Contact Sheet", parts[1]["text"])
        self.assertEqual(parts[2]["inline_data"]["mime_type"], "image/jpeg")
        self.assertGreater(len(parts[2]["inline_data"]["data"]), 100)
        self.assertEqual(parts[3], {"text": "NEW photo from the camera right now:"})
        self.assertEqual(parts[4]["inline_data"]["data"], "NEW_CAM_B64")
        self.assertIn("exclusively the matching caregiver note", parts[5]["text"])

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

    def test_resolve_exact_place_description_returns_stored_json_description(self) -> None:
        places = [
            PlaceMemory(
                name="web_living_room_f912baba",
                note="This is the living room, where you watch football and drink beer!",
                images_b64=["IMG1"],
            ),
            PlaceMemory(
                name="web_your_kitchen_8ba4aa57",
                note="This your the kitchen, where you cook food and talk to your wife.",
                images_b64=["IMG2"],
            ),
        ]
        # Case 1: Model outputs #2 or stripped "2 ..."
        self.assertEqual(
            resolve_exact_place_description("#2 This your the kitchen", places),
            "This your the kitchen, where you cook food and talk to your wife.",
        )
        self.assertEqual(
            resolve_exact_place_description("2 This your the kitchen", places),
            "This your the kitchen, where you cook food and talk to your wife.",
        )
        # Case 2: Model paraphrases ("You are in the kitchen") -> still returns exact JSON description
        self.assertEqual(
            resolve_exact_place_description("You are in the kitchen.", places),
            "This your the kitchen, where you cook food and talk to your wife.",
        )
        # Case 3: Model outputs NONE
        self.assertEqual(
            resolve_exact_place_description("NONE", places),
            MSG_UNRECOGNIZED_PLACE,
        )

    @patch("memoryaid.vision.requests.get")
    @patch("memoryaid.vision.requests.post")
    def test_query_vision_api_falls_back_on_404_or_claude_model_name(self, mock_post: MagicMock, mock_get: MagicMock) -> None:
        import memoryaid.vision as vision_mod
        vision_mod._DISCOVERED_MODEL = ""
        vision_mod._FAILED_404_MODELS.clear()
        mock_get_resp = MagicMock()
        mock_get_resp.status_code = 200
        mock_get_resp.json.return_value = {"models": [{"name": "models/gemini-flash-latest", "supportedGenerationMethods": ["generateContent"]}]}
        mock_get.return_value = mock_get_resp

        resp_404 = MagicMock()
        resp_404.status_code = 404
        resp_404.text = "models/gemini-3-flash-preview not found"

        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "You're in David's bedroom."}]
                    }
                }
            ]
        }

        mock_post.side_effect = [resp_404, resp_200]
        payload = build_messages_payload([], "NEW_B64")

        text, _size_bytes, _latency = query_vision_api(
            payload, api_key="test-key", model="claude-haiku-5-5", timeout=5, retry_delay=0.01
        )
        self.assertEqual(text, "You're in David's bedroom.")
        self.assertEqual(mock_post.call_count, 2)
        vision_mod._DISCOVERED_MODEL = ""
        vision_mod._FAILED_404_MODELS.clear()

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

    @patch("memoryaid.memories.requests.post")
    @patch("memoryaid.app.transcribe_audio_with_gemini")
    def test_successful_hold_to_record_button_flow(
        self, mock_transcribe: MagicMock, mock_upload_post: MagicMock
    ) -> None:
        mock_transcribe.return_value = "This is the living room where we watch movies."
        mock_upload_resp = MagicMock()
        mock_upload_resp.status_code = 201
        mock_upload_resp.json.return_value = {
            "id": "pi-living-room-123",
            "filename": "living_room_where_watch-pi-livin.jpg",
            "description": "This is the living room where we watch movies.",
            "uploadedAt": "2026-11-08T12:00:00.000Z",
        }
        mock_upload_post.return_value = mock_upload_resp
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

            # 1st press of Button 1: speaks active prompt, captures photo, starts recording
            app.handle_record_press()
            mock_speaker.speak.assert_called_once_with(MSG_RECORD_ACTIVE)
            mock_recorder.start_recording.assert_called_once()
            mock_cam.capture_base64.assert_called_once()

            # 2nd press of Button 1: stops recording, transcribes, saves locally, speaks saved & uploading prompts, uploads
            app.handle_record_press()
            mock_recorder.stop_recording.assert_called_once()
            mock_transcribe.assert_called_once()
            self.assertEqual(
                [c.args[0] for c in mock_speaker.speak.call_args_list],
                [MSG_RECORD_ACTIVE, MSG_RECORD_SAVED, MSG_RECORD_UPLOADING],
            )
            mock_upload_post.assert_called_once()

            # Verify the new memory was written to disk and loaded into app.places
            self.assertEqual(len(app.places), 1)
            self.assertEqual(app.places[0].note, "This is the living room where we watch movies.")
            self.assertTrue((Path(tmp) / "recorded_notes.txt").is_file())

    def test_config_loading_from_custom_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / "custom.env"
            env_path.write_text(
                "GEMINI_API_KEY=AIzaSyTest123\n"
                "FISH_API_KEY=fish_test_key_456\n"
                "FISH_REFERENCE_ID=fish_voice_ref_789\n"
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
                self.assertEqual(cfg.fish_api_key, "fish_test_key_456")
                self.assertEqual(cfg.fish_reference_id, "fish_voice_ref_789")
                self.assertEqual(cfg.model, "gemini-2.5-pro")
                self.assertEqual(cfg.bluetooth_mac, "AA:BB:CC:DD:EE:FF")
                self.assertEqual(cfg.audio_device, "bluealsa:DEV=AA:BB:CC:DD:EE:FF,PROFILE=a2dp")
                self.assertEqual(cfg.button_pin, 17)
                self.assertEqual(cfg.record_button_pin, 27)
                self.assertEqual(cfg.speech_rate, 155)
                self.assertTrue(cfg.save_last_capture)

    @patch("memoryaid.speech.requests.post")
    def test_fish_audio_tts_synthesizes_valid_wav(self, mock_post: MagicMock) -> None:
        fake_wav = b"RIFF" + b"\x00" * 100
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = fake_wav
        mock_post.return_value = mock_resp

        wav_bytes = synthesize_fish_audio_wav(
            text="You're at Joe's Cafe.",
            api_key="fish_test_key",
            reference_id="ref_123",
        )
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertEqual(len(wav_bytes), len(fake_wav))

    def test_mock_camera_loads_and_resizes_image(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            img_path = tmp_dir / "test.jpg"
            last_cap = tmp_dir / "last_capture.jpg"
            cap_dir = tmp_dir / "captured_photos"
            _create_dummy_image(img_path, size=(1280, 960), color=(50, 100, 150))
            cam = MockCamera(
                image_path=img_path,
                save_last_capture=True,
                last_capture_path=last_cap,
                captures_dir=cap_dir,
            )
            cam.start()
            b64 = cam.capture_base64()
            raw = base64.b64decode(b64)
            with Image.open(io.BytesIO(raw)) as img:
                self.assertEqual(img.size, (640, 480))
            self.assertTrue(last_cap.is_file())
            self.assertTrue((cap_dir / "latest.jpg").is_file())
            self.assertEqual(len(list(cap_dir.glob("capture_*.jpg"))), 1)

    def test_electret_mic_wav_normalization_removes_dc_and_boosts_gain(self) -> None:
        import array
        import wave

        # Create a 16-bit mono WAV with a +1000 DC offset and quiet speech signal (+/- 500)
        samples = array.array("h", [1000 + (500 if i % 2 == 0 else -500) for i in range(400)])
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples.tobytes())

        normalized = normalize_electret_wav_bytes(buf.getvalue())
        with wave.open(io.BytesIO(normalized), "rb") as out_wf:
            out_samples = array.array("h")
            out_samples.frombytes(out_wf.readframes(out_wf.getnframes()))

        # DC offset should be ~0 and peak should be boosted significantly above 500
        self.assertAlmostEqual(sum(out_samples) / len(out_samples), 0.0, delta=5.0)
        self.assertGreater(max(abs(s) for s in out_samples), 4000)


if __name__ == "__main__":
    unittest.main()
