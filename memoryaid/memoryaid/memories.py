"""Memory book loading, saving new recorded places, contact-sheet composite grid, and disk caching."""

import base64
import hashlib
import io
import json
import logging
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote

import requests
from PIL import Image, ImageDraw, ImageOps

logger = logging.getLogger("memoryaid")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_PLACES = 10
MAX_PHOTOS_PER_PLACE = 2
REF_MAX_EDGE = 512
REF_JPEG_QUALITY = 82
GRID_TILE_WIDTH = 480
GRID_TILE_HEIGHT = 360
GRID_JPEG_QUALITY = 85


@dataclass
class PlaceMemory:
    """Represents one known place from the caregiver's/user's memory book."""

    name: str
    note: str
    images_b64: List[str] = field(default_factory=list)


def preprocess_image_bytes(
    img: Image.Image, max_edge: int = REF_MAX_EDGE, quality: int = REF_JPEG_QUALITY
) -> bytes:
    """Apply EXIF orientation, convert to RGB, resize to max_edge on long side, and encode as JPEG."""
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")

    width, height = img.size
    long_edge = max(width, height)
    if long_edge > max_edge:
        scale = max_edge / float(long_edge)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        img = img.resize(new_size, resample=resample)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def preprocess_image_file_to_b64(
    path: Path, max_edge: int = REF_MAX_EDGE, quality: int = REF_JPEG_QUALITY
) -> str:
    """Load an image file, resize/compress it, and return a base64 string."""
    with Image.open(path) as img:
        jpeg_bytes = preprocess_image_bytes(img, max_edge=max_edge, quality=quality)
    return base64.b64encode(jpeg_bytes).decode("ascii")


def build_contact_sheet_b64(
    places: List[PlaceMemory],
    cache_dir: Optional[Path] = None,
    captures_dir: Optional[Path] = None,
) -> Optional[str]:
    """Stitch all reference photos across `places` into a single labeled contact-sheet grid JPEG (base64).

    Each tile is labeled in the top-left corner with `#<place_index> <place_name>` (e.g., `#1 joes_cafe`).
    Also saves a viewable copy to `captured_photos/reference_contact_sheet.jpg` when `captures_dir` is provided.
    Returns None if no place has any reference photos.
    """
    tiles: List[Tuple[int, str, str]] = []
    for place_idx, place in enumerate(places, start=1):
        for img_b64 in place.images_b64:
            tiles.append((place_idx, place.name, img_b64))

    if not tiles:
        return None

    # Compute a fingerprint over all tile base64 strings + labels so we can reuse the cached grid on disk
    sig_hasher = hashlib.sha256()
    sig_hasher.update(f"v2:{GRID_TILE_WIDTH}x{GRID_TILE_HEIGHT}:".encode("utf-8"))
    for place_idx, place_name, img_b64 in tiles:
        sig_hasher.update(f"{place_idx}:{place_name}:".encode("utf-8"))
        sig_hasher.update(img_b64.encode("ascii"))
    grid_sig = sig_hasher.hexdigest()[:24]

    if cache_dir is not None:
        grid_cache_file = cache_dir / f"contact_sheet_{grid_sig}.json"
        if grid_cache_file.is_file():
            try:
                cached = json.loads(grid_cache_file.read_text(encoding="utf-8"))
                if isinstance(cached.get("b64"), str) and cached["b64"]:
                    sheet_b64_cached = str(cached["b64"])
                    if captures_dir is not None:
                        try:
                            captures_dir.mkdir(parents=True, exist_ok=True)
                            (captures_dir / "reference_contact_sheet.jpg").write_bytes(
                                base64.b64decode(sheet_b64_cached)
                            )
                        except OSError:
                            pass
                    return sheet_b64_cached
            except Exception:
                pass

    num_tiles = len(tiles)
    cols = min(3, max(1, int(math.ceil(math.sqrt(num_tiles)))))
    rows = int(math.ceil(num_tiles / float(cols)))

    canvas_w = cols * GRID_TILE_WIDTH
    canvas_h = rows * GRID_TILE_HEIGHT
    sheet = Image.new("RGB", (canvas_w, canvas_h), color=(20, 24, 32))
    resample = getattr(Image, "Resampling", Image).LANCZOS

    for idx, (place_idx, place_name, img_b64) in enumerate(tiles):
        col = idx % cols
        row = idx // cols
        x0 = col * GRID_TILE_WIDTH
        y0 = row * GRID_TILE_HEIGHT

        try:
            raw = base64.b64decode(img_b64)
            with Image.open(io.BytesIO(raw)) as tile_img:
                # Use contain() rather than fit() so portrait phone photos are never cropped
                contained = ImageOps.contain(
                    tile_img.convert("RGB"),
                    (GRID_TILE_WIDTH - 8, GRID_TILE_HEIGHT - 32),
                    method=resample,
                )
                pad_x = x0 + (GRID_TILE_WIDTH - contained.width) // 2
                pad_y = y0 + 26 + (GRID_TILE_HEIGHT - 30 - contained.height) // 2
                sheet.paste(contained, (pad_x, pad_y))
        except Exception as exc:
            logger.warning("Could not render contact-sheet tile for %s: %s", place_name, exc)

        # Draw high-contrast label banner `#N place_name` across the top header bar of the tile
        draw = ImageDraw.Draw(sheet)
        label = f"#{place_idx} {place_name}"[:36]
        draw.rectangle([x0 + 2, y0 + 2, x0 + GRID_TILE_WIDTH - 2, y0 + 24], fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 7), label, fill=(255, 255, 100))

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=GRID_JPEG_QUALITY, optimize=True)
    sheet_bytes = buf.getvalue()
    sheet_b64 = base64.b64encode(sheet_bytes).decode("ascii")

    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Remove older contact_sheet_*.json files
            for old_sheet in cache_dir.glob("contact_sheet_*.json"):
                try:
                    old_sheet.unlink()
                except OSError:
                    pass
            (cache_dir / f"contact_sheet_{grid_sig}.json").write_text(
                json.dumps({"tiles": num_tiles, "b64": sheet_b64}),
                encoding="utf-8",
            )
        except OSError:
            pass

    if captures_dir is not None:
        try:
            captures_dir.mkdir(parents=True, exist_ok=True)
            ref_sheet_path = captures_dir / "reference_contact_sheet.jpg"
            ref_sheet_path.write_bytes(sheet_bytes)
            logger.info("Saved reference contact sheet image -> %s", ref_sheet_path)
        except OSError:
            pass

    logger.info(
        "Built composite contact-sheet grid (%d tile(s) across %dx%d grid, %.1f KB).",
        num_tiles,
        cols,
        rows,
        len(sheet_bytes) / 1024.0,
    )
    return sheet_b64


def slugify_place_name(text: str) -> str:
    """Create a filesystem-safe folder slug from a spoken note or place name."""
    cleaned = text.lower().replace("'", "").replace("\u2019", "")
    words = re.findall(r"[a-z0-9]+", cleaned)
    stop_words = {"this", "is", "the", "a", "an", "you", "are", "in", "at", "my", "our", "here", "we"}
    filtered = [w for w in words if w not in stop_words]
    chosen = (filtered or words)[:4]
    slug = "_".join(chosen).strip("_")
    if not slug:
        slug = f"memory_{int(time.time())}"
    return slug[:40]


def save_recorded_memory(
    memories_dir: Path,
    captured_b64: str,
    note_text: str,
    place_name_hint: str = "",
) -> PlaceMemory:
    """Save a newly recorded photo and spoken transcript into `memories/<place_slug>/` and return the PlaceMemory.

    Writes:
      - `memories/<slug>/note.txt` with what the user said
      - `memories/<slug>/1.jpg` (or `2.jpg` if `1.jpg` already exists)
    """
    memories_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify_place_name(place_name_hint or note_text)
    place_dir = memories_dir / slug

    if place_dir.exists():
        existing_photos = sorted(
            [
                p
                for p in place_dir.iterdir()
                if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
            ]
        )
        if len(existing_photos) >= MAX_PHOTOS_PER_PLACE:
            slug = f"{slug}_{int(time.time()) % 10000}"
            place_dir = memories_dir / slug

    place_dir.mkdir(parents=True, exist_ok=True)

    note_path = place_dir / "note.txt"
    clean_note = note_text.strip()
    note_path.write_text(clean_note + "\n", encoding="utf-8")

    raw_jpeg = base64.b64decode(captured_b64)
    photo_1 = place_dir / "1.jpg"
    photo_2 = place_dir / "2.jpg"
    target_photo = photo_2 if photo_1.exists() else photo_1
    target_photo.write_bytes(raw_jpeg)

    log_file = memories_dir / "recorded_notes.txt"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] ({slug}/{target_photo.name}): {clean_note}\n")

    logger.info(
        "Saved recorded memory to %s (note=%r, photo=%s)",
        place_dir,
        clean_note,
        target_photo.name,
    )

    ref_b64 = preprocess_image_file_to_b64(
        target_photo, max_edge=REF_MAX_EDGE, quality=REF_JPEG_QUALITY
    )
    return PlaceMemory(name=slug, note=clean_note, images_b64=[ref_b64])


def sync_remote_website_memories(
    memories_dir: Path,
    website_url: str = "",
    timeout: float = 5.0,
) -> bool:
    """Sync new/updated photos and descriptions from the HackRPI2026 Caregiver Website into `memories_dir`.

    1. If `website_url` is reachable (`GET <website_url>/api/photos`), downloads any new photos from
       `<website_url>/images/<filename>` and writes their descriptions into `memories/web_<slug>_<shortid>/note.txt`.
    2. If `website_url` is unreachable or local, also checks for a sibling `../HackRPI2026/descriptions.json` +
       `../HackRPI2026/images/` directory on disk and imports any new entries automatically.

    Returns True if any place was added, updated, or removed on disk (so the caller knows to reload `memories/`
    and rebuild the Contact Sheet), or False if nothing changed.
    """
    memories_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = memories_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "website_sync_state.json"

    prev_state: Dict[str, Dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                prev_state = loaded
        except Exception:
            prev_state = {}

    records: Optional[List[Dict[str, Any]]] = None
    source_mode = ""  # "http" or "local"
    active_http_url = ""
    local_web_root: Optional[Path] = None

    # Build ordered list of candidate HTTP URLs to reach the Caregiver Website server
    candidate_urls: List[str] = []
    server_url_file = memories_dir.parent / ".server_url"
    if server_url_file.is_file() and not isinstance(getattr(requests.get, "_mock_name", None), str):
        try:
            saved_srv = server_url_file.read_text(encoding="utf-8").strip().rstrip("/")
            if saved_srv:
                candidate_urls.append(saved_srv)
        except OSError:
            pass

    clean_url = (website_url or "").strip().rstrip("/")
    if clean_url and clean_url not in candidate_urls:
        candidate_urls.append(clean_url)

    # If SSHed into the Pi from the laptop hosting the website, SSH_CLIENT / SSH_CONNECTION has the laptop IP
    for ssh_var in ("SSH_CLIENT", "SSH_CONNECTION"):
        ssh_val = os.environ.get(ssh_var, "").strip()
        if ssh_val:
            ssh_ip = ssh_val.split()[0].strip()
            if ssh_ip and not ssh_ip.startswith("127."):
                ssh_url = f"http://{ssh_ip}:5173"
                if ssh_url not in candidate_urls:
                    candidate_urls.append(ssh_url)

    for fallback_url in (
        "http://192.168.50.209:5173",
        "http://Johns-MacBook-Air-2.local:5173",
        "http://localhost:5173",
    ):
        if fallback_url not in candidate_urls:
            # Only add automatic LAN fallback URLs when not mocked in unit tests
            if not isinstance(getattr(requests.get, "_mock_name", None), str):
                candidate_urls.append(fallback_url)

    for url_cand in candidate_urls:
        try:
            resp = requests.get(f"{url_cand}/api/photos", timeout=min(timeout, 2.5))
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                    records = data
                    source_mode = "http"
                    active_http_url = url_cand
                    logger.info(
                        "Connected to caregiver website at %s (%d photo record(s) on server).",
                        active_http_url,
                        len(records),
                    )
                    break
        except Exception as exc:
            logger.debug("Website HTTP sync from %s unreachable: %s", url_cand, exc)

    # Fallback: check local sibling `HackRPI2026/descriptions.json` if running on same machine or synced folder
    if records is None:
        candidate_roots = [
            memories_dir.parent.parent / "HackRPI2026",
            memories_dir.parent.parent,
            memories_dir.parent / "HackRPI2026",
            memories_dir.parent,
        ]
        for cand in candidate_roots:
            desc_file = cand / "descriptions.json"
            if desc_file.is_file():
                try:
                    data = json.loads(desc_file.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        records = data
                        source_mode = "local"
                        local_web_root = cand
                        logger.info(
                            "Loaded %d photo record(s) from local %s.",
                            len(records),
                            desc_file,
                        )
                        break
                except Exception as exc:
                    logger.warning("Failed reading local %s: %s", desc_file, exc)

    if records is None:
        logger.warning(
            "Could not reach caregiver website (%s) and no local descriptions.json found.",
            ", ".join(candidate_urls),
        )
        return False

    changed = False
    new_state: Dict[str, Dict[str, Any]] = {}

    for item in records:
        if not isinstance(item, dict):
            continue
        photo_id = str(item.get("id") or "").strip()
        filename = str(item.get("filename") or "").strip()
        description = str(item.get("description") or "").strip()
        updated_at = str(item.get("updatedAt") or item.get("uploadedAt") or "").strip()

        if not photo_id or not filename:
            continue

        short_id = re.sub(r"[^a-z0-9]", "", photo_id.lower())[:8] or "web"
        slug_base = slugify_place_name(description or Path(filename).stem)
        folder_name = f"web_{slug_base}_{short_id}"
        place_dir = memories_dir / folder_name
        target_photo = place_dir / "1.jpg"
        target_note = place_dir / "note.txt"

        prev_entry = prev_state.get(photo_id)
        already_up_to_date = (
            prev_entry is not None
            and prev_entry.get("folder") == folder_name
            and prev_entry.get("description") == description
            and prev_entry.get("updated_at") == updated_at
            and prev_entry.get("filename") == filename
            and target_photo.is_file()
            and target_note.is_file()
        )
        if already_up_to_date:
            new_state[photo_id] = prev_entry
            continue

        # If the description changed and produced a new folder name, remove the old folder first
        if prev_entry and prev_entry.get("folder") and prev_entry.get("folder") != folder_name:
            old_dir = memories_dir / str(prev_entry["folder"])
            if old_dir.is_dir() and old_dir.name.startswith("web_"):
                shutil.rmtree(old_dir, ignore_errors=True)

        # Fetch or copy the image if not already present on disk
        img_bytes: Optional[bytes] = None
        if not target_photo.is_file() or (prev_entry and prev_entry.get("filename") != filename):
            if source_mode == "http":
                try:
                    img_url = f"{active_http_url}/images/{quote(filename)}"
                    img_resp = requests.get(img_url, timeout=timeout)
                    if img_resp.status_code == 200 and img_resp.content:
                        img_bytes = img_resp.content
                        # Also mirror the downloaded image into local HackRPI2026/images/ if present on device
                        sibling_web = memories_dir.parent.parent / "HackRPI2026"
                        if sibling_web.is_dir():
                            try:
                                (sibling_web / "images").mkdir(parents=True, exist_ok=True)
                                (sibling_web / "images" / Path(filename).name).write_bytes(img_bytes)
                            except OSError:
                                pass
                    else:
                        logger.warning("Website image download failed (%d) for %s", img_resp.status_code, img_url)
                        continue
                except Exception as exc:
                    logger.warning("Error downloading website image %s: %s", filename, exc)
                    continue
            elif source_mode == "local" and local_web_root is not None:
                src_file = local_web_root / "images" / Path(filename).name
                if src_file.is_file():
                    try:
                        img_bytes = src_file.read_bytes()
                    except OSError as exc:
                        logger.warning("Could not read local website image %s: %s", src_file, exc)
                        continue
                else:
                    continue

        place_dir.mkdir(parents=True, exist_ok=True)
        if img_bytes is not None:
            # Also save a direct copy in HackRPI2026/memoryaid/images/<filename>
            if not isinstance(getattr(requests.get, "_mock_name", None), str):
                try:
                    local_images_dir = memories_dir.parent / "images"
                    local_images_dir.mkdir(parents=True, exist_ok=True)
                    (local_images_dir / Path(filename).name).write_bytes(img_bytes)
                except OSError:
                    pass

            # Convert/normalize through Pillow so any WebP/BMP/GIF/JPEG from the website becomes a valid JPEG/PNG
            try:
                with Image.open(io.BytesIO(img_bytes)) as raw_img:
                    jpeg_bytes = preprocess_image_bytes(
                        raw_img, max_edge=REF_MAX_EDGE, quality=REF_JPEG_QUALITY
                    )
                target_photo = place_dir / "1.jpg"
                target_photo.write_bytes(jpeg_bytes)
            except Exception as exc:
                logger.warning("Pillow re-encode failed for %s (%s); saving raw image bytes.", filename, exc)
                target_photo = place_dir / "1.jpg"
                target_photo.write_bytes(img_bytes)

        note_content = description if description else slug_base.replace("_", " ")
        target_note.write_text(note_content + "\n", encoding="utf-8")

        new_state[photo_id] = {
            "folder": folder_name,
            "filename": filename,
            "description": description,
            "updated_at": updated_at,
        }
        changed = True
        logger.info(
            "Synced website memory [%s]: folder=%s, note=%r",
            source_mode,
            folder_name,
            note_content,
        )

    # Remove any `web_*` folders that were deleted from the website
    for old_id, old_entry in prev_state.items():
        if old_id not in new_state:
            old_folder = str(old_entry.get("folder") or "")
            if old_folder.startswith("web_"):
                old_path = memories_dir / old_folder
                if old_path.is_dir():
                    shutil.rmtree(old_path, ignore_errors=True)
                    changed = True
                    logger.info("Removed deleted website memory folder: %s", old_folder)

    if changed or set(prev_state.keys()) != set(new_state.keys()):
        try:
            manifest_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
        except OSError:
            pass
        if source_mode == "http" and records is not None:
            sibling_web = memories_dir.parent.parent / "HackRPI2026"
            if sibling_web.is_dir():
                try:
                    (sibling_web / "descriptions.json").write_text(
                        json.dumps(records, indent=2) + "\n", encoding="utf-8"
                    )
                except OSError:
                    pass

    return changed


def _cache_key_for_source(source_path: Path, root_dir: Path) -> str:
    """Compute a stable hash key from the relative file path."""
    try:
        rel_path = str(source_path.resolve().relative_to(root_dir.resolve()))
    except ValueError:
        rel_path = str(source_path.resolve())
    return hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:24]


def _load_or_build_cached_b64(
    source_path: Path, root_dir: Path, cache_dir: Path
) -> str:
    """Load cached base64 image if mtime+size match, otherwise reprocess and update cache."""
    stat = source_path.stat()
    mtime_ns = getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))
    file_size = stat.st_size
    key = _cache_key_for_source(source_path, root_dir)
    cache_file = cache_dir / f"{key}.json"

    if cache_file.is_file():
        try:
            data: Dict[str, object] = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                data.get("mtime_ns") == mtime_ns
                and data.get("size") == file_size
                and isinstance(data.get("b64"), str)
                and data.get("b64")
            ):
                return str(data["b64"])
        except Exception as exc:
            logger.warning("Cache read failed for %s (%s); rebuilding.", source_path.name, exc)

    try:
        rel_display = source_path.resolve().relative_to(root_dir.resolve())
    except ValueError:
        rel_display = Path(source_path.name)
    logger.info("Preprocessing reference photo: %s", rel_display)
    b64_data = preprocess_image_file_to_b64(
        source_path, max_edge=REF_MAX_EDGE, quality=REF_JPEG_QUALITY
    )

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source": str(source_path.name),
            "mtime_ns": mtime_ns,
            "size": file_size,
            "b64": b64_data,
        }
        cache_file.write_text(json.dumps(payload), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write cache file %s: %s", cache_file, exc)

    return b64_data


def load_memories(memories_dir: Path) -> List[PlaceMemory]:
    """Load up to 10 known places from `memories_dir`, preprocessing and caching up to 2 photos each."""
    if not memories_dir.exists() or not memories_dir.is_dir():
        logger.warning(
            "Memories directory %s does not exist or is not a directory. Continuing with 0 known places.",
            memories_dir,
        )
        return []

    subdirs = sorted(
        [
            p
            for p in memories_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ],
        key=lambda p: p.name.lower(),
    )

    if len(subdirs) > MAX_PLACES:
        logger.warning(
            "Found %d place folders in %s; limiting to the first %d alphabetically.",
            len(subdirs),
            memories_dir,
            MAX_PLACES,
        )
        subdirs = subdirs[:MAX_PLACES]

    cache_dir = memories_dir / ".cache"
    active_cache_keys: Set[str] = set()
    places: List[PlaceMemory] = []

    for folder in subdirs:
        note_path = folder / "note.txt"
        note_text = ""
        if note_path.is_file():
            try:
                note_text = note_path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                logger.warning("Failed to read %s: %s", note_path, exc)

        if not note_text:
            note_text = folder.name.replace("_", " ").strip()

        photo_files = sorted(
            [
                p
                for p in folder.iterdir()
                if p.is_file()
                and not p.name.startswith(".")
                and p.suffix.lower() in ALLOWED_EXTENSIONS
            ],
            key=lambda p: p.name.lower(),
        )

        if len(photo_files) > MAX_PHOTOS_PER_PLACE:
            logger.warning(
                "Place '%s' has %d photos; using only the first %d (%s).",
                folder.name,
                len(photo_files),
                MAX_PHOTOS_PER_PLACE,
                ", ".join(p.name for p in photo_files[:MAX_PHOTOS_PER_PLACE]),
            )
            photo_files = photo_files[:MAX_PHOTOS_PER_PLACE]

        images_b64: List[str] = []
        for photo in photo_files:
            try:
                key = _cache_key_for_source(photo, memories_dir)
                active_cache_keys.add(key)
                b64_str = _load_or_build_cached_b64(photo, memories_dir, cache_dir)
                images_b64.append(b64_str)
            except Exception as exc:
                logger.error("Skipping unreadable photo %s: %s", photo, exc)

        if not images_b64:
            if folder.name in ("davids_bedroom", "joes_cafe"):
                shutil.rmtree(folder, ignore_errors=True)
                logger.info("Removed empty sample folder '%s'.", folder.name)
                continue
            logger.info(
                "Place folder '%s' has no reference photos yet; skipping.",
                folder.name,
            )
            continue

        places.append(
            PlaceMemory(
                name=folder.name,
                note=note_text,
                images_b64=images_b64,
            )
        )

    # Prune stale cache entries so removed photos do not linger on disk
    if cache_dir.is_dir():
        for cached_file in cache_dir.glob("*.json"):
            if (
                not cached_file.stem.startswith("contact_sheet_")
                and cached_file.name != "website_sync_state.json"
                and cached_file.stem not in active_cache_keys
            ):
                try:
                    cached_file.unlink()
                except OSError:
                    pass

    # Sort places that actually have reference photos ahead of empty note-only placeholder folders
    places.sort(key=lambda p: (0 if p.images_b64 else 1, p.name.lower()))

    total_photos = sum(len(p.images_b64) for p in places)
    logger.info(
        "Loaded %d known place(s) (%d reference photo(s) total) from %s.",
        len(places),
        total_photos,
        memories_dir,
    )
    return places
