"""Memory book loading, saving new recorded places, reference image preprocessing, and disk caching."""

import base64
import hashlib
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

from PIL import Image, ImageOps

logger = logging.getLogger("memoryaid")

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_PLACES = 10
MAX_PHOTOS_PER_PLACE = 2
REF_MAX_EDGE = 512
REF_JPEG_QUALITY = 75


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

    # If the folder already has 2 photos and a different note, create a uniquely numbered folder
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

    # Write note.txt
    note_path = place_dir / "note.txt"
    clean_note = note_text.strip()
    note_path.write_text(clean_note + "\n", encoding="utf-8")

    # Decode the captured JPEG and save as 1.jpg (or 2.jpg)
    raw_jpeg = base64.b64decode(captured_b64)
    photo_1 = place_dir / "1.jpg"
    photo_2 = place_dir / "2.jpg"
    target_photo = photo_2 if photo_1.exists() else photo_1
    target_photo.write_bytes(raw_jpeg)

    # Also append to a master log file `memories/recorded_notes.txt` so every entry is easily readable in one file
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

        places.append(
            PlaceMemory(
                name=folder.name,
                note=note_text,
                images_b64=images_b64,
            )
        )
        if not images_b64:
            logger.info(
                "Place folder '%s' has no reference photos yet; loaded caregiver note only.",
                folder.name,
            )

    # Prune stale cache entries so removed photos do not linger on disk
    if cache_dir.is_dir():
        for cached_file in cache_dir.glob("*.json"):
            if cached_file.stem not in active_cache_keys:
                try:
                    cached_file.unlink()
                except OSError:
                    pass

    total_photos = sum(len(p.images_b64) for p in places)
    logger.info(
        "Loaded %d known place(s) (%d reference photo(s) total) from %s.",
        len(places),
        total_photos,
        memories_dir,
    )
    return places
