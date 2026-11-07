# Memory Book (`memories/`)

Each subfolder inside `memories/` represents one familiar place in the person's life.

## Folder Structure

```text
memories/
  joes_cafe/
    note.txt
    1.jpg
    2.jpg
  davids_bedroom/
    note.txt
    1.jpg
```

## Rules & Limits

- **Up to 10 places**: Folders are loaded in alphabetical order. If more than 10 folders exist, the first 10 alphabetically are used.
- **1 to 2 photos per place**: Place `.jpg`, `.jpeg`, or `.png` images inside each folder. Up to 2 photos (sorted alphabetically by filename) are loaded per place.
- **`note.txt`**: A short plain-text note describing what the place is, who is there, and the person's routine or connection to it. If `note.txt` is missing or empty, the folder name (with underscores replaced by spaces) is used as the description.
- **Automatic caching**: On startup, Memory Aid resizes reference photos to at most 512 px on the long edge (JPEG quality 75, EXIF orientation applied) and caches them in `memories/.cache/`. Only new or modified photos are reprocessed.

## How to Add or Update a Place

1. Create a new folder under `memories/` with a simple name (for example, `memories/living_room`).
2. Add 1 or 2 clear, wide-angle photos taken from where the person usually stands or sits (`1.jpg`, `2.jpg`).
3. Create `note.txt` inside that folder with 1–3 comforting sentences including names and routines.
4. Restart the service so the new place is cached and loaded:
   ```bash
   sudo systemctl restart memoryaid
   ```
