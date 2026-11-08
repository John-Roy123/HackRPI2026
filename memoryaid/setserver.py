#!/usr/bin/env python3
"""Programmatically set the Caregiver Website server IP/URL for Memory Aid and sync photos immediately.

Usage:
    python3 setserver.py "192.168.50.209"
    python3 setserver.py "192.168.50.209:5173"
    python3 setserver.py "http://192.168.50.209:5173"
"""

import argparse
import re
import sys
from pathlib import Path


def normalize_server_url(raw_input: str, default_port: int = 5173) -> str:
    """Convert an IP address, hostname, or URL into a clean `http://<host>:<port>` base URL."""
    val = raw_input.strip().strip("'\"")
    if not val:
        raise ValueError("Server IP or URL cannot be empty.")

    if not val.startswith(("http://", "https://")):
        val = f"http://{val}"

    val = val.rstrip("/")

    # Add :5173 if no port is specified after the host
    scheme, rest = val.split("://", 1)
    host_part = rest.split("/", 1)[0]
    if ":" not in host_part:
        val = f"{scheme}://{host_part}:{default_port}"

    return val


def _update_env_file(env_path: Path, server_url: str) -> bool:
    """Update or append WEBSITE_URL=<server_url> in an env file if it exists and is writable."""
    if not env_path.is_file():
        return False
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("WEBSITE_URL="):
                new_lines.append(f"WEBSITE_URL={server_url}")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"WEBSITE_URL={server_url}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return True
    except OSError:
        return False


def _update_source_defaults(memoryaid_root: Path, server_url: str) -> None:
    """Update the default fallback URL in memoryaid/config.py and memoryaid/memories.py."""
    config_py = memoryaid_root / "memoryaid" / "config.py"
    if config_py.is_file():
        text = config_py.read_text(encoding="utf-8")
        text = re.sub(
            r'website_url:\s*str\s*=\s*"http[^"]*"',
            f'website_url: str = "{server_url}"',
            text,
        )
        text = re.sub(
            r'merged\.get\("WEBSITE_URL",\s*"http[^"]*"\)',
            f'merged.get("WEBSITE_URL", "{server_url}")',
            text,
        )
        text = re.sub(
            r'website_url\s*=\s*"http://[^"]+"',
            f'website_url = "{server_url}"',
            text,
        )
        config_py.write_text(text, encoding="utf-8")

    memories_py = memoryaid_root / "memoryaid" / "memories.py"
    if memories_py.is_file():
        text = memories_py.read_text(encoding="utf-8")
        text = re.sub(
            r'for fallback_url in \(\s*"http://[^"]+",',
            f'for fallback_url in (\n        "{server_url}",',
            text,
        )
        memories_py.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Set the Caregiver Website server IP/URL and immediately sync photos onto the device."
    )
    parser.add_argument(
        "ip_or_url",
        help='Server IP address or URL, e.g. "192.168.50.209" or "http://192.168.50.209:5173"',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5173,
        help="Server port if not included in ip_or_url (default: 5173)",
    )
    args = parser.parse_args()

    try:
        server_url = normalize_server_url(args.ip_or_url, default_port=args.port)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    # Locate the `memoryaid` project folder whether run from `HackRPI2026/` or `HackRPI2026/memoryaid/`
    script_dir = Path(__file__).resolve().parent
    if (script_dir / "memoryaid" / "__init__.py").is_file():
        memoryaid_root = script_dir
    elif (script_dir / "memoryaid" / "memoryaid" / "__init__.py").is_file():
        memoryaid_root = script_dir / "memoryaid"
    else:
        memoryaid_root = script_dir

    # 1. Write `.server_url` (read automatically by load_config)
    server_url_file = memoryaid_root / ".server_url"
    server_url_file.write_text(server_url + "\n", encoding="utf-8")
    print(f"[OK] Saved server URL: {server_url} -> {server_url_file}")

    # 2. Update config.py / memories.py defaults and any .env files
    _update_source_defaults(memoryaid_root, server_url)
    for env_candidate in (
        memoryaid_root / "memoryaid.env",
        memoryaid_root / "memoryaid.env.example",
        Path("/etc/memoryaid.env"),
    ):
        if _update_env_file(env_candidate, server_url):
            print(f"[OK] Updated WEBSITE_URL in {env_candidate}")

    # 3. Immediately test connection and pull photos from the server
    sys.path.insert(0, str(memoryaid_root))
    try:
        from memoryaid.memories import load_memories, sync_remote_website_memories

        memories_dir = memoryaid_root / "memories"
        print(f"[..] Contacting {server_url}/api/photos and syncing photos into {memories_dir} ...")
        changed = sync_remote_website_memories(memories_dir=memories_dir, website_url=server_url)
        places = load_memories(memories_dir)
        total_photos = sum(len(p.images_b64) for p in places)
        print(
            f"[OK] Sync complete (changed={changed}): {len(places)} active place(s), {total_photos} photo(s) ready on device."
        )
        for idx, p in enumerate(places, start=1):
            print(f"     #{idx} {p.name}: {p.note!r}")
    except Exception as exc:
        print(f"[WARN] Server URL updated, but immediate sync test failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
