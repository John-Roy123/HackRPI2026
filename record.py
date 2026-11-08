#!/usr/bin/env python3
"""Command-line shortcut to toggle the Memory Aid Record & Upload flow for proof-of-concept demos.

Usage:
  1. Interactive toggle (starts recording + takes photo, press Enter to stop, save & upload):
       python3 record.py

  2. Direct description (takes photo, saves description & uploads immediately):
       python3 record.py "This is the kitchen where we make coffee"
"""

import sys
from pathlib import Path

# Ensure the memoryaid package is importable whether run from HackRPI2026/ or HackRPI2026/memoryaid/
HERE = Path(__file__).resolve().parent
if (HERE / "memoryaid" / "__init__.py").is_file():
    sys.path.insert(0, str(HERE))
elif (HERE / "memoryaid" / "memoryaid" / "__init__.py").is_file():
    sys.path.insert(0, str(HERE / "memoryaid"))

from memoryaid.__main__ import main  # noqa: E402


if __name__ == "__main__":
    user_args = sys.argv[1:]
    if not any(a in ("-r", "--record") for a in user_args):
        # If the user passed a description string without -r/--record, forward it to --record
        if user_args and not user_args[0].startswith("-"):
            description = " ".join(user_args)
            sys.exit(main(["--record", description]))
        else:
            sys.exit(main(["--record", *user_args]))
    sys.exit(main(user_args))
