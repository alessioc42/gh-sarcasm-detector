from __future__ import annotations

import sys

from .compress import run_compress
from .config import Config
from .import_raw import run_import
from .jobs import run_jobs, run_status


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        print("Usage: python -m sarcasm_detector {import|compress|run|status}")
        return 0 if argv and argv[0] in {"-h", "--help"} else 1

    config = Config.from_env()
    command = argv[0]

    if command == "import":
        run_import(config)
        return 0
    if command == "compress":
        run_compress(config)
        return 0
    if command == "run":
        run_jobs(config)
        return 0
    if command == "status":
        run_status(config)
        return 0

    print(f"Unknown command: {command}", file=sys.stderr)
    print(
        "Usage: python -m sarcasm_detector {import|compress|run|status}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
