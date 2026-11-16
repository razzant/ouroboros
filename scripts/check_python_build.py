#!/usr/bin/env python3
"""Refuse a desktop build whose python-standalone is not the pinned Python.

Every build script runs this with the python-standalone it is about to ship, before it
installs anything:

    python-standalone/bin/python3 scripts/check_python_build.py

A python-standalone left from an earlier pin (a reused build tree that skipped the downloader)
would otherwise ship again under a new version: the launcher is frozen from a venv of this same
interpreter, so nothing else would notice. One that cannot start fails its build line by itself.
Only the CPython version is compared: a tree from another python-build-standalone release of the
same CPython (a SQLite or OpenSSL respin) passes. The pin is ``PY_VERSION`` in
``scripts/download_python_standalone.sh``. Stdlib only.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

PIN_SCRIPT = pathlib.Path(__file__).resolve().parent / "download_python_standalone.sh"


def pinned_version(pin_script: pathlib.Path = PIN_SCRIPT) -> str:
    match = re.search(r'^PY_VERSION="(\d+\.\d+\.\d+)"$', pin_script.read_text(encoding="utf-8"), re.MULTILINE)
    if match is None:
        raise SystemExit(f"ERROR: no PY_VERSION pin in {pin_script}.")
    return match.group(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect", default="", help="expected version (default: the download script's pin)")
    expected = parser.parse_args(argv).expect or pinned_version()
    running = "%d.%d.%d" % sys.version_info[:3]
    if running != expected:
        print(f"ERROR: {sys.executable} is Python {running}; this checkout pins {expected}. Replace python-standalone "
              "with scripts/download_python_standalone.sh (Windows: .ps1) before building.", file=sys.stderr)
        return 1
    print(f"python-standalone: Python {running}, the pinned version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
