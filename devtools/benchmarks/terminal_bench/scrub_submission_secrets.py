"""Scrub secret values from a TB submission pack copy before ``harbor upload``.

Submitted leaderboard jobs become PUBLIC. Runs launched with
``OUROBOROS_BENCH_ALLOW_CONTAINER_SECRETS=1`` persist live provider keys inside
every trial (``agent/ouroboros-data/settings.json``), and values may leak into
logs. This tool must be run on a COPY of the job directory (never the archived
original) before the first upload.

Two passes:
1. Structural: every ``settings.json`` under an ``ouroboros-data`` directory
   gets known secret fields (``common.secrets.SECRET_KEYS`` + extras) blanked.
2. Value sweep: every literal occurrence of every secret VALUE collected from
   ``--secrets-from`` sources is replaced with ``<REDACTED:NAME>`` in text
   files; a secret found inside a non-text file is a hard error.

A final independent verify pass re-scans the tree for every secret value and
exits non-zero on any hit. Secret values are never printed.

Usage:
    python scrub_submission_secrets.py --root <job_dir_copy> \
        --secrets-from ~/ouro/data/settings.json --secrets-from ~/ouro/file1.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.secrets import SECRET_KEYS  # noqa: E402

# Fields blanked in embedded settings.json files, beyond common.secrets.
EXTRA_SECRET_FIELDS = (
    "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_PASSWORD",
    "OUROBOROS_NETWORK_PASSWORD",
    "TELEGRAM_BOT_TOKEN",
)
MIN_SECRET_LEN = 8  # ignore trivially short values (avoid mass false rewrites)


def collect_secrets(sources: list[Path]) -> dict[str, str]:
    """Collect name -> value pairs from settings.json / ``name: value`` files."""
    secrets: dict[str, str] = {}

    def add(name: str, value: object) -> None:
        if isinstance(value, str) and len(value) >= MIN_SECRET_LEN:
            secrets[name] = value

    for src in sources:
        text = src.read_text(encoding="utf-8", errors="replace")
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict):
            for key, value in data.items():
                upper = key.upper()
                if any(tag in upper for tag in ("KEY", "TOKEN", "PASSWORD", "SECRET")):
                    add(key, value)
            continue
        for line in text.splitlines():
            if ":" not in line:
                continue
            name, _, value = line.partition(":")
            add(name.strip() or "unnamed", value.strip())
    return secrets


def _blank_settings_fields(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, ValueError):
        return 0
    if not isinstance(data, dict):
        return 0
    blanked = 0
    for field in (*SECRET_KEYS, *EXTRA_SECRET_FIELDS):
        if isinstance(data.get(field), str) and data[field]:
            data[field] = ""
            blanked += 1
    if blanked:
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return blanked


def _sweep_file(path: Path, secrets: dict[str, str]) -> tuple[int, list[str]]:
    """Replace secret values in one file; returns (replacements, binary_hits)."""
    try:
        raw = path.read_bytes()
    except OSError:
        return 0, []
    hits = [name for name, value in secrets.items() if value.encode() in raw]
    if not hits:
        return 0, []
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return 0, hits  # secret inside a non-text file: caller fails hard
    replaced = 0
    for name in hits:
        value = secrets[name]
        replaced += text.count(value)
        text = text.replace(value, f"<REDACTED:{name}>")
    path.write_text(text, encoding="utf-8")
    return replaced, []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--secrets-from", action="append", required=True, type=Path, dest="sources"
    )
    args = parser.parse_args()

    root: Path = args.root
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    secrets = collect_secrets(args.sources)
    if not secrets:
        parser.error("no secret values collected from --secrets-from sources")

    files = [p for p in root.rglob("*") if p.is_file()]

    blanked_fields = 0
    for path in files:
        if path.name == "settings.json" and "ouroboros-data" in path.parts:
            blanked_fields += _blank_settings_fields(path)

    replacements = 0
    binary_failures: list[str] = []
    for path in files:
        count, binary_hits = _sweep_file(path, secrets)
        replacements += count
        if binary_hits:
            binary_failures.append(f"{path}: {sorted(set(binary_hits))}")

    if binary_failures:
        print("SECRET VALUES IN NON-TEXT FILES (fix manually):", file=sys.stderr)
        for line in binary_failures:
            print(f"  {line}", file=sys.stderr)
        return 2

    # Independent verify pass: re-read everything, zero tolerance.
    leftovers = 0
    for path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        for name, value in secrets.items():
            if value.encode() in raw:
                leftovers += 1
                print(f"LEFTOVER {name} in {path}", file=sys.stderr)
    print(
        f"secrets={len(secrets)} files={len(files)} "
        f"settings_fields_blanked={blanked_fields} value_replacements={replacements} "
        f"verify_leftovers={leftovers}"
    )
    return 1 if leftovers else 0


if __name__ == "__main__":
    raise SystemExit(main())
