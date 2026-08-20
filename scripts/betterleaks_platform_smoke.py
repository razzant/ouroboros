#!/usr/bin/env python3
"""Run a redaction-safe Betterleaks runtime smoke.

The script deliberately emits only three bounded facts: engine version, finding
count, and detector/confidence classification.  It never prints scanner output
or the generated candidate used by the positive case.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import string
import subprocess
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ouroboros import platform_layer  # noqa: E402
from ouroboros.betterleaks_runtime import (  # noqa: E402
    BETTERLEAKS_VERSION,
    resolve_betterleaks,
)

_SAFE_RULE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_REPORT_LIMIT = 1024 * 1024


def _candidate() -> str:
    alphabet = string.ascii_letters + string.digits
    suffix = "".join(alphabet[(index * 17 + 11) % len(alphabet)] for index in range(82))
    return "github_pat_" + suffix


def _scan(binary: str, content: str, root: pathlib.Path, name: str) -> list[dict]:
    report = root / f"{name}.json"
    ignore = root / "empty-ignore"
    ignore.touch(exist_ok=True)
    env = dict(os.environ)
    for key in (
        "BETTERLEAKS_CONFIG",
        "BETTERLEAKS_CONFIG_TOML",
        "GITLEAKS_CONFIG",
        "GITLEAKS_CONFIG_TOML",
    ):
        env.pop(key, None)
    completed = subprocess.run(
        [
            binary,
            "stdin",
            "--set-attr",
            f"path={name}.txt",
            "--no-banner",
            "--no-color",
            "--redact=100",
            "--confidence",
            "low",
            "--max-archive-depth",
            "1",
            "--max-decode-depth",
            "1",
            "--exit-code",
            "0",
            "--report-format",
            "json",
            "--report-path",
            str(report),
            "--gitleaks-ignore-path",
            str(ignore),
        ],
        input=content,
        text=True,
        encoding="utf-8",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=30,
        check=False,
        env=env,
        **platform_layer.subprocess_hidden_kwargs(),
    )
    if completed.returncode != 0 or not report.is_file():
        raise RuntimeError("scanner process failed")
    if report.stat().st_size > _REPORT_LIMIT:
        raise RuntimeError("scanner report exceeded the smoke bound")
    raw = report.read_text(encoding="utf-8")
    value = json.loads(raw)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise RuntimeError("scanner report shape is invalid")
    return value


def _classification(rows: list[dict]) -> str:
    if len(rows) != 1:
        raise RuntimeError("positive smoke did not produce exactly one finding")
    row = rows[0]
    attributes = row.get("Attributes")
    if not isinstance(attributes, dict):
        raise RuntimeError("finding attributes are missing")
    rule = str(row.get("RuleID") or "").strip().lower()
    confidence = str(attributes.get("confidence") or "").strip().lower()
    if not _SAFE_RULE.fullmatch(rule) or confidence != "high":
        raise RuntimeError("finding classification is unexpected")
    return f"{rule}:{confidence}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--bundle-root",
        type=pathlib.Path,
        help="resolve only from this extracted/installed package resource root",
    )
    mode.add_argument(
        "--managed-runtime",
        action="store_true",
        help="resolve only from the exact managed runtime under the active data root",
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    args = build_parser().parse_args(argv)
    state = (
        resolve_betterleaks(bundle_bases=[args.bundle_root], include_managed=False)
        if args.bundle_root is not None
        else resolve_betterleaks(bundle_bases=[], include_managed=True)
    )
    count = 0
    classification = "runtime_error"
    try:
        if not state.ready:
            raise RuntimeError("runtime unavailable")
        positive = _candidate()
        with tempfile.TemporaryDirectory(prefix="ouroboros-betterleaks-smoke-") as temporary:
            root = pathlib.Path(temporary)
            clean_rows = _scan(state.binary_path, "hello from Ouroboros\n", root, "clean")
            if clean_rows:
                raise RuntimeError("clean smoke produced a finding")
            rows = _scan(
                state.binary_path,
                f'credential = "{positive}"\n',
                root,
                "positive",
            )
            # Full redaction is part of the smoke contract, but the raw report
            # and generated candidate never become output.
            if any(positive in path.read_text(encoding="utf-8") for path in root.glob("*.json")):
                raise RuntimeError("scanner report was not fully redacted")
            count = len(rows)
            classification = _classification(rows)
    except (OSError, UnicodeError, ValueError, RuntimeError, subprocess.SubprocessError):
        print(f"version={BETTERLEAKS_VERSION}")
        print("count=0")
        print("classification=runtime_error")
        return 1

    print(f"version={BETTERLEAKS_VERSION}")
    print(f"count={count}")
    print(f"classification={classification}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
