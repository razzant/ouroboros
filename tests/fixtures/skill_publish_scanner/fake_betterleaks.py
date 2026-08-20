#!/usr/bin/env python3
"""Deterministic offline Betterleaks process double used by focused tests."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import stat
import sys
import time

SCRIPT = pathlib.Path(__file__)
MODE_PATH = SCRIPT.with_suffix(".mode")
TRACE_PATH = SCRIPT.with_suffix(".trace.jsonl")
FIND_RE = re.compile(r"FAKE_(FIND|ALLOW):([A-Za-z0-9._/-]+):([^\s]+)")


def _mode() -> str:
    try:
        return MODE_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _trace(payload: dict) -> None:
    with TRACE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _flag(args: list[str], name: str) -> str:
    try:
        return args[args.index(name) + 1]
    except (ValueError, IndexError):
        return ""


def _env_presence() -> dict[str, bool]:
    return {
        name: name in os.environ
        for name in (
            "BETTERLEAKS_CONFIG",
            "BETTERLEAKS_CONFIG_TOML",
            "GITLEAKS_CONFIG",
            "GITLEAKS_CONFIG_TOML",
            "OPENAI_API_KEY",
            "GITHUB_TOKEN",
        )
    }


def _version(mode: str) -> int:
    _trace({"command": "version", "env_presence": _env_presence()})
    if mode == "timeout_version":
        time.sleep(30)
    if mode == "invalid_version":
        sys.stdout.write("9.9.9\n")
        return 0
    if mode == "version_nonzero":
        return 3
    sys.stdout.write("1.8.1\n")
    return 0


def _config_show(args: list[str], mode: str) -> int:
    config_path = pathlib.Path(_flag(args, "--config"))
    try:
        config = config_path.read_bytes()
    except OSError:
        config = b""
    _trace(
        {
            "command": "config_show",
            "config_sha256": hashlib.sha256(config).hexdigest(),
            "env_presence": _env_presence(),
        }
    )
    if mode == "timeout_config":
        time.sleep(30)
    if mode == "config_nonzero":
        return 4
    if mode == "config_invalid_utf8":
        sys.stdout.buffer.write(b"\xff")
        return 0
    if mode == "config_oversized":
        sys.stdout.buffer.write(b"x" * (1024 * 1024))
        return 0
    sys.stdout.buffer.write(b"resolved fake ruleset\n")
    return 0


def _row(
    *,
    path: pathlib.Path,
    line: int,
    column: int,
    detector: str,
    confidence: str,
) -> dict:
    return {
        "RuleID": detector,
        "StartLine": line,
        "StartColumn": column,
        "File": str(path),
        "Attributes": {"path": str(path), "confidence": confidence},
        "Secret": "never-expose-candidate",
        "Match": "never-expose-match",
        "Fingerprint": "never-expose-fingerprint",
        "Description": "never-expose-description",
    }


def _detect(args: list[str], mode: str) -> int:
    source = pathlib.Path(_flag(args, "--source"))
    config = pathlib.Path(_flag(args, "--config"))
    ignore = pathlib.Path(_flag(args, "--gitleaks-ignore-path"))
    report = pathlib.Path(_flag(args, "--report-path"))
    audit = "--ignore-gitleaks-allow" in args
    files = sorted(path for path in source.rglob("*") if path.is_file())
    _trace(
        {
            "command": "detect",
            "audit": audit,
            "config_outside_source": source not in config.parents,
            "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
            "ignore_empty": ignore.read_bytes() == b"",
            "ignore_outside_source": source not in ignore.parents,
            "private_root_mode": stat.S_IMODE(source.parent.stat().st_mode),
            "projection_mode": stat.S_IMODE(source.stat().st_mode),
            "env_presence": _env_presence(),
            "flags": {
                "no_git": "--no-git" in args,
                "redact": next((item for item in args if item.startswith("--redact")), ""),
                "confidence": _flag(args, "--confidence"),
                "archive_depth": _flag(args, "--max-archive-depth"),
                "decode_depth": _flag(args, "--max-decode-depth"),
                "exit_code": _flag(args, "--exit-code"),
                "timeout": _flag(args, "--timeout"),
                "format": _flag(args, "--report-format"),
                "validation": "--validation" in args,
                "baseline": "--baseline-path" in args,
            },
            "source_names": [path.relative_to(source).as_posix() for path in files],
        }
    )
    if mode == "timeout_detect":
        time.sleep(30)
    if mode == "detect_nonzero":
        return 5
    if mode == "report_missing":
        return 0
    if mode == "report_invalid_utf8":
        report.write_bytes(b"\xff")
        return 0
    if mode == "report_invalid_json":
        report.write_text("{", encoding="utf-8")
        return 0
    if mode == "report_bad_root":
        report.write_text("{}", encoding="utf-8")
        return 0
    if mode == "report_oversized":
        report.write_text("[" + (" " * 4096) + "]", encoding="utf-8")
        return 0

    rows: list[dict] = []
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), 1):
            for match in FIND_RE.finditer(line):
                kind, detector, confidence = match.groups()
                if kind == "ALLOW" and not audit:
                    continue
                rows.append(
                    _row(
                        path=path,
                        line=line_number,
                        column=match.start() + 1,
                        detector=detector,
                        confidence=confidence,
                    )
                )
    if mode.startswith("report_") and not rows:
        rows.append(
            _row(
                path=(source / "fixture.txt"),
                line=1,
                column=1,
                detector="fixture-rule",
                confidence="medium",
            )
        )
    if mode == "report_bad_line":
        rows[0]["StartLine"] = 0
    elif mode == "report_bad_column":
        rows[0]["StartColumn"] = 0
    elif mode == "report_bad_path":
        rows[0]["Attributes"]["path"] = str(source.parent / "outside.txt")
    elif mode == "report_legacy_path":
        rows[0]["Attributes"].pop("path", None)
    elif mode == "report_missing_confidence":
        rows[0]["Attributes"].pop("confidence", None)
    elif mode == "report_unsafe_detector":
        rows[0]["RuleID"] = "unsafe detector/value"
    if mode == "report_null":
        payload = None
    else:
        payload = rows or None
    report.write_text(json.dumps(payload), encoding="utf-8")
    return 0


def main() -> int:
    args = sys.argv[1:]
    mode = _mode()
    if args[:1] == ["version"]:
        return _version(mode)
    if args[:2] == ["config", "show"]:
        return _config_show(args, mode)
    if args[:1] == ["detect"]:
        return _detect(args, mode)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
