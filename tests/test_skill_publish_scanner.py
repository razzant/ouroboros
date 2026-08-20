"""Offline process-double contract tests for the Betterleaks adapter."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import sys
from dataclasses import dataclass

import pytest

import ouroboros.skill_publish_scanner as scanner
from ouroboros.platform_layer import IS_WINDOWS
from ouroboros.skill_publish_scanner import (
    ScannerExecutable,
    scan_named_bytes,
)

pytestmark = pytest.mark.serial

_FAKE_SOURCE = pathlib.Path(__file__).parent / "fixtures" / "skill_publish_scanner" / "fake_betterleaks.py"


@dataclass(frozen=True)
class _FakeScanner:
    executable: ScannerExecutable
    mode_path: pathlib.Path
    trace_path: pathlib.Path

    def mode(self, value: str) -> None:
        self.mode_path.write_text(value, encoding="utf-8")

    def trace(self) -> list[dict]:
        if not self.trace_path.is_file():
            return []
        return [json.loads(line) for line in self.trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _fake_scanner(tmp_path: pathlib.Path) -> _FakeScanner:
    script = tmp_path / "fake_betterleaks.py"
    shutil.copy2(_FAKE_SOURCE, script)
    if IS_WINDOWS:
        binary = tmp_path / "betterleaks.cmd"
        binary.write_text(
            f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n',
            encoding="utf-8",
        )
    else:
        binary = tmp_path / "betterleaks"
        shutil.copy2(script, binary)
        os.chmod(binary, 0o700)
        script = binary
    return _FakeScanner(
        executable=ScannerExecutable(
            path=binary.resolve(),
            identity="fake-betterleaks-v1",
        ),
        mode_path=script.with_suffix(".mode"),
        trace_path=script.with_suffix(".trace.jsonl"),
    )


def _scan(
    fake: _FakeScanner,
    tmp_path: pathlib.Path,
    files,
    **kwargs,
):
    return scan_named_bytes(
        files,
        executable=fake.executable,
        drive_root=tmp_path / "drive",
        owner_task_id="task-fixture",
        timeout_sec=kwargs.pop("timeout_sec", 5),
        **kwargs,
    )


def test_fake_scanner_two_pass_classification_env_and_neutral_projection(
    tmp_path,
    monkeypatch,
):
    fake = _fake_scanner(tmp_path)
    monkeypatch.setenv("BETTERLEAKS_CONFIG", "/untrusted/config")
    monkeypatch.setenv("GITLEAKS_CONFIG_TOML", "untrusted")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-enter-scanner")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-enter-scanner")
    files = [
        ("high.txt", b"FAKE_FIND:provider-key:high\n"),
        ("medium.txt", b"FAKE_FIND:generic-password:medium\n"),
        ("unknown.txt", b"FAKE_FIND:unclassified:HIGH\n"),
        ("allowed.txt", b"FAKE_ALLOW:provider-key:high\n"),
        (".betterleaksignore", b"FAKE_FIND:ignore-content:low\n"),
        (".gitleaksignore", b"ordinary ignore bytes\n"),
        (".betterleaks.toml", b"ordinary payload config bytes\n"),
    ]

    result = _scan(fake, tmp_path, files, scope="session")

    assert result.status == "findings"
    assert result.engine == "betterleaks"
    assert result.version == "1.8.1"
    assert result.ruleset_sha256 == hashlib.sha256(b"resolved fake ruleset\n").hexdigest()
    assert len(result.scan_contract_sha256) == 64
    assert result.blocker_count == 1
    assert result.warning_count == 3
    assert result.audited_false_positive_count == 1
    by_path = {item.path: item for item in result.findings}
    assert by_path["high.txt"].disposition == "blocker"
    assert by_path["medium.txt"].disposition == "warning"
    assert by_path["unknown.txt"].confidence == "unknown"
    assert by_path["unknown.txt"].disposition == "warning"
    assert by_path["allowed.txt"].disposition == "audited_false_positive"
    assert by_path[".betterleaksignore"].disposition == "warning"
    assert {item.verification for item in result.findings} == {"not_attempted"}

    trace = fake.trace()
    assert [row["command"] for row in trace] == [
        "version",
        "config_show",
        "detect",
        "detect",
    ]
    detects = [row for row in trace if row["command"] == "detect"]
    assert [row["audit"] for row in detects] == [False, True]
    for row in trace:
        assert not any(row["env_presence"].values())
    for row in detects:
        assert row["config_outside_source"] is True
        assert row["config_sha256"] == scanner._HOST_CONFIG_SHA256
        assert row["ignore_empty"] is True
        assert row["ignore_outside_source"] is True
        if not IS_WINDOWS:
            assert row["private_root_mode"] == 0o700
            assert row["projection_mode"] == 0o700
        assert row["flags"] == {
            "no_git": True,
            "redact": "--redact=100",
            "confidence": "low",
            "archive_depth": "1",
            "decode_depth": "1",
            "exit_code": "0",
            "timeout": "5",
            "format": "json",
            "validation": False,
            "baseline": False,
        }
        assert ".betterleaksignore" not in row["source_names"]
        assert ".gitleaksignore" not in row["source_names"]
        assert sum("ouroboros-scanned-ignore" in name for name in row["source_names"]) == 2
        assert ".betterleaks.toml" in row["source_names"]

    ledger = tmp_path / "drive" / "state" / "process_ledger.jsonl"
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert {row["scope"] for row in records} == {"session"}
    assert {row["purpose"] for row in records} == {
        "skill_publish_scanner_version",
        "skill_publish_scanner_ruleset",
        "skill_publish_scanner_detect",
    }


def test_safe_finding_allowlist_discards_candidate_bearing_schema(tmp_path):
    fake = _fake_scanner(tmp_path)
    result = _scan(
        fake,
        tmp_path,
        {"fixture.txt": b"FAKE_FIND:provider-key:high\n"},
    )

    assert set(dataclasses.asdict(result.findings[0])) == {
        "path",
        "line",
        "detector",
        "confidence",
        "reason",
        "verification",
        "disposition",
    }
    serialized = json.dumps(dataclasses.asdict(result), sort_keys=True)
    assert "never-expose-candidate" not in serialized
    assert "never-expose-match" not in serialized
    assert "never-expose-fingerprint" not in serialized
    assert "never-expose-description" not in serialized
    assert str(tmp_path) not in serialized


def test_duplicate_multiset_is_preserved_and_audit_difference_is_exact(tmp_path):
    fake = _fake_scanner(tmp_path)
    files = {
        "duplicates.txt": (
            b"FAKE_FIND:generic-password:medium FAKE_FIND:generic-password:medium\n"
            b"FAKE_ALLOW:provider-key:high FAKE_ALLOW:provider-key:high\n"
        )
    }

    result = _scan(fake, tmp_path, files)

    assert result.warning_count == 2
    assert result.audited_false_positive_count == 2
    assert [item.disposition for item in result.findings].count("warning") == 2
    assert [item.disposition for item in result.findings].count("audited_false_positive") == 2


def test_derived_scan_forces_allow_ignoring_without_audit_disposition(tmp_path):
    fake = _fake_scanner(tmp_path)

    result = _scan(
        fake,
        tmp_path,
        {"derived.md": b"FAKE_ALLOW:provider-key:high\n"},
        honor_inline_allowances=False,
    )

    assert result.blocker_count == 1
    assert result.audited_false_positive_count == 0
    assert result.findings[0].disposition == "blocker"
    detects = [row for row in fake.trace() if row["command"] == "detect"]
    assert len(detects) == 1
    assert detects[0]["audit"] is True


def test_clean_null_legacy_path_unknown_confidence_and_detector_sanitization(tmp_path):
    fake = _fake_scanner(tmp_path)
    clean = _scan(fake, tmp_path, {"clean.txt": b"ordinary\n"})
    assert clean.status == "clean"
    assert clean.findings == ()

    fake.mode("report_legacy_path")
    legacy = _scan(
        fake,
        tmp_path,
        {"legacy.txt": b"FAKE_FIND:legacy-rule:medium\n"},
    )
    assert legacy.findings[0].path == "legacy.txt"

    fake.mode("report_missing_confidence")
    unknown = _scan(
        fake,
        tmp_path,
        {"unknown.txt": b"FAKE_FIND:legacy-rule:medium\n"},
    )
    assert unknown.findings[0].confidence == "unknown"
    assert unknown.findings[0].disposition == "warning"

    fake.mode("report_unsafe_detector")
    sanitized = _scan(
        fake,
        tmp_path,
        {"rule.txt": b"FAKE_FIND:legacy-rule:medium\n"},
    )
    assert sanitized.findings[0].detector == "unsafe_detector_value"


@pytest.mark.parametrize(
    "mode",
    [
        "detect_nonzero",
        "report_missing",
        "report_invalid_utf8",
        "report_invalid_json",
        "report_bad_root",
        "report_bad_line",
        "report_bad_column",
        "report_bad_path",
    ],
)
def test_malformed_or_incomplete_report_fails_closed(tmp_path, mode):
    fake = _fake_scanner(tmp_path)
    fake.mode(mode)

    result = _scan(
        fake,
        tmp_path,
        {"fixture.txt": b"FAKE_FIND:fixture-rule:medium\n"},
    )

    assert result.status == "scanner_error"
    assert result.reason_code == "scanner_report_invalid"
    assert result.findings == ()


def test_oversized_report_fails_closed(tmp_path, monkeypatch):
    fake = _fake_scanner(tmp_path)
    fake.mode("report_oversized")
    monkeypatch.setattr(scanner, "_MAX_REPORT_BYTES", 32)

    result = _scan(fake, tmp_path, {"fixture.txt": b"ordinary\n"})

    assert result.reason_code == "scanner_report_invalid"


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("invalid_version", "scanner_corrupt"),
        ("version_nonzero", "scanner_corrupt"),
        ("config_nonzero", "scanner_ruleset_invalid"),
        ("config_invalid_utf8", "scanner_ruleset_invalid"),
    ],
)
def test_version_and_ruleset_probe_failures_are_closed(tmp_path, mode, reason):
    fake = _fake_scanner(tmp_path)
    fake.mode(mode)

    result = _scan(fake, tmp_path, {"fixture.txt": b"ordinary\n"})

    assert result.status == "scanner_error"
    assert result.reason_code == reason
    assert set(dataclasses.asdict(result)) == {
        "status",
        "engine",
        "version",
        "ruleset_sha256",
        "scan_contract_sha256",
        "findings",
        "blocker_count",
        "warning_count",
        "audited_false_positive_count",
        "reason_code",
        "repair_hint",
    }


def test_oversized_ruleset_probe_fails_closed(tmp_path, monkeypatch):
    fake = _fake_scanner(tmp_path)
    fake.mode("config_oversized")
    monkeypatch.setattr(scanner, "_MAX_RULESET_OUTPUT_BYTES", 32)

    result = _scan(fake, tmp_path, {"fixture.txt": b"ordinary\n"})

    assert result.reason_code == "scanner_ruleset_invalid"


def test_detect_timeout_kills_process_group_and_returns_safe_reason(tmp_path, monkeypatch):
    fake = _fake_scanner(tmp_path)
    fake.mode("timeout_detect")
    monkeypatch.setattr(scanner, "_HOST_TIMEOUT_GRACE_SEC", 0.1)

    result = _scan(
        fake,
        tmp_path,
        {"fixture.txt": b"ordinary\n"},
        timeout_sec=1,
    )

    assert result.status == "scanner_error"
    assert result.reason_code == "scanner_timeout"
    assert str(tmp_path) not in json.dumps(dataclasses.asdict(result))


@pytest.mark.parametrize(
    "files",
    [
        [("../escape.txt", b"x")],
        [("/absolute.txt", b"x")],
        [("same.txt", b"x"), ("same.txt", b"y")],
        [("windows\\path.txt", b"x")],
    ],
)
def test_invalid_or_duplicate_named_paths_are_refused_before_spawn(tmp_path, files):
    fake = _fake_scanner(tmp_path)

    result = _scan(fake, tmp_path, files)

    assert result.reason_code == "scanner_input_invalid"
    assert fake.trace() == []


def test_missing_and_corrupt_resolved_executable_are_typed(tmp_path):
    missing = scan_named_bytes(
        {"fixture.txt": b"ordinary\n"},
        executable=ScannerExecutable(None, "", "missing"),
        drive_root=tmp_path / "drive",
    )
    corrupt = scan_named_bytes(
        {"fixture.txt": b"ordinary\n"},
        executable=ScannerExecutable(tmp_path / "missing-binary", "identity", "ready"),
        drive_root=tmp_path / "drive",
    )
    corrupt_without_path = scan_named_bytes(
        {"fixture.txt": b"ordinary\n"},
        executable=ScannerExecutable(None, "", "corrupt"),
        drive_root=tmp_path / "drive",
    )

    assert missing.reason_code == "scanner_missing"
    assert corrupt.reason_code == "scanner_corrupt"
    assert corrupt_without_path.reason_code == "scanner_corrupt"
    assert missing.repair_hint
    assert corrupt.repair_hint


def test_private_projection_and_raw_reports_are_removed_on_return(tmp_path, monkeypatch):
    fake = _fake_scanner(tmp_path)
    real_factory = scanner.tempfile.TemporaryDirectory
    roots: list[pathlib.Path] = []

    def tracked_factory(*args, **kwargs):
        created = real_factory(*args, **kwargs)
        roots.append(pathlib.Path(created.name))
        return created

    monkeypatch.setattr(scanner.tempfile, "TemporaryDirectory", tracked_factory)

    result = _scan(fake, tmp_path, {"fixture.txt": b"ordinary\n"})

    assert result.status == "clean"
    assert len(roots) == 1
    assert not roots[0].exists()
