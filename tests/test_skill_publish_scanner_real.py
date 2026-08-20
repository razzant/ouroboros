"""Pinned Betterleaks v1.8.1 contract; skips when no explicit cache exists."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import subprocess
import tempfile
import zipfile

import pytest

import ouroboros.skill_publish_scanner as scanner
from ouroboros.betterleaks_runtime import resolve_betterleaks
from ouroboros.skill_publish_scanner import ScannerExecutable, scan_named_bytes

pytestmark = pytest.mark.serial

_FIXTURE_ROOT = pathlib.Path(__file__).parent / "fixtures" / "skill_publish_scanner"
_CONTEXT_FIXTURES = {
    "google_workspace_context.fixture": (
        "google_workspace_context.py",
        "26e330659241958418e3513dd000f3a4d65680e5d23619bdbe11e7760d51b4de",
    ),
    "email_context.fixture": (
        "email_context.py",
        "681188f4713d4d240f615a16e2d4496aa0182449d3280280b726a1c29d0cb564",
    ),
    "read_ai_context.fixture": (
        "read_ai_context.py",
        "5bc0608c1fbb3d011d8226b72a914421ea93996800c2797b72c2f8c7c3101c5e",
    ),
}
_EXPECTED_RULESET_BYTES = 287882
_EXPECTED_RULESET_SHA256 = "7c34b6ee2980139a97156b9b7e818813c8c13b7bc3d15b11704115f1f7d5027a"


def _binary() -> pathlib.Path:
    configured = str(os.environ.get("OUROBOROS_BETTERLEAKS_TEST_BINARY") or "").strip()
    data_root = str(
        os.environ.get("OUROBOROS_TEST_LIVE_DATA_ROOT") or os.environ.get("OUROBOROS_DATA_DIR") or ""
    ).strip()
    managed_state = (
        resolve_betterleaks(
            data_root=pathlib.Path(data_root),
            bundle_bases=[],
            include_managed=True,
        )
        if data_root
        else None
    )
    candidates = [
        pathlib.Path(configured).expanduser() if configured else None,
        pathlib.Path(managed_state.binary_path) if managed_state is not None and managed_state.ready else None,
        pathlib.Path.home() / ".claudexor" / "cache" / "betterleaks" / "v1.8.1" / "betterleaks",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    pytest.skip("pinned Betterleaks v1.8.1 binary is not provisioned")


def _executable(binary: pathlib.Path) -> ScannerExecutable:
    return ScannerExecutable(
        path=binary,
        identity=hashlib.sha256(binary.read_bytes()).hexdigest(),
    )


def _safe_absence(candidate: str, serialized: str) -> None:
    if candidate in serialized:
        pytest.fail("a generated candidate survived safe result projection", pytrace=False)


def test_pinned_engine_version_ruleset_and_contextual_fixture_multiset(tmp_path):
    binary = _binary()
    version = subprocess.run(
        [str(binary), "version"],
        env=scanner._scanner_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    assert version.returncode == 0
    assert version.stdout.decode("utf-8").strip() == "1.8.1"

    config_path = tmp_path / "host-config.toml"
    config_path.write_bytes(scanner._HOST_CONFIG_BYTES)
    shown = subprocess.run(
        [
            str(binary),
            "config",
            "show",
            "--config",
            str(config_path),
            "--no-banner",
            "--no-color",
            "--log-level",
            "error",
        ],
        env=scanner._scanner_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=20,
        check=False,
    )
    assert shown.returncode == 0
    assert len(shown.stdout) == _EXPECTED_RULESET_BYTES
    assert hashlib.sha256(shown.stdout).hexdigest() == _EXPECTED_RULESET_SHA256

    files = {}
    for physical_name, (logical_name, expected_digest) in _CONTEXT_FIXTURES.items():
        content = (_FIXTURE_ROOT / physical_name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_digest
        files[logical_name] = content
    result = scan_named_bytes(
        files,
        executable=_executable(binary),
        drive_root=tmp_path / "drive",
        scope="session",
    )

    assert result.status == "findings"
    assert result.ruleset_sha256 == _EXPECTED_RULESET_SHA256
    assert result.blocker_count == 0
    assert result.warning_count == 8
    assert result.audited_false_positive_count == 0
    assert {(item.path, item.detector, item.confidence, item.disposition) for item in result.findings} == {
        ("email_context.py", "generic-password", "medium", "warning")
    }


def test_pinned_engine_high_warning_sentinel_and_redaction_contract(tmp_path):
    binary = _binary()
    provider_tail = "".join(("aB3dE5fG", "7hJ9kL2m", "N4pQ6rS8", "tV0wX2yZ", "5cD7"))
    provider_candidate = "".join(("gh", "p_", provider_tail))
    generic_parts = ("8f3a9c1d", "7b2e4a6f", "0d5c8e1b", "3a7f9d2c")
    generic_candidate = "".join(reversed(generic_parts))
    password_candidate = "".join(("ordinary", "-fixture-", "password-", "12345"))
    private_body = "".join(
        (
            "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBK",
            "cwggSjAgEAAoIBAQC7FixtureOnlyMaterial",
            "ABCD1234efgh5678IJKL9012mnop3456",
        )
    )
    private_candidate = "\n".join(
        (
            "-----BEGIN PRIVATE KEY-----",
            private_body,
            "-----END PRIVATE KEY-----",
        )
    )
    sentinel = "mock_token"
    materialized = (
        f"provider = {provider_candidate!r}\n"
        f"access_token = {generic_candidate!r}\n"
        f"password = {password_candidate!r}\n"
        f"private_key = {private_candidate!r}\n"
        f"placeholder_token = {sentinel!r}\n"
    ).encode("utf-8")

    result = scan_named_bytes(
        {"generated_corpus.py": materialized},
        executable=_executable(binary),
        drive_root=tmp_path / "drive",
    )

    safe_rows = {(item.detector, item.confidence) for item in result.findings}
    assert ("github-pat", "high") in safe_rows
    assert ("private-key", "high") in safe_rows
    assert ("generic-api-key", "medium") in safe_rows
    assert ("generic-password", "low") in safe_rows
    assert result.blocker_count >= 2
    assert result.warning_count >= 2
    assert all(item.line != 5 for item in result.findings)
    serialized = json.dumps(dataclasses.asdict(result), sort_keys=True)
    for generated in (
        provider_candidate,
        generic_candidate,
        password_candidate,
        private_body,
    ):
        _safe_absence(generated, serialized)


def test_pinned_engine_annotation_audit_and_derived_forced_ignore(tmp_path):
    binary = _binary()
    candidate = "".join(
        (
            "gh",
            "p_",
            "aB3dE5fG",
            "7hJ9kL2m",
            "N4pQ6rS8",
            "tV0wX2yZ",
            "5cD7",
        )
    )
    annotated = f"token = {candidate!r}  # betterleaks:allow\n".encode("utf-8")

    payload = scan_named_bytes(
        {"annotated.py": annotated},
        executable=_executable(binary),
        drive_root=tmp_path / "payload-drive",
    )
    derived = scan_named_bytes(
        {"annotated.py": annotated},
        executable=_executable(binary),
        drive_root=tmp_path / "derived-drive",
        honor_inline_allowances=False,
    )

    assert payload.blocker_count == 0
    assert payload.audited_false_positive_count == 1
    assert payload.findings[0].disposition == "audited_false_positive"
    assert derived.blocker_count == 1
    assert derived.audited_false_positive_count == 0
    _safe_absence(candidate, json.dumps(dataclasses.asdict(payload)))
    _safe_absence(candidate, json.dumps(dataclasses.asdict(derived)))


def test_pinned_engine_scans_one_archive_and_decode_layer(tmp_path):
    binary = _binary()
    candidate = "".join(
        (
            "gh",
            "p_",
            "aB3dE5fG",
            "7hJ9kL2m",
            "N4pQ6rS8",
            "tV0wX2yZ",
            "5cD7",
        )
    )
    inner = f"token = {candidate!r}\n".encode("utf-8")
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("inner.py", inner)
    encoded = base64.b64encode(inner)

    result = scan_named_bytes(
        {
            "archive.zip": archive_buffer.getvalue(),
            "encoded.txt": encoded,
        },
        executable=_executable(binary),
        drive_root=tmp_path / "drive",
    )

    assert result.blocker_count == 2
    assert {item.path for item in result.findings} == {
        "archive.zip!inner.py",
        "encoded.txt",
    }
    _safe_absence(candidate, json.dumps(dataclasses.asdict(result)))


def _raw_findings(
    binary: pathlib.Path,
    source: pathlib.Path,
    *,
    config: pathlib.Path,
    ignore: pathlib.Path,
    report: pathlib.Path,
) -> list[dict]:
    run = subprocess.run(
        [
            str(binary),
            "detect",
            "--no-git",
            "--source",
            str(source),
            "--config",
            str(config),
            "--gitleaks-ignore-path",
            str(ignore),
            "--redact=100",
            "--confidence",
            "low",
            "--max-archive-depth",
            "1",
            "--max-decode-depth",
            "1",
            "--exit-code",
            "0",
            "--timeout",
            "30",
            "--report-format",
            "json",
            "--report-path",
            str(report),
            "--no-banner",
            "--no-color",
            "--log-level",
            "error",
        ],
        env=scanner._scanner_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=40,
        check=False,
    )
    assert run.returncode == 0
    root = json.loads(report.read_text(encoding="utf-8"))
    return root if isinstance(root, list) else []


def test_pinned_engine_payload_ignore_autodiscovery_is_physically_neutralized(tmp_path):
    binary = _binary()
    candidate = "".join(
        (
            "gh",
            "p_",
            "aB3dE5fG",
            "7hJ9kL2m",
            "N4pQ6rS8",
            "tV0wX2yZ",
            "5cD7",
        )
    )
    with tempfile.TemporaryDirectory() as raw_tmp:
        root = pathlib.Path(raw_tmp)
        source = root / "source"
        source.mkdir()
        content = f"token = {candidate!r}\n".encode("utf-8")
        (source / "fixture.py").write_bytes(content)
        config = root / "host.toml"
        config.write_bytes(scanner._HOST_CONFIG_BYTES)
        ignore = root / "host-ignore"
        ignore.write_bytes(b"")
        first = _raw_findings(
            binary,
            source,
            config=config,
            ignore=ignore,
            report=root / "first.json",
        )
        assert len(first) == 1
        fingerprint = first[0].get("Fingerprint")
        assert isinstance(fingerprint, str) and fingerprint
        payload_ignore = (fingerprint + "\n").encode("utf-8")
        (source / ".betterleaksignore").write_bytes(payload_ignore)
        second = _raw_findings(
            binary,
            source,
            config=config,
            ignore=ignore,
            report=root / "second.json",
        )
        assert second == []

    result = scan_named_bytes(
        [
            ("fixture.py", content),
            (".betterleaksignore", payload_ignore),
            (".gitleaksignore", payload_ignore),
            (".betterleaks.toml", b"this payload config is intentionally invalid ="),
        ],
        executable=_executable(binary),
        drive_root=tmp_path / "drive",
    )

    assert result.blocker_count == 1
    assert any(item.path == "fixture.py" for item in result.findings)
    _safe_absence(candidate, json.dumps(dataclasses.asdict(result)))


def test_pinned_engine_scans_neutralized_payload_ignore_contents(tmp_path):
    binary = _binary()
    candidate = "".join(
        (
            "gh",
            "p_",
            "aB3dE5fG",
            "7hJ9kL2m",
            "N4pQ6rS8",
            "tV0wX2yZ",
            "5cD7",
        )
    )
    content = f"value = {candidate!r}\n".encode("utf-8")

    result = scan_named_bytes(
        {
            ".betterleaksignore": content,
            ".gitleaksignore": content,
        },
        executable=_executable(binary),
        drive_root=tmp_path / "drive",
    )

    assert result.blocker_count == 2
    assert {item.path for item in result.findings} == {
        ".betterleaksignore",
        ".gitleaksignore",
    }
    _safe_absence(candidate, json.dumps(dataclasses.asdict(result)))
