"""Redaction-safe Betterleaks adapter for immutable publish bytes."""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from ouroboros.betterleaks_runtime import (
    BETTERLEAKS_INSTALL_COMMAND,
    BETTERLEAKS_VERSION,
)
from ouroboros.platform_layer import (
    kill_process_tree,
    merge_hidden_kwargs,
    subprocess_new_group_kwargs,
)
from ouroboros.process_custody import spawn_supervised

BETTERLEAKS_ENGINE = "betterleaks"

_HOST_CONFIG_BYTES = b'title = "Ouroboros skill publish"\n\n[extend]\nuseDefault = true\n'
_HOST_CONFIG_SHA256 = hashlib.sha256(_HOST_CONFIG_BYTES).hexdigest()
_PAYLOAD_IGNORE_FILENAMES = frozenset({".betterleaksignore", ".gitleaksignore"})
_SCANNER_ENV_KEYS = frozenset(
    {
        "PATH",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
    }
)
_MAX_VERSION_OUTPUT_BYTES = 256
_MAX_RULESET_OUTPUT_BYTES = 512 * 1024
_MAX_REPORT_BYTES = 16 * 1024 * 1024
_CONFIG_SHOW_TIMEOUT_SEC = 15.0
_HOST_TIMEOUT_GRACE_SEC = 5.0
_DETECTOR_RE = re.compile(r"[^A-Za-z0-9._-]+")

ScannerAvailability = Literal["ready", "missing", "corrupt"]
ScannerStatus = Literal["clean", "findings", "scanner_error"]
FindingConfidence = Literal["low", "medium", "high", "unknown"]
FindingDisposition = Literal["blocker", "warning", "audited_false_positive"]
ProcessScope = Literal["task", "session"]

VALID_SCANNER_REASON_CODES = frozenset(
    {
        "",
        "scanner_missing",
        "scanner_corrupt",
        "scanner_timeout",
        "scanner_report_invalid",
        "scanner_ruleset_invalid",
        "scanner_input_invalid",
    }
)


@dataclass(frozen=True)
class ScannerExecutable:
    """Dependency-injected runtime identity; no runtime resolver import."""

    path: pathlib.Path | None
    identity: str
    status: ScannerAvailability = "ready"


@dataclass(frozen=True)
class SecretFinding:
    """Candidate-free finding safe for UI, model, logs, and receipts."""

    path: str
    line: int
    detector: str
    confidence: FindingConfidence
    reason: str
    verification: Literal["not_attempted"]
    disposition: FindingDisposition


@dataclass(frozen=True)
class SecretScanResult:
    """Complete safe classification for one named-byte scan."""

    status: ScannerStatus
    engine: str
    version: str
    ruleset_sha256: str
    scan_contract_sha256: str
    findings: tuple[SecretFinding, ...]
    blocker_count: int
    warning_count: int
    audited_false_positive_count: int
    reason_code: str = ""
    repair_hint: str = ""


@dataclass(frozen=True)
class _ScannerContract:
    version: str
    ruleset_sha256: str
    ruleset_byte_count: int
    scan_contract_sha256: str


@dataclass(frozen=True)
class _ParsedFinding:
    path: str
    line: int
    column: int
    detector: str
    confidence: FindingConfidence

    def coordinate(self) -> tuple[str, int, int, str, str]:
        return (self.path, self.line, self.column, self.detector, self.confidence)


class _ReportInvalid(RuntimeError):
    pass


def _repair_hint(reason_code: str) -> str:
    if reason_code == "scanner_input_invalid":
        return "Capture canonical unique payload paths, then retry."
    return f"Run `{BETTERLEAKS_INSTALL_COMMAND}`, then retry."


def _failure(
    reason_code: str,
    *,
    version: str = "",
    ruleset_sha256: str = "",
    scan_contract_sha256: str = "",
) -> SecretScanResult:
    if reason_code not in VALID_SCANNER_REASON_CODES or not reason_code:
        reason_code = "scanner_report_invalid"
    return SecretScanResult(
        status="scanner_error",
        engine=BETTERLEAKS_ENGINE,
        version=version,
        ruleset_sha256=ruleset_sha256,
        scan_contract_sha256=scan_contract_sha256,
        findings=(),
        blocker_count=0,
        warning_count=0,
        audited_false_positive_count=0,
        reason_code=reason_code,
        repair_hint=_repair_hint(reason_code),
    )


def _scanner_env() -> dict[str, str]:
    """Minimal process substrate with all scanner/provider credentials absent."""
    return {key: value for key in _SCANNER_ENV_KEYS if (value := os.environ.get(key)) is not None}


def _normalize_named_bytes(
    named_bytes: Mapping[str, bytes] | Iterable[tuple[str, bytes]],
) -> list[tuple[str, bytes]] | None:
    items = named_bytes.items() if isinstance(named_bytes, Mapping) else named_bytes
    normalized: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    try:
        for raw_path, raw_content in items:
            rel = str(raw_path or "")
            pure = pathlib.PurePosixPath(rel)
            if (
                not rel
                or "\x00" in rel
                or "\\" in rel
                or pure.is_absolute()
                or pathlib.PureWindowsPath(rel).is_absolute()
                or any(part in {"", ".", ".."} for part in pure.parts)
                or pure.as_posix() != rel
                or rel in seen
            ):
                return None
            if not isinstance(raw_content, (bytes, bytearray, memoryview)):
                return None
            seen.add(rel)
            normalized.append((rel, bytes(raw_content)))
    except (TypeError, ValueError):
        return None
    return sorted(normalized, key=lambda item: item[0])


def _neutral_physical_path(
    original: str,
    *,
    original_paths: set[str],
    physical_paths: set[str],
) -> str:
    pure = pathlib.PurePosixPath(original)
    if pure.name.lower() not in _PAYLOAD_IGNORE_FILENAMES:
        return original
    token = hashlib.sha256(original.encode("utf-8")).hexdigest()
    parent = pure.parent
    suffix = 0
    while True:
        name = f".ouroboros-scanned-ignore-{token}.payload"
        if suffix:
            name = f".ouroboros-scanned-ignore-{token}-{suffix}.payload"
        candidate = (parent / name).as_posix()
        if candidate not in original_paths and candidate not in physical_paths:
            return candidate
        suffix += 1


def _materialize_projection(
    projection: pathlib.Path,
    items: list[tuple[str, bytes]],
) -> dict[str, str]:
    original_paths = {path for path, _content in items}
    physical_paths: set[str] = set()
    path_map: dict[str, str] = {}
    for original, content in items:
        physical = _neutral_physical_path(
            original,
            original_paths=original_paths,
            physical_paths=physical_paths,
        )
        target = projection.joinpath(*pathlib.PurePosixPath(physical).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        try:
            os.chmod(target, 0o600)
        except OSError:
            pass
        physical_paths.add(physical)
        path_map[physical] = original
    return path_map


def _spawn_kwargs() -> dict[str, object]:
    return merge_hidden_kwargs(subprocess_new_group_kwargs())


def _run_capture(
    cmd: list[str],
    *,
    drive_root: pathlib.Path,
    scope: ProcessScope,
    owner_task_id: str,
    purpose: str,
    timeout_sec: float,
    output_limit: int,
    cwd: pathlib.Path,
) -> tuple[str, bytes]:
    try:
        proc = spawn_supervised(
            cmd,
            drive_root=drive_root,
            purpose=purpose,
            scope=scope,
            owner_task_id=owner_task_id,
            new_process_group=False,
            cwd=str(cwd),
            env=_scanner_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            **_spawn_kwargs(),
        )
    except (OSError, RuntimeError, ValueError):
        return "failed", b""
    try:
        stdout, _stderr = proc.communicate(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        return "timeout", b""
    if proc.returncode != 0:
        return "failed", b""
    data = bytes(stdout or b"")
    if not data or len(data) > output_limit:
        return "failed", b""
    return "ok", data


def _contract_digest(
    executable: ScannerExecutable,
    *,
    version: str,
    ruleset_sha256: str,
    timeout_sec: int,
    honor_inline_allowances: bool,
) -> str:
    payload = {
        "binary_identity": str(executable.identity or ""),
        "engine": BETTERLEAKS_ENGINE,
        "host_config_sha256": _HOST_CONFIG_SHA256,
        "passes": (["normal", "audit_ignore_allow"] if honor_inline_allowances else ["forced_ignore_allow"]),
        "ruleset_sha256": ruleset_sha256,
        "timeout_sec": int(timeout_sec),
        "version": version,
        "flags": [
            "no_git",
            "redact_100",
            "confidence_low",
            "archive_depth_1",
            "decode_depth_1",
            "exit_code_0",
            "no_validation",
            "host_config",
            "host_ignore",
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _probe_scanner_contract(
    executable: ScannerExecutable,
    *,
    config_path: pathlib.Path,
    drive_root: pathlib.Path,
    scope: ProcessScope,
    owner_task_id: str,
    timeout_sec: int,
    honor_inline_allowances: bool,
    cwd: pathlib.Path,
) -> tuple[str, _ScannerContract | None]:
    assert executable.path is not None
    binary = str(executable.path)
    status, version_raw = _run_capture(
        [binary, "version"],
        drive_root=drive_root,
        scope=scope,
        owner_task_id=owner_task_id,
        purpose="skill_publish_scanner_version",
        timeout_sec=min(_CONFIG_SHOW_TIMEOUT_SEC, float(timeout_sec) + _HOST_TIMEOUT_GRACE_SEC),
        output_limit=_MAX_VERSION_OUTPUT_BYTES,
        cwd=cwd,
    )
    if status == "timeout":
        return "scanner_timeout", None
    try:
        version = version_raw.decode("utf-8").strip() if status == "ok" else ""
    except UnicodeDecodeError:
        version = ""
    if version != BETTERLEAKS_VERSION:
        return "scanner_corrupt", None

    status, ruleset_raw = _run_capture(
        [
            binary,
            "config",
            "show",
            "--config",
            str(config_path),
            "--no-banner",
            "--no-color",
            "--log-level",
            "error",
        ],
        drive_root=drive_root,
        scope=scope,
        owner_task_id=owner_task_id,
        purpose="skill_publish_scanner_ruleset",
        timeout_sec=_CONFIG_SHOW_TIMEOUT_SEC,
        output_limit=_MAX_RULESET_OUTPUT_BYTES,
        cwd=cwd,
    )
    if status == "timeout":
        return "scanner_timeout", None
    if status != "ok":
        return "scanner_ruleset_invalid", None
    try:
        ruleset_raw.decode("utf-8")
    except UnicodeDecodeError:
        return "scanner_ruleset_invalid", None
    ruleset_sha256 = hashlib.sha256(ruleset_raw).hexdigest()
    return (
        "",
        _ScannerContract(
            version=version,
            ruleset_sha256=ruleset_sha256,
            ruleset_byte_count=len(ruleset_raw),
            scan_contract_sha256=_contract_digest(
                executable,
                version=version,
                ruleset_sha256=ruleset_sha256,
                timeout_sec=timeout_sec,
                honor_inline_allowances=honor_inline_allowances,
            ),
        ),
    )


def _report_original_path(
    raw_path: object,
    *,
    projection: pathlib.Path,
    path_map: dict[str, str],
) -> str:
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise _ReportInvalid
    outer_path, separator, archive_member = raw_path.partition("!")
    if separator:
        member = pathlib.PurePosixPath(archive_member.replace("\\", "/"))
        if (
            not archive_member
            or "!" in archive_member
            or member.is_absolute()
            or pathlib.PureWindowsPath(archive_member).is_absolute()
            or any(part in {"", ".", ".."} for part in member.parts)
        ):
            raise _ReportInvalid
        archive_suffix = "!" + member.as_posix()
    else:
        archive_suffix = ""
    normalized = outer_path.replace("\\", os.sep)
    candidate = pathlib.Path(normalized)
    try:
        resolved = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (projection / candidate).resolve(strict=False)
        )
        physical = resolved.relative_to(projection.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        raise _ReportInvalid from None
    original = path_map.get(physical)
    if original is None:
        raise _ReportInvalid
    return original + archive_suffix


def _sanitize_detector(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    sanitized = _DETECTOR_RE.sub("_", raw.strip()).strip("._-")[:128]
    return sanitized or "unknown"


def _parse_report(
    raw: bytes,
    *,
    projection: pathlib.Path,
    path_map: dict[str, str],
) -> list[_ParsedFinding]:
    try:
        decoded = raw.decode("utf-8")
        root = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise _ReportInvalid from None
    if root is None:
        return []
    if not isinstance(root, list):
        raise _ReportInvalid

    findings: list[_ParsedFinding] = []
    for row in root:
        if not isinstance(row, dict):
            raise _ReportInvalid
        attributes = row.get("Attributes")
        if attributes is None:
            attributes = {}
        if not isinstance(attributes, dict):
            raise _ReportInvalid
        report_path = attributes.get("path")
        if report_path in (None, ""):
            report_path = row.get("File")
        original_path = _report_original_path(
            report_path,
            projection=projection,
            path_map=path_map,
        )
        line = row.get("StartLine")
        column = row.get("StartColumn")
        if (
            not isinstance(line, int)
            or isinstance(line, bool)
            or line <= 0
            or not isinstance(column, int)
            or isinstance(column, bool)
            or column <= 0
        ):
            raise _ReportInvalid
        raw_confidence = attributes.get("confidence")
        confidence: FindingConfidence = raw_confidence if raw_confidence in {"low", "medium", "high"} else "unknown"
        findings.append(
            _ParsedFinding(
                path=original_path,
                line=line,
                column=column,
                detector=_sanitize_detector(row.get("RuleID")),
                confidence=confidence,
            )
        )
    return findings


def _detect_command(
    binary: pathlib.Path,
    *,
    projection: pathlib.Path,
    config_path: pathlib.Path,
    ignore_path: pathlib.Path,
    report_path: pathlib.Path,
    timeout_sec: int,
    ignore_inline_allowances: bool,
) -> list[str]:
    cmd = [
        str(binary),
        "detect",
        "--no-git",
        "--source",
        str(projection),
        "--config",
        str(config_path),
        "--gitleaks-ignore-path",
        str(ignore_path),
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
        str(timeout_sec),
        "--report-format",
        "json",
        "--report-path",
        str(report_path),
        "--no-banner",
        "--no-color",
        "--log-level",
        "error",
    ]
    if ignore_inline_allowances:
        cmd.append("--ignore-gitleaks-allow")
    return cmd


def _run_detect(
    binary: pathlib.Path,
    *,
    projection: pathlib.Path,
    config_path: pathlib.Path,
    ignore_path: pathlib.Path,
    report_path: pathlib.Path,
    path_map: dict[str, str],
    timeout_sec: int,
    ignore_inline_allowances: bool,
    drive_root: pathlib.Path,
    scope: ProcessScope,
    owner_task_id: str,
) -> tuple[str, list[_ParsedFinding] | None]:
    cmd = _detect_command(
        binary,
        projection=projection,
        config_path=config_path,
        ignore_path=ignore_path,
        report_path=report_path,
        timeout_sec=timeout_sec,
        ignore_inline_allowances=ignore_inline_allowances,
    )
    try:
        proc = spawn_supervised(
            cmd,
            drive_root=drive_root,
            purpose="skill_publish_scanner_detect",
            scope=scope,
            owner_task_id=owner_task_id,
            new_process_group=False,
            cwd=str(projection.parent),
            env=_scanner_env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_spawn_kwargs(),
        )
    except (OSError, RuntimeError, ValueError):
        return "scanner_corrupt", None
    try:
        proc.wait(timeout=float(timeout_sec) + _HOST_TIMEOUT_GRACE_SEC)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except (OSError, subprocess.SubprocessError):
            pass
        return "scanner_timeout", None
    if proc.returncode != 0:
        return "scanner_report_invalid", None
    try:
        if not report_path.is_file():
            return "scanner_report_invalid", None
        size = report_path.stat().st_size
        if size <= 0 or size > _MAX_REPORT_BYTES:
            return "scanner_report_invalid", None
        raw_report = report_path.read_bytes()
        if len(raw_report) != size:
            return "scanner_report_invalid", None
        return "", _parse_report(
            raw_report,
            projection=projection,
            path_map=path_map,
        )
    except (OSError, _ReportInvalid):
        return "scanner_report_invalid", None


def _safe_finding(
    finding: _ParsedFinding,
    *,
    disposition: FindingDisposition | None = None,
) -> SecretFinding:
    selected = disposition or ("blocker" if finding.confidence == "high" else "warning")
    reason = {
        "blocker": "High-confidence secret candidate detected.",
        "warning": "Scanner finding requires review before publication.",
        "audited_false_positive": "Inline allowance was surfaced by the audit pass.",
    }[selected]
    return SecretFinding(
        path=finding.path,
        line=finding.line,
        detector=finding.detector,
        confidence=finding.confidence,
        reason=reason,
        verification="not_attempted",
        disposition=selected,
    )


def _finding_sort_key(finding: SecretFinding) -> tuple[object, ...]:
    disposition_rank = {"blocker": 0, "warning": 1, "audited_false_positive": 2}
    return (
        finding.path,
        finding.line,
        finding.detector,
        finding.confidence,
        disposition_rank[finding.disposition],
    )


def _successful_result(
    contract: _ScannerContract,
    findings: Iterable[SecretFinding],
) -> SecretScanResult:
    ordered = tuple(sorted(findings, key=_finding_sort_key))
    blockers = sum(item.disposition == "blocker" for item in ordered)
    warnings = sum(item.disposition == "warning" for item in ordered)
    audited = sum(item.disposition == "audited_false_positive" for item in ordered)
    return SecretScanResult(
        status="findings" if ordered else "clean",
        engine=BETTERLEAKS_ENGINE,
        version=contract.version,
        ruleset_sha256=contract.ruleset_sha256,
        scan_contract_sha256=contract.scan_contract_sha256,
        findings=ordered,
        blocker_count=blockers,
        warning_count=warnings,
        audited_false_positive_count=audited,
    )


def scan_named_bytes(
    named_bytes: Mapping[str, bytes] | Iterable[tuple[str, bytes]],
    *,
    executable: ScannerExecutable,
    drive_root: pathlib.Path,
    scope: ProcessScope = "task",
    owner_task_id: str = "",
    timeout_sec: int = 30,
    honor_inline_allowances: bool = True,
) -> SecretScanResult:
    """Scan exact named bytes without granting payload-owned scanner policy.

    Payload mode (the default) runs normal and audit passes and reports the
    audit-only multiset as ``audited_false_positive``. Derived-text callers set
    ``honor_inline_allowances=False`` for one forced-ignore pass.
    """
    items = _normalize_named_bytes(named_bytes)
    if items is None or scope not in {"task", "session"} or not isinstance(timeout_sec, int) or timeout_sec <= 0:
        return _failure("scanner_input_invalid")
    if executable.status == "missing":
        return _failure("scanner_missing")
    if executable.status != "ready":
        return _failure("scanner_corrupt")
    if executable.path is None:
        return _failure("scanner_missing")
    if not str(executable.identity or ""):
        return _failure("scanner_corrupt")
    binary = pathlib.Path(executable.path)
    try:
        if not binary.is_absolute() or not binary.is_file():
            return _failure("scanner_corrupt")
    except OSError:
        return _failure("scanner_corrupt")

    try:
        with tempfile.TemporaryDirectory(prefix="ouroboros-skill-publish-scan-") as tmp:
            root = pathlib.Path(tmp)
            try:
                os.chmod(root, 0o700)
            except OSError:
                pass
            config_path = root / "host-config.toml"
            ignore_path = root / "host-ignore"
            config_path.write_bytes(_HOST_CONFIG_BYTES)
            ignore_path.write_bytes(b"")
            for private_file in (config_path, ignore_path):
                try:
                    os.chmod(private_file, 0o600)
                except OSError:
                    pass

            reason, contract = _probe_scanner_contract(
                executable,
                config_path=config_path,
                drive_root=pathlib.Path(drive_root),
                scope=scope,
                owner_task_id=owner_task_id,
                timeout_sec=timeout_sec,
                honor_inline_allowances=honor_inline_allowances,
                cwd=root,
            )
            if contract is None:
                return _failure(reason)

            projection = root / "projection"
            projection.mkdir(mode=0o700)
            path_map = _materialize_projection(projection, items)
            normal_report = root / "normal-report.json"
            reason, normal = _run_detect(
                binary,
                projection=projection,
                config_path=config_path,
                ignore_path=ignore_path,
                report_path=normal_report,
                path_map=path_map,
                timeout_sec=timeout_sec,
                ignore_inline_allowances=not honor_inline_allowances,
                drive_root=pathlib.Path(drive_root),
                scope=scope,
                owner_task_id=owner_task_id,
            )
            if normal is None:
                return _failure(
                    reason,
                    version=contract.version,
                    ruleset_sha256=contract.ruleset_sha256,
                    scan_contract_sha256=contract.scan_contract_sha256,
                )
            if not honor_inline_allowances:
                return _successful_result(
                    contract,
                    (_safe_finding(item) for item in normal),
                )

            audit_report = root / "audit-report.json"
            reason, audit = _run_detect(
                binary,
                projection=projection,
                config_path=config_path,
                ignore_path=ignore_path,
                report_path=audit_report,
                path_map=path_map,
                timeout_sec=timeout_sec,
                ignore_inline_allowances=True,
                drive_root=pathlib.Path(drive_root),
                scope=scope,
                owner_task_id=owner_task_id,
            )
            if audit is None:
                return _failure(
                    reason,
                    version=contract.version,
                    ruleset_sha256=contract.ruleset_sha256,
                    scan_contract_sha256=contract.scan_contract_sha256,
                )

            normal_counts = collections.Counter(item.coordinate() for item in normal)
            audited_only: list[_ParsedFinding] = []
            for item in audit:
                coordinate = item.coordinate()
                if normal_counts[coordinate] > 0:
                    normal_counts[coordinate] -= 1
                else:
                    audited_only.append(item)
            safe_findings = [*(_safe_finding(item) for item in normal)]
            safe_findings.extend(_safe_finding(item, disposition="audited_false_positive") for item in audited_only)
            return _successful_result(contract, safe_findings)
    except (OSError, RuntimeError, ValueError):
        return _failure("scanner_report_invalid")


__all__ = [
    "BETTERLEAKS_ENGINE",
    "BETTERLEAKS_VERSION",
    "VALID_SCANNER_REASON_CODES",
    "ScannerExecutable",
    "SecretFinding",
    "SecretScanResult",
    "scan_named_bytes",
]
