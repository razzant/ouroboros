"""Selected-skill publication preflight contract and cache tests."""

from __future__ import annotations

import asyncio
import inspect
import pathlib
import types
from typing import get_args, get_type_hints

import pytest

from ouroboros import skill_publish_scanner as scanner_adapter
from ouroboros.contracts.skill_manifest import SkillManifest
from ouroboros.gateway import skill_publish as preflight
from ouroboros.skill_loader import LoadedSkill, SkillReviewState
from ouroboros.skill_publish_result import (
    SkillPublishDestinationError,
    parse_skill_publish_destination,
)
from ouroboros.skill_publish_scanner import SecretFinding, SecretScanResult
from ouroboros.skill_publish_snapshot import (
    CapturedPublishManifest,
    CapturedSkillFile,
    SkillPublishSnapshot,
    SkillPublishSnapshotError,
)
from ouroboros.skill_review_status import (
    STATUS_BLOCKERS,
    STATUS_CLEAN,
    STATUS_PENDING,
    STATUS_WARNINGS,
)

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_CONTRACT_A = "c" * 64
_DESTINATION = "https://raw.githubusercontent.com/ExampleOwner/ExampleHub/main/catalog.json"


@pytest.fixture(autouse=True)
def _clear_preflight_cache():
    preflight._PREFLIGHT_SCAN_CACHE.clear()
    yield
    preflight._PREFLIGHT_SCAN_CACHE.clear()


def _loaded(
    *,
    status: str = STATUS_CLEAN,
    source: str = "external",
    review_hash: str = _HASH_A,
    version: str = "1.2.3",
    profile: str = "",
    collision: bool = False,
    load_error: str = "",
) -> LoadedSkill:
    return LoadedSkill(
        name="demo",
        skill_dir=pathlib.Path("/fixture/demo"),
        manifest=SkillManifest(
            name="demo",
            description="Demo skill.",
            version=version,
            type="instruction",
        ),
        content_hash=_HASH_A,
        review=SkillReviewState(
            status=status,
            content_hash=review_hash,
            review_profile=profile,
        ),
        load_error=load_error,
        source=source,
        identity_collision=collision,
    )


def _snapshot(*, content_hash: str = _HASH_A, version: str = "1.2.3") -> SkillPublishSnapshot:
    manifest = CapturedSkillFile.from_bytes(
        "SKILL.md",
        (f"---\nname: demo\ndescription: Demo skill.\nversion: {version}\ntype: instruction\n---\n# Demo\n").encode(),
    )
    payload = CapturedSkillFile.from_bytes("payload.txt", b"safe payload\n")
    return SkillPublishSnapshot(
        skill="demo",
        source="external",
        manifest_file=manifest,
        manifest=CapturedPublishManifest(
            path="SKILL.md",
            name="demo",
            description="Demo skill.",
            version=version,
            skill_type="instruction",
            when_to_use="",
        ),
        content_hash=content_hash,
        full_files=(manifest, payload),
        public_files=(manifest, payload),
        control_files=(),
    )


def _finding(index: int, *, disposition: str = "warning") -> SecretFinding:
    return SecretFinding(
        path=f"files/{index:03d}.txt",
        line=index + 1,
        detector="fixture-detector",
        confidence="high" if disposition == "blocker" else "medium",
        reason="High-confidence secret candidate detected."
        if disposition == "blocker"
        else "Scanner finding requires review before publication.",
        verification="not_attempted",
        disposition=disposition,  # type: ignore[arg-type]
    )


def _scan(
    findings: tuple[SecretFinding, ...] = (),
    *,
    status: str | None = None,
    contract: str = _CONTRACT_A,
    reason_code: str = "",
    repair_hint: str = "",
) -> SecretScanResult:
    return SecretScanResult(
        status=status or ("findings" if findings else "clean"),  # type: ignore[arg-type]
        engine="betterleaks",
        version="1.8.1",
        ruleset_sha256="d" * 64 if contract else "",
        scan_contract_sha256=contract,
        findings=findings,
        blocker_count=sum(item.disposition == "blocker" for item in findings),
        warning_count=sum(item.disposition == "warning" for item in findings),
        audited_false_positive_count=sum(item.disposition == "audited_false_positive" for item in findings),
        reason_code=reason_code,
        repair_hint=repair_hint,
    )


def _contract(
    contract: str = _CONTRACT_A,
    *,
    status: str = "ready",
    reason_code: str = "",
    repair_hint: str = "",
) -> preflight.ScannerContractResult:
    return preflight.ScannerContractResult(
        status=status,  # type: ignore[arg-type]
        engine="betterleaks",
        version="1.8.1" if status == "ready" else "",
        ruleset_sha256="d" * 64 if status == "ready" else "",
        scan_contract_sha256=contract if status == "ready" else "",
        reason_code=reason_code,
        repair_hint=repair_hint,
    )


def _patch_domain(
    monkeypatch: pytest.MonkeyPatch,
    loaded: LoadedSkill,
    snapshot: SkillPublishSnapshot,
    scan: SecretScanResult,
    calls: list[dict[str, object]],
    *,
    executable_identity: str = "e" * 64,
) -> None:
    monkeypatch.setattr(
        preflight,
        "discover_selected_skill_candidates",
        lambda *_a, **_kw: [loaded],
    )
    monkeypatch.setattr(
        preflight,
        "capture_skill_publish_candidate",
        lambda _loaded: snapshot,
    )
    monkeypatch.setattr(
        preflight,
        "resolve_betterleaks",
        lambda **_kw: types.SimpleNamespace(
            binary_path="/fixture/betterleaks",
            binary_sha256=executable_identity,
            status="ready",
        ),
    )
    monkeypatch.setattr(
        preflight,
        "probe_scanner_contract",
        lambda **_kw: _contract(scan.scan_contract_sha256 or _CONTRACT_A),
    )

    def fake_scan(_named_bytes, **kwargs):
        calls.append(dict(kwargs))
        return scan

    monkeypatch.setattr(preflight, "scan_named_bytes", fake_scan)


def _build(tmp_path: pathlib.Path):
    return preflight.build_skill_publish_preflight(
        tmp_path,
        "demo",
        repo_path="",
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )


def test_destination_parser_is_shared_case_preserving_and_closed():
    assert parse_skill_publish_destination(_DESTINATION) == (
        "ExampleOwner",
        "ExampleHub",
        "main",
    )
    for invalid in (
        "http://raw.githubusercontent.com/o/r/main/catalog.json",
        "https://github.com/o/r/main/catalog.json",
        "https://raw.githubusercontent.com/o/r/catalog.json",
        "https://raw.githubusercontent.com/o/r/main/catalog.json?ref=x",
    ):
        with pytest.raises(SkillPublishDestinationError) as caught:
            parse_skill_publish_destination(invalid)
        assert caught.value.reason_code == "hub_destination_invalid"


def test_preflight_state_vocabulary_matches_gateway_contract():
    from ouroboros.gateway.contracts import SkillPublishPreflightResponse

    state_type = get_type_hints(
        SkillPublishPreflightResponse,
        include_extras=True,
    )["state"]
    assert frozenset(get_args(state_type)) == preflight.PREFLIGHT_STATES


@pytest.mark.parametrize(
    (
        "status",
        "profile",
        "review_hash",
        "version",
        "scan",
        "expected_state",
        "reason_code",
        "publication_ready",
    ),
    [
        (STATUS_CLEAN, "", _HASH_A, "1.2.3", _scan(), "ready", "", True),
        (
            STATUS_WARNINGS,
            "",
            _HASH_A,
            "1.2.3",
            _scan(),
            "warnings",
            "warnings_present",
            True,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_A,
            "1.2.3",
            _scan((_finding(0),)),
            "warnings",
            "warnings_present",
            True,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_A,
            "1.2.3",
            _scan((_finding(0, disposition="blocker"),)),
            "needs_attention",
            "secret_blocked",
            False,
        ),
        (
            STATUS_BLOCKERS,
            "",
            _HASH_A,
            "1.2.3",
            _scan(),
            "needs_attention",
            "review_blockers",
            False,
        ),
        (
            STATUS_PENDING,
            "",
            _HASH_A,
            "1.2.3",
            _scan(),
            "needs_attention",
            "review_pending",
            False,
        ),
        (
            STATUS_CLEAN,
            "owner_attested",
            _HASH_A,
            "1.2.3",
            _scan(),
            "needs_attention",
            "review_owner_attested",
            False,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_B,
            "1.2.3",
            _scan(),
            "needs_attention",
            "review_stale",
            False,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_A,
            "",
            _scan(),
            "needs_attention",
            "manifest_version_missing",
            False,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_A,
            "1.2.3",
            _scan(
                status="scanner_error",
                contract="",
                reason_code="scanner_missing",
                repair_hint="Install Betterleaks, then retry.",
            ),
            "repairable",
            "scanner_missing",
            False,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_A,
            "1.2.3",
            _scan(
                status="scanner_error",
                contract="",
                reason_code="scanner_corrupt",
                repair_hint="Repair Betterleaks, then retry.",
            ),
            "repairable",
            "scanner_corrupt",
            False,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_A,
            "1.2.3",
            _scan(
                status="scanner_error",
                contract="",
                reason_code="scanner_timeout",
                repair_hint="Repair Betterleaks, then retry.",
            ),
            "repairable",
            "scanner_timeout",
            False,
        ),
        (
            STATUS_CLEAN,
            "",
            _HASH_A,
            "1.2.3",
            _scan(
                status="scanner_error",
                contract="",
                reason_code="scanner_report_invalid",
                repair_hint="Repair Betterleaks, then retry.",
            ),
            "repairable",
            "scanner_report_invalid",
            False,
        ),
    ],
)
def test_preflight_state_matrix_scans_current_bytes_before_review_classification(
    tmp_path,
    monkeypatch,
    status,
    profile,
    review_hash,
    version,
    scan,
    expected_state,
    reason_code,
    publication_ready,
):
    loaded = _loaded(
        status=status,
        profile=profile,
        review_hash=review_hash,
        version=version,
    )
    snapshot = _snapshot(version=version)
    calls: list[dict[str, object]] = []
    _patch_domain(monkeypatch, loaded, snapshot, scan, calls)

    outcome = _build(tmp_path)

    assert outcome.status_code == 200
    assert outcome.payload["state"] == expected_state
    assert outcome.payload["reason_code"] == reason_code
    assert outcome.payload["publication_ready"] is publication_ready
    assert outcome.payload["task_start_allowed"] is True
    assert outcome.payload["repository"] == "ExampleOwner/ExampleHub"
    assert outcome.payload["snapshot_hash"] == _HASH_A
    assert len(calls) == 1
    assert calls[0]["scope"] == "session"
    assert calls[0]["owner_task_id"] == ""
    assert len(preflight._PREFLIGHT_SCAN_CACHE) == (0 if expected_state == "repairable" else 1)


def test_contract_probe_failure_is_repairable_without_detect_or_cache(tmp_path, monkeypatch):
    calls: list[dict[str, object]] = []
    _patch_domain(monkeypatch, _loaded(), _snapshot(), _scan(), calls)
    monkeypatch.setattr(
        preflight,
        "probe_scanner_contract",
        lambda **_kw: _contract(
            status="scanner_error",
            reason_code="scanner_ruleset_invalid",
            repair_hint="Repair Betterleaks, then retry.",
        ),
    )

    def unexpected_detect(*_args, **_kwargs):
        raise AssertionError("failed contract probe reached detect")

    monkeypatch.setattr(preflight, "scan_named_bytes", unexpected_detect)
    payload = _build(tmp_path).payload

    assert payload["state"] == "repairable"
    assert payload["reason_code"] == "scanner_ruleset_invalid"
    assert payload["scanner"]["status"] == "scanner_error"
    assert len(preflight._PREFLIGHT_SCAN_CACHE) == 0


@pytest.mark.parametrize("reason_code", ["scanner_missing", "scanner_timeout"])
def test_preflight_repair_hint_keeps_installer_authority(tmp_path, monkeypatch, reason_code):
    result = scanner_adapter._failure(reason_code)
    _patch_domain(monkeypatch, _loaded(), _snapshot(), result, [])
    payload = _build(tmp_path).payload
    assert (payload["state"], payload["reason_code"]) == ("repairable", reason_code)
    assert payload["repair_hint"] == result.repair_hint


@pytest.mark.parametrize(
    "capture_reason",
    [
        "snapshot_skill_unreadable",
        "snapshot_manifest_missing",
        "snapshot_manifest_invalid",
        "snapshot_payload_unreadable",
        "snapshot_payload_too_large",
    ],
)
def test_known_broken_skill_capture_is_http_200_agent_repairable(tmp_path, monkeypatch, capture_reason):
    loaded = _loaded(load_error="raw loader detail that must not leave")
    monkeypatch.setattr(
        preflight,
        "discover_selected_skill_candidates",
        lambda *_a, **_kw: [loaded],
    )

    def fail_capture(_loaded):
        raise SkillPublishSnapshotError(capture_reason)

    monkeypatch.setattr(preflight, "capture_skill_publish_candidate", fail_capture)

    outcome = _build(tmp_path)

    assert outcome.status_code == 200
    assert outcome.payload["state"] == "needs_attention"
    assert outcome.payload["publication_ready"] is False
    assert outcome.payload["task_start_allowed"] is True
    assert outcome.payload["snapshot_hash"] == ""
    assert outcome.payload["scanner"]["status"] == "not_run"
    assert outcome.payload["reason_code"] == capture_reason
    assert "raw loader detail" not in str(outcome.payload)


def test_discovered_invalid_external_manifest_remains_agent_repairable(tmp_path, monkeypatch):
    skill_dir = tmp_path / "skills" / "external" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: [unterminated\n---\n",
        encoding="utf-8",
    )

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("broken manifest reached scanner")

    monkeypatch.setattr(preflight, "scan_named_bytes", unexpected_scan)

    outcome = preflight.build_skill_publish_preflight(
        tmp_path,
        "broken",
        repo_path="",
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )

    assert outcome.status_code == 200
    assert outcome.payload["state"] == "needs_attention"
    assert outcome.payload["task_start_allowed"] is True
    assert outcome.payload["reason_code"] == "snapshot_skill_unreadable"
    assert "manifest parse error" not in str(outcome.payload)


@pytest.mark.parametrize("layout", ["data_bucket", "repo_root"])
def test_selected_card_survives_manifest_deletion_without_global_discovery(
    tmp_path,
    monkeypatch,
    layout,
):
    from ouroboros.skill_loader import discover_skills

    skill_dir = tmp_path / "skills" / "external" / "recover" if layout == "data_bucket" else tmp_path / "recover"
    repo_path = "" if layout == "data_bucket" else str(skill_dir)
    skill_dir.mkdir(parents=True)
    manifest = skill_dir / "SKILL.md"
    manifest.write_text(
        "---\nname: recover\ndescription: Recover.\nversion: 1.0.0\ntype: instruction\n---\n# Recover\n",
        encoding="utf-8",
    )
    assert [skill.name for skill in discover_skills(tmp_path, repo_path=repo_path)] == ["recover"]
    manifest.unlink()
    assert discover_skills(tmp_path, repo_path=repo_path) == []

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("manifestless skill reached scanner")

    monkeypatch.setattr(preflight, "scan_named_bytes", unexpected_scan)
    outcome = preflight.build_skill_publish_preflight(
        tmp_path,
        "recover",
        repo_path=repo_path,
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )

    assert outcome.status_code == 200
    assert outcome.payload["state"] == "needs_attention"
    assert outcome.payload["publication_ready"] is False
    assert outcome.payload["task_start_allowed"] is True
    assert outcome.payload["scanner"]["status"] == "not_run"
    assert outcome.payload["reason_code"] == "snapshot_manifest_missing"


@pytest.mark.parametrize("layout", ["data_group", "repo_root"])
def test_manifestless_container_with_manifested_children_is_not_selected(
    tmp_path,
    monkeypatch,
    layout,
):
    from ouroboros.skill_loader import (
        discover_selected_skill_candidates,
        discover_skills,
    )

    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    container = drive_root / "skills" / "demo" if layout == "data_group" else tmp_path / "demo"
    repo_path = "" if layout == "data_group" else str(container)
    child = container / "child"
    child.mkdir(parents=True)
    (child / "SKILL.md").write_text(
        "---\nname: child\ndescription: Child.\nversion: 1.0.0\ntype: instruction\n---\n# Child\n",
        encoding="utf-8",
    )

    ordinary = discover_skills(drive_root, repo_path=repo_path)
    assert [skill.skill_dir for skill in ordinary] == [child.resolve()]
    assert (
        discover_selected_skill_candidates(
            drive_root,
            "demo",
            repo_path=repo_path,
        )
        == []
    )

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("manifestless grouping container reached scanner")

    monkeypatch.setattr(preflight, "scan_named_bytes", unexpected_scan)
    outcome = preflight.build_skill_publish_preflight(
        drive_root,
        "demo",
        repo_path=repo_path,
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )

    assert outcome.status_code == 404
    assert outcome.payload["reason_code"] == "skill_not_found"
    assert outcome.payload["task_start_allowed"] is False


def test_manifestless_target_participates_in_hidden_identity_collision(tmp_path, monkeypatch):
    from ouroboros.skill_loader import discover_skills

    (tmp_path / "skills" / "external" / "demo").mkdir(parents=True)
    checkout = tmp_path / "checkout"
    valid = checkout / "demo"
    valid.mkdir(parents=True)
    (valid / "SKILL.md").write_text(
        "---\nname: demo\ndescription: Demo.\nversion: 1.0.0\ntype: instruction\n---\n# Demo\n",
        encoding="utf-8",
    )
    ordinary = discover_skills(tmp_path, repo_path=str(checkout))
    assert len(ordinary) == 1 and ordinary[0].identity_collision is False

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("ambiguous skill reached scanner")

    monkeypatch.setattr(preflight, "scan_named_bytes", unexpected_scan)
    outcome = preflight.build_skill_publish_preflight(
        tmp_path,
        "demo",
        repo_path=str(checkout),
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )

    assert outcome.status_code == 409
    assert outcome.payload["reason_code"] == "skill_identity_ambiguous"
    assert outcome.payload["task_start_allowed"] is False


def test_manifestless_bucket_is_unknown_and_seeded_native_stays_unsupported(
    tmp_path,
):
    native_bucket = tmp_path / "skills" / "native"
    native_bucket.mkdir(parents=True)
    unknown = preflight.build_skill_publish_preflight(
        tmp_path,
        "native",
        repo_path="",
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )
    assert (unknown.status_code, unknown.payload["reason_code"]) == (
        404,
        "skill_not_found",
    )

    seeded = native_bucket / "seeded"
    seeded.mkdir()
    (seeded / ".seed-origin").write_text("seeded\n", encoding="utf-8")
    unsupported = preflight.build_skill_publish_preflight(
        tmp_path,
        "seeded",
        repo_path="",
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )
    assert (unsupported.status_code, unsupported.payload["reason_code"]) == (
        409,
        "skill_source_unsupported",
    )


@pytest.mark.parametrize(
    ("case", "status_code", "reason_code"),
    [
        ("token", 409, "github_token_missing"),
        ("missing", 404, "skill_not_found"),
        ("ambiguous", 409, "skill_identity_ambiguous"),
        ("source", 409, "skill_source_unsupported"),
        ("admission", 503, "worker_pool_unavailable"),
    ],
)
def test_preflight_hard_refusal_matrix(tmp_path, monkeypatch, case, status_code, reason_code):
    loaded = _loaded(
        collision=case == "ambiguous",
        source="native" if case == "source" else "external",
    )
    monkeypatch.setattr(
        preflight,
        "discover_selected_skill_candidates",
        lambda *_a, **_kw: [] if case == "missing" else [loaded],
    )
    outcome = preflight.build_skill_publish_preflight(
        tmp_path,
        "demo",
        repo_path="",
        catalog_url=_DESTINATION,
        github_token_configured=case != "token",
        admission_problem=(503, "worker_pool_unavailable", "Unavailable") if case == "admission" else None,
    )

    assert outcome.status_code == status_code
    assert outcome.payload["ok"] is False
    assert outcome.payload["state"] == "hard_block"
    assert outcome.payload["task_start_allowed"] is False
    assert outcome.payload["reason_code"] == reason_code


def test_preflight_rejects_malformed_slug_and_destination_without_discovery(tmp_path, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "discover_selected_skill_candidates",
        lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("unexpected discovery")),
    )
    malformed = preflight.build_skill_publish_preflight(
        tmp_path,
        "../demo",
        repo_path="",
        catalog_url=_DESTINATION,
        github_token_configured=True,
    )
    invalid_hub = preflight.build_skill_publish_preflight(
        tmp_path,
        "demo",
        repo_path="",
        catalog_url="https://example.invalid/catalog.json",
        github_token_configured=True,
    )
    assert (malformed.status_code, malformed.payload["reason_code"]) == (
        400,
        "skill_invalid",
    )
    assert (invalid_hub.status_code, invalid_hub.payload["reason_code"]) == (
        409,
        "hub_destination_invalid",
    )


def test_cache_hit_recomputes_review_and_byte_or_runtime_identity_changes_miss(tmp_path, monkeypatch):
    loaded = _loaded()
    snapshots = [_snapshot(), _snapshot(), _snapshot(content_hash=_HASH_B), _snapshot()]
    snapshot_index = 0
    runtime_identity = "e" * 64
    contract_identity = _CONTRACT_A
    scan_calls = 0

    monkeypatch.setattr(
        preflight,
        "discover_selected_skill_candidates",
        lambda *_a, **_kw: [loaded],
    )

    def capture(_loaded):
        return snapshots[snapshot_index]

    monkeypatch.setattr(preflight, "capture_skill_publish_candidate", capture)

    def resolve(**_kw):
        return types.SimpleNamespace(
            binary_path="/fixture/betterleaks",
            binary_sha256=runtime_identity,
            status="ready",
        )

    monkeypatch.setattr(preflight, "resolve_betterleaks", resolve)
    monkeypatch.setattr(
        preflight,
        "probe_scanner_contract",
        lambda **_kw: _contract(contract_identity),
    )

    def scan_named(_named_bytes, **_kwargs):
        nonlocal scan_calls
        scan_calls += 1
        return _scan(contract=contract_identity)

    monkeypatch.setattr(preflight, "scan_named_bytes", scan_named)

    assert _build(tmp_path).payload["state"] == "ready"
    loaded.review.content_hash = _HASH_B
    assert _build(tmp_path).payload["reason_code"] == "review_stale"
    assert scan_calls == 1

    loaded.review.content_hash = _HASH_A
    loaded.review.status = STATUS_BLOCKERS
    assert _build(tmp_path).payload["reason_code"] == "review_blockers"
    assert scan_calls == 1

    loaded.review.status = STATUS_CLEAN
    contract_identity = "g" * 64
    assert _build(tmp_path).payload["state"] == "ready"
    assert scan_calls == 2

    snapshot_index = 2
    loaded.review.content_hash = _HASH_B
    assert _build(tmp_path).payload["snapshot_hash"] == _HASH_B
    assert scan_calls == 3

    snapshot_index = 3
    loaded.review.content_hash = _HASH_A
    runtime_identity = "f" * 64
    contract_identity = "h" * 64
    assert _build(tmp_path).payload["state"] == "ready"
    assert scan_calls == 4


def test_cache_and_response_keep_bounded_prefix_but_full_counts(tmp_path, monkeypatch):
    findings = tuple(_finding(index) for index in range(137))
    loaded = _loaded()
    calls: list[dict[str, object]] = []
    _patch_domain(monkeypatch, loaded, _snapshot(), _scan(findings), calls)

    payload = _build(tmp_path).payload

    assert len(payload["findings"]) == preflight._MAX_PREFLIGHT_FINDINGS
    assert payload["omitted_count"] == 37
    assert payload["warning_count"] == 137
    assert payload["blocker_count"] == 0
    assert len({row["path"] for row in payload["findings"]}) == 100
    assert _build(tmp_path).payload["findings"] == payload["findings"]
    assert len(calls) == 1
    cached_rows = list(preflight._PREFLIGHT_SCAN_CACHE._entries.values())
    assert len(cached_rows) == 1
    assert len(cached_rows[0].projection.findings) == 100
    assert cached_rows[0].projection.omitted_count == 37
    assert not isinstance(cached_rows[0].projection, SecretScanResult)
    assert len(preflight._PREFLIGHT_SCAN_CACHE._entries) <= preflight._SCAN_CACHE_CAPACITY


def test_finding_beyond_visible_prefix_still_blocks_from_full_counts(tmp_path, monkeypatch):
    findings = tuple(_finding(index) for index in range(101)) + (_finding(999, disposition="blocker"),)
    calls: list[dict[str, object]] = []
    _patch_domain(monkeypatch, _loaded(), _snapshot(), _scan(findings), calls)

    payload = _build(tmp_path).payload

    assert payload["state"] == "needs_attention"
    assert payload["reason_code"] == "secret_blocked"
    assert payload["blocker_count"] == 1
    assert payload["warning_count"] == 101
    assert payload["omitted_count"] == 2
    assert payload["findings"][0]["disposition"] == "blocker"
    assert sum(row["disposition"] == "warning" for row in payload["findings"]) == 99


def test_lru_key_includes_skill_snapshot_and_scan_contract():
    cache = preflight._SafeScanCache(capacity=4)
    first = preflight._safe_scan_projection(_scan(contract="1" * 64))
    second = preflight._safe_scan_projection(_scan(contract="2" * 64))
    cache.put(
        skill="demo",
        snapshot_hash=_HASH_A,
        executable_identity="e" * 64,
        projection=first,
    )
    cache.put(
        skill="demo",
        snapshot_hash=_HASH_A,
        executable_identity="f" * 64,
        projection=second,
    )
    assert set(cache._entries) == {
        ("demo", _HASH_A, "1" * 64),
        ("demo", _HASH_A, "2" * 64),
    }


def test_lru_capacity_evicts_oldest_exact_contract():
    cache = preflight._SafeScanCache(capacity=2)
    for index in range(3):
        projection = preflight._safe_scan_projection(_scan(contract=str(index) * 64))
        cache.put(
            skill=f"demo-{index}",
            snapshot_hash=_HASH_A,
            executable_identity=f"runtime-{index}",
            projection=projection,
        )
    assert len(cache) == 2
    assert ("demo-0", _HASH_A, "0" * 64) not in cache._entries


def test_handler_offloads_complete_sync_builder(monkeypatch, tmp_path):
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
    expected = preflight.SkillPublishPreflightOutcome(
        preflight._response(
            ok=True,
            skill="demo",
            repository="ExampleOwner/ExampleHub",
            state="ready",
            publication_ready=True,
            task_start_allowed=True,
        )
    )
    monkeypatch.setattr(preflight, "build_skill_publish_preflight", lambda *_a, **_kw: expected)

    async def fake_to_thread(func, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(preflight.asyncio, "to_thread", fake_to_thread)
    request = types.SimpleNamespace(
        path_params={"skill": "demo"},
        app=types.SimpleNamespace(
            state=types.SimpleNamespace(
                drive_root=tmp_path,
                supervisor_ready_event=None,
            )
        ),
    )

    response = asyncio.run(preflight.api_skill_publish_preflight(request))

    assert response.status_code == 200
    assert calls and calls[0][0] is preflight.build_skill_publish_preflight
    assert calls[0][1] == (tmp_path, "demo")


def test_api_route_preserves_typed_domain_status_and_exact_response_shape(monkeypatch, tmp_path):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    outcome = preflight._hard_block(
        skill="demo",
        repository="ExampleOwner/ExampleHub",
        reason_code="github_token_missing",
        summary="GitHub authority is not configured.",
        repair_hint="Configure authority, then retry.",
        status_code=409,
    )
    monkeypatch.setattr(
        preflight,
        "build_skill_publish_preflight",
        lambda *_a, **_kw: outcome,
    )
    app = Starlette(
        routes=[
            Route(
                "/api/skills/{skill}/publish-preflight",
                preflight.api_skill_publish_preflight,
                methods=["POST"],
            )
        ]
    )
    app.state.drive_root = tmp_path
    app.state.supervisor_ready_event = None

    response = TestClient(app).post("/api/skills/demo/publish-preflight")

    assert response.status_code == 409
    assert response.json() == outcome.payload
    assert set(response.json()) == {
        "ok",
        "skill",
        "repository",
        "state",
        "publication_ready",
        "task_start_allowed",
        "snapshot_hash",
        "review",
        "scanner",
        "findings",
        "omitted_count",
        "blocker_count",
        "warning_count",
        "audited_false_positive_count",
        "reason_code",
        "summary",
        "repair_hint",
    }


def test_authoritative_tool_uses_shared_destination_resolver_and_never_gateway_cache():
    from ouroboros.gateway import skill_publish as preflight
    from ouroboros.tools import skill_publish as authoritative

    # One resolver binds the destination for the preflight answer and the publishing tool.
    for module in (authoritative, preflight):
        source = inspect.getsource(module)
        assert "resolve_publish_destination" in source
        assert "_parse_hub_destination" not in source
    source = inspect.getsource(authoritative)
    assert "gateway.skill_publish" not in source
    assert "_PREFLIGHT_SCAN_CACHE" not in source


def test_passive_index_has_no_scanner_or_runtime_callsite():
    from ouroboros.gateway import extensions

    source = inspect.getsource(extensions._build_extensions_index)
    assert "scan_named_bytes" not in source
    assert "resolve_betterleaks" not in source
