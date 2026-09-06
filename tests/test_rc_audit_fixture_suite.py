"""RC audit fixture suite (ABI-7b, F13/F14) — the verification hook of the
ADOPTION ABI-7 row for the RC auditor half.

Runs ``scripts/rc_audit.py`` over fixture installs built from the shared N−1
catalog ``tests/fixtures/nminus1/`` — REAL bytes authored by the previous
minor, not synthetic shapes (F14; shared property with the ABI-2 quarantine
suite and the ABI-7a updater shim suite, whose N−1 byte forms are inline):

- ``settings_v6.113.4.json`` — written by the v6.113.4 ``config.save_settings``
  itself in an isolated root (carries the retired comma-list keys and
  ``OUROBOROS_SCOPE_REVIEW_FLOOR`` exactly as a real N−1 install did; all
  secret fields empty).
- ``task_result_v6.113.4.json`` — written by the v6.113.4
  ``task_results.write_task_result`` (no ``_schema_version`` stamp; carries a
  real stored ``cost_usd`` alias key).
- ``telegram_SKILL_v6.113.4.md`` — the bundled telegram extension manifest at
  f0313064 (the commit before ABI-1 added ``plugin_api``): a real pre-7.0
  extension manifest without the field.

Pinned semantics: N−1 install → exit 1 with the expected check ids; clean
7.0 install → exit 0; broken/unreadable install → exit 2; strict read-only
guarantee (byte-for-byte fixture-tree snapshot before/after, no new files).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "rc_audit.py"
FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "nminus1"


def _load_module():
    spec = importlib.util.spec_from_file_location("rc_audit_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(data_root: pathlib.Path, *extra: str, isolated_root: pathlib.Path,
         env_extra: dict | None = None):
    env = dict(os.environ)
    env.update({
        "OUROBOROS_APP_ROOT": str(isolated_root),
        "OUROBOROS_REPO_DIR": str(isolated_root / "repo"),
        "OUROBOROS_DATA_DIR": str(isolated_root / "data"),
        "OUROBOROS_SETTINGS_PATH": str(isolated_root / "data" / "settings.json"),
    })
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(data_root), *extra],
        capture_output=True, text=True, env=env, timeout=120,
    )


def _build_nminus1_install(root: pathlib.Path) -> pathlib.Path:
    data = root / "data"
    (data / "skills" / "external" / "telegram").mkdir(parents=True)
    (data / "task_results").mkdir()
    (data / "state").mkdir()
    shutil.copyfile(FIXTURES / "settings_v6.113.4.json", data / "settings.json")
    shutil.copyfile(FIXTURES / "task_result_v6.113.4.json",
                    data / "task_results" / "tsk_n1_fixture.json")
    shutil.copyfile(FIXTURES / "telegram_SKILL_v6.113.4.md",
                    data / "skills" / "external" / "telegram" / "SKILL.md")
    (data / "state" / "ui_preferences.json").write_text(
        json.dumps({"project_last_viewed": {"p1": 1}}), encoding="utf-8")
    return data


def _build_clean_70_install(root: pathlib.Path) -> pathlib.Path:
    data = root / "data"
    (data / "skills" / "external" / "telegram").mkdir(parents=True)
    (data / "task_results").mkdir()
    data.joinpath("settings.json").write_text(
        json.dumps({"TOTAL_BUDGET": "10", "OUROBOROS_REVIEWER_SLOTS": ""}, indent=2),
        encoding="utf-8")
    data.joinpath("task_results", "tsk_clean.json").write_text(
        json.dumps({
            "_schema_version": 1, "task_id": "tsk_clean", "status": "done",
            "summary": "clean 7.0 row",
        }), encoding="utf-8")
    # The CURRENT bundled manifest declares plugin_api and negotiates cleanly.
    shutil.copyfile(REPO / "skills" / "telegram" / "SKILL.md",
                    data / "skills" / "external" / "telegram" / "SKILL.md")
    return data


def _tree_snapshot(root: pathlib.Path):
    snapshot = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            snapshot[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()).hexdigest()
    return snapshot


# ------------------------------------------------------------- N−1 install


def test_nminus1_fixture_install_exits_1_with_the_expected_checks(tmp_path):
    data = _build_nminus1_install(tmp_path / "install")
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1, result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    incompatible = [f for f in report["findings"] if f["severity"] == "incompatible"]
    by_check = {}
    for f in incompatible:
        by_check.setdefault(f["check_id"], []).append(f["subject"])

    assert "settings.json:OUROBOROS_SCOPE_REVIEW_FLOOR" in by_check["retired-setting"]
    # D04 (owner 1B): the N−1 document carried the flat timeout pair at its
    # defaults, so retiring them must be VISIBLE to an upgrading install — a
    # default-valued ghost is exactly the one nobody would think to look for.
    for key in ("OUROBOROS_SOFT_TIMEOUT_SEC", "OUROBOROS_HARD_TIMEOUT_SEC"):
        assert f"settings.json:{key}" in by_check["retired-setting"]
    # The real N−1 settings document carried the comma-list keys as defaults.
    comma_subjects = set(by_check["comma-list"])
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS",
                "OUROBOROS_SCOPE_REVIEW_MODEL"):
        assert f"settings.json:{key}" in comma_subjects
    assert any("telegram" in s for s in by_check["plugin-api"])
    assert any("tsk_n1_fixture" in s for s in by_check["schema-stamp"])
    # ABI-2: the Q8=B consequence MUST be named, verbatim semantics.
    schema_findings = [f for f in incompatible if f["check_id"] == "schema-stamp"]
    assert any("Q8=B" in f["detail"] and "BY DESIGN" in f["detail"]
               for f in schema_findings)

    # Stored gateway-alias keys are notes (read-tolerance kept), never blocking.
    alias_findings = [f for f in report["findings"] if f["check_id"] == "gateway-alias"]
    assert alias_findings and all(f["severity"] == "note" for f in alias_findings)

    # F13: the owner-attestation list is printed, not silently absorbed.
    assert "OWNER ATTESTATION" in result.stdout
    assert len(report["owner_attestation"]) >= 5
    assert any("fail_tasks" in note for note in report["prose_notes"])


def _real_content_hash(skill_dir: pathlib.Path) -> str:
    """The runtime's own review-staleness hash over the fixture payload."""
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text
    from ouroboros.skill_loader import compute_content_hash

    manifest = parse_skill_manifest_text(
        (skill_dir / "SKILL.md").read_text(encoding="utf-8"))
    return compute_content_hash(
        skill_dir, manifest_entry=manifest.entry, manifest_scripts=manifest.scripts)


def _write_review_pass(data: pathlib.Path, state_name: str, content_hash: str,
                       **extra) -> None:
    review_dir = data / "state" / "skills" / state_name
    review_dir.mkdir(parents=True)
    (review_dir / "review.json").write_text(
        json.dumps({"status": "pass", "content_hash": content_hash, **extra}),
        encoding="utf-8")


def test_grandfathered_hash_bound_pass_downgrades_plugin_api_to_a_note(tmp_path):
    """The grandfather note requires a PASS bound to the payload's CURRENT
    bytes — the fixture stores the REAL computed hash, exactly as review did."""
    data = _build_nminus1_install(tmp_path / "install")
    skill_dir = data / "skills" / "external" / "telegram"
    _write_review_pass(data, "telegram", _real_content_hash(skill_dir))
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1  # other incompatibilities remain
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "note" for f in plugin_findings)
    assert any("GRANDFATHERED" in f["detail"] for f in plugin_findings)


def test_stale_hash_bound_pass_is_incompatible_not_a_grandfather_note(tmp_path):
    """A stored PASS whose hash does not match the current payload bytes is a
    STALE review: the runtime refuses to load it, so the audit must report an
    incompatibility, never the grandfather note (adversarial finding 2)."""
    data = _build_nminus1_install(tmp_path / "install")
    _write_review_pass(data, "telegram", "a" * 64)  # not the payload's hash
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "incompatible" for f in plugin_findings)
    assert any("STALE" in f["detail"] for f in plugin_findings)


def test_review_state_lookup_uses_the_directory_basename_not_manifest_name(tmp_path):
    """Runtime/state identity is the skill DIRECTORY basename
    (skill_loader.load_skill); a PASS stored under the manifest's display name
    for a differently-named directory must NOT grandfather the skill."""
    data = _build_nminus1_install(tmp_path / "install")
    skill_dir = data / "skills" / "external" / "telegram"
    renamed = skill_dir.with_name("telegram_fork")
    skill_dir.rename(renamed)
    # manifest.name stays "telegram": state under that name is the WRONG dir.
    _write_review_pass(data, "telegram", _real_content_hash(renamed))
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "incompatible" for f in plugin_findings)
    # Stored under the basename, the same PASS grandfathers again.
    _write_review_pass(data, "telegram_fork", _real_content_hash(renamed))
    result = _run(data, "--json", str(tmp_path / "report2.json"),
                  isolated_root=tmp_path / "isol")
    report = json.loads((tmp_path / "report2.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "note" for f in plugin_findings)


def test_pass_with_blockers_never_grandfathers_even_under_advisory_enforcement(tmp_path):
    """Adversarial fix-round 3, finding 1: the audit's grandfather predicate
    is the PluginAPI refusal path's own — only clean|warnings. Advisory
    enforcement makes a BLOCKERS verdict EXECUTABLE (skill_review_gate), but
    the runtime grandfather (plugin_api_admission_refusal_outcome) never
    grandfathers blockers, so the audit must report INCOMPATIBLE ("would be
    pending"), never the grandfather note."""
    data = _build_nminus1_install(tmp_path / "install")
    skill_dir = data / "skills" / "external" / "telegram"
    _write_review_pass(
        data, "telegram", _real_content_hash(skill_dir),
        status="blockers",
        findings=[{"item": "bug_hunting", "verdict": "FAIL",
                   "severity": "critical", "reason": "pinned blocker"}],
    )
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol",
                  env_extra={"OUROBOROS_REVIEW_ENFORCEMENT": "advisory"})
    assert result.returncode == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "incompatible" for f in plugin_findings)
    assert not any("GRANDFATHERED" in f["detail"] for f in plugin_findings)


def test_symlinked_skill_is_judged_by_its_resolved_target_identity(tmp_path):
    """Adversarial fix-round 3, finding 2: the runtime resolves the skill
    directory FIRST (skill_loader.load_skill) and derives state identity from
    the resolved basename — a symlinked skill must be judged by its TARGET's
    review state, never by the link's lexical name."""
    data = _build_nminus1_install(tmp_path / "install")
    target = tmp_path / "payloads" / "telegram_target"
    target.parent.mkdir(parents=True)
    (data / "skills" / "external" / "telegram").rename(target)
    link = data / "skills" / "external" / "telegram_link"
    link.symlink_to(target, target_is_directory=True)

    # A PASS stored under the LINK's lexical name is the WRONG state dir.
    _write_review_pass(data, "telegram_link", _real_content_hash(target))
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "incompatible" for f in plugin_findings)

    # Stored under the resolved TARGET basename, the same PASS grandfathers.
    _write_review_pass(data, "telegram_target", _real_content_hash(target))
    result = _run(data, "--json", str(tmp_path / "report2.json"),
                  isolated_root=tmp_path / "isol")
    report = json.loads((tmp_path / "report2.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "note" for f in plugin_findings)
    assert any("GRANDFATHERED" in f["detail"] for f in plugin_findings)


def test_identity_collision_is_a_blocking_finding_never_a_grandfather(tmp_path):
    """Adversarial fix-round 3, finding 2 (collision half): two directories
    sanitising to ONE runtime identity are refused by the runtime BEFORE any
    review-state read — the audit must report a blocking finding and must not
    bind either payload to the shared (ambiguous) review state, even when a
    matching PASS is stored under that identity."""
    data = _build_nminus1_install(tmp_path / "install")
    fixture = FIXTURES / "telegram_SKILL_v6.113.4.md"
    a = data / "skills" / "external" / "telegram fork"
    b = data / "skills" / "external" / "telegram+fork"
    for d in (a, b):
        d.mkdir(parents=True)
        shutil.copyfile(fixture, d / "SKILL.md")
    _write_review_pass(data, "telegram_fork", _real_content_hash(a))
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    collisions = [f for f in report["findings"]
                  if f["check_id"] == "unauditable-source"
                  and "telegram_fork" in f["subject"]]
    assert collisions and all(f["severity"] == "incompatible" for f in collisions)
    assert any("one runtime identity" in f["detail"] for f in collisions)
    # Neither colliding directory reached the plugin-api judgment.
    assert not any("fork" in f["subject"] for f in report["findings"]
                   if f["check_id"] == "plugin-api")


def test_native_seed_pass_without_seed_origin_is_not_grandfathered(tmp_path):
    """Adversarial fix-round 2, claim 2: the auditor judges admission through
    the runtime's own ``load_review_state`` — a ``native_seed`` PASS whose
    ``.seed-origin`` launcher-provenance marker is gone demotes to pending at
    runtime, so the audit must report an incompatibility, never the
    grandfather note. With the marker present the SAME state grandfathers."""
    data = _build_nminus1_install(tmp_path / "install")
    # native_seed provenance lives in the NATIVE bucket (only there is the
    # top-level .seed-origin marker hash-exempt payload).
    (data / "skills" / "native").mkdir()
    skill_dir = data / "skills" / "native" / "telegram"
    (data / "skills" / "external" / "telegram").rename(skill_dir)
    _write_review_pass(data, "telegram", _real_content_hash(skill_dir),
                       review_profile="native_seed")
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "incompatible" for f in plugin_findings)

    # The admission read is READ-ONLY: no state dir may be created as a side
    # effect of judging admission (skill_state_dir_path, never mkdir).
    assert not (data / "state" / "skills" / "telegram" / "extension_calls").exists()
    before = _tree_snapshot(data)

    # Contrast: with launcher provenance intact (.seed-origin is hash-exempt)
    # the runtime serves the PASS and the audit grandfathers it again.
    (skill_dir / ".seed-origin").write_text("native\n", encoding="utf-8")
    result = _run(data, "--json", str(tmp_path / "report2.json"),
                  isolated_root=tmp_path / "isol")
    report = json.loads((tmp_path / "report2.json").read_text(encoding="utf-8"))
    plugin_findings = [f for f in report["findings"] if f["check_id"] == "plugin-api"]
    assert plugin_findings
    assert all(f["severity"] == "note" for f in plugin_findings)
    assert any("GRANDFATHERED" in f["detail"] for f in plugin_findings)
    after = {k: v for k, v in _tree_snapshot(data).items() if ".seed-origin" not in k}
    assert after == before  # the two audits wrote nothing into the install


# ----------------------------------------------------------- clean install


def test_clean_70_install_exits_0(tmp_path):
    data = _build_clean_70_install(tmp_path / "install")
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["incompatible"] == 0
    # The attestation list prints even on a clean install (F13).
    assert "OWNER ATTESTATION" in result.stdout


# ------------------------------------------------------- unreadable install


def test_broken_settings_document_exits_2(tmp_path):
    data = tmp_path / "install" / "data"
    data.mkdir(parents=True)
    (data / "settings.json").write_text("{not json", encoding="utf-8")
    result = _run(data, isolated_root=tmp_path / "isol")
    assert result.returncode == 2
    assert "INSTALL UNREADABLE" in result.stderr


def test_missing_data_root_exits_2(tmp_path):
    result = _run(tmp_path / "nope" / "data", isolated_root=tmp_path / "isol")
    assert result.returncode == 2


def test_data_root_without_settings_exits_2(tmp_path):
    data = tmp_path / "install" / "data"
    data.mkdir(parents=True)
    result = _run(data, isolated_root=tmp_path / "isol")
    assert result.returncode == 2


# ------------------------------------------------- exit contract (finding 3)


def test_malformed_manifest_is_a_blocking_unauditable_source_never_exit_0(tmp_path):
    """A mandatory source the audit cannot parse must never yield exit 0: the
    otherwise-clean install carries one malformed skill manifest and the audit
    reports it as a blocking ``unauditable-source`` finding (exit 1)."""
    data = _build_clean_70_install(tmp_path / "install")
    broken = data / "skills" / "external" / "broken_skill"
    broken.mkdir(parents=True)
    (broken / "skill.json").write_text("{not a manifest", encoding="utf-8")
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    findings = [f for f in report["findings"] if f["check_id"] == "unauditable-source"]
    assert findings and all(f["severity"] == "incompatible" for f in findings)


def test_unreadable_skills_tree_is_an_audit_failure_exit_2_not_clean(tmp_path):
    """Adversarial fix-round 2, claim 3a: the audit's traversal must not stand
    on the runtime's fail-soft listdir — an unreadable skills directory is an
    audit failure (exit 2), never an empty walk that audits clean."""
    import pytest
    if os.name == "nt":
        pytest.skip("chmod-based unreadable probes are POSIX-only")
    if os.geteuid() == 0:
        pytest.skip("permission probes are meaningless as root")
    data = _build_clean_70_install(tmp_path / "install")
    locked = data / "skills" / "external"
    locked.chmod(0)
    try:
        result = _run(data, isolated_root=tmp_path / "isol")
    finally:
        locked.chmod(0o755)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "audit traversal failed" in result.stderr


def test_unreadable_task_results_dir_is_an_audit_failure_exit_2_not_clean(tmp_path):
    """Adversarial fix-round 3, finding 3: task_results is a mandatory audit
    source of the same class as the skills tree — Path.glob would suppress a
    PermissionError on supported Python 3.10 and the unreadable directory
    would audit clean; the strict lister maps it to exit 2."""
    import pytest
    if os.name == "nt":
        pytest.skip("chmod-based unreadable probes are POSIX-only")
    if os.geteuid() == 0:
        pytest.skip("permission probes are meaningless as root")
    data = _build_clean_70_install(tmp_path / "install")
    locked = data / "task_results"
    locked.chmod(0)
    try:
        result = _run(data, isolated_root=tmp_path / "isol")
    finally:
        locked.chmod(0o755)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "audit traversal failed" in result.stderr


def test_symlink_loop_at_task_results_is_an_audit_failure_exit_2(tmp_path):
    """Adversarial fix-round 4, finding 2: ``Path.is_dir()`` folds ELOOP into
    plain False — a symlink loop standing where the mandatory ``task_results``
    source lives would silently skip the whole history scan. The strict
    ``os.stat`` pre-check raises instead (exit 2)."""
    data = _build_clean_70_install(tmp_path / "install")
    shutil.rmtree(data / "task_results")
    os.symlink("task_results", data / "task_results")
    result = _run(data, isolated_root=tmp_path / "isol")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "audit traversal failed" in result.stderr


def test_dangling_ui_preferences_symlink_is_an_audit_failure_exit_2(tmp_path):
    """Adversarial fix-round 4, finding 2: a DANGLING symlink is not absence —
    ``Path.is_file()`` reads it as False and the source would silently audit
    clean; only a truly absent path is a legitimate skip."""
    data = _build_clean_70_install(tmp_path / "install")
    (data / "state").mkdir()
    os.symlink("nonexistent-target.json", data / "state" / "ui_preferences.json")
    result = _run(data, isolated_root=tmp_path / "isol")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "dangling symlink" in result.stderr


def test_truly_absent_optional_sources_still_audit_clean(tmp_path):
    """Fix-round 4 contrast pin: genuine absence of ``task_results`` and
    ``state/ui_preferences.json`` stays a legitimate skip — the strict stat
    probe must not turn a minimal install into an audit failure."""
    data = _build_clean_70_install(tmp_path / "install")
    shutil.rmtree(data / "task_results")
    result = _run(data, isolated_root=tmp_path / "isol")
    assert result.returncode == 0, result.stdout + result.stderr


def test_symlink_loop_in_a_declared_entry_is_a_blocking_finding_not_a_crash(tmp_path):
    """Adversarial fix-round 4, finding 3: compute_content_hash resolves the
    manifest-DECLARED entry unguarded (skill_loader._add_if_confined) — a
    symlink loop there raises RuntimeError on supported 3.10, which passed the
    OSError-only handlers and crashed as Python's bare exit 1 (no report).
    Now: blocking unauditable-source finding, report still written."""
    data = _build_nminus1_install(tmp_path / "install")
    skill_dir = data / "skills" / "external" / "telegram"
    _write_review_pass(data, "telegram", _real_content_hash(skill_dir))
    # The fixture manifest declares `entry: plugin.py`; make that entry a
    # self-referencing symlink so its resolve() hits the 3.10 loop detector.
    os.symlink("plugin.py", skill_dir / "plugin.py")
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    unauditable = [f for f in report["findings"]
                   if f["check_id"] == "unauditable-source"]
    assert any("telegram" in f["subject"] and "hash failed" in f["detail"]
               for f in unauditable)
    assert not any("GRANDFATHERED" in f["detail"] for f in report["findings"])


def test_runtime_error_from_the_audit_walk_maps_to_exit_2(tmp_path, monkeypatch, capsys):
    """Adversarial fix-round 4, finding 3 (class backstop): the top-level
    handler treats RuntimeError like OSError — a 3.10 pathlib symlink loop
    raised from ANY path the audit (or a runtime classifier it consumes)
    resolves is exit 2, never Python's bare exit 1. The N−1 install is the
    one whose legacy telegram manifest actually reaches the review-state
    read."""
    module = _load_module()
    data = _build_nminus1_install(tmp_path / "install")

    def _loop_boom(*_a, **_k):
        raise RuntimeError("Symlink loop from 'state/skills/telegram'")

    monkeypatch.setattr(module, "load_review_state", _loop_boom)
    rc = module.main([str(data)])
    assert rc == 2
    assert "audit traversal failed" in capsys.readouterr().err


def test_data_root_resolve_symlink_loop_exits_2(tmp_path, monkeypatch, capsys):
    """Adversarial fix-round 3, finding 4: a symlink loop under the data-root
    argument raises RuntimeError from the 3.10 pathlib resolver — that must
    map to exit 2 (INSTALL UNREADABLE), never Python's bare exit 1."""
    module = _load_module()
    real_resolve = pathlib.Path.resolve

    def _resolve(self, *args, **kwargs):
        if self.name == "loop-data":
            raise RuntimeError("Symlink loop from 'loop-data'")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", _resolve)
    rc = module.main([str(tmp_path / "loop-data")])
    assert rc == 2
    assert "data root does not resolve" in capsys.readouterr().err


def test_report_path_resolve_symlink_loop_exits_2(tmp_path, monkeypatch, capsys):
    """Adversarial fix-round 3, finding 4: the report-path resolve must catch
    RuntimeError (3.10 pathlib symlink-loop detector) exactly like OSError —
    exit 2 (REPORT UNWRITABLE), never a bare Python exit 1."""
    module = _load_module()
    data = _build_clean_70_install(tmp_path / "install")
    real_resolve = pathlib.Path.resolve

    def _resolve(self, *args, **kwargs):
        if self.name == "loop-report.json":
            raise RuntimeError("Symlink loop from 'loop-report.json'")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", _resolve)
    rc = module.main([str(data), "--json", str(tmp_path / "loop-report.json")])
    assert rc == 2
    assert "REPORT UNWRITABLE" in capsys.readouterr().err


def test_report_path_resolve_failure_exits_2_not_python_exit_1(tmp_path, monkeypatch, capsys):
    """Adversarial fix-round 2, claim 3b: an OSError from resolving the
    report path itself maps to exit 2 (REPORT UNWRITABLE), never Python's
    bare exit 1 that automation reads as \"incompatibilities found\"."""
    module = _load_module()
    data = _build_clean_70_install(tmp_path / "install")
    real_resolve = pathlib.Path.resolve

    def _resolve(self, *args, **kwargs):
        if self.name == "boom-report.json":
            raise OSError("resolve exploded")
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "resolve", _resolve)
    rc = module.main([str(data), "--json", str(tmp_path / "boom-report.json")])
    assert rc == 2
    assert "REPORT UNWRITABLE" in capsys.readouterr().err


def test_report_write_failure_exits_2_not_1(tmp_path):
    """A report-write OSError is an audit failure (exit 2) — a bare Python
    exit 1 would read as "incompatibilities found" to automation."""
    data = _build_clean_70_install(tmp_path / "install")
    result = _run(data, "--json", str(tmp_path / "no" / "such" / "dir" / "r.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 2
    assert "REPORT UNWRITABLE" in result.stderr


# ------------------------------------------- runtime-parity discovery (finding 5)


def test_orphan_and_hidden_skill_dirs_are_ignored_like_the_runtime(tmp_path):
    """Crash leftovers (.replaced-/.staging-/.tmp-) and hidden directories are
    never loaded by the runtime, so they must not become audit findings."""
    data = _build_clean_70_install(tmp_path / "install")
    for name in ("telegram.replaced-20260901", "x.staging-1", "y.tmp-2", ".hidden"):
        orphan = data / "skills" / "external" / name
        orphan.mkdir(parents=True)
        (orphan / "skill.json").write_text("{not a manifest", encoding="utf-8")
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["incompatible"] == 0


# ------------------------------------------------ bytecode plane (finding 4)


def test_pycache_prefix_inside_the_audited_root_is_refused_loudly(tmp_path):
    """Interpreter startup writes stdlib bytecode under PYTHONPYCACHEPREFIX
    BEFORE the script's first line, so a prefix inside the audited install
    cannot be silently tolerated: the audit refuses (exit 2) instead of
    printing a false read-only report over the already-touched tree."""
    data = _build_nminus1_install(tmp_path / "install")
    env_extra = {"PYTHONPYCACHEPREFIX": str(data / "state" / "pycache")}
    env = dict(os.environ)
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    env.update(env_extra)
    isolated = tmp_path / "isol"
    env.update({
        "OUROBOROS_APP_ROOT": str(isolated),
        "OUROBOROS_REPO_DIR": str(isolated / "repo"),
        "OUROBOROS_DATA_DIR": str(isolated / "data"),
        "OUROBOROS_SETTINGS_PATH": str(isolated / "data" / "settings.json"),
    })
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(data)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 2
    assert "READ-ONLY GUARANTEE VIOLATED" in result.stderr


def test_pycache_prefix_with_dont_write_bytecode_leaves_the_tree_untouched(tmp_path):
    """The launcher mode that IS safe: PYTHONDONTWRITEBYTECODE suppresses the
    startup writes and the script's own flag covers everything after — the
    audit runs and the audited tree stays byte-for-byte identical."""
    data = _build_nminus1_install(tmp_path / "install")
    before = _tree_snapshot(data)
    env = dict(os.environ)
    isolated = tmp_path / "isol"
    env.update({
        "PYTHONPYCACHEPREFIX": str(data / "state" / "pycache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "OUROBOROS_APP_ROOT": str(isolated),
        "OUROBOROS_REPO_DIR": str(isolated / "repo"),
        "OUROBOROS_DATA_DIR": str(isolated / "data"),
        "OUROBOROS_SETTINGS_PATH": str(isolated / "data" / "settings.json"),
    })
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(data)],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert not (data / "state" / "pycache").exists()
    assert _tree_snapshot(data) == before


# --------------------------------------------- source provenance (finding 6)


def test_tree_sha_appends_dirty_suffix_for_tracked_changes(monkeypatch):
    module = _load_module()

    class _Out:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    state = {"dirty": True}

    def _fake_run(cmd, **kwargs):
        if "rev-parse" in cmd:
            return _Out("f" * 40 + "\n")
        assert "--untracked-files=no" in cmd  # tracked-scope dirty check
        return _Out(" M ouroboros/config.py\n" if state["dirty"] else "")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    assert module._tree_sha() == "f" * 40 + "-dirty"
    state["dirty"] = False
    assert module._tree_sha() == "f" * 40


def test_tree_sha_is_fail_closed_when_the_dirty_check_fails(monkeypatch):
    """Adversarial fix-round 2, claim 4: rev-parse OK but git status failing
    (or erroring) must never yield a bare SHA that reads as proven-clean —
    the suffix names the unproven state."""
    module = _load_module()

    class _Out:
        def __init__(self, stdout, returncode=0):
            self.returncode = returncode
            self.stdout = stdout

    mode = {"kind": "rc"}

    def _fake_run(cmd, **kwargs):
        if "rev-parse" in cmd:
            return _Out("f" * 40 + "\n")
        if mode["kind"] == "rc":
            return _Out("", returncode=128)
        raise OSError("status exploded")

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    assert module._tree_sha() == "f" * 40 + "-unknown-dirty-state"
    mode["kind"] = "raise"
    assert module._tree_sha() == "f" * 40 + "-unknown-dirty-state"


def test_repo_root_wins_the_import_resolution():
    module = _load_module()
    assert sys.path[0] == str(module.REPO_ROOT)


# --------------------------------------------------------------- read-only


def test_audit_is_byte_for_byte_read_only_over_the_install(tmp_path):
    data = _build_nminus1_install(tmp_path / "install")
    before = _tree_snapshot(data)
    result = _run(data, isolated_root=tmp_path / "isol")
    assert result.returncode == 1
    after = _tree_snapshot(data)
    assert before == after  # same files, same bytes — nothing created or touched


def test_report_file_is_refused_inside_the_audited_root(tmp_path):
    data = _build_nminus1_install(tmp_path / "install")
    result = _run(data, "--json", str(data / "rc_report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 2
    assert not (data / "rc_report.json").exists()


# ------------------------------------------------------------ scope schema


def test_scope_document_matches_the_design_note_schema():
    module = _load_module()
    scope = module.build_scope()
    assert scope["abi"] == "7.0"
    assert set(scope["sources"]) == {"tree", "inventories_frozen_at"}
    assert scope["sources"]["inventories_frozen_at"] == module.INVENTORIES_FROZEN_AT
    ids = {c["id"] for c in scope["checks"]}
    assert ids == {"gateway-alias", "retired-setting", "comma-list",
                   "plugin-api", "schema-stamp"}
    aliases = [c for c in scope["checks"] if c["id"] == "gateway-alias"]
    assert {c["removed"] for c in aliases} == {
        "cost_usd", "cost_usd_with_children", "telegram_chat_id",
        "project_last_viewed", "project_hidden",
    }  # the five frozen ABI-3 aliases (F11 inventory)
    schema_checks = [c for c in scope["checks"] if c["id"] == "schema-stamp"]
    assert len(schema_checks) == 1
    assert "Q8=B" in schema_checks[0]["consequence"]


def test_comma_list_class_is_snapped_from_settings_defaults_not_hardcoded():
    from ouroboros.settings_defaults import (
        RETIRED_COMMA_LIST_SETTING_KEYS,
        RETIRED_SETTING_KEYS,
    )

    assert set(RETIRED_COMMA_LIST_SETTING_KEYS) <= set(RETIRED_SETTING_KEYS)
    module = _load_module()
    scope = module.build_scope()
    comma_keys = {c["key"] for c in scope["checks"] if c["id"] == "comma-list"}
    assert comma_keys == set(RETIRED_COMMA_LIST_SETTING_KEYS)
    retired_keys = {c["key"] for c in scope["checks"] if c["id"] == "retired-setting"}
    assert retired_keys == set(RETIRED_SETTING_KEYS) - set(RETIRED_COMMA_LIST_SETTING_KEYS)


def test_scope_only_flag_prints_the_scope_and_exits_0(tmp_path):
    env = dict(os.environ)
    isolated = tmp_path / "isol"
    env.update({
        "OUROBOROS_APP_ROOT": str(isolated),
        "OUROBOROS_REPO_DIR": str(isolated / "repo"),
        "OUROBOROS_DATA_DIR": str(isolated / "data"),
        "OUROBOROS_SETTINGS_PATH": str(isolated / "data" / "settings.json"),
    })
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "irrelevant"), "--scope-only"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert result.returncode == 0
    scope = json.loads(result.stdout)
    assert scope["abi"] == "7.0"


# ----------------------------------------------------- fixture provenance


def test_nminus1_fixtures_are_the_real_previous_minor_byte_forms():
    """The catalog documents (not synthetic shapes): the settings document
    carries the N−1 defaults for the retired keys, the task result is
    unstamped with the stored cost alias, the manifest is an extension
    without the plugin_api field — and none carries a secret value."""
    settings = json.loads((FIXTURES / "settings_v6.113.4.json").read_text("utf-8"))
    for key in ("OUROBOROS_SCOPE_REVIEW_FLOOR", "OUROBOROS_REVIEW_MODELS",
                "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL"):
        assert key in settings
    secretish = [k for k in settings
                 if k.endswith(("_API_KEY", "_TOKEN", "_CREDENTIALS", "_PASSWORD"))
                 or k == "GITHUB_TOKEN"]
    assert secretish and all(not settings[k] for k in secretish)

    row = json.loads((FIXTURES / "task_result_v6.113.4.json").read_text("utf-8"))
    assert "_schema_version" not in row
    assert "cost_usd" in row

    manifest_text = (FIXTURES / "telegram_SKILL_v6.113.4.md").read_text("utf-8")
    from ouroboros.contracts.skill_manifest import parse_skill_manifest_text

    manifest = parse_skill_manifest_text(manifest_text)
    assert manifest.type == "extension"
    assert manifest.plugin_api is None


def test_unparseable_ui_preferences_is_a_blocking_unauditable_source(tmp_path):
    """External-audit correction lane (base 8827fd2c), item 3: a PRESENT but
    unparseable ``state/ui_preferences.json`` silently audited clean (bare
    ``except JSONDecodeError: return``) — the one surviving instance of the
    pattern the fix-round-1 contract retired: a malformed mandatory source is
    NEVER a clean exit 0. Now it is a blocking ``unauditable-source`` finding
    (exit 1), same class as an unparseable skill manifest."""
    data = _build_clean_70_install(tmp_path / "install")
    (data / "state").mkdir()
    (data / "state" / "ui_preferences.json").write_text(
        "{this is not json", encoding="utf-8")
    result = _run(data, "--json", str(tmp_path / "report.json"),
                  isolated_root=tmp_path / "isol")
    assert result.returncode == 1, result.stdout + result.stderr
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    unauditable = [f for f in report["findings"]
                   if f["check_id"] == "unauditable-source"]
    assert any("ui_preferences" in f["subject"] and "does not parse" in f["detail"]
               for f in unauditable), report["findings"]


def test_non_object_ui_preferences_still_audits_clean(tmp_path):
    """Contrast pin: a file that PARSES to a determinate non-object holds no
    stored keys — the legacy-key audit has a truthful clean answer (the
    runtime drops the value wholesale), so it must not become a false block."""
    data = _build_clean_70_install(tmp_path / "install")
    (data / "state").mkdir()
    (data / "state" / "ui_preferences.json").write_text("[1, 2]", encoding="utf-8")
    result = _run(data, isolated_root=tmp_path / "isol")
    assert result.returncode == 0, result.stdout + result.stderr


def test_retired_setting_migration_names_the_successor_when_the_table_has_one():
    """A retired key whose retirement table names a successor must not be told
    «no replacement knob»: the wall-clock pair points at the activity model (the
    same truth the first-boot notice states), the truly knob-less keys keep the
    knob-less text."""
    from ouroboros.settings_defaults import RETIRED_SETTING_SUCCESSORS

    rc_audit = _load_module()
    checks = {c["key"]: c for c in rc_audit.build_scope()["checks"] if c["id"] == "retired-setting"}
    for key, successors in RETIRED_SETTING_SUCCESSORS.items():
        assert "no replacement knob" not in checks[key]["migration"], key
        for successor in successors:
            assert successor in checks[key]["migration"], (key, successor)
    knobless = [k for k in checks if k not in RETIRED_SETTING_SUCCESSORS]
    assert knobless and all("no replacement knob" in checks[k]["migration"] for k in knobless)
