"""Release identity and the local acceptance facts beside the review verdict.

Pins: the release digest (payload + dependency declaration, per-install notes out, the
same value from a directory and from an index file list); the one-time conversion of
persisted records (only records that covered the unchanged bytes move; stale stays
stale; free replay still finds old history rows); the gate's acceptance bases (trusted
source, install-time owner skip, owner edit) and the strict review-source blocker rule;
the owner-edit seam; turn-on-when-ready.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import pathlib

import pytest

from ouroboros import skill_identity as si


def _skill(root: pathlib.Path, name: str = "probe", *, extra: str = "") -> pathlib.Path:
    skill = root / name
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\nversion: 1.0.0\ntype: instruction\n{extra}---\n\n# {name}\n",
        encoding="utf-8")
    (skill / "notes.md").write_text("body\n", encoding="utf-8")
    return skill


# --- the digest -----------------------------------------------------------------------


@pytest.fixture
def digest_memo(monkeypatch):
    """A memo and device verdicts of this test's own (both are process-wide)."""
    monkeypatch.setattr(si, "_DIGESTS", collections.OrderedDict())
    monkeypatch.setattr(si, "_LOCAL_DEVICES", {})


def _payload_opens(monkeypatch, skill: pathlib.Path) -> list:
    opened, real = [], pathlib.Path.open
    monkeypatch.setattr(pathlib.Path, "open", lambda self, *args, **kwargs: (
        opened.append(self.name) if self.parent == skill else None) or real(self, *args, **kwargs))
    return opened


@pytest.mark.skipif(os.name == "nt", reason="POSIX change time; Windows hashes every time")
def test_a_kept_digest_reads_no_payload_but_an_edit_that_restores_size_and_mtime_is_hashed_again(
        tmp_path, monkeypatch, digest_memo):
    """P3 at the payload: an unchanged file is not read again, and a same-size edit whose mtime is set back
    (``utime``, ``touch -r``) still moves the change time, so the next digest reads the new bytes."""
    from ouroboros.skill_loader import compute_content_hash

    monkeypatch.setattr(si, "_RACY_NS", 0)  # keep a just-written file at once (the clock-tick guard has its own test)
    skill = _skill(tmp_path / "external")
    if (skill / "notes.md").stat().st_ctime_ns % 1_000_000_000 == 0:
        pytest.skip("a whole-second filesystem clock: there only the clock-tick guard keeps the key exact")
    first = compute_content_hash(skill)
    opened = _payload_opens(monkeypatch, skill)
    assert compute_content_hash(skill) == first and "notes.md" not in opened
    info = (skill / "notes.md").stat()
    (skill / "notes.md").write_text("BODY\n", encoding="utf-8")  # the same size
    os.utime(skill / "notes.md", ns=(info.st_atime_ns, info.st_mtime_ns))
    assert (skill / "notes.md").stat().st_mtime_ns == info.st_mtime_ns
    opened.clear()  # the edit's own write
    assert compute_content_hash(skill) != first and opened.count("notes.md") == 1


def test_a_file_changed_within_a_clock_tick_of_its_hash_is_read_every_time(tmp_path, monkeypatch, digest_memo):
    """A coarse filesystem clock (one second on HFS+ and ext3) can give a later write the same change time:
    a file that changed within ``_RACY_NS`` of being hashed is never served from the memo; once its change
    time lies further back, the digest is kept and the file is not read again."""
    from ouroboros.skill_loader import compute_content_hash

    skill = _skill(tmp_path / "external")
    first = compute_content_hash(skill)
    opened = _payload_opens(monkeypatch, skill)
    assert compute_content_hash(skill) == first and "notes.md" in opened
    if os.name != "nt":
        monkeypatch.setattr(si, "_RACY_NS", 0)  # as if the change time were old enough
        compute_content_hash(skill)
        opened.clear()
        assert compute_content_hash(skill) == first and "notes.md" not in opened


@pytest.mark.skipif(os.name == "nt", reason="Windows hashes every time anyway")
def test_a_filesystem_the_store_does_not_call_local_is_hashed_every_time(tmp_path, monkeypatch, digest_memo):
    """Network, FUSE and unknown filesystems may keep change times dishonestly (a client's attribute cache,
    a userspace driver): there the memo is never consulted, however old the change time."""
    from ouroboros import store
    from ouroboros.skill_loader import compute_content_hash

    monkeypatch.setattr(si, "_RACY_NS", 0)
    monkeypatch.setattr(store, "_filesystem_type", lambda path: "nfs")
    skill = _skill(tmp_path / "external")
    first = compute_content_hash(skill)
    opened = _payload_opens(monkeypatch, skill)
    assert compute_content_hash(skill) == first and "notes.md" in opened


@pytest.mark.parametrize("note,bucket", sorted(si.PER_INSTALL_NOTE_BUCKETS.items()))
def test_per_install_notes_stay_out_only_in_their_own_bucket(tmp_path, note, bucket):
    """A note the host writes into its own bucket never enters the release; the same name
    anywhere else is skill content, so it cannot carry bytes past the review binding."""
    from ouroboros.skill_loader import compute_content_hash

    own = _skill(tmp_path / bucket)
    clean = compute_content_hash(own)
    (own / note).write_text(json.dumps({"installed_at": "t1"}), encoding="utf-8")
    first = compute_content_hash(own)
    (own / note).write_text(json.dumps({"installed_at": "t2"}), encoding="utf-8")
    assert compute_content_hash(own) == first == clean  # its own bucket, top level: outside
    nested = own / "data"
    nested.mkdir()
    (nested / note).write_text("runtime-readable bytes\n", encoding="utf-8")
    assert compute_content_hash(own) != clean  # a nested file of that name is payload
    other_bucket = "user" if bucket != "user" else "elsewhere"
    other = _skill(tmp_path / other_bucket)
    plain = compute_content_hash(other)
    (other / note).write_text("runtime-readable bytes\n", encoding="utf-8")
    assert compute_content_hash(other) != plain  # another place: the name is payload
    (other / "notes.md").write_text("changed\n", encoding="utf-8")
    assert compute_content_hash(other) != plain


def test_declaration_is_part_of_the_release(tmp_path):
    from ouroboros.skill_loader import compute_content_hash

    plain = compute_content_hash(_skill(tmp_path / "a"))
    declared = _skill(tmp_path / "ouroboroshub", extra="install_specs:\n  - kind: pip\n    package: requests\n")
    with_specs = compute_content_hash(declared)
    assert si.release_declaration(declared) == [{"kind": "pip", "package": "requests"}]
    # A legacy sidecar's recorded declaration is what the installer used: in its bucket it decides.
    sidecar = json.dumps({"install_specs": {"raw": [{"kind": "pip", "package": "httpx"}], "auto": []}})
    (declared / ".ouroboroshub.json").write_text(sidecar, encoding="utf-8")
    assert si.release_declaration(declared) == [{"kind": "pip", "package": "httpx"}]
    assert compute_content_hash(declared) not in {with_specs, plain}
    # Outside its bucket a file of that name is content and never replaces the manifest's declaration.
    elsewhere = _skill(tmp_path / "external", extra="install_specs:\n  - kind: pip\n    package: requests\n")
    (elsewhere / ".ouroboroshub.json").write_text(sidecar, encoding="utf-8")
    assert si.release_declaration(elsewhere) == [{"kind": "pip", "package": "requests"}]


def test_directory_and_index_file_list_agree(tmp_path):
    from ouroboros.skill_loader import compute_content_hash

    skill = _skill(tmp_path / "hub", extra="install_specs:\n  - kind: pip\n    package: requests\n")
    (skill / "lib").mkdir()
    (skill / "lib" / "a-b.py").write_text("x = 1\n", encoding="utf-8")
    (skill / "lib-z.txt").write_text("z\n", encoding="utf-8")
    files = [{"path": p.relative_to(skill).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(skill.rglob("*")) if p.is_file()]
    from_index = si.release_digest_from_files(files, [{"kind": "pip", "package": "requests"}])
    assert from_index == compute_content_hash(skill)
    assert si.is_release_digest(from_index)


def test_version_two_never_equals_version_one(tmp_path):
    from ouroboros.skill_loader import compute_content_hash, reduce_skill_content_hash

    skill = _skill(tmp_path / "native", "seeded")
    (skill / ".seed-origin").write_text("seeded_from=skills\n", encoding="utf-8")
    current = compute_content_hash(skill)
    legacy = si.legacy_content_hashes(skill)
    assert current not in legacy and len(legacy) == 2  # post-6.31 (marker exempt) and older
    pairs = [(p.relative_to(skill).as_posix(), hashlib.sha256(p.read_bytes()).digest())
             for p in sorted(skill.rglob("*")) if p.is_file() and p.name != ".seed-origin"]
    assert legacy[0] == reduce_skill_content_hash(pairs)


# --- the one-time conversion ------------------------------------------------------------


def _state(drive_root, name="probe"):
    path = drive_root / "state" / "skills" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_conversion_moves_only_records_that_covered_the_unchanged_bytes(tmp_path):
    from ouroboros.skill_loader import compute_content_hash, find_skill

    drive_root = tmp_path / "data"
    skill_dir = _skill(drive_root / "skills" / "external")
    (skill_dir / ".ouroboroshub.json").write_text(json.dumps({"installed_at": "then"}), encoding="utf-8")
    legacy = si.legacy_content_hashes(skill_dir)[0]
    digest = compute_content_hash(skill_dir)
    state = _state(drive_root)
    (state / "review.json").write_text(json.dumps({
        "content_hash": legacy, "status": "warnings", "reviewer_models": ["m"],
        "findings": [{"item": "bug_hunting", "verdict": "FAIL", "severity": "warning", "model": "m"}],
        "author_disposition": {"disposition": "accept", "rationale": "ok", "subject_hash": legacy,
                               "reviewer_signal": "warnings", "enforcement": "advisory", "source": "author",
                               "recorded_at": "2026-10-01T00:00:00Z", "action": "finish"}}), encoding="utf-8")
    (state / "grants.json").write_text(json.dumps({"content_hash": legacy, "granted_keys": [],
                                                   "requested_keys": []}), encoding="utf-8")
    (state / "owner_attestation.json").write_text(json.dumps({"content_hash": "f" * 64}), encoding="utf-8")
    receipt = {"published": {"content_hash": legacy, "slug": "probe"}}
    (state / "ouroboroshub.json").write_text(json.dumps(receipt), encoding="utf-8")

    summary = si.convert_skill_identities(drive_root)
    assert sorted(summary["converted"]["probe"]) == ["grants.json", "ouroboroshub.json", "review.json"]
    review = json.loads((state / "review.json").read_text(encoding="utf-8"))
    assert review["content_hash"] == digest and review["legacy_content_hashes"] == [legacy]
    assert review["author_disposition"]["subject_hash"] == digest
    assert review["findings"][0]["item"] == "bug_hunting"  # evidence itself is untouched
    assert json.loads((state / "grants.json").read_text(encoding="utf-8"))["content_hash"] == digest
    assert json.loads((state / "owner_attestation.json").read_text(encoding="utf-8"))["content_hash"] == "f" * 64
    receipt["published"]["content_hash"] = digest
    assert json.loads((state / "ouroboroshub.json").read_text(encoding="utf-8")) == receipt
    loaded = find_skill(drive_root, "probe")
    assert loaded.review.gate_for(loaded.content_hash)["executable_review"] is True

    # Idempotent: the marker ends the pass; a re-run of the record step is a no-op.
    assert si.convert_skill_identities(drive_root) == {"skipped": True}
    assert si.convert_skill_records(drive_root, "probe", [legacy], digest) == []


def test_conversion_keeps_stale_records_stale(tmp_path):
    from ouroboros.skill_loader import find_skill

    drive_root = tmp_path / "data"
    _skill(drive_root / "skills" / "external")
    state = _state(drive_root)
    (state / "review.json").write_text(json.dumps({"content_hash": "0" * 64, "status": "clean", "findings": [
        {"item": "x", "verdict": "PASS", "model": "m"}]}), encoding="utf-8")
    receipt = {"published": {"content_hash": "0" * 64, "slug": "probe"}}
    (state / "ouroboroshub.json").write_text(json.dumps(receipt), encoding="utf-8")
    si.convert_skill_identities(drive_root)
    assert json.loads((state / "ouroboroshub.json").read_text(encoding="utf-8")) == receipt
    assert json.loads((state / "review.json").read_text(encoding="utf-8"))["content_hash"] == "0" * 64
    loaded = find_skill(drive_root, "probe")
    assert loaded.review.gate_for(loaded.content_hash)["executable_review"] is False


def test_identity_conversion_retries_after_a_record_write_error(tmp_path, monkeypatch):
    drive = tmp_path / "data"
    skill = _skill(drive / "skills" / "external")
    state = _state(drive)
    (state / "review.json").write_text(json.dumps({"content_hash": si.legacy_content_hashes(skill)[0]}),
                                      encoding="utf-8")
    original = si.convert_skill_records
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise OSError("write interrupted")
        patch.setattr(si, "convert_skill_records", fail)
        assert si.convert_skill_identities(drive)["errors"]
    assert not (drive / "state" / "skills" / si.CONVERSION_MARKER).exists()
    assert si.convert_skill_records is original
    assert si.convert_skill_identities(drive)["converted"] == {"probe": ["review.json"]}
    assert si.convert_skill_identities(drive) == {"skipped": True}


@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("manifest_name", ["SKILL.md", "skill.json"])
def test_legacy_hub_conversion_uses_only_the_original_manifest_declaration(tmp_path, edited, manifest_name):
    from ouroboros.skill_install_record import convert_legacy_records, read_install_record
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_source_catalog import install_listing

    drive = tmp_path / "data"
    payload = _skill(drive / "skills" / "ouroboroshub", extra="dependencies:\n  pip: [requests]\n")
    if manifest_name == "skill.json":
        (payload / "SKILL.md").unlink()
        (payload / manifest_name).write_text(json.dumps({"name": "probe", "description": "d", "version": "1.0.0",
            "type": "instruction", "dependencies": {"pip": ["requests"]}}), encoding="utf-8")
    digest = si.payload_digest(payload)
    files = [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in payload.iterdir() if p.is_file()]
    sidecar = {"source": "ouroboroshub", "slug": "probe", "version": "1.0.0", "files": files,
               "catalog_url": "https://raw.githubusercontent.com/razzant/OuroborosHub/main/catalog.json"}
    (payload / ".ouroboroshub.json").write_text(json.dumps(sidecar), encoding="utf-8")
    if edited:
        manifest = payload / manifest_name
        manifest.write_text(manifest.read_text(encoding="utf-8").replace("requests", "httpx"), encoding="utf-8")
    row = {"id": "community", "kind": "hub", "name": "Community", "url": "https://github.com/razzant/OuroborosHub"}
    assert convert_legacy_records(drive, [row])["converted"] == ["probe"]
    record = read_install_record(drive, "probe")
    loaded = find_skill(drive, "probe")
    listing = install_listing(drive, loaded, {"community": (row, {"entries": [
        {"name": "probe", "version": "1.0.0", "digest": digest}]})})["install"]
    assert listing["modified"] is edited
    if not edited:
        assert record["release"]["digest"] == loaded.content_hash == digest
        assert listing["update_available"] is None
    else:
        assert record["release"]["digest"] == si.release_digest_from_files(files)


# --- the gate ----------------------------------------------------------------------------


def _gate(status="pending", *, enforcement="blocking", acceptance=None, findings=None, stale=True, digest="d" * 64):
    from ouroboros.skill_review_status import skill_review_gate

    return skill_review_gate(status, stale=stale, enforcement=enforcement, findings=findings or [],
                             current_hash=digest, acceptance=acceptance)


@pytest.fixture
def not_cyber(monkeypatch):
    import ouroboros.tools.review_helpers as helpers

    monkeypatch.setattr(helpers, "review_enforcement_blocks", lambda _mode: True)


def test_acceptance_bases_admit_only_their_exact_bytes(not_cyber):
    digest = "d" * 64
    for basis, reason in (("trusted", "trusted_source"), ("owner_skip", "owner_skip")):
        gate = _gate(acceptance={"basis": basis, "release_digest": digest})
        assert gate["executable_review"] is True and gate["blocking_reason"] == reason
        assert _gate(acceptance={"basis": basis, "release_digest": "e" * 64})["executable_review"] is False
    assert _gate(acceptance={"basis": "review", "release_digest": digest})["executable_review"] is False
    assert _gate(acceptance={"basis": "cyber", "release_digest": digest})["executable_review"] is False
    assert _gate(acceptance={"owner_edit_digest": digest})["blocking_reason"] == "owner_edit"
    assert _gate(acceptance={"owner_edit_digest": "e" * 64})["executable_review"] is False
    failed = [{"item": "skill_preflight", "verdict": "FAIL", "model": "deterministic_preflight"}]
    refused = _gate(acceptance={"basis": "trusted", "release_digest": digest}, findings=failed, stale=False)
    assert refused["executable_review"] is False  # a fresh preflight failure outranks the basis


def test_review_source_blockers_need_the_owner_even_under_advisory(not_cyber):
    blockers = dict(status="blockers", enforcement="advisory", stale=False)
    strict = _gate(acceptance={"basis": "review", "release_digest": "x", "strict_review": True}, **blockers)
    assert strict["executable_review"] is False and strict["blocking_reason"] == "blocker_findings_need_owner"
    for acceptance in (None, {"basis": "review", "strict_review": False}):
        assert _gate(acceptance=acceptance, **blockers)["executable_review"] is True
    assert _gate(acceptance={"strict_review": True}, status="warnings", enforcement="advisory",
                 stale=False)["executable_review"] is True


@pytest.mark.parametrize("status", ["blockers", "pending"])
def test_author_acceptance_is_not_a_basis_for_a_review_source(not_cyber, status):
    """Under Advisory the author's finish runs ordinary skills' current bytes; a skill from
    a source that reviews each skill needs a review without blockers or the owner's Skip."""
    from ouroboros.review_records import build_author_disposition
    from ouroboros.skill_review_status import executable_review_requirement, skill_review_gate

    digest = "d" * 64
    author = build_author_disposition(disposition="accepted", rationale="findings addressed",
                                      subject_hash=digest, enforcement="advisory", action="finish")
    if status == "pending":
        author["review_reference"] = {
            "surface": "skill", "basis": "unavailable", "job_id": "job-1", "content_hash": digest,
            "finished_at": "2026-10-17T00:00:00Z", "status": "failed", "review_status": "pending",
            "preflight": {"content_hash": digest, "status": "pass"}}
    kwargs = dict(stale=False, enforcement="advisory", findings=[], author_disposition=author, current_hash=digest)
    ordinary = skill_review_gate(status, **kwargs)
    assert ordinary["executable_review"] is True and ordinary["blocking_reason"] == "author_accepted_advisory"
    strict = skill_review_gate(status, acceptance={"basis": "review", "strict_review": True}, **kwargs)
    assert strict["executable_review"] is False and strict["author_accepted"] is False
    assert strict["blocking_reason"] == {"blockers": "blocker_findings_need_owner", "pending": "review_pending"}[status]
    assert "Skip review" in executable_review_requirement(strict, strict_review=True)


def test_install_record_feeds_the_loaded_gate(tmp_path, not_cyber):
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_loader import auto_grant_if_enabled, find_skill

    drive_root = tmp_path / "data"
    skill_dir = _skill(drive_root / "skills" / "external")
    loaded = find_skill(drive_root, "probe")
    write_install_record(drive_root, "probe", {"source": {"id": "ouroboros"}, "basis": "trusted",
                                               "release": {"digest": loaded.content_hash}})
    loaded = find_skill(drive_root, "probe")
    assert loaded.review.gate_for(loaded.content_hash)["blocking_reason"] == "trusted_source"
    assert auto_grant_if_enabled(drive_root, loaded).granted is False  # nothing requested
    (skill_dir / "notes.md").write_text("edited\n", encoding="utf-8")
    edited = find_skill(drive_root, "probe")
    assert edited.review.gate_for(edited.content_hash)["executable_review"] is False


def test_trusted_install_still_meets_the_deterministic_preflight(tmp_path, not_cyber, monkeypatch):
    import ouroboros.skill_review as sr
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_loader import find_skill

    drive_root = tmp_path / "data"
    _skill(drive_root / "skills" / "ouroboroshub")
    loaded = find_skill(drive_root, "probe")
    write_install_record(drive_root, "probe", {"source": {"id": "ouroboros"}, "basis": "trusted",
                                               "release": {"digest": loaded.content_hash}})
    assert find_skill(drive_root, "probe").source == "ouroboroshub"  # the record proves the install
    monkeypatch.setattr("ouroboros.tools.skill_preflight._handle_skill_preflight",
                        lambda *a, **k: json.dumps({"ok": True}))
    assert sr.run_acceptance_preflight(object(), drive_root, "probe")["ok"] is True
    monkeypatch.setattr("ouroboros.tools.skill_preflight._handle_skill_preflight",
                        lambda *a, **k: json.dumps({"ok": False, "error": "binary payload"}))
    assert sr.run_acceptance_preflight(object(), drive_root, "probe")["ok"] is False
    refused = find_skill(drive_root, "probe")
    gate = refused.review.gate_for(refused.content_hash)
    assert gate["executable_review"] is False and refused.review.status == "pending"


# --- the owner-edit seam -------------------------------------------------------------------


def _admitted(drive_root, *, owner_request):
    from ouroboros.contracts.task_constraint import normalize_task_constraint
    from ouroboros.skill_loader import compute_content_hash
    from ouroboros.skill_repair_admission import record_repair_admission

    skill_dir = _skill(drive_root / "skills" / "external")
    record_repair_admission(drive_root, "probe", task_id="t1", owner_request=owner_request,
                            base_content_hash=compute_content_hash(skill_dir))
    constraint = normalize_task_constraint({"mode": "skill_repair", "skill_name": "probe",
                                            "payload_root": "skills/external/probe"})
    return skill_dir, constraint


@pytest.mark.parametrize("owner_request", [True, False])
def test_admitted_owner_task_edits_keep_running(tmp_path, owner_request, not_cyber):
    from ouroboros.skill_acceptance import OWNER_EDIT_FILENAME
    from ouroboros.skill_loader import find_skill
    from ouroboros.skill_repair_admission import advance_repair_expected_hash

    drive_root = tmp_path / "data"
    skill_dir, constraint = _admitted(drive_root, owner_request=owner_request)
    (skill_dir / "notes.md").write_text("fixed on the owner's request\n", encoding="utf-8")
    advance_repair_expected_hash(drive_root, constraint, task_id="t1")
    loaded = find_skill(drive_root, "probe")
    gate = loaded.review.gate_for(loaded.content_hash)
    assert gate["executable_review"] is owner_request
    assert (drive_root / "state" / "skills" / "probe" / OWNER_EDIT_FILENAME).exists() is owner_request
    # A later edit from anywhere else leaves the skill unreviewed.
    (skill_dir / "notes.md").write_text("edited elsewhere\n", encoding="utf-8")
    later = find_skill(drive_root, "probe")
    assert later.review.gate_for(later.content_hash)["executable_review"] is False


@pytest.mark.parametrize("stamp,owner_request", [
    ({"source_ref": {"chat_id": 1, "client_message_id": "owner-1"}}, True),
    ({"origin_suppressed": True}, True),
    ({}, False),
    ({"source_ref": {}, "origin_suppressed": "yes"}, False),
])
def test_promotion_admits_owner_edits_only_under_the_owner_door_stamp(tmp_path, monkeypatch, stamp, owner_request):
    """The seam's input: a selected-skill task promoted from the owner's own message (the
    door's ref, or its suppressed-origin stamp) records its revisions as owner edits; a
    promotion without that stamp (a task, a schedule, a consciousness wake-up) does not."""
    import types

    import supervisor.workers as workers
    from ouroboros.skill_repair_admission import load_repair_admission

    _skill(tmp_path / "skills" / "external")
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued = []
    ctx = types.SimpleNamespace(enqueue_task=enqueued.append, persist_queue_snapshot=lambda **_kwargs: True,
                                load_state=lambda: {"owner_chat_id": 1})
    result = workers.promote_chat_to_task({
        "type": "promote_chat_to_task", "task_id": "repair01", "objective": "Fix probe as asked", "chat_id": 1,
        "task_constraint": {"mode": "skill_repair", "skill_name": "probe", "payload_root": "skills/external/probe"},
        **stamp,
    }, ctx)
    assert result["status"] == "scheduled" and len(enqueued) == 1
    assert load_repair_admission(tmp_path, "probe")["owner_request"] is owner_request


def test_owner_edit_rebinds_unchanged_grant_requests(tmp_path):
    from ouroboros.skill_acceptance import record_owner_edit
    from ouroboros.skill_loader import compute_content_hash, load_skill_grants, save_skill_grants

    drive_root = tmp_path / "data"
    skill_dir = _skill(drive_root / "skills" / "external")
    save_skill_grants(drive_root, "probe", [], content_hash=compute_content_hash(skill_dir), requested_keys=[],
                      granted_permissions=["inject_chat"], requested_permissions=["inject_chat"])
    (skill_dir / "notes.md").write_text("owner edit\n", encoding="utf-8")
    digest = record_owner_edit(drive_root, skill_dir, task_id="t1")
    # The manifest requests nothing, so the old request set no longer matches: no carry.
    assert load_skill_grants(drive_root, "probe")["content_hash"] != digest


def test_owner_state_protects_the_acceptance_facts(tmp_path, monkeypatch):
    """Generic agent writes cannot forge the gate's acceptance facts: filename, stem and
    every wrapper agree; a payload file of the same name stays ordinary payload."""
    from ouroboros import config as cfg
    from ouroboros.contracts.skill_payload_policy import (
        SKILL_OWNER_STATE_FILENAMES,
        SKILL_OWNER_STATE_STEMS,
        is_skill_owner_state_target,
    )
    from ouroboros.gateway import files as gateway_files
    from ouroboros.tools import core

    data_root = tmp_path / "data"
    monkeypatch.setattr(cfg, "DATA_DIR", data_root, raising=True)
    for stem in ("install", "owner_edit", "update_backup"):
        assert f"{stem}.json" in SKILL_OWNER_STATE_FILENAMES and stem in SKILL_OWNER_STATE_STEMS
        for parts in (("state", "skills"), ("State", "Skills")):
            target = data_root.joinpath(*parts) / "probe" / f"{stem}.json"
            assert is_skill_owner_state_target(target, data_root) is True
            assert core._is_skill_owner_state_target(target, data_root) is True
            assert gateway_files._is_skill_owner_state_target(target) is True
            assert gateway_files._is_owner_only_file(target) is True
        payload = data_root / "skills" / "external" / "probe" / f"{stem}.json"
        assert is_skill_owner_state_target(payload, data_root) is False
        assert gateway_files._is_skill_owner_state_target(payload) is False


# --- turn on when ready -------------------------------------------------------------------


def _trusted_install(drive_root, *, turn_on=True):
    from ouroboros.skill_install_record import write_install_record
    from ouroboros.skill_loader import find_skill

    _skill(drive_root / "skills" / "external")
    loaded = find_skill(drive_root, "probe")
    write_install_record(drive_root, "probe", {"source": {"id": "ouroboros"}, "basis": "trusted",
                                               "release": {"digest": loaded.content_hash},
                                               "turn_on_when_ready": turn_on})


def test_turn_on_when_ready_enables_an_accepted_install(tmp_path, not_cyber, monkeypatch):
    from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready
    from ouroboros.skill_loader import load_enabled

    drive_root = tmp_path / "data"
    _trusted_install(drive_root)
    result = reconcile_turn_on_when_ready(drive_root, "probe", repo_path="")
    assert result["action"] == "enabled" and load_enabled(drive_root, "probe") is True
    assert reconcile_turn_on_when_ready(drive_root, "probe", repo_path="")["action"] == "already_on"


def test_turn_on_when_ready_never_overrides_a_choice_or_a_missing_basis(tmp_path, not_cyber, monkeypatch):
    from ouroboros.skill_install_record import read_install_record, write_install_record
    from ouroboros.skill_lifecycle_actions import reconcile_turn_on_when_ready
    from ouroboros.skill_loader import load_enabled, save_enabled

    drive_root = tmp_path / "data"
    _trusted_install(drive_root, turn_on=False)
    assert reconcile_turn_on_when_ready(drive_root, "probe", repo_path="")["action"] == "not_requested"
    _trusted_install(drive_root)
    save_enabled(drive_root, "probe", False, actor="owner_ui")
    assert reconcile_turn_on_when_ready(drive_root, "probe", repo_path="")["action"] == "owner_choice"
    save_enabled(drive_root, "probe", False, actor="load_error_revert")
    assert reconcile_turn_on_when_ready(drive_root, "probe", repo_path="")["action"] == "enabled"
    record = read_install_record(drive_root, "probe")
    write_install_record(drive_root, "other", {**record, "basis": "review", "entry": "other"})
    _skill(drive_root / "skills" / "external", "other")
    result = reconcile_turn_on_when_ready(drive_root, "other", repo_path="")
    assert result["action"] == "not_ready" and result["missing"] == ["acceptance"]
    assert load_enabled(drive_root, "other") is False


# --- review outside the lane ------------------------------------------------------------------


def test_lent_lane_lets_other_jobs_run_and_is_taken_back(tmp_path):
    import asyncio
    import threading

    from ouroboros import skill_lifecycle_queue as q

    lent = threading.Event()
    other_ran = threading.Event()
    release = threading.Event()
    order = []

    def long_review():
        with q.lend_lifecycle_lane():
            lent.set()
            assert release.wait(5)
        order.append("persist")
        return "reviewed"

    async def main():
        async def review_runner():
            return await q.run_blocking_preserving_cancellation(long_review)

        async def quick_runner():
            other_ran.set()
            order.append("other")
            return "other"

        review = asyncio.create_task(q.run_lifecycle_job(kind="review", target="a", runner=review_runner,
                                                         options=q.LifecycleJobOptions(drive_root=tmp_path)))
        assert await asyncio.to_thread(lent.wait, 5)
        other = await asyncio.wait_for(q.run_lifecycle_job(kind="install", target="b", runner=quick_runner,
                                                           options=q.LifecycleJobOptions(drive_root=tmp_path)), 5)
        assert other == "other"
        release.set()
        assert await asyncio.wait_for(review, 5) == "reviewed"

    asyncio.run(main())
    assert order == ["other", "persist"]
    with q.lend_lifecycle_lane():  # outside a lane job it is a no-op
        pass


def test_lent_lane_is_retaken_within_the_lane_deadline(tmp_path, monkeypatch):
    """Another holder of the cross-process lock past the lane's deadline fails the lending
    job loudly; the lane is whole again afterwards."""
    import asyncio

    from ouroboros import skill_lifecycle_queue as q
    from ouroboros.platform_layer import file_lock_exclusive_nb

    monkeypatch.setattr(q, "_lifecycle_deadline_sec", lambda: 0.3)
    holder = {}

    def review():
        with q.lend_lifecycle_lane():
            holder["fh"] = (tmp_path / "state" / "skill_lifecycle.lock").open("a+")
            file_lock_exclusive_nb(holder["fh"].fileno())  # e.g. another process's job

    async def run(kind, func):
        async def runner():
            return await q.run_blocking_preserving_cancellation(func)

        return await asyncio.wait_for(q.run_lifecycle_job(kind=kind, target="a", runner=runner,
                                                          options=q.LifecycleJobOptions(drive_root=tmp_path)), 5)

    with pytest.raises(TimeoutError, match="held beyond the deadline"):
        asyncio.run(run("review", review))
    holder["fh"].close()
    assert asyncio.run(run("install", lambda: "next")) == "next"


@pytest.mark.parametrize("moved_on", [False, True])
def test_a_verdict_lands_only_while_its_bytes_are_installed(tmp_path, moved_on):
    """The panel ran with the lane lent out; back under the lane, a verdict persists only
    for the bytes still installed (an edit, update or uninstall meanwhile drops it)."""
    import types

    from ouroboros.skill_loader import find_skill, skill_state_dir_path
    from ouroboros.skill_review import SkillReviewOutcome, _persist_projection

    drive_root = tmp_path / "data"
    skill_dir = _skill(drive_root / "skills" / "external")
    skill = find_skill(drive_root, "probe")
    outcome = SkillReviewOutcome(skill_name="probe", status="clean", findings=[], reviewer_models=["m1", "m2"],
                                 content_hash=skill.content_hash)
    if moved_on:
        (skill_dir / "notes.md").write_text("changed during the panel\n", encoding="utf-8")
    _persist_projection(types.SimpleNamespace(), skill, drive_root, outcome)
    result = outcome
    persisted = (skill_state_dir_path(drive_root, "probe") / "review.json").is_file()
    assert persisted is not moved_on
    assert result.status == ("pending" if moved_on else "clean")
    assert ("payload changed while the review ran" in (result.error or "")) is moved_on
