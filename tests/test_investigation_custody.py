"""Evidence-only investigations: one no-apply reading of custody, no apply debt."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_registration_policy import (
    INVESTIGATION_AUTHORITY,
    candidate_apply_possible,
    independent_review_without_apply,
)
from ouroboros.subagents import delegated_run_shape


@pytest.mark.parametrize("facts,possible", [
    ({"authority_source": INVESTIGATION_AUTHORITY, "snapshot_id": "snap", "target_root": "/repo"}, False),
    ({"authority_source": "readonly"}, False),
    ({}, False),  # no source over no capture binding: nothing captured to apply
    ({"snapshot_id": "legacy-snap"}, True),  # a capture binding without a recorded source stays possible
    ({"resource_ref": {"workspace_kind": "directory", "strategy": "copy"}}, True),
    ({"authority_source": "acting_constraint", "snapshot_id": "snap"}, True),
    ({"authority_source": "external_workspace_root"}, True),
    ({"authority_source": "skill_payload"}, True),
    ({"authority_source": "something_new"}, True),  # unknown is conservative, never permission
])
def test_one_reading_for_custody_entries_and_raw_rows(facts, possible):
    entry = custody.RunCustody(run_id="r", task_id="t", route_id="x", **facts)
    assert candidate_apply_possible(entry) is possible
    assert candidate_apply_possible(dict(facts)) is possible


def test_independence_needs_a_review_source_and_positive_no_apply_evidence():
    review = {"source": "review_substrate:plan_review", "authority_source": INVESTIGATION_AUTHORITY}
    assert independent_review_without_apply(review) is True
    # A row without the authority fact needs its recorded readonly Ask shape (STARTED row or
    # stored request); absent shape facts are no proof that the run never wrote.
    legacy = {"source": "review_substrate:task_acceptance"}
    assert independent_review_without_apply({**legacy, "access": "readonly", "mode": "ask"}) is True
    assert independent_review_without_apply({**legacy, "request": {"access": "readonly", "mode": "ask"}}) is True
    assert independent_review_without_apply(legacy) is False
    assert independent_review_without_apply({**legacy, "access": "workspace_write", "mode": "agent"}) is False
    assert independent_review_without_apply({"source": "delegated_subagent",
                                              "authority_source": INVESTIGATION_AUTHORITY}) is False  # author research
    assert independent_review_without_apply({"source": "review_substrate", "snapshot_id": "s"}) is False
    assert independent_review_without_apply({"authority_source": INVESTIGATION_AUTHORITY}) is False  # unknown source


def test_an_investigation_owes_no_apply_debt(tmp_path):
    settled = dict(task_id="t", route_id="x", settled=True, snapshot_id="snap")
    state = {
        "inv": custody.RunCustody(run_id="inv", authority_source=INVESTIGATION_AUTHORITY, **settled),
        "act": custody.RunCustody(run_id="act", authority_source="acting_constraint", **settled),
    }
    assert [row.run_id for row in custody.undisposed_patches(tmp_path, state)] == ["act"]


def test_integration_refuses_an_investigation_before_every_branch():
    from ouroboros.tools.subagent_integration_delegated import _delegated_disposition_refusal

    entry = custody.RunCustody(run_id="inv", task_id="t", route_id="x", settled=True, snapshot_id="snap",
                               execution_root="/copy", authority_source=INVESTIGATION_AUTHORITY)
    refusal = _delegated_disposition_refusal(custody.OWNED, entry, "inv")
    assert "INTEGRATE_DELEGATED_INVESTIGATION_EVIDENCE" in refusal
    acting = custody.RunCustody(run_id="act", task_id="t", route_id="x", settled=True, snapshot_id="snap",
                                execution_root="/copy", authority_source="acting_constraint")
    assert _delegated_disposition_refusal(custody.OWNED, acting, "act") == ""


def test_a_run_without_write_authority_gets_investigation_authority(tmp_path):
    from ouroboros.tools.delegate_integration import _mutation_authority

    ctx = SimpleNamespace(repo_dir=str(tmp_path), workspace_root=None, workspace_mode="", task_constraint=None,
                          is_workspace_mode=lambda: False, active_repo_dir=lambda: tmp_path)
    record, refusal = _mutation_authority(ctx, delegated_run_shape(False, "full"))
    assert refusal is None and record["source"] == INVESTIGATION_AUTHORITY and record["capture_mode"] == "evidence"
    record, refusal = _mutation_authority(ctx, delegated_run_shape(False, "readonly"))
    assert refusal is None and (record["source"], record["capture_mode"]) == ("readonly", "none")


def test_the_investigation_instruction_rides_only_a_writable_investigation():
    from ouroboros.delegate_start_instructions import INVESTIGATION_INSTRUCTION, access_instruction

    assert INVESTIGATION_INSTRUCTION in access_instruction("full", investigation=True)
    assert INVESTIGATION_INSTRUCTION not in access_instruction("full")
    assert access_instruction("readonly", investigation=True) == access_instruction("readonly")
    assert access_instruction("unknown", investigation=True) == ""


def test_schedule_schema_distinguishes_native_authority_from_checking_access(tmp_path):
    from ouroboros.tools.control import get_tools
    from ouroboros.tools.delegate_integration import _mutation_authority

    schema = next(tool.schema for tool in get_tools() if tool.name == "schedule_subagent")
    props = schema["parameters"]["properties"]
    emitted = "\n".join([schema["description"], *(prop.get("description", "") for prop in props.values())])
    for condition in ("native child/nanny", "Select readonly for read-only session execution",
                      "no task or product-write authority", "evidence-only private checking copy",
                      "do not call integrate_delegated_patch"):
        assert condition in emitted
    ctx = SimpleNamespace(repo_dir=str(tmp_path), workspace_root=None, workspace_mode="",
                          task_constraint=None, is_workspace_mode=lambda: False,
                          active_repo_dir=lambda: tmp_path)
    for access in ("full", "workspace_write", "readonly"):
        authority, refusal = _mutation_authority(ctx, delegated_run_shape(False, access))
        assert refusal is None
        assert authority["source"] == ("readonly" if access == "readonly" else INVESTIGATION_AUTHORITY)


def test_an_investigation_continuation_is_told_nothing_was_applied():
    from ouroboros.delegate_continuation import continuation_prompt

    text = continuation_prompt({"continuation_of": "r1", "prior_patch_disposition": "evidence_retained",
                                "prior_terminal_state": "succeeded"}, "continue")
    assert "evidence-only investigation" in text and "none of its changes" in text


def _retention_world(monkeypatch, *, landed=True):
    """The one dispose writer's two effects, observed: the row and the snapshot release."""
    rows, released = [], []

    def record(drive, entry, *, disposition, **row):
        rows.append({"disposition": disposition, **row})
        if landed:
            entry.patch_disposed = disposition
        return landed

    monkeypatch.setattr(custody, "record_patch_disposed", record)
    monkeypatch.setattr("ouroboros.subagent_worktrees.remove_execution_snapshot",
                        lambda key: released.append(key) or True)
    return rows, released


def test_a_ready_investigation_capture_is_retained_then_released(tmp_path, monkeypatch):
    from ouroboros.headless_status import ARTIFACT_STATUS_READY_WITH_CHANGES
    from ouroboros.tools.subagent_integration_delegated import retain_investigation_evidence

    rows, released = _retention_world(monkeypatch)
    entry = custody.RunCustody(run_id="inv", task_id="t", route_id="x", settled=True, snapshot_id="snap",
                               execution_root="/copy", authority_source=INVESTIGATION_AUTHORITY)
    block = retain_investigation_evidence(tmp_path, entry, {"status": ARTIFACT_STATUS_READY_WITH_CHANGES,
                                                            "patch_artifact": "p"})
    assert block["disposition"] == custody.EVIDENCE_RETAINED and block["patch_artifact"] == "p"
    assert [row["disposition"] for row in rows] == [custody.EVIDENCE_RETAINED] and released == ["snap"]
    assert "engine_receipt" not in rows[0]  # a Git snapshot has no engine operation to receipt


def test_an_unwritten_retention_row_keeps_the_private_copy(tmp_path, monkeypatch):
    from ouroboros.headless_status import ARTIFACT_STATUS_READY_WITH_CHANGES
    from ouroboros.tools.subagent_integration_delegated import retain_investigation_evidence

    rows, released = _retention_world(monkeypatch, landed=False)
    entry = custody.RunCustody(run_id="inv", task_id="t", route_id="x", settled=True, snapshot_id="snap",
                               execution_root="/copy", authority_source=INVESTIGATION_AUTHORITY)
    block = retain_investigation_evidence(tmp_path, entry, {"status": ARTIFACT_STATUS_READY_WITH_CHANGES})
    assert len(rows) == 1 and released == [] and "disposition" not in block
    assert "private copy is kept" in block["note"]


@pytest.mark.parametrize("authority,status", [
    ("acting_constraint", "ready_with_changes"),  # a candidate patch: the live owner decides
    (INVESTIGATION_AUTHORITY, "failed"),          # not ready: the copy stays for inspection
])
def test_retention_leaves_candidates_and_unready_captures_alone(tmp_path, monkeypatch, authority, status):
    from ouroboros.tools.subagent_integration_delegated import retain_investigation_evidence

    rows, released = _retention_world(monkeypatch)
    entry = custody.RunCustody(run_id="r", task_id="t", route_id="x", settled=True, snapshot_id="snap",
                               execution_root="/copy", authority_source=authority)
    block = {"status": status}
    assert retain_investigation_evidence(tmp_path, entry, block) is block and rows == [] and released == []


@pytest.mark.parametrize("engine_status,retained", [("discarded", True), ("pending", False)])
def test_a_directory_investigation_discards_its_engine_copy_with_a_receipt(tmp_path, monkeypatch,
                                                                          engine_status, retained):
    from ouroboros.tools.subagent_integration_delegated import retain_investigation_evidence

    rows, released = _retention_world(monkeypatch)
    decided = []

    class Gateway:
        def decide_run(self, run_id, body, *, idempotency_key):
            decided.append((run_id, body, idempotency_key))
            return {"status": engine_status}

    entry = custody.RunCustody(run_id="inv", task_id="t", route_id="x", settled=True,
                               resource_ref={"workspace_kind": "directory", "strategy": "copy"},
                               authority_source=INVESTIGATION_AUTHORITY)
    block = retain_investigation_evidence(tmp_path, entry, {"status": "ready", "capture_kind": "engine_directory"},
                                          Gateway())
    assert [(run_id, body) for run_id, body, _key in decided] == [("inv", {"action": "discard"})]
    assert released == []  # no Git snapshot to release
    if retained:
        assert rows[0]["engine_receipt"] == {"status": "discarded"}
        assert block["disposition"] == custody.EVIDENCE_RETAINED
    else:
        assert rows == [] and "unconfirmed" in block["note"] and "disposition" not in block


def _real_investigation(tmp_path, monkeypatch):
    """A settled investigation in a real private copy (a linked worktree) whose checks left named
    evidence the patch never carries (``.review-evidence/`` stays out of the result commit) beside
    an ordinary edit."""
    from pathlib import Path

    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from tests.test_delegated_run_isolation import _nanny_ctx, _seed_target

    target = _seed_target(tmp_path)
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(target_root=target, task_id="t-nanny", snapshot_id="inv-snap")
    copy = Path(handle.path)
    (copy / ".review-evidence").mkdir()
    (copy / ".review-evidence" / "coverage.xml").write_bytes(b"<coverage line-rate='0.91'/>\n" * 64)
    (copy / ".review-evidence" / "notes.txt").write_text("reproduced with the fixture\n")
    (copy / "tracked.txt").write_text("checked\n")
    entry = custody.RunCustody(
        run_id="inv-1", task_id="t-nanny", route_id="r", snapshot_id=handle.snapshot_id,
        execution_root=handle.path, baseline_sha=handle.baseline_sha, target_root=str(target),
        authority_source=INVESTIGATION_AUTHORITY, settled=True, ledger_root=str(custody.custody_root(ctx)))
    custody._CUSTODY[entry.run_id] = entry
    return target, custody.custody_root(ctx), entry, copy, handle


def _retained_files(block):
    from pathlib import Path

    root = Path(block["named_evidence"]["evidence_dir"]) / "files"
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob("*")) if path.is_file()}


def test_a_real_capture_retains_named_evidence_the_patch_excludes_then_releases(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.workspace_patch_rules import NAMED_EVIDENCE_EXCLUSION

    target, drive, entry, copy, handle = _real_investigation(tmp_path, monkeypatch)
    expected = {name: (copy / name).read_bytes() for name in (".review-evidence/coverage.xml",
                                                               ".review-evidence/notes.txt")}
    try:
        block = capture_terminal_patch_for_drive(drive, entry)
        manifest = json.loads(Path(block["manifest_artifact"]).read_text())
        assert {row["path"]: row["reason"] for row in manifest["untracked_excluded"]} == dict.fromkeys(
            (".review-evidence/coverage.xml", ".review-evidence/notes.txt"), NAMED_EVIDENCE_EXCLUSION)
        patch = Path(block["patch_artifact"]).read_text()
        assert "tracked.txt" in patch and ".review-evidence" not in patch
        assert block["disposition"] == custody.EVIDENCE_RETAINED == entry.patch_disposed
        assert block["named_evidence"]["status"] == "retained" and _retained_files(block) == expected
        assert not copy.exists() and find_execution_snapshot(handle.snapshot_id) is None
        assert (target / "tracked.txt").read_text() == "one\ntwo\n"  # nothing applied
        again = capture_terminal_patch_for_drive(drive, entry)  # a reread keeps the presentation
        assert again["disposition"] == custody.EVIDENCE_RETAINED and _retained_files(again) == expected
    finally:
        custody._CUSTODY.clear()


def test_a_named_link_is_a_recorded_gap_and_the_copy_is_still_released(tmp_path, monkeypatch):
    """Evidence is retained whole (no size bound); a link is recorded with its reason, never
    followed, and no longer keeps the copy: nothing more could ever be retained from it."""
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    target, drive, entry, copy, _handle = _real_investigation(tmp_path, monkeypatch)
    (copy / ".review-evidence" / "outside").symlink_to(target / "tracked.txt")
    expected = {name: (copy / name).read_bytes() for name in (".review-evidence/coverage.xml",
                                                               ".review-evidence/notes.txt")}
    try:
        block = capture_terminal_patch_for_drive(drive, entry)
        named = block["named_evidence"]
        assert named["status"] == "retained" and _retained_files(block) == expected
        assert {"path": ".review-evidence/outside", "reason": "evidence_symlink"} in named["gaps"]
        assert block["disposition"] == custody.EVIDENCE_RETAINED and not copy.exists()
        assert (target / "tracked.txt").read_text() == "one\ntwo\n"  # the link's target was never read or moved
    finally:
        custody._CUSTODY.clear()


def test_a_named_output_over_the_old_bound_is_retained_whole_and_the_copy_is_released(tmp_path, monkeypatch):
    """Investigations are no longer bounded at 64 MiB: an output past that old bound is retained
    whole, byte-exact and streamed, and the copy is released at once (nothing stays only there)."""
    import hashlib
    import tracemalloc
    from pathlib import Path

    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    target, drive, entry, copy, _handle = _real_investigation(tmp_path, monkeypatch)
    big = copy / ".review-evidence" / "run" / "results.bin"
    big.parent.mkdir(parents=True)
    size = 65 * 1024 * 1024  # past the retired 64 MiB investigation bound
    with big.open("wb") as stream:
        stream.truncate(size)  # sparse on the source side; the retained copy is whole
    digest = hashlib.sha256()
    with big.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    try:
        tracemalloc.start()
        try:
            block = capture_terminal_patch_for_drive(drive, entry)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        named = block["named_evidence"]
        assert named["status"] == "retained" and named["gaps"] == [], named
        assert {row["path"]: (row["size"], row["sha256"]) for row in named["files"]}[
            ".review-evidence/run/results.bin"] == (size, digest.hexdigest())
        assert not (named.get("inputs") or {}).get("limit_bytes")  # no bound is part of the receipt
        assert peak < 16 * 1024 * 1024, peak  # streamed, never held in memory whole
        kept = Path(named["evidence_dir"]) / "files" / ".review-evidence" / "run" / "results.bin"
        assert kept.stat().st_size == size
        assert block["disposition"] == custody.EVIDENCE_RETAINED and not copy.exists()
    finally:
        custody._CUSTODY.clear()


def test_a_capture_interrupted_before_retention_is_finished_by_the_next_capture(tmp_path, monkeypatch):
    from ouroboros.tools import subagent_integration_delegated
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    target, drive, entry, copy, _handle = _real_investigation(tmp_path, monkeypatch)
    retain = subagent_integration_delegated.retain_investigation_evidence
    try:
        # A crash after PATCH_CAPTURED landed and before the evidence was retained.
        monkeypatch.setattr(subagent_integration_delegated, "retain_investigation_evidence",
                            lambda drive, entry, block, gateway=None, **kwargs: block)
        capture_terminal_patch_for_drive(drive, entry)
        assert entry.patch_captured and not entry.patch_disposed and copy.is_dir()
        monkeypatch.setattr(subagent_integration_delegated, "retain_investigation_evidence", retain)
        block = capture_terminal_patch_for_drive(drive, entry)
        assert block["disposition"] == custody.EVIDENCE_RETAINED and not copy.exists()
        assert ".review-evidence/coverage.xml" in _retained_files(block)
    finally:
        custody._CUSTODY.clear()


def _fresh_start_consumer(monkeypatch, drive, target, task_id, access="readonly"):
    """The registered delegate_start entry with only its settings/transport faked."""
    import json

    from ouroboros import claudexor_daemon, config
    from ouroboros.tools import delegate
    from ouroboros.tools.registry import ToolContext
    from tests._delegated_transport_shared import _LiveRunStub

    class Gateway(_LiveRunStub):
        engine_version = config.CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION

        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly", "workspace_write", "full"]}]}

        def ensure_full_access(self, root):
            pass

        def start_run(self, request, *, idempotency_key=""):
            starts.append((request, idempotency_key))
            return {"runId": "fresh-run"}

    starts = []
    row = {"subagent_id": "investigation-fixture", "name": "Fixture",
           "recommended_use": "Regression consumer", "access": access, "effort": "low",
           "route": {"kind": "agent_session", "target_id": "some-route=weak-model"}}
    monkeypatch.setattr(config, "runtime_settings", lambda: {
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [row]})})
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: Gateway())
    ctx = ToolContext(repo_dir=target, drive_root=drive)
    ctx.task_id = task_id
    handler = next(tool.handler for tool in delegate.get_tools() if tool.name == "delegate_start")
    return ctx, handler, starts


@pytest.mark.parametrize("access", ["full", "workspace_write", "readonly"])
def test_fresh_public_start_after_an_unretained_investigation_capture_keeps_evidence(tmp_path, monkeypatch, access):
    import json

    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools import subagent_integration_delegated
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    target, drive, entry, copy, handle = _real_investigation(tmp_path, monkeypatch)
    entry.invocation_id = "original-invocation"
    assert custody.record_started(drive, entry)
    assert custody.emit(drive, custody.SETTLED, {"run_id": entry.run_id, "state": "succeeded"})
    expected = {name: (copy / name).read_bytes() for name in (
        ".review-evidence/coverage.xml", ".review-evidence/notes.txt", "tracked.txt")}
    try:
        with monkeypatch.context() as crashed:  # the capture row landed; retention never ran
            crashed.setattr(subagent_integration_delegated, "retain_investigation_evidence",
                            lambda drive, entry, block, gateway=None, **kwargs: block)
            block = capture_terminal_patch_for_drive(drive, entry)
        assert entry.patch_captured and not entry.patch_disposed and "disposition" not in block
        custody._CUSTODY.clear()  # admission must read the durable facts, not this worker's memo
        ctx, start, starts = _fresh_start_consumer(monkeypatch, drive, target, entry.task_id, access)
        for decision in ("apply", "reject"):
            assert "INTEGRATE_DELEGATED_INVESTIGATION_EVIDENCE" in _integrate_delegated_patch(
                ctx, entry.run_id, decision, "No disposition is owed")
        result = json.loads(start(ctx, prompt="Check again", subagent_id="investigation-fixture"))
        assert result["status"] == "started", result
        assert len(starts) == 1 and starts[0][1] != entry.invocation_id
        request, invocation = starts[0]
        assert request["access"] == access
        assert request["mode"] == ("ask" if access == "readonly" else "agent")
        runs = custody.replay(drive)
        assert runs["fresh-run"].invocation_id == invocation
        prior = runs[entry.run_id]
        assert prior.invocation_id == entry.invocation_id and prior.patch_captured and not prior.patch_disposed
        assert prior.settled and prior.authority_source == INVESTIGATION_AUTHORITY
        assert find_execution_snapshot(handle.snapshot_id) is not None and copy.is_dir()
        assert {name: (copy / name).read_bytes() for name in expected} == expected
        assert (target / "tracked.txt").read_text() == "one\ntwo\n"  # neither run applied anything
    finally:
        custody._CUSTODY.clear()


@pytest.mark.parametrize("authority,settled,pending,blocker", [
    ("acting_constraint", True, False, "undisposed_patch_run_ids"),
    ("", True, False, "undisposed_patch_run_ids"),  # historical binding without authority
    ("unknown-authority", True, False, "undisposed_patch_run_ids"),
    (INVESTIGATION_AUTHORITY, False, False, "open_run_ids"),
    ("readonly", False, False, "open_run_ids"),
    (INVESTIGATION_AUTHORITY, False, True, "pending_invocation_ids"),
])
def test_fresh_public_start_keeps_other_candidate_open_and_pending_obligations(
        tmp_path, monkeypatch, authority, settled, pending, blocker):
    import json

    from ouroboros.delegate_start_claims import claimed_start_request

    ctx, start, starts = _fresh_start_consumer(monkeypatch, tmp_path, tmp_path, "owner")
    if pending:
        assert custody.record_start_requested(tmp_path, invocation_id="prior", task_id=ctx.task_id,
                                              snapshot_id="snap", authority_source=authority,
                                              request={"prompt": "Original pending work"})
    else:
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id="prior", task_id=ctx.task_id, route_id="some-route", snapshot_id="snap",
            authority_source=authority))
        if settled:
            assert custody.emit(tmp_path, custody.SETTLED, {"run_id": "prior", "state": "succeeded"})
    try:
        custody._CUSTODY.clear()
        result = json.loads(start(ctx, prompt="Check again", subagent_id="investigation-fixture"))
        assert result["status"] == "started" and starts
        claimed, refusal = claimed_start_request(
            tmp_path, actor_ctx=ctx, task_id=ctx.task_id, invocation_id="new-independent",
            request={"prompt": "Check independently"})
        assert claimed and not refusal
        assert custody.invocation_record(tmp_path, "new-independent")["state"] == "pending"
    finally:
        custody._CUSTODY.clear()
