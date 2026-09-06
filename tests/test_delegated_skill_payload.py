"""Delegated skill-payload capability (R1): exact-resource selector, standalone
private snapshot, payload capture adapter, and the parent-only CAS apply.

The restored D10 target class: a top-level task delegates ONE exact non-native
skill payload to the configured harness through a private standalone Git
snapshot; the harness never touches the live payload; the parent applies the
captured harness-authored diff explicitly; the existing skill review goes stale.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.subagent_worktrees import (
    find_execution_snapshot,
    provision_payload_snapshot,
    remove_execution_snapshot,
)


@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    """Same seam as the transport suite: the owned-daemon lifecycle has its own
    focused tests; here every case supplies a fake gateway class."""
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )


def _seed_skill(data: pathlib.Path, name: str = "alpha", bucket: str = "external") -> pathlib.Path:
    skill = data / "skills" / bucket / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"# {name}\n\nA test skill.\n", encoding="utf-8")
    (skill / "plugin.py").write_text("VALUE = 1\n", encoding="utf-8")
    (skill / "notes.txt").write_text("PENDING\n", encoding="utf-8")
    return skill


def _payload_ctx(tmp_path: pathlib.Path, monkeypatch):
    """A genuine TOP-LEVEL context (self_modification profile, no workspace)."""
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snaps"))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    configured = {
        "enabled": True,
        "items": [{
            "subagent_id": "payload-session",
            "name": "Payload session",
            "recommended_use": "Edit an exact delegated skill payload.",
            "route": {
                "kind": "agent_session",
                "target_id": "some-route=weak-model",
                "credential_profile_id": "",
            },
            "effort": "low",
        }],
    }
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", json.dumps(configured))
    ctx = ToolContext(repo_dir=repo, drive_root=data)
    ctx.task_id = "t-payload"
    ctx.task_metadata = {"root_task_id": "t-payload"}
    from ouroboros.subagent_runtime import select_subagent_snapshot

    ctx._payload_subagent_snapshot = select_subagent_snapshot(
        {"OUROBOROS_SUBAGENTS": json.dumps(configured)},
        subagent_id="payload-session",
    )[0]
    return ctx


def _exact_payload_start(ctx, prompt: str, **params):
    from ouroboros.subagent_runtime import exact_start

    return exact_start(ctx, prompt, {
        "snapshot": ctx._payload_subagent_snapshot,
        **params,
    })


class _StartStub:
    """The minimal gateway a payload delegate_start touches."""

    def __init__(self, seen):
        from ouroboros.config import CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION

        self.engine_version = CLAUDEXOR_DELEGATED_MARKER_MIN_VERSION
        self._seen = seen

    def handshake(self, **_kw):
        return {}

    def agent_capabilities(self):
        return {"harnesses": [{
            "id": "some-route", "enabled": True, "status": "ok",
            "accessProfilesSupported": ["readonly", "workspace_write"],
        }]}

    def quota_snapshots(self):
        return []

    def find_project_id(self, root):
        return "prj-existing"

    def register_project(self, root):
        raise AssertionError("must reuse the registration")

    def start_run(self, request, *, idempotency_key=""):
        self._seen["request"] = request
        self._seen["idempotency_key"] = idempotency_key
        return {"runId": "run-p1", "runDir": "/tmp/run-p1"}

    def close(self):
        pass


def _start_payload_run(ctx, monkeypatch, *, skill_name="alpha", bucket="external"):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    seen: dict = {}
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _StartStub(seen))
    delegate._CUSTODY.clear()
    payload = json.loads(_exact_payload_start(
        ctx, "edit the skill", root="skill_payload", bucket=bucket,
        skill_name=skill_name,
    ))
    return payload, seen


def _terminal_wait(ctx, monkeypatch, *, run_id="run-p1",
                   effective_access="workspace_write"):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    class _Stub:
        def handshake(self, **_kw):
            return {}

        def get_run(self, rid, **_kw):
            return {"lastSeq": 9, "summary": {
                "state": "succeeded", "spendUsd": 0.0,
                "effectiveAccess": effective_access,
            }}

        def cancel_run(self, rid, reason=""):
            raise AssertionError(f"the run must NOT be cancelled ({reason})")

        def remove_project(self, pid):
            pass

        def close(self):
            pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Stub())
    return json.loads(delegate._delegate_wait(ctx, run_id, wait_sec=1))


# -- 1A: selector, authority, custody shape -------------------------------------


def test_payload_start_provisions_standalone_snapshot_with_semantic_ref(tmp_path, monkeypatch):
    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    payload, seen = _start_payload_run(ctx, monkeypatch)
    assert payload["status"] == "started", payload
    request = seen["request"]
    # The mutating shape rides the exact binding, never a workspace derivation.
    assert request["access"] == "workspace_write" and request["mode"] == "agent"
    assert request["execution"] == {"isolation": "live", "delegated": True}
    exec_root = pathlib.Path(str(request["scope"]["root"]))
    assert exec_root.resolve().is_relative_to((tmp_path / "snaps").resolve())
    # STANDALONE snapshot: its own .git, the live payload has none and is intact.
    assert (exec_root / ".git").is_dir()
    assert (exec_root / "SKILL.md").read_text(encoding="utf-8").startswith("# alpha")
    assert not (skill / ".git").exists()
    assert pathlib.Path(payload["authority_target_root"]).resolve() == skill.resolve()
    # Durable custody carries the granted shape and the semantic reference.
    entry = custody.replay(tmp_path / "data")["run-p1"]
    assert entry.authority_source == "skill_payload"
    assert entry.access == "workspace_write" and entry.isolation == "live"
    ref = entry.resource_ref
    assert ref["source"] == "external" and ref["skill_name"] == "alpha"
    assert ref["target_root"] == str(skill.resolve()) and ref["payload_hash"]
    # Gate fix 3: the child is TOLD editing this payload is its assignment; the
    # contradictory blanket "skills" ban is narrowed for payload runs only.
    instructions = str(request["instructions"])
    assert "PAYLOAD ASSIGNMENT" in instructions and "'alpha'" in instructions
    assert "runtime controls, skills, or memory" not in instructions
    custody._CUSTODY.clear()


def test_selector_argument_shapes_refuse_typed(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate

    ctx = _payload_ctx(tmp_path, monkeypatch)
    _seed_skill(tmp_path / "data")
    for kwargs, reason in (
        (dict(root="external_workspace", bucket="external", skill_name="alpha"),
         "unsupported_root"),
        (dict(root="skill_payload", bucket="external", skill_name="alpha",
              retry_of="tok1"), "selector_on_retry"),
        (dict(root="skill_payload", bucket="external"), "payload_selector_incomplete"),
        (dict(bucket="external", skill_name="alpha"), "payload_selector_incomplete"),
    ):
        out = json.loads(delegate._delegate_start(ctx, "x", **kwargs))
        assert out["status"] == "refused" and out["reason"] == reason, out


def test_native_missing_and_child_targets_refuse_before_any_gateway(tmp_path, monkeypatch):
    import ouroboros.claudexor_daemon as daemon
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.registry import ToolContext

    ctx = _payload_ctx(tmp_path, monkeypatch)
    native = _seed_skill(tmp_path / "data", name="native-ish", bucket="native")
    (native / ".seed-origin").write_text("seeded\n", encoding="utf-8")

    def _no_gateway():
        raise AssertionError("the refusal must land BEFORE any gateway work")

    monkeypatch.setattr(daemon, "ensure_owned_gateway", _no_gateway)
    out = json.loads(_exact_payload_start(
        ctx, "x", root="skill_payload", bucket="native", skill_name="native-ish"))
    assert out["reason"] == "payload_target_unresolved", out
    out = json.loads(_exact_payload_start(
        ctx, "x", root="skill_payload", bucket="external", skill_name="ghost"))
    assert out["reason"] == "payload_target_unresolved", out
    assert "manifest" in out["detail"].lower() or "SKILL.md" in out["detail"], out
    # A read-only CHILD gets an AUTHORITY denial (Fable F4), not a lookup-
    # flavored refusal: policy is checked before the binding is even built.
    child = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data",
                        task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    child.task_id = "t-child"
    child.task_metadata = {"parent_task_id": "t-payload"}
    child._payload_subagent_snapshot = ctx._payload_subagent_snapshot
    out = json.loads(_exact_payload_start(
        child, "x", root="skill_payload", bucket="external", skill_name="alpha"))
    assert out["reason"] == "payload_delegation_forbidden", out
    assert "AUTHORITY denial" in out["detail"], out


@pytest.mark.parametrize("runtime_mode", ["light", "advanced", "pro"])
def test_markerless_native_delegates_as_external_and_rebinds_by_marker(
    runtime_mode,
    tmp_path,
    monkeypatch,
):
    import ouroboros.safety as safety
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.delegate_integration import _rebind_payload_reference
    from ouroboros.tools.registry import ToolRegistry

    ctx = _payload_ctx(tmp_path, monkeypatch)
    payload = _seed_skill(
        tmp_path / "data",
        name="user-native",
        bucket="native",
    )
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", runtime_mode)
    monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, ""))
    seen: dict = {}
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _StartStub(seen))
    custody._CUSTODY.clear()
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry.set_context(ctx)

    started = json.loads(
        registry.execute(
            "delegate_start",
            {
                "subagent_id": "payload-session",
                "prompt": "edit notes.txt",
                "root": "skill_payload",
                "bucket": "external",
                "skill_name": "user-native",
            },
        )
    )
    assert started["status"] == "started", started
    assert pathlib.Path(started["authority_target_root"]).resolve() == payload.resolve()
    entry = custody.replay(tmp_path / "data")["run-p1"]
    assert entry.resource_ref["source"] == "external"
    assert entry.resource_ref["target_root"] == str(payload.resolve())

    (payload / ".seed-origin").write_text("launcher-seed\n", encoding="utf-8")
    rebound, _binding, refusal = _rebind_payload_reference(
        ctx,
        entry.resource_ref,
        entry.target_root,
        tool="integrate_delegated_patch",
        context="test",
    )
    assert rebound is None
    assert "payload_target_unresolved" in refusal
    custody._CUSTODY.clear()


def test_second_delegation_on_same_payload_is_refused_cheaply(tmp_path, monkeypatch):
    import ouroboros.claudexor_daemon as daemon

    ctx = _payload_ctx(tmp_path, monkeypatch)
    _seed_skill(tmp_path / "data")
    payload, _ = _start_payload_run(ctx, monkeypatch)
    assert payload["status"] == "started"

    def _no_gateway():
        raise AssertionError("busy refusal must land BEFORE any gateway work")

    monkeypatch.setattr(daemon, "ensure_owned_gateway", _no_gateway)
    out = json.loads(_exact_payload_start(
        ctx, "second", root="skill_payload", bucket="external", skill_name="alpha"))
    assert out["reason"] == "replacement_requires_settlement", out
    assert out["open_run_ids"] == ["run-p1"]
    custody._CUSTODY.clear()


def test_wait_after_start_replays_recorded_shape_and_does_not_cancel(tmp_path, monkeypatch):
    ctx = _payload_ctx(tmp_path, monkeypatch)
    _seed_skill(tmp_path / "data")
    payload, _ = _start_payload_run(ctx, monkeypatch)
    assert payload["status"] == "started"
    # The stub's cancel_run raises: a re-derivation (readonly) would cancel the
    # workspace_write run as widened on this very first wait (the R1-2 defect).
    out = _terminal_wait(ctx, monkeypatch)
    assert out["status"] == "terminal", out
    assert out["access_evidence"]["effective"] == "workspace_write"
    assert out["workspace_capture"]["status"] in ("ready_no_changes", "ready_with_changes")
    custody._CUSTODY.clear()


def test_duplicate_started_rows_keep_first_binding_facts(tmp_path):
    drive = tmp_path
    entry = custody.RunCustody(
        run_id="run-d", task_id="t-a", route_id="r",
        snapshot_id="snap1", execution_root="/x/exec", baseline_sha="b1",
        target_root="/x/target", authority_source="skill_payload",
        resource_ref={"skill_name": "alpha", "payload_hash": "h1"})
    custody.record_started(drive, entry, shape={
        "access": "workspace_write", "mode": "agent", "isolation": "live",
        "delegated": True, "root": "/x/exec"})
    # A later idempotent STARTED row minted WITHOUT the binding facts.
    custody.record_started(drive, custody.RunCustody(run_id="run-d", task_id="t-a",
                                                     route_id="r"))
    replayed = custody.replay(drive)["run-d"]
    assert replayed.snapshot_id == "snap1" and replayed.baseline_sha == "b1"
    assert replayed.target_root == "/x/target"
    assert replayed.authority_source == "skill_payload"
    assert replayed.resource_ref["payload_hash"] == "h1"
    assert replayed.access == "workspace_write" and replayed.delegated is True
    custody._CUSTODY.clear()


def test_pending_invocation_and_retry_records_carry_the_resource_ref(tmp_path):
    drive = tmp_path
    ref = {"root": "skill_payload", "source": "external", "skill_name": "alpha",
           "target_root": "/x/target", "payload_hash": "h1"}
    custody.record_start_requested(
        drive, run_id="", task_id="t-a", idempotency_key="k", invocation_id="inv1",
        max_seconds=60, request={"prompt": "x"}, project_id="p", project_owned=False,
        route="r", root_task_id="t-a", parent_task_id="", snapshot_id="snap1",
        execution_root="/x/exec", baseline_sha="b1", target_root="/x/target",
        authority_source="skill_payload", resource_ref=ref)
    record = custody.invocation_record(drive, "inv1")
    assert record["resource_ref"] == ref
    pending = custody.pending_invocations(drive)
    assert pending and pending[0]["resource_ref"] == ref


# -- 1B: standalone snapshot + capture adapter -----------------------------------


def test_snapshot_copies_symlinks_as_symlinks_and_drops_escapes(tmp_path):
    data = tmp_path / "data"
    skill = _seed_skill(data)
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    os.symlink("SKILL.md", skill / "rel.md")                    # confined relative
    os.symlink(str(skill / "plugin.py"), skill / "abs.py")      # confined absolute
    os.symlink(str(outside), skill / "esc.txt")                 # escaping
    handle = provision_payload_snapshot(
        target_root=skill, task_id="t1", snapshot_id="snapS",
        worktree_root=tmp_path / "snaps", data_dir=data)
    snap = pathlib.Path(handle.path)
    assert handle.standalone is True and handle.payload_hash
    assert (snap / "rel.md").is_symlink()
    assert os.readlink(snap / "rel.md") == "SKILL.md"
    assert (snap / "abs.py").is_symlink()
    rewritten = os.readlink(snap / "abs.py")
    assert not os.path.isabs(rewritten)                          # rewritten relative
    assert (snap / "abs.py").resolve() == (snap / "plugin.py").resolve()
    assert not os.path.lexists(snap / "esc.txt")                 # not copied
    record = find_execution_snapshot("snapS", data_dir=data)
    assert record and record["standalone"] is True
    # Standalone cleanup: only the private dir + registry row disappear.
    assert remove_execution_snapshot("snapS", worktree_root=tmp_path / "snaps", data_dir=data)
    assert not snap.exists() and skill.is_dir()


def _payload_entry(handle, skill, *, run_id="run-p1", settled=True):
    entry = custody.RunCustody(
        run_id=run_id, task_id="t-payload", route_id="some-route",
        snapshot_id=handle.snapshot_id, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=str(skill.resolve()),
        authority_source="skill_payload", settled=settled,
        access="workspace_write", mode="agent", isolation="live", delegated=True,
        resource_ref={"root": "skill_payload", "source": "external",
                      "skill_name": skill.name, "target_root": str(skill.resolve()),
                      "payload_hash": handle.payload_hash})
    custody._CUSTODY[entry.run_id] = entry
    return entry


def _provisioned(tmp_path, monkeypatch, *, name="alpha"):
    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data", name=name)
    handle = provision_payload_snapshot(
        target_root=skill, task_id="t-payload", snapshot_id="snapP")
    custody._CUSTODY.clear()
    return ctx, skill, handle


def test_capture_transports_utf8_with_nul_and_loader_junk_stays_out(tmp_path, monkeypatch):
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    # The harness edits the SNAPSHOT: a UTF-8-with-NUL file (git's binary
    # heuristic would veto it in a text diff), a plain edit, a deletion, junk.
    (exec_root / "table.txt").write_bytes("col1\0col2\nrow\0data\n".encode("utf-8"))
    (exec_root / "plugin.py").write_text("VALUE = 2\n", encoding="utf-8")
    (exec_root / "notes.txt").unlink()
    (exec_root / "node_modules").mkdir()
    (exec_root / "node_modules" / "junk.js").write_text("x\n", encoding="utf-8")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "ready_with_changes", capture
    manifest = json.loads(pathlib.Path(capture["manifest_artifact"]).read_text())
    assert manifest["capture_kind"] == "skill_payload"
    assert set(manifest["tracked_changed"]) == {"table.txt", "plugin.py", "notes.txt"}
    assert manifest["blocked_reserved_paths"] == []
    assert manifest["result_content_hash"] and manifest["baseline_payload_hash"]
    assert manifest["result_content_hash"] != manifest["baseline_payload_hash"]
    custody._CUSTODY.clear()


def test_non_utf8_addition_is_a_typed_capture_failure(tmp_path, monkeypatch):
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    (pathlib.Path(handle.path) / "blob.bin").write_bytes(b"\xff\xfe\x00\x01binary")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "failed", capture
    assert entry.patch_captured is False
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, out
    assert find_execution_snapshot("snapP") is not None    # snapshot preserved
    custody._CUSTODY.clear()


# -- 1C: parent-only apply ---------------------------------------------------------


def _captured(tmp_path, monkeypatch, edit=None):
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    if edit is None:
        (exec_root / "notes.txt").write_text("DONE\n", encoding="utf-8")
        (exec_root / "extra.txt").write_bytes("nul\0ok\n".encode("utf-8"))
    else:
        edit(exec_root)
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    return ctx, skill, handle, entry, capture


def test_apply_from_a_foreign_cwd_writes_the_live_payload(tmp_path, monkeypatch):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    assert capture["status"] == "ready_with_changes", capture
    state_dir = tmp_path / "data" / "state" / "skills" / "alpha"
    state_dir.mkdir(parents=True)
    (state_dir / "grants.json").write_text('{"granted": []}\n', encoding="utf-8")
    grants_before = (state_dir / "grants.json").read_bytes()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    old_cwd = os.getcwd()
    os.chdir(elsewhere)   # a foreign process cwd must not misroute the apply
    try:
        out = _integrate_delegated_patch(ctx, "run-p1", "apply", "looks good")
    finally:
        os.chdir(old_cwd)
    assert "✅ Integrated" in out, out
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "DONE\n"
    assert (skill / "extra.txt").read_bytes() == "nul\0ok\n".encode("utf-8")
    assert not (skill / ".git").exists()
    assert entry.patch_disposed == "applied"
    assert find_execution_snapshot("snapP") is None
    # Lifecycle state stays byte-identical; the stale review is a HASH fact.
    assert (state_dir / "grants.json").read_bytes() == grants_before
    assert "STALE" in out and "skill_review" in out
    # The extension-reconcile marker was queued for the mutated skill; the
    # receipt says QUEUED and never claims the reconcile completed (Sol P2-2).
    assert "QUEUED" in out and "reconciled off" not in out, out
    from ouroboros.extension_reconcile_queue import list_extension_reconcile_requests

    requests = list_extension_reconcile_requests(tmp_path / "data")
    assert any(r["skill"] == "alpha" for r in requests), requests
    custody._CUSTODY.clear()


def test_stale_cas_conflict_preserves_material_and_changes_nothing(tmp_path, monkeypatch):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    (skill / "plugin.py").write_text("VALUE = 99  # drifted\n", encoding="utf-8")
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_CONFLICT" in out, out
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n"
    assert entry.patch_disposed == ""
    assert find_execution_snapshot("snapP") is not None
    assert pathlib.Path(capture["patch_artifact"]).exists()
    custody._CUSTODY.clear()


def test_already_applied_content_disposes_idempotently(tmp_path, monkeypatch):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    # A crashed earlier attempt landed the patch but never recorded disposition.
    subprocess.run(["git", "apply", capture["patch_artifact"]], cwd=str(skill),
                   capture_output=True, check=True)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "ALREADY carries" in out, out
    assert entry.patch_disposed == "applied"
    assert find_execution_snapshot("snapP") is None
    custody._CUSTODY.clear()


def test_reserved_path_patch_refuses_whole_apply_and_preserves_candidate(tmp_path, monkeypatch):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    def edit(exec_root):
        (exec_root / "notes.txt").write_text("DONE\n", encoding="utf-8")
        (exec_root / ".clawhub.json").write_text('{"forged": true}\n', encoding="utf-8")

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch, edit=edit)
    assert capture["status"] == "ready_with_changes", capture
    manifest = json.loads(pathlib.Path(capture["manifest_artifact"]).read_text())
    assert manifest["blocked_reserved_paths"] == [".clawhub.json"]
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_DELEGATED_RESERVED_PATHS" in out, out
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n"
    assert not (skill / ".clawhub.json").exists()
    assert entry.patch_disposed == ""
    assert pathlib.Path(capture["patch_artifact"]).exists()
    assert find_execution_snapshot("snapP") is not None
    custody._CUSTODY.clear()


def test_moved_or_deleted_target_is_refused_at_apply(tmp_path, monkeypatch):
    import shutil

    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    shutil.rmtree(skill)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "payload_target_unresolved" in out or "payload_target_moved" in out, out
    assert entry.patch_disposed == ""
    assert find_execution_snapshot("snapP") is not None
    custody._CUSTODY.clear()


def test_reject_needs_no_live_target_and_releases_the_snapshot(tmp_path, monkeypatch):
    import shutil

    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    shutil.rmtree(skill)   # owner deleted the skill; reject must still work
    out = _integrate_delegated_patch(ctx, "run-p1", "reject", "not wanted")
    assert "🚫 Rejected" in out, out
    assert entry.patch_disposed == "rejected"
    assert find_execution_snapshot("snapP") is None
    custody._CUSTODY.clear()


# -- golden registry-level E2E ---------------------------------------------------


def test_registry_golden_e2e_start_wait_apply_review_stale(tmp_path, monkeypatch):
    import ouroboros.safety as safety
    from ouroboros.gateways import claudexor as gw
    from ouroboros.skill_loader import load_skill
    from ouroboros.tools.registry import ToolRegistry

    ctx = _payload_ctx(tmp_path, monkeypatch)
    data = tmp_path / "data"
    skill = _seed_skill(data)
    # A VALID script manifest so the closing preflight+review leg (Sol P2-1)
    # exercises the real deterministic preflight, not a manifest-parse failure.
    (skill / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Test skill.\nversion: 0.1.0\n"
        "type: script\nruntime: python3\nscripts:\n  - name: run.py\n"
        "    description: Run.\n---\n", encoding="utf-8")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
    sibling = _seed_skill(data, name="beta")
    sibling_bytes = (sibling / "SKILL.md").read_bytes()
    # A PASS review bound to the CURRENT payload content.
    loaded = load_skill(skill, data)
    state_dir = data / "state" / "skills" / "alpha"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "review.json").write_text(json.dumps({
        "status": "pass", "content_hash": loaded.content_hash}), encoding="utf-8")
    (state_dir / "enabled.json").write_text('{"enabled": false}\n', encoding="utf-8")
    enabled_before = (state_dir / "enabled.json").read_bytes()
    loaded = load_skill(skill, data)   # re-read WITH the review state on disk
    assert loaded.review.status == "clean"
    assert not loaded.review.is_stale_for(loaded.content_hash)

    monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=data)
    registry.set_context(ctx)
    seen: dict = {}
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _StartStub(seen))
    custody._CUSTODY.clear()

    started = json.loads(registry.execute("delegate_start", {
        "subagent_id": "payload-session",
        "prompt": "flip PENDING to DONE in notes.txt",
        "root": "skill_payload", "bucket": "external", "skill_name": "alpha"}))
    assert started["status"] == "started", started
    exec_root = pathlib.Path(str(seen["request"]["scope"]["root"]))
    # The deterministic "harness" edits the private snapshot only.
    (exec_root / "notes.txt").write_text("DONE\n", encoding="utf-8")
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n"

    out = _terminal_wait(ctx, monkeypatch)
    assert out["status"] == "terminal"
    capture = out["workspace_capture"]
    assert capture["status"] == "ready_with_changes", capture
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n"

    applied = registry.execute("integrate_delegated_patch", {
        "run_id": "run-p1", "decision": "apply", "reason": "golden"})
    assert "✅ Integrated" in applied, applied
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "DONE\n"
    assert not (skill / ".git").exists()
    # Sibling skill + lifecycle sidecars byte-identical; enablement unchanged.
    assert (sibling / "SKILL.md").read_bytes() == sibling_bytes
    assert (state_dir / "enabled.json").read_bytes() == enabled_before
    # The old PASS review is now STALE for the new content — reachable, not faked.
    refreshed = load_skill(skill, data)
    assert refreshed.review.is_stale_for(refreshed.content_hash)

    # Sol P2-1: CLOSE the loop with the REAL skill preflight + review path over
    # the APPLIED content — reviewer LLM faked deterministically, no live model.
    from tests.test_skill_review_persist_guard import _pass_actor

    monkeypatch.setattr(
        "ouroboros.skill_review._run_skill_advisory_pre_review",
        lambda *_a, **_kw: {"status": "empty"})
    monkeypatch.setattr(
        "ouroboros.tools.review._handle_multi_model_review",
        lambda *_a, **_kw: json.dumps(
            {"results": [_pass_actor("fake/a"), _pass_actor("fake/b")]}))
    preflight = json.loads(registry.execute("skill_preflight", {"skill": "alpha"}))
    assert preflight.get("ok") is True, preflight
    review_out = registry.execute("skill_review", {"skill": "alpha"})
    from ouroboros.skill_loader import load_review_state

    persisted = load_review_state(data, "alpha")
    assert persisted.status == "clean", (persisted.status, review_out[:800])
    # The fresh verdict is bound to the APPLIED content hash, so the loop ends
    # with an executable review for exactly the delegated result.
    assert persisted.content_hash == refreshed.content_hash
    closed = load_skill(skill, data)
    assert not closed.review.is_stale_for(closed.content_hash)
    # Nothing about the delegated run fabricates grants or authorship: no grant
    # state appears and the persisted verdict carries no run attribution.
    assert not (state_dir / "grants.json").exists()
    review_doc = json.loads((state_dir / "review.json").read_text(encoding="utf-8"))
    assert not review_doc.get("auto_granted_keys")
    assert "run-p1" not in json.dumps(review_doc)
    assert (state_dir / "enabled.json").read_bytes() == enabled_before
    custody._CUSTODY.clear()


def test_legacy_disabled_claude_code_edit_blocks_the_selector_call(tmp_path, monkeypatch):
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    _seed_skill(data)
    registry = ToolRegistry(repo_dir=repo, drive_root=data)
    contract = build_task_contract({"description": "x",
                                    "disabled_tools": ["claude_code_edit"]})
    registry.set_context(ToolContext(repo_dir=repo, drive_root=data,
                                     task_metadata={"task_contract": contract}))
    blocked = registry.execute("delegate_start", {
        "prompt": "x", "root": "skill_payload", "bucket": "external",
        "skill_name": "alpha"})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in blocked and "disabled_tools" in blocked


# -- 5-lane gate fix batch ---------------------------------------------------------


def test_file_replaced_by_escaping_symlink_rides_as_deletion(tmp_path, monkeypatch):
    """Gate fix 1b (reviewer repro): plugin.py → /tmp/.../secret.txt symlink in the
    snapshot must NOT carry stale baseline content or the symlink itself — the
    inventory drops the escape, so the candidate stages the path as a DELETION."""
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    secret = tmp_path / "secret.txt"
    secret.write_text("SECRET\n", encoding="utf-8")
    (exec_root / "plugin.py").unlink()
    os.symlink(str(secret), exec_root / "plugin.py")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "ready_with_changes", capture
    patch_bytes = pathlib.Path(capture["patch_artifact"]).read_bytes()
    assert b"deleted file mode 100644" in patch_bytes
    assert b"120000" not in patch_bytes and b"SECRET" not in patch_bytes
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "✅ Integrated" in out, out
    # The live payload never receives the symlink; the honest candidate deleted it.
    assert not (skill / "plugin.py").is_symlink()
    assert not (skill / "plugin.py").exists()
    custody._CUSTODY.clear()


def test_escaping_symlink_patch_is_refused_whole_at_apply(tmp_path, monkeypatch):
    """Gate fix 1a: apply-time containment judges the CANDIDATE — a patch hunk
    introducing a symlink whose target escapes the live payload refuses the
    WHOLE apply with the candidate preserved."""
    import hashlib

    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    assert capture["status"] == "ready_with_changes"
    # Forge a candidate introducing an escaping symlink (crafted in a scratch
    # repo; the guard must judge patch CONTENT, not trust capture provenance).
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=scratch, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit",
                    "--allow-empty", "-q", "-m", "base"], cwd=scratch, check=True)
    os.symlink("../../secret.txt", scratch / "evil_link")
    subprocess.run(["git", "add", "-A"], cwd=scratch, check=True)
    forged = subprocess.run(["git", "diff", "--cached", "--binary", "HEAD"],
                            cwd=scratch, capture_output=True, check=True).stdout
    patch_path = pathlib.Path(capture["patch_artifact"])
    patch_path.write_bytes(forged)
    manifest_path = pathlib.Path(capture["manifest_artifact"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["sha256"] = hashlib.sha256(forged).hexdigest()
    manifest["tracked_changed"] = ["evil_link"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_DELEGATED_RESERVED_PATHS" in out and "evil_link" in out, out
    assert not os.path.lexists(skill / "evil_link")
    assert entry.patch_disposed == ""
    assert patch_path.exists()                      # candidate preserved
    assert find_execution_snapshot("snapP") is not None
    custody._CUSTODY.clear()


def test_child_git_config_diff_driver_does_not_execute_at_capture(tmp_path, monkeypatch):
    """Gate fix 2 (reviewer repro): a snapshot-local `diff.evil.command` written
    by the child must not execute in the PARENT at capture, and the patch must
    carry the real content, not driver output."""
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    pwned = tmp_path / "pwned.txt"
    (exec_root / ".git" / "config").write_text(
        "[diff \"evil\"]\n\tcommand = sh -c 'touch %s; echo'\n" % pwned,
        encoding="utf-8")
    (exec_root / ".gitattributes").write_text("*.py diff=evil\n", encoding="utf-8")
    (exec_root / "plugin.py").write_text("VALUE = 2\n", encoding="utf-8")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "ready_with_changes", capture
    assert not pwned.exists(), "child-controlled git config executed in the parent"
    patch_bytes = pathlib.Path(capture["patch_artifact"]).read_bytes()
    assert b"+VALUE = 2" in patch_bytes
    custody._CUSTODY.clear()


def test_payload_instructions_variant_is_payload_only(tmp_path, monkeypatch):
    """Gate fix 3: ordinary runs keep the blanket ban byte-identically; only a
    payload run gets the narrowed ban plus the explicit permission block."""
    from ouroboros.subagents import delegated_run_shape
    from ouroboros.tools.delegate import _HOST_INSTRUCTIONS, _host_instructions

    ordinary = _host_instructions(delegated_run_shape(False))
    assert "runtime controls, skills, or memory" in ordinary
    assert "PAYLOAD ASSIGNMENT" not in ordinary
    payload = _host_instructions(delegated_run_shape(True), payload_skill="alpha")
    assert "runtime controls, skills, or memory" not in payload
    assert "runtime controls or memory" in payload
    assert "PAYLOAD ASSIGNMENT" in payload and "'alpha'" in payload
    assert "runtime controls, skills, or memory" in _HOST_INSTRUCTIONS  # source intact


def test_idempotent_already_applied_branch_runs_the_finalizer(tmp_path, monkeypatch):
    """Gate fix 4: the already-applied branch reconciles and reports staleness
    exactly like a fresh apply (it used to dispose and skip both)."""
    from ouroboros.extension_reconcile_queue import list_extension_reconcile_requests
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    subprocess.run(["git", "apply", capture["patch_artifact"]], cwd=str(skill),
                   capture_output=True, check=True)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "ALREADY carries" in out, out
    assert "STALE" in out and "skill_review" in out
    # Sol P2-2: the receipt claims a QUEUED request, never a completed reconcile.
    assert "QUEUED" in out and "reconciled off" not in out, out
    requests = list_extension_reconcile_requests(tmp_path / "data")
    assert any(r["skill"] == "alpha" for r in requests), requests
    custody._CUSTODY.clear()


def test_reconcile_queue_failure_degrades_the_receipt_honestly(tmp_path, monkeypatch):
    """Gate fix 4: a failed reconcile queue-write must not claim the extension
    was reconciled off — the receipt states the failure; the apply stands."""
    import ouroboros.extension_reconcile_queue as reconcile_queue

    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)

    def _boom(*_a, **_k):
        raise OSError("queue disk full")

    monkeypatch.setattr(reconcile_queue, "request_extension_reconcile", _boom)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "✅ Integrated" in out and "STALE" in out, out
    assert "could NOT be queued" in out, out
    assert "reconciled off until re-review" not in out, out
    assert entry.patch_disposed == "applied"
    custody._CUSTODY.clear()


def test_parallel_starts_on_same_payload_yield_exactly_one_winner(tmp_path, monkeypatch):
    """Gate fix 5 + Sol delta 5a: deterministic REQUESTED→STARTED window, x50.

    Both starts pass the cheap early check and provision; the winner then claims
    (durable START_REQUESTED) and is HELD inside the gateway POST — after its
    request row, before its STARTED row. The loser runs its locked busy predicate
    strictly inside that window: the old two-pass read could miss a holder whose
    transition landed between the passes; the single-pass snapshot cannot. Fifty
    fresh-drive iterations prove the interleaving is stable, not schedule-lucky."""
    import threading

    import ouroboros.tools.delegate as delegate
    import ouroboros.tools.delegate_integration as integration
    from ouroboros.gateways import claudexor as gw

    real_provision = integration._provision_payload_snapshot
    real_replay = custody.replay
    for iteration in range(50):
        base = tmp_path / f"iter{iteration}"
        base.mkdir()
        ctx = _payload_ctx(base, monkeypatch)
        _seed_skill(base / "data")
        delegate._CUSTODY.clear()
        barrier = threading.Barrier(2, timeout=30)
        in_window = threading.Event()   # winner: REQUESTED durable, STARTED not
        release = threading.Event()     # loser observed busy; let winner finish

        def _synced(*args, **kwargs):
            result = real_provision(*args, **kwargs)
            barrier.wait()
            if threading.current_thread().name != "winner":
                assert in_window.wait(30), "winner never reached its window"
            return result

        class _WindowStub(_StartStub):
            def start_run(self, request, *, idempotency_key=""):
                in_window.set()
                assert release.wait(30), "the loser never settled"
                return super().start_run(request, idempotency_key=idempotency_key)

        def _replay_hook(drive_root, rows=None):
            state = real_replay(drive_root, rows=rows)
            if (threading.current_thread().name != "winner"
                    and in_window.is_set() and not release.is_set()):
                # Mutant-killer: the winner's STARTED row lands durably AFTER
                # this pass returned and BEFORE the pending projection runs —
                # the exact hole a two-pass (re-reading) predicate had. Only a
                # single shared row snapshot still reports busy here.
                release.set()
                thread.join(timeout=30)
            return state

        seen: dict = {}
        monkeypatch.setattr(custody, "replay", _replay_hook)
        monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _WindowStub(seen))
        monkeypatch.setattr(integration, "_provision_payload_snapshot", _synced)
        monkeypatch.setattr(delegate, "_provision_payload_snapshot", _synced)
        winner_out: list = []

        def _winner():
            winner_out.append(json.loads(_exact_payload_start(
                ctx, "start winner", root="skill_payload", bucket="external",
                skill_name="alpha")))

        thread = threading.Thread(target=_winner, name="winner")
        thread.start()
        loser = json.loads(_exact_payload_start(
            ctx, "start loser", root="skill_payload", bucket="external",
            skill_name="alpha"))
        release.set()
        thread.join(timeout=60)
        assert not thread.is_alive(), f"iteration {iteration}: winner hung"
        assert loser["status"] == "refused", (iteration, loser)
        assert loser["reason"] == "payload_delegation_busy", (iteration, loser)
        assert winner_out and winner_out[0]["status"] == "started", (iteration, winner_out)
    custody._CUSTODY.clear()


def test_snapshot_root_under_runtime_data_refuses_provisioning(tmp_path):
    """Gate fix 6 (reviewer repro): a snapshot root resolving inside the runtime
    data root is refused — no child-writable Git repo inside live state."""
    data = tmp_path / "data"
    skill = _seed_skill(data)
    with pytest.raises(ValueError, match="runtime data"):
        provision_payload_snapshot(
            target_root=skill, task_id="t1", snapshot_id="snapBad",
            worktree_root=data / "state" / "snaps", data_dir=data)
    assert not (data / "state" / "snaps").exists()


def test_registry_save_failure_leaves_no_orphan_snapshot_dir(tmp_path, monkeypatch):
    """Gate fix 7: a registry-write failure removes the snapshot directory —
    an unregistered directory would be invisible to disposal/retention."""
    import ouroboros.subagent_worktrees as worktrees

    data = tmp_path / "data"
    skill = _seed_skill(data)

    def _boom(*_a, **_k):
        raise OSError("registry disk full")

    monkeypatch.setattr(worktrees, "_save_registry", _boom)
    with pytest.raises(OSError, match="registry disk full"):
        provision_payload_snapshot(
            target_root=skill, task_id="t1", snapshot_id="snapReg",
            worktree_root=tmp_path / "snaps", data_dir=data)
    leftovers = list((tmp_path / "snaps").glob("dlgp_*")) if (tmp_path / "snaps").exists() else []
    assert leftovers == [], leftovers


def test_first_wins_is_keyed_on_the_first_shape_carrying_row(tmp_path):
    """Gate fix 8b: a recorded delegated=False survives a later True, and an
    empty resource_ref is never 'filled' by a later row — in BOTH the replay
    and the in-process memo (8a: same merge, no raw replacement)."""
    drive = tmp_path
    custody._CUSTODY.clear()
    first = custody.RunCustody(run_id="run-f", task_id="t-a", route_id="r")
    custody.record_started(drive, first, shape={
        "access": "readonly", "mode": "ask", "isolation": "", "delegated": False})
    # Later duplicate claims a WIDER shape and a filled-in resource_ref.
    later = custody.RunCustody(
        run_id="run-f", task_id="t-a", route_id="r",
        access="workspace_write", mode="agent", isolation="live", delegated=True,
        resource_ref={"skill_name": "late", "payload_hash": "hX"})
    custody.record_started(drive, later, shape={
        "access": "workspace_write", "mode": "agent", "isolation": "live",
        "delegated": True})
    replayed = custody.replay(drive)["run-f"]
    assert replayed.delegated is False and replayed.access == "readonly"
    assert replayed.mode == "ask" and replayed.resource_ref == {}
    status, memo = custody.lookup(drive, "t-a", "run-f")
    assert status == custody.OWNED
    assert memo.delegated is False and memo.access == "readonly"
    assert memo.resource_ref == {}
    # The memo answers EXACTLY what a restart would replay.
    for attr in ("access", "mode", "isolation", "delegated", "resource_ref",
                 "snapshot_id", "target_root", "authority_source"):
        assert getattr(memo, attr) == getattr(replayed, attr), attr
    custody._CUSTODY.clear()


def test_recovered_pending_invocation_object_carries_the_shape(tmp_path, monkeypatch):
    """Gate fix 8c: recovery copies access/mode/isolation/delegated onto the
    in-memory object, not only the resource_ref."""
    drive = tmp_path
    custody._CUSTODY.clear()
    ref = {"root": "skill_payload", "source": "external", "skill_name": "alpha",
           "target_root": "/x/target", "payload_hash": "h1"}
    body = {"prompt": "x", "access": "workspace_write", "mode": "agent",
            "execution": {"isolation": "live", "delegated": True},
            "scope": {"root": "/x/exec"}, "primaryHarness": "r"}
    custody.record_start_requested(
        drive, run_id="", task_id="t-a", idempotency_key="k", invocation_id="invR",
        max_seconds=60, request=body, project_id="p", project_owned=False,
        route="r", root_task_id="t-a", parent_task_id="", snapshot_id="snap1",
        execution_root="/x/exec", baseline_sha="b1", target_root="/x/target",
        authority_source="skill_payload", resource_ref=ref)
    record = custody.pending_invocations(drive)[0]
    monkeypatch.setattr(custody, "_reconcile_one", lambda *_a, **_k: {"ok": True})

    class _Gw:
        def start_run(self, request, *, idempotency_key=""):
            return {"runId": "run-rec"}

    custody._recover_pending_invocation(drive, _Gw(), record)
    obj = custody._CUSTODY["run-rec"]
    assert obj.access == "workspace_write" and obj.mode == "agent"
    assert obj.isolation == "live" and obj.delegated is True
    assert obj.resource_ref == ref and obj.authority_source == "skill_payload"
    custody._CUSTODY.clear()


def test_unknown_root_via_registry_is_typed_unsupported_root(tmp_path, monkeypatch):
    """Gate fix 9: an unknown root value falls through to the handler's TYPED
    unsupported_root refusal, never an untyped binding ValueError."""
    import ouroboros.safety as safety
    from ouroboros.tools.registry import ToolRegistry

    ctx = _payload_ctx(tmp_path, monkeypatch)
    _seed_skill(tmp_path / "data")
    monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, ""))
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry.set_context(ctx)
    out = registry.execute("delegate_start", {
        "subagent_id": "payload-session", "prompt": "x", "root": "repo",
        "bucket": "external", "skill_name": "alpha"})
    parsed = json.loads(out)
    assert parsed["status"] == "refused", parsed
    assert parsed["reason"] == "unsupported_root", parsed


# -- Sol scope-review fix batch (P1 trust defects, P2 contract gaps) ----------------


def _capture_manifest(capture) -> dict:
    return json.loads(pathlib.Path(capture["manifest_artifact"]).read_text(encoding="utf-8"))


def test_child_forged_index_only_blob_is_invisible_to_capture(tmp_path, monkeypatch):
    """Sol P1-1 (reviewer repro): a non-UTF-8 blob the child staged ONLY into the
    snapshot's own .git/index (absent from the worktree) must not exist for the
    capture — the parent diffs a FRESH parent-owned index seeded from the
    RECORDED baseline commit and never reads child .git/index — while a
    legitimate worktree edit still captures and applies."""
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    (exec_root / "notes.txt").write_text("DONE\n", encoding="utf-8")  # legitimate edit
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1"}
    blob = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"], cwd=str(exec_root),
        input=b"\x00\xff\xfe forged opaque bytes", capture_output=True, env=env, check=True)
    sha = blob.stdout.decode("ascii").strip()
    subprocess.run(
        ["git", "update-index", "--add", "--cacheinfo", f"100644,{sha},evil.bin"],
        cwd=str(exec_root), capture_output=True, env=env, check=True)
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "ready_with_changes", capture
    manifest = _capture_manifest(capture)
    assert manifest["tracked_changed"] == ["notes.txt"], manifest
    patch_bytes = pathlib.Path(capture["patch_artifact"]).read_bytes()
    assert b"evil.bin" not in patch_bytes and b"\xff\xfe" not in patch_bytes
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "✅ Integrated" in out, out
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "DONE\n"
    assert not (skill / "evil.bin").exists()
    custody._CUSTODY.clear()


def test_symlinked_git_config_is_typed_capture_failure_without_writethrough(
        tmp_path, monkeypatch):
    """Sol P1-2 (reviewer repro): .git/config -> sentinel must fail capture typed
    and the sentinel must stay byte-identical (nothing writes through the link)."""
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    sentinel = tmp_path / "sentinel.cfg"
    sentinel.write_bytes(b"[core]\n\tsentinel = untouched\n")
    before = sentinel.read_bytes()
    config = exec_root / ".git" / "config"
    config.unlink()
    os.symlink(str(sentinel), config)
    (exec_root / "notes.txt").write_text("DONE\n", encoding="utf-8")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "failed", capture
    note = _capture_manifest(capture)["note"]
    assert "snapshot git metadata untrusted" in note and ".git/config" in note
    assert sentinel.read_bytes() == before
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n"
    custody._CUSTODY.clear()


def test_symlinked_git_dir_is_typed_capture_failure(tmp_path, monkeypatch):
    """Sol P1-2: the whole .git replaced by a symlink to an outside directory is
    refused before any parent git operation."""
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    outside = tmp_path / "outside_git"
    shutil.move(str(exec_root / ".git"), str(outside))
    os.symlink(str(outside), exec_root / ".git")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "failed", capture
    note = _capture_manifest(capture)["note"]
    assert "snapshot git metadata untrusted" in note
    assert ".git is not a real directory" in note
    custody._CUSTODY.clear()


def test_schema_and_docs_split_git_staging_from_payload_live_apply():
    """Sol P2-3 pin: integrate_delegated_patch's schema and the deep-delegation
    docs describe the Git lane as staged-into-active-root and the payload lane
    as a live non-Git CAS apply — no universal 'staged' claim covers both."""
    from ouroboros.tools.subagent_integration import get_tools

    entry = next(t for t in get_tools() if t.name == "integrate_delegated_patch")
    desc = entry.schema["description"]
    assert "LIVE apply into the non-Git payload" in desc
    assert "nothing is staged into your active root" in desc
    decision = entry.schema["parameters"]["properties"]["decision"]["description"]
    assert "STAGED into your active root" in decision
    assert "applied LIVE into the non-Git payload" in decision
    arch = (pathlib.Path(__file__).resolve().parents[1] / "docs" /
            "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "staging substrate differs" in arch
    assert "A SKILL-PAYLOAD target captures through the payload adapter" in arch
    assert "QUEUES the extension reconcile request" in arch


def test_registry_and_custody_baseline_disagreement_fails_capture_typed(
        tmp_path, monkeypatch):
    """Sol P1-1: the baseline identity is the HOST registry's; a custody row that
    disagrees (or a missing registry record) is a typed failure, not a diff
    against whichever sha happens to be replayed."""
    from ouroboros.tools.delegate import _capture_terminal_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    entry = _payload_entry(handle, skill)
    entry.baseline_sha = "0" * 40  # forged/corrupt custody row
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "failed", capture
    assert "disagree on the baseline" in _capture_manifest(capture)["note"]
    custody._CUSTODY.clear()


# -- Sol P1 representation batch (raw bytes, modes, post-apply assert) --------------


def test_existing_gitattributes_crlf_edit_transports_raw_bytes(tmp_path, monkeypatch):
    """Sol P1 (reviewer repro a): a payload-authored `.gitattributes text eol=lf`
    must not forge the staged content — a CRLF edit rides RAW, applies cleanly,
    and the live raw bytes equal the candidate with equal loader hashes."""
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.delegate_integration import payload_content_hash
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    (skill / ".gitattributes").write_text("notes.txt text eol=lf\n", encoding="utf-8")
    handle = provision_payload_snapshot(
        target_root=skill, task_id="t-payload", snapshot_id="snapP")
    custody._CUSTODY.clear()
    (pathlib.Path(handle.path) / "notes.txt").write_bytes(b"DONE\r\nWITH CRLF\r\n")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "ready_with_changes", capture
    manifest = _capture_manifest(capture)
    assert manifest["tracked_changed"] == ["notes.txt"]
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "✅ Integrated" in out, out
    assert (skill / "notes.txt").read_bytes() == b"DONE\r\nWITH CRLF\r\n"
    assert payload_content_hash(skill) == manifest["result_content_hash"]
    custody._CUSTODY.clear()


def test_child_added_gitattributes_and_crlf_file_transport_raw(tmp_path, monkeypatch):
    """Sol P1 (reviewer repro b): a CHILD-added `.gitattributes` is ordinary raw
    content and cannot LF-normalize the sibling CRLF file it names."""
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.delegate_integration import payload_content_hash
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    (exec_root / ".gitattributes").write_text("* text eol=lf\n", encoding="utf-8")
    (exec_root / "table.csv").write_bytes(b"a,b\r\n1,2\r\n")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "ready_with_changes", capture
    manifest = _capture_manifest(capture)
    assert set(manifest["tracked_changed"]) == {".gitattributes", "table.csv"}
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "✅ Integrated" in out, out
    assert (skill / "table.csv").read_bytes() == b"a,b\r\n1,2\r\n"
    assert payload_content_hash(skill) == manifest["result_content_hash"]
    custody._CUSTODY.clear()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows has no POSIX executable bit: os.chmod cannot flip 0644->0755, "
           "so the mode-only divergence this test pins cannot exist there")
def test_exec_bit_only_flip_is_typed_unreviewable_metadata_change(tmp_path, monkeypatch):
    """Sol P1 (reviewer repro c): 0644→0755 with identical bytes is invisible to
    the payload review hash — a typed unreviewable_metadata_change refusal, so
    the stale review can never stay falsely authoritative through a success."""
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle = _provisioned(tmp_path, monkeypatch)
    exec_root = pathlib.Path(handle.path)
    os.chmod(exec_root / "plugin.py", 0o755)
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "failed", capture
    manifest = _capture_manifest(capture)
    assert manifest.get("refusal_kind") == "unreviewable_metadata_change"
    assert manifest.get("normalized_mode_paths") == ["plugin.py"]
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, out
    assert "unreviewable_metadata_change" in out, out
    assert not os.access(skill / "plugin.py", os.X_OK)   # live payload untouched
    assert find_execution_snapshot("snapP") is not None  # snapshot preserved
    custody._CUSTODY.clear()


def test_symlink_topology_change_with_equal_loader_hash_is_typed_refusal(
        tmp_path, monkeypatch):
    """Sol P1 (reviewer repro d): retargeting a confined symlink between two
    equal-content files changes the patch but not the loader hash (it reads
    THROUGH links) — a typed unreviewable_metadata_change refusal, no apply."""
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    (skill / "a.txt").write_text("same\n", encoding="utf-8")
    (skill / "b.txt").write_text("same\n", encoding="utf-8")
    os.symlink("a.txt", skill / "link.txt")
    handle = provision_payload_snapshot(
        target_root=skill, task_id="t-payload", snapshot_id="snapP")
    custody._CUSTODY.clear()
    exec_root = pathlib.Path(handle.path)
    (exec_root / "link.txt").unlink()
    os.symlink("b.txt", exec_root / "link.txt")
    entry = _payload_entry(handle, skill)
    capture = _capture_terminal_patch(ctx, entry)
    assert capture["status"] == "failed", capture
    manifest = _capture_manifest(capture)
    assert manifest.get("refusal_kind") == "unreviewable_metadata_change"
    assert manifest.get("tracked_changed") == ["link.txt"]
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out, out
    assert os.readlink(skill / "link.txt") == "a.txt"    # live payload untouched
    custody._CUSTODY.clear()


def test_post_apply_hash_mismatch_yields_no_success_and_ambiguous_state(
        tmp_path, monkeypatch):
    """Sol P1 (reviewer repro e): if the LIVE loader hash after a real apply is
    not the recorded result hash, no success receipt is emitted, nothing is
    disposed (the stale-extension reconcile marker IS still queued — final Sol
    scope P1), and the PENDING apply intent routes the next integrate to
    the existing APPLY_AMBIGUOUS owner-recovery machinery."""
    import ouroboros.tools.delegate_integration as integration
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    assert capture["status"] == "ready_with_changes", capture
    real = integration.payload_content_hash
    calls = {"n": 0}

    def _hash_diverges_after_apply(root):
        calls["n"] += 1     # call 1 = pre-apply CAS check, call 2 = post-apply
        return "0" * 64 if calls["n"] >= 2 else real(root)

    monkeypatch.setattr(integration, "payload_content_hash", _hash_diverges_after_apply)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_APPLY_HASH_MISMATCH" in out, out
    assert "✅" not in out and "No success is claimed" in out
    assert entry.patch_disposed == ""
    assert find_execution_snapshot("snapP") is not None  # forensics preserved
    from ouroboros.extension_reconcile_queue import list_extension_reconcile_requests

    assert list_extension_reconcile_requests(tmp_path / "data") != []
    monkeypatch.setattr(integration, "payload_content_hash", real)
    again = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_DELEGATED_APPLY_AMBIGUOUS" in again, again
    custody._CUSTODY.clear()


def test_apply_hash_mismatch_queues_stale_extension_reconcile_marker(
        tmp_path, monkeypatch):
    """Final Sol scope P1 («Reconcile stale extensions after apply-hash
    mismatch»): the mismatch branch used to return before any reconcile
    queueing, so a stale enabled extension's subscriptions/companions stayed
    live although the payload DID mutate. The marker must be queued WITHOUT
    any success/disposition record, APPLY_AMBIGUOUS routing and the forensic
    material (snapshot + captured patch + verdict) must be preserved."""
    import ouroboros.tools.delegate_integration as integration
    from ouroboros.extension_reconcile_queue import list_extension_reconcile_requests
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    assert capture["status"] == "ready_with_changes", capture
    real = integration.payload_content_hash
    calls = {"n": 0}

    def _hash_diverges_after_apply(root):
        calls["n"] += 1     # call 1 = pre-apply CAS check, call 2 = post-apply
        return "0" * 64 if calls["n"] >= 2 else real(root)

    monkeypatch.setattr(integration, "payload_content_hash", _hash_diverges_after_apply)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_APPLY_HASH_MISMATCH" in out, out
    # The reconcile marker IS set, typed with the mismatch-specific reason.
    requests = list_extension_reconcile_requests(tmp_path / "data")
    assert [r["skill"] for r in requests] == ["alpha"], requests
    assert requests[0]["reason"] == "delegated_payload_apply_hash_mismatch"
    # The receipt reports the QUEUED marker and never a completed reconcile.
    assert "QUEUED" in out and "reconciled off" not in out, out
    # No success/disposition is recorded; forensic custody is intact.
    assert "No success is claimed" in out and "✅" not in out
    assert entry.patch_disposed == ""
    assert find_execution_snapshot("snapP") is not None
    assert pathlib.Path(capture["patch_artifact"]).exists()
    assert "Verdict:" in out, out
    # The durable apply intent stays PENDING: APPLY_AMBIGUOUS answers next.
    monkeypatch.setattr(integration, "payload_content_hash", real)
    again = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_DELEGATED_APPLY_AMBIGUOUS" in again, again
    custody._CUSTODY.clear()


def test_mismatch_reconcile_queue_failure_keeps_honest_ambiguity(
        tmp_path, monkeypatch):
    """A reconcile queue-write failure in the mismatch branch must not fake the
    marker or a success: the receipt reports the failed queueing, nothing is
    disposed, and the PENDING intent still routes to APPLY_AMBIGUOUS."""
    import ouroboros.extension_reconcile_queue as reconcile_queue
    import ouroboros.tools.delegate_integration as integration
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    real = integration.payload_content_hash
    calls = {"n": 0}

    def _hash_diverges_after_apply(root):
        calls["n"] += 1
        return "0" * 64 if calls["n"] >= 2 else real(root)

    def _boom(*_a, **_k):
        raise OSError("queue disk full")

    monkeypatch.setattr(integration, "payload_content_hash", _hash_diverges_after_apply)
    monkeypatch.setattr(reconcile_queue, "request_extension_reconcile", _boom)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_APPLY_HASH_MISMATCH" in out, out
    assert "could NOT be queued" in out and "✅" not in out, out
    assert entry.patch_disposed == ""
    assert find_execution_snapshot("snapP") is not None
    custody._CUSTODY.clear()
