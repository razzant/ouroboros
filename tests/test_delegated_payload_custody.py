"""Payload custody: no-repository apply, the payload lock, and orphan disposition.

Sibling of ``tests/test_delegated_skill_payload.py`` (which owns the capability
itself). This module covers the custody obligations around it: that a Git
repository living ABOVE the runtime data root cannot make ``git apply`` skip
every hunk in silence, that a provable non-mutation is refused typed instead of
reported as a success, that a settled run whose OWNER TASK is terminal stops
holding its skill payload hostage forever, and that such an orphan may be
disposed by a live top-level task holding the same target.
"""

from __future__ import annotations

import json
import pathlib
import subprocess

from ouroboros import delegate_custody as custody
from tests.test_delegated_skill_payload import (  # noqa: F401 - shared fixtures
    _captured,
    _exact_payload_start,
    _payload_ctx,
    _payload_entry,
    _provisioned,
    _seed_skill,
    _start_payload_run,
    _terminal_wait,
    _owned_gateway_uses_each_test_transport,
)


def _ancestor_repo(tmp_path: pathlib.Path) -> None:
    """A Git worktree ABOVE the runtime data root — the live shape that broke
    the payload apply (the operator's own checkout containing ``data/``)."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)


# -- PC-F01: the payload apply runs in NO-REPOSITORY mode -----------------------


def test_ancestor_git_repo_above_the_payload_cannot_silently_skip_the_apply(
        tmp_path, monkeypatch):
    """With an ancestor ``.git`` above the payload, git treats the payload as a
    subdirectory prefix: every hunk is skipped at rc=0 and ``--numstat`` prints
    nothing. The apply must still land the bytes."""
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    _ancestor_repo(tmp_path)
    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    assert capture["status"] == "ready_with_changes", capture
    patch_text = pathlib.Path(capture["patch_artifact"]).read_text(
        encoding="utf-8", errors="replace")
    # A GIT-FORMAT patch is what makes this fixture load-bearing: a plain
    # unified diff applies fine under an ancestor repo and would pin nothing.
    assert "diff --git" in patch_text, patch_text[:400]
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "✅ Integrated" in out, out
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "DONE\n"
    assert not (skill / ".git").exists()
    assert entry.patch_disposed == "applied"
    custody._CUSTODY.clear()


def test_touched_path_reader_needs_the_ceiling_at_the_payload_parent(
        tmp_path, monkeypatch):
    """The empirical core of the fix, asserted directly on the numstat reader:
    blind without a ceiling, still blind with the ceiling pinned AT the payload,
    correct only with the ceiling at the payload's PARENT."""
    from ouroboros.subagent_worktrees import isolated_git_env
    from ouroboros.tools.subagent_integration import _patch_touched_paths

    _ancestor_repo(tmp_path)
    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    patch_path = pathlib.Path(capture["patch_artifact"])

    blind, blind_err = _patch_touched_paths(patch_path, skill, env=isolated_git_env())
    assert blind == set() and not blind_err, (blind, blind_err)

    at_payload, _ = _patch_touched_paths(patch_path, skill, env={
        **isolated_git_env(), "GIT_CEILING_DIRECTORIES": str(skill.resolve())})
    assert at_payload == set(), at_payload

    at_parent, parent_err = _patch_touched_paths(patch_path, skill, env={
        **isolated_git_env(),
        "GIT_CEILING_DIRECTORIES": str(skill.resolve().parent)})
    assert not parent_err and "notes.txt" in at_parent, (at_parent, parent_err)
    custody._CUSTODY.clear()


def test_a_no_op_apply_is_refused_typed_even_with_no_recorded_result_hash(
        tmp_path, monkeypatch):
    """The guard sits BEFORE the result-hash conditional: a manifest that never
    recorded a result content hash used to route a provable non-mutation straight
    into the success finalizer."""
    import ouroboros.tools.delegate_integration as integration
    from ouroboros.extension_reconcile_queue import list_extension_reconcile_requests
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    manifest_path = pathlib.Path(capture["manifest_artifact"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("result_content_hash", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    real = integration.payload_content_hash
    calls = {"n": 0}

    def _baseline_after_apply(root):
        calls["n"] += 1     # call 1 = pre-apply CAS check, call 2 = post-apply
        return handle.payload_hash if calls["n"] >= 2 else real(root)

    monkeypatch.setattr(integration, "payload_content_hash", _baseline_after_apply)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_APPLY_NO_OP" in out, out
    assert "✅" not in out and "provable non-mutation" in out
    # PC-F09: the consequence and the closing move are named at the point of need.
    assert "decision='reject'" in out
    assert "Done with warnings" in out and "delegated_custody_unreconciled" in out
    # Nothing disposed, no reconcile queued, and the apply intent is RESOLVED.
    assert entry.patch_disposed == ""
    assert entry.patch_apply_pending is False
    assert list_extension_reconcile_requests(tmp_path / "data") == []
    custody._CUSTODY.clear()


def test_no_op_refusal_keeps_the_retry_lane_open(tmp_path, monkeypatch):
    """The resolved intent must not route the next attempt into APPLY_AMBIGUOUS:
    once the payload really mutates, the same run integrates normally."""
    import ouroboros.tools.delegate_integration as integration
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    real = integration.payload_content_hash
    calls = {"n": 0}

    def _baseline_after_apply(root):
        calls["n"] += 1
        return handle.payload_hash if calls["n"] >= 2 else real(root)

    monkeypatch.setattr(integration, "payload_content_hash", _baseline_after_apply)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_APPLY_NO_OP" in out, out
    # The first attempt really did apply the bytes, so restore the payload before
    # retrying under the real hash function.
    (skill / "notes.txt").write_text("PENDING\n", encoding="utf-8")
    (skill / "extra.txt").unlink()
    monkeypatch.setattr(integration, "payload_content_hash", real)
    again = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "APPLY_AMBIGUOUS" not in again, again
    assert "✅ Integrated" in again, again
    custody._CUSTODY.clear()


def test_apply_hash_mismatch_receipt_names_a_followable_recovery(
        tmp_path, monkeypatch):
    """The mismatch receipt used to prescribe ``decision='acknowledge_ambiguous'``
    — a value absent from the decision enum. The real flag is the boolean."""
    import ouroboros.tools.delegate_integration as integration
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    real = integration.payload_content_hash
    calls = {"n": 0}

    def _hash_diverges_after_apply(root):
        calls["n"] += 1
        return "0" * 64 if calls["n"] >= 2 else real(root)

    monkeypatch.setattr(integration, "payload_content_hash", _hash_diverges_after_apply)
    out = _integrate_delegated_patch(ctx, "run-p1", "apply", "")
    assert "INTEGRATE_APPLY_HASH_MISMATCH" in out, out
    assert "acknowledge_ambiguous=true after inspection" in out, out
    assert "decision='acknowledge_ambiguous'" not in out, out
    assert "decision='reject'" in out, out
    assert "Done with warnings" in out and "delegated_custody_unreconciled" in out
    custody._CUSTODY.clear()


# -- PC-F09: the consequence is discoverable from the tool schema ---------------


def test_integrate_schema_states_the_finalization_consequence_and_the_reject_exit():
    from ouroboros.tools.subagent_integration import get_tools

    entry = next(e for e in get_tools() if e.name == "integrate_delegated_patch")
    description = str(entry.schema["description"])
    assert "Done with warnings" in description
    assert "delegated_custody_unreconciled" in description
    assert "reject is the closing move" in description
    assert "terminal owner's orphan" in description
    assert "Applying requires the caller's active Git root or fresh payload binding" in description
    assert "Rejecting a terminal-owner orphan requires only the owner's terminality" in description
    assert "release a dead task's locks and snapshot" in description
    assert entry.schema["parameters"]["properties"]["decision"]["enum"] == [
        "apply", "reject"]


# -- PC-F11E: a terminal owner stops holding the payload hostage ----------------


def _other_task_ctx(tmp_path, monkeypatch, task_id: str = "t-second"):
    """A SECOND live top-level context on the same drive. It must carry a
    different task_id: the same actor hits the per-actor
    ``replacement_requires_settlement`` gate before the busy check runs."""
    ctx = _payload_ctx(tmp_path, monkeypatch)
    ctx.task_id = task_id
    ctx.task_metadata = {"root_task_id": task_id}
    return ctx


def _held_payload(tmp_path, monkeypatch):
    """A settled, UNDISPOSED payload run owned by ``t-payload``."""
    ctx = _payload_ctx(tmp_path, monkeypatch)
    skill = _seed_skill(tmp_path / "data")
    payload, _ = _start_payload_run(ctx, monkeypatch)
    assert payload["status"] == "started", payload
    waited = _terminal_wait(ctx, monkeypatch)
    assert waited.get("state") == "succeeded", waited
    return ctx, skill


def test_settled_run_of_a_terminal_owner_releases_the_payload(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    ctx, skill = _held_payload(tmp_path, monkeypatch)
    data = tmp_path / "data"
    second = _other_task_ctx(tmp_path, monkeypatch)

    # While the owner is LIVE the lock holds and the refusal names the owner.
    write_task_result(data, "t-payload", "running")
    refused = json.loads(_exact_payload_start(
        second, "second", root="skill_payload", bucket="external",
        skill_name="alpha"))
    assert refused["reason"] == "payload_delegation_busy", refused
    assert refused["holder_owner_task_id"] == "t-payload", refused
    assert "owner task is still live" in refused["detail"]
    assert "delegate_wait it and integrate_delegated_patch its capture" \
        not in refused["detail"]

    # Once the owner task is terminal the payload is free again.
    write_task_result(data, "t-payload", STATUS_FAILED)
    started, _ = _start_payload_run(second, monkeypatch)
    assert started["status"] == "started", started
    custody._CUSTODY.clear()


def test_pending_invocation_of_a_terminal_owner_releases_the_payload(
        tmp_path, monkeypatch):
    """The second projection: a worker death between the accepted POST and the
    STARTED row leaves only a request row, which had NO liveness axis at all."""
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    _payload_ctx(tmp_path, monkeypatch)     # installs the runtime env for this tmp
    skill = _seed_skill(tmp_path / "data")
    data = tmp_path / "data"
    assert custody.emit(data, custody.START_REQUESTED, {
        "invocation_id": "inv-dead", "task_id": "t-dead",
        "authority_source": "skill_payload", "target_root": str(skill.resolve()),
        "request": {"mode": "agent"},
    })
    second = _other_task_ctx(tmp_path, monkeypatch)

    write_task_result(data, "t-dead", "running")
    refused = json.loads(_exact_payload_start(
        second, "second", root="skill_payload", bucket="external",
        skill_name="alpha"))
    assert refused["reason"] == "payload_delegation_busy", refused
    assert refused["holder"] == "inv-dead", refused
    # A pending holder has no replayed run row, so the owner must come off the
    # invocation record — `custody.lookup` alone answers UNKNOWN here.
    assert refused["holder_owner_task_id"] == "t-dead", refused

    write_task_result(data, "t-dead", STATUS_FAILED)
    started, _ = _start_payload_run(second, monkeypatch)
    assert started["status"] == "started", started
    custody._CUSTODY.clear()


def test_unprovable_owner_terminality_keeps_the_payload_locked(tmp_path, monkeypatch):
    """Fail-closed: no task_result row at all means unknown, and unknown keeps
    the lock. There is deliberately no time-based release."""
    ctx, skill = _held_payload(tmp_path, monkeypatch)
    second = _other_task_ctx(tmp_path, monkeypatch)
    refused = json.loads(_exact_payload_start(
        second, "second", root="skill_payload", bucket="external",
        skill_name="alpha"))
    assert refused["reason"] == "payload_delegation_busy", refused
    assert "terminality cannot be proven" in refused["detail"]
    custody._CUSTODY.clear()


# -- PC-F11B: a terminal owner's orphan is disposable by a live top-level task --


def _payload_orphan(tmp_path, monkeypatch):
    """A captured, settled, UNDISPOSED payload patch owned by a task that is
    already terminal, plus a SECOND live top-level task on the same drive."""
    from ouroboros.task_results import STATUS_FAILED, write_task_result

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    assert capture["status"] == "ready_with_changes", capture
    write_task_result(tmp_path / "data", "t-payload", STATUS_FAILED)
    return _other_task_ctx(tmp_path, monkeypatch), skill, entry, capture


def _disposed_rows(tmp_path):
    rows = list(custody._iter_rows(custody.event_log_path(tmp_path / "data")))
    return [r for r in rows if str(r.get("type") or "") == custody.PATCH_DISPOSED]


def test_top_level_task_may_reject_a_terminal_owners_payload_orphan(
        tmp_path, monkeypatch):
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    second, skill, entry, capture = _payload_orphan(tmp_path, monkeypatch)
    out = _integrate_delegated_patch(second, "run-p1", "reject", "not wanted")
    assert "🚫 Rejected" in out, out
    assert "orphan of terminal task t-payload" in out, out
    assert entry.patch_disposed == "rejected"
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n"
    assert find_execution_snapshot("snapP") is None
    rows = _disposed_rows(tmp_path)
    assert [r["disposed_by_task_id"] for r in rows] == ["t-second"], rows
    assert [r["task_id"] for r in rows] == ["t-payload"], rows
    custody._CUSTODY.clear()


def test_top_level_task_may_apply_a_terminal_owners_payload_orphan(
        tmp_path, monkeypatch):
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    second, skill, entry, capture = _payload_orphan(tmp_path, monkeypatch)
    out = _integrate_delegated_patch(second, "run-p1", "apply", "looks good")
    assert "✅ Integrated" in out, out
    assert "orphan of terminal task t-payload" in out, out
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "DONE\n"
    assert entry.patch_disposed == "applied"
    rows = _disposed_rows(tmp_path)
    assert [r["disposed_by_task_id"] for r in rows] == ["t-second"], rows
    custody._CUSTODY.clear()


def test_executor_classifies_an_orphan_payload_no_op_as_a_failure(
        tmp_path, monkeypatch):
    from types import SimpleNamespace

    import ouroboros.tools.delegate_integration as integration
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.task_results import STATUS_FAILED, write_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    owner, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    assert capture["status"] == "ready_with_changes", capture
    write_task_result(tmp_path / "data", "t-payload", STATUS_FAILED)
    second = _other_task_ctx(tmp_path, monkeypatch)
    real = integration.payload_content_hash
    calls = {"n": 0}

    def _baseline_after_apply(root):
        calls["n"] += 1
        return handle.payload_hash if calls["n"] >= 2 else real(root)

    monkeypatch.setattr(integration, "payload_content_hash", _baseline_after_apply)
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    # v7 typed dispatch: the executor consumes ``execute_result`` (a ToolResult);
    # the text producer stays the upstream one, adapted through the legacy seam.
    tools = SimpleNamespace(
        CODE_TOOLS=set(),
        _ctx=second,
        execute_result=lambda name, args: LegacyTextResultAdapter.from_text(
            name, _integrate_delegated_patch(second, args["run_id"], args["decision"], "")),
    )
    logs = tmp_path / "executor-logs"
    logs.mkdir()
    executed = _execute_single_tool(
        tools,
        {"id": "orphan-no-op", "function": {"name": "integrate_delegated_patch",
                                               "arguments": json.dumps({
                                                   "run_id": "run-p1", "decision": "apply"})}},
        logs,
        task_id="t-second",
    )

    assert executed["result"].splitlines()[0].startswith("⚠️ INTEGRATE_APPLY_NO_OP")
    assert "orphan of terminal task t-payload" in executed["result"]
    assert executed["is_error"] is True
    assert executed["result_meta"]["status"] == "integration_blocked"
    assert entry.patch_disposed == ""
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "DONE\n"
    custody._CUSTODY.clear()


def test_a_live_owners_run_is_still_not_owned_by_another_task(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, skill, handle, entry, capture = _captured(tmp_path, monkeypatch)
    write_task_result(tmp_path / "data", "t-payload", "running")
    second = _other_task_ctx(tmp_path, monkeypatch)
    out = _integrate_delegated_patch(second, "run-p1", "reject", "")
    assert "INTEGRATE_DELEGATED_NOT_OWNED" in out, out
    assert "once the owner is terminal" in out, out
    assert entry.patch_disposed == ""
    assert (skill / "notes.txt").read_text(encoding="utf-8") == "PENDING\n"
    custody._CUSTODY.clear()


def test_non_top_level_profiles_may_not_dispose_an_orphan(tmp_path, monkeypatch):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    second, skill, entry, capture = _payload_orphan(tmp_path, monkeypatch)
    for constraint, direct_chat in (
            (TaskConstraint(mode="skill_repair"), False),
            (TaskConstraint(mode="acting_subagent", surface="worktree"), False),
            (None, True),                       # direct chat = operator_control
    ):
        second.task_constraint = constraint
        second.is_direct_chat = direct_chat
        out = _integrate_delegated_patch(second, "run-p1", "reject", "")
        assert "INTEGRATE_DELEGATED_NOT_OWNED" in out, (constraint, out)
        assert entry.patch_disposed == ""
    custody._CUSTODY.clear()


def test_wait_and_cancel_authority_did_not_widen_for_an_orphan(tmp_path, monkeypatch):
    """`_owned_run` governs wait/cancel/answer and is deliberately NOT widened:
    cancelling or answering a foreign run destroys work instead of closing an
    obligation."""
    import ouroboros.tools.delegate as delegate

    second, skill, entry, capture = _payload_orphan(tmp_path, monkeypatch)
    waited = json.loads(delegate._delegate_wait(second, "run-p1", wait_sec=1))
    assert waited["reason"] == "run_not_owned", waited
    cancelled = json.loads(delegate._delegate_cancel(second, "run-p1", "stop"))
    assert cancelled["reason"] == "run_not_owned", cancelled
    assert entry.patch_disposed == ""
    custody._CUSTODY.clear()


def test_git_lane_orphan_is_disposable_by_a_top_level_task_on_the_same_root(
        tmp_path, monkeypatch):
    """Owner decision 1=A carries no lane qualifier: the Git lane behaves the
    same way, and its own guards (recorded target == active root, protected
    paths, proven drift) are unchanged."""
    from ouroboros.task_results import STATUS_FAILED, write_task_result
    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_delegated_run_isolation import _git, _isolated_entry, _nanny_ctx, _seed_target

    target = _seed_target(tmp_path)
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(
        target_root=target, task_id="t-nanny", snapshot_id="snapG")
    custody._CUSTODY.clear()
    (pathlib.Path(handle.path) / "newfile.py").write_text("print('hi')\n", encoding="utf-8")
    entry = _isolated_entry(ctx, target, handle)
    assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"

    drive = custody.custody_root(ctx)
    write_task_result(drive, "t-nanny", STATUS_FAILED)
    second = _nanny_ctx(tmp_path, target, monkeypatch)
    second.task_id = "t-second"

    out = _integrate_delegated_patch(second, "run-1", "apply", "adopted")
    assert "✅ Integrated" in out, out
    assert "orphan of terminal task t-nanny" in out, out
    assert (target / "newfile.py").read_text(encoding="utf-8") == "print('hi')\n"
    assert "newfile.py" in _git(target, "diff", "--cached", "--name-only").stdout
    assert entry.patch_disposed == "applied"
    rows = [r for r in custody._iter_rows(custody.event_log_path(drive))
            if str(r.get("type") or "") == custody.PATCH_DISPOSED]
    assert [r["disposed_by_task_id"] for r in rows] == ["t-second"], rows
    custody._CUSTODY.clear()


def test_git_lane_orphan_from_a_different_root_cannot_apply_but_may_reject(
        tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_FAILED, write_task_result
    from ouroboros.subagent_worktrees import provision_execution_snapshot
    from ouroboros.tools.delegate import _capture_terminal_patch
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch
    from tests.test_delegated_run_isolation import _isolated_entry, _nanny_ctx, _seed_target

    target = _seed_target(tmp_path)
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(
        target_root=target, task_id="t-nanny", snapshot_id="snapG")
    custody._CUSTODY.clear()
    (pathlib.Path(handle.path) / "newfile.py").write_text("print('hi')\n", encoding="utf-8")
    entry = _isolated_entry(ctx, target, handle)
    assert _capture_terminal_patch(ctx, entry)["status"] == "ready_with_changes"
    write_task_result(custody.custody_root(ctx), "t-nanny", STATUS_FAILED)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    second = _nanny_ctx(tmp_path, target, monkeypatch)
    second.task_id = "t-second"
    second.workspace_root = str(elsewhere)
    out = _integrate_delegated_patch(second, "run-1", "apply", "")
    assert "INTEGRATE_DELEGATED_TARGET_MISMATCH" in out, out
    assert not (target / "newfile.py").exists()
    assert entry.patch_disposed == ""

    rejected = _integrate_delegated_patch(second, "run-1", "reject", "release orphan")
    assert "🚫 Rejected" in rejected, rejected
    assert "orphan of terminal task t-nanny" in rejected, rejected
    assert entry.patch_disposed == "rejected"
    rows = [row for row in custody._iter_rows(
        custody.event_log_path(custody.custody_root(ctx)))
        if str(row.get("type") or "") == custody.PATCH_DISPOSED]
    assert [row["disposed_by_task_id"] for row in rows] == ["t-second"], rows
    custody._CUSTODY.clear()
