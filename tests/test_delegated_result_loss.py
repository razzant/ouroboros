"""Disposing a delegated Git writer whose result may be gone.

When the private copy is gone and nothing else holds the run's changes, an ordinary
reject closes the obligation, records the result as lost and applies nothing; apply
names that reject as the exit. Anything that still holds the result (a saved capture,
a result pin, a commit or staged change Git still keeps for the deleted checkout) is
recovered or kept, and an unreadable or malformed carrier proves nothing. Whoever
recorded a decision reaches current custody and its readers without a chain replay.

Every case drives the public handler over real Git copies, pins and capture files.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shutil
import subprocess

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import store
from ouroboros.subagent_worktrees import (
    find_execution_snapshot,
    provision_execution_snapshot,
    release_snapshot_checkout,
    remove_execution_snapshot,
)
from ouroboros.task_results import STATUS_FAILED, load_task_result, write_task_result
from ouroboros.tools.subagent_integration import _integrate_delegated_patch
from tests.test_delegated_run_isolation import _git, _HealthEnv, _nanny_ctx, _seed_target

RID, SNAP, OWNER = "run-lost", "snapLost", "t-nanny"
RESULT_REF = f"refs/ouroboros/delegated-result/{SNAP}"
BASELINE_REF = f"refs/ouroboros/delegated/{SNAP}"


def _writer(tmp_path, monkeypatch, *, settled=True):
    """A finished writer whose child edited only its private copy (real durable rows)."""
    target = _seed_target(tmp_path)
    ctx = _nanny_ctx(tmp_path, target, monkeypatch)
    handle = provision_execution_snapshot(target_root=target, task_id=OWNER, snapshot_id=SNAP)
    (pathlib.Path(handle.path) / "tracked.txt").write_text("one\ntwo\nCHILD-EDIT\n", encoding="utf-8")
    drive = custody.custody_root(ctx)
    assert custody.record_started(drive, custody.RunCustody(
        run_id=RID, task_id=OWNER, route_id="some-route", snapshot_id=SNAP, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=str(target), authority_source="external_workspace_root"))
    if settled:
        assert custody.emit(drive, custody.SETTLED, {"run_id": RID, "task_id": OWNER, "state": "succeeded"})
    custody._CUSTODY.clear()
    return target, ctx, handle, drive


def _saved(drive, handle):
    """Capture through the ordinary core, then release the checkout as a saved result does."""
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    block = capture_terminal_patch_for_drive(drive, custody.replay(drive)[RID])
    assert block["status"] == "ready_with_changes", block
    if pathlib.Path(handle.path).exists():
        assert release_snapshot_checkout(SNAP)
    assert not pathlib.Path(handle.path).exists() and find_execution_snapshot(SNAP)["result_sha"]
    custody._CUSTODY.clear()
    return custody.delegated_capture_dir(drive, OWNER, SNAP)


def _rows(drive, kind):
    return [row for row in custody._iter_rows(custody.event_log_path(drive)) if row.get("type") == kind]


def _state(target):
    """The target's working files, index and HEAD: what an applied nothing must leave alone."""
    files = {p.relative_to(target).as_posix(): p.read_bytes() for p in sorted(target.rglob("*"))
             if p.is_file() and ".git" not in p.relative_to(target).parts}
    return (files, _git(target, "status", "--porcelain").stdout, _git(target, "rev-parse", "HEAD").stdout)


def _ref(target, ref):
    return _git(target, "for-each-ref", "--format=%(objectname)", ref).stdout.strip()


def _verdict(out):
    match = re.search(r"Verdict: (\S+?\.json)", out)
    assert match, out
    return json.loads(pathlib.Path(match.group(1)).read_text(encoding="utf-8"))


def _ways_on(out):
    """Each run_command argv the answer names, decoded as its tool call carries it."""
    found, at = [], out.find("run_command(cmd=")
    while at >= 0:
        argv, end = json.JSONDecoder().raw_decode(out, at + len("run_command(cmd="))
        found.append(argv)
        at = out.find("run_command(cmd=", end)
    return found


def _untouched(drive, target, before, *, owed=True):
    assert _rows(drive, custody.PATCH_DISPOSED) == []
    assert custody.replay(drive)[RID].patch_disposed == ""
    assert [row.run_id for row in custody.undisposed_patches(drive)] == ([RID] if owed else [])
    assert _state(target) == before
    custody._CUSTODY.clear()


@pytest.mark.parametrize("record", ["kept", "removed"])
def test_apply_names_the_reject_exit_and_reject_records_the_loss(tmp_path, monkeypatch, record):
    """The copy is gone by hand (its record kept) or with its record and pins (an older
    collector), and nothing saved the child's edit: apply says reject records the loss;
    reject records an ordinary rejection that applied nothing and frees the owner's debt."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    if record == "kept":
        shutil.rmtree(handle.path)
        assert find_execution_snapshot(SNAP) is not None and _ref(target, BASELINE_REF)
    else:
        assert remove_execution_snapshot(SNAP) and not _ref(target, BASELINE_REF)
    write_task_result(drive, OWNER, STATUS_FAILED, delegated_runs_unreconciled=[f"patch:{RID}"])
    before = _state(target)

    applied = _integrate_delegated_patch(ctx, RID, "apply", "")
    assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in applied and "the result is lost" in applied, applied
    assert "decision='reject') records the loss" in applied and "retry this call" not in applied, applied
    _untouched(drive, target, before)

    out = _integrate_delegated_patch(ctx, RID, "reject", "copy deleted by hand")
    assert "🚫 Rejected delegated run run-lost as LOST" in out and "changed no files" in out, out
    assert "unknown where unknown" in out and "copy deleted by hand" in out, out
    [row] = _rows(drive, custody.PATCH_DISPOSED)
    assert (row["disposition"], row["task_id"], row["disposed_by_task_id"]) == ("rejected", OWNER, OWNER)
    assert row["result_lost"] == {"execution_root": handle.path, "saved_capture": "none",
                                  "copy_record": "without_result" if record == "kept" else "absent"}
    verdict = _verdict(out)
    assert (verdict["outcome"], verdict["applied"], verdict["files"]) == ("result_lost", False, [])
    entry = custody.replay(drive)[RID]
    assert (entry.patch_disposed, entry.patch_disposed_by, entry.patch_result_lost) == ("rejected", OWNER, True)
    assert entry.patch_captured is False  # never set for display convenience
    assert custody.undisposed_patches(drive) == [] and SNAP not in custody.open_snapshot_ids(drive)
    assert f"patch:{RID}" not in load_task_result(drive, OWNER)["delegated_runs_unreconciled"]
    from ouroboros.context_health import build_health_invariants
    assert "DELEGATED PATCH AWAITS DISPOSITION" not in build_health_invariants(_HealthEnv(drive))
    assert _state(target) == before
    assert find_execution_snapshot(SNAP) is None and not _ref(target, BASELINE_REF)

    custody._CUSTODY.clear()
    again = _integrate_delegated_patch(ctx, RID, "reject", "")
    assert "INTEGRATE_DELEGATED_ALREADY_DISPOSED" in again and "rejected" in again, again
    assert len(_rows(drive, custody.PATCH_DISPOSED)) == 1 and _state(target) == before
    custody._CUSTODY.clear()


def test_a_change_the_child_staged_in_its_gone_copy_is_not_loss(tmp_path, monkeypatch):
    """The same edit as the true-loss case above, but staged: Git's entry for the deleted
    checkout still holds it in its index, so nothing is called lost or pruned."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    _git(handle.path, "add", "tracked.txt")
    admin = pathlib.Path(_git(handle.path, "rev-parse", "--absolute-git-dir").stdout.strip())
    shutil.rmtree(handle.path)
    assert _git(target, "--git-dir", str(admin), "diff-index", "--cached", handle.baseline_sha, check=False).stdout
    before, index = _state(target), (admin / "index").read_bytes()
    for decision in ("reject", "apply"):
        out = _integrate_delegated_patch(ctx, RID, decision, "")
        assert "loss is not proven" in out and "still stages changes against its baseline" in out, out
        assert "LOST" not in out, out
        custody._CUSTODY.clear()
    _untouched(drive, target, before)
    assert (admin / "index").read_bytes() == index  # the staged material is kept as it was


def test_an_earlier_failed_capture_is_no_saved_result(tmp_path, monkeypatch):
    """A readable failed manifest proves only that an earlier capture produced nothing usable."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    cap = custody.delegated_capture_dir(drive, OWNER, SNAP)
    cap.mkdir(parents=True)
    (cap / "workspace_patch.json").write_text(json.dumps({"status": "failed", "errors": [
        {"type": "result_commit_failed", "message": "diff exploded"}]}), encoding="utf-8")
    shutil.rmtree(handle.path)
    out = _integrate_delegated_patch(ctx, RID, "reject", "")
    assert "as LOST" in out, out
    assert _rows(drive, custody.PATCH_DISPOSED)[0]["result_lost"]["saved_capture"] == "failed"
    custody._CUSTODY.clear()


def test_a_released_saved_capture_is_an_ordinary_reject(tmp_path, monkeypatch):
    """An absent checkout alone is not loss: the saved capture is the result and survives the reject."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    cap = _saved(drive, handle)
    out = _integrate_delegated_patch(ctx, RID, "reject", "not wanted")
    assert "🚫 Rejected delegated run run-lost's captured patch" in out and "LOST" not in out, out
    [row] = _rows(drive, custody.PATCH_DISPOSED)
    assert "result_lost" not in row and row["disposed_by_task_id"] == OWNER
    assert custody.replay(drive)[RID].patch_result_lost is False
    assert "CHILD-EDIT" in (cap / "workspace.patch").read_text(encoding="utf-8")  # material retained
    custody._CUSTODY.clear()


def test_a_pinned_result_without_its_capture_is_recovered_before_any_decision(tmp_path, monkeypatch):
    """Capture pinned commit C, then its saved files went: the capture owner restores the
    checkout from the pin (as a continuation does), so apply shows it for inspection first
    and nothing is ever called lost."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    cap = _saved(drive, handle)
    pinned = find_execution_snapshot(SNAP)["result_sha"]
    shutil.rmtree(cap)
    before = _state(target)

    shown = _integrate_delegated_patch(ctx, RID, "apply", "")
    assert "INTEGRATE_DELEGATED_INSPECT_CAPTURE" in shown, shown
    assert "CHILD-EDIT" in (cap / "workspace.patch").read_text(encoding="utf-8")
    assert _state(target) == before and _rows(drive, custody.PATCH_DISPOSED) == []
    custody._CUSTODY.clear()
    out = _integrate_delegated_patch(ctx, RID, "apply", "inspected")
    assert "✅ Integrated" in out, out
    assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\nCHILD-EDIT\n"
    [row] = _rows(drive, custody.PATCH_DISPOSED)
    assert row["disposition"] == "applied" and "result_lost" not in row and pinned
    custody._CUSTODY.clear()


def test_a_surviving_result_pin_without_a_record_is_not_loss(tmp_path, monkeypatch):
    """The record was deleted by hand but Git still pins C: nothing is released or called lost."""
    from ouroboros.subagent_worktrees import _record_file

    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    cap = _saved(drive, handle)
    shutil.rmtree(cap)
    _record_file(handle.path).unlink()
    before = _state(target)
    out = _integrate_delegated_patch(ctx, RID, "reject", "")
    assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out and "loss is not proven" in out, out
    assert RESULT_REF in out and "LOST" not in out, out
    _untouched(drive, target, before)
    assert _ref(target, RESULT_REF)


def test_a_commit_the_child_made_in_its_gone_copy_is_not_loss(tmp_path, monkeypatch):
    """Git keeps a hand-deleted checkout's entry, and its HEAD still reaches what the child committed."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    _git(handle.path, "-c", "user.email=c@c", "-c", "user.name=child", "commit", "-qam", "child commit")
    child = _git(handle.path, "rev-parse", "HEAD").stdout.strip()
    shutil.rmtree(handle.path)
    before = _state(target)
    for decision in ("reject", "apply"):
        out = _integrate_delegated_patch(ctx, RID, decision, "")
        assert "loss is not proven" in out and child[:12] in out and "LOST" not in out, out
        custody._CUSTODY.clear()
    _untouched(drive, target, before)
    assert child in _git(target, "worktree", "list", "--porcelain").stdout  # nothing pruned it


def test_a_continuation_never_takes_its_predecessors_pin_as_its_own_result(tmp_path, monkeypatch):
    """A same-tree successor shares its predecessor's copy record, whose pin is the
    predecessor's capture: its own gone checkout is neither restored from it nor lost."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    _saved(drive, handle)
    assert custody.record_started(drive, custody.RunCustody(
        run_id="run-next", task_id=OWNER, route_id="some-route", snapshot_id=SNAP, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=str(target), authority_source="external_workspace_root",
        continuation_of=RID, capture_id="inv-next", snapshot_task_id=OWNER))
    assert custody.emit(drive, custody.SETTLED, {"run_id": "run-next", "task_id": OWNER, "state": "succeeded"})
    custody._CUSTODY.clear()
    assert custody.replay(drive)[RID].superseded_by == "run-next"
    before = _state(target)
    out = _integrate_delegated_patch(ctx, "run-next", "reject", "")
    assert "loss is not proven" in out and "may be its predecessor run-lost's result" in out, out
    assert "LOST" not in out and "fails the same way" in out, out
    assert not pathlib.Path(handle.path).exists()  # the predecessor's result was not checked out as this run's
    assert not (custody.delegated_capture_dir(drive, OWNER, "inv-next") / "workspace_patch.json").exists()
    assert _rows(drive, custody.PATCH_DISPOSED) == [] and _state(target) == before
    custody._CUSTODY.clear()


@pytest.mark.parametrize("carrier, payload", [
    ("manifest", "{not json"), ("manifest", '["ready_with_changes"]'), ("manifest", '"ready_with_changes"'),
    ("manifest", "7"), ("manifest", "null"), ("record", "{not json")])
def test_an_unreadable_carrier_never_proves_loss(tmp_path, monkeypatch, carrier, payload):
    """Corrupt bytes, or well-formed JSON that is no manifest, refuse typed and stay as they are."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    shutil.rmtree(handle.path)
    if carrier == "manifest":
        broken = custody.delegated_capture_dir(drive, OWNER, SNAP) / "workspace_patch.json"
        broken.parent.mkdir(parents=True)
    else:  # another copy's record: the scan cannot rule out that it names this run
        from ouroboros.subagent_worktrees import _records_dir
        broken = _records_dir() / "other-copy-0123456789ab.json"
    broken.write_text(payload, encoding="utf-8")
    before = _state(target)
    for decision in ("reject", "apply"):
        out = _integrate_delegated_patch(ctx, RID, decision, "")
        assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out and "loss is not proven" in out, out
        assert "could not" in out and "LOST" not in out, out
        custody._CUSTODY.clear()
    assert broken.read_text(encoding="utf-8") == payload
    _untouched(drive, target, before)


@pytest.mark.parametrize("unreadable", ["worktrees", "index"])
def test_unreadable_git_metadata_cannot_release_a_staged_result(tmp_path, monkeypatch, unreadable):
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    _git(handle.path, "add", "tracked.txt")
    admin = pathlib.Path(_git(handle.path, "rev-parse", "--absolute-git-dir").stdout.strip())
    index = (admin / "index").read_bytes()
    shutil.rmtree(handle.path)
    before = _state(target)
    real_stat = pathlib.Path.stat
    mode = admin.parent.stat().st_mode

    def denied_stat(path, *args, **kwargs):
        if path == admin / "index":
            raise PermissionError(13, "metadata unreadable", str(path))
        return real_stat(path, *args, **kwargs)

    with monkeypatch.context() as limited:
        try:
            if unreadable == "worktrees":
                admin.parent.chmod(0)
                try:
                    list(admin.parent.iterdir())
                except PermissionError:
                    pass
                else:
                    pytest.skip("This filesystem/user does not enforce directory read modes")
            else:
                limited.setattr(pathlib.Path, "stat", denied_stat)
            for decision in ("reject", "apply"):
                out = _integrate_delegated_patch(ctx, RID, decision, "")
                assert "loss is not proven" in out and "could not" in out, out
                assert "LOST" not in out, out
                custody._CUSTODY.clear()
        finally:
            admin.parent.chmod(mode)
    _untouched(drive, target, before)
    assert (admin / "index").read_bytes() == index


def test_loss_stays_behind_terminality_and_apply_ambiguity(tmp_path, monkeypatch):
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch, settled=False)
    shutil.rmtree(handle.path)
    before = _state(target)
    live = _integrate_delegated_patch(ctx, RID, "reject", "")
    assert "INTEGRATE_DELEGATED_NOT_TERMINAL" in live, live
    _untouched(drive, target, before, owed=False)

    assert custody.emit(drive, custody.SETTLED, {"run_id": RID, "task_id": OWNER, "state": "failed"})
    assert custody.record_patch_apply_started(drive, custody.replay(drive)[RID], target_root=str(target))
    custody._CUSTODY.clear()
    ambiguous = _integrate_delegated_patch(ctx, RID, "reject", "")
    assert "INTEGRATE_DELEGATED_APPLY_AMBIGUOUS" in ambiguous, ambiguous
    _untouched(drive, target, before)

    out = _integrate_delegated_patch(ctx, RID, "reject", "inspected the target", acknowledge_ambiguous=True)
    assert "as LOST" in out and "changed no files" in out, out
    assert [row["reason"] for row in _rows(drive, custody.PATCH_APPLY_RESOLVED)] == ["owner_acknowledged"]
    assert custody.replay(drive)[RID].patch_apply_pending is False
    custody._CUSTODY.clear()


def test_a_present_copy_that_cannot_be_captured_is_kept_without_a_blind_retry(tmp_path, monkeypatch):
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)

    def _denied(*_a, **_kw):
        raise PermissionError(13, "Permission denied", str(handle.path))

    monkeypatch.setattr("ouroboros.subagent_worktrees.capture_result_commit", _denied)
    before = _state(target)
    for decision in ("apply", "reject"):
        out = _integrate_delegated_patch(ctx, RID, decision, "")
        assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in out and "Permission denied" in out, out
        assert "fails the same way until that cause changes" in out and "retry this call" not in out, out
        custody._CUSTODY.clear()
    assert pathlib.Path(handle.path, "tracked.txt").read_text(encoding="utf-8").endswith("CHILD-EDIT\n")
    _untouched(drive, target, before)


def test_payload_and_directory_results_keep_their_own_contracts(tmp_path, monkeypatch):
    from ouroboros.tools.delegate_integration import result_loss

    _target, _ctx, handle, drive = _writer(tmp_path, monkeypatch)
    shutil.rmtree(handle.path)
    entry = custody.replay(drive)[RID]
    assert result_loss(drive, entry)[0]["copy_record"] == "without_result"
    for shape in ({"authority_source": "skill_payload"},
                  {"resource_ref": {"workspace_kind": "directory", "strategy": "copy"}}):
        facts, why = result_loss(drive, dataclasses.replace(entry, **shape))
        assert facts == {} and "its own" in why, why
    custody._CUSTODY.clear()


def test_an_orphans_disposer_reaches_current_custody_and_its_readers(tmp_path, monkeypatch):
    """The owner ended; another top-level task rejects its lost orphan. The decision
    stays the owner's row, the verdict the disposer's, and readers name the disposer."""
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.external_runs import episode_disclosure
    from ouroboros.owner_stop_receipt import EVENT_FIELD, FIELD, receipt_text, retained_run_facts, retained_work_fields

    target, _ctx, handle, drive = _writer(tmp_path, monkeypatch)
    shutil.rmtree(handle.path)
    write_task_result(drive, OWNER, STATUS_FAILED)
    disposer = _nanny_ctx(tmp_path, target, monkeypatch)
    disposer.task_id = "t-second"

    out = _integrate_delegated_patch(disposer, RID, "reject", "owner gone, copy gone")
    assert "orphan of terminal task t-nanny" in out and "as LOST" in out, out
    [row] = _rows(drive, custody.PATCH_DISPOSED)
    assert (row["task_id"], row["disposed_by_task_id"]) == (OWNER, "t-second")
    assert [v["task_id"] for v in _rows(drive, "delegate_run_patch_verdict")] == ["t-second"]
    custody._CUSTODY.clear()
    entry = custody.replay(drive)[RID]
    assert entry.patch_disposed_by == "t-second"

    line = episode_disclosure(drive, OWNER, [{"run_id": RID}])
    assert "no patch captured, rejected as lost" in line and "by task t-second" in line, line
    shown = capture_projection(drive, entry, reader_task_id=OWNER)
    assert (shown["status"], shown["disposition"], shown["disposed_by"], shown["result_lost"]) == (
        "missing", "rejected", "t-second", True), shown
    assert "still owed" not in shown["note"] and "t-second" in shown["note"], shown
    row = {"run_id": RID, "state": "succeeded", **retained_run_facts(drive, entry)}
    assert row["patch"]["disposition"] == "rejected" and row["patch"]["capture"]["disposed_by"] == "t-second"
    audit = {FIELD: {"task_id": OWNER, "runs": [row]}}
    owner = retained_work_fields({"delegate_terminal_reconciliation": audit})[EVENT_FIELD]["text"]
    assert "file changes lost: the private copy was gone and nothing was saved, rejected by task t-second" in owner
    assert "captured file changes are missing" not in owner, owner
    model = receipt_text(audit)
    assert "disposition rejected, result lost, disposed by t-second" in model, model
    assert "capture receipt unconfirmed" not in model, model
    custody._CUSTODY.clear()


def test_a_decision_without_a_recorded_disposer_reads_not_recorded_without_replay(tmp_path, monkeypatch):
    """Rows written before the disposer was recorded (or projected) stay honest gaps."""
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.delegate_custody_current import run
    from ouroboros.external_runs import episode_disclosure

    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    _saved(drive, handle)
    assert custody.emit(drive, custody.PATCH_DISPOSED, {"run_id": RID, "task_id": OWNER, "disposition": "rejected"})
    with store.open_store(drive, write=True) as txn:  # a current row a pre-projection reducer wrote
        facts = txn.custody_row(RID)
        facts.pop("patch_disposed_by", None)
        txn.put_custody(RID, facts, state="closed")

    def _replayed(*_a, **_kw):
        raise AssertionError("a reader replayed the custody chain")

    monkeypatch.setattr(custody, "_iter_rows", _replayed)
    custody._CUSTODY.clear()
    entry = run(drive, RID)
    assert entry.patch_disposed == "rejected" and entry.patch_disposed_by == ""
    shown = capture_projection(drive, entry, reader_task_id=OWNER)
    assert shown["disposed_by"] == "not_recorded" and "NOT APPLIED" not in shown["note"], shown
    line = episode_disclosure(drive, OWNER, [{"run_id": RID}])
    assert "patch captured, rejected (disposer not recorded)" in line, line
    custody._CUSTODY.clear()


# -- carriers a loss proof still reads (review rl-20261029T121637) ------------


def test_an_unlistable_copy_records_directory_proves_nothing(tmp_path, monkeypatch):
    """A records directory that cannot be listed is unread, never "no record"; a missing one stays empty."""
    from ouroboros.subagent_worktrees import _records, _records_dir

    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    shutil.rmtree(handle.path)
    assert _records(tmp_path / "no-data", strict=True) == []
    directory, before = _records_dir(), _state(target)
    mode = directory.stat().st_mode
    directory.chmod(0)
    try:
        try:
            os.listdir(directory)
        except PermissionError:
            pass
        else:
            pytest.skip("this filesystem/user does not enforce directory read modes")
        assert _records(op="probe") == []  # the lenient scan still skips what it cannot list
        with pytest.raises(PermissionError):
            _records(op="probe", strict=True)
        for decision in ("reject", "apply"):
            out = _integrate_delegated_patch(ctx, RID, decision, "")
            assert "loss is not proven" in out and "private-copy records could not be read" in out, out
            assert "LOST" not in out, out
            custody._CUSTODY.clear()
    finally:
        directory.chmod(mode)
    _untouched(drive, target, before)
    assert find_execution_snapshot(SNAP)["path"] == handle.path


@pytest.mark.parametrize("held", ["staged", "committed", "nothing"])
def test_a_relative_worktree_gitdir_still_finds_what_git_holds(tmp_path, monkeypatch, held):
    """With worktree.useRelativePaths (Git 2.48+) the entry records its checkout relative to itself:
    a staged or committed change in the gone checkout is still kept, and true loss still rejects."""
    for key, value in (("GIT_CONFIG_COUNT", "1"), ("GIT_CONFIG_KEY_0", "worktree.useRelativePaths"),
                       ("GIT_CONFIG_VALUE_0", "true")):
        monkeypatch.setenv(key, value)
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    admin = pathlib.Path(_git(handle.path, "rev-parse", "--absolute-git-dir").stdout.strip())
    if pathlib.Path((admin / "gitdir").read_text(encoding="utf-8").strip()).is_absolute():
        pytest.skip("this Git records absolute worktree paths")
    if held == "staged":
        _git(handle.path, "add", "tracked.txt")
    elif held == "committed":
        _git(handle.path, "-c", "user.email=c@c", "-c", "user.name=child", "commit", "-qam", "child commit")
    head, index = _git(handle.path, "rev-parse", "HEAD").stdout.strip(), (admin / "index").read_bytes()
    shutil.rmtree(handle.path)
    before = _state(target)
    out = _integrate_delegated_patch(ctx, RID, "reject", "")
    if held == "nothing":
        assert "as LOST" in out and not admin.exists(), out  # its empty entry goes with the reject
    else:
        assert "loss is not proven" in out and "LOST" not in out, out
        assert ("still stages changes" if held == "staged" else head[:12]) in out, out
        _untouched(drive, target, before)
        assert (admin / "index").read_bytes() == index
        assert _git(target, f"--git-dir={admin}", "rev-parse", "HEAD").stdout.strip() == head
    custody._CUSTODY.clear()


def test_a_continuation_restores_its_own_captured_result_never_its_predecessors(tmp_path, monkeypatch):
    """A same-tree successor captured its own C2, then its saved files and its copy went: C2 is
    checked out again, shown, then applied; its predecessor's C1 is never offered in its place."""
    from ouroboros.subagent_worktrees import recreate_snapshot_checkout
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive

    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    _saved(drive, handle)
    c1 = find_execution_snapshot(SNAP)["result_sha"]
    assert custody.record_started(drive, custody.RunCustody(
        run_id="run-next", task_id=OWNER, route_id="some-route", snapshot_id=SNAP, execution_root=handle.path,
        baseline_sha=handle.baseline_sha, target_root=str(target), authority_source="external_workspace_root",
        continuation_of=RID, capture_id="inv-next", snapshot_task_id=OWNER))
    recreate_snapshot_checkout(SNAP)  # the hand-over a continuation's start performs
    (pathlib.Path(handle.path) / "tracked.txt").write_text("one\ntwo\nCHILD-EDIT\nSUCCESSOR-EDIT\n", encoding="utf-8")
    assert custody.emit(drive, custody.SETTLED, {"run_id": "run-next", "task_id": OWNER, "state": "succeeded"})
    custody._CUSTODY.clear()
    block = capture_terminal_patch_for_drive(drive, custody.replay(drive)["run-next"])
    assert block["status"] == "ready_with_changes", block
    c2, cap = find_execution_snapshot(SNAP)["result_sha"], custody.delegated_capture_dir(drive, OWNER, "inv-next")
    shutil.rmtree(cap)
    if pathlib.Path(handle.path).exists():
        shutil.rmtree(handle.path)
    custody._CUSTODY.clear()
    assert c2 != c1 and custody.replay(drive)["run-next"].patch_captured
    before = _state(target)

    shown = _integrate_delegated_patch(ctx, "run-next", "apply", "")
    assert "INTEGRATE_DELEGATED_INSPECT_CAPTURE" in shown, shown
    assert "SUCCESSOR-EDIT" in (cap / "workspace.patch").read_text(encoding="utf-8")
    assert _state(target) == before and _rows(drive, custody.PATCH_DISPOSED) == []
    custody._CUSTODY.clear()
    out = _integrate_delegated_patch(ctx, "run-next", "apply", "inspected")
    assert "✅ Integrated" in out, out
    assert (target / "tracked.txt").read_text(encoding="utf-8") == "one\ntwo\nCHILD-EDIT\nSUCCESSOR-EDIT\n"
    [row] = _rows(drive, custody.PATCH_DISPOSED)
    assert (row["run_id"], row["disposition"]) == ("run-next", "applied") and "result_lost" not in row
    assert custody.replay(drive)[RID].patch_disposed == custody.SUPERSEDED
    custody._CUSTODY.clear()


@pytest.mark.parametrize("history", ["reflog_and_orig_head", "orig_head_only", "unreadable_reflog"])
def test_a_commit_the_gone_checkouts_head_history_still_reaches_is_not_loss(tmp_path, monkeypatch, history):
    """The child committed, then reset HEAD and index to the baseline: Git's entry still reaches that
    commit through ORIG_HEAD and its HEAD reflog. Nothing is called lost or pruned; the answer names
    the commit and how to keep or give it up, after which reject records the loss. A HEAD history
    that cannot be read proves nothing either."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    admin = pathlib.Path(_git(handle.path, "rev-parse", "--absolute-git-dir").stdout.strip())
    reflog = admin / "logs" / "HEAD"
    child = ""
    if history != "unreadable_reflog":
        _git(handle.path, "-c", "user.email=c@c", "-c", "user.name=child", "commit", "-qam", "child result")
        child = _git(handle.path, "rev-parse", "HEAD").stdout.strip()
        _git(handle.path, "reset", "-q", "--mixed", handle.baseline_sha)
        if history == "orig_head_only":
            reflog.unlink()
    shutil.rmtree(handle.path)
    before = _state(target)
    mode = reflog.stat().st_mode if reflog.exists() else 0
    if history == "unreadable_reflog":
        reflog.chmod(0)
        try:
            reflog.read_bytes()
        except PermissionError:
            pass
        else:
            reflog.chmod(mode)
            pytest.skip("this filesystem/user does not enforce file read modes")
    try:
        for decision in ("reject", "apply"):
            out = _integrate_delegated_patch(ctx, RID, decision, "")
            assert "loss is not proven" in out and "LOST" not in out, out
            ways = [argv[3:7] for argv in _ways_on(out)]
            assert (f"still records commit {child[:12]}" in out and ["worktree", "remove", "--force", "--force"] in ways
                    if child else "could not be read" in out), out
            custody._CUSTODY.clear()
    finally:
        if history == "unreadable_reflog" and reflog.exists():  # a wrongful reject prunes it with its entry
            reflog.chmod(mode)
    _untouched(drive, target, before)
    assert admin.exists()
    if child:
        assert _git(target, f"--git-dir={admin}", "rev-parse", "ORIG_HEAD").stdout.strip() == child
        [give_up] = [argv for argv in _ways_on(out) if "worktree" in argv]
        _git(target, *give_up[1:])  # the exit, run as the answer's argv
        out = _integrate_delegated_patch(ctx, RID, "reject", "the reset commit is not needed")
        assert "as LOST" in out, out
    custody._CUSTODY.clear()


def test_files_an_earlier_failed_capture_saved_stay_kept_and_named(tmp_path, monkeypatch):
    """A copy provisioned before the one-primitive update captures through the legacy producer, which
    saves file outputs before its Git diff. That diff failed, then the copy went: the tracked edit
    is lost and the saved binary is not. Apply and reject name what remains, the reject keeps it,
    and no reader says nothing was kept."""
    import zipfile

    from ouroboros import workspace_patch_capture
    from ouroboros.delegate_collection import capture_projection
    from ouroboros.owner_stop_receipt import EVENT_FIELD, FIELD, receipt_text, retained_run_facts, retained_work_fields
    from ouroboros.subagent_worktrees import _put_record
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.workspace_file_outputs import file_output_changes

    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    _put_record({**find_execution_snapshot(SNAP), "file_baseline": {}})
    binary = b"\x00legacy saved binary result\xff"
    pathlib.Path(handle.path, "report.bin").write_bytes(binary)
    original = workspace_patch_capture._append_git_output

    def failing_diff(cmd, root, fh, **kwargs):  # the real late Git call, failing the way Git fails
        return original(["git", "diff", "--no-such-option"], root, fh, **kwargs)

    with monkeypatch.context() as limited:
        limited.setattr(workspace_patch_capture, "_append_git_output", failing_diff)
        block = capture_terminal_patch_for_drive(drive, custody.replay(drive)[RID])
    assert block["status"] == "failed", block
    cap = pathlib.Path(block["manifest_artifact"]).parent
    [saved] = file_output_changes(json.loads((cap / "workspace_patch.json").read_text(encoding="utf-8")), cap)
    assert saved["path"] == "report.bin"
    shutil.rmtree(handle.path)
    custody._CUSTODY.clear()
    before = _state(target)

    applied = _integrate_delegated_patch(ctx, RID, "apply", "")
    assert "INTEGRATE_DELEGATED_CAPTURE_FAILED" in applied and "report.bin" in applied, applied
    assert "decision='reject')" in applied and "nothing was kept" not in applied, applied
    _untouched(drive, target, before)

    out = _integrate_delegated_patch(ctx, RID, "reject", "the patch is gone")
    assert "as LOST" in out and "report.bin" in out and "nothing was kept" not in out, out
    [row] = _rows(drive, custody.PATCH_DISPOSED)
    assert (row["result_lost"]["saved_capture"], row["result_lost"]["saved_files"]) == ("failed", 1), row
    # The verdict covers the lost tracked edit and the kept binary alike: neither was applied.
    assert (_verdict(out)["outcome"], _verdict(out)["files"]) == ("result_lost", ["report.bin", "tracked.txt"])
    with zipfile.ZipFile(saved["archive_path"]) as archive:
        assert archive.read(saved["member"]) == binary  # kept, never deleted
    custody._CUSTODY.clear()
    entry = custody.replay(drive)[RID]
    shown = capture_projection(drive, entry, reader_task_id=OWNER)
    assert shown["result_lost"] and shown["file_output_count"] == 1, shown
    assert "nothing was kept" not in shown["note"] and "1 file change" in shown["note"], shown
    facts = {"run_id": RID, "state": "succeeded", **retained_run_facts(drive, entry)}
    audit = {FIELD: {"task_id": OWNER, "runs": [facts]}}
    owner = retained_work_fields({"delegate_terminal_reconciliation": audit})[EVENT_FIELD]["text"]
    assert "nothing was saved" not in owner and "1 saved file change kept" in owner, owner
    assert "1 saved file change kept" in receipt_text(audit)
    assert _state(target) == before
    custody._CUSTODY.clear()


# -- the way on, as worded, and a decision's later readers (review rl-20261029T145748) --------


@pytest.mark.parametrize("held", ["staged", "committed"])
@pytest.mark.parametrize("folder", ["plain", "it's a space"])
def test_the_keep_and_give_up_commands_run_as_worded_and_keep_exact_bytes(tmp_path, monkeypatch, folder, held):
    """The child staged or committed a binary file, then its copy went. Each way on the refusal names is a
    run_command argv, run as given even under a folder named with an apostrophe and a space: the kept
    patch or branch holds the exact bytes, Git's entry goes only by the give-up argv, and the reject
    after it records the loss and leaves what was kept alone."""
    base = tmp_path / folder
    base.mkdir()
    target, ctx, handle, drive = _writer(base, monkeypatch)
    binary = bytes(range(256)) * 4
    pathlib.Path(handle.path, "report.bin").write_bytes(binary)
    _git(handle.path, "add", "report.bin")
    if held == "committed":
        _git(handle.path, "-c", "user.email=c@c", "-c", "user.name=child", "commit", "-qm", "child result")
    admin = pathlib.Path(_git(handle.path, "rev-parse", "--absolute-git-dir").stdout.strip())
    shutil.rmtree(handle.path)
    before = _state(target)
    out = _integrate_delegated_patch(ctx, RID, "reject", "")
    assert "loss is not proven" in out and "LOST" not in out, out
    keep, give_up = _ways_on(out)
    patch = base / "kept" / "staged result.patch"
    patch.parent.mkdir()
    if held == "staged":
        assert "--binary" in keep and "--full-index" in keep, keep
        keep = [arg.replace("<file>", str(patch)) for arg in keep]
    else:
        keep = [arg.replace("<name>", "kept-result") for arg in keep]
    subprocess.run(keep, cwd=patch.parent, check=True, capture_output=True)
    _untouched(drive, target, before)
    assert admin.exists()

    def kept_bytes():
        if held == "committed":
            return subprocess.run(["git", "-C", str(target), "cat-file", "blob", "kept-result:report.bin"],
                                  check=True, capture_output=True).stdout
        restored = base / "restored"
        shutil.rmtree(restored, ignore_errors=True)
        _git(base, "init", "-q", str(restored))
        _git(restored, "apply", str(patch))
        return (restored / "report.bin").read_bytes()

    assert kept_bytes() == binary
    subprocess.run(give_up, cwd=patch.parent, check=True, capture_output=True)
    assert not admin.exists()
    custody._CUSTODY.clear()
    out = _integrate_delegated_patch(ctx, RID, "reject", "kept by hand")
    assert "as LOST" in out, out
    assert kept_bytes() == binary and _state(target) == before
    custody._CUSTODY.clear()


@pytest.mark.parametrize("orig_head", ["corrupt", "empty", "missing"])
def test_a_present_orig_head_that_names_no_commit_is_not_absence(tmp_path, monkeypatch, orig_head):
    """Git writes ORIG_HEAD into a new checkout's entry. Present but naming no commit, it may have
    named the child's result, so neither decision proves loss or prunes the entry; absent, it holds
    nothing and reject records the loss."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    admin = pathlib.Path(_git(handle.path, "rev-parse", "--absolute-git-dir").stdout.strip())
    shutil.rmtree(handle.path)
    orig, payload = admin / "ORIG_HEAD", {"corrupt": b"not a commit\n", "empty": b""}.get(orig_head)
    if payload is None:
        orig.unlink(missing_ok=True)
        assert "as LOST" in _integrate_delegated_patch(ctx, RID, "reject", "") and not admin.exists()
        custody._CUSTODY.clear()
        return
    orig.write_bytes(payload)
    before = _state(target)
    for decision in ("reject", "apply"):
        out = _integrate_delegated_patch(ctx, RID, decision, "")
        assert "loss is not proven" in out and "ORIG_HEAD" in out and "LOST" not in out, out
        custody._CUSTODY.clear()
    _untouched(drive, target, before)
    assert orig.read_bytes() == payload


@pytest.mark.parametrize("collector, saved", [("foreground", "nothing"), ("body", "nothing"), ("body", "binary"),
                                              ("body", "unreadable")])
@pytest.mark.parametrize("reader", ["owner", "other"])
def test_output_collected_after_a_lost_reject_is_delivered_with_its_decision(tmp_path, monkeypatch, collector, saved, reader):
    """The engine's terminal answer was observed; then the copy went and its owner rejected the run as
    lost before its output or money were collected. The delayed delivery collects them and captures
    the gone copy again, itself or through the maintenance child's job. Under the body the first delivery
    comes while that job is owed, and still wakes the child: it shows what the run saved (the binary an
    earlier failed capture kept, with readers that reach it; nothing after a true loss; an unknown count
    once that manifest cannot be read), and the child's later result agrees, including after a torn
    manifest. A different delivery reader receives absolute source addresses rather than the owner's
    relative prefix; these address checks do not grant that reader access.
    Each capture carries the recorded decision at top level and in the collection block, and the unpaid
    ledger stays owed."""
    import copy
    import zipfile

    from ouroboros import delegate_capture_job as job
    from ouroboros import delegate_collection as body
    from ouroboros import delegate_custody_current as current
    from ouroboros import maintenance, owner_stop_receipt, usage_accounting, workspace_patch_capture
    from ouroboros.subagent_worktrees import _put_record
    from ouroboros.tool_access_roots import resource_root_path
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.delegate_terminal_evidence import collect_and_deliver_terminal
    from ouroboros.workspace_file_outputs import file_output_changes
    from tests.test_delegate_body_collection import terminal

    target, ctx, handle, drive = _writer(tmp_path, monkeypatch, settled=False)
    assert body.observe_run(drive, current.run(drive, RID), terminal(RID))
    binary = b"\x00saved beside a lost patch\xff"
    manifest = custody.delegated_capture_dir(drive, OWNER, SNAP) / "workspace_patch.json"
    if saved != "nothing":  # a legacy copy's capture saves its file outputs, then its late Git diff fails
        _put_record({**find_execution_snapshot(SNAP), "file_baseline": {}})
        pathlib.Path(handle.path, "report.bin").write_bytes(binary)
        original = workspace_patch_capture._append_git_output

        def failing_diff(cmd, root, fh, **kwargs):
            return original(["git", "diff", "--no-such-option"], root, fh, **kwargs)

        with monkeypatch.context() as limited:
            limited.setattr(workspace_patch_capture, "_append_git_output", failing_diff)
            assert capture_terminal_patch_for_drive(drive, current.run(drive, RID))["status"] == "failed"
        [kept] = file_output_changes(json.loads(manifest.read_text(encoding="utf-8")), manifest.parent)
    shutil.rmtree(handle.path)
    assert "as LOST" in _integrate_delegated_patch(ctx, RID, "reject", "copy gone")
    [row] = _rows(drive, custody.PATCH_DISPOSED)
    assert row["result_lost"].get("saved_files", 0) == (0 if saved == "nothing" else 1), row
    if saved == "unreadable":
        manifest.write_bytes(b'{"status": "fai')  # torn after the decision
    custody._CUSTODY.clear()
    before, entry = _state(target), current.run(drive, RID)
    reading = copy.copy(ctx)
    if reader == "other":
        reading.task_id = "t-second"

    def unpaid(*_a, **_kw):
        raise OSError("controlled ledger gap")

    def deliver():
        return collect_and_deliver_terminal(reading, RID, terminal(RID), body._shape(entry), entry, None)

    woken, deliveries, runtime = [], [], None
    monkeypatch.setattr(usage_accounting, "record_subscription_session", unpaid)
    monkeypatch.setattr(maintenance.child, "request", lambda: woken.append(RID))  # the test runs the child's job
    try:
        if collector == "body":
            monkeypatch.setattr(body.BodyRuntime, "tick", lambda self: None)  # its own pass stays parked
            runtime = body.start_body_runtime(drive, gateway_factory=lambda: pytest.fail("no transport is owed"))
        deliveries.append(deliver())
        assert woken == ([RID] if collector == "body" else []), woken
        if collector == "body":  # that first delivery came while the child still owed its capture
            assert current.run(drive, RID).capture_job["state"] == "pending"
            assert job.run(drive, lambda: False)["items"] == 1
            deliveries.append(deliver())
    finally:
        if runtime is not None:
            runtime.close(wait=True)
    for payload in deliveries:
        assert payload["primary_output"]["text"] == "complete answer", payload
        for shown in (payload["workspace_capture"], payload["collection"]["workspace_capture"]):
            assert (shown.get("disposition"), shown.get("result_lost"), shown.get("disposed_by")) == (
                "rejected", True, OWNER), shown
            assert "No apply or reject is owed" in shown["note"], shown
            if saved == "binary":
                assert shown["file_output_count"] == 1, shown
                assert "the 1 file change its failed capture saved is kept" in shown["note"], shown
                read = resource_root_path(reading, "artifact_store") / shown["manifest_read"]["path"]
                assert read.resolve() == manifest.resolve(), shown
                assert kept["archive_path"] in [item["path"] for item in shown["file_outputs"]], shown
            elif saved == "unreadable":
                assert "file_output_count" not in shown and "is unknown" in shown["note"], shown
            else:
                assert shown["status"] == "missing" and "nothing was kept" in shown["note"], shown
            assert saved == "nothing" or "nothing was kept" not in shown["note"], shown
            owner_line = owner_stop_receipt._owner_run({"run_id": RID, "owner_task_id": OWNER,
                "patch": {"capture": shown, "disposition": "rejected"}}, "")
            if saved == "unreadable":
                assert "unknown" in owner_line and "nothing was saved" not in owner_line, owner_line
            elif saved == "nothing":
                assert "nothing was saved" in owner_line, owner_line
        assert (payload["settlement"]["settled"], payload["settlement"]["ledger_recorded"]) == (False, False)
    if saved != "nothing":
        with zipfile.ZipFile(kept["archive_path"]) as archive:
            assert archive.read(kept["member"]) == binary  # kept, never deleted
    fresh = current.run(drive, RID)
    assert fresh.output_complete and not fresh.settled and current.collection_pending(fresh)
    assert (fresh.patch_disposed, fresh.patch_disposed_by, fresh.patch_result_lost) == ("rejected", OWNER, True)
    assert len(_rows(drive, custody.PATCH_DISPOSED)) == 1 and _state(target) == before
    custody._CUSTODY.clear()


@pytest.mark.parametrize("reflog", ["missing", "baseline_only", "expired_by_git", "retained_child", "retained_old_only", "wholly_malformed",
                                    "malformed_beside_baseline", "unterminated_record"])
def test_a_head_reflog_git_reads_only_in_part_is_not_absence(tmp_path, monkeypatch, reflog):
    """Git skips, with exit 0, each HEAD reflog line it cannot parse, a final record without its newline
    included. A present reflog it reads only in part may have recorded the child's result, so neither
    decision proves loss or prunes the entry. Missing, emptied by Git itself or holding only baseline
    records, it holds nothing and reject records the loss; a retained commit is named with the way on."""
    target, ctx, handle, drive = _writer(tmp_path, monkeypatch)
    admin = pathlib.Path(_git(handle.path, "rev-parse", "--absolute-git-dir").stdout.strip())
    log, child = admin / "logs" / "HEAD", ""
    if reflog in ("retained_child", "retained_old_only"):
        _git(handle.path, "-c", "user.email=c@c", "-c", "user.name=child", "commit", "-qam", "child result")
        child = _git(handle.path, "rev-parse", "HEAD").stdout.strip()
        _git(handle.path, "reset", "-q", "--mixed", handle.baseline_sha)
        (admin / "ORIG_HEAD").unlink()  # only the reflog still reaches it
        if reflog == "retained_old_only":
            # Git's default delete keeps the reset record's old value; log -g prints only its new value.
            _git(handle.path, "reflog", "delete", "HEAD@{1}")
    shutil.rmtree(handle.path)
    valid = log.read_bytes()  # as Git wrote it: baseline records, and the child's when it committed
    if reflog == "missing":
        log.unlink()
    elif reflog == "expired_by_git":
        _git(target, f"--git-dir={admin}", "reflog", "expire", "--expire=now", "--expire-unreachable=now", "HEAD")
        assert log.read_bytes() == b""
    elif reflog not in ("baseline_only", "retained_child", "retained_old_only"):
        log.write_bytes({"wholly_malformed": b"not a record\n", "malformed_beside_baseline": valid + b"not a record\n",
                         "unterminated_record": valid.rstrip(b"\n")}[reflog])
    if reflog in ("missing", "baseline_only", "expired_by_git"):
        assert "as LOST" in _integrate_delegated_patch(ctx, RID, "reject", "") and not admin.exists()
        custody._CUSTODY.clear()
        return
    kept, before = log.read_bytes(), _state(target)
    for decision in ("reject", "apply"):
        out = _integrate_delegated_patch(ctx, RID, decision, "")
        assert "loss is not proven" in out and "LOST" not in out, out
        assert (f"still records commit {child[:12]}" if child else "reflog records") in out, out
        custody._CUSTODY.clear()
    _untouched(drive, target, before)
    assert log.read_bytes() == kept
