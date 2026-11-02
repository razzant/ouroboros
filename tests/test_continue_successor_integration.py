"""Owner Continue quiz (2026-10-14), option 1, at the real integration entry.

A root the owner's explicit Continue created may accept or reject the FINISHED
native result of a direct child of its recorded predecessor. Its verdict is its
own record and names the original parent; nothing else widens.
"""
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import subagent_worktrees as trees
from ouroboros.artifacts import task_artifact_dir_path
from ouroboros.owner_continue import claim_on_predecessor, successor_id, work_order_text
from ouroboros.task_results import write_task_result
from ouroboros.task_status import load_effective_task_result
from ouroboros.tools.join_ledger import (
    _child_result_sha256,
    _discard_child_result,
    _is_own_child,
    _record_current_child_result_disposition,
    _resume_child_task,
)
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools.subagent_integration import _integrate_subagent_patch, get_tools
from ouroboros.workspace_copies import admitted_copy_metadata
from ouroboros.workspace_patch_capture import write_workspace_patch_artifacts
from supervisor.events_subagent_admission import _resolve_subagent_constraint
from tests._store_shared import read_result, result_signature, seed_result

# Actual Git subprocess fixtures follow the repository's real-process lane.
pytestmark = pytest.mark.serial

NONCE = "continue-nonce-1"
SUCCESSOR = successor_id("pred", NONCE)
LINEAGE_KEYS = {"decision_authority", "original_parent_task_id", "child_result_sha256"}


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout


def repo(path, files):
    path.mkdir()
    git(path, "init", "-q")
    git(path, "config", "user.name", "Fixture")
    git(path, "config", "user.email", "fixture@example.invalid")
    for name, content in files.items():
        (path / name).write_bytes(content)
    git(path, "add", "-A")
    git(path, "commit", "-qm", "fixture")
    return path


@pytest.fixture
def world(tmp_path, monkeypatch):
    system = repo(tmp_path / "system", {"system.txt": b"own body\n"})
    project = repo(tmp_path / "project", {"a.txt": b"base\n"})
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "copies"))
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.delenv("OUROBOROS_BOOT_RUNTIME_MODE", raising=False)
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    # Keep the boundary's isolated HOME: treating the entire fixture as HOME
    # would also declare data's parent (and its sibling projects) the app root.
    # The worker-side queue projection is its own tested seam; here it names who is live.
    live = set()
    monkeypatch.setattr("ouroboros.task_status.task_has_live_queue_ownership",
                        lambda _root, task_id: task_id in live)
    return SimpleNamespace(tmp=tmp_path, system=system, project=project, data=data, live=live)


def interrupted_root(data, task_id="pred"):
    """A root a restart interrupted: the recorded technical cause Continue requires."""
    write_task_result(data, task_id, "running", root_task_id=task_id, chat_id=7, project_id="")
    write_task_result(data, task_id, "cancelled",
                      cancel_origin={"source": "snapshot_restore", "reason": "server_shutdown"})


def continue_from(data, predecessor="pred", nonce=NONCE):
    """The owner's Continue claim, written by its one writer before the successor exists."""
    successor = successor_id(predecessor, nonce)
    claim_on_predecessor(data, predecessor, nonce, {
        "verb": "continue", "binding_version": 1, "predecessor_task_id": predecessor,
        "action_nonce": nonce, "successor_task_id": successor, "cause": "snapshot_restore"})
    return successor


def copy_child(w, task_id="child", parent="pred", edit=b"base\nchild\n"):
    """A finished self_worktree child of PARENT, captured as headless finalization does."""
    constraint, folder, mode, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=w.system, DRIVE_ROOT=w.data), tid=task_id,
        requested_constraint={"mode": "acting_subagent", "surface": "self_worktree"},
        workspace_root=str(w.project), workspace_mode="external", base_sha="", parent_task_id=parent)
    assert not error, error
    binding = admitted_copy_metadata(folder)
    task = {"id": task_id, "parent_task_id": parent, "root_task_id": parent,
            "delegation_role": "subagent", "workspace_root": folder, "workspace_mode": mode,
            "task_constraint": constraint, "workspace_copy": binding, "metadata": {"workspace_copy": binding}}
    (Path(folder) / "a.txt").write_bytes(edit)
    artifacts = task_artifact_dir_path(w.data, task_id, create=True)
    _, manifest = write_workspace_patch_artifacts(Path(folder), artifacts, task=task)
    assert manifest["status"] == "ready_with_changes", manifest
    write_task_result(w.data, task_id, "completed", result="helper finished", artifact_status="ready",
                      **{key: value for key, value in task.items() if key != "id"})
    current = load_effective_task_result(w.data, task_id)
    assert (current["status"], current["artifact_status"], current["root_task_id"]) == (
        "completed", "ready", parent)
    return Path(folder), artifacts, manifest


def shared_child(w, folder, task_id="shared", parent="pred"):
    """A finished external_workspace child of PARENT that wrote FOLDER directly."""
    constraint, root, mode, error = _resolve_subagent_constraint(
        SimpleNamespace(REPO_DIR=w.system, DRIVE_ROOT=w.data), tid=task_id,
        requested_constraint={"mode": "acting_subagent", "surface": "external_workspace",
                              "write_root": str(folder)},
        workspace_root=str(folder), workspace_mode="external", base_sha="", parent_task_id=parent)
    assert not error, error
    (folder / "a.txt").write_bytes(b"base\nshared\n")
    task = {"id": task_id, "workspace_root": root, "workspace_mode": mode, "task_constraint": constraint}
    _, manifest = write_workspace_patch_artifacts(
        folder, task_artifact_dir_path(w.data, task_id, create=True), task=task)
    assert manifest["status"] == "ready_with_changes", manifest
    write_task_result(w.data, task_id, "completed", result="Files written in the shared folder",
                      parent_task_id=parent, root_task_id=parent, delegation_role="subagent",
                      workspace_root=root, task_constraint=constraint, artifact_status="ready")
    current = load_effective_task_result(w.data, task_id)
    assert (current["status"], current["artifact_status"]) == ("completed", "ready")


def successor_ctx(w, task_id=SUCCESSOR, **extra):
    return ToolContext(repo_dir=w.system, drive_root=w.data, task_id=task_id,
                       workspace_root=w.project, workspace_mode="external", **extra)


def invoke(ctx, **args):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute_result("integrate_subagent_patch", args).text


def verdict(w, decider, child):
    path = task_artifact_dir_path(w.data, decider) / f"subagent_patch_verdict_{child}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def dispositions(w, decider):
    """The decider's attested patch verdicts: its ``subagent_patch_verdict`` custody rows."""
    from ouroboros.delegate_custody_current import marks
    return marks(w.data, decider).get("patch_verdicts", [])


def test_successor_applies_a_finished_copy_after_its_execution_root_is_gone(world):
    interrupted_root(world.data)
    folder, artifacts, manifest = copy_child(world)
    successor = continue_from(world.data)
    consumed = _child_result_sha256(load_effective_task_result(world.data, "child"))
    product = {name: (artifacts / name).read_bytes() for name in ("workspace.patch", "workspace_patch.json")}
    signatures = result_signature(world.data, "child"), result_signature(world.data, "pred")
    # Retention collects the child's execution copy first; the canonical capture is the product.
    assert trees.remove_worktree(task_id="child") and not folder.exists()

    out = _integrate_subagent_patch(successor_ctx(world), task_id="child", reason="finished helper work")

    assert "✅ Integrated subagent patch from child" in out and "owner-Continue successor of pred" in out, out
    assert (world.project / "a.txt").read_bytes() == b"base\nchild\n"
    assert git(world.project, "diff", "--cached", "--name-only").splitlines() == [b"a.txt"]
    receipt = verdict(world, successor, "child")
    assert (receipt["parent_task_id"], receipt["child_task_id"], receipt["outcome"]) == (successor, "child", "applied")
    assert (receipt["decision_authority"], receipt["original_parent_task_id"]) == ("owner_continue", "pred")
    assert receipt["child_result_sha256"] == consumed and receipt["patch_sha256"] == manifest["sha256"]
    assert Path(receipt["target_root"]) == world.project.resolve() and receipt["reason"] == "finished helper work"
    row = dispositions(world, successor)[-1]
    assert (row["child_task_id"], row["disposition"], row["applied"], row["original_parent_task_id"]) == (
        "child", "applied", True, "pred")
    # The author's record, evaluation and product, and the predecessor, stay exactly as recorded.
    assert (result_signature(world.data, "child"), result_signature(world.data, "pred")) == signatures
    assert {name: (artifacts / name).read_bytes() for name in product} == product
    assert not load_effective_task_result(world.data, "child").get("child_result_disposition")
    assert not (task_artifact_dir_path(world.data, "pred") / "subagent_patch_verdict_child.json").exists()


@pytest.mark.parametrize("decision", ["apply", "reject"])
def test_successor_decides_a_shared_result_and_its_own_child_stays_ordinary(world, decision):
    interrupted_root(world.data)
    shared = repo(world.tmp / "shared", {"a.txt": b"base\n"})
    shared_child(world, shared)
    successor = continue_from(world.data)
    ctx = successor_ctx(world)

    out = invoke(ctx, task_id="shared", decision=decision, reason="checked by the successor")

    assert ("Verified external_workspace child" if decision == "apply" else "🚫 Rejected") in out, out
    assert "owner-Continue successor of pred" in out, out
    receipt = verdict(world, successor, "shared")
    assert receipt["outcome"] == ("verified_shared_workspace" if decision == "apply" else "rejected")
    assert receipt["applied"] is False and receipt["original_parent_task_id"] == "pred"
    assert (shared / "a.txt").read_bytes() == b"base\nshared\n"  # verified or rejected, never rewritten
    assert not load_effective_task_result(world.data, "shared").get("child_result_disposition")

    own = repo(world.tmp / "own", {"a.txt": b"base\n"})
    shared_child(world, own, task_id="own", parent=successor)
    out = invoke(ctx, task_id="own", decision=decision, reason="my own helper")
    assert "owner-Continue" not in out, out
    assert not LINEAGE_KEYS & set(verdict(world, successor, "own"))
    assert load_effective_task_result(world.data, "own")["child_result_disposition"] == (
        "integrated" if decision == "apply" else "irrelevant")


def test_a_continue_chain_reaches_the_first_predecessors_child(world):
    interrupted_root(world.data)
    shared = repo(world.tmp / "shared", {"a.txt": b"base\n"})
    shared_child(world, shared)
    first = continue_from(world.data)
    interrupted_root(world.data, first)
    second = continue_from(world.data, first, "continue-nonce-2")
    out = invoke(successor_ctx(world, task_id=second), task_id="shared")
    assert "Verified external_workspace child" in out and "owner-Continue successor of pred" in out, out
    assert verdict(world, second, "shared")["original_parent_task_id"] == "pred"


REFUSALS = {
    "unrelated_root": "is not a direct child of this task",
    "metadata_only": "is not a direct child of this task",
    "automatic_retry": "is not a direct child of this task",
    "forged_claim": "is not a direct child of this task",
    "grandchild": "is not a direct child of this task",
    "running_child": "it is not provably finished",
    "live_owner": "it is not provably finished",
    "already_decided": "already marked this exact result integrated",
    "genesis": "not a native self_worktree/external_workspace result",
    "legacy_copy": "records no source to return to",
}


@pytest.mark.parametrize("decision", ["apply", "reject"])
@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_only_a_recorded_continue_of_a_finished_direct_child_widens_lineage(world, case, decision):
    interrupted_root(world.data)
    copy_child(world)
    target, caller, extra = "child", SUCCESSOR, {}
    if case != "metadata_only":
        continue_from(world.data)
    if case == "unrelated_root":
        caller = "unrelated-root"
    elif case == "metadata_only":
        extra = {"task_metadata": {"root_task_id": caller, "continuation": {"predecessor_task_id": "pred"}},
                 "task_contract": {"predecessor_authority": {"source": {"task_id": "pred"}}}}
    elif case == "automatic_retry":
        caller = f"{SUCCESSOR}-retry"
        write_task_result(world.data, caller, "running", root_task_id=caller, supersedes_task_id=SUCCESSOR,
                          original_task_id=SUCCESSOR, timeout_retry_from=SUCCESSOR)
    elif case == "forged_claim":
        row = read_result(world.data, "pred")
        row["continued_by"]["binding"]["cause"] = "edited after hashing"
        seed_result(world.data, "pred", row)
    elif case == "grandchild":
        copy_child(world, task_id="grand", parent="child")
        target = "grand"
    elif case in ("running_child", "genesis", "legacy_copy"):
        row = read_result(world.data, "child")
        if case == "running_child":
            row["status"] = "running"
        elif case == "genesis":
            row["task_constraint"] = {**row["task_constraint"], "surface": "genesis"}
        else:
            row.pop("workspace_copy", None)
            row["metadata"].pop("workspace_copy", None)
        seed_result(world.data, "child", row)
    elif case == "live_owner":
        world.live.add("child")
    elif case == "already_decided":
        pred = ToolContext(repo_dir=world.system, drive_root=world.data, task_id="pred")
        assert _record_current_child_result_disposition(
            pred, "child", "integrated", "applied before the restart").startswith("OK:")
    ctx = successor_ctx(world, task_id=caller, **extra)
    before = result_signature(world.data, target)

    out = _integrate_subagent_patch(ctx, task_id=target, decision=decision)

    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and REFUSALS[case] in out, out
    assert (world.project / "a.txt").read_bytes() == b"base\n"
    assert not (task_artifact_dir_path(world.data, caller) / f"subagent_patch_verdict_{target}.json").exists()
    assert result_signature(world.data, target) == before


def test_successor_meets_the_ordinary_integrity_and_target_refusals(world):
    interrupted_root(world.data)
    (world.project / "a.txt").write_bytes(b"parent uncommitted\n")
    _, artifacts, _ = copy_child(world, edit=b"child\n")
    successor = continue_from(world.data)
    ctx = successor_ctx(world)
    assert "INTEGRATE_TARGET_FORBIDDEN" in _integrate_subagent_patch(
        ctx, task_id="child", target_root=str(world.system))
    (world.project / "a.txt").write_bytes(b"concurrent parent\n")
    out = _integrate_subagent_patch(ctx, task_id="child")
    assert "INTEGRATE_CONFLICT" in out and "changed since copy" in out, out
    assert (world.project / "a.txt").read_bytes() == b"concurrent parent\n"
    receipt = verdict(world, successor, "child")
    assert receipt["outcome"] == "conflict" and receipt["original_parent_task_id"] == "pred"
    (artifacts / "workspace.patch").write_bytes(b"tampered\n")
    assert "INTEGRATE_PATCH_CORRUPT" in _integrate_subagent_patch(ctx, task_id="child")
    assert (world.project / "a.txt").read_bytes() == b"concurrent parent\n"


def test_successor_verification_uses_its_own_current_read_authority(world):
    interrupted_root(world.data)
    shared = repo(world.tmp / "shared", {"a.txt": b"base\n"})
    shared_child(world, shared)
    successor = continue_from(world.data)
    ctx = successor_ctx(world, task_contract={"resource_policy": {"protected_artifacts": [{
        "id": "black-box", "paths": [str(shared / "a.txt")], "deny": ["read_bytes"]}]}})
    out = invoke(ctx, task_id="shared")
    assert "INTEGRATE_TARGET_FORBIDDEN" in out and "Captured result retained" in out, out
    receipt = verdict(world, successor, "shared")
    assert receipt["outcome"] == "shared_workspace_read_refused" and receipt["original_parent_task_id"] == "pred"
    assert not load_effective_task_result(world.data, "shared").get("child_result_disposition")
    # A refused read is an observation, not the decision: the corrected attempt proceeds.
    out = _integrate_subagent_patch(successor_ctx(world), task_id="shared", reason="read granted now")
    assert "Verified external_workspace child" in out and "owner-Continue successor of pred" in out, out
    assert [row["disposition"] for row in dispositions(world, successor)] == [
        "shared_workspace_read_refused", "verified_shared_workspace"]


def test_deciding_a_result_gives_the_successor_no_control_over_old_helpers(world):
    interrupted_root(world.data)
    copy_child(world)
    continue_from(world.data)
    ctx = successor_ctx(world)
    assert "✅ Integrated" in _integrate_subagent_patch(ctx, task_id="child", reason="take it")
    assert not _is_own_child(ctx, world.data, "child")
    assert not _is_own_child(ctx, world.data, "child", root_tree=True)
    assert "is not a child of this task" in _discard_child_result(ctx, "child", "drop it")
    assert "is not a child of this task" in _resume_child_task(ctx, "child", "go on")
    assert "CHILD_RESULT_LINEAGE_FORBIDDEN" in _record_current_child_result_disposition(
        ctx, "child", "integrated", "mark it")
    assert not load_effective_task_result(world.data, "child").get("child_result_disposition")


def test_tool_description_and_continue_work_order_name_the_narrow_capability():
    schema = next(entry.schema for entry in get_tools() if entry.name == "integrate_subagent_patch")
    assert "the FINISHED native result of a direct child of its recorded predecessor" in schema["description"]
    assert "old helpers stay outside its control" in schema["description"]
    assert "a decision already recorded on that exact result stands" in schema["description"]
    text = work_order_text("pred", "owner_restart", {})
    assert "its helpers are not yours to steer" in text
    assert "integrate_subagent_patch(task_id=...)" in text and "continue_from=<run_id>" in text


@pytest.mark.parametrize("prior, refusal", [
    ("apply", "pred already marked this exact result integrated"),
    ("reject", "pred already marked this exact result irrelevant"),
    ("apply_tree_row_lost", "pred already recorded the decision 'applied'"),
])
def test_the_predecessors_own_decision_stands(world, monkeypatch, prior, refusal):
    interrupted_root(world.data)
    copy_child(world)
    pred = ToolContext(repo_dir=world.system, drive_root=world.data, task_id="pred",
                       workspace_root=world.project, workspace_mode="external")
    with monkeypatch.context() as patched:
        if prior == "apply_tree_row_lost":  # its verdict landed, its tree row did not
            patched.setattr("ouroboros.tools.join_ledger._record_current_child_result_disposition",
                            lambda *_a, **_k: "⚠️ CHILD_RESULT_DISPOSITION_WRITE_FAILED: fixture")
        out = _integrate_subagent_patch(pred, task_id="child", decision="reject" if prior == "reject" else "apply",
                                        reason="decided before the restart")
    assert ("🚫 Rejected" if prior == "reject" else "✅ Integrated") in out, out
    project = (world.project / "a.txt").read_bytes(), git(world.project, "diff", "--cached", "--name-only")
    successor = continue_from(world.data)
    for decision in ("apply", "reject"):
        out = _integrate_subagent_patch(successor_ctx(world), task_id="child", decision=decision)
        assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and refusal in out, out
        if prior == "apply_tree_row_lost":
            assert "covers every revision" in out and "this exact result" not in out, out
    assert ((world.project / "a.txt").read_bytes(),
            git(world.project, "diff", "--cached", "--name-only")) == project
    assert not (task_artifact_dir_path(world.data, successor) / "subagent_patch_verdict_child.json").exists()


@pytest.mark.parametrize("kind, decision, recorded", [
    ("shared", "apply", "verified_shared_workspace"),
    ("shared", "reject", "rejected"),
    ("copy", "apply", "applied"),
])
def test_one_decision_stands_for_the_same_successor_and_along_the_chain(world, kind, decision, recorded):
    interrupted_root(world.data)
    if kind == "shared":
        shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}), task_id="child")
    else:
        copy_child(world)
    first = continue_from(world.data)
    out = _integrate_subagent_patch(successor_ctx(world, task_id=first), task_id="child",
                                    decision=decision, reason="the first successor decides")
    assert "owner-Continue successor of pred" in out and "INTEGRATE_DISPOSITION_FAILED" not in out, out
    receipt = verdict(world, first, "child")
    project = (world.project / "a.txt").read_bytes(), git(world.project, "diff", "--cached", "--name-only")
    for again in ("apply", "reject"):  # the same successor repeating either decision changes nothing
        out = _integrate_subagent_patch(successor_ctx(world, task_id=first), task_id="child", decision=again)
        assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and f"{first} already recorded the decision {recorded!r}" in out, out
        assert f"{recorded!r} for this exact result" in out, out
    interrupted_root(world.data, first)
    second = continue_from(world.data, first, "continue-nonce-2")
    for again in ("apply", "reject"):  # B decided: C, its successor, cannot override
        out = _integrate_subagent_patch(successor_ctx(world, task_id=second), task_id="child", decision=again)
        assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and f"{first} already recorded the decision {recorded!r}" in out, out
        assert f"{recorded!r} for this exact result" in out, out
    assert verdict(world, first, "child") == receipt
    assert [row["disposition"] for row in dispositions(world, first)] == [recorded]
    assert not (task_artifact_dir_path(world.data, second) / "subagent_patch_verdict_child.json").exists()
    assert ((world.project / "a.txt").read_bytes(),
            git(world.project, "diff", "--cached", "--name-only")) == project


def test_a_conflict_is_not_the_decision_and_the_corrected_retry_decides_once(world):
    interrupted_root(world.data)
    (world.project / "a.txt").write_bytes(b"parent uncommitted\n")
    copy_child(world, edit=b"child\n")
    successor = continue_from(world.data)
    ctx = successor_ctx(world)
    (world.project / "a.txt").write_bytes(b"concurrent parent\n")
    assert "INTEGRATE_CONFLICT" in _integrate_subagent_patch(ctx, task_id="child")
    (world.project / "a.txt").write_bytes(b"parent uncommitted\n")  # the source is back at the copy's baseline
    out = _integrate_subagent_patch(ctx, task_id="child", reason="after restoring the source")
    assert "✅ Integrated" in out and "owner-Continue successor of pred" in out, out
    assert (world.project / "a.txt").read_bytes() == b"child\n"
    assert [row["disposition"] for row in dispositions(world, successor)] == ["conflict", "applied"]
    assert "already recorded the decision 'applied'" in _integrate_subagent_patch(ctx, task_id="child")


def test_a_decision_binds_the_exact_result_it_judged(world):
    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    assert "🚫 Rejected" in _integrate_subagent_patch(successor_ctx(world, task_id=first), task_id="shared",
                                                     decision="reject", reason="not this result")
    judged = verdict(world, first, "shared")["child_result_sha256"]
    row = read_result(world.data, "shared")
    row["result"] = "Files written in the shared folder, then a corrected final report"
    seed_result(world.data, "shared", row)
    assert _child_result_sha256(load_effective_task_result(world.data, "shared")) != judged
    interrupted_root(world.data, first)
    second = continue_from(world.data, first, "continue-nonce-2")
    out = _integrate_subagent_patch(successor_ctx(world, task_id=second), task_id="shared", reason="the new result")
    assert "Verified external_workspace child" in out, out
    assert verdict(world, second, "shared")["child_result_sha256"] not in ("", judged)


def test_unreadable_decision_records_refuse_and_an_unlanded_decision_is_disclosed(world, monkeypatch):
    from ouroboros import store

    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    successor = continue_from(world.data)
    ctx = successor_ctx(world)

    def unreadable(_root, _task_id):
        raise store.StoreError("fixture: custody marks unreadable")

    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.delegate_custody_current.marks", unreadable)
        out = _integrate_subagent_patch(ctx, task_id="shared")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and "decision records cannot be read" in out, out
    assert not (task_artifact_dir_path(world.data, successor) / "subagent_patch_verdict_shared.json").exists()

    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.delegate_custody.emit", lambda *_a, **_k: False)
        out = _integrate_subagent_patch(ctx, task_id="shared", reason="decided while custody is down")
    assert "Verified external_workspace child" in out and "INTEGRATE_DISPOSITION_FAILED" in out, out
    assert verdict(world, successor, "shared")["custody_row_write_failed"] is True


def test_a_decision_whose_custody_row_never_landed_still_stands(world, monkeypatch):
    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.delegate_custody.emit", lambda *_a, **_k: False)
        out = invoke(successor_ctx(world, task_id=first), task_id="shared", reason="custody down")
    assert "Verified external_workspace child" in out and "INTEGRATE_DISPOSITION_FAILED" in out, out
    receipt = verdict(world, first, "shared")
    assert receipt["custody_row_write_failed"] is True and not dispositions(world, first)
    for again in ("apply", "reject"):
        out = _integrate_subagent_patch(successor_ctx(world, task_id=first), task_id="shared", decision=again)
        assert f"{first} already recorded the decision 'verified_shared_workspace'" in out, out
    interrupted_root(world.data, first)
    second = continue_from(world.data, first, "continue-nonce-2")
    out = _integrate_subagent_patch(successor_ctx(world, task_id=second), task_id="shared", decision="reject")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and f"{first} already recorded" in out, out
    assert verdict(world, first, "shared") == receipt and not dispositions(world, first)
    assert not (task_artifact_dir_path(world.data, second) / "subagent_patch_verdict_shared.json").exists()


@pytest.mark.parametrize("body", ["{torn", json.dumps({"parent_task_id": "someone-else",
                                                       "child_task_id": "shared", "outcome": "rejected"})])
def test_an_unreadable_retained_verdict_refuses(world, body):
    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    path = task_artifact_dir_path(world.data, first, create=True) / "subagent_patch_verdict_shared.json"
    path.write_text(body, encoding="utf-8")
    out = invoke(successor_ctx(world, task_id=first), task_id="shared")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and "decision records cannot be read" in out, out
    assert path.read_text(encoding="utf-8") == body


def test_a_verdict_on_the_deciders_own_execution_drive_counts(world):
    from ouroboros.headless import task_artifacts_dir, task_state_dir

    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    store = task_artifacts_dir(task_state_dir(world.data, first) / "data", first)
    (store / "subagent_patch_verdict_shared.json").write_text(json.dumps({
        "parent_task_id": first, "child_task_id": "shared", "outcome": "rejected",
        "child_result_sha256": _child_result_sha256(load_effective_task_result(world.data, "shared"))}))
    out = invoke(successor_ctx(world, task_id=first), task_id="shared")
    assert f"{first} already recorded the decision 'rejected'" in out, out


def test_retained_decision_lookup_holds_the_existing_publication_lock(world, monkeypatch):
    from ouroboros import task_custody
    from ouroboros.tools.patch_verdict import retained_verdicts

    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    ordinary = task_custody.task_artifact_stores
    probes = []

    def observed_stores(root, tid):
        with task_custody.task_custody_lock(root, tid, timeout_sec=0.05) as acquired:
            probes.append(acquired)
        return ordinary(root, tid)

    monkeypatch.setattr(task_custody, "task_artifact_stores", observed_stores)
    assert list(retained_verdicts(world.data, first, "shared")) == []
    assert probes == [False]  # promotion/GC cannot move the source between these reads
    with task_custody.task_custody_lock(world.data, first, timeout_sec=0.05) as acquired:
        assert acquired  # the lookup released it


def test_retained_decision_survives_execution_drive_removal_after_promotion(world):
    import shutil
    from ouroboros.headless import task_artifacts_dir, task_state_dir
    from ouroboros.task_custody import task_custody_lock

    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    own_drive = task_state_dir(world.data, first) / "data"
    own_store = task_artifacts_dir(own_drive, first)
    payload = json.dumps({"parent_task_id": first, "child_task_id": "shared", "outcome": "rejected",
                          "child_result_sha256": _child_result_sha256(load_effective_task_result(world.data, "shared"))})
    name = "subagent_patch_verdict_shared.json"
    (own_store / name).write_text(payload, encoding="utf-8")
    interrupted_root(world.data, first)
    second = continue_from(world.data, first, "continue-nonce-2")
    # Fixture models the existing publish-before-drive-removal order; not a whole GC test.
    with task_custody_lock(world.data, first) as acquired:
        assert acquired
        canonical = task_artifact_dir_path(world.data, first, create=True)
        shutil.copyfile(own_store / name, canonical / name)
        shutil.rmtree(own_drive)
    out = invoke(successor_ctx(world, task_id=second), task_id="shared")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and f"{first} already recorded" in out, out
    assert (canonical / name).read_text(encoding="utf-8") == payload


def test_retained_verdict_file_symlink_is_not_followed(world):
    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    external = world.tmp / "foreign-verdict.json"
    external.write_text(json.dumps({"parent_task_id": first, "child_task_id": "shared", "outcome": "rejected"}),
                        encoding="utf-8")
    path = task_artifact_dir_path(world.data, first, create=True) / "subagent_patch_verdict_shared.json"
    path.symlink_to(external)
    out = invoke(successor_ctx(world, task_id=first), task_id="shared")
    assert "decision records cannot be read" in out, out
    assert path.is_symlink() and json.loads(external.read_text(encoding="utf-8"))["outcome"] == "rejected"


def test_retained_decision_follows_actual_collision_safe_promotion(world):
    import shutil
    from ouroboros.headless import task_artifacts_dir, task_state_dir
    from ouroboros.task_custody import _child_store_plan, _publish, task_custody_lock

    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    own_drive = task_state_dir(world.data, first) / "data"
    own_store = task_artifacts_dir(own_drive, first)
    canonical = task_artifact_dir_path(world.data, first, create=True)
    name = "subagent_patch_verdict_shared.json"
    base = {"parent_task_id": first, "child_task_id": "shared",
            "child_result_sha256": _child_result_sha256(load_effective_task_result(world.data, "shared"))}
    (canonical / name).write_text(json.dumps({**base, "outcome": "shared_workspace_read_refused"}), encoding="utf-8")
    payload = json.dumps({**base, "outcome": "rejected", "custody_row_write_failed": True})
    (own_store / name).write_text(payload, encoding="utf-8")
    staging = world.tmp / "staged-verdict"
    staging.mkdir()
    with task_custody_lock(world.data, first) as acquired:
        assert acquired
        plan = _child_store_plan(world.data, own_drive, first, {}, {}, staging)
        assert len(plan) == 1 and plan[0]["rel"] != name
        _publish(world.data, first, plan)
        shutil.rmtree(own_drive)
    assert (canonical / plan[0]["rel"]).read_text(encoding="utf-8") == payload
    interrupted_root(world.data, first)
    second = continue_from(world.data, first, "continue-nonce-2")
    out = invoke(successor_ctx(world, task_id=second), task_id="shared")
    assert "INTEGRATE_LINEAGE_FORBIDDEN" in out and f"{first} already recorded" in out, out


def test_retained_decisions_ignore_a_dotted_childs_files_but_keep_real_collision_copies(world):
    import hashlib
    from ouroboros.task_custody import _collision
    from ouroboros.tools.patch_verdict import retained_verdicts

    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)
    store = task_artifact_dir_path(world.data, first, create=True)
    for other in ("shared.x", "shared.0123abcd"):
        (store / f"subagent_patch_verdict_{other}.json").write_text(json.dumps(
            {"parent_task_id": first, "child_task_id": other, "outcome": "rejected"}), encoding="utf-8")
    assert list(retained_verdicts(world.data, first, "shared")) == []
    out = invoke(successor_ctx(world, task_id=first), task_id="shared", decision="reject", reason="its own decision")
    assert "decision records cannot be read" not in out and "🚫 Rejected" in out, out
    own = (store / "subagent_patch_verdict_shared.json").read_bytes()
    _collision(store / "subagent_patch_verdict_shared.json", hashlib.sha256(own).hexdigest()).write_bytes(own)
    assert [row["outcome"] for row in retained_verdicts(world.data, first, "shared")] == ["rejected", "rejected"]
    assert [row["child_task_id"] for row in retained_verdicts(world.data, first, "shared.x")] == ["shared.x"]


def test_a_decision_with_no_retained_record_does_not_claim_one(world, monkeypatch):
    interrupted_root(world.data)
    shared_child(world, repo(world.tmp / "shared", {"a.txt": b"base\n"}))
    first = continue_from(world.data)

    def unwritable(*_a, **_k):
        raise OSError("fixture: artifact store unwritable")

    with monkeypatch.context() as patched:
        patched.setattr("ouroboros.tools.patch_verdict.atomic_write_json", unwritable)
        patched.setattr("ouroboros.delegate_custody.emit", lambda *_a, **_k: False)
        out = invoke(successor_ctx(world, task_id=first), task_id="shared", reason="stores down")
    assert "Verdict: (unwritten)" in out and "INTEGRATE_DISPOSITION_FAILED" in out, out
    assert "retained verdict artifact" not in out and "verdict artifact was not written either" in out, out
