"""A confirmed retry receives settled work without inheriting live control."""

from __future__ import annotations

import json

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.utils import utc_now_iso
from tests._shared import write_queue_document


pytestmark = pytest.mark.serial


@pytest.fixture(autouse=True)
def isolated_retry_state(monkeypatch):
    from ouroboros.tools import delegate

    memo = {}
    monkeypatch.setattr(custody, "_CUSTODY", memo)
    monkeypatch.setattr(delegate, "_CUSTODY", memo)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")


def retry_context(tmp_path, monkeypatch, *, chain=("retry-a", "retry-b"), split=False,
                  fixed_ownership_clock=False):
    """Durable host retry mirrors and the supervisor's current execution census."""
    data, repo, target = (tmp_path / name for name in ("data", "repo", "target"))
    for path in (data, repo, target):
        path.mkdir(exist_ok=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data))
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snapshots"))
    for index, task_id in enumerate(chain):
        fields = {"root_task_id": chain[0], "parent_task_id": "", "delegation_role": "root"}
        if index:
            fields.update(supersedes_task_id=chain[index - 1],
                          original_task_id=chain[index - 1], timeout_retry_from=chain[index - 1])
        if index + 1 < len(chain):
            fields.update(superseded_by=chain[index + 1], retry_task_id=chain[index + 1])
        write_task_result(data, task_id, "running" if index + 1 == len(chain) else "interrupted",
                          reason_code="idle_timeout" if index + 1 < len(chain) else "", **fields)
    drive = tmp_path / "child-drive" if split else data
    drive.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id=chain[-1],
                      workspace_root=target, workspace_mode="external")
    ctx.budget_drive_root = data
    ctx.task_metadata = {"root_task_id": chain[0], "parent_task_id": "", "delegation_role": "root"}
    queue_snapshot(ctx, [chain[-1]])
    if fixed_ownership_clock:
        from types import SimpleNamespace
        from ouroboros import task_status

        # No supervisor refreshes this synthetic queue during capture or EOF
        # reads. Only these consumers opt in; stale-authority tests stay real.
        clock = task_status.time
        now = clock.time()
        monkeypatch.setattr(task_status, "time", SimpleNamespace(
            time=lambda: now, monotonic=clock.monotonic, sleep=clock.sleep))
    return ctx


def queue_snapshot(ctx, active, *, ts=None):
    data = custody.custody_root(ctx)
    rows = [{"id": task_id, "task": {
        **(load_task_result(data, task_id) or {}), "id": task_id, "type": "task",
    }} for task_id in active]
    path = data / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_queue_document(path, {"schema_version": 1, "ts": ts or utc_now_iso(),
                                "running": rows, "pending": []})
    return path


def durable_run(ctx, *, owner="retry-a", run_id="run-retry", state="succeeded", **fields):
    data = custody.custody_root(ctx)
    fields.setdefault("access", "readonly")
    entry = custody.RunCustody(run_id=run_id, task_id=owner, route_id="test-route",
                              model="test-model", ledger_root=str(data),
                              root_task_id=ctx.task_metadata["root_task_id"],
                              target_root=str(ctx.workspace_root), **fields)
    assert custody.record_started(data, entry)
    if state is not None:
        assert custody.emit(data, custody.SETTLED, {
            "run_id": run_id, "task_id": owner, "state": state,
            "cost_usd": 0.25, "cost_final": True, "spend_disclosed": True,
        })
    custody._CUSTODY.clear()
    return custody.replay(data)[run_id]


def call(ctx, tool, **arguments):
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry.execute_result(tool, arguments)


class TerminalGateway:
    engine_version = "99.0.0"

    def __init__(self, *, state="succeeded", output="completed work", effective_access="readonly"):
        self.state, self.output, self.effective_access = state, output, effective_access
        self.reads = []

    def get_run(self, run_id, **_kwargs):
        self.reads.append(run_id)
        return {"lastSeq": 9, "primaryOutput": self.output, "summary": {
            "state": self.state, "spendUsd": 0.25, "effectiveAccess": self.effective_access,
        }}

    def close(self):
        pass

    def cancel_run(self, *_args, **_kwargs):
        raise AssertionError("a retry result reader must never cancel its predecessor's run")

    def remove_project(self, *_args, **_kwargs):
        raise AssertionError("a retry result reader must never retire the predecessor's project")


def bind_gateway(monkeypatch, gateway):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor

    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda *a, **k: gateway)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: gateway)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: gateway)


@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
@pytest.mark.parametrize("owner,chain", [
    ("retry-a", ("retry-a", "retry-b")),
    ("retry-a", ("retry-a", "retry-b", "retry-c")),
    ("retry-b", ("retry-a", "retry-b", "retry-c")),
])
def test_registered_wait_retrieves_completed_ancestor_without_changing_starter(
    tmp_path, monkeypatch, state, owner, chain,
):
    ctx = retry_context(tmp_path, monkeypatch, chain=chain)
    held = durable_run(ctx, owner=owner, state=state)
    bind_gateway(monkeypatch, TerminalGateway(state=state))
    result = call(ctx, "delegate_wait", run_id=held.run_id)
    assert result.status == "ok", result.text
    payload = json.loads(result.text)
    assert payload["status"] == "terminal" and payload["state"] == state
    assert payload["primary_output"] == "completed work"
    status, replayed = custody.lookup(custody.custody_root(ctx), ctx.task_id, held.run_id)
    assert status == custody.FOREIGN and replayed.task_id == owner
    assert load_task_result(custody.custody_root(ctx), owner)["status"] == "interrupted"
    rows = list(custody._iter_rows(custody.event_log_path(custody.custody_root(ctx))))
    assert len([row for row in rows if row["type"] == custody.SETTLED]) == 1


def test_successor_wait_does_not_enter_live_containment_or_settlement(tmp_path, monkeypatch):
    from ouroboros.tools import delegate

    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx)
    bind_gateway(monkeypatch, TerminalGateway(effective_access="full"))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("terminal result authority cannot enter live run management")

    monkeypatch.setattr(delegate, "_halt_breached_run", unexpected)
    monkeypatch.setattr(custody, "settle_run", unexpected)
    result = call(ctx, "delegate_wait", run_id=held.run_id)
    assert result.status == "ok", result.text
    assert json.loads(result.text)["state"] == "succeeded"


@pytest.mark.parametrize("case", [
    "common_root", "narrative", "one_sided", "wrong_root", "cycle", "old_execution_live",
    "parallel_attempt", "malformed_parallel_attempt", "snapshot_stale", "snapshot_missing", "leaf_live", "closed_absent",
    "review_owned", "stale_successor", "stale_custody_memo",
])
def test_registered_wait_refuses_unproven_successor_without_touching_gateway(
    tmp_path, monkeypatch, case,
):
    ctx = retry_context(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    fields = {"source": "review_substrate", "review_slot_id": "slot-a"} if case == "review_owned" else {}
    state = None if case in {"leaf_live", "closed_absent", "stale_custody_memo"} else "succeeded"
    held = durable_run(ctx, state=state, **fields)
    if case == "closed_absent":
        assert custody.emit(data, custody.CLOSED_ABSENT, {"run_id": held.run_id, "task_id": held.task_id})
    elif case == "stale_custody_memo":
        held.settled, held.terminal_state = True, "succeeded"
        custody._CUSTODY[held.run_id] = held
    elif case in {"common_root", "narrative"}:
        write_task_result(data, "retry-a", "interrupted", superseded_by="", retry_task_id="")
        write_task_result(data, "retry-b", "running", original_task_id="", timeout_retry_from="",
                          supersedes_task_id="", predecessor_task_id="retry-a" if case == "narrative" else "")
    elif case == "one_sided":
        write_task_result(data, "retry-a", "interrupted", superseded_by="", retry_task_id="")
    elif case == "wrong_root":
        write_task_result(data, "retry-b", "running", root_task_id="different-root")
    elif case == "cycle":
        write_task_result(data, "retry-a", "interrupted", original_task_id="retry-b",
                          supersedes_task_id="retry-b", timeout_retry_from="retry-b")
        write_task_result(data, "retry-b", "running", superseded_by="retry-a", retry_task_id="retry-a")
    elif case == "old_execution_live":
        queue_snapshot(ctx, ["retry-a", "retry-b"])
    elif case in {"parallel_attempt", "malformed_parallel_attempt"}:
        write_task_result(data, "retry-other", "running", root_task_id="retry-a",
                          parent_task_id="", delegation_role="root",
                          original_task_id="retry-a",
                          timeout_retry_from="retry-a" if case == "parallel_attempt" else "")
        queue_snapshot(ctx, ["retry-b", "retry-other"])
    elif case == "snapshot_stale":
        queue_snapshot(ctx, ["retry-b"], ts="2000-01-01T00:00:00+00:00")
    elif case == "snapshot_missing":
        (data / "state" / "queue_snapshot.json").unlink()
    elif case == "stale_successor":
        write_task_result(data, "retry-b", "interrupted", superseded_by="retry-c", retry_task_id="retry-c")
        write_task_result(data, "retry-c", "running", root_task_id="retry-a",
                          parent_task_id="", delegation_role="root",
                          original_task_id="retry-b", timeout_retry_from="retry-b", supersedes_task_id="retry-b")
        queue_snapshot(ctx, ["retry-c"])
    gateway = TerminalGateway()
    bind_gateway(monkeypatch, gateway)
    result = call(ctx, "delegate_wait", run_id=held.run_id)
    assert result.status == "error", result.text
    assert json.loads(result.text)["reason"] == "run_not_owned"
    assert gateway.reads == []


@pytest.mark.parametrize("state", [None, "succeeded"])
@pytest.mark.parametrize("tool,options", [
    ("delegate_cancel", {"reason": "finished"}),
    ("delegate_answer", {"interaction_id": "question-one", "answers": [{"question_id": "q", "free_text": "yes"}]}),
])
def test_retry_never_inherits_cancel_or_answer(tmp_path, monkeypatch, tool, options, state):
    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx, state=state)
    gateway = TerminalGateway()
    bind_gateway(monkeypatch, gateway)
    result = call(ctx, tool, run_id=held.run_id, **options)
    assert result.status == "error", result.text
    assert json.loads(result.text)["reason"] == "run_not_owned"
    assert gateway.reads == []


@pytest.mark.parametrize("state", [None, "succeeded"])
def test_retry_can_message_a_live_same_tree_run_but_not_a_settled_one(tmp_path, monkeypatch, state):
    from tests.test_delegate_message import _Stub, _row

    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx, state=state)
    gateway = _Stub(result={"accepted": True, "outcome": "accepted"}, harnesses=[_row("test-route")])
    bind_gateway(monkeypatch, gateway)
    result = call(ctx, "delegate_message", run_id=held.run_id, text="continue")
    payload = json.loads(result.text)
    if state is None:
        assert result.status == "ok", result.text
        assert payload["status"] == "accepted" and payload["accepted"] is True
        assert len(gateway.posts) == 1 and gateway.posts[0]["run_id"] == held.run_id
        assert custody.lookup(custody.custody_root(ctx), ctx.task_id, held.run_id)[0] == custody.FOREIGN
    else:
        assert result.status == "error" and payload["reason"] == "run_settled"
        assert gateway.reads == gateway.posts == []


def test_terminal_result_read_refuses_a_now_live_engine_without_live_effects(tmp_path, monkeypatch):
    ctx = retry_context(tmp_path, monkeypatch)
    held = durable_run(ctx)
    gateway = TerminalGateway(state="running")
    bind_gateway(monkeypatch, gateway)
    result = call(ctx, "delegate_wait", run_id=held.run_id,
                  checkpoint_after_sec=1, checkpoint_reason="bound this contradictory-engine fixture")
    assert result.status == "error", result.text
    assert json.loads(result.text)["status"] == "refused"
    assert gateway.reads == [held.run_id]


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("disposed", [False, True])
def test_successor_reads_canonical_capture_before_and_after_disposition(
    tmp_path, monkeypatch, split, disposed,
):
    # B is neither C's root nor its parent in the ordinary file-read lineage.
    ctx = retry_context(tmp_path, monkeypatch, chain=("retry-a", "retry-b", "retry-c"), split=split)
    held = durable_run(ctx, owner="retry-b", snapshot_id="capture-one",
                       execution_root=str(tmp_path / "snapshot"))
    data = custody.custody_root(ctx)
    capture = custody.delegated_capture_dir(data, held.task_id, held.snapshot_id)
    capture.mkdir(parents=True)
    patch = capture / "workspace.patch"
    patch.write_text("--- a/result.txt\n+++ b/result.txt\n@@ -1 +1 @@\n-old\n+retained work\n", encoding="utf-8")
    manifest = capture / "workspace_patch.json"
    manifest.write_text(json.dumps({"status": "ready_with_changes"}), encoding="utf-8")
    assert custody.record_patch_captured(data, held, status="ready_with_changes")
    if disposed:
        assert custody.record_patch_disposed(data, held, disposition="rejected", disposed_by_task_id=ctx.task_id)
    custody._CUSTODY.clear()
    bind_gateway(monkeypatch, TerminalGateway())
    waited = call(ctx, "delegate_wait", run_id=held.run_id)
    assert waited.status == "ok", waited.text
    references = json.loads(waited.text)["workspace_capture"]
    result = call(ctx, "read_file", **references["patch_read"])
    assert result.status == "ok" and "retained work" in result.text, result.text
    manifest_read = call(ctx, "read_file", **references["manifest_read"])
    assert manifest_read.status == "ok" and "ready_with_changes" in manifest_read.text
    private = capture.parent.parent / "private-notes.txt"
    private.write_text("unrelated owner notes", encoding="utf-8")
    refused = call(ctx, "read_file", root="artifact_store", path=str(private))
    assert "unrelated owner notes" not in refused.text
    assert refused.status == "error", refused.text
