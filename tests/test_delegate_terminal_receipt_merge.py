"""Terminal custody disclosures merge with CURRENT inside the result write (#1573).

A maintenance refresh fixes its audit, then a late ``delegate_wait`` publishes the
retained-output locator and discharges its closing receipt before the refresh's
own write lands. That write merges through ``write_task_result``'s existing field
projector and reducer floor over CURRENT, so the newer locator survives; the debt
lists stay the writing audit's.
"""

from __future__ import annotations

import contextvars
import copy
import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current, delegate_terminal, obligations, task_results
from ouroboros.delegate_terminal import terminal_custody_notice
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_store import signature
from ouroboros.tools.registry import ToolContext
from tests.test_delegate_late_output_publication import discharged_retry, refresh, source_page
from tests.test_delegate_retry_output import (
    TerminalGateway,
    bind_gateway,
    call,
    durable_run,
    read_whole,
    receipts,
    wait_result,
)
from tests.test_delegate_retry_output import (
    isolated_retry_state as isolated_retry_state,  # autouse fixtures of the retry suites
)
from tests.test_delegate_retry_output import (
    reader_coverage as reader_coverage,
)



def _receipts(row):
    return {item["run_id"]: item for item in row["delegate_terminal_reconciliation"]["terminal_runs"]}


def _stored(root, task_id):
    """The stored row and its write signature: a no-op refresh leaves both untouched."""
    return signature(root, task_id), load_task_result(root, task_id)


def custody_unanswerable(patch):
    """The store cannot answer custody (the audit's readability probe), as an unopenable log once did."""
    patch.setattr(delegate_custody_current, "unreadable", lambda _root: True)


@pytest.mark.parametrize("incomplete_locator", [False, True])
def test_stale_sibling_refresh_cannot_erase_a_late_published_locator(
    tmp_path, monkeypatch, incomplete_locator,
):
    ctx, held, before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    if incomplete_locator:
        from ouroboros.delegate_output import _stage_full_output

        artifact_a = _stage_full_output(ctx, held.run_id, '{"primary_output": "old preview"}')
        assert custody.emit(data, custody.OUTPUT_SPILLED, {
            "run_id": held.run_id, "artifact": artifact_a["source"]["path"],
            "source": artifact_a["source"], "sha256": artifact_a["sha256"],
            "staged": True, "full_content": False,
        })
        before = load_task_result(data, held.task_id)
        assert _receipts(before)[held.run_id]["staged_output"]["complete"] is False
        assert obligations.members(data, "delegated_runs") == {}
    # Another run of the same task settles: the maintenance refresh has a real row to land.
    sibling = durable_run(ctx, run_id="run-sibling", state="failed")
    assert set(obligations.members(data, "delegated_runs")[held.task_id]["closed_runs"]) == {sibling.run_id}
    write = task_results.write_task_result
    armed, late = [True], []

    def late_wait_first(root, task_id, *args, **kwargs):
        if armed and task_id == held.task_id and "delegate_terminal_reconciliation" in kwargs:
            armed.clear()
            # The refresh's audit is fixed. The late wait runs in its own execution
            # context, like the worker thread that owns it, and lands first. The
            # recorder swallows exceptions, so its state is asserted after the call.
            late.append(contextvars.Context().run(wait_result, ctx, held.run_id))
            late.append(load_task_result(root, task_id))
            late.append(obligations.members(data, "delegated_runs"))
        return write(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "write_task_result", late_wait_first)
    stale = delegate_terminal.refresh_terminal_reconciliation(data, held.task_id)
    wait, published, owed = late
    artifact = wait["output_delivery"]["artifact"]
    assert _receipts(published)[held.run_id]["staged_output"]["sha256"] == artifact["sha256"]
    assert owed == {}, "the late publication discharged its closing duty before the stale write"
    after = load_task_result(data, held.task_id)
    receipts = _receipts(after)
    assert "staged_output" in receipts[held.run_id], "a stale audit erased the late-published locator"
    assert receipts[held.run_id]["staged_output"]["sha256"] == artifact["sha256"]
    assert receipts[held.run_id]["staged_output"]["complete"] is True
    if incomplete_locator:
        assert artifact["sha256"] != artifact_a["sha256"]
    assert source_page(ctx, held.task_id, artifact)["status"] == "available"
    assert receipts[sibling.run_id]["state"] == "failed"
    old = _receipts(before)
    assert {key: receipts[key] for key in ("old-failed", "old-cancelled")} == {
        key: old[key] for key in ("old-failed", "old-cancelled")}
    # Its audit added nothing the publication had not landed: no write, no "refreshed" claim.
    assert stale is False
    for key in ("status", "reason_code", "outcome_axes", "accounted_upper_bound_usd",
                "cost_final", "delegated_runs_started", "delegated_runs_settled"):
        assert after[key] == before[key]
    assert after["delegated_runs_unreconciled"] == []
    assert obligations.members(data, "delegated_runs") == {}
    assert obligations.members(data, "custody_open") == {}
    raw = _stored(data, held.task_id)
    assert delegate_terminal.refresh_terminal_reconciliation(data, held.task_id) is False
    assert _stored(data, held.task_id) == raw


class _EngineGateway(TerminalGateway):
    """A bounded engine preview whose artifact route fails (A) or serves the whole file (B)."""

    def __init__(self, full, *, serve):
        super().__init__(output={"kind": "answer", "path": "final/answer.md", "text": full[:120_000],
                                 "bytes": len(full.encode("utf-8")), "truncated": True})
        self.full, self.serve = full, serve

    def get_run_artifact(self, _run_id, _path):
        if not self.serve:
            raise OSError("artifact route unavailable")
        return self.full.encode("utf-8")


def _published_refs(reader, task_id, run_id):
    text = call(reader, "get_task_result", task_id=task_id).text
    summary = json.loads(text.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])
    view = summary.get("delegated_custody") or {}
    rows = [row for row in (*view.get("terminal_runs", []), *view.get("predecessor_terminal_runs", []))
            if row["run_id"] == run_id]
    assert len(rows) <= 1
    return [ref for row in rows for ref in (row["staged_output"], *row.get("other_staged_outputs", []))]


def _digests(reader, task_id, run_id):
    return {ref["sha256"]: ref["complete"] for ref in _published_refs(reader, task_id, run_id)}


def test_a_delayed_raw_audit_cannot_unpublish_a_later_complete_answer(tmp_path, monkeypatch):
    """Retry retains incomplete A; maintenance snapshots A while its closing receipt is owed;
    the retry then publishes complete B and discharges; the raw-A write lands last."""
    ctx, held, before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    write = task_results.write_task_result
    full = "late retained work " * 12_000

    def lost_publication(root, task_id, *args, **kwargs):
        if task_id == held.task_id and "delegate_terminal_reconciliation" in kwargs:
            return load_task_result(root, task_id)  # A's own refresh timed out on the lock
        return write(root, task_id, *args, **kwargs)

    bind_gateway(monkeypatch, _EngineGateway(full, serve=False))
    with monkeypatch.context() as patch:
        patch.setattr(task_results, "write_task_result", lost_publication)
        artifact_a = wait_result(ctx, held.run_id)["output_delivery"]["artifact"]
    owed = obligations.members(data, "delegated_runs")[held.task_id]["closed_runs"][held.run_id]
    assert (owed["output_source"], owed["output_complete"]) == (artifact_a["source"], False)

    bind_gateway(monkeypatch, _EngineGateway(full, serve=True))
    armed, late, audits = [True], [], []
    audit = delegate_terminal._audit_task_custody

    def observed_audit(root, mine, result, **kwargs):
        audit(root, mine, result, **kwargs)
        audits.append(copy.deepcopy(result))

    def late_wait_first(root, task_id, *args, **kwargs):
        if armed and task_id == held.task_id and "delegate_terminal_reconciliation" in kwargs:
            armed.clear()
            late.append(contextvars.Context().run(wait_result, ctx, held.run_id))
            late.append(load_task_result(root, task_id))
            late.append(obligations.members(data, "delegated_runs"))
        return write(root, task_id, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(delegate_terminal, "_audit_task_custody", observed_audit)
        patch.setattr(task_results, "write_task_result", late_wait_first)
        delegate_terminal.backfill_terminal_reconciliations(data)
    wait_b, published, discharged = late
    artifact_b = wait_b["output_delivery"]["artifact"]
    raw_a = {row["run_id"]: row for row in audits[0]["terminal_runs"]}[held.run_id]["staged_output"]
    assert (raw_a["sha256"], raw_a["complete"]) == (artifact_a["sha256"], False), "the raw snapshot held A"
    b = _receipts(published)[held.run_id]["staged_output"]
    assert (b["sha256"], b["complete"]) == (artifact_b["sha256"], True) != (artifact_a["sha256"], False)
    assert discharged == {}, "B's publication discharged its closing duty before the raw-A write"

    custody._CUSTODY.clear()
    reader = ToolContext(repo_dir=ctx.repo_dir, drive_root=data, task_id="physical-reader")
    refs = {ref["sha256"]: ref for ref in _published_refs(reader, held.task_id, held.run_id)}
    assert artifact_b["sha256"] in refs, "the delayed raw-A audit unpublished the later complete B"
    assert set(refs) == {artifact_a["sha256"], artifact_b["sha256"]}
    for artifact, status in ((artifact_b, "available"), (artifact_a, "incomplete")):
        recipe = refs[artifact["sha256"]]["source_read"]["arguments"]
        assert recipe == {"task_id": held.task_id, "delegated_output_sha256": artifact["sha256"]}
        page = source_page(reader, held.task_id, artifact)
        assert (page["status"], page["run_id"], page["staging_task_id"]) == (status, held.run_id, ctx.task_id)
        assert page["text"] == Path(artifact["abs_path"]).read_text(encoding="utf-8")[:80]
        pinned = call(reader, "get_task_result", task_id=ctx.task_id,
                      delegated_output_sha256=artifact["sha256"]).text
        assert json.loads(pinned)["delegated_output_source"]["reason"] == "source_not_published"
    unknown = call(reader, "get_task_result", task_id=held.task_id, delegated_output_sha256="f" * 64).text
    assert json.loads(unknown)["delegated_output_source"]["reason"] == "source_not_published"
    after = load_task_result(data, held.task_id)
    for key in ("status", "reason_code", "outcome_axes", "accounted_upper_bound_usd",
                "cost_final", "delegated_runs_started", "delegated_runs_settled"):
        assert after[key] == before[key]
    assert after["delegated_runs_unreconciled"] == []
    assert obligations.members(data, "delegated_runs") == {}
    raw = _stored(data, held.task_id)
    assert delegate_terminal.refresh_terminal_reconciliation(data, held.task_id) is False
    assert _stored(data, held.task_id) == raw
    # Visibility is not acknowledgement: only the authorized whole read of the CURRENT
    # custody source (B) earns D7; the stale preview A never does.
    assert not receipts(ctx)
    read_whole(ctx, artifact_a)
    assert not receipts(ctx)
    read_whole(ctx, artifact_b)
    assert [(row["sha256"], row["reader_task_id"]) for row in receipts(ctx)] == [
        (artifact_b["sha256"], ctx.task_id)]


def test_a_failed_kill_lane_audit_keeps_every_published_digest(tmp_path, monkeypatch):
    """A and B are published; a kill-lane refresh then meets custody the store cannot
    answer. Its failure stays disclosed debt, never a clean audit or an acknowledgement."""
    ctx, held, before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    full = "late retained work " * 12_000
    durable_run(ctx, run_id="run-open", state=None)  # still open: the kill lanes refresh its debt
    artifacts = []
    for serve in (False, True):
        bind_gateway(monkeypatch, _EngineGateway(full, serve=serve))
        artifacts.append(wait_result(ctx, held.run_id)["output_delivery"]["artifact"])
    artifact_a, artifact_b = artifacts
    reader = ToolContext(repo_dir=ctx.repo_dir, drive_root=data, task_id="physical-reader")
    published = {artifact_a["sha256"]: False, artifact_b["sha256"]: True}
    assert _digests(reader, held.task_id, held.run_id) == published
    assert load_task_result(data, held.task_id)["delegated_runs_unreconciled"] == ["run-open"]
    with monkeypatch.context() as patch:
        custody_unanswerable(patch)
        assert custody.custody_log_unreadable(data)
        assert delegate_terminal.refresh_terminal_reconciliation(
            data, held.task_id, trigger="kill_path_clear") is True
    custody._CUSTODY.clear()
    failed = load_task_result(data, held.task_id)
    seen = {"after_failed_audit": _digests(reader, held.task_id, held.run_id)}
    assert custody.emit(data, custody.SETTLED, {"run_id": "run-open", "task_id": held.task_id,
                                                "state": "failed", "cost_usd": 0.0, "cost_final": True,
                                                "spend_disclosed": True})
    assert refresh(data, monkeypatch) == [held.task_id]  # the addressed heal sees current custody only
    seen["after_heal"] = _digests(reader, held.task_id, held.run_id)
    assert seen == {"after_failed_audit": published, "after_heal": published}, "a published digest was lost"

    audit = failed["delegate_terminal_reconciliation"]
    assert audit["audit_status"] == "failed"
    assert failed["delegated_runs_unreconciled"] == audit["unreconciled"] == [
        "delegated_run_state_unknown:custody_log_unreadable"]
    assert terminal_custody_notice(failed).startswith("Current delegated execution could not be verified")
    for key in ("status", "reason_code", "outcome_axes", "accounted_upper_bound_usd",
                "cost_final", "delegated_runs_started", "delegated_runs_settled"):
        assert failed[key] == before[key]
    healed = load_task_result(data, held.task_id)
    assert healed["delegated_runs_unreconciled"] == []
    assert obligations.members(data, "delegated_runs") == {}
    for artifact, status in ((artifact_b, "available"), (artifact_a, "incomplete")):
        assert source_page(reader, held.task_id, artifact)["status"] == status
    assert not receipts(ctx), "neither the failed audit nor discovery acknowledges"


def test_a_failed_kill_audit_terminal_write_keeps_the_tasks_published_digest(tmp_path, monkeypatch):
    """The task's own loop exit publishes its run's retained output. The reaper then kills it,
    the kill audit meets custody the store cannot answer, and the reaper's one terminal write carries
    that failed audit (``task_reaper`` not-will_retry shape) straight to ``write_task_result``."""
    from types import SimpleNamespace

    from ouroboros.delegate_custody_reconcile import release_task_runs
    from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
    from ouroboros.utils import append_jsonl
    from supervisor.cancel_publication import _audit_delegated_runs_on_kill, _custody_disclosure_fields

    ctx, _held, _before = discharged_retry(tmp_path, monkeypatch)
    data = custody.custody_root(ctx)
    own = durable_run(ctx, owner=ctx.task_id, run_id="run-own")
    artifact = wait_result(ctx, own.run_id)["output_delivery"]["artifact"]
    release_task_runs(data, ctx.task_id)  # the normal loop exit records its receipts
    reader = ToolContext(repo_dir=ctx.repo_dir, drive_root=data, task_id="physical-reader")
    published = {artifact["sha256"]: True}
    assert load_task_result(data, ctx.task_id)["status"] == "running"
    assert _digests(reader, ctx.task_id, own.run_id) == published
    with monkeypatch.context() as patch:
        custody_unanswerable(patch)
        audit = _audit_delegated_runs_on_kill(SimpleNamespace(DRIVE_ROOT=data, append_jsonl=append_jsonl),
                                              ctx.task_id, trigger="reaper_timeout")
        assert audit["audit_status"] == "failed"
        write_task_result(data, ctx.task_id, "failed", reason_code="timeout",
                          outcome_axes=terminal_outcome_axes(
                              lifecycle="failed", execution=EXECUTION_INFRA_FAILED,
                              reason_code="timeout", review_trigger="supervisor_terminal"),
                          **_custody_disclosure_fields(audit), result="Task killed by timeout after 1s.")
    custody._CUSTODY.clear()
    killed = load_task_result(data, ctx.task_id)
    assert (killed["status"], killed["delegated_runs_unreconciled"]) == (
        "failed", ["delegated_run_state_unknown:custody_log_unreadable"])
    assert _digests(reader, ctx.task_id, own.run_id) == published, "the kill write unpublished the digest"
    assert source_page(reader, ctx.task_id, artifact)["status"] == "available"


# --- precedence controls on the real recorder -------------------------------------

TASK = "merge-task"
LOCATOR = {"run_dir": "/engine/runs/run-a"}
STAGED_A = {"kind": "task_source", "root": "artifact_store", "size": 3, "sha256": "a" * 64,
            "path": f"source_handles/tool_results/delegated-run-run-a-{'a' * 64}.json",
            "task_id": "stager", "complete": True}
STAGED_B = {**STAGED_A, "sha256": "b" * 64,
            "path": f"source_handles/tool_results/delegated-run-run-a-{'b' * 64}.json"}


def _receipt(run_id, **facts):
    return {"run_id": run_id, "state": "succeeded", "model": "m", "profile_id": "",
            "selected_subagent_id": "", **facts}


def _audit(*receipts, open_ids=()):
    return {"task_id": TASK, "trigger": "sweep_refresh", "outcomes": [], "audit_status": "ok",
            "open_run_ids": list(open_ids), "pending_invocation_ids": [],
            "undisposed_patch_run_ids": [], "deferred_project_retirements": [],
            "terminal_runs": sorted(receipts, key=lambda row: row["run_id"]),
            "unreconciled": list(open_ids)}


def _seed(root, audit):
    write_task_result(root, TASK, "failed", reason_code="tool_failure",
                      delegate_terminal_reconciliation=audit,
                      delegated_runs_unreconciled=list(audit["unreconciled"]))


def test_a_missing_locator_never_rewrites_and_a_carried_one_wins(tmp_path):
    record = delegate_terminal.record_terminal_reconciliation
    _seed(tmp_path, _audit(_receipt("run-a", engine_locator=LOCATOR, staged_output=STAGED_A)))
    raw = _stored(tmp_path, TASK)
    # An audit read before the run's locators existed has nothing to write.
    assert record(tmp_path, TASK, _audit(_receipt("run-a"))) is False
    assert _stored(tmp_path, TASK) == raw
    # A key the audit itself carries is the custody row's own fact and wins; the ref it
    # replaces stays published beside it, whichever order the audits land in.
    for own, other in ((STAGED_B, STAGED_A), (STAGED_A, STAGED_B)):
        assert record(tmp_path, TASK, _audit(_receipt("run-a", staged_output=own))) is True
        (row,) = load_task_result(tmp_path, TASK)["delegate_terminal_reconciliation"]["terminal_runs"]
        assert row == _receipt("run-a", engine_locator=LOCATOR, staged_output=own,
                               other_staged_outputs=[other])
    raw = _stored(tmp_path, TASK)
    assert record(tmp_path, TASK, _audit(_receipt("run-a", staged_output=STAGED_A))) is False
    assert _stored(tmp_path, TASK) == raw


@pytest.mark.parametrize("addressed", [False, True])
def test_the_audit_owns_its_debt_lists(tmp_path, addressed):
    record = delegate_terminal.record_terminal_reconciliation
    _seed(tmp_path, _audit(_receipt("run-a", staged_output=STAGED_A), open_ids=["run-b"]))
    assert TASK in obligations.members(tmp_path, "delegated_runs")
    # A fresher audit settled run-b: debt it cleared is never unioned back.
    fresh = _audit(_receipt("run-a"), _receipt("run-b", state="failed"))
    assert record(tmp_path, TASK, fresh, addressed=addressed) is True
    row = load_task_result(tmp_path, TASK)
    audit = row["delegate_terminal_reconciliation"]
    assert row["delegated_runs_unreconciled"] == audit["unreconciled"] == audit["open_run_ids"] == []
    assert audit["terminal_runs"] == [_receipt("run-a", staged_output=STAGED_A),
                                      _receipt("run-b", state="failed")]
    assert TASK not in obligations.members(tmp_path, "delegated_runs")
    # An older audit that still saw run-b open writes that debt, which stays owed for
    # the backfill; only a current-custody refresh keeps the receipt it no longer sees.
    assert record(tmp_path, TASK, _audit(_receipt("run-a"), open_ids=["run-b"]), addressed=addressed) is True
    row = load_task_result(tmp_path, TASK)
    audit = row["delegate_terminal_reconciliation"]
    assert row["delegated_runs_unreconciled"] == audit["open_run_ids"] == ["run-b"]
    assert TASK in obligations.members(tmp_path, "delegated_runs")
    assert [item["run_id"] for item in audit["terminal_runs"]] == (
        ["run-a", "run-b"] if addressed else ["run-a"])
    assert audit["terminal_runs"][0]["staged_output"] == STAGED_A
    assert (row["status"], row["reason_code"]) == ("failed", "tool_failure")


def test_the_result_writer_floor_keeps_every_published_digest(tmp_path):
    """A wholesale envelope (the kill/reap/cancel writers' shape) merges under the result lock."""
    _seed(tmp_path, _audit(_receipt("run-a", staged_output=STAGED_A, other_staged_outputs=[STAGED_B]),
                           _receipt("run-b"), open_ids=["run-c"]))
    # A fresh audit that omits both settled receipts: only the one that published a digest stays.
    stored = write_task_result(tmp_path, TASK, "failed", delegated_runs_unreconciled=[],
                               delegate_terminal_reconciliation=_audit(_receipt("run-c", state="failed")))
    audit = stored["delegate_terminal_reconciliation"]
    assert audit["terminal_runs"] == [
        _receipt("run-a", staged_output=STAGED_A, other_staged_outputs=[STAGED_B]),
        _receipt("run-c", state="failed")]
    assert stored["delegated_runs_unreconciled"] == audit["unreconciled"] == audit["open_run_ids"] == []
    # A failed audit knows no receipt: every stored one stays and its debt is the failure itself.
    failed = {"task_id": TASK, "trigger": "reaper_timeout", "outcomes": [], "audit_status": "failed",
              "unreconciled": ["delegated_run_state_unknown:custody_log_unreadable"]}
    stored = write_task_result(tmp_path, TASK, "failed", delegate_terminal_reconciliation=failed,
                               delegated_runs_unreconciled=list(failed["unreconciled"]))
    assert stored["delegate_terminal_reconciliation"]["terminal_runs"] == audit["terminal_runs"]
    assert stored["delegated_runs_unreconciled"] == failed["unreconciled"]
    assert TASK in obligations.members(tmp_path, "delegated_runs")
    assert (stored["status"], stored["reason_code"]) == ("failed", "tool_failure")
