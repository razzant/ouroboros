"""A saved row's ``outcome_axes.lifecycle`` names the status its write actually landed.

The loop derives ``lifecycle: completed`` for every attempt that ended; the pipeline
then settles ``failed`` for a failed execution or an already failed row, the exception
writer stores ``failed`` over the same derivation, and the writer's projector and
monotonic guard may land yet another word. Readers normalize the pair, but the raw row
and the attempt's own events carried the contradiction. These checks read the actual
SQLite body, the public projection and the events the real pipeline emits. Pre-store
events state the status observed before the store (the outbox stamp's word, not a
promise about a later race); post-store ones state the stored row. The stored
``loop_outcome`` stays the loop's own evidence.
"""
from __future__ import annotations

import copy
import json
import sqlite3
import time
from types import SimpleNamespace

import pytest

from ouroboros.agent_task_pipeline import emit_task_results
from ouroboros.outcomes import public_task_result, terminal_outcome_axes
from ouroboros.task_results import load_task_result, write_task_result
from supervisor.events_worker_reports import _handle_task_metrics
from tests._store_shared import read_result, result_signature, seed_result


def _raw_body(root, task_id):
    """The stored JSON exactly as the store holds it: no reader, no normalizer."""
    with sqlite3.connect(root / "state" / "current.sqlite") as conn:
        (body,) = conn.execute("SELECT body FROM task_bodies WHERE task_id=?", (task_id,)).fetchone()
    return json.loads(body)


def _without_lifecycle(axes):
    return {key: value for key, value in axes.items() if key != "lifecycle"}


@pytest.mark.parametrize("usage, text, seeded, expected", [
    ({"rounds": 1, "execution_status": "infra_failed", "reason_code": "provider_unavailable"},
     "Preserved incomplete work.", None, "failed"),
    # An already failed row keeps failed whatever this attempt's execution says.
    ({"rounds": 1}, "Done.", "failed", "failed"),
    # Positive control: a clean finish is completed on every carrier, before and after.
    ({"rounds": 1}, "Done.", None, "completed"),
], ids=["infra_failed", "existing_failed_row", "completed"])
def test_carriers_name_the_resolved_terminal_status(tmp_path, monkeypatch, usage, text, seeded, expected):
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", lambda *args, **kwargs: None)
    task = {"id": "parity-task", "chat_id": 1, "type": "task", "text": "Owner objective",
            "_is_direct_chat": True, "_skip_post_task_synthesis": True}
    if seeded:
        write_task_result(tmp_path, task["id"], seeded, result="earlier failure")
    env = SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    pending = []
    emit_task_results(env, None, None, pending, task, text, dict(usage),
                      {"tool_calls": [], "reasoning_notes": []}, time.time(), logs)

    raw = _raw_body(tmp_path, task["id"])
    assert raw["status"] == expected
    assert raw["outcome_axes"]["lifecycle"]["status"] == expected
    # The loop's own derivation is retained as written: evidence, not the row's word.
    assert raw["loop_outcome"]["outcome_axes"]["lifecycle"]["status"] == "completed"
    loop_axes = raw["loop_outcome"]["outcome_axes"]
    assert raw["outcome_axes"]["execution"] == loop_axes["execution"]
    assert raw["reason_code"] == raw["loop_outcome"]["reason_code"]
    public = public_task_result(load_task_result(tmp_path, task["id"]))
    assert public["status"] == public["outcome_axes"]["lifecycle"]["status"] == expected

    evaluation = next(json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()
                      if json.loads(line).get("type") == "task_eval")
    metric = next(row for row in pending if row["type"] == "task_metrics")
    done = next(row for row in pending if row["type"] == "task_done")
    forwarded = []
    _handle_task_metrics(metric, SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], append_jsonl=lambda *_a, **_k: None,
        bridge=SimpleNamespace(push_log=forwarded.append)))
    final = next(row for row in pending if row.get("type") == "send_message")
    carriers = [evaluation["outcome_axes"], metric["outcome_axes"], done["outcome_axes"],
                forwarded[0]["outcome_axes"]]
    carriers += [final["progress_meta"]["outcome_axes"]] if "outcome_axes" in final.get("progress_meta", {}) else []
    for axes in carriers:
        assert axes["lifecycle"]["status"] == expected
        # Only the lifecycle word is aligned; the attempt's other axes stay its own.
        assert _without_lifecycle(axes) == _without_lifecycle(metric["outcome_axes"])
    assert done["status"] == expected
    assert metric["outcome_axes"]["execution"]["status"] == loop_axes["execution"]["status"]


@pytest.mark.parametrize("cancel_lands", ["before_attempt", "between_observation_and_store"])
def test_a_cancel_that_wins_keeps_its_row_while_pre_store_events_keep_their_observation(
        tmp_path, monkeypatch, cancel_lands):
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", lambda *args, **kwargs: None)
    task = {"id": "late-task", "chat_id": 1, "type": "task", "text": "Owner objective",
            "_is_direct_chat": True, "_skip_post_task_synthesis": True}

    def cancel():
        write_task_result(tmp_path, task["id"], "cancelled", reason_code="owner_cancelled",
                          outcome_axes=terminal_outcome_axes(lifecycle="cancelled", execution="cancelled",
                                                             reason_code="owner_cancelled"))

    if cancel_lands == "before_attempt":
        cancel()
    else:
        write_task_result(tmp_path, task["id"], "running", result="started")

        def racing_review_evidence(*_args, **_kwargs):
            cancel()  # after task_eval/task_metrics and the final frame, before the result store
            return {}
        monkeypatch.setattr("ouroboros.review_evidence.collect_review_evidence", racing_review_evidence)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    pending = []
    emit_task_results(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), None, None, pending, task,
                      "Preserved incomplete work.",
                      {"rounds": 1, "execution_status": "infra_failed", "reason_code": "provider_unavailable"},
                      {"tool_calls": [], "reasoning_notes": []}, time.time(), logs)
    raw = _raw_body(tmp_path, task["id"])
    # The settled cancel wins; the late failed write was refused whole.
    assert raw["status"] == raw["outcome_axes"]["lifecycle"]["status"] == "cancelled"
    assert raw["outcome_axes"]["execution"]["status"] == "cancelled" and "loop_outcome" not in raw
    done = next(row for row in pending if row["type"] == "task_done")
    assert done["status"] == done["outcome_axes"]["lifecycle"]["status"] == "cancelled"
    assert done["outcome_axes"]["execution"]["status"] == "infra_failed", "the attempt's own evaluation stays"
    # Pre-store carriers state the observation their stamp made, not the later winner.
    evaluation = next(json.loads(line) for line in (logs / "events.jsonl").read_text().splitlines()
                      if json.loads(line).get("type") == "task_eval")
    metric = next(row for row in pending if row["type"] == "task_metrics")
    final = next(row for row in pending if row.get("type") == "send_message")
    for axes in (evaluation["outcome_axes"], metric["outcome_axes"], final["progress_meta"]["outcome_axes"]):
        assert axes["lifecycle"]["status"] == "failed"


def test_the_exception_writer_stores_a_failed_lifecycle_and_keeps_its_loop_evidence(tmp_path):
    from ouroboros.agent import _task_exception_terminal

    env = SimpleNamespace(drive_root=tmp_path)
    logs = tmp_path / "logs"
    logs.mkdir(exist_ok=True)
    boom = RuntimeError("boom")
    boom._ouroboros_loop_usage = {"rounds": 1, "execution_id": "exec_boom"}
    boom._ouroboros_loop_trace = {"reasoning_notes": [], "tool_calls": [
        {"tool": "write_file", "result": "ok", "trace_ref": {"call_id": "tool_write_1"}}]}
    _task_exception_terminal(env, {"id": "exc-task", "type": "task"}, boom, logs)
    raw = _raw_body(tmp_path, "exc-task")
    assert raw["status"] == raw["outcome_axes"]["lifecycle"]["status"] == "failed"
    assert raw["outcome_axes"]["execution"]["status"] == "infra_failed"
    assert raw["reason_code"] == "task_exception"
    assert raw["loop_outcome"]["outcome_axes"]["lifecycle"]["status"] == "completed"
    # The loop's trace refs are stored once, at the top level; the saved outcome keeps the rest.
    assert raw["trace_refs"]["execution_id"] == "exec_boom"
    assert [ref["call_id"] for ref in raw["trace_refs"]["tool_call_refs"]] == ["tool_write_1"]
    assert "trace_refs" not in raw["loop_outcome"] and raw["loop_outcome"]["usage"]["total_rounds"] == 1
    # A late exception at the worker boundary never rewrites a row that already settled.
    write_task_result(tmp_path, "done-task", "completed", outcome_axes=_axes("completed", execution="ok"))
    signature, settled = result_signature(tmp_path, "done-task"), _raw_body(tmp_path, "done-task")
    _task_exception_terminal(env, {"id": "done-task", "type": "task"}, RuntimeError("late"), logs,
                             keep_terminal=True)
    assert result_signature(tmp_path, "done-task") == signature and _raw_body(tmp_path, "done-task") == settled


def _axes(lifecycle, execution="infra_failed"):
    axes = terminal_outcome_axes(lifecycle=lifecycle, execution=execution, reason_code="provider_unavailable")
    axes["lifecycle"]["observed_at"] = "kept"  # unknown lifecycle detail survives alignment
    axes["objective"] = {"status": "fail", "source": "task_acceptance_review", "reason": "kept"}
    return axes


def test_the_writer_names_the_landed_status_without_touching_the_caller_or_other_axes(tmp_path):
    incoming = _axes("completed")
    before = copy.deepcopy(incoming)
    write_task_result(tmp_path, "w-1", "failed", outcome_axes=incoming, reason_code="provider_unavailable")
    raw = _raw_body(tmp_path, "w-1")
    assert raw["status"] == raw["outcome_axes"]["lifecycle"]["status"] == "failed"
    assert raw["outcome_axes"]["lifecycle"]["observed_at"] == "kept"
    assert _without_lifecycle(raw["outcome_axes"]) == _without_lifecycle(before)
    assert raw["reason_code"] == "provider_unavailable"
    assert incoming == before, "the caller's axes are never mutated"


def test_a_projected_status_is_the_one_its_axes_name(tmp_path):
    write_task_result(tmp_path, "p-1", "running", result="started")
    seen = []

    def settle(current, incoming):
        seen.append(incoming["outcome_axes"]["lifecycle"]["status"])
        return {**incoming, "status": "cancelled"}

    write_task_result(tmp_path, "p-1", "completed", _field_projector=settle, outcome_axes=_axes("completed"))
    raw = _raw_body(tmp_path, "p-1")
    assert seen == ["completed"], "the projector sees what the caller sent"
    assert raw["status"] == raw["outcome_axes"]["lifecycle"]["status"] == "cancelled"
    assert raw["outcome_axes"]["execution"]["status"] == "infra_failed"


def test_a_status_change_that_carries_no_axes_renames_the_kept_ones(tmp_path):
    write_task_result(tmp_path, "r-1", "running", outcome_axes=_axes("running", execution="pending"))
    write_task_result(tmp_path, "r-1", "failed", result="worker lost")
    raw = _raw_body(tmp_path, "r-1")
    assert raw["status"] == raw["outcome_axes"]["lifecycle"]["status"] == "failed"
    assert raw["outcome_axes"]["lifecycle"]["observed_at"] == "kept"
    assert _without_lifecycle(raw["outcome_axes"]) == _without_lifecycle(_axes("running", execution="pending"))
    # Same-status enrichment afterwards changes only what it carries.
    write_task_result(tmp_path, "r-1", "failed", result="worker lost, logs kept")
    assert _raw_body(tmp_path, "r-1")["outcome_axes"] == raw["outcome_axes"]


def test_a_projector_no_op_and_a_rejected_late_write_change_nothing(tmp_path):
    write_task_result(tmp_path, "g-1", "cancelled", outcome_axes=_axes("cancelled", execution="cancelled"))
    settled = _raw_body(tmp_path, "g-1")
    signature = result_signature(tmp_path, "g-1")
    write_task_result(tmp_path, "g-1", "completed", _field_projector=lambda current, incoming: None,
                      outcome_axes=_axes("completed", execution="ok"))
    # Natural completion over a settled cancel and a stale running mirror are both refused.
    write_task_result(tmp_path, "g-1", "completed", outcome_axes=_axes("completed", execution="ok"))
    write_task_result(tmp_path, "g-1", "running", outcome_axes=_axes("running", execution="ok"))
    assert result_signature(tmp_path, "g-1") == signature
    assert _raw_body(tmp_path, "g-1") == settled
    assert settled["outcome_axes"]["lifecycle"]["status"] == "cancelled"


def test_stored_rows_are_not_migrated_and_odd_axes_stay_as_written(tmp_path):
    # A historical row with the old contradiction: an enrichment write that carries no
    # axes leaves them exactly as stored; the public reader already resolves the pair.
    legacy = {"status": "failed", "result": "old", "outcome_axes": _axes("completed")}
    seed_result(tmp_path, "h-1", legacy)
    write_task_result(tmp_path, "h-1", "failed", result="enriched")
    stored = read_result(tmp_path, "h-1")
    assert stored["result"] == "enriched"
    assert stored["outcome_axes"] == legacy["outcome_axes"]
    assert public_task_result(stored)["outcome_axes"]["lifecycle"]["status"] == "failed"
    # Axes without a lifecycle mapping, or with a malformed one, are kept as sent.
    for task_id, axes in (("h-2", {"execution": {"status": "failed"}}),
                          ("h-3", {"lifecycle": "completed", "execution": {"status": "failed"}})):
        write_task_result(tmp_path, task_id, "failed", outcome_axes=axes)
        assert read_result(tmp_path, task_id)["outcome_axes"] == axes


def test_lifecycle_named_copies_only_when_the_word_differs():
    from ouroboros.task_results import lifecycle_named

    axes = {"lifecycle": {"status": "failed", "extra": 1}, "review": {"status": "skipped"}}
    assert lifecycle_named(axes, "failed") is axes
    renamed = lifecycle_named(axes, "cancelled")
    assert renamed == {"lifecycle": {"status": "cancelled", "extra": 1}, "review": {"status": "skipped"}}
    assert axes["lifecycle"]["status"] == "failed" and renamed["review"] is axes["review"]
    for odd in (None, [], {}, {"lifecycle": None}, {"lifecycle": "failed"}):
        assert lifecycle_named(odd, "cancelled") is odd
