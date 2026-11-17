"""A delivered answer lists its direct children once per wait-free finalization segment (#1528).

The real final-answer gate (``loop.run_llm_loop`` -> ``loop_delivery._no_tool_final_answer``) runs a scripted model
over one parent with four dispositioned children. ``task_status.select_lineage`` runs once per real listing and
``task_custody.store_artifact_view`` once per materialized child. A child change at a boundary between segments
(the services' stop, the result review, admission's drain) or inside one reaches the delivery decision before the
answer leaves; a content-only change without registration keeps the child's identity and delivers.
"""
from __future__ import annotations

import contextlib
import json
import queue

import pytest

CHILDREN = ("child1", "child2", "child3", "child4")


def _child(root, task_id):
    from ouroboros.artifacts import (_register_task_artifact_records, artifact_record, store_task_artifact_bytes,
                                     task_artifact_dir_path)
    from ouroboros.task_results import write_task_result

    store = task_artifact_dir_path(root, task_id, create=True)
    report = store_task_artifact_bytes(root, task_id, "report.md", f"report of {task_id}".encode())
    (store / "notes.md").write_bytes(b"registered notes v1")
    _register_task_artifact_records(store, [artifact_record(store / "notes.md")])
    (store / "scratch.txt").write_bytes(b"unregistered scratch")
    write_task_result(root, task_id, "completed", parent_task_id="parent1", root_task_id="parent1",
                      delegation_role="subagent", result=f"result of {task_id}", artifact_status="ready",
                      artifacts=[{"kind": "report", "name": "report.md", "path": str(store / "report.md"),
                                  "sha256": report["sha256"]}])
    return store


def _dispose(root, task_id):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.task_tree_ledger import tree_ledger_append
    from ouroboros.tools.join_ledger import _child_result_sha256

    row = load_effective_task_result(root, task_id)
    written = tree_ledger_append("parent1", "decision", "the parent consumed the handoff", task_id="parent1",
                                 role="orchestrator", payload={"type": "child_result_disposition",
                                                               "child_task_id": task_id, "disposition": "integrated",
                                                               "child_result_sha256": _child_result_sha256(row)},
                                 allow_child_result_disposition=True, data_root=root)
    assert written.startswith("OK:"), written


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    stores = {task_id: _child(root, task_id) for task_id in CHILDREN}
    for task_id in CHILDREN:
        _dispose(root, task_id)
    return root, stores


class Listings:
    """Count the real listings and artifact views; every wrapper returns what the reader returned."""

    def __init__(self, monkeypatch):
        from ouroboros import finalization_timing, task_custody, task_status

        self.rows, self.views = [], 0
        real_select, real_view = task_status.select_lineage, task_custody.store_artifact_view

        def select(*args, **kwargs):
            self.rows.append({"pass": finalization_timing._FINALIZATION_CHILDREN.get(),
                              "direct": kwargs.get("direct_only")})
            return real_select(*args, **kwargs)

        def view(*args, **kwargs):
            self.views += 1
            return real_view(*args, **kwargs)

        monkeypatch.setattr(task_status, "select_lineage", select)
        monkeypatch.setattr(task_custody, "store_artifact_view", view)

    def passes(self):
        seen = []
        for row in self.rows:
            if row["pass"] is not None and not any(row["pass"] is known for known in seen):
                seen.append(row["pass"])
        return seen


def _finish(answer):
    return {"content": None, "tool_calls": [{"id": "finish-selection", "type": "function", "function": {
        "name": "finish_task", "arguments": json.dumps({"action": "finish", "answer": answer})}}]}


def run_gate(root, monkeypatch, answers, *, admission=False):
    import ouroboros.loop as loop
    from ouroboros import finalization_timing
    from ouroboros.tools.registry import ToolRegistry

    calls, script = [], iter(answers)

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        calls.append(len(request_messages))
        finalization_timing.mark_last_answer({})
        answer = next(script)
        return ({"role": "assistant", **answer} if isinstance(answer, dict)
                else {"role": "assistant", "content": answer}), 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=root, drive_root=root)
    registry._ctx.task_id = "parent1"
    registry._ctx._owner_directives = []
    registry._ctx.task_metadata = {"budget_drive_root": str(root), "root_task_id": "parent1"}
    if admission:  # the ordinary close: admission's lock, then one drain before the last decision
        import threading
        from types import SimpleNamespace

        registry._ctx.owner_message_admission_lock = threading.Lock()
        registry._ctx.owner_message_admission_agent = SimpleNamespace(
            _busy=True, _current_task_id="parent1", _accepting_owner_messages=True)
    result, _usage, trace = loop.run_llm_loop(
        messages=[{"role": "user", "content": "do the work"}], tools=registry, llm=FakeLLM(), drive_logs=root,
        emit_progress=lambda *_a, **_k: None, incoming_messages=queue.Queue(), task_id="parent1", drive_root=root)
    return {"result": result, "model_calls": len(calls), "candidate": trace.get("delivery_candidate") or {},
            "dispositions": trace.get("child_result_dispositions")}


def _holds(monkeypatch):
    from ouroboros import loop_delivery

    seen, real = [], loop_delivery.observe_completion_hold

    def observe(tools, ctx, origin, reason, subreason=""):
        seen.append(reason)
        return real(tools, ctx, origin, reason, subreason)

    monkeypatch.setattr(loop_delivery, "observe_completion_hold", observe)
    return seen


def test_a_delivered_answer_lists_its_children_once_per_segment(tree, monkeypatch):
    from ouroboros import loop_delivery

    root, _stores = tree
    listings = Listings(monkeypatch)
    scoped = run_gate(root, monkeypatch, ["Final answer."])
    scoped_rows, scoped_views, passes = list(listings.rows), listings.views, listings.passes()
    assert scoped["result"] == "Final answer." and scoped["model_calls"] == 1
    # Candidate/gates and post-service reducers share reads; the final evidence check reads fresh.
    assert len(passes) == 2
    assert [sum(row["pass"] is known for row in scoped_rows) for known in passes] == [1, 1]
    assert all(row["direct"] is True for row in scoped_rows if row["pass"] is not None), "a direct read stays direct"

    listings.rows.clear()
    listings.views = 0
    with monkeypatch.context() as patch:  # the same answer with every reader listing for itself
        patch.setattr(loop_delivery, "finalization_child_scope", contextlib.nullcontext)
        unscoped = run_gate(root, monkeypatch, ["Final answer."])
    assert not listings.passes()
    assert unscoped["result"] == scoped["result"]
    keep = ("content_sha256", "evidence_fingerprint", "subject_sha256", "evidence_current", "degraded")
    assert {key: unscoped["candidate"].get(key) for key in keep} == {key: scoped["candidate"].get(key) for key in keep}
    assert scoped["candidate"]["evidence_current"] is True
    assert unscoped["dispositions"] == scoped["dispositions"]
    assert len(scoped_rows) < len(listings.rows) and scoped_views < listings.views, (
        len(scoped_rows), len(listings.rows), scoped_views, listings.views)


def _change(root, stores, kind):
    from ouroboros.artifacts import _register_task_artifact_records, artifact_record
    from ouroboros.task_results import write_task_result

    store = stores["child1"]
    if kind == "registration":
        (store / "notes.md").write_bytes(b"registered notes v2, registered again")
        _register_task_artifact_records(store, [artifact_record(store / "notes.md")])
    elif kind == "status":
        write_task_result(root, "child1", "completed", artifact_status="failed")
    elif kind == "late_output":
        (store / "late-output.txt").write_bytes(b"a late output after the terminal row")
    elif kind == "deletion":
        (store / "scratch.txt").unlink()
    elif kind == "content_only":
        (store / "notes.md").write_bytes(b"registered notes, rewritten without registration")
        return
    _dispose(root, "child1")  # the parent's disposition stays current: only the evidence moved


def _inject_once(monkeypatch, owner, name, act, *, after=False, when=lambda *a, **k: True):
    real, done = getattr(owner, name), []

    def wrapper(*args, **kwargs):
        if done or not when(*args, **kwargs):
            return real(*args, **kwargs)
        done.append(True)
        if after:
            result = real(*args, **kwargs)
            act()
            return result
        act()
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapper)
    return done


@pytest.mark.parametrize("boundary,kind", [
    ("services", "registration"), ("review", "status"), ("drain", "late_output"), ("services", "deletion"),
    ("inside_first_segment", "status"),
    ("last_projection", "status"), ("last_projection", "registration"),
    ("last_projection", "late_output"), ("last_projection", "deletion"),
])
def test_a_child_change_between_or_inside_segments_holds_the_retained_answer(tree, monkeypatch, boundary, kind):
    import ouroboros.loop as loop
    from ouroboros.tools import review_result
    from ouroboros.tools.join_ledger import _child_result_sha256, _load_direct_child_results

    root, stores = tree
    holds = _holds(monkeypatch)
    before = {row["task_id"]: _child_result_sha256(row) for row in _load_direct_child_results(root, "parent1", "parent1")}
    seen_inside = []

    def act():
        _change(root, stores, kind)
        if boundary == "inside_first_segment":  # this pass's projection still answers the earlier listing
            seen_inside.append({row["task_id"]: _child_result_sha256(row)
                                for row in _load_direct_child_results(root, "parent1", "parent1")})

    if boundary == "services":
        done = _inject_once(monkeypatch, loop, "_finalize_task_services", act)
    elif boundary == "review":
        done = _inject_once(monkeypatch, review_result, "result_review_at_delivery", act)
    elif boundary == "drain":
        done = _inject_once(monkeypatch, loop, "_drain_incoming_messages", act,  # the final close's drain only
                            when=lambda *a, **k: getattr(getattr(k.get("owner_ctx"), "owner_message_admission_agent",
                                                                 None), "_accepting_owner_messages", True) is False)
    elif boundary == "last_projection":
        projections = []

        def third_projection(*args, **kwargs):
            projections.append(True)
            return len(projections) == 3

        done = _inject_once(monkeypatch, loop, "_project_child_result_dispositions", act,
                            after=True, when=third_projection)
    else:  # the finish door's last check inside the first segment
        done = _inject_once(monkeypatch, loop, "_maybe_inject_finalization_nudges", act, after=True)
    gate = run_gate(root, monkeypatch, ["First answer.", _finish("Second answer.")], admission=boundary == "drain")
    after = {row["task_id"]: _child_result_sha256(row) for row in _load_direct_child_results(root, "parent1", "parent1")}
    assert done and after["child1"] != before["child1"], "the change moved the child's identity"
    if boundary == "inside_first_segment":
        assert seen_inside == [before], "inside a segment its one listing answers"
    assert "evidence_changed" in holds, holds
    assert gate["result"] == "Second answer." and gate["model_calls"] == 2
    assert gate["candidate"]["evidence_current"] is True


def test_a_content_only_change_keeps_the_childs_identity_and_delivers(tree, monkeypatch):
    import ouroboros.loop as loop
    from ouroboros.tools.join_ledger import _child_result_sha256, _load_direct_child_results

    root, stores = tree
    holds = _holds(monkeypatch)
    before = {row["task_id"]: _child_result_sha256(row) for row in _load_direct_child_results(root, "parent1", "parent1")}
    done = _inject_once(monkeypatch, loop, "_finalize_task_services", lambda: _change(root, stores, "content_only"))
    gate = run_gate(root, monkeypatch, ["First answer."])
    after = {row["task_id"]: _child_result_sha256(row) for row in _load_direct_child_results(root, "parent1", "parent1")}
    assert done and after == before
    assert gate["result"] == "First answer." and gate["model_calls"] == 1 and holds == []
