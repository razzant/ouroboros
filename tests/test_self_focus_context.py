"""The author's own dated focus reaches the actual next Main request."""

from copy import deepcopy
import json

import pytest

from ouroboros.focus import normalize_focus
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.loop_messages import append_context_facts
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_loop_compaction import _ctx
from tests.test_main_authored_context import call, main_loop  # noqa: F401
from tests.test_subscription_main_wait import (  # noqa: F401
    MODEL, main_call, live_wait, setup,
)

pytestmark = pytest.mark.serial


def _facts(messages):
    return next(row["content"] for row in reversed(messages)
                if row.get(HOST_CONTEXT_KIND_KEY) == "ouroboros_context_facts")


def _record(root, text, *, author="task", date="2026-10-10T12:00:00Z"):
    focus = normalize_focus(text, {"reader": "recent_tasks"}, task_id=author, authored_at=date)
    write_task_result(root, "task", "running", focus=focus)
    return focus


def test_main_focus_is_retained_in_sent_tail_after_folding_and_revised(main_loop, monkeypatch):  # noqa: F811
    from ouroboros import loop

    f = main_loop
    f.ctx.task_metadata = {"root_task_id": "authored-main", "delegation_role": "root"}
    write_task_result(f.ctx.drive_root, "authored-main", "running")
    first, revised = "Compare the two retained estimates.", "Use the corrected estimate; the earlier one was withdrawn."
    measured, observations = [], {}
    original = loop._measure_round_main_fit

    def measure(ctx, **kwargs):
        result = original(ctx, **kwargs)
        measured.append(deepcopy(ctx.messages))
        return result

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)

    def after_write(kwargs):
        observations["first"] = deepcopy(load_task_result(f.ctx.drive_root, "authored-main"))
        observations["first_measured"] = deepcopy(measured[-1])
        return call("read_file", {"path": "evidence.txt"}, "read")

    def after_fold(kwargs):
        observations["fold_status"] = f.ctx._context_view_receipt["status"]
        f.incoming.put({"text": "Use the corrected estimate; the earlier one was withdrawn.", "msg_id": "correction"})
        return call("update_focus", {"text": revised, "source_ref": {"reader": "recent_tasks"}}, "revise")

    def final(kwargs):
        observations["last_measured"] = deepcopy(measured[-1])
        return {"content": "done"}

    answer, _, _ = f.run([
        call("update_focus", {"text": first, "source_ref": {"reader": "recent_tasks"}}, "focus"),
        after_write,
        call("compact_context", {"working_note": "The estimates and their exact source are retained.",
                                 "keep_unit_ids": []}, "fold"),
        after_fold,
        final,
    ])
    assert answer == "done"
    stored = observations["first"]["focus"]
    assert stored["text"] == first and stored["source_handle"]
    first_line = _facts(f.inputs[1]["messages"])
    assert first in first_line and stored["authored_at"] in first_line
    assert "self-authored focus" in first_line and "not owner instructions" in first_line
    assert first in str(observations["first_measured"])
    assert f.inputs[1]["messages"][:len(f.inputs[0]["messages"])] == f.inputs[0]["messages"]
    assert observations["fold_status"] == "applied"
    assert first in _facts(f.inputs[3]["messages"])
    last_line = _facts(f.inputs[-1]["messages"])
    assert revised in last_line and first not in last_line
    assert revised in str(observations["last_measured"])
    from ouroboros.llm_attempt import _physical_candidate
    from tests.test_host_context_wire import _wire

    for rows in (_physical_candidate({"messages": f.inputs[-1]["messages"]})["messages"],
                 _wire(monkeypatch, f.inputs[-1]["messages"], f.inputs[-1]["tools"])["messages"]):
        assert any(row.get("content") == last_line for row in rows)
        assert all(HOST_CONTEXT_KIND_KEY not in row for row in rows)


def test_same_round_focus_change_appends_without_rewriting_sent_history(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    _record(tmp_path, "First working thought")
    assert append_context_facts(ctx)
    before = deepcopy(ctx.messages)
    assert "First working thought" in _facts(before)
    assert not append_context_facts(ctx)
    _record(tmp_path, "Revised working thought", date="2026-10-10T12:01:00Z")
    assert append_context_facts(ctx)
    assert ctx.messages[:len(before)] == before
    assert "Revised working thought" in _facts(ctx.messages)
    assert not append_context_facts(ctx)


def test_cold_same_task_uses_canonical_focus_not_execution_root_copy(tmp_path):
    canonical, execution = tmp_path / "canonical", tmp_path / "execution"
    original = _record(canonical, "Canonical current thought")
    _record(execution, "Outdated execution copy")
    ctx = _ctx(execution)
    ctx.context_fit_plan = None
    ctx.tools._ctx.task_metadata = {"budget_drive_root": str(canonical)}
    assert append_context_facts(ctx)
    assert "Canonical current thought" in _facts(ctx.messages)
    assert "Outdated execution copy" not in _facts(ctx.messages)
    cold = _ctx(execution)
    cold.context_fit_plan = None
    cold.tools._ctx.task_metadata = json.loads(json.dumps(ctx.tools._ctx.task_metadata))
    assert append_context_facts(cold)
    assert "Canonical current thought" in _facts(cold.messages)
    assert load_task_result(canonical, "task")["focus"] == original


@pytest.mark.parametrize("author,shown", [("task", True), ("predecessor", False)])
def test_focus_keeps_its_author_across_a_new_task_id(tmp_path, author, shown):
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    original = _record(tmp_path, "Dated predecessor thought", author=author)
    assert append_context_facts(ctx)
    assert ("Dated predecessor thought" in _facts(ctx.messages)) is shown
    assert load_task_result(tmp_path, "task")["focus"] == original


def test_missing_focus_is_optional_and_failed_read_is_visible(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    write_task_result(tmp_path, "task", "running")
    assert append_context_facts(ctx)
    assert "self-authored focus" not in _facts(ctx.messages)

    def unavailable(*args, **kwargs):
        raise OSError("unreadable focus source")

    monkeypatch.setattr("ouroboros.task_results.load_task_result", unavailable)
    ctx.round_idx += 1
    assert append_context_facts(ctx)
    assert "own focus unavailable" in _facts(ctx.messages)


def test_unreadable_record_is_not_quarantined_and_recovery_refreshes_same_round(tmp_path):
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    _record(tmp_path, "Retain this dated thought")
    path = tmp_path / "task_results" / "task.json"
    original = path.read_bytes()
    assert append_context_facts(ctx)
    assert "Retain this dated thought" in _facts(ctx.messages)
    before = deepcopy(ctx.messages)
    path.write_bytes(b"{broken record")
    assert append_context_facts(ctx)
    assert path.read_bytes() == b"{broken record"
    assert not (path.parent / "quarantine").exists()
    assert "own focus unavailable" in _facts(ctx.messages)
    assert ctx.messages[:len(before)] == before
    assert not append_context_facts(ctx)
    path.write_bytes(original)
    assert append_context_facts(ctx)
    assert "Retain this dated thought" in _facts(ctx.messages)


@pytest.mark.parametrize("destination", [MODEL, "openai::gpt-5.6-terra"])
def test_waiting_route_reprepare_reads_new_focus_into_the_measured_request(main_call, destination):  # noqa: F811
    from ouroboros import loop, usage_accounting as ua
    from ouroboros.loop_model_call import _reprepare_waiting_main

    ctx = main_call[0]
    focus = normalize_focus("Inspect the newly corrected source.", {"reader": "recent_tasks"},
                            task_id=ctx.task_id, authored_at="2026-10-10T12:02:00Z")
    write_task_result(ctx.drive_root, ctx.task_id, "running", focus=focus)
    physical = loop._physical_context_for_fit(loop._measure_round_main_fit(ctx, automatic_pass_used=False))
    with ua.bind_physical_attempt_context(physical):
        prepared = _reprepare_waiting_main(ctx, {"messages": deepcopy(ctx.messages), "model": destination,
                                                "model_role": "main", "tools": deepcopy(ctx.tool_schemas)})
    assert focus["text"] in _facts(ctx.messages)
    assert any(focus["text"] in str(row.get("content")) for row in prepared.kwargs["messages"])
    assert ctx.accumulated_usage["_context_prompt_estimate"] > 0
