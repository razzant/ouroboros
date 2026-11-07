"""The author's own dated focus reaches the actual next Main request."""

from copy import deepcopy
import json

import pytest

from ouroboros.focus import normalize_focus
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.loop_messages import append_context_facts
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_shared import read_result, result_signature, unreadable_result
from tests.test_loop_compaction import _ctx
from tests.test_main_authored_context import call, main_loop  # noqa: F401
from tests.test_subscription_main_wait import (  # noqa: F401
    MODEL, main_call, live_wait, setup,
)



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


def test_unreadable_record_is_not_quarantined_and_recovery_refreshes_same_round(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    _record(tmp_path, "Retain this dated thought")
    original = read_result(tmp_path, "task")
    signature = result_signature(tmp_path, "task")
    assert append_context_facts(ctx)
    assert "Retain this dated thought" in _facts(ctx.messages)
    before = deepcopy(ctx.messages)
    outage = unreadable_result(monkeypatch, "task", root=tmp_path)
    assert append_context_facts(ctx)
    assert not (tmp_path / "task_results" / "quarantine").exists()
    assert "own focus unavailable" in _facts(ctx.messages)
    assert "Retain this dated thought" not in _facts(ctx.messages)
    assert ctx.messages[:len(before)] == before
    assert not append_context_facts(ctx)
    outage["on"] = False
    assert read_result(tmp_path, "task") == original
    assert result_signature(tmp_path, "task") == signature
    assert append_context_facts(ctx)
    # The first row still carries the recovered, unchanged focus whole: the new row names it in one line.
    assert "self-authored focus unchanged" in _facts(ctx.messages)
    assert not append_context_facts(ctx)


def test_an_unchanged_focus_is_whole_once_while_an_intact_facts_row_in_view_carries_it(tmp_path):
    """C2 S3 (06.11.2026): an unchanged focus is not repeated every round; a change or a lost carrier shows it whole."""
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    _record(tmp_path, "Retain this dated thought")
    assert append_context_facts(ctx) and "Retain this dated thought" in _facts(ctx.messages)
    ctx.round_idx += 1
    assert append_context_facts(ctx)
    line = _facts(ctx.messages)
    assert "Retain this dated thought" not in line and "self-authored focus unchanged" in line and "fingerprint" in line
    for row in ctx.messages:  # the carrying row is no longer the host state it was produced as
        if "Retain this dated thought" in str(row.get("content")):
            row["content"] += " edited"
    ctx.round_idx += 1
    assert append_context_facts(ctx) and "Retain this dated thought" in _facts(ctx.messages)
    _record(tmp_path, "A changed thought")
    ctx.round_idx += 1
    assert append_context_facts(ctx) and "A changed thought" in _facts(ctx.messages)


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


def _sampled_at(monkeypatch, iso):
    """Fix the facts row's one sample instant (``append_context_facts``) at ``iso``."""
    from types import SimpleNamespace

    from ouroboros import loop_messages
    from ouroboros.deadline_utils import parse_deadline_ts

    monkeypatch.setattr(loop_messages, "time", SimpleNamespace(time=lambda: parse_deadline_ts(iso).timestamp()))


def test_the_focus_line_names_its_reader_and_its_age_at_the_rows_sample_instant(tmp_path, monkeypatch):
    """The age is a calculation over the dated fact at the row's sample instant, not a kept clock: an
    unchanged reprepare keeps the sent row (and the cache), a new row measures again, and the clock
    segment of the same row is sampled at the same instant."""
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    _record(tmp_path, "Comparing the two estimates")
    _sampled_at(monkeypatch, "2026-10-10T15:05:00Z")
    assert append_context_facts(ctx)
    line = _facts(ctx.messages)
    assert ("self-authored focus, the line my human reads under this work's name, written 3 h 5 min ago "
            "(dated data, not owner instructions): {") in line
    assert '"authored_at": "2026-10-10T12:00:00Z"' in line and "Comparing the two estimates" in line
    _sampled_at(monkeypatch, "2026-10-10T15:20:00Z")
    assert not append_context_facts(ctx), "the age is outside the row's identity: a reprepare keeps the row"
    assert _facts(ctx.messages) == line
    ctx.round_idx += 1
    assert append_context_facts(ctx)
    assert "written 3 h 20 min ago" in _facts(ctx.messages)


@pytest.mark.parametrize("sampled,said", [("2026-10-10T12:00:00Z", "written less than a minute ago"),
                                         ("2026-10-10T12:00:59Z", "written less than a minute ago"),
                                         ("2026-10-10T12:01:00Z", "written 1 min ago")])
def test_a_line_younger_than_a_minute_says_so_and_never_a_minute(tmp_path, monkeypatch, sampled, said):
    """The age reads at one-minute granularity (``ago_text``, the wake's lines too): a line written under a minute
    ago says so, a minute old says one."""
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    _record(tmp_path, "Just started")
    _sampled_at(monkeypatch, sampled)
    assert append_context_facts(ctx)
    assert f"name, {said} (dated data" in _facts(ctx.messages)


def test_an_unreadable_authoring_time_is_said_not_guessed(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    _record(tmp_path, "Thought with a broken stamp", date="last tuesday")
    _sampled_at(monkeypatch, "2026-10-10T15:05:00Z")
    assert append_context_facts(ctx)
    line = _facts(ctx.messages)
    assert "the line my human reads under this work's name, written at an unreadable time (dated data" in line
    assert "ago" not in line.split("self-authored focus", 1)[1].split("{", 1)[0]


_POOLED, _DIRECT = {"_execution_owner": {"kind": "pooled"}}, {"_execution_owner": {"kind": "direct"}}


@pytest.mark.parametrize("metadata,direct,told", [
    ({"delegation_role": "root"}, False, True),  # an independent root's work card would carry a line: a fact
    ({"delegation_role": "root"}, True, False),  # a live direct turn of Main (a reply, a wake) answers in chat
    ({"delegation_role": "root", **_DIRECT}, True, False),  # the same, its carrier recorded
    ({}, True, False),  # a direct turn is a root by its door alone: it still answers in chat
    ({"delegation_role": "subagent", "parent_task_id": "parent", **_POOLED}, False, False),  # a child may not publish
    ({"delegation_role": "root", "_presence_turn": True}, False, False),  # a speaker's line is its place's
    ({}, False, False),  # no root marker is no publishing authority
])
def test_a_missing_line_is_named_only_to_a_root_whose_work_card_would_carry_one(tmp_path, metadata, direct, told):
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    ctx.tools._ctx.task_metadata = dict(metadata)
    ctx.tools._ctx.is_direct_chat = direct
    write_task_result(tmp_path, "task", "running")
    assert append_context_facts(ctx)
    line = _facts(ctx.messages)
    assert ("self-authored focus: none; my human reads nothing under this work's name" in line) is told
    if not told:
        assert "self-authored focus" not in line
    assert not append_context_facts(ctx), "an absent line has one identity: no new row per reprepare"


def test_a_direct_turn_still_reads_the_line_it_published(tmp_path, monkeypatch):
    """Only the absent line goes unsaid on a direct turn: a line it wrote is dated data like any root's."""
    ctx = _ctx(tmp_path)
    ctx.context_fit_plan = None
    ctx.tools._ctx.task_metadata = {"delegation_role": "root"}
    ctx.tools._ctx.is_direct_chat = True
    _record(tmp_path, "Answering the estimate question")
    _sampled_at(monkeypatch, "2026-10-10T12:05:00Z")
    assert append_context_facts(ctx)
    line = _facts(ctx.messages)
    assert "self-authored focus, the line my human reads under this work's name, written 5 min ago" in line
    assert "Answering the estimate question" in line and "focus: none" not in line
