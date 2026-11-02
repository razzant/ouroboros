"""A Place focus carries no invented round cap; a limit that was really set still binds.

The Places producer (``presence_runner._build_task``) admits a focus without ``inline_max_rounds``;
the agent bootstrap reads only a cap that a retained pre-Places turn actually carries. These
scenarios run the real Place producer, agent bootstrap, loop, terminal pipeline and runner
(``install_bootstrap_harness``: only inference is scripted): a focus works past ten productive
rounds and finishes on the ordinary path, while the owner's global ``OUROBOROS_MAX_ROUNDS`` and a
retained legacy cap still end their runs at their own number.
"""
from __future__ import annotations

import contextvars
import json
from dataclasses import replace

import pytest

from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.presence_runner import presence_event_identity, presence_turn_task_id, run_presence_turn
from ouroboros.task_results import load_task_result
from tests.test_presence_continuation_bootstrap import admission as legacy_admission, event as legacy_event
from tests.test_presence_continuation_bootstrap import install_bootstrap_harness, tool_call
from tests.test_presence_contact_wait import start
from tests.test_presence_owner_contact import contact_runtime as _contact_runtime, focus  # noqa: F401

contact_runtime = _contact_runtime
pytestmark = pytest.mark.serial

ANSWER = "Checked twelve sources; the card and the HTML page are both delivered."
FORCED = "Best answer at the limit: the sources read so far are listed; the rest is unchecked."


def _work(tool, rounds):
    return [{"content": None, "tool_calls": [tool_call(tool, {}, f"{tool}-{index}")]} for index in range(rounds)]


def _finish():
    return {"content": None, "tool_calls": [tool_call("finish_task", {"action": "finish", "answer": ANSWER}, "done")]}


def _round_limit_prompts(h):
    return [index for index, messages in enumerate(h.inputs) if "[ROUND_LIMIT]" in json.dumps(messages)]


@pytest.mark.parametrize("metadata,expected", [
    ({}, None), ({"inline_max_rounds": None}, None),  # absent stays absent: no inline cap at all
    ({"inline_max_rounds": 3}, 3), ({"inline_max_rounds": 200}, 200),  # a recorded positive cap binds
    # Present but malformed is a typo, never "no bound": the finite legacy fallback.
    ({"inline_max_rounds": 0}, 10), ({"inline_max_rounds": -2}, 10), ({"inline_max_rounds": True}, 10),
    ({"inline_max_rounds": False}, 10), ({"inline_max_rounds": "8"}, 10), ({"inline_max_rounds": 2.5}, 10),
])
def test_bootstrap_reads_only_a_recorded_inline_cap(metadata, expected):
    from ouroboros.loop import carried_inline_round_cap

    assert carried_inline_round_cap(metadata) == expected


@pytest.mark.parametrize("global_rounds", ["unlimited", None], ids=["unlimited", "document-default"])
def test_a_place_focus_works_past_ten_rounds_and_finishes_on_the_ordinary_path(contact_runtime, monkeypatch,
                                                                                global_rounds):
    r = contact_runtime
    if global_rounds is None:
        monkeypatch.delenv("OUROBOROS_MAX_ROUNDS", raising=False)
    else:
        monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", global_rounds)
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    h.script = [*_work("knowledge_list", 12), _finish()]

    result = start(r, h).result.result(timeout=120)

    ctx = h.agents[0].tools._ctx
    assert "inline_max_rounds" not in ctx.task_metadata and ctx.task_metadata["wake_cause"] == "focus"
    # Calls made, forced [ROUND_LIMIT] prompts and the bootstrap's cap: no forced call before the author's finish.
    assert (h.calls, _round_limit_prompts(h), ctx.inline_max_rounds) == (13, [], None)
    assert (result.outcome, result.text) == ("message", ANSWER)
    row = load_task_result(r.root, result.task_id)
    assert row["status"] == "completed" and row["reason_code"] != "round_limit"
    assert row["outcome_axes"]["execution"]["status"] == "ok" and row["result"] == ANSWER


def test_the_owner_global_round_limit_still_ends_a_place_focus(contact_runtime, monkeypatch):
    r = contact_runtime
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "5")
    h = install_bootstrap_harness(monkeypatch, r.root, review_mode="off")
    h.script = [*_work("knowledge_list", 5), {"content": FORCED, "tool_calls": []}]

    result = start(r, h).result.result(timeout=120)

    assert h.calls == 6 and _round_limit_prompts(h) == [5]
    assert "MAX_ROUNDS (5)" in json.dumps(h.inputs[5])
    row = load_task_result(r.root, result.task_id)
    assert row["reason_code"] == "round_limit" and row["result"] == FORCED


def test_a_retained_pre_places_turn_keeps_its_recorded_cap(tmp_path, monkeypatch):
    """A turn admitted before Places carries its frozen cap through the same producer and bootstrap."""
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "unlimited")
    h = install_bootstrap_harness(monkeypatch, tmp_path / "data", review_mode="off")
    admitted = legacy_admission()
    ceiling = replace(admitted.capability_ceiling, inline_max_rounds=3)
    admitted = replace(admitted, inline_max_rounds=3, capability_ceiling=replace(
        ceiling, digest=presence_ceiling_payload(ceiling)["digest"]))
    h.script = [*_work("chat_history", 3), {"content": FORCED, "tool_calls": []}]
    turn = legacy_event()

    execution = h.executions.start_thread(
        presence_turn_task_id(admitted.binding_id, turn.source_event_id),
        identity=presence_event_identity(admitted.binding_id, turn),
        admit=lambda: h.gate.acquire(turn.conversation_key),
        run=lambda lease: run_presence_turn(admission=admitted, event=turn, repo_dir=h.repo, drive_root=h.data,
                                            agent_factory=h.factory, admitted=lease),
        context=contextvars.copy_context())[0]
    result = execution.result.result(timeout=120)

    assert h.agents[0].tools._ctx.inline_max_rounds == 3
    assert h.calls == 4 and _round_limit_prompts(h) == [3]
    assert "MAX_ROUNDS (3)" in json.dumps(h.inputs[3])
    assert load_task_result(h.data, result.task_id)["reason_code"] == "round_limit"
