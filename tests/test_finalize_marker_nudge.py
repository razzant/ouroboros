"""P2: the one-shot final-answer-marker nudge. The agent did real work and has prose but no
FINAL ANSWER marker -> state the protocol's capture rule (no extractor change). Ordered AFTER
the skill notice and the one pre-delivery notice. The A3 "expected output but nothing attempted" re-loop is
gone (#1748): prose can itself be the deliverable.
v6.60.0: the marker nudge is PROTOCOL-GATED — it fires only for tasks whose contract
declares answer_protocol="final_answer_line" (the fixtures declare it)."""
from __future__ import annotations

import tempfile
import types as _t
from pathlib import Path

from ouroboros import loop as L


def _ctx_tools(monkeypatch, expected_output="The answer is 42", answer_protocol="final_answer_line"):
    # Pre-latch the earlier nudges so we exercise the marker decision in isolation.
    monkeypatch.setattr(L, "_skill_finalization_message", lambda *a, **k: "")
    contract = {"expected_output": expected_output}
    if answer_protocol:
        contract["answer_protocol"] = answer_protocol
    drive_root = Path(tempfile.mkdtemp())
    ctx = _t.SimpleNamespace(
        task_contract=contract, task_metadata={}, drive_root=drive_root,
        _skill_finalization_injected=True, _pre_delivery_noticed=True,
    )
    return drive_root, ctx, _t.SimpleNamespace(_ctx=ctx)


def _trace(tool_calls=None):
    return {"reasoning_notes": [], "tool_calls": tool_calls if tool_calls is not None else [{"tool": "run_command", "status": "ok"}]}


def test_marker_nudge_fires_on_work_plus_prose_without_marker(monkeypatch):
    dr, ctx, tools = _ctx_tools(monkeypatch)
    msgs: list = []
    fired = L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(), "I computed it: the value is 42.", msgs, lambda *_: None)
    assert fired is True
    assert getattr(ctx, "_final_marker_nudged") is True
    assert "FINAL ANSWER" in msgs[-1]["content"]


def test_marker_nudge_one_shot(monkeypatch):
    dr, ctx, tools = _ctx_tools(monkeypatch)
    msgs: list = []
    assert L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(), "answer 42", msgs, lambda *_: None) is True
    assert L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(), "answer 42", msgs, lambda *_: None) is False


def test_marker_nudge_suppressed_when_marker_present(monkeypatch):
    dr, ctx, tools = _ctx_tools(monkeypatch)
    assert L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(), "done.\nFINAL ANSWER: 42", [], lambda *_: None) is False


def test_marker_nudge_fires_without_expected_output_under_protocol(monkeypatch):
    """The protocol gate is sufficient: GAIA-shaped contracts declare
    answer_protocol="final_answer_line" with the question in `objective` and
    expected_output EMPTY — the nudge must still fire (a v6.56.0 GAIA run
    finalized a last-round refusal with an empty typed answer because the old
    extra expected_output gate suppressed the only salvage surface)."""
    dr, ctx, tools = _ctx_tools(monkeypatch, expected_output="")
    msgs: list = []
    fired = L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(), "some prose", msgs, lambda *_: None)
    assert fired is True
    assert getattr(ctx, "_final_marker_nudged") is True
    assert "FINAL ANSWER" in msgs[-1]["content"]


def test_marker_nudge_suppressed_without_answer_protocol(monkeypatch):
    """v6.60.0: no answer_protocol on the contract -> the marker nudge NEVER fires,
    even with expected_output + real work + marker-less prose (ordinary tasks are
    never prompted to emit FINAL ANSWER lines)."""
    dr, ctx, tools = _ctx_tools(monkeypatch, answer_protocol="")
    msgs: list = []
    fired = L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(), "I computed it: 42.", msgs, lambda *_: None)
    assert fired is False
    assert getattr(ctx, "_final_marker_nudged", False) is False


def test_prose_only_deliverable_buys_no_extra_round(monkeypatch):
    """A3 removed (#1748; base witness: the former ``test_marker_nudge_yields_to_a3_no_op``
    asserted this exact turn re-looped with "without having attempted it"). An expected
    output answered in plain prose with no tool calls gets no re-loop, no latch and no
    injected text, with or without the answer protocol (the marker notice needs real work)."""
    for protocol in ("final_answer_line", ""):
        dr, ctx, tools = _ctx_tools(monkeypatch, answer_protocol=protocol)
        msgs: list = []
        notes: list = []
        fired = L._maybe_inject_finalization_nudges(
            tools, dr, "t", _trace(tool_calls=[]), "The answer is 42, because 6 x 7.", msgs, notes.append)
        assert fired is False and msgs == [] and notes == []
        assert not hasattr(ctx, "_noop_attempt_nudged")
        assert getattr(ctx, "_final_marker_nudged", False) is False


def test_marker_notice_states_the_protocol_without_commands(monkeypatch):
    dr, ctx, tools = _ctx_tools(monkeypatch)
    msgs: list = []
    assert L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(), "the value is 42", msgs, lambda *_: None)
    text = msgs[-1]["content"]
    assert "`FINAL ANSWER: <answer>`" in text and "the response above has none" in text
    assert "If you are done" not in text and "keep working" not in text
