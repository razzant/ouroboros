"""P2: the one-shot final-answer-marker nudge. The agent did real work and has prose but no
FINAL ANSWER marker -> remind it to mark its OWN answer (no extractor change). Ordered AFTER
the skill/red/FR3-verify and A3 no-op nudges; mutually exclusive with the A3 no-op case.
v6.60.0: the marker nudge is PROTOCOL-GATED — it fires only for tasks whose contract
declares answer_protocol="final_answer_line" (the fixtures declare it)."""
from __future__ import annotations

import tempfile
import types as _t
from pathlib import Path

from ouroboros import loop as L


def _ctx_tools(monkeypatch, expected_output="The answer is 42", answer_protocol="final_answer_line"):
    # Pre-latch the earlier nudges so we exercise the A3 / marker decision in isolation.
    monkeypatch.setattr(L, "_skill_finalization_message", lambda *a, **k: "")
    contract = {"expected_output": expected_output}
    if answer_protocol:
        contract["answer_protocol"] = answer_protocol
    ctx = _t.SimpleNamespace(
        task_contract=contract, task_metadata={},
        _skill_finalization_injected=True, _verify_red_nudged=True, _verify_nudged=True,
    )
    return Path(tempfile.mkdtemp()), ctx, _t.SimpleNamespace(_ctx=ctx)


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


def test_marker_nudge_yields_to_a3_no_op(monkeypatch):
    # No tool calls + no reviewable effects + no marker -> A3 no-op nudge owns this turn,
    # NOT the marker nudge (mutual exclusivity).
    dr, ctx, tools = _ctx_tools(monkeypatch)
    msgs: list = []
    fired = L._maybe_inject_finalization_nudges(tools, dr, "t", _trace(tool_calls=[]), "I cannot proceed", msgs, lambda *_: None)
    assert fired is True
    assert getattr(ctx, "_noop_attempt_nudged", False) is True
    assert getattr(ctx, "_final_marker_nudged", False) is False
