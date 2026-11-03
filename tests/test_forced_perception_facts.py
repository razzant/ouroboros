"""The facts reach the mind before its last word (chat-voice C2, perception).

A forced rail hands its ONE model call the limitations it will record beside the
answer as typed ``key=value`` facts, never the owner-facing ``⚠️`` notice; and a
planning hold the mind was told about that the owner's hurry then released reaches
it exactly once. Every other gate transition already arrives by its own route (the
mind's own ``plan_task`` result, the settled-ask task message, the forced prompt),
so it buys no round.

New module rather than ``tests/test_owner_hurry_s3.py`` (933 lines, would cross the
1000-line target) or ``tests/test_delivery_forced_finalization.py`` (a frozen giant).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ouroboros import owner_hurry as oh
from tests._delivery_candidate_shared import write_child, write_confirmed_disposition_fixture
from tests.test_delivery_forced_finalization import _forced_test_context
from tests.test_owner_hurry_s3 import _plan_record


def _acceptance_ctx(tmp_path, *, latched):
    """A root's delivery context, latched by the owner's hurry when ``latched`` (the helper the deleted
    acceptance tests of ``test_owner_hurry_s3`` shared)."""
    ctx = SimpleNamespace(is_direct_chat=False, drive_root=str(tmp_path), budget_drive_root=str(tmp_path),
                          task_metadata={}, task_contract={}, task_attempt=1, task_id="t-acc")
    if latched:
        ctx._owner_hurry_latch = {"msg_id": "hurry:x", "requested_at": "", "reason": oh.REASON_OWNER_HURRY}
    return ctx

OWNER_NOTICE = (
    "\n\n⚠️ Plan review: 1 open objection remains open — triad_a/o1 (ask a1, revision 1): no; "
    "work proceeded under the owner-selected advisory enforcement."
)


def _held(ctx, *, content="answer"):
    import ouroboros.loop as loop_mod

    trace = {"reasoning_notes": []}
    messages: list = []
    held = loop_mod._enforce_plan_gate(content, messages, SimpleNamespace(_ctx=ctx), trace, lambda _m: None)
    return held, messages, trace


@pytest.fixture
def open_blocking_install(monkeypatch):
    import ouroboros.loop as loop_mod

    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr(oh, "_plan_facts", lambda _ctx: (_plan_record("held"), {}, False))


def test_an_open_advisory_gate_costs_no_extra_round_on_first_sight(tmp_path, monkeypatch):
    """Q4=A: no reminder before every final. The marker is not seeded by an observation."""
    import ouroboros.loop as loop_mod

    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "advisory")
    monkeypatch.setattr(oh, "_plan_facts", lambda _ctx: (_plan_record("held"), {}, False))
    ctx = _acceptance_ctx(tmp_path, latched=False)
    ctx.task_metadata = {"force_plan": True}
    held, messages, trace = _held(ctx)
    assert held is False and messages == []
    assert trace["plan_gate"]["allow"] is True
    assert not hasattr(ctx, "_plan_gate_told_identity")


def test_a_hold_released_by_owner_hurry_reaches_the_mind_once(tmp_path, open_blocking_install):
    """Fires only on a change the mind cannot see otherwise, one shot, typed not rendered."""
    ctx = _acceptance_ctx(tmp_path, latched=False)
    ctx.task_metadata = {"force_plan": True}
    held, messages, _trace = _held(ctx)
    assert held is True and messages[-1]["content"].startswith("[PLAN_REVIEW_HOLD]")
    told = ctx._plan_gate_told_identity
    assert told

    ctx._owner_hurry_latch = _acceptance_ctx(tmp_path, latched=True)._owner_hurry_latch
    held, messages, trace = _held(ctx, content="the draft")
    assert held is True and [m["role"] for m in messages] == ["assistant", "user"]
    released = messages[-1]["content"]
    assert released.startswith("[PLAN_REVIEW_RELEASED]")
    for fact in ("plan_review_open=true", "plan_holding=1", "owner_hurry_local_advisory=true", "configured_enforcement=blocking"):
        assert fact in released
    assert "⚠️" not in released and "advisory enforcement" not in released.split("\n")[1]
    assert ctx._plan_gate_told_identity != told and ctx._plan_gate_release_told is True
    assert trace["reasoning_notes"] == ["Released plan-review gate reported before final response."]

    held, messages, _trace = _held(ctx)
    assert held is False and messages == []


def test_a_gate_the_mind_cleared_itself_buys_no_round(tmp_path, monkeypatch, open_blocking_install):
    """The latch never bills the mind for its own act: a hold whose objection the raiser
    withdrew, released while the hurry latch is armed, appends nothing."""
    ctx = _acceptance_ctx(tmp_path, latched=False)
    ctx.task_metadata = {"force_plan": True}
    assert _held(ctx)[0] is True

    ctx._owner_hurry_latch = _acceptance_ctx(tmp_path, latched=True)._owner_hurry_latch
    monkeypatch.setattr(oh, "_plan_facts", lambda _ctx: (_plan_record("awaiting"), {}, False))
    held, messages, _trace = _held(ctx)
    assert held is False and messages == []
    assert not getattr(ctx, "_plan_gate_release_told", False)


ADVISORY_OPEN = {
    "required": True, "self_opened": True, "enforcement": "advisory", "status": "objections_open", "allow": True,
    "open_objections": [{"seat_id": "triad_a", "id": "o1", "text": "no", "holds": True, "not_adjudicated": False,
                         "ask_id": "a1", "revision": 1}],
    "holding": ["triad_a:o1"], "pending_asks": [], "asks_dispatched": 1, "force_plan_met": True, "revision": 1,
    "plan_review_class": "objections_open",
}


def _round_limit_prompt(tmp_path, monkeypatch, decision, *, deferred_child):
    if deferred_child:
        write_child(tmp_path)
        write_confirmed_disposition_fixture(tmp_path, disposition="deferred", rationale="defer it")
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    captured = []
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda _llm, messages, *_a, **_k: (
        captured.append([dict(m) for m in messages]) or {"content": "Forced answer."}, 0.0))
    monkeypatch.setattr(loop, "_plan_gate_decision", lambda *_a, **_k: dict(decision))
    monkeypatch.setattr(loop, "_plan_gate_disclosure",
                        lambda *_a, **_k: OWNER_NOTICE if decision.get("open_objections") else "")
    text, usage, _returned = loop._handle_round_limit(limit_ctx)
    assert text == "Forced answer." and len(captured) == 1
    last = next(m for m in reversed(captured[0]) if m["role"] == "user")
    content = last["content"]
    if isinstance(content, list):
        content = "".join(str(block.get("text") or "") for block in content if isinstance(block, dict))
    return str(content), usage


def test_the_forced_prompt_carries_typed_facts_not_the_owner_notice(tmp_path, monkeypatch):
    content, usage = _round_limit_prompt(tmp_path, monkeypatch, ADVISORY_OPEN, deferred_child=True)
    block = content[content.index("[TASK_STATE_FACTS]"):]
    for fact in ("plan_review_open=true", "plan_gate_status=objections_open", "plan_open_objections=1",
                 "plan_review_enforcement=advisory", "children_deferred=1"):
        assert fact in block
    assert "⚠️" not in block and "owner-selected" not in block
    # The owner-facing notice still rides the recorded result, where it belongs.
    assert OWNER_NOTICE.strip() in usage["terminal_host_notice"]


def test_a_clean_gate_adds_no_facts_block(tmp_path, monkeypatch):
    clean = {"required": True, "self_opened": True, "enforcement": "advisory", "status": "clear", "allow": True,
             "open_objections": [], "holding": [], "pending_asks": [], "asks_dispatched": 1, "force_plan_met": True}
    content, usage = _round_limit_prompt(tmp_path, monkeypatch, clean, deferred_child=False)
    assert "[TASK_STATE_FACTS]" not in content
    assert "terminal_plan_review_open" not in usage
