"""The acceptance fence's TYPED OUTCOME through the loop's middle layer.

``_begin_task_acceptance_fence`` / ``_end_task_acceptance_fence`` answer
ok | refused(reason) | unknown(waited_sec) instead of a bool with a DEBUG line:
an answer that has not arrived is a gap, every refusal or gap leaves ONE durable
``supervisor_ack_unavailable`` row, and only ``sealed`` is a seal — an unexplained
``released`` on a terminal outcome is decided by the local mailbox (#406).
"""

from __future__ import annotations

import json
import queue as stdqueue
from types import SimpleNamespace

import pytest

from tests.test_acceptance_fence import _isolated_queue
from tests.test_acceptance_fence_transport import WAIT_SEC, _Supervisor, _pooled_agent


@pytest.fixture
def short_wait(monkeypatch):
    from ouroboros import runtime_limits

    monkeypatch.setattr(runtime_limits, "get_acceptance_fence_ack_wait_sec", lambda: WAIT_SEC)


def _loop_ctx(tmp_path, agent, task_id="root-1", **extra):
    ctx = SimpleNamespace(
        task_metadata={"root_task_id": task_id}, task_id=task_id, drive_root=tmp_path,
        _task_acceptance_fence_token=None, _task_acceptance_sealed_fence_token=None,
        _task_acceptance_fence_generation=None, _task_acceptance_queue_descendants=[],
        **extra,
    )
    if agent is not None:
        ctx.begin_acceptance_fence = agent._begin_acceptance_fence
        ctx.inspect_acceptance_fence = agent._inspect_acceptance_fence
        ctx.end_acceptance_fence = agent._end_acceptance_fence
    return ctx


def _unavailable_rows(tmp_path):
    path = tmp_path / "logs" / "events.jsonl"
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row.get("type") == "supervisor_ack_unavailable"]


# --- the middle layer: ok | refused(reason) | unknown(waited_sec) ---------------------------


def test_one_lost_ack_and_resend_puts_the_fence_up(monkeypatch, tmp_path, short_wait):
    from ouroboros.loop import _begin_task_acceptance_fence

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    events: stdqueue.Queue = stdqueue.Queue()
    supervisor = _Supervisor(events, tmp_path, lose_acks=1)
    ctx = _loop_ctx(tmp_path, _pooled_agent(tmp_path, events, running=monkeypatch))
    try:
        fence_ok, token = _begin_task_acceptance_fence(ctx, "root-1")
    finally:
        supervisor.stop()
    assert fence_ok and token == ctx._task_acceptance_fence_token
    assert queue_mod.ACCEPTANCE_FENCES["root-1"]["token"] == token
    assert ctx._task_acceptance_fence_outcome.status == "ok"
    assert _unavailable_rows(tmp_path) == []  # the fence is up: no model round was owed


def test_never_acked_begin_is_typed_unknown_with_one_durable_row(monkeypatch, tmp_path, short_wait):
    from ouroboros.loop import _begin_task_acceptance_fence

    _isolated_queue(monkeypatch, tmp_path)
    ctx = _loop_ctx(tmp_path, _pooled_agent(tmp_path, stdqueue.Queue()))
    outcome, token = _begin_task_acceptance_fence(ctx, "root-1")
    assert not outcome and token is None
    assert (outcome.status, outcome.op) == ("unknown", "begin")
    assert outcome.waited_sec >= WAIT_SEC * 2 - 0.1
    assert ctx._task_acceptance_fence_outcome is outcome  # the next package reads it here
    rows = _unavailable_rows(tmp_path)
    assert len(rows) == 1
    assert {key: rows[0][key] for key in ("task_id", "root_task_id", "op", "outcome")} == {
        "task_id": "root-1", "root_task_id": "root-1", "op": "begin", "outcome": "unknown"}
    assert rows[0]["waited_sec"] == outcome.waited_sec and "reason" in rows[0]


def test_healthy_supervisor_leaves_no_unavailable_row(monkeypatch, tmp_path, short_wait):
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence

    _isolated_queue(monkeypatch, tmp_path)
    events: stdqueue.Queue = stdqueue.Queue()
    supervisor = _Supervisor(events, tmp_path)
    ctx = _loop_ctx(tmp_path, _pooled_agent(tmp_path, events, running=monkeypatch))
    try:
        assert _begin_task_acceptance_fence(ctx, "root-1")[0]
        assert _begin_task_acceptance_fence(ctx, "root-1")[0]  # refresh through inspect
        sealed = _end_task_acceptance_fence(ctx, outcome="terminal")
    finally:
        supervisor.stop()
    assert sealed and sealed.status == "ok"
    assert ctx._task_acceptance_sealed_fence_token
    assert _unavailable_rows(tmp_path) == []


def test_refused_begin_is_typed_refused_with_its_reason(monkeypatch, tmp_path, short_wait):
    from ouroboros.loop import _begin_task_acceptance_fence

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    queue_mod.transition_acceptance_fence(
        action="begin", token="a" * 32, root_task_id="root-1", task_id="root-1")
    queue_mod.transition_acceptance_fence(action="end", token="a" * 32, outcome="terminal")
    events: stdqueue.Queue = stdqueue.Queue()
    supervisor = _Supervisor(events, tmp_path)
    ctx = _loop_ctx(tmp_path, _pooled_agent(tmp_path, events, running=monkeypatch))
    try:
        outcome, token = _begin_task_acceptance_fence(ctx, "root-1")
    finally:
        supervisor.stop()
    assert not outcome and token is None
    assert (outcome.status, outcome.reason) == ("refused", "sealed")  # the row's typed state, not prose
    rows = _unavailable_rows(tmp_path)
    assert [(row["op"], row["outcome"], row["reason"]) for row in rows] == [("begin", "refused", "sealed")]


def test_begin_with_stale_token_rebinds_through_fresh_begin(tmp_path):
    """A lost end/inspect ack leaves a stale local token while the supervisor already
    released the fence: a REFUSED inspection drops the binding and begins afresh."""
    from ouroboros.loop import _begin_task_acceptance_fence

    ctx = _loop_ctx(tmp_path, None)
    ctx._task_acceptance_fence_token, ctx._task_acceptance_fence_generation = "stale-token", 0
    ctx.inspect_acceptance_fence = lambda **_kwargs: (_ for _ in ()).throw(
        RuntimeError("acceptance fence inspect failed"))
    ctx.begin_acceptance_fence = lambda **_kwargs: {
        "ok": True, "status": "active", "token": "fresh-token",
        "owner_message_generation": 1, "queue_descendants": [],
    }
    ok, token = _begin_task_acceptance_fence(ctx, "root-1")
    assert ok and token == "fresh-token"
    assert ctx._task_acceptance_fence_token == "fresh-token"
    assert ctx._task_acceptance_fence_generation == 1


def test_unanswered_inspection_keeps_the_binding_and_asks_nothing_more(tmp_path):
    """The other direction: no answer is a gap, not a release — the token and the
    known generation stay, and no second request is stacked on a silent supervisor."""
    from ouroboros.loop import _begin_task_acceptance_fence

    begins: list = []
    ctx = _loop_ctx(tmp_path, None)
    ctx._task_acceptance_fence_token, ctx._task_acceptance_fence_generation = "held-token", 3
    ctx.inspect_acceptance_fence = lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("no ack"))
    ctx.begin_acceptance_fence = lambda **kwargs: begins.append(kwargs) or {"token": "never"}
    outcome, token = _begin_task_acceptance_fence(ctx, "root-1")
    assert not outcome and outcome.status == "unknown" and token == "held-token"
    assert begins == []
    assert (ctx._task_acceptance_fence_token, ctx._task_acceptance_fence_generation) == ("held-token", 3)


def test_end_failure_drops_binding_so_next_begin_is_fresh(tmp_path):
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence

    ctx = _loop_ctx(tmp_path, None)
    ctx._task_acceptance_fence_token, ctx._task_acceptance_fence_generation = "token-1", 0

    def failing_end(**_kwargs):
        raise TimeoutError("supervisor did not acknowledge acceptance fence token-1")

    ctx.end_acceptance_fence = failing_end
    ended = _end_task_acceptance_fence(ctx, outcome="revision")
    assert not ended and (ended.status, ended.op) == ("unknown", "end")
    assert ctx._task_acceptance_fence_token is None
    assert ctx._task_acceptance_fence_generation is None
    assert [(row["op"], row["outcome"]) for row in _unavailable_rows(tmp_path)] == [("end", "unknown")]

    ctx.begin_acceptance_fence = lambda **_kwargs: {
        "ok": True, "status": "active", "token": "token-2",
        "owner_message_generation": 0, "queue_descendants": [],
    }
    ok, token = _begin_task_acceptance_fence(ctx, "root-1")
    assert ok and token == "token-2"


def test_end_carries_the_known_generation(tmp_path):
    """``end`` is never sent without ``expected_generation`` once a generation was known —
    an unanswered refresh must not erase it (the compare-and-seal would vanish with it)."""
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence

    sent: list = []
    ctx = _loop_ctx(tmp_path, None)
    ctx.begin_acceptance_fence = lambda **_kwargs: {"token": "t", "owner_message_generation": 3}
    ctx.inspect_acceptance_fence = lambda **_kwargs: (_ for _ in ()).throw(TimeoutError("no ack"))
    ctx.end_acceptance_fence = lambda **kwargs: sent.append(kwargs) or {"ok": True, "status": "sealed"}
    assert _begin_task_acceptance_fence(ctx, "root-1")[0]
    assert not _begin_task_acceptance_fence(ctx, "root-1")[0]  # the refresh went unanswered
    assert _end_task_acceptance_fence(ctx, outcome="terminal")
    assert sent == [{"token": "t", "outcome": "terminal", "expected_generation": 3}]


@pytest.mark.parametrize("begin_answer,expected", [("bare-token", 0), ({"token": "t", "owner_message_generation": 2}, 2)])
def test_a_begin_answer_never_leaves_the_generation_unknown(tmp_path, begin_answer, expected):
    """``end`` omits ``expected_generation`` for ``None`` and a seal without the queue's
    compare-and-seal is the blind seal (#406): whatever shape ``begin`` answered in, the
    end carries a number — a fresh fence starts at 0, and a wrong 0 is refused by the queue."""
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence

    sent: list = []
    ctx = _loop_ctx(tmp_path, None)
    ctx.begin_acceptance_fence = lambda **_kwargs: begin_answer
    ctx.end_acceptance_fence = lambda **kwargs: sent.append(kwargs) or {"ok": True, "status": "sealed"}
    assert _begin_task_acceptance_fence(ctx, "root-1")[0]
    assert ctx._task_acceptance_fence_generation == expected
    assert _end_task_acceptance_fence(ctx, outcome="terminal")
    assert sent[0]["expected_generation"] == expected


# --- ``released`` is not a seal --------------------------------------------------------------


def _owner_mail(tmp_path, task_id="root-1"):
    from ouroboros.owner_mailbox import write_owner_message

    assert write_owner_message(tmp_path, "Use the blue variant", task_id, msg_id="owner-blue")


@pytest.mark.parametrize("mail,expected_mismatch", [(True, True), (False, False)])
def test_released_is_not_a_seal_and_owner_mail_forces_revision(tmp_path, mail, expected_mismatch):
    from ouroboros.loop import _end_task_acceptance_fence

    ctx = _loop_ctx(tmp_path, None)
    ctx._task_acceptance_fence_token, ctx._task_acceptance_fence_generation = "token-1", 0
    ctx.end_acceptance_fence = lambda **_kwargs: {"ok": True, "status": "released", "row_absent": True}
    if mail:
        _owner_mail(tmp_path)
    ended = _end_task_acceptance_fence(ctx, outcome="terminal")
    assert ended  # the supervisor answered; an absent row is not a refusal
    assert ctx._task_acceptance_sealed_fence_token is None  # and it is not a seal either
    assert ctx._task_acceptance_fence_generation_mismatch is expected_mismatch


def test_a_sealed_answer_is_a_seal_without_consulting_the_mailbox(tmp_path):
    from ouroboros.loop import _end_task_acceptance_fence

    ctx = _loop_ctx(tmp_path, None)
    ctx._task_acceptance_fence_token, ctx._task_acceptance_fence_generation = "token-1", 0
    ctx.end_acceptance_fence = lambda **_kwargs: {"ok": True, "status": "sealed"}
    _owner_mail(tmp_path)  # queue authority already compared the generation
    assert _end_task_acceptance_fence(ctx, outcome="terminal")
    assert ctx._task_acceptance_sealed_fence_token == "token-1"
    assert ctx._task_acceptance_fence_generation_mismatch is False


def test_lost_generation_mismatch_ack_cannot_produce_a_blind_seal(monkeypatch, tmp_path, short_wait):
    """#406: the first ``end(terminal)`` is applied as ``released + generation_mismatch`` and
    its ack is lost; the re-send finds no row. Owner mail is durably written before the
    generation moves, so the local mailbox — not the bare ``released`` — decides."""
    from ouroboros.loop import _begin_task_acceptance_fence, _end_task_acceptance_fence

    queue_mod, _pending = _isolated_queue(monkeypatch, tmp_path)
    events: stdqueue.Queue = stdqueue.Queue()
    agent = _pooled_agent(tmp_path, events, running=monkeypatch)
    ctx = _loop_ctx(tmp_path, agent)
    supervisor = _Supervisor(events, tmp_path)
    try:
        assert _begin_task_acceptance_fence(ctx, "root-1")[0]
        with queue_mod._queue_lock:  # steering: durable mail first, then the generation
            _owner_mail(tmp_path)
            queue_mod.ACCEPTANCE_FENCES["root-1"]["owner_message_generation"] += 1
        supervisor.lose_acks = 1
        ended = _end_task_acceptance_fence(ctx, outcome="terminal")
    finally:
        supervisor.stop()
    assert [evt.get("expected_generation") for evt in supervisor.seen if evt["action"] == "end"] == [0, 0]
    assert ended and queue_mod.ACCEPTANCE_FENCES == {}
    assert ctx._task_acceptance_sealed_fence_token is None
    assert ctx._task_acceptance_fence_generation_mismatch is True  # the caller revises, never seals
