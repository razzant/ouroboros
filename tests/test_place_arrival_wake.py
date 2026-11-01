"""A place arrival's host-recorded ``place_arrival.wake`` decides whether it ends an ordinary wait.

``place_intake.offer_to_author`` writes the fact on the conversation author's mailbox entry. A typed
``True`` wakes that author's wait whatever senders/tasks it selected; a typed ``False`` is context
that the next genuine drain delivers, never a wake, even when any mail would wake the wait; a
missing or non-bool fact keeps the ordinary mail rule. Driven through the real consumers:
``await_messages`` in-slot, its warm park (``direct_owner_wait``), the cold sleep-wake pass
(``supervisor.sleep_wake``), ``wait_task``/``wait_tasks``, a review panel's wait,
``delegate_wait`` (``delegate_supervision.supervised_wait``) and a provider-outage backoff
(``loop_transport.transport_wait_step``). Real connectors' reactions through these consumers:
``tests/test_place_reaction_connectors.py``.
"""

from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros.owner_mailbox import (
    KIND_OWNER_PAUSE,
    acknowledged_task_message_ids,
    write_owner_message,
    write_task_message,
)
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_model_sleep import _cold_park, _ctx, _mail, _result
from tests.test_place_focus import focus as focus

_LEGACY = object()


def _place(root, task_id, wake, *, msg_id="place-event:e1", author_of="presence-turn"):
    arrival = {"source_event_id": msg_id, "message": {"ts": "1.0"}}
    if wake is not _LEGACY:
        arrival["wake"] = wake
    assert write_task_message(root, json.dumps({"event": "reaction:+1"}), task_id, source_task_id=author_of,
                              provenance="place_event", msg_id=msg_id, place_arrival=arrival)


def _delivered(root, ctx):
    """The ordinary round-top drain: the only owner of delivery and acknowledgement."""
    from ouroboros.loop_round_limits import _drain_incoming_messages

    messages: list = []
    _drain_incoming_messages(messages, queue.Queue(), root, ctx.task_id, queue.Queue(),
                             ctx._loop_mailbox_seen_ids, owner_ctx=ctx)
    return json.dumps(messages, ensure_ascii=False)


# -- the shared reader ------------------------------------------------------------------------------

@pytest.mark.parametrize("wake, selected, expected", [
    (True, {"senders": ["peer-a"]}, "mail:presence-turn"),       # unrelated selector: still the author's wake
    (True, {"tasks": ["child"]}, "mail:presence-turn"),
    (False, {"any_mail": True}, ""),                             # context only, even under any mail
    (False, {"senders": ["presence-turn"]}, ""),                 # ... and even when its sender is selected
    (_LEGACY, {"any_mail": True}, "mail:presence-turn"),         # written before the fact: ordinary mail
    (_LEGACY, {"senders": ["peer-a"]}, ""),
    ("false", {"any_mail": True}, "mail:presence-turn"),         # not a bool: no typed fact at all
    (0, {"senders": ["peer-a"]}, ""),
    (None, {"any_mail": True}, "mail:presence-turn"),
])
def test_the_typed_wake_fact_decides_and_anything_else_keeps_the_mail_rule(tmp_path, wake, selected, expected):
    from ouroboros import model_sleep

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _place(tmp_path, "sleeper", wake)
    assert model_sleep.wake_reason(ctx, {"senders": [], "tasks": [], "runs": [], "any_mail": False, **selected}) == expected
    assert acknowledged_task_message_ids(tmp_path, "sleeper", attempt_key=1) == set(), "a peek acknowledges nothing"


def test_controls_owner_words_and_descendants_keep_their_precedence_over_place_mail(tmp_path):
    from ouroboros import model_sleep

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    chosen = {"senders": ["peer-a"], "tasks": [], "runs": [], "any_mail": False}
    _place(tmp_path, "sleeper", True, msg_id="place-event:wake")
    assert write_owner_message(tmp_path, "keep going", "sleeper", msg_id="owner-1", kind="owner_text")
    assert model_sleep.wake_reason(ctx, chosen) == "owner_text"
    _mail(tmp_path, "sleeper", "owner_pause", kind=KIND_OWNER_PAUSE, msg_id="pause-1")
    assert model_sleep.wake_reason(ctx, chosen) == "control:owner_pause"
    # A passive arrival ahead of a child's escalation never hides it.
    other = _ctx(tmp_path, "author")
    _result(tmp_path, "author")
    _place(tmp_path, "author", False, msg_id="place-event:quiet")
    assert write_task_message(tmp_path, "blocked", "author", source_task_id="child", provenance="descendant_task",
                              msg_id="child-1")
    assert model_sleep.wake_reason(other, {"senders": [], "tasks": [], "runs": [], "any_mail": False}) == "mail:child"


# -- await_messages in-slot ---------------------------------------------------------------------------

def test_in_slot_wait_does_not_end_on_passive_place_mail_and_its_drain_still_delivers_it(tmp_path):
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _place(tmp_path, "sleeper", False)
    out = json.loads(_await_messages(ctx, 1))
    assert out["reason"] == "timeout" and out["pending"] is False
    assert acknowledged_task_message_ids(tmp_path, "sleeper", attempt_key=1) == set()
    assert "reaction:+1" in _delivered(tmp_path, ctx), "the next genuine drain still delivers it"
    assert "place-event:e1" in acknowledged_task_message_ids(tmp_path, "sleeper", attempt_key=1)


@pytest.mark.parametrize("wake", [True, _LEGACY])
def test_in_slot_wait_ends_on_a_waking_or_legacy_place_arrival(tmp_path, wake):
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _place(tmp_path, "sleeper", wake)
    out = json.loads(_await_messages(ctx, 5))
    assert out["reason"] == "owner_mailbox_pending" and out["pending"] is True


# -- await_messages warm: arming check and the park ---------------------------------------------------

def test_a_warm_sleep_on_other_senders_wakes_at_once_for_a_waking_arrival(tmp_path):
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    ctx = _ctx(tmp_path)
    _place(tmp_path, "sleeper", True)
    ready = json.loads(_await_messages(ctx, mode="warm", senders=["peer-a"]))
    assert ready["reason"] == "ready" and ready["woke_by"] == "mail:presence-turn"
    assert not getattr(ctx, "_model_sleep", None)


def test_the_warm_park_wakes_for_a_waking_arrival_landing_after_arming(tmp_path):
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "peer-a")
    ctx = _ctx(tmp_path)
    armed = json.loads(_await_messages(ctx, mode="warm", senders=["peer-a"], wake_after_sec=30))
    assert armed["reason"] == "sleep_armed"
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    _place(tmp_path, "sleeper", True)
    assert direct_owner_wait(ctx, checkpoint) == "mail:presence-turn"
    assert acknowledged_task_message_ids(tmp_path, "sleeper", attempt_key=1) == set()


def test_a_warm_any_mail_sleep_arms_over_passive_place_mail_and_a_selected_terminal_still_wakes_it(tmp_path):
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _place(tmp_path, "sleeper", False)
    armed = json.loads(_await_messages(ctx, mode="warm"))      # no selectors: any mail would wake it
    assert armed["reason"] == "sleep_armed", armed
    ctx._model_sleep = None
    ctx._owner_wait_requested = ""
    # Passive mail plus an independently ready selected task: the task wakes it, the mail waits for the drain.
    _result(tmp_path, "child")
    armed = json.loads(_await_messages(ctx, mode="warm", tasks=["child"], wake_after_sec=30))
    assert armed["reason"] == "sleep_armed"
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    _result(tmp_path, "child", "completed")
    assert direct_owner_wait(ctx, checkpoint) == "task:child:completed"
    assert acknowledged_task_message_ids(tmp_path, "sleeper", attempt_key=1) == set()
    assert "reaction:+1" in _delivered(tmp_path, ctx)


def test_the_warm_park_keeps_sleeping_over_passive_place_mail_until_its_chosen_time(tmp_path):
    from ouroboros.owner_wait import checkpoint_owner_wait, direct_owner_wait
    from ouroboros.tools.control_task_results import _await_messages

    _result(tmp_path, "sleeper")
    _result(tmp_path, "presence-turn")
    ctx = _ctx(tmp_path)
    armed = json.loads(_await_messages(ctx, mode="warm", senders=["presence-turn"], wake_after_sec=2))
    assert armed["reason"] == "sleep_armed"
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    _place(tmp_path, "sleeper", False)
    assert direct_owner_wait(ctx, checkpoint) == "timeout"


# -- cold: the supervisor's sleep-wake pass -----------------------------------------------------------

def test_a_cold_sleep_on_another_sender_wakes_for_a_waking_arrival_and_a_pause_still_vetoes(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    queue_mod, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    _result(tmp_path, "peer-a")
    _cold_park(tmp_path, monkeypatch, workers, senders=["peer-a"])
    assert wake_ready_sleepers(queue_mod) == []
    owner_pause.install_fence(tmp_path, "sleeper", request_id="p")
    _place(tmp_path, "sleeper", True)
    vetoed = wake_ready_sleepers(queue_mod)
    assert vetoed[0]["error"] == "sleep_wake_vetoed" and vetoed[0]["veto"] == "owner_pause"
    assert budget_pause_row(tmp_path, "sleeper")["sleep_ready"]["reason"] == "mail:presence-turn"
    owner_pause.release_fence(tmp_path, "sleeper", reason="test")
    granted = wake_ready_sleepers(queue_mod)
    assert granted[0]["ok"] is True and granted[0]["ready"]["reason"] == "mail:presence-turn"


def test_a_cold_any_mail_sleep_is_not_woken_by_passive_place_mail(tmp_path, monkeypatch):
    from ouroboros.budget_pause import budget_pause_row
    from supervisor.sleep_wake import wake_ready_sleepers

    queue_mod, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    row = _cold_park(tmp_path, monkeypatch, workers, wake_after_sec=3600)
    assert row["sleep"]["any_mail"] is True
    _place(tmp_path, "sleeper", False)
    assert wake_ready_sleepers(queue_mod) == []
    assert not budget_pause_row(tmp_path, "sleeper").get("sleep_ready")
    _mail(tmp_path, "sleeper", "real mail", sender="peer-b")
    assert wake_ready_sleepers(queue_mod)[0]["ready"]["reason"] == "mail:peer-b"


# -- wait_task / wait_tasks mailbox early return ------------------------------------------------------

def _waiter(tmp_path):
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "child", "running", root_task_id="parent", parent_task_id="parent")
    return SimpleNamespace(drive_root=tmp_path, task_id="parent", task_attempt=1,
                           task_metadata={"root_task_id": "parent"}, _loop_mailbox_seen_ids=set())


def test_wait_tasks_does_not_return_early_for_passive_place_mail(tmp_path):
    from ouroboros.tools.control_task_results import _wait_attention_poll, _wait_for_task, _wait_for_tasks

    ctx = _waiter(tmp_path)
    _place(tmp_path, "parent", False)
    assert _wait_attention_poll(ctx, "", ["child"])({}, {}) is None
    assert "early_return" not in json.loads(_wait_for_tasks(ctx, ["child"], timeout_sec=0))
    assert "unread message for this task" not in _wait_for_task(ctx, "child", timeout_sec=0)
    assert acknowledged_task_message_ids(tmp_path, "parent", attempt_key=1) == set()
    assert "reaction:+1" in _delivered(tmp_path, ctx)


@pytest.mark.parametrize("wake", [True, _LEGACY, "false"])
def test_wait_tasks_returns_early_for_a_waking_or_legacy_place_arrival(tmp_path, wake):
    from ouroboros.tools.control_task_results import _wait_attention_poll, _wait_for_tasks

    ctx = _waiter(tmp_path)
    _place(tmp_path, "parent", wake)
    assert _wait_attention_poll(ctx, "", ["child"])({}, {})["reason"] == "owner_mailbox_pending"
    assert json.loads(_wait_for_tasks(ctx, ["child"], timeout_sec=0))["early_return"]["reason"] == "owner_mailbox_pending"
    assert acknowledged_task_message_ids(tmp_path, "parent", attempt_key=1) == set()


def test_wait_tasks_still_returns_for_genuine_mail_behind_passive_place_mail(tmp_path):
    from ouroboros.tools.control_task_results import _wait_attention_poll

    ctx = _waiter(tmp_path)
    hook = _wait_attention_poll(ctx, "", ["child"])
    _place(tmp_path, "parent", False)
    assert hook({}, {}) is None
    assert write_owner_message(tmp_path, "Use option B.", "parent", msg_id="owner-b")
    assert hook({}, {})["reason"] == "owner_mailbox_pending"


# -- an inline author's direct owner-question park (Presence keeps its conversation there) ------------

def _question_park(tmp_path, seconds):
    import datetime

    from ouroboros.owner_wait import checkpoint_owner_wait

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    ctx._owner_wait_requested = "q1"
    ctx._owner_wait_deadline_at = (datetime.datetime.now(datetime.timezone.utc)
                                   + datetime.timedelta(seconds=seconds)).isoformat()
    checkpoint = checkpoint_owner_wait(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    assert checkpoint["quiz_id"] == "q1" and "sleep" not in checkpoint
    return ctx, checkpoint


def test_a_direct_question_park_keeps_waiting_over_passive_place_mail_until_its_bound(tmp_path):
    from ouroboros.owner_wait import direct_owner_wait

    ctx, checkpoint = _question_park(tmp_path, 2)
    _place(tmp_path, "sleeper", False)
    assert direct_owner_wait(ctx, checkpoint) == "timeout"
    assert acknowledged_task_message_ids(tmp_path, "sleeper", attempt_key=1) == set()
    assert "reaction:+1" in _delivered(tmp_path, ctx)


def test_a_direct_question_park_wakes_for_waking_mail_named_past_a_passive_arrival(tmp_path):
    from ouroboros.owner_wait import direct_owner_wait

    ctx, checkpoint = _question_park(tmp_path, 30)
    _place(tmp_path, "sleeper", False, msg_id="place-event:quiet")
    _place(tmp_path, "sleeper", True, msg_id="place-event:wake", author_of="presence-next")
    assert direct_owner_wait(ctx, checkpoint) == "mail:presence-next"


# -- a provider-outage backoff --------------------------------------------------------------------------

@pytest.mark.parametrize("wake, heard", [(False, False), (True, True), (_LEGACY, True)])
def test_a_transport_backoff_hears_only_waking_or_legacy_place_mail(tmp_path, monkeypatch, wake, heard):
    import time

    from ouroboros import loop_transport

    _place(tmp_path, "sleeper", wake)
    observed = []
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", lambda _seconds, check: observed.append(check()))
    tools = SimpleNamespace(_ctx=SimpleNamespace(task_attempt=1, task_metadata={}))
    assert loop_transport.transport_wait_step(
        loop_transport.TransportWaitEpisode(started_monotonic=time.monotonic()), tools=tools,
        error_kind="transport_unavailable", drive_root=tmp_path, drive_logs=tmp_path / "logs", task_id="sleeper",
        model="unused", emit_progress=lambda *_a, **_k: None, incoming_messages=None, owner_msg_seen=set())
    assert observed == [heard]
    assert acknowledged_task_message_ids(tmp_path, "sleeper", attempt_key=1) == set(), "a peek acknowledges nothing"


# -- a review panel's wait ----------------------------------------------------------------------------

@pytest.mark.parametrize("wake, expected", [(False, ""), (True, "mail:presence-turn")])
def test_a_review_wait_hears_only_a_waking_place_arrival(tmp_path, wake, expected):
    from ouroboros.review_waiting import wake_reason

    _result(tmp_path, "sleeper")
    ctx = _ctx(tmp_path)
    _place(tmp_path, "sleeper", wake)
    assert wake_reason(ctx, {"selection": {"senders": ["peer-a"], "any_mail": False}}) == expected
    assert wake_reason(ctx, {}) == expected


# -- delegate_wait: the supervised wait over a delegated run ----------------------------------------
# Only the physical observer (``wait_once``) is replaced; the arrival is written by the production
# producer (``place_intake.offer_to_author``) into the live author's mailbox.

def _author(focus):
    import threading

    from ouroboros.conversation_lease import PresenceTurnGate

    root, _place_row, _admission, event = focus
    lease = PresenceTurnGate(state_root=root / "state").acquire(event.conversation_key)
    lease.bind_author("author")
    lease.intake_agent = SimpleNamespace(_owner_message_admission_lock=threading.RLock(),
                                         _accepting_owner_messages=True, _busy=True, _current_task_id="author")
    ctx = SimpleNamespace(task_id="author", task_attempt=1, drive_root=root, budget_drive_root=str(root),
                          task_metadata={}, _loop_mailbox_seen_ids=set())
    return lease, ctx


def _arrive(focus, wake, source_event_id="event1"):
    from dataclasses import replace

    from ouroboros.place_intake import offer_to_author
    from ouroboros.presence_runner import capture_presence_event

    root, _place_row, admission, event = focus
    event = replace(event, source_event_id=source_event_id)
    task = capture_presence_event(admission, event, drive_root=root)
    assert offer_to_author(root, task, event, wake=wake)
    return "place-event:" + task["metadata"]["presence_event_identity"]


def _observer(*steps):
    """Each read returns the next status; ``then`` (optional) runs before it, as an event landing mid-wait."""
    calls = []

    def wait_once(_ctx, run_id, *_args, **_kwargs):
        assert len(calls) < len(steps), "the wait kept polling past its last scripted read"
        status, then = steps[len(calls)]
        calls.append(status)
        if then:
            then()
        return json.dumps({"status": status, "run_id": run_id, "last_seq": len(calls),
                           **({"state": "succeeded"} if status == "terminal" else {})})
    return wait_once, calls


def _wake_ids(wake):
    return [row.get("msg_id") for row in wake.get("wake_events") or () if row.get("type") == "addressed_message"]


def test_delegate_wait_sleeps_over_a_passive_arrival_and_the_drain_still_delivers_it(focus):
    import ouroboros.delegate_supervision as supervision

    lease, ctx = _author(focus)
    try:
        quiet = _arrive(focus, False)
        wait_once, calls = _observer(("no_progress", None), ("terminal", None))
        wake = json.loads(supervision.supervised_wait(ctx, "run-1", wait_once=wait_once).text)
        assert calls == ["no_progress", "terminal"], "a passive arrival must survive the quiet poll"
        assert wake["status"] == "terminal" and quiet not in _wake_ids(wake)
        assert supervision.acknowledge_pending_wake(ctx, wake)
        assert quiet not in acknowledged_task_message_ids(focus[0], "author", attempt_key=1)
        assert quiet not in ctx._loop_mailbox_seen_ids
        rendered = _delivered(focus[0], ctx)
        assert "Place event; context only" in rendered and ctx._presence_delivered[-1]["wake"] is False
        assert quiet in acknowledged_task_message_ids(focus[0], "author", attempt_key=1)
    finally:
        lease.release()


def test_delegate_wait_wakes_for_a_later_waking_arrival_and_leaves_the_passive_one_unread(focus):
    import ouroboros.delegate_supervision as supervision

    lease, ctx = _author(focus)
    try:
        quiet = _arrive(focus, False)
        landed = []
        wait_once, calls = _observer(("no_progress", None),
                                     ("no_progress", lambda: landed.append(_arrive(focus, True, "event2"))))
        wake = json.loads(supervision.supervised_wait(ctx, "run-1", wait_once=wait_once).text)
        assert calls == ["no_progress", "no_progress"] and _wake_ids(wake) == landed
        assert supervision.acknowledge_pending_wake(ctx, wake)
        acked = acknowledged_task_message_ids(focus[0], "author", attempt_key=1)
        assert landed[0] in acked and quiet not in acked
        assert "Place event; context only" in _delivered(focus[0], ctx)
        assert quiet in acknowledged_task_message_ids(focus[0], "author", attempt_key=1)
    finally:
        lease.release()


@pytest.mark.parametrize("genuine", ["owner", "peer", "deadline"])
def test_delegate_wait_still_wakes_for_genuine_input_beside_a_passive_arrival(focus, genuine):
    import ouroboros.delegate_supervision as supervision

    lease, ctx = _author(focus)
    try:
        quiet = _arrive(focus, False)
        if genuine == "owner":
            assert write_owner_message(focus[0], "Use option B.", "author", msg_id="owner-b")
        elif genuine == "peer":
            assert write_task_message(focus[0], "tests done", "author", source_task_id="peer",
                                      provenance="peer_task", relation="sibling", msg_id="peer-1")
        else:
            ctx.task_metadata = {"deadline_at": "2000-01-01T00:00:00+00:00"}
        wait_once, calls = _observer(("no_progress", None))
        wake = json.loads(supervision.supervised_wait(ctx, "run-1", wait_once=wait_once).text)
        expected = {"owner": ["owner-b"], "peer": ["peer-1"], "deadline": []}[genuine]
        assert _wake_ids(wake) == expected
        if genuine == "deadline":
            assert {"type": "deadline"} in wake["wake_events"] and calls == []
        assert supervision.acknowledge_pending_wake(ctx, wake)
        assert quiet not in acknowledged_task_message_ids(focus[0], "author", attempt_key=1)
    finally:
        lease.release()


@pytest.mark.parametrize("wake", [_LEGACY, "false", None])
def test_delegate_wait_keeps_waking_for_place_mail_without_a_typed_fact(tmp_path, wake):
    import ouroboros.delegate_supervision as supervision

    ctx = SimpleNamespace(task_id="author", task_attempt=1, drive_root=tmp_path, budget_drive_root=str(tmp_path),
                          task_metadata={}, _loop_mailbox_seen_ids=set())
    _place(tmp_path, "author", wake)
    wait_once, calls = _observer(("no_progress", None))
    wake_out = json.loads(supervision.supervised_wait(ctx, "run-1", wait_once=wait_once).text)
    assert calls == ["no_progress"] and _wake_ids(wake_out) == ["place-event:e1"]


def test_delegate_wait_wakes_at_once_for_a_waking_arrival(focus):
    import ouroboros.delegate_supervision as supervision

    lease, ctx = _author(focus)
    try:
        loud = _arrive(focus, True)
        wait_once, calls = _observer(("no_progress", None))
        wake = json.loads(supervision.supervised_wait(ctx, "run-1", wait_once=wait_once).text)
        assert calls == ["no_progress"] and _wake_ids(wake) == [loud]
    finally:
        lease.release()
