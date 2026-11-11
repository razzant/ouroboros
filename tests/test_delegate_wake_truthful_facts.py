"""Truthful wake facts: the task-scoped child cursor across run ids and recovery, whole-sleep facts at the
single wake-publication point (when the wait was armed, how long it slept, in which mode, what woke it),
and the leaf route's declared live-input capability. A run's dated observation facts are the body's
(``test_body_feed``)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ouroboros.delegate_supervision as supervision
from ouroboros import task_tree_ledger
from ouroboros.delegate_supervision import acknowledge_pending_wake
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
from tests._run_wait_shared import wake_now


def _ctx(tmp_path, task_id="parent"):
    return SimpleNamespace(
        task_id=task_id, task_attempt=1, drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_metadata={"root_task_id": "root", "delegation_role": "subagent"},
    )


def _child(tmp_path, task_id="child", status=STATUS_RUNNING):
    return write_task_result(tmp_path, task_id, status, parent_task_id="parent",
                             root_task_id="root", delegation_role="subagent")


def _settled(run_id, seq=1):
    return lambda *_a, **_k: json.dumps({"status": "completed", "run_id": run_id, "last_seq": seq})


def _child_events(wake, kind="child_terminal"):
    return [event for event in wake.get("wake_events", []) if event.get("type") == kind]


def _state(ctx):
    return supervision._load_state(ctx, "run-o")


# -- A: the committed child-delivery cursor is task-scoped ---------------------


@pytest.mark.parametrize("carry", [True, False])
def test_acked_child_events_are_not_reannounced_under_a_new_run_id(tmp_path, monkeypatch, carry):
    """After a run-id switch an ACKED settled child and an acked beacon stay
    delivered, while a child settling after the switch emits exactly once. With the
    carry removed (the old rebuild) the delivered terminal is re-announced."""
    if not carry:
        def _old_rebuild(ctx, run_id):  # the pre-fix rebuild, verbatim in effect
            try:
                data = json.loads(supervision._state_path(ctx).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            if not isinstance(data, dict) or (str(run_id) and str(data.get("run_id") or "") != str(run_id)):
                data = {"schema": 1, "run_id": str(run_id), "journal_cursor": 0}
            return data

        monkeypatch.setattr(supervision, "_load_state", _old_rebuild)
    _child(tmp_path, "done", status=STATUS_COMPLETED)
    _child(tmp_path, "late")
    assert task_tree_ledger.tree_ledger_append(
        "root", "blocker", "done needs input", task_id="done", data_root=tmp_path,
    ).startswith("OK:")
    ctx = _ctx(tmp_path)
    first = json.loads(wake_now(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert [e["child_task_id"] for e in _child_events(first)] == ["done"]
    assert len(_child_events(first, "child_attention_beacon")) == 1
    assert acknowledge_pending_wake(ctx, first)

    def _late_settles(*_a, **_k):
        write_task_result(tmp_path, "late", STATUS_COMPLETED, result="late result")
        return json.dumps({"status": "completed", "run_id": "run-2", "last_seq": 1})

    second = json.loads(wake_now(ctx, "run-2", wait_once=_late_settles).text)
    announced = [e["child_task_id"] for e in _child_events(second)]
    if carry:
        assert announced == ["late"]
        assert _child_events(second, "child_attention_beacon") == []
    else:
        assert "done" in announced  # the defect the carry repairs
    assert acknowledge_pending_wake(ctx, second)
    if carry:
        third = json.loads(wake_now(ctx, "run-3", wait_once=_settled("run-3")).text)
        assert _child_events(third) == [] and _child_events(third, "child_attention_beacon") == []


def test_an_unacked_child_terminal_survives_the_run_switch_and_emits_once(tmp_path):
    _child(tmp_path, "done", status=STATUS_COMPLETED)
    ctx = _ctx(tmp_path)
    first = json.loads(wake_now(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert [e["child_task_id"] for e in _child_events(first)] == ["done"]
    # Never acknowledged: its cursor never committed, so the new run re-detects it once.
    second = json.loads(wake_now(ctx, "run-2", wait_once=_settled("run-2")).text)
    assert [e["child_task_id"] for e in _child_events(second)] == ["done"]
    assert acknowledge_pending_wake(ctx, second)
    third = json.loads(wake_now(ctx, "run-3", wait_once=_settled("run-3")).text)
    assert _child_events(third) == []


@pytest.mark.parametrize("carried", [True, False])
def test_recovery_handoff_and_restore_keep_the_committed_cursor(tmp_path, carried):
    import ouroboros.delegate_recovery as recovery

    cursor = {"attention_after_ts": "2026-09-26T00:00:00+00:00", "attention_seen": ["b1"],
              "children": {"done": {"status": "completed", "updated_at": "t", "result_sha256": "x"}}}
    path = supervision._state_path(_ctx(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "run_id": "run-old", "coordination_cursor": cursor}),
                    encoding="utf-8")
    row = {"task_id": "parent", "run_id": "run-new", "journal_cursor": 0}
    if carried:
        row["coordination_cursor"] = dict(cursor)
    recovery._restore_wait_checkpoint(tmp_path, row)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["schema"] == 2 and set(restored["runs"]) == {"run-old", "run-new"}
    assert restored["runs"]["run-new"]["status"] == "adopted"
    # D1 preserves task-wide delivery facts even when this holder carries none.
    assert restored["coordination_cursor"] == cursor


def test_prepare_handoff_row_snapshots_the_committed_cursor(tmp_path):
    from tests.test_delegate_pending_recovery import _pending_handoff

    cursor = {"children": {"done": {"status": "completed"}}}
    path = tmp_path / "state" / "delegate_supervision" / "t-cursor.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "run_id": "", "coordination_cursor": cursor}),
                    encoding="utf-8")
    _custody, recovery, *_rest = _pending_handoff(tmp_path, "t-cursor")
    assert recovery._read(tmp_path, "t-cursor")["coordination_cursor"] == cursor


# -- B: whole-sleep facts at the single publication point ---------------------


class _Clock:
    def __init__(self, start=1_000_000.0):
        self.now = start

    def time(self):
        return self.now


def _fake_time(monkeypatch, clock):
    import time as real_time

    monkeypatch.setattr(supervision, "time", SimpleNamespace(
        time=clock.time, sleep=lambda _s: None, strftime=real_time.strftime, gmtime=real_time.gmtime))


def test_every_wake_measures_the_whole_sleep_and_replays_it_exactly(tmp_path, monkeypatch):
    """The wake's sleep facts are the wait's own: armed then, slept this long in this mode, woken by this; the
    tick's quiet-renewal counters are gone. A replay repeats the recorded facts, nothing added since."""
    clock = _Clock()
    _fake_time(monkeypatch, clock)
    monkeypatch.setattr("ouroboros.tools.delegate._read_run", lambda _ctx, run_id, _seq, gateway=None: json.dumps(
        {"status": "completed", "run_id": run_id, "last_seq": 7}))
    ctx = _ctx(tmp_path)
    armed = {"tool": "delegate_wait", "runs": ["run-1"], "armed_at": "2026-11-09T00:00:00+00:00",
             "armed_epoch": clock.now, "mode": "cold", "since_seq": 2}
    clock.now += 15.0
    wake = json.loads(supervision._publish(ctx, armed, "run:run-1").text)
    assert wake["sleep"] == {"entered_at": "2026-11-09T00:00:00+00:00", "slept_sec": 15.0, "mode": "cold",
                             "woke_by": "run:run-1"}
    assert wake["last_seq"] == 7 and supervision._load_state(ctx, "run-1")["journal_cursor"] == 7
    clock.now += 500.0
    replay = json.loads(wake_now(ctx, "run-1").text)  # replayed before any new read
    assert replay.pop("replay_of")["fresh_observation"] is False  # labelled as a replay (#2992)
    assert replay == wake                        # recorded facts, nothing added since


def test_a_quiet_status_wake_drops_the_tick_advice_but_keeps_the_paused_note(tmp_path, monkeypatch):
    from ouroboros.delegate_progress import paused_note

    monkeypatch.setattr(supervision, "_control_wakes", lambda _ctx: [{"type": "deadline"}])
    tick = {"status": "no_progress", "run_id": "run-q", "last_seq": 1, "waited_sec": 3,
            "quiet_for_sec": 3, "reason": "non_terminal", "note": "or delegate_cancel if it is stuck"}
    quiet = json.loads(wake_now(_ctx(tmp_path, "t-quiet"), "run-q",
                                wait_once=lambda *_a, **_k: json.dumps(tick)).text)
    assert quiet["wake_events"] == [{"type": "deadline"}]
    assert not {"waited_sec", "quiet_for_sec", "note"} & set(quiet)
    assert quiet["sleep"]["mode"] == "answered_at_call"
    monkeypatch.setattr(supervision, "_control_wakes", lambda _ctx: [])

    pending = [{"interaction_id": "i1", "timeout_at": None}]
    paused = json.loads(wake_now(_ctx(tmp_path, "t-paused"), "run-q", wait_once=lambda *_a, **_k: json.dumps({
        **tick, "waiting_on_user": True, "pending_interactions": pending})).text)
    assert paused["note"] == paused_note(pending)
    assert "delegate_cancel" not in paused["note"] and "waited_sec" not in paused
    # A terminal wake is not quiet: its own fields stay, and it carries the sleep too.
    terminal = json.loads(wake_now(_ctx(tmp_path, "t-term"), "run-q", wait_once=lambda *_a, **_k: json.dumps({
        "status": "completed", "run_id": "run-q", "last_seq": 1, "note": "terminal note"})).text)
    assert terminal["note"] == "terminal note" and "sleep" in terminal


def test_an_old_pending_wake_without_sleep_replays_unchanged(tmp_path):
    ctx = _ctx(tmp_path)
    stored = {"status": "completed", "run_id": "run-1", "supervision_wake_id": "w-old"}
    path = supervision._state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "run_id": "run-1", "pending_wake": {
        "wake_id": "w-old", "attempt_key": "1", "payload": stored}}), encoding="utf-8")
    replay = json.loads(wake_now(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert "published_at" not in replay.pop("replay_of")  # a replay label without an invented time (#2992)
    assert replay == stored                       # nothing invented for an old record


def test_a_retired_ticks_sleep_entry_never_leaks_into_a_new_wake(tmp_path):
    """A state the retired tick left (its ``sleep_entry``, a ``sleeping`` status) labels nothing now: the wake's
    sleep facts are the new wait's own, never carried over from the old entry."""
    ctx = _ctx(tmp_path)
    path = supervision._state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "run_id": "run-1", "status": "sleeping", "sleep_entry": {
        "entered_at": "2026-09-26T00:00:00+00:00", "entered_at_unix": 1.0}}), encoding="utf-8")
    wake = json.loads(wake_now(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert wake["sleep"]["slept_sec"] < 60 and not {"opened_after", "previous_entered_at"} & set(wake["sleep"])


def test_a_spilled_wake_keeps_the_sleep_and_live_input_facts(tmp_path, monkeypatch):
    import ouroboros.tool_capabilities as capabilities

    monkeypatch.setattr(capabilities, "tool_result_limit", lambda _name: 900)
    ctx = _ctx(tmp_path)
    ctx.task_contract = {"delegation_budget": {"intent_note": "x" * 5000}}
    wake = json.loads(wake_now(ctx, "run-s", wait_once=_settled("run-s")).text)
    assert wake["wake_delivery"]["complete"] is False
    assert wake["sleep"]["mode"] == "answered_at_call" and wake["leaf_live_input"] == "unknown"


# -- leaf_live_input: read once at entry, stamped on every wake ----------------


class _Catalog:
    def __init__(self, rows=None, fail=False):
        self.rows, self.fail, self.reads, self.closed = rows or [], fail, [], False

    def agent_capabilities(self, *, timeout_sec=None):
        self.reads.append(timeout_sec)
        if self.fail:
            raise RuntimeError("catalog unreadable")
        return {"harnesses": self.rows}

    def close(self):
        self.closed = True


def _owned(tmp_path, monkeypatch, run_id="run-l", route="fixture"):
    from ouroboros import delegate_custody as custody

    monkeypatch.setitem(custody._CUSTODY, run_id, custody.RunCustody(
        run_id=run_id, task_id="parent", route_id=route, model="m"))


@pytest.mark.parametrize("rows,fail,expected", [
    ([{"id": "fixture", "liveInput": "mid_turn"}], False, "mid_turn"),
    ([{"id": "fixture"}], False, "none"),                 # an engine without the field
    ([], True, "unknown"),                                # failure is a fact, never a refusal
])
def test_leaf_live_input_is_read_once_per_publication_and_replayed_as_recorded(
    tmp_path, monkeypatch, rows, fail, expected,
):
    _owned(tmp_path, monkeypatch)
    catalog = _Catalog(rows, fail)
    monkeypatch.setattr(supervision, "_loop_gateway", lambda _entry: catalog)

    def wait_once(_ctx, run_id, *_a, **_k):
        return json.dumps({"status": "completed", "run_id": run_id, "last_seq": 4})

    ctx = _ctx(tmp_path)
    wake = json.loads(wake_now(ctx, "run-l", wait_once=wait_once).text)
    assert wake["leaf_live_input"] == expected
    assert catalog.reads == [supervision._TICK_SEC]    # once per publication
    assert catalog.closed                               # the publication's transport is dropped
    replay = json.loads(wake_now(ctx, "run-l").text)
    assert replay["leaf_live_input"] == expected and catalog.reads == [supervision._TICK_SEC]


def test_an_unowned_run_reads_no_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision, "_loop_gateway", lambda _entry: (_ for _ in ()).throw(
        AssertionError("no route, no catalog read")))
    wake = json.loads(wake_now(_ctx(tmp_path), "run-x", wait_once=_settled("run-x")).text)
    assert wake["leaf_live_input"] == "unknown"


# -- questions that hold nothing (claudexor #628) -----------------------------------------------


def _quiet_question_tick(**extra):
    pending = [{"interaction_id": "i1", "timeout_at": None, "delivery": "async"}]
    return pending, {"status": "no_progress", "run_id": "run-a", "state": "running", "last_seq": 1,
                     "waited_sec": 3, "waiting_on_user": True, "pending_interactions": pending,
                     "answered": True, **extra}


@pytest.mark.parametrize("whole_set_async", [True, False], ids=["whole-set-async", "fitted-rows-only"])
def test_a_quiet_wake_keeps_the_whole_sets_meaning_never_its_fitted_rows(tmp_path, monkeypatch, whole_set_async):
    """The payload's ``async_questions_only`` (judged before any preview) decides the note, and a run's recorded
    work (``run_activity``) reads the body's same fact; rows that merely look async in a fitted preview never
    re-decide it."""
    from ouroboros.delegate_progress import paused_note
    from ouroboros.delegate_feed import run_activity

    monkeypatch.setattr(supervision, "_control_wakes", lambda _ctx: [])
    pending, tick = _quiet_question_tick(**({"async_questions_only": True} if whole_set_async else {}))
    ctx = _ctx(tmp_path, f"t-{whole_set_async}")
    wake = json.loads(wake_now(ctx, "run-a", wait_once=lambda *_a, **_k: json.dumps(tick)).text)
    assert wake["note"] == paused_note(pending, async_only=whole_set_async)
    assert ("PAUSED" in wake["note"]) is not whole_set_async
    assert run_activity({"answered": True, "run_state": "running", "waiting_on_user": True,
                         "async_questions_only": whole_set_async}) is whole_set_async


@pytest.mark.parametrize("async_only", [True, False])
def test_a_compact_question_wake_does_not_invent_a_null_expiry(tmp_path, monkeypatch, async_only):
    """A compact observation retains attention but has no rows from which to claim expiry."""
    monkeypatch.setattr(supervision, "_control_wakes", lambda _ctx: [])
    _pending, tick = _quiet_question_tick(async_questions_only=async_only)
    tick.pop("pending_interactions")
    wake = json.loads(wake_now(_ctx(tmp_path), "run-a", wait_once=lambda *_a, **_k: json.dumps(tick)).text)
    assert "expiry details are unavailable" in wake["note"]
    assert "timeout_at is null" not in wake["note"] and "NO automatic expiry" not in wake["note"]
    assert ("PAUSED" in wake["note"]) is not async_only


@pytest.mark.parametrize("async_only", [True, False])
def test_expiry_absent_is_not_an_observed_null(async_only):
    from ouroboros.delegate_progress import waiting_expiry_clause

    kind = "async" if async_only else "blocking"
    assert "unavailable" in waiting_expiry_clause([{"delivery": kind}], async_only)
    assert "timeout_at is null" in waiting_expiry_clause([{"delivery": kind, "timeout_at": None}], async_only)


@pytest.mark.parametrize("oversize", ["wake_events", "sleep"], ids=["fitted-envelope", "last-resort-envelope"])
def test_both_spilled_envelopes_carry_the_question_facts(tmp_path, oversize):
    payload = {"status": "no_progress", "run_id": "run-a", "state": "running", "last_seq": 1,
               "waiting_on_user": True, "async_questions_only": True, "supervision_wake_id": "wake-1",
               "continuation": "same_session"}
    if oversize == "wake_events":
        payload["wake_events"] = [{"type": "addressed", "msg_id": f"m{i}", "text": "x" * 4000} for i in range(12)]
    else:
        payload["sleep"] = {"entered_at": "t", "detail": "y" * 40_000}
    shown = json.loads(supervision._render_wake_payload(_ctx(tmp_path, "t-spill"), payload).text)
    assert shown["wake_delivery"]["complete"] is False
    assert shown["waiting_on_user"] is True and shown["async_questions_only"] is True
