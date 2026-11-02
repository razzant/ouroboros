"""Truthful supervising-wait facts: the task-scoped child cursor across run ids and
recovery, whole-sleep facts at the single wake-publication point, dated observation
facts, and the leaf route's declared live-input capability."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import ouroboros.delegate_supervision as supervision
from ouroboros import task_tree_ledger
from ouroboros.delegate_supervision import acknowledge_pending_wake, supervised_wait
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result


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
    first = json.loads(supervised_wait(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert [e["child_task_id"] for e in _child_events(first)] == ["done"]
    assert len(_child_events(first, "child_attention_beacon")) == 1
    assert acknowledge_pending_wake(ctx, first)

    def _late_settles(*_a, **_k):
        write_task_result(tmp_path, "late", STATUS_COMPLETED, result="late result")
        return json.dumps({"status": "completed", "run_id": "run-2", "last_seq": 1})

    second = json.loads(supervised_wait(ctx, "run-2", wait_once=_late_settles).text)
    announced = [e["child_task_id"] for e in _child_events(second)]
    if carry:
        assert announced == ["late"]
        assert _child_events(second, "child_attention_beacon") == []
    else:
        assert "done" in announced  # the defect the carry repairs
    assert acknowledge_pending_wake(ctx, second)
    if carry:
        third = json.loads(supervised_wait(ctx, "run-3", wait_once=_settled("run-3")).text)
        assert _child_events(third) == [] and _child_events(third, "child_attention_beacon") == []


def test_an_unacked_child_terminal_survives_the_run_switch_and_emits_once(tmp_path):
    _child(tmp_path, "done", status=STATUS_COMPLETED)
    ctx = _ctx(tmp_path)
    first = json.loads(supervised_wait(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert [e["child_task_id"] for e in _child_events(first)] == ["done"]
    # Never acknowledged: its cursor never committed, so the new run re-detects it once.
    second = json.loads(supervised_wait(ctx, "run-2", wait_once=_settled("run-2")).text)
    assert [e["child_task_id"] for e in _child_events(second)] == ["done"]
    assert acknowledge_pending_wake(ctx, second)
    third = json.loads(supervised_wait(ctx, "run-3", wait_once=_settled("run-3")).text)
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
    clock = _Clock()
    _fake_time(monkeypatch, clock)
    ticks = []

    def wait_once(_ctx, run_id, _window, _cursor, **_k):
        ticks.append(1)
        clock.now += 3.0
        if len(ticks) <= 4:
            return json.dumps({"status": "no_progress", "run_id": run_id, "last_seq": len(ticks),
                               "waited_sec": 3, "note": "or delegate_cancel if it is stuck"})
        return json.dumps({"status": "completed", "run_id": run_id, "last_seq": 7})

    ctx = _ctx(tmp_path)
    wake = json.loads(supervised_wait(ctx, "run-1", since_seq=2, wait_once=wait_once).text)
    sleep = wake["sleep"]
    assert sleep["slept_sec"] == 15.0          # the whole call, not the last 3 s tick
    assert sleep["quiet_renewals"] == 4
    assert sleep["journal_advances"] == 5       # cursor 2 at entry -> 7 at the wake
    assert "opened_after" not in sleep
    clock.now += 500.0
    replay = json.loads(supervised_wait(ctx, "run-1", wait_once=lambda *_a, **_k: (_ for _ in ()).throw(
        AssertionError("a pending wake replays before any new observation"))).text)
    assert replay == wake                        # recorded facts, nothing added since


def test_a_quiet_status_wake_drops_the_tick_advice_but_keeps_the_paused_note(tmp_path, monkeypatch):
    from ouroboros.delegate_progress import paused_note

    calls: list = []
    monkeypatch.setattr(supervision, "_control_wakes", lambda _ctx: calls.append(1) or (
        [{"type": "deadline"}] if len(calls) > 1 else []))
    tick = {"status": "no_progress", "run_id": "run-q", "last_seq": 1, "waited_sec": 3,
            "quiet_for_sec": 3, "reason": "non_terminal", "note": "or delegate_cancel if it is stuck"}
    quiet = json.loads(supervised_wait(_ctx(tmp_path, "t-quiet"), "run-q",
                                       wait_once=lambda *_a, **_k: json.dumps(tick)).text)
    assert quiet["wake_events"] == [{"type": "deadline"}]
    assert not {"waited_sec", "quiet_for_sec", "note"} & set(quiet)
    assert quiet["sleep"]["quiet_renewals"] == 0

    pending = [{"interaction_id": "i1", "timeout_at": None}]
    calls.clear()
    paused = json.loads(supervised_wait(_ctx(tmp_path, "t-paused"), "run-q", wait_once=lambda *_a, **_k: json.dumps({
        **tick, "waiting_on_user": True, "pending_interactions": pending})).text)
    assert paused["note"] == paused_note(pending)
    assert "delegate_cancel" not in paused["note"] and "waited_sec" not in paused
    # A terminal wake is not quiet: its own fields stay, and it carries the sleep too.
    calls.clear()
    terminal = json.loads(supervised_wait(_ctx(tmp_path, "t-term"), "run-q", wait_once=lambda *_a, **_k: json.dumps({
        "status": "completed", "run_id": "run-q", "last_seq": 1, "note": "terminal note"})).text)
    assert terminal["note"] == "terminal note" and "sleep" in terminal


def test_an_old_pending_wake_without_sleep_replays_unchanged(tmp_path):
    ctx = _ctx(tmp_path)
    stored = {"status": "completed", "run_id": "run-1", "supervision_wake_id": "w-old"}
    path = supervision._state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "run_id": "run-1", "pending_wake": {
        "wake_id": "w-old", "attempt_key": "1", "payload": stored}}), encoding="utf-8")
    replay = json.loads(supervised_wait(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert replay == stored                       # nothing invented for an old record


@pytest.mark.parametrize("prior,label", [
    ("adopted", "worker_loss_adoption"), ("sleeping", "interrupted_sleep"), ("awake", None),
])
def test_a_new_call_opens_a_new_sleep_labelled_by_the_prior_status(tmp_path, prior, label):
    ctx = _ctx(tmp_path)
    path = supervision._state_path(ctx)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "run_id": "run-1", "status": prior, "sleep_entry": {
        "entered_at": "2026-09-26T00:00:00+00:00", "entered_at_unix": 1.0}}), encoding="utf-8")
    wake = json.loads(supervised_wait(ctx, "run-1", wait_once=_settled("run-1")).text)
    assert wake["sleep"].get("opened_after") == label
    assert wake["sleep"]["slept_sec"] < 60        # never carried over from the old entry
    assert ("previous_entered_at" in wake["sleep"]) is (prior == "sleeping")


def test_a_spilled_wake_keeps_the_sleep_and_live_input_facts(tmp_path, monkeypatch):
    import ouroboros.tool_capabilities as capabilities

    monkeypatch.setattr(capabilities, "tool_result_limit", lambda _name: 900)
    ctx = _ctx(tmp_path)
    ctx.task_contract = {"delegation_budget": {"intent_note": "x" * 5000}}
    wake = json.loads(supervised_wait(ctx, "run-s", wait_once=_settled("run-s")).text)
    assert wake["wake_delivery"]["complete"] is False
    assert wake["sleep"]["quiet_renewals"] == 0 and wake["leaf_live_input"] == "unknown"


# -- D1: dated observation facts (freshness is never the file's mtime) ---------


def test_observation_facts_separate_answered_reads_from_failures(tmp_path, monkeypatch):
    _fake_time(monkeypatch, _Clock())
    replies = iter([
        {"status": "no_progress", "run_id": "run-o", "state": "running", "last_seq": 1},
        {"status": "observation_pending", "run_id": "run-o", "reason": "daemon_unreachable"},
    ])
    seen = []

    def wait_once(ctx, *_a, **_k):
        try:
            return json.dumps(next(replies))
        except StopIteration:
            seen.append(_state(ctx))
            return json.dumps({"status": "completed", "run_id": "run-o", "last_seq": 2})

    ctx = _ctx(tmp_path)
    supervised_wait(ctx, "run-o", wait_once=wait_once)
    before_wake = seen[0]
    assert before_wake["observation_failure"] == "daemon_unreachable"
    assert before_wake["observation"]["answered"] is False
    answered_at = before_wake["last_answered_observation_at"]
    assert answered_at                          # the earlier ANSWERED read, not advanced
    assert "state_written_at" not in before_wake
    after = _state(ctx)
    assert after["observation_failure"] == "" and after["last_answered_observation_at"] >= answered_at


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
    ([{"id": "other", "liveInput": "mid_turn"}], False, "unknown"),
    ([], True, "unknown"),                                # failure is a fact, never a refusal
])
def test_leaf_live_input_is_read_once_at_entry_and_stamped_on_every_wake(
    tmp_path, monkeypatch, rows, fail, expected,
):
    _owned(tmp_path, monkeypatch)
    catalog = _Catalog(rows, fail)
    monkeypatch.setattr(supervision, "_loop_gateway", lambda: catalog)
    monkeypatch.setattr(supervision.time, "sleep", lambda _s: None)
    ticks = []

    def wait_once(_ctx, run_id, *_a, **_k):
        ticks.append(1)
        status = "no_progress" if len(ticks) < 4 else "completed"
        return json.dumps({"status": status, "run_id": run_id, "last_seq": len(ticks)})

    ctx = _ctx(tmp_path)
    wake = json.loads(supervised_wait(ctx, "run-l", wait_once=wait_once).text)
    assert wake["leaf_live_input"] == expected
    assert catalog.reads == [supervision._TICK_SEC]    # once per call, never per tick
    assert catalog.closed                               # the one-read transport is dropped
    replay = json.loads(supervised_wait(ctx, "run-l", wait_once=wait_once).text)
    assert replay["leaf_live_input"] == expected and catalog.reads == [supervision._TICK_SEC]


def test_an_unowned_run_reads_no_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(supervision, "_loop_gateway", lambda: (_ for _ in ()).throw(
        AssertionError("no route, no catalog read")))
    wake = json.loads(supervised_wait(_ctx(tmp_path), "run-x", wait_once=_settled("run-x")).text)
    assert wake["leaf_live_input"] == "unknown"
