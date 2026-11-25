"""``delegate_wait`` is a sleep on runs; its wake is published once and acknowledged after whole delivery.

Driven through the real arming (``delegate_supervision.run_wait`` -> ``model_sleep.request_default_sleep``), the
sleep's own readiness (``model_sleep.wake_reason`` with the ``run_wait`` sources), the owed delivery at the round
top (``native_task_wait.deliver_owed``), the one wake seam (``deliver_wake``) and the deferred acknowledgement the
next saved state flushes (``working_checkpoint``). Only the one addressed read of the run a wake speaks for is
scripted (``tools.delegate._read_run``); the body that records terminals and questions is the custody
writer itself (``delegate_collection.observe_run``). Each guard is shown firing beside the case it lets through.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests._run_wait_shared import _NANNY, _asks, _body, _ended, _nanny, _notices, _run, reads as reads
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines  # noqa: F401

def _armed(result):
    from ouroboros.delegate_shared import delegate_payload

    return delegate_payload(result)


def _pending(ctx, run_id):
    from ouroboros import delegate_supervision as ds

    state = ds._load_state(ctx, run_id)
    return (state.get("pending_wake") or {}).get("wake_id", ""), (state.get("last_acknowledged_wake") or {}).get(
        "wake_id", "")


@pytest.mark.parametrize("continuation", [True, False])
def test_already_pending_unselected_question_reaches_immediate_wait_result(tmp_path, reads, continuation):
    from ouroboros import delegate_supervision as ds, model_sleep
    from ouroboros.working_checkpoint import flush_content_acks

    ctx, _limit = _nanny(tmp_path)
    if not continuation:
        ctx.owner_wait_callback = None
    _run(tmp_path, "run-selected")
    _run(tmp_path, "run-other")
    _asks(tmp_path, "run-other", "unselected-question")
    first = ds.run_wait(ctx, "run-selected")
    payload = _armed(first)
    assert payload["sleep"]["woke_by"] == "run_question:run-other"
    assert "unselected-question" in first.text and payload["questions"][0]["interaction_source"]
    assert not getattr(ctx, "_sleep_interaction_seen", None), "publication alone exposes nothing"
    row = {"result_meta": {"tool_result_meta": first.meta}}
    ds.deliver_wake(ctx, row, {"result_partial": True}, "shortened")
    assert not getattr(ctx, "_sleep_interaction_seen", None), "unshown interactions stay unseen"
    ctx.task_attempt = 2  # another attempt must keep this quiet wake's unshown question sources
    assert _armed(ds.run_wait(ctx, "run-selected"))["supervision_wake_id"] == payload["supervision_wake_id"]
    ds.deliver_wake(ctx, row, {"result": first.text}, first.text)
    flush_content_acks(ctx)
    assert ctx._sleep_interaction_seen == {"run-other": ["unselected-question"]}
    assert not model_sleep.wake_reason(ctx, {"runs": ["run-selected"]})
    ctx.owner_wait_callback = lambda *_a, **_k: "unknown"
    assert _armed(ds.run_wait(ctx, "run-selected"))["reason"] == "sleep_armed"


@pytest.mark.parametrize("shape", ["outer_cut", "inner_spill"])
def test_unselected_question_exposure_waits_for_the_exact_wake_source(tmp_path, monkeypatch, shape):
    from tests.test_delegate_wake_mail_delivery import _call, _crowded, _nanny as registry_nanny, _page, _ready, _shown, _wide

    registry, ctx = registry_nanny(tmp_path, monkeypatch, lambda _ctx, rid, *_a, **_kw: json.dumps(
        {"status": "no_progress", "run_id": rid, "last_seq": 0}), runs=())
    _run(tmp_path, "run-selected")
    _run(tmp_path, "run-other")
    _asks(tmp_path, "run-other", "unselected-question")
    ctx.task_contract = {"delegation_budget": {"intent_note": "x" * (30_000 if shape == "inner_spill" else 4_000)}}
    fit, messages, trace = _wide() if shape == "inner_spill" else _crowded()
    wake = _call(registry, tmp_path, "wait", "delegate_wait", {"run_id": "run-selected"})
    _shown(registry, wake, messages, trace, fit)
    source, = ctx._wake_sources.values()
    assert not getattr(ctx, "_sleep_interaction_seen", None)
    half, fit = source["size"] // 2, _wide()[0]
    _shown(registry, _call(registry, tmp_path, "half", "read_file", _page(source, 0, half)), messages, trace, fit)
    _ready(registry, messages, trace)
    assert not getattr(ctx, "_sleep_interaction_seen", None), "half a source is not a delivered notice"
    _shown(registry, _call(registry, tmp_path, "rest", "read_file", _page(source, half, source["size"] + 1)),
           messages, trace, fit)
    assert ctx._sleep_interaction_seen == {"run-other": ["unselected-question"]}
    _ready(registry, messages, trace)
    next_wait = _call(registry, tmp_path, "again", "delegate_wait", {"run_id": "run-selected"})
    assert json.loads(next_wait["result"])["reason"] == "sleep_armed"


@pytest.mark.parametrize("sleep_mode,mode", [(None, "cold"), ("warm", "warm")])
def test_a_wait_on_a_quiet_run_sleeps_and_its_result_is_owed_to_the_next_request(tmp_path, reads, sleep_mode, mode):
    """Nothing there yet: the call arms the task's ordinary sleep on its runs (no slot, no tick, no read) and owes its
    result; the run's terminal wakes the sleep, and the first request after it carries one wake, published then and
    acknowledged only after the next saved state, never before."""
    from ouroboros import delegate_supervision as ds, model_sleep
    from ouroboros.native_task_wait import deliver_owed
    from ouroboros.working_checkpoint import flush_content_acks

    ctx, limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    armed = _armed(ds.run_wait(ctx, "run-1", sleep_mode=sleep_mode))
    assert armed["reason"] == "sleep_armed" and armed["mode"] == mode, "omitted: the body takes the run, nothing keeps it"
    assert ctx._model_sleep["runs"] == ["run-1"] and ctx._model_sleep["any_mail"] and ctx._model_sleep["run_wait"]
    assert ctx._pending_task_wait["tool"] == "delegate_wait" and ctx._pending_task_wait["runs"] == ["run-1"]
    assert reads == [] and _pending(ctx, "run-1") == ("", ""), "arming reads no run and publishes no wake"
    from ouroboros import delegate_custody as custody

    trail = [row for row in map(json.loads, custody.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines())
             if row["type"] == "delegate_supervision_wait_entered"]
    assert [(row["run_id"], row["outcome"], row["mode"]) for row in trail] == [("run-1", "sleep_armed", mode)]
    chosen = dict(ctx._model_sleep)
    assert model_sleep.wake_reason(ctx, chosen) == "", "a quiet run does not wake its sleep"
    _ended(tmp_path, "run-1")
    assert model_sleep.wake_reason(ctx, chosen) == "run:run-1"
    ctx._model_sleep, ctx._owner_wait_requested = None, ""  # the park took it; the loop woke on the run
    assert deliver_owed(limit) and ctx._pending_task_wait is None
    (notice,) = _notices(limit)
    wake = _body(notice)
    assert wake["status"] == "terminal" and wake["run_id"] == "run-1" and wake["sleep"]["woke_by"] == "run:run-1"
    assert "The delegate_wait you chose answers here" in notice and reads == [("run-1", 0)]
    assert _pending(ctx, "run-1") == (wake["supervision_wake_id"], ""), "published; not acknowledged before a save"
    flush_content_acks(ctx)  # the next saved state (``save_ready``) flushes exactly this
    assert _pending(ctx, "run-1") == ("", wake["supervision_wake_id"])
    assert not deliver_owed(limit), "owed once: nothing more to deliver"


def test_what_is_already_there_answers_at_once_and_nothing_parks(tmp_path, reads):
    """A wait whose run already ended, or which already asks, answers through its own result: one published wake, no
    sleep armed, nothing owed. Its acknowledgement follows the tool result's whole delivery (``deliver_wake``)."""
    from ouroboros import delegate_supervision as ds

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-done")
    _ended(tmp_path, "run-done")
    result = ds.run_wait(ctx, "run-done")
    assert result.meta["supervision_wake_id"] and _armed(result)["status"] == "terminal"
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
    _run(tmp_path, "run-asks")
    _asks(tmp_path, "run-asks")
    asked = _armed(ds.run_wait(ctx, "run-asks"))
    assert asked["status"] == "waiting_on_user" and asked["sleep"]["woke_by"] == "run_question:run-asks"
    assert [iid for row in asked["pending_interactions"] for iid in [row["interaction_id"]]] == ["q1"]


@pytest.mark.parametrize("sleep_mode,mode", [(None, "cold"), ("cold", "cold"), ("warm", "warm")])
def test_a_named_sleep_mode_arms_that_sleep_omitted_keeps_the_default_and_ready_answers_at_once(
        tmp_path, reads, sleep_mode, mode):
    """``delegate_wait`` chooses its sleep as the task waits do (``native_task_wait.sleep_choice``): warm keeps the
    process, cold releases it, and an omitted mode keeps the owner's default (here cold: nothing keeps the process).
    A run that already ended answers through the result in that mode, and nothing parks."""
    from ouroboros import delegate_supervision as ds

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-done")
    _ended(tmp_path, "run-done")
    assert _armed(ds.run_wait(ctx, "run-done", sleep_mode=sleep_mode))["status"] == "terminal"
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
    _run(tmp_path, "run-quiet")
    armed = _armed(ds.run_wait(ctx, "run-quiet", sleep_mode=sleep_mode))
    assert armed["reason"] == "sleep_armed" and armed["mode"] == mode == ctx._model_sleep["mode"]
    assert ctx._pending_task_wait["arguments"].get("sleep_mode") == sleep_mode and reads == [("run-done", 0)]


@pytest.mark.parametrize("sleep_mode", ["warm", "cold"])
def test_with_no_continuation_owner_what_is_there_answers_in_a_named_sleep_mode_and_only_a_sleep_is_refused(
        tmp_path, reads, sleep_mode):
    """A result already there needs nothing to sleep under: it answers in whatever mode was named, owner or not. Only
    a call that would arm a sleep is refused when the task has no continuation owner; an unknown mode is refused
    first, even over a ready run."""
    from ouroboros import delegate_custody as custody, delegate_supervision as ds
    from ouroboros.delegate_shared import AGENT_FAULT_CODE

    ctx, _limit = _nanny(tmp_path)
    owner, ctx.owner_wait_callback = ctx.owner_wait_callback, None
    _run(tmp_path, "run-done")
    _ended(tmp_path, "run-done")
    unknown = ds.run_wait(ctx, "run-done", sleep_mode="hot")
    assert _armed(unknown)["reason"] == "sleep_mode_refused" and unknown.code == AGENT_FAULT_CODE and not reads
    assert _armed(ds.run_wait(ctx, "run-done", sleep_mode=sleep_mode))["status"] == "terminal"
    _run(tmp_path, "run-quiet")
    refused = ds.run_wait(ctx, "run-quiet", sleep_mode=sleep_mode, checkpoint_after_sec=300, checkpoint_reason="look")
    assert _armed(refused)["reason"] == "sleep_mode_refused" and refused.code == AGENT_FAULT_CODE
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
    assert reads == [("run-done", 0)]
    kinds = [json.loads(line)["type"] for line in custody.event_log_path(tmp_path).read_text().splitlines()]
    assert "delegate_supervision_checkpoint_scheduled" not in kinds, "a refused wait schedules no checkpoint"
    ctx.owner_wait_callback = owner  # the same call that is accepted and armed does schedule one
    assert _armed(ds.run_wait(ctx, "run-quiet", sleep_mode=sleep_mode, checkpoint_after_sec=300,
                              checkpoint_reason="look"))["reason"] == "sleep_armed"
    kinds = [json.loads(line)["type"] for line in custody.event_log_path(tmp_path).read_text().splitlines()]
    assert kinds.count("delegate_supervision_checkpoint_scheduled") == 1


def test_an_in_slot_wait_keeps_the_call_after_what_an_earlier_call_of_the_batch_armed_or_staged(
        tmp_path, monkeypatch, reads):
    """``in_slot`` waits in this call on the sleep's own sources although the task could sleep. What an earlier call
    of the batch armed or staged comes first, before any slot is taken, as the task waits do: its unshown hold answers
    at once as ``dependency_held`` in any mode (once delivered, the same wait sleeps), and its armed wait refuses the
    next one (``armed_conflict``), which would otherwise hold that wait's wake behind this call."""
    from ouroboros import delegate_custody as custody, delegate_supervision as ds, model_sleep

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    beats, waited_on = [], ["run-1"]  # each beat ends the run the slot waits on
    monkeypatch.setattr(model_sleep, "declare_in_slot_wait", lambda *_a, **_k: None)
    monkeypatch.setattr(model_sleep, "end_in_slot_wait", lambda *_a, **_k: None)
    monkeypatch.setattr("supervisor.owner_stop.in_process_stop_now", lambda _ctx: False)
    monkeypatch.setattr(ds.time, "sleep", lambda _sec: beats.append(waited_on[0]) or _ended(tmp_path, waited_on[0]))
    assert _armed(ds.run_wait(ctx, "run-1", sleep_mode="in_slot"))["status"] == "terminal" and beats == ["run-1"]
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
    _run(tmp_path, "run-2")
    model_sleep.hold_notice(ctx, [{"task_id": "held-child", "identity": "hold-1"}])
    for sleep_mode in ("in_slot", None):
        waited_on[0] = "run-2"
        held = ds.run_wait(ctx, "run-2", sleep_mode=sleep_mode, checkpoint_after_sec=300, checkpoint_reason="look")
        assert _armed(held)["reason"] == "dependency_held" and held.meta == {model_sleep.HOLD_NOTICE: True}
    rows = custody.event_log_path(tmp_path).read_text().splitlines()
    assert not any(json.loads(row)["type"] == "delegate_supervision_checkpoint_scheduled" for row in rows), "held: none"
    assert beats == ["run-1"] and not getattr(ctx, "_model_sleep", None)
    assert not getattr(ctx, "_pending_task_wait", None)
    ctx._held_dependencies_pending = []  # the batch delivered the hold
    assert _armed(ds.run_wait(ctx, "run-2"))["reason"] == "sleep_armed"
    _run(tmp_path, "run-3")
    waited_on[0] = "run-3"
    refused = _armed(ds.run_wait(ctx, "run-3", sleep_mode="in_slot"))
    assert refused["reason"] == "wait_already_armed" and beats == ["run-1"]
    assert ctx._pending_task_wait["runs"] == ["run-2"] and reads == [("run-1", 0)]


@pytest.mark.parametrize("local_first", [False, True])
def test_a_run_only_this_process_holds_is_read_at_once_unless_a_selected_durable_run_is_ready(tmp_path, reads,
                                                                                              local_first):
    """``started_uncustodied``: a start whose STARTED row never landed is still this task's run, held in this process
    alone (``custody.lookup``). Neither the sleep nor the body reads that custody, so a wait selecting it parks nothing
    and answers now, in either selection order: with what is already there, a selected durable run's terminal or
    question, as the ordinary wait would, else with the in-process run's one read. A run nobody holds and a sibling's
    are refused as before, reading nothing."""
    from ouroboros import delegate_custody as custody, delegate_supervision as ds

    ctx, _limit = _nanny(tmp_path)
    held = {rid: custody.RunCustody(run_id=rid, task_id=owner, route_id="route", model="m", project_id="p",
                                    project_owned=False)
            for rid, owner in (("run-local", _NANNY), ("run-sibling", "t-b"))}
    custody._CUSTODY.update(held)
    try:
        nobody = _armed(ds.run_wait(ctx, "run-nobody"))
        assert nobody["reason"] == "run_ownership_unknown" and "hint" not in nobody
        # A task id given as a run id: the public wait names the read that answers it.
        from ouroboros.task_results import write_task_result
        from ouroboros.tools.control_task_results import _status_root

        write_task_result(_status_root(ctx), "kidtask09", "completed", result="done")
        task = _armed(ds.run_wait(ctx, "kidtask09"))
        assert task["reason"] == "run_ownership_unknown" and task["hint"] == (
            "That id names a task, not a delegated run: get_task_result(task_id='kidtask09') reads it.")
        # An id this task's custody OWNS is never called a task, even when a task record shares it.
        write_task_result(_status_root(ctx), "run-local", "completed", result="done")
        mixed = _armed(ds.run_wait(ctx, ["run-local", "run-nobody"]))
        assert mixed["reason"] == "run_ownership_unknown" and "hint" not in mixed
        assert _armed(ds.run_wait(ctx, "run-sibling"))["reason"] == "run_not_owned" and reads == []
        for durable, ready, status, woke_by in (("run-quiet", None, "no_progress", None),
                                                ("run-done", _ended, "terminal", "run:run-done"),
                                                ("run-asks", _asks, "waiting_on_user", "run_question:run-asks")):
            _run(tmp_path, durable)
            if ready is not None:
                ready(tmp_path, durable)
            result = ds.run_wait(ctx, ["run-local", durable] if local_first else [durable, "run-local"])
            answered, read = _armed(result), "run-local" if ready is None else durable
            sleep = answered.get("sleep") or {}
            assert (answered.get("run_id"), answered.get("status"), sleep.get("mode"), sleep.get("woke_by")) == (
                read, status, "answered_at_call", woke_by), answered
            assert reads[-1][0] == read, reads
            assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
            assert ds.acknowledge_pending_wake(ctx, result.text)  # the next selection of run-local reads it afresh
        assert [run_id for run_id, _seq in reads] == ["run-local", "run-done", "run-asks"]
    finally:
        for rid in held:
            custody._CUSTODY.pop(rid, None)


def test_an_unacknowledged_wake_is_replayed_marked_before_any_sleep_is_armed(tmp_path, reads):
    """S2: a wake published but never acknowledged (a refused ACK, a lost process) is the next wait's answer, marked as
    the same wake and not a new observation; only after it is acknowledged does the next wait sleep again."""
    from ouroboros import delegate_supervision as ds

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    _ended(tmp_path, "run-1")
    first = ds.run_wait(ctx, "run-1")
    wake_id = first.meta["supervision_wake_id"]
    replay = ds.run_wait(ctx, "run-1")
    payload = _armed(replay)
    assert payload["supervision_wake_id"] == wake_id and payload["replay_of"]["fresh_observation"] is False
    assert len(reads) == 1 and not getattr(ctx, "_model_sleep", None), "a replay reads nothing and arms nothing"
    assert ds.acknowledge_pending_wake(ctx, first.text)
    assert _pending(ctx, "run-1") == ("", wake_id)
    again = ds.run_wait(ctx, "run-1")  # its terminal is still there: answered again, as a new wake
    assert again.meta["supervision_wake_id"] != wake_id and len(reads) == 2
    _run(tmp_path, "run-quiet")  # nothing ready on this run: still its stored wake answers, before any sleep
    stored = {"status": "no_progress", "run_id": "run-quiet", "supervision_wake_id": "stored"}
    ds._save_state(ctx, {"run_id": "run-quiet", "pending_wake": {"wake_id": "stored", "attempt_key": "1",
                                                                 "payload": stored, "created_at": "2026-11-09T00:00:00"}})
    quiet = _armed(ds.run_wait(ctx, "run-quiet"))
    assert quiet["supervision_wake_id"] == "stored" and quiet["replay_of"]["published_at"] == "2026-11-09T00:00:00"
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)


def test_an_old_tick_wake_with_mail_is_delivered_once(tmp_path, reads):
    """S5: a pending wake the retired tick stored (schema 2, mail inside) reaches the next wait once, its mail
    acknowledged with it; the wait after that observes again instead of replaying it."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    assert write_owner_message(tmp_path, "Keep the old shape.", _NANNY, msg_id="old-mail")
    events = [{"type": "addressed_message", "msg_id": "old-mail", "kind": "owner_text", "provenance": "owner",
               "text": "Keep the old shape.", "ts": ""}]  # the tick put mail inside; a wake now carries controls only
    old = {"status": "progress", "run_id": "run-1", "wake_events": events, "supervision_wake_id": "tick-wake"}
    ds._save_state(ctx, {"run_id": "run-1", "status": "wake_pending", "pending_wake": {
        "wake_id": "tick-wake", "attempt_key": "1", "payload": old, "mailbox_ids": ["old-mail"],
        "seen_mailbox_ids": ["old-mail"], "interaction_ids": [], "created_at": "2026-11-01T00:00:00+00:00"}})
    replay = ds.run_wait(ctx, "run-1")
    assert _armed(replay)["supervision_wake_id"] == "tick-wake" and "replay_of" in _armed(replay)
    assert ds.acknowledge_pending_wake(ctx, replay.text)
    assert drain_owner_entries(tmp_path, _NANNY, set(), 1) == [], "its mail was acknowledged with it"
    assert _armed(ds.run_wait(ctx, "run-1"))["reason"] == "sleep_armed", "delivered once: the next wait sleeps"


def test_a_crash_between_publication_and_acknowledgement_replays_the_same_wake_through_two_restarts(
        tmp_path, reads):
    """Restart x2 and a crash in delivery: the owed descriptor rides the continuation without its publication, so each
    restored process delivers the one stored wake (marked as a replay), never a second publication; the first saved
    state after a whole delivery acknowledges it."""
    from ouroboros import delegate_supervision as ds, owner_wait
    from ouroboros.native_task_wait import deliver_owed
    from ouroboros.working_checkpoint import flush_content_acks

    ctx, limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    _armed(ds.run_wait(ctx, "run-1"))
    saved = json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 2, [], set())))  # the park's source
    ctx._model_sleep, ctx._owner_wait_requested = None, ""
    _ended(tmp_path, "run-1")
    assert deliver_owed(limit)
    published = _body(_notices(limit)[0])["supervision_wake_id"]
    ctx._pending_content_acks = []  # the process dies before its next saved state
    for restart in (1, 2):
        restored, again = _nanny(tmp_path)
        owner_wait.restore_continuation_state(SimpleNamespace(_ctx=restored), saved, [], {}, {}, set())
        assert restored._pending_task_wait["tool"] == "delegate_wait" and "wake" not in restored._pending_task_wait
        assert deliver_owed(again)
        wake = _body(_notices(again)[0])
        assert wake["supervision_wake_id"] == published and "replay_of" in wake, restart
        assert len(reads) == 1, "no second observation, no second publication"
        if restart == 1:
            restored._pending_content_acks = []  # this one dies too
    flush_content_acks(restored)
    assert _pending(restored, "run-1") == ("", published)


def test_forty_one_runs_in_one_wait_need_no_count_limit(tmp_path, reads):
    """E4: the 32-id selection cap is gone with the tick; a wait on 41 runs arms like a wait on one."""
    from ouroboros import delegate_supervision as ds

    ctx, _limit = _nanny(tmp_path)
    runs = [_run(tmp_path, f"run-{index}").run_id for index in range(41)]
    armed = _armed(ds.run_wait(ctx, runs))
    assert armed["reason"] == "sleep_armed" and armed["runs"] == runs and len(ctx._model_sleep["runs"]) == 41


def test_the_sleep_on_runs_wakes_on_what_its_tick_woke_on_and_on_nothing_else(tmp_path, monkeypatch, reads):
    """The four tick wake sources are sleep sources now: a selected run's terminal, a question from any run this task
    owns (selected or not), addressed mail, the task's cancel or spent deadline, a direct child's terminal or
    attention; a quiet run and another task's mail wake nothing."""
    from ouroboros import cancel_intents, delegate_supervision as ds, model_sleep
    from ouroboros.owner_mailbox import write_task_message
    from ouroboros.task_results import write_task_result

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-sel")
    _run(tmp_path, "run-other")
    chosen = ds._choice({"runs": ["run-sel"]})
    assert model_sleep.wake_reason(ctx, chosen) == ""
    assert write_task_message(tmp_path, "for someone else", "elsewhere", msg_id="not-mine", source_task_id="peer")
    assert model_sleep.wake_reason(ctx, chosen) == "", "mail addressed to another task wakes nothing"
    _asks(tmp_path, "run-other")
    assert model_sleep.wake_reason(ctx, chosen) == "run_question:run-other", "an unselected own run's question"
    ctx._sleep_interaction_seen = {"run-other": ["q1"]}
    assert model_sleep.wake_reason(ctx, chosen) == ""
    write_task_result(tmp_path, "kid", "completed", parent_task_id=_NANNY, root_task_id=_NANNY, chat_id=0,
                      delegation_role="subagent")
    assert model_sleep.wake_reason(ctx, chosen) == "coordination:kid", "a direct child's terminal"
    state = ds._load_state(ctx, "")
    state["coordination_cursor"] = ds._coordination_wakes(ctx, state)[1]  # acknowledged: the cursor moved past it
    ds._save_state(ctx, state)
    assert model_sleep.wake_reason(ctx, chosen) == ""
    assert write_task_message(tmp_path, "a note for the nanny", _NANNY, msg_id="mine", source_task_id="peer")
    assert model_sleep.wake_reason(ctx, chosen) == "mail:peer"
    ctx._loop_mailbox_seen_ids = {"mine"}
    assert model_sleep.wake_reason(ctx, chosen) == ""
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda _root, task_id: task_id == _NANNY)
    assert model_sleep.wake_reason(ctx, chosen) == "control:cancelled"
    monkeypatch.setattr(cancel_intents, "cancel_pending", lambda _root, _task_id: False)
    ctx.task_metadata["deadline_at"] = "2020-01-01T00:00:00+00:00"
    assert model_sleep.wake_reason(ctx, chosen) == "control:deadline"
    ctx.task_metadata.pop("deadline_at")
    _ended(tmp_path, "run-sel")
    assert model_sleep.wake_reason(ctx, chosen) == "run:run-sel"


def test_a_sleepers_tick_reads_no_child_whose_end_its_last_wake_showed(tmp_path, monkeypatch):
    """Every assignment tick asks each sleeper on runs whether a direct child ended or asks for attention. An end the
    last acknowledged wake already showed ends no sleep, so the tick reads neither that child's row nor its replica (a
    nanny with dozens of ended children decoded them all on every tick). The running child is read and wakes it when it
    ends; a beacon of an ended child still wakes it, and so does the end of a retry an ended child is redirected to,
    even one that names no parent and is a child only through the ended child's queued row."""
    from ouroboros import delegate_supervision as ds, model_sleep, task_status, task_store
    from ouroboros.task_results import write_task_result
    from ouroboros.task_tree_ledger import tree_ledger_append

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-sel")
    chosen = ds._choice({"runs": ["run-sel"]})

    def child(task_id, status):
        write_task_result(tmp_path, task_id, status, parent_task_id=_NANNY, root_task_id=_NANNY, chat_id=0,
                          delegation_role="subagent")

    def acknowledged():  # the nanny's next wake showed what is there: the cursor moves past it
        state = ds._load_state(ctx, "")
        state["coordination_cursor"] = ds._coordination_wakes(ctx, state)[1]
        ds._save_state(ctx, state)

    ended = {"ended-1", "ended-2", "ended-3"}
    for task_id in sorted(ended):
        child(task_id, "completed")
    child("busy", "running")
    acknowledged()
    read, documents, effective = [], task_store._documents, task_status.effective_task_result
    monkeypatch.setattr(task_store, "_documents", lambda pages: [
        read.append(task_id) or (task_id, document) for task_id, document in documents(pages)])
    monkeypatch.setattr(task_status, "effective_task_result", lambda root, row, **kw: read.append(
        row.get("task_id")) or effective(root, row, **kw))
    assert model_sleep.wake_reason(ctx, chosen) == ""
    assert "busy" in read and not ended & set(read), "only the child whose end is not shown yet is read"
    assert tree_ledger_append(_NANNY, "blocker", "stuck", task_id="ended-2", role="subagent",
                              needs_parent_attention=True, data_root=tmp_path).startswith("OK")
    assert model_sleep.wake_reason(ctx, chosen) == "coordination:ended-2", "an ended child's beacon still wakes"
    acknowledged()
    child("busy", "completed")
    assert model_sleep.wake_reason(ctx, chosen) == "coordination:busy"
    acknowledged()
    child("retry-1", "completed")
    task_store.patch_document(tmp_path, "ended-1", lambda current: {**current, "retry_task_id": "retry-1"})
    assert model_sleep.wake_reason(ctx, chosen) == "coordination:retry-1"
    acknowledged()
    # a retry that names no parent is this nanny's child only through the queued row of the ended child it redirects
    write_task_result(tmp_path, "retry-3", "completed", chat_id=0)
    task_store.patch_document(tmp_path, "ended-3", lambda current: {**current, "retry_task_id": "retry-3"})
    queued = {"id": "ended-3", "parent_task_id": _NANNY, "delegation_role": "subagent"}
    (tmp_path / "state" / "queue_snapshot.json").write_text(
        json.dumps({"pending": [], "running": [{"id": "ended-3", "task": queued}]}), encoding="utf-8")
    assert model_sleep.wake_reason(ctx, chosen) == "coordination:retry-3", "the queue still recovers a redirected child"
    acknowledged()
    read.clear()
    assert model_sleep.wake_reason(ctx, chosen) == "" and "busy" not in read


def test_a_sleeper_wakes_only_on_what_its_acknowledging_read_shows(tmp_path, monkeypatch):
    """The tick's read that skips ended children only says that nothing is new: on an event it reads every child again,
    as the wake's acknowledging read does. While an ended child cannot be read, both reads are blind and the sleep stays
    quiet, as before; it never wakes on a sibling's end that its wake cannot show (every acknowledged delegate_wait
    would then return at once, showing nothing)."""
    from ouroboros import delegate_supervision as ds, model_sleep, task_status
    from ouroboros.task_results import write_task_result

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-sel")
    chosen = ds._choice({"runs": ["run-sel"]})
    for task_id, status in (("ended", "completed"), ("busy", "running"), ("busy", "completed")):
        write_task_result(tmp_path, task_id, status, parent_task_id=_NANNY, root_task_id=_NANNY, chat_id=0,
                          delegation_role="subagent")
        if status == "running":
            state = ds._load_state(ctx, "")
            state["coordination_cursor"] = ds._coordination_wakes(ctx, state)[1]  # acknowledged: "ended" shown
            ds._save_state(ctx, state)
    effective = task_status.effective_task_result

    def unreadable(root, row, **kw):
        if row.get("task_id") == "ended":
            raise OSError("the ended child's store cannot be read")
        return effective(root, row, **kw)

    monkeypatch.setattr(task_status, "effective_task_result", unreadable)
    assert ds._coordination_wakes(ctx, ds._load_state(ctx, ""))[0] == [], "the acknowledging read is blind"
    assert model_sleep.wake_reason(ctx, chosen) == "", "so the tick is quiet too"
    monkeypatch.setattr(task_status, "effective_task_result", effective)
    assert model_sleep.wake_reason(ctx, chosen) == "coordination:busy"


def test_the_wake_read_is_one_detail_read_and_never_the_journal(tmp_path):
    """(c) The body is the only journal reader: a wake's addressed read of its run is one detail read, no window, no
    run-event stream and no speech, whatever the engine serves."""
    from ouroboros.tools import delegate

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    calls = []

    class Gateway:
        engine_version = "9.9.9"

        def get_run(self, run_id, **_kw):
            calls.append(("get_run", run_id))
            return {"summary": {"runId": run_id, "state": "running"}, "lastSeq": 4}

        def open_run_events(self, *_a, **_kw):
            pytest.fail("a wake's read must not read the run's journal")

        def __getattr__(self, name):
            calls.append((name,))
            return lambda *_a, **_k: {}

    ctx.emit_progress_fn = lambda *_a, **_k: pytest.fail("a wake's read publishes no speech")
    payload = json.loads(delegate._read_run(ctx, "run-1", 2, gateway=Gateway()))
    assert payload["status"] in {"progress", "no_progress"} and payload["last_seq"] == 4
    assert calls.count(("get_run", "run-1")) == 1


def test_a_turn_with_no_continuation_owner_waits_in_its_slot_on_the_same_sources(tmp_path, monkeypatch, reads):
    """A Presence turn cannot sleep: its delegate_wait keeps the process (yielding acting capacity on its runs) until a
    source of the same sleep is ready, then answers through its result; a task that can sleep never does this."""
    from ouroboros import delegate_supervision as ds, model_sleep

    ctx, _limit = _nanny(tmp_path)
    ctx.owner_wait_callback = None
    _run(tmp_path, "run-1")
    marks, beats = [], []
    monkeypatch.setattr(model_sleep, "declare_in_slot_wait", lambda _ctx, key, _until, **_k: marks.append(key))
    monkeypatch.setattr(model_sleep, "end_in_slot_wait", lambda _ctx, key: marks.append("end:" + key))
    monkeypatch.setattr("supervisor.owner_stop.in_process_stop_now", lambda _ctx: False)

    def beat(_sec):
        beats.append(1)
        _ended(tmp_path, "run-1")

    monkeypatch.setattr(ds.time, "sleep", beat)
    wake = _armed(ds.run_wait(ctx, "run-1"))
    assert wake["status"] == "terminal" and beats == [1] and marks == ["delegate:run-1", "end:delegate:run-1"]
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)


def test_the_unknown_outcome_hold_parks_a_host_armed_sleep_and_its_wake_is_acknowledged_before_dispatch(
        tmp_path, monkeypatch, reads):
    """The hold buys no model call and holds no slot: with nothing ready the loop's round top (``round_top``) arms
    the task's own warm sleep on its leaves and parks it through the ordinary warm park; the next round top publishes
    the wake, carries it whole as the receipt and acknowledges it synchronously. That round goes back to its top
    once (``"redrain"``), so mail that came after its drain joins the receipt in the resumed request; that top sends
    the receipt and holds nothing, even when the latch's tombstone was not written (it may hold again after it)."""
    from ouroboros import delegate_hold, delegate_supervision as ds

    ctx, limit = _nanny(tmp_path)
    ctx.exact_model_route, ctx.task_metadata["configured_subagent"] = True, {"config_fingerprint": "fp"}
    _run(tmp_path, "leaf")
    ds.write_unknown_hold(ctx, "leaf", {"run_ids": ["leaf"], "run_id": "leaf", "hold_cycles": 1})
    tools, limit.messages, limit.drive_logs, limit.task_id = SimpleNamespace(_ctx=ctx), [], tmp_path, _NANNY
    parks = []

    def park(_ctx, checkpoint):  # the worker's warm park; the leaf ends while the hold sleeps
        parks.append((checkpoint["reason"], dict(ctx._model_sleep)))
        _ended(tmp_path, "leaf")
        return "run:leaf"

    ctx.owner_wait_callback = park

    def top():
        return delegate_hold.round_top(tools, limit, {}, lambda *_a: None, new_input=False)

    assert top() == "park" and reads == []
    [(reason, sleep)] = parks
    assert reason == "sleep" and sleep["mode"] == "warm" and sleep["run_wait"]["hold"] is True
    assert not getattr(ctx, "_model_sleep", None) and not ctx._owner_wait_requested, "the park ended the sleep"
    assert top() == "redrain" and len(parks) == 1
    receipt = limit.messages[-1]["content"]
    assert receipt.startswith("[DELEGATED LEAF WAKE / UNKNOWN-HOLD RESUME]") and '"status": "terminal"' in receipt
    assert _pending(ctx, "leaf")[0] == "" and _pending(ctx, "leaf")[1] in receipt, "acknowledged before dispatch"
    assert not ds.read_unknown_hold(ctx).get("run_id"), "the latch is a tombstone now"
    assert top() == "", "the loop's next top drains, then sends the receipt"

    _run(tmp_path, "leaf-2")
    ds.write_unknown_hold(ctx, "leaf-2", {"run_ids": ["leaf-2"], "run_id": "leaf-2", "hold_cycles": 1})
    _ended(tmp_path, "leaf-2")
    real_write = delegate_hold.write_unknown_hold

    def no_tombstone(ctx_, run_id, fields):
        if not run_id:
            raise OSError("controlled tombstone failure")
        return real_write(ctx_, run_id, fields)

    monkeypatch.setattr(delegate_hold, "write_unknown_hold", no_tombstone)
    assert top() == "redrain" and ds.read_unknown_hold(ctx).get("run_id") == "leaf-2", "drains first, latch kept"
    assert limit.messages[-1]["content"].startswith("[DELEGATED LEAF WAKE") and _pending(ctx, "leaf-2")[0] == ""
    assert top() == "" and len(parks) == 1, "that top sends the receipt instead of holding again"


def test_a_hold_resumed_on_owner_words_after_its_park_sleeps_again_on_its_next_latch(tmp_path, reads):
    """The park marks the hold as parked; owner words at the next top resume it before its wake is read. The mark must
    not outlive that resume: a later latch with nothing ready would read it as a woken park and publish a wake on no
    source (``hold_wake(..., woke=True)``), buying a round on nothing."""
    from ouroboros import delegate_hold, delegate_supervision as ds

    ctx, limit = _nanny(tmp_path)
    ctx.exact_model_route, ctx.task_metadata["configured_subagent"] = True, {"config_fingerprint": "fp"}
    _run(tmp_path, "leaf")
    tools, limit.messages, limit.drive_logs, limit.task_id = SimpleNamespace(_ctx=ctx), [], tmp_path, _NANNY
    parks = []
    ctx.owner_wait_callback = lambda _ctx, checkpoint: parks.append(checkpoint["reason"]) or "run:leaf"

    def top(new_input=False):
        return delegate_hold.round_top(tools, limit, {}, lambda *_a: None, new_input=new_input)

    ds.write_unknown_hold(ctx, "leaf", {"run_ids": ["leaf"], "run_id": "leaf", "hold_cycles": 1})
    assert top() == "park" and ctx._delegate_hold_parked is True
    assert top(new_input=True) == "resume" and ctx._delegate_hold_parked is False, "the owner's words resumed it"
    ds.write_unknown_hold(ctx, "leaf", {"run_ids": ["leaf"], "run_id": "leaf", "hold_cycles": 1})  # a new unknown
    assert top() == "park" and len(parks) == 2, "nothing is ready: it sleeps again"
    assert _pending(ctx, "leaf") == ("", ""), "no wake was published on nothing"


def test_the_frozen_delegate_wait_contract_says_its_result_is_owed_to_the_next_request(tmp_path):
    """Frozen ABI: the five delegate verbs, delegate_wait's arguments and its str publisher are unchanged; the meaning of
    its result changed and the description says so (a signature equality alone would not show that). What each mode
    keeps is said in the tool's own lead, in wait_task's words: the custom-tool wire drops parameter descriptions; the
    parameter says it in the one wording the task waits splice in."""
    import inspect

    from ouroboros.delegate_supervision import delegate_wait_entry
    from ouroboros.openai_chat_custom import project_function_tools_to_openai_custom
    from ouroboros.tools import delegate

    entries = {entry.name: entry for entry in delegate.get_tools()}
    assert set(entries) == {"delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer", "delegate_message"}
    from ouroboros.native_task_wait import SLEEP_MODE_WORDS, SLEEP_MODES, SLEEP_PARAMS

    wait = entries["delegate_wait"]
    properties = wait.schema["parameters"]["properties"]
    assert set(properties) == {"run_id", "runs", "since_seq", "checkpoint_after_sec", "checkpoint_reason", "sleep_mode"}
    assert set(properties["sleep_mode"]["enum"]) == set(SLEEP_MODES), "the task waits' modes, no other"
    held = properties["sleep_mode"]["description"]
    assert SLEEP_MODE_WORDS in held and SLEEP_MODE_WORDS in SLEEP_PARAMS["sleep_mode"]["description"], "one wording"
    assert "with no continuation owner it waits in this call" in held
    assert not any(character.isdigit() for character in held), "what stays held, in words and no numbers"
    assert list(inspect.signature(wait.handler).parameters) == list(inspect.signature(delegate_wait_entry).parameters)
    assert wait.handler.__wrapped__ is delegate_wait_entry, "the one ToolResult->str publisher wraps the entry"
    text = " ".join(wait.schema["description"].split())
    modes = "Explicit in_slot keeps the process and same-call result while yielding acting capacity; warm/cold choose"
    wire = project_function_tools_to_openai_custom([{"type": "function", "function": wait.schema}]).wire_tools()
    lead = wire[0]["custom"]["description"]  # some providers see only this projection: it must say what each mode does
    no_owner = "Without a continuation owner an omitted or in_slot mode waits in-process on the same sources, and warm"
    for said in (modes, "(warm keeps your process, cold releases it)", no_owner):
        assert said in text and said in lead, said
    assert "this wait's result is then owed to your next model request" in text
    assert "Otherwise a sleep returns sleep_armed" in text and "it returns sleep_armed" not in text
    assert "unless named live custody or a conversation turn keeps it warm" in text
    assert "yields acting capacity" in text and "with no further call" in text
    assert "Each wake is delivered once and counts as shown only when shown whole" in text
    assert "a wait_sec an old transcript passes is accepted and sizes nothing" in text
    assert "wait_sec" in inspect.signature(delegate_wait_entry).parameters, "accepted, though the schema offers none"


def test_a_warm_park_acknowledges_what_its_stored_source_holds_as_a_cold_park_does(tmp_path):
    """E3: a deferred acknowledgement (a wake delivered in this batch) is flushed once the warm park's continuation
    source is stored, before the park waits; with no wait requested nothing is parked and nothing is flushed."""
    from ouroboros import owner_wait

    ctx, limit = _nanny(tmp_path)
    flushed = []
    ctx._pending_content_acks = [lambda: flushed.append("ack")]
    owner_wait.wait_after_tools(ctx, [], {}, {}, 1, [], set())
    assert flushed == [], "no wait requested: nothing parks and nothing is flushed"
    seen_at_park = []
    ctx._owner_wait_requested = "quiz-1"
    ctx.owner_wait_callback = lambda _ctx, _checkpoint: seen_at_park.append(list(flushed)) or "answer"
    owner_wait.wait_after_tools(ctx, [{"role": "user", "content": "x"}], {}, {}, 1, [], set())
    assert seen_at_park == [["ack"]], "flushed before the park waits, after its source was stored"


def test_read_coverage_of_a_wake_source_rides_the_continuation(tmp_path, monkeypatch):
    """E3: what of an exact wake source was delivered is carried by the continuation that saves that transcript, so a
    new process completes the source with the rest of it; another task's coverage never rides."""
    from ouroboros import delegate_output, owner_wait

    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})
    ctx, _limit = _nanny(tmp_path)
    key = f"wake:{_NANNY}|path|sha"
    delegate_output._covered_whole(key, 0, 50, 100)
    delegate_output._covered_whole("wake:other|path|sha", 0, 100, 100)
    state = json.loads(json.dumps(owner_wait.continuation_state(ctx, [], {}, {}, 1, [], set())))
    assert state["read_coverage"] == {key: [[0, 50]]}
    monkeypatch.setattr(delegate_output, "_READ_COVERAGE", {})  # a new process
    restored, _ = _nanny(tmp_path)
    owner_wait.restore_continuation_state(SimpleNamespace(_ctx=restored), state, [], {}, {}, set())
    assert not delegate_output._covered_whole(key, 50, 99, 100) and delegate_output._covered_whole(key, 99, 100, 100)
    assert "wake:other|path|sha" not in delegate_output._READ_COVERAGE


def _review_run(root, *, run_id="review-leaf", wave="review-wave", operation="review-operation",
                surface="multi_model_review", task_id=_NANNY, requested=True):
    from ouroboros import delegate_custody as custody

    token = f"invocation-{run_id}"
    common = {"task_id": task_id, "root_task_id": task_id, "invocation_id": token, "route": "route",
              "source": "review_substrate", "surface": surface, "slot_id": "seat", "operation_id": operation,
              "review_wave_id": wave, "review_slot_id": "seat"}
    if requested:
        assert custody.emit(root, custody.START_REQUESTED, {**common, "idempotency_key": token,
            "request": {"prompt": "synthetic review"}})
    assert custody.emit(root, custody.STARTED, {**common, "run_id": run_id, "ledger_root": str(root)})
    return run_id


def _ordinary_wait(ctx, tool, runs):
    from ouroboros.tools import delegate
    from ouroboros.tools.control_task_results import await_messages_entry

    entry = (next(entry for entry in delegate.get_tools() if entry.name == tool)
             if tool == "delegate_wait" else await_messages_entry())
    return entry.handler(ctx, runs=runs)


@pytest.fixture
def no_review_wait_effects(monkeypatch, stub_delegate_engines):
    def forbidden(*_args, **_kwargs):
        pytest.fail("A review wait or exact settled collection tried to observe a leaf or dispatch a reviewer")

    for name in ("ouroboros.tools.delegate._read_run", "ouroboros.delegate_supervision._wait_in_slot",
                 "ouroboros.review_substrate.run_review_request", "ouroboros.review_substrate.ReviewCoordinator.run",
                 "ouroboros.llm.LLMClient.chat", "ouroboros.claudexor_daemon.ensure_owned_gateway",
                 "ouroboros.gateways.claudexor.ClaudexorGateway"):
        monkeypatch.setattr(name, forbidden)


@pytest.mark.parametrize("tool", ["delegate_wait", "await_messages"])
@pytest.mark.parametrize("surface,attempt_tool,collector", [
    ("change", "review_change", "review_change"),
    ("commit_gate", "commit_reviewed", "commit_reviewed"),
])
def test_review_wait_redirect_binds_original_collector(tmp_path, no_review_wait_effects, monkeypatch, tool, surface,
                                                      attempt_tool, collector):
    """Even a mixed selector parks nothing. Its reference resolves in the actual collector, not a mock router."""
    from ouroboros import review_ledger, review_state
    from ouroboros.tools.review_change import _handle_review_change
    from ouroboros.tools.review_collection import resolve_reference

    ctx, _ = _nanny(tmp_path)
    ctx.repo_dir = tmp_path
    review = _review_run(tmp_path)
    _run(tmp_path, "ordinary-leaf")
    row = {"seat_id": "seat", "operation_id": "review-operation", "status": "responded"}
    exact = review_ledger.ReviewLedgerRecord(
        record_id="review-exact", task_id=_NANNY, surface=surface, review_wave_id="review-wave",
        verdict={"aggregate": "PASS"}, rows=[row], brief={"dispute_input": {}})
    review_ledger.write_record(tmp_path, exact)
    # Same owner, wave and slot are not enough: the other operation must not get the pointer.
    review_ledger.write_record(tmp_path, review_ledger.ReviewLedgerRecord(
        record_id="review-neighbor", task_id=_NANNY, surface=surface, review_wave_id="review-wave",
        verdict={"aggregate": "FAIL"}, rows=[{**row, "operation_id": "another-operation"}]))
    attempt = review_state.CommitAttemptRecord(
        ts="2026-01-01T00:00:00+00:00", commit_message="test", status="reviewed", task_id=_NANNY,
        repo_key=review_state.make_repo_key(tmp_path), tool_name=attempt_tool, attempt=3, paid=True,
        review_retry_key="review-wave", review_record_id=exact.record_id,
        triad_raw_results=[{"slot_id": "seat", "operation_id": "review-operation", "status": "ok"}])
    review_state.update_state(tmp_path, lambda state: state.record_attempt(attempt))

    with monkeypatch.context() as reads:
        def forbidden(*_args, **_kwargs):
            pytest.fail("Wait redirect searched historical attempts or archived review segments")

        reads.setattr(review_state, "_load_state_unlocked", forbidden)
        reads.setattr(review_ledger, "_index_segments", forbidden)
        if tool == "delegate_wait":  # the named checkpoint address works even if the hot index no longer lists it
            ctx._left_review_wait = {"task_id": _NANNY, "retry_key": "review-wave", "record_id": exact.record_id}
            reads.setattr(review_ledger, "task_record_page", forbidden)
        response = json.loads(_ordinary_wait(ctx, tool, ["ordinary-leaf", review]))
    assert response["reason"] == "review_panel_owns_run" and response["slept"] is False
    assert [run["run_id"] for run in response["review_runs"]] == [review]
    assert "No selected source was waited on" in response["detail"]
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)
    assert not getattr(ctx, "_owner_wait_requested", None)
    assert not (tmp_path / "state/delegate_supervision" / f"{_NANNY}.json").exists()

    recipe = response["review_runs"][0]["collection"]
    assert recipe["status"] == "collect_existing" and recipe["tool"] == collector
    assert set(recipe["arguments"]) == {"review_reference"}
    _, original, retained = resolve_reference(ctx, recipe["arguments"]["review_reference"],
                                              commit=collector == "commit_reviewed")
    assert original.attempt == 3 and original.paid and retained["record_id"] == exact.record_id
    if collector == "review_change":
        result = json.loads(_handle_review_change(ctx, **recipe["arguments"]))
        assert result["record_id"] == exact.record_id and result["aggregate"] == "PASS"
        assert result["reused"] is True and result["cost"] == {"usd": 0.0, "unknown": False}

    ordinary = json.loads(_ordinary_wait(ctx, tool, ["ordinary-leaf"]))
    assert ordinary["reason"] == "sleep_armed"


def test_reviewer_wait_without_continuation_owner_returns_a_gap_instead_of_waiting(tmp_path, no_review_wait_effects):
    from ouroboros.delegate_supervision import delegate_wait_entry

    ctx, _ = _nanny(tmp_path)
    ctx.owner_wait_callback = None
    legacy = _review_run(tmp_path, requested=False)
    response = _armed(delegate_wait_entry(ctx, run_id=legacy))
    gap = response["review_runs"][0]["collection"]
    assert response["reason"] == "review_panel_owns_run" and response["slept"] is False
    assert gap["status"] == "reference_unavailable" and "review_reference" not in gap
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)


@pytest.mark.parametrize("selector", ["foreign-review", "unknown-run"])
@pytest.mark.parametrize("tool", ["delegate_wait", "await_messages"])
def test_ordinary_wait_unknown_or_foreign_review_does_not_offer_a_reference(tmp_path, no_review_wait_effects, tool,
                                                                          selector):
    ctx, _ = _nanny(tmp_path)
    _review_run(tmp_path, run_id="foreign-review", task_id="other-task")
    response = _ordinary_wait(ctx, tool, [selector])
    if tool == "delegate_wait":
        assert json.loads(response)["reason"] == (
            "run_not_owned" if selector == "foreign-review" else "run_ownership_unknown")
    else:
        assert "not delegated runs of this task" in response
    assert "review_reference" not in response and "review_runs" not in response
    assert not getattr(ctx, "_model_sleep", None) and not getattr(ctx, "_pending_task_wait", None)


def test_review_wait_wrong_wave_never_offers_a_guessed_reference(tmp_path, no_review_wait_effects):
    from ouroboros.review_ledger import ReviewLedgerRecord, write_record

    ctx, _ = _nanny(tmp_path)
    review = _review_run(tmp_path)
    write_record(tmp_path, ReviewLedgerRecord(record_id="not-exact", task_id=_NANNY, surface="change",
        verdict={"aggregate": "FAIL"}, review_wave_id="different-wave",
        rows=[{"seat_id": "seat", "operation_id": "review-operation"}]))
    response = json.loads(_ordinary_wait(ctx, "await_messages", [review]))
    gap = response["review_runs"][0]["collection"]
    assert gap["status"] == "reference_unavailable" and gap["reader"]["tool"] == "review_status"
    assert "not-exact" not in json.dumps(response)


def test_review_wait_skill_offers_exact_free_reader_not_a_skill_review_reissue(tmp_path, no_review_wait_effects):
    from ouroboros.review_ledger import ReviewLedgerRecord, write_record

    ctx, _ = _nanny(tmp_path)
    review = _review_run(tmp_path, surface="skill_review")
    write_record(tmp_path, ReviewLedgerRecord(record_id="skill-exact", task_id=_NANNY, surface="skill",
        verdict={"aggregate": "FAIL"}, review_wave_id="review-wave",
        rows=[{"seat_id": "seat", "operation_id": "review-operation"}]))
    response = json.loads(_ordinary_wait(ctx, "delegate_wait", [review]))
    action = response["review_runs"][0]["collection"]
    assert action["status"] == "read_existing" and action["surface"] == "skill"
    assert action["tool"] == "query_data" and action["arguments"] == {
        "op": "json", "kind": "review_ledger", "ids": ["skill-exact"]}


def test_review_wait_plan_offers_free_plan_read_without_an_ask(tmp_path, no_review_wait_effects):
    from ouroboros import plan_record

    ctx, _ = _nanny(tmp_path)
    review = _review_run(tmp_path, surface="plan_review")
    # Keep ordinary plan-record readers exercised; synthetic seats are already settled so no provider is needed.
    plan_record.append_ask(tmp_path, _NANNY, {"ask_id": "ask-1", "revision": None,
        "retry_key": "review-wave", "participants": [
            {"seat_id": "seat", "status": "failed", "operation_id": "review-operation"}]})
    response = json.loads(_ordinary_wait(ctx, "await_messages", [review]))
    action = response["review_runs"][0]["collection"]
    assert action == {"status": "read_existing", "surface": "intention", "ask_id": "ask-1",
                      "tool": "plan_task", "arguments": {}}


@pytest.mark.parametrize("differs", [True, False, None], ids=["differs", "ordinary", "undisclosed"])
def test_a_terminal_wake_repeats_the_receipt_in_one_line(tmp_path, monkeypatch, differs):
    """c50-4: the finished run's terminal ``delegate_wait`` payload repeats what still differs from the request,
    from that run's custody rows: the applied access when the engine applied another, and my human's later words
    whose last outcome is not ``delivered`` (``accepted`` is not read). In the ordinary case nothing on record
    differs, and the line stops there. The spilled envelope keeps it."""
    from ouroboros import delegate_message_store, delegate_supervision as ds

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    _ended(tmp_path, "run-1")
    effective = {True: "workspace_write", False: "readonly", None: ""}[differs]
    monkeypatch.setattr("ouroboros.tools.delegate._read_run", lambda _ctx, rid, seq, gateway=None: json.dumps({
        "status": "terminal", "run_id": rid, "state": "succeeded", "last_seq": 9,
        "access_evidence": {"requested": "readonly", "effective": effective, "verified": True}}))
    frames = [{"frame": {"owner_words": {"address": "chat 1 / a"}, "ts": "1"}, "outcome": "delivered"},
              {"frame": {"owner_words": {"address": "chat 1 / b"}, "ts": "2"}, "outcome": "accepted" if differs else "delivered"}]
    monkeypatch.setattr(delegate_message_store, "placed_words", lambda _ctx, run_id: frames if run_id == "run-1" else [])
    wake = _armed(ds.run_wait(ctx, "run-1"))
    assert wake["status"] == "terminal"
    if differs:
        assert wake["receipt"] == ("access applied workspace_write, requested readonly [observed]; my human's later "
                                   "words: 1 accepted (taken by the engine, not read) [observed]")
    elif differs is None:  # the engine finished without disclosing the profile: unknown is said, never "as asked"
        assert wake["receipt"] == "access applied: not disclosed by the engine (requested readonly) [observed]"
    else:
        assert "receipt" not in wake, "nothing observed to differ: no line, no assurance"
    spilled = ds._render_wake_payload(ctx, {**wake, "wake_events": [{"type": "x", "text": "y" * 400_000}]})
    assert _armed(spilled).get("receipt") == wake.get("receipt"), "the shortened envelope keeps the line, or none"


@pytest.mark.parametrize("lookup", ["raised", "answered_empty"])
def test_a_failed_custody_lookup_is_no_proof_the_record_is_absent(tmp_path, monkeypatch, lookup):
    """The wake's own read of the run's custody (``delegate_supervision._held_run``): a lookup that raised says the
    record was unavailable at this read; only a lookup that answered and found no row says there is none. The
    wait's admission keeps its ordinary lookup: only the read inside that seam is made to fail."""
    from ouroboros import delegate_custody as custody, delegate_supervision as ds

    ctx, _limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    _ended(tmp_path, "run-1")
    monkeypatch.setattr("ouroboros.tools.delegate._read_run", lambda _ctx, rid, seq, gateway=None: json.dumps({
        "status": "terminal", "run_id": rid, "state": "succeeded", "last_seq": 9,
        "access_evidence": {"requested": "readonly", "effective": "readonly", "verified": True}}))

    def failing(*_a, **_k):
        if lookup == "raised":
            raise OSError("store busy")
        return custody.UNKNOWN, None

    real_seam = ds._held_run

    def seam(ctx_, run_id):  # only this module's name for custody, only for this read: no other reader sees it
        ds.custody = SimpleNamespace(lookup=failing, custody_root=custody.custody_root)
        try:
            return real_seam(ctx_, run_id)
        finally:
            ds.custody = custody

    monkeypatch.setattr(ds, "_held_run", seam)
    wake = _armed(ds.run_wait(ctx, "run-1"))
    if lookup == "raised":
        assert wake["receipt"] == "this run's custody record was unavailable at this read [observed]"
    else:
        assert wake["receipt"] == "no custody record of this run [observed]"


def test_a_runs_applied_access_is_compared_with_its_own_recorded_request(monkeypatch):
    """A run is judged against what IT recorded: its START_REQUESTED body, else its own STARTED access. An older
    record holds neither, and the evidence's ``requested`` is then today's derivation from the live context
    (``delegate._read_run``): nobody asked it of this run, so the line names the applied access and the gap."""
    from types import SimpleNamespace

    from ouroboros import delegate_message_store
    from ouroboros.helper_receipt import run_result_line

    monkeypatch.setattr(delegate_message_store, "placed_words", lambda _ctx, _run: [])
    ctx = SimpleNamespace(drive_root="/nonexistent", task_id="nanny")
    derived = {"requested": "workspace_write", "effective": "readonly"}  # the live derivation beside what ran
    body_only = SimpleNamespace(run_id="run-2", access="", request_event={"request": {"access": "readonly"}})
    assert run_result_line(ctx, body_only, derived) == "", "its own body asked readonly: nothing differs"
    started_only = SimpleNamespace(run_id="run-2", access="readonly", request_event={})
    assert run_result_line(ctx, started_only, derived) == ""
    assert run_result_line(ctx, started_only, {"requested": "readonly", "effective": "workspace_write"}) == (
        "access applied workspace_write, requested readonly [observed]")
    both = SimpleNamespace(run_id="run-2", access="readonly", request_event={"request": {"access": "workspace_write"}})
    assert run_result_line(ctx, both, {"effective": "workspace_write"}) == "", "the body is read before STARTED"
    assert run_result_line(ctx, both, {"effective": "readonly"}) == (
        "access applied readonly, requested workspace_write [observed]")
    old = SimpleNamespace(run_id="run-2", access="", request_event={})
    assert run_result_line(ctx, old, derived) == (
        "access applied readonly; this run's original request is not on record [observed]")
    assert run_result_line(ctx, old, {"requested": "workspace_write"}) == (
        "access applied: not disclosed by the engine (requested not on record) [observed]")


def test_an_old_record_without_its_request_is_not_compared_with_todays_derivation(tmp_path, monkeypatch):
    """The real chain: a STARTED row without access and without a request body, read by today's nanny whose live
    context derives workspace_write; the engine applied readonly. The wake names that and the missing request."""
    from ouroboros import delegate_supervision as ds
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import Engine
    from tests._run_wait_shared import held_run

    ctx, _ = _nanny(tmp_path)
    engine = Engine(tmp_path)
    job = engine.accept("first", {"access": "readonly"}, state="succeeded")
    rid = engine.commands[job]["run"]
    entry = held_run(tmp_path, rid, ctx.task_id, access="")
    assert not entry.access and not entry.request_event
    assert delegate._derive_authority(ctx).access == "workspace_write", "today's derivation differs from what ran"
    gateway = engine.gateway()
    read = delegate._read_run
    monkeypatch.setattr(delegate, "_read_run", lambda ctx_, rid_, seq=None, **_: read(ctx_, rid_, seq, gateway=gateway))
    assert json.loads(delegate._read_run(ctx, rid)).get("status") == "terminal"  # the body collects the terminal
    wake = _armed(ds.run_wait(ctx, rid))
    assert wake.get("status") == "terminal", wake
    assert wake["receipt"] == "access applied readonly; this run's original request is not on record [observed]"
