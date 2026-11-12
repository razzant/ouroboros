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

from tests._budget_pause_exact_helpers import _loop_ctx
from tests._stub_delegate_engines import stub_delegate_engines as stub_delegate_engines  # noqa: F401

_NANNY = "nanny"


def _run(root, run_id, task_id=_NANNY):
    from ouroboros import delegate_custody as custody, delegate_custody_current as current

    assert custody.emit(root, custody.STARTED, {
        "run_id": run_id, "invocation_id": f"invoke-{run_id}", "task_id": task_id, "root_task_id": task_id,
        "route": "route", "source": "delegate_start", "access": "readonly"})
    return current.run(root, run_id)


def _ended(root, run_id, state="succeeded"):
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_collection import observe_run

    assert observe_run(root, current.run(root, run_id), {"summary": {"runId": run_id, "state": state}, "lastSeq": 9})


def _asks(root, run_id, question="q1"):
    from ouroboros import delegate_custody_current as current
    from ouroboros.delegate_collection import observe_run

    assert observe_run(root, current.run(root, run_id), {"summary": {"runId": run_id, "state": "running"}, "lastSeq": 3,
        "pendingInteractions": [{"interactionId": question, "questions": [{"id": "pick", "question": "Which?"}]}]})


def _nanny(root, task_id=_NANNY):
    from ouroboros.task_results import write_task_result

    write_task_result(root, task_id, "running", root_task_id=task_id, chat_id=0)
    ctx, limit = _loop_ctx(root, task_id)
    ctx.task_metadata, ctx._loop_mailbox_seen_ids = {"root_task_id": task_id}, set()
    limit.tool_schemas, limit.round_idx, limit.llm_trace = [], 2, {"tool_calls": []}
    (root / "logs").mkdir(exist_ok=True)
    return ctx, limit


@pytest.fixture
def reads(monkeypatch):
    """The one addressed read a wake makes: a terminal for an ended run, its question set, else a quiet state."""
    from ouroboros import delegate_custody_current as current

    made = []

    def observe(ctx, run_id, since_seq, gateway=None):
        made.append((run_id, since_seq))
        run = current.run(ctx.drive_root, run_id)
        if run is not None and run.terminal_state:
            return json.dumps({"status": "terminal", "run_id": run_id, "state": run.terminal_state, "last_seq": 9})
        if run is not None and run.pending_interaction_ids:
            return json.dumps({"status": "waiting_on_user", "run_id": run_id, "state": "running", "last_seq": 3,
                               "pending_interactions": [{"interaction_id": iid, "questions": [{"id": "pick"}]}
                                                        for iid in run.pending_interaction_ids]})
        return json.dumps({"status": "no_progress", "run_id": run_id, "state": "running", "last_seq": since_seq})

    monkeypatch.setattr("ouroboros.tools.delegate._read_run", observe)
    return made


def _armed(result):
    from ouroboros.delegate_shared import delegate_payload

    return delegate_payload(result)


def _pending(ctx, run_id):
    from ouroboros import delegate_supervision as ds

    state = ds._load_state(ctx, run_id)
    return (state.get("pending_wake") or {}).get("wake_id", ""), (state.get("last_acknowledged_wake") or {}).get(
        "wake_id", "")


def _notices(limit):
    return [row["content"] for row in limit.messages if "[BEGIN_TASK_WAIT_OUTPUT]" in str(row.get("content"))]


def _body(notice):
    return json.loads(notice.split("[BEGIN_TASK_WAIT_OUTPUT]\n", 1)[1].rsplit("\n[END_TASK_WAIT_OUTPUT]", 1)[0])


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


def test_a_wait_on_a_quiet_run_sleeps_and_its_result_is_owed_to_the_next_request(tmp_path, reads):
    """Nothing there yet: the call arms the task's ordinary sleep on its runs (no slot, no tick, no read) and owes its
    result; the run's terminal wakes the sleep, and the first request after it carries one wake, published then and
    acknowledged only after the next saved state, never before."""
    from ouroboros import delegate_supervision as ds, model_sleep
    from ouroboros.native_task_wait import deliver_owed
    from ouroboros.working_checkpoint import flush_content_acks

    ctx, limit = _nanny(tmp_path)
    _run(tmp_path, "run-1")
    armed = _armed(ds.run_wait(ctx, "run-1"))
    assert armed["reason"] == "sleep_armed" and armed["mode"] == "cold", "the body takes the run: nothing keeps it"
    assert ctx._model_sleep["runs"] == ["run-1"] and ctx._model_sleep["any_mail"] and ctx._model_sleep["run_wait"]
    assert ctx._pending_task_wait["tool"] == "delegate_wait" and ctx._pending_task_wait["runs"] == ["run-1"]
    assert reads == [] and _pending(ctx, "run-1") == ("", ""), "arming reads no run and publishes no wake"
    from ouroboros import delegate_custody as custody

    trail = [row for row in map(json.loads, custody.event_log_path(tmp_path).read_text(encoding="utf-8").splitlines())
             if row["type"] == "delegate_supervision_wait_entered"]
    assert [(row["run_id"], row["outcome"], row["mode"]) for row in trail] == [("run-1", "sleep_armed", "cold")]
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
        assert _armed(ds.run_wait(ctx, "run-nobody"))["reason"] == "run_ownership_unknown"
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
    events = ds._addressed_wakes(ctx, {})
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
    the wake, carries it whole as the receipt and acknowledges it synchronously before the resumed round is sent."""
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
    assert top() == "resume" and len(parks) == 1
    receipt = limit.messages[-1]["content"]
    assert receipt.startswith("[DELEGATED LEAF WAKE / UNKNOWN-HOLD RESUME]") and '"status": "terminal"' in receipt
    assert _pending(ctx, "leaf")[0] == "" and _pending(ctx, "leaf")[1] in receipt, "acknowledged before dispatch"
    assert not ds.read_unknown_hold(ctx).get("run_id"), "the latch is a tombstone now"


def test_the_frozen_delegate_wait_contract_says_its_result_is_owed_to_the_next_request(tmp_path):
    """Frozen ABI: the five delegate verbs, delegate_wait's arguments and its str publisher are unchanged; the meaning of
    its result changed and the description says so (a signature equality alone would not show that)."""
    import inspect

    from ouroboros.delegate_supervision import delegate_wait_entry
    from ouroboros.tools import delegate

    entries = {entry.name: entry for entry in delegate.get_tools()}
    assert set(entries) == {"delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer", "delegate_message"}
    wait = entries["delegate_wait"]
    assert set(wait.schema["parameters"]["properties"]) == {
        "run_id", "runs", "since_seq", "checkpoint_after_sec", "checkpoint_reason"}
    assert list(inspect.signature(wait.handler).parameters) == list(inspect.signature(delegate_wait_entry).parameters)
    assert wait.handler.__wrapped__ is delegate_wait_entry, "the one ToolResult->str publisher wraps the entry"
    text = " ".join(wait.schema["description"].split())
    assert "this wait's result is then owed to your next model request" in text
    assert "it returns sleep_armed" in text and "yields acting capacity" in text and "with no further call" in text
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
