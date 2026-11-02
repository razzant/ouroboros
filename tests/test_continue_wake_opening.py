"""#1376 Q2=A: Continue of a consciousness wake, a place focus or a card-less drafted start.

The settled card offered Continue on these interrupted roots (typed technical causes), and
the admission then refused them as ``owner_source_missing``: the opening their producer
recorded was not a source it accepted. Each test drives the real producer (the own-clock
wake stamps, the Place admission and focus task builder, ``start_task``), the real result
writer and the real ``admit_continuation``. The recorded opening is attributed context:
never owner words, an owner ingress, a Presence speaker or a turn lease, and the successor
keeps the work's recorded autonomy and room by value.
"""

from __future__ import annotations

import json
import types
from dataclasses import replace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_consciousness import FLOOR, T0
from tests.test_consciousness import clock as _alarm_clock
from tests.test_owner_continue import NONCE
from tests.test_root_effort_ingress import _pool_ready

clock = _alarm_clock  # the alarm clock fixture whose real launch stamps a wake's producer number

WORDS = "hello, can you look at the failing build?"


@pytest.fixture(autouse=True)
def _isolated_projects_root(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path_factory.mktemp("projects_root")))


def _focus_root(tmp_path, monkeypatch, *, owner=False, reason_code="provider_unavailable"):
    """A place focus built by the real Place admission and ``_build_task``, interrupted."""
    from ouroboros import places
    from ouroboros.presence_admission import admit_presence_turn
    from ouroboros.presence_runner import PresenceTurnEvent, _build_task
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    place = places.connect_place(tmp_path, connector="bridge", provider="chat", account_id="a",
                                 conversation_id="r", wakes=["mention"])
    if owner:
        places.edit_place(tmp_path, place.place_id, "confirm_owner_account", author="owner",
                          platform_actor_id="person", account_id="a")
    monkeypatch.setattr("ouroboros.presence_admission.focus_enabled", lambda _root: True)
    monkeypatch.setattr("ouroboros.presence_admission.focus_allowance",
                        lambda _root: {"status": "available", "remaining_usd": 2, "settled_usd": 1})
    monkeypatch.setattr("ouroboros.presence_context._own_work_section", lambda *_args: "")
    actor = {"platform_actor_id": "person", "display_name": "Person", **({"authenticated": True} if owner else {})}
    event = PresenceTurnEvent("event1", "chat", "a", "r", "", "chat:a:r", actor, {}, {}, WORDS, facts=("mention",))
    admission = admit_presence_turn(drive_root=tmp_path, authenticated_transport_skill="bridge", provider="chat",
                                    account_id="a", conversation_id="r")
    task = _build_task(replace(admission, allowance={"status": "available", "remaining_usd": 2}), event,
                       drive_root=tmp_path, staged_files=())
    # The presence runner's start record, then the turn's own terminal (the same writer).
    write_task_result(tmp_path, task["id"], STATUS_RUNNING, create_only=True, strict_existing_dict=True,
                      metadata=task["metadata"], chat_id=task["chat_id"], _is_direct_chat=True,
                      source=task["source"], result="Task is running.")
    write_task_result(tmp_path, task["id"], "failed", reason_code=reason_code, metadata=task["metadata"],
                      result="The model provider was unavailable.")
    return workers, task


def _wake_root(tmp_path, monkeypatch, *, reason_code="provider_unavailable", metadata=None):
    """An own-clock wake (its producer's stamps, ``metadata`` when its alarm launched it) whose own turn was
    interrupted."""
    from ouroboros.consciousness_wake import wake_task_metadata
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    _queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _pool_ready(monkeypatch, workers)
    metadata = metadata or wake_task_metadata("act", "heartbeat")
    write_task_result(tmp_path, "wake-1", STATUS_RUNNING, chat_id=1, metadata=metadata, _is_direct_chat=True,
                      result="Task is running.")
    write_task_result(tmp_path, "wake-1", "failed", reason_code=reason_code, metadata=metadata,
                      result="The model provider was unavailable.")
    return workers, metadata


def _successor(workers, ack):
    assert ack["ok"] is True, ack
    return next(task for task in workers.PENDING if task["id"] == ack["successor_task_id"])


def _no_owner_words(row):
    from ouroboros.dialogue_provenance import run_origin
    from ouroboros.loop_messages import _initialize_owner_directives

    metadata = row["metadata"]
    assert metadata["owner_corpus"] == []
    ctx = types.SimpleNamespace(task_metadata=metadata, _owner_directives=[])
    _initialize_owner_directives(ctx, [{"role": "user", "content": row["text"]}])
    assert ctx._owner_directives == [], "the recorded opening never becomes an owner directive"
    assert "Owner's original instruction" not in row["text"]
    assert "origin_message_ref" not in row and "origin_message_ref" not in metadata
    assert run_origin(row)["owner_ingress"] is False


def test_an_interrupted_place_focus_continues_from_its_recorded_event(tmp_path, monkeypatch):
    from ouroboros.consciousness_authority import disabled_tools_for
    from ouroboros.dialogue_provenance import presence_record_binding
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.place_reply import reply_carrier
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation

    workers, focus = _focus_root(tmp_path, monkeypatch)
    recorded = load_task_result(tmp_path, focus["id"])
    assert continuation_offer(recorded, focus["id"])["eligible"] is True

    row = _successor(workers, admit_continuation(focus["id"], action_nonce=NONCE))

    metadata = row["metadata"]
    wake = metadata["continuation"]["owner_sources"]["wake"]
    assert wake == {"source": "consciousness_wake", "task_id": focus["id"], "wake_cause": "focus",
                    "focus": focus["metadata"]["focus"], "observed_text": WORDS}
    assert wake["focus"]["actor"]["owner_confirmed"] is False
    # Consciousness work in the same room, at the level it was recorded under.
    assert row["chat_id"] == focus["chat_id"]
    assert metadata["work_kind"] == metadata["initiator"] == "consciousness"
    assert metadata["consciousness_autonomy"] == focus["metadata"]["consciousness_autonomy"]
    assert metadata["disabled_tools"] == disabled_tools_for(focus["metadata"]["consciousness_autonomy"])
    assert metadata["focus"] == focus["metadata"]["focus"] and "wake_cause" not in metadata
    assert metadata["origin_wake_task_id"] == focus["id"]
    # The place's related work, never its speaker: no turn, no reply carrier, no lease.
    assert metadata["presence_binding_authority"] == {"binding_id": focus["metadata"]["presence"]["binding_id"]}
    assert presence_record_binding(row) == focus["metadata"]["presence"]["binding_id"]
    assert "presence" not in metadata and "_presence_turn" not in row and not row.get("_is_direct_chat")
    assert reply_carrier(metadata) == {}
    _no_owner_words(row)
    text = row["text"]
    assert WORDS in text and text.index("The event's words") < text.index(WORDS)
    assert "not an owner instruction" in text


def test_an_owner_confirmed_sender_stays_a_recorded_fact(tmp_path, monkeypatch):
    """The place recorded the sender as the owner's confirmed account: the opening keeps that
    fact by value; the work stays consciousness work and no owner ingress is minted."""
    from supervisor.continuation_admission import admit_continuation

    workers, focus = _focus_root(tmp_path, monkeypatch, owner=True)
    assert focus["metadata"]["focus"]["actor"]["owner_confirmed"] is True

    row = _successor(workers, admit_continuation(focus["id"], action_nonce=NONCE))

    wake = row["metadata"]["continuation"]["owner_sources"]["wake"]
    assert wake["focus"]["actor"]["owner_confirmed"] is True
    assert row["metadata"]["work_kind"] == "consciousness"
    assert '"owner_confirmed": true' in row["text"]
    _no_owner_words(row)


def test_an_own_clock_wake_continues_from_its_recorded_reason(tmp_path, monkeypatch):
    from supervisor.continuation_admission import admit_continuation

    workers, metadata = _wake_root(tmp_path, monkeypatch)

    row = _successor(workers, admit_continuation("wake-1", action_nonce=NONCE))

    assert row["metadata"]["continuation"]["owner_sources"]["wake"] == {
        "source": "consciousness_wake", "task_id": "wake-1", "wake_cause": "own_clock", "wake_reason": "heartbeat"}
    assert row["metadata"]["work_kind"] == "consciousness" and row["metadata"]["consciousness_autonomy"] == "act"
    assert row["metadata"]["continuation"]["predecessor_task_id"] == "wake-1"
    assert row["metadata"]["origin_wake_task_id"] == "wake-1"
    assert "presence_binding_authority" not in row["metadata"] and "presence" not in row["metadata"]
    assert '"wake_reason": "heartbeat"' in row["text"] and "The event's words" not in row["text"]
    _no_owner_words(row)


def test_the_recorded_level_rides_by_value_under_the_existing_contract(tmp_path, monkeypatch):
    """No new permission policy: the successor's authority is the existing
    ``apply_consciousness_authority`` over the level its work recorded, not a fresh read and
    not a new min(recorded, current) rule (the owner's later setting governs new wakes)."""
    from ouroboros.consciousness_authority import ACT_DISABLED
    from supervisor.continuation_admission import admit_continuation

    workers, _metadata = _wake_root(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "observe")

    row = _successor(workers, admit_continuation("wake-1", action_nonce=NONCE))

    assert row["metadata"]["consciousness_autonomy"] == "act"
    assert set(row["metadata"]["disabled_tools"]) == set(ACT_DISABLED) - {"tell_task"}
    assert row["metadata"]["runtime_mode_cap"] == "light"


def test_a_chained_continue_keeps_the_wake_opening(tmp_path, monkeypatch):
    """The successor is not a wake (it never inherits ``wake_cause``); its own Continue reads
    the opening its admission chained, exactly as a start basis is chained."""
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_restart_retention import _pool_events

    workers, focus = _focus_root(tmp_path, monkeypatch)
    first = _successor(workers, admit_continuation(focus["id"], action_nonce=NONCE))
    _pool_events(workers, monkeypatch)
    monkeypatch.setattr(workers, "WORKERS", {})
    # The successor never reaches a worker: the ordinary shutdown writer settles its row.
    assert workers.kill_workers(terminal_status="cancelled", result_reason="server shutdown",
                                stop_source="server_shutdown")
    assert load_task_result(tmp_path, first["id"])["cancel_origin"]["source"] == "server_shutdown"

    second = _successor(workers, admit_continuation(first["id"], action_nonce=NONCE))

    assert (second["metadata"]["continuation"]["owner_sources"]["wake"]
            == first["metadata"]["continuation"]["owner_sources"]["wake"])
    assert second["metadata"]["presence_binding_authority"] == first["metadata"]["presence_binding_authority"]
    assert second["metadata"]["origin_wake_task_id"] == first["metadata"]["origin_wake_task_id"] == focus["id"]
    assert WORDS in second["text"]
    _no_owner_words(second)


@pytest.mark.parametrize("damage,gap", [
    (lambda meta: meta["presence"].pop("observed_text"), "wake_opening_unreadable"),
    (lambda meta: meta["focus"].pop("place_id"), "wake_opening_unreadable"),
    (lambda meta: meta.update(initiator="owner_turn"), "wake_opening_unreadable"),
    (lambda meta: meta.update(work_kind="owner"), "wake_opening_unreadable"),
    (lambda meta: meta.pop("wake_cause"), "original_owner_message_text"),
], ids=["no_event_words", "no_place", "not_consciousness", "owner_kind", "not_a_wake"])
def test_an_unreadable_or_foreign_wake_record_is_a_named_gap(tmp_path, monkeypatch, damage, gap):
    from ouroboros.task_results import load_task_result
    from tests.test_continue_quiz_origin import _edit_result, _refused

    workers, focus = _focus_root(tmp_path, monkeypatch)
    _edit_result(tmp_path, focus["id"], lambda row: damage(row["metadata"]))
    _refused(tmp_path, workers, focus["id"], gap)
    assert "continued_by" not in load_task_result(tmp_path, focus["id"])


@pytest.mark.parametrize("terminal,refusal", [
    ({"status": "cancelled", "cancel_origin": {"source": "http_single", "reason": "Stop now"}}, "stopped_by_owner"),
    ({"status": "cancelled", "cancel_origin": {"source": "panic"}}, "stopped_by_owner"),
    ({"reason_code": "budget_exhausted"}, "hard_limit_reached"),
    ({"deadline_at": "2000-01-01T00:00:00+00:00"}, "hard_limit_reached"),
    ({"status": "completed", "reason_code": ""}, "author_finished"),
], ids=["stop", "panic", "money", "deadline", "answered"])
def test_a_wake_opening_never_outranks_a_stop_or_a_whole_work_bound(tmp_path, monkeypatch, terminal, refusal):
    from ouroboros.task_results import load_task_result
    from supervisor.continuation_admission import admit_continuation
    from tests.test_continue_quiz_origin import _edit_result

    workers, focus = _focus_root(tmp_path, monkeypatch)
    _edit_result(tmp_path, focus["id"], lambda row: row.update(terminal))

    outcome = admit_continuation(focus["id"], action_nonce=NONCE)

    assert outcome["ok"] is False and outcome["error"] == refusal, outcome
    assert workers.PENDING == [] and "continued_by" not in load_task_result(tmp_path, focus["id"])


def test_a_drafted_start_without_an_answered_card_continues_on_its_drafted_objective(tmp_path, monkeypatch):
    """Rewritten from ``test_a_wake_started_root_without_an_answered_card_is_not_continued``
    (#1376 Q2=A): the producer's ``origin_message_absent`` with no card opens on the drafted
    objective alone, labelled as the drafting task's words; no owner text is invented."""
    from supervisor.continuation_admission import admit_continuation
    from tests.test_continue_quiz_origin import OBJECTIVE, WAKE, _interrupted_wake_root

    workers, root = _interrupted_wake_root(tmp_path, monkeypatch, answers=False)
    assert root["metadata"]["origin_message_absent"] is True and root["metadata"]["owner_corpus"] == []

    row = _successor(workers, admit_continuation(root["id"], action_nonce=NONCE))

    start = row["metadata"]["continuation"]["owner_sources"]["start"]
    assert start["answers"] == [] and start["author"] == {"kind": "task", "task_id": WAKE}
    assert start["objective"].startswith(OBJECTIVE)
    assert f"task {WAKE} drafted its objective" in row["text"] and OBJECTIVE in row["text"]
    assert "answered cards" not in row["text"]
    _no_owner_words(row)
    assert json.dumps(row["metadata"]["owner_corpus"]) == "[]"


def test_a_technical_continue_takes_no_consciousness_slot_and_keeps_the_money_gate(tmp_path, monkeypatch):
    """An owner Continue is a named technical continuation (``enqueue_task(continuation=True)``), never a
    spontaneous start: with every consciousness slot taken it is admitted and takes none, while a fresh root
    is still refused, and an allowance at zero or unknown still refuses the Continue (its claim kept for
    the same nonce). A replay and the saved queue keep the same successor and its marker."""
    from ouroboros.consciousness_allowance import STATUS_EXHAUSTED, STATUS_UNKNOWN
    from supervisor import queue
    from supervisor.continuation_admission import admit_continuation

    workers, _metadata = _wake_root(tmp_path, monkeypatch)
    available = {"status": "available", "limit_usd": 20.0, "settled_usd": 4.0, "accounted_usd": 4.0,
                 "remaining_usd": 16.0, "unknown_unmetered": 0, "resets_at": ""}
    monkeypatch.setattr("ouroboros.config.get_consciousness_max_tasks", lambda: 1)

    def window(fields):
        monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", lambda *_a, **_k: dict(fields))

    def started(task_id):
        return {"id": task_id, "type": "task", "text": task_id, "chat_id": 1, "root_task_id": task_id,
                "delegation_role": "root", "metadata": {"initiator": "consciousness",
                                                        "usage_category": "consciousness_task",
                                                        "consciousness_autonomy": "act"}}

    window(available)
    assert "_admission_blocked" not in queue.enqueue_task(started("started-1"))
    assert queue.live_consciousness_root_count() == 1  # the only slot is taken
    for refusal, code in (({"status": STATUS_EXHAUSTED, "limit_usd": 0.0}, "consciousness_allowance_exhausted"),
                          ({"status": STATUS_UNKNOWN, "error": "unreadable"}, "consciousness_allowance_unknown")):
        window(refusal)
        assert admit_continuation("wake-1", action_nonce=NONCE)["error"] == code
    window(available)

    ack = admit_continuation("wake-1", action_nonce=NONCE)

    row = _successor(workers, ack)
    assert row["_consciousness_continuation"] is True
    assert "root_cost_ceiling_usd" not in row["metadata"], "no producer number recorded: none is invented"
    assert queue.live_consciousness_root_count() == 1, "the successor takes no slot"
    assert queue.enqueue_task(started("started-2"))["_admission_blocked"] == "consciousness_task_limit"
    again = admit_continuation("wake-1", action_nonce=NONCE)
    assert again["ok"] is True and again["successor_task_id"] == ack["successor_task_id"]
    assert [task["id"] for task in workers.PENDING].count(ack["successor_task_id"]) == 1
    assert queue.persist_queue_snapshot(reason="test_continue_slot") is True
    saved = json.loads(queue.QUEUE_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert [entry["task"]["_consciousness_continuation"] for entry in saved["pending"]
            if entry["id"] == ack["successor_task_id"]] == [True]


AVAILABLE_TO_THE_DOOR = {"status": "available", "limit_usd": 20.0, "settled_usd": 4.0, "accounted_usd": 4.0,
                         "remaining_usd": 16.0, "unknown_unmetered": 0, "resets_at": ""}


def _scope(root, task):
    """The usage scope ``task``'s own start binds (``agent.py``: the whole-work billing, ``task_usage_scope``)."""
    from ouroboros.usage_admission import task_billing_fields
    from ouroboros.usage_origin import task_usage_scope

    root_id = str(task.get("root_task_id") or task["id"])
    return task_usage_scope(task, drive_root=root, root_task_id=root_id,
                            parent_task_id=str(task.get("parent_task_id") or ""),
                            billing=task_billing_fields(task, root_id, 50.0, root))


def _stop(root, task):
    """The stop ``task``'s own start resolves in its own scope: the scope, its ceiling, the known spend the loop
    compares with it (``_loop_tree_accounting``) and who authored it."""
    from ouroboros import task_pacing, usage_accounting as ua
    from ouroboros.loop_budget import _loop_tree_accounting

    scope = _scope(root, task)
    ctx = types.SimpleNamespace()
    with ua.usage_scope(scope):
        ceiling = task_pacing.resolve_task_cost_ceiling(ctx, 1000.0)
        tree = _loop_tree_accounting(refresh=True, max_age_sec=0.0, strict=True)
    return scope, ceiling, (tree or {}).get("settled_usd"), task_pacing.cost_stop_authority(ctx)


def _gate(root, task, own_cost, monkeypatch):
    """The loop's own budget decision after a spending round (``loop._check_budget_limits``), in ``task``'s scope with
    the stop its start resolves: ``(result, usage, sends, ledger unchanged)``. A model send it would make is counted;
    ``None`` lets the next round's paid dispatch go ahead."""
    from ouroboros import loop, task_pacing, usage_accounting as ua
    from tests._budget_limits_helpers import _make_args
    from tests._usage_store_testing import ledger_rows

    sends = []
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: sends.append("paid send") or ({}, 0.0))
    tool_ctx = types.SimpleNamespace(task_id=task["id"], drive_root=root, budget_drive_root=root,
                                     owner_wait_callback=None, task_metadata=task.get("metadata") or {},
                                     is_direct_chat=False, task_contract=task.get("task_contract"))
    before = ledger_rows(root)
    with ua.usage_scope(_scope(root, task)):
        ceiling = task_pacing.resolve_task_cost_ceiling(tool_ctx, 1000.0)
        args = _make_args(task_id=task["id"], accumulated_usage={"cost": own_cost}, cost_ceiling=ceiling,
                          budget_remaining_usd=1000.0, drive_logs=root / "logs")
        args["ctx"].tools = types.SimpleNamespace(_ctx=tool_ctx)
        result = loop._check_budget_limits(**args)
    return result, args["ctx"].accumulated_usage, sends, ledger_rows(root) == before


def test_a_continued_wake_keeps_its_producer_total_for_the_whole_work(clock, tmp_path, monkeypatch):
    """The alarm's real launch records what is left of the day as the wake's producer TOTAL. Continue hands
    that number on unchanged: the successor's stop compares the whole work's group spend, so the predecessor's
    spend counts once and the successor has only the rest. The loop's own budget gate lets the next paid round
    go ahead below the total and stops before one at it. A replay and a second Continue keep it (never a fresh
    grant), and a child the real scheduler starts reads it as its root's producer number."""
    import queue as queues

    from ouroboros import consciousness, task_pacing
    from ouroboros.agent import OuroborosAgent
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools import control
    from ouroboros.tools.registry import ToolContext
    from supervisor.continuation_admission import admit_continuation
    from tests.test_available_subagents_runtime import _api_row, _settings
    from tests.test_billing_group import _spend
    from tests.test_original_tree_cost_ceiling import _schedule

    monkeypatch.setenv("TOTAL_BUDGET", "1000")
    monkeypatch.setattr("ouroboros.config.get_consciousness_max_tasks", lambda: 5)
    day = dict(AVAILABLE_TO_THE_DOOR, settled_usd=16.0, accounted_usd=16.0, remaining_usd=4.0)
    monkeypatch.setattr(consciousness, "allowance_window", lambda *_a, **_k: dict(day))
    assert clock.clock.tick(T0 + FLOOR + 1) == "launched"
    metadata = clock.launches[0]["metadata"]
    total = metadata["root_cost_ceiling_usd"]
    assert total == 4.0  # min(the per-task cap $50, the $4 left of the day)
    workers, _metadata = _wake_root(tmp_path, monkeypatch, metadata=metadata)
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window",
                        lambda *_a, **_k: dict(AVAILABLE_TO_THE_DOOR))
    wake_scope, wake_stop, spent, _authority = _stop(tmp_path, {"id": "wake-1", "metadata": metadata})
    assert (wake_stop.state, wake_stop.ceiling_usd, wake_stop.basis) == ("active", total, "producer_allowance")
    assert spent == 0
    _spend(tmp_path, wake_scope, 2.5)  # what the wake spent before its provider failed

    ack = admit_continuation("wake-1", action_nonce=NONCE)

    row = _successor(workers, ack)
    assert row["metadata"]["root_cost_ceiling_usd"] == total
    scope, stop, spent, authority = _stop(tmp_path, row)
    assert scope.billing_group_id == "wake-1" and authority == task_pacing.COST_STOP_PRODUCER
    assert (stop.state, stop.ceiling_usd, stop.basis) == ("active", total, "producer_allowance")
    assert spent == pytest.approx(2.5), "the predecessor's spend is counted once: $1.50 of the total is left"
    _spend(tmp_path, scope, 1.0)  # the whole work: $3.50 of its $4.00
    result, _usage, sends, unchanged = _gate(tmp_path, row, 1.0, monkeypatch)
    assert result is None and sends == [] and unchanged, "below the total the next paid round goes ahead"
    _spend(tmp_path, scope, 0.5)  # the whole work reaches its total; the successor's own calls are $1.50 of it
    result, usage, sends, unchanged = _gate(tmp_path, row, 1.5, monkeypatch)
    assert result is not None and sends == [] and unchanged, "stopped before another paid dispatch"
    assert usage["cost_stop_spend_basis"] == task_pacing.SPEND_BASIS_TREE and usage["reason_code"] == "budget_exhausted"
    assert "Task tree spent $4.000" in result[0] and "own calls $1.500" in result[0] and "$4.00" in result[0]
    replay = admit_continuation("wake-1", action_nonce=NONCE)
    assert replay["ok"] is True and replay["successor_task_id"] == row["id"]
    assert load_task_result(tmp_path, row["id"])["metadata"]["root_cost_ceiling_usd"] == total
    # The successor's own start record, then the real scheduler starts a child from its resolved stop.
    OuroborosAgent._persist_running_record(types.SimpleNamespace(env=types.SimpleNamespace(drive_root=tmp_path)),
                                           dict(row))
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    successor_ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    successor_ctx.task_id, successor_ctx.event_queue, successor_ctx._cost_ceiling = row["id"], queues.Queue(), stop
    successor_ctx.task_metadata = {**row["metadata"], "root_task_id": row["id"]}
    successor_ctx.task_contract = row["task_contract"]
    member = _schedule(successor_ctx)  # its stored row, event and payload carry one number
    assert member["root_cost_ceiling_usd"] == total and member["root_task_id"] == row["id"]
    _scope, member_stop, member_spent, member_authority = _stop(tmp_path, member)
    assert member_authority == task_pacing.COST_STOP_PRODUCER and member_spent == pytest.approx(4.0)
    assert (member_stop.state, member_stop.ceiling_usd, member_stop.basis) == ("active", total, "root_resolved_ceiling")
    # Its provider fails too: the next Continue keeps the same total over the same group, never a fresh one.
    write_task_result(tmp_path, row["id"], "failed", reason_code="provider_unavailable", metadata=row["metadata"],
                      result="The model provider was unavailable.")
    workers.PENDING[:] = [task for task in workers.PENDING if task["id"] != row["id"]]  # it left the queue to run

    second = _successor(workers, admit_continuation(row["id"], action_nonce=NONCE))

    assert second["metadata"]["root_cost_ceiling_usd"] == total
    _scope, second_stop, spent, _authority = _stop(tmp_path, second)
    assert (second_stop.state, second_stop.ceiling_usd) == ("active", total) and spent >= total


def test_a_continued_place_focus_keeps_its_producer_total(tmp_path, monkeypatch):
    """A Place focus's real admission records what is left of the allowance (at most the per-task cap) as its
    producer number; its Continue resolves the same active stop over the focus's group."""
    from ouroboros import task_pacing
    from supervisor.continuation_admission import admit_continuation

    workers, focus = _focus_root(tmp_path, monkeypatch)
    total = focus["metadata"]["root_cost_ceiling_usd"]
    assert 0 < total <= 2

    row = _successor(workers, admit_continuation(focus["id"], action_nonce=NONCE))

    assert row["metadata"]["root_cost_ceiling_usd"] == total
    scope, stop, spent, authority = _stop(tmp_path, row)
    assert scope.billing_group_id == focus["id"] and authority == task_pacing.COST_STOP_PRODUCER
    assert (stop.state, stop.ceiling_usd, stop.basis, spent) == ("active", total, "producer_allowance", 0)
