"""What holds my own held wake (ouroboros/consciousness_hold.py): the cause its records prove, the room a Resume would
add, and the host's words for it as the consciousness state line says them."""

import datetime as dt

import pytest

from ouroboros import consciousness_hold as hold, consciousness_view as view
from tests._store_shared import seed_result


def _pause(rail="graceful_ceiling", *, state="paused", reason="budget", scope="root", ceiling=2.99,
           basis="producer_allowance", cap=50.0):
    row = {"pause_id": "p", "state": state, "reason": reason, "rail": rail, "scope": scope, "task_attempt": 1}
    if basis is not None:
        row["cost_ceiling"] = {"state": "active", "ceiling_usd": ceiling, "root_cap_usd": cap,
                               "planning_margin_usd": None, "basis": basis}
    return row


def _launched(source="allowance", ceiling=2.99, **extra):
    from ouroboros.consciousness_wake import wake_task_metadata
    return {**wake_task_metadata("act", "heartbeat", root_cost_ceiling_usd=ceiling, root_cost_ceiling_source=source),
            **extra}


CHILD_AT_CEILING = _pause(basis="root_resolved_ceiling")


@pytest.mark.parametrize("metadata,own,members,cause", [
    # The launch stamp names the limit that set the ceiling the wake paused at.
    (_launched("allowance"), _pause(), (), "allowance"),
    (_launched("both"), _pause(), (), "allowance_and_cap"),
    (_launched("per_task_cap"), _pause(), (), "per_task_cap"),
    # A per-task cap of 0 left the allowance alone to set it (stamped "allowance"); a saved null cap changes nothing.
    (_launched("allowance"), _pause(cap=None), (), "allowance"),
    # The cap was raised and Resume pressed: the re-resolved cap is above the ceiling, the stamp still says the cap.
    (_launched("per_task_cap", ceiling=10.0), _pause(ceiling=10.0, cap=80.0), (), "per_task_cap"),
    # A record from before the stamp proves the launch ceiling only, never which limit.
    (_launched(""), _pause(), (), "launch_ceiling"),
    (_launched(""), _pause(basis=None), (), "launch_ceiling"),
    # Child first: the tree's latch refused the wake while a member paused at the inherited number.
    (_launched("allowance"), _pause("dispatch_refused", basis=None), (CHILD_AT_CEILING,), "allowance"),
    # A member at a number of its own (an explicit profile's) proves nothing about the wake's launch ceiling.
    (_launched("allowance"), _pause("dispatch_refused", basis=None),
     (_pause(basis="min(global_pct, root_cap_minus_margin, non_root_member, root_resolved_ceiling)"),), "per_task_fence"),
    (_launched("allowance"), _pause("dispatch_refused", basis=None), (_pause(basis="root_resolved_ceiling", ceiling=1.0),),
     "per_task_fence"),
    # The cap was lowered after launch, before the task started: its fence refused first.
    (_launched("allowance"), _pause("dispatch_refused", basis=None), (), "per_task_fence"),
    (_launched("allowance"), _pause("dispatch_refused", scope="global", basis=None), (), "total_budget"),
    (_launched("allowance"), _pause("global_exhausted", scope="global", basis=None), (), "total_budget"),
    # Pauses that are not money keep their own reason.
    (_launched("allowance"), _pause("owner_pause", reason="owner", basis=None), (), "owner"),
    (_launched("allowance"), _pause("model_sleep", reason="sleep", scope="task", basis=None), (CHILD_AT_CEILING,), "sleep"),
    # A technical pause goes on by itself only with a typed cause (budget_pause: cause + auto_resume); an older,
    # untyped one waits for a Resume.
    (_launched("allowance"), {**_pause("billing_authority_unreadable", reason="authority", basis=None),
                              "cause": "read_failed", "auto_resume": True}, (), "technical"),
    (_launched("allowance"), _pause("billing_authority_unreadable", reason="authority", basis=None), (), "technical_resume"),
    # No live pause holds it (a Resume granted, a resumed record, no row at all): the pending wake is only queued.
    (_launched("allowance"), _pause(state="resume_granted"), (), "queued"),
    (_launched("allowance"), _pause(state="resumed"), (), "queued"),
    (_launched("allowance"), {}, (), "queued"),
    # A saved ceiling that is not the launch number, or an allowance the owner's Resume lifted, proves nothing.
    (_launched("allowance"), _pause(ceiling=5.0), (), ""),
    ({**{k: v for k, v in _launched("allowance").items() if k != "root_cost_ceiling_usd"},
      "producer_allowance_lifted": {"ceiling_usd": 2.99}}, _pause(), (), ""),
])
def test_a_held_wake_is_said_held_by_its_allowance_only_where_its_records_prove_it(metadata, own, members, cause):
    held = hold.hold_cause(metadata, own, members)
    assert held["cause"] == cause
    stamped = cause in {"allowance", "allowance_and_cap", "per_task_cap", "launch_ceiling"}
    assert held["ceiling_usd"] == (metadata["root_cost_ceiling_usd"] if stamped else None)


FREES = "Stop now ends it and frees the clock for its next check."
HELD_NOTE = f"No spending estimate is available for Resume. Raising the limit alone resumes nothing. {FREES}"
NEXT_WAITS = "The next own wake waits until it ends."


@pytest.mark.parametrize("cause,ceiling,said,note", [
    ("allowance", 2.99, "Its own wake-up stopped at its allowance ($2.99).", HELD_NOTE),
    # A positive sub-cent ceiling stays an amount.
    ("allowance", 0.004, "Its own wake-up stopped at its allowance ($0.0040).", HELD_NOTE),
    ("allowance_and_cap", 50.0, "Its own wake-up stopped at its allowance ($50.00, the per-task cap too).", HELD_NOTE),
    # A host that estimates Resume's room (live_wake_hold.resume_additional_usd) keeps that raising the limit alone
    # resumes nothing; the estimate itself is the client's line (wakeResumeRoomWords), as Costs and Main say it.
    ("allowance+estimate", 2.99, "Its own wake-up stopped at its allowance ($2.99).",
     "Raising the limit alone resumes nothing; Resume checks what is left now and, if there is room, continues this "
     f"wake without resetting its spend. {FREES}"),
    ("launch_ceiling", 2.99, "Its own wake-up is paused at the spending limit it started with ($2.99).",
     f"Raising a limit alone resumes nothing; Resume checks what is left. {FREES}"),
    ("per_task_cap", 10.0, "Its own wake-up is paused at the per-task cost cap it started with ($10.00).",
     f"Changing the setting in Costs does not raise it: raise this work's cap in chat, then Resume. {FREES}"),
    # A stamp from before the number was kept names the limit alone.
    ("per_task_cap", None, "Its own wake-up is paused at the per-task cost cap it started with.",
     f"Changing the setting in Costs does not raise it: raise this work's cap in chat, then Resume. {FREES}"),
    # The total budget is said for what it is, as a past stop (its pause keeps the rail after a raise), and raising it
    # is not a Resume (owner 27.11: the block's words); whether it has room now is the client's line.
    ("total_budget", None, "Its own wake-up stopped when the total budget was spent.",
     f"Raising Total budget alone resumes nothing. {FREES}"),
    ("per_task_fence", None, "Its own wake-up is paused at a spending limit of its own work; its card shows which.",
     f"Resume once that limit has room. {FREES}"),
    ("owner", None, "Its own wake-up is paused; its card shows who paused it.",
     "The next own wake waits until it is resumed or stopped."),
    ("sleep", None, "Its own wake-up sleeps until what it waits for happens, then goes on by itself; its card shows what.",
     NEXT_WAITS),
    ("technical", None, "Its own wake-up is paused: a record it needs could not be read; its card shows which.",
     f"It goes on by itself once that record can be read. {NEXT_WAITS}"),
    ("technical_resume", None, "Its own wake-up is paused: a record it needs could not be read; its card shows which.",
     f"Resume continues it once that record can be read. {FREES}"),
    ("restart", None, "Its own wake-up is held since Ouroboros stopped or restarted.", f"Resume continues it. {FREES}"),
    ("panic", None, "Its own wake-up was stopped by Panic.", f"Resume continues it. {FREES}"),
    ("queued", None, "Its own wake-up waits in the queue for a worker and starts by itself.", None),
    # Only a cause nothing proves keeps the either-or.
    ("", None, "Its own wake-up is paused or waiting in the queue; its card shows why.", NEXT_WAITS),
    ("a cause of a newer host", None, "Its own wake-up is paused or waiting in the queue; its card shows why.", NEXT_WAITS),
])
def test_the_held_wake_sentence_names_every_proven_cause_and_its_next_step(cause, ceiling, said, note):
    cause, _, estimated = cause.partition("+")
    snapshot = {"live_wake_task_id": "w1", "live_wake_held": True, "next_wake_at": "2026-10-24T09:30:00+00:00",
                "live_wake_hold": {"cause": cause, "ceiling_usd": ceiling, "started_at": "2026-10-24T09:10:00+00:00",
                                   **({"resume_additional_usd": 1.5} if estimated else {})}}
    described = view.describe_consciousness_state(snapshot, True)
    assert described["status"] == "wake_paused" and described["detail"] == said
    assert described.get("detail_note") == note
    words = described["detail"] + described.get("detail_note", "")
    # No clock and no id: the block names the wake with its start in the viewer's zone; Costs alone says when room returns.
    assert "09:" not in words and "w1" not in words
    # A held own wake holds back only the next own wake: no sentence claims that nothing wakes.
    assert "No new wake" not in words and "no wake" not in words.lower()
    # Off still wins over any hold, and a note the snapshot carried from an earlier hold never outlives it.
    off = view.describe_consciousness_state({**snapshot, "detail_note": HELD_NOTE}, False)
    assert off["status"] == "disabled" and "detail_note" not in off


@pytest.mark.parametrize("pause,cause,ceiling", [
    (_pause("owner_pause", reason="owner", basis=None), "owner", None),
    (_pause("model_sleep", reason="sleep", scope="task", basis=None), "sleep", None),
    (_pause("billing_authority_unreadable", reason="authority", basis=None), "technical_resume", None),
    (_pause(), "allowance", 2.99),  # a money stop falls back to the launch stamp, which a never-renewed wake still proves
])
def test_an_unreadable_task_record_loses_only_what_it_would_have_proven(tmp_path, monkeypatch, pause, cause, ceiling):
    """Only a money stop reads the task record (the ceiling an earlier Resume may have moved): an owner Pause, a sleep or a
    technical pause keeps its type from the pause row, and a record that cannot be read costs nothing else."""
    metadata = _launched()
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "paused", "metadata": metadata, "budget_pause": pause})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    assert hold.held_wake_cause(tmp_path, "wake-x") == {"cause": cause, "ceiling_usd": ceiling}

    def unreadable(*_args, **_kwargs):
        raise OSError("task store busy")

    monkeypatch.setattr("ouroboros.task_results.load_task_result", unreadable)
    assert hold.held_wake_cause(tmp_path, "wake-x") == {"cause": cause, "ceiling_usd": ceiling}


def test_an_unreadable_pause_leaves_the_cause_unknown_and_a_readable_absent_one_is_queued(monkeypatch, tmp_path):
    """Queued and unreadable are told apart: a record that answers with no live pause is a wake waiting for a worker
    (no start yet to name); one that cannot be read proves nothing."""
    metadata = _launched()
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "queued", "metadata": metadata})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    assert hold.held_wake_cause(tmp_path, "wake-x") == {"cause": "queued", "ceiling_usd": None}

    def unreadable(_root, _task_id):
        raise OSError("store busy")

    monkeypatch.setattr("ouroboros.store_hot_facts.controls", unreadable)
    assert hold.held_wake_cause(tmp_path, "wake-x") == {"cause": "", "ceiling_usd": None}


@pytest.mark.parametrize("error,room", [
    ("independent_cap_exhausted", 0.0), ("work_cap_export_unverified", 0.0),
    ("consciousness_allowance_unknown", None),
])
def test_resume_room_distinguishes_exhausted_hard_caps_from_unreadable_money(tmp_path, monkeypatch, error, room):
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_resume_decision",
                        lambda *_a, **_k: {"error": error})
    assert hold._resume_room(tmp_path, "wake-x", {}, {}) == {"resume_additional_usd": room}


@pytest.mark.parametrize("fence,cause", [
    ({"state": "requested"}, "owner"),
    ({"state": "paused"}, "owner"),
    ({"state": "released"}, "allowance"),
    ({"state": "paused", "selected_members": {"wake-x": {"grant_id": "g"}}}, "allowance"),
])
def test_owner_pause_over_a_parked_budget_wake_takes_precedence(tmp_path, monkeypatch, fence, cause):
    """Pause-all need not rewrite a parked wake's old budget row: its root fence owns the current hold. Either way the
    wake's own start, which its pause row keeps, names it in its block."""
    metadata = _launched()
    started = dt.datetime(2026, 11, 27, 17, 20, 16, tzinfo=dt.timezone.utc)
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "paused", "metadata": metadata,
                                    "budget_pause": {**_pause(), "started_at": started.timestamp()}, "owner_pause": fence})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    held = hold.held_wake_cause(tmp_path, "wake-x")
    assert held == {"cause": cause, "ceiling_usd": 2.99 if cause == "allowance" else None,
                    "started_at": started.isoformat()}
    described = view.describe_consciousness_state({"live_wake_task_id": "wake-x", "live_wake_held": True,
                                                  "live_wake_hold": held}, True)
    assert ("Stop now" in described.get("detail_note", "")) is (cause == "allowance")


QUEUE_HOLDS = [
    ({}, "queued"),  # nothing holds the pending row: it waits for a worker
    ({"_budget_pause_resume": {"grant_id": "g"}}, "queued"),  # a granted Resume on its way to a worker
    # The tick enqueued its own wake while only the total budget was empty: paused before dispatch, in memory only
    # (worker_assignment), no pause row in the store. The owner's own case (27.11).
    ({"_budget_pause": {"status": "paused_before_dispatch", "scope": "global", "auto_resume": False}}, "total_budget"),
    ({"_budget_pause_hold": {"reason": "owner_restart_hold"}}, "restart"),
    ({"_budget_pause_hold": {"reason": "saved_work_hold"}}, "restart"),
    ({"_budget_pause_hold": {"reason": "panic_hold"}}, "panic"),
    ({"_budget_pause_hold": {"reason": "saved_work_hold", "door": "panic"}}, "panic"),  # a working restore of a Panic
    # An owner's Pause before its body started parks the row in memory too (events_budget): said as the owner's.
    ({"_budget_pause": {"status": "paused_before_dispatch", "cause": "owner_pause", "scope": "root"}}, "owner"),
    # A refusal (a consumed or superseded grant, an owner hold) is no queue position and proves no cause.
    ({"_budget_pause_hold": {"reason": "stale_queue_row_grant_consumed"}}, ""),
    ({"_owner_hold": True}, ""),
    ({"_budget_pause_resume": {"grant_id": "g", "refusal": {"reason": "stale_resume_grant_superseded"}}}, ""),
    # A park past its hard deadline cannot start: Close is its offer, never a queue position.
    ({"deadline_at": "2020-01-01T00:00:00+00:00"}, ""),
    ({"deadline_at": "2999-01-01T00:00:00+00:00"}, "queued"),
]


@pytest.mark.parametrize("facts,cause", QUEUE_HOLDS)
def test_a_pending_wake_is_queued_only_when_its_queue_row_holds_nothing(tmp_path, monkeypatch, facts, cause):
    """The store records no live pause for these rows; what holds them is the pool's own row (work_stop's reading), so
    "queued … starts by itself" is said only of a row that carries none."""
    metadata = _launched()
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "queued", "metadata": metadata})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata, **facts}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == cause


def test_an_unselected_budget_latch_of_its_tree_is_no_queue_position(tmp_path, monkeypatch):
    """A tree's money latch that has not selected the row holds it (work_stop.current_stop's tree stop); once the
    latch is gone the same row is queued."""
    metadata = _launched()
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "queued", "metadata": metadata})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    monkeypatch.setattr("supervisor.queue.BUDGET_ROOT_FENCES", {"wake-x": {"status": "active", "fence_id": "f1"}})
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == ""
    monkeypatch.setattr("supervisor.queue.BUDGET_ROOT_FENCES", {})
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == "queued"


@pytest.mark.parametrize("reason,cause", [("sleep", "restart"), ("authority", "restart")])
def test_a_restart_hold_over_a_self_waking_pause_is_the_hold(tmp_path, monkeypatch, reason, cause):
    """A saved sleep or technical pause that a restart held no longer goes on by itself (restart_retention): the hold
    the queue row carries is what holds it now; without that hold the pause keeps its own door."""
    metadata = _launched()
    pause = {**_pause("model_sleep", reason=reason, scope="task", basis=None), "cause": "read_failed", "auto_resume": True}
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "paused", "metadata": metadata, "budget_pause": pause})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata,
                                                      "_budget_pause_hold": {"reason": "owner_restart_hold"}}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == cause
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == {"sleep": "sleep", "authority": "technical"}[reason]


@pytest.mark.parametrize("started,named", [(1795800016.05, True), (1e20, False), ("soon", False)])
def test_a_bad_start_time_drops_only_the_time_never_the_cause(tmp_path, monkeypatch, started, named):
    metadata = _launched()
    pause = {**_pause("global_exhausted", scope="global", basis=None), "started_at": started}
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "paused", "metadata": metadata, "budget_pause": pause})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    held = hold.held_wake_cause(tmp_path, "wake-x")
    assert held["cause"] == "total_budget"
    assert ("started_at" in held) is named


def test_a_held_wake_whose_carrier_row_is_gone_at_lookup_is_unknown_never_queued(tmp_path, monkeypatch):
    """The clock listed the wake held under one lock; dispatch or removal can take the row before this read takes
    another. A row that is not there proves nothing (``""``); a free row there is queued."""
    metadata = _launched()
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "queued", "metadata": metadata})
    monkeypatch.setattr("supervisor.queue.PENDING", [])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == ""
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == "queued"


@pytest.mark.parametrize("durable", [
    # An older, untyped technical pause still stored (it would say "Resume continues it").
    {"budget_pause": {**_pause("authority_unreadable", reason="authority", scope="task", basis=None), "cause": ""}},
    # A closed owner fence of its tree (it would say the owner's Pause).
    {"budget_pause": _pause(), "owner_pause": {"state": "paused"}},
])
def test_a_gone_carrier_row_is_unknown_before_any_stored_fact_is_read(tmp_path, monkeypatch, durable):
    """The stored facts outlive the row the clock saw: with the row gone, none of them is what holds it now."""
    metadata = _launched()
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "paused", "metadata": metadata, **durable})
    monkeypatch.setattr("supervisor.queue.PENDING", [])
    assert hold.held_wake_cause(tmp_path, "wake-x") == {"cause": "", "ceiling_usd": None}
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] != "", "with its row there the stored fact still speaks"


WRITER_WAIT = {"_budget_pause_hold": {"reason": "continuation_writer_unsettled"}}


def test_a_writer_wait_is_no_wait_for_a_worker(tmp_path, monkeypatch):
    """The worker skips a row whose predecessor's writers have not settled (work_stop.writer_wait): it is not waiting
    for a worker, so "queued" is not said of it."""
    metadata = _launched()
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "queued", "metadata": metadata})
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata, **WRITER_WAIT}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == ""


@pytest.mark.parametrize("veto", [
    {"_budget_pause_hold": {"reason": "stale_queue_row_grant_consumed"}},  # a refusal
    WRITER_WAIT,
    {"_owner_hold": True},
])
@pytest.mark.parametrize("reason,self_waking", [("sleep", "sleep"), ("authority", "technical")])
def test_a_queue_veto_takes_the_self_waking_promise_from_a_pause(tmp_path, monkeypatch, veto, reason, self_waking):
    """A sleep or a typed technical pause goes on by itself only while its queue row would let it (the sleeper's own
    veto: a refusal, a tree stop, a writer wait); with one it proves no cause. Without one it keeps its words."""
    metadata = _launched()
    pause = {**_pause("model_sleep", reason=reason, scope="task", basis=None), "cause": "read_failed", "auto_resume": True}
    seed_result(tmp_path, "wake-x", {"task_id": "wake-x", "status": "paused", "metadata": metadata, "budget_pause": pause})
    marker = {"_budget_pause": {"reason": reason, "rail": "model_sleep"}}
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata, **marker, **veto}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == ""
    monkeypatch.setattr("supervisor.queue.PENDING", [{"id": "wake-x", "metadata": metadata, **marker}])
    assert hold.held_wake_cause(tmp_path, "wake-x")["cause"] == self_waking
