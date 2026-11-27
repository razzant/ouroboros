"""The owner's Turn on of consciousness after Panic (``supervisor/panic_choice.py``): their plain Start from any door
turns the background back on with no choice, and only the desktop's confirmed press hands Ouroboros's own wakes the
choice over stopped work.

Real queue rows and their published snapshot, the real clock's ``start`` and ``stop``, the real switch preview and
the mind's control door over isolated state with invented works. Nothing is dispatched and no worker, process or
Panic is started: a released row stays in the fixture's pending list. Refusals are decisions read from state.
"""
import copy
import os
import time

import pytest

from ouroboros.consciousness import AGENT_STOP, BackgroundConsciousness, held_stop_text
from ouroboros.control_actor import host_control, panic_release
from ouroboros.installation_stop import PANIC_RECEIPT
from ouroboros.task_results import write_task_result
from ouroboros.work_stop import panic_hold_id, release_refusal
from supervisor import events_budget as eb
from supervisor import panic_choice
from supervisor.panic_choice import (NOT_AS_SHOWN, active_choice, bind_launcher_start, choice_decision, last_panic,
                                     start_needs_proof, withdraw, worker_view)
from tests._queue_state import gateway as _gateway

gateway = _gateway
PANIC_LINE = "Panic turned consciousness off."


def _held(world, task_id, title, **facts):
    facts = {"root_task_id": task_id, **facts}
    row = {"id": task_id, "chat_id": 1, "type": "task", "_attempt": 1, "admitted_dispatch": "none", "title": title,
           **facts}
    write_task_result(world.root, task_id, "scheduled", chat_id=1, title=title, **facts)
    eb.hold_budget_row(row, reason=eb.HOLD_PANIC)
    world.pending.append(row)
    return row


@pytest.fixture
def after_panic(gateway, monkeypatch):
    """An installation a Panic stopped an hour ago: consciousness off by it and its receipt saved. No work yet."""
    from ouroboros import consciousness
    from supervisor import queue, state

    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", gateway.root / "state" / "queue_snapshot.json")
    monkeypatch.setattr(consciousness.own_time_seed, "settle", lambda root: False)
    state.update_state(lambda live: live.update({
        PANIC_RECEIPT: {"mtime_ns": int((time.time() - 3600) * 1e9)},
        "bg_consciousness_enabled": False, "bg_consciousness_stop_source": "panic"}))
    gateway.state, gateway.queue = state, queue
    gateway.publish = lambda: queue.publish_queue_snapshot(reason="fixture") or True
    gateway.clock = BackgroundConsciousness(gateway.root, gateway.root / "repo", lambda: 1)
    gateway.publish()
    panic_choice._PENDING_PIPE.clear()
    panic_choice._OFF_HERE[0] = False
    return gateway


@pytest.fixture
def stopped(after_panic):
    """The same, with two invented works under their own Panic holds in the published queue."""
    after_panic.report = _held(after_panic, "report", "Weekly report")
    after_panic.digest = _held(after_panic, "digest", "News digest")
    after_panic.publish()
    return after_panic


def _state(world):
    return world.state.load_state()


def _on(world):
    return _state(world).get("bg_consciousness_enabled")


def _shown(world):
    """What the switch reads before the turn: the host's passive preview for this install."""
    from ouroboros.consciousness_view import read_switch_preview

    return read_switch_preview(world.root, "on")


def _press(world, preview_or_basis):
    """Apply the basis confirmed in the desktop native dialog over the closed shell path."""
    basis = preview_or_basis["after_panic"]["basis"] if isinstance(preview_or_basis, dict) else str(preview_or_basis)
    return world.clock.start(basis=basis)


def _wake(world, task_id="wake-1", *, level="act", cause="own_clock", **lineage):
    from ouroboros.consciousness_wake import wake_task_metadata

    metadata = wake_task_metadata(level, "own clock")
    if cause != "own_clock":
        metadata = {**metadata, "wake_cause": cause}
    record = {"metadata": metadata, **(lineage or {"root_task_id": task_id})}
    write_task_result(world.root, task_id, "running", chat_id=1, **record)
    return {"id": task_id, **record}


_CHOSEN = object()


def _wake_resumes(task_id, actor="wake-1", hold=_CHOSEN, **event):
    """A wake's request through the mind's door. By default the conditional one, with the hold its Drive state showed
    (the chosen one; a fixed stranger when none is chosen); ``hold=None``: the ordinary Resume, without
    ``chosen_hold``."""
    from supervisor.events_mind_controls import _apply

    if hold is _CHOSEN:
        hold = ((active_choice() or {}).get("holds") or {}).get(task_id, "0" * 16)
    return _apply({"action": "resume_task", "task_id": task_id, "actor_task_id": actor, "reason": "worth continuing",
                   "action_nonce": f"{actor}:{task_id}", **event, **({} if hold is None else {"chosen_hold": hold})})


def _named(row, title):
    return f"«{title}» (task_id {row['id']}, chosen_hold {panic_hold_id(row)})"


# --- after Panic the owner's plain Start turns it on; only the switch's confirmed press hands over a choice -------

def test_after_panic_the_owners_plain_start_turns_it_on_and_hands_over_no_choice(stopped):
    """Every owner door's ``/bg start`` (app, browser, phone, Telegram, POST /api/command, headless) is this call: as
    in 7.12.1 it turns consciousness on, returns no work and records no choice (the owner 27.11). The mind's own Start
    stays refused, and the desktop's confirmed press can still hand the choice over afterwards."""
    before = (copy.deepcopy(stopped.pending), _state(stopped))
    assert stopped.clock.start(source=AGENT_STOP) == held_stop_text("panic")  # never the mind's to lift
    assert _on(stopped) is False and _state(stopped) == before[1] and stopped.pending == before[0]

    assert stopped.clock.start().startswith("Consciousness on")
    assert _on(stopped) is True and "bg_consciousness_stop_source" not in _state(stopped)
    assert active_choice() is None and "bg_consciousness_panic_choice" not in _state(stopped)
    assert stopped.pending == before[0] and not stopped.running, "the Start returns no work"
    _wake(stopped)
    assert _wake_resumes("report")["error"] == "panic_requires_owner_resume"  # a wake has no choice to stand on

    stopped.publish()  # the supervisor's refresh of its queue certificate, as a live run keeps it fresh
    preview = _shown(stopped)  # the holds still wait: the desktop's press hands the choice over, as at the turn
    assert preview["after_panic"]["why"] == "held" and preview["after_panic"]["choice"] is True
    assert _press(stopped, preview).startswith("Consciousness is on. No task was started")
    assert active_choice() == {
        "panic": last_panic(_state(stopped)),
        "holds": {"report": panic_hold_id(stopped.report), "digest": panic_hold_id(stopped.digest)}}


@pytest.mark.parametrize("state, why", [
    ({"bg_consciousness_enabled": False, "bg_consciousness_stop_source": "panic"}, "panic"),
    ({"bg_consciousness_enabled": False, "bg_consciousness_stop_source": "owner"}, ""),
    ({"bg_consciousness_enabled": False, "bg_consciousness_stop_source": "agent_tool"}, ""),
    ({"bg_consciousness_enabled": False}, ""),  # a stop older than the source key: the ordinary Start, as before
    ({"bg_consciousness_enabled": True, "bg_consciousness_stop_source": "panic"}, ""),
    ({"bg_consciousness_enabled": False, "_recovery": {"unconfirmed": ["bg_consciousness_stop_source"]}}, "unknown"),
    ({"bg_consciousness_stop_source": "owner", "_recovery": {"unconfirmed": ["bg_consciousness_enabled"]}}, "unknown"),
    ({"bg_consciousness_enabled": False, "bg_consciousness_stop_source": "unknown"}, "unknown"),  # kept by a later Off
    ({}, "unknown")])
def test_which_stops_need_the_confirmed_press(state, why):
    assert start_needs_proof(state) == why


def test_the_switch_says_before_the_press_what_turning_on_means(stopped):
    preview = _shown(stopped)

    assert preview["lines"][-5:] == [
        PANIC_LINE,
        "Nothing starts right now: 2 tasks stopped by Panic stay stopped at this press.",
        "Confirmed in the desktop app, Ouroboros may on its later wakes decide which of them to resume, even one the "
        "Panic was pressed for.",
        "A task you stopped or paused separately stays stopped, and deadlines, money and autonomy limits stay in force.",
        "Turning consciousness off, or restarting Ouroboros, ends that choice."]
    assert "panic_start" not in preview
    assert preview["after_panic"]["why"] == "panic" and preview["after_panic"]["choice"] is True
    assert preview["after_panic"]["held"] == 2 and len(preview["after_panic"]["basis"]) == 32
    assert _on(stopped) is False and active_choice() is None and "bg_consciousness_panic_choice" not in _state(stopped)


def test_the_confirmed_turn_turns_consciousness_on_saves_what_was_shown_and_starts_nothing(stopped):
    before = copy.deepcopy(stopped.pending)
    preview = _shown(stopped)

    reply = _press(stopped, preview)

    assert _on(stopped) is True and "bg_consciousness_stop_source" not in _state(stopped)
    assert "bg_consciousness_panic_choice" not in _state(stopped), "Panic choice is never persisted to state.json"
    assert reply.endswith("No task was started; on its wakes Ouroboros may now decide which of the 2 tasks stopped "
                          "by Panic to resume.")
    assert active_choice() == {
        "panic": last_panic(_state(stopped)),
        "holds": {"report": panic_hold_id(stopped.report), "digest": panic_hold_id(stopped.digest)}}
    assert stopped.pending == before and not stopped.running, "the press itself returns no work"


def test_with_no_stopped_work_the_confirmed_turn_still_turns_it_on(after_panic):
    preview = _shown(after_panic)
    assert preview["lines"][-2:] == [
        PANIC_LINE,
        "Nothing starts right now, and this turn hands over no choice to resume Panic-stopped tasks."]
    assert preview["after_panic"]["choice"] is False and preview["after_panic"]["held"] == 0

    reply = _press(after_panic, preview)

    assert _on(after_panic) is True and active_choice() is None
    assert "Panic" not in reply


@pytest.mark.parametrize("setting, value", [("get_consciousness_autonomy", "observe"), ("get_consciousness_daily_usd", 0)])
def test_in_advises_or_without_allowance_it_turns_on_and_hands_over_no_choice(stopped, monkeypatch, setting, value):
    from ouroboros import runtime_limits

    with monkeypatch.context() as patch:
        patch.setattr(runtime_limits, setting, lambda: value)
        preview = _shown(stopped)
        assert preview["lines"][-2:] == [
            PANIC_LINE,
            "Nothing starts right now: 2 tasks stopped by Panic stay stopped, and this turn hands over no choice to "
            "resume them."]

        reply = _press(stopped, preview)

    assert _on(stopped) is True and active_choice() is None
    assert reply.endswith("2 tasks stopped by Panic stay stopped.")
    _wake(stopped)  # even at Acts again, a turn that was shown no choice gave none
    assert _wake_resumes("report")["error"] == "panic_requires_owner_resume"
    assert panic_hold_id(stopped.report)


def test_an_unread_queue_turns_it_on_and_hands_over_no_choice(stopped):
    (stopped.root / "state" / "queue.alive").unlink()  # no certificate: the snapshot is not a fresh fact
    preview = _shown(stopped)
    assert preview["lines"][-2:] == [
        PANIC_LINE,
        "Nothing starts right now: whether Panic stopped any task could not be read, so this turn hands over no "
        "choice to resume tasks."]

    _press(stopped, preview)

    assert _on(stopped) is True and active_choice() is None


def test_without_panic_the_start_is_the_ordinary_one(after_panic):
    after_panic.state.update_state(lambda live: live.update({"bg_consciousness_stop_source": "owner"}))
    preview = _shown(after_panic)
    assert "after_panic" not in preview and "panic_start" not in preview
    assert not any("Panic" in line for line in preview["lines"])

    assert after_panic.clock.start().startswith("Consciousness on")
    assert _on(after_panic) is True and active_choice() is None
    assert after_panic.clock.start().startswith("Consciousness is already on.")


def test_manual_owner_resume_works_while_consciousness_stays_off_after_panic(stopped):
    """Before any Turn on, per-task Resume works on every surface: my human's own press, through the door
    ``api_task_resume`` binds, and never a call that names no door."""
    from ouroboros.control_actor import owner_door
    from supervisor.queue_transitions import resume_budget_paused_task

    hold = panic_hold_id(stopped.report)
    assert resume_budget_paused_task("report", expected_panic_hold_id=hold)["error"] == "panic_requires_owner_resume"
    with owner_door("api_task_resume"):
        outcome = resume_budget_paused_task("report", expected_panic_hold_id=hold)
    assert outcome["ok"] is True, outcome
    assert eb.BUDGET_HOLD_KEY not in stopped.report
    assert panic_hold_id(stopped.digest), "other Panic-stopped tasks remain held"
    assert _on(stopped) is False and active_choice() is None


# --- a new Panic, Off or any change to what was shown leaves an earlier attempt without effect -------------------

@pytest.mark.parametrize("change", [
    "new-panic", "after-an-off", "holds-changed", "mode-changed", "allowance-changed", "already-applied", "new-process",
])
def test_changed_state_off_or_new_panic_leaves_an_earlier_attempt_without_effect(stopped, monkeypatch, change):
    basis = _shown(stopped)["after_panic"]["basis"]
    if change == "new-panic":
        stopped.state.update_state(lambda live: live.update({PANIC_RECEIPT: {"mtime_ns": int(time.time() * 1e9)}}),
                                   confirm=(PANIC_RECEIPT,))
    elif change == "after-an-off":
        stopped.clock.stop(source="owner")
    elif change == "holds-changed":
        stopped.pending.remove(stopped.digest)
        stopped.publish()
    elif change == "mode-changed":
        from ouroboros import runtime_limits
        monkeypatch.setattr(runtime_limits, "get_consciousness_autonomy", lambda: "observe")
    elif change == "allowance-changed":
        from ouroboros import runtime_limits
        monkeypatch.setattr(runtime_limits, "get_consciousness_daily_usd", lambda: 0)
    elif change == "already-applied":
        assert _press(stopped, basis).startswith("Consciousness on")
        stopped.clock.stop(source="owner")
    else:
        monkeypatch.setattr(panic_choice, "_PROCESS_NONCE", "f" * 32)
    panic_choice._OFF_HERE[0] = False
    before = (copy.deepcopy(stopped.pending), _on(stopped), _state(stopped).get("bg_consciousness_stop_source"))

    reply = _press(stopped, basis)

    assert reply == NOT_AS_SHOWN and "stayed" not in reply
    assert (stopped.pending, _on(stopped), _state(stopped).get("bg_consciousness_stop_source")) == before
    assert _on(stopped) is False and before[2] == ("owner" if change == "already-applied" else "panic")
    assert active_choice() is None


def test_launcher_start_handoff_posts_standard_command_reply_and_withdraw_clears_pending(stopped, monkeypatch):
    """The private-pipe handoff calls the bound clock and emits Main's standard ``command_reply`` row; ``withdraw``
    clears pending entries and advances the epoch in memory without waiting."""
    from supervisor import message_bus

    sent = []
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: object())
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))
    bind_launcher_start(lambda: stopped.clock)
    try:
        preview = _shown(stopped)
        epoch = panic_choice._EPOCH[0]
        panic_choice._run_launcher_start(preview["after_panic"]["basis"], epoch)
        assert _on(stopped) is True
        assert len(sent) == 1 and sent[0][0] == 1 and sent[0][1].startswith("🧠 Consciousness on:")
        assert sent[0][2] == {"role": "system", "system_type": "command_reply"}

        # A stale basis over a newer epoch emits the standard refusal row and changes nothing.
        stopped.state.update_state(lambda live: live.update(bg_consciousness_enabled=False, bg_consciousness_stop_source="panic"),
                                   confirm=("bg_consciousness_enabled", "bg_consciousness_stop_source"))
        stopped.clock._enabled = False
        panic_choice._PENDING_PIPE[:] = [(preview["after_panic"]["basis"], panic_choice._EPOCH[0])]
        withdraw()
        assert panic_choice._PENDING_PIPE == []
        panic_choice._run_launcher_start(preview["after_panic"]["basis"], epoch)
        assert _on(stopped) is False and sent[-1][1] == f"🧠 {NOT_AS_SHOWN}"
    finally:
        bind_launcher_start(None)


# --- a later Off keeps the stop's reason and takes the choice back, here and in the saved state -----------------

@pytest.mark.parametrize("by", ["owner", "agent_tool"])
def test_an_off_of_a_clock_stopped_in_an_unknown_way_keeps_that_reason(after_panic, by):
    state = after_panic.state

    def recovered(live):  # a backup recovery: neither the switch nor who stopped it is proven
        live.update(bg_consciousness_enabled=False, bg_consciousness_stop_source="owner")
        state.mark_unconfirmed(live, "bg_consciousness_enabled")
        state.mark_unconfirmed(live, "bg_consciousness_stop_source")

    state.update_state(recovered)
    assert start_needs_proof(_state(after_panic)) == "unknown"

    after_panic.clock.stop(source=by)  # an explicit Off confirms the switch; it does not make the stop an ordinary one

    saved = _state(after_panic)
    assert saved["bg_consciousness_stop_source"] == "unknown" and start_needs_proof(saved) == "unknown"
    refused = after_panic.clock.start(source=AGENT_STOP)  # never the mind's to lift
    assert refused.startswith("Background consciousness stayed off: ") and _on(after_panic) is False
    preview = _shown(after_panic)
    assert preview["lines"][-2:] == [
        "How consciousness was stopped could not be read.",
        "Nothing starts right now, and this turn hands over no choice to resume Panic-stopped tasks."]
    assert _press(after_panic, preview).startswith("Consciousness on")
    assert _on(after_panic) is True and "bg_consciousness_stop_source" not in _state(after_panic)


def test_an_off_after_panic_stays_panics_stop(stopped):
    stopped.clock.stop(source="owner")
    assert _state(stopped)["bg_consciousness_stop_source"] == "panic"
    refused = stopped.clock.start(source=AGENT_STOP)  # never the mind's to lift
    assert refused.startswith("Background consciousness stayed off: ") and _on(stopped) is False
    assert stopped.clock.start().startswith("Consciousness on") and active_choice() is None  # the owner's, no choice


@pytest.mark.parametrize("saved", [True, False])
def test_an_off_taken_while_a_confirmed_turn_on_decides_lands_after_it_and_ends_its_choice(stopped, monkeypatch, saved):
    """The confirmed Turn on decides and records its choice under the clock's lock; an Off taken meanwhile waits for
    that lock, then ends the choice in memory whether or not its own write is saved, and no later ordinary Start
    brings it back."""
    import threading

    from ouroboros import consciousness

    if not saved:  # the Off's own write is lost: it holds in this process only
        monkeypatch.setattr(consciousness, "_persist_stop", lambda source: " Not saved: the stop holds in this process only.")
    real, off = panic_choice.note_start, {}

    def deciding(shown):
        off["thread"] = threading.Thread(target=lambda: stopped.clock.stop(source="owner"), daemon=True)
        off["thread"].start()
        off["thread"].join(0.3)
        off["waited"] = off["thread"].is_alive()
        real(shown)

    monkeypatch.setattr(panic_choice, "note_start", deciding)
    _press(stopped, _shown(stopped))
    off["thread"].join(10)

    assert off["waited"] and not off["thread"].is_alive(), "the Off waited for the Start's decision"
    assert active_choice() is None and _on(stopped) is (not saved)
    monkeypatch.setattr(panic_choice, "note_start", real)
    assert stopped.clock.start().startswith("Consciousness on") and active_choice() is None


def test_an_off_whose_write_failed_still_takes_the_choice_back(stopped, monkeypatch):
    """The stop this process took revokes the choice in memory at once; even a restart over the un-saved state starts
    with zero choice and never resurrects the revoked permission."""
    from supervisor.state import StateUnavailable

    _press(stopped, _shown(stopped))
    _wake(stopped)
    real = stopped.state.update_state
    with monkeypatch.context() as patch:
        patch.setattr(stopped.state, "update_state", lambda *a, **kw: (_ for _ in ()).throw(StateUnavailable("lock_timeout", "update")))
        assert "the stop holds in this process only" in stopped.clock.stop(source="owner")
    assert stopped.state.update_state is real
    assert _on(stopped) is True, "the saved state never heard of that Off"
    before = copy.deepcopy(stopped.pending)

    assert active_choice() is None
    assert _wake_resumes("report")["error"] == "panic_requires_owner_resume" and stopped.pending == before
    assert stopped.clock.start().startswith("Consciousness on")  # their ordinary Start after their own Off
    assert active_choice() is None
    assert _wake_resumes("report")["error"] == "panic_requires_owner_resume" and stopped.pending == before

    # A new server/controller boot over the same state also starts with zero Panic choice.
    restarted = BackgroundConsciousness(stopped.root, stopped.root / "repo", lambda: 1)
    assert restarted.enabled is True and active_choice() is None
    assert _wake_resumes("report")["error"] == "panic_requires_owner_resume" and stopped.pending == before


def test_choice_lasts_only_until_restart_and_already_on_consciousness_can_hand_over_remaining_holds(stopped):
    """Option 1 (26.11): Panic choice lives only in this server run. After an ordinary Restart/Update where background
    consciousness auto-restores as already on, zero prior Panic choice survives and no dialog opens at boot; the owner
    can hand over remaining Panic-stopped tasks on desktop without toggling Off -> On or resetting the wake clock."""
    from supervisor.panic_choice import wake_fact

    _press(stopped, _shown(stopped))
    _wake(stopped, "wake-1")
    assert _wake_resumes("report", "wake-1")["ok"] is True
    stopped.publish()
    assert eb.BUDGET_HOLD_KEY not in stopped.report and panic_hold_id(stopped.digest)

    # Simulate an ordinary Restart / Update: new controller run over the same drive_root with consciousness already on.
    stopped.clock = BackgroundConsciousness(stopped.root, stopped.root / "repo", lambda: 1)
    assert stopped.clock.enabled is True, "ordinary background consciousness stays on across Restart/Update"
    assert active_choice() is None, "prior run's Panic choice does not survive Restart/Update"
    assert stopped.clock.status_snapshot().get("panic_choice_available") is True

    wake2 = _wake(stopped, "wake-2")
    assert wake_fact(stopped.root, wake2) == ""
    assert _wake_resumes("digest", "wake-2")["error"] == "panic_requires_owner_resume"
    assert stopped.clock.start() == "Consciousness is already on." and active_choice() is None

    # Owner initiates handover on desktop while consciousness is already on (no Off -> On, wake schedule preserved).
    scheduled_wake = stopped.clock.next_wake_at
    preview = _shown(stopped)
    assert preview["places"] == []
    assert preview["after_panic"]["why"] == "held" and preview["after_panic"]["choice"] is True
    assert preview["after_panic"]["held"] == 1 and len(preview["after_panic"]["basis"]) == 32
    assert preview["lines"] == [
        "Consciousness is already on; tasks stopped by Panic still wait for your decision in this run.",
        "Nothing starts right now: 1 task stopped by Panic stays stopped at this press.",
        "Confirmed in the desktop app, Ouroboros may on its later wakes decide whether to resume it, even one the "
        "Panic was pressed for.",
        "A task you stopped or paused separately stays stopped, and deadlines, money and autonomy limits stay in force.",
        "Turning consciousness off, or restarting Ouroboros, ends that choice."]

    reply = _press(stopped, preview)
    assert reply == ("Consciousness is on. No task was started; on its wakes Ouroboros may now decide which of the "
                     "1 task stopped by Panic to resume.")
    assert stopped.clock.enabled is True and stopped.clock.next_wake_at == scheduled_wake
    assert "panic_choice_available" not in stopped.clock.status_snapshot()
    assert _shown(stopped) != preview and "after_panic" not in _shown(stopped)
    assert _named(stopped.digest, "News digest") in wake_fact(stopped.root, wake2)
    assert _wake_resumes("digest", "wake-2")["ok"] is True


def test_a_recovered_receipt_is_unknown_until_written_again(stopped):
    """``panic_ingress_receipt`` is a control: after a backup recovery its saved value proves nothing; Panic choice is
    never stored in ``state.json`` at all."""
    from supervisor import state as state_module

    assert "panic_ingress_receipt" in state_module.CONTROL_KEYS
    assert "bg_consciousness_panic_choice" not in state_module.CONTROL_KEYS
    _press(stopped, _shown(stopped))
    _wake(stopped)
    stopped.state.update_state(lambda live: state_module.mark_unconfirmed(live, "panic_ingress_receipt"))
    before = copy.deepcopy(stopped.pending)

    with pytest.raises(LookupError):
        last_panic(_state(stopped))
    assert _wake_resumes("report")["error"] == "panic_selection_changed" and stopped.pending == before


def test_after_a_recovery_a_confirmed_turn_turns_on_and_hands_over_no_choice(stopped):
    from supervisor import state as state_module

    stopped.state.update_state(lambda live: state_module.mark_unconfirmed(live, "panic_ingress_receipt"))
    preview = _shown(stopped)
    assert preview["lines"][-2:] == [
        PANIC_LINE,
        "Nothing starts right now: whether Panic stopped any task could not be read, so this turn hands over no "
        "choice to resume tasks."]

    _press(stopped, preview)

    assert _on(stopped) is True and active_choice() is None


def test_the_boot_that_receives_a_panic_confirms_its_receipt():
    """``auto_resume_after_restart`` writes the receipt as a decision, so a recovered one becomes known again."""
    import ast
    import inspect

    from supervisor import worker_chat_lane

    calls = [node for node in ast.walk(ast.parse(inspect.getsource(worker_chat_lane.auto_resume_after_restart)))
             if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "update_state"]
    assert any("PANIC_RECEIPT" in ast.unparse(call) and any(kw.arg == "confirm" for kw in call.keywords) for call in calls)


# --- the choice, decided from facts read at the grant -----------------------------------------------------------

GRANTED = dict(wake=True, enabled=True, autonomy="act", last_panic=7, task_id="report", hold_id="h1")


@pytest.mark.parametrize("change, decision", [
    ({}, "granted"),
    ({"wake": False}, "none"), ({"task_id": "other"}, "none"),
    ({"enabled": False}, "changed"), ({"enabled": None}, "changed"), ({"autonomy": "observe"}, "changed"),
    ({"last_panic": 8}, "changed"), ({"last_panic": None}, "changed"),
    ({"hold_id": "h2"}, "changed"),  # stopped again
    ({"hold_id": ""}, "changed"),    # its hold released, or one whose occurrence cannot be told
])
def test_the_choice_decision(change, decision):
    chosen = {"panic": 7, "holds": {"report": "h1", "digest": "h9"}}
    assert choice_decision(chosen, **{**GRANTED, **change}) == decision
    assert choice_decision(None, **GRANTED) == choice_decision({"panic": 7}, **GRANTED) == "none"
    assert choice_decision({"panic": None, "holds": {"report": "h1"}}, **{**GRANTED, "last_panic": None}) == "changed"


def test_a_wake_returns_one_shown_work_through_its_own_hold(stopped):
    _press(stopped, _shown(stopped))
    _wake(stopped)
    digest_before = copy.deepcopy(stopped.digest)

    outcome = _wake_resumes("report")

    assert outcome["ok"], outcome
    assert outcome["panic_release"] == {"kind": "consciousness_choice", "task_id": "report"}
    assert eb.BUDGET_HOLD_KEY not in stopped.report
    assert stopped.digest == digest_before and panic_hold_id(stopped.digest), "each work is its own decision"
    assert not stopped.running


@pytest.mark.parametrize("actor", ["ordinary-task", "advises", "focus", "helper-of-the-wake"])
def test_the_choice_is_a_root_wakes_of_its_own_clock_above_advises(stopped, actor):
    _press(stopped, _shown(stopped))
    if actor == "advises":
        _wake(stopped, actor, level="observe")
    elif actor == "focus":
        _wake(stopped, actor, cause="focus")
    elif actor == "helper-of-the-wake":
        _wake(stopped, "wake-1")
        _wake(stopped, actor, root_task_id="wake-1", parent_task_id="wake-1", delegation_role="subagent")
    else:
        write_task_result(stopped.root, actor, "running", root_task_id=actor, chat_id=1)

    assert _wake_resumes("report", actor)["error"] == "panic_requires_owner_resume"
    assert panic_hold_id(stopped.report)


@pytest.mark.parametrize("now", ["advises-now", "off", "off-then-on-again", "panic-latched", "next-panic-controls",
                                 "next-panic-received", "switch-unconfirmed", "list-resume"])
def test_what_the_wake_finds_changed_at_the_grant(stopped, monkeypatch, now):
    from ouroboros import runtime_limits, server_control
    from ouroboros.server_control import PANIC_CONTROL_KEYS, _panic_controls

    _press(stopped, _shown(stopped))
    _wake(stopped)  # the wake was created at Acts, with the choice standing
    state, expected, event = stopped.state, "panic_selection_changed", {}
    if now == "advises-now":  # the level my human set since is read at the grant, not the wake's own stamp
        monkeypatch.setattr(runtime_limits, "get_consciousness_autonomy", lambda: "observe")
    elif now.startswith("off"):
        stopped.clock.stop(source="owner")
        assert active_choice() is None
        if now == "off-then-on-again":  # an ordinary Start after their own Off carries no choice
            assert stopped.clock.start().startswith("Consciousness on") and active_choice() is None
        expected = "panic_requires_owner_resume"
    elif now == "panic-latched":
        monkeypatch.setattr(server_control, "_restart_stop_requested", True)
        assert active_choice() is None
        expected = "panic_requires_owner_resume"
    elif now == "next-panic-controls":
        state.update_state(_panic_controls, confirm=PANIC_CONTROL_KEYS)
        expected = "panic_requires_owner_resume"
    elif now == "next-panic-received":  # its controls write was lost: the receipt alone names a newer Panic
        state.update_state(lambda live: live.update({PANIC_RECEIPT: {"mtime_ns": int(time.time() * 1e9)}}))
    elif now == "switch-unconfirmed":  # a state recovered from its backup
        state.update_state(lambda live: live.update({state.RECOVERY_KEY: {"unconfirmed": ["bg_consciousness_enabled"]}}))
    else:  # a general return never stands on the choice
        expected, event = "panic_requires_owner_resume", {"list_call": "call-1", "general": True, "hold": None}
    before = copy.deepcopy(stopped.pending)

    assert _wake_resumes("report", **event)["error"] == expected
    assert stopped.pending == before and panic_hold_id(stopped.report)


@pytest.mark.parametrize("other", ["stopped-again", "occurrence-unknown", "hold-released", "not-shown", "their-pause",
                                   "deadline-passed"])
def test_a_named_work_answers_to_its_own_stop_and_never_to_the_general_resume(stopped, other):
    from ouroboros.owner_pause import install_fence

    if other == "deadline-passed":  # a work past its hard deadline is never among those shown
        _held(stopped, "late", "Overdue check", deadline_at="2026-01-01T00:00:00Z")
        stopped.publish()
    preview = _shown(stopped)
    assert preview["lines"][-4].startswith("Nothing starts right now: 2 tasks stopped by Panic")
    _press(stopped, preview)
    _wake(stopped)
    target, expected = "report", "panic_selection_changed"
    if other == "stopped-again":  # not the hold they were shown
        stopped.report[eb.BUDGET_HOLD_KEY]["held_at"] = "2099-01-01T00:00:00+00:00"
    elif other == "occurrence-unknown":
        stopped.report[eb.BUDGET_HOLD_KEY].pop("held_at")
    elif other == "hold-released":  # my human returned it themselves; what is left is not the chosen stop
        stopped.report.pop(eb.BUDGET_HOLD_KEY)
    elif other == "not-shown":  # stopped work the choice does not name stays the owner's
        _held(stopped, "later", "Added afterwards")
        target, expected = "later", "panic_requires_owner_resume"
    elif other == "their-pause":  # a separate Pause of the owner's on the same work
        install_fence(stopped.root, "report", request_id="their-pause", by={"actor": "owner"})
        expected = "selection_owner_held"
    else:
        target, expected = "late", "panic_requires_owner_resume"
    before = copy.deepcopy(stopped.pending)

    refused = _wake_resumes(target)

    assert refused["ok"] is False and refused["error"] == expected, refused
    assert "panic_release" not in refused or other == "their-pause"
    assert stopped.pending == before and not stopped.running


def test_the_grant_asks_again_under_the_queue_lock(stopped, monkeypatch):
    """The conditional door itself refuses a release whose choice no longer stands (``choice_stands``)."""
    from supervisor.queue_transitions import resume_budget_paused_task

    _press(stopped, _shown(stopped))
    hold = panic_hold_id(stopped.report)
    withdraw()  # withdrawn in controller memory after the door was entered
    before = copy.deepcopy(stopped.pending)

    with panic_release({"kind": "consciousness_choice", "task_id": "report"}):
        assert resume_budget_paused_task("report", expected_panic_hold_id=hold)["error"] == "panic_selection_changed"
    assert stopped.pending == before


def test_only_the_request_that_says_so_stands_on_the_choice(stopped):
    """The conditional return is chosen by the request itself: the wake's ordinary Resume stays the ordinary one (the
    Panic hold refuses it), and a conditional one that mixes in a list or an accepted unknown start does nothing."""
    _press(stopped, _shown(stopped))
    _wake(stopped)
    before = copy.deepcopy(stopped.pending)

    ordinary = _wake_resumes("report", hold=None)
    assert ordinary["error"] == "panic_requires_owner_resume" and "panic_release" not in ordinary
    for mixed in ({"list_call": "call-1", "general": True}, {"accept_unknown": ["run-1"]}):
        assert _wake_resumes("report", **mixed)["error"] == "invalid_panic_selection"
    assert stopped.pending == before and panic_hold_id(stopped.report)
    assert _wake_resumes("report")["ok"] is True


@pytest.mark.parametrize("parent, expected", [
    ("held", "selection_owner_held"),        # its parent still waits under its own Panic hold
    ("returned", None),                      # the wake returned the parent first: the helper is a choice of its own
    ("finished", None),                      # the parent ended without a Stop
    ("stopped", "selection_owner_held"),     # the parent was stopped
    ("cut-off", "selection_owner_held"),     # the parent was interrupted and nothing holds it for a return
    ("unknown", "selection_authority_unavailable")])
def test_a_helper_returns_alone_and_only_on_its_parents_standing(stopped, parent, expected):
    helper = _held(stopped, "helper", "Report helper", root_task_id="report", parent_task_id="ghost"
                   if parent == "unknown" else "report", delegation_role="subagent")
    stopped.publish()
    _press(stopped, _shown(stopped))
    _wake(stopped)
    assert active_choice()["holds"]["helper"] == panic_hold_id(helper), "a helper is among the works handed over"
    if parent == "returned":
        assert _wake_resumes("report")["ok"] is True
    elif parent in ("finished", "stopped", "cut-off"):
        stopped.pending.remove(stopped.report)
        status = {"finished": "completed", "stopped": "cancelled", "cut-off": "interrupted"}[parent]
        write_task_result(stopped.root, "report", status, root_task_id="report", chat_id=1, title="Weekly report")
    before = copy.deepcopy(stopped.pending)

    outcome = _wake_resumes("helper")

    if expected:
        assert outcome["error"] == expected and stopped.pending == before and panic_hold_id(helper)
    else:
        assert outcome["ok"] is True, outcome
        assert eb.BUDGET_HOLD_KEY not in helper and outcome["panic_release"]["task_id"] == "helper"
        assert panic_hold_id(stopped.digest), "no cascade: other works keep their holds"


@pytest.mark.parametrize("parent, expected", [
    ("running", None), ("warm-paused", "selection_owner_held"), ("nowhere", "selection_authority_unavailable"),
    ("retry-running", None), ("retry-gone", "selection_authority_unavailable")])
def test_a_parent_counts_as_returned_only_when_shown_live(stopped, monkeypatch, parent, expected):
    """The pure predicate over inert facts: a live row, never one parked warm under a Pause; the root read as its
    physical actor (a timeout retry); a parent neither live nor finished leaves the helper standing."""
    from ouroboros import cancel_intents
    from supervisor.panic_choice import parent_refusal

    helper = _held(stopped, "helper", "Report helper", root_task_id="report", parent_task_id="report",
                   delegation_role="subagent")
    stopped.pending.remove(stopped.report)
    actor = "report-retry" if parent.startswith("retry") else "report"
    monkeypatch.setattr(cancel_intents, "_validated_single_cancel_target", lambda drive, tid: actor)
    write_task_result(stopped.root, "report", "scheduled", root_task_id="report", chat_id=1, title="Weekly report")
    if parent != "nowhere" and parent != "retry-gone":
        meta = {"task": {"id": actor, "root_task_id": "report"},
                **({"owner_wait": {"state": "waiting", "reason": "owner_pause"}} if parent == "warm-paused" else {})}
        monkeypatch.setitem(stopped.queue.RUNNING, actor, meta)
    if parent == "retry-gone":
        write_task_result(stopped.root, actor, "scheduled", root_task_id="report", chat_id=1, title="Weekly report")

    with stopped.queue._queue_lock:
        refusal = parent_refusal(stopped.queue, helper)

    assert (refusal or {}).get("error") == expected


def test_a_wake_never_chooses_inside_its_own_tree(stopped):
    _held(stopped, "mine", "Wake's own helper", root_task_id="wake-1", parent_task_id="wake-1",
          delegation_role="subagent")
    stopped.publish()
    _press(stopped, _shown(stopped))
    _wake(stopped)
    assert _wake_resumes("mine")["error"] == "panic_requires_owner_resume"


def test_an_off_waits_out_a_grant_being_written_and_admits_none_after(stopped):
    """The grant holds ``grant_lock`` from its last check to its write: an Off taken meanwhile is visible at once to
    every later check, and returns only after that grant is written; nothing written is taken back."""
    import threading

    from supervisor.panic_choice import grant_lock

    _press(stopped, _shown(stopped))
    _wake(stopped)
    returned = threading.Event()
    with grant_lock():
        off = threading.Thread(target=lambda: (withdraw(), returned.set()), daemon=True)
        off.start()
        assert not returned.wait(0.3), "the Off waits for the grant in progress"
        assert active_choice() is None, "later checks already see the Off"
    off.join(5)
    assert returned.is_set()
    assert _wake_resumes("digest", hold=panic_hold_id(stopped.digest))["error"] == "panic_requires_owner_resume"
    assert panic_hold_id(stopped.digest)


def test_after_their_off_and_on_the_desktop_can_hand_over_again(stopped):
    """Off ends the choice and an ordinary On never brings it back; while the holds remain, a new desktop press may."""
    _press(stopped, _shown(stopped))
    stopped.clock.stop(source="owner")
    assert stopped.clock.start().startswith("Consciousness on") and active_choice() is None

    preview = _shown(stopped)
    assert preview["after_panic"]["why"] == "held" and preview["after_panic"]["choice"] is True
    _press(stopped, preview)
    _wake(stopped)
    assert _wake_resumes("report")["ok"] is True


def test_a_wakes_choice_keeps_the_trees_money_cap():
    from supervisor.budget_resume import _exact_tree_money_required

    assert _exact_tree_money_required({}, "budget", {}) is False, "my human's own press keeps the existing rule"
    with panic_release({"kind": "consciousness_choice", "task_id": "report"}):
        assert _exact_tree_money_required({}, "budget", {}) is True


# --- what the wake itself is told ------------------------------------------------------------------------------

def test_a_wake_reads_its_right_only_while_it_has_one(stopped, monkeypatch):
    from ouroboros import runtime_limits
    from supervisor.panic_choice import wake_fact

    wake = _wake(stopped)
    assert wake_fact(stopped.root, wake) == "", "no choice was made"
    _press(stopped, _shown(stopped))

    fact = wake_fact(stopped.root, wake)
    assert fact == (
        "After Panic my human left me the choice of the work it stopped. Still stopped and mine to decide: "
        f"{_named(stopped.report, 'Weekly report')}; {_named(stopped.digest, 'News digest')}. "
        "resume_task(task_id, chosen_hold) with the hold shown here returns one, alone; none is owed, and one of them "
        "may be what the Panic was pressed for. If its stop changed since, the request is refused and nothing else is "
        "resumed. A helper returns alone, once its parent runs again or finished without a Stop. A separate Stop or "
        "Pause, a deadline and money still apply. Off, a restart or a new Panic ends this choice.")
    for other in (_wake(stopped, "advises", level="observe"), _wake(stopped, "focus", cause="focus"),
                  {"id": "task", "root_task_id": "task", "metadata": {}}, None):
        assert wake_fact(stopped.root, other) == ""

    assert _wake_resumes("report")["ok"]
    stopped.publish()
    assert (_named(stopped.digest, "News digest") in wake_fact(stopped.root, wake)
            and "report" not in wake_fact(stopped.root, wake))
    with monkeypatch.context() as patch:
        patch.setattr(runtime_limits, "get_consciousness_autonomy", lambda: "observe")
        assert wake_fact(stopped.root, wake) == ""
    stopped.clock.stop(source="owner")
    assert wake_fact(stopped.root, wake) == ""


def test_spawned_worker_projection_is_session_bound_and_never_grants_authority(stopped, monkeypatch):
    """A spawned wake worker renders ``wake_fact`` from ``_panic_choice_view`` only within the same custody session,
    and that projection can never authorize a release once controller memory withdrew or restarted."""
    from supervisor.panic_choice import wake_fact

    _press(stopped, _shown(stopped))
    view = worker_view()
    assert isinstance(view, dict) and view["holds"]["report"] == panic_hold_id(stopped.report)
    wake = {**_wake(stopped), "_panic_choice_view": view}
    stale = {**_wake(stopped, "wake-stale"), "_panic_choice_view": {**view, "session": "0" * 32}}

    with monkeypatch.context() as patch:
        patch.setenv("OUROBOROS_IN_WORKER", "1")
        assert _named(stopped.report, "Weekly report") in wake_fact(stopped.root, wake)
        assert wake_fact(stopped.root, stale) == "", "a previous server session's view is ignored"

    withdraw()  # controller memory clears; the task dict still holds ``_panic_choice_view``
    assert _wake_resumes("report")["error"] == "panic_requires_owner_resume"
    assert panic_hold_id(stopped.report)


def test_the_drive_state_section_carries_the_fact_for_that_wake_alone(stopped):
    from types import SimpleNamespace

    from ouroboros.context_sections import _drive_state_section

    _press(stopped, _shown(stopped))
    env = lambda task: SimpleNamespace(drive_path=lambda rel: stopped.root / rel, task=task)
    said = lambda task: "left me the choice of the work it stopped" in str(_drive_state_section(env(task)))

    assert said(_wake(stopped)) and not said({"id": "main-turn", "root_task_id": "main-turn", "metadata": {}})


def test_the_release_lifts_only_the_owner_bar_of_a_panic_hold():
    stop = {"cause": "panic", "release": "owner"}
    assert release_refusal(stop, {"actor": "ouroboros"})["error"] == "panic_requires_owner_resume"
    with panic_release({"kind": "consciousness_choice"}):
        assert release_refusal(stop, {"actor": "ouroboros"}) is None
        assert release_refusal({"release": "none"}, {"actor": "ouroboros"})["error"] == "stop_not_resumable"
        with host_control():  # the host's own act never inherits it
            assert release_refusal(stop, {})["error"] == "panic_requires_owner_resume"
    assert release_refusal(stop, {"actor": "ouroboros"})["error"] == "panic_requires_owner_resume"


def test_last_panic_reads_the_receipt_as_a_control():
    assert last_panic({}) is None, "no Panic was ever received"
    assert last_panic({"panic_ingress_receipt": {"mtime_ns": 1_700_000_000_500_000_000}}) == 1_700_000_000_500_000_000
    for unknown in ({"panic_ingress_receipt": {"mtime_ns": 5}, "_recovery": {"unconfirmed": ["panic_ingress_receipt"]}},
                    {"panic_ingress_receipt": {"mtime_ns": 5}, "_state_read": {"quality": "unavailable"}}):
        with pytest.raises(LookupError):
            last_panic(unknown)


def test_ordinary_polls_do_not_read_queue_snapshot_without_pending_panic_holds(after_panic, monkeypatch):
    """Ordinary ``status_snapshot`` / ``handover_available`` polls never touch ``load_queue_document`` when there is no
    Panic, nor after a historical Panic once no Panic-held task waits in the live ``PENDING`` queue."""
    from ouroboros import queue_observation

    def forbidden(*_a, **_kw):
        raise AssertionError("ordinary poll must not read queue_snapshot.json")

    # 1. Historical Panic receipt still in state.json, consciousness turned back on, no Panic-held rows in PENDING.
    _press(after_panic, _shown(after_panic))
    assert after_panic.clock.enabled is True
    after_panic.pending.append({"id": "ordinary-1", "root_task_id": "ordinary-1", "chat_id": 1, "type": "task"})
    monkeypatch.setattr(queue_observation, "load_queue_document", forbidden)
    snap = after_panic.clock.status_snapshot()
    assert snap["enabled"] is True and "panic_choice_available" not in snap

    # 2. Fresh install that never had a Panic receipt (even without owns_live_queue).
    after_panic.state.update_state(lambda live: live.pop(PANIC_RECEIPT, None))
    monkeypatch.setattr(after_panic.queue, "INITIALIZED", False)
    snap_fresh = after_panic.clock.status_snapshot()
    assert snap_fresh["enabled"] is True and "panic_choice_available" not in snap_fresh


def test_panic_latch_blocks_confirmation_and_choice_without_panic_entering_panic_choice(stopped, monkeypatch):
    """Emergency ``execute_panic_stop`` has zero reference to ``panic_choice``; the existing irreversible Panic latch
    blocks confirmations, starts, ticks and choice grants from the reader/grant side."""
    import inspect

    from ouroboros import server_control
    from supervisor.panic_choice import confirm_start, handover_available, receive_launcher_start, wake_fact

    for fn in (server_control.execute_panic_stop, server_control._panic_controls):
        src = inspect.getsource(fn)
        assert "panic_choice" not in src and "bg_consciousness_panic_choice" not in src

    preview = _shown(stopped)
    basis = preview["after_panic"]["basis"]
    _press(stopped, preview)
    wake = _wake(stopped)
    assert active_choice() is not None

    next_basis = "a" * 32
    monkeypatch.setattr(server_control, "_restart_stop_requested", True)
    assert active_choice() is None
    assert handover_available(stopped.root) is False
    assert receive_launcher_start(next_basis) is False
    assert confirm_start(stopped.root, basis) is None
    assert wake_fact(stopped.root, wake) == ""
    assert "Panic" in stopped.clock.start(basis=basis)
    assert stopped.clock.tick() == "panic_stop"
    assert _wake_resumes("report")["error"] == "panic_requires_owner_resume"
