"""Contracts for INTERACTIVE transport-wait episodes (direct-chat turns):
the raw idle-timeout bound measured from episode entry, its
None-aware minimum with an explicit deadline, the final free redial at that
bound, the untouched managed rails, the mailbox wake of a direct turn, notes
that preserve ordinary progress and incident delivery. Shared fixtures live in
``tests/test_loop_transport_wait.py``.
"""

from __future__ import annotations

import json
import queue
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
import ouroboros.loop_transport as loop_transport
from ouroboros.config import get_finalization_grace_sec
from ouroboros.loop import run_llm_loop
from ouroboros.tools.registry import ToolRegistry
from tests.test_loop_transport_wait import (
    _FakeClock,
    _loop_kwargs,
    _read_network_wait_events,
    _transport_failing_call,
)

INTERACTIVE_DETAIL = "interactive_wait_window_exhausted"

# A managed episode's owner-visible texts are a frozen contract: these are the
# literal wordings the base emits, and the ONLY notes a managed episode gets.
BASE_MANAGED_ENTRY = (
    "🌐 The request did not reach the provider — waiting and "
    "redialing automatically (failed attempts are $0). Stop cancels."
)
BASE_MANAGED_RECOVERY = "🌐 Provider connection restored after 1.5 min — resuming."


class _NoteRecorder:
    """emit_progress fake honoring the ``incident=`` keyword contract."""

    def __init__(self):
        self.texts = []
        self.incidents = []

    def __call__(self, text, *, incident=None):
        self.texts.append(text)
        self.incidents.append(incident)


def _ctx(**flags):
    return SimpleNamespace(task_metadata={}, task_attempt=None, **flags)


def _enter(tmp_path, ctx, notes, task_id="t-i"):
    return loop_transport.reconcile_transport_wait(
        None, ctx, msg_present=False, error_kind="transport_unavailable",
        drive_logs=tmp_path, task_id=task_id, model="m", emit_progress=notes,
    )


def _step(episode, tmp_path, tools, notes, task_id="t-i", error_kind="transport_unavailable"):
    return loop_transport.transport_wait_step(
        episode, tools=tools, error_kind=error_kind,
        drive_root=None, drive_logs=tmp_path, task_id=task_id, model="m",
        emit_progress=notes, incoming_messages=None, owner_msg_seen=set(),
    )


def _run_until_terminal(episode, tmp_path, tools, notes, task_id="t-i", limit=50):
    """Redial until the step terminalizes; returns the number of granted redials."""
    redials = 0
    while _step(episode, tmp_path, tools, notes, task_id=task_id):
        redials += 1
        assert redials < limit, "an interactive episode must terminalize on its own"
    return redials


# ------------------------------------------------------------ the idle bound

@pytest.mark.parametrize("flag", ["is_direct_chat"])
def test_interactive_bound_is_the_raw_idle_getter_measured_from_entry(tmp_path, monkeypatch, flag):
    """The bound is the configured idle timeout itself, the same literal window
    the queue's silence rail applies to managed records, which these turns never
    have; it starts at episode entry, its expiry ends the episode with its own
    detail, and the waiting rows carry the shrinking window."""
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 90)
    notes = _NoteRecorder()
    ctx = _ctx(**{flag: True})
    episode = _enter(tmp_path, ctx, notes)

    assert episode.interactive is True
    assert episode.wait_bound_sec == 90.0
    redials = _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes)

    assert redials >= 3
    assert sum(clock.sleeps) <= 90.0
    assert episode.final_redial_done is True  # the last grant reserved the margin
    events = _read_network_wait_events(tmp_path)
    assert events[-1]["phase"] == "ended"
    assert events[-1]["detail"] == INTERACTIVE_DETAIL
    windows = [row["window_remaining_sec"] for row in events if row["phase"] == "waiting"]
    assert windows[0] == 90.0
    assert windows == sorted(windows, reverse=True)


@pytest.mark.parametrize("metadata", ["not-a-dict", {"deadline_at": "garbage"}, None])
def test_malformed_task_metadata_keeps_the_interactive_bound(tmp_path, monkeypatch, metadata):
    """No parseable deadline means no deadline window: the idle bound still
    binds, so a malformed metadata carrier can neither unbound the wait nor
    crash the step."""
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    notes = _NoteRecorder()
    ctx = SimpleNamespace(task_metadata=metadata, task_attempt=None, is_direct_chat=True)
    episode = _enter(tmp_path, ctx, notes)
    _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes)

    assert 0.0 < sum(clock.sleeps) <= 60.0
    assert _read_network_wait_events(tmp_path)[-1]["detail"] == INTERACTIVE_DETAIL


def test_zero_idle_bound_terminalizes_immediately_with_zero_wait_wording(tmp_path, monkeypatch):
    """A spent-at-entry bound (the getter clamps at 60 s, but the step must not
    depend on it) ends the episode before any sleep, and the terminal text
    then says no window was left instead of claiming a wait."""
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 0)
    ctx = _ctx(is_direct_chat=True)
    notes = _NoteRecorder()
    episode = _enter(tmp_path, ctx, notes)

    assert _step(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes) is False
    assert clock.sleeps == []
    assert _read_network_wait_events(tmp_path)[-1]["detail"] == INTERACTIVE_DETAIL
    assert episode.waited_sec == 0.0
    text = loop_transport.provider_terminal_fallback_text(
        {}, is_context_overflow=False, is_transport_wait=True,
        waited_sec=episode.waited_sec, interactive=episode.interactive,
        is_deadline_exhausted=False,
    )
    assert "no wait window was left" in text
    assert "this turn" in text


# ------------------------------------------- minimum with an explicit deadline

def test_explicit_deadline_shorter_than_the_bound_keeps_its_deadline_detail(tmp_path, monkeypatch):
    """When both windows exist the shorter one binds: an owner deadline closing
    before the idle bound ends the episode with a deadline detail, and every
    waiting row reports that shorter window."""
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 900)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 8)
    ctx = _ctx(is_direct_chat=True)
    ctx.task_metadata = {"deadline_at": deadline.isoformat()}
    notes = _NoteRecorder()
    episode = _enter(tmp_path, ctx, notes)
    _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes)

    events = _read_network_wait_events(tmp_path)
    assert events[-1]["detail"] == "deadline_after_final_redial"
    assert all(row["window_remaining_sec"] <= 8.5 for row in events if row["phase"] == "waiting")
    assert sum(clock.sleeps) < 900.0


def test_explicit_deadline_longer_than_the_bound_yields_the_interactive_detail(tmp_path, monkeypatch):
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    deadline = datetime.now(timezone.utc) + timedelta(seconds=get_finalization_grace_sec() + 3600)
    ctx = _ctx(is_direct_chat=True)
    ctx.task_metadata = {"deadline_at": deadline.isoformat()}
    notes = _NoteRecorder()
    episode = _enter(tmp_path, ctx, notes)
    _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes)

    events = _read_network_wait_events(tmp_path)
    assert events[-1]["detail"] == INTERACTIVE_DETAIL
    assert sum(clock.sleeps) <= 60.0
    assert max(row["window_remaining_sec"] for row in events if row["phase"] == "waiting") <= 60.0


@pytest.mark.parametrize("final_redial_done", [False, True])
@pytest.mark.parametrize("earlier", ["bound", "deadline"])
def test_both_windows_spent_attributes_the_rail_that_expired_first(
    tmp_path, monkeypatch, earlier, final_redial_done,
):
    """When a step finds BOTH windows already spent (a process stall inside a
    sleep overshoots them), the detail names the rail that expired EARLIER by
    signed lateness — clamping first would erase the ordering — on both exits:
    the plain exhaustion check and the post-final-redial check."""
    clock = _FakeClock(monkeypatch)
    bound_late, deadline_late = (40.0, 5.0) if earlier == "bound" else (5.0, 40.0)
    ctx = _ctx(is_direct_chat=True)
    ctx.task_metadata = {
        "deadline_at": (
            datetime.now(timezone.utc)
            + timedelta(seconds=get_finalization_grace_sec() - deadline_late)
        ).isoformat(),
    }
    episode = loop_transport.TransportWaitEpisode(
        started_monotonic=clock.now - (60.0 + bound_late), interactive=True,
        wait_bound_sec=60.0, final_redial_done=final_redial_done,
    )
    notes = _NoteRecorder()

    assert _step(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes) is False
    assert clock.sleeps == []
    detail = _read_network_wait_events(tmp_path)[-1]["detail"]
    if earlier == "bound":
        assert detail == INTERACTIVE_DETAIL
    else:
        assert detail == ("deadline_after_final_redial" if final_redial_done else "deadline_exhausted")


@pytest.mark.parametrize("case", ["bound_earlier", "deadline_earlier", "managed"])
def test_deadline_refused_redial_uses_the_same_signed_attribution(tmp_path, monkeypatch, case):
    """A redial refused by the deadline admission gate ends the episode under
    the same attribution rule as every other exit: with both windows spent,
    the rail that expired earlier names the detail (the refusal itself stays
    visible in the loop's `llm_not_dispatched` row); a managed episode has no
    bound and keeps `deadline_refused_dispatch` with no exhaustion note."""
    clock = _FakeClock(monkeypatch)
    bound_late, deadline_late = {
        "bound_earlier": (60.0, 5.0), "deadline_earlier": (5.0, 60.0), "managed": (None, 5.0),
    }[case]
    ctx = _ctx() if case == "managed" else _ctx(is_direct_chat=True)
    ctx.task_metadata = {
        "deadline_at": (
            datetime.now(timezone.utc)
            + timedelta(seconds=get_finalization_grace_sec() - deadline_late)
        ).isoformat(),
    }
    if bound_late is None:
        episode = loop_transport.TransportWaitEpisode(started_monotonic=clock.now - 1000.0)
    else:
        episode = loop_transport.TransportWaitEpisode(
            started_monotonic=clock.now - (60.0 + bound_late), interactive=True,
            wait_bound_sec=60.0,
        )
    notes = _NoteRecorder()

    assert _step(
        episode, tmp_path, SimpleNamespace(_ctx=ctx), notes, error_kind="deadline_exhausted",
    ) is False
    assert clock.sleeps == []
    detail = _read_network_wait_events(tmp_path)[-1]["detail"]
    assert detail == {
        "bound_earlier": INTERACTIVE_DETAIL,
        "deadline_earlier": "deadline_refused_dispatch",
        "managed": "deadline_refused_dispatch",
    }[case]
    if case == "managed":
        assert notes.texts == []  # a managed exhaustion is its terminal result, not a note
    else:
        assert "this turn ends as an outage" in notes.texts[-1]


# ------------------------------------------- final free redial at the bound

def test_final_free_redial_reserves_the_margin_at_the_interactive_bound(tmp_path, monkeypatch):
    """The last grant before the interactive bound closes sleeps to remaining
    minus the named margin (round-top overhead eats ~1 s), and the next step
    terminalizes with the interactive detail — the same Q14 shape the owner
    deadline has."""
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    episode = loop_transport.TransportWaitEpisode(
        started_monotonic=clock.now, interactive=True, wait_bound_sec=60.0,
        wait_iterations=10,  # backoff already at the cap
    )
    clock.now += 40.0  # 20 s of the bound left
    tools = SimpleNamespace(_ctx=_ctx(is_direct_chat=True))
    notes = _NoteRecorder()

    assert _step(episode, tmp_path, tools, notes) is True
    assert episode.final_redial_done is True
    assert clock.sleeps == [pytest.approx(20.0 - loop_transport._FINAL_REDIAL_MARGIN_SEC)]
    assert _step(episode, tmp_path, tools, notes) is False
    events = _read_network_wait_events(tmp_path)
    assert events[-1]["detail"] == INTERACTIVE_DETAIL
    assert events[-1]["elapsed_sec"] == pytest.approx(60.0 - loop_transport._FINAL_REDIAL_MARGIN_SEC)


# --------------------------------------------------- managed rails untouched

def test_managed_task_without_deadline_keeps_no_local_bound(tmp_path, monkeypatch):
    """A managed task's wait has no loop-local ceiling: only its existing rails
    (deadline, budget, Stop, absolute ceiling) bound it, however long the idle
    timeout is — its waiting notes keep that rail alive — and its notes carry
    no incident toast (they render as live-card rows)."""
    clock = _FakeClock(monkeypatch)
    notes = _NoteRecorder()
    episode = _enter(tmp_path, _ctx(), notes)

    assert episode.interactive is False
    assert episode.wait_bound_sec is None
    clock.now += 100_000.0  # far beyond any idle timeout
    episode.wait_iterations = 10
    assert _step(episode, tmp_path, SimpleNamespace(_ctx=_ctx()), notes) is True
    waiting = [row for row in _read_network_wait_events(tmp_path) if row["phase"] == "waiting"]
    assert waiting and "window_remaining_sec" not in waiting[-1]
    assert len(notes.texts) == 2  # entry + the periodic note
    assert notes.incidents == [None, None]


def test_finalize_now_terminal_threads_the_interactive_facts(tmp_path, monkeypatch):
    """A finalize_now landing mid-episode composes the terminal from the
    episode's own facts: the turn class and the wall time it actually waited."""
    clock = _FakeClock(monkeypatch)
    episode = loop_transport.TransportWaitEpisode(
        started_monotonic=clock.now, interactive=True, wait_bound_sec=900.0,
        wait_iterations=2, redials=2,
    )
    clock.now += 30.0
    seen = {}

    def _terminal(**kwargs):
        seen.update(kwargs)
        return "", {}, {}

    loop_transport.finalize_now_transport_terminal(
        episode, drive_logs=tmp_path, task_id="t", model="m",
        handle_provider_unavailable=_terminal,
    )
    assert seen["interactive"] is True
    assert seen["waited_sec"] == pytest.approx(30.0)
    assert seen["wait_cause"] == "transport_unavailable"
    assert _read_network_wait_events(tmp_path)[-1]["detail"] == "finalize_now"


# ------------------------------------------------ direct-turn mailbox wake

def test_direct_turn_mailbox_message_wakes_the_sleep_and_reaches_the_round_top(tmp_path, monkeypatch):
    """A direct turn keeps accepting owner mailbox messages while it waits: the
    episode's wake check sees the message before sleeping, the round top
    delivers it into the transcript, and the free redial carries it. The
    message is written from inside the first failing dispatch, so it lands
    after that round's drain and before the episode's first wait whatever the
    host's speed (a timer raced a cold process's setup)."""
    from ouroboros.owner_mailbox import write_owner_message

    seen = []

    def fake_call(_llm, messages, _model, _tools, _effort, _max_retries, _drive_logs,
                  _task_id, _round_idx, _event_queue, accumulated_usage, *_a, **_k):
        seen.append(json.dumps(messages))
        if len(seen) == 1:
            write_owner_message(tmp_path, "also check the brakes", "t-wait")
            accumulated_usage["_last_llm_error_kind"] = "transport_unavailable"
            return None, 0.0
        accumulated_usage.pop("_last_llm_error_kind", None)
        return {"role": "assistant", "content": "done"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.is_direct_chat = True
    start = time.monotonic()
    result, usage, _trace = run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    elapsed = time.monotonic() - start

    assert result == "done"
    assert usage.get("reason_code") is None
    assert elapsed < 3.5  # the wake check saw the message before the 4 s backoff: no sleep ran
    assert len(seen) == 2
    assert "also check the brakes" not in seen[0]
    assert "also check the brakes" in seen[1]


# --------------------------------------------- owner notes and the toast seam

def test_interactive_notes_promise_no_cancellation_and_the_managed_note_still_does(tmp_path, monkeypatch):
    """Direct turns retain their plain wait notes and their separate Stop
    control; a managed task keeps the cancellation promise in its notes."""
    clock = _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    managed = _NoteRecorder()
    _enter(tmp_path, _ctx(), managed, task_id="t-managed")
    assert managed.texts[0].endswith("Stop cancels.")
    assert managed.incidents == [None]

    for flag in ("is_direct_chat",):
        notes = _NoteRecorder()
        ctx = _ctx(**{flag: True})
        episode = _enter(tmp_path, ctx, notes, task_id=f"t-{flag}")
        episode.last_note_monotonic = clock.now - 10_000.0  # force a periodic note
        _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes)

        assert len(notes.texts) >= 3  # entry, at least one periodic, exhaustion
        assert all("cancel" not in text.lower() for text in notes.texts)
        assert "waiting and redialing automatically" in notes.texts[0]
        assert "this turn ends as an outage" in notes.texts[-1]
        assert all("chat turn" not in text for text in notes.texts)


def test_direct_turn_waits_with_plain_notes_and_no_toast_pair(tmp_path, monkeypatch):
    """A direct turn (owner chat or Presence) is interactive for the wait
    bound but its progress rows render on the live card, so none of its notes
    — entry, periodic, exhaustion — carries the incident pair."""
    _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    ctx = _ctx(is_direct_chat=True)
    notes = _NoteRecorder()
    episode = _enter(tmp_path, ctx, notes, task_id="direct1")

    assert episode.interactive is True
    assert episode.wait_bound_sec == 60.0
    _run_until_terminal(episode, tmp_path, SimpleNamespace(_ctx=ctx), notes, task_id="direct1")
    assert len(notes.texts) >= 2
    assert notes.incidents == [None] * len(notes.texts)
    assert _read_network_wait_events(tmp_path)[-1]["detail"] == INTERACTIVE_DETAIL


@pytest.mark.parametrize("flags", [{}, {"is_direct_chat": True}])
def test_error_kind_change_closure_is_an_interactive_note(tmp_path, monkeypatch, flags):
    """A redial that reaches the provider and fails differently closes the
    episode with a durable row for every episode; the owner note naming the
    fresh kind is an interactive turn's only closure surface, so a managed
    episode gets none."""
    _FakeClock(monkeypatch)
    ctx = _ctx(**flags)
    notes = _NoteRecorder()
    episode = _enter(tmp_path, ctx, notes, task_id="t-kind")
    assert loop_transport.reconcile_transport_wait(
        episode, ctx, msg_present=False, error_kind="provider_transient",
        drive_logs=tmp_path, task_id="t-kind", model="m", emit_progress=notes,
    ) is None

    assert _read_network_wait_events(tmp_path)[-1]["detail"] == "error_kind_changed:provider_transient"
    if not flags:
        assert notes.texts == [BASE_MANAGED_ENTRY]  # the managed closure is its row, not a note
        return
    assert "got past the connect phase and failed as provider_transient" in notes.texts[-1]
    assert "ordinary failure policy resumes" in notes.texts[-1]
    incident = notes.incidents[-1]
    assert incident is None


@pytest.mark.parametrize("closure", ["recovered", "error_kind_changed"])
def test_managed_episode_owner_texts_are_byte_identical_to_base(tmp_path, monkeypatch, closure):
    """The managed wordings are a frozen contract: entry still ends with the
    Stop promise its cancel authority honors, recovery still says resuming,
    and the other closure writes its durable row and no note at all —
    the literal texts the base emits, nothing more."""
    clock = _FakeClock(monkeypatch)
    ctx = _ctx()
    notes = _NoteRecorder()
    episode = _enter(tmp_path, ctx, notes, task_id="t-managed")
    clock.now += 90.0
    outcome = {
        "recovered": dict(msg_present=True, error_kind=""),
        "error_kind_changed": dict(msg_present=False, error_kind="provider_transient"),
    }[closure]
    assert loop_transport.reconcile_transport_wait(
        episode, ctx, drive_logs=tmp_path, task_id="t-managed", model="m", emit_progress=notes, **outcome,
    ) is None

    expected = [BASE_MANAGED_ENTRY] + ([BASE_MANAGED_RECOVERY] if closure == "recovered" else [])
    assert notes.texts == expected
    assert notes.incidents == [None] * len(expected)
    last = _read_network_wait_events(tmp_path)[-1]
    assert (last["phase"], last.get("detail")) == {
        "recovered": ("recovered", None),
        "error_kind_changed": ("ended", "error_kind_changed:provider_transient"),
    }[closure]


def test_agent_progress_seam_projects_an_explicit_incident():
    """Explicit incidents retain their typed projection; ordinary notes stay plain."""
    from ouroboros.agent import OuroborosAgent

    events = queue.Queue()
    agent = SimpleNamespace(
        _last_progress_ts=None, _event_queue=events, _current_chat_id=7,
        _current_task_id="eph1",
        tools=SimpleNamespace(_ctx=SimpleNamespace(is_direct_chat=True)),
        _subagent_progress_meta=lambda _event: {},
    )
    OuroborosAgent._emit_progress(
        agent, "waiting",
        incident={"task_incident": "network_wait", "toast_once": "eph1:network_wait:entered:1"},
    )
    event = events.get_nowait()
    assert event["is_progress"] is True
    assert event["task_id"] == "eph1"
    assert event["progress_meta"] == {
        "task_incident": "network_wait",
        "toast_once": "eph1:network_wait:entered:1",
        # A transport-wait note is the host talking about the turn, so the card
        # shows the row and keeps its title; the toast pair is unaffected.
        "narration": False,
    }
    OuroborosAgent._emit_progress(agent, "plain note")
    assert "task_incident" not in events.get_nowait().get("progress_meta", {})


def test_direct_turn_end_to_end_keeps_plain_wait_notes(tmp_path, monkeypatch):
    """The native turn waits through the real round gate and keeps ordinary notes."""
    fake_call, _calls = _transport_failing_call(fail_times=99)
    _FakeClock(monkeypatch)
    monkeypatch.setattr(loop_transport, "get_task_idle_timeout_sec", lambda: 60)
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.delenv("USE_LOCAL_FALLBACK", raising=False)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.is_direct_chat = True
    notes = _NoteRecorder()
    kwargs = _loop_kwargs(tmp_path, registry, [])
    kwargs["emit_progress"] = notes
    result, usage, _trace = run_llm_loop(**kwargs)

    assert notes.texts and all(inc is None for inc in notes.incidents)
    assert usage.get("reason_code") == "provider_unavailable"
    assert "this turn waited and redialed for" in result
