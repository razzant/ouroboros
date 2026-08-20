"""The wait window: what it may promise, how it polls, and what the human sees.

Split verbatim out of ``tests/test_delegated_subagent_transport.py`` by theme. This
module owns the configured wait ceiling and its clamp against the nanny's own
deadline, the bound every read the wait issues carries, and the live stream that keeps
flowing to the human while the model waits.
"""

from __future__ import annotations

import datetime
import json

import pytest

from ouroboros.gateways import claudexor as cx

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _owned_gateway_uses_each_test_transport,
    _StreamingStub,
    _delegating_ctx,
    _isolation_stub,
    _nanny_ctx,
    _write_attempt,
)


def test_the_configured_wait_ceiling_cannot_promise_more_than_the_tool_can_serve():
    """`OUROBOROS_DELEGATE_WAIT_MAX_SEC` accepted up to 86,400 while `delegate_wait`'s
    own per-call executor timeout is 2100 and the tool is neither per-call-timeout
    configurable nor deadline-clamped — so everything above the window max bought a
    KILLED tool call instead of the graceful typed no-progress return the wait
    exists to give. F5 (grok blocking): the window max is a HARD 1800, decoupled
    from the ToolEntry timeout, and the whole chain is STRICT —
    window (1800) < tool-kill (2100) < lease absolute ceiling (2400) — so a full
    window plus its teardown always fits under the executor timeout, and the
    executor timeout always fits under the idle-rail lease."""
    import os

    from ouroboros.config import (
        DELEGATE_WAIT_CEILING_SEC,
        DELEGATE_WAIT_WINDOW_MAX_SEC,
        get_delegate_wait_max_sec,
    )
    from ouroboros.delegate_progress import EXTERNAL_WAIT_LEASE_CEILING_SEC
    from ouroboros.loop_tool_execution import _DEADLINE_CLAMPED_TOOLS, _PER_CALL_TIMEOUT_TOOLS
    from ouroboros.tools.delegate import get_tools

    entry = next(e for e in get_tools() if e.schema["name"] == "delegate_wait")
    assert DELEGATE_WAIT_CEILING_SEC == entry.timeout_sec
    # The strict inequality chain, pinned by value so no member can drift onto
    # another: a window EQUAL to the executor timeout has zero teardown margin.
    assert DELEGATE_WAIT_WINDOW_MAX_SEC < DELEGATE_WAIT_CEILING_SEC < EXTERNAL_WAIT_LEASE_CEILING_SEC
    assert (DELEGATE_WAIT_WINDOW_MAX_SEC, DELEGATE_WAIT_CEILING_SEC,
            EXTERNAL_WAIT_LEASE_CEILING_SEC) == (1800, 2100, 2400)
    # ...and neither escape hatch applies to this tool, which is why the ToolEntry
    # value really is the bound. The task deadline is a separate concern and is
    # honoured INSIDE the tool (see the wait-window test below), which is why the
    # outer clamp still must not apply: it would thread-kill the graceful return.
    assert "delegate_wait" not in _PER_CALL_TIMEOUT_TOOLS
    assert "delegate_wait" not in _DEADLINE_CLAMPED_TOOLS

    previous = os.environ.get("OUROBOROS_DELEGATE_WAIT_MAX_SEC")
    os.environ["OUROBOROS_DELEGATE_WAIT_MAX_SEC"] = "7200"
    try:
        # The configurable max clamps to the hard window max — NOT to the
        # ToolEntry timeout: raising the executor timeout must never silently
        # widen the askable window.
        assert get_delegate_wait_max_sec() == DELEGATE_WAIT_WINDOW_MAX_SEC
    finally:
        if previous is None:
            os.environ.pop("OUROBOROS_DELEGATE_WAIT_MAX_SEC", None)
        else:
            os.environ["OUROBOROS_DELEGATE_WAIT_MAX_SEC"] = previous


def _wait_against_a_live_run(ctx, tmp_path, monkeypatch, *, wait_sec):
    """Drive `_delegate_wait` against a run that stays RUNNING and never advances its
    cursor, so the WINDOW itself is the only thing that can end the wait. Returns
    (payload, elapsed_sec)."""
    import time

    import ouroboros.gateways.claudexor as gw
    import ouroboros.tools.delegate as delegate

    run_dir = tmp_path / "rundir"
    run_dir.mkdir(exist_ok=True)

    class _AliveStub:
        def handshake(self, **_kw): return {"compatible": True, "protocolMajor": 3}

        def get_run(self, rid, *, timeout_sec=None):
            return {"lastSeq": 0, "summary": {
                "state": "running", "effectiveAccess": "workspace_write",
                "runDir": str(run_dir),
            }}

        def close(self): pass

    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _AliveStub())
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    try:
        started = time.monotonic()
        out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=wait_sec, since_seq=0))
        return out, time.monotonic() - started
    finally:
        delegate._CUSTODY.clear()


def test_wait_payload_carries_elapsed_and_cap_facts(tmp_path, monkeypatch):
    """Nanny facts against premature cancels: the wait payload states how long
    the run has ACTUALLY been going and what its cap really is, from the durable
    start row — a nanny that cannot see these confabulated "exceeded the cap" at
    153s of a 180s run and cancelled, discarding the whole spend. Facts only:
    no auto-timeout, no threshold."""
    from ouroboros import delegate_custody as custody

    ctx = _delegating_ctx(tmp_path, acting=True)
    drive = custody.custody_root(ctx)
    assert custody.emit(drive, custody.STARTED, {
        "run_id": "run-1", "task_id": "t-nanny", "route": "some-route",
        "model": "m", "max_seconds": 180,
    })

    out, _ = _wait_against_a_live_run(ctx, tmp_path, monkeypatch, wait_sec=1)

    assert out["status"] == "no_progress", out
    assert out["max_seconds"] == 180
    assert isinstance(out["elapsed_seconds"], int) and out["elapsed_seconds"] >= 0


def test_wait_payload_facts_stay_null_for_a_row_that_predates_them(tmp_path, monkeypatch):
    """Absent facts stay absent: a run whose STARTED row predates `max_seconds`
    (or whose row never landed) reports nulls, never invented numbers."""
    ctx = _delegating_ctx(tmp_path, acting=True)

    out, _ = _wait_against_a_live_run(ctx, tmp_path, monkeypatch, wait_sec=1)

    assert out["status"] == "no_progress", out
    assert out["elapsed_seconds"] is None
    assert out["max_seconds"] is None


def test_the_wait_window_never_outlives_the_nannys_own_deadline(tmp_path, monkeypatch):
    """`delegate_wait` is deliberately NOT in `_DEADLINE_CLAMPED_TOOLS`, so nothing
    upstream cuts it: measured, a 2100s window with ten seconds of task deadline left
    kept the full 2100s outer timeout while `web_search` was clamped to 1s, and a real
    call with an 8s window against a 2s deadline returned after 8.0s — the task slid
    six seconds past its own deadline mid-tool, the exact defect that clamp exists for.

    The bound belongs HERE rather than in that set: the outer clamp is a thread-kill,
    while the wait's whole contract is the graceful typed `no_progress` return. So the
    window narrows to the remaining deadline and the caller still gets its answer, in
    time to finalize.

    The deadline is set ABOVE the finalization reserve on purpose. With a 3s deadline the
    reserve subtraction drives the window to the `max(1, …)` FLOOR, and a `<= 3` assertion
    then holds for a reason that has nothing to do with the clamp — the arithmetic being
    pinned here would be untested. The floor is exercised separately at the end."""
    from ouroboros.deadline_utils import deadline_remaining_sec
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    ctx = _delegating_ctx(tmp_path, acting=True)
    reserve = int(effective_finalization_reserve_sec(ctx))
    ctx.task_metadata = dict(ctx.task_metadata or {})
    ctx.task_metadata["deadline_at"] = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=reserve + 5)).isoformat()

    out, elapsed = _wait_against_a_live_run(ctx, tmp_path, monkeypatch, wait_sec=8)

    # It waited the DEADLINE (minus the reserve), not the asked-for window: strictly
    # under the 8s asked for AND strictly over the 1s floor, so this measures the clamp
    # itself. It also came back BEFORE the deadline rather than being killed after it.
    assert out["status"] == "no_progress", out
    assert 1 < out["waited_sec"] <= 5, out
    assert elapsed < 6.0, elapsed
    # The clamp's ARITHMETIC is the `waited_sec <= 5` line above: the granted window
    # never targets the grace. This wall-clock line is the smoke over it, and it gets
    # one second of tolerance: between stamping the deadline and returning, the runner
    # itself spends time (imports, stub spawn, polls) — windows-latest measured 10.6ms
    # PAST the exact boundary on a run whose twin had passed, i.e. the strict `>` was
    # racing runner speed, not pinning the clamp.
    assert deadline_remaining_sec(ctx) > reserve - 1.0,         "the wait blew past the finalization grace by more than runner overhead"

    # The floor, kept from the original scenario: a deadline SHORTER than the reserve
    # still yields a positive window and a graceful typed return, never 0 or negative.
    ctx.task_metadata["deadline_at"] = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=3)).isoformat()
    out, elapsed = _wait_against_a_live_run(ctx, tmp_path, monkeypatch, wait_sec=8)
    assert out["status"] == "no_progress" and out["waited_sec"] == 1, out
    assert elapsed < 3.0, elapsed
    assert deadline_remaining_sec(ctx) > 0, "the wait ate the whole remaining deadline"


def test_a_wait_with_no_deadline_keeps_the_full_window_it_asked_for(tmp_path, monkeypatch):
    """The clamp is NARROW-ONLY, and `deadline_remaining_sec` answers 0.0 both for "no
    deadline" and for "the deadline is behind us" — so a clamp that skipped the
    positive-remaining guard would shrink EVERY ordinary deadline-less wait to one
    second. This is the control that keeps that path byte-identical."""
    from ouroboros.deadline_utils import deadline_remaining_sec

    ctx = _delegating_ctx(tmp_path, acting=True)
    assert deadline_remaining_sec(ctx) == 0.0, ctx.task_metadata

    out, _ = _wait_against_a_live_run(ctx, tmp_path, monkeypatch, wait_sec=2)
    assert out["status"] == "no_progress" and out["waited_sec"] == 2, out


def test_the_wait_leaves_the_grace_it_needs_to_answer_at_all(tmp_path, monkeypatch):
    """The clamp now targets the remaining deadline MINUS the finalization grace, for
    the reason `_deadline_clamped_timeout` subtracts it for the network tools. While the
    wait returned on the first advance this never bit, because a busy run came back in
    seconds; now that the window is really held, aiming at the whole remaining deadline
    means routinely returning at the instant there is no time left to emit an answer."""
    from ouroboros.config import get_finalization_grace_sec

    grace = int(get_finalization_grace_sec())
    ctx = _delegating_ctx(tmp_path, acting=True)
    ctx.task_metadata = dict(ctx.task_metadata or {})
    ctx.task_metadata["deadline_at"] = (
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(seconds=grace + 4)).isoformat()

    out, elapsed = _wait_against_a_live_run(ctx, tmp_path, monkeypatch, wait_sec=8)

    # Without the reserve it would have held the asked-for 8s and come back with only
    # the grace period left; with it, the window is what remains ABOVE the grace.
    assert out["waited_sec"] <= 4, out
    assert elapsed < 6.0, elapsed


@pytest.mark.parametrize("seconds_left, ask, expected_window", [
    pytest.param(lambda reserve: 0.5, 8, 1, id="half_a_second_left_is_still_a_deadline"),
    pytest.param(lambda reserve: -5.0, 8, 1, id="a_deadline_already_behind_us"),
    pytest.param(None, 2, 2, id="no_deadline_at_all_keeps_the_whole_window"),
    pytest.param(lambda reserve: reserve + 60, 2, 2, id="a_comfortable_remainder_binds_on_the_ask"),
])
def test_the_clamp_keys_on_whether_a_deadline_exists_not_on_int_of_what_is_left(
        tmp_path, monkeypatch, seconds_left, ask, expected_window):
    """The clamp above is only as good as the question it asks, and it used to ask
    ``int(deadline_remaining_sec(ctx)) > 0`` — a test for "is there time left" that two
    real shapes answer with a flat NO, and both of them are the shapes where the clamp
    matters MOST:

    * half a second of deadline left truncates to ``int(0.5) == 0``, and
    * a deadline already spent leaves a NEGATIVE remainder.

    In both, the clamp was skipped entirely and the wait held its whole asked-for window
    — up to the 1800s ceiling — while the task it belongs to was already out of time.
    A wait that outlives its task's deadline is exactly the defect the clamp exists for,
    and it was open in the last instant before the deadline and every instant after.

    What separates those from the one case that legitimately takes the full window is
    not the SIZE of the remainder but whether a deadline EXISTS at all:
    ``deadline_remaining_sec`` answers a flat 0.0 for "no deadline set", so only
    ``parse_deadline_ts`` on the metadata can tell "nothing to obey" from "nothing left".
    Both spent shapes land on the ``max(1, …)`` floor and still return the graceful typed
    payload rather than being killed mid-tool; the deadline-less wait still gets what it
    asked for, and so does a wait with room to spare (the clamp NARROWS, it never
    lengthens and never shrinks a window that fits)."""
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    ctx = _delegating_ctx(tmp_path, acting=True)
    if seconds_left is not None:
        ctx.task_metadata = dict(ctx.task_metadata or {})
        ctx.task_metadata["deadline_at"] = (
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                seconds=seconds_left(float(effective_finalization_reserve_sec(ctx))))
        ).isoformat()

    out, elapsed = _wait_against_a_live_run(ctx, tmp_path, monkeypatch, wait_sec=ask)

    assert out["status"] == "no_progress", out
    assert out["waited_sec"] == expected_window, out
    # `waited_sec` is what the payload CLAIMS; the wall clock is what the task actually
    # spent, and the seconds held past a spent deadline are what the caller pays for.
    assert elapsed < expected_window + 2.0, elapsed


# -- the window is a WINDOW: the timer waits, the human's stream does not ------


class _FinishesOnTheSecondPoll(_StreamingStub):
    """A streaming run that goes terminal on its SECOND answer — so which poll the wait
    does or does not issue decides whether the model is told the run is over."""

    def get_run(self, rid, *, timeout_sec=None):
        detail = super().get_run(rid, timeout_sec=timeout_sec)
        if self.seq >= 2:
            detail["summary"]["state"] = "succeeded"
            detail["summary"]["spendUsd"] = 0.0
            detail["primaryOutput"] = "done"
        return detail


def _wait_against_a_streaming_run(ctx, tmp_path, monkeypatch, *, wait_sec, stub=None, since_seq=0):
    """Drive `_delegate_wait` against a run that keeps advancing. Returns
    (payload, elapsed_sec)."""
    import time

    import ouroboros.gateways.claudexor as gw
    import ouroboros.tools.delegate as delegate

    stub = stub if stub is not None else _StreamingStub()
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: stub)
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-a", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    try:
        started = time.monotonic()
        out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=wait_sec,
                                                 since_seq=since_seq))
        return out, time.monotonic() - started
    finally:
        delegate._CUSTODY.clear()


def test_a_streaming_run_no_longer_wakes_the_model_per_event_batch(tmp_path, monkeypatch):
    """THE defect: the advance check sat BEFORE the deadline check, so the only path that
    ever consulted the caller's window was the SILENT one. A run that streams — which is
    what a healthy Claudexor run does — tripped it on the very first poll, the loop never
    reached its sleep, and `wait_sec` bought nothing. Measured on one real task: 18 nanny
    rounds, a 3s median gap, 861,177 prompt tokens and $0.39 spent narrating a run that
    was doing fine.

    The window is now held, and the advances arrive as a SEQUENCE rather than one wake-up
    each."""
    out, elapsed = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch, wait_sec=4)

    assert elapsed >= 3.5, f"the wait returned early on an advance: {elapsed}s"
    assert out["status"] == "progress", out
    assert out["waited_sec"] == 4, out
    seqs = [row["seq"] for row in out["advances"]]
    assert len(seqs) >= 2, out["advances"]
    assert seqs == sorted(set(seqs)), seqs
    assert out["last_seq"] == seqs[-1], out
    assert isinstance(out["quiet_for_sec"], int)


def test_the_human_keeps_the_live_stream_while_the_model_waits(tmp_path, monkeypatch):
    """The owner's binding correction: hold the TIMER, never the stream. Every advance
    reaches the live progress surface the instant the loop sees it — the human's view of
    the delegated run gets richer (the harness's own event titles, at observation time,
    instead of the nanny's paraphrase one round later), and it is the same frame the
    supervisor's idle enforcer stamps `last_progress_at` from."""
    import time

    emitted = []
    ctx = _nanny_ctx(tmp_path)
    ctx.emit_progress_fn = lambda text: emitted.append((text, time.monotonic()))

    started = time.monotonic()
    out, _ = _wait_against_a_streaming_run(ctx, tmp_path, monkeypatch, wait_sec=4)
    returned = time.monotonic()

    assert len(emitted) == len(out["advances"]), (emitted, out["advances"])
    assert emitted[0][1] < started + 1.0, "the first advance must not wait for the window"
    # <=, not <: Windows's monotonic clock ticks at ~15.6ms, so an emit and the
    # return inside one tick read EQUAL — the invariant is observed-by-return.
    assert all(at <= returned for _, at in emitted)
    assert any("running tests" in text for text, _ in emitted), emitted
    assert all("run-1" in text for text, _ in emitted), emitted


def test_the_wait_adopts_the_standing_tail_before_it_starts_watching(tmp_path, monkeypatch):
    """A run does not go quiet while nobody is waiting on it, so the first `get_run` of a
    window routinely answers with a tail the caller has already been shown. Without
    adopting that tail as HISTORY, the window's first advance re-announced all of it as
    new session events — to the human's live stream and to the model alike. Here the
    daemon publishes four rows per cursor step, and the caller says it has read to the
    cursor the first poll returns: every advance must then carry the four rows that step
    actually added, never the eight rows standing on the timeline."""
    stub = _StreamingStub(batch=4, title="step")
    out, _ = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch, wait_sec=2, stub=stub, since_seq=1)

    assert out["advances"], out
    assert [len(row["events"]) for row in out["advances"]] == [4] * len(out["advances"]), \
        out["advances"]


def test_a_progress_emit_failure_never_aborts_the_wait(tmp_path, monkeypatch):
    """The progress channel is narration. A broken one must not abort a wait that is
    holding a live, possibly overpowered, run."""
    def _boom(_text):
        raise RuntimeError("progress channel is gone")

    ctx = _nanny_ctx(tmp_path)
    ctx.emit_progress_fn = _boom

    out, elapsed = _wait_against_a_streaming_run(ctx, tmp_path, monkeypatch, wait_sec=2)

    assert out["status"] == "progress", out
    assert elapsed >= 1.5, elapsed


def test_a_terminal_state_still_returns_immediately_mid_window(tmp_path, monkeypatch):
    """The terminal check runs FIRST on every poll, so holding the window costs nothing
    when the run finishes: a 120s window returns the moment the state goes terminal."""
    import ouroboros.delegate_custody as dc

    dc._CUSTODY.clear()
    out, elapsed = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch,
        wait_sec=120, stub=_FinishesOnTheSecondPoll())
    dc._CUSTODY.clear()

    assert out["status"] == "terminal", out
    assert elapsed < 10.0, elapsed


def test_a_containment_breach_still_halts_mid_window(tmp_path, monkeypatch):
    """The other early exit, pinned at a window where "immediate" and "at expiry" are
    actually distinguishable — the existing breach coverage all runs at wait_sec=1."""
    import time

    import ouroboros.tools.delegate as delegate

    run_dir = tmp_path / "run-1"
    home = tmp_path / "operator-home"
    home.mkdir()
    monkeypatch.setattr(cx, "operator_home", lambda: home)
    _isolation_stub(monkeypatch, run_dir=run_dir)
    _write_attempt(run_dir, isolated=False, home_dir=str(home))

    ctx = _delegating_ctx(tmp_path, acting=True)
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    started = time.monotonic()
    out = json.loads(delegate._delegate_wait(ctx, "run-1", wait_sec=120, since_seq=0))
    elapsed = time.monotonic() - started
    delegate._CUSTODY.clear()

    assert out["status"] == "refused" and out["reason"] == "home_isolation_not_applied", out
    assert elapsed < 10.0, elapsed


class _SlowPollStub(_StreamingStub):
    """A streaming daemon that is SLOW to answer, and records when each read STARTED.

    Every other stub in this file replies instantly, so the poll the loop issues after its
    window is already spent cost nothing and no test could see it. A real `get_run` carries
    a read timeout — the client's 60-second default, or whatever bound the caller passed —
    and that read is what the task pays for in wall clock past its own deadline. So this
    honours the bound the way httpx does: it RAISES at the bound rather than answering
    late, which is the only reason a bound is worth anything.
    """

    def __init__(self, *, read_sec=1.2, **kwargs):
        import time

        super().__init__(**kwargs)
        self.read_sec, self.reads, self._clock = read_sec, [], time.monotonic

    def get_run(self, rid, *, timeout_sec=None):
        import time

        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        self.reads.append(self._clock())
        bound = self.read_sec if timeout_sec is None else min(self.read_sec, float(timeout_sec))
        time.sleep(bound)
        if bound < self.read_sec:
            raise ClaudexorUnavailable("daemon_unreachable",
                                       "Claudexor daemon unreachable: ReadTimeout")
        return super().get_run(rid)


def test_every_poll_is_bounded_by_what_the_window_has_left(tmp_path, monkeypatch):
    """The bound belongs on EVERY poll, not just the one after the window is spent.

    A poll started a moment BEFORE expiry carries the client's own 60-second read
    default, so it can answer long after the window — and after the task deadline the
    clamp above exists to protect. Measured against an unbounded in-loop poll: 2.11s of
    wall for a window that reported `waited_sec=1`, with the deadline already crossed.
    The window is the ceiling for the transport too, so what each poll may ASK for is
    what the window still has (never below the floor a bound is useful at, and never
    above the read default the client would have used anyway — a bound that WIDENS the
    ask is not a bound)."""
    from ouroboros.delegate_progress import bounded_poll
    from ouroboros.gateways.claudexor import _READ_TIMEOUT_SEC, SHORT_POLL_TIMEOUT_SEC

    asked = []

    class _Recorder:
        def get_run(self, rid, *, timeout_sec=None):
            asked.append(timeout_sec)
            return {"lastSeq": 1, "summary": {"state": "running"}}

    gateway = _Recorder()
    bounded_poll(gateway, "run-1", 40.0)      # plenty of window left -> ask for it
    bounded_poll(gateway, "run-1", 0.5)       # nearly spent -> the floor, not 60s
    bounded_poll(gateway, "run-1", -3.0)      # spent -> the floor, still bounded
    assert asked == [40.0, SHORT_POLL_TIMEOUT_SEC, SHORT_POLL_TIMEOUT_SEC], asked
    assert all(value is not None for value in asked), \
        "an unbounded poll is the client's 60s default, which outlives any window"

    # The other direction, which a floor alone got backwards: a long window has MORE
    # than the client's own read default left, and `max()` handed that surplus to the
    # transport as the ask (measured: 1797.0 for a 1800s window). A hung daemon then
    # stopped failing at sixty seconds and held the whole window — reported afterwards
    # as a wait that saw nothing. A bound NARROWS in both directions or it is decoration.
    asked.clear()
    bounded_poll(gateway, "run-1", 1797.0)
    bounded_poll(gateway, "run-1", 61.0)
    assert asked == [_READ_TIMEOUT_SEC, _READ_TIMEOUT_SEC], \
        f"a bound above the client's own default grants a hung read MORE rope: {asked}"


class _BoundRecordingStub(_StreamingStub):
    """A healthy streaming daemon that records the BOUND every read was given.

    `_SlowPollStub` proves what a bound COSTS; this one proves each read HAS one, which
    is a fact about the call site rather than about the helper it calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bounds, self.handshake_bounds = [], []

    def handshake(self, **kwargs):
        self.handshake_bounds.append(kwargs.get("timeout_sec"))
        return super().handshake(**kwargs)

    def get_run(self, rid, *, timeout_sec=None):
        self.bounds.append(timeout_sec)
        return super().get_run(rid, timeout_sec=timeout_sec)


class _DiesAfter(_StreamingStub):
    """A streaming daemon that answers ``answers`` polls and then stops answering.

    Not SLOW — gone: a daemon that was restarted, started 503ing, or lost its socket
    while the wait was holding its window. The transport reports exactly that as the
    typed `ClaudexorUnavailable` this stub raises."""

    def __init__(self, *, answers=1, **kwargs):
        super().__init__(**kwargs)
        self.answers = answers

    def get_run(self, rid, *, timeout_sec=None):
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        detail = super().get_run(rid, timeout_sec=timeout_sec)
        if self.seq > self.answers:
            raise ClaudexorUnavailable(
                "daemon_unreachable", "Claudexor daemon unreachable: ConnectError: [Errno 61]")
        return detail


def test_every_read_the_wait_issues_carries_a_bound_and_the_window_is_it(tmp_path, monkeypatch):
    """The helper's arithmetic is pinned above; this pins WHICH READS GET IT.

    A test that calls `bounded_poll` directly says nothing about the call site, so the
    call site could go back to bounding only the poll after the window is spent —
    `progress.bounded_poll(gateway, rid, 0.0) if spent else gateway.get_run(rid)` — and
    186 tests stayed green while every in-window read carried the client's 60s default
    again. So the wait is driven end to end and every read it issues is inspected: the
    opening one, each in-loop one, and the last one after the window is spent."""
    import ouroboros.gateways.claudexor as gw
    import ouroboros.tools.delegate as delegate

    monkeypatch.setattr(gw, "SHORT_POLL_TIMEOUT_SEC", 0.4)   # so the floor is visible
    monkeypatch.setattr(delegate, "_POLL_INTERVAL_SEC", 1.0)  # several in-loop polls, cheaply
    window = 3
    stub = _BoundRecordingStub()

    out, _ = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch, wait_sec=window, stub=stub)

    assert out["status"] == "progress" and out["waited_sec"] == window, out
    assert len(stub.bounds) >= 3, stub.bounds
    assert all(value is not None for value in stub.bounds), \
        f"an unbounded read inherits the client's 60s default, which outlives the window: {stub.bounds}"
    assert stub.handshake_bounds and all(v is not None for v in stub.handshake_bounds), \
        f"the opening handshake is paid for out of the same window: {stub.handshake_bounds}"
    # The OPENING read is bounded by the whole window (the clock starts before the
    # connection), and every later one by what is left — so the sequence never rises.
    assert stub.bounds[0] <= window, stub.bounds
    assert stub.bounds == sorted(stub.bounds, reverse=True), stub.bounds
    assert all(gw.SHORT_POLL_TIMEOUT_SEC <= value <= window for value in stub.bounds), stub.bounds
    assert all(value <= gw._READ_TIMEOUT_SEC for value in stub.bounds), stub.bounds
    # ...and the last one, issued once the window is spent, sits on the floor.
    assert stub.bounds[-1] == pytest.approx(gw.SHORT_POLL_TIMEOUT_SEC), stub.bounds


def test_a_daemon_that_dies_mid_window_is_refused_not_reported_as_a_quiet_wait(
        tmp_path, monkeypatch):
    """A failing poll may only become an expiry once there is a window to expire.

    Merging the spent-window poll into the general one widened its `ClaudexorUnavailable`
    swallow from that ONE call to EVERY call, and the result is a fabrication rather than
    a gap: measured on a 1800s window whose daemon died three seconds in, the model was
    handed `status: progress`, `waited_sec: 1800` and "The run advanced 1 time(s) during
    the 1800s you asked to wait" after 3.0s of wall. Before any advance it reads worse —
    `no_progress` over a full window, with a note inviting a `delegate_cancel` of a live
    overpowered run because the transport blipped.

    A daemon that dies while the window still has time is the typed refusal it always
    was: the caller's own `except ClaudexorUnavailable` turns it into a `_fail`, and the
    nanny is told the transport is gone rather than told a story about a quiet run."""
    import ouroboros.tools.delegate as delegate

    monkeypatch.setattr(delegate, "_POLL_INTERVAL_SEC", 0.2)
    window = 600
    stub = _DiesAfter(answers=1)

    out, elapsed = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch, wait_sec=window, stub=stub)

    assert out["status"] == "refused", out
    assert out["reason"] == "daemon_unreachable", out
    assert out["run_id"] == "run-1", out
    # The duration is the tell: the wall clock is seconds, so any `waited_sec` at all
    # would be a window nobody waited, and the advance it saw is not a completed wait.
    assert "waited_sec" not in out and "advances" not in out, out
    assert elapsed < 30.0, elapsed


def test_a_daemon_that_dies_only_at_the_spent_window_poll_still_expires_gracefully(
        tmp_path, monkeypatch):
    """A failed final poll expires a spent window instead of refusing the wait, and preserves prior progress too."""
    from types import SimpleNamespace

    import ouroboros.tools.delegate as delegate

    clock = {"now": 0.0}

    def sleep(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(
        delegate,
        "time",
        SimpleNamespace(monotonic=lambda: clock["now"], sleep=sleep),
    )
    stub = _DiesAfter(answers=1)

    out, _ = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch, wait_sec=1, stub=stub)

    assert out["status"] == "progress", out
    assert out["waited_sec"] == 1, out
    assert stub.seq == 2, f"the spent window still owes its last poll: {stub.seq}"
    assert clock["now"] == pytest.approx(1.0)


def test_the_last_poll_of_a_spent_window_is_bounded_not_skipped(tmp_path, monkeypatch):
    """The window used to be checked only at the TOP of the loop, so the sleep that
    consumed the last of it was always followed by one more `gateway.get_run` — and that
    call is not free. It carries the gateway's 60s read default, so against a slow daemon
    the wall clock outran the deadline the clamp above exists to protect: measured, a call
    reporting `waited_sec=1` spent 3.42s, 1.92s past the deadline, mid-tool.

    Skipping that poll entirely was the wrong half of the trade (see the terminal case
    below). It is BOUNDED instead: still exactly one read past the window, but one whose
    cost is `SHORT_POLL_TIMEOUT_SEC`, not a minute — and a daemon that cannot answer
    inside the bound expires the window gracefully rather than failing the tool, which is
    the second stub here. The typed payload is unchanged either way; both expiry returns
    render through the same `_expired()`.

    The window is measured from BEFORE the connection (the opening handshake and first
    poll are part of what this call holds), so the ask here is comfortably longer than
    the opening read — otherwise the window is already spent when the daemon first
    answers and there is no second poll to bound, which is its own correct behaviour and
    a different case from this one.
    """
    import ouroboros.gateways.claudexor as gw

    stub = _SlowPollStub(read_sec=1.2)

    window = 3
    out, elapsed = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch, wait_sec=window, stub=stub)

    assert out["status"] == "progress" and out["waited_sec"] == window, out
    # The wait's own deadline: the window starts BEFORE the opening read, so it expires
    # `window` seconds after that read began. Exactly ONE read may start at or after it.
    assert stub.reads, "the wait never polled at all"
    deadline = stub.reads[0] + window
    assert len([at for at in stub.reads if at >= deadline]) == 1, \
        f"the window paid for more than one last poll: {stub.reads}, deadline {deadline}"
    assert len(stub.reads) == 2, stub.reads
    # ...and the wall clock the TASK pays stays within the BOUND of that deadline, rather
    # than within a 60s read of it.
    assert elapsed < window + gw.SHORT_POLL_TIMEOUT_SEC + 0.6, elapsed

    # A daemon slower than the bound: the read is cut AT the bound, and an expiry is what
    # the caller gets — never a transport refusal raised out of a tool holding a live run.
    monkeypatch.setattr(gw, "SHORT_POLL_TIMEOUT_SEC", 0.4)
    slower = _SlowPollStub(read_sec=1.2)
    out, elapsed = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch, wait_sec=window, stub=slower)

    assert out["status"] == "progress" and out["waited_sec"] == window, out
    assert len(slower.reads) == 2, slower.reads
    assert elapsed < window + 0.4 + 0.6, elapsed
    # The BOUND is what ended that read, not the daemon: it cost the bound, not the 1.2s
    # the daemon wanted — which is the whole of what a per-request timeout buys here.
    last_read_sec = (slower.reads[0] + elapsed) - slower.reads[1]
    assert last_read_sec < 0.4 + 0.3, last_read_sec


def test_a_run_that_finishes_during_the_last_sleep_is_reported_terminal(tmp_path, monkeypatch):
    """The cost of the OTHER half of that trade, and the reason the last poll is bounded
    rather than dropped. Returning straight after the sleep judged terminal state and
    containment breach on data read BEFORE it, so a run that succeeded during that sleep
    came back `status: progress`, `state: running`, with no settlement — and the model paid
    another full-context nanny round for a run that was already done, which is the exact
    cost this whole window exists to remove. At a window of 3s or less (the deadline
    clamp's own floor produces 1s) the second poll never happened at all.
    """
    import ouroboros.delegate_custody as dc

    dc._CUSTODY.clear()
    out, elapsed = _wait_against_a_streaming_run(
        _nanny_ctx(tmp_path), tmp_path, monkeypatch,
        wait_sec=1, stub=_FinishesOnTheSecondPoll())
    dc._CUSTODY.clear()

    assert out["status"] == "terminal", out
    assert out["state"] == "succeeded", out
    assert out["settlement"]["settled"] is True, out["settlement"]
    assert elapsed < 6.0, elapsed


def test_bounded_poll_retries_the_git_atomic_object_race_once():
    """The CI gate learned to tolerate the engine's transient Git atomic-object
    ENOENT (938094a9) while the production poll kept propagating it — so CI could
    pass on an engine whose live delegate_wait still failed. One immediate re-read,
    only for exactly that shape, only while the window has time left."""
    from ouroboros.delegate_progress import bounded_poll, is_transient_git_object_race

    class _RaceOnce:
        def __init__(self):
            self.calls = 0
        def get_run(self, run_id, timeout_sec=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError(
                    "ENOENT: no such file or directory, open "
                    "'/x/.git/objects/ab/tmp_obj_h4x'")
            return {"state": "running"}

    gw = _RaceOnce()
    assert bounded_poll(gw, "run-1", 60.0) == {"state": "running"}
    assert gw.calls == 2

    # A spent window does not retry (the expiring poll owns that path), and any
    # OTHER failure propagates untouched on the first read.
    gw2 = _RaceOnce()
    try:
        bounded_poll(gw2, "run-1", 0.0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised and gw2.calls == 1

    class _RealFailure:
        def get_run(self, run_id, timeout_sec=None):
            raise RuntimeError("ENOENT: no such file or directory, open '/x/data/config.json'")

    try:
        bounded_poll(_RealFailure(), "run-1", 60.0)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert not is_transient_git_object_race(RuntimeError("connection refused"))
