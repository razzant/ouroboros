"""The one read a ``delegate_wait`` wake makes of its run: what it may promise and how it is bounded.

``tools/delegate._read_run`` is ONE detail read under the transport's read bound, narrowed by
the nanny's own deadline. Its payload states the run's dated facts, a row with no recorded grant
is still held to the live derivation, and every read it issues carries a bound. The run's speech
is the body's (``delegate_feed``), never this read's.
"""

from __future__ import annotations

import datetime
import json
import pytest
from ouroboros.gateways import claudexor as cx

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _delegating_ctx,
    _isolation_stub,
    _nanny_ctx,
    _owned_gateway_uses_each_test_transport,
    _write_attempt,
)


class _AliveStub:
    """A run that stays RUNNING and never advances its cursor; it records the bound each call carried."""

    def __init__(self, run_dir):
        self.run_dir, self.read_bounds, self.handshake_bounds = run_dir, [], []

    def handshake(self, **kwargs):
        self.handshake_bounds.append(kwargs.get("timeout_sec"))
        return {"compatible": True, "protocolMajor": 3}

    def get_run(self, rid, *, timeout_sec=None):
        self.read_bounds.append(timeout_sec)
        return {"lastSeq": 0, "summary": {
            "state": "running", "effectiveAccess": "workspace_write", "runDir": str(self.run_dir)}}

    def close(self):
        pass


def _read_a_live_run(ctx, tmp_path, monkeypatch):
    """One wake's read of a run that stays RUNNING. Returns (payload, the stub that served it)."""
    import ouroboros.gateways.claudexor as gw
    import ouroboros.tools.delegate as delegate

    run_dir = tmp_path / "rundir"
    run_dir.mkdir(exist_ok=True)
    stub = _AliveStub(run_dir)
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: stub)
    delegate._CUSTODY.clear()
    delegate._CUSTODY["run-1"] = delegate._RunCustody(
        task_id="t-nanny", route_id="some-route", model="m",
        project_id="prj", project_owned=False,
    )
    try:
        return json.loads(delegate._read_run(ctx, "run-1", 0)), stub
    finally:
        delegate._CUSTODY.clear()


def test_a_wakes_read_carries_elapsed_and_cap_facts(tmp_path, monkeypatch):
    """Nanny facts against premature cancels: the payload states how long the run has
    ACTUALLY been going and what its cap really is, from the durable start row — a nanny
    that cannot see these confabulated "exceeded the cap" at 153s of a 180s run and
    cancelled, discarding the whole spend. Facts only: no auto-timeout, no threshold."""
    from ouroboros import delegate_custody as custody

    ctx = _delegating_ctx(tmp_path, acting=True)
    drive = custody.custody_root(ctx)
    assert custody.emit(drive, custody.STARTED, {
        "run_id": "run-1", "task_id": "t-nanny", "route": "some-route",
        "model": "m", "max_seconds": 180,
    })

    out, _ = _read_a_live_run(ctx, tmp_path, monkeypatch)

    assert out["status"] == "no_progress", out
    assert out["max_seconds"] == 180
    assert isinstance(out["elapsed_seconds"], int) and out["elapsed_seconds"] >= 0
    assert "waited_sec" not in out and "advances" not in out, "one read holds no window"


def test_a_wakes_read_facts_stay_null_for_a_row_that_predates_them(tmp_path, monkeypatch):
    """Absent facts stay absent: a run whose STARTED row predates `max_seconds`
    (or whose row never landed) reports nulls, never invented numbers."""
    ctx = _delegating_ctx(tmp_path, acting=True)

    out, _ = _read_a_live_run(ctx, tmp_path, monkeypatch)

    assert out["status"] == "no_progress", out
    assert out["elapsed_seconds"] is None
    assert out["max_seconds"] is None


@pytest.mark.parametrize("seconds_left, expected", [
    pytest.param(None, (60, 60), id="no_deadline_keeps_the_transport_bound"),
    pytest.param(lambda reserve: reserve + 30, (25, 30), id="a_deadline_narrows_the_bound_above_its_reserve"),
    pytest.param(lambda reserve: 0.5, (5, 5), id="half_a_second_left_is_still_a_deadline_on_the_floor"),
])
def test_the_read_never_outlives_the_nannys_own_deadline(tmp_path, monkeypatch, seconds_left, expected):
    """``delegate_wait`` is NOT in ``_DEADLINE_CLAMPED_TOOLS``: the outer clamp is a thread-kill, while the read's
    contract is a typed answer. So the read's own bound narrows to the remaining deadline minus the finalization
    reserve (``window_within_deadline``), floored at one second; only "no deadline set" keeps the transport's own
    read bound. Keyed on whether a deadline EXISTS, not on ``int`` of what is left: half a second and a spent
    deadline both land on the floor instead of the full bound."""
    from ouroboros.gateways.claudexor import _READ_TIMEOUT_SEC, SHORT_POLL_TIMEOUT_SEC
    from ouroboros.task_pacing import effective_finalization_reserve_sec

    ctx = _delegating_ctx(tmp_path, acting=True)
    if seconds_left is not None:
        ctx.task_metadata = dict(ctx.task_metadata or {})
        ctx.task_metadata["deadline_at"] = (
            datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                seconds=seconds_left(float(effective_finalization_reserve_sec(ctx))))
        ).isoformat()

    out, stub = _read_a_live_run(ctx, tmp_path, monkeypatch)

    assert out["status"] == "no_progress", out
    [bound] = stub.read_bounds
    assert (_READ_TIMEOUT_SEC, SHORT_POLL_TIMEOUT_SEC) == (60, 5), "the bounds this table is written in"
    assert expected[0] <= bound <= expected[1], (bound, expected)
    assert stub.handshake_bounds and all(value is not None for value in stub.handshake_bounds)


def test_a_containment_breach_of_a_row_without_a_grant_halts_the_read(tmp_path, monkeypatch):
    """A row with no recorded grant has only the read's live derivation to hold it to: an unconfined home under a
    writable profile is halted at once as a containment fault."""
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
    out = json.loads(delegate._read_run(ctx, "run-1", 0))
    delegate._CUSTODY.clear()

    assert out["status"] == "refused" and out["reason"] == "home_isolation_not_applied", out


def test_every_poll_is_bounded_by_what_its_caller_has_left(tmp_path, monkeypatch):
    """What each poll may ASK for is what its caller still has (never below the floor a
    bound is useful at, and never above the read default the client would have used
    anyway — a bound that WIDENS the ask is not a bound)."""
    from ouroboros.delegate_progress import bounded_poll
    from ouroboros.gateways.claudexor import _READ_TIMEOUT_SEC, SHORT_POLL_TIMEOUT_SEC

    asked = []

    class _Recorder:
        def get_run(self, rid, *, timeout_sec=None):
            asked.append(timeout_sec)
            return {"lastSeq": 1, "summary": {"state": "running"}}

    gateway = _Recorder()
    bounded_poll(gateway, "run-1", 40.0)      # plenty of time left -> ask for it
    bounded_poll(gateway, "run-1", 0.5)       # nearly spent -> the floor, not 60s
    bounded_poll(gateway, "run-1", -3.0)      # spent -> the floor, still bounded
    assert asked == [40.0, SHORT_POLL_TIMEOUT_SEC, SHORT_POLL_TIMEOUT_SEC], asked
    assert all(value is not None for value in asked), \
        "an unbounded poll is the client's 60s default, which outlives any deadline"

    # The direction a floor alone got backwards: max() handed a long remainder's
    # surplus to the transport ask (1797.0), so a hung daemon held all of it.
    # A bound NARROWS both directions or it is decoration.
    asked.clear()
    bounded_poll(gateway, "run-1", 1797.0)
    bounded_poll(gateway, "run-1", 61.0)
    assert asked == [_READ_TIMEOUT_SEC, _READ_TIMEOUT_SEC], \
        f"a bound above the client's own default grants a hung read MORE rope: {asked}"


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
