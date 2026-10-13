"""Truthful waiting D + E: the unsettled wait_task body and the one wait window.

D. A wait_task that returns BEFORE its child settled (timeout, mailbox interrupt,
beacon) carries the compact per-child projection wait_tasks already owns, plus the
child's own open delegated runs as dated observation facts read once from its
supervision record. A settled child, or a caller whose known hash no longer
matches, still gets the full single-child handoff.

E. wait_task, wait_tasks and await_messages share ``_wait_window``: under a deadline
the window stops a margin inside the executor's emit window (the finalization
reserve subtracted once), and each entry timeout stays at least ceiling + margin,
so a full window ends inside its kill timer instead of in TOOL_TIMEOUT.
"""

from __future__ import annotations

import datetime as _dt
import json
import pathlib
import time
from types import SimpleNamespace

import pytest

import ouroboros.delegate_custody as dc
import ouroboros.task_pacing as pacing
from ouroboros.deadline_utils import utc_now
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, write_task_result
from ouroboros.tools import control_task_results as mod

REPO = pathlib.Path(__file__).resolve().parents[1]
CHILD = "nannychild"


def _iso(seconds_ago: float) -> str:
    return (utc_now() - _dt.timedelta(seconds=seconds_ago)).isoformat()


def _seed_child(drive, *, status=STATUS_RUNNING, record=None, runs=("run-a",)):
    dc._CUSTODY.clear()
    write_task_result(drive, CHILD, status, result="partial answer",
                      trace_summary="step " * 50, parent_task_id="parent")
    for run_id in runs:
        dc.record_started(drive, dc.RunCustody(
            run_id=run_id, task_id=CHILD, route_id="codex", ledger_root=str(drive)),
            shape={"max_seconds": 1800})
    # A review panel's run on the same task is its panel's, never this child's leaf.
    dc.record_started(drive, dc.RunCustody(
        run_id="run-review", task_id=CHILD, route_id="codex", source="review_substrate",
        category="task_acceptance_review", ledger_root=str(drive)))
    if record is not None:
        path = drive / "state" / "delegate_supervision" / f"{CHILD}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record if isinstance(record, str) else json.dumps(record), encoding="utf-8")


def _live_record(**extra):
    record = {
        "schema": 1, "run_id": "run-a", "journal_cursor": 7, "status": "sleeping",
        "sleep_entry": {"run_id": "run-a", "entered_at": _iso(600)},
        "observation": {"at": _iso(3), "answered": True, "status": "progress", "run_state": "running",
                        "reason": ""},
        "last_answered_observation_at": _iso(3),
    }
    record.update(extra)
    return record


def _projection(text: str) -> dict:
    return json.loads(text)["tasks"][CHILD]


def _ctx(drive, **extra):
    return SimpleNamespace(drive_root=drive, task_id="parent", **extra)


# ---------------------------------------------------------------- D: the body


@pytest.mark.parametrize("early", [
    None,
    {"reason": "owner_mailbox_pending", "delivery": "pending_loop_drain"},
    {"reason": "child_attention_beacon", "beacons": [], "beacons_remaining": 0},
])
def test_an_unsettled_return_is_compact_with_dated_leaf_rows(tmp_path, monkeypatch, early):
    _seed_child(tmp_path, record=_live_record())
    real_wait = mod.wait_for_effective_tasks

    def _wait(root, ids, **kw):
        out = real_wait(root, ids, **{**kw, "on_poll": None})
        if early is not None:
            out["early_return"] = early
        return out

    monkeypatch.setattr(mod, "wait_for_effective_tasks", _wait)

    text = mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0)

    assert json.loads(text)["tasks"][CHILD]["status"] == "running"
    assert "[SUBTASK_OUTCOME]" not in text and "[SUBTASK_TRACE]" not in text
    projection = _projection(text)
    assert projection["status"] == "running" and projection["result"] == "partial answer"
    rows = projection["delegated_runs"]
    assert [row["run_id"] for row in rows] == ["run-a"], "review-owned runs are the panel's"
    assert rows[0]["max_seconds"] == 1800 and rows[0]["started_at"] != "unknown"
    supervision = rows[0]["supervision"]
    assert supervision["observed_run_state"] == "running" and supervision["observation_failure"] == ""
    assert supervision["journal_cursor"] == 7 and supervision["hold_entered_at"] != "unknown"
    assert 0 <= supervision["observation_age_sec"] < 60
    # Dated facts only: no written-at stamp and no liveness verdict of any spelling.
    flat = json.dumps(projection)
    for word in ("state_written_at", "alive", "liveness"):
        assert word not in flat


@pytest.mark.parametrize("record, reason", [
    (None, "no_supervision_record"),
    ("{not json", "unreadable"),
    (json.dumps(["a", "list"]), "unreadable"),
])
def test_a_missing_or_unreadable_record_is_the_fact_unknown(tmp_path, record, reason):
    _seed_child(tmp_path, record=record)

    rows = _projection(mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0))["delegated_runs"]

    assert rows[0]["supervision"] == {"state": "unknown", "reason": reason}


def test_record_facts_attach_only_to_the_run_they_name(tmp_path):
    _seed_child(tmp_path, record=_live_record(), runs=("run-a", "run-b"))

    rows = _projection(mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0))["delegated_runs"]

    by_run = {row["run_id"]: row["supervision"] for row in rows}
    assert by_run["run-b"] == {"state": "unknown", "reason": "record_names_another_run"}
    assert by_run["run-a"]["observed_run_state"] == "running"


def test_a_failed_observation_is_named_and_the_last_answer_keeps_its_date(tmp_path):
    answered_at = _iso(900)
    _seed_child(tmp_path, record=_live_record(
        observation={"at": _iso(2), "answered": False, "status": "observation_pending",
                     "reason": "daemon_unreachable"},
        last_answered_observation_at=answered_at))

    supervision = _projection(mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0))[
        "delegated_runs"][0]["supervision"]

    assert supervision["observation_failure"] == "daemon_unreachable"
    assert supervision["last_answered_observation_at"] == answered_at
    assert supervision["observation_age_sec"] >= 899


def test_a_record_without_observation_facts_says_unknown_not_fresh(tmp_path):
    """The supervisor rewrites its record on every quiet renewal, failed reads
    included, so the record's own freshness proves nothing about the leaf."""
    _seed_child(tmp_path, record={"schema": 1, "run_id": "run-a", "journal_cursor": 3,
                                  "status": "sleeping", "updated_at": _iso(1)})

    supervision = _projection(mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0))[
        "delegated_runs"][0]["supervision"]

    assert supervision["last_answered_observation_at"] == "unknown"
    assert supervision["observation_failure"] == "unknown"
    assert supervision["observed_run_state"] == "unknown" and supervision["observation_age_sec"] is None


def test_the_record_is_read_once_after_the_wait_without_retrying(tmp_path, monkeypatch):
    """Windows: a racing os.replace makes the read fail; that is a fact, not a
    loop, and no handle is open while the wait sleeps."""
    _seed_child(tmp_path, record=_live_record())
    order = []
    real_wait, real_read = mod.wait_for_effective_tasks, pathlib.Path.read_text

    def _wait(*args, **kw):
        order.append("wait")
        return real_wait(*args, **kw)

    def _read(self, *args, **kw):
        if self.name == f"{CHILD}.json" and self.parent.name == "delegate_supervision":
            order.append("read")
            raise PermissionError("sharing violation")
        return real_read(self, *args, **kw)

    monkeypatch.setattr(mod, "wait_for_effective_tasks", _wait)
    monkeypatch.setattr(pathlib.Path, "read_text", _read)

    rows = _projection(mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0))["delegated_runs"]

    assert order == ["wait", "read"]
    assert rows[0]["supervision"] == {"state": "unknown", "reason": "unreadable"}


def test_a_settled_child_keeps_the_full_handoff(tmp_path):
    _seed_child(tmp_path, status=STATUS_COMPLETED, record=_live_record())

    text = mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0)

    assert _projection(text)["status"] == STATUS_COMPLETED
    assert _projection(text)["result"] == "partial answer"


def test_a_changed_known_hash_returns_the_full_body_and_a_match_stays_compact(tmp_path):
    _seed_child(tmp_path, record=_live_record())
    ctx = _ctx(tmp_path)
    sha = _projection(mod._wait_for_task(ctx, CHILD, timeout_sec=0))["child_result_sha256"]

    changed = mod._wait_for_task(ctx, CHILD, timeout_sec=0, known_result_sha256="0" * 64)
    assert _projection(changed)["child_result_sha256"] == sha
    assert _projection(changed)["result"] == "partial answer"

    same = _projection(mod._wait_for_task(ctx, CHILD, timeout_sec=0, known_result_sha256=sha))
    assert same["result_unchanged"] is True and "result" not in same and "trace_summary" not in same


def test_the_batch_and_the_single_wait_share_one_projection(tmp_path, monkeypatch):
    _seed_child(tmp_path, record=_live_record())
    calls = []
    real = mod._compact_child_projection
    monkeypatch.setattr(mod, "_compact_child_projection",
                        lambda tid, data, known: calls.append(tid) or real(tid, data, known))

    batch = json.loads(mod._wait_for_tasks(_ctx(tmp_path), [CHILD], timeout_sec=0))["tasks"][CHILD]
    single = _projection(mod._wait_for_task(_ctx(tmp_path), CHILD, timeout_sec=0))

    assert calls == [CHILD, CHILD]
    assert single == batch


# ---------------------------------------------------------------- E: the window


def _deadline_ctx(drive, remaining_sec):
    return _ctx(drive, task_metadata={
        "deadline_at": (utc_now() + _dt.timedelta(seconds=remaining_sec)).isoformat()})


@pytest.mark.parametrize("requested, clamp, minimum, margin, remaining, expected", [
    (180, 3600, 0, 30, None, (180, "requested")),
    (0, 3600, 0, 30, None, (0, "requested")),       # the zero-second peek survives
    (-4, 3600, 0, 30, None, (0, "minimum")),
    (10_000, 7200, 0, 30, None, (7200, "ceiling")),
    (3600, 3600, 0, 30, 1000.5, (910, "deadline")),  # emit window 940, less the margin once
    (3600, 3600, 0, 30, 90.5, (0, "deadline")),      # emit window 30: nothing left to wait
    (3600, 3600, 0, 30, -5, (0, "deadline")),        # spent deadline: one peek
    (0, 1800, 1, 1, None, (1, "minimum")),           # await_messages keeps its minimum ...
    (600, 1800, 1, 1, 400.5, (339, "deadline")),     # ... and its one-second margin
])
def test_the_one_window_ladder(tmp_path, monkeypatch, requested, clamp, minimum, margin, remaining, expected):
    monkeypatch.setattr(pacing, "effective_finalization_reserve_sec", lambda ctx: 60.0)
    ctx = _ctx(tmp_path) if remaining is None else _deadline_ctx(tmp_path, remaining)

    assert mod._wait_window(ctx, requested, clamp=clamp, minimum=minimum, margin=margin) == expected


_WINDOWED = {
    "wait_task": (mod._WAIT_TASK_CLAMP_SEC, 0, 30),
    "wait_tasks": (mod._WAIT_TASKS_CLAMP_SEC, 0, 30),
    "await_messages": (None, 1, 1),
}


def _executor_margins(tmp_path, monkeypatch, name, get_timeout, settings_timeout):
    import ouroboros.config as config_mod
    import ouroboros.loop_tool_execution as lte

    monkeypatch.setattr(config_mod, "get_per_call_timeout_ceiling_sec", lambda: 1800)
    monkeypatch.setattr(lte, "load_settings", lambda: {"OUROBOROS_TOOL_TIMEOUT_SEC": settings_timeout})
    monkeypatch.setattr(pacing, "effective_finalization_reserve_sec", lambda ctx: 60.0)
    clamp, minimum, margin = _WINDOWED[name]
    clamp = 1800 if clamp is None else clamp
    margins = []
    for remaining in (None, 95.5, 400.5, 3000.5, 3650.5, 7000.5, 7300.5, 20_000.5):
        ctx = _ctx(tmp_path) if remaining is None else _deadline_ctx(tmp_path, remaining)
        executor = lte._get_tool_timeout(SimpleNamespace(_ctx=ctx, get_timeout=get_timeout), name, {})
        window, _bound = mod._wait_window(ctx, 10**6, clamp=clamp, minimum=minimum, margin=margin)
        if window > 0:
            margins.append(executor - window)
    return margins, margin


@pytest.mark.parametrize("name", sorted(_WINDOWED))
@pytest.mark.parametrize("settings_timeout", [0, 60, 99_999])
def test_the_executor_kill_timer_outlives_every_window_by_the_margin(tmp_path, monkeypatch, name,
                                                                     settings_timeout):
    """Exactly one reserve subtraction on both sides of the executor seam: with and
    without a deadline, and whatever OUROBOROS_TOOL_TIMEOUT_SEC says."""
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=REPO, drive_root=tmp_path)
    margins, margin = _executor_margins(tmp_path, monkeypatch, name, registry.get_timeout, settings_timeout)

    assert margins and min(margins) >= margin, margins


def test_the_old_equal_window_and_kill_timeout_fails_the_ordering(tmp_path, monkeypatch):
    """The guard's other direction: a 7200 s entry around a 7200 s window leaves the
    full window to end in TOOL_TIMEOUT with an abandoned thread."""
    margins, margin = _executor_margins(tmp_path, monkeypatch, "wait_tasks", lambda name: 7200, 0)

    assert min(margins) < margin


def test_a_narrowed_window_is_named_in_both_results(tmp_path, monkeypatch):
    monkeypatch.setattr(pacing, "effective_finalization_reserve_sec", lambda ctx: 60.0)
    write_task_result(tmp_path, "livechild", STATUS_RUNNING, result="")
    ctx = _deadline_ctx(tmp_path, 90.5)

    single = mod._wait_for_task(ctx, "livechild", timeout_sec=600)
    batch = json.loads(mod._wait_for_tasks(ctx, ["livechild"], timeout_sec=600))

    assert json.loads(single)["window_sec"] == 0
    assert json.loads(single)["window_bound"] == "deadline"
    assert (batch["window_sec"], batch["window_bound"], batch["timeout_sec"]) == (0.0, "deadline", 0.0)
    assert batch["wait_expired_with_live_children"]["requested_timeout_sec"] == 600.0

    plain = _ctx(tmp_path)
    assert "[WAIT_WINDOW]" not in mod._wait_for_task(plain, "livechild", timeout_sec=0)
    assert "window_bound" not in json.loads(mod._wait_for_tasks(plain, ["livechild"], timeout_sec=0))


def test_the_poll_sleep_never_overshoots_the_window(tmp_path):
    from ouroboros.task_status import wait_for_effective_tasks

    write_task_result(tmp_path, "livechild", STATUS_RUNNING, result="")
    start = time.monotonic()

    out = wait_for_effective_tasks(tmp_path, ["livechild"], timeout_sec=1.0, poll_interval_sec=2.0)

    assert out["timed_out"] is True and time.monotonic() - start < 1.5


def test_a_slow_final_read_still_ends_inside_the_kill_timer(tmp_path, monkeypatch):
    """The materializing read after the loop is not bounded by the window; the
    settlement margin is what leaves it room before the executor's kill timer."""
    import ouroboros.loop_tool_execution as lte
    import ouroboros.task_status as task_status

    monkeypatch.setattr(pacing, "effective_finalization_reserve_sec", lambda ctx: 60.0)
    monkeypatch.setattr(lte, "load_settings", lambda: {})
    write_task_result(tmp_path, "livechild", STATUS_RUNNING, result="")
    real = task_status.load_effective_task_result

    def _slow(root, tid, *args, **kw):
        if kw.get("materialize_artifacts", True):
            time.sleep(0.5)
        return real(root, tid, *args, **kw)

    monkeypatch.setattr(task_status, "load_effective_task_result", _slow)
    ctx = _deadline_ctx(tmp_path, 91.9)
    executor = lte._deadline_clamped_timeout(SimpleNamespace(_ctx=ctx), "wait_tasks", 7230)
    start = time.monotonic()

    out = json.loads(mod._wait_for_tasks(ctx, ["livechild"], timeout_sec=600))

    took = time.monotonic() - start
    assert out["window_bound"] == "deadline" and 0 < out["window_sec"] <= 1
    assert took < out["window_sec"] + 1.5 and executor - took >= 28
