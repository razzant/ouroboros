"""Tests for evolution/consciousness status snapshots."""

import json

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


def test_evolution_status_waits_for_owner_chat(monkeypatch):
    from supervisor import queue as queue_module

    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(
        queue_module,
        "load_state",
        lambda: {
            "evolution_mode_enabled": True,
            "owner_chat_id": None,
            "evolution_cycle": 3,
            "evolution_consecutive_failures": 0,
            "last_evolution_task_at": "",
        },
    )
    monkeypatch.setattr(queue_module, "budget_remaining", lambda st, **_kwargs: 25.0)

    snapshot = queue_module.get_evolution_status_snapshot()

    assert snapshot["status"] == "waiting_for_owner_chat"
    assert snapshot["enabled"] is True
    assert snapshot["owner_chat_bound"] is False


def test_evolution_status_reports_waiting_for_idle(monkeypatch):
    from supervisor import queue as queue_module

    monkeypatch.setattr(queue_module, "PENDING", [{"id": "task-1", "type": "task"}])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(
        queue_module,
        "load_state",
        lambda: {
            "evolution_mode_enabled": True,
            "owner_chat_id": 7,
            "evolution_cycle": 4,
            "evolution_consecutive_failures": 0,
            "last_evolution_task_at": "",
        },
    )
    monkeypatch.setattr(queue_module, "budget_remaining", lambda st, **_kwargs: 25.0)

    snapshot = queue_module.get_evolution_status_snapshot()

    assert snapshot["status"] == "waiting_for_idle"
    assert snapshot["pending_count"] == 1


@pytest.mark.parametrize("enabled,failures,remaining,status", [
    (False, 0, 1.25, "disabled"),  # formerly inferred as budget_stopped
    (True, 0, 0.0, "budget_blocked"),  # the ordinary exhausted limit
    (True, 0, 0.003, "idle_ready"),  # shown as $0.00, yet positive money the tick admits on
])
def test_evolution_status_names_no_count_stop_or_reserve(monkeypatch, enabled, failures, remaining, status):
    from supervisor import queue as queue_module

    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(
        queue_module,
        "load_state",
        lambda: {
            "evolution_mode_enabled": enabled,
            "owner_chat_id": 7,
            "evolution_cycle": 6,
            "evolution_consecutive_failures": failures,
            "last_evolution_task_at": "2026-03-31T10:00:00Z",
        },
    )
    monkeypatch.setattr(queue_module, "budget_remaining", lambda st, **_kwargs: remaining)

    snapshot = queue_module.get_evolution_status_snapshot()

    assert snapshot["status"] == status
    assert (snapshot["budget_remaining_usd"], snapshot["consecutive_failures"]) == (round(remaining, 2), failures)
    assert snapshot["budget_reserve_usd"] == 0.0


@pytest.fixture
def queued_cycle(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor import queue

    cycle = {"id": "cycle", "type": "evolution"}
    write_task_result(tmp_path, "cycle", "scheduled", root_task_id="cycle")
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [cycle])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(queue, "load_state", lambda: {"evolution_mode_enabled": True, "owner_chat_id": 1})
    monkeypatch.setattr(queue, "budget_remaining", lambda *_args, **_kwargs: 20.0)
    monkeypatch.setattr(queue, "_read_evolution_campaign", lambda: {})
    return queue, cycle


@pytest.mark.parametrize("row,fence,expected", [
    ({}, {}, "queued"),
    ({}, {"status": "active", "fence_id": "money"}, "budget_stopped"),
    ({"_budget_pause": {"status": "paused_before_dispatch"}}, {"status": "active", "cause": "owner_pause"}, "queued"),
    ({"_budget_pause": {"status": "paused_before_dispatch"},
      "_budget_pause_hold": {"reason": "panic_hold"}}, {}, "queued"),
    ({"_budget_pause_hold": {"reason": "budget", "release": "none"}}, {}, "queued"),
    ({"_budget_pause_hold": {"reason": "new_reason"}}, {}, "queued"),
])
def test_only_a_current_money_stop_released_by_resume_is_budget_stopped(queued_cycle, row, fence, expected):
    queue, cycle = queued_cycle
    cycle.update(row)
    queue.BUDGET_ROOT_FENCES["cycle"] = fence
    assert queue.get_evolution_status_snapshot()["status"] == expected


def test_owner_pause_over_a_money_parked_cycle(queued_cycle, tmp_path, monkeypatch):
    from ouroboros import owner_pause

    queue, cycle = queued_cycle
    cycle["_budget_pause"] = {"status": "paused_before_dispatch", "replay_safe": True}

    def expect_status(expected):
        snapshot = queue.get_evolution_status_snapshot()
        assert (snapshot["queued_task_id"], snapshot["status"]) == ("cycle", expected)
        assert ("Resume" in snapshot["detail"]) == (expected == "budget_stopped")

    fence, _ = owner_pause.install_fence(tmp_path, "cycle", request_id="pause-cycle")
    expect_status("queued")
    owner_pause.set_fence_state(tmp_path, "cycle", fence_id=fence["fence_id"], state="paused")
    expect_status("queued")
    owner_pause.release_fence(tmp_path, "cycle", reason="owner_resume")
    expect_status("budget_stopped")

    def unreadable(*_args, **_kwargs):
        raise OSError("synthetic owner fence read failure")

    monkeypatch.setattr(owner_pause, "read_fence", unreadable)
    expect_status("state_unknown")


def test_evolution_control_restart_and_running_precede_a_parked_cycle(queued_cycle, monkeypatch):
    queue, cycle = queued_cycle
    cycle["_budget_pause"] = {"status": "paused_before_dispatch"}
    live, campaign = {}, {"active_transaction": {"commit_sha": "abc", "restart_required": True}}
    monkeypatch.setattr(queue, "load_state", lambda: live)
    monkeypatch.setattr(queue, "_read_evolution_campaign", lambda: campaign)
    queue.RUNNING["other"] = {"task": {"id": "other", "type": "evolution"}}
    assert queue.get_evolution_status_snapshot()["status"] == "state_unknown"
    live["evolution_mode_enabled"] = True
    assert queue.get_evolution_status_snapshot()["status"] == "waiting_for_restart_verify"
    campaign.clear()
    assert queue.get_evolution_status_snapshot()["status"] == "running"
    queue.RUNNING.clear()
    assert queue.get_evolution_status_snapshot()["status"] == "budget_stopped"


def test_a_subscription_main_is_not_budget_blocked_at_the_reached_limit(monkeypatch):
    from supervisor import queue as queue_module

    monkeypatch.setenv("OUROBOROS_MODEL", "claudexor::codex=gpt-6-astra")
    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(queue_module, "load_state", lambda: {
        "evolution_mode_enabled": True, "owner_chat_id": 7, "evolution_cycle": 6})
    monkeypatch.setattr(queue_module, "budget_remaining", lambda st, **_kwargs: 0.0)
    assert queue_module.get_evolution_status_snapshot()["status"] == "idle_ready"


@pytest.mark.parametrize("case,settled,status,shown,admitted", [
    ("positive_subcent", 4.997, "idle_ready", 0.0, True),
    ("overspent", 5.25, "budget_blocked", 0.0, False),  # the store reports no room, never a negative one
    ("unknown", 4.997, "accounting_unavailable", None, False),
])
def test_the_state_endpoint_reports_what_the_tick_then_admits(tmp_path, monkeypatch, case, settled, status, shown, admitted):
    """The real GET /api/state handler classifies the money the real tick admits on; its
    budget_remaining_usd keeps its cents (owner decision 2026-11-02 removed the $2 reserve)."""
    import asyncio
    import types

    from starlette.requests import Request

    from ouroboros import usage_accounting as ua
    from ouroboros.gateway.state import api_state
    from supervisor import queue, state
    from tests.test_evolution_redesign import _owner_campaign, _settle

    pending, notices, _ctx = _owner_campaign(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 5.0)
    _settle(tmp_path, "earlier-work", settled)
    if case == "disabled":
        state.update_state(lambda live: live.update(evolution_mode_enabled=False))
    if case == "unknown":
        def unreadable(*_args, **_kwargs):
            raise OSError("usage store unreadable")
        monkeypatch.setattr(ua, "usage_projection", unreadable)
    request = Request({"type": "http", "method": "GET", "path": "/api/state", "headers": [], "query_string": b"",
                       "scheme": "http", "server": ("test", 80), "client": ("test", 1),
                       "app": types.SimpleNamespace(state=types.SimpleNamespace(drive_root=tmp_path, app_start=0.0))})

    evolution = json.loads(asyncio.run(api_state(request)).body)["evolution_state"]
    queue.enqueue_evolution_task_if_needed()

    assert [row["type"] for row in pending] == (["evolution"] if admitted else [])
    assert (evolution["status"], evolution["budget_remaining_usd"]) == (status, shown)
    assert notices == []
    events_path = tmp_path / "logs" / "events.jsonl"
    events = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
    assert ("evolution_accounting_unavailable" in events) is (case == "unknown")


def test_consciousness_status_snapshot_exposes_the_alarm_facts(monkeypatch, tmp_path):
    from ouroboros import consciousness as clock_module
    from ouroboros.consciousness import BackgroundConsciousness
    from supervisor import state

    monkeypatch.setattr(state, "load_state", lambda: {"bg_consciousness_enabled": True, "owner_chat_id": 1})
    monkeypatch.setattr(clock_module, "allowance_window", lambda root, now=None, **_display_read: {
        "status": "available", "limit_usd": 20.0, "settled_usd": 3.0, "accounted_usd": 3.0, "remaining_usd": 17.0, "resets_at": ""})
    monkeypatch.setattr("supervisor.queue.live_consciousness_roots", lambda: [])
    clock = BackgroundConsciousness(tmp_path, tmp_path / "repo", lambda: 1, now=1_800_000_000.0)
    snapshot = clock.status_snapshot()

    # #1603: no early-wake reason exists to expose.
    assert snapshot["enabled"] is True and "pending_reason" not in snapshot
    assert snapshot["next_wake_at"] and snapshot["last_wake_at"] == "" and snapshot["live_wake_task_id"] == ""
    assert snapshot["spent_24h_usd"] == 3.0 and snapshot["daily_usd"] == 20.0


def test_evolution_data_strips_legacy_checkpoint_result_status(tmp_path, monkeypatch):
    from ouroboros.evolution_checkpoints import CHECKPOINTS_REL
    from ouroboros.gateway import control

    repo = tmp_path / "repo"
    repo.mkdir()
    checkpoint_path = tmp_path / CHECKPOINTS_REL
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(
        json.dumps({
            "task_id": "evo-legacy",
            "status": "completed",
            "result_status": "failed",
            "reason_code": "legacy_error",
            "loop_outcome": {
                "result_status": "failed",
                "compat_result_status": "failed",
            },
        })
        + "\n",
        encoding="utf-8",
    )

    asked = []

    def fake_projection(repo_dir, data_dir, *, force=False):
        asked.append((repo_dir, data_dir, force))
        return {"versions": [], "saved_at": "", "refresh": {"state": "idle"}, "install": {}}

    monkeypatch.setattr("ouroboros.evolution_history.history.projection", fake_projection)
    app = Starlette(routes=[Route("/api/evolution-data", endpoint=control.api_evolution_data)])
    app.state.drive_root = tmp_path
    app.state.repo_dir = repo

    response = TestClient(app).get("/api/evolution-data?force=1")
    payload = response.json()
    assert asked == [(str(repo), str(tmp_path), True)] and response.headers["cache-control"] == "no-store"

    checkpoint = payload["checkpoints"][0]
    assert "result_status" not in checkpoint
    assert "result_status" not in checkpoint["loop_outcome"]
    assert "compat_result_status" not in checkpoint["loop_outcome"]
    assert checkpoint["outcome_axes"]["execution"]["status"] == "failed"


def test_evolution_data_polls_do_not_reread_an_unchanged_checkpoint_journal(tmp_path, monkeypatch):
    """The page asks every two seconds while a refresh runs; the unbounded journal is read once, then only appends."""
    import pathlib

    from ouroboros.evolution_checkpoints import CHECKPOINTS_REL
    from ouroboros.gateway import control

    journal = tmp_path / CHECKPOINTS_REL
    journal.parent.mkdir(parents=True, exist_ok=True)
    row = lambda n, **extra: json.dumps({"task_id": f"evo-{n}", "result_status": "failed", **extra}) + "\n"  # noqa: E731
    journal.write_text("".join(row(n) for n in range(150)) + row(150, kind="cycle_outcome"), encoding="utf-8")
    opened = []
    real_open = pathlib.Path.open
    monkeypatch.setattr(pathlib.Path, "open", lambda self, *a, **k: (opened.append(self.name) if self == journal else None)
                        or real_open(self, *a, **k))
    monkeypatch.setattr("ouroboros.evolution_history.history.projection", lambda *_a, **_k: {
        "versions": [], "saved_at": "", "refresh": {"state": "running"}, "install": {}})
    app = Starlette(routes=[Route("/api/evolution-data", endpoint=control.api_evolution_data)])
    app.state.drive_root = tmp_path
    app.state.repo_dir = tmp_path / "repo"
    client = TestClient(app)

    first, second = (client.get("/api/evolution-data").json()["checkpoints"] for _ in range(2))
    assert [c["task_id"] for c in first] == [f"evo-{n}" for n in range(50, 150)] and second == first
    assert "result_status" not in first[0] and opened == [journal.name], "an unchanged journal is opened once"
    with journal.open("a", encoding="utf-8") as handle:
        handle.write(row(151))
    third = client.get("/api/evolution-data").json()["checkpoints"]
    assert [c["task_id"] for c in third][-2:] == ["evo-149", "evo-151"] and len(third) == 100
    assert opened == [journal.name] * 3, "the append is read (one open) after the test's own"


def test_budget_remaining_uses_valid_projection_and_self_computes_on_limit_mismatch(monkeypatch):
    """De-triplication seam pin: a passed projection is used verbatim (zero
    ledger reads) ONLY when its limit_usd equals the limit budget_remaining
    reads itself; a mismatched limit (settings hot-reload race, wrong-limit
    caller) falls through to self-computation."""
    from ouroboros import usage_accounting as ua
    from supervisor import state

    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 10.0)
    calls = {"projection": 0}

    def _projection(_root, **_kwargs):
        calls["projection"] += 1
        return {"limit_usd": 10.0, "remaining_known_usd": 3.75}

    monkeypatch.setattr(ua, "usage_projection", _projection)

    valid = {"limit_usd": 10.0, "remaining_known_usd": 4.25}
    assert state.budget_remaining({}, projection=valid) == 4.25
    assert calls == {"projection": 0}

    stale = {"limit_usd": 5.0, "remaining_known_usd": 4.25}
    assert state.budget_remaining({}, projection=stale) == 3.75
    assert calls == {"projection": 1}
