"""Tests for evolution/consciousness status snapshots."""

import json

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


def test_evolution_status_reports_budget_stop_when_disabled_after_run(monkeypatch):
    from supervisor import queue as queue_module

    monkeypatch.setattr(queue_module, "PENDING", [])
    monkeypatch.setattr(queue_module, "RUNNING", {})
    monkeypatch.setattr(
        queue_module,
        "load_state",
        lambda: {
            "evolution_mode_enabled": False,
            "owner_chat_id": 7,
            "evolution_cycle": 6,
            "evolution_consecutive_failures": 0,
            "last_evolution_task_at": "2026-03-31T10:00:00Z",
        },
    )
    monkeypatch.setattr(queue_module, "budget_remaining", lambda st, **_kwargs: 1.25)

    snapshot = queue_module.get_evolution_status_snapshot()

    assert snapshot["status"] == "budget_stopped"
    assert snapshot["budget_remaining_usd"] == 1.25


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
