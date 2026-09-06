"""Owner-facing heartbeat presentation regressions for v6.64."""

from __future__ import annotations

import inspect
import pathlib
import time
import types


def _heartbeat_ctx(tmp_path, task):
    pushed = []
    return types.SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={task["id"]: {"task": task, "started_at": time.time()}},
        bridge=types.SimpleNamespace(push_log=pushed.append),
    ), pushed


def test_routine_heartbeat_is_not_rendered_as_chat_message() -> None:
    from supervisor import queue

    source = inspect.getsource(queue._enforce_task_timeouts_locked)
    assert "running for" not in source
    assert "heartbeat_lag=" not in source


def test_liveness_and_incident_controls_remain_active() -> None:
    from supervisor import queue

    # The enforce loop keeps the DECISION; the grace-window side effects (finalize_now
    # control + owner incident toast) live in task_reaper.request_finalization_grace.
    source = "\n".join((
        inspect.getsource(queue._enforce_task_timeouts_locked),
        inspect.getsource(queue._request_finalization_grace),
    ))
    for invariant in (
        "last_heartbeat_at",
        "get_task_idle_timeout_sec",
        "get_task_abs_ceiling_sec",
        "deadline_reached",
        "finalization_requested_at",
        '"task_incident": terminal_reason',
        '"is_progress": True',
    ):
        assert invariant in source


def test_root_heartbeat_carries_one_live_subtree_projection(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from supervisor.events import _handle_task_heartbeat

    monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_kw: {
        "accounted_usd": 76.82, "reserved_usd": 1.25,
        "unresolved_upper_bound_usd": 0.5, "unknown_unmetered": 2,
        "non_final_rows": 3, "integrity_degraded": False, "cost_final": True,
    })
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-hb", "type": "task", "root_task_id": "root-hb",
    })
    _handle_task_heartbeat({"task_id": "root-hb", "phase": "running"}, ctx)

    frame = pushed[0]
    assert frame["accounted_upper_bound_usd_with_children"] == 76.82
    assert frame["cost_final"] is False
    assert frame["cost_with_children_partial"] is True
    assert (frame["reserved_usd"], frame["unresolved_upper_bound_usd"]) == (1.25, 0.5)
    assert frame["unknown_unmetered"] == 2


def test_child_heartbeat_does_not_read_or_publish_live_cost(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from supervisor.events import _handle_task_heartbeat

    monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_kw: (
        _ for _ in ()
    ).throw(AssertionError("child heartbeat must not read the ledger")))
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "child-hb", "type": "task", "root_task_id": "root-hb",
        "parent_task_id": "root-hb", "delegation_role": "subagent",
    })
    _handle_task_heartbeat({"task_id": "child-hb", "phase": "running"}, ctx)

    assert "cost_accounting_status" not in pushed[0]
    assert "accounted_upper_bound_usd_with_children" not in pushed[0]


def test_root_heartbeat_keeps_ledger_failure_honest(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from supervisor.events import _handle_task_heartbeat

    monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_kw: (
        _ for _ in ()
    ).throw(OSError("ledger unavailable")))
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-hb", "type": "task", "root_task_id": "root-hb",
    })
    _handle_task_heartbeat({"task_id": "root-hb", "phase": "running"}, ctx)

    frame = pushed[0]
    assert frame["cost_accounting_status"] == "unavailable"
    assert frame["accounted_upper_bound_usd_with_children"] is None
    assert frame["cost_final"] is False


def test_root_heartbeat_omits_empty_or_legacy_only_cost(tmp_path, monkeypatch):
    from ouroboros import usage_accounting
    from supervisor.events import _handle_task_heartbeat

    monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_kw: {
        "accounted_usd": 0.0, "attempt_counts": {"metadata_only": 4},
        "subscription_sessions": 0, "unknown_unmetered": 0,
        "non_final_rows": 0, "integrity_degraded": False,
    })
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-empty", "type": "task", "root_task_id": "root-empty",
    })
    _handle_task_heartbeat({"task_id": "root-empty", "phase": "running"}, ctx)

    assert "cost_accounting_status" not in pushed[0]
    assert "accounted_upper_bound_usd_with_children" not in pushed[0]


def test_root_heartbeat_keeps_unknown_zero_cost_nullable_and_measured_zero_numeric(
    tmp_path, monkeypatch,
):
    from ouroboros import usage_accounting
    from supervisor.events import _handle_task_heartbeat

    usage = {
        "accounted_usd": 0.0, "attempt_counts": {"settled": 1},
        "subscription_sessions": 0, "unknown_unmetered": 1,
        "non_final_rows": 1, "integrity_degraded": False,
    }
    monkeypatch.setattr(usage_accounting, "usage_projection", lambda *_a, **_kw: usage)
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-unknown", "type": "task", "root_task_id": "root-unknown",
    })
    _handle_task_heartbeat({"task_id": "root-unknown", "phase": "running"}, ctx)
    unknown = pushed[0]
    assert unknown["cost_accounting_status"] == "available"
    assert unknown["accounted_upper_bound_usd_with_children"] is None
    assert unknown["unknown_unmetered"] == 1

    usage = {**usage, "unknown_unmetered": 0, "non_final_rows": 0}
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-free", "type": "task", "root_task_id": "root-free",
    })
    _handle_task_heartbeat({"task_id": "root-free", "phase": "running"}, ctx)
    assert pushed[0]["accounted_upper_bound_usd_with_children"] == 0.0


def test_a_retired_liveness_knob_leaves_no_deprecation_chatter(tmp_path, monkeypatch) -> None:
    """Retirement is silent by construction, for the whole class.

    Three knobs (the planning-scout heartbeat, and D04's flat soft/hard wall
    clock) were once accepted as typed no-ops that logged a
    ``deprecated_settings_ignored`` row. ``config.RETIRED_SETTING_KEYS`` drops
    all three on settings load, so there is no stored value left for the
    supervisor to be loud about — and no half-retired key it still names."""
    from ouroboros import config
    from supervisor import queue

    retired = ("OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC",
               "OUROBOROS_SOFT_TIMEOUT_SEC", "OUROBOROS_HARD_TIMEOUT_SEC")
    for key in retired:
        assert key in config.RETIRED_SETTING_KEYS
        monkeypatch.setenv(key, "121")
    queue.init(tmp_path)
    queue.refresh_timeouts_from_settings({key: 121 for key in retired})
    assert not (tmp_path / "logs" / "events.jsonl").exists()


def test_owner_visible_incidents_use_canonical_message_seam() -> None:
    repo = pathlib.Path(__file__).resolve().parents[1]
    for relpath in (
        "server.py",
        "supervisor/workers.py",
        # F2.2: spans moved out of workers.py keep the same seam invariant.
        "supervisor/worker_assignment.py",
        "supervisor/worker_health.py",
    ):
        source = (repo / relpath).read_text(encoding="utf-8")
        assert "get_bridge().send_message(" not in source
        assert "bridge.send_message(" not in source


def test_cancel_failure_is_progress_incident_not_chat_bubble(monkeypatch) -> None:
    import supervisor.queue as q
    from supervisor.events import _handle_cancel_task

    sent = []
    # Phase A: the handler drives the TYPED custody outcome directly (the boolean
    # facade collapsed already_settled into a false "✅ cancel").
    monkeypatch.setattr(
        q, "cancel_task_custody",
        lambda task_id, **_kw: q.CANCEL_FAILED if task_id == "cancel-me" else q.CANCEL_NOT_FOUND,
    )

    class _Ctx:
        @staticmethod
        def load_state():
            return {"owner_chat_id": 9}

        @staticmethod
        def send_with_budget(*args, **kwargs):
            sent.append((args, kwargs))

    _handle_cancel_task({"task_id": "cancel-me"}, _Ctx())

    assert len(sent) == 1
    args, kwargs = sent[0]
    assert args[0] == 9
    assert args[1].startswith("❌ cancel cancel-me")
    assert "watchdog" in args[1]  # the intent stays open and is retried
    assert kwargs == {
        "is_progress": True,
        "task_id": "cancel-me",
        "progress_meta": {
            "task_incident": "cancellation_fault",
            "toast_once": "cancel-me:cancellation_fault",
        },
    }


def test_cancel_event_keeps_requested_retry_id_for_owner_presentation(monkeypatch) -> None:
    import supervisor.queue as q
    from supervisor.events import _handle_cancel_task

    captured = []
    monkeypatch.setattr(
        q,
        "cancel_task_custody",
        lambda task_id, **_kwargs: (
            captured.append(("custody", task_id)) or q.CANCEL_FAILED
        ),
    )

    class _Ctx:
        DRIVE_ROOT = pathlib.Path("/unused")

        @staticmethod
        def load_state():
            return {"owner_chat_id": 9}

        @staticmethod
        def send_with_budget(*args, **kwargs):
            captured.append((args, kwargs))

    _handle_cancel_task(
        {"task_id": "physical-retry-7f3a", "requested_task_id": "stable-root"},
        _Ctx(),
    )

    assert captured[0] == ("custody", "physical-retry-7f3a")
    args, kwargs = captured[1]
    assert args[1].startswith("❌ cancel stable-root")
    assert "physical-retry-7f3a" not in args[1]
    assert kwargs["task_id"] == "stable-root"
    assert kwargs["progress_meta"] == {
        "task_incident": "cancellation_fault",
        "toast_once": "stable-root:cancellation_fault",
        "cancel_physical_task_id": "physical-retry-7f3a",
    }
