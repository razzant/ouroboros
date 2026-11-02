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


def _ledger(monkeypatch, answer):
    """Replace the one ledger read of the live root heartbeat (``usage_breakdown`` with
    custody coverage) and record each call; ``answer`` is the bucket or the error it raises."""
    from ouroboros import usage_accounting

    calls = []

    def read(drive_root, **kwargs):
        calls.append(kwargs)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(usage_accounting, "usage_breakdown", read)
    return calls


def _live_read(calls, root_task_id):
    """The root heartbeat read its own subtree once, stale-allowed and with custody coverage."""
    [call] = calls
    assert call["root_task_id"] == root_task_id
    assert call["allow_stale"] is True and call["include_custody"] is True


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
    from supervisor.events import _handle_task_heartbeat

    calls = _ledger(monkeypatch, {
        "accounted_usd": 76.82, "reserved_usd": 1.25,
        "unresolved_upper_bound_usd": 0.5, "unknown_unmetered": 2,
        "non_final_rows": 3, "integrity_degraded": False, "cost_final": True,
    })
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-hb", "type": "task", "root_task_id": "root-hb",
    })
    _handle_task_heartbeat({"task_id": "root-hb", "phase": "running"}, ctx)

    _live_read(calls, "root-hb")
    frame = pushed[0]
    assert frame["accounted_upper_bound_usd_with_children"] == 76.82
    assert frame["cost_final"] is False
    assert frame["cost_with_children_partial"] is True
    assert (frame["reserved_usd"], frame["unresolved_upper_bound_usd"]) == (1.25, 0.5)
    assert frame["unknown_unmetered"] == 2


def test_child_heartbeat_does_not_read_or_publish_live_cost(tmp_path, monkeypatch):
    from supervisor.events import _handle_task_heartbeat

    calls = _ledger(monkeypatch, AssertionError("child heartbeat must not read the ledger"))
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "child-hb", "type": "task", "root_task_id": "root-hb",
        "parent_task_id": "root-hb", "delegation_role": "subagent",
    })
    _handle_task_heartbeat({"task_id": "child-hb", "phase": "running"}, ctx)

    assert calls == []
    assert "cost_accounting_status" not in pushed[0]
    assert "accounted_upper_bound_usd_with_children" not in pushed[0]


def test_root_heartbeat_keeps_ledger_failure_honest(tmp_path, monkeypatch):
    from supervisor.events import _handle_task_heartbeat

    calls = _ledger(monkeypatch, OSError("ledger unavailable"))
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-hb", "type": "task", "root_task_id": "root-hb",
    })
    _handle_task_heartbeat({"task_id": "root-hb", "phase": "running"}, ctx)

    _live_read(calls, "root-hb")
    frame = pushed[0]
    assert frame["cost_accounting_status"] == "unavailable"
    assert frame["accounted_upper_bound_usd_with_children"] is None
    assert frame["cost_final"] is False


def test_root_heartbeat_omits_empty_or_legacy_only_cost(tmp_path, monkeypatch):
    from supervisor.events import _handle_task_heartbeat

    calls = _ledger(monkeypatch, {
        "accounted_usd": 0.0, "attempt_counts": {"metadata_only": 4},
        "subscription_sessions": 0, "unknown_unmetered": 0,
        "non_final_rows": 0, "integrity_degraded": False,
    })
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-empty", "type": "task", "root_task_id": "root-empty",
    })
    _handle_task_heartbeat({"task_id": "root-empty", "phase": "running"}, ctx)

    _live_read(calls, "root-empty")
    assert "cost_accounting_status" not in pushed[0]
    assert "accounted_upper_bound_usd_with_children" not in pushed[0]


def test_root_heartbeat_keeps_unknown_zero_cost_nullable_and_measured_zero_numeric(
    tmp_path, monkeypatch,
):
    from supervisor.events import _handle_task_heartbeat

    usage = {
        "accounted_usd": 0.0, "attempt_counts": {"settled": 1},
        "subscription_sessions": 0, "unknown_unmetered": 1,
        "non_final_rows": 1, "integrity_degraded": False,
    }
    calls = _ledger(monkeypatch, usage)
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-unknown", "type": "task", "root_task_id": "root-unknown",
    })
    _handle_task_heartbeat({"task_id": "root-unknown", "phase": "running"}, ctx)
    _live_read(calls, "root-unknown")
    unknown = pushed[0]
    assert unknown["cost_accounting_status"] == "available"
    assert unknown["accounted_upper_bound_usd_with_children"] is None
    assert unknown["unknown_unmetered"] == 1

    calls = _ledger(monkeypatch, {**usage, "unknown_unmetered": 0, "non_final_rows": 0})
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-free", "type": "task", "root_task_id": "root-free",
    })
    _handle_task_heartbeat({"task_id": "root-free", "phase": "running"}, ctx)
    _live_read(calls, "root-free")
    assert pushed[0]["accounted_upper_bound_usd_with_children"] == 0.0


def test_root_heartbeat_with_an_empty_ledger_and_an_open_run_is_unknown_not_absent(tmp_path):
    """No ledger row yet, but a delegated run of this root is still open: the real read
    publishes an unknown amount with its custody, never ``$0`` and never nothing."""
    from ouroboros import delegate_custody as custody, usage_store
    from supervisor.events import _handle_task_heartbeat

    usage_store.migrate_from_journal(tmp_path)  # the server's startup phase; a stale display read never migrates
    entry = custody.RunCustody(run_id="open-run", task_id="root-open", root_task_id="root-open", route_id="test")
    assert custody.record_started(tmp_path, entry)
    ctx, pushed = _heartbeat_ctx(tmp_path, {
        "id": "root-open", "type": "task", "root_task_id": "root-open",
    })
    _handle_task_heartbeat({"task_id": "root-open", "phase": "running"}, ctx)

    frame = pushed[0]
    assert frame["cost_accounting_status"] == "available"
    assert frame["accounted_upper_bound_usd_with_children"] is None
    assert frame["cost_custody"]["open_run_ids"] == ["open-run"]
    assert frame["cost_final"] is False and frame["cost_with_children_partial"] is True


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


def test_cancel_failure_wakes_the_intent_driver_and_says_nothing(monkeypatch) -> None:
    """The cancel_task event is a wake for the durable intent's driver. A FAILED
    outcome used to send a "❌ cancel … did not settle" row to the owner chat;
    seven of the eight phantom cancelled-subagent cards in Main were that row,
    drained 23–58 minutes after the watchdog had already settled the intent.
    The intent stays open for the watchdog; the card and supervisor.jsonl are
    the record, so the event publishes nothing for any outcome."""
    import supervisor.queue as q
    from supervisor.events_runtime_controls import _drive_cancel_task_event as _handle_cancel_task

    sent = []
    seen = []
    monkeypatch.setattr(
        q, "drive_cancel_intent_scope",
        lambda task_id, **_kw: seen.append(task_id) or (
            q.CANCEL_FAILED if task_id == "cancel-me" else q.CANCEL_NOT_FOUND
        ),
    )

    class _Ctx:
        @staticmethod
        def load_state():
            return {"owner_chat_id": 9}

        @staticmethod
        def send_with_budget(*args, **kwargs):
            sent.append((args, kwargs))

    _handle_cancel_task({"task_id": "cancel-me"}, _Ctx())

    assert seen == ["cancel-me"]
    assert sent == []


def test_cancel_event_drives_the_physical_retry_id_without_owner_presentation(monkeypatch) -> None:
    """The event names the physical id the tool resolved; the logical
    ``requested_task_id`` used to shape an owner notice and now shapes nothing."""
    import supervisor.queue as q
    from supervisor.events_runtime_controls import _drive_cancel_task_event as _handle_cancel_task

    captured = []
    monkeypatch.setattr(
        q,
        "drive_cancel_intent_scope",
        lambda task_id, **_kwargs: (
            captured.append(("driver", task_id)) or q.CANCEL_FAILED
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

    assert captured == [("driver", "physical-retry-7f3a")]


def test_routine_cancel_outcomes_publish_nothing_to_the_owner_chat(monkeypatch) -> None:
    """#624: custody publishes the settled truth itself (task_summary/salvage in
    the task's own thread; the tool result to the calling agent). A second host
    acknowledgement for cancelled / already-settled / not-found reached the
    GLOBAL owner chat as an untyped assistant row with no task identity — and
    for a pending-drop race the "had already finished" wording misstated the
    causal history. The event is a wake for ``drive_cancel_intent_scope`` and
    publishes nothing for any outcome, FAILED included (the test above)."""
    import supervisor.queue as q
    from supervisor.events_runtime_controls import _drive_cancel_task_event as _handle_cancel_task

    for outcome in (q.CANCEL_CANCELLED, q.CANCEL_ALREADY_SETTLED, q.CANCEL_NOT_FOUND, q.CANCEL_FAILED):
        sent = []
        seen = []
        monkeypatch.setattr(
            q, "drive_cancel_intent_scope",
            lambda task_id, **_kw: seen.append(task_id) or outcome,
        )

        class _Ctx:
            DRIVE_ROOT = pathlib.Path("/unused")

            @staticmethod
            def load_state():
                return {"owner_chat_id": 9}

            @staticmethod
            def send_with_budget(*args, **kwargs):
                sent.append((args, kwargs))

        _handle_cancel_task({"task_id": "child-A", "requested_task_id": "child-A"}, _Ctx())
        assert seen == ["child-A"], outcome  # custody still runs
        assert sent == [], (outcome, sent)   # no Main-chat speech, no untyped row

    # A missing task id never reaches the driver and still says nothing.
    sent = []
    monkeypatch.setattr(q, "drive_cancel_intent_scope", lambda task_id, **_kw: q.CANCEL_CANCELLED)

    class _Ctx2:
        @staticmethod
        def load_state():
            return {"owner_chat_id": 9}

        @staticmethod
        def send_with_budget(*args, **kwargs):
            sent.append((args, kwargs))

    _handle_cancel_task({"task_id": ""}, _Ctx2())
    assert sent == []
    # Custody owns presentation; this event handler must not send its own row.
    assert "send_with_budget(" not in inspect.getsource(_handle_cancel_task)
