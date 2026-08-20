"""Owner-facing heartbeat presentation regressions for v6.64."""

from __future__ import annotations

import inspect
import json
import pathlib


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


def test_retired_timeout_defaults_are_quiet_but_custom_value_is_loud(tmp_path, monkeypatch) -> None:
    from supervisor import queue

    monkeypatch.setattr(queue, "_timeout_deprecation_emitted", False)
    queue.init(tmp_path)
    events = tmp_path / "logs" / "events.jsonl"
    assert not events.exists()

    monkeypatch.setenv("OUROBOROS_SOFT_TIMEOUT_SEC", "601")
    queue.init(tmp_path)
    row = json.loads(events.read_text(encoding="utf-8"))
    assert row["type"] == "deprecated_settings_ignored"
    assert row["keys"] == ["OUROBOROS_SOFT_TIMEOUT_SEC"]


def test_the_deprecation_notice_now_reads_the_environment_not_a_parameter(
    tmp_path, monkeypatch,
) -> None:
    """`load_settings` drops every retired key before a reader sees it, so a
    settings document can no longer carry a non-default value and the parameters
    that used to ferry one were always the two defaults. The environment is the
    only surviving source, so that is where the notice looks."""
    import json

    from supervisor import queue

    monkeypatch.setattr(queue, "_timeout_deprecation_emitted", False)
    for key, default in queue.RETIRED_LIVENESS_ENV_DEFAULTS:
        monkeypatch.setenv(key, default)
    queue.init(tmp_path)
    events = tmp_path / "logs" / "events.jsonl"
    assert not events.exists(), "the defaults must stay quiet"

    monkeypatch.setattr(queue, "_timeout_deprecation_emitted", False)
    monkeypatch.setenv("OUROBOROS_HARD_TIMEOUT_SEC", "1801")
    queue.init(tmp_path)
    row = json.loads(events.read_text(encoding="utf-8"))
    assert row["type"] == "deprecated_settings_ignored"
    assert row["keys"] == ["OUROBOROS_HARD_TIMEOUT_SEC"]


def test_a_reload_no_longer_probes_the_settings_document_for_retired_keys() -> None:
    """A document cannot answer either way once the key is stripped from it; a
    probe there would be a question with no possible answer."""
    import inspect

    from supervisor import queue

    source = inspect.getsource(queue.refresh_timeouts_from_settings)
    for key, _default in queue.RETIRED_LIVENESS_ENV_DEFAULTS:
        assert key not in source, key
    assert "get_finalization_grace_sec(settings)" in source


def test_retired_planning_heartbeat_default_is_quiet_but_custom_value_is_loud(
    tmp_path, monkeypatch,
) -> None:
    """The planning-scout heartbeat knob is retired: ``RETIRED_SETTING_KEYS`` strips it
    from the settings document before any reader sees it, so the ENVIRONMENT is the only
    surviving source of a value. That is what the notice reads, and it distinguishes the
    two cases honestly — the shipped default stays quiet (nothing was customized), while
    a non-default env value still earns exactly one ``deprecated_settings_ignored`` row
    naming the key. Silence for a value the owner really set would lose the fact that a
    knob they tuned no longer does anything."""
    from supervisor import queue

    monkeypatch.setattr(queue, "_timeout_deprecation_emitted", False)
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC", "120")
    queue.init(tmp_path)
    events = tmp_path / "logs" / "events.jsonl"
    assert not events.exists()

    monkeypatch.setattr(queue, "_timeout_deprecation_emitted", False)
    monkeypatch.setenv("OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC", "121")
    queue.init(tmp_path)
    row = json.loads(events.read_text(encoding="utf-8"))
    assert row["type"] == "deprecated_settings_ignored"
    assert row["keys"] == ["OUROBOROS_PLAN_TASK_SWARM_HEARTBEAT_STALE_SEC"]


def test_status_text_names_the_live_rails_and_not_the_retired_numbers(tmp_path, monkeypatch) -> None:
    """The owner status line printed `soft=600s, hard=1800s` beside a note that
    they were ignored — two numbers no rail has read since idle/deadline/ceiling/
    reaper replaced them. Naming the live rails is the whole truth; the numbers
    were an invitation to tune something that does not exist."""
    from supervisor import state

    monkeypatch.setattr(state, "load_state", lambda: {"owner_id": 1, "session_id": "s"})
    monkeypatch.setattr(state, "budget_remaining", lambda: (0.0, 0.0, 0.0))
    text = state.status_text({}, [], {})

    assert "active_liveness: idle+deadline+absolute_ceiling+reaper" in text
    assert "soft=" not in text and "hard=" not in text
    assert "legacy_timeouts_ignored" not in text
    # ...and the two arguments are gone from the signature, not merely ignored.
    assert set(inspect.signature(state.status_text).parameters) == {
        "workers_dict", "pending_list", "running_dict"}


def test_the_worker_pool_keeps_no_copy_of_the_retired_or_budget_globals() -> None:
    """Three module globals nothing read: two retired liveness keys and a third
    copy of the budget limit whose live authority is supervisor.state. A global
    that is written and never read reads as configuration to the next person."""
    from supervisor import workers

    for name in ("SOFT_TIMEOUT_SEC", "HARD_TIMEOUT_SEC", "TOTAL_BUDGET_LIMIT"):
        assert not hasattr(workers, name), name
    source = inspect.getsource(workers.init)
    for name in ("SOFT_TIMEOUT_SEC", "HARD_TIMEOUT_SEC", "TOTAL_BUDGET_LIMIT"):
        assert name not in source, name
    # ...and no longer accepts them either: the pool binds what it reads.
    parameters = set(inspect.signature(workers.init).parameters)
    assert parameters == {"repo_dir", "drive_root", "max_workers",
                          "branch_dev", "branch_stable"}
    assert "queue.init(drive_root)" in source


def test_the_queue_never_rebinds_the_retired_timeout_constants() -> None:
    """They are constants, not state: whatever settings carry, the rails see the
    same two numbers, so ``init`` must not perform a binding that suggests it
    could be otherwise."""
    from supervisor import queue

    source = inspect.getsource(queue.init)
    assert "global DRIVE_ROOT, FINALIZATION_GRACE_SEC" in source
    assert set(inspect.signature(queue.init).parameters) == {"drive_root"}
    for name in ("SOFT_TIMEOUT_SEC", "HARD_TIMEOUT_SEC"):
        assert not hasattr(queue, name), name


def test_the_queue_snapshot_path_has_one_owner(tmp_path, monkeypatch) -> None:
    """The queue kept a private copy of the snapshot path, rebound by ITS init.
    Two inits, two answers: whichever ran last for a given root won, and a test or
    boot path that ran only one of them read or wrote the wrong file. The path
    belongs to supervisor.state with the other state paths; the queue reads it
    through the module at use time."""
    import inspect

    from supervisor import queue, state

    assert not hasattr(queue, "QUEUE_SNAPSHOT_PATH")
    assert "QUEUE_SNAPSHOT_PATH" not in inspect.getsource(queue.init)
    for reader in (queue.persist_queue_snapshot, queue.restore_pending_from_snapshot):
        assert "_state.QUEUE_SNAPSHOT_PATH" in inspect.getsource(reader), reader.__name__

    # Rebinding the one owner is what the readers see — at use time, not import time.
    monkeypatch.setattr(state, "QUEUE_SNAPSHOT_PATH", tmp_path / "state" / "snap.json")
    monkeypatch.setattr(queue, "PENDING", [])
    monkeypatch.setattr(queue, "RUNNING", {})
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    assert queue.persist_queue_snapshot(reason="one-owner-probe") is True
    assert (tmp_path / "state" / "snap.json").is_file()


def test_the_bench_container_no_longer_carries_the_retired_liveness_keys() -> None:
    """A forwarded no-op key makes a benchmark run look configured when it is not."""
    repo = pathlib.Path(__file__).resolve().parents[1]
    source = (repo / "devtools" / "benchmarks" / "terminal_bench"
              / "harbor_installed_agent.py").read_text("utf-8")
    assert '"OUROBOROS_SOFT_TIMEOUT_SEC"' not in source
    assert '"OUROBOROS_HARD_TIMEOUT_SEC"' not in source
    assert '"OUROBOROS_TOOL_TIMEOUT_SEC"' in source  # the live one stays


def test_owner_visible_incidents_use_canonical_message_seam() -> None:
    repo = pathlib.Path(__file__).resolve().parents[1]
    for relpath in ("server.py", "supervisor/workers.py"):
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
