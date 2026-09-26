"""TZ-2 B3: time facts are projections, not invented calendar limits."""

import datetime
import time

from ouroboros import config, context, model_wait


def test_started_and_optional_ceiling_are_explicitly_sourced(tmp_path, monkeypatch):
    task = {"id": "clock", "started_at": time.time() - 10}
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: None)
    unlimited = context.task_execution_clock_fact(task, None)
    assert unlimited["started_at"] and unlimited["absolute_ceiling_at"] is None
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 100)
    estimated = context.task_execution_clock_fact(task, None)
    remaining = (datetime.datetime.fromisoformat(estimated["absolute_ceiling_at"])
                 - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert 87 <= remaining <= 93
    assert "may move" in estimated["absolute_ceiling_at_basis"]
    assert context.task_execution_clock_fact({"id": "clock"}, None)["absolute_ceiling_at"] is None
    with model_wait.task_model_wait_scope(task=task, drive_root=tmp_path, event_queue=None,
                                          worker_slot_held=True) as owner:
        monkeypatch.setattr(owner, "executed_seconds", lambda **_kwargs: 7)
        live = context.task_execution_clock_fact(task, None)
    live_remaining = (datetime.datetime.fromisoformat(live["absolute_ceiling_at"])
                      - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert 91 <= live_remaining <= 95
