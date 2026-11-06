"""Slow read-only consumers must leave the server loop available to other work."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from ouroboros.gateway import control, logs, schedules




@pytest.mark.parametrize("surface", ["updates", "logs", "schedules"])
def test_slow_reader_does_not_hold_event_loop(monkeypatch, tmp_path, surface):
    entered, release = threading.Event(), threading.Event()
    request = SimpleNamespace(path_params={"name": "events"}, query_params={},
                              app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)))

    def blocked_read(*_args, **_kwargs):
        entered.set()
        assert release.wait(2), "reader blocked the event loop before it could release the read"
        return [] if surface == "logs" else {"available": False}

    if surface == "updates":
        monkeypatch.setattr(control, "_managed_update_payload", blocked_read)
        endpoint = control.api_update_status
    elif surface == "logs":
        monkeypatch.setattr(logs, "read_rotated_jsonl_entries", blocked_read)
        endpoint = logs.api_logs_tail
    else:
        from supervisor import followup_policy, queue

        monkeypatch.setattr(queue, "load_schedule_store", lambda _root: {})
        monkeypatch.setattr(followup_policy, "observed_store", lambda _root, store: store)
        monkeypatch.setattr(queue, "schedule_activity_projection", blocked_read)
        endpoint = schedules.api_schedules_list

    async def scenario():
        read = asyncio.create_task(endpoint(request))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            # This coroutine must progress while the real endpoint is still reading.
            assert not read.done()
        finally:
            release.set()
        response = await read
        assert response.status_code == 200
        assert json.loads(response.body) == (
            {"name": "events", "entries": []} if surface == "logs" else {"available": False}
        )

    asyncio.run(scenario())


def test_offloaded_schedules_keep_unreadable_store_status(monkeypatch, tmp_path):
    from supervisor import queue

    def unreadable(_root):
        raise queue.ScheduleStoreUnreadable("schedule data is unreadable")

    monkeypatch.setattr(queue, "load_schedule_store", unreadable)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(drive_root=tmp_path)))
    response = asyncio.run(schedules.api_schedules_list(request))
    assert response.status_code == 503
    assert "unreadable" in json.loads(response.body)["error"]
