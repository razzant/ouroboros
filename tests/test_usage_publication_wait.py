"""Elapsed bounds still apply to real storage contention."""
import asyncio
from types import SimpleNamespace

import pytest

from ouroboros import _usage_wait, config, model_wait, queue_observation, store
from ouroboros.llm_attempt import PhysicalDispatchInterrupted


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    owner = SimpleNamespace(task={}, control_reason=lambda: "")
    monkeypatch.setattr(_usage_wait, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(model_wait, "current_model_wait", lambda: owner)
    monkeypatch.setattr(model_wait, "dispatch_deadline_remaining_sec", lambda: None)
    monkeypatch.setattr(_usage_wait, "_check", lambda: None)
    monkeypatch.setattr(_usage_wait, "_hold", lambda *_args: None)
    return now, owner


def _sleep(monkeypatch, callback):
    monkeypatch.setattr(queue_observation, "_sleep", callback)

    async def sleeping(seconds):
        callback(seconds)

    monkeypatch.setattr(_usage_wait.asyncio, "sleep", sleeping)


def _retry(mode, callback):
    if mode == "async":
        return asyncio.run(_usage_wait.retry_presend_async(callback, check=False))
    return _usage_wait.retry_presend(callback, check=False)


@pytest.mark.parametrize(("bound", "mode"), [("interactive", "async"), ("unowned", "sync")])
def test_other_real_elapsed_bounds_still_apply_during_storage_contention(monkeypatch, clock, mode, bound):
    now, owner = clock
    if bound == "interactive":
        owner.task["_is_direct_chat"] = True
        monkeypatch.setattr(config, "get_task_idle_timeout_sec", lambda: 2.0)
    else:
        monkeypatch.setattr(model_wait, "current_model_wait", lambda: None)
        monkeypatch.setattr(_usage_wait, "USAGE_LOCK_TIMEOUT_SEC", 2.0)
    error = store.StoreBusy("same storage contention", reason="contention")

    def probe():
        now[0] += 3
        raise error

    _sleep(monkeypatch, lambda _: pytest.fail("the existing elapsed bound must end this wait"))
    if bound == "interactive":
        with pytest.raises(PhysicalDispatchInterrupted, match="accounting_wait_expired"):
            _retry(mode, probe)
    else:
        with pytest.raises(store.StoreBusy) as caught:
            _retry(mode, probe)
        assert caught.value is error
