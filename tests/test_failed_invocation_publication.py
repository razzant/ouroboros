"""Failed custody publication joins physical work, not passive bookkeeping."""
from concurrent.futures import Future, ThreadPoolExecutor
import threading

import pytest

from ouroboros import budget_pause, loop_tool_execution, owner_pause, tool_custody
from tests.test_batch4_producer_custody import _registry
from tests.test_invocation_ownership import owner, run



@pytest.mark.parametrize("real_hold", [False, True])
def test_failed_publication_joins_body_and_real_holds_without_passive_callback(tmp_path, monkeypatch, real_hold):
    registry, queue, _ = _registry(tmp_path, monkeypatch)
    entered, finish_body, failed, passive_entered, release_passive, release_cleanup = [
        threading.Event() for _ in range(6)]
    real_add = Future.add_done_callback
    cleanup_threads, effects = [], []

    def delayed_done(self, callback):
        if getattr(callback, "__name__", "") == "done":
            def delayed(value):
                passive_entered.set()
                assert release_passive.wait(10)
                callback(value)
            return real_add(self, delayed)
        return real_add(self, callback)

    def body(*_args, **_kwargs):
        if real_hold:
            release = tool_custody.hold_invocation()
            def cleanup():
                try:
                    assert release_cleanup.wait(10)
                    effects.append("cleanup completed")
                finally:
                    release()
            thread = threading.Thread(target=cleanup)
            cleanup_threads.append(thread)
            thread.start()
        entered.set()
        assert finish_body.wait(10)
        return "physically complete"

    def interrupted(*_args):
        assert entered.wait(5)
        raise RuntimeError("caller interrupted before result")

    def unavailable(*_args, **_kwargs):
        failed.set()
        raise OSError("residual publication unavailable")

    registry.override_handler("knowledge_read", body)
    monkeypatch.setattr(Future, "add_done_callback", delayed_done)
    monkeypatch.setattr(loop_tool_execution, "future_result", interrupted)
    monkeypatch.setattr(tool_custody, "retain_invocation", unavailable)

    def call():
        with owner(registry):
            return run(registry)

    with ThreadPoolExecutor(1) as executor:
        caller = executor.submit(call)
        try:
            assert failed.wait(5)
            # Catch the original bad ordering: the caller pin is already gone,
            # but the body has not ended when the failed-publication wait begins.
            for _ in range(1000):
                with budget_pause._TOOL_LOCK:
                    rows = budget_pause._TOOL_FUTURES[budget_pause.tool_scope_key(registry._ctx)]
                    row = next(iter(rows.values()))
                    released = row["holds"] == int(real_hold)
                if released:
                    break
                threading.Event().wait(.005)
            assert released and not row["future"].done()
            with queue._queue_lock, owner_pause.launch_lock(tmp_path, "root", wait=False):
                pass
            finish_body.set()
            assert passive_entered.wait(5) and row["future"].done()
            assert not release_passive.is_set() and not row["done"]
            if real_hold:
                assert row["holds"] == 1 and not caller.done()
                assert effects == [] and not row["settled"].is_set()
                release_cleanup.set()
                cleanup_threads[0].join(5)
                assert effects == ["cleanup completed"] and not cleanup_threads[0].is_alive()
            with pytest.raises(RuntimeError, match="caller interrupted before result"):
                caller.result(timeout=2)
            assert row["settled"].is_set() and row["holds"] == 0
            assert not release_passive.is_set(), "return must not need the passive callback"
        finally:
            finish_body.set()
            release_cleanup.set()
            release_passive.set()
            for thread in cleanup_threads:
                thread.join(5)
            try:
                caller.result(timeout=5)
            except RuntimeError:
                pass
