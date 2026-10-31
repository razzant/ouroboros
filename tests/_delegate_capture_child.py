"""Exercise maintenance's real addressed job with an in-process transport for barrier tests.

The native subprocess boundary has its own test. This transport lets recovery tests
control exact capture/Stop interleavings without a daemon or an unrelated history job.
"""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def capture_child(monkeypatch):
    from ouroboros import delegate_capture_job as job, delegate_collection as body, maintenance
    from ouroboros.tools import delegate_integration

    threads, errors = {}, []
    original, capture = job.request_or_result, delegate_integration.capture_terminal_patch_for_drive

    def with_transport(root, entry, **kwargs):
        runtime = body._RUNTIMES.get(str(root.resolve()))
        if (kwargs.get("publish") is False and entry.resource_ref.get("workspace_kind") == "directory"
                and runtime is not None and runtime.gateway_factory is not None):
            kwargs["gateway"] = runtime.gateway_factory()
        return capture(root, entry, **kwargs)

    def run(root, runtime):
        end = time.monotonic() + 15
        try:
            while job.due(root, {}, {}, True) and (runtime is None or runtime.active):
                maintenance.run_job(root, job.NAME, should_stop=lambda: runtime is not None and not runtime.active)
                if time.monotonic() >= end:
                    raise AssertionError("capture fixture did not release its result lock")
                time.sleep(.01)
        except BaseException as exc:
            errors.append(exc)

    def requested(root, entry, **kwargs):
        result = original(root, entry, **kwargs)
        if result.get("status") == "pending" and not (root in threads and threads[root].is_alive()):
            runtime = body._RUNTIMES.get(str(root.resolve()))
            thread = threads[root] = threading.Thread(target=run, args=(root, runtime))
            thread.start()
        return result

    def drain(root):
        thread = threads.pop(root, None)
        if thread is None:
            return False
        thread.join(20)
        assert not thread.is_alive()
        if errors:
            raise errors.pop(0)
        return True

    monkeypatch.setattr(maintenance.child, "request", lambda: None)
    monkeypatch.setattr(job, "request_or_result", requested)
    monkeypatch.setattr(delegate_integration, "capture_terminal_patch_for_drive", with_transport)
    yield SimpleNamespace(drain=drain, threads=threads)
    for root in list(threads):
        drain(root)
    assert not errors
