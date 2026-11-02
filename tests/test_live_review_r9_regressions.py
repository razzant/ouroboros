"""Regressions from review round R9 (astra, 22.10): synthetic provider ports and a disposable root only."""
import asyncio
import errno
import json
import threading

import pytest

from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('phase', ['tool_job', 'executor'])
@pytest.mark.parametrize('log_fails', [False, True])
def test_cancelled_finish_attempts_timing_after_custody(host_world, monkeypatch, phase, log_fails):
    """R9-3: the call.timing evidence line is attempted after every custody step, whichever await the
    cancellation lands on, and a failed write never replaces the cancellation."""
    host = host_world.host
    attempted = []
    settled = []
    observe = host.observe
    def observe_timing(direction, raw):
        if json.loads(raw).get('type') == 'call.timing':
            assert settled
            attempted.append(True)
            if log_fails:
                raise OSError(errno.ENOSPC, 'synthetic timing-only write failure')
        return observe(direction, raw)
    async def scenario():
        entered, release = asyncio.Event(), asyncio.Event()
        entered_thread, release_thread = threading.Event(), threading.Event()
        if phase == 'tool_job':
            async def job():
                entered.set()
                await release.wait()
                settled.append(True)
            host.tool_jobs.add(asyncio.create_task(job()))
        else:
            shutdown = host.executor.shutdown
            def slow_shutdown(*args, **kwargs):
                entered_thread.set()
                assert release_thread.wait(5)
                result = shutdown(*args, **kwargs)
                settled.append(True)
                return result
            monkeypatch.setattr(host.executor, 'shutdown', slow_shutdown)
        monkeypatch.setattr(host, 'observe', observe_timing)
        finishing = asyncio.create_task(host.finish())
        try:
            if phase == 'tool_job':
                await asyncio.wait_for(entered.wait(), 5)
            else:
                assert await asyncio.to_thread(entered_thread.wait, 5)
            finishing.cancel()
            await asyncio.sleep(0)
        finally:
            release.set()
            release_thread.set()
        with pytest.raises(asyncio.CancelledError):
            await finishing
        assert settled and attempted == [True]
    asyncio.run(scenario())
