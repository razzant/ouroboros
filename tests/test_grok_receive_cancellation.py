"""A meter read completing at teardown must not swallow the reader's cancellation."""
import asyncio

import pytest

from ouroboros.realtime_session import JsonSession
from skills.grok_realtime.plugin import GrokSession as Session


@pytest.mark.parametrize('cancel', [False, True])
def test_meter_read_completion_cannot_outlive_cancelled_connection(cancel):
    async def run():
        session = object.__new__(Session)
        session.pending = []
        async def advance(): pass
        session.advance_context = advance
        session.convert = lambda event: [event]
        async def read():
            if cancel:
                asyncio.get_running_loop().call_soon(task.cancel)
            return {'synthetic':'native row ready at teardown'}
        session.read = read
        task = asyncio.create_task(session.receive())
        if cancel:
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert await task == [{'synthetic':'native row ready at teardown'}]
    asyncio.run(run())


def test_timed_out_meter_read_is_cancelled_and_joined():
    async def run():
        session = object.__new__(JsonSession)
        stopped = asyncio.Event()
        async def read():
            try: await asyncio.Future()
            finally: stopped.set()
        session.read = read
        with pytest.raises(asyncio.TimeoutError):
            await session.timed_read(.001)
        assert stopped.is_set()
    asyncio.run(run())
