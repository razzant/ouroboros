"""Cancellation waits for owned effects, then still terminates the caller."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from ouroboros.live_conversation import ConversationHost
from ouroboros.realtime_session import joined
from tests.test_live_conversation import host_world as host_world


@pytest.mark.parametrize('cancel', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_joined_preserves_cancellation_and_the_owned_outcome(cancel, fails):
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        settled = []
        async def work():
            entered.set()
            await release.wait()
            settled.append(True)
            if fails:
                raise ValueError('owned failure')
            return 'owned result'
        waiting = asyncio.create_task(joined(work()))
        await entered.wait()
        if cancel:
            waiting.cancel()
            await asyncio.sleep(0)
            waiting.cancel()
            await asyncio.sleep(0)
            assert not waiting.done() and not settled
        release.set()
        if fails:
            with pytest.raises(ValueError, match='owned failure'):
                await waiting
        elif cancel:
            with pytest.raises(asyncio.CancelledError):
                await waiting
        else:
            assert await waiting == 'owned result'
        assert settled == [True]
    asyncio.run(run())


def mailbox_host(monkeypatch, tmp_path):
    from ouroboros import loop_round_limits
    entered, release = threading.Event(), threading.Event()
    checkpoints = []
    row = {'role': 'user', 'content': 'Retain the completed drain.'}
    def drain(rows, *args, **kwargs):
        entered.set()
        assert release.wait(30)
        rows.append(row)
        return {}
    monkeypatch.setattr(loop_round_limits, '_drain_incoming_messages', drain)
    host = SimpleNamespace(incoming=object(), _tools_lock=asyncio.Lock(),
        ctx=SimpleNamespace(drive_root=tmp_path, task_id='mailbox-cancellation'),
        event_queue=None, owner_seen=set(), messages=[], schemas=[], _context_changed=lambda: None,
        checkpoint=checkpoints.append, model=SimpleNamespace(vision=True))
    host.mailbox = lambda: ConversationHost.mailbox(host)
    return host, entered, release, checkpoints, row


@pytest.mark.parametrize('cancel', [False, True])
def test_real_mailbox_joins_drain_retains_rows_and_propagates_cancel(monkeypatch, tmp_path, cancel):
    async def run():
        host, entered, release, checkpoints, row = mailbox_host(monkeypatch, tmp_path)
        task = asyncio.create_task(host.mailbox())
        try:
            assert await asyncio.to_thread(entered.wait, 30)
            if cancel:
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert await task == [row]
            assert host.messages == [row] and checkpoints == ['ready']
            assert host._pending_mail_answer, 'a committed drain owns its answer even when its waiter was cancelled'
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())


def test_feature_renewal_joins_cancelled_real_mailbox(monkeypatch, tmp_path):
    from skills.live_conversation.driver import _Call
    async def run():
        host, entered, release, checkpoints, row = mailbox_host(monkeypatch, tmp_path)
        transition = asyncio.Event()
        host.hub = SimpleNamespace(send_state=lambda *args: transition.set())
        class LocalCall(_Call):
            async def provider(self):
                await asyncio.Event().wait()
            async def peers(self):
                assert await asyncio.to_thread(entered.wait, 30)
                self.renewing = True
        host.state = {}
        call = LocalCall(host)
        before = set(asyncio.all_tasks())
        call.writer = asyncio.create_task(call.save_rows())
        consume = asyncio.create_task(call.consume(SimpleNamespace(final_captions=lambda: [], renewal_ready=True)))
        try:
            await asyncio.wait_for(transition.wait(), 30)
            await asyncio.sleep(0)  # let the first cancellation reach joined()
            release.set()
            done, _ = await asyncio.wait([consume], timeout=5)
            assert consume in done, 'renewal still joins a mailbox loop after its drain settled'
            assert await consume == ''
            assert host.messages == [row] and checkpoints == ['ready']
        finally:
            release.set()
            children = asyncio.all_tasks() - before
            for child in children:
                child.cancel()
            await asyncio.gather(*children, return_exceptions=True)
    asyncio.run(run())


@pytest.mark.parametrize('cancel', [False, True])
def test_caption_completes_history_handoff_before_cancel(host_world, monkeypatch, cancel):
    from ouroboros import dialogue_store
    host = host_world.host
    entered, release = threading.Event(), threading.Event()
    original = dialogue_store.append
    def append(*args, **kwargs):
        entered.set()
        assert release.wait(30)
        return original(*args, **kwargs)
    monkeypatch.setattr(dialogue_store, 'append', append)
    async def run():
        task = asyncio.create_task(host.caption('ouroboros', 'Retained speech', 'joined-caption', completed=True))
        try:
            assert await asyncio.to_thread(entered.wait, 30)
            if cancel:
                task.cancel()
            release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
            assert ('ouroboros', 'joined-caption') in host.captioned
            assert host.messages[-1]['content'] == 'Retained speech' and host._caption_dirty
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
    asyncio.run(run())
