"""A synchronous extension WS handler never stalls the ASGI event loop.

#1195 F3 review follow-up: `ws._dispatch_extension_message` mirrors the HTTP
dispatcher — a coroutine handler is awaited, a synchronous one (including its
synchronous execution-barrier wait) runs in a worker thread. The proof is a
heartbeat coroutine that keeps ticking while the handler is blocked.
"""
from __future__ import annotations

import asyncio
import builtins
import json
import threading

from ouroboros import extension_loader
from tests._extension_loader_shared import _prepare_extension
from tests._extension_loader_shared import (  # noqa: F401  (autouse fixture applies on import)
    _clear_loader_state,
)

FAST_WS_PLUGIN = (
    "def _fast(msg):\n"
    "    return {'ok': True}\n"
    "def register(api):\n"
    "    api.register_ws_handler('fast', _fast)\n"
)


BLOCKING_WS_PLUGIN = (
    "import builtins\n"
    "def _slow(msg):\n"
    "    builtins._ouro_ws_entered.set()\n"
    "    released = builtins._ouro_ws_release.wait(timeout=2)\n"
    "    return {'released': released}\n"
    "def register(api):\n"
    "    api.register_ws_handler('slow', _slow)\n"
)


def test_sync_ws_handler_runs_off_the_event_loop(tmp_path, monkeypatch):
    from ouroboros.gateway import ws as ws_module

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "ws_off_loop", BLOCKING_WS_PLUGIN, permissions=["ws_handler"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))
    builtins._ouro_ws_entered = threading.Event()
    builtins._ouro_ws_release = threading.Event()
    sent: list = []

    class _FakeApp:
        class state:  # noqa: N801
            drive_root = None

    class _FakeWebSocket:
        def __init__(self, app):
            self.app = app

        async def send_text(self, text):
            sent.append(json.loads(text))

    app = _FakeApp()
    app.state.drive_root = drive_root
    msg_type = extension_loader.extension_surface_name(loaded.name, "slow")

    async def main():
        ticks = 0
        dispatch = asyncio.create_task(
            ws_module._dispatch_extension_message(_FakeWebSocket(app), {"type": msg_type}, msg_type)
        )
        # The handler is standing inside its blocking wait; the loop must still turn.
        await asyncio.to_thread(builtins._ouro_ws_entered.wait, 5)
        for _ in range(20):
            await asyncio.sleep(0.01)
            ticks += 1
        builtins._ouro_ws_release.set()
        assert await asyncio.wait_for(dispatch, 5) is True
        return ticks

    try:
        ticks = asyncio.run(main())
    finally:
        del builtins._ouro_ws_entered
        del builtins._ouro_ws_release
        extension_loader.unload_extension(loaded.name)
    # The release is set by the loop AFTER ticking; a handler that had frozen the
    # loop could never see it and would report the timeout instead.
    assert ticks == 20
    assert sent and sent[-1]["type"] == msg_type + ".reply", sent
    assert sent[-1]["data"] == {"released": True}, "the event loop was blocked while the sync handler ran"


def test_ws_model_cost_disclosure_runs_off_the_event_loop(tmp_path, monkeypatch):
    """Disclosure reads settings under the money lock; it must not freeze the socket loop."""
    from ouroboros.gateway import ws as ws_module

    loaded, repo_root, drive_root = _prepare_extension(
        tmp_path, "ws_disclose_off_loop", FAST_WS_PLUGIN, permissions=["ws_handler"],
    )
    monkeypatch.setenv("OUROBOROS_SKILLS_REPO_PATH", str(repo_root))

    def blocking_disclose(*_args, **_kwargs):
        builtins._ouro_ws_entered.set()
        assert builtins._ouro_ws_release.wait(timeout=2)

    monkeypatch.setattr(
        "ouroboros.extension_process_runner.disclose_inprocess_extension_dispatch",
        blocking_disclose,
    )
    builtins._ouro_ws_entered = threading.Event()
    builtins._ouro_ws_release = threading.Event()
    sent: list = []

    class _FakeApp:
        class state:  # noqa: N801
            drive_root = None

    class _FakeWebSocket:
        def __init__(self, app):
            self.app = app

        async def send_text(self, text):
            sent.append(json.loads(text))

    app = _FakeApp()
    app.state.drive_root = drive_root
    msg_type = extension_loader.extension_surface_name(loaded.name, "fast")

    async def main():
        ticks = 0
        dispatch = asyncio.create_task(
            ws_module._dispatch_extension_message(_FakeWebSocket(app), {"type": msg_type}, msg_type)
        )
        await asyncio.to_thread(builtins._ouro_ws_entered.wait, 5)
        for _ in range(20):
            await asyncio.sleep(0.01)
            ticks += 1
        builtins._ouro_ws_release.set()
        assert await asyncio.wait_for(dispatch, 5) is True
        return ticks

    try:
        ticks = asyncio.run(main())
    finally:
        del builtins._ouro_ws_entered
        del builtins._ouro_ws_release
        extension_loader.unload_extension(loaded.name)
    assert ticks == 20
    assert sent and sent[-1]["data"] == {"ok": True}
