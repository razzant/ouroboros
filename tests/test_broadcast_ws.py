from __future__ import annotations

import asyncio
import json


class _DeadWebSocket:
    async def send_text(self, _text):
        raise RuntimeError("dead client")


def test_broadcast_partial_failure_uses_module_data_dir(tmp_path, monkeypatch):
    from ouroboros.gateway import ws

    monkeypatch.delenv("OUROBOROS_DATA_DIR", raising=False)
    monkeypatch.setattr(ws, "DATA_DIR", tmp_path)

    with ws._ws_lock:
        original_clients = list(ws._ws_clients)
        ws._ws_clients.clear()
        ws._ws_clients.append(_DeadWebSocket())
    try:
        asyncio.run(ws.broadcast_ws({"type": "unit_test"}))
    finally:
        with ws._ws_lock:
            ws._ws_clients.clear()
            ws._ws_clients.extend(original_clients)

    events_path = tmp_path / "logs" / "events.jsonl"
    rows = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["type"] == "broadcast_partial_failure"
    assert rows[-1]["msg_type"] == "unit_test"
    assert rows[-1]["dead_clients"] == 1


def test_a_closed_loop_does_not_leak_the_unawaited_broadcast_coroutine(monkeypatch):
    """`broadcast_ws_sync` built the coroutine BEFORE handing it to
    `run_coroutine_threadsafe` and swallowed the RuntimeError a closed or
    stopped loop raises — dropping the coroutine object on the floor. Nothing
    ever awaited it, so CPython emitted `RuntimeWarning: coroutine
    'broadcast_ws' was never awaited` from whatever code happened to be
    running when the object was collected. That is how the warning surfaced
    inside unrelated tests (tests/test_extensions_api.py::test_api_extension*,
    where an earlier test left a finished loop in the module global): a leak
    reported against an innocent bystander. On the pre-fix code that mechanism
    reproduces only under the CI-shaped FULL run — not from that file alone,
    and not on every full run: the loop is left behind deterministically (by
    the three `with TestClient(srv.app)` lifespan tests — `server.lifespan` is
    the only caller of `ws.set_event_loop` and never resets it to None; this
    test's own monkeypatch write is restored), but whether a broadcast then fires, and
    which test is running when the object is collected, is not. Hence a pin on
    the leaking line itself rather than on the bystander.

    The stale-loop case itself is deliberate: a broadcast with no live loop is
    a no-op, not an error.
    """
    import warnings

    from ouroboros.gateway import ws

    loop = asyncio.new_event_loop()
    loop.close()
    monkeypatch.setattr(ws, "_event_loop", loop)

    # `record=True` + "always", not "error": the warning is raised from the
    # coroutine's __del__, so turning it into an exception makes it UNRAISABLE
    # (a pytest side note nobody fails on) instead of a caught assertion.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ws.broadcast_ws_sync({"type": "unit_test_closed_loop"})

    assert [str(w.message) for w in caught] == []
