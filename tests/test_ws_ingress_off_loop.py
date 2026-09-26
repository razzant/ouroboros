"""A held web chat acceptance never stalls the ASGI loop (TZ-1 PR1 review F1).

``ws_endpoint`` hands an owner chat frame to ``bridge.ui_send``. That acceptance
takes the single host's ingress lock and writes the canonical ``chat.jsonl`` row
through a locked durable append BEFORE it enqueues and echoes. A skill delivery
scanning retained chat under that lock, or a slow disk, used to stall the whole
event loop with it: every other HTTP and WebSocket handler waited, Stop and
Panic included. The acceptance now runs off the loop through
``gateway._helpers.run_sync_to_completion`` and keeps its custody: the row, the
queue item and the echo still complete in that order, even when the socket task
is cancelled while the acceptance stands on the lock. A later chat frame on
this socket waits its turn, but the receive loop never waits: the SPA sends
Panic and Restart on the SAME socket as chat, so a command frame is admitted
inline while that socket's chat acceptance is still held (review F2).

A late quiz answer (``POST /api/decisions`` on an expired card) is forwarded
through the named ingress, which waits for that same lock — behind a socket
acceptance that now holds it off the loop. Its wait runs off the loop too, with
the same custody: the decision request's cancellation settles row → queue →
echo first, and a retry rejoins the one delivery.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time

import anyio
import pytest
from starlette.applications import Starlette
from starlette.routing import Route, WebSocketRoute
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from ouroboros.gateway.state import api_health
from ouroboros.gateway.ws import ws_endpoint
from supervisor import message_bus


@pytest.fixture()
def bridge(tmp_path, monkeypatch):
    """A real LocalChatBridge on a pytest drive, registered as THE host bridge."""
    drive = tmp_path / "drive"
    (drive / "logs").mkdir(parents=True)
    bridge = message_bus.LocalChatBridge({})
    monkeypatch.setattr(message_bus, "_BRIDGE", bridge)
    monkeypatch.setattr(message_bus, "DATA_DIR", drive)
    monkeypatch.setattr(message_bus, "load_state", lambda: {})
    bridge.drive = drive
    return bridge


def _chat_frame(client_message_id: str, text: str = "hello") -> str:
    return json.dumps({"type": "chat", "content": text, "client_message_id": client_message_id})


def _row_ids(drive) -> list[str]:
    path = drive / "logs" / "chat.jsonl"
    if not path.exists():
        return []
    return [json.loads(line)["client_message_id"]
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _witness_echoes(bridge):
    """Record what the bridge can see at echo time: queue depth and rows on disk."""
    echoes: list[tuple[dict, int, list[str]]] = []
    bridge._broadcast_fn = lambda payload: echoes.append((payload, bridge._inbox.qsize(), _row_ids(bridge.drive)))
    return echoes


def _gate_ui_send(bridge, entered: threading.Event, release: threading.Event | None = None):
    """Flag the moment the socket hands the frame to the bridge (an instance attribute:
    the socket resolves ``bridge.ui_send`` at call time, so the real method still runs)."""
    original = bridge.ui_send

    def ui_send(text, **kwargs):
        entered.set()
        if release is not None:
            assert release.wait(5), "the test never released the held acceptance"
        return original(text, **kwargs)

    bridge.ui_send = ui_send


def test_a_held_ingress_lock_never_stalls_other_handlers(bridge):
    """Consumer regression: with the ingress lock held by another writer, an
    owner chat frame is standing in its acceptance; an unrelated HTTP handler
    and another socket's command frame must still be served meanwhile."""
    held, release, timed_out, entered = (threading.Event() for _ in range(4))
    echoes = _witness_echoes(bridge)
    _gate_ui_send(bridge, entered)

    def hold_ingress():  # a skill delivery mid-scan under the single host's ingress lock
        with message_bus._INGRESS_LOCK:
            held.set()
            if not release.wait(timeout=6.0):
                timed_out.set()

    holder = threading.Thread(target=hold_ingress, name="ingress-holder", daemon=True)
    holder.start()
    assert held.wait(5)
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint), Route("/api/health", api_health)])
    try:
        # One client context = one portal: the sockets and the GET share ONE event loop.
        with TestClient(app) as client, client.websocket_connect("/ws") as owner, \
                client.websocket_connect("/ws") as other:
            owner.send_text(_chat_frame("held-1"))
            assert entered.wait(5), "the chat frame never reached the bridge"
            # The acceptance is standing on the held lock. The loop must still serve
            # an unrelated HTTP handler and another socket's command (the Stop/Panic rail).
            started = time.monotonic()
            response = client.get("/api/health")
            elapsed = time.monotonic() - started
            assert response.status_code == 200
            assert not timed_out.is_set() and elapsed < 3.0, (
                f"the event loop was blocked behind the held ingress lock (health answered after {elapsed:.2f}s)")
            other.send_text(json.dumps({"type": "command", "cmd": "/stop"}))
            deadline = time.monotonic() + 3.0
            while bridge._inbox.qsize() < 1 and time.monotonic() < deadline:
                time.sleep(0.01)
            assert bridge._inbox.qsize() == 1, "a command frame queued behind the held owner ingress"
            assert not timed_out.is_set()
            assert echoes == [] and _row_ids(bridge.drive) == []  # nothing accepted while the lock is held
            release.set()
            holder.join(5)
            deadline = time.monotonic() + 5.0
            while not echoes and time.monotonic() < deadline:
                time.sleep(0.01)
    finally:
        release.set()
    # Custody order survived the off-loop hop: the row was on disk and the item
    # queued (behind the command that never waited) before the echo went out.
    assert [(q, ids) for _payload, q, ids in echoes] == [(2, ["held-1"])], echoes
    assert echoes[0][0]["ingress_accepted"] is True and echoes[0][0]["client_message_id"] == "held-1"
    updates = bridge.get_updates(offset=0, timeout=1)
    assert [u["message"]["text"] for u in updates] == ["/stop", "hello"]
    assert updates[1]["message"]["accepted_source_row"]["client_message_id"] == "held-1"


class _OpenSocket:
    """A socket that delivers its frames, then stays open forever."""

    app = None  # never consulted for a chat frame

    def __init__(self, frames):
        self.frames = list(frames)
        self.sent: list[dict] = []

    async def accept(self):
        return None

    async def receive_text(self):
        if self.frames:
            return self.frames.pop(0)
        await asyncio.Event().wait()

    async def send_text(self, text):
        self.sent.append(json.loads(text))


def test_a_cancelled_socket_task_still_settles_the_held_acceptance(bridge):
    """Cancel the socket task while the acceptance is held: the canonical row,
    the queue item and the echo still complete, in that order, before the
    cancellation is observed — custody is never abandoned mid-write."""
    entered, release = threading.Event(), threading.Event()
    echoes = _witness_echoes(bridge)
    _gate_ui_send(bridge, entered, release)
    socket = _OpenSocket([_chat_frame("cancel-1", "settle me")])

    async def main():
        task = asyncio.create_task(ws_endpoint(socket))
        assert await asyncio.to_thread(entered.wait, 5)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done(), "the socket task abandoned its held acceptance"
        assert echoes == [] and _row_ids(bridge.drive) == []
        release.set()
        try:
            await asyncio.wait_for(task, 5)
        except asyncio.CancelledError:
            pass
        assert task.cancelled()

    asyncio.run(main())
    assert [(q, ids) for _payload, q, ids in echoes] == [(1, ["cancel-1"])], echoes
    assert echoes[0][0]["ingress_accepted"] is True
    assert socket.sent == []  # no initialization notice: the acceptance succeeded
    update = bridge.get_updates(offset=0, timeout=1)[0]["message"]
    assert update["text"] == "settle me"
    assert update["accepted_source_row"]["client_message_id"] == "cancel-1"


def test_an_acceptance_failure_still_answers_the_socket(bridge):
    """The initialization notice (the socket's only failure reply) survives the
    off-loop hop: a refused canonical write reaches the sender, and nothing is
    echoed or queued."""
    echoes = _witness_echoes(bridge)

    def refuse(*_a, **_k):
        raise OSError("disk refused")

    bridge.ui_send = refuse
    socket = _OpenSocket([_chat_frame("refused-1")])

    async def main():
        task = asyncio.create_task(ws_endpoint(socket))
        deadline = time.monotonic() + 5.0
        while not socket.sent and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await asyncio.wait_for(task, 5)
        except asyncio.CancelledError:
            pass

    asyncio.run(main())
    assert socket.sent and socket.sent[0]["system_type"] == "initialization_notice", socket.sent
    assert echoes == [] and bridge.get_updates(offset=0, timeout=0) == []



def _record_handoffs(bridge, release: threading.Event | None = None):
    """Instrument the bridge callback the socket hands every frame to: record each
    chat acceptance entering it (held until ``release`` when given) and each command
    admitted. Nothing consumes the queue, so an admitted /panic is never executed."""
    handoffs: list[tuple[str, str]] = []
    chat_entered = threading.Event()
    original = bridge.ui_send

    def ui_send(text, **kwargs):
        if kwargs.get("broadcast") is False:
            handoffs.append(("command", text))
        else:
            handoffs.append(("chat", kwargs["client_message_id"]))
            chat_entered.set()
            if release is not None:
                assert release.wait(5), "the test never released the held acceptance"
        return original(text, **kwargs)

    bridge.ui_send = ui_send
    return handoffs, chat_entered


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.01)
    return predicate()


def _command_frame(cmd: str) -> str:
    return json.dumps({"type": "command", "cmd": cmd})


def _queued(bridge) -> list[str]:
    with bridge._inbox.mutex:
        return [item["client_message_id"] or item["text"] for item in bridge._inbox.queue]


def test_a_same_socket_panic_is_admitted_while_its_chat_acceptance_is_held(bridge):
    """Consumer regression (TZ-1 PR1 review F2): ``confirmAndSendPanic`` uses the one
    SPA socket that also carries chat. With that socket's chat acceptance standing on
    a held ingress lock, its next frame — Panic — must still be received and admitted
    on one event loop; the Emergency Stop never waits for ordinary acceptance work."""
    held, release, timed_out = threading.Event(), threading.Event(), threading.Event()
    frames, echoed = _witness_custody(bridge, chat_echoes=1)
    handoffs, chat_entered = _record_handoffs(bridge)

    def hold_ingress():  # a skill delivery mid-scan under the single host's ingress lock
        with message_bus._INGRESS_LOCK:
            held.set()
            if not release.wait(timeout=6.0):
                timed_out.set()

    holder = threading.Thread(target=hold_ingress, name="ingress-holder", daemon=True)
    holder.start()
    assert held.wait(5)
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    try:
        with TestClient(app) as client, client.websocket_connect("/ws") as owner:
            owner.send_text(_chat_frame("held-1"))
            assert chat_entered.wait(5), "the chat frame never reached the bridge"
            owner.send_text(_command_frame("/panic"))
            assert _wait_for(lambda: ("command", "/panic") in handoffs), (
                "Panic waited behind this socket's held chat acceptance")
            assert not timed_out.is_set() and message_bus._INGRESS_LOCK.locked()
            assert frames == [] and _row_ids(bridge.drive) == [] and _queued(bridge) == ["/panic"]
            release.set()
            holder.join(5)
            assert echoed.wait(5), frames
    finally:
        release.set()
    assert handoffs == [("chat", "held-1"), ("command", "/panic")]
    [(echo, queued, rows)] = frames
    assert echo["client_message_id"] == "held-1" and echo["ingress_accepted"] is True
    assert queued == ["/panic", "held-1"] and rows == ["held-1"]  # row → queue → echo


def test_same_socket_chat_frames_settle_in_receive_order_while_commands_pass(bridge):
    """Later chat frames on the socket wait for the held one — they never race it for
    the lock — while commands between them are admitted on receipt. After release the
    rows, queue items and echoes follow receive order, each echo after its row and item."""
    release = threading.Event()
    frames, echoed = _witness_custody(bridge, chat_echoes=3)
    handoffs, chat_entered = _record_handoffs(bridge, release)
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    try:
        with TestClient(app) as client, client.websocket_connect("/ws") as owner:
            owner.send_text(_chat_frame("c-1", "first"))
            assert chat_entered.wait(5)
            owner.send_text(_chat_frame("c-2", "second"))
            owner.send_text(_command_frame("/restart"))
            owner.send_text(_chat_frame("c-3", "third"))
            owner.send_text(_command_frame("/status"))  # its admission proves c-3 was received
            assert _wait_for(lambda: len(handoffs) == 3), handoffs
            assert handoffs == [("chat", "c-1"), ("command", "/restart"), ("command", "/status")]
            assert frames == [] and _row_ids(bridge.drive) == []
            release.set()
            assert echoed.wait(5), frames
    finally:
        release.set()
    assert handoffs[3:] == [("chat", "c-2"), ("chat", "c-3")]
    assert _row_ids(bridge.drive) == ["c-1", "c-2", "c-3"]
    assert [frame["client_message_id"] for frame, *_ in frames] == ["c-1", "c-2", "c-3"]
    for frame, queued, rows in frames:
        assert frame["client_message_id"] in queued and frame["client_message_id"] in rows, frames
    updates = bridge.get_updates(offset=0, timeout=1)
    assert [u["message"]["text"] for u in updates] == ["/restart", "/status", "first", "second", "third"]


def test_a_disconnect_settles_every_received_chat_before_the_socket_task_returns(bridge):
    """The owner closes the socket (TestClient then cancels the session's anyio scope)
    while its first chat is held and a second waits behind it: the socket task returns
    only after both settled row → queue → echo, in receive order."""
    release = threading.Event()
    frames, echoed = _witness_custody(bridge, chat_echoes=2)
    handoffs, chat_entered = _record_handoffs(bridge, release)
    app = Starlette(routes=[WebSocketRoute("/ws", ws_endpoint)])
    with TestClient(app) as client:
        owner = client.websocket_connect("/ws")
        owner.__enter__()
        closer = threading.Thread(target=owner.__exit__, args=(None, None, None), daemon=True)
        try:
            owner.send_text(_chat_frame("d-1", "first"))
            assert chat_entered.wait(5)
            owner.send_text(_chat_frame("d-2", "second"))
            owner.send_text(_command_frame("/status"))  # its admission proves d-2 was received
            assert _wait_for(lambda: ("command", "/status") in handoffs), handoffs
            closer.start()  # disconnect, then the session's anyio scope is cancelled
            closer.join(0.3)
            assert closer.is_alive(), "the socket task returned with received chat frames unsettled"
            assert frames == [] and _row_ids(bridge.drive) == []
            release.set()
            closer.join(5)
            assert not closer.is_alive()
            assert echoed.wait(5), frames
        finally:  # a failed assertion still closes the session instead of hanging the client
            release.set()
            if closer.ident is None:
                closer.start()
            closer.join(10)
    assert _row_ids(bridge.drive) == ["d-1", "d-2"]
    assert [frame["client_message_id"] for frame, *_ in frames] == ["d-1", "d-2"]
    for frame, queued, rows in frames:
        assert frame["client_message_id"] in queued and frame["client_message_id"] in rows, frames


class _ClosingSocket(_OpenSocket):
    """A socket whose client disconnects after its frames; later sends fail."""

    def __init__(self, frames, *, disconnect: bool):
        super().__init__(frames)
        self.disconnect = disconnect
        self.closed = False

    async def receive_text(self):
        if self.frames or not self.disconnect:
            return await super().receive_text()
        self.closed = True
        raise WebSocketDisconnect(1001)

    async def send_text(self, text):
        if self.closed:
            raise RuntimeError("websocket is closed")
        await super().send_text(text)


@pytest.mark.parametrize("disconnect", [False, True])
def test_a_failed_acceptance_answers_in_order_and_the_next_chat_still_lands(bridge, disconnect):
    """A refused acceptance answers its sender with the initialization notice before
    the next chat on the socket is accepted. If the socket is already gone, the
    undeliverable notice is dropped quietly, the next chat still lands, and the
    socket task ends cleanly — no task exception is left unretrieved."""
    release = threading.Event()
    timeline: list[str] = []
    bridge._broadcast_fn = lambda payload: timeline.append(f"echo:{payload['client_message_id']}")
    original = bridge.ui_send

    def refuse_first(text, **kwargs):
        if kwargs["client_message_id"] == "refused-1":
            assert release.wait(5)
            raise OSError("disk refused")
        return original(text, **kwargs)

    bridge.ui_send = refuse_first
    socket = _ClosingSocket([_chat_frame("refused-1"), _chat_frame("ok-2", "next")], disconnect=disconnect)
    sent = socket.send_text

    async def send_text(text):
        await sent(text)
        timeline.append(f"sent:{json.loads(text)['system_type']}")

    socket.send_text = send_text
    unretrieved: list[dict] = []

    async def main():
        asyncio.get_running_loop().set_exception_handler(lambda _loop, context: unretrieved.append(context))
        task = asyncio.create_task(ws_endpoint(socket))
        await asyncio.sleep(0.05)
        assert not task.done() and timeline == []
        release.set()
        deadline = time.monotonic() + 5.0
        while "echo:ok-2" not in timeline and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if disconnect:
            await asyncio.wait_for(task, 5)  # returns cleanly, nothing re-raised
        else:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)

    asyncio.run(main())
    expected = ["echo:ok-2"] if disconnect else ["sent:initialization_notice", "echo:ok-2"]
    assert timeline == expected
    assert _row_ids(bridge.drive) == ["ok-2"] and not unretrieved
    [update] = bridge.get_updates(offset=0, timeout=1)
    assert update["message"]["text"] == "next"


@pytest.mark.parametrize("mode", ["asyncio", "anyio"])
def test_a_cancelled_socket_settles_every_received_chat_in_order(bridge, mode):
    """Cancel the socket task — raw ``Task.cancel()`` twice, or anyio's level
    cancellation — while its first chat is held and a second waits behind it: the
    task stays in custody until both settled in receive order."""
    release = threading.Event()
    frames, _echoed = _witness_custody(bridge, chat_echoes=2)
    handoffs, chat_entered = _record_handoffs(bridge, release)
    socket = _OpenSocket([_chat_frame("x-1", "first"), _chat_frame("x-2", "second")])
    scopes: list[anyio.CancelScope] = []

    async def endpoint():
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await ws_endpoint(socket)

    async def main():
        task = asyncio.create_task(endpoint())
        assert await asyncio.to_thread(chat_entered.wait, 5)
        await asyncio.sleep(0.01)  # the socket is now waiting for a third frame
        if mode == "anyio":
            scopes[0].cancel()
        else:
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "the cancelled socket task abandoned its received chats"
        assert frames == [] and handoffs == [("chat", "x-1")]
        release.set()
        if mode == "anyio":
            await asyncio.wait_for(task, 5)  # the scope absorbs its own cancellation
        else:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)

    try:
        asyncio.run(main())
    finally:
        release.set()
    assert handoffs == [("chat", "x-1"), ("chat", "x-2")]
    assert _row_ids(bridge.drive) == ["x-1", "x-2"]
    assert [frame["client_message_id"] for frame, *_ in frames] == ["x-1", "x-2"]
    for frame, queued, rows in frames:
        assert frame["client_message_id"] in queued and frame["client_message_id"] in rows, frames


_LATE_ID = "quiz_late_answer:task-1:q1"
_LATE_BODY = {"request_id": "r1", "decision_id": "quiz:task-1:q1", "option_index": 1}


def _expired_card(bridge, monkeypatch):
    """A root task asked q1 and ended, so an answer is forwarded as the owner's own
    message; ``forwarding`` flags the answer arriving at the named ingress."""
    from ouroboros.gateway import task_decision as td
    from ouroboros.owner_quiz import reconcile_terminal, record_asked

    record_asked(bridge.drive, "task-1", quiz_id="q1", question="Which db?",
                 options=["sqlite", "postgres"], assumption="sqlite meanwhile", chat_id=1)
    reconcile_terminal(bridge.drive, "task-1")
    monkeypatch.setattr(td, "request_drive_root", lambda request: bridge.drive)
    monkeypatch.setattr(td, "_live_root_task", lambda task_id: (None, "task_not_live"))
    forwarding = threading.Event()
    accept = message_bus.accept_local_message

    def accept_named(*args, **kwargs):
        forwarding.set()
        return accept(*args, **kwargs)

    monkeypatch.setattr(message_bus, "accept_local_message", accept_named)
    return td, forwarding


def _witness_custody(bridge, chat_echoes: int):
    """Record each broadcast with the queued items and rows on disk at that moment."""
    frames: list[tuple[dict, list[str], list[str]]] = []
    echoed = threading.Event()

    def record(payload):
        with bridge._inbox.mutex:
            queued = [item["client_message_id"] or item["text"] for item in bridge._inbox.queue]
        frames.append((payload, queued, _row_ids(bridge.drive)))
        if sum(1 for frame, *_ in frames if frame.get("type") == "chat") >= chat_echoes:
            echoed.set()

    bridge._broadcast_fn = record
    return frames, echoed


def test_a_late_quiz_answer_behind_a_held_socket_acceptance_never_stalls_the_loop(bridge, monkeypatch):
    """Consumer regression (TZ-1 PR1 review R1): the socket acceptance holds the
    ingress lock off the loop while its canonical row reads session state (which may
    wait on the state lock). The late answer needs that lock; forwarded inline from
    the async decision handler it froze the loop behind the socket's worker. On one
    loop the unrelated GET and another socket's command must still be served, and
    after release both acceptances settle in lock order."""
    td, forwarding = _expired_card(bridge, monkeypatch)
    holding, release, timed_out, command_queued = (threading.Event() for _ in range(4))
    frames, echoed = _witness_custody(bridge, chat_echoes=2)

    def state_read_under_ingress_lock():
        holding.set()
        if not release.wait(timeout=6.0):
            timed_out.set()
        return {}

    monkeypatch.setattr(message_bus, "load_state", state_read_under_ingress_lock)
    ui_send = bridge.ui_send

    def ui_send_flagging_commands(text, **kwargs):
        ui_send(text, **kwargs)
        if kwargs.get("broadcast") is False:
            command_queued.set()

    bridge.ui_send = ui_send_flagging_commands
    app = Starlette(routes=[
        WebSocketRoute("/ws", ws_endpoint), Route("/api/health", api_health),
        Route("/api/decisions", td.api_decision_answer, methods=["POST"]),
    ])
    answered: dict = {}
    try:
        # One client context = one portal: sockets, the POST and the GET share ONE event loop.
        with TestClient(app) as client, client.websocket_connect("/ws") as owner, \
                client.websocket_connect("/ws") as other:
            owner.send_text(_chat_frame("held-1"))
            assert holding.wait(5), "the socket acceptance never reached its locked row"
            assert message_bus._INGRESS_LOCK.locked()
            poster = threading.Thread(
                target=lambda: answered.update(response=client.post("/api/decisions", json=_LATE_BODY)),
                name="late-answer", daemon=True,
            )
            poster.start()
            assert forwarding.wait(5), "the late answer never reached the named ingress"
            started = time.monotonic()
            response = client.get("/api/health")
            elapsed = time.monotonic() - started
            assert response.status_code == 200
            assert not timed_out.is_set(), (
                f"the event loop froze behind the late answer's ingress wait (health answered after {elapsed:.2f}s)")
            other.send_text(json.dumps({"type": "command", "cmd": "/stop"}))
            assert command_queued.wait(5) and not timed_out.is_set(), "a command queued behind the late answer"
            assert poster.is_alive() and frames == []
            assert _row_ids(bridge.drive) == ["quiz_answer:task-1:q1"]  # the card's history row only
            release.set()
            poster.join(10)
            assert echoed.wait(5), frames
    finally:
        release.set()
    assert answered["response"].status_code == 200, answered["response"].text
    body = answered["response"].json()
    assert body["answered_after_terminal"] is True and body["forwarded"] is True
    assert _row_ids(bridge.drive) == ["quiz_answer:task-1:q1", "held-1", _LATE_ID]
    chats = {frame["client_message_id"]: (queued, rows) for frame, queued, rows in frames if frame.get("type") == "chat"}
    assert set(chats) == {"held-1", _LATE_ID}
    for client_message_id, (queued, rows) in chats.items():  # row → queue → echo, each in custody order
        assert client_message_id in queued and client_message_id in rows, (client_message_id, queued, rows)
    assert [frame["type"] for frame, *_ in frames].count("quiz_state") == 1
    updates = bridge.get_updates(offset=0, timeout=1)
    assert [u["message"]["text"] for u in updates] == ["/stop", "hello", "2. postgres"]


def test_a_cancelled_late_answer_request_settles_its_delivery_and_a_retry_rejoins(bridge, monkeypatch):
    """Cancel the decision request while its late answer stands on the held ingress
    lock: the owner's row, the queue item and the echo still complete, in that order,
    before the cancellation is observed. The same request retried afterwards rejoins
    that delivery instead of enqueueing a second owner turn."""
    td, forwarding = _expired_card(bridge, monkeypatch)
    held, release, timed_out = threading.Event(), threading.Event(), threading.Event()
    frames, _echoed = _witness_custody(bridge, chat_echoes=1)

    def hold_ingress():  # a socket acceptance or skill delivery mid-write under the lock
        with message_bus._INGRESS_LOCK:
            held.set()
            if not release.wait(timeout=6.0):
                timed_out.set()

    holder = threading.Thread(target=hold_ingress, name="ingress-holder", daemon=True)
    holder.start()
    assert held.wait(5)

    async def main():
        task = asyncio.create_task(td.answer_decision(bridge.drive, dict(_LATE_BODY)))
        assert await asyncio.to_thread(forwarding.wait, 5)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "the cancelled request abandoned its late delivery (or blocked the loop)"
        assert frames == [] and _row_ids(bridge.drive) == ["quiz_answer:task-1:q1"]
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 5)
        assert task.cancelled()

    try:
        asyncio.run(main())
    finally:
        release.set()
        holder.join(5)
    assert not timed_out.is_set()
    [(echo, queued, rows)] = frames
    assert echo["client_message_id"] == _LATE_ID and echo["ingress_accepted"] is True
    assert queued == [_LATE_ID] and rows == ["quiz_answer:task-1:q1", _LATE_ID]

    status, body = asyncio.run(td.answer_decision(bridge.drive, dict(_LATE_BODY)))
    assert status == 200 and body["duplicate"] is True and body["forwarded"] is True
    assert _row_ids(bridge.drive) == ["quiz_answer:task-1:q1", _LATE_ID]
    assert [frame["type"] for frame, *_ in frames] == ["chat", "quiz_state"]  # no second echo; the card settles
    [update] = bridge.get_updates(offset=0, timeout=1)
    assert update["message"]["text"] == "2. postgres"
    assert update["message"]["accepted_source_row"]["client_message_id"] == _LATE_ID
