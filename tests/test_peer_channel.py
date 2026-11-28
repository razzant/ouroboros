"""Tests T3 and T5 for cross-machine channel sockets, session monotonicity, and offline delivery."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import pathlib
import select
import shutil
import socket
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import peer_channel, peer_channel_queue, peer_door, peer_links, peer_replies, place_projection, places
from ouroboros.execution_domain import EngineClients
from ouroboros.install_identity import acting_as
from tests._peer_letters_shared import init_control_state
from tests.test_peer_messages import (
    _approve,
    _enable_consciousness,
    _poll_until,
)

pytestmark = pytest.mark.skipif(
    not peer_channel.is_door_supported(),
    reason="POSIX Unix-domain channel tests only",
)


class UnixSocketForwarder:
    """Four-mode Unix-domain socket forwarder for channel tests (no SSH):
    - ``absent``: socket file does not exist (ENOENT)
    - ``accept_close``: accepts connection and closes immediately without bytes (RemoteProtocolError)
    - ``accept_silent``: accepts connection and holds silently without responding (ReadTimeout)
    - ``forward``: transparent bidirectional byte relay to ``target_path``
    """

    def __init__(self, listen_path: pathlib.Path, target_path: pathlib.Path, *, mode: str = "forward") -> None:
        self.listen_path = pathlib.Path(listen_path)
        self.target_path = pathlib.Path(target_path)
        self._lock = threading.Lock()
        self._mode = mode
        self._stop = threading.Event()
        self._server: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._conns: list[socket.socket] = []
        self._apply_mode(mode)

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> None:
        assert mode in {"absent", "refuse", "accept_close", "accept_silent", "forward"}
        self._stop_listener()
        self._apply_mode(mode)

    def _apply_mode(self, mode: str) -> None:
        with self._lock:
            self._mode = mode
        if mode == "absent":
            self.listen_path.unlink(missing_ok=True)
            return
        self._stop.clear()
        self.listen_path.parent.mkdir(parents=True, exist_ok=True)
        self.listen_path.unlink(missing_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.listen_path))
        os.chmod(self.listen_path, 0o600)
        if mode == "refuse":
            srv.close()  # leave stale socket inode without listener -> ECONNREFUSED on connect(2)
            return
        srv.listen(64)
        srv.settimeout(0.05)
        self._server = srv
        self._thread = threading.Thread(target=self._accept_loop, args=(srv, mode), daemon=True)
        self._thread.start()

    def _accept_loop(self, srv: socket.socket, mode: str) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._lock:
                self._conns.append(conn)
            if mode == "accept_close":
                with contextlib.suppress(OSError):
                    conn.close()
            elif mode == "accept_silent":
                threading.Thread(target=self._hold_silent, args=(conn,), daemon=True).start()
            elif mode == "forward":
                threading.Thread(target=self._relay, args=(conn,), daemon=True).start()

    def _hold_silent(self, conn: socket.socket) -> None:
        conn.settimeout(0.05)
        try:
            while not self._stop.is_set():
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                except socket.timeout:
                    continue
                except OSError:
                    break
        finally:
            with contextlib.suppress(OSError):
                conn.close()

    def _relay(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            upstream.connect(str(self.target_path))
            with self._lock:
                self._conns.append(upstream)
            client.setblocking(False)
            upstream.setblocking(False)
            sockets = [client, upstream]
            while not self._stop.is_set() and sockets:
                readable, _, _ = select.select(sockets, [], [], 0.05)
                for src in readable:
                    dst = upstream if src is client else client
                    try:
                        chunk = src.recv(65536)
                    except OSError:
                        chunk = b""
                    if not chunk:
                        with contextlib.suppress(OSError):
                            dst.shutdown(socket.SHUT_WR)
                        sockets.remove(src)
                        continue
                    try:
                        dst.setblocking(True)
                        dst.sendall(chunk)
                        dst.setblocking(False)
                    except OSError:
                        sockets.clear()
                        break
        except OSError:
            pass
        finally:
            for s in (client, upstream):
                if s is not None:
                    with contextlib.suppress(OSError):
                        s.close()

    def _stop_listener(self) -> None:
        self._stop.set()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()
            self._server = None
        with self._lock:
            conns, self._conns = list(self._conns), []
        for conn in conns:
            with contextlib.suppress(OSError):
                conn.close()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def close(self) -> None:
        self._stop_listener()
        self.listen_path.unlink(missing_ok=True)


@contextlib.contextmanager
def live_channel_pair(tmp_path: pathlib.Path, monkeypatch, *, b_lets_a_start: bool = True, a_lets_b_start: bool = False):
    """Two installations (A opener, B receiver) with real ASGI letters doors over Unix sockets
    and two controllable 4-mode forwarders (no SSH)."""
    if not peer_channel.is_door_supported():
        pytest.skip("POSIX Unix-domain channel tests only")
    short_dir = pathlib.Path(tempfile.mkdtemp(prefix="obc-", dir="/tmp"))
    root_a, root_b = short_dir / "a" / "data", short_dir / "b" / "data"
    for r in (root_a, root_b):
        (r / "memory").mkdir(parents=True)
        init_control_state(r, enabled=False)

    # In-process multi-install fixture: bind each door's ASGI requests to its own install identity.
    real_create_app = peer_door.create_peer_door_app

    def _scoped_door_app(drive_root, **kwargs):
        inner = real_create_app(drive_root, **kwargs)
        target_root = pathlib.Path(drive_root)

        async def _wrapped(scope, receive, send):
            with acting_as(target_root):
                await inner(scope, receive, send)

        return _wrapped

    monkeypatch.setattr(peer_door, "create_peer_door_app", _scoped_door_app)

    # Start both doors on one dedicated event loop thread.
    ready = threading.Event()
    stop_ev = threading.Event()
    boots = [
        SimpleNamespace(peer_door_task=None, peer_door_server=None, peer_door_listener=contextlib.ExitStack()),
        SimpleNamespace(peer_door_task=None, peer_door_server=None, peer_door_listener=contextlib.ExitStack()),
    ]
    err: list[BaseException] = []

    def _loop_runner():
        async def _main():
            try:
                await peer_door.serve_peer_door(root_a, boots[0])
                await peer_door.serve_peer_door(root_b, boots[1])
            except BaseException as exc:  # noqa: BLE001
                err.append(exc)
            finally:
                ready.set()
            while not stop_ev.is_set():
                await asyncio.sleep(0.02)
            from ouroboros.server_boot import stop_embedded_services

            for b in boots:
                await stop_embedded_services(b)

        asyncio.run(_main())

    thread = threading.Thread(target=_loop_runner, name="test-channel-doors", daemon=True)
    thread.start()
    assert ready.wait(timeout=10.0)
    if err:
        stop_ev.set()
        thread.join(timeout=10.0)
        raise err[0]

    fwd_a_to_b: UnixSocketForwarder | None = None
    fwd_b_to_a: UnixSocketForwarder | None = None
    extra_fwds: list[UnixSocketForwarder] = []
    try:
        door_a = pathlib.Path(peer_channel.door_status(root_a)["socket_path"])
        door_b = pathlib.Path(peer_channel.door_status(root_b)["socket_path"])

        with acting_as(root_b):
            inv = peer_links.create_invite(root_b, label="Laptop", own_label="Velm",
                                           note="Channel peer", may_start=b_lets_a_start, remote=True)
        link_id = inv["link_id"]
        sess1 = peer_channel.next_channel_session()

        sock_l = peer_channel.opener_socket_path(root_a, link_id)
        sock_r = peer_channel.receiver_socket_path(root_b, link_id, sess1)
        fwd_a_to_b = UnixSocketForwarder(sock_l, door_b, mode="forward")
        fwd_b_to_a = UnixSocketForwarder(sock_r, door_a, mode="forward")

        with acting_as(root_a):
            view_a = peer_links.accept_invite(root_a, inv["invite"], session=sess1, label="Velm",
                                              own_label="Laptop", may_start=a_lets_b_start)
        with acting_as(root_b):
            place_b = peer_links.read_registry(root_b)["links"][link_id]["place_id"]
        for r in (root_a, root_b):
            with acting_as(r):
                peer_links.confirm(r, link_id)

        def rotate_session(new_session: str = "") -> tuple[str, UnixSocketForwarder]:
            nonlocal fwd_b_to_a
            with acting_as(root_a):
                sess = new_session or peer_channel.allocate_channel_session(root_a, link_id)
            new_r = peer_channel.receiver_socket_path(root_b, link_id, sess)
            new_fwd = UnixSocketForwarder(new_r, door_a, mode="forward")
            extra_fwds.append(new_fwd)
            if fwd_a_to_b.mode != "forward":
                fwd_a_to_b.set_mode("forward")
            with acting_as(root_a):
                peer_channel.announce_channel(root_a, link_id, sess)
            old_fwd = fwd_b_to_a
            fwd_b_to_a = new_fwd
            if old_fwd is not None:
                old_fwd._stop_listener()  # leave file removal to R6 cleanup verification
            return sess, new_fwd

        ctx = SimpleNamespace(
            root_a=root_a, root_b=root_b, link_id=link_id, place_a=view_a["place_id"], place_b=place_b,
            door_a=door_a, door_b=door_b, session=sess1, fwd_a_to_b=fwd_a_to_b, fwd_b_to_a=fwd_b_to_a,
            rotate_session=rotate_session, invite=inv["invite"],
        )
        yield ctx
    finally:
        for f in (*extra_fwds, fwd_a_to_b, fwd_b_to_a):
            if f is not None:
                f.close()
        stop_ev.set()
        thread.join(timeout=10.0)
        shutil.rmtree(short_dir, ignore_errors=True)


def _flush_threads_idle() -> bool:
    return not any(
        t.is_alive() and t.name.startswith(("peer-reply-", "peer-replies-", "peer-channel-"))
        for t in threading.enumerate()
    )


def _message_texts_in_room(root: pathlib.Path, place_id: str) -> list[str]:
    from ouroboros import dialogue_store as ds
    from ouroboros.places import conversation_chat_id, place_records
    from ouroboros.presence_bindings import conversation_key

    with acting_as(root):
        place = place_records(root).places[place_id]
        key = conversation_key(place.provider, place.account_id, place.conversation_id, "")
        chat_id = conversation_chat_id(key)
        if not ds.present(root):
            return []
        return [str(entry.row.get("text") or "") for entry in ds.iter_probes(root, "chat", [("chat_id", chat_id)])]


def _incoming_message_ids(root: pathlib.Path, link_id: str) -> list[str]:
    folder = peer_links.token_path(root, link_id).parent / "messages"
    if not folder.is_dir():
        return []
    ids = []
    for p in sorted(folder.iterdir()):
        if p.suffix == ".json":
            ids.append(json.loads(p.read_text(encoding="utf-8"))["letter_id"])
    return ids


# ── T3: Derived socket naming, 12-hex session validation, monotonicity & cleanup

def test_t3_derived_socket_names_session_format_and_monotonic_announce(tmp_path, monkeypatch):
    """Derived socket names come strictly from link_id & 12-hex session; invalid sessions are refused;
    no session can name door.sock or another link; stale/receiver-advance announces are refused
    channel_session_stale; equal session only resumes; new session cleans only older r-files of this link."""
    with live_channel_pair(tmp_path, monkeypatch) as ch:
        tag = peer_channel.link_tag(ch.link_id)
        assert peer_channel.opener_socket_path(ch.root_a, ch.link_id).name == f"l-{tag}.sock"
        assert peer_channel.receiver_socket_path(ch.root_b, ch.link_id, ch.session).name == f"r-{tag}-{ch.session}.sock"

        # Invalid session formats rejected everywhere (accept, introduce, announce).
        bad_sessions = ["", "12345678", "0123456789ab0", "ABCDEF012345", "../door.sock", "door.sock", 123456789012]
        for bad in bad_sessions:
            with pytest.raises(peer_links.PeerLinkError) as e1:
                peer_channel.validate_session(bad)
            assert e1.value.code == "invalid_session"
            with acting_as(ch.root_a), pytest.raises(peer_links.PeerLinkError) as e2:
                peer_links.accept_invite(ch.root_a, ch.invite, session=bad, may_start=False)
            assert e2.value.code == "invalid_session"
            with acting_as(ch.root_b), pytest.raises(peer_links.PeerLinkError) as e3:
                peer_links.introduce(ch.root_b, "tok", {
                    "link_id": ch.link_id, "install_id": "x", "session": bad, "key_sha256": "0" * 64,
                })
            assert e3.value.code == "invalid_session"
            with acting_as(ch.root_b), pytest.raises(peer_links.PeerLinkError) as e4:
                peer_channel.set_channel_session(ch.root_b, ch.link_id, bad)
            assert e4.value.code == "invalid_session"

        # No valid 12-hex session can ever point at door.sock or another link's socket.
        other_tag = peer_channel.link_tag("f" * 32)
        for cand in ("000000000000", "ffffffffffff", ch.session):
            p = peer_channel.receiver_socket_path(ch.root_b, ch.link_id, cand)
            assert p.name != peer_channel.DOOR_SOCKET_NAME
            assert p.name.startswith(f"r-{tag}-") and other_tag not in p.name

        # Monotonic session rules:
        s1 = ch.session
        s_older = f"{max(0, int(s1, 16) - 1):012x}"
        s_newer = peer_channel.next_channel_session(s1)
        s_future = peer_channel.next_channel_session(s_newer)

        # Create a sibling link's r-socket AND a future-session r-socket in B's door dir to verify selective cleanup.
        door_dir_b = peer_channel.verify_door_dir(ch.root_b)
        other_sock = door_dir_b / f"r-{other_tag}-{s1}.sock"
        future_sock = door_dir_b / f"r-{tag}-{s_future}.sock"
        other_fwd = UnixSocketForwarder(other_sock, ch.door_a, mode="forward")
        future_fwd = UnixSocketForwarder(future_sock, ch.door_a, mode="forward")
        try:
            # 1. Lower session refused channel_session_stale, nothing changes.
            with acting_as(ch.root_b), pytest.raises(peer_links.PeerLinkError) as e_old:
                peer_channel.set_channel_session(ch.root_b, ch.link_id, s_older)
            assert e_old.value.code == "channel_session_stale"
            assert peer_links.read_registry(ch.root_b)["links"][ch.link_id]["channel_session"] == s1

            # 2. Larger session sent from receiver side (to opener) is refused channel_session_stale.
            with acting_as(ch.root_a), pytest.raises(peer_links.PeerLinkError) as e_rev:
                peer_channel.set_channel_session(ch.root_a, ch.link_id, s_newer)
            assert e_rev.value.code == "channel_session_stale"
            with acting_as(ch.root_b), pytest.raises(peer_links.PeerLinkError) as e_rev2:
                peer_channel.announce_channel(ch.root_b, ch.link_id, s_newer)
            assert e_rev2.value.code == "channel_session_stale"

            # 3. Equal session accepted from BOTH sides without modifying session or deleting sockets.
            with acting_as(ch.root_b):
                eq_b = peer_channel.announce_channel(ch.root_b, ch.link_id, s1)
            assert eq_b["ok"] is True and eq_b["session"] == s1
            with acting_as(ch.root_a):
                eq_a = peer_channel.announce_channel(ch.root_a, ch.link_id, s1)
            assert eq_a["ok"] is True and eq_a["session"] == s1
            assert (door_dir_b / f"r-{tag}-{s1}.sock").exists()

            # 4. Strictly newer session from opener updates session and cleans ONLY older r-files of THIS link.
            s2, _new_fwd = ch.rotate_session(s_newer)
            assert s2 == s_newer
            assert peer_links.read_registry(ch.root_b)["links"][ch.link_id]["channel_session"] == s2
            assert peer_links.read_registry(ch.root_a)["links"][ch.link_id]["channel_session"] == s2
            assert not (door_dir_b / f"r-{tag}-{s1}.sock").exists(), "older r-socket of own link removed"
            assert (door_dir_b / f"r-{tag}-{s2}.sock").exists(), "active r-socket kept"
            assert future_sock.exists(), "higher-session r-socket NOT removed"
            assert other_sock.exists(), "another link's r-socket NOT removed"
            assert ch.door_b.exists(), "door.sock NOT removed"
        finally:
            other_fwd.close()
            future_fwd.close()


# ── T5(a)-(d): 4-mode forwarder unavailability, zero pass burn, R6 delivery ──

def test_t5_unavailability_modes_zero_pass_burn_and_r6_delivery(tmp_path, monkeypatch):
    """T5(a),(b),(c),(d): absent -> waiting (0 passes); accept_close & accept_silent -> uncertain
    (0 passes even past _MAX_PASSES, no timers, owed & queued stay open); R6 new session delivers
    each once with same letter_id and 0 model calls on Off receiver."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_ATTEMPT_SEC", 0.35)
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (60.0,))

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        _enable_consciousness(ch.root_a, monkeypatch, enabled=False)
        _enable_consciousness(ch.root_b, monkeypatch, enabled=False)
        # Count model turns dispatched on B (must remain 0 while B is Off).
        b_turns: list = []
        from ouroboros import presence_carrier
        orig_dispatch = presence_carrier.dispatch_presence_turn
        monkeypatch.setattr(presence_carrier, "dispatch_presence_turn",
                            lambda *a, **k: (b_turns.append(a), orig_dispatch(*a, **k))[1])

        ctx_a = SimpleNamespace(task_id="task-a-1", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)

        # Create an owed automatic reply on A (simulating a finished focus turn on A replying to B).
        from ouroboros.task_results import STATUS_COMPLETED, write_task_result
        with acting_as(ch.root_a):
            assert ch.place_a in places.place_records(ch.root_a).places
            owed_text = "Auto reply queued while channel down [T5]."
            pres_meta = {
                "place_id": ch.place_a, "delivery_reporting_version": 1,
                "event": {"event_id": "ev-t5", "message": {"message_id": "in-1", "peer_message": True}},
            }
            peer_links.record_owed(ch.root_a, ch.link_id, "prod-t5", place_id=ch.place_a)
            _approve(SimpleNamespace(root_b=ch.root_a, place_b=ch.place_a, link_id=ch.link_id),
                     "prod-t5", owed_text, {"presence": pres_meta})
            write_task_result(ch.root_a, "prod-t5", STATUS_COMPLETED, result=owed_text,
                              terminal_origin="model_final", chat_id=1, source="presence",
                              metadata={"presence": pres_meta})
            owed_lid = peer_replies.message_letter_id(ch.link_id, "prod-t5:terminal")

        # (a) Mode: absent (socket file missing) -> waiting, 0 passes, no timer scheduled.
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            res_absent = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Tool note absent [T5a]").text)
            assert res_absent["status"] == "queued" and res_absent["reason"] == "channel_unavailable", res_absent
            lid_a = res_absent["letter_id"]
            assert peer_links.sending_record(ch.root_a, ch.link_id, lid_a) is None, "nothing sent -> no sent/ record"
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid_a) is not None
            _poll_until(_flush_threads_idle, timeout=10.0, label="absent server pass done")
            settled, wait = peer_replies._one_pass(ch.root_a, ch.link_id, "prod-t5")
            assert (settled, wait) == (False, None), "no timer retry scheduled on absent channel"
            assert peer_links.owed_refs(ch.root_a, ch.link_id) == [(ch.link_id, "prod-t5")]
            assert peer_links.sending_record(ch.root_a, ch.link_id, owed_lid) is None

        # (b) Mode: accept_close (RemoteProtocolError) -> uncertain, 0 passes even after _MAX_PASSES + 2 tries!
        ch.fwd_a_to_b.set_mode("accept_close")
        with acting_as(ch.root_a):
            res_close = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Tool note close [T5b]").text)
            assert res_close["status"] == "queued" and res_close["reason"] == "channel_unconfirmed", res_close
            lid_b = res_close["letter_id"]
            for _ in range(peer_replies._MAX_PASSES + 2):
                _poll_until(_flush_threads_idle, timeout=10.0, label="close server pass done")
                s_rec = peer_links.sending_record(ch.root_a, ch.link_id, lid_b)
                assert s_rec is not None and int(s_rec["passes"]) == 0
                settled, wait = peer_replies._one_pass(ch.root_a, ch.link_id, "prod-t5")
                assert (settled, wait) == (False, None)
                peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
            assert peer_links.owed_refs(ch.root_a, ch.link_id) == [(ch.link_id, "prod-t5")]
            assert int(peer_links.sending_record(ch.root_a, ch.link_id, owed_lid)["passes"]) == 0

        # (c) Mode: accept_silent (ReadTimeout) -> uncertain, 0 passes, no timer, still owed & queued.
        ch.fwd_a_to_b.set_mode("accept_silent")
        with acting_as(ch.root_a):
            res_silent = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Tool note silent [T5c]").text)
            assert res_silent["status"] == "queued" and res_silent["reason"] == "channel_unconfirmed", res_silent
            lid_c = res_silent["letter_id"]
            assert int(peer_links.sending_record(ch.root_a, ch.link_id, lid_c)["passes"]) == 0
            _poll_until(_flush_threads_idle, timeout=10.0, label="silent server pass done")
            settled, wait = peer_replies._one_pass(ch.root_a, ch.link_id, "prod-t5")
            assert (settled, wait) == (False, None)
            assert int(peer_links.sending_record(ch.root_a, ch.link_id, owed_lid)["passes"]) == 0
            assert _incoming_message_ids(ch.root_b, ch.link_id) == []

        # (d) R6 new session -> restore normal read timeout so real SQLite init never races a 0.35s timeout;
        # all 4 letters arrive ONCE with their exact original letter_ids; 0 model calls on B.
        monkeypatch.setattr(peer_replies, "_ATTEMPT_SEC", 15.0)
        ch.rotate_session()
        expected_ids = sorted([lid_a, lid_b, lid_c, owed_lid])
        _poll_until(lambda: sorted(_incoming_message_ids(ch.root_b, ch.link_id)) == expected_ids,
                    timeout=15.0, label="all queued & owed delivered on R6 new session")
        _poll_until(_flush_threads_idle, timeout=30.0, label="flush threads done")
        with acting_as(ch.root_a):
            assert peer_links.owed_refs(ch.root_a, ch.link_id) == []
            assert peer_channel_queue.queued_messages(ch.root_a, ch.link_id) == []
            for lid in expected_ids:
                assert peer_links.sending_record(ch.root_a, ch.link_id, lid) is None
        assert b_turns == [], "Consciousness Off receiver must make zero model calls"


# ── T5(e),(f): Equal-session announce, verified exchange, cached client across session change

def test_t5_equal_session_verified_exchange_and_cached_client_rotation(tmp_path, monkeypatch):
    """T5(e),(f): equal-session announce and verified exchange without reconnect each flush queued/owed;
    a PeerGateway cached in EngineClients BEFORE session rotation reaches the NEW socket after rotation
    for status, stop-state, cancel, and BodyRuntime._peer_link_signature."""
    from ouroboros.delegate_collection import BodyRuntime
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=True, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-a-eq", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)

        # (e.1) Equal-session announce flushes queued message without rotating session.
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q1 = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Queued for equal session").text)
            assert q1["status"] == "queued"
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel.announce_channel(ch.root_a, ch.link_id, ch.session)
        _poll_until(lambda: q1["letter_id"] in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=10.0, label="equal-session announce flushed queued message")
        _poll_until(_flush_threads_idle, timeout=30.0)

        # (e.2) Grok-A1: Verified status GET is strictly passive (does NOT flush); verified POST DOES flush!
        ctx_b = SimpleNamespace(task_id="task-b-vx", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_b, budget_drive_root=ch.root_b)
        ch.fwd_b_to_a.set_mode("absent")
        with acting_as(ch.root_b):
            qb = json.loads(_message_ouroboros(ctx_b, place_id=ch.place_b, text="Queued on B for verified exchange").text)
            assert qb["status"] == "queued"
        ch.fwd_b_to_a.set_mode("forward")
        with acting_as(ch.root_a):
            link_a = peer_links.outgoing_link(ch.place_a, drive_root=ch.root_a)
            with peer_channel.peer_http_client(channel_path=peer_channel.channel_socket_for_link(link_a)) as client:
                r = client.get("/api/letters/probe-nonexistent", headers=link_a.auth_headers())
                assert r.status_code == 404
        time.sleep(0.15)
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert qb["letter_id"] not in _incoming_message_ids(ch.root_a, ch.link_id), (
            "passive GET /api/letters/{id} must not trigger outbound sends"
        )
        # Now A sends a verified POST exchange -> B flushes its queued message to A!
        with acting_as(ch.root_a):
            res_post = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Verified POST trigger").text)
            assert res_post["status"] == "received", res_post
        _poll_until(lambda: qb["letter_id"] in _incoming_message_ids(ch.root_a, ch.link_id),
                    timeout=10.0, label="verified POST exchange flushed B's queued message to A")
        _poll_until(_flush_threads_idle, timeout=30.0)

        # (f) Long-lived PeerGateway cached in EngineClients BEFORE session rotation (receiver B -> A, whose
        # derived socket path changes from r-<tag>-<s1>.sock to r-<tag>-<s2>.sock!).
        ref_b = {"kind": "peer", "place_id": ch.place_b, "link_id": ch.link_id}
        clients = EngineClients()
        try:
            with acting_as(ch.root_b):
                gw_before = clients.get(ref_b)
                sig_before = BodyRuntime._peer_link_signature(SimpleNamespace(root=ch.root_b), ref_b)
                assert sig_before == (ch.link_id, peer_channel.TRANSPORT_CHANNEL, "")
                # Probe before rotation -> 404 letter_not_found over s1.
                with pytest.raises(Exception) as exc1:
                    gw_before.get_run("missing-1")
                assert getattr(exc1.value, "code", "") == "letter_not_found"

            # Rotate session: old r-socket is unlinked by R6 cleanup!
            s2, _ = ch.rotate_session()
            assert s2 != ch.session

            with acting_as(ch.root_b):
                gw_after = clients.get(ref_b)
                assert gw_after is gw_before, "same cached PeerGateway reused across session rotation"
                sig_after = BodyRuntime._peer_link_signature(SimpleNamespace(root=ch.root_b), ref_b)
                assert sig_after == sig_before == (ch.link_id, peer_channel.TRANSPORT_CHANNEL, ""), (
                    "live channel link signature must stay stable across session rotation"
                )
                # Same cached client now reaches A over r-<tag>-<s2>.sock for get_run, stop-state, cancel!
                for call in (
                    lambda: gw_after.get_run("missing-2"),
                    lambda: gw_after.get_run("missing-2", for_stop=True),
                    lambda: gw_after.cancel_run("missing-2"),
                ):
                    with pytest.raises(Exception) as exc2:
                        call()
                    assert getattr(exc2.value, "code", "") == "letter_not_found"
        finally:
            clients.drop()


# ── T5(g),(h),(i): Policy matrix on announce, speech_basis recheck, assignment never auto-retried

@pytest.mark.parametrize("condition,expected_posts", [
    ("clean", 1),
    ("panic", 0),
    ("paused", 0),
    ("silent", 0),
    ("rules_changed", 0),
    ("revoked", 0),
])
def test_t5_announce_matrix_and_speech_basis_check(tmp_path, monkeypatch, condition, expected_posts):
    """T5(g),(h): announce x {clean, panic, paused, silent, rules_changed, revoked} yields 1 or 0 POSTs,
    and queued message_ouroboros checks saved speech_basis before resending."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id=f"task-{condition}", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text=f"Queued {condition}").text)
            assert q["status"] == "queued"
            assert q["letter_id"]

        # Apply condition before channel comes back.
        if condition == "panic":
            monkeypatch.setattr(place_projection, "switch_state", lambda root: "panic")
        elif condition == "paused":
            with acting_as(ch.root_a):
                places.edit_place(ch.root_a, ch.place_a, "set_state", author=places.AUTHOR_OWNER,
                                  state=places.STATE_PAUSED)
        elif condition == "silent":
            with acting_as(ch.root_a):
                places.edit_place(ch.root_a, ch.place_a, "set_state", author=places.AUTHOR_OWNER,
                                  state=places.STATE_SILENT)
        elif condition == "rules_changed":
            with acting_as(ch.root_a):
                places.edit_place(ch.root_a, ch.place_a, "add_note", author=places.AUTHOR_OWNER,
                                  text="Strict new rule: never send queued words.")
        elif condition == "revoked":
            with acting_as(ch.root_a):
                peer_links.revoke(ch.root_a, ch.link_id)

        ch.fwd_a_to_b.set_mode("forward")
        if condition != "revoked":
            with acting_as(ch.root_a):
                peer_channel.announce_channel(ch.root_a, ch.link_id, ch.session)
        else:
            peer_channel_queue.flush_link_sync(ch.root_a, ch.link_id)
        _poll_until(_flush_threads_idle, timeout=30.0)
        time.sleep(0.15)

        got = _incoming_message_ids(ch.root_b, ch.link_id)
        assert len(got) == expected_posts, (condition, got)
        if condition in {"rules_changed", "revoked"}:
            with acting_as(ch.root_a):
                assert peer_channel_queue.queued_messages(ch.root_a, ch.link_id) == []
        elif condition in {"panic", "paused", "silent"}:
            with acting_as(ch.root_a):
                assert len(peer_channel_queue.queued_messages(ch.root_a, ch.link_id)) == 1


def test_t5_assignment_with_unknown_outcome_is_never_auto_retried(tmp_path, monkeypatch):
    """T5(i) + C10: PeerGateway.start_run over accept_close / accept_silent raises unconfirmed and is never
    queued or auto-retried when the channel returns (verified via receiver _accept_new spy + task store + positive control)."""
    from ouroboros.gateway import peer_letters_api
    from ouroboros.gateways.peer import UNCONFIRMED, PeerGateway
    from ouroboros.peer_letters import letter_task_id
    from ouroboros.task_results import load_task_result
    from tests._peer_letters_shared import ready_queue

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=True, a_lets_b_start=False) as ch:
        ready_queue(monkeypatch, ch.root_b)
        accepted_ids: list[str] = []
        orig_accept_new = peer_letters_api._accept_new

        def _spy_accept_new(request, caller, letter, sha, facts):
            accepted_ids.append(str(letter.get("letter_id") or ""))
            return orig_accept_new(request, caller, letter, sha, facts)

        monkeypatch.setattr(peer_letters_api, "_accept_new", _spy_accept_new)

        with acting_as(ch.root_a):
            link = peer_links.outgoing_link(ch.place_a, drive_root=ch.root_a)
            gw = PeerGateway(link, timeout_sec=0.35)
        try:
            for idx, mode in enumerate(("accept_close", "accept_silent"), start=1):
                ch.fwd_a_to_b.set_mode(mode)
                with acting_as(ch.root_a), pytest.raises(Exception) as exc:
                    gw.start_run({"letter_id": f"assign-{idx}", "text": "Do work", "protocol": 1})
                assert getattr(exc.value, "code", "") == UNCONFIRMED
            # Channel recovers (both session rotation and equal-session announce): NO assignment is auto-retried.
            ch.rotate_session()
            with acting_as(ch.root_a):
                peer_channel.announce_channel(ch.root_a, ch.link_id)
            _poll_until(_flush_threads_idle, timeout=30.0)
            time.sleep(0.15)
            with acting_as(ch.root_a):
                assert peer_channel_queue.queued_messages(ch.root_a, ch.link_id) == []
            assert accepted_ids == [], f"unconfirmed assignments must never auto-retry: {accepted_ids}"
            with acting_as(ch.root_b):
                for lid in ("assign-1", "assign-2"):
                    assert load_task_result(ch.root_b, letter_task_id(ch.place_b, lid)) is None
            # Positive control: an explicit assignment POST over the recovered channel DOES hit _accept_new.
            # Use normal timeout (not the fault-injection 0.35s timeout) so real store init never races a tight deadline.
            with acting_as(ch.root_a):
                live_gw = PeerGateway(link)
                try:
                    started = live_gw.start_run({"letter_id": "assign-live-control", "text": "Live check", "protocol": 1})
                finally:
                    live_gw.close()
            assert started["runId"] == "assign-live-control"
            assert accepted_ids == ["assign-live-control"]
        finally:
            gw.close()


# ── Review Round 1 regression tests (C1-C7, C11, A1, C16) ────────────────────

def test_t5_503_then_connect_refused_preserves_uncertain_outcome_and_sending_record(tmp_path, monkeypatch):
    """C1: If attempt 0 gets HTTP 503 (network reached) and attempt 1 hits channel connect failure,
    send_peer_message preserves state='uncertain', keeps sent/ intact, and never returns 'waiting'."""
    from ouroboros.gateways.peer import PeerGateway, _unavailable

    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0.01)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        attempts = {"n": 0}

        def _fake_send(self, text, *, letter_id, reply_to="", timeout_sec=None):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise _unavailable("unconfirmed", "receiver returned 503", 503)
            err = _unavailable("channel_unavailable", "channel refused connection on attempt 2")
            err.channel_connect_failed = True
            err.channel_transport_error = True
            raise err

        monkeypatch.setattr(PeerGateway, "send_message", _fake_send)
        lid = "msg-c1-503-then-refused"
        with acting_as(ch.root_a):
            res = peer_replies.send_peer_message(
                ch.root_a, ch.place_a, letter_id=lid, text="Important words",
                origin={"kind": "tool", "task_id": "task-c1"},
            )
            assert res["state"] == "uncertain", res
            assert res["code"] == "unconfirmed", res
            assert res.get("channel_wait") is True, res
            s_rec = peer_links.sending_record(ch.root_a, ch.link_id, lid)
            assert s_rec is not None and int(s_rec["passes"]) == 1, s_rec


def test_t5_queue_write_failure_preserves_sending_record_for_retry_of(tmp_path, monkeypatch):
    """C2: When channel is down and writing queued/ fails, sent/ is NOT deleted beforehand so
    retry_of=<letter_id> still resolves the original letter and delivers once channel & storage recover."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-c2", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")

        def _fail_queue(*_a, **_k):
            raise peer_links.PeerLinkError("peer_links_unreadable", "disk full", 503)

        orig_record = peer_channel_queue.record_queued_message
        monkeypatch.setattr(peer_channel_queue, "record_queued_message", _fail_queue)
        with acting_as(ch.root_a):
            failed = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="C2 durable message").text)
            assert failed["ok"] is False and failed["reason"] == "peer_links_unreadable", failed
            lid = failed["letter_id"]
            assert peer_links.sending_record(ch.root_a, ch.link_id, lid) is not None, (
                "sent/ record must remain intact when queue persistence fails"
            )

        # Storage and channel recover: retry_of=<lid> succeeds with the exact same letter_id!
        monkeypatch.setattr(peer_channel_queue, "record_queued_message", orig_record)
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            retried = json.loads(
                _message_ouroboros(ctx_a, place_id=ch.place_a, text="C2 durable message", retry_of=lid).text
            )
            assert retried["status"] == "received" and retried["letter_id"] == lid, retried
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [lid]


def test_t5_queued_tool_message_reconciles_before_automatic_reply_including_startup(tmp_path, monkeypatch):
    """C3: When a focus turn queued a message_ouroboros tool send while the channel was down and also
    recorded the same words as its automatic reply, flush_link_sync and resume_owed deliver the queued
    tool letter FIRST so _tool_sent suppresses any duplicate automatic reply."""
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        _enable_consciousness(ch.root_a, monkeypatch, enabled=False)
        _enable_consciousness(ch.root_b, monkeypatch, enabled=False)
        for idx, trigger_mode in enumerate(("flush_sync", "resume_owed"), start=1):
            prod_id = f"prod-c3-{idx}"
            msg_text = f"Exact focus reply text #{idx}"
            pres_meta = {
                "wake_cause": "focus",
                "place_id": ch.place_a,
                "transport_skill": peer_links.CONNECTOR,
                "delivery_reporting_version": 1,
                "event": {"event_id": f"ev-c3-{idx}", "message": {"message_id": f"in-c3-{idx}", "peer_message": True}},
            }
            ctx_a = SimpleNamespace(
                task_id=prod_id,
                task_metadata={"wake_cause": "focus", "presence": pres_meta},
                drive_root=ch.root_a,
                budget_drive_root=ch.root_a,
            )
            ch.fwd_a_to_b.set_mode("absent")
            with acting_as(ch.root_a):
                q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text=msg_text).text)
                assert q["status"] == "queued", q
                tool_lid = q["letter_id"]
                peer_links.record_owed(ch.root_a, ch.link_id, prod_id, place_id=ch.place_a)
                _approve(SimpleNamespace(root_b=ch.root_a, place_b=ch.place_a, link_id=ch.link_id),
                         prod_id, msg_text, {"presence": pres_meta})
                write_task_result(ch.root_a, prod_id, STATUS_COMPLETED, result=msg_text,
                                  terminal_origin="model_final", chat_id=1, source="presence",
                                  metadata={"presence": pres_meta})
                # Direct _deliver_output while channel still down must NOT duplicate.
                settled, wait = peer_replies._one_pass(ch.root_a, ch.link_id, prod_id)
                assert (settled, wait) == (False, None)

            before_ids = set(_incoming_message_ids(ch.root_b, ch.link_id))
            ch.fwd_a_to_b.set_mode("forward")
            with acting_as(ch.root_a):
                if trigger_mode == "flush_sync":
                    peer_channel_queue.flush_link_sync(ch.root_a, ch.link_id)
                else:
                    peer_replies.resume_owed(ch.root_a, link_id=ch.link_id, at_start=True)
            _poll_until(_flush_threads_idle, timeout=30.0)
            _poll_until(lambda: tool_lid in _incoming_message_ids(ch.root_b, ch.link_id), timeout=10.0)
            _poll_until(lambda: peer_links.owed_refs(ch.root_a, ch.link_id) == [], timeout=10.0)
            after_ids = set(_incoming_message_ids(ch.root_b, ch.link_id)) - before_ids
            assert after_ids == {tool_lid}, f"{trigger_mode} sent duplicate automatic reply: {after_ids}"


def test_t5_hold_notice_write_failure_keeps_queued_message_until_confirmed(tmp_path, monkeypatch):
    """C4: If record_notice fails while holding a queued message (e.g. rules_changed), the queued
    record stays on disk until a later flush writes the hold notice cleanly."""
    from ouroboros import place_reply
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-c4", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="C4 held message").text)
            assert q["status"] == "queued"
            lid = q["letter_id"]
            places.edit_place(ch.root_a, ch.place_a, "add_note", author=places.AUTHOR_OWNER,
                              text="Rule changed before flush.")

        orig_notice = place_reply.record_notice
        monkeypatch.setattr(place_reply, "record_notice", lambda *a, **k: (_ for _ in ()).throw(OSError("store locked")))
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is not None, (
                "queued record must NOT be deleted when record_notice fails"
            )
            assert _incoming_message_ids(ch.root_b, ch.link_id) == []

        # Once notice storage recovers, the hold notice is recorded and the queued entry retires.
        monkeypatch.setattr(place_reply, "record_notice", orig_notice)
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is None
        texts = _message_texts_in_room(ch.root_a, ch.place_a)
        assert any("Message held:" in t for t in texts), texts


def test_t5_revoke_during_send_never_recreates_queue_and_preserves_uncertainty(tmp_path, monkeypatch):
    """C5: Revoking a link while a channel send is failing never leaves a queued/ file behind
    and preserves sent/ + uncertain state when a transport error happened."""
    from ouroboros.gateways.peer import PeerGateway, _unavailable
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-c5", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)

        def _revoke_then_transport_fail(self, text, *, letter_id, reply_to="", timeout_sec=None):
            with acting_as(ch.root_a):
                peer_links.revoke(ch.root_a, ch.link_id)
            err = _unavailable("peer_unreachable", "channel broke during revoke")
            err.channel_transport_error = True
            raise err

        monkeypatch.setattr(PeerGateway, "send_message", _revoke_then_transport_fail)
        with acting_as(ch.root_a):
            res = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="C5 racing revoke").text)
            assert res["status"] == "refused" and res["reason"] in {"unconfirmed", "delivery_outcome_unknown"}, res
            lid = res["letter_id"]
            _poll_until(_flush_threads_idle, timeout=10.0, label="revoked server pass done")
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is None, (
                "queued/ must never be recreated on a revoked link"
            )
            assert peer_links.sending_record(ch.root_a, ch.link_id, lid) is not None, (
                "uncertain sent/ record must remain preserved for reconciliation"
            )
            # Direct record_queued_message on a revoked link also refuses cleanly.
            with pytest.raises(peer_links.PeerLinkError) as exc:
                peer_channel_queue.record_queued_message(
                    ch.root_a, ch.link_id, "msg-revoked-direct", place_id=ch.place_a,
                    account_id="acc", text="hi", reply_to="", origin={"kind": "tool"},
                    approved_basis="abc", coverage_kind="owner",
                )
            assert exc.value.code == "peer_link_unavailable"


def test_t5_queued_tool_message_honors_originating_task_pause_and_stop_fences(tmp_path, monkeypatch):
    """C6: Queued message_ouroboros checks the originating task's Pause fence (stays queued until Resume)
    and Stop state (holds terminally without sending)."""
    from ouroboros import cancel_intents, owner_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        root_pause_id = "task-c6-pause-root"
        child_pause_id = "task-c6-pause-child"
        stop_task_id = "task-c6-stop"
        with acting_as(ch.root_a):
            write_task_result(ch.root_a, root_pause_id, STATUS_RUNNING, chat_id=1)
            write_task_result(ch.root_a, child_pause_id, STATUS_RUNNING, chat_id=1,
                              metadata={"root_task_id": root_pause_id})
            write_task_result(ch.root_a, stop_task_id, STATUS_RUNNING, chat_id=1)

        ctx_paused = SimpleNamespace(
            task_id=child_pause_id, root_task_id=root_pause_id,
            task_metadata={"work_kind": "owner", "root_task_id": root_pause_id},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        ctx_stopped = SimpleNamespace(
            task_id=stop_task_id, root_task_id=stop_task_id,
            task_metadata={"work_kind": "owner", "root_task_id": stop_task_id},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            qp = json.loads(_message_ouroboros(ctx_paused, place_id=ch.place_a, text="C6 paused child note").text)
            qs = json.loads(_message_ouroboros(ctx_stopped, place_id=ch.place_a, text="C6 stopped task note").text)
            assert qp["status"] == "queued" and qs["status"] == "queued"
            # Pause the first task's tree; Stop the second task.
            owner_pause.install_fence(ch.root_a, root_pause_id, request_id="req-pause-c6")
            cancel_intents.request_cancel(ch.root_a, stop_task_id, reason="owner stopped", source="owner")

        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
            # The Stop above already woke the link; this call may coalesce into that pass.
            _poll_until(_flush_threads_idle, timeout=30.0, label="flush threads done")
            assert _incoming_message_ids(ch.root_b, ch.link_id) == [], "neither paused nor stopped task may POST"
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, qp["letter_id"]) is not None, (
                "paused task's queued message must stay queued for Resume"
            )
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, qs["letter_id"]) is None, (
                "stopped task's queued message must be held terminally and retired"
            )

        # Resume the paused tree -> its queued message now delivers automatically via the real wake!
        with acting_as(ch.root_a):
            owner_pause.release_fence(ch.root_a, root_pause_id, reason="owner_resume_consumed")
        _poll_until(lambda: _incoming_message_ids(ch.root_b, ch.link_id) == [qp["letter_id"]],
                    timeout=10.0, label="release_fence woke queue and delivered paused task's letter")
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert any("Message held:" in t for t in _message_texts_in_room(ch.root_a, ch.place_a))


def test_t5_queued_tool_message_retries_on_receiver_429_retry_after(tmp_path, monkeypatch):
    """C7: When flushing a queued tool message hits receiver HTTP 429 with retry_after,
    flush_queued_messages executes bounded follow-ups with the same letter_id until delivered."""
    from ouroboros.gateways.peer import PeerGateway, _unavailable
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.03, 0.05))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_MAX_SEC", 0.2)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-c7", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="C7 retry after 429").text)
            assert q["status"] == "queued"
            lid = q["letter_id"]

        orig_send = PeerGateway.send_message
        seen = {"n": 0}

        def _rate_limited_once(self, text, *, letter_id, reply_to="", timeout_sec=None):
            seen["n"] += 1
            if seen["n"] == 1:
                err = _unavailable("presence_rate_limited", "try again soon", 429, refusal="presence_rate_limited")
                err.retry_after = 0.03
                raise err
            return orig_send(self, text, letter_id=letter_id, reply_to=reply_to, timeout_sec=timeout_sec)

        monkeypatch.setattr(PeerGateway, "send_message", _rate_limited_once)
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
        _poll_until(lambda: _incoming_message_ids(ch.root_b, ch.link_id) == [lid],
                    timeout=15.0, label="same letter received after 429 retry deadline")
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is None
        assert seen["n"] == 2
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [lid]


def test_t5_reconnect_announce_during_active_flush_is_retained_and_delivers(tmp_path, monkeypatch):
    """C11: If a reconnect announce arrives while an earlier flush is blocked on a dead socket,
    trigger_link_flush records a pending wake so the flush re-runs and delivers on the live socket."""
    from ouroboros.gateways.peer import PeerGateway, _unavailable
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-c11", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="C11 mid-flush wake").text)
            assert q["status"] == "queued"
            lid = q["letter_id"]

        in_first_flush = threading.Event()
        release_first_flush = threading.Event()
        orig_send = PeerGateway.send_message
        call_count = {"n": 0}

        def _slow_dead_first(self, text, *, letter_id, reply_to="", timeout_sec=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                in_first_flush.set()
                release_first_flush.wait(timeout=5.0)
                err = _unavailable("peer_unreachable", "old dead socket timed out")
                err.channel_transport_error = True
                raise err
            return orig_send(self, text, letter_id=letter_id, reply_to=reply_to, timeout_sec=timeout_sec)

        monkeypatch.setattr(PeerGateway, "send_message", _slow_dead_first)
        ch.fwd_a_to_b.set_mode("forward")
        peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id)
        assert in_first_flush.wait(timeout=5.0), "first flush did not enter send_message"

        # While the first flush is STILL active and blocked, a reconnect announce arrives!
        with acting_as(ch.root_a):
            peer_channel.announce_channel(ch.root_a, ch.link_id, ch.session)
        release_first_flush.set()

        # Without any second announce, the retained wake must re-run and deliver the queued message.
        _poll_until(lambda: lid in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=5.0, label="pending wake re-ran flush and delivered")
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert call_count["n"] == 2


def test_t5_backlog_flush_checks_active_first_and_enumerates_once(tmp_path, monkeypatch):
    """A1: Flushing a multi-letter backlog parses queued files once (O(N), not O(N^2)), because
    trigger_link_flush checks the active-flush set before touching disk."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-a1", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        lids = []
        with acting_as(ch.root_a):
            for i in range(10):
                q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text=f"Backlog msg {i}").text)
                assert q["status"] == "queued"
                lids.append(q["letter_id"])

        scan_calls = {"n": 0}
        orig_queued_messages = peer_channel_queue.queued_messages

        def _count_scans(*a, **k):
            scan_calls["n"] += 1
            return orig_queued_messages(*a, **k)

        monkeypatch.setattr(peer_channel_queue, "queued_messages", _count_scans)
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel.announce_channel(ch.root_a, ch.link_id, ch.session)
        _poll_until(lambda: sorted(_incoming_message_ids(ch.root_b, ch.link_id)) == sorted(lids),
                    timeout=15.0, label="all 10 backlog letters delivered")
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert 1 <= scan_calls["n"] <= 2, f"one initial pass and at most one progress pass: {scan_calls['n']}"


def test_t5_message_ouroboros_rule_edit_during_safety_refuses_and_unchanged_resumes(tmp_path, monkeypatch):
    """C16: speech_basis for message_ouroboros is captured BEFORE check_safety; an owner rule edit
    during Safety refuses rules_changed with 0 POSTs and 0 queued files, while an unchanged approval
    queues and later delivers without another model turn."""
    from ouroboros import safety
    from ouroboros.tools.registry import ToolContext, ToolRegistry

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        repo = tmp_path / "repo"
        repo.mkdir()
        ctx = ToolContext(repo_dir=repo, drive_root=ch.root_a, task_id="task-c16",
                          task_metadata={"work_kind": "owner"})
        reg = ToolRegistry(repo_dir=repo, drive_root=ch.root_a)
        reg.set_context(ctx)

        # Case 1: Owner edits place rules DURING check_safety -> refused rules_changed, 0 POST, 0 queued.
        def _safety_with_rule_edit(name, args, **_kw):
            if name == "message_ouroboros":
                with acting_as(ch.root_a):
                    places.edit_place(ch.root_a, ch.place_a, "add_note", author=places.AUTHOR_OWNER,
                                      text="Owner edited rule mid-safety review.")
            return True, ""

        monkeypatch.setattr(safety, "check_safety", _safety_with_rule_edit)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            res_edited = json.loads(
                reg.execute_result("message_ouroboros", {"place_id": ch.place_a, "text": "Stale words"}).text
            )
            assert res_edited["ok"] is False and res_edited["reason"] == "rules_changed", res_edited
            assert peer_channel_queue.queued_messages(ch.root_a, ch.link_id) == []
            assert _incoming_message_ids(ch.root_b, ch.link_id) == []

        # Case 2: No rule edit during check_safety -> queued with captured basis and resumes cleanly!
        monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, ""))
        with acting_as(ch.root_a):
            res_ok = json.loads(
                reg.execute_result("message_ouroboros", {"place_id": ch.place_a, "text": "Fresh approved words"}).text
            )
            assert res_ok["status"] == "queued", res_ok
            lid = res_ok["letter_id"]
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel.announce_channel(ch.root_a, ch.link_id, ch.session)
        _poll_until(lambda: lid in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=10.0, label="unchanged Safety-approved message delivered")
        _poll_until(_flush_threads_idle, timeout=30.0)


# ── Review Round 2 regression tests (Q1-Q5, Q7) ──────────────────────────────

def test_q1_outbound_get_and_status_read_never_wake_channel_queue(tmp_path, monkeypatch):
    """Q1: Outbound GET/HEAD via PeerGateway (including 200 status/state reads) never wakes
    peer_channel_queue or delivers queued speech."""
    from ouroboros.gateways.peer import PeerGateway
    from ouroboros.tools.peer_delegate import _message_ouroboros
    from tests._peer_letters_shared import ready_queue

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=True, a_lets_b_start=False) as ch:
        ready_queue(monkeypatch, ch.root_b)
        with acting_as(ch.root_a):
            link_a = peer_links.outgoing_link(ch.place_a, drive_root=ch.root_a)
            gw = PeerGateway(link_a, timeout_sec=5.0)
        try:
            # Create one real work letter on B so GET /api/letters/{id} returns HTTP 200.
            with acting_as(ch.root_a):
                started = gw.start_run({"letter_id": "q1-work-1", "text": "Count", "protocol": 1})
                assert started["runId"] == "q1-work-1"
            # Now queue a message while channel is down, then restore forwarder without announcing.
            ch.fwd_a_to_b.set_mode("absent")
            ctx_a = SimpleNamespace(task_id="task-q1", task_metadata={"work_kind": "owner"},
                                    drive_root=ch.root_a, budget_drive_root=ch.root_a)
            with acting_as(ch.root_a):
                q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Q1 queued note").text)
                assert q["status"] == "queued"
            ch.fwd_a_to_b.set_mode("forward")
            wake_calls: list[tuple] = []
            orig_note = peer_channel_queue.note_verified_exchange
            monkeypatch.setattr(peer_channel_queue, "note_verified_exchange",
                                lambda *a, **k: (wake_calls.append(a), orig_note(*a, **k))[1])
            with acting_as(ch.root_a):
                detail = gw.get_run("q1-work-1")
                assert detail["runId"] == "q1-work-1"
                stop_detail = gw.get_run("q1-work-1", for_stop=True)
                assert stop_detail["runId"] == "q1-work-1"
            time.sleep(0.15)
            _poll_until(_flush_threads_idle, timeout=30.0)
            assert wake_calls == [], f"outbound GET must never call note_verified_exchange: {wake_calls}"
            assert _incoming_message_ids(ch.root_b, ch.link_id) == [], "outbound GET must never flush queued speech"
        finally:
            gw.close()


def test_q2_announce_during_failed_send_before_queue_write_still_delivers(tmp_path, monkeypatch):
    """Q2: If a channel announce completes AFTER a send fails on an absent socket and BEFORE
    record_queued_message writes queued/ (while send_peer_message holds _SENDING), releasing
    the claim still triggers delivery with no second announce."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-q2-race", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        orig_record = peer_channel_queue.record_queued_message
        raced = {"done": False}

        def _announce_right_before_write(*a, **k):
            if not raced["done"]:
                raced["done"] = True
                ch.fwd_a_to_b.set_mode("forward")
                with acting_as(ch.root_a):
                    peer_channel.announce_channel(ch.root_a, ch.link_id, ch.session)
            return orig_record(*a, **k)

        monkeypatch.setattr(peer_channel_queue, "record_queued_message", _announce_right_before_write)
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Q2 race note").text)
            assert q["status"] in {"queued", "received"}, q
            lid = q["letter_id"]

        assert raced["done"] is True
        _poll_until(lambda: lid in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=8.0, label="queued message delivered despite announce racing queue write")
        _poll_until(_flush_threads_idle, timeout=30.0)


def test_q2_resume_stop_and_sleep_release_wake_channel_queue(tmp_path, monkeypatch):
    """Q2: reopen_for_resume, cold model_sleep release (set_budget_pause STATE_RESUMED), and
    cancel_intents.request_cancel each wake peer_channel_queue automatically after reconnect."""
    from ouroboros import budget_pause, cancel_intents, owner_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        t_reopen, t_sleep, t_stop = "task-q2-reopen", "task-q2-sleep", "task-q2-stop"
        with acting_as(ch.root_a):
            for tid in (t_reopen, t_sleep, t_stop):
                write_task_result(ch.root_a, tid, STATUS_RUNNING, chat_id=1)

        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q_reopen = json.loads(_message_ouroboros(
                SimpleNamespace(task_id=t_reopen, root_task_id=t_reopen, task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a),
                place_id=ch.place_a, text="Q2 reopen_for_resume note",
            ).text)
            q_sleep = json.loads(_message_ouroboros(
                SimpleNamespace(task_id=t_sleep, root_task_id=t_sleep, task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a),
                place_id=ch.place_a, text="Q2 model_sleep note",
            ).text)
            q_stop = json.loads(_message_ouroboros(
                SimpleNamespace(task_id=t_stop, root_task_id=t_stop, task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a),
                place_id=ch.place_a, text="Q2 post-reconnect stop note",
            ).text)
            assert q_reopen["status"] == q_sleep["status"] == q_stop["status"] == "queued"

            fence_r, _ = owner_pause.install_fence(ch.root_a, t_reopen, request_id="req-q2-r")
            owner_pause.install_fence(ch.root_a, t_stop, request_id="req-q2-s")
            budget_pause.set_budget_pause(
                ch.root_a, t_sleep,
                {"pause_id": "p-q2-sleep", "reason": "sleep", "state": budget_pause.STATE_PAUSED},
            )

        # Channel reconnects while all three tasks are paused/sleeping -> all three stay parked.
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel.announce_channel(ch.root_a, ch.link_id, ch.session)
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert _incoming_message_ids(ch.root_b, ch.link_id) == []

        # 1. reopen_for_resume wakes and delivers q_reopen automatically.
        with acting_as(ch.root_a):
            owner_pause.reopen_for_resume(ch.root_a, t_reopen, t_reopen,
                                          fence_id=fence_r["fence_id"], grant_id="grant-q2-r")
        _poll_until(lambda: q_reopen["letter_id"] in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=8.0, label="reopen_for_resume delivered queued letter")

        # 2. Cold model_sleep release (STATE_RESUMED) wakes and delivers q_sleep automatically.
        with acting_as(ch.root_a):
            budget_pause.set_budget_pause(
                ch.root_a, t_sleep,
                {"pause_id": "p-q2-sleep", "reason": "sleep", "state": budget_pause.STATE_RESUMED},
            )
        _poll_until(lambda: q_sleep["letter_id"] in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=8.0, label="model_sleep release delivered queued letter")

        # 3. Post-reconnect Stop wakes queue, retires q_stop without sending, and writes hold notice.
        with acting_as(ch.root_a):
            cancel_intents.request_cancel(ch.root_a, t_stop, reason="stopped after reconnect", source="owner")
        _poll_until(lambda: peer_channel_queue.queued_message(ch.root_a, ch.link_id, q_stop["letter_id"]) is None,
                    timeout=8.0, label="request_cancel retired queued letter")
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert q_stop["letter_id"] not in _incoming_message_ids(ch.root_b, ch.link_id)
        assert any("Message held:" in t for t in _message_texts_in_room(ch.root_a, ch.place_a))


def test_q3_429_not_before_survives_wakes_and_lets_other_letters_proceed(tmp_path, monkeypatch):
    """Q3: Receiver 429 retry_after persists as not_before on the queued record; external wakes
    never POST before not_before or burn passes, while other queued letters proceed immediately."""
    from ouroboros.gateways.peer import PeerGateway, _unavailable
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.05, 0.05, 0.05))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_MAX_SEC", 5.0)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-q3-429", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            qa = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Q3 rate-limited letter A").text)
            qb = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Q3 other letter B").text)
            assert qa["status"] == qb["status"] == "queued"
            lid_a, lid_b = qa["letter_id"], qb["letter_id"]

        orig_send = PeerGateway.send_message
        a_times: list[float] = []

        def _send_hook(self, text, *, letter_id, reply_to="", timeout_sec=None):
            if letter_id == lid_a:
                a_times.append(time.monotonic())
                if len(a_times) == 1:
                    err = _unavailable("presence_rate_limited", "slow down", 429, refusal="presence_rate_limited")
                    err.retry_after = 2.5
                    raise err
            return orig_send(self, text, letter_id=letter_id, reply_to=reply_to, timeout_sec=timeout_sec)

        monkeypatch.setattr(PeerGateway, "send_message", _send_hook)
        ch.fwd_a_to_b.set_mode("forward")
        peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id)
        _poll_until(lambda: lid_b in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=15.0, label="other letter B delivered immediately")
        _poll_until(lambda: len(a_times) >= 1, timeout=15.0, label="first 429 attempt of A")
        with acting_as(ch.root_a):
            _poll_until(lambda: (peer_links.sending_record(ch.root_a, ch.link_id, lid_a) or {}).get("not_before"))

        # Fire 4 rapid external wakes while not_before (2.5s) has NOT elapsed.
        for _ in range(4):
            time.sleep(0.08)
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)

        assert len(a_times) == 1, f"wakes before not_before must not re-POST A (got {len(a_times)} POSTs)"
        # Wait for not_before to pass -> A delivers cleanly on its 2nd POST!
        _poll_until(lambda: lid_a in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=15.0, label="letter A delivered once not_before passed")
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert len(a_times) == 2
        assert a_times[1] - a_times[0] >= 2.4, f"second POST arrived too early: {a_times[1] - a_times[0]:.3f}s"


def test_q3_429_after_earlier_uncertain_schedules_followup_with_same_letter_id(tmp_path, monkeypatch):
    """Q3: A 429 with retry_after arriving after an earlier uncertain attempt still schedules
    bounded follow-up and delivers with the exact same letter_id without a new announce."""
    from ouroboros.gateways.peer import PeerGateway, _unavailable
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.04, 0.04, 0.04))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_MAX_SEC", 1.0)
    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0.0)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-q3-unc429", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("accept_close")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Q3 uncertain then 429").text)
            assert q["status"] == "queued" and q["reason"] == "channel_unconfirmed", q
            lid = q["letter_id"]

        orig_send = PeerGateway.send_message
        calls = {"n": 0}

        def _429_then_ok(self, text, *, letter_id, reply_to="", timeout_sec=None):
            calls["n"] += 1
            if calls["n"] == 1:
                err = _unavailable("presence_rate_limited", "busy", 429, refusal="presence_rate_limited")
                err.retry_after = 0.05
                raise err
            return orig_send(self, text, letter_id=letter_id, reply_to=reply_to, timeout_sec=timeout_sec)

        monkeypatch.setattr(PeerGateway, "send_message", _429_then_ok)
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
        _poll_until(lambda: _incoming_message_ids(ch.root_b, ch.link_id) == [lid],
                    timeout=15.0, label="same letter received after uncertain send and 429")
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert calls["n"] == 2, f"expected follow-up POST after uncertain+429, got {calls['n']}"
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [lid]
        assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is None


def test_q3_intermediate_parent_cascade_stop_holds_grandchild_letter(tmp_path, monkeypatch):
    """Q3: A cascade Stop on an intermediate parent holds its grandchild's queued letter
    before teardown marks the grandchild cancelled."""
    from ouroboros import cancel_intents
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        root_t, mid_t, grand_t = "task-q3-root", "task-q3-mid", "task-q3-grand"
        with acting_as(ch.root_a):
            write_task_result(ch.root_a, root_t, STATUS_RUNNING, chat_id=1)
            write_task_result(ch.root_a, mid_t, STATUS_RUNNING, chat_id=1,
                              parent_task_id=root_t, metadata={"root_task_id": root_t})
            write_task_result(ch.root_a, grand_t, STATUS_RUNNING, chat_id=1,
                              parent_task_id=mid_t, metadata={"root_task_id": root_t})

        ch.fwd_a_to_b.set_mode("absent")
        ctx_g = SimpleNamespace(task_id=grand_t, root_task_id=root_t,
                                task_metadata={"work_kind": "owner", "root_task_id": root_t},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_g, place_id=ch.place_a, text="Q3 grandchild note").text)
            assert q["status"] == "queued"
            cancel_intents.request_cancel(ch.root_a, mid_t, scope=cancel_intents.SCOPE_CASCADE,
                                          reason="parent cascade stopped", source="owner")

        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
        _poll_until(_flush_threads_idle, timeout=30.0)
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [], "grandchild letter must not leave after parent cascade Stop"
        assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, q["letter_id"]) is None
        assert any("Message held:" in t for t in _message_texts_in_room(ch.root_a, ch.place_a))


def test_q4_surviving_queue_file_after_hold_notice_is_retired_not_sent(tmp_path, monkeypatch):
    """Q4: If a crash leaves a queued file after its terminal hold notice was written, recovery
    retires the queue file and NEVER sends it even if the original place rules are restored."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx_a = SimpleNamespace(task_id="task-q4-hold", task_metadata={"work_kind": "owner"},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx_a, place_id=ch.place_a, text="Q4 held crash note").text)
            assert q["status"] == "queued"
            lid = q["letter_id"]
            rec = peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid)
            assert rec is not None
            # Simulate crash AFTER hold notice is written and BEFORE settle_queued_message deletes queued/.
            ok = peer_channel_queue._hold_queued_message(
                ch.root_a, place_id=ch.place_a, link_id=ch.link_id, account_id=rec["account_id"],
                letter_id=lid, text=rec["text"], reply_to=rec["reply_to"], origin=rec["origin"],
                outcome={"state": "held", "code": "rules_changed", "message": "rules changed; nothing was sent"},
                identity=peer_channel_queue._hold_ids(lid, rec)["rules_changed"], earlier_uncertain=False, states=set(),
            )
            assert ok is True
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is not None

        # Rules are intact and channel returns: surviving queue file must be retired without sending!
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
        assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is None
        assert _incoming_message_ids(ch.root_b, ch.link_id) == []


def test_q4_unreadable_queue_file_keeps_automatic_reply_pending(tmp_path, monkeypatch):
    """Q4: If the deterministic queued tool-message file is unreadable, reconcile_matching_queued
    returns pending so the automatic reply does not send a duplicate under a different letter_id."""
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        prod = "prod-q4-dup"
        words = "Q4 shared words between tool and automatic reply"
        pres_meta = {
            "place_id": ch.place_a, "delivery_reporting_version": 1,
            "event": {"event_id": "ev-q4", "message": {"message_id": "in-q4", "peer_message": True}},
        }
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            ctx_p = SimpleNamespace(
                task_id=prod, task_metadata={"work_kind": "presence", "wake_cause": "focus", "presence": pres_meta},
                drive_root=ch.root_a, budget_drive_root=ch.root_a,
            )
            q = json.loads(_message_ouroboros(ctx_p, place_id=ch.place_a, text=words).text)
            assert q["status"] == "queued"
            tool_lid = q["letter_id"]
            peer_links.record_owed(ch.root_a, ch.link_id, prod, place_id=ch.place_a)
            _approve(SimpleNamespace(root_b=ch.root_a, place_b=ch.place_a, link_id=ch.link_id),
                     prod, words, {"presence": pres_meta})
            write_task_result(ch.root_a, prod, STATUS_COMPLETED, result=words,
                              terminal_origin="model_final", chat_id=1, source="presence",
                              metadata={"presence": pres_meta})
            # Corrupt the exact queued tool file so reading it fails.
            q_path = peer_channel_queue._queued_path(ch.root_a, ch.link_id, tool_lid)
            q_path.write_text("{corrupt-json", encoding="utf-8")

        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            settled, wait = peer_replies._one_pass(ch.root_a, ch.link_id, prod)
        assert (settled, wait) == (False, None), "unreadable matching queued file must keep automatic reply pending"
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [], "automatic reply must not send under a second id"


def test_q5_channel_prewire_nothing_sent_for_assignment_and_answer(tmp_path, monkeypatch):
    """Q5: Pre-wire channel validation codes and channel_connect_failed (absent/refuse) are
    NOTHING_SENT across start_run, delegate_to_ouroboros, and answer_interaction (before before_send),
    while accept_close/accept_silent stay UNCONFIRMED and retry_of stays unknown."""
    from ouroboros import delegate_custody as custody, safety
    from ouroboros.gateways.peer import NOTHING_SENT, UNCONFIRMED, PeerGateway
    from ouroboros.tools.peer_delegate import _delegate_to_ouroboros

    for code in ("peer_channel_refused", "peer_door_unsupported", "peer_door_dir_invalid",
                 "invalid_session", "channel_session_invalid", "channel_end_missing"):
        assert code in NOTHING_SENT, f"{code} missing from NOTHING_SENT"

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=True, a_lets_b_start=False) as ch:
        monkeypatch.setattr(safety, "check_peer_assignment", lambda *a, **k: (True, ""))
        with acting_as(ch.root_a):
            link = peer_links.outgoing_link(ch.place_a, drive_root=ch.root_a)
            gw = PeerGateway(link, timeout_sec=0.35)
        try:
            # 1. PeerGateway.start_run & answer_interaction: absent/refuse -> channel_unavailable (not UNCONFIRMED)
            #    and answer_interaction never calls before_send when socket is absent.
            for mode in ("absent", "refuse"):
                ch.fwd_a_to_b.set_mode(mode)
                with acting_as(ch.root_a), pytest.raises(Exception) as exc_s:
                    gw.start_run({"letter_id": f"q5-s-{mode}", "text": "Work", "protocol": 1})
                assert getattr(exc_s.value, "code", "") == "channel_unavailable"
                assert getattr(exc_s.value, "channel_connect_failed", False) is True

                called = {"n": 0}
                with acting_as(ch.root_a), pytest.raises(Exception) as exc_a:
                    gw.answer_interaction(
                        "q5-run-1", "q-1", [{"questionId": "q-1", "freeText": "Yes"}],
                        before_send=lambda *_a: called.__setitem__("n", called["n"] + 1),
                    )
                assert getattr(exc_a.value, "code", "") == "channel_unavailable"
                assert getattr(exc_a.value, "channel_connect_failed", False) is True
                if mode == "absent":
                    assert called["n"] == 0, "before_send must not run when channel socket is absent"

            # 2. accept_close / accept_silent -> UNCONFIRMED on both start_run and answer_interaction.
            for mode in ("accept_close", "accept_silent"):
                ch.fwd_a_to_b.set_mode(mode)
                with acting_as(ch.root_a), pytest.raises(Exception) as exc_u:
                    gw.answer_interaction("q5-run-1", "q-1", [{"questionId": "q-1", "freeText": "Yes"}])
                assert getattr(exc_u.value, "code", "") == UNCONFIRMED

            # 3. delegate_to_ouroboros: first send on absent/refuse closes unsent (definitely_unrun=True);
            #    accept_close stays unconfirmed; retry_of on absent stays unknown (not definitely_unrun).
            ctx_a = SimpleNamespace(
                task_id="task-q5-del", root_task_id="task-q5-del",
                task_metadata={"work_kind": "owner", "root_task_id": "task-q5-del"},
                drive_root=ch.root_a, budget_drive_root=ch.root_a,
            )
            for mode in ("absent", "refuse"):
                ch.fwd_a_to_b.set_mode(mode)
                with acting_as(ch.root_a):
                    res = json.loads(_delegate_to_ouroboros(
                        ctx_a, place_id=ch.place_a, title=f"Q5 {mode}", text=f"Do {mode}",
                    ).text)
                assert res["ok"] is False and res["reason"] == "channel_unavailable", res
                assert res.get("definitely_unrun") is True, f"first send on {mode} must be definitely_unrun: {res}"
                inv_id = res["pending_invocation_id"]
                with acting_as(ch.root_a):
                    assert custody.invocation_record(ch.root_a, inv_id)["state"] == "failed_definite"

            # accept_close leaves status=unconfirmed without definitely_unrun; retry_of on absent stays unknown!
            ch.fwd_a_to_b.set_mode("accept_close")
            with acting_as(ch.root_a):
                res_close = json.loads(_delegate_to_ouroboros(
                    ctx_a, place_id=ch.place_a, title="Q5 close", text="Do close",
                ).text)
            assert res_close["status"] == "unconfirmed" and not res_close.get("definitely_unrun"), res_close
            pending_id = res_close["pending_invocation_id"]
            ch.fwd_a_to_b.set_mode("absent")
            with acting_as(ch.root_a):
                res_retry = json.loads(_delegate_to_ouroboros(
                    ctx_a, place_id=ch.place_a, title="Q5 close", text="Do close", retry_of=pending_id,
                ).text)
            assert not res_retry.get("definitely_unrun"), f"retry_of must never claim definitely_unrun: {res_retry}"
        finally:
            gw.close()


def test_addressed_clarification_reports_channel_unavailable_without_pinning(tmp_path, monkeypatch):
    from ouroboros.delegate_interactions import _delegate_answer
    from ouroboros.delegate_message_store import read_peer_answer_pin
    from ouroboros.tools.registry import ToolContext
    from tests._peer_letters_shared import safety_reads
    from tests.test_peer_delegate import _seed_recovered_peer_run

    safety_reads(monkeypatch)
    with live_channel_pair(tmp_path, monkeypatch) as pair:
        run_id, iid = "offline-clarification", "q-1"
        entry = _seed_recovered_peer_run(pair, run_id, resume_start_state="running")
        ctx = ToolContext(repo_dir=tmp_path, drive_root=pair.root_a)
        ctx.task_id, ctx.task_metadata = entry.task_id, {"root_task_id": entry.task_id}
        reply_to = {"starter_task_id": entry.task_id, "run_id": run_id, "interaction_id": iid, "question_ids": [iid]}
        answers = [{"question_id": iid, "free_text": "Yes"}]
        pair.fwd_a_to_b.set_mode("absent")
        with acting_as(pair.root_a):
            direct = json.loads(_delegate_answer(ctx, run_id, iid, answers).text)
            addressed = json.loads(_delegate_answer(ctx, run_id, iid, answers, reply_to=reply_to).text)
            assert (direct["status"], direct["reason"]) == ("refused", "channel_unavailable"), direct
            assert addressed == direct
            assert read_peer_answer_pin(ctx, entry, iid) is None
            assert peer_channel_queue.queued_messages(pair.root_a, pair.link_id) == []


def test_q7_announce_channel_validates_ack_before_advancing_session(tmp_path, monkeypatch):
    """Q7: announce_channel parses and validates {ok: true, link_id, session} on HTTP 200;
    malformed or mismatched 200 responses raise malformed_response (502) without advancing session."""
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        with acting_as(ch.root_a):
            next_sess = peer_channel.allocate_channel_session(ch.root_a, ch.link_id)
            orig_sess = peer_links.read_registry(ch.root_a)["links"][ch.link_id]["channel_session"]

        bad_responses = [
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
            b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 14\r\nConnection: close\r\n\r\n{"ok": false }',
            (f'{{"ok": true, "link_id": "{"f" * 32}", "session": "{next_sess}"}}').encode("utf-8"),
            (f'{{"ok": true, "link_id": "{ch.link_id}", "session": "{orig_sess}"}}').encode("utf-8"),
        ]
        raw_frames = []
        for item in bad_responses:
            if item.startswith(b"HTTP/1.1"):
                raw_frames.append(item)
            else:
                raw_frames.append(
                    f"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {len(item)}\r\n"
                    f"Connection: close\r\n\r\n".encode("ascii") + item
                )

        ch.fwd_a_to_b.close()
        sock_l = peer_channel.opener_socket_path(ch.root_a, ch.link_id)
        for frame in raw_frames:
            sock_l.unlink(missing_ok=True)
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            srv.bind(str(sock_l))
            os.chmod(sock_l, 0o600)
            srv.listen(4)
            srv.settimeout(2.0)

            def _serve_one(s=srv, payload=frame):
                try:
                    conn, _ = s.accept()
                    with conn:
                        conn.recv(4096)
                        conn.sendall(payload)
                except OSError:
                    pass
                finally:
                    s.close()

            t = threading.Thread(target=_serve_one, daemon=True)
            t.start()
            try:
                with acting_as(ch.root_a), pytest.raises(peer_links.PeerLinkError) as exc:
                    peer_channel.announce_channel(ch.root_a, ch.link_id, next_sess)
                assert exc.value.code == "malformed_response" and exc.value.status == 502, (exc.value.code, exc.value.status)
            finally:
                t.join(timeout=3.0)
                sock_l.unlink(missing_ok=True)

            with acting_as(ch.root_a):
                assert peer_links.read_registry(ch.root_a)["links"][ch.link_id]["channel_session"] == orig_sess
