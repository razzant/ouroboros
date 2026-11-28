"""Durable per-letter queue, compare-and-set settlement and two-process delivery."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from ouroboros import (
    cancel_intents,
    peer_channel,
    peer_channel_queue,
    peer_letters,
    peer_links,
    peer_replies,
    places,
)
from ouroboros.gateways.peer import PeerGateway, _unavailable
from ouroboros.install_identity import acting_as
from ouroboros.presence_delivery import recorded_states
from ouroboros.task_results import STATUS_CANCELLED, STATUS_COMPLETED, STATUS_RUNNING, write_task_result
from tests.test_peer_channel import (
    _flush_threads_idle,
    _incoming_message_ids,
    live_channel_pair,
)
from tests.test_peer_messages import _enable_consciousness, _poll_until
from tests.test_peer_owed_replies import _approve

pytestmark = pytest.mark.skipif(not peer_channel.is_door_supported(), reason="POSIX letters door only")


def _cleanup_link_flusher(root: pathlib.Path, link_id: str, letter_ids: list[str]) -> None:
    """Remove test queued files and wake any sleeping coordinator so _flush_threads_idle() finishes promptly."""
    for lid in letter_ids:
        peer_channel_queue.settle_queued_message(root, link_id, lid)
    key = (peer_channel_queue._canonical_root(root), link_id)
    with peer_channel_queue._FLUSHING_LOCK:
        st = peer_channel_queue._FLUSHING_LINKS.get(key)
        if st is not None:
            st["wake_event"].set()
    _poll_until(_flush_threads_idle, timeout=10.0, label="test cleanup flusher idle")


# ── C1: ContextVar approval snapshot isolation + fail-closed registry dispatch ─

def test_r3_c1_shared_tool_context_approval_scope_isolated(tmp_path, monkeypatch):
    """C1: A timed-out worker exiting message_approval_scope on a shared ToolContext never erases
    the active call's snapshot; rule change (add_note) after Safety still refuses with rules_changed,
    and registry-dispatched calls with a missing snapshot refuse fail-closed."""
    from ouroboros import safety
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_context import ToolContext

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx = ToolContext(
            repo_dir=pathlib.Path.cwd(),
            drive_root=ch.root_a,
            task_id="task-r3-c1",
            task_metadata={"work_kind": "owner", "root_task_id": "task-r3-c1"},
        )
        t1_entered = threading.Event()
        t2_in_safety = threading.Event()
        t1_exited = threading.Event()

        def _t1():
            with acting_as(ch.root_a), peer_letters.message_approval_scope(
                ctx, "message_ouroboros", {"place_id": ch.place_a, "text": "Old timed-out message"},
            ):
                t1_entered.set()
                assert t2_in_safety.wait(timeout=5.0)
            t1_exited.set()

        def _fake_safety(name, args, **kwargs):
            if name == "message_ouroboros":
                t2_in_safety.set()
                assert t1_exited.wait(timeout=5.0)
                with acting_as(ch.root_a):
                    places.edit_place(
                        ch.root_a, ch.place_a, "add_note",
                        author=places.AUTHOR_OWNER, text="Strict new rule added after Safety.",
                    )
            return True, ""

        monkeypatch.setattr(safety, "check_safety", _fake_safety)
        t1 = threading.Thread(target=_t1, daemon=True)
        t1.start()
        assert t1_entered.wait(timeout=5.0)
        try:
            reg = ToolRegistry(repo_dir=pathlib.Path.cwd(), drive_root=ch.root_a)
            reg.set_context(ctx)
            with acting_as(ch.root_a):
                res = reg.execute_result("message_ouroboros", {"place_id": ch.place_a, "text": "Approved under old rule"})
            payload = json.loads(res.text)
            assert payload.get("status") != "received", payload
            assert payload.get("reason") == "rules_changed", payload
            assert _incoming_message_ids(ch.root_b, ch.link_id) == []
        finally:
            t2_in_safety.set()
            t1.join(timeout=5.0)

        # Fail-closed: if registry-dispatched with no ContextVar approval snapshot, refuse before POST.
        @peer_letters.contextlib.contextmanager
        def _empty_scope(*_a, **_k):
            yield

        monkeypatch.setattr(peer_letters, "message_approval_scope", _empty_scope)
        monkeypatch.setattr(safety, "check_safety", lambda *a, **k: (True, ""))
        with acting_as(ch.root_a):
            res2 = reg.execute_result("message_ouroboros", {"place_id": ch.place_a, "text": "No snapshot"})
        payload2 = json.loads(res2.text)
        assert payload2.get("reason") == "approval_snapshot_missing", payload2
        assert _incoming_message_ids(ch.root_b, ch.link_id) == []


# ── Real Two-Process Harness (a): cross-process contention + mid-POST SIGKILL ──

def test_r3_two_process_a_contention_and_worker_exit_before_delivery(tmp_path, monkeypatch):
    """A real worker publishes the intent/wake and can die while the server's POST is still pending."""
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ready = tmp_path / "worker-wake.json"
        root = ch.root_a
        text = "Two-process words survive publisher exit"
        worker_env = {**os.environ, "OUROBOROS_IN_WORKER": "1", "PYTHONPATH": str(pathlib.Path.cwd())}
        worker = f"""
import json, pathlib, time
from types import SimpleNamespace
from ouroboros import peer_channel_queue
from ouroboros.gateways.peer import PeerGateway
from ouroboros.install_identity import acting_as
from ouroboros.tools.peer_delegate import _message_ouroboros
root = pathlib.Path({str(root)!r})
def on_post(*a, **k):
    raise AssertionError('worker must not POST')
PeerGateway.send_message = on_post
peer_channel_queue.bind_worker_event_queue(SimpleNamespace(put=lambda row: pathlib.Path({str(ready)!r}).write_text(json.dumps(row))))
with acting_as(root):
    ctx = SimpleNamespace(task_id='task-2p', task_metadata={{'work_kind':'owner'}}, drive_root=root, budget_drive_root=root)
    result = json.loads(_message_ouroboros(ctx, place_id={ch.place_a!r}, text={text!r}).text)
    assert result['status'] == 'queued', result
for _ in range(1500):
    time.sleep(.02)
"""
        started, release = threading.Event(), threading.Event()
        orig = PeerGateway.send_message
        posts = []

        def observed(self, *a, **kw):
            posts.append(kw["letter_id"])
            started.set()
            assert release.wait(10)
            return orig(self, *a, **kw)

        monkeypatch.setattr(PeerGateway, "send_message", observed)
        proc = subprocess.Popen([sys.executable, "-c", worker], env=worker_env)
        try:
            _poll_until(ready.exists, timeout=15, label="worker published peer_link_wake")
            evt = json.loads(ready.read_text())
            assert evt["link_id"] == ch.link_id
            with acting_as(root):
                assert len(peer_channel_queue.queued_messages(root, ch.link_id)) == 1
            peer_channel_queue.trigger_link_flush(root, evt["link_id"])
            assert started.wait(5)
            peer_channel_queue.trigger_link_flush(root, ch.link_id)  # coalesces during POST
            proc.kill()
            proc.wait(timeout=5)
            release.set()
            _poll_until(_flush_threads_idle, timeout=10)
            assert len(posts) == 1 and _incoming_message_ids(ch.root_b, ch.link_id) == posts
        finally:
            release.set()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)



def test_r3_two_process_worker_never_flushes_and_emits_peer_link_wake(tmp_path, monkeypatch):
    """Server-only delivery: a worker subprocess (OUROBOROS_IN_WORKER=1) never starts a flush thread
    and instead emits peer_link_wake to the supervisor event queue; server dispatches it to flush."""
    from supervisor.events import EVENT_HANDLERS

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            basis = peer_letters.speech_basis(ch.root_a, ch.place_a, kind="owner")
            acc = peer_links.read_registry(ch.root_a)["links"][ch.link_id]["peer_install_id"]
        lid = peer_replies.message_letter_id(ch.link_id, "task-worker-wake", "", "Worker wake letter")
        ev_file = tmp_path / "emitted_events.json"
        worker_env = {**os.environ, "OUROBOROS_IN_WORKER": "1", "PYTHONPATH": str(pathlib.Path.cwd())}
        code = f"""
import json, pathlib, threading
from ouroboros import peer_channel_queue
from ouroboros.install_identity import acting_as

events = []
class _Q:
    def put(self, item):
        events.append(item)

peer_channel_queue.bind_worker_event_queue(_Q())
root = pathlib.Path({str(ch.root_a)!r})
with acting_as(root):
    peer_channel_queue.record_queued_message(
        root, {ch.link_id!r}, {lid!r}, place_id={ch.place_a!r}, account_id={acc!r},
        text="Worker wake letter", reply_to="", origin={{"kind": "tool", "task_id": "task-worker-wake"}},
        approved_basis={basis!r}, coverage_kind="owner", root_task_id="task-worker-wake", wake=True,
    )
flush_threads = [t.name for t in threading.enumerate() if t.name.startswith(("peer-reply-", "peer-channel-"))]
assert flush_threads == [], f"worker must never start flush threads: {{flush_threads}}"
pathlib.Path({str(ev_file)!r}).write_text(json.dumps(events), encoding="utf-8")
"""
        res = subprocess.run([sys.executable, "-c", code], env=worker_env, capture_output=True, text=True, timeout=15.0)
        assert res.returncode == 0, (res.stdout, res.stderr)
        emitted = json.loads(ev_file.read_text(encoding="utf-8"))
        assert emitted == [{"type": "peer_link_wake", "drive_root": str(ch.root_a), "link_id": ch.link_id}], emitted

        # Channel returns; server dispatches the worker's peer_link_wake through EVENT_HANDLERS -> delivers!
        ch.fwd_a_to_b.set_mode("forward")
        EVENT_HANDLERS["peer_link_wake"](emitted[0], SimpleNamespace(DRIVE_ROOT=ch.root_a))
        _poll_until(lambda: lid in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=10.0, label="server delivered via peer_link_wake")
        _poll_until(_flush_threads_idle, timeout=10.0)


# ── Parent (b) + C4 + K6: 429 retry_after=3600 + 4 wakes + persist failure ───

def test_r3_b_429_retry_after_3600_wakes_and_persist_failure(tmp_path, monkeypatch):
    """Parent (b), K6, C4:
    1) Receiver 429 retry_after=3600 persists the full deadline; already elapsed time is not added again.
       Four external wakes 0.4s apart make ZERO early POSTs.
    2) The second letter keeps its independent receiver deadline through wakes and coordinator restart,
       keeps the letter pending on disk, and survives simulated restart."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (30.0, 120.0, 480.0))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_MAX_SEC", 900.0)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx = SimpleNamespace(
            task_id="task-r3-b", root_task_id="task-r3-b",
            task_metadata={"work_kind": "owner", "root_task_id": "task-r3-b"},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q1 = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="K6 3600s letter").text)
            q2 = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="C4 persist-fail letter").text)
            lid1, lid2 = q1["letter_id"], q2["letter_id"]

        orig_send = PeerGateway.send_message
        posts1: list[float] = []
        posts2: list[float] = []
        allow_deliver_2 = {"on": False}

        def _hook(self, text, *, letter_id, reply_to="", timeout_sec=None):
            if letter_id == lid1:
                posts1.append(time.monotonic())
                err = _unavailable("consciousness_allowance_exhausted", "daily reset", 429,
                                   refusal="consciousness_allowance_exhausted")
                err.retry_after = 3600.0
                raise err
            if letter_id == lid2 and not allow_deliver_2["on"]:
                posts2.append(time.monotonic())
                err = _unavailable("presence_rate_limited", "wait 30s", 429, refusal="presence_rate_limited")
                err.retry_after = 30.0
                raise err
            return orig_send(self, text, letter_id=letter_id, reply_to=reply_to, timeout_sec=timeout_sec)

        monkeypatch.setattr(PeerGateway, "send_message", _hook)
        ch.fwd_a_to_b.set_mode("forward")
        t0 = time.time()
        try:
            with acting_as(ch.root_a):
                rec1 = peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid1)
                status1, wait1 = peer_channel_queue._flush_one_queued(ch.root_a, ch.link_id, rec1)
                updated1 = peer_links.sending_record(ch.root_a, ch.link_id, lid1)
            assert status1["code"] == "retry_not_before" and 0 < wait1 <= 3600.0, (status1, wait1)
            assert float(updated1["not_before"]) - t0 >= 3590.0, updated1
            assert len(posts1) == 1
            assert time.monotonic() + wait1 >= posts1[0] + 3600.0

            # Trigger flush for lid2 and fire 4 external wakes 0.4s apart for both letters.
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)
            _poll_until(lambda: len(posts2) == 1, timeout=5.0, label="first 429 on lid2")
            for _ in range(4):
                time.sleep(0.4)
                peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)
            assert len(posts1) == 1, f"K6: zero early POSTs for retry_after=3600 (got {len(posts1)})"
            assert len(posts2) == 1, f"C4: zero early POSTs before lid2's saved deadline (got {len(posts2)})"
            with acting_as(ch.root_a):
                assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid2) is not None, "must survive on disk"

            # Simulate restart (clear the in-memory retry state) -> lid2 survives and delivers!
            peer_channel_queue.settle_queued_message(ch.root_a, ch.link_id, lid1)
            with peer_channel_queue._FLUSHING_LOCK:
                peer_channel_queue._UNPERSISTED_RETRY.clear()
            allow_deliver_2["on"] = True
            # sent/ carries the receiver's original wait even across local memory loss.
            with acting_as(ch.root_a):
                fallback = peer_links.sending_record(ch.root_a, ch.link_id, lid2)
                assert float(fallback["not_before"]) > time.time()
                monkeypatch.setattr(peer_channel_queue, "time", SimpleNamespace(
                    time=lambda: fallback["not_before"] + 1, monotonic=time.monotonic, sleep=time.sleep))
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)
            _poll_until(lambda: lid2 in _incoming_message_ids(ch.root_b, ch.link_id),
                        timeout=10.0, label="lid2 survived restart and delivered")
            _poll_until(_flush_threads_idle, timeout=10.0)
        finally:
            _cleanup_link_flusher(ch.root_a, ch.link_id, [lid1, lid2])


# ── Parent (c) + K1: Delivered-but-unacked tool letter + queue_error + auto reply ─

def test_r3_c_delivered_unacked_tool_letter_plus_queue_error_automatic_reply_single_id(tmp_path, monkeypatch):
    """No POST before queue durability. After repair the same ID deduplicates a dropped acknowledgement."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", ())
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        prod = "prod-r3-k1"
        words = "Exact shared words delivered once."
        pres_meta = {
            "place_id": ch.place_a, "delivery_reporting_version": 1,
            "event": {"event_id": "ev-k1", "message": {"message_id": "in-k1", "peer_message": True}},
        }
        ctx = SimpleNamespace(
            task_id=prod, root_task_id=prod,
            task_metadata={"work_kind": "consciousness", "wake_cause": "focus", "presence": pres_meta},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        orig_send = PeerGateway.send_message
        dropped = {"done": False}

        def _deliver_then_drop(self, text, *, letter_id, reply_to="", timeout_sec=None):
            res = orig_send(self, text, letter_id=letter_id, reply_to=reply_to, timeout_sec=timeout_sec)
            if not dropped["done"]:
                dropped["done"] = True
                err = _unavailable("unconfirmed", "socket dropped after receiver saved")
                err.channel_transport_error = True
                raise err
            return res

        def _fail_queue(*_a, **_k):
            raise peer_links.PeerLinkError("peer_links_unreadable", "disk full on queued/", 503)

        monkeypatch.setattr(PeerGateway, "send_message", _deliver_then_drop)
        orig_record_q = peer_channel_queue.record_queued_message
        monkeypatch.setattr(peer_channel_queue, "record_queued_message", _fail_queue)
        with acting_as(ch.root_a):
            tool_res = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text=words).text)
            tool_lid = tool_res["letter_id"]
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, tool_lid) is None
            assert peer_links.sending_record(ch.root_a, ch.link_id, tool_lid) is not None
            assert _incoming_message_ids(ch.root_b, ch.link_id) == []
            assert peer_links.sending_record(ch.root_a, ch.link_id, tool_lid)["pre_transmission"] is True

        # Resubmit the durable intent; the receiver saves it but the acknowledgement is dropped.
        monkeypatch.setattr(peer_channel_queue, "record_queued_message", orig_record_q)
        with acting_as(ch.root_a):
            retry = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text=words, retry_of=tool_lid).text)
            assert retry["status"] == "queued" and retry["letter_id"] == tool_lid
            assert _incoming_message_ids(ch.root_b, ch.link_id) == [tool_lid]
            peer_links.record_owed(ch.root_a, ch.link_id, prod, place_id=ch.place_a)
            _approve(SimpleNamespace(root_b=ch.root_a, place_b=ch.place_a, link_id=ch.link_id),
                     prod, words, {"presence": pres_meta})
            write_task_result(ch.root_a, prod, STATUS_COMPLETED, result=words,
                              terminal_origin="model_final", chat_id=1, source="presence",
                              metadata={"presence": pres_meta})
        peer_replies.deliver_owed(ch.root_a, prod)
        _poll_until(_flush_threads_idle, timeout=10.0)
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [tool_lid]
        with acting_as(ch.root_a):
            assert peer_links.owed_refs(ch.root_a, ch.link_id) == []
            assert peer_links.sending_record(ch.root_a, ch.link_id, tool_lid) is None


# ── Parent (d) + C3: Pre-wire failure + queue write failure then restart ─────

def test_r3_d_pre_wire_failure_plus_queue_error_restart_never_uncertain(tmp_path, monkeypatch):
    """Parent (d) + C3: connect_failed + queue_error retains a pre_transmission zero-pass envelope for retry_of,
    and neither earlier_state nor startup _reconcile_sending ever marks it 'uncertain' / 'may have arrived'."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx = SimpleNamespace(
            task_id="task-r3-c3", root_task_id="task-r3-c3",
            task_metadata={"work_kind": "owner", "root_task_id": "task-r3-c3"},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        ch.fwd_a_to_b.set_mode("absent")
        monkeypatch.setattr(
            peer_channel_queue, "record_queued_message",
            lambda *a, **k: (_ for _ in ()).throw(peer_links.PeerLinkError("peer_links_unreadable", "disk full", 503)),
        )
        with acting_as(ch.root_a):
            res = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="C3 unsent words").text)
            lid = res["letter_id"]
            acc = peer_links.read_registry(ch.root_a)["links"][ch.link_id]["peer_install_id"]
            s_rec = peer_links.sending_record(ch.root_a, ch.link_id, lid)
            assert s_rec is not None and int(s_rec["passes"]) == 0 and s_rec.get("pre_transmission") is True
            # retry_of lookup still finds the retained envelope:
            found, err = peer_replies.resolve_message_retry(ch.root_a, ch.link_id, acc, lid)
            assert found is not None and not err
            # earlier_state MUST NOT say uncertain:
            earlier, states, _rec, _e = peer_replies.earlier_state(ch.root_a, ch.link_id, acc, lid)
            assert earlier == "none", (earlier, states)
            # Startup reconciliation MUST NOT publish an 'uncertain' receipt:
            peer_replies._reconcile_sending(ch.root_a, peer_links.sending_records(ch.root_a))
            after_states = recorded_states(ch.root_a, peer_links.CONNECTOR, acc, lid, peer_replies.MESSAGE_PART)
            assert "uncertain" not in after_states, after_states
            # Even if the link is paused afterward, retrying never says "may have arrived":
            places.edit_place(ch.root_a, ch.place_a, "set_state", author=places.AUTHOR_OWNER, state=places.STATE_PAUSED)
            held = peer_replies.send_peer_message(
                ch.root_a, ch.place_a, letter_id=lid, text="C3 unsent words",
                origin={"kind": "tool", "task_id": "task-r3-c3"}, link_id=ch.link_id, account_id=acc,
            )
            assert held["state"] == "held" and "may have arrived" not in str(held.get("message") or ""), held


# ── Parent (e) + K4: SCOPE_SINGLE root Stop keeps running child's letter ─────

def test_r3_e_single_task_stop_of_root_keeps_running_child_queued_letter(tmp_path, monkeypatch):
    """Parent (e) + K4: SCOPE_SINGLE Stop of a root while its child runs lets the child's queued
    letter deliver; settled SCOPE_CASCADE Stop of a root holds and retires a child's queued letter."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        root_s, child_s = "task-r3-root-single", "task-r3-child-single"
        root_c, child_c = "task-r3-root-cascade", "task-r3-child-cascade"
        with acting_as(ch.root_a):
            write_task_result(
                ch.root_a, root_s, STATUS_CANCELLED, chat_id=1,
                cancel_origin={"request_id": "req-single", "scope": cancel_intents.SCOPE_SINGLE, "source": "owner"},
            )
            write_task_result(ch.root_a, child_s, STATUS_RUNNING, chat_id=1,
                              parent_task_id=root_s, metadata={"root_task_id": root_s})
            write_task_result(
                ch.root_a, root_c, STATUS_CANCELLED, chat_id=1,
                cancel_origin={"request_id": "req-cascade", "scope": cancel_intents.SCOPE_CASCADE, "source": "owner"},
            )
            write_task_result(ch.root_a, child_c, STATUS_RUNNING, chat_id=1,
                              parent_task_id=root_c, metadata={"root_task_id": root_c})

        ctx_s = SimpleNamespace(task_id=child_s, root_task_id=root_s,
                                task_metadata={"work_kind": "owner", "root_task_id": root_s},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ctx_c = SimpleNamespace(task_id=child_c, root_task_id=root_c,
                                task_metadata={"work_kind": "owner", "root_task_id": root_c},
                                drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            qs = json.loads(_message_ouroboros(ctx_s, place_id=ch.place_a, text="Child of single-stopped root").text)
            qc = json.loads(_message_ouroboros(ctx_c, place_id=ch.place_a, text="Child of cascade-stopped root").text)
            assert qs["status"] == "queued" and qc["status"] == "refused" and qc["reason"] == "task_stopped"

        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            peer_channel_queue.flush_queued_messages(ch.root_a, ch.link_id)
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [qs["letter_id"]]
        with acting_as(ch.root_a):
            qid = peer_replies.message_letter_id(ch.link_id, child_c, "", "Child of cascade-stopped root")
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, qid) is None


# ── A1: O(1) reconcile_matching_queued ───────────────────────────────────────

def test_r3_a1_reconcile_matching_queued_is_o1(tmp_path, monkeypatch):
    """A1: reconcile_matching_queued does O(1) direct lookup by deterministic tool letter_id and
    reads ZERO unrelated queued files."""
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        with acting_as(ch.root_a):
            basis = peer_letters.speech_basis(ch.root_a, ch.place_a, kind="owner")
            acc = peer_links.read_registry(ch.root_a)["links"][ch.link_id]["peer_install_id"]
            for i in range(20):
                lid = peer_replies.message_letter_id(ch.link_id, f"other-{i}", "", f"unrelated {i}")
                peer_channel_queue.record_queued_message(
                    ch.root_a, ch.link_id, lid, place_id=ch.place_a, account_id=acc,
                    text=f"unrelated {i}", reply_to="", origin={"kind": "tool", "task_id": f"other-{i}"},
                    approved_basis=basis, coverage_kind="owner", root_task_id=f"other-{i}", wake=False,
                )
        qdir = peer_channel_queue._queued_dir(ch.root_a, ch.link_id)
        reads = {"n": 0}
        orig_read_text = pathlib.Path.read_text

        def _count_reads(self, *a, **k):
            if self.parent == qdir and self.exists():
                reads["n"] += 1
            return orig_read_text(self, *a, **k)

        monkeypatch.setattr(pathlib.Path, "read_text", _count_reads)
        for i in range(15):
            assert not peer_channel_queue.matching_queued(
                ch.root_a, ch.link_id, producer=f"auto-{i}", text=f"reply {i}", reply_to="",
            )
        assert reads["n"] == 0, f"expected 0 unrelated queued file reads, got {reads['n']}"


# ── K2, K3, K5 ───────────────────────────────────────────────────────────────

def test_r3_k2_cross_process_verified_exchange_before_enqueue_wakes(tmp_path, monkeypatch):
    """K2: A verified exchange in another process right before worker writes queued/ advances
    channel_wake_seq so wake_after_enqueue wakes delivery instead of stranding the letter."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx = SimpleNamespace(
            task_id="task-r3-k2", root_task_id="task-r3-k2",
            task_metadata={"work_kind": "owner", "root_task_id": "task-r3-k2"},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        ch.fwd_a_to_b.set_mode("accept_close")
        orig_record = peer_channel_queue.record_queued_message
        raced = {"done": False}

        def _other_proc_verified(*a, **k):
            if not raced["done"]:
                raced["done"] = True
                ch.fwd_a_to_b.set_mode("forward")
                code = (
                    "import pathlib; from ouroboros import peer_channel_queue; "
                    f"peer_channel_queue.note_verified_exchange(pathlib.Path({str(ch.root_a)!r}), {ch.link_id!r})"
                )
                subprocess.run([sys.executable, "-c", code], check=True, timeout=15.0)
            return orig_record(*a, **k)

        monkeypatch.setattr(peer_channel_queue, "record_queued_message", _other_proc_verified)
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="K2 cross-proc wake").text)
            assert q["status"] in {"queued", "received"}, q
            lid = q["letter_id"]
        _poll_until(lambda: lid in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=10.0, label="K2 letter delivered after cross-process verified exchange")
        _poll_until(_flush_threads_idle, timeout=10.0)


def test_r3_k3_outgoing_mutating_receipt_wakes_with_receiver_off(tmp_path, monkeypatch):
    """K3: A validated outgoing POST receipt wakes sender's queued + owed work even when receiver
    Consciousness is Off, while outbound GETs stay strictly passive."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        _enable_consciousness(ch.root_b, monkeypatch, enabled=False)
        ch.fwd_a_to_b.set_mode("absent")
        ctx = SimpleNamespace(
            task_id="task-r3-k3", root_task_id="task-r3-k3",
            task_metadata={"work_kind": "owner", "root_task_id": "task-r3-k3"},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        prod, owed_words = "prod-r3-k3", "K3 owed automatic reply"
        pres_meta = {
            "place_id": ch.place_a, "delivery_reporting_version": 1,
            "event": {"event_id": "ev-k3", "message": {"message_id": "in-k3", "peer_message": True}},
        }
        with acting_as(ch.root_a):
            q_old = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="K3 queued letter").text)
            old_lid = q_old["letter_id"]
            peer_links.record_owed(ch.root_a, ch.link_id, prod, place_id=ch.place_a)
            _approve(SimpleNamespace(root_b=ch.root_a, place_b=ch.place_a, link_id=ch.link_id),
                     prod, owed_words, {"presence": pres_meta})
            write_task_result(ch.root_a, prod, STATUS_COMPLETED, result=owed_words,
                              terminal_origin="model_final", chat_id=1, source="presence",
                              metadata={"presence": pres_meta})

        # Channel returns; passive GET must NOT wake queued/owed!
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(ch.root_a):
            link = peer_links.outgoing_link(ch.place_a, drive_root=ch.root_a)
            gw = PeerGateway(link)
            try:
                with pytest.raises(Exception):
                    gw.get_run("nonexistent-k3")
            finally:
                gw.close()
        time.sleep(0.15)
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [], "passive GET must not wake queue"

        # Now sender completes one fresh verified outgoing POST -> wakes queued + owed!
        with acting_as(ch.root_a):
            fresh = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="K3 fresh POST").text)
            assert fresh["status"] == "received", fresh
        _poll_until(lambda: old_lid in _incoming_message_ids(ch.root_b, ch.link_id)
                    and peer_links.owed_refs(ch.root_a, ch.link_id) == [],
                    timeout=10.0, label="K3 outgoing receipt flushed queued and owed")
        _poll_until(_flush_threads_idle, timeout=10.0)


def test_r3_k5_resume_owed_converges_on_single_link_coordinator(tmp_path, monkeypatch):
    """K5: resume_links_work is deleted; concurrent trigger_link_flush and resume_owed share the
    single per-link coordinator without parallel flushers or double-POSTing across 429 not_before."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    assert not hasattr(peer_channel_queue, "resume_links_work")
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.05, 0.05, 0.05))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_MAX_SEC", 900.0)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx = SimpleNamespace(
            task_id="task-r3-k5", root_task_id="task-r3-k5",
            task_metadata={"work_kind": "owner", "root_task_id": "task-r3-k5"},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="K5 single coordinator").text)
            lid = q["letter_id"]

        in_post = threading.Event()
        release_post = threading.Event()
        posts: list[float] = []
        active_flushers = {"cur": 0, "max": 0}
        orig_flush_one = peer_channel_queue._flush_one_queued

        def _spy_flush_one(*a, **k):
            active_flushers["cur"] += 1
            active_flushers["max"] = max(active_flushers["max"], active_flushers["cur"])
            try:
                return orig_flush_one(*a, **k)
            finally:
                active_flushers["cur"] -= 1

        def _hook(self, text, *, letter_id, reply_to="", timeout_sec=None):
            posts.append(time.monotonic())
            in_post.set()
            release_post.wait(timeout=5.0)
            err = _unavailable("presence_rate_limited", "wait 300s", 429, refusal="presence_rate_limited")
            err.retry_after = 300.0
            raise err

        monkeypatch.setattr(peer_channel_queue, "_flush_one_queued", _spy_flush_one)
        monkeypatch.setattr(PeerGateway, "send_message", _hook)
        ch.fwd_a_to_b.set_mode("forward")
        try:
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id)
            assert in_post.wait(timeout=5.0)
            peer_replies.resume_owed(ch.root_a, link_id=ch.link_id, at_start=False)
            time.sleep(0.15)
            release_post.set()
            time.sleep(0.3)
            assert active_flushers["max"] == 1, f"expected single per-link coordinator, saw {active_flushers['max']}"
            assert len(posts) == 1, f"expected 1 POST across 300s 429, saw {len(posts)}"
        finally:
            release_post.set()
            _cleanup_link_flusher(ch.root_a, ch.link_id, [lid])


# ── New Defects #1 & #2: CAS settle_queued_message; owed replies vs tree Stop ─

def test_r3_new_defect1_settle_queued_message_compare_and_delete(tmp_path, monkeypatch):
    """New defect 1: settle_queued_message(..., seen=ver) refuses to unlink if the queued file
    changed since ver, and unlinks cleanly when ver matches."""
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        with acting_as(ch.root_a):
            basis = peer_letters.speech_basis(ch.root_a, ch.place_a, kind="owner")
            acc = peer_links.read_registry(ch.root_a)["links"][ch.link_id]["peer_install_id"]
            lid = "msg-cas-test-0001"
            peer_channel_queue.record_queued_message(
                ch.root_a, ch.link_id, lid, place_id=ch.place_a, account_id=acc,
                text="CAS message", reply_to="", origin={"kind": "tool", "task_id": "t-cas"},
                approved_basis=basis, coverage_kind="owner", root_task_id="t-cas", wake=False,
            )
            v1 = peer_channel_queue.queued_version(ch.root_a, ch.link_id, lid)
            assert v1 is not None
            # A newer server outcome changes the row; stale v1 must not delete it.
            assert peer_channel_queue._write_queue_outcome(ch.root_a, ch.link_id, lid, v1,
                {"state": "waiting", "code": "retry_not_before"}, observed_retry=1)
            assert peer_channel_queue.settle_queued_message(ch.root_a, ch.link_id, lid, seen=v1) is False
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is not None
            # Matching current version deletes cleanly!
            v2 = peer_channel_queue.queued_version(ch.root_a, ch.link_id, lid)
            assert v2 != v1
            assert peer_channel_queue.settle_queued_message(ch.root_a, ch.link_id, lid, seen=v2) is True
            assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is None


def test_r3_new_defect2_owed_automatic_reply_unaffected_by_later_tree_stop(tmp_path, monkeypatch):
    """New defect 2: An approved owed automatic reply of a settled producer is governed by place state,
    Panic, and conversation rules (speech_basis), NOT by a later Stop of the producer's task tree."""
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        root_t, prod_t = "task-r3-owed-root", "task-r3-owed-prod"
        owed_text = "Settled automatic reply survives later tree Stop."
        pres_meta = {
            "place_id": ch.place_a, "delivery_reporting_version": 1,
            "event": {"event_id": "ev-d2", "message": {"message_id": "in-d2", "peer_message": True}},
        }
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            write_task_result(ch.root_a, root_t, STATUS_RUNNING, chat_id=1)
            peer_links.record_owed(ch.root_a, ch.link_id, prod_t, place_id=ch.place_a)
            _approve(SimpleNamespace(root_b=ch.root_a, place_b=ch.place_a, link_id=ch.link_id),
                     prod_t, owed_text, {"presence": pres_meta})
            write_task_result(ch.root_a, prod_t, STATUS_COMPLETED, result=owed_text,
                              terminal_origin="model_final", chat_id=1, source="presence",
                              parent_task_id=root_t, metadata={"presence": pres_meta, "root_task_id": root_t})
            # Later Stop of the producer's task tree while the reply is waiting for the channel:
            cancel_intents.request_cancel(ch.root_a, root_t, scope=cancel_intents.SCOPE_CASCADE,
                                          reason="stopped after settlement", source="owner")
            write_task_result(
                ch.root_a, root_t, STATUS_CANCELLED, chat_id=1,
                cancel_origin={"request_id": "req-d2", "scope": cancel_intents.SCOPE_CASCADE, "source": "owner"},
            )

        ch.fwd_a_to_b.set_mode("forward")
        peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)
        expected_lid = peer_replies.message_letter_id(ch.link_id, f"{prod_t}:terminal")
        _poll_until(lambda: expected_lid in _incoming_message_ids(ch.root_b, ch.link_id),
                    timeout=10.0, label="settled owed automatic reply delivered despite later tree Stop")
        _poll_until(_flush_threads_idle, timeout=10.0)
        with acting_as(ch.root_a):
            assert peer_links.owed_refs(ch.root_a, ch.link_id) == []


# ── I3 Guard Tests: persist-under-claim, under-claim deadline re-check, hold on persist fail ──

def test_r3_i3_429_persists_before_a_coalesced_wake_can_observe_the_letter(tmp_path, monkeypatch):
    """A wake during persistence joins the running pass and then reads the new deadline."""
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.05,))
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx = SimpleNamespace(task_id="task-r3-i3", task_metadata={"work_kind": "owner"},
                              drive_root=ch.root_a, budget_drive_root=ch.root_a)
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="I3 persist before next pass").text)
        lid = q["letter_id"]
        stale = peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid)
        written, release = threading.Event(), threading.Event()
        raw = peer_links.set_sending_deadline
        posts = []

        def hold_write(*a, **kw):
            written.set()
            assert release.wait(10)
            return raw(*a, **kw)

        def limited(*_a, **kw):
            posts.append(kw["letter_id"])
            err = _unavailable("presence_rate_limited", "wait 60s", 429, refusal="presence_rate_limited")
            err.retry_after = 60.0
            raise err

        monkeypatch.setattr(peer_links, "set_sending_deadline", hold_write)
        monkeypatch.setattr(PeerGateway, "send_message", limited)
        ch.fwd_a_to_b.set_mode("forward")
        try:
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id)
            assert written.wait(5)
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id)
            assert posts == [lid]
            release.set()
            _poll_until(lambda: (peer_links.sending_record(ch.root_a, ch.link_id, lid) or {}).get("not_before"))
            # Even with a stale object, the pass reads the updated row before deciding.
            with acting_as(ch.root_a):
                status, wait = peer_channel_queue._flush_one_queued(ch.root_a, ch.link_id, stale)
            assert status["code"] == "retry_not_before" and wait > 30 and posts == [lid]
        finally:
            release.set()
            _cleanup_link_flusher(ch.root_a, ch.link_id, [lid])



def test_r3_i3_persist_failure_holds_letter_and_notices_room_until_saved(tmp_path, monkeypatch):
    """I3/C4 guard: when the sending deadline write fails on 429, the letter is held (zero timers, zero POSTs
    even after retry_after elapses while persistence still fails) and a non-terminal room notice is
    written once without retiring queued/; once storage recovers, a wake persists and delivers."""
    from tests.test_peer_channel import _message_texts_in_room
    from ouroboros.tools.peer_delegate import _message_ouroboros

    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.05, 0.05, 0.05))
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_MAX_SEC", 5.0)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False, a_lets_b_start=False) as ch:
        ctx = SimpleNamespace(
            task_id="task-r3-i3-hold", root_task_id="task-r3-i3-hold",
            task_metadata={"work_kind": "owner", "root_task_id": "task-r3-i3-hold"},
            drive_root=ch.root_a, budget_drive_root=ch.root_a,
        )
        ch.fwd_a_to_b.set_mode("absent")
        with acting_as(ch.root_a):
            q = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="I3 persist fail hold").text)
            lid = q["letter_id"]

        orig_send = PeerGateway.send_message
        fail_persist = {"on": True}
        posts: list[float] = []

        def _hook(self, text, *, letter_id, reply_to="", timeout_sec=None):
            if letter_id == lid:
                posts.append(time.monotonic())
                if len(posts) == 1:
                    err = _unavailable("presence_rate_limited", "short 429", 429, refusal="presence_rate_limited")
                    err.retry_after = 0.05
                    raise err
            return orig_send(self, text, letter_id=letter_id, reply_to=reply_to, timeout_sec=timeout_sec)

        atomic_write = peer_links.atomic_write_json

        def _fail_sent_deadline(path, record, **kw):
            if record.get("letter_id") == lid and record.get("not_before") and fail_persist["on"]:
                raise OSError("deadline write unavailable")
            return atomic_write(path, record, **kw)

        monkeypatch.setattr(PeerGateway, "send_message", _hook)
        monkeypatch.setattr(peer_links, "atomic_write_json", _fail_sent_deadline)
        ch.fwd_a_to_b.set_mode("forward")
        try:
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)
            _poll_until(lambda: len(posts) == 1, timeout=10.0, label="initial 429 attempt")
            _poll_until(_flush_threads_idle, timeout=10.0, label="no retry timer while persist failed")
            # Wait past 0.05s retry_after and fire 3 wakes while persist STILL fails -> must stay held!
            time.sleep(0.15)
            for _ in range(3):
                peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)
                _poll_until(_flush_threads_idle, timeout=10.0)
            assert len(posts) == 1, f"must stay held while deadline persist still fails (got {len(posts)} POSTs)"
            with acting_as(ch.root_a):
                assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is not None
            room_texts = _message_texts_in_room(ch.root_a, ch.place_a)
            delay_notices = [t for t in room_texts if "Message delayed:" in t]
            assert len(delay_notices) == 1, room_texts

            # Storage recovers -> wake persists the retry fact and delivers cleanly (CAS settles queued file)!
            fail_persist["on"] = False
            peer_channel_queue.trigger_link_flush(ch.root_a, ch.link_id, external=True)
            _poll_until(lambda: lid in _incoming_message_ids(ch.root_b, ch.link_id),
                        timeout=10.0, label="delivered once persist recovered")
            _poll_until(_flush_threads_idle, timeout=10.0)
            assert len(posts) == 2
            with acting_as(ch.root_a):
                assert peer_channel_queue.queued_message(ch.root_a, ch.link_id, lid) is None
        finally:
            fail_persist["on"] = False
            _cleanup_link_flusher(ch.root_a, ch.link_id, [lid])
