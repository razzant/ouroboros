"""Delivery after a retry, the originating worker's confirmed exit, or a settled producer's tree Stop."""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, peer_channel, peer_channel_queue as delivery, peer_letters, peer_links, peer_replies
from ouroboros.gateways.peer import PeerGateway, _unavailable
from ouroboros.install_identity import acting_as
from ouroboros.task_results import STATUS_CANCELLED, STATUS_COMPLETED, STATUS_RUNNING, write_task_result
from ouroboros.tools.peer_delegate import _message_ouroboros
from tests.test_peer_channel import _approve, _flush_threads_idle, _incoming_message_ids, live_channel_pair
from tests.test_peer_messages import _poll_until

pytestmark = pytest.mark.skipif(not peer_channel.is_door_supported(), reason="POSIX letters door")


def test_live_channel_recovers_after_first_pass_503_without_announce(tmp_path, monkeypatch):
    monkeypatch.setattr(peer_replies, "_FOLLOW_UP_SEC", (0.05, 0.1, 0.2))
    monkeypatch.setattr(peer_replies, "_RETRY_PAUSE_SEC", 0)
    original, calls = PeerGateway.send_message, []

    def booting(self, text, **kw):
        calls.append(kw["letter_id"])
        if len(calls) <= 3:
            raise _unavailable("unconfirmed", "peer not accepting letters yet", 503)
        return original(self, text, **kw)

    monkeypatch.setattr(PeerGateway, "send_message", booting)
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as ch:
        ctx = SimpleNamespace(task_id="503-producer", task_metadata={"work_kind": "owner"},
                              drive_root=ch.root_a, budget_drive_root=ch.root_a)
        with acting_as(ch.root_a):
            result = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text="Wait for readiness").text)
        _poll_until(_flush_threads_idle, timeout=8)
        assert calls == [result["letter_id"]] * 4, (result, calls)
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [result["letter_id"]]


def test_channel_retry_of_after_deadline_wakes_idle_delivery(tmp_path, monkeypatch):
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as ch:
        ctx = SimpleNamespace(task_id="resume-one-letter", task_metadata={"work_kind": "owner"},
                              drive_root=ch.root_a, budget_drive_root=ch.root_a)
        lid, words = "msg-" + "2" * 32, "Past due letter"
        with acting_as(ch.root_a):
            account = peer_links.read_registry(ch.root_a)["links"][ch.link_id]["peer_install_id"]
            delivery.record_queued_message(
                ch.root_a, ch.link_id, lid, place_id=ch.place_a, account_id=account, text=words, reply_to="",
                origin={"kind": "tool", "task_id": ctx.task_id},
                approved_basis=peer_letters.speech_basis(ch.root_a, ch.place_a, kind="owner"),
                coverage_kind="owner", wake=False)
            result = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text=words, retry_of=lid).text)
        _poll_until(_flush_threads_idle, timeout=8)
        assert _incoming_message_ids(ch.root_b, ch.link_id) == [lid], result


def test_confirmed_worker_exit_after_commit_before_wake_delivers_once(tmp_path, monkeypatch):
    from ouroboros import custody_owed
    from supervisor.worker_health import owe_worker_exit

    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as ch:
        _poll_until(_flush_threads_idle, timeout=8)
        text, tid = "Written before worker exit", "exiting-producer"
        lid = peer_replies.message_letter_id(ch.link_id, tid, "", text)
        marker = tmp_path / "intent-durable"
        with acting_as(ch.root_a):
            account = peer_links.read_registry(ch.root_a)["links"][ch.link_id]["peer_install_id"]
            basis = peer_letters.speech_basis(ch.root_a, ch.place_a, kind="owner")
        script = f"""
import pathlib, threading
from ouroboros import peer_channel_queue
from ouroboros.install_identity import acting_as
root = pathlib.Path({str(ch.root_a)!r})
def commit_before_wake(*args):
    pathlib.Path({str(marker)!r}).touch()
    threading.Event().wait(30)
peer_channel_queue.trigger_link_flush = commit_before_wake
with acting_as(root):
    peer_channel_queue.record_queued_message(
        root, {ch.link_id!r}, {lid!r}, place_id={ch.place_a!r}, account_id={account!r},
        text={text!r}, reply_to='', origin={{'kind': 'tool', 'task_id': {tid!r}}},
        approved_basis={basis!r}, coverage_kind='owner')
"""
        wire, posts = PeerGateway.send_message, []

        def counted(self, *a, **kw):
            posts.append(kw["letter_id"])
            return wire(self, *a, **kw)

        monkeypatch.setattr(PeerGateway, "send_message", counted)
        with subprocess.Popen([sys.executable, "-c", script], env={**os.environ, "OUROBOROS_IN_WORKER": "1",
                              "PYTHONPATH": str(pathlib.Path.cwd())}) as process:
            try:
                _poll_until(marker.exists, timeout=15, label="worker committed, wake not emitted")
                assert delivery.queued_message(ch.root_a, ch.link_id, lid)["text"] == text
                assert posts == []
            finally:
                if process.poll() is None:
                    process.kill()  # exactly the subprocess started above
                process.wait(timeout=5)
            proc = SimpleNamespace(pid=process.pid, exitcode=process.returncode,
                                   is_alive=lambda: process.poll() is None)
            owe_worker_exit(SimpleNamespace(proc=proc, process_birth="synthetic-worker-birth"), ch.root_a)
            custody_owed.drain()  # the ordinary deferred owner, off the queue lock
            _poll_until(_flush_threads_idle, timeout=8)
            assert posts == [lid] and _incoming_message_ids(ch.root_b, ch.link_id) == [lid], posts


def test_later_tree_stop_does_not_poison_adopted_approved_automatic_reply(tmp_path, monkeypatch):
    with live_channel_pair(tmp_path, monkeypatch, b_lets_a_start=False) as ch:
        root, lid, tid = ch.root_a, ch.link_id, "settled-producer"
        ancestor, words = "producer-tree", "This settled reply remains owed."
        meta = {"place_id": ch.place_a, "delivery_reporting_version": 1,
                "event": {"event_id": "arrival", "message": {"message_id": "input-1", "peer_message": True}}}
        ch.fwd_a_to_b.set_mode("accept_close")  # one ordinary tool attempt with an uncertain acknowledgement
        with acting_as(root):
            write_task_result(root, ancestor, STATUS_RUNNING, chat_id=1)
            write_task_result(root, tid, STATUS_RUNNING, chat_id=1, parent_task_id=ancestor)
            ctx = SimpleNamespace(task_id=tid, task_metadata={"work_kind": "presence", "presence": meta,
                "wake_cause": "focus", "root_task_id": ancestor}, drive_root=root, budget_drive_root=root,
                root_task_id=ancestor)
            tool = json.loads(_message_ouroboros(ctx, place_id=ch.place_a, text=words).text)
            assert tool["status"] == "queued" and tool["reason"] == "channel_unconfirmed", tool
            assert tool["letter_id"] == peer_replies.message_letter_id(lid, tid, "input-1", words)
            peer_links.record_owed(root, lid, tid, place_id=ch.place_a)
            _approve(SimpleNamespace(root_b=root, place_b=ch.place_a, link_id=lid), tid, words, {"presence": meta})
            write_task_result(root, tid, STATUS_COMPLETED, result=words, terminal_origin="model_final", chat_id=1,
                              source="presence", parent_task_id=ancestor, metadata={"presence": meta})
            # Complete the ancestor Stop before delivering anything on the working channel.
            with monkeypatch.context() as quiet:
                quiet.setattr(delivery, "trigger_link_flush", lambda *a, **kw: None)
                cancel_intents.request_cancel(root, ancestor, scope=cancel_intents.SCOPE_CASCADE,
                                              reason="tree stopped after settlement", source="owner")
                write_task_result(root, ancestor, STATUS_CANCELLED, chat_id=1,
                    cancel_origin={"request_id": "request", "scope": cancel_intents.SCOPE_CASCADE, "source": "owner"})
        ch.fwd_a_to_b.set_mode("forward")
        with acting_as(root):
            delivery.flush_link_sync(root, lid)  # queued tool hold is handled before the approved owed reply
            assert peer_links.owed_refs(root, lid) == []
        _poll_until(_flush_threads_idle, timeout=8)
        assert _incoming_message_ids(ch.root_b, lid) == [tool["letter_id"]]
