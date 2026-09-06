"""Wait-local empty proofs avoid replay without acquiring delivery authority."""

import json
import os
import queue
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import loop_transport, owner_mailbox as mailbox
from ouroboros.tools.control_task_results import _wait_attention_poll


def acknowledged_history(root, count=1000):
    path, ack = mailbox._mailbox_path(root, "parent"), mailbox._ack_path(root, "parent")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps({"msg_id": f"m{i}", "kind": "task_message", "text": "recorded"}) + "\n"
                            for i in range(count)), encoding="utf-8")
    ack.write_text("".join(json.dumps({"msg_id": f"m{i}"}) + "\n" for i in range(count)), encoding="utf-8")
    return path, ack


def count_reads(monkeypatch, paths):
    counts = Counter()
    original = Path.read_text
    def read(path, *args, **kwargs):
        if path in paths:
            counts[path] += 1
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    return counts


def test_unchanged_wait_reads_full_mailbox_and_acks_only_once(tmp_path, monkeypatch):
    path, ack = acknowledged_history(tmp_path)
    counts = count_reads(monkeypatch, {path, ack})
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="parent", task_attempt=1,
                          task_metadata={}, _loop_mailbox_seen_ids=set())
    hook = _wait_attention_poll(ctx, "", [])
    assert [hook({}, {}) for _ in range(6)] == [None] * 6
    assert counts == {path: 2, ack: 1}  # original full reader, one cold miss
    assert ctx._loop_mailbox_seen_ids == set()
    assert mailbox.write_owner_message(tmp_path, "new directive", "parent", msg_id="new")
    assert hook({}, {})["reason"] == "owner_mailbox_pending"
    assert "new" not in mailbox.acknowledged_task_message_ids(tmp_path, "parent", attempt_key=1)


def test_transport_episode_reuses_empty_proof_and_checks_incoming_queue_each_tick(tmp_path, monkeypatch):
    path, ack = acknowledged_history(tmp_path)
    counts = count_reads(monkeypatch, {path, ack})
    incoming = queue.Queue()
    def sleep(_seconds, check):
        assert all(not check() for _ in range(6))
        incoming.put("arrived in memory")
        assert check() is True
        assert incoming.get_nowait() == "arrived in memory"
    monkeypatch.setattr(loop_transport, "interruptible_wait_sleep", sleep)
    episode = loop_transport.TransportWaitEpisode(started_monotonic=time.monotonic())
    tools = SimpleNamespace(_ctx=SimpleNamespace(task_attempt=1, task_metadata={}))
    for _ in range(2):
        assert loop_transport.transport_wait_step(episode, tools=tools, error_kind="transport_unavailable",
            drive_root=tmp_path, drive_logs=tmp_path / "logs", task_id="parent", model="unused",
            emit_progress=lambda *_a, **_k: None, incoming_messages=incoming, owner_msg_seen=set())
    assert counts == {path: 2, ack: 1}
    assert loop_transport.TransportWaitEpisode().mailbox_peek is not episode.mailbox_peek


@pytest.mark.parametrize("change", ["ack_only", "replace", "truncate", "same_size", "delete_ack", "delete_mailbox"])
def test_each_source_fingerprint_change_invalidates_empty_proof(tmp_path, change):
    path, ack = acknowledged_history(tmp_path, 2)
    peek = mailbox.OwnerMailboxPeek()
    assert not peek.pending(tmp_path, "parent", set(), 1)
    if change == "ack_only":
        ack.write_text(json.dumps({"msg_id": "m1"}) + "\n", encoding="utf-8")
    elif change == "delete_ack":
        ack.unlink()
    elif change == "delete_mailbox":
        path.unlink()
        assert not peek.pending(tmp_path, "parent", set(), 1)
        mailbox.write_owner_message(tmp_path, "recreated", "parent", msg_id="new")
    elif change == "same_size":
        before = path.stat()
        path.write_bytes(path.read_bytes().replace(b'"m0"', b'"xx"'))
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000))
        assert path.stat().st_size == before.st_size
    else:
        replacement = json.dumps({"msg_id": "new", "text": "new owner text"}) + "\n"
        if change == "truncate":
            path.write_text(replacement, encoding="utf-8")
        else:
            sibling = path.with_suffix(".replacement")
            sibling.write_text(replacement, encoding="utf-8")
            sibling.replace(path)
    assert peek.pending(tmp_path, "parent", set(), 1)


def test_attempt_seen_task_and_execution_root_are_part_of_the_empty_proof(tmp_path):
    mailbox.write_owner_message(tmp_path, "durable owner text", "parent", msg_id="owner")
    mailbox.acknowledge_task_messages(tmp_path, "parent", ["owner"], wake_id="test", attempt_key=1)
    peek = mailbox.OwnerMailboxPeek()
    assert not peek.pending(tmp_path, "parent", set(), 1)
    assert peek.pending(tmp_path, "parent", set(), 2)
    assert not peek.pending(tmp_path, "parent", {"owner"}, 2)
    assert peek.pending(tmp_path, "parent", set(), 2)
    mailbox.write_owner_message(tmp_path, "other task", "other", msg_id="owner")
    assert not peek.pending(tmp_path, "parent", set(), 1)
    assert peek.pending(tmp_path, "other", set(), 1)
    other = tmp_path / "another-execution"
    mailbox.write_owner_message(other, "other execution", "parent", msg_id="owner")
    assert not peek.pending(tmp_path, "parent", set(), 1)
    assert peek.pending(other, "parent", set(), 1)
    assert not peek.pending(tmp_path, "parent", set(), None)  # unscoped ACK reader
    assert peek.pending(tmp_path, "parent", set(), "None")  # a distinct scoped attempt


@pytest.mark.parametrize("kind", [mailbox.KIND_HURRY, mailbox.KIND_FINALIZE_NOW])
def test_revocation_and_retry_keep_the_existing_control_semantics(tmp_path, kind):
    mailbox.write_owner_message(tmp_path, "control", "parent", msg_id="old", kind=kind)
    mailbox.revoke_owner_control(tmp_path, "parent", "old")
    peek = mailbox.OwnerMailboxPeek()
    assert not peek.pending(tmp_path, "parent", set(), 1)
    mailbox.write_owner_message(tmp_path, "current control", "parent", msg_id="current", kind=kind)
    assert peek.pending(tmp_path, "parent", set(), 1)
    assert not peek.pending(tmp_path, "parent", {"current"}, 1)
    assert not mailbox.acknowledged_task_message_ids(tmp_path, "parent", attempt_key=1)
    assert mailbox.reset_attempt_controls_for_retry(tmp_path, "parent") == 1
    assert not peek.pending(tmp_path, "parent", set(), 2)
    assert mailbox.drain_owner_entries(tmp_path, "parent", set(), 2) == []


@pytest.mark.parametrize("source", ["mailbox", "ack"])
@pytest.mark.parametrize("fault", ["torn", "malformed", "read_error"])
def test_failed_or_incomplete_read_is_never_a_cached_empty_result(tmp_path, monkeypatch, source, fault):
    path, ack = acknowledged_history(tmp_path, 2)
    target = path if source == "mailbox" else ack
    if fault == "torn":
        target.write_bytes(target.read_bytes().rstrip(b"\n"))
    elif fault == "malformed":
        with target.open("ab") as handle:
            handle.write(b"not-json\n")
    real = Path.read_text
    calls = []
    def read(p, *args, **kwargs):
        if p == target:
            calls.append(p)
            if fault == "read_error":
                raise PermissionError("temporary mailbox read failure")
        return real(p, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", read)
    peek = mailbox.OwnerMailboxPeek()
    for _ in range(3):
        previous = len(calls)
        assert not loop_transport._owner_signal_pending(None, tmp_path, "parent", {"m0", "m1"}, 1, peek)
        assert len(calls) > previous


def test_stat_error_does_not_make_fail_soft_empty_sticky(tmp_path, monkeypatch):
    path, ack = acknowledged_history(tmp_path, 2)
    original = Path.stat
    def denied(p, *args, **kwargs):
        if p == ack:
            raise PermissionError("temporary stat failure")
        return original(p, *args, **kwargs)
    peek = mailbox.OwnerMailboxPeek()
    with monkeypatch.context() as patch:
        patch.setattr(Path, "stat", denied)
        assert not loop_transport._owner_signal_pending(None, tmp_path, "parent", set(), 1, peek)
    counts = count_reads(monkeypatch, {path, ack})
    assert not peek.pending(tmp_path, "parent", set(), 1)
    assert counts == {path: 2, ack: 1}


def test_append_between_read_and_final_stat_is_not_cached_as_empty(tmp_path, monkeypatch):
    path, _ack = acknowledged_history(tmp_path, 2)
    original = Path.read_text
    reads = 0
    def append_after_read(p, *args, **kwargs):
        nonlocal reads
        content = original(p, *args, **kwargs)
        if p == path:
            reads += 1
            if reads == 2:  # after ACK classification, just after drain's full read
                mailbox.write_owner_message(tmp_path, "arrived after read", "parent", msg_id="racing")
        return content
    monkeypatch.setattr(Path, "read_text", append_after_read)
    peek = mailbox.OwnerMailboxPeek()
    assert not peek.pending(tmp_path, "parent", set(), 1)
    assert peek.pending(tmp_path, "parent", set(), 1)
