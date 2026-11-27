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


@pytest.mark.parametrize("change", ["ack_only", "truncate", "delete_mailbox"])
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


@pytest.mark.parametrize(("fault", "source"), [("torn", "ack"), ("malformed", "ack"), ("read_error", "mailbox")])
def test_failed_or_incomplete_read_is_never_a_cached_empty_result(tmp_path, monkeypatch, source, fault):
    path, ack = acknowledged_history(tmp_path, 2)
    target = path if source == "mailbox" else ack
    if fault == "torn":
        # Remove the whole native line terminator. A remaining CR is still
        # a complete newline when the production text reader normalizes it.
        target.write_bytes(target.read_bytes().rstrip(b"\r\n"))
        assert not target.read_bytes().endswith((b"\r", b"\n"))
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


@pytest.mark.parametrize("ack_state", ["missing", "torn", "unreadable"])
def test_acknowledged_owner_source_and_wait_completeness_remain_distinct(tmp_path, monkeypatch, ack_state):
    assert mailbox.write_owner_message(tmp_path, "Enable the repaired skill", "parent", msg_id="owner")
    ack = mailbox._ack_path(tmp_path, "parent")
    if ack_state != "missing":
        assert mailbox.acknowledge_task_messages(tmp_path, "parent", ["owner"], wake_id="delivered")
    if ack_state == "torn":
        ack.write_bytes(ack.read_bytes().rstrip(b"\r\n"))
        assert not ack.read_bytes().endswith((b"\r", b"\n"))
    elif ack_state == "malformed":
        with ack.open("ab") as stream:
            stream.write(b"not-json\n")
    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == ack and ack_state == "unreadable":
            raise PermissionError("controlled ACK read failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    delivery_status = {}
    normal = mailbox.drain_owner_entries(tmp_path, "parent", _read_status=delivery_status)
    assert delivery_status["complete"] is (ack_state in {"missing", "valid"})
    assert bool(normal) is (ack_state in {"missing", "unreadable"})
    # Exact owner-source lookup reads acknowledged content without requiring
    # the ACK projection; it still resolves all mailbox revocations.
    source_status = {}
    source = mailbox.drain_owner_entries(tmp_path, "parent", include_acknowledged=True, _read_status=source_status)
    assert source_status == {"complete": True}
    assert [row["msg_id"] for row in source] == ["owner"]
    assert mailbox.revoke_owner_control(tmp_path, "parent", "owner")
    source = mailbox.drain_owner_entries(tmp_path, "parent", include_acknowledged=True, _read_status=source_status)
    assert source == [] and source_status == {"complete": True}


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


@pytest.mark.parametrize("words", ["plain words", "Ещё одно сообщение"])
def test_a_torn_row_keeps_its_bytes_and_the_next_rows_stay_whole_and_reach_the_drain_once(tmp_path, monkeypatch, words):
    from ouroboros import utils

    real, torn = utils._write_fd_fully, []

    def tear_inside_a_character(fd, data, path):
        if not torn and str(path).endswith("worker.jsonl"):
            torn.append(data)
            os.write(fd, data[:data.index(words.encode("utf-8")) + 1])
            raise OSError("injected torn mailbox write")
        return real(fd, data, path)

    monkeypatch.setattr(utils, "_write_fd_fully", tear_inside_a_character)

    def place(msg_id):
        return mailbox.write_task_message(tmp_path, words, "worker", source_task_id="s", provenance="place_event", msg_id=msg_id,
                                          place_arrival={"source_event_id": msg_id, "message": {}, "wake": True})

    assert place("place-event:first") is False, "the bool API: a torn append is no success"
    path = mailbox._mailbox_path(tmp_path, "worker")
    prefix = path.read_bytes()
    assert prefix and not prefix.endswith(b"\n")
    assert mailbox.place_event_retained(tmp_path, "worker", "place-event:first") is False, "a torn row cannot be delivered"
    assert place("place-event:first") is True and place("place-event:first") is True, "once whole, deduplicated"
    assert place("place-event:second") is True
    assert mailbox.write_owner_message(tmp_path, "owner words", "worker", msg_id="owner-1")  # an unrelated writer keeps working
    raw = path.read_bytes()
    assert raw.startswith(prefix + b"\n"), "the torn bytes stay exactly as they landed, the next row on its own line"
    rows = [json.loads(line) for line in raw[len(prefix) + 1:].decode("utf-8").splitlines()]
    assert [row["msg_id"] for row in rows] == ["place-event:first", "place-event:second", "owner-1"]
    status = {}
    drained = mailbox.drain_owner_entries(tmp_path, "worker", set(), _read_status=status)
    assert [entry["msg_id"] for entry in drained] == ["place-event:first", "place-event:second", "owner-1"]
    assert status["complete"] is False, "a skipped row is said: the read covered less than the whole file"
    assert mailbox.mailbox_holds(tmp_path, "worker", "place-event:second")


# --- The row lookup decodes lazily and reads exactly as every mailbox reader does -------------------------------------


def spy_on_decodes(monkeypatch):
    """The mailbox lines the module hands to its decoder, in order: the module's own ``json.loads``, traced."""
    lines, real = [], json.loads

    def traced(text, *args, **kwargs):
        lines.append(text)
        return real(text, *args, **kwargs)

    monkeypatch.setattr(mailbox, "json", SimpleNamespace(**{**vars(json), "loads": traced}))
    return lines


def write_rows(root, rows, *, eol=None):
    path = mailbox._mailbox_path(root, "worker")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    if eol is None:
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    else:
        path.write_bytes(b"".join(line.encode("utf-8") + eol for line in lines))
    return path, lines


@pytest.mark.parametrize("eol", [None, b"\r\n"], ids=["native", "crlf"])
def test_the_row_lookup_decodes_only_up_to_the_first_whole_row_of_the_id(tmp_path, monkeypatch, eol):
    path, lines = write_rows(
        tmp_path,
        [{"msg_id": f"m{i}", "kind": "task_message", "text": f"row {i}\rkeep"} for i in range(40)],
        eol=eol,
    )
    decoded = spy_on_decodes(monkeypatch)
    assert mailbox._holds(path, "m0") and decoded == lines[:1], "the first row is the one: one decode, not forty"
    decoded.clear()
    assert mailbox._holds(path, "m7") and decoded == lines[:8], "no row after the match is decoded"
    decoded.clear()
    assert mailbox.mailbox_holds(tmp_path, "worker", "m3") and decoded == lines[:4], "the public lookup shares the one reader"
    decoded.clear()
    assert not mailbox._holds(path, "absent") and decoded == lines, "a miss reads every row, as it always did"
    decoded.clear()
    assert mailbox.mail_records(path) == [json.loads(line) for line in lines] and decoded == lines


def test_a_place_event_retry_is_found_without_decoding_the_mailbox_behind_it(tmp_path, monkeypatch):
    def place(msg_id):
        return mailbox.write_task_message(tmp_path, "arrived", "worker", source_task_id="s", provenance="place_event",
                                          msg_id=msg_id, place_arrival={"source_event_id": msg_id, "message": {}, "wake": True})

    ids = [f"place-event:{i}" for i in range(25)]
    assert all(place(msg_id) for msg_id in ids)
    path = mailbox._mailbox_path(tmp_path, "worker")
    kept = path.read_bytes()
    lines = {line for line in kept.decode("utf-8").split("\n") if line}
    decoded = spy_on_decodes(monkeypatch)
    assert place(ids[0]) is True, "the retry of the first row is deduplicated under the mail lock"
    assert [json.loads(line)["msg_id"] for line in decoded if line in lines] == [ids[0]], "one decode: the row it found"
    assert path.read_bytes() == kept, "nothing was appended"
    assert mailbox.place_event_retained(tmp_path, "worker", ids[-1]) and not mailbox.place_event_retained(tmp_path, "worker", "gone")


@pytest.mark.parametrize("unreadable", [b'{"msg_id": "torn", "kind": "task_message", "text": "\xd0', b"not json at all",
                                        b"[1, 2, 3]", b"  \t "], ids=["torn-character", "foreign-bytes", "no-row", "blank"])
def test_the_lookup_skips_what_it_cannot_parse_exactly_as_every_mailbox_reader_does(tmp_path, monkeypatch, unreadable):
    """An unreadable line costs only itself: the rows around it, one holding a literal U+2028 inside its words and
    one in Cyrillic, are found whole by the lookup, the list reader and the public answer alike."""
    whole = [{"msg_id": "first", "text": "a b"}, {"msg_id": "second", "text": "тест"}, {"msg_id": "third"}]
    path = mailbox._mailbox_path(tmp_path, "worker")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(unreadable + b"\n" + b"".join(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n" for row in whole))
    assert mailbox.mail_records(path) == whole == list(mailbox.iter_mail_records(path))
    for row in whole:
        assert mailbox._holds(path, row["msg_id"]) and mailbox.mailbox_holds(tmp_path, "worker", row["msg_id"])
    assert not mailbox._holds(path, "torn") and not mailbox.mailbox_holds(tmp_path, "worker", "torn")
    decoded = spy_on_decodes(monkeypatch)
    assert mailbox._holds(path, "second")
    assert decoded[-1] == json.dumps(whole[1], ensure_ascii=False) and json.dumps(whole[2], ensure_ascii=False) not in decoded


def test_an_absent_or_unreadable_mailbox_answers_as_it_did(tmp_path, monkeypatch):
    absent = mailbox._mailbox_path(tmp_path, "worker")
    assert mailbox.mail_records(absent) == [] and list(mailbox.iter_mail_records(absent)) == []
    assert not mailbox._holds(absent, "m") and not mailbox.mailbox_holds(tmp_path, "worker", "m")
    assert not mailbox.mailbox_holds(tmp_path, "../not-a-task", "m"), "a name that is no task names no mailbox"
    path, _lines = write_rows(tmp_path, [{"msg_id": "m"}])
    real_open = Path.open

    def denied(p, *args, **kwargs):
        if p == path:
            raise PermissionError("mailbox unreadable")
        return real_open(p, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    assert mailbox.mailbox_holds(tmp_path, "worker", "m") is False, "the public answer stays no, never an error"
    with pytest.raises(PermissionError):
        mailbox._holds(path, "m")  # the lock holder's lookup proves nothing it could not read


def test_first_match_lookup_and_relay_idempotence_stop_before_eof(tmp_path, monkeypatch):
    from ouroboros.owner_relay import write_relay

    quote = {"text": "keep", "attachments": [],
             "source": {"chat_id": 1, "label": "Main", "ts": "2026-11-23T00:00:00Z",
                        "address": "row:1@2026-11-23T00:00:00Z#a", "client_message_id": ""},
             "owner_row": {"text": "keep", "chat_id": 1, "ts": "2026-11-23T00:00:00Z"}}
    first, fresh = write_relay(tmp_path, "worker", relay_id="r0", note="", quotes=[quote],
                               chat_id=1, teller_task_id="", work_reply=True)
    assert fresh is True
    path = mailbox._mailbox_path(tmp_path, "worker")
    with path.open("ab") as handle:
        pad = "x" * 1024
        handle.writelines((json.dumps({"msg_id": f"m{i}", "kind": "task_message", "text": pad}) + "\n").encode("utf-8")
                          for i in range(1, 150))
    file_size = path.stat().st_size
    assert file_size > 120_000

    positions, real_open = [], Path.open

    class PositionSpy:
        def __init__(self, inner):
            self._inner = inner

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *exc):
            positions.append(self._inner.raw.tell() if hasattr(self._inner, "raw") else self._inner.tell())
            return self._inner.__exit__(*exc)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(Path, "open", lambda p, mode="r", *a, **kw:
                        PositionSpy(real_open(p, mode, *a, **kw)) if p == path else real_open(p, mode, *a, **kw))

    assert mailbox._holds(path, first["msg_id"]) is True
    assert len(positions) == 1 and 0 < positions[0] < file_size // 4

    positions.clear()
    prior, fresh_again = write_relay(tmp_path, "worker", relay_id="r0", note="", quotes=[quote],
                                     chat_id=1, teller_task_id="", work_reply=True)
    assert fresh_again is False and prior["msg_id"] == first["msg_id"]
    assert len(positions) == 1 and 0 < positions[0] < file_size // 4
    assert path.stat().st_size == file_size

    positions.clear()
    assert len(mailbox.mail_records(path)) == 150 and positions == [file_size]


def test_literal_cr_inside_line_does_not_split_under_crlf(tmp_path):
    path = mailbox._mailbox_path(tmp_path, "worker")
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = {"msg_id": "m-keep", "kind": "owner_text", "text": "a\rb\u2028c"}
    path.write_bytes(
        b'garbage\r{"msg_id": "hidden-by-cr"}\r\n'
        + json.dumps(kept, ensure_ascii=False).encode("utf-8") + b"\r\n"
    )
    assert mailbox.mail_records(path) == [kept]
    assert not mailbox._holds(path, "hidden-by-cr")
    assert mailbox._holds(path, "m-keep")
