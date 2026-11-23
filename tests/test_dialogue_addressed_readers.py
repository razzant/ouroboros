"""Addressed readers use SQL truth, preserve owner bytes, and disclose incomplete import."""
from contextlib import contextmanager
import json
import pathlib

import pytest

from ouroboros import chat_chain, dialogue_import as di, dialogue_store as ds, notice_receipts, owner_words
from ouroboros.gateway.routing_decision import _origin_message, handle_routing_decision
from ouroboros.project_dialogue import (
    append_chat_annotation, build_owner_message_ref, latest_chat_annotations, resolve_owner_message_source,
    resolve_legacy_continuation_narrative,
)
from tests._dialogue import seed


def owner(client_id="owner-one", *, chat_id=1, **extra):
    return {"ts": "2026-10-17T00:00:01+00:00", "direction": "in", "chat_id": chat_id,
            "client_message_id": client_id, "text": "  Слова владельца\n\tцеликом  ", **extra}


@contextmanager
def metered(monkeypatch):
    """Measure actual SQL instructions, including short-lived composite transactions."""
    counts = {"statements": 0, "steps": 0, "transactions": 0}
    transaction = ds.transaction

    @contextmanager
    def traced(*args, **kwargs):
        counts["transactions"] += 1
        with transaction(*args, **kwargs) as conn:
            conn.set_trace_callback(lambda _: counts.__setitem__("statements", counts["statements"] + 1))
            conn.set_progress_handler(lambda: counts.__setitem__("steps", counts["steps"] + 1) or 0, 1)
            try:
                yield conn
            finally:
                conn.set_trace_callback(None)
                conn.set_progress_handler(None, 0)

    with monkeypatch.context() as patched:
        patched.setattr(ds, "transaction", traced)
        yield counts


@pytest.mark.parametrize("reverse", [False, True])
def test_probes_page_overlapping_keys_without_repeats_or_foreign_rows(tmp_path, reverse):
    receipts = []
    for index in range(9):
        receipts.append(ds.append(tmp_path, "chat", owner(f"own-{index}", task_id="selected")))
        ds.append(tmp_path, "chat", owner(f"foreign-{index}", chat_id=22))
    upper = ds.stream_end(tmp_path, "chat")
    ds.append(tmp_path, "chat", owner("later", task_id="selected"))
    entries = list(ds.iter_probes(tmp_path, "chat", [("chat_id", 1), ("task_id", "selected")],
                                 lower=receipts[1].seq, upper=upper, batch=2, newest_first=reverse))
    expected = [row.seq for row in receipts[1:]]
    assert [row.seq for row in entries] == (expected[::-1] if reverse else expected)
    assert list(ds.iter_probes(tmp_path, "chat", [], batch=2)) == []
    with pytest.raises(ValueError, match="positive"):
        list(ds.iter_probes(tmp_path, "chat", [("chat_id", 1)], batch=0))


def test_import_marker_only_intersects_unimported_range(tmp_path):
    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner("older")]}, live={"chat": [owner()]})
    ds.ensure(tmp_path)
    [segment] = di.pending_segments(tmp_path)
    live = ds.rows_by_client_id(tmp_path, "owner-one")[0]
    assert ds.range_coverage(tmp_path, "chat", upper=live.seq) == {
        "imported": False, "pending": 1, "mismatch": 0, "gaps": 0}
    assert ds.range_coverage(tmp_path, "chat", lower=live.seq)["imported"]
    assert ds.range_coverage(tmp_path, "chat", upper=0)["imported"]
    with pytest.raises(ds.StoreMigrating, match="still importing"):
        ds.require_imported(tmp_path, "chat", upper=live.seq)
    ds.require_imported(tmp_path, "chat", lower=live.seq)
    di.import_segment(tmp_path, "chat", segment["name"])
    assert ds.range_coverage(tmp_path, "chat")["imported"]
    ds.require_imported(tmp_path, "chat")


def test_origin_and_notice_import_never_claim_absence_while_import_is_pending(tmp_path):
    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner()]})
    ds.ensure(tmp_path)
    append_chat_annotation(tmp_path, "owner-one", action="route_decision", status="needs_manual_target",
                           routing_token="route-one", options=[{"action": "steer_task", "task_id": "target"}])
    status, payload = handle_routing_decision(tmp_path, request_id="click-one",
                                            decision_id="routing:owner-one:route-one", option_index=0)
    assert status == 503 and payload["error"] == "history_importing"
    assert "still importing" in payload["reason"]
    assert notice_receipts.import_upgrade_receipts(tmp_path) == {}
    from ouroboros.startup_migrations import prepare_startup_state
    assert prepare_startup_state(tmp_path)["imported"]
    with pytest.raises(ds.StoreMigrating):
        notice_receipts.recorded(tmp_path, 1, "legacy_memory_notice")
    [segment] = di.pending_segments(tmp_path)
    di.import_segment(tmp_path, "chat", segment["name"])
    assert _origin_message(tmp_path, "owner-one") == owner()
    assert _origin_message(tmp_path, "absent") == {}
    assert notice_receipts.import_upgrade_receipts(tmp_path) == {}
    assert not notice_receipts.recorded(tmp_path, 1, "legacy_memory_notice")


def test_notice_import_does_not_block_current_store_and_late_receipts_are_found(tmp_path):
    from ouroboros.startup_migrations import prepare_startup_state

    row = {"chat_id": 1, "direction": "system", "type": "legacy_memory_notice"}
    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [row]}, live={"chat": [{**row, "chat_id": 2}]})
    assert prepare_startup_state(tmp_path)["imported"]
    assert notice_receipts.recorded(tmp_path, 2, row["type"])
    with pytest.raises(ds.StoreMigrating):
        notice_receipts.recorded(tmp_path, 1, row["type"])
    [segment] = di.pending_segments(tmp_path)
    di.import_segment(tmp_path, "chat", segment["name"])
    assert notice_receipts.recorded(tmp_path, 1, row["type"])
    assert not notice_receipts.recorded(tmp_path, 3, row["type"])


@pytest.mark.parametrize("reader", ["owner", "exact-owner", "publication", "narrative"])
def test_absence_readers_answer_over_the_readable_rows_of_a_root_with_a_torn_line(tmp_path, reader):
    """A line the import could not read is an unreadable old row, as it was for the chain readers:
    answers are given over the readable rows, so one torn line in the archive never makes a reader
    refuse forever (a finished task's Done card, an owed notice, a routing click), a partial owner ref
    included. Readers with a coverage field disclose the interval (``range_coverage``)."""
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.terminal_projection import _already_in_chat

    def result_ref(task_id):
        return {"kind": "task_result", "task_id": task_id, "reader": "get_task_result"}

    summary = {"task_id": "selected", "chat_id": 1, "type": "task_summary",
               "summary_kind": "authored_root_summary", "summary_id": "task-narrative:selected",
               "result_ref": result_ref("selected"), "source_coverage": {"task_result": result_ref("selected")},
               "text": "The retained answer"}
    for gapped in (False, True):
        root = tmp_path / ("gapped" if gapped else "complete")
        rows = [owner("selected"), summary, {"chat_id": 1, "direction": "system", "type": "legacy_memory_notice"}]
        seed(root, live={"chat": rows + (["{broken\n"] if gapped else [])})
        ds.ensure(root)
        reads = {
            "owner": lambda key: resolve_owner_message_source(root, build_owner_message_ref(
                chat_id=1, client_message_id=key, ts=owner()["ts"], text=owner()["text"])),
            "exact-owner": lambda key: resolve_owner_message_source(root, owner_row_source(owner(key))[0]),
            "routing": lambda key: _origin_message(root, key),
            "words": lambda key: owner_words._inbound_rows(root, {key: "steer_task"}),
            "notices": lambda key: notice_receipts.recorded(root, 1 if key == "selected" else 22, "legacy_memory_notice"),
            "publication": lambda key: _already_in_chat(root, {**summary, "task_id": key,
                                                                "summary_id": f"task-narrative:{key}"}),
            "narrative": lambda key: resolve_legacy_continuation_narrative(root, key, result_ref(key)),
        }
        assert reads[reader]("selected")
        assert not reads[reader]("absent")
        assert bool(ds.range_coverage(root, "chat")["gaps"]) is gapped, "the torn line stays disclosed by coverage"


@pytest.mark.parametrize("source_room,exact", [(1, False), (None, False), (1, True)],
                         ids=["recorded-room", "legacy-room-unknown", "physical-address"])
def test_a_torn_line_leaves_the_routing_card_open_until_its_source_is_readable(tmp_path, monkeypatch, source_room, exact):
    from types import SimpleNamespace
    from ouroboros import chat_annotations as journal, routing_wait
    from supervisor import workers

    seed(tmp_path, live={"chat": ["{broken\n"]})
    ds.ensure(tmp_path)
    monkeypatch.setattr(journal, "_COMPACT_AT_BYTES", 1)
    dispatched = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put_nowait=dispatched.append))
    monkeypatch.setattr(routing_wait, "wait_for_routing_annotation", lambda *args: {"status": "delivered"})
    original = owner()
    address = chat_chain.format_address(chat_chain.row_address(original))
    append_chat_annotation(tmp_path, "owner-one", action="route_decision", status="needs_manual_target",
                           routing_token="route-one", source_chat_id=source_room,
                           source_row_address=address if exact else "", options=[{"action": "steer_task", "task_id": "target"}])
    before = latest_chat_annotations(tmp_path)
    assert before["owner-one"]["status"] == "needs_manual_target"
    request = {"request_id": "click-one", "decision_id": "routing:owner-one:route-one", "option_index": 0}
    status, payload = handle_routing_decision(tmp_path, **request)
    # Over the readable rows the origin is not there: the card says so and stays open, never a 503 forever.
    assert status == 409 and payload["error"] == "origin_text_unavailable" and payload["state"] == "open"
    assert latest_chat_annotations(tmp_path) == before and dispatched == []
    assert exact or "unread line" in payload["reason"]
    ds.append(tmp_path, "chat", original)
    if not exact:
        # An old card names only an id: two distinct readable rows with it are told apart by nothing it saved, so
        # the same card read against such a history stays open, and the owner can retry the same click.
        journal_path = pathlib.Path("logs/chat_annotations.jsonl")
        twin = tmp_path / "twin"
        seed(twin, live={"chat": [owner("owner-one", text="Another request"), original]})
        (twin / journal_path).write_bytes((tmp_path / journal_path).read_bytes())
        status, payload = handle_routing_decision(twin, **request)
        assert (status, payload["state"], dispatched) == (409, "open", [])
    # The torn line hides only itself: the one readable row answers the card, exact or not.
    status, payload = handle_routing_decision(tmp_path, **request)
    assert status == 200 and payload["dispatched"] == "delivered"
    # The owner's words travel as his verified quote; the steer note is only his (absent) comment.
    [event] = dispatched
    assert (event["quotes"][0]["text"], event["quotes"][0]["source"]["address"]) == (original["text"], address)
    assert latest_chat_annotations(tmp_path)["owner-one"]["status"] == "delivered"


@pytest.mark.parametrize("reader", ["owner", "routing", "notices"])
def test_named_reader_has_equal_sql_work_with_one_or_thirty_foreign_rooms(tmp_path, monkeypatch, reader):
    row = owner()
    ref = build_owner_message_ref(chat_id=1, client_message_id=row["client_message_id"], ts=row["ts"], text=row["text"])
    counts = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        ds.append(root, "chat", row)
        ds.append(root, "chat", {"chat_id": 1, "direction": "system", "type": "legacy_memory_notice", "ts": row["ts"]})
        for room in range(rooms):
            for number in range(10):
                ds.append(root, "chat", owner(f"foreign-{room}-{number}", chat_id=room + 10))
        reads = {
            "owner": lambda: resolve_owner_message_source(root, ref),
            "routing": lambda: _origin_message(root, "owner-one"),
            "words": lambda: owner_words._inbound_rows(root, {"owner-one": "steer_task"}),
            "notices": lambda: notice_receipts.import_upgrade_receipts(root),
        }
        with metered(monkeypatch) as measured:
            result = reads[reader]()
        counts.append(dict(measured))
        expected = {"1:legacy_memory_notice": {"chat_id": 1, "type": "legacy_memory_notice",
                                             "ts": row["ts"], "recorded": True}}
        assert result == (expected if reader == "notices" else [row] if reader == "words" else row)
        assert not (root / "logs" / "chat.jsonl").exists()
    assert counts[0] == counts[1] and counts[0]["steps"] > 0


def test_annotation_compaction_uses_stored_ids_and_keeps_pending_import(tmp_path, monkeypatch):
    from ouroboros import chat_annotations as journal

    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner()]})
    monkeypatch.setattr(journal, "_COMPACT_AT_BYTES", 1)
    append_chat_annotation(tmp_path, "owner-one", action="steer_task", status="delivered")
    append_chat_annotation(tmp_path, "absent", action="steer_task", status="delivered")
    assert set(latest_chat_annotations(tmp_path)) == {"owner-one", "absent"}
    [segment] = di.pending_segments(tmp_path)
    di.import_segment(tmp_path, "chat", segment["name"])
    append_chat_annotation(tmp_path, "owner-one", action="steer_task", status="delivered")
    assert set(latest_chat_annotations(tmp_path)) == {"owner-one"}


def test_annotation_compaction_over_a_torn_line_keeps_every_source_and_the_newest_act(tmp_path, monkeypatch):
    """A gap cannot prove an annotation's source absent, so nothing is dropped for it; the sidecar
    still compacts to the newest act per message instead of growing without bound."""
    from ouroboros import chat_annotations as journal

    seed(tmp_path, live={"chat": ["{broken\n"]})
    ds.ensure(tmp_path)
    monkeypatch.setattr(journal, "_COMPACT_AT_BYTES", 1)
    for status in ("queued", "delivering", "delivered"):
        append_chat_annotation(tmp_path, "unreadable-source", action="steer_task", status=status)
    assert ds.range_coverage(tmp_path, "chat")["gaps"]
    assert latest_chat_annotations(tmp_path)["unreadable-source"]["status"] == "delivered"
    sidecar = tmp_path / "logs" / "chat_annotations.jsonl"
    assert len(sidecar.read_text(encoding="utf-8").splitlines()) == 1, "compacted to the newest act"


def test_legacy_and_scoped_receipts_keep_append_order_through_compaction(tmp_path, monkeypatch):
    from ouroboros import chat_annotations as journal

    ds.append(tmp_path, "chat", owner())
    monkeypatch.setattr(journal, "_COMPACT_AT_BYTES", 1)
    times = iter(["2024-01-01T10:00:00Z", "2024-01-01T10:01:00Z", "2024-01-01T09:59:00Z"])
    monkeypatch.setattr(journal, "utc_now_iso", lambda: next(times))
    for room, token, status in [(1, "first", "pending"), (1, "second", "scheduled"), (None, "first", "delivered")]:
        assert append_chat_annotation(tmp_path, "owner-one", source_chat_id=room, routing_token=token,
                                      status=status, action="steer_task")
    assert journal.latest_picker_annotation(tmp_path, "owner-one", "first")[1]["status"] == "delivered"
    assert journal.scoped_chat_annotations(tmp_path)[1, "owner-one"]["status"] == "delivered"
    assert [row["status"] for row in journal.scoped_chat_annotations(tmp_path, by_token=True).values()] == [
        "scheduled", "delivered"]


def test_annotation_compaction_asks_the_store_once_however_many_messages_it_keeps(tmp_path, monkeypatch):
    """Past the threshold every append compacts. A transaction per annotated message cost the owner's busy server
    seconds an append (each releases the GIL; 823 messages, 14.11); one statement asks for every source, and the
    rows kept stay the per-message answer: each source's newest act, none of an absent source."""
    from ouroboros import chat_annotations as journal

    counts = []
    for messages in (1, 30):
        root = tmp_path / str(messages)
        ids = [f"owner-{number}" for number in range(messages)]
        # Another room's row first: a root caches its cutover fact from its second store access on, so neither
        # measured compaction pays that one-time read.
        ds.append(root, "chat", owner("foreign", chat_id=22))
        for message_id in ids:
            ds.append(root, "chat", owner(message_id))
            append_chat_annotation(root, message_id, action="steer_task", status="queued", routing_token="act")
        quoted = 'ответ "в" \\22'  # carried only by another room's outbound row: any chat row keeps a source
        ds.append(root, "chat", {**owner(quoted, chat_id=22), "direction": "out"})
        append_chat_annotation(root, quoted, action="steer_task", status="queued", routing_token="act")
        append_chat_annotation(root, "absent", action="steer_task", status="delivered", routing_token="act")
        with monkeypatch.context() as patched, metered(monkeypatch) as measured:
            patched.setattr(journal, "_COMPACT_AT_BYTES", 1)
            append_chat_annotation(root, ids[-1], action="steer_task", status="delivered", routing_token="act")
        counts.append((measured["transactions"], measured["statements"]))
        sidecar = (root / "logs" / "chat_annotations.jsonl").read_text(encoding="utf-8")
        kept = [(row["client_message_id"], row["status"]) for row in map(json.loads, sidecar.splitlines())]
        assert kept == [*((name, "queued") for name in ids[:-1]), (quoted, "queued"), (ids[-1], "delivered")]
    assert counts[0] == counts[1], counts


def test_continuation_narrative_is_addressed_to_its_task_and_checks_result_identity(tmp_path):
    result_ref = {"kind": "task_result", "task_id": "selected", "reader": "get_task_result"}
    summary = {"task_id": "selected", "chat_id": 1, "type": "task_summary",
               "summary_kind": "authored_root_summary", "summary_id": "task-narrative:selected",
               "result_ref": result_ref, "source_coverage": {"task_result": result_ref}, "text": "Kept answer"}
    assert resolve_legacy_continuation_narrative(tmp_path, "selected", result_ref) is None
    receipt = ds.append(tmp_path, "chat", summary)
    ds.append(tmp_path, "chat", {**summary, "task_id": "foreign", "text": "Another task"})
    ds.append(tmp_path, "chat", {**summary, "result_ref": {**result_ref, "task_id": "foreign"}, "text": "Wrong ref"})
    actual = resolve_legacy_continuation_narrative(tmp_path, "selected", result_ref)
    assert actual["text"] == "Kept answer" and actual["source"]["history_id"] == f"chat:{receipt.seq}"
    assert actual["source"]["path"] == str(ds.path(tmp_path))
    assert resolve_legacy_continuation_narrative(tmp_path, "foreign", result_ref) is None


def test_quiz_last_inbound_keeps_older_valid_stamp_and_discloses_pending_import(tmp_path):
    from types import SimpleNamespace
    from ouroboros.tools.core_artifacts import _quiz_host_facts

    ctx = SimpleNamespace(task_metadata={})
    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner()]})
    assert "history is still importing" in _quiz_host_facts(ctx, tmp_path, "asking", 1)
    [segment] = di.pending_segments(tmp_path)
    di.import_segment(tmp_path, "chat", segment["name"])
    ds.append(tmp_path, "chat", owner("invalid-stamp", ts="invalid"))
    ds.append(tmp_path, "chat", owner("foreign", chat_id=99))
    facts = _quiz_host_facts(ctx, tmp_path, "asking", 1)
    assert "2026-10-17 00:00 UTC" in facts and "still importing" not in facts
    assert "your last message in this chat: unknown." in _quiz_host_facts(ctx, tmp_path, "asking", 22)


def test_quiz_last_inbound_only_needs_coverage_after_the_known_owner_row(tmp_path):
    from types import SimpleNamespace
    from ouroboros.tools.core_artifacts import _quiz_host_facts

    ctx = SimpleNamespace(task_metadata={})
    seed(tmp_path, live={"chat": [owner(), "{broken\n"]})
    facts = _quiz_host_facts(ctx, tmp_path, "asking", 1)
    assert "your last message in this chat: unknown" in facts and "invalid intervals" in facts
    ds.append(tmp_path, "chat", owner("later", ts="2026-10-17T01:00:00+00:00"))
    facts = _quiz_host_facts(ctx, tmp_path, "asking", 1)
    assert "2026-10-17 01:00 UTC" in facts and "invalid intervals" not in facts


def test_routing_recent_text_rows_report_only_missing_parts_of_the_window(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import server_routing_context as routing

    monkeypatch.setattr(routing, "addressable_roots", lambda *args: routing.RootSnapshot())
    monkeypatch.setattr(routing, "_recent_root_results", lambda *args: ([], {}))
    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner("archive")]})
    ds.ensure(tmp_path)
    ds.append(tmp_path, "chat", owner())
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path)
    partial = routing._main_routing_manifest(ctx)
    assert partial["omissions"]["dialogue_gaps"] == ["history is still importing"]
    shown = partial["recent_canonical_dialogue"][0]
    assert "text" not in shown and shown["chars"] == len(owner()["text"].strip())
    address = shown["reader"]["arguments"]
    assert address["from"] == address["to"]
    assert chat_chain.resolve_row(tmp_path, address["from"])[0] == owner()
    for index in range(20):
        ds.append(tmp_path, "chat", owner(f"recent-{index}"))
        ds.append(tmp_path, "chat", {"chat_id": 1, "type": "task_summary"})
    covered = routing._main_routing_manifest(ctx)
    assert covered["omissions"]["dialogue_gaps"] == []
    assert [row["client_message_id"] for row in covered["recent_canonical_dialogue"]] == [f"recent-{i}" for i in range(20)]
    # A Reply saved before ``target_home``, to the archived row: while that segment imports, then after it failed, the
    # room rule cannot read its target. The manifest keeps its Projects and every row, as the Main turn receives it;
    # that row reads in its written chat (the reader before the rule), and the gap names why.
    from ouroboros.chat_references import reference_from_row
    from ouroboros.projects_registry import create_project

    create_project(tmp_path, "alpha", name="Alpha")
    legacy = {key: value for key, value in reference_from_row(owner("archive")).items() if key != "target_home"}
    ds.append(tmp_path, "chat", owner("legacy-reply", reply_to=legacy))
    for gap in ("history is still importing", "a segment could not be imported"):
        if gap == "a segment could not be imported":
            [segment] = di.pending_segments(tmp_path)
            (tmp_path / "archive" / segment["name"]).write_text("changed after cutover\n", encoding="utf-8")
            di.import_segment(tmp_path, "chat", segment["name"])
        held = routing._main_routing_manifest(ctx)
        assert held["omissions"]["dialogue_gaps"] == [gap] and [p["project_id"] for p in held["projects"]] == ["alpha"]
        assert [row["client_message_id"] for row in held["recent_canonical_dialogue"]] == [
            *(f"recent-{i}" for i in range(1, 20)), "legacy-reply"]
        assert held["recent_canonical_dialogue"][-1]["reader"]["arguments"]["room_id"] == "1"
        assert routing._decision_turn_metadata(ctx, 1, "", {}, routing.RootSnapshot())["main_routing_manifest"] == held

    def failing(error):
        def row_rooms(*_args):
            raise error
        return row_rooms

    # Any other store failure of the rule is no import gap: the manifest fails as before, and its handler logs it.
    monkeypatch.setattr("ouroboros.room_membership.row_rooms", failing(ds.StoreBusy("contention")))
    with pytest.raises(ds.StoreBusy):
        routing._main_routing_manifest(ctx)
    clean = tmp_path / "clean"
    ds.append(clean, "chat", owner())  # a stream imported whole: there a plain StoreError is no import gap either
    monkeypatch.setattr("ouroboros.room_membership.row_rooms", failing(ds.StoreError("dialogue store unavailable")))
    with pytest.raises(ds.StoreError, match="unavailable"):
        routing._main_routing_manifest(SimpleNamespace(DRIVE_ROOT=clean))


def test_routing_manifest_holds_a_legacy_reply_whose_target_imports_in_progress(tmp_path, monkeypatch):
    """The room rule also reads a legacy Reply's target from progress: with chat imported whole and that target's
    progress segment pending, then failed, the manifest keeps its Projects and the row in its written chat, and
    names the gap."""
    from types import SimpleNamespace
    from ouroboros import server_routing_context as routing
    from ouroboros.chat_references import reference_from_row
    from ouroboros.projects_registry import create_project

    monkeypatch.setattr(routing, "addressable_roots", lambda *args: routing.RootSnapshot())
    monkeypatch.setattr(routing, "_recent_root_results", lambda *args: ([], {}))
    progress = {"chat_id": 1, "ts": "2026-10-16T00:00:00+00:00", "direction": "out", "task_id": "writer",
                "is_progress": True, "text": "Work in progress"}
    legacy = {key: value for key, value in reference_from_row(progress).items() if key != "target_home"}
    seed(tmp_path, archives={"progress_20261016T000000.jsonl": [progress]},
         live={"chat": [owner("legacy-reply", reply_to=legacy)]})
    ds.ensure(tmp_path)
    create_project(tmp_path, "alpha", name="Alpha")
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path)
    for gap in ("history is still importing", "a segment could not be imported"):
        if gap == "a segment could not be imported":
            [segment] = di.pending_segments(tmp_path)
            (tmp_path / "archive" / segment["name"]).write_text("changed after cutover\n", encoding="utf-8")
            di.import_segment(tmp_path, "progress", segment["name"])
        assert ds.range_coverage(tmp_path, "chat")["imported"]
        assert ds.range_coverage(tmp_path, "progress")["mismatch" if "could not" in gap else "pending"] == 1
        held = routing._main_routing_manifest(ctx)
        assert held["omissions"]["dialogue_gaps"] == [gap] and [p["project_id"] for p in held["projects"]] == ["alpha"]
        [row] = held["recent_canonical_dialogue"]
        assert row["client_message_id"] == "legacy-reply" and row["reader"]["arguments"]["room_id"] == "1"
        assert routing._decision_turn_metadata(ctx, 1, "", {}, routing.RootSnapshot())["main_routing_manifest"] == held


def test_routing_recent_text_readers_name_a_room_that_shows_each_row(tmp_path, monkeypatch):
    """Each recent row's reader follows the room rule memory_read reads by, and every reader is executed. A work moved
    from Main into a Project keeps its ``row:1@…`` addresses and ``chat_id`` 1, and its readers name the Project,
    which returns each row, as it returns that work's reply written to a transport chat; an unmoved row reads in the
    chat it was written in; a row no room shows (A2A) is read there too, and that reading says it did not return
    the row, never a silent complete."""
    from types import SimpleNamespace
    from ouroboros import server_routing_context as routing
    from tests.test_memory_view_room import _moved_work, _returned

    monkeypatch.setattr(routing, "addressable_roots", lambda *args: routing.RootSnapshot())
    monkeypatch.setattr(routing, "_recent_root_results", lambda *args: ([], {}))
    root, rooms, moved, registry = _moved_work(tmp_path)
    abroad = {"ts": "2026-09-05T10:36:00+00:00", "chat_id": 777, "direction": "out", "task_id": "mover",
              "text": "answered on the transport", "transport": {"provider": "telegram", "conversation_id": "c7"}}
    ds.append(root, "chat", abroad)
    moved = [*moved, chat_chain.format_address(chat_chain.row_address(abroad))]
    ds.append(root, "chat", owner("a2a-row", chat_id=-5))
    rows = routing._main_routing_manifest(SimpleNamespace(DRIVE_ROOT=root))["recent_canonical_dialogue"]
    by_room = {}
    for row in rows:
        args = row["reader"]["arguments"]
        assert args["from"] == args["to"] and args["rows"] is True
        by_room.setdefault(args["room_id"], []).append(args["from"])
        found, text = _returned(registry, args["room_id"], args["from"], args["to"])
        if row["chat_id"] == -5:
            assert args["room_id"] == "-5" and found == [] and text.split("\n")[0].endswith("did not return that row")
        else:
            assert found == [args["from"]], (row, text)
            assert args["room_id"] == (str(rooms["alpha"]) if args["from"] in moved else str(row["chat_id"]))
    assert set(moved) <= set(by_room[str(rooms["alpha"])]) and by_room["1"] and not set(moved) & set(by_room["1"])
    assert {row["chat_id"] for row in rows if row["reader"]["arguments"]["from"] in moved} == {1, 777}


def test_routing_recent_text_discloses_recorded_invalid_intervals(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import server_routing_context as routing

    monkeypatch.setattr(routing, "addressable_roots", lambda *args: routing.RootSnapshot())
    monkeypatch.setattr(routing, "_recent_root_results", lambda *args: ([], {}))
    seed(tmp_path, live={"chat": [owner(), "{broken\n"]})
    ds.ensure(tmp_path)
    result = routing._main_routing_manifest(SimpleNamespace(DRIVE_ROOT=tmp_path))
    assert result["omissions"]["dialogue_gaps"] == ["recorded invalid intervals"]
    shown = result["recent_canonical_dialogue"][0]
    assert "text" not in shown and shown["chars"] == len(owner()["text"].strip())
    address = shown["reader"]["arguments"]
    assert address["from"] == address["to"]
    assert chat_chain.resolve_row(tmp_path, address["from"])[0] == owner()


def test_empty_accepted_wake_keeps_late_rows_with_older_timestamps(tmp_path):
    from ouroboros.consciousness_wake import _chat_window

    first, window = _chat_window(tmp_path, None, 2_000_000_000, set())
    assert first == [] and window["boundary"]["upper"] == 0
    row = owner(ts="2000-01-01T00:00:00+00:00")
    ds.append(tmp_path, "chat", row)
    rows, second = _chat_window(tmp_path, window["boundary"], 2_000_000_001, set())
    assert [entry for _seq, entry in rows] == [row]
    assert second["basis"] == "accepted_boundary"
    assert _chat_window(tmp_path, second["boundary"], 2_000_000_002, set())[0] == []


def test_initialization_reads_existing_dialogue_evidence_and_never_creates_a_store(tmp_path):
    from ouroboros import store
    from supervisor.state_initialization import initialization_decision, supervisor_evidence

    assert supervisor_evidence(tmp_path) == ("none", "")
    assert not ds.present(tmp_path)
    ds.create(tmp_path)
    assert supervisor_evidence(tmp_path) == ("none", "")
    ds.append(tmp_path, "progress", {"chat_id": 1, "text": "scaffolding alone is not owner history"})
    assert supervisor_evidence(tmp_path) == ("none", "")
    ds.append(tmp_path, "chat", owner())
    assert supervisor_evidence(tmp_path) == ("present", str(pathlib.Path("state", "dialog.sqlite")))  # OS separators
    assert initialization_decision(tmp_path)["reason"] == "prior_history_without_state"
    store.close_connections()
    ds.path(tmp_path).write_bytes(b"unreadable dialogue store")
    assert supervisor_evidence(tmp_path)[0] == "unknown"
    assert initialization_decision(tmp_path)["reason"] == "initialization_evidence_unknown"


@pytest.mark.parametrize("named_time", [False, True])
def test_legacy_owner_binding_keeps_the_missing_chat_id_convention(tmp_path, named_time):
    row = owner()
    row.pop("chat_id")
    ds.append(tmp_path, "chat", row)
    ds.append(tmp_path, "chat", owner("foreign", chat_id=22))
    ref = build_owner_message_ref(chat_id=1, client_message_id="", ts=row["ts"] if named_time else "", text=row["text"])
    assert resolve_owner_message_source(tmp_path, ref) == row
    assert resolve_owner_message_source(tmp_path, {**ref, "text_sha256": "0" * 64}) is None
    # A later card may repeat these words (even in the same clock tick). It has its
    # own identity and must not shadow a pre-identity inbound source or its corpus.
    quiz = {"ts": row["ts"], "chat_id": 1, "direction": "out", "type": "quiz_answer",
            "source": "owner_quiz_answer", "client_message_id": "quiz_answer:asker:choice",
            "text": "host answer frame", "quiz": {"state": "answered", "comment": row["text"]}}
    ds.append(tmp_path, "chat", quiz)
    assert resolve_owner_message_source(tmp_path, ref) == row
    words = owner_words._message_rows(tmp_path, "legacy-task", None, ref, "origin_message")
    assert [word["text"] for word in words] == [row["text"]]
    exact_quiz = build_owner_message_ref(chat_id=1, client_message_id=quiz["client_message_id"],
                                         ts=quiz["ts"], text=row["text"])
    assert resolve_owner_message_source(tmp_path, exact_quiz) == quiz


def test_mismatched_import_is_not_coverage_or_absence(tmp_path):
    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner()]})
    ds.ensure(tmp_path)
    [segment] = di.pending_segments(tmp_path)
    path = tmp_path / "archive" / segment["name"]
    path.write_text("changed after cutover\n", encoding="utf-8")
    di.import_segment(tmp_path, "chat", segment["name"])
    assert ds.range_coverage(tmp_path, "chat") == {"imported": False, "pending": 0, "mismatch": 1, "gaps": 0}
    with pytest.raises(ds.StoreError, match="could not be imported"):
        ds.require_imported(tmp_path, "chat")
    with pytest.raises(ds.StoreError, match="could not be imported"):
        _origin_message(tmp_path, "owner-one")
    assert ds.range_coverage(tmp_path, "chat", lower=segment["base"] + segment["size"])["imported"]


@pytest.mark.parametrize("reader", ["identity", "room", "task"])
def test_memory_readers_do_equal_work_with_one_or_thirty_foreign_rooms(tmp_path, monkeypatch, reader):
    from ouroboros import chat_chain
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.memory import Memory

    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        row = owner(chat_id=777, task_id="selected")
        ds.append(root, "chat", row)
        for room in range(rooms):
            for number in range(3):
                ds.append(root, "chat", owner(f"foreign-{room}-{number}", chat_id=room + 10))
        reads = {
            "identity": lambda: chat_chain.resolve_row(root, chat_chain.row_address(row))[0],
            "room": lambda: [r for _a, r, _p in chat_chain.iter_room_rows(root, 777)],
            "evidence": lambda: read_room_source(root, 777),
            "task": lambda: Memory(root).read_task_recent("chat.jsonl", "selected", 5),
        }
        with metered(monkeypatch) as measured:
            result = reads[reader]()
        assert result
        if reader == "evidence":
            assert len(result["rows"]) == 1 and result["rows"][0]["text"] == row["text"]
        elif reader == "task":
            # #4012: the task reader also names the row in the retained store — ``dialogue_store.Row.history_id``,
            # the store's own coordinate ("<stream>:<seq>"), which the mind's clipped progress line quotes. The
            # row's data, its order, the read's volume and its gaps are what they were; the id is that coordinate.
            task_rows, task_coverage = result
            stored = ds.probe_rows(root, "chat", [("task_id", "selected")], limit=5)
            assert task_rows == [{**row, "history_id": stored[0].history_id}]
            assert stored[0].history_id == "chat:0"
            assert task_coverage["shown"] == task_coverage["matched"] == 1 and task_coverage["gaps"] == []
        else:
            assert result == (row if reader == "identity" else [row])
        measurements.append(dict(measured))
    assert measurements[0] == measurements[1] and measurements[0]["steps"] > 0


def test_legacy_origin_timestamp_does_not_scan_the_source_rooms_history(tmp_path, monkeypatch):
    import json
    from ouroboros import chat_chain
    from ouroboros.projects_registry import create_project

    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        project = create_project(root, "selected", origin="reconcile")
        row = owner(client_id="", ts="2026-10-01T00:00:00+00:00")
        ref = build_owner_message_ref(chat_id=1, client_message_id="", ts=row["ts"], text=row["text"])
        (root / "state/project_task_bindings.json").write_text(json.dumps({"bindings": {
            "selected": {"project_id": project["id"], "project_chat_id": project["chat_id"], "source_ref": ref}
        }}), encoding="utf-8")
        ds.append(root, "chat", row)
        for room in range(rooms):
            for number in range(3):
                # The foreign histories share Main's physical address, as before promotion.
                ds.append(root, "chat", owner(f"foreign-{room}-{number}", task_id=f"other-{room}"))
        own = {"chat_id": project["chat_id"], "text": "own reply", "direction": "out"}
        ds.append(root, "chat", own)
        with metered(monkeypatch) as measured:
            rows = [entry.row for entry in chat_chain.room_rows(root, project["chat_id"])]
        assert rows == [row, own]
        measurements.append(dict(measured))
    assert measurements[0] == measurements[1] and measurements[0]["steps"] > 0


def test_memory_readers_disclose_pending_failed_and_invalid_import(tmp_path):
    from ouroboros import chat_chain
    from ouroboros.chronicle_store import ChronicleStore
    from ouroboros.dialogue_evidence import read_room_source
    from ouroboros.memory import Memory

    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner(chat_id=777), "{broken\n"]})
    ds.ensure(tmp_path)
    assert "history_importing" in Memory(tmp_path).chat_history()
    assert read_room_source(tmp_path, 777)["coverage"]["chat"]["gaps"]
    assert ChronicleStore(tmp_path).ensure_activated()["kind"] == "import_pending"
    with pytest.raises(ds.StoreMigrating):
        list(chat_chain.iter_room_rows(tmp_path, 777))
    di.run_dialogue_import(tmp_path)
    receipt = ChronicleStore(tmp_path).ensure_activated()
    assert receipt["kind"] == "activation" and receipt["metadata"]["dialogue_gaps"]
    assert "completeness unknown" in Memory(tmp_path).chat_history()
    missing = tmp_path / "mismatch"
    seed(missing, archives={"chat_20261016T000000.jsonl": [owner(chat_id=777)]})
    ds.ensure(missing)
    (missing / "archive/chat_20261016T000000.jsonl").write_text("changed\n")
    di.run_dialogue_import(missing)
    assert "history_import_failed" in Memory(missing).chat_history()
    assert ChronicleStore(missing).ensure_activated()["kind"] == "import_refused"


def test_memory_recall_keeps_the_frozen_public_golden(tmp_path):
    import json
    from ouroboros.memory import Memory
    from tests.test_dialogue_reader_golden import GOLDEN, conversation, _normalize

    conversation(tmp_path)
    di.run_dialogue_import(tmp_path)
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))["memory"]
    assert _normalize(Memory(tmp_path).chat_history(count=20), tmp_path) == expected


@pytest.mark.parametrize("reader", ["reentry", "page", "wake"])
def test_presence_readers_do_equal_sql_work_with_one_or_thirty_foreign_rooms(tmp_path, monkeypatch, reader):
    from ouroboros import presence_continuation as pc
    from ouroboros.consciousness_wake import _chat_window
    from ouroboros.presence_delivery import PresenceDeliveryRecorder
    from tests.test_presence_continuation_seams import _row, KEY
    from tests.test_presence_delivery import _payload

    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        for room in range(rooms):
            for number in range(3):
                ds.append(root, "chat", owner(f"foreign-{room}-{number}", chat_id=room + 10))
        recorder = PresenceDeliveryRecorder(root)
        recorder.record("transport", _payload())
        cursor = pc._chat_cursor(root)
        _rows, window = _chat_window(root, None, 0, set())
        row = _row(KEY, "new own words")
        ds.append(root, "chat", row)
        history = pc.capture_reentry_history(root, cursor)
        reads = {
            "reentry": lambda: pc._conversation_rows_since(root, cursor, KEY),
            "page": lambda: pc.reentry_history_page(root, history, KEY, history["start"]),
            "receipt": lambda: PresenceDeliveryRecorder(root).record("transport", _payload()),
            "wake": lambda: _chat_window(root, window["boundary"], 0, set()),
        }
        with metered(monkeypatch) as measured:
            result = reads[reader]()
        if reader == "reentry":
            assert result == ([row], [])
        elif reader == "page":
            assert result["rows"] == [row] and result["interval_exhausted"] and not result["gaps"]
        elif reader == "receipt":
            assert result["duplicate"] and result["history_coverage"] == "indexed"
        else:
            assert [value for _seq, value in result[0]] == [row]
            assert result[1]["basis"] == "accepted_boundary"
        measurements.append(dict(measured))
    assert measurements[0] == measurements[1] and measurements[0]["steps"] > 0


@pytest.mark.parametrize("active", [False, True])
def test_addressed_room_summary_does_equal_sql_work_with_foreign_history(tmp_path, monkeypatch, active):
    from ouroboros.projects_registry import create_project, projects_summary

    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        project = create_project(root, "selected", origin="reconcile")
        ds.ensure(root)
        if active:
            ds.append(root, "chat", owner(chat_id=project["chat_id"]), feed=True)
        for room in range(rooms):
            for number in range(3):
                ds.append(root, "chat", owner(f"foreign-{room}-{number}", chat_id=room + 10))
        with metered(monkeypatch) as measured:
            [result] = projects_summary(root, chat_id=project["chat_id"])
        assert result["has_thread_activity"] is active and result["visible_revision"] == int(active)
        measurements.append(dict(measured))
    assert measurements[0] == measurements[1] and measurements[0]["steps"] > 0


@pytest.mark.parametrize("stream", ["chat", "progress"])
def test_telegram_tail_does_equal_sql_work_with_foreign_history(tmp_path, monkeypatch, stream):
    from tests.test_telegram_subagent_cards import _load_plugin

    _load_plugin()
    from tg_sub_test.lib.telegram_state import _jsonl_tail

    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        for room in range(rooms):
            ds.append(root, stream, {"chat_id": room + 10, "text": "older foreign"})
        for index in range(4):
            ds.append(root, stream, {"chat_id": 1, "text": f"latest-{index}"})
        with metered(monkeypatch) as measured:
            rows, omitted = _jsonl_tail(root / f"logs/{stream}.jsonl", max_entries=3, tail_bytes=262144)
        assert [row["text"] for row in rows] == ["latest-1", "latest-2", "latest-3"] and omitted
        measurements.append(dict(measured))
    assert measurements[0] == measurements[1] and measurements[0]["steps"] > 0


@pytest.mark.parametrize("stream", ["chat", "progress"])
def test_benchmark_dialogue_queries_do_equal_work_with_foreign_rooms(tmp_path, monkeypatch, stream):
    from devtools.benchmarks.terminal_bench import atif

    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        own = {"task_id": "selected", "direction": "out", "text": "own answer"}
        ds.append(root, stream, own)
        for room in range(rooms):
            ds.append(root, stream, {"chat_id": room + 10, "task_id": f"other-{room}",
                                     "direction": "out", "text": "foreign answer"})
        # Trace the converter's exact SQL through the common VM meter. Its real
        # read-only connection is exercised by terminal_bench/test_atif.py.
        monkeypatch.setattr(atif, "_open_dialogue", lambda _path: ds.transaction(root, write=False))
        with metered(monkeypatch) as measured:
            rows = atif._dialogue_rows(root, stream, task_id="selected", final=stream == "chat")
        assert rows == [own]
        measurements.append(dict(measured))
    assert measurements[0] == measurements[1] and measurements[0]["steps"] > 0


def test_the_sidebar_keeps_every_project_while_the_archive_imports(tmp_path):
    from ouroboros import dialogue_import as di
    from ouroboros.projects_registry import create_project, projects_summary

    quiet = create_project(tmp_path, "quiet-room", name="Quiet")
    seed(tmp_path, archives={"chat_20261001T000000.jsonl": [{"ts": "2026-10-01T00:00:00Z", "direction": "in",
                                                             "chat_id": 1, "text": "old"}]})
    di.cutover(tmp_path)  # the archive segment is planned, not imported yet
    listed = projects_summary(tmp_path)
    assert [row["id"] for row in listed] == [quiet["id"]] and listed[0]["has_thread_activity"] is False
    with pytest.raises(ds.StoreMigrating):  # the addressed room still answers its typed outcome
        projects_summary(tmp_path, chat_id=quiet["chat_id"])
    di.run_dialogue_import(tmp_path)
    assert projects_summary(tmp_path, chat_id=quiet["chat_id"])[0]["has_thread_activity"] is False


def test_a_pre_identity_project_opening_resolves_to_its_one_original_and_refuses_a_rival(tmp_path):
    """A binding older than client ids and times names only a room and a checksum. Its Project still holds the one
    readable original of those words (lens refs, retained opening, memory membership, room rows), every such address
    of a read checked by one walk of its room; a second distinct row with the same words leaves nothing to project,
    and the retained copy says so instead of picking either."""
    from ouroboros import memory_inventory
    from ouroboros.project_dialogue import _source_ref_identity, project_origin_rows, source_refs_for_project
    from ouroboros.projects_registry import create_project

    project = create_project(tmp_path, "selected", origin="reconcile")
    original = owner(client_id="", ts="2026-01-02T00:00:00+00:00")
    ref = build_owner_message_ref(chat_id=1, client_message_id="", ts="", text=original["text"])
    (tmp_path / "state/project_task_bindings.json").write_text(json.dumps({"bindings": {"legacy": {
        "project_id": project["id"], "project_chat_id": project["chat_id"], "source_ref": ref,
        "source_text": original["text"]}}}), encoding="utf-8")
    ds.append(tmp_path, "chat", original)
    ds.append(tmp_path, "chat", owner("other", text="Other words"))
    key = _source_ref_identity(ref)

    def projected():
        gaps: set = set()
        return (source_refs_for_project(tmp_path, project["chat_id"]),
                [row["text"] for row in project_origin_rows(tmp_path, project["chat_id"], gaps=gaps)], gaps,
                memory_inventory._membership_facts(tmp_path).projects_by_source_key.get(key),
                [entry.row for entry in chat_chain.room_rows(tmp_path, project["chat_id"])])

    assert projected() == ([ref], [original["text"]], set(), frozenset({project["chat_id"]}), [original])
    ds.append(tmp_path, "chat", owner("rival", ts="2026-01-03T00:00:00+00:00"))  # the same words, a distinct row
    assert projected() == ([], [], {"The visibility of a retained project opening could not be verified."}, None, [])


@pytest.mark.parametrize("reader", ["refs", "origins", "membership", "room"])
def test_older_project_bindings_resolve_in_one_read_however_many_a_room_holds(tmp_path, monkeypatch, reader):
    """Every pre-upgrade binding names its opening by a partial address. One read resolves all of them together
    (their client ids in one probe), so a room with twenty such bindings costs the SQL work of a room with one: no
    lookup per binding for the lens refs, the retained openings, the memory membership or the room's rows (Opus
    measured 381 ms for the membership of 1,000 such bindings)."""
    from ouroboros import memory_inventory
    from ouroboros.project_dialogue import project_origin_rows, source_refs_for_project
    from ouroboros.projects_registry import create_project

    work = []
    for count in (1, 20):
        root = tmp_path / str(count)
        project = create_project(root, "selected", origin="reconcile")
        bindings, rows = {}, []
        for index in range(count):
            row = owner(f"opening-{index}", ts=f"2026-10-01T00:00:{index:02d}+00:00")
            ds.append(root, "chat", row)
            rows.append(row)
            bindings[f"task-{index}"] = {"project_id": project["id"], "project_chat_id": project["chat_id"],
                                         "source_text": row["text"], "source_ref": build_owner_message_ref(
                                             chat_id=1, client_message_id=row["client_message_id"], ts=row["ts"],
                                             text=row["text"])}
        (root / "state/project_task_bindings.json").write_text(json.dumps({"bindings": bindings}), encoding="utf-8")
        # Each filled root starts the measured read from the same premise: its chat confirmed imported, read here
        # and not under the meter, so neither count pays the process's first coverage read for the other.
        assert ds.range_coverage(root, "chat")["imported"]
        reads = {
            "refs": lambda: source_refs_for_project(root, project["chat_id"]),
            "origins": lambda: project_origin_rows(root, project["chat_id"]),
            "membership": lambda: list(memory_inventory._membership_facts(root).projects_by_source_key),
            "room": lambda: [entry.row for entry in chat_chain.room_rows(root, project["chat_id"])],
        }
        with metered(monkeypatch) as measured:
            result = reads[reader]()
        assert result == rows if reader == "room" else len(result) == count, result
        work.append((measured["transactions"], measured["statements"]))
    assert work[0] == work[1] and work[0][0] > 0


def test_a_legacy_selector_names_the_cross_room_origin_by_its_physical_row(tmp_path):
    """A Project task opened from Main records its origin with that row's digest, while a skill's owner_source still
    sends the four older fields. Such a selector is the task's own source when its one original is that physical row
    (also against an origin recorded before digests); another row, an ambiguous selector or a task with no recorded
    origin stays refused."""
    from types import SimpleNamespace
    from ouroboros.owner_relay import owner_row_source
    from ouroboros.owner_source import resolve_owner_source
    from ouroboros.task_results import write_task_result

    row = owner()
    ds.append(tmp_path, "chat", row)
    other = owner("other-one", text="Unrelated words")
    ds.append(tmp_path, "chat", other)
    recorded, words = owner_row_source(row)
    legacy = build_owner_message_ref(chat_id=1, client_message_id=row["client_message_id"], ts=row["ts"], text=row["text"])
    write_task_result(tmp_path, "room-task", "running", chat_id=77, origin_message_ref=recorded, origin_message_text=words)
    write_task_result(tmp_path, "older-task", "running", chat_id=77, origin_message_ref=legacy, origin_message_text=words)

    def source(ref, task_id="room-task"):
        ctx = SimpleNamespace(task_id=task_id, drive_root=tmp_path, current_chat_id=77, task_metadata={})
        return resolve_owner_source(ctx, {"kind": "chat", "ref": ref}).get("ref")

    assert source(legacy) == source(recorded) == source(legacy, "older-task") == legacy
    stranger = build_owner_message_ref(chat_id=1, client_message_id="other-one", ts=other["ts"], text=other["text"])
    assert source(stranger) is None and source(legacy, "no-origin") is None
    ds.append(tmp_path, "chat", owner("twin", ts=row["ts"]))  # the same words at the same time, a distinct row
    unnamed = build_owner_message_ref(chat_id=1, client_message_id="", ts=row["ts"], text=row["text"])
    assert source(unnamed) is None, "the saved fields cannot tell the two rows apart"


@pytest.mark.parametrize("reader", ["origins", "refs"])
def test_pre_identity_openings_of_one_room_share_one_walk_of_it(tmp_path, monkeypatch, reader):
    """Two old bindings that name their openings only by Main and a checksum: one read walks Main's rows once for
    both (no index holds a checksum), so the second costs no second walk; each still answers with its own one
    original, and no other room's history is read."""
    from ouroboros.project_dialogue import project_origin_rows, source_refs_for_project
    from ouroboros.projects_registry import create_project

    work = []
    for count in (1, 2):
        root = tmp_path / str(count)
        project = create_project(root, "selected", origin="reconcile")
        bindings, texts = {}, []
        for index in range(2):
            row = owner(client_id="", ts=f"2026-01-0{index + 1}T00:00:00+00:00", text=f"Opening words {index}")
            ds.append(root, "chat", row)  # both originals stay in Main whichever bindings name them
            if index < count:
                texts.append(row["text"])
                bindings[f"legacy-{index}"] = {"project_id": project["id"], "project_chat_id": project["chat_id"],
                                               "source_text": row["text"], "source_ref": build_owner_message_ref(
                                                   chat_id=1, client_message_id="", ts="", text=row["text"])}
        for room in range(3):
            ds.append(root, "chat", owner(f"foreign-{room}", chat_id=room + 10))
        (root / "state/project_task_bindings.json").write_text(json.dumps({"bindings": bindings}), encoding="utf-8")
        read = {"origins": lambda: [row["text"] for row in project_origin_rows(root, project["chat_id"])],
                "refs": lambda: [ref["text_sha256"] for ref in source_refs_for_project(root, project["chat_id"])]}[reader]
        with metered(monkeypatch) as measured:
            result = read()
        assert len(result) == count and (reader == "refs" or result == texts), result
        work.append((measured["transactions"], measured["statements"]))
    assert work[0] == work[1] and work[0][0] > 0
