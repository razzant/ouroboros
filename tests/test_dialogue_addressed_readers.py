"""Addressed readers use SQL truth, preserve owner bytes, and disclose incomplete import."""
from contextlib import contextmanager
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
    counts = {"statements": 0, "steps": 0}
    transaction = ds.transaction

    @contextmanager
    def traced(*args, **kwargs):
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


@pytest.mark.parametrize("reader", ["owner", "routing", "words", "notices", "publication", "narrative"])
def test_absence_readers_answer_over_the_readable_rows_of_a_root_with_a_torn_line(tmp_path, reader):
    """A line the import could not read is an unreadable old row, as it was for the chain readers:
    absence-only answers are given over the readable rows, so one torn line in the archive never
    makes a reader refuse forever (a finished task's Done card, an owed notice, a routing click).
    Readers with a coverage field disclose the interval (``range_coverage``)."""
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


def test_a_torn_line_leaves_the_routing_card_open_until_its_source_is_readable(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ouroboros import project_dialogue as pd, routing_wait
    from supervisor import workers

    seed(tmp_path, live={"chat": ["{broken\n"]})
    ds.ensure(tmp_path)
    monkeypatch.setattr(pd, "_COMPACT_AT_BYTES", 1)
    dispatched = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put_nowait=dispatched.append))
    monkeypatch.setattr(routing_wait, "wait_for_routing_annotation", lambda *args: {"status": "delivered"})
    append_chat_annotation(tmp_path, "owner-one", action="route_decision", status="needs_manual_target",
                           routing_token="route-one", options=[{"action": "steer_task", "task_id": "target"}])
    before = latest_chat_annotations(tmp_path)
    assert before["owner-one"]["status"] == "needs_manual_target"
    request = {"request_id": "click-one", "decision_id": "routing:owner-one:route-one", "option_index": 0}
    status, payload = handle_routing_decision(tmp_path, **request)
    # Over the readable rows the origin is not there: the card says so and stays open, never a 503 forever.
    assert status == 409 and payload["error"] == "origin_text_unavailable" and payload["state"] == "open"
    assert latest_chat_annotations(tmp_path) == before and dispatched == []
    ds.append(tmp_path, "chat", owner())
    status, payload = handle_routing_decision(tmp_path, **request)
    assert status == 200 and payload["dispatched"] == "delivered"
    # The owner's words travel as his verified quote; the steer note is only his (absent) comment.
    assert [event["quotes"][0]["text"] for event in dispatched] == [owner()["text"]]
    assert latest_chat_annotations(tmp_path)["owner-one"]["status"] == "delivered"


@pytest.mark.parametrize("reader", ["owner", "routing", "words", "notices"])
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
    from ouroboros import project_dialogue as pd

    seed(tmp_path, archives={"chat_20261016T000000.jsonl": [owner()]})
    monkeypatch.setattr(pd, "_COMPACT_AT_BYTES", 1)
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
    from ouroboros import project_dialogue as pd

    seed(tmp_path, live={"chat": ["{broken\n"]})
    ds.ensure(tmp_path)
    monkeypatch.setattr(pd, "_COMPACT_AT_BYTES", 1)
    for status in ("queued", "delivering", "delivered"):
        append_chat_annotation(tmp_path, "unreadable-source", action="steer_task", status=status)
    assert ds.range_coverage(tmp_path, "chat")["gaps"]
    assert latest_chat_annotations(tmp_path)["unreadable-source"]["status"] == "delivered"
    sidecar = tmp_path / "logs" / "chat_annotations.jsonl"
    assert len(sidecar.read_text(encoding="utf-8").splitlines()) == 1, "compacted to the newest act"


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


@pytest.mark.parametrize("reader", ["identity", "room", "evidence", "task"])
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
            "task": lambda: Memory(root).read_task_recent("chat.jsonl", "selected", 5)[0],
        }
        with metered(monkeypatch) as measured:
            result = reads[reader]()
        assert result
        if reader == "evidence":
            assert len(result["rows"]) == 1 and result["rows"][0]["text"] == row["text"]
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


@pytest.mark.parametrize("reader", ["reentry", "page", "receipt", "wake"])
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

    plugin = _load_plugin()
    measurements = []
    for rooms in (1, 30):
        root = tmp_path / str(rooms)
        for room in range(rooms):
            ds.append(root, stream, {"chat_id": room + 10, "text": "older foreign"})
        for index in range(4):
            ds.append(root, stream, {"chat_id": 1, "text": f"latest-{index}"})
        with metered(monkeypatch) as measured:
            rows, omitted = plugin._jsonl_tail(root / f"logs/{stream}.jsonl", max_entries=3, tail_bytes=262144)
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
