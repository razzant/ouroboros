"""Exact quoted-row navigation through the production store and history endpoint.

An address, never the quote's words/task/chat label, supplies identity. These cases
deliberately put its home outside the quoting task's room and duplicate its origin in
another room. A successful page must bind the SAME stored history id after conversion.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.chat_chain import format_address, row_address
from ouroboros.gateway.history_contracts import ChatHistoryResponse
from ouroboros.gateway.schema import validate_ingress
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.store import StoreError
from tests import _dialogue as dialogue_test
from tests.test_history_around import _read, _say, _ts


def _save(root, *, stream="chat", **fields):
    row = {"chat_id": 1, "ts": _ts(40), "direction": "in", "text": "Same words", **fields}
    ds.append(root, stream, row)
    return row, format_address(row_address(row))


def _found(root, address, chat_id=1):
    status, page = _read(root, chat_id=chat_id, around=address, n_human=8, n_progress=8)
    assert status == 200, page
    assert not validate_ingress(page, ChatHistoryResponse), validate_ingress(page, ChatHistoryResponse)
    assert page["anchor"]["state"] == "found", page
    assert page["anchor"]["row_address"] == address
    assert page["anchor"]["chat_id"] == chat_id
    row = next(row for row in page["messages"] if row.get("row_address") == address)
    assert row["history_id"] == page["anchor"]["history_id"]
    return page


def _origin(row):
    return build_owner_message_ref(chat_id=row["chat_id"], client_message_id=row.get("client_message_id", ""),
                                   ts=row["ts"], text=row["text"])


def test_old_owner_row_without_task_uses_exact_identity_and_continues_older(tmp_path):
    _say(tmp_path, 0, 40)
    _, address = _save(tmp_path, client_message_id="source")
    _, other = _save(tmp_path, client_message_id="same-words-same-time")
    _say(tmp_path, 50, 400)
    assert address != other
    assert address not in {row.get("row_address") for row in _read(tmp_path)[1]["messages"]}
    page = _found(tmp_path, address)
    assert len(page["messages"]) == 8
    assert "after_cursor" not in page and page["has_more"]
    assert page["anchor"]["history_id"] != _found(tmp_path, other)["anchor"]["history_id"]
    assert _read(tmp_path, cursor=page["page_cursor"])[1]["messages"] == page["messages"]
    older = _read(tmp_path, cursor=page["next_cursor"])[1]
    assert max(row["history_position"]["offset"] for row in older["messages"]) < min(
        row["history_position"]["offset"] for row in page["messages"])


def test_home_is_written_room_even_when_quoting_task_uses_it_as_another_origin(tmp_path):
    a, b = [create_project(tmp_path, name) for name in ("triage", "s1")]
    row, address = _save(tmp_path, chat_id=a["chat_id"], client_message_id="owner-a")
    bind_task_to_project(tmp_path, "s1-task", b["id"], origin={"ref": _origin(row), "text": row["text"]})
    status, page = _read(tmp_path, around=address)
    assert status == 200 and page["anchor"] == {"state": "moved", "row_address": address, "chat_id": a["chat_id"]}
    assert page["messages"] == [] and page["page_cursor"] is None
    a_page, b_page = _found(tmp_path, address, a["chat_id"]), _found(tmp_path, address, b["chat_id"])
    assert a_page["anchor"]["history_id"] == b_page["anchor"]["history_id"]


def test_conversion_after_quote_moves_exact_history_id_not_earliest_task_row(tmp_path):
    _save(tmp_path, ts=_ts(39), task_id="converted", direction="out", text="First task row")
    _, address = _save(tmp_path, task_id="converted", direction="out", text="The referenced row")
    before = _found(tmp_path, address)["anchor"]["history_id"]
    x = create_project(tmp_path, "converted-room")
    bind_task_to_project(tmp_path, "converted", x["id"], origin={"absent": "system"})
    assert _read(tmp_path, around=address)[1]["anchor"]["chat_id"] == x["chat_id"]
    after = _found(tmp_path, address, x["chat_id"])
    assert after["anchor"]["history_id"] == before
    task = _read(tmp_path, chat_id=x["chat_id"], around="task:converted")[1]
    assert task["anchor"]["history_id"] != before, "task references deliberately keep task semantics"


@pytest.mark.parametrize("chat_state", ["missing", "mismatch", "importing", "unreadable"])
def test_available_progress_survives_every_chat_lookup_miss(tmp_path, chat_state, monkeypatch):
    if chat_state == "importing":
        from ouroboros import dialogue_import
        dialogue_test.seed(tmp_path, archives={"chat_20261013T110000.jsonl": [{
            "chat_id": 1, "ts": _ts(1), "direction": "in", "text": "archive"}]})
        dialogue_import.cutover(tmp_path)
    row, address = _save(tmp_path, stream="progress", direction="out", task_id="narrator", narration=True)
    if chat_state == "mismatch":
        _save(tmp_path, text="A different chat row at that same timestamp")
    if chat_state == "unreadable":
        original = ds.rows_by_identity
        def rows(root, chat_id, ts, *, stream="chat"):
            if stream == "chat":
                raise StoreError("injected chat read failure")
            return original(root, chat_id, ts, stream=stream)
        monkeypatch.setattr(ds, "rows_by_identity", rows)
    page = _found(tmp_path, address)
    assert page["anchor"]["history_id"].startswith("progress:")
    assert "chat_around_unpositioned" in page["window"]["truncated_by"]


def test_both_complete_missing_sources_are_absent_but_partial_import_retries(tmp_path):
    from ouroboros import dialogue_import
    row = {"chat_id": 1, "ts": _ts(40), "direction": "in", "text": "not imported yet"}
    address = format_address(row_address(row))
    dialogue_test.seed(tmp_path, archives={"chat_20261013T110000.jsonl": [row]})
    dialogue_import.cutover(tmp_path)
    status, page = _read(tmp_path, around=address)
    assert status == 503 and page["reason_code"] == "migrating"
    dialogue_test.import_all(tmp_path)
    _found(tmp_path, address)
    missing = address[:-12] + "000000000000"
    assert _read(tmp_path, around=missing)[1]["anchor"]["state"] == "absent"


def test_unknown_progress_is_not_hidden_by_a_complete_chat_miss(tmp_path, monkeypatch):
    _, address = _save(tmp_path)
    missing = address[:-12] + "000000000000"
    original = ds.rows_by_identity
    def rows(root, chat_id, ts, *, stream="chat"):
        if stream == "progress":
            raise StoreError("injected progress read failure")
        return original(root, chat_id, ts, stream=stream)
    monkeypatch.setattr(ds, "rows_by_identity", rows)
    status, page = _read(tmp_path, around=missing)
    assert status == 503 and page["reason_code"] == "history_source_unavailable"
    _found(tmp_path, address)  # symmetric: an available exact chat match survives this failure


@pytest.mark.parametrize("lifecycle", ["deleting", "tombstoned"])
def test_inactive_home_does_not_fall_back_to_another_origin_room(tmp_path, lifecycle):
    from ouroboros.projects_registry import begin_project_deletion, complete_project_deletion
    a, b = [create_project(tmp_path, name) for name in ("home", "other-origin")]
    row, address = _save(tmp_path, chat_id=a["chat_id"], client_message_id="original")
    bind_task_to_project(tmp_path, "other", b["id"], origin={"ref": _origin(row), "text": row["text"]})
    begin_project_deletion(tmp_path, a["id"])
    if lifecycle == "tombstoned":
        complete_project_deletion(tmp_path, a["id"])
    for chat in (1, a["chat_id"]):
        status, page = _read(tmp_path, chat_id=chat, around=address)
        assert status == 200 and page["anchor"]["state"] == "unavailable"
        assert page["messages"] == []


@pytest.mark.parametrize("chat", [0, -100])
def test_existing_hidden_rows_are_unavailable_not_absent(tmp_path, chat):
    _, address = _save(tmp_path, chat_id=chat)
    status, page = _read(tmp_path, around=address)
    assert status == 200 and page["anchor"]["state"] == "unavailable"


def test_retained_context_cannot_replace_missing_original(tmp_path):
    project = create_project(tmp_path, "retained")
    row = {"chat_id": 1, "ts": _ts(40), "direction": "in", "text": "Only a retained copy", "client_message_id": "retained"}
    bind_task_to_project(tmp_path, "retained-task", project["id"], origin={"ref": _origin(row), "text": row["text"]})
    ds.append(tmp_path, "chat", {"chat_id": 1, "ts": _ts(41), "direction": "in", "text": "other"})
    address = format_address(row_address(row))
    assert _read(tmp_path, chat_id=project["chat_id"], around=address)[1]["anchor"]["state"] == "absent"


@pytest.mark.parametrize("kind", ["project_completion_summary", "steer_not_delivered"])
def test_pinned_main_and_origin_notices_keep_their_authoritative_home(tmp_path, kind):
    project = create_project(tmp_path, "bound")
    bind_task_to_project(tmp_path, "bound-task", project["id"], origin={"absent": "system"})
    _, address = _save(tmp_path, task_id="bound-task", direction="system", type=kind)
    page = _found(tmp_path, address)
    assert page["anchor"]["chat_id"] == 1
    assert _read(tmp_path, chat_id=project["chat_id"], around=address)[1]["anchor"]["chat_id"] == 1


def test_a_project_works_main_notice_is_found_in_main_and_its_project_and_homes_there(tmp_path):
    """One row in two rooms (ARCHITECTURE §3 "Room contract"): found where asked in either; from an
    unrelated room its home is its work's Project, never ambiguous."""
    project, other = create_project(tmp_path, "bound"), create_project(tmp_path, "other")
    bind_task_to_project(tmp_path, "bound-task", project["id"], origin={"absent": "system"})
    _, address = _save(tmp_path, task_id="bound-task", direction="out", type="main_notice")
    assert _found(tmp_path, address)["anchor"]["chat_id"] == 1
    found = _read(tmp_path, chat_id=project["chat_id"], around=address)[1]["anchor"]
    assert found["state"] == "found" and found["chat_id"] == project["chat_id"]
    moved = _read(tmp_path, chat_id=other["chat_id"], around=address)[1]["anchor"]
    assert moved["state"] == "moved" and moved["chat_id"] == project["chat_id"]


def test_hash_prefix_ambiguity_never_selects_a_convenient_match(tmp_path, monkeypatch):
    from ouroboros import chat_chain
    first, _ = _save(tmp_path, client_message_id="a")
    _save(tmp_path, client_message_id="b")
    digest = chat_chain.source_row_id
    # Force an otherwise infeasible 48-bit collision at the real hash seam, not a lookup response.
    monkeypatch.setattr(chat_chain, "source_row_id", lambda row: "a" * 12 + digest(row)[12:])
    status, page = _read(tmp_path, around=f"row:1@{first['ts']}#{'a' * 12}")
    assert status == 200 and page["anchor"]["state"] == "ambiguous" and page["messages"] == []


def test_bad_row_addresses_are_refused_and_cursor_is_mutually_exclusive(tmp_path):
    _, address = _save(tmp_path)
    for invalid in ("row:1@now#bad", "row:x@now#000000000000", "row:"):
        assert _read(tmp_path, around=invalid)[0] == 400
    cursor = _read(tmp_path)[1]["page_cursor"]
    assert _read(tmp_path, around=address, cursor=cursor)[0] == 400


def test_strict_authority_decides_only_the_destination_never_each_page_row(tmp_path, monkeypatch):
    from ouroboros import projects_registry
    project = create_project(tmp_path, "worked")
    addresses = []
    for index in range(5):
        bind_task_to_project(tmp_path, f"work-{index}", project["id"], origin={"absent": "system"})
        addresses.append(_save(tmp_path, ts=_ts(41 + index), task_id=f"work-{index}", direction="out",
                               text=f"Work {index}")[1])
    address = addresses[2]
    strict_reads = []
    original = projects_registry.get_reserved_project
    def counted(root, project_id, *, strict=False):
        strict_reads.append(project_id if strict else None)
        return original(root, project_id, strict=strict)
    monkeypatch.setattr(projects_registry, "get_reserved_project", counted)
    page = _found(tmp_path, address, project["chat_id"])
    assert {"Work 0", "Work 4"} <= {row.get("text") for row in page["messages"]}
    assert [read for read in strict_reads if read] == [project["id"]], "only the quoted row's room is strict"


def test_unanswerable_lineage_of_other_page_rows_is_ordinary_unknown_never_a_bare_error(tmp_path, monkeypatch):
    from ouroboros import projects_registry
    for index in range(6):
        _save(tmp_path, ts=_ts(41 + index), task_id=f"loose-{index}", direction="out", text=f"Loose work {index}")
    _, address = _save(tmp_path, ts=_ts(47), client_message_id="owner-row")
    _, quoted_work = _save(tmp_path, ts=_ts(50), task_id="loose-quoted", direction="out", text="Quoted work")
    monkeypatch.setattr(projects_registry, "_read_lineage", lambda root, task_ids, **_kw: None)  # unreadable authority
    assert {"Loose work 4", "Loose work 5"} <= {row.get("text") for row in _found(tmp_path, address)["messages"]}
    # The quoted row's own room still needs readable authority: its task's lineage decides it.
    status, page = _read(tmp_path, around=quoted_work)
    assert status == 503 and page["reason_code"] == "history_source_unavailable", page


def _rewrite(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_a_malformed_unrelated_room_does_not_block_a_healthy_home(tmp_path):
    from ouroboros.projects_registry import _registry_path
    home, other = [create_project(tmp_path, name) for name in ("healthy-home", "broken-neighbour")]
    _, address = _save(tmp_path, chat_id=home["chat_id"], client_message_id="home-row")

    def break_routing(project_id):
        def change(data):
            next(row for row in data["projects"] if row["id"] == project_id)["routing_generation"] = -1
        _rewrite(_registry_path(tmp_path), change)

    break_routing(other["id"])  # its identity stays valid; only its own routing is malformed
    assert _read(tmp_path, around=address)[1]["anchor"] == {
        "state": "moved", "row_address": address, "chat_id": home["chat_id"]}
    _found(tmp_path, address, home["chat_id"])
    break_routing(home["id"])  # the home's own authority unreadable: unknown, never a guessed room
    status, page = _read(tmp_path, around=address)
    assert status == 503 and page["reason_code"] == "history_source_unavailable", page


def test_a_malformed_binding_row_neither_hides_origins_nor_blocks_exact_reads(tmp_path):
    from ouroboros.projects_registry import _bindings_path
    project = create_project(tmp_path, "origin-room")
    row, address = _save(tmp_path, client_message_id="origin-owner")
    bind_task_to_project(tmp_path, "origin-task", project["id"], origin={"ref": _origin(row), "text": row["text"]})
    _rewrite(_bindings_path(tmp_path), lambda data: data["bindings"].update({"torn-binding": "not a binding row"}))
    status, page = _read(tmp_path, chat_id=project["chat_id"])
    canonical = [message for message in page["messages"]
                 if message.get("row_address") == address and not message.get("origin_projected")]
    assert status == 200 and len(canonical) == 1, page["messages"]
    _found(tmp_path, address, project["chat_id"])


def test_a_same_time_different_row_beside_an_unreadable_gap_is_unknown_not_absent(tmp_path):
    said = {"chat_id": 1, "ts": _ts(40), "direction": "in", "text": "what was said"}
    stale = format_address(row_address({**said, "text": "what was never said"}))
    for root, lines in ((tmp_path / "torn", [said, '{"ts": "%s", "chat_id": 1, "text": "torn\n' % _ts(45)]),
                        (tmp_path / "intact", [said])):
        dialogue_test.seed(root, live={"chat": lines})
        dialogue_test.import_all(root)
    status, page = _read(tmp_path / "torn", around=stale)
    assert status == 503 and page["reason_code"] == "history_source_unavailable", page
    assert _read(tmp_path / "intact", around=stale)[1]["anchor"]["state"] == "absent"
