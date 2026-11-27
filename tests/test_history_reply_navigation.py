"""Exact quoted-row navigation through the production store and history endpoint.

An address, never the quote's words/task/chat label, supplies identity. These cases
deliberately put its home outside the quoting task's room and duplicate its origin in
another room. A successful page must bind the SAME stored history id after conversion.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from ouroboros import dialogue_store as ds
from ouroboros.chat_chain import format_address, row_address
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_contracts import ChatHistoryResponse
from ouroboros.gateway.schema import validate_ingress
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.store import StoreError
from tests import _dialogue as dialogue_test
from tests._dialogue import healthy_main_fixture  # noqa: F401 -- the "healthy_main" fixture
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


SCOPE = "scope-frozen-at-apply"  # a serving generation's frozen data-profile scope (#3294)


def _scoped_read(root, scope, **params):
    """The endpoint's read as a scoped document makes it: ``app.state.served_ui`` names the serving
    generation's frozen scope (#3294), whose marker the browser demands on every successful page."""
    query = {"chat_id": "1", **{key: str(value) for key, value in params.items()}}
    request = SimpleNamespace(query_params=query, app=SimpleNamespace(
        state=SimpleNamespace(served_ui=SimpleNamespace(serving_profile_scope=lambda: scope))))
    response = asyncio.run(make_chat_history_endpoint(root)(request))
    return response.status_code, json.loads(response.body)


def test_every_successful_answer_of_an_exact_read_carries_the_serving_scope(tmp_path):
    """The non-found exact answers ("moved": the row lives in another room; "absent": both sources
    complete) are successful pages of the same endpoint as the found page, so a scoped document
    consumes them only when each carries its serving generation's marker (#3294: the browser refuses
    a 200 whose marker differs or is absent); a scopeless generation invents none."""
    home, other = [create_project(tmp_path, name) for name in ("home", "other-origin")]
    row, address = _save(tmp_path, chat_id=home["chat_id"], client_message_id="original")
    bind_task_to_project(tmp_path, "other", other["id"], origin={"ref": _origin(row), "text": row["text"]})
    missing = address[:-12] + "000000000000"
    for around, state in ((address, "moved"), (missing, "absent")):
        status, page = _scoped_read(tmp_path, SCOPE, around=around)
        assert status == 200 and page["anchor"]["state"] == state, page
        assert page["profile_scope"] == SCOPE, page
    status, found = _scoped_read(tmp_path, SCOPE, chat_id=home["chat_id"], around=address)
    assert status == 200 and found["anchor"]["state"] == "found" and found["profile_scope"] == SCOPE
    for around in (address, missing):  # a rootless generation before apply: no marker is invented
        status, page = _scoped_read(tmp_path, "", around=around)
        assert status == 200 and "profile_scope" not in page, page
    assert "profile_scope" not in _read(tmp_path, around=address)[1]  # no served_ui: unchanged


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


@pytest.mark.parametrize("chat_state", ["missing", "mismatch", "unreadable"])
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


# DESIGN "Replies": an owner's answer to a card is shown on its question's card, never as a row of its own. The
# question row names the answer row (`quiz.answer_row_address`), and a page around that answer stands on the question.
QUESTION = {"quiz_id": "qz-1", "options": [{"label": "Yes"}, {"label": "No"}], "stake": "", "assumption": "",
            "state": "open"}


def _card(root, at, gap, *, block=True, comment=""):
    """A Main question, ``gap`` owner messages, the owner's recorded answer (and, with ``block``, its task's
    question block keeping the stored answer row's copy): (question receipt, answer receipt, answer address)."""
    asked = ds.append(root, "chat", {"ts": _ts(at), "direction": "out", "chat_id": 1, "task_id": "asker",
                                     "type": "quiz", "text": "Use blue and gold in the table legend?", "quiz": dict(QUESTION)})
    _say(root, at + 1, gap)
    answered = {**QUESTION, "state": "answered", "answered_index": 1, "answered_at": _ts(at + gap + 1),
                **({"comment": comment} if comment else {})}
    answer = ds.append(root, "chat", {"ts": _ts(at + gap + 1), "direction": "system", "chat_id": 1,
                                      "type": "quiz_answer", "source": "owner_quiz_answer", "task_id": "asker",
                                      "client_message_id": "quiz_answer:asker:qz-1",
                                      "text": "[Owner quiz answer] quiz qz-1 — No", "quiz": answered})
    if block:
        from ouroboros.task_results import write_task_result

        write_task_result(root, "asker", "running", owner_quiz={"qz-1": {**answered, "delivery": {
            "history_at": _ts(at + gap + 1), "answer_row": dict(answer.row)}}})
    return asked, answer, format_address(row_address(answer.row))


def _question(page):
    (card,) = [row for row in page["messages"] if row.get("msg_type") == "quiz"]
    return card


def test_answer_quote_can_reopen_after_its_question_imports(tmp_path):
    from ouroboros import dialogue_import

    question = {"ts": "2024-04-09T10:05:00+00:00", "chat_id": 1, "direction": "out", "task_id": "asker",
                "type": "quiz", "text": "Use blue and gold in the table legend?", "quiz": dict(QUESTION)}
    dialogue_test.seed(tmp_path, archives={"chat_20240409T110000.jsonl": [question]})
    dialogue_import.cutover(tmp_path)
    _, address = _save(tmp_path, ts="2024-04-09T10:14:00+00:00", task_id="asker", direction="system", type="quiz_answer",
                       source="owner_quiz_answer", quiz={**QUESTION, "state": "answered", "answered_index": 0})
    status, page = _read(tmp_path, around=address)
    assert status == 503 and page["reason_code"] == "migrating", page
    dialogue_test.import_all(tmp_path)
    status, page = _read(tmp_path, around=address)
    assert status == 200 and page["anchor"]["state"] == "found", page
    assert _question(page)["quiz"]["answer_row_address"] == address


@pytest.mark.parametrize("gap_first", [True, False], ids=["before_answer", "after_answer"])
@pytest.mark.parametrize("imported", [False, True], ids=["pending", "unreadable_line"])
def test_missing_question_only_depends_on_gaps_before_its_answer(tmp_path, gap_first, imported, healthy_main):
    from ouroboros import dialogue_import

    answer = {"ts": "2024-04-09T10:14:00+00:00", "chat_id": 1, "task_id": "asker", "direction": "system", "type": "quiz_answer",
              "source": "owner_quiz_answer", "text": "Choice", "quiz": {**QUESTION, "state": "answered", "answered_index": 1}}
    earlier, later = "chat_20240409T110000.jsonl", "chat_20240409T120000.jsonl"
    gap_segment, answer_segment = (earlier, later) if gap_first else (later, earlier)
    dialogue_test.seed(tmp_path, archives={gap_segment: ["{torn line\n"], answer_segment: [answer]})
    dialogue_import.cutover(tmp_path)
    dialogue_import.import_segment(tmp_path, "chat", answer_segment)
    if imported:
        dialogue_import.import_segment(tmp_path, "chat", gap_segment)
    address = format_address(row_address(answer))
    status, page = _read(tmp_path, around=address)
    if gap_first:
        assert status == 503 and page["reason_code"] == ("history_source_unavailable" if imported else "migrating"), page
    else:
        assert status == 200 and page["anchor"] == {"row_address": address, "state": "unavailable"}, page


def test_readable_question_needs_no_chat_coverage_check(tmp_path, monkeypatch):
    from ouroboros import dialogue_positions
    from ouroboros.chat_references import quiz_question_row

    asked, answer, _ = _card(tmp_path, 0, 1)

    def unexpected(*args, **kwargs):
        pytest.fail("A matching question already proves the destination")

    monkeypatch.setattr(ds, "require_imported", unexpected)
    monkeypatch.setattr(dialogue_positions, "skipped_between", unexpected)
    assert quiz_question_row(tmp_path, answer.row, answer_seq=answer.seq).seq == asked.seq


@pytest.mark.parametrize("block", [True, False], ids=["question_block", "window_only"])
def test_an_answered_question_names_its_answer_row_the_card_shows(tmp_path, block):
    asked, answer, address = _card(tmp_path, 0, 2, block=block, comment="and soon")
    status, page = _read(tmp_path)
    assert status == 200 and not validate_ingress(page, ChatHistoryResponse), validate_ingress(page, ChatHistoryResponse)
    card = _question(page)
    assert card["history_id"] == f"chat:{asked.seq}" and card["quiz"]["state"] == "answered"
    assert card["quiz"]["answer_row_address"] == address, card["quiz"]


def test_an_open_or_closed_question_names_no_answer_row(tmp_path):
    ds.append(tmp_path, "chat", {"ts": _ts(0), "direction": "out", "chat_id": 1, "task_id": "asker", "type": "quiz",
                                 "text": "Use blue and gold in the table legend?", "quiz": dict(QUESTION)})
    assert "answer_row_address" not in _question(_read(tmp_path)[1])["quiz"]
    closed = {**QUESTION, "state": "closed", "closed_reason": "No longer needed", "closed_at": _ts(1)}
    ds.append(tmp_path, "chat", {"ts": _ts(1), "direction": "system", "chat_id": 1, "type": "quiz_closed",
                                 "task_id": "asker", "client_message_id": "quiz_closed:asker:qz-1",
                                 "text": "No longer needed", "quiz": closed})
    card = _question(_read(tmp_path)[1])
    assert card["quiz"]["state"] == "closed" and "answer_row_address" not in card["quiz"]


def test_an_answer_far_from_its_question_pages_at_the_question_and_keeps_its_own_identity(tmp_path):
    _say(tmp_path, 0, 5)
    asked, answer, address = _card(tmp_path, 5, 60)
    _say(tmp_path, 100, 30)
    status, page = _read(tmp_path, around=address, n_human=8, n_progress=8)
    assert status == 200 and not validate_ingress(page, ChatHistoryResponse), validate_ingress(page, ChatHistoryResponse)
    # The anchor is the answer's own, never the question's: a client lands on the card part naming it.
    assert page["anchor"] == {"state": "found", "row_address": address, "chat_id": 1, "history_id": f"chat:{answer.seq}"}
    card = _question(page)
    assert card["history_id"] == f"chat:{asked.seq}" != page["anchor"]["history_id"]
    assert card["quiz"]["state"] == "answered" and card["quiz"]["answer_row_address"] == address
    assert address not in {row.get("row_address") for row in page["messages"]}, "no row of its own on the page"
    # The page is an ordinary around page: it re-reads and continues older from the question.
    assert _read(tmp_path, cursor=page["page_cursor"])[1]["messages"] == page["messages"] and page["has_more"]


def _late(root, at, quiz_id, text, *, snapshot=True, chat_id=1):
    """The owner's late answer to ``asker``'s ``quiz_id`` as its row records it (``late_answer``); a row saved before
    answer snapshots were kept names only the asker and the quiz."""
    late = {"task_id": "asker", "quiz_id": quiz_id}
    if snapshot:
        late["answer_snapshot"] = {"task_id": "asker", "quiz": {**QUESTION, "quiz_id": quiz_id, "state": "answered",
                                                                 "answered_after_terminal": True}}
    return _save(root, chat_id=chat_id, ts=_ts(at), client_message_id=f"quiz_late_answer:asker:{quiz_id}", text=text,
                 late_answer=late)


def _recorded(root, at, quiz_id, words, *, decision="quiz_answer", chat_id=1):
    """The decision door's record of the owner's answer to (or closure of) ``asker``'s live question: a System row in
    the question's room, named ``<decision>:asker:<quiz>``, holding the card with the delivery it still owed when
    written (``task_decision._record_quiz_answer_history``, ``events_chat_delivery._handle_send_quiz_state``)."""
    settled = ({"state": "answered", "comment": words, "answered_at": _ts(at)} if decision == "quiz_answer" else
               {"state": "closed", "closed_reason": words, "closed_at": _ts(at)})
    settled["delivery"] = {"kind": "answer" if decision == "quiz_answer" else "closure", "state": "owed"}
    return _save(root, chat_id=chat_id, ts=_ts(at), direction="system", type=decision, task_id="asker",
                 source="owner_quiz_answer" if decision == "quiz_answer" else "owner_question_close",
                 client_message_id=f"{decision}:asker:{quiz_id}", text=f"[{decision}] quiz {quiz_id}",
                 quiz={**QUESTION, "quiz_id": quiz_id, **settled})


LATE_WORDS = "HTML, with the totals on top"


def _asked(root, quiz_id, *, chat_id=1, named=True, files=None, at=0):
    """``asker``'s published question card, named and holding the publication it still owed as ``message_bus.send_quiz``
    stores it (before ``mark_published``), or as cards were before they were named."""
    owed = {"question_delivery": {"kind": "publication", "state": "owed"}} if named else {}
    asked = ds.append(root, "chat", {"ts": _ts(at), "direction": "out", "chat_id": chat_id, "task_id": "asker", "type": "quiz",
                                     "text": "**Which format for the weekly report?**\n\nThe [draft](https://example.invalid) "
                                             "compares three.", "quiz": {**QUESTION, "quiz_id": quiz_id, **owed},
                                     **({"attachments": files} if files else {}),
                                     **({"client_message_id": f"quiz_publication:asker:{quiz_id}"} if named else {})})
    return format_address(row_address(asked.row))


@pytest.mark.parametrize("opening", ["late_answer", "recorded_answer", "retained_copy"])
def test_a_project_opened_by_an_answer_quotes_its_question_where_its_press_lands(tmp_path, opening):
    """Owner choice A, 23.11: the answer that began a Project's work stands first in it under its question's quote,
    marked as the host's (``derived_question``), the preview the question's first line and nothing of its files,
    whatever recorded the answer: a late answer's own row; the decision door's record of an answer to a live
    question, the row a Project started by it opens with (on Main's card it is no row of its own); or, that row
    gone, the binding's retained card fact. The quote names Main, exactly where the strict press lands."""
    from ouroboros.owner_relay import owner_row_source

    question = _asked(tmp_path, "qz-late", files=[{"name": "layouts.png", "kind": "image", "unavailable": "too_large"}])
    room = create_project(tmp_path, "weekly-report", name="Weekly report")
    if opening == "retained_copy":
        ref = build_owner_message_ref(chat_id=1, client_message_id="quiz_answer:asker:qz-late", ts=_ts(5), text=LATE_WORDS)
        typed = {"source": "owner_quiz_answer", "source_task_id": "asker", "quiz_id": "qz-late", "content": LATE_WORDS}
        origin = {"ref": ref, "text": LATE_WORDS, "snapshot": {"projectable": True, "input": typed}}
    else:
        row, _ = (_late if opening == "late_answer" else _recorded)(tmp_path, 5, "qz-late", LATE_WORDS)
        ref, words = owner_row_source(row)
        origin = {"ref": ref, "text": words}
    bind_task_to_project(tmp_path, "worker", room["id"], origin=origin)
    status, page = _read(tmp_path, chat_id=room["chat_id"])
    assert status == 200 and not validate_ingress(page, ChatHistoryResponse), validate_ingress(page, ChatHistoryResponse)
    [opened] = [message for message in page["messages"] if message.get("role") == "user"]
    assert opened["text"] == LATE_WORDS and bool(opened.get("origin_projected")) is (opening != "late_answer")
    quote = opened["reply_to"]
    assert {key: quote.get(key) for key in ("address", "chat_id", "role", "author_label", "kind", "snippet", "task_id",
                                            "room_label", "room_label_kind", "derived_question")} == {
        "address": question, "chat_id": 1, "role": "assistant", "author_label": "Ouroboros", "kind": "question",
        "snippet": "Which format for the weekly report?", "task_id": "asker", "room_label": "Main",
        "room_label_kind": "main", "derived_question": True}
    assert not {"attachment_failures", "chars"} & set(quote), quote
    status, there = _read(tmp_path, chat_id=room["chat_id"], around=question)
    assert status == 200 and there["anchor"] == {"state": "moved", "row_address": question, "chat_id": 1}


def test_a_quote_is_placed_by_the_authority_its_press_reads_never_by_a_softer_page_read(tmp_path, monkeypatch):
    """Review 1: a page read keeps unanswerable lineage an ordinary unknown, and its softer rule puts the asker's
    question in Main; the strict press cannot place it now. The quote follows the press: no quote, the answer itself
    still readable, never a Main the jump does not know."""
    from ouroboros import projects_registry

    question = _asked(tmp_path, "qz-late")
    _, address = _late(tmp_path, 5, "qz-late", LATE_WORDS)
    monkeypatch.setattr(projects_registry, "_read_lineage", lambda root, task_ids, **_kw: None)  # unreadable now
    status, jump = _read(tmp_path, around=question)
    assert status == 503 and jump["reason_code"] == "history_source_unavailable", jump
    status, page = _read(tmp_path)
    shown = next(message for message in page["messages"] if message.get("row_address") == address)
    assert status == 200 and shown["text"] == LATE_WORDS and not (shown.get("reply_to") or {}).get("address"), shown


def test_a_pages_quotes_cost_its_own_scope_however_many_questions_they_name(tmp_path, monkeypatch):
    """Review 1: a page's quotes cost one read of its scope, never one per question: the strict room authority (the
    registry parses included, lent to every home check) and the cards of an asker published before cards were named
    (one read of that asker's rows) cost a page answering three of its questions what they cost a page answering one."""
    from ouroboros import project_admission, projects_registry

    parses, probes = [], []
    parse, probe = project_admission._strict_admission_snapshot, ds.iter_probes

    def counted_parse(root, **kw):
        parses.append(root)
        return parse(root, **kw)

    def counted_probe(root, stream, keys, **kw):
        probes.append(tuple(keys))
        return probe(root, stream, probes[-1], **kw)

    for module in (project_admission, projects_registry):  # the registry facade holds its own name for it
        monkeypatch.setattr(module, "_strict_admission_snapshot", counted_parse)
    monkeypatch.setattr(ds, "iter_probes", counted_probe)
    costs = {}
    for count in (1, 3):
        root = tmp_path / f"{count}-questions"
        room = create_project(root, "planning", name="Planning")
        quizzes = [f"qz-{index}" for index in range(count)]
        questions = {quiz: _asked(root, quiz, chat_id=room["chat_id"], named=False, at=index)
                     for index, quiz in enumerate(quizzes)}
        answers = {quiz: _late(root, 10 + index, quiz, f"Answer {quiz}", chat_id=room["chat_id"])[1]
                   for index, quiz in enumerate(quizzes)}
        parses.clear()
        probes.clear()
        shown = {message.get("row_address"): message for message in _read(root, chat_id=room["chat_id"])[1]["messages"]}
        assert {quiz: shown[answers[quiz]]["reply_to"]["address"] for quiz in quizzes} == questions
        assert not any("room_label" in shown[address]["reply_to"] for address in answers.values())  # found here
        costs[count] = len(parses), probes.count((("task_id", "asker"),))
    assert costs[3] == costs[1] and costs[1][1] == 1, costs


def test_only_the_owners_rooms_that_show_the_answer_quote_its_question(tmp_path):
    """Showing is not placing: the room must be the owner's and show the answer itself. The Project the answer opened
    quotes it, as Main does; a Project with no link to it, a room that is not the owner's (a place, the hidden
    partition) quote nothing. A live frame other rooms share has one preview for all of them, so no room's quote."""
    from ouroboros.chat_references import owner_frame_fields, question_quote
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID

    _asked(tmp_path, "qz-late")
    row, _ = _late(tmp_path, 5, "qz-late", LATE_WORDS)
    opened, unlinked = (create_project(tmp_path, slug, name=name) for slug, name in
                        (("weekly-report", "Weekly report"), ("garden", "Garden")))
    bind_task_to_project(tmp_path, "worker", opened["id"], origin={"ref": _origin(row), "text": row["text"]})
    in_main = question_quote(tmp_path, 1, row)
    assert in_main["derived_question"] is True and in_main["address"] and "room_label" not in in_main
    assert question_quote(tmp_path, opened["chat_id"], row)["room_label"] == "Main"
    for room in (unlinked["chat_id"], HIDDEN_CHAT_ID, 700_001):  # 700001: a place's room, neither Main nor a Project
        assert question_quote(tmp_path, room, row) == {}, room
    assert owner_frame_fields(tmp_path, 1, row)["reply_to"] == in_main  # a frame its own room alone shows
    shared = owner_frame_fields(tmp_path, 1, {**row, "room_revisions": {"1": 2, str(opened["chat_id"]): 1}})
    assert shared["also_rooms"] == [opened["chat_id"]] and "reply_to" not in shared


@pytest.mark.parametrize("decision", ["quiz_answer", "quiz_closed"])
def test_a_landed_question_shows_its_saved_decision_after_its_block_and_answer_left(tmp_path, decision):
    """The press lands on the original card in its actual state: with its asker's block gone and its answer far
    beyond the landing page, the card settles by the record the decision door named for it, never reads as open;
    the record is evidence for the card, never a row of the page. Neither the card's own publication nor the record
    says what is owed now: with no block, the card names no delivery."""
    question = _asked(tmp_path, "qz-late")
    _say(tmp_path, 1, 60)
    _, record = _recorded(tmp_path, 70, "qz-late", LATE_WORDS, decision=decision)
    status, page = _read(tmp_path, around=question, n_human=8, n_progress=8)
    assert status == 200 and not validate_ingress(page, ChatHistoryResponse), validate_ingress(page, ChatHistoryResponse)
    quiz = _question(page)["quiz"]
    assert not {"question_delivery", "delivery"} & set(quiz), quiz
    assert record not in {message.get("row_address") for message in page["messages"]}
    if decision == "quiz_answer":
        assert (quiz["state"], quiz.get("comment"), quiz.get("answer_row_address")) == ("answered", LATE_WORDS, record)
    else:
        assert (quiz["state"], quiz.get("closed_reason")) == ("closed", LATE_WORDS) and "answer_row_address" not in quiz


@pytest.mark.parametrize("project_id, name, label, kind", [
    ("asking-room", "Asking room", "Asking room", None),
    ("proj_5e1f", "", "Project", "project"),
    ("user-main", "Main", "Main", None),
    ("user-project", "Project", "Project", None),
], ids=["named", "never_named", "named_Main", "named_Project"])
def test_a_late_answer_names_the_questions_room_until_it_is_unavailable(tmp_path, project_id, name, label, kind):
    """The quote names the Project holding the question by its name (one never named reads ``Project``, never an id,
    with ``room_label_kind="project"`` so clients translate only host chrome and never a user Project named ``Main``
    or ``Project``); a source no available room shows is not exposed in another room: the quote leaves with its room."""
    from ouroboros.projects_registry import begin_project_deletion

    source, other = create_project(tmp_path, project_id, name=name), create_project(tmp_path, "later", name="Later")
    _asked(tmp_path, "qz-late", chat_id=source["chat_id"])
    row, address = _late(tmp_path, 5, "qz-late", "yes", chat_id=source["chat_id"])
    bind_task_to_project(tmp_path, "worker", other["id"], origin={"ref": _origin(row), "text": row["text"]})

    def opening():
        status, page = _read(tmp_path, chat_id=other["chat_id"])
        assert status == 200 and not validate_ingress(page, ChatHistoryResponse), validate_ingress(page, ChatHistoryResponse)
        return next(m for m in page["messages"] if m.get("row_address") == address)

    reply = opening()["reply_to"]
    assert reply["room_label"] == label
    assert reply.get("room_label_kind") == kind
    begin_project_deletion(tmp_path, source["id"])
    assert not (opening().get("reply_to") or {}).get("address")


def test_a_rows_own_reply_is_never_replaced_by_the_late_answers_question(tmp_path):
    """A row that already names what it replies to keeps that Reply; the derived quote only fills a row without one."""
    from ouroboros.chat_references import reference_from_row

    _asked(tmp_path, "qz-late")
    other, _ = _save(tmp_path, ts=_ts(2), client_message_id="other", text="Another message")
    own = reference_from_row(other)
    _, address = _late(tmp_path, 5, "qz-late", "yes")
    replying = ds.append(tmp_path, "chat", {"chat_id": 1, "ts": _ts(6), "direction": "in", "text": "yes, that one",
                                            "client_message_id": "quiz_late_answer:asker:qz-late-2", "reply_to": own,
                                            "late_answer": {"task_id": "asker", "quiz_id": "qz-late"}}).row
    shown = {m.get("row_address"): m for m in _read(tmp_path)[1]["messages"]}
    assert shown[format_address(row_address(replying))]["reply_to"]["address"] == own["address"]
    assert shown[address]["reply_to"]["kind"] == "question"


def test_a_row_without_a_readable_recorded_question_quotes_nothing(tmp_path):
    """Nothing is guessed: an owner message that records no question, and a late answer whose question cannot be
    read, carry no quote."""
    _, plain = _save(tmp_path, ts=_ts(1), client_message_id="plain", text=LATE_WORDS)
    _, orphan = _late(tmp_path, 2, "qz-gone", "yes")
    shown = {m.get("row_address"): m for m in _read(tmp_path)[1]["messages"]}
    assert not (shown[plain].get("reply_to") or {}).get("address")
    assert not (shown[orphan].get("reply_to") or {}).get("address")


def test_an_ordinary_exact_row_still_pages_at_itself(tmp_path):
    _say(tmp_path, 0, 5)
    _card(tmp_path, 5, 60)
    _say(tmp_path, 100, 30)
    row, address = _save(tmp_path, ts=_ts(140), client_message_id="plain-owner", text="Not an answer")
    _say(tmp_path, 141, 30)
    page = _found(tmp_path, address)
    assert [m for m in page["messages"] if m.get("msg_type") == "quiz"] == []


def _receipt_before_forwarding(root, at, quiz_id, *, chat_id=1):
    """A decision receipt as the door writes it before forwarding (``_record_quiz_answer_history`` stores
    ``quiz=dict(block)``): the settled card with its delivery snapshot still owed."""
    settled = {**QUESTION, "quiz_id": quiz_id, "state": "answered", "answered_index": 1, "answered_at": _ts(at),
               "delivery": {"kind": "answer", "state": "owed"}}
    return ds.append(root, "chat", {"ts": _ts(at), "direction": "system", "chat_id": chat_id,
                                    "type": "quiz_answer", "source": "owner_quiz_answer", "task_id": "asker",
                                    "client_message_id": f"quiz_answer:asker:{quiz_id}",
                                    "text": f"[quiz_answer] quiz {quiz_id}", "quiz": settled})


def test_an_evicted_questions_receipt_settles_its_card_never_a_stale_pending(tmp_path):
    """The receipt was frozen before forwarding, the card as published still owed its publication, and the
    question's block has since been evicted: the card settles by the receipt — state, choice, answer address —
    while delivery status comes from the current block alone, so no "pending" outlives the block that owed it."""
    _asked(tmp_path, "qz-1")
    receipt = _receipt_before_forwarding(tmp_path, 1, "qz-1")
    quiz = _question(_read(tmp_path)[1])["quiz"]
    assert quiz["state"] == "answered" and quiz["answer_row_address"] == format_address(row_address(receipt.row))
    assert "question_delivery" not in quiz


def test_a_delivered_block_still_names_the_cards_current_delivery(tmp_path):
    """The block's own delivery remains what the card shows: authority moves to it, not away from it."""
    from ouroboros.task_results import write_task_result

    _asked(tmp_path, "qz-1")
    _receipt_before_forwarding(tmp_path, 1, "qz-1")
    settled = {**QUESTION, "state": "answered", "answered_index": 1,
               "delivery": {"kind": "answer", "state": "delivered"}}
    write_task_result(tmp_path, "asker", "running", owner_quiz={"qz-1": settled})
    assert _question(_read(tmp_path)[1])["quiz"]["question_delivery"] == {"kind": "answer", "state": "delivered"}


@pytest.mark.parametrize("block", [False, True], ids=["evicted", "delivered_block"])
def test_mains_project_pointer_reads_delivery_only_from_the_current_block(tmp_path, block):
    """Main's pointer settles an evicted question by its receipt yet never carries its frozen delivery; a
    current delivered block still names the delivery, block authority intact."""
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "plan-room", name="Plan room")
    settled = {} if not block else {"owner_quiz": {"qz-main": {**QUESTION, "quiz_id": "qz-main", "state": "answered",
                                                                 "answered_index": 1,
                                                                 "delivery": {"kind": "answer", "state": "delivered"}}}}
    write_task_result(tmp_path, "asker", "running", project_id=project["id"], chat_id=project["chat_id"], **settled)
    _asked(tmp_path, "qz-main", chat_id=project["chat_id"])
    _receipt_before_forwarding(tmp_path, 1, "qz-main", chat_id=project["chat_id"])
    pointer = next(message for message in _read(tmp_path)[1]["messages"]
                   if message.get("system_type") == "project_question_pointer")
    assert pointer["quiz_state"] == "answered" and pointer["text"] == "You answered in Plan room"
    if block:
        assert pointer["question_delivery"] == {"kind": "answer", "state": "delivered"}
    else:
        assert "question_delivery" not in pointer


def test_a_one_off_legacy_lookup_stops_at_its_first_exact_hit(tmp_path, monkeypatch):
    """The ordinary single lookup (no shared cache) reads no more of the asker's history than before the
    derived quote: it stops at the first exact card, where the per-read map scans the asker once."""
    from ouroboros.chat_references import published_question

    real = ds.iter_probes
    pulled: list[int] = []

    def counting(root, stream, keys, **kwargs):
        probes = real(root, stream, keys, **kwargs)
        if keys != [("task_id", "asker")]:
            yield from probes
            return
        for row in probes:
            pulled.append(1)
            yield row

    monkeypatch.setattr(ds, "iter_probes", counting)
    _asked(tmp_path, "qz-first", named=False)
    for index in range(60):
        ds.append(tmp_path, "chat", {"ts": _ts(10 + index), "direction": "out", "chat_id": 1, "task_id": "asker",
                                     "type": "quiz", "text": f"Older {index}",
                                     "quiz": {**QUESTION, "quiz_id": f"old-{index}"}})
    assert published_question(tmp_path, "asker", "qz-first").row["quiz"]["quiz_id"] == "qz-first"
    assert len(pulled) == 1, pulled  # first exact hit: 61 cards of the asker, one row read


def test_a_legacy_question_published_in_two_rooms_quotes_neither(tmp_path):
    """One never-named question witnessed in two rooms is the migration's ambiguous case: neither room's
    quote picks the first publication by insertion order, while a unique legacy publication still quotes."""
    from ouroboros.chat_references import question_quote

    first, second, unique = (create_project(tmp_path, slug) for slug in ("first-room", "second-room", "unique-room"))
    _asked(tmp_path, "qz-amb", chat_id=first["chat_id"], named=False)
    _asked(tmp_path, "qz-amb", chat_id=second["chat_id"], named=False)
    row, _ = _late(tmp_path, 5, "qz-amb", "yes", chat_id=first["chat_id"])
    bind_task_to_project(tmp_path, "worker", second["id"], origin={"ref": _origin(row), "text": row["text"]})
    assert question_quote(tmp_path, first["chat_id"], row) == {}
    assert question_quote(tmp_path, second["chat_id"], row) == {}
    _asked(tmp_path, "qz-one", chat_id=unique["chat_id"], named=False)
    alone, _ = _late(tmp_path, 6, "qz-one", "yes", chat_id=unique["chat_id"])
    bind_task_to_project(tmp_path, "worker-2", unique["id"], origin={"ref": _origin(alone), "text": alone["text"]})
    assert question_quote(tmp_path, unique["chat_id"], alone)["derived_question"] is True


@pytest.mark.parametrize("unread", ["importing", "skipped_line"])
def test_a_legacy_question_quotes_only_from_a_whole_chat_history(tmp_path, monkeypatch, unread):
    """Review 3: before cards were named, one question could stand in two rooms, and a part of the chat history not
    read yet (an archive still importing, a line the import skipped) may hold the other one, so the first card found
    proves no uniqueness. A page's quotes read the asker's legacy cards only once the whole chat history is readable,
    confirmed before that read: an import ending while the cards are read vouches for nothing. A named publication
    quotes at once, and the ordinary single lookup keeps its first exact hit."""
    from ouroboros import dialogue_import
    from ouroboros.chat_references import published_question
    from ouroboros.project_registry_recovery import initialize_new_root

    initialize_new_root(tmp_path)  # a root with its room map: Main reads its live present while older parts import

    def card(quiz_id, chat_id, at):  # published before cards were named: no publication name
        return {"ts": _ts(at), "direction": "out", "chat_id": chat_id, "task_id": "asker", "type": "quiz",
                "text": f"Which format for {quiz_id}?", "quiz": {**QUESTION, "quiz_id": quiz_id}}

    read, unread_part = "chat_20261001T000000.jsonl", "chat_20261002T000000.jsonl"
    dialogue_test.seed(tmp_path, archives={read: [card("qz-two-rooms", 1, 1), card("qz-one-room", 1, 2)],
                                           unread_part: [card("qz-two-rooms", 7, 3)] if unread == "importing" else
                                           ["{torn line\n"]})
    dialogue_import.cutover(tmp_path)
    dialogue_import.import_segment(tmp_path, "chat", read)
    if unread == "skipped_line":
        dialogue_import.import_segment(tmp_path, "chat", unread_part)
    _asked(tmp_path, "qz-named", at=4)
    _say(tmp_path, 5, 4)  # the newest page fills above the import frontier, as a room with a live present does
    answers = {_late(tmp_path, at, quiz_id, "yes")[1]: quiz_id
               for at, quiz_id in ((10, "qz-two-rooms"), (11, "qz-one-room"), (12, "qz-named"))}
    real = ds.iter_probes

    def import_ends_while_cards_are_read(root, stream, keys, **kwargs):
        # Read before the history were confirmed whole, the asker's cards would see the import end mid-read.
        yield from real(root, stream, keys, **kwargs)
        if list(keys) == [("task_id", "asker")] and ds.range_coverage(root, "chat")["pending"]:
            dialogue_import.import_segment(root, "chat", unread_part)

    def quoted():
        status, page = _read(tmp_path, n_human=4)
        assert status == 200 and not validate_ingress(page, ChatHistoryResponse), validate_ingress(page, ChatHistoryResponse)
        return {answers[row["row_address"]]: (row.get("reply_to") or {}).get("snippet")
                for row in page["messages"] if row.get("row_address") in answers}

    monkeypatch.setattr(ds, "iter_probes", import_ends_while_cards_are_read)
    named = "Which format for the weekly report?"
    assert quoted() == {"qz-two-rooms": None, "qz-one-room": None, "qz-named": named}
    assert published_question(tmp_path, "asker", "qz-two-rooms").row["chat_id"] == 1  # the one-off lookup: first hit
    if unread == "importing":
        dialogue_import.import_segment(tmp_path, "chat", unread_part)
        assert quoted() == {"qz-two-rooms": None, "qz-one-room": "Which format for qz-one-room?", "qz-named": named}
    else:
        assert quoted() == {"qz-two-rooms": None, "qz-one-room": None, "qz-named": named}
