"""``GET /api/chat/search``: a room's saved messages read for a word search, over the production store and room rule.

The search reads what the room's history shows (``gateway.chat_search``) and answers its saved messages as candidates,
the fields their shown words are drawn from, newest first, in bounded reads (rows and a soft byte target, a row never
split) whose cursor binds the room, its view, the query and the frozen end. Whether a message's shown words hold the
query is the client's (web/tests/chat_search.test.js; the real renderer in tests/test_history_reply_browser.py): the
host leaves out only an owner's literal ASCII words without an ASCII query. A page without a candidate keeps its
continuation; the import frontier and unreadable lines are said, never "no match"; a room holding imported earlier
messages is searched in its reading order. Every answer is checked against ``ChatSearchResponse``.
"""
from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

from ouroboros import dialogue_store as ds
from ouroboros.gateway import chat_search
from ouroboros.gateway.chat_search import make_chat_search_endpoint
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.history_contracts import ChatSearchResponse
from ouroboros.gateway.schema import validate_ingress
from ouroboros.projects_registry import bind_task_to_project, create_project
from tests import _dialogue as dialogue_test


def _ts(index):
    return "2026-10-13T{:02d}:{:02d}:{:02d}Z".format(index // 3600, index // 60 % 60, index % 60)


def _history(root, **params):
    query = {"chat_id": "1", **{key: str(value) for key, value in params.items()}}
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params=query)))
    return response.status_code, json.loads(response.body)


def _search(root, **params):
    query = {"chat_id": "1", **{key: str(value) for key, value in params.items()}}
    response = asyncio.run(make_chat_search_endpoint(root)(SimpleNamespace(query_params=query)))
    body = json.loads(response.body)
    assert not validate_ingress(body, ChatSearchResponse), validate_ingress(body, ChatSearchResponse)
    return response.status_code, body


def _found(root, **params):
    status, page = _search(root, **params)
    assert status == 200, page
    return page


def _say(root, index, text, *, chat_id=1, direction="in", stream="chat", **extra):
    row = {"chat_id": chat_id, "ts": _ts(index), "direction": direction, "text": text, **extra}
    ds.append(root, stream, row)
    return row


def _words(candidate):
    """The words a candidate's row is drawn from: a caption, a Project answer, else its text (a question's too)."""
    return candidate.get("caption") or candidate.get("completion_answer") or candidate["text"]


def _texts(page):
    return [_words(candidate) for candidate in page["candidates"]]


def _all(root, chat_id=1, **params):
    """Every page of one search, in order: the matches newest first and the pages' spans."""
    pages, cursor = [], None
    while True:
        page = _found(root, chat_id=chat_id, **params, **({"cursor": cursor} if cursor else {}))
        pages.append(page)
        cursor = page["next_cursor"]
        if not page["has_more"]:
            assert cursor is None
            return pages


def test_only_an_owners_literal_ascii_words_without_an_ascii_query_are_left_out(tmp_path):
    """The host never decides whether shown words hold the query: Markdown, a System sentence (shown in the install's
    language), non-ASCII words and every row for a non-ASCII query are candidates. An owner's ASCII words are drawn
    literally, so one without the ASCII query (white space as one space, letters lower case) cannot show it."""
    _say(tmp_path, 1, "Die Straße endet hier")
    _say(tmp_path, 2, "STRASSE gesperrt")
    _say(tmp_path, 3, "pattern a.b*c literally")
    _say(tmp_path, 4, "line one\n\n   line TWO")
    _say(tmp_path, 5, "**bold** words from Ouroboros", direction="out")
    _say(tmp_path, 6, "A host notice", direction="system", type="restart_notice")
    _say(tmp_path, 7, "pattern a.b*c, said with ** marks", format="markdown")
    assert _texts(_found(tmp_path, q="strasse")) == [
        "A host notice", "**bold** words from Ouroboros", "STRASSE gesperrt", "Die Straße endet hier"]
    assert _texts(_found(tmp_path, q="A.B*C")) == ["pattern a.b*c, said with ** marks", "A host notice",
                                                    "**bold** words from Ouroboros", "pattern a.b*c literally",
                                                    "Die Straße endet hier"], "non-ASCII words are always the client's"
    assert "line one\n\n   line TWO" in _texts(_found(tmp_path, q="  one   line\ttwo "))
    assert "line one\n\n   line TWO" not in _texts(_found(tmp_path, q="two line"))
    page = _found(tmp_path, q="Straße")
    assert len(page["candidates"]) == 7, "a non-ASCII query is the client's to read in every row"
    assert page["query"] == "Straße" and page["searched"] == {
        "rows": 7, "reached_ts": _ts(1), "beginning": True, "importing": False, "gaps": []}
    assert _found(tmp_path, q="Cafe\u0301   noir")["query"] == "Café noir", "the query as searched is NFC, one space"


def test_a_candidate_carries_the_fields_its_words_are_drawn_from_never_its_media_or_facts(tmp_path):
    _say(tmp_path, 1, "See **this**", direction="out", format="markdown", sender_label="Ouroboros", task_id="t-1",
         client_message_id="secret-id", cost={"usd": 1})
    _say(tmp_path, 2, "🧠 Memory: kept", direction="system", type="legacy_memory_notice", narration=True)
    candidates = _found(tmp_path, q="see")["candidates"]
    assert candidates[1] == {"history_id": candidates[1]["history_id"], "row_address": candidates[1]["row_address"],
                             "role": "assistant", "ts": _ts(1), "text": "See **this**", "markdown": True,
                             "sender_label": "Ouroboros", "task_id": "t-1"}
    assert candidates[0]["role"] == "system" and candidates[0]["system_type"] == "legacy_memory_notice"
    assert candidates[0]["narration"] is True


def test_a_match_is_the_rows_identity_in_history_and_lands_there(tmp_path):
    for index in range(300):
        _say(tmp_path, index, f"m{index}")
    target = _say(tmp_path, 300, "the far older needle", direction="out")
    for index in range(301, 700):
        _say(tmp_path, index, f"n{index}")
    match, = _found(tmp_path, q="NEEDLE")["candidates"]
    assert match["role"] == "assistant" and match["ts"] == target["ts"]
    status, around = _history(tmp_path, around=match["row_address"], n_human=8, n_progress=4)
    assert status == 200 and around["anchor"]["state"] == "found", around
    assert around["anchor"]["history_id"] == match["history_id"]
    landed = next(row for row in around["messages"] if row.get("row_address") == match["row_address"])
    assert landed["text"] == "the far older needle"


def test_only_the_saved_messages_the_room_shows_are_searched(tmp_path):
    project = create_project(tmp_path, "search-room", name="Search room")
    bind_task_to_project(tmp_path, "bound-work", project["id"], origin={"absent": "system"})
    _say(tmp_path, 1, "needle from the owner")
    _say(tmp_path, 2, "needle from Ouroboros", direction="out")
    _say(tmp_path, 3, "needle as a host notice", direction="system", type="restart_notice")
    _say(tmp_path, 4, "plain words", note="needle only in metadata", attachments=[{"name": "needle.txt"}])
    _say(tmp_path, 5, "needle", text_placeholder=True)  # the host's words for an owner who sent none, and no file
    _say(tmp_path, 6, "needle in a routing grounding", direction="out", type="routing_options")
    _say(tmp_path, 7, "needle answer evidence", direction="out", type="quiz_answer", quiz={"quiz_id": "q"})
    _say(tmp_path, 8, "needle terminal summary", direction="out", type="task_summary", task_id="old")
    _say(tmp_path, 9, "needle skill review", direction="out", type="skill_review")
    _say(tmp_path, 10, "needle placed in a card", direction="out", task_id="t", card_row="timeline")
    _say(tmp_path, 11, "needle of an agent-to-agent talk", chat_id=-5)
    _say(tmp_path, 12, "needle with no known author", direction="debug")
    _say(tmp_path, 13, "needle narration", stream="progress", task_id="t", content="needle narration")
    _say(tmp_path, 14, "needle in the Project", chat_id=project["chat_id"])
    _say(tmp_path, 15, "needle written under Main by the bound work", direction="out", task_id="bound-work")
    _say(tmp_path, 16, "(file attached)", text_placeholder=True, attachments=[{"name": "needle-map.pdf"}])
    main = _found(tmp_path, q="needle")
    # An owner row's file cards show their names, so a row with attachments is a candidate whatever its words (the
    # client reads the cards' names, never the metadata note); a placeholder's words are marked as not shown.
    assert _texts(main) == ["(file attached)", "plain words", "needle as a host notice", "needle from Ouroboros",
                            "needle from the owner"]
    assert [match["role"] for match in main["candidates"]] == ["user", "user", "system", "assistant", "user"]
    placeholder, plain = main["candidates"][0], main["candidates"][1]
    assert placeholder["text_placeholder"] is True and placeholder["attachments"] == [
        {"name": "needle-map.pdf", "kind": "file", "available": False}]
    assert plain["attachments"] == [{"name": "needle.txt", "kind": "file", "available": False}]
    assert "text_placeholder" not in plain
    room = _found(tmp_path, chat_id=project["chat_id"], q="needle")
    assert _texts(room) == ["needle written under Main by the bound work", "needle in the Project"]


def test_a_childs_words_are_its_cards_content_by_the_rule_history_shares(tmp_path):
    """A child's own words are lines of its task card (``is_conversation_card_content``), by its stored lineage or,
    on a row written before rows carried it, by the lineage its task result names; its root's words and a notice
    about the child that stands alone are messages of the room."""
    from ouroboros.task_results import write_task_result

    lineage = {"delegation_role": "subagent", "parent_task_id": "root-1", "root_task_id": "root-1"}
    write_task_result(tmp_path, "root-1", "completed")
    write_task_result(tmp_path, "kid-old", "completed", role="researcher", **lineage)
    _say(tmp_path, 1, "kingfisher from the root", direction="out", task_id="root-1")
    _say(tmp_path, 2, "kingfisher from the child", direction="out", task_id="kid-1", subagent_task_id="kid-1", **lineage)
    _say(tmp_path, 3, "kingfisher from a legacy child", direction="out", task_id="kid-old")
    _say(tmp_path, 4, "kingfisher steer not delivered", direction="system", type="steer_not_delivered", task_id="kid-1")
    assert _texts(_found(tmp_path, q="kingfisher")) == ["kingfisher steer not delivered", "kingfisher from the root"]
    status, page = _history(tmp_path, n_human=8, n_progress=4)
    shown = {row.get("text"): row for row in page["messages"]}
    assert status == 200 and shown["kingfisher from a legacy child"]["delegation_role"] == "subagent", \
        "history draws the legacy child's words in its card, by the same task result"


def test_a_search_reads_its_room_and_the_room_rules_lineage_never_a_task_result_or_journal(tmp_path, monkeypatch):
    """A speaker's lineage is the room rule's indexed facts, typed or an older body's metadata role: no task result
    is read (its effective projection reads the queue snapshot and the events journal for a task stored as running),
    nor the routing annotations journal only history's rows carry."""
    from ouroboros import project_dialogue, task_store
    from ouroboros.task_results import write_task_result
    from tests._store_shared import seed_result

    write_task_result(tmp_path, "root-1", "running")
    seed_result(tmp_path, "kid-meta", {"chat_id": 1, "status": "completed", "metadata": {
        "delegation_role": "subagent", "parent_task_id": "root-1", "root_task_id": "root-1"}})
    _say(tmp_path, 1, "kingfisher from the owner", client_message_id="msg-1")
    _say(tmp_path, 2, "kingfisher from a running root", direction="out", task_id="root-1")
    _say(tmp_path, 3, "kingfisher from a metadata-role child", direction="out", task_id="kid-meta")
    reads, documents, annotations = [], task_store.display_documents, project_dialogue.latest_chat_annotations
    monkeypatch.setattr(task_store, "display_documents", lambda *args, **kwargs: (
        reads.append("task result"), documents(*args, **kwargs))[1])
    monkeypatch.setattr(project_dialogue, "latest_chat_annotations", lambda *args, **kwargs: (
        reads.append("annotations"), annotations(*args, **kwargs))[1])
    assert _texts(_found(tmp_path, q="kingfisher")) == ["kingfisher from a running root", "kingfisher from the owner"]
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 1)
    pages = _all(tmp_path, q="kingfisher")  # each continued page as well, never once per search
    assert len(pages) >= 2 and [text for page in pages for text in _texts(page)] == [
        "kingfisher from a running root", "kingfisher from the owner"]
    assert reads == []
    status, page = _history(tmp_path, n_human=8, n_progress=4)
    assert status == 200 and {"task result", "annotations"} <= set(reads), "history's own projection reads both"
    shown = {row.get("text"): row for row in page["messages"]}
    assert shown["kingfisher from a metadata-role child"]["delegation_role"] == "subagent", \
        "history draws the metadata-role child's words in its card, which the search left out"


def test_main_finds_no_copy_of_a_projects_question_which_its_project_finds(tmp_path):
    """Main draws a Project's question only as a copy of that room's, and a settled one not at all (chat_decision.js),
    so its search lists none, open, answered or closed (``history_paging.message_elsewhere``, as unread reads them);
    the Project's own search finds them all."""
    from ouroboros.owner_quiz import record_answered, record_asked, record_closed

    project = create_project(tmp_path, "asking-room", name="Asking room")
    for index, quiz_id in enumerate(("open", "answered", "closed")):
        question = f"Which heron path, {quiz_id}?"
        record_asked(tmp_path, "asker", quiz_id=quiz_id, question=question, options=["a", "b"], wait_for_answer=True)
        _say(tmp_path, index, question, chat_id=project["chat_id"], direction="out", type="quiz", task_id="asker",
             quiz={"quiz_id": quiz_id, "question": question, "options": ["a", "b"], "wait_for_answer": True})
    assert record_answered(tmp_path, "asker", quiz_id="answered", option_index=0, request_id="answer")["ok"]
    assert record_closed(tmp_path, "asker", quiz_id="closed", reason="not needed now", actor="owner",
                         request_id="close")["ok"]
    _say(tmp_path, 5, "a heron path seen from Main")
    status, page = _history(tmp_path, n_human=8, n_progress=4)
    assert status == 200 and [row["system_type"] for row in page["messages"] if row.get("quiz_id")] == [
        "project_question_pointer"] * 3, "Main's history holds them only as copies of the Project's"
    assert _texts(_found(tmp_path, q="heron path")) == ["a heron path seen from Main"]
    assert _texts(_found(tmp_path, chat_id=project["chat_id"], q="heron path")) == [
        "Which heron path, closed?", "Which heron path, answered?", "Which heron path, open?"]


def test_a_settled_questions_card_carries_the_owners_own_answer_as_history_replays_it(tmp_path):
    """A card settled as answered shows the owner's words (question_presentation.js ``renderOwnerAnswer``): its
    candidate carries them as ``comment``, from the asking task's question lifecycle or, once that projection is gone,
    the page's own answer row, as history replays the card. An open or closed card's lifecycle and the answer's
    evidence row are never words."""
    from ouroboros.owner_quiz import record_answered, record_asked, record_closed

    project = create_project(tmp_path, "asking-room", name="Asking room")
    chat = project["chat_id"]
    for index, quiz_id in enumerate(("open", "answered", "closed", "chosen")):
        question = f"Which heron path, {quiz_id}?"
        record_asked(tmp_path, "asker", quiz_id=quiz_id, question=question, options=["a", "b"])
        _say(tmp_path, index, question, chat_id=chat, direction="out", type="quiz", task_id="asker",
             quiz={"quiz_id": quiz_id, "question": question, "options": ["a", "b"]})
    assert record_answered(tmp_path, "asker", quiz_id="answered", option_index=None, request_id="answer",
                           comment="Neither, follow the kingfisher upstream")["ok"]
    assert record_answered(tmp_path, "asker", quiz_id="chosen", option_index=1, request_id="chosen",
                           comment="b, but only at dawn")["ok"]
    assert record_closed(tmp_path, "asker", quiz_id="closed", reason="not needed now", actor="owner",
                         request_id="close")["ok"]
    # An older card whose lifecycle projection is gone: the page's answer row (card evidence, never a message) speaks.
    _say(tmp_path, 8, "Which heron path, evidence?", chat_id=chat, direction="out", type="quiz", task_id="gone",
         quiz={"quiz_id": "old", "question": "Which heron path, evidence?", "options": ["a", "b"]})
    _say(tmp_path, 9, "", chat_id=chat, direction="system", type="quiz_answer", task_id="gone",
         quiz={"quiz_id": "old", "state": "answered", "comment": "Kingfisher pool first"})
    cards = {row["quiz"]["quiz_id"]: row["quiz"] for row in _found(tmp_path, chat_id=chat, q="heron path")["candidates"]}
    assert set(cards) == {"open", "answered", "closed", "chosen", "old"}, "the answer's evidence row is no candidate"
    assert {quiz_id: card.get("comment") for quiz_id, card in cards.items()} == {
        "open": None, "answered": "Neither, follow the kingfisher upstream", "closed": None,
        "chosen": "b, but only at dawn", "old": "Kingfisher pool first"}
    status, page = _history(tmp_path, chat_id=chat, n_human=12, n_progress=4)
    shown = {row["quiz"]["quiz_id"]: row["quiz"] for row in page["messages"] if row.get("msg_type") == "quiz"}
    assert status == 200 and {quiz_id: card.get("comment") if card.get("state") == "answered" else None
                              for quiz_id, card in shown.items()} == {
        quiz_id: card.get("comment") for quiz_id, card in cards.items()}, "the words the room's card draws"


def test_a_settled_questions_answer_on_a_newer_page_is_found_by_one_bounded_probe(tmp_path, monkeypatch):
    """Once the asking task's question lifecycle is gone, the answer row lies after its question, often on a newer page
    than the one that finds the question (the room's feed already holds it there): one indexed probe of the asking
    task's later chat rows, below the search's frozen end and within ``_PROBE_ROWS``, carries its words. An answer
    that arrived after the search began, or one past the probe's bound, leaves the card as its row froze it."""
    project = create_project(tmp_path, "asking-room", name="Asking room")
    chat = project["chat_id"]
    _say(tmp_path, 1, "Which heron path, far?", chat_id=chat, direction="out", type="quiz", task_id="gone",
         quiz={"quiz_id": "far", "question": "Which heron path, far?", "options": ["a", "b"]})
    _say(tmp_path, 2, "Which heron path, late?", chat_id=chat, direction="out", type="quiz", task_id="later",
         quiz={"quiz_id": "late", "question": "Which heron path, late?", "options": ["a", "b"]})
    for index in range(3, 7):
        _say(tmp_path, index, f"filler {index}", chat_id=chat)
    _say(tmp_path, 7, "", chat_id=chat, direction="system", type="quiz_answer", task_id="gone",
         quiz={"quiz_id": "far", "state": "answered", "comment": "Kingfisher pool first"})
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 3)
    first = _found(tmp_path, chat_id=chat, q="heron path")
    assert first["candidates"] == [] and first["has_more"], "the answer's page holds no question"
    _say(tmp_path, 8, "", chat_id=chat, direction="system", type="quiz_answer", task_id="later",
         quiz={"quiz_id": "late", "state": "answered", "comment": "arrived after the search began"})
    cards = {}
    cursor = first["next_cursor"]
    while cursor:
        page = _found(tmp_path, chat_id=chat, q="heron path", cursor=cursor)
        cards.update({row["quiz"]["quiz_id"]: row["quiz"] for row in page["candidates"] if row.get("msg_type") == "quiz"})
        cursor = page["next_cursor"]
    assert {quiz_id: card.get("comment") for quiz_id, card in cards.items()} == {
        "far": "Kingfisher pool first", "late": None}, "the probe reads below the search's frozen end"

    def comments():  # a new search, every page
        return {row["quiz"]["quiz_id"]: row["quiz"].get("comment") for page in _all(tmp_path, chat_id=chat, q="heron path")
                for row in page["candidates"] if row.get("msg_type") == "quiz"}
    assert comments() == {"far": "Kingfisher pool first", "late": "arrived after the search began"}
    # A question whose task speaks once before its answer, a page of rows later.
    _say(tmp_path, 10, "Which heron path, bounded?", chat_id=chat, direction="out", type="quiz", task_id="bounded",
         quiz={"quiz_id": "bounded", "question": "Which heron path, bounded?", "options": ["a", "b"]})
    _say(tmp_path, 11, "the bounded task speaks", chat_id=chat, direction="out", task_id="bounded")
    for index in range(12, 15):
        _say(tmp_path, index, f"filler {index}", chat_id=chat)
    _say(tmp_path, 15, "", chat_id=chat, direction="system", type="quiz_answer", task_id="bounded",
         quiz={"quiz_id": "bounded", "state": "answered", "comment": "past the probe"})
    assert comments()["bounded"] == "past the probe"
    monkeypatch.setattr(chat_search, "_PROBE_ROWS", 1)  # the probe's one row is the task's own words before it
    assert comments()["bounded"] is None, "an answer past the probe's bound leaves the card as its row froze it"


def test_the_search_field_takes_no_more_than_the_host_searches():
    """The field holds the host's query bound in code points (``clipQuery``, never a UTF-16 ``maxlength``): words the
    host would refuse cannot be typed (web/tests/chat_search.test.js pins an emoji query)."""
    import pathlib

    module = pathlib.Path(__file__).resolve().parents[1] / "web" / "modules" / "chat_search.js"
    assert f"const MAX_QUERY_CHARS = {chat_search._MAX_QUERY_CHARS};" in module.read_text(encoding="utf-8")


def _scheduled(root, index, child, *, chat_id=1):
    """The parent's progress row naming its child, the only lineage of a child whose result never entered the store."""
    _say(root, index, "Child scheduled", chat_id=chat_id, stream="progress", task_id="root-1", content="Child scheduled",
         subagent_event="scheduled", delegation_role="subagent", parent_task_id="root-1", root_task_id="root-1",
         subagent_task_id=child)


def test_a_legacy_childs_words_named_by_the_pages_progress_are_its_cards_content(tmp_path):
    """No stored lineage and no task result: the page's progress row naming the child is its lineage, as the feed
    reads it (``history_paging.latest_arrival``'s ``known_children``); the root's words and a notice about the child
    that stands alone stay messages."""
    _scheduled(tmp_path, 1, "kid-old")
    _say(tmp_path, 2, "kingfisher from the root", direction="out", task_id="root-1")
    _say(tmp_path, 3, "kingfisher from a progress-only child", direction="out", task_id="kid-old")
    _say(tmp_path, 4, "kingfisher steer not delivered", direction="system", type="steer_not_delivered", task_id="kid-old")
    page = _found(tmp_path, q="kingfisher")
    assert _texts(page) == ["kingfisher steer not delivered", "kingfisher from the root"]
    assert page["searched"]["beginning"]


def test_a_childs_lineage_named_only_outside_the_page_stays_unknown(tmp_path, monkeypatch):
    """The page's own evidence only: a child named by a progress row older than everything the page read (and the
    page not at the room's beginning) is not known there, so its words stay a match, as that page of the feed shows."""
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 5)
    _scheduled(tmp_path, 1, "kid-far")
    for index in range(2, 20):
        _say(tmp_path, index, f"m{index}")
    _say(tmp_path, 20, "kingfisher from a far child", direction="out", task_id="kid-far")
    page = _found(tmp_path, q="kingfisher")
    assert _texts(page) == ["kingfisher from a far child"] and not page["searched"]["beginning"]


def test_pages_continue_below_their_cut_and_a_page_without_a_match_keeps_its_continuation(tmp_path, monkeypatch):
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 50)
    for index in range(300):
        _say(tmp_path, index, "needle early" if index == 10 else "needle late" if index == 250 else f"m{index}")
    pages = _all(tmp_path, q="needle")
    assert [len(page["candidates"]) for page in pages] == [1, 0, 0, 0, 0, 1]
    assert [_texts(page) for page in pages if page["candidates"]] == [["needle late"], ["needle early"]]
    assert [page["searched"]["rows"] for page in pages] == [50] * 6, "each read is bounded; together they read all"
    reached = [page["searched"]["reached_ts"] for page in pages]
    assert reached == sorted(reached, reverse=True) and reached[-1] == _ts(0)
    assert [page["searched"]["beginning"] for page in pages] == [False] * 5 + [True]


def _sized(root, count, *, chat_id=1):
    """``count`` messages of Ouroboros (every one a candidate), two in every seven long ones; their texts, oldest first."""
    texts = [("long " * 300 if index % 7 in (3, 4) else "short") + f" #{index}" for index in range(count)]
    for index, text in enumerate(texts):
        _say(root, index, text, chat_id=chat_id, direction="out")
    return texts


def test_a_page_ends_at_its_byte_target_on_a_whole_row_and_its_continuation_skips_none(tmp_path, monkeypatch):
    """The stored bytes a page reads are bounded too: the row that crosses the soft target is read whole (one longer
    than the target alone is a page), and the next page starts right below it, so the chain reads every row once."""
    texts = _sized(tmp_path, 40)
    monkeypatch.setattr(chat_search, "_SCAN_BYTES", 1200)
    pages = _all(tmp_path, q="#")
    assert [text for page in pages for text in _texts(page)] == texts[::-1], "every row once, newest first, none split"
    assert len(pages) > 5 and [page["searched"]["beginning"] for page in pages][-1]
    for page in pages:
        texts_read = _texts(page)
        assert all(text.startswith("short") for text in texts_read[:-1]), \
            "a long row (longer than the target alone) is read whole and ends its page"
    assert any(len(page["candidates"]) == 1 and _texts(page)[0].startswith("long") for page in pages), \
        "one longer than the target alone is a page of its own"
    monkeypatch.setattr(chat_search, "_SCAN_BYTES", 10 ** 9)
    assert _texts(_found(tmp_path, q="#")) == texts[::-1], "a target above the room is one page"


def test_a_room_in_reading_order_continues_on_its_byte_target_without_skipping(tmp_path, monkeypatch):
    room = 7002
    texts = _sized(tmp_path, 30, chat_id=room)
    for order in range(12):  # imported later, read earlier than every live row
        text = ("long " * 300 if order % 5 == 1 else "short") + f" imported #{order}"
        _say(tmp_path, 100 + order, text, chat_id=room, direction="out", type="place_arrival", place_id="pl-1",
             historical_import={"id": f"imp-{order}"},
             source_order={"time": f"2026-10-12T00:00:{order:02d}Z", "order": order})
        texts.insert(order, text)
    monkeypatch.setattr(chat_search, "_SCAN_BYTES", 1500)
    pages = _all(tmp_path, chat_id=room, q="#")
    assert [text for page in pages for text in _texts(page)] == texts[::-1]
    assert len(pages) > 4 and pages[-1]["searched"]["beginning"]


def _hydrated(monkeypatch):
    """Every stored row a read decoded (``dialogue_store._row``, the one place a body becomes a row): its count and
    stored bytes."""
    seen, decode = {"rows": 0, "bytes": 0}, ds._row

    def counted(record):
        seen["rows"] += 1
        seen["bytes"] += record["length"]
        return decode(record)
    monkeypatch.setattr(ds, "_row", counted)
    return seen


def test_a_search_of_long_messages_decodes_each_stored_row_once_in_both_orders(tmp_path, monkeypatch):
    """The byte target bounds what a page decodes, not only what it answers: a chain of pages over long messages reads
    each stored body once, in stored order and in reading order alike, never the rest of a probe page past its cut."""
    long_text = "word " * 13_100  # about 64 KiB stored

    def searched(chat_id):
        stored = sum(row.length for row in ds.iter_rows(tmp_path, "chat") if row.row.get("chat_id") == chat_id)
        seen = _hydrated(monkeypatch)
        pages = _all(tmp_path, chat_id=chat_id, q="#")
        monkeypatch.undo()
        assert sum(len(page["candidates"]) for page in pages) == 200 and pages[-1]["searched"]["beginning"]
        assert len(pages) > 20, "a page ends near its byte target"
        return {**seen, "stored": stored, "pages": len(pages)}

    for index in range(200):
        _say(tmp_path, index, f"{long_text}#{index}", direction="out")
    measured = [searched(1)]
    room = 7003
    for order in range(200):  # imported later, read earlier: the room reads in its reading order
        _say(tmp_path, 1000 + order, f"{long_text}imported #{order}", chat_id=room, direction="out", type="place_arrival",
             place_id="pl-1", historical_import={"id": f"imp-{order}"},
             source_order={"time": f"2026-10-12T00:{order // 60:02d}:{order % 60:02d}Z", "order": order})
    measured.append(searched(room))
    assert all(seen["rows"] <= 200 + seen["pages"] and seen["bytes"] <= seen["stored"] * 1.1 for seen in measured), measured


def test_a_request_reads_a_bounded_slice_of_the_room_never_its_history(tmp_path, monkeypatch):
    for index in range(1000):
        _say(tmp_path, index, f"m{index}")
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 50)
    decoded, probe = [], ds.probe_rows

    def counted(*args, **kwargs):
        rows = probe(*args, **kwargs)
        decoded.append(len(rows))
        return rows
    monkeypatch.setattr(ds, "probe_rows", counted)
    page = _found(tmp_path, q="absent words")
    assert page["candidates"] == [] and page["has_more"] and page["searched"]["rows"] == 50
    assert sum(decoded) <= 200, decoded  # one probe page of the room's newest rows


def test_the_cursor_binds_its_room_query_view_and_frozen_end(tmp_path, monkeypatch):
    for index in range(30):
        _say(tmp_path, index, f"needle {index}")
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 5)
    first = _found(tmp_path, q="needle")
    _say(tmp_path, 30, "needle that arrived after the search began")
    second = _found(tmp_path, q="needle", cursor=first["next_cursor"])
    assert _texts(second) == [f"needle {index}" for index in range(24, 19, -1)], "the set is frozen at the first read"
    assert _texts(_found(tmp_path, q="needle"))[0] == "needle that arrived after the search began"
    for params, status, reason in (
            ({"q": "other words"}, 400, "search_cursor_invalid"),
            ({"q": "NEEDLE"}, 400, "search_cursor_invalid"),
            ({"q": "needle", "chat_id": 7}, 409, "search_view_changed"),
            ({"q": "needle", "cursor": first["next_cursor"][:-4] + "AAAA"}, 400, "search_cursor_invalid"),
            ({"q": "   "}, 400, "search_query_invalid"),
            ({"q": "x" * 201}, 400, "search_query_invalid")):
        sent = {"cursor": first["next_cursor"], **params}
        code, body = _search(tmp_path, **sent)
        assert (code, body["reason_code"], body["candidates"], body["next_cursor"]) == (status, reason, [], sent["cursor"])
    project = create_project(tmp_path, "later-room", name="Later")
    bind_task_to_project(tmp_path, "later-work", project["id"], origin={"absent": "system"})
    code, body = _search(tmp_path, q="needle", cursor=first["next_cursor"])
    assert (code, body["reason_code"]) == (409, "search_view_changed"), "a binding may move rows out of the room"


def test_the_import_frontier_is_older_history_not_yet_searched_never_no_match(tmp_path):
    from ouroboros import dialogue_import

    dialogue_test.seed(tmp_path, archives={"chat_20261013T110000.jsonl": [
        {"chat_id": 1, "ts": _ts(1), "direction": "in", "text": "archived needle"}]})
    dialogue_import.cutover(tmp_path)
    _say(tmp_path, 5, "live needle")
    page = _found(tmp_path, q="needle")
    assert _texts(page) == ["live needle"]
    assert (page["has_more"], page["next_cursor"]) == (False, None)
    assert page["searched"]["importing"] and not page["searched"]["beginning"]
    dialogue_test.import_all(tmp_path)
    page = _found(tmp_path, q="needle")
    assert _texts(page) == ["live needle", "archived needle"] and page["searched"]["beginning"]


def test_lines_the_import_could_not_store_are_named_gaps(tmp_path):
    dialogue_test.seed(tmp_path, archives={"chat_20261013T110000.jsonl": [
        {"chat_id": 1, "ts": _ts(1), "direction": "in", "text": "needle before"},
        "{this line is not json\n",
        {"chat_id": 1, "ts": _ts(3), "direction": "in", "text": "needle after"}]})
    dialogue_test.import_all(tmp_path)
    page = _found(tmp_path, q="needle")
    assert _texts(page) == ["needle after", "needle before"]
    assert page["searched"]["gaps"] == ["malformed_jsonl"] and page["searched"]["beginning"]


def test_a_room_of_imported_earlier_messages_is_searched_in_its_reading_order(tmp_path, monkeypatch):
    room = 7001
    _say(tmp_path, 100, "alpha said live", chat_id=room, sender_label="Ann")
    # Imported later, read earlier: their provider times precede the live row they were stored after.
    for order, minute in enumerate((10, 20)):
        _say(tmp_path, 200 + order, f"alpha imported {order}", chat_id=room, type="place_arrival",
             place_id="pl-1", historical_import={"id": f"imp-{order}"}, sender_label="Bo",
             source_order={"time": _ts(minute), "order": order})
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 1)
    pages = _all(tmp_path, chat_id=room, q="alpha")
    assert [_texts(page) for page in pages] == [["alpha said live"], ["alpha imported 1"], ["alpha imported 0"]]
    assert pages[0]["candidates"][0]["sender_label"] == "Ann" and pages[-1]["searched"]["beginning"]
    assert [page["searched"]["reached_ts"] for page in pages] == [_ts(100), _ts(20), _ts(10)]
    # Its cut is a reading key as history's own cursors hold one (``history_paging._row_key``); any other is refused.
    cursor = pages[0]["next_cursor"]
    state = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    for cut in ([state["cut"][0], state["cut"][1], -1], ["x" * 65, None, 0], [state["cut"][0], True, 0]):
        forged = chat_search.encode_cursor({**state, "cut": cut})
        code, body = _search(tmp_path, chat_id=room, q="alpha", cursor=forged)
        assert (code, body["reason_code"]) == (400, "search_cursor_invalid"), cut


def test_project_lifecycle_rows_carry_their_answer_and_ending_for_the_client_to_draw(tmp_path):
    """A lifecycle row is drawn as its authored answer, else its System text (project_answer.js): the candidate carries
    both, the text read as history reads a lifecycle row, and the typed ending its System row is painted from."""
    room = create_project(tmp_path, "mirror-room", name="Mirror room")
    for index, kind in enumerate(("project_started", "project_handoff", "project_completion_summary")):
        _say(tmp_path, index, "**hidden** lifecycle boilerplate", direction="system", type=kind,
             project_id=room["id"], completion_answer=f"Kingfisher **answer** {index}")
    notice = {"phase": "failed", "cause": "The heron work stopped.", "undescribed": []}
    _say(tmp_path, 10, "visible fallback notice", direction="system", type="project_completion_summary",
         project_id=room["id"], project_name="Mirror room", task_name="Heron count", completion_answer="",
         completion_notice=notice)
    _say(tmp_path, 11, "ordinary visible words", direction="out", type="message", completion_answer="hidden payload")
    ordinary, fallback, *answers = _found(tmp_path, q="kingfisher")["candidates"]
    assert [candidate["completion_answer"] for candidate in answers] == [f"Kingfisher **answer** {i}" for i in (2, 1, 0)]
    assert {candidate["text"] for candidate in answers} == {"hidden lifecycle boilerplate"}, "history's plain reading"
    assert all(candidate["role"] == "system" and candidate["project_id"] == room["id"] for candidate in answers)
    assert (fallback["completion_notice"], fallback["task_name"], fallback["project_name"]) == (
        notice, "Heron count", "Mirror room")
    assert "completion_answer" not in ordinary, "a field another row happens to carry is not a lifecycle answer"


def test_main_work_portals_are_not_messages_but_legacy_notices_stay_searchable(tmp_path):
    room = create_project(tmp_path, "portal-room", name="Portal room")
    for index, kind in enumerate(("project_started", "project_handoff")):
        _say(tmp_path, index, "portal-only falcon", direction="system", type=kind,
             task_id=f"work-{index}", project_id=room["id"], completion_answer="portal-only falcon answer")
        _say(tmp_path, index + 2, "standalone legacy falcon", direction="system", type=kind,
             project_id=room["id"])
    _say(tmp_path, 10, "hidden lifecycle text", direction="system", type="project_completion_summary",
         task_id="finished-work", project_id=room["id"], completion_answer="visible falcon answer")
    assert _texts(_found(tmp_path, q="falcon")) == [
        "visible falcon answer", "standalone legacy falcon", "standalone legacy falcon"], \
        "Main renders a started or handed-over work's rows as its portal, not a message"


def test_owner_attachments_reach_the_client_as_names_and_the_ascii_filter_reads_the_whole_stored_text(tmp_path):
    """The bubble hides the web composer's file-name tail from the owner's words (chat_attachments.js): its candidate
    carries the names (with what decides whether its card shows one) and the source that rule reads. The host's ASCII
    filter never leaves out a row with attachments, whose cards' names are words too; on a row without any it reads the
    whole stored text."""
    names = [f"bird-{index}.txt" for index in range(27)]
    tail = "\n".join([f"[Attached file: {name}]" for name in names[:25]] + ["[2 more attached files]"])
    _say(tmp_path, 1, "Visible caption.\n\n" + tail, source="web", attachments=[{"name": name} for name in names])
    _say(tmp_path, 2, "[Attached file: x.txt]", source="telegram", attachments=[{"name": "x.txt", "unavailable": "missing"}])
    _say(tmp_path, 3, "Owner words without the query and no file.")
    telegram, web = _found(tmp_path, q="caption")["candidates"]
    assert telegram["source"] == "telegram" and telegram["attachments"] == [{"name": "x.txt", "kind": "file",
                                                                            "available": False}]
    assert web["source"] == "web" and web["attachments"] == [{"name": name, "kind": "file", "available": False}
                                                             for name in names]
    assert set(web) <= {"history_id", "row_address", "role", "ts", "text", "source", "attachments"}, "never a file's URL"
    assert len(_found(tmp_path, q="attached file")["candidates"]) == 2, "a row without the words or a file is left out"


def test_media_files_links_and_questions_carry_what_their_bubble_draws_from_never_media(tmp_path):
    media = {"download_url": "/api/tasks/t/artifacts/heron.png", "mime": "image/png"}
    _say(tmp_path, 1, "Photo attachment", direction="out", type="photo", caption="A heron on the weir", **media)
    _say(tmp_path, 2, "Photo attachment", direction="out", type="photo")  # no address: its bubble shows these words
    _say(tmp_path, 3, "📎 heron.pdf", direction="out", type="document", caption="", filename="heron.pdf",
         download_url="/api/files/download?path=heron.pdf")
    _say(tmp_path, 4, "links", direction="out", type="links", title="Heron sites",
         actions=[{"label": "Weir", "url": "https://example.org/weir", "extra": "x"}])
    # A question row as the host writes it (supervisor/message_bus.py send_quiz): the question is its text.
    _say(tmp_path, 5, "Which **heron**?", direction="out", type="quiz", task_id="asker",
         quiz={"quiz_id": "q", "options": [{"label": "Weir", "detail": "Below the mill", "recommended": True},
                                           {"label": "Pool"}],
               "stake": "Nesting ends soon", "assumption": "Weir", "state": "open", "wait_for_answer": True})
    _say(tmp_path, 7, "Which older heron?", direction="out", type="quiz", task_id="asker",
         quiz={"quiz_id": "legacy", "options": ["Weir", 3], "option_details": ["Below the mill", None]})
    _say(tmp_path, 6, "card", direction="out", type="card", card={"card_id": "c", "kind": "k", "title": "Heron card"})
    legacy, quiz, links, document, plain, photo = _found(tmp_path, q="heron")["candidates"]
    assert (photo["msg_type"], photo["caption"]) == ("photo", "A heron on the weir") and "download_url" not in photo
    assert "msg_type" not in plain and plain["text"] == "Photo attachment"
    assert (document["msg_type"], document["caption"], document["filename"]) == ("document", "", "heron.pdf"), \
        "its card shows the name (the file itself is never read)"
    assert "download_url" not in document and "mime" not in document
    assert (links["msg_type"], links["title"], links["actions"]) == (
        "links", "Heron sites", [{"label": "Weir", "url": "https://example.org/weir"}])
    # Its card's words (chat_search_words.js draws them as the card does): never its lifecycle, assumption or
    # recommendation; an option of another shape stays one the card refuses.
    assert (quiz["msg_type"], quiz["text"], quiz["quiz"]) == ("quiz", "Which **heron**?", {
        "quiz_id": "q", "options": [{"label": "Weir", "detail": "Below the mill"}, {"label": "Pool"}],
        "stake": "Nesting ends soon"})
    assert legacy["quiz"] == {"quiz_id": "legacy", "options": ["Weir", None], "option_details": ["Below the mill", ""]}


def test_the_search_runs_off_the_event_loop(tmp_path, monkeypatch):
    import threading

    _say(tmp_path, 1, "needle")
    on_loop, search = [], chat_search.search_room

    def recorded(*args, **kwargs):
        on_loop.append(threading.current_thread() is threading.main_thread())
        return search(*args, **kwargs)
    monkeypatch.setattr(chat_search, "search_room", recorded)
    assert _texts(_found(tmp_path, q="needle")) == ["needle"]
    assert on_loop == [False], "the room's reads, selection and encoding run in a worker thread"


def test_an_unreadable_store_keeps_the_answers_shape(tmp_path, monkeypatch):
    from ouroboros.store import StoreError, StoreMigrating

    _say(tmp_path, 1, "needle")
    for error, reason in ((StoreMigrating("importing"), "migrating"), (StoreError("unreadable"), "history_source_unavailable")):
        monkeypatch.setattr(chat_search, "search_room", lambda *args, error=error, **kwargs: (_ for _ in ()).throw(error))
        code, body = _search(tmp_path, q="needle")
        assert (code, body["reason_code"], body["candidates"], body["searched"]["importing"]) == (
            503, reason, [], reason == "migrating")


def test_unreadable_room_authority_is_unavailable_never_no_match(tmp_path, monkeypatch):
    """Which saved messages a room shows is its rule's: the Project registry and the bindings that move a work's words
    into its Project. Unreadable, a search cannot say what the room holds: 503 ``history_source_unavailable`` (the
    surface's "could not be searched"), never a read that reached the beginning with no match. A continuation keeps
    its cursor, and goes on where it was once they read again."""
    from ouroboros.projects_registry import _bindings_path, _registry_path

    project = create_project(tmp_path, "authority-room", name="Authority room")
    bind_task_to_project(tmp_path, "bound-work", project["id"], origin={"absent": "system"})
    for index in range(12):
        _say(tmp_path, index, f"heron {index}", direction="out", task_id="bound-work")
    room = project["chat_id"]
    monkeypatch.setattr(chat_search, "_SCAN_ROWS", 5)
    first = _found(tmp_path, chat_id=room, q="heron")
    assert _texts(first) == [f"heron {index}" for index in range(11, 6, -1)] and first["has_more"]
    for path in (_bindings_path(tmp_path), _registry_path(tmp_path)):
        healthy = path.read_bytes()
        path.write_text("{not json", encoding="utf-8")
        for chat_id, cursor in ((room, None), (room, first["next_cursor"]), (1, None)):
            code, body = _search(tmp_path, chat_id=chat_id, q="heron", **({"cursor": cursor} if cursor else {}))
            assert (code, body.get("reason_code"), body["candidates"], body["searched"]["beginning"]) == (
                503, "history_source_unavailable", [], False), (path.name, chat_id, body)
            assert (body["has_more"], body["next_cursor"]) == (bool(cursor), cursor), "the same request can be repeated"
        path.write_bytes(healthy)
    second = _found(tmp_path, chat_id=room, q="heron", cursor=first["next_cursor"])
    assert _texts(second) == [f"heron {index}" for index in range(6, 1, -1)], "the continuation goes on where it was"


def test_a_row_whose_room_the_rule_cannot_read_is_unavailable_never_left_out(tmp_path, monkeypatch):
    """A row naming only its task lives where the task's recorded lineage puts it (``RoomResolver``): while the task
    store cannot answer, the rule never guesses the row out of its Project and the search says it is unavailable."""
    from ouroboros import projects_registry
    from ouroboros.store import StoreError
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "lineage-room", name="Lineage room")
    bind_task_to_project(tmp_path, "bound-work", project["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "follow-up", "completed", parent_task_id="bound-work", root_task_id="bound-work")
    _say(tmp_path, 1, "osprey from the bound work's follow-up", direction="out", task_id="follow-up")
    room = project["chat_id"]

    def unreadable(*args, **kwargs):
        raise StoreError("unreadable")
    with monkeypatch.context() as patch:  # a process that remembers no lineage yet (as after a restart)
        patch.setattr(projects_registry, "_TASK_LINEAGE", {})
        patch.setattr(projects_registry, "open_store", unreadable)
        code, body = _search(tmp_path, chat_id=room, q="osprey")
    assert (code, body.get("reason_code"), body["candidates"], body["searched"]["beginning"]) == (
        503, "history_source_unavailable", [], False), body
    assert _texts(_found(tmp_path, chat_id=room, q="osprey")) == ["osprey from the bound work's follow-up"]


def test_a_bound_works_descendant_the_lens_cannot_name_is_unavailable_never_left_out(tmp_path, monkeypatch):
    """A Project reads its bound work's descendants by the task store's lineage (``history_paging.room_lens``): a
    descendant's row written under Main with its task id alone is found only through them. While that store cannot
    answer, the search says it is unavailable, never a room read without them."""
    from ouroboros import task_store
    from ouroboros.store import StoreError
    from ouroboros.task_results import write_task_result

    project = create_project(tmp_path, "lens-room", name="Lens room")
    bind_task_to_project(tmp_path, "bound-work", project["id"], origin={"absent": "system"})
    write_task_result(tmp_path, "follow-up", "completed", parent_task_id="bound-work", root_task_id="bound-work")
    _say(tmp_path, 1, "plover from the bound work's follow-up", direction="out", task_id="follow-up")
    room = project["chat_id"]

    def unreadable(*args, **kwargs):
        raise StoreError("unreadable")
    with monkeypatch.context() as patch:
        patch.setattr(task_store, "task_facts", unreadable)
        code, body = _search(tmp_path, chat_id=room, q="plover")
    assert (code, body.get("reason_code"), body["candidates"], body["searched"]["beginning"]) == (
        503, "history_source_unavailable", [], False), body
    assert _texts(_found(tmp_path, chat_id=room, q="plover")) == ["plover from the bound work's follow-up"]
