"""``GET /api/chat/search``: a room's saved messages found by their words, over the production store and room rule.

The search reads what the room's history shows (``gateway.chat_search``): literal case-folded words of saved messages,
newest first, in bounded reads whose cursor binds the room, its view, the query and the frozen end. A page without a
match keeps its continuation; the import frontier and unreadable lines are said, never "no match"; a room holding
imported earlier messages is searched in its reading order. Every answer is checked against ``ChatSearchResponse``.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace


from ouroboros import dialogue_store as ds
from ouroboros.gateway import chat_search
from ouroboros.gateway.chat_search import make_chat_search_endpoint, snippet
from ouroboros.gateway.history_contracts import ChatSearchResponse
from ouroboros.gateway.history import make_chat_history_endpoint
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


def _words(match):
    return "".join(part["text"] for part in match["snippet"])


def _texts(page):
    return [_words(match) for match in page["matches"]]


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


def test_words_are_literal_case_folded_and_their_white_space_is_one_space(tmp_path):
    _say(tmp_path, 1, "Die Straße endet hier")
    _say(tmp_path, 2, "STRASSE gesperrt", direction="out")
    _say(tmp_path, 3, "pattern a.b*c literally")
    _say(tmp_path, 4, "regex would take aXbbc")
    _say(tmp_path, 5, "line one\n\n   line two")
    _say(tmp_path, 6, "Новогодняя ЁЛКА")
    _say(tmp_path, 7, "ёлка во дворе")
    assert _texts(_found(tmp_path, q="strasse")) == ["STRASSE gesperrt", "Die Straße endet hier"]
    assert _texts(_found(tmp_path, q="A.B*C")) == ["pattern a.b*c literally"], "never a pattern"
    assert _texts(_found(tmp_path, q="  one   line\ttwo ")) == ["line one line two"]
    assert _found(tmp_path, q="two line")["matches"] == []
    page = _found(tmp_path, q="ёлка")
    assert _texts(page) == ["ёлка во дворе", "Новогодняя ЁЛКА"]
    assert [[part for part in match["snippet"] if part["match"]] for match in page["matches"]] == [
        [{"text": "ёлка", "match": True}], [{"text": "ЁЛКА", "match": True}]]
    assert page["query"] == "ёлка" and page["searched"] == {
        "rows": 7, "reached_ts": _ts(1), "beginning": True, "importing": False, "gaps": []}


def test_a_match_is_the_rows_identity_in_history_and_lands_there(tmp_path):
    for index in range(300):
        _say(tmp_path, index, f"m{index}")
    target = _say(tmp_path, 300, "the far older needle", direction="out")
    for index in range(301, 700):
        _say(tmp_path, index, f"n{index}")
    match, = _found(tmp_path, q="NEEDLE")["matches"]
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
    _say(tmp_path, 5, "needle", text_placeholder=True)
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
    main = _found(tmp_path, q="needle")
    assert _texts(main) == ["needle as a host notice", "needle from Ouroboros", "needle from the owner"]
    assert [match["role"] for match in main["matches"]] == ["system", "assistant", "user"]
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
    pages = _all(tmp_path, q="kingfisher", limit=1)  # each continued page as well, never once per search
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
             quiz={"quiz_id": quiz_id, "options": ["a", "b"], "wait_for_answer": True})
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
    assert [len(page["matches"]) for page in pages] == [1, 0, 0, 0, 0, 1]
    assert [_texts(page) for page in pages if page["matches"]] == [["needle late"], ["needle early"]]
    assert [page["searched"]["rows"] for page in pages] == [50] * 6, "each read is bounded; together they read all"
    reached = [page["searched"]["reached_ts"] for page in pages]
    assert reached == sorted(reached, reverse=True) and reached[-1] == _ts(0)
    assert [page["searched"]["beginning"] for page in pages] == [False] * 5 + [True]
    # A page's limit of matches ends it early as well; the next page goes on right below its last match.
    first, second = _found(tmp_path, q="m", limit=3), None
    second = _found(tmp_path, q="m", limit=3, cursor=first["next_cursor"])
    assert _texts(first) + _texts(second) == [f"m{index}" for index in range(299, 293, -1)]


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
    assert page["matches"] == [] and page["has_more"] and page["searched"]["rows"] == 50
    assert sum(decoded) <= 200, decoded  # one probe page of the room's newest rows


def test_the_cursor_binds_its_room_query_view_and_frozen_end(tmp_path):
    for index in range(30):
        _say(tmp_path, index, f"needle {index}")
    first = _found(tmp_path, q="needle", limit=5)
    _say(tmp_path, 30, "needle that arrived after the search began")
    second = _found(tmp_path, q="needle", limit=5, cursor=first["next_cursor"])
    assert _texts(second) == [f"needle {index}" for index in range(24, 19, -1)], "the set is frozen at the first read"
    assert _texts(_found(tmp_path, q="needle", limit=1)) == ["needle that arrived after the search began"]
    for params, status, reason in (
            ({"q": "other words"}, 400, "search_cursor_invalid"),
            ({"q": "needle", "chat_id": 7}, 409, "search_view_changed"),
            ({"q": "needle", "cursor": first["next_cursor"][:-4] + "AAAA"}, 400, "search_cursor_invalid"),
            ({"q": "   "}, 400, "search_query_invalid"),
            ({"q": "x" * 201}, 400, "search_query_invalid")):
        sent = {"cursor": first["next_cursor"], **params}
        code, body = _search(tmp_path, **sent)
        assert (code, body["reason_code"], body["matches"], body["next_cursor"]) == (status, reason, [], sent["cursor"])
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


def test_a_room_of_imported_earlier_messages_is_searched_in_its_reading_order(tmp_path):
    room = 7001
    _say(tmp_path, 100, "alpha said live", chat_id=room, sender_label="Ann")
    # Imported later, read earlier: their provider times precede the live row they were stored after.
    for order, minute in enumerate((10, 20)):
        _say(tmp_path, 200 + order, f"alpha imported {order}", chat_id=room, type="place_arrival",
             place_id="pl-1", historical_import={"id": f"imp-{order}"}, sender_label="Bo",
             source_order={"time": _ts(minute), "order": order})
    pages = _all(tmp_path, chat_id=room, q="alpha", limit=1)
    assert [_texts(page) for page in pages] == [["alpha said live"], ["alpha imported 1"], ["alpha imported 0"]]
    assert pages[0]["matches"][0]["sender_label"] == "Ann" and pages[-1]["searched"]["beginning"]
    assert [page["searched"]["reached_ts"] for page in pages] == [_ts(100), _ts(20), _ts(10)]


def test_snippets_window_on_words_and_map_folded_matches_back_to_their_text():
    long = "word " * 40 + "the Kingfisher flashed " + "tail " * 60
    parts = snippet(long, "kingfisher")
    assert parts[0] == {"text": "…", "match": False} and parts[-1] == {"text": "…", "match": False}
    assert [part["text"] for part in parts if part["match"]] == ["Kingfisher"]
    words = "".join(part["text"] for part in parts[1:-1])
    assert words.startswith("word ") and len(words) <= 161 and not words.endswith(" ")
    assert [part for part in snippet("Die Straße endet", "strasse") if part["match"]] == [{"text": "Straße", "match": True}]
    assert snippet("two hits: hit and HIT", "hit") == [
        {"text": "two ", "match": False}, {"text": "hit", "match": True}, {"text": "s: ", "match": False},
        {"text": "hit", "match": True}, {"text": " and ", "match": False}, {"text": "HIT", "match": True}]


def test_visible_project_answers_use_their_presented_text_and_author(tmp_path):
    room = create_project(tmp_path, "mirror-room", name="Mirror room")
    for index, kind in enumerate(("project_started", "project_handoff", "project_completion_summary")):
        _say(tmp_path, index, "hidden lifecycle boilerplate", direction="system", type=kind,
             project_id=room["id"], completion_answer=f"Kingfisher answer {index}")
    matches = _found(tmp_path, q="kingfisher")["matches"]
    assert len(matches) == 3
    assert all(match["role"] == "assistant" for match in matches)
    assert _texts(_found(tmp_path, q="kingfisher")) == [f"Kingfisher answer {i}" for i in (2, 1, 0)]
    assert not _found(tmp_path, q="hidden lifecycle boilerplate")["matches"], "hidden host text is not the message"
    _say(tmp_path, 10, "visible fallback notice", direction="system", type="project_completion_summary",
         project_id=room["id"], completion_answer="")
    fallback, = _found(tmp_path, q="visible fallback")["matches"]
    assert fallback["role"] == "system"
    _say(tmp_path, 11, "ordinary visible words", direction="out", type="message", completion_answer="hidden payload")
    assert not _found(tmp_path, q="hidden payload")["matches"], "unrelated fields are not rendered answers"


def test_casefold_expansion_never_duplicates_original_snippet_characters():
    for words, query in (("Die Straße endet hier", "s"), ("ßß", "s"), ("İstanbul", "i")):
        parts = snippet(words, query)
        assert "".join(part["text"] for part in parts) == words


def test_main_work_portals_are_not_messages_but_legacy_notices_stay_searchable(tmp_path):
    room = create_project(tmp_path, "portal-room", name="Portal room")
    for index, kind in enumerate(("project_started", "project_handoff")):
        _say(tmp_path, index, "portal-only falcon", direction="system", type=kind,
             task_id=f"work-{index}", project_id=room["id"], completion_answer="portal-only falcon answer")
        _say(tmp_path, index + 2, "standalone legacy falcon", direction="system", type=kind,
             project_id=room["id"])
    _say(tmp_path, 10, "hidden lifecycle text", direction="system", type="project_completion_summary",
         task_id="finished-work", project_id=room["id"], completion_answer="visible falcon answer")
    assert not _found(tmp_path, q="portal-only")['matches'], "Main renders these rows as work portals, not bubbles"
    assert _texts(_found(tmp_path, q="falcon")) == [
        "visible falcon answer", "standalone legacy falcon", "standalone legacy falcon"]


def test_search_uses_web_caption_words_without_hiding_authored_or_other_transport_text(tmp_path):
    tail = "[Attached file: bird.txt]"
    attachment = [{"name": "bird.txt", "unavailable": "missing"}]
    _say(tmp_path, 1, "Here is the document.\n\n" + tail, source="web", attachments=attachment)
    _say(tmp_path, 2, tail, source="telegram", attachments=attachment)
    _say(tmp_path, 3, "Authored literal: " + tail, source="web", attachments=attachment)
    _say(tmp_path, 4, "Wrong name.\n\n" + tail, source="web", attachments=[{"name": "other.txt"}])
    _say(tmp_path, 5, tail, source="web", attachments=attachment)
    _say(tmp_path, 6, tail, source="web")
    assert _texts(_found(tmp_path, q="document")) == ["Here is the document."]
    assert _texts(_found(tmp_path, q="Attached file")) == [
        tail, "Wrong name. " + tail, "Authored literal: " + tail, tail]
    assert not _found(tmp_path, q="bird.txt] Here")['matches']


def test_generated_attachment_tail_uses_the_same_bounded_names_and_overflow_as_the_composer(tmp_path):
    names = [f"bird-{index}.txt" for index in range(27)]
    tail = "\n".join([f"[Attached file: {name}]" for name in names[:25]] + ["[2 more attached files]"])
    _say(tmp_path, 1, "Visible caption.\n\n" + tail, source="web", attachments=[{"name": name} for name in names])
    assert _texts(_found(tmp_path, q="caption")) == ["Visible caption."]
    assert not _found(tmp_path, q="attached files")['matches']


def test_the_search_runs_off_the_event_loop(tmp_path, monkeypatch):
    import threading

    _say(tmp_path, 1, "needle")
    on_loop, search = [], chat_search.search_room

    def recorded(*args, **kwargs):
        on_loop.append(threading.current_thread() is threading.main_thread())
        return search(*args, **kwargs)
    monkeypatch.setattr(chat_search, "search_room", recorded)
    assert _texts(_found(tmp_path, q="needle")) == ["needle"]
    assert on_loop == [False], "the room's reads, matching and encoding run in a worker thread"


def test_an_unreadable_store_keeps_the_answers_shape(tmp_path, monkeypatch):
    from ouroboros.store import StoreError, StoreMigrating

    _say(tmp_path, 1, "needle")
    for error, reason in ((StoreMigrating("importing"), "migrating"), (StoreError("unreadable"), "history_source_unavailable")):
        monkeypatch.setattr(chat_search, "search_room", lambda *args, error=error, **kwargs: (_ for _ in ()).throw(error))
        code, body = _search(tmp_path, q="needle")
        assert (code, body["reason_code"], body["matches"], body["searched"]["importing"]) == (
            503, reason, [], reason == "migrating")


def test_unreadable_room_authority_is_unavailable_never_no_match(tmp_path):
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
    first = _found(tmp_path, chat_id=room, q="heron", limit=5)
    assert _texts(first) == [f"heron {index}" for index in range(11, 6, -1)] and first["has_more"]
    for path in (_bindings_path(tmp_path), _registry_path(tmp_path)):
        healthy = path.read_bytes()
        path.write_text("{not json", encoding="utf-8")
        for chat_id, cursor in ((room, None), (room, first["next_cursor"]), (1, None)):
            code, body = _search(tmp_path, chat_id=chat_id, q="heron", **({"cursor": cursor} if cursor else {}))
            assert (code, body.get("reason_code"), body["matches"], body["searched"]["beginning"]) == (
                503, "history_source_unavailable", [], False), (path.name, chat_id, body)
            assert (body["has_more"], body["next_cursor"]) == (bool(cursor), cursor), "the same request can be repeated"
        path.write_bytes(healthy)
    second = _found(tmp_path, chat_id=room, q="heron", limit=5, cursor=first["next_cursor"])
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
    assert (code, body.get("reason_code"), body["matches"], body["searched"]["beginning"]) == (
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
    assert (code, body.get("reason_code"), body["matches"], body["searched"]["beginning"]) == (
        503, "history_source_unavailable", [], False), body
    assert _texts(_found(tmp_path, chat_id=room, q="plover")) == ["plover from the bound work's follow-up"]
