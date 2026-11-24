"""A mark of reading follows delivery (BIBLE P1, "the unread is named"): a reply, a quiet finish and a ``fold_read``
finish count as read only what their work was shown whole or read (``memory_reading``); a delegated draft covers
only what its writer was shown or read; a finish written before the stamp keeps the positional rule."""
from types import SimpleNamespace

import pytest

from ouroboros import chat_chain, memory_reading as reading, memory_view as mv, memory_view_conversations as conversations
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory_floor import FORMS_VERSION
from ouroboros.tools.chronicle import _memory_read, _write_page, publish_covered_page
from ouroboros.tools.registry import ToolContext
from tests import _memory_inventory_shared as shared
from tests._store_shared import seed_result
from tests.test_memory_room_liveness import _rank, _room, _thread
from tests.test_memory_view_room import MAIN_TASK, _addr, _install, _ts, _view

HELPER = {"kind": "helper", "task_id": "h1", "focus": {"role": "subagent", "task_id": "h1"}}


def _own(root, room):
    """The room's own focus view: its lane-1 rows by address, with their stream positions."""
    snapshot = _view(root, {"id": "focus1", "type": "task", "chat_id": int(room)}, room_id=str(room))[0]
    assert snapshot.spec.room_lanes and str(snapshot.room["room_id"]) == str(room)
    return snapshot, {item["address"]: item["pos"] for item in snapshot.room["lane1"]}


def _finish(minute, room, task, mark=None, **extra):
    """The canonical finish row of a work in ``room``, with what it was shown there (``mark``) or without (legacy)."""
    return shared.msg(_ts(minute), "Done.", chat_id=int(room), direction="system", type="task_summary", task_id=task,
                      root_task_id=task, summary_kind="terminal_root_projection", status="completed",
                      outcome_authority="canonical_task_result_after_finalization", terminal_origin="model_final",
                      **({"read_spans": mark} if mark is not None else {}), **extra)


def _append(root, *rows):
    shared.append(root / "logs" / "chat.jsonl", *rows)


def _ctx(snapshot, addressed=()):
    """A work's tool context after one accepted request whose projection drew ``addressed`` forms."""
    facts = {"floor": {"forms_version": FORMS_VERSION, "addressed": [list(pair) for pair in addressed]}}
    plan = SimpleNamespace(core=SimpleNamespace(memory_view_json=mv.snapshot_json(snapshot)),
                           projection=lambda _mode: SimpleNamespace(memory_facts=facts))
    return SimpleNamespace(context_fit_plan=plan, active_context_mode="max", task_metadata={})


# --- recording ------------------------------------------------------------------------------------

def test_a_view_that_held_older_rows_as_a_range_records_only_the_rows_it_held_whole(tmp_path):
    """The recording side: the rows the request drew as openings or pointers (F7p) are not delivered; the whole tail
    is one span, and a full view records every row. The work's usage carries the stamp for its finish."""
    _install(tmp_path)
    room, said, delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(4)]
    _append(tmp_path, *rows)
    snapshot, pos = _own(tmp_path, room)
    at = [pos[_addr(row)] for row in rows]
    ranged, usage = _ctx(snapshot, [("F7p", [_addr(rows[0]), "bare:" + _addr(rows[1])])]), {}
    reading.note_view(ranged, usage)
    assert usage["read_spans"] == {"room": room, "spans": [[at[2], at[3]]], "events": [], "messages": []}
    gap = _ctx(snapshot, [("F7p", [_addr(rows[1])])])  # a shortened row between two whole rows ends a span
    reading.note_view(gap)
    assert reading.shown(gap, room)["spans"] == [[at[0], at[0]], [at[2], at[3]]]
    full = _ctx(snapshot)
    reading.note_view(full)
    assert reading.shown(full, room)["spans"] == [[at[0], at[3]]]


def test_an_exact_read_records_the_rows_it_returned_whole_in_the_room_it_read(tmp_path):
    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(3)]
    _append(tmp_path, *rows)
    _snapshot, pos = _own(tmp_path, room)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader01", current_chat_id=int(room))
    assert reading.shown(ctx, room)["spans"] == []
    text = _memory_read(ctx, room_id=room, rows=True, **{"from": _addr(rows[1]), "to": _addr(rows[2])})
    assert "word 1" in text and "word 2" in text
    assert reading.shown(ctx, room)["spans"] == [[pos[_addr(rows[1])], pos[_addr(rows[2])]]]


# --- consideration --------------------------------------------------------------------------------

@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "legacy"])
def test_my_reply_counts_as_read_only_what_its_work_was_shown_and_a_legacy_finish_keeps_the_position(tmp_path, stamped):
    """A reply after a view that held the last rows whole and older ones as a range considers only the last rows;
    the older rows stay unread (rank 2). A finish written before the stamp keeps today's rule: every earlier row."""
    _install(tmp_path)
    room, said, delivered = _thread()
    old, ranged, seen, last = (said(20 + n, text) for n, text in enumerate(("old", "held as a range", "seen", "last")))
    _append(tmp_path, old, ranged, seen, last)
    _snapshot, pos = _own(tmp_path, room)
    mark = {"room": room, "spans": [[pos[_addr(seen)], pos[_addr(last)]]], "events": []}
    _append(tmp_path, delivered(28), _finish(29, room, "p1", mark if stamped else None))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert [_rank(snapshot, room, row) for row in (old, ranged, seen, last)] == ([2, 2, 1, 1] if stamped else [1, 1, 1, 1])


def test_a_reply_after_full_delivery_considers_every_row_as_before(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(3)]
    _append(tmp_path, *rows)
    snapshot, _pos = _own(tmp_path, room)
    full = _ctx(snapshot)
    reading.note_view(full)
    _append(tmp_path, delivered(28), _finish(29, room, "p1", reading.stamp(full, room)))
    assert [_rank(_view(tmp_path, MAIN_TASK)[0], room, row) for row in rows] == [1, 1, 1]


def test_a_running_works_reply_counts_nothing_until_its_finish_says_what_it_was_shown(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    asked = said(20, "can you look?")
    seed_result(tmp_path, "p1", {"status": "running"})
    _append(tmp_path, asked, delivered(28))
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 2
    seed_result(tmp_path, "p1", {"status": "completed"})  # finished with no finish row in hand: the legacy position
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 1


@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "legacy"])
def test_a_quiet_finish_never_counts_words_that_arrived_after_its_last_request(tmp_path, stamped):
    _install(tmp_path)
    room, said, delivered = _thread()
    seen = said(20, "seen before the last request")
    _append(tmp_path, delivered(19), seen)
    _snapshot, pos = _own(tmp_path, room)
    late = said(21, "arrived while the work finished")
    mark = {"room": room, "spans": [[pos[_addr(seen)], pos[_addr(seen)]]], "events": []}
    _append(tmp_path, late, _finish(22, room, "w1", mark if stamped else None, wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, seen), _rank(snapshot, room, late)) == ((1, 2) if stamped else (1, 1))


def test_an_exact_read_of_older_rows_then_a_quiet_finish_counts_them(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    older, newer = said(20, "older words"), said(21, "newer words")
    _append(tmp_path, delivered(19), older, newer)
    snapshot, _pos = _own(tmp_path, room)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="w1", current_chat_id=int(room))
    ctx.context_fit_plan = _ctx(snapshot, [("F7p", [_addr(older)])]).context_fit_plan  # the view held it as an opening
    ctx.active_context_mode = "max"
    reading.note_view(ctx)
    assert reading.inside(reading.shown(ctx, room)["spans"], _own(tmp_path, room)[1][_addr(older)]) is False
    _memory_read(ctx, room_id=room, rows=True, **{"from": _addr(older), "to": _addr(older)})
    _append(tmp_path, _finish(22, room, "w1", reading.stamp(ctx, room), wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, older), _rank(snapshot, room, newer)) == (1, 1)


def test_a_wordless_reaction_mine_or_theirs_considers_nothing(tmp_path):
    """Pinned: a reaction is no worded reply and no recorded person's words; a person's word before it stays unread."""
    _install(tmp_path)
    room, said, delivered = _thread()
    asked = said(20, "anyone?")
    theirs = shared.msg(_ts(21), "", chat_id=int(room), source="presence:github",
                        transport={"provider": "github", "account_id": "me", "conversation_id": "repo.1", "thread_id": "i1",
                                   "actor": {"platform_actor_id": "u8"},
                                   "message": {"reaction": {"schema_version": 1, "action": "added",
                                                            "values": [{"name": "eyes"}]}}})
    _append(tmp_path, delivered(19), asked, theirs)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 2
    mine = shared.msg(_ts(22), "", chat_id=int(room), direction="out", type="presence_delivery", task_id="p2",
                      source="presence:github", transport={"provider": "github", "account_id": "me",
                                                           "conversation_id": "repo.1", "thread_id": "i1",
                                                           "delivery": {"state": "delivered"},
                                                           "message": {"reaction": {"schema_version": 1,
                                                                                    "action": "added",
                                                                                    "values": [{"name": "eyes"}]}}})
    _append(tmp_path, mine)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 2


def test_a_failed_send_records_no_delivery_and_a_delivered_one_does(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    asked = said(20, "a question")
    failed = shared.msg(_ts(21), "Looking.", chat_id=int(room), direction="system", type="presence_delivery",
                        task_id="p1", source="presence:github",
                        transport={"provider": "github", "account_id": "me", "conversation_id": "repo.1",
                                   "thread_id": "i1", "delivery": {"state": "failed"}})
    _append(tmp_path, delivered(19), asked, failed)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 2
    _append(tmp_path, delivered(22))
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 1


def test_each_fold_folds_only_the_rows_its_work_read_and_keeps_its_own_sentence(tmp_path):
    """Two folds keep two sentences: the later one never stands for rows the earlier one told, and a row neither
    work was shown stays out of both."""
    _install(tmp_path)
    room, said, delivered = _thread()
    first, second, unseen = said(20, "first words"), said(21, "second words"), said(22, "nobody was shown these")
    _append(tmp_path, delivered(19), first, second, unseen)
    _snapshot, pos = _own(tmp_path, room)
    one = {"room": room, "spans": [[pos[_addr(first)], pos[_addr(first)]]], "events": []}
    two = {"room": room, "spans": [[pos[_addr(first)], pos[_addr(second)]]], "events": []}
    _append(tmp_path, _finish(23, room, "f1", one, wake_silent=True, quiet_effective=True, fold_read=True,
                              wake_summary="The first question settled."),
            _finish(24, room, "f2", two, wake_silent=True, quiet_effective=True, fold_read=True,
                    wake_summary="The second question settled."))
    snapshot, text = _view(tmp_path, MAIN_TASK)
    folds = _room(snapshot, room)["folds"]
    assert [(fold["sentence"], fold["count"]) for fold in folds] == [("The first question settled.", 1),
                                                                    ("The second question settled.", 1)]
    assert "«The first question settled.»: 1 words read and folded" in text
    assert "«The second question settled.»: 1 words read and folded" in text
    assert _rank(snapshot, room, unseen) == 2 and "first words" not in text and "second words" not in text


# --- the waiting line -----------------------------------------------------------------------------

def test_an_overheard_room_with_considered_untold_words_waits_and_one_nobody_considered_does_not(tmp_path):
    """A room I only heard (no reply, no word of my human) owes a telling once an act of mine counted its words;
    before that it owes nothing."""
    _install(tmp_path)
    room, said, _delivered = _thread()
    heard = said(20, "a person's words I heard")
    _append(tmp_path, heard)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _room(snapshot, room)["overheard"] and _rank(snapshot, room, heard) == 2
    assert room not in {item["room_id"] for owed in conversations.owed(snapshot.conversations) for item in owed["rooms"]}
    _snapshot, pos = _own(tmp_path, room)
    mark = {"room": room, "spans": [[pos[_addr(heard)], pos[_addr(heard)]]], "events": []}
    _append(tmp_path, _finish(21, room, "w1", mark, wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _room(snapshot, room)["overheard"] and _rank(snapshot, room, heard) == 1
    assert room in {item["room_id"] for owed in conversations.owed(snapshot.conversations) for item in owed["rooms"]}


# --- delegated coverage ---------------------------------------------------------------------------

def test_a_helper_draft_covers_only_the_rows_it_was_shown_and_names_what_stayed_open(tmp_path):
    """A delegated writer's ``{to}`` seals exactly the rows it read; the rest stays open and the receipt names it.
    With nothing read the page is a typed refusal naming the gap; the mind's own page keeps its coverage."""
    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(4)]
    _append(tmp_path, *rows)
    store = ChronicleStore(tmp_path)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="h1", current_chat_id=int(room))
    page = {"room_id": room, "covers": {"to": _addr(rows[3])}, "text": "What was said.", "quotes": None,
            "expected_sequence": None}
    refused = _write_page(ctx, tmp_path, store, HELPER, page)
    assert '"reason": "not_read"' in refused and "not shown to this writer whole or read" in refused
    assert store.sealed_row_refs(room) == set()  # nothing was written
    _memory_read(ctx, room_id=room, rows=True, **{"from": _addr(rows[1]), "to": _addr(rows[2])})
    result, covers, facts, _stamp = publish_covered_page(tmp_path, store, HELPER, room, "What was said.",
                                                         {"to": _addr(rows[3])}, reader=ctx)
    assert result.ok and covers["count"] == 2 and facts["rows"] == 2
    assert facts["left_open"]["not shown to this writer whole or read"]["count"] == 2
    sha = [chat_chain.row_address(row)["row_sha256"] for row in rows]
    assert store.sealed_row_refs(room) == {sha[1], sha[2]}  # exactly the rows it read
    mind, mind_covers, _facts, _stamp = publish_covered_page(tmp_path, store, shared.MIND, room, "All of it.",
                                                             {"to": _addr(rows[3])})
    assert mind.ok and mind_covers["count"] == 2  # the mind's page takes the rest, as before


def test_the_fallback_writer_and_a_delegated_writer_share_one_coverage_predicate():
    entries = [({"row_sha256": "a" * 64, "chat_id": 5, "ts": "2026-09-05T10:20:00+00:00"}, {"ts": "t1", "task_id": "x"}, 7),
               ({"row_sha256": "b" * 64, "chat_id": 5, "ts": "2026-09-05T10:21:00+00:00"}, {"ts": "t2"}, 9)]
    covers = reading.restrict({"room_id": "5", "mode": "open_to"}, entries)
    assert (covers["rows"], covers["count"], covers["stream_span"], covers["task_ids"], covers["note_ids"]) == (
        ["a" * 64, "b" * 64], 2, [7, 9], ["x"], [])


# --- waiting drafts -------------------------------------------------------------------------------

def test_helper_drafts_awaiting_my_decision_are_counted_apart_from_decided_ones(tmp_path):
    from ouroboros.memory_waiting import measure, text as waiting_text
    from ouroboros.tools.chronicle import page_covers

    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(2)]
    _append(tmp_path, *rows)
    store = ChronicleStore(tmp_path)
    drafts = []
    for row in rows:
        coverage = page_covers(tmp_path, room, from_addr=_addr(row), to_addr=_addr(row))["covers"]
        drafts.append(store.publish_page(room_id=room, text="A helper's telling.", covers=coverage, author=HELPER))
    assert all(draft.ok for draft in drafts) and len(store.waiting_drafts()) == 2
    assert store.decide(drafts[0].record["id"], True, shared.MIND, "Right.").ok
    assert [room_id for room_id, _ts in store.waiting_drafts()] == [room]
    waiting = measure([], {}, [], (), store.waiting_drafts())
    assert waiting["drafts"]["count"] == 1 and waiting["drafts"]["in_rooms"] == 1
    assert "helper drafts awaiting my decision 1 in 1 room" in waiting_text(waiting)
    assert "drafts" not in measure([], {}, [], ())  # a view captured without the census names none


# --- scope refinements: disjoint delivery, the thread snapshot, wordless speech, the waiting line --------------

def test_an_exact_read_and_a_later_view_stay_disjoint_spans_and_never_mark_the_rows_between(tmp_path):
    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(5)]
    _append(tmp_path, *rows)
    snapshot, pos = _own(tmp_path, room)
    at = [pos[_addr(row)] for row in rows]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="w1", current_chat_id=int(room))
    _memory_read(ctx, room_id=room, rows=True, **{"from": _addr(rows[0]), "to": _addr(rows[0])})
    ctx.context_fit_plan, ctx.active_context_mode = _ctx(snapshot, [("F7p", [_addr(r) for r in rows[:4]])]).context_fit_plan, "max"
    reading.note_view(ctx)
    mark = reading.shown(ctx, room)
    assert mark["spans"] == [[at[0], at[0]], [at[4], at[4]]]
    assert [reading.names(mark, p, {}) for p in at] == [True, False, False, False, True]


def test_the_waking_events_thread_snapshot_counts_its_messages_as_shown_whole(tmp_path):
    """The thread snapshot rides every accepted request whole: its messages count by provider id, a message it
    did not hold does not."""
    _install(tmp_path)
    room, said, delivered = _thread()
    base = {"provider": "github", "account_id": "me", "conversation_id": "repo.1", "thread_id": "i1",
            "actor": {"platform_actor_id": "u7"}}
    held, other = (shared.msg(_ts(20 + n), f"thread word {n}", chat_id=int(room), source="presence:github",
                              transport={**base, "message": {"message_id": f"m{n}"}}) for n in (1, 2))
    _append(tmp_path, delivered(19), held, other)
    snapshot, _pos = _own(tmp_path, room)
    ctx = _ctx(snapshot, [("F7p", [_addr(held), _addr(other)])])  # the room view held both as openings
    ctx.task_metadata = {"presence": {"event": {"source_event_id": "e9", "conversation": {"thread_history": {
        "data": {"messages": [{"message_id": "m1", "text": "thread word 1"}]}}}}}}
    reading.note_view(ctx)
    mark = reading.stamp(ctx, room)
    _snap, pos = _own(tmp_path, room)
    assert mark["messages"] == ["m1"] and mark["events"] == ["e9"]
    assert not any(reading.inside(mark["spans"], pos[_addr(row)]) for row in (held, other))  # openings, not whole
    _append(tmp_path, _finish(25, room, "w1", mark, wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, held), _rank(snapshot, room, other)) == (1, 2)


@pytest.mark.parametrize("wordless", ["sticker", "reaction"])
def test_a_wordless_delivery_is_my_speech_bounded_like_a_reply_unless_it_is_a_typed_reaction(tmp_path, wordless):
    """A sticker or file I sent (no words, no reaction key) is my speech: it counts what its work was shown, never
    more. A typed wordless reaction counts nothing at all."""
    _install(tmp_path)
    room, said, delivered = _thread()
    old, seen = said(20, "an old word"), said(21, "a word the work was shown")
    _append(tmp_path, delivered(19), old, seen)
    _snapshot, pos = _own(tmp_path, room)
    message = ({"sticker": {"file_id": "s1"}} if wordless == "sticker"
               else {"reaction": {"schema_version": 1, "action": "added", "values": [{"name": "eyes"}]}})
    sent = shared.msg(_ts(22), "", chat_id=int(room), direction="out", type="presence_delivery", task_id="p2",
                      source="presence:github", transport={"provider": "github", "account_id": "me",
                                                           "conversation_id": "repo.1", "thread_id": "i1",
                                                           "delivery": {"state": "delivered"}, "message": message})
    mark = {"room": room, "spans": [[pos[_addr(seen)], pos[_addr(seen)]]], "events": []}
    _append(tmp_path, sent, _finish(23, room, "p2", mark))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    # The earlier delivery (no finish in hand, not running) keeps the positional rule for rows before it only.
    assert (_rank(snapshot, room, old), _rank(snapshot, room, seen)) == ((2, 1) if wordless == "sticker" else (2, 2))


def test_a_room_where_i_spoke_owes_a_telling_only_for_words_someone_of_mine_was_shown(tmp_path):
    """The waiting line lists rooms with untold words delivered or read; a room whose open words nobody was shown
    stays a count and owes nothing, and my own reply there brings it back."""
    _install(tmp_path)
    room, said, delivered = _thread()
    first = delivered(19)
    seal_rows = [first]
    from tests.test_memory_room_liveness import seal
    _append(tmp_path, first)
    seal(tmp_path, room, first)  # my earlier reply is told: the room is one where I spoke
    unread = said(20, "nobody was shown this")
    _append(tmp_path, unread)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert not _room(snapshot, room)["overheard"] and _rank(snapshot, room, unread) == 2
    owed = {item["room_id"] for entry in conversations.owed(snapshot.conversations) for item in entry["rooms"]}
    assert room not in owed and seal_rows
    _append(tmp_path, delivered(21))
    owed = {item["room_id"] for entry in conversations.owed(_view(tmp_path, MAIN_TASK)[0].conversations)
            for item in entry["rooms"]}
    assert room in owed
