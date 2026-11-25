"""A mark of reading follows delivery (BIBLE P1, "the unread is named"): a reply, a quiet finish and a ``fold_read``
finish count as read only what their work was shown whole or read (``memory_reading``), never a row after the act;
a delegated draft covers only rows and notes its writer was shown or read, and a part only members it held; a finish
written before the stamp keeps the positional rule. Each guard is pinned in both directions."""
import contextlib
import dataclasses
from types import SimpleNamespace

import pytest

from ouroboros import (chat_chain, memory_coverage as coverage, memory_reading as reading, memory_view as mv,
                       memory_view_conversations as conversations)
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory_floor import FORMS_VERSION
from ouroboros.tool_call_log import _ACTIVE_CALL
from ouroboros.tools.chronicle import _memory_read, _write_page, _write_part, publish_covered_page
from ouroboros.tools.registry import ToolContext
from tests import _memory_inventory_shared as shared
from tests._store_shared import seed_result
from tests.test_memory_room_liveness import _rank, _room, _thread
from tests.test_memory_view_room import MAIN_TASK, _addr, _install, _ts, _view

HELPER = {"kind": "helper", "task_id": "h1", "focus": {"role": "subagent", "task_id": "h1"}}
BASE = {"provider": "github", "account_id": "me", "conversation_id": "repo.1", "thread_id": "i1"}


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


def _mark(room, *spans, **more):
    return {"room": room, "spans": [list(s) for s in spans], "events": [], "messages": [], **more}


def _append(root, *rows):
    shared.append(root / "logs" / "chat.jsonl", *rows)


def _ctx(snapshot, addressed=()):
    """A work's context after one accepted request whose projection drew ``addressed`` forms."""
    facts = {"floor": {"forms_version": FORMS_VERSION, "addressed": [list(pair) for pair in addressed]}}
    plan = SimpleNamespace(core=SimpleNamespace(memory_view_json=mv.snapshot_json(snapshot)),
                           projection=lambda _mode: SimpleNamespace(memory_facts=facts))
    return SimpleNamespace(context_fit_plan=plan, active_context_mode="max", task_metadata={})


@contextlib.contextmanager
def _call(call_id):
    token = _ACTIVE_CALL.set({"tool_call_id": call_id})
    try:
        yield
    finally:
        _ACTIVE_CALL.reset(token)


def read(ctx, call_id="c1", deliver=True, **args):
    """An exact read under its own tool call, then (``deliver``) an accepted request carrying its whole result."""
    with _call(call_id):
        text = _memory_read(ctx, **args)
    if deliver:
        reading.note_view(ctx, None, [{"role": "tool", "tool_call_id": call_id, "content": text}])
    return text


def _entry(pos, **fields):
    row = shared.msg(_ts(pos % 60), f"row {pos}", chat_id=7, **fields)
    return chat_chain.row_address(row), row, pos


def _room_shown(ctx, room):
    return (reading.record_of(ctx)["rooms"].get(str(room)) or {})


# --- recording ------------------------------------------------------------------------------------

def test_a_view_that_held_older_rows_as_a_range_records_only_the_rows_it_held_whole(tmp_path):
    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(4)]
    _append(tmp_path, *rows)
    snapshot, pos = _own(tmp_path, room)
    at = [pos[_addr(row)] for row in rows]
    ranged, usage = _ctx(snapshot, [("F7p", [_addr(rows[0]), "bare:" + _addr(rows[1])])]), {}
    reading.note_view(ranged, usage)
    assert usage["read_spans"]["room"] == room and usage["read_spans"]["spans"] == [[at[2], at[3]]]
    assert reading.RECORD in usage  # the record travels with the work's saved usage
    gap = _ctx(snapshot, [("F7p", [_addr(rows[1])])])  # a shortened row between two whole rows ends a span
    reading.note_view(gap)
    assert _room_shown(gap, room)["spans"] == [[at[0], at[0]], [at[2], at[3]]]
    full = _ctx(snapshot)
    reading.note_view(full)
    assert _room_shown(full, room)["spans"] == [[at[0], at[3]]]


def test_spans_join_rows_in_sorted_order_and_a_task_row_between_ends_them():
    """Display order never widens or empties a span, and a position that is not whole ends a run."""
    assert reading.runs([33, 28, 29, 30, 32], [33, 28, 29, 30, 32]) == [[28, 33]]
    assert reading.runs([10, 12, 14], [10, 14]) == [[10, 10], [14, 14]]
    assert reading.runs([10, 14], [10, 14]) == [[10, 14]]


def test_the_recorder_on_a_real_context_plan_stamps_the_whole_rows_and_a_failure_counts_nothing(monkeypatch):
    from ouroboros import memory_floor as floor
    from tests.test_memory_view_lever import plan

    p, snap = plan()
    lane1 = snap.room["lane1"]
    whole, usage = SimpleNamespace(context_fit_plan=p, active_context_mode="max", task_metadata={}), {}
    reading.note_view(whole, usage)
    assert usage["read_spans"]["spans"] == [[lane1[0]["pos"], lane1[2]["pos"]]]
    chosen = floor.select_view(p, "max", ["F2:" + lane1[1]["address"]])  # my reply as an opening
    part, usage = SimpleNamespace(context_fit_plan=chosen, active_context_mode="max", task_metadata={}), {}
    reading.note_view(part, usage)
    assert usage["read_spans"]["spans"] == [[lane1[0]["pos"], lane1[0]["pos"]], [lane1[2]["pos"], lane1[2]["pos"]]]
    monkeypatch.setattr(reading, "whole_positions", lambda *_a: 1 / 0)
    broken, usage = SimpleNamespace(context_fit_plan=p, active_context_mode="max", task_metadata={}), {}
    reading.note_view(broken, usage)
    assert usage["read_spans"] == reading.NOTHING  # an unrecorded view counts nothing, never the positional rule


def test_the_stamp_rides_a_real_terminal_publication_both_task_done_writers_and_a_reload(tmp_path, monkeypatch):
    """The carriage end to end on the real writers: the terminal fields copy it (an empty stamp when nothing was
    recorded), the room's finish row and both task-done writers carry it, and the stored result reloads it."""
    from ouroboros import headless, loop
    from ouroboros.task_finalization import terminal_result_fields
    from ouroboros.task_results import load_task_result
    from ouroboros.terminal_projection import _project_row
    from supervisor import events, events_task_done
    from tests._wake_summary_shared import seed_wake

    mark = _mark("7", [1, 2])
    assert terminal_result_fields({"read_spans": mark})["read_spans"] == mark
    assert terminal_result_fields({})["read_spans"] == reading.NOTHING
    stored = seed_wake(tmp_path, read_spans=mark)
    assert load_task_result(tmp_path, stored["task_id"])["read_spans"] == mark  # the reload
    assert _project_row(stored["task_id"], stored, {}, {"chat_id": 1})["read_spans"] == mark  # the room's finish row
    ctx, captured = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING={}), []
    monkeypatch.setattr(headless, "terminal_task_files_ready", lambda *a: True)
    monkeypatch.setattr(events, "_finish_task_done_dispatch", lambda *a, **kw: captured.append(kw["task_done_event"]))
    monkeypatch.setattr(events, "_authoritative_terminal_cost", lambda *a: {})
    events_task_done._handle_task_done({"task_id": stored["task_id"], "status": "completed"}, ctx)
    events_task_done._resolve_lifecycle_fault({"task_id": stored["task_id"], "status": "completed"}, ctx, "running")
    assert [event.get("read_spans") for event in captured] == [mark, mark]
    calls = []
    monkeypatch.setattr(reading, "note_view", lambda *args: calls.append(args))
    monkeypatch.setattr(loop, "_observe_transcript_send", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.tools.compact_context.record_context_view", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.own_time_seed.spend_offer", lambda *a, **k: None)
    monkeypatch.setattr("ouroboros.context_input_selection.capture_historical_inputs", lambda *a, **k: None)
    usage, messages = {}, [{"role": "user", "content": "x"}]
    loop._record_transcript_prefix(SimpleNamespace(), messages, 0, usage, None, "t", None, [])
    assert calls and calls[0][1] is usage and calls[0][2] is messages


def test_a_work_that_never_wrote_its_own_finish_counts_nothing_and_a_stored_old_row_stays(tmp_path):
    """The lifecycle writer gives a new work, a new attempt and a host-ended work the empty stamp; a stored row from
    before the rule is never rewritten; a reopened work starts empty."""
    from ouroboros.task_results import load_task_result, write_task_result

    write_task_result(tmp_path, "born01", "running")
    assert load_task_result(tmp_path, "born01")["read_spans"] == reading.NOTHING
    from ouroboros.task_store import patch_document  # a new attempt never inherits its predecessor's reading
    patch_document(tmp_path, "born01", lambda current: {**current, "read_spans": _mark("7", [0, 10 ** 9])})
    write_task_result(tmp_path, "born01", "running", task_attempt=2)
    assert load_task_result(tmp_path, "born01")["read_spans"] == reading.NOTHING
    seed_result(tmp_path, "old001", {"status": "running"})  # started before the rule
    write_task_result(tmp_path, "old001", "failed", result="host ended it")
    assert load_task_result(tmp_path, "old001")["read_spans"] == reading.NOTHING  # a terminal without one
    seed_result(tmp_path, "old002", {"status": "completed"})
    assert "read_spans" not in load_task_result(tmp_path, "old002")  # never rewritten: positional


def test_an_exact_read_counts_only_once_an_accepted_request_carried_it_whole(tmp_path):
    """Staged when the tool runs, credited when a request delivered it; a delivered result cut in the middle credits
    the rows it held whole and not the rest."""
    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n} " + "x" * 50) for n in range(3)]
    _append(tmp_path, *rows)
    _snapshot, pos = _own(tmp_path, room)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader01", current_chat_id=int(room))
    text = read(ctx, deliver=False, room_id=room, rows=True, **{"from": _addr(rows[0]), "to": _addr(rows[2])})
    assert _room_shown(ctx, room).get("read") in (None, [])  # run, not yet delivered
    head, tail = text.split("word 1", 1)
    reading.note_view(ctx, None, [{"role": "tool", "tool_call_id": "c1", "content": head + "…cut…" + tail.split("word 2")[0][:5]}])
    assert _room_shown(ctx, room)["read"] == [[pos[_addr(rows[0])], pos[_addr(rows[0])]]]
    read(ctx, call_id="c2", room_id=room, rows=True, **{"from": _addr(rows[1]), "to": _addr(rows[2])})
    assert _room_shown(ctx, room)["read"] == [[pos[_addr(rows[0])], pos[_addr(rows[2])]]]  # consecutive: one span


def test_a_row_longer_than_one_page_counts_once_its_windows_reach_its_end(tmp_path, monkeypatch):
    import ouroboros.tools.chronicle as chronicle

    _install(tmp_path)
    room, said, _delivered = _thread()
    long_row = said(20, "L" * 3000)
    _append(tmp_path, long_row)
    _snapshot, pos = _own(tmp_path, room)
    monkeypatch.setattr(chronicle, "tool_result_limit", lambda _name: 1200)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="reader01", current_chat_id=int(room))
    start, n = 0, 0
    while True:
        n += 1
        text = read(ctx, call_id=f"w{n}", room_id=room, rows=True, start=start,
                    **{"from": _addr(long_row), "to": _addr(long_row)})
        if "start=" not in text.split("\n")[1]:
            break
        assert not _room_shown(ctx, room).get("read")  # a window alone is not the row
        start = int(text.split("start=")[1].split(" ")[0])
    assert n > 2 and _room_shown(ctx, room)["read"] == [[pos[_addr(long_row)], pos[_addr(long_row)]]]


def test_a_record_longer_than_one_result_counts_once_its_windows_reach_its_end(tmp_path, monkeypatch):
    """The record reader keeps the row reader's window bookkeeping; its continuation comes from the result's window
    line, so a short note whose body names ``next_start`` counts at once."""
    import ouroboros.tools.chronicle as chronicle

    _install(tmp_path)
    store = ChronicleStore(tmp_path)
    long_note = store.write_note(room_id="1", task_id="t9", text="ab" * 2200, author=shared.MIND).record["id"]
    literal = store.write_note(room_id="1", task_id="t9", text="see next_start=7 in the log", author=shared.MIND)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="h1", current_chat_id=1)
    read(ctx, call_id="lit", node_id=literal.record["id"])
    assert literal.record["id"] in reading.record_of(ctx)["nodes"]
    monkeypatch.setattr(chronicle, "tool_result_limit", lambda _name: 1000)
    start, n = 0, 0
    while True:
        n += 1
        text = read(ctx, call_id=f"w{n}", node_id=long_note, start=start)
        line = next(each for each in text.split("\n") if each.startswith("chars "))
        if "next_start=" not in line:
            break
        assert long_note not in reading.record_of(ctx)["nodes"]  # a window alone is not the record
        start = int(line.split("next_start=")[1].split(" ")[0].rstrip(";"))
    assert n > 2 and long_note in reading.record_of(ctx)["nodes"]


def test_an_idle_request_does_no_reading_work(monkeypatch):
    record = {"pending": [], "rooms": {}, "windows": {}}
    monkeypatch.setattr(reading, "_tool_texts", lambda *a: 1 / 0)
    reading._promote(record, [{"role": "tool", "tool_call_id": "x", "content": "y"}])  # nothing pending: no scan
    from ouroboros.task_store import DISPLAY_DROPPED
    assert "read_spans" in DISPLAY_DROPPED


def test_a_thread_snapshot_names_a_message_by_message_id_id_or_ts(tmp_path):
    """F27/F28: a waking thread's snapshot keys its messages as the connector does (GitHub ``id``, Slack ``ts``);
    each names its stored row, and a message absent from the snapshot stays unread."""
    _install(tmp_path)
    room, said, delivered = _thread()
    by_id, by_ts, absent = (shared.msg(_ts(20 + n), text, chat_id=int(room), source="presence:github",
                                       transport={**BASE, "actor": {"platform_actor_id": "u7"}, "message": message})
                            for n, (text, message) in enumerate((("keyed by id", {"id": "99"}),
                                                                 ("keyed by ts", {"ts": "1700.000100"}),
                                                                 ("not in the snapshot", {"id": "100"}))))
    _append(tmp_path, delivered(19), by_id, by_ts, absent)
    ctx, usage = _ctx(_view(tmp_path, MAIN_TASK)[0]), {}
    thread = {"data": {"messages": [{"id": "99"}, {"ts": "1700.000100"}]}}
    ctx.task_metadata = {"presence": {"event": {"conversation_id": "repo.1", "source_event_id": "e1",
                                                 "conversation": {"thread_history": thread}}}}
    reading.note_view(ctx, usage)
    _append(tmp_path, _finish(25, room, "w1", usage[reading.STAMP], wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert [_rank(snapshot, room, row) for row in (by_id, by_ts, absent)] == [1, 1, 2]


def test_a_record_told_through_an_account_or_shown_as_a_room_count_is_not_drawn_whole():
    story = [{"id": "whole"}, {"id": "told", "told_by": "acc1"}, {"id": "counted", "kind": "legacy", "room_id": "5"},
             {"id": "other", "kind": "legacy", "room_id": "6"}]
    snapshot = SimpleNamespace(room={}, story=story)
    assert reading._drawn_records(snapshot, {"F3": ["5"]}) == ["whole", "other"]


def test_the_record_lives_in_the_works_usage_and_survives_a_cold_resume(tmp_path):
    _install(tmp_path)
    room, said, _delivered = _thread()
    row = said(20, "a word")
    _append(tmp_path, row)
    usage = {}
    first = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="w1", current_chat_id=int(room))
    first._accumulated_usage = usage
    read(first, room_id=room, rows=True, **{"from": _addr(row), "to": _addr(row)})
    resumed = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="w1", current_chat_id=int(room))
    resumed._accumulated_usage = dict(usage)  # the saved state, restored in a fresh worker
    assert _room_shown(resumed, room)["read"] and not _room_shown(
        ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="w2", current_chat_id=int(room)), room)


# --- consideration --------------------------------------------------------------------------------

@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "legacy"])
def test_my_reply_counts_as_read_only_what_its_work_was_shown_and_a_legacy_finish_keeps_the_position(tmp_path, stamped):
    _install(tmp_path)
    room, said, delivered = _thread()
    old, ranged, seen, last = (said(20 + n, text) for n, text in enumerate(("old", "held as a range", "seen", "last")))
    _append(tmp_path, old, ranged, seen, last)
    _snapshot, pos = _own(tmp_path, room)
    mark = _mark(room, [pos[_addr(seen)], pos[_addr(last)]])
    _append(tmp_path, delivered(28), _finish(29, room, "p1", mark if stamped else None))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert [_rank(snapshot, room, row) for row in (old, ranged, seen, last)] == ([2, 2, 1, 1] if stamped else [1, 1, 1, 1])


@pytest.mark.parametrize("stored", ["stamped", "running", "running with a stamp", "ended before stamps"])
def test_a_reply_whose_finish_row_is_not_in_the_room_reads_its_works_stored_stamp(tmp_path, stored):
    """The finish row may come later (post-task work), never (a stopped inline turn), or elsewhere (a cross-room reply):
    the work's stored result says what the reply counts; only a work ended before stamps keeps the old position."""
    _install(tmp_path)
    room, said, delivered = _thread()
    old, seen = said(20, "old words"), said(21, "words the work was shown")
    _append(tmp_path, old, seen)
    _snapshot, pos = _own(tmp_path, room)
    document = {"stamped": {"status": "completed", "read_spans": _mark(room, [pos[_addr(seen)], pos[_addr(seen)]])},
                "running": {"status": "running"}, "ended before stamps": {"status": "completed"},
                "running with a stamp": {"status": "running", "read_spans": _mark(room, [0, 10 ** 9])}}[stored]
    seed_result(tmp_path, "p1", document)
    _append(tmp_path, delivered(22))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    expected = {"stamped": (2, 1), "running": (2, 2), "running with a stamp": (2, 2), "ended before stamps": (1, 1)}[stored]
    assert (_rank(snapshot, room, old), _rank(snapshot, room, seen)) == expected


@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "legacy"])
def test_a_quiet_finish_never_counts_words_that_arrived_after_its_last_request(tmp_path, stamped):
    _install(tmp_path)
    room, said, delivered = _thread()
    seen = said(20, "seen before the last request")
    _append(tmp_path, delivered(19), seen)
    _snapshot, pos = _own(tmp_path, room)
    late = said(21, "arrived while the work finished")
    mark = _mark(room, [pos[_addr(seen)], pos[_addr(seen)]])
    _append(tmp_path, late, _finish(22, room, "w1", mark if stamped else None, wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, seen), _rank(snapshot, room, late)) == ((1, 2) if stamped else (1, 1))


def test_a_works_stamp_counts_only_up_to_its_own_act(tmp_path):
    """F7: a span the work was shown that reaches past its reply counts only the rows before the reply."""
    _install(tmp_path)
    room, said, delivered = _thread()
    before = said(20, "before my reply")
    _append(tmp_path, before)
    reply = delivered(21)
    after = said(22, "after my reply")
    _append(tmp_path, reply, after)
    _snapshot, pos = _own(tmp_path, room)
    seed_result(tmp_path, "p1", {"status": "completed",
                                 "read_spans": _mark(room, [pos[_addr(before)], pos[_addr(after)]])})
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, before), _rank(snapshot, room, after)) == (1, 2)


@pytest.mark.parametrize("sender", ["task_id", "transport origin"])
def test_a_sealed_reply_still_reads_its_works_stored_stamp(tmp_path, sender):
    """A page sealed my reply; the words after the page that the reply's work was shown stay counted, found through
    the work's stored stamp by the reply's task id or, when that is blank, the delivery's recorded sender."""
    from tests.test_memory_room_liveness import seal

    _install(tmp_path)
    room, said, delivered = _thread()
    first = delivered(18)
    _append(tmp_path, first)
    seal(tmp_path, room, first)
    word, unseen = said(20, "a word the work was shown"), said(21, "a word it was not shown")
    reply = delivered(22)
    if sender == "transport origin":
        reply = {**reply, "task_id": "", "transport": {**(reply.get("transport") or {}), "origin": {"task_id": "p1"}}}
    else:
        reply = {**reply, "task_id": "p1"}
    _append(tmp_path, word, unseen, reply)
    _snapshot, pos = _own(tmp_path, room)
    seed_result(tmp_path, "p1", {"status": "completed", "read_spans": _mark(room, [pos[_addr(word)]] * 2)})
    seal(tmp_path, room, reply)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, word), _rank(snapshot, room, unseen)) == (1, 2)


@pytest.mark.parametrize("kind", ["quiet", "fold"])
@pytest.mark.parametrize("stamped", [True, False], ids=["stamped", "before the rule"])
def test_only_a_stamped_act_makes_an_overheard_room_owe_and_an_old_one_keeps_its_rank(tmp_path, stamped, kind):
    """A quiet or ``fold_read`` finish from before the rule makes no overheard room owe; it still ranks the word it
    considered as before (the word after it stays unread)."""
    _install(tmp_path)
    room, said, _delivered = _thread()
    heard = said(20, "a person's words I heard")
    _append(tmp_path, heard)
    _snapshot, pos = _own(tmp_path, room)
    fold = {"fold_read": True, "wake_summary": "Heard it; nothing owed."} if kind == "fold" else {}
    late = said(23, "a word after the finish")
    _append(tmp_path, _finish(22, room, "w1", _mark(room, [pos[_addr(heard)]] * 2) if stamped else None,
                              wake_silent=True, quiet_effective=True, **fold), late)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _room(snapshot, room)["overheard"] and (room in _owed_rooms(snapshot)) is stamped
    ranks = [_rank(snapshot, room, row) for row in ([late] if kind == "fold" else [heard, late])]
    assert ranks == ([2] if kind == "fold" else [1, 2])  # a folded word is drawn by its fold, not as a word


def _read_stamp(root, room, rows, task):
    """A listening work that read ``rows`` exactly (its own view held them only as pointers): its finish stamp."""
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task, current_chat_id=int(room))
    ctx._accumulated_usage = usage = {}
    snapshot, positions = _own(root, room)
    ctx.context_fit_plan, ctx.active_context_mode = _ctx(snapshot, [("F7p", list(positions))]).context_fit_plan, "max"
    for n, row in enumerate(rows):
        read(ctx, call_id=f"{task}{n}", room_id=room, rows=True, **{"from": _addr(row), "to": _addr(row)})
    return reading.finish_stamp(usage)


def _awaits(snapshot, room):
    return next((item["awaiting"][room] for item in conversations.owed(snapshot.conversations)
                 if room in item["awaiting"]), None)


def test_a_row_an_old_fold_tells_awaits_once_a_stamped_act_counts_it(tmp_path):
    """Debt is counted per row, whichever fold tells it: a stamped reread of a word an old fold tells makes an
    overheard room owe; a word an old quiet finish considered is not one nobody considered."""
    _install(tmp_path)
    room, said, _delivered = _thread()
    folded, old_heard, fresh, tail = (said(minute, text) for minute, text in (
        (20, "behind the old fold"), (22, "the old quiet finish heard this"), (26, "a new arrival"), (27, "unread")))
    _append(tmp_path, folded, _finish(21, room, "oldfold1", fold_read=True, wake_summary="Earlier sentence.",
                                      wake_silent=True, quiet_effective=True),
            old_heard, _finish(25, room, "oldquiet1", wake_silent=True, quiet_effective=True), fresh, tail)
    assert _awaits(_view(tmp_path, MAIN_TASK)[0], room) is None
    _append(tmp_path, _finish(32, room, "listener1", _read_stamp(tmp_path, room, [folded], "listener1"),
                              wake_silent=True, quiet_effective=True))
    assert _awaits(_view(tmp_path, MAIN_TASK)[0], room) == (1, 2)
    _append(tmp_path, _finish(34, room, "listener2", _read_stamp(tmp_path, room, [fresh], "listener2"),
                              wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _awaits(snapshot, room) == (2, 1) and _rank(snapshot, room, old_heard) == 1


def test_a_works_newest_finish_decides_what_its_sealed_reply_counts(tmp_path):
    from tests.test_memory_room_liveness import seal

    _install(tmp_path)
    room, said, delivered = _thread()
    first = delivered(18)
    _append(tmp_path, first)
    seal(tmp_path, room, first)
    word = said(20, "a word the second finish names")
    _append(tmp_path, word)
    _snapshot, pos = _own(tmp_path, room)
    reply = delivered(21)
    older, newer = (_finish(22 + n, room, "p1", mark) for n, mark in enumerate(
        (_mark(room), _mark(room, [pos[_addr(word)]] * 2))))
    _append(tmp_path, reply, older, newer, said(25, "a later word"))  # the room's newest row is a place row
    for row in (reply, older, newer):
        seal(tmp_path, room, row)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _rank(snapshot, room, word) == 1 and room in _owed_rooms(snapshot)


def test_a_sealed_reply_counts_in_a_room_whose_display_order_is_not_its_stream_order(tmp_path):
    """Imported history makes the open rows read by send time: the lowest open position is the lowest stream
    position in both passes, so the sealed reply's stamp still counts its word."""
    from ouroboros import presence_history as history
    from ouroboros.memory_inventory import open_rows_by_room
    from ouroboros.presence_bindings import conversation_key
    from tests.test_memory_room_liveness import seal

    _install(tmp_path)
    room, said, delivered = _thread()
    live, reply = said(29, "a live question"), delivered(30)
    _append(tmp_path, live, reply)
    _snapshot, pos = _own(tmp_path, room)
    seed_result(tmp_path, "p1", {"status": "completed", "read_spans": _mark(room, [pos[_addr(live)]] * 2)})
    seal(tmp_path, room, reply)
    sent = said(5, "")["ts"]
    key = conversation_key(*(BASE[k] for k in ("provider", "account_id", "conversation_id", "thread_id")))
    event = {**BASE, "conversation_key": key, "source_event_id": "older-history-message",
             "actor": {"platform_actor_id": "u7", "display_name": "Participant"},
             "conversation": {"title": "History"}, "message": {"message_id": "old1", "sent_at": sent},
             "text": "earlier chat history"}
    admission = SimpleNamespace(place=SimpleNamespace(place_id="example-place"), binding_id="b1",
                                transport_skill="bridge", behavior_skill="", profile_fingerprint="")
    request = history.parse_history_request({"import_id": "historic", "records": [
        {"event": event, "source_timestamp": sent, "authorship": "participant"}]})
    assert history.record_history(tmp_path, admission, request)[0] == 200
    assert [entry[2] for entry in open_rows_by_room(tmp_path)[room]][0] > pos[_addr(live)]  # display ≠ stream order
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _rank(snapshot, room, live) == 1 and room in _owed_rooms(snapshot)


def test_the_draft_census_asks_no_status_of_my_own_pages(tmp_path, monkeypatch):
    from tests.test_memory_room_liveness import seal

    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(3)]
    _append(tmp_path, *rows)
    for row in rows:
        seal(tmp_path, room, row)
    asked = []
    real = ChronicleStore._status
    monkeypatch.setattr(ChronicleStore, "_status", lambda self, db, record: asked.append(record["id"])
                        or real(self, db, record))
    assert ChronicleStore(tmp_path).waiting_drafts() == [] and asked == []


def test_an_act_never_names_a_row_after_it_by_a_message_or_event_id(tmp_path):
    """A same-id edit arriving after a quiet finish stays unread; the original it was shown stays counted."""
    _install(tmp_path)
    room, _said, delivered = _thread()
    original = shared.msg(_ts(20), "the original words", chat_id=int(room), source="presence:github",
                          transport={**BASE, "actor": {"platform_actor_id": "u7"}, "message": {"message_id": "m1"}})
    _append(tmp_path, delivered(19), original)
    mark = _mark(room, messages=["repo.1:m1"], events=["e1"])
    edit = shared.msg(_ts(22), "edited words", chat_id=int(room), source="presence:github",
                      transport={**BASE, "actor": {"platform_actor_id": "u7"}, "message": {"message_id": "m1"}},
                      presence_provenance={"source_event_id": "e1"})
    _append(tmp_path, _finish(21, room, "w1", mark, wake_silent=True, quiet_effective=True), edit)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, original), _rank(snapshot, room, edit)) == (1, 2)


def test_a_stamp_of_another_room_still_counts_the_waking_event_and_thread_messages_it_was_given(tmp_path):
    """A place turn moved into a Project stamps the Project room; in the thread its waking word still counts, other
    words of the thread do not."""
    _install(tmp_path)
    room, _said, delivered = _thread()
    woke = shared.msg(_ts(20), "the waking word", chat_id=int(room), source="presence:github",
                      transport={**BASE, "actor": {"platform_actor_id": "u7"}, "message": {"message_id": "m5"}},
                      presence_provenance={"source_event_id": "e5"})
    other = shared.msg(_ts(21), "another word", chat_id=int(room), source="presence:github",
                       transport={**BASE, "actor": {"platform_actor_id": "u8"}, "message": {"message_id": "m6"}})
    _append(tmp_path, woke, other, delivered(22), _finish(23, room, "p1", _mark("999", [0, 10 ** 9], events=["e5"])))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, woke), _rank(snapshot, room, other)) == (1, 2)


def test_an_exact_read_of_another_room_counts_for_my_act_there(tmp_path):
    """A Main work read a thread's word exactly and replied there: the reply counts that word; another word it did
    not read stays unread."""
    _install(tmp_path)
    room, said, delivered = _thread()
    word, other = said(20, "the word Main read exactly"), said(21, "a word nobody read")
    _append(tmp_path, word, other)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="p1", current_chat_id=1)
    ctx._accumulated_usage = {}
    read(ctx, room_id=room, rows=True, **{"from": _addr(word), "to": _addr(word)})
    mark = ctx._accumulated_usage[reading.STAMP]
    assert mark["room"] != room and list(mark["reads"]) == [room]
    _append(tmp_path, delivered(22), _finish(23, "1", "p1", mark))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, word), _rank(snapshot, room, other)) == (1, 2)


def test_an_exact_read_of_older_rows_then_a_quiet_finish_counts_them(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    older, newer = said(20, "older words"), said(21, "newer words")
    _append(tmp_path, delivered(19), older, newer)
    snapshot, pos = _own(tmp_path, room)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="w1", current_chat_id=int(room))
    ctx.context_fit_plan = _ctx(snapshot, [("F7p", [_addr(older)])]).context_fit_plan  # the view held it as an opening
    ctx.active_context_mode, usage = "max", {}
    ctx._accumulated_usage = usage
    reading.note_view(ctx, usage)
    assert not reading.inside(usage["read_spans"]["spans"], pos[_addr(older)])
    read(ctx, room_id=room, rows=True, **{"from": _addr(older), "to": _addr(older)})
    _append(tmp_path, _finish(22, room, "w1", usage["read_spans"], wake_silent=True, quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, older), _rank(snapshot, room, newer)) == (1, 1)


def test_a_wordless_reaction_mine_or_theirs_considers_nothing(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    asked = said(20, "anyone?")
    reaction = {"reaction": {"schema_version": 1, "action": "added", "values": [{"name": "eyes"}]}}
    theirs = shared.msg(_ts(21), "", chat_id=int(room), source="presence:github",
                        transport={**BASE, "actor": {"platform_actor_id": "u8"}, "message": reaction})
    _append(tmp_path, delivered(19), asked, theirs)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 2
    mine = shared.msg(_ts(22), "", chat_id=int(room), direction="out", type="presence_delivery", task_id="p2",
                      source="presence:github", transport={**BASE, "delivery": {"state": "delivered"}, "message": reaction})
    _append(tmp_path, mine)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 2


@pytest.mark.parametrize("wordless", ["sticker", "reaction"])
def test_a_wordless_delivery_is_my_speech_bounded_like_a_reply_unless_it_is_a_typed_reaction(tmp_path, wordless):
    _install(tmp_path)
    room, said, delivered = _thread()
    old, seen = said(20, "an old word"), said(21, "a word the work was shown")
    _append(tmp_path, delivered(19), old, seen)
    _snapshot, pos = _own(tmp_path, room)
    message = ({"sticker": {"file_id": "s1"}} if wordless == "sticker"
               else {"reaction": {"schema_version": 1, "action": "added", "values": [{"name": "eyes"}]}})
    sent = shared.msg(_ts(22), "", chat_id=int(room), direction="out", type="presence_delivery", task_id="p2",
                      source="presence:github", transport={**BASE, "delivery": {"state": "delivered"}, "message": message})
    _append(tmp_path, sent, _finish(23, room, "p2", _mark(room, [pos[_addr(seen)], pos[_addr(seen)]])))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert (_rank(snapshot, room, old), _rank(snapshot, room, seen)) == ((2, 1) if wordless == "sticker" else (2, 2))


def test_a_failed_send_records_no_delivery_and_a_delivered_one_does(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    asked = said(20, "a question")
    failed = shared.msg(_ts(21), "Looking.", chat_id=int(room), direction="system", type="presence_delivery",
                        task_id="p1", source="presence:github", transport={**BASE, "delivery": {"state": "failed"}})
    _append(tmp_path, delivered(19), asked, failed)
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 2
    _append(tmp_path, delivered(22))
    assert _rank(_view(tmp_path, MAIN_TASK)[0], room, asked) == 1


def test_each_stamped_fold_keeps_its_sentence_and_sealing_its_finish_keeps_its_rows(tmp_path):
    """Two stamped folds keep two sentences; a row neither work was shown stays out of both; sealing the earlier
    finish row (a page over that task) leaves its sentence telling its row."""
    _install(tmp_path)
    room, said, delivered = _thread()
    first, second, unseen = said(20, "first words"), said(21, "second words"), said(22, "nobody was shown these")
    _append(tmp_path, delivered(19), first, second, unseen)
    _snapshot, pos = _own(tmp_path, room)
    one, two = _mark(room, [pos[_addr(first)]] * 2), _mark(room, [pos[_addr(first)], pos[_addr(second)]])
    fold_one = _finish(23, room, "f1", one, wake_silent=True, quiet_effective=True, fold_read=True,
                       wake_summary="The first question settled.")
    _append(tmp_path, fold_one, _finish(24, room, "f2", two, wake_silent=True, quiet_effective=True, fold_read=True,
                                        wake_summary="The second question settled."))
    expected = [("The first question settled.", 1), ("The second question settled.", 1)]
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert [(f["sentence"], f["count"]) for f in _room(snapshot, room)["folds"]] == expected
    assert _rank(snapshot, room, unseen) == 2 and "first words" not in text and "second words" not in text
    # Drawn as the newest fold's line and one count line for the earlier ones, with their reader.
    assert "«The second question settled.»: 1 words read and folded" in text
    assert "«The first question settled.»: 1 words" not in text and text.count("words read and folded") == 2
    assert "1 earlier folds of 1 words read and folded; memory_read(room_id=" in text
    from tests.test_memory_room_liveness import seal
    seal(tmp_path, room, fold_one)
    assert [(f["sentence"], f["count"]) for f in _room(_view(tmp_path, MAIN_TASK)[0], room)["folds"]] == expected


def test_three_hundred_stamped_folds_draw_two_lines_not_three_hundred(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    rows = [delivered(0)]
    for n in range(300):
        rows.append({**said(2, f"word {n}"), "ts": f"2026-09-05T{10 + n // 60:02d}:{n % 60:02d}:30+00:00"})
    _append(tmp_path, *rows)
    _snapshot, pos = _own(tmp_path, room)
    _append(tmp_path, *({**_finish(0, room, f"f{n:03d}", _mark(room, [pos[_addr(rows[n + 1])]] * 2),
                                wake_silent=True, quiet_effective=True, fold_read=True,
                                wake_summary=f"Settled question {n:03d}."), "ts": f"2026-09-06T{n // 60:02d}:{n % 60:02d}:00+00:00"}
                              for n in range(300)))
    snapshot, text = _view(tmp_path, MAIN_TASK)
    assert len(_room(snapshot, room)["folds"]) == 300
    assert "«Settled question 299.»: 1 words read and folded" in text and text.count("words read and folded") == 2
    assert "299 earlier folds of 299 words read and folded" in text


def test_the_join_of_sealed_replies_stays_flat_as_sealed_rows_grow(tmp_path, monkeypatch):
    """Item 7's join walks each room's sealed rows back only to its first open row and reads each room's open rows
    once: the sealed rows walked, the open rows read and the works asked do not grow with a room's paged history."""
    _install(tmp_path)
    room, said, delivered = _thread()
    seen, opened, asked = {"n": 0}, {"n": 0}, []
    real_rows, real_marks = conversations.memory_inventory.open_rows_by_room, reading.reply_marks

    class Walked(list):  # counts the steps of every walk back over a room's sealed rows
        def __reversed__(self):
            for entry in list.__reversed__(self):
                seen["n"] += 1
                yield entry

    class Opened(list):  # counts every pass over a room's open rows (a join comparing per sealed row grows here)
        def __iter__(self):
            for entry in list.__iter__(self):
                opened["n"] += 1
                yield entry

    def rows_by_room(*args, sealed=None, **kwargs):
        out = real_rows(*args, sealed=sealed, **kwargs)
        for key in list(sealed or {}):
            sealed[key] = Walked(sealed[key])
        return {key: Opened(rows) for key, rows in out.items()}

    monkeypatch.setattr(conversations.memory_inventory, "open_rows_by_room", rows_by_room)
    monkeypatch.setattr(reading, "reply_marks", lambda root, tasks: asked.append(set(tasks)) or real_marks(root, tasks))
    counts = []
    for batch in range(2):
        rows = [{**(delivered(1) if n % 2 else said(1, f"old word {batch}.{n}")), "task_id": f"old{batch}{n}",
                 "ts": f"2026-09-0{5 + batch}T{n // 60:02d}:{n % 60:02d}:00+00:00"} for n in range(40 + 160 * batch)]
        _append(tmp_path, *rows)
        from ouroboros.tools.chronicle import page_covers

        covers = page_covers(tmp_path, room, to_addr=_addr(rows[-1]))["covers"]
        assert ChronicleStore(tmp_path).publish_page(room_id=room, text="Paged.", covers=covers, author=shared.MIND).ok
        _append(tmp_path, {**said(1, f"an open word {batch}"), "ts": f"2026-09-0{5 + batch}T23:00:00+00:00"})
        seen["n"], opened["n"], asked[:] = 0, 0, []
        _view(tmp_path, MAIN_TASK)
        counts.append((seen["n"], sum(len(ids) for ids in asked), opened["n"]))
    assert counts[1][0] <= counts[0][0] + 4 and counts[1][1] <= counts[0][1] + 1  # 200 more sealed rows: flat
    assert counts[1][2] <= counts[0][2] + 20
    from ouroboros.tools.chronicle import page_covers  # now nothing is open there: its sealed replies ask nothing
    last = [row for row in real_rows(tmp_path).get(room, ())][-1]
    covers = page_covers(tmp_path, room, to_addr=chat_chain.format_address(last[0]))["covers"]
    assert ChronicleStore(tmp_path).publish_page(room_id=room, text="All.", covers=covers, author=shared.MIND).ok
    asked[:] = []
    _view(tmp_path, MAIN_TASK)
    assert not any(task.startswith("old") for ids in asked for task in ids)


def test_the_join_reads_a_rooms_open_rows_once_however_many_sealed_rows_follow_them(tmp_path, monkeypatch):
    """Sealed rows newer than the room's first open row are joined by one comparison each against one lowest open
    position: the passes over the open rows do not grow with them (a join comparing per sealed row would)."""
    from ouroboros.tools.chronicle import page_covers

    _install(tmp_path)
    room, said, delivered = _thread()
    _append(tmp_path, said(1, "an early word that stays open"))
    opened, real_rows = {"n": 0}, conversations.memory_inventory.open_rows_by_room

    class Opened(list):
        def __iter__(self):
            for entry in list.__iter__(self):
                opened["n"] += 1
                yield entry

    monkeypatch.setattr(conversations.memory_inventory, "open_rows_by_room", lambda *a, **k: {
        key: Opened(rows) for key, rows in real_rows(*a, **k).items()})
    counts = []
    for batch, size in enumerate((20, 220)):
        rows = [{**said(1, f"sealed word {batch}.{n}"), "ts": f"2026-09-0{6 + batch}T{n // 60:02d}:{n % 60:02d}:00+00:00"}
                for n in range(size)]
        _append(tmp_path, *rows, {**said(1, f"a later word {batch}"), "ts": f"2026-09-0{6 + batch}T23:00:00+00:00"})
        covers = page_covers(tmp_path, room, from_addr=_addr(rows[0]), to_addr=_addr(rows[-1]))["covers"]
        assert ChronicleStore(tmp_path).publish_page(room_id=room, text="Paged.", covers=covers, author=shared.MIND).ok
        opened["n"] = 0
        _view(tmp_path, MAIN_TASK)
        counts.append(opened["n"])
    assert counts[1] <= counts[0] + 20  # 200 more sealed rows after the first open one: the open rows read alike


def test_acts_before_a_rooms_first_open_row_stay_out_of_its_indexes(tmp_path, monkeypatch):
    """A fold or quiet finish a page already sealed, older than every open row, names no open row: none enters."""
    from ouroboros.tools.chronicle import page_covers

    _install(tmp_path)
    room, said, delivered = _thread()
    rows = [delivered(1)]
    for n in range(30):
        word = {**said(2, f"word {n}"), "ts": f"2026-09-05T11:{n:02d}:00+00:00"}
        _append(tmp_path, word)
        _snapshot, pos = _own(tmp_path, room)
        rows += [word, {**_finish(0, room, f"f{n:02d}", _mark(room, [pos[_addr(word)]] * 2), wake_silent=True,
                                  quiet_effective=True, fold_read=True, wake_summary=f"Fold {n}."),
                        "ts": f"2026-09-05T11:{n:02d}:30+00:00"}]
        _append(tmp_path, rows[-1])
    covers = page_covers(tmp_path, room, to_addr=_addr(rows[-1]))["covers"]
    assert ChronicleStore(tmp_path).publish_page(room_id=room, text="Paged.", covers=covers, author=shared.MIND).ok
    _append(tmp_path, {**said(2, "an open word"), "ts": "2026-09-05T13:00:00+00:00"})
    added = []
    real = reading.Acts.add
    monkeypatch.setattr(reading.Acts, "add", lambda self, pos, mark: added.append(self.room) or real(self, pos, mark))
    _view(tmp_path, MAIN_TASK)
    assert added.count(room) == 0


def test_unstamped_folds_keep_the_last_fold_rule_exactly(tmp_path):
    _install(tmp_path)
    room, said, delivered = _thread()
    words = [said(20 + n, f"word {n}") for n in range(3)]
    folds = [_finish(30 + n, room, f"f{n}", None, wake_silent=True, quiet_effective=True, fold_read=True,
                     wake_summary=f"Fold {n}.") for n in range(3)]
    _append(tmp_path, delivered(19), words[0], folds[0], words[1], folds[1], words[2], folds[2])
    item = _room(_view(tmp_path, MAIN_TASK)[0], room)
    assert [(f["sentence"], f["count"]) for f in item["folds"]] == [("Fold 2.", 4)]  # every row up to it, my reply too, as before


@pytest.mark.parametrize("shape", ["tail", "fragmented", "misses"])
def test_consideration_and_the_fold_walk_stay_flat_as_stamped_acts_grow(monkeypatch, shape):
    """One index per room: each row's keys are read once, and the fold walk examines each claimed row once and never
    a row no fold names, however many folds there are (operation counts, not timing)."""
    calls = {"n": 0}
    real = reading.row_keys

    def counted(meta):
        calls["n"] += 1
        return real(meta)

    monkeypatch.setattr(reading, "row_keys", counted)
    rows = [({"row_sha256": f"{n:064x}"}, {"presence_provenance": {"source_event_id": f"e{n}"}}, n * 2) for n in range(2000)]
    for acts_n in (10, 1000):
        acts, calls["n"] = reading.Acts("7"), 0
        for n in range(acts_n):
            at = 4000 + n if shape == "misses" else n * 4
            spans = ([] if shape == "misses" else [[at - 2, at - 1]] if shape == "fragmented"
                     else [[max(0, at - 300), at]])
            acts.add(at, _mark("7", *spans, events=[] if shape == "misses" else [f"e{n}"]))
        considered = sum(acts.considers(row) for row in rows)
        claims = acts.claims(rows)
        assert calls["n"] <= len(rows)  # keys once per row, not once per act
        assert len(claims) == considered and acts.ops <= considered + acts_n  # each claimed row once, each key once
        if shape == "misses":
            assert considered == 0 and acts.ops == 0


# --- the waiting line -----------------------------------------------------------------------------

def _owed_rooms(snapshot):
    return {item["room_id"] for entry in conversations.owed(snapshot.conversations) for item in entry["rooms"]}


def test_an_overheard_room_waits_only_for_words_an_act_counted_never_for_a_speakerless_row_or_a_note(tmp_path):
    _install(tmp_path)
    room, said, _delivered = _thread()
    heard = said(20, "a person's words I heard")
    revision = shared.msg(_ts(21), "## Summary\nRevised.", chat_id=int(room), source="presence:github",
                          transport={**BASE, "actor": {"platform": "github", "attribution": "unknown"}})
    _append(tmp_path, heard, revision)
    ChronicleStore(tmp_path).write_note(room_id=room, task_id="tn", text="unrelated", author=shared.MIND)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _room(snapshot, room)["overheard"] and room not in _owed_rooms(snapshot)
    _snapshot, pos = _own(tmp_path, room)
    _append(tmp_path, _finish(22, room, "w1", _mark(room, [pos[_addr(heard)]] * 2), wake_silent=True,
                              quiet_effective=True))
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _room(snapshot, room)["overheard"] and room in _owed_rooms(snapshot)


def test_a_room_where_i_spoke_owes_a_telling_only_for_words_someone_of_mine_was_shown(tmp_path):
    from tests.test_memory_room_liveness import seal

    _install(tmp_path)
    room, said, delivered = _thread()
    first = delivered(19)
    _append(tmp_path, first)
    seal(tmp_path, room, first)
    unread = said(20, "nobody was shown this")
    _append(tmp_path, unread)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    assert _rank(snapshot, room, unread) == 2 and room not in _owed_rooms(snapshot)
    _append(tmp_path, delivered(21))
    assert room in _owed_rooms(_view(tmp_path, MAIN_TASK)[0])


def test_words_another_conversation_drew_whole_count_nothing_only_an_act_in_their_room_does(tmp_path):
    """A Main turn that saw a thread's word whole stamps nothing for that thread: the word stays unread there."""
    from tests.test_memory_room_liveness import seal

    _install(tmp_path)
    room, said, delivered = _thread()
    first = delivered(19)
    _append(tmp_path, first)
    seal(tmp_path, room, first)
    word = said(20, "a word Main was shown")
    _append(tmp_path, word)
    main = _view(tmp_path, MAIN_TASK)[0]
    ctx, usage = _ctx(main), {}
    reading.note_view(ctx, usage)
    assert "seen" not in usage["read_spans"] and set(usage["read_spans"]) >= {"room", "spans", "events", "messages"}
    _append(tmp_path, shared.msg(_ts(25), "Done.", chat_id=1, direction="system", type="task_summary", task_id="m1",
                                 root_task_id="m1", summary_kind="terminal_root_projection", status="completed",
                                 read_spans=usage["read_spans"]))
    after = _view(tmp_path, MAIN_TASK)[0]
    assert _rank(after, room, word) == 2 and room not in _owed_rooms(after)


def test_the_waiting_line_states_facts_and_offers_no_seal(tmp_path):
    from ouroboros.memory_waiting import owed_listing

    _install(tmp_path)
    room, said, delivered = _thread()
    early, mine, late = said(18, "a word my reply counted"), delivered(19), said(20, "a word nobody considered")
    _append(tmp_path, early, mine, late)
    listed = owed_listing(_view(tmp_path, MAIN_TASK)[0].conversations)[1]
    assert "2 words read or said here await a page, 1 more nobody considered" in listed
    assert "seal" not in listed and "covers=" not in listed and f"newest open row {_addr(late)}" in listed


def test_a_page_write_makes_no_capture_and_cannot_fail_on_one(tmp_path, monkeypatch):
    _install(tmp_path)
    room, said, delivered = _thread()
    counted, unread = said(18, "a word my reply counted"), said(20, "a word nobody considered")
    _append(tmp_path, counted, delivered(19), unread)
    monkeypatch.setattr(conversations, "capture", lambda *a, **k: 1 / 0)
    result, covers, facts, _s, _rows = publish_covered_page(tmp_path, ChronicleStore(tmp_path), shared.MIND, room, "All.",
                                                     {"to": _addr(unread)})
    assert result.ok and covers["count"] == 3 and "sealed_words_no_act_counted" not in facts


# --- delegated coverage ---------------------------------------------------------------------------

def test_a_helper_draft_covers_only_the_rows_it_read_and_names_what_stayed_open(tmp_path):
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
    read(ctx, deliver=False, room_id=room, rows=True, **{"from": _addr(rows[1]), "to": _addr(rows[2])})
    assert '"reason": "not_read"' in _write_page(ctx, tmp_path, store, HELPER, page)  # same response: not delivered
    reading.note_view(ctx, None, [{"role": "tool", "tool_call_id": "c1", "content": _memory_read(
        ctx, room_id=room, rows=True, **{"from": _addr(rows[1]), "to": _addr(rows[2])})}])
    result, covers, facts, _stamp, _rows = publish_covered_page(tmp_path, store, HELPER, room, "What was said.",
                                                         {"to": _addr(rows[3])}, reader=ctx)
    assert result.ok and covers["count"] == 2 and facts["left_open"]["not shown to this writer whole or read"]["count"] == 2
    sha = [chat_chain.row_address(row)["row_sha256"] for row in rows]
    assert store.sealed_row_refs(room) == {sha[1], sha[2]}
    mind = publish_covered_page(tmp_path, store, shared.MIND, room, "All of it.", {"to": _addr(rows[3])})
    assert mind[0].ok and mind[1]["count"] == 2  # the mind's page takes the rest, as before


def test_a_helper_seals_only_notes_it_was_shown_and_a_note_page_alone_needs_them(tmp_path):
    _install(tmp_path)
    room, said, _delivered = _thread()
    row = shared.msg(_ts(20), "a word with a task", chat_id=int(room), source="presence:github", task_id="t9",
                     transport={**BASE, "actor": {"platform_actor_id": "u7"}})
    _append(tmp_path, row)
    store = ChronicleStore(tmp_path)
    note = store.write_note(room_id=room, task_id="t9", text="My note on t9.", author=shared.MIND).record["id"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="h1", current_chat_id=int(room))
    with pytest.raises(coverage.NotShown) as refused:
        publish_covered_page(tmp_path, store, HELPER, room, "Notes.", {"note_ids": [note]}, reader=ctx)
    assert note in refused.value.left[coverage.NOTES_LEFT]
    read(ctx, room_id=room, rows=True, **{"from": _addr(row), "to": _addr(row)})
    _result, covers, facts, _s, _rows = publish_covered_page(tmp_path, store, HELPER, room, "Rows.", {"to": _addr(row)}, reader=ctx)
    assert covers["note_ids"] == [] and note in facts["left_open"][coverage.NOTES_LEFT]
    other = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="h2", current_chat_id=int(room))
    read(other, call_id="n1", node_id=note)
    done = publish_covered_page(tmp_path, ChronicleStore(tmp_path), {**HELPER, "task_id": "h2"}, room, "Note.",
                                {"note_ids": [note]}, reader=other)
    assert done[0].ok and done[1]["note_ids"] == [note]


def test_a_helper_part_folds_only_members_it_read_or_drafted(tmp_path):
    """Refused before the reads, published after them; its own drafts need no read."""
    from tests.test_child_chronicle_drafts import LEGACY

    _install(tmp_path)
    store = ChronicleStore(tmp_path)
    legacy = [f"legacy-b9{n}-r1" for n in (0, 1)]  # two adjacent sections, as the one-time import lays them down
    assert store.publish([{"id": rid, "kind": "legacy", "room_id": "1", "text": f"old era {n}", "author": LEGACY,
                           "metadata": {"legacy_type": "era", "legacy_block": 90 + n}}
                          for n, rid in enumerate(legacy)]).ok
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="h1", current_chat_id=1)
    part = {"member_ids": legacy, "text": "Both sections.", "room_id": "1", "quotes": None,
            "expected_sequence": store.room_head("1")}
    refused = _write_part(ctx, tmp_path, store, HELPER, part)
    assert '"reason": "not_read"' in refused and all(ident in refused for ident in legacy)
    for n, ident in enumerate(legacy):
        read(ctx, call_id=f"r{n}", node_id=ident)
    folded = _write_part(ctx, tmp_path, store, HELPER, {**part, "expected_sequence": store.room_head("1")})
    assert '"ok": true' in folded
    # Its own drafts need no read: a fresh work of the same helper folds two of its pages unread.
    from ouroboros.tools.chronicle import page_covers

    room, said, _delivered = _thread()
    rows = [said(30 + n, f"word {n}") for n in range(2)]
    _append(tmp_path, *rows)
    pages = [store.publish_page(room_id=room, text=f"page {n}", author=HELPER,
                                covers=page_covers(tmp_path, room, from_addr=_addr(row), to_addr=_addr(row))["covers"])
             for n, row in enumerate(rows)]
    fresh = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="h1", current_chat_id=int(room))
    own = {"member_ids": [page.record["id"] for page in pages], "text": "My pages.", "room_id": room, "quotes": None,
           "expected_sequence": store.room_head(room)}
    assert '"ok": true' in _write_part(fresh, tmp_path, store, HELPER, own)


def test_a_helper_conversation_page_over_a_note_it_read_alone_is_accepted(tmp_path):
    """Astra 11: the room map, bounds and tasks are rebuilt from the retained note when no row was shown."""
    _install(tmp_path)
    room, _said, _delivered = _thread()
    row = shared.msg(_ts(20), "a row with a note", chat_id=int(room), source="presence:github", task_id="t9",
                     transport={**BASE, "actor": {"platform_actor_id": "u7"}})
    _append(tmp_path, row)
    store = ChronicleStore(tmp_path)
    note = store.write_note(room_id=room, task_id="t9", text="Read me.", author=shared.MIND).record["id"]
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="h1", current_chat_id=int(room))
    refused = _write_page(ctx, tmp_path, store, HELPER, {"room_id": room, "covers": {"note_ids": [note]},
                                                          "text": "x", "quotes": None, "expected_sequence": None})
    assert "memory_read(node_id=" in refused  # the reader that reads a note
    read(ctx, call_id="note", node_id=note)
    result, covers, facts, _stamp, _rows = publish_covered_page(tmp_path, store, HELPER, room, "The note alone.",
                                                         {"to": _addr(row), "conversation": True}, reader=ctx)
    assert result.ok and covers["rows"] == ["note:" + note] and covers["count"] == 0
    assert sum(map(len, covers["rooms"].values())) == 1 and facts["left_open"]


def test_a_fallback_page_never_seals_a_row_of_running_work(tmp_path):
    """F19 through the fallback's real input: a running work's row stays out of the covers; a finished task's row is
    given through its task's stamp line and sealed as before."""
    from ouroboros import memory_fallback as mf

    _install(tmp_path)
    seed_result(tmp_path, "run1", {"status": "running"})
    owner = shared.msg(_ts(40), "my human's words", chat_id=1)
    summary = shared.msg(_ts(41), "A finished task's report.", chat_id=1, direction="system",
                         type="task_summary", task_id="finished", root_task_id="finished")
    live = shared.msg(_ts(42), "Still going.", chat_id=1, direction="out", task_id="run1")
    _append(tmp_path, owner, summary, live)
    entries = [(chat_chain.row_address(row), row, chat_chain.stream_position_of(tmp_path, chat_chain.row_address(row)))
               for row in (owner, summary, live)]
    segment = SimpleNamespace(from_addr=entries[0][0], to_addr=entries[-1][0], rows=entries)
    draft = mf.build_writer_input(tmp_path, ChronicleStore(tmp_path), mf.FallbackUnit("page", "1", "open:1",
                                                                                      segment=segment),
                                  budget_tokens=None)
    assert draft.covers["rows"] == [entries[0][0]["row_sha256"], entries[1][0]["row_sha256"]]


def test_owner_rows_whole_in_a_childs_view_count_as_received(tmp_path):
    _install(tmp_path)
    room, said, _delivered = _thread()
    row = said(20, "a row the work order carried")
    _append(tmp_path, row)
    snapshot = _view(tmp_path, MAIN_TASK)[0]
    child = dataclasses.replace(snapshot, spec=dataclasses.replace(snapshot.spec, room_lanes=False),
                                owner_words_sources={"rows": [{"row_sha256": chat_chain.row_address(row)["row_sha256"],
                                                               "room_id": room}]})
    ctx = _ctx(child)
    reading.note_view(ctx)
    entry = (chat_chain.row_address(row), row, 0)
    covers = {"room_id": room, "rows": [entry[0]["row_sha256"]], "request": {}}
    assert coverage.bound_delegated(tmp_path, reading.record_of(ctx), room, covers, [entry])[0]["count"] == 1
    with pytest.raises(coverage.NotShown):
        coverage.bound_delegated(tmp_path, reading.record_of(_ctx(snapshot)), room, covers, [entry])


def test_running_work_stays_open_for_both_writers_by_its_rows_task_or_the_selector_that_named_it(tmp_path):
    _install(tmp_path)
    seed_result(tmp_path, "run1", {"status": "running"})
    done, live, bound = (_entry(n, **fields) for n, fields in ((5, {"task_id": "done1"}), (6, {"task_id": "run1"}),
                                                                 (7, {})))  # the last: my human's words, bound
    everything = {"rooms": {"7": {"read": [[0, 10]]}}, "nodes": []}
    for record in (None, everything):  # the fallback writer (None) and a helper that read them all
        covers, kept, left = coverage.bound_delegated(tmp_path, record, "7", {"room_id": "7", "request": {}},
                                                     [done, live, bound])
        assert [e[2] for e in kept] == [5, 7] and left["running work"]["count"] == 1
        covers, kept, left = coverage.bound_delegated(tmp_path, record, "7", {"request": {"task_ids": ["run1"]}},
                                                     [done, bound])
        assert [e[2] for e in kept] == [5] and left["running work"]["count"] == 1
    assert coverage.bound_delegated(tmp_path, None, "7", {"request": {}}, [done])[1] == [done]


def test_a_task_row_between_whole_spoken_rows_counts_for_a_helper_only_when_read(tmp_path):
    spoken, task = _entry(11), _entry(12, direction="system", type="task_summary", task_id="x1")
    record = {"rooms": {"7": {"spans": [[10, 14]], "read": []}}}
    kept = coverage.bound_delegated(tmp_path, record, "7", {"request": {}}, [spoken, task])[1]
    assert kept == [spoken]
    record["rooms"]["7"]["read"] = [[12, 12]]
    assert coverage.bound_delegated(tmp_path, record, "7", {"request": {}}, [spoken, task])[1] == [spoken, task]


def test_the_fallback_writer_and_a_delegated_writer_share_one_coverage_rule():
    entries = [({"row_sha256": "a" * 64, "chat_id": 5}, {"ts": "t1", "task_id": "x"}, 7),
               ({"row_sha256": "b" * 64, "chat_id": 5}, {"ts": "t2"}, 9)]
    covers = coverage.restrict({"room_id": "5", "mode": "open_to"}, entries)
    assert (covers["rows"], covers["count"], covers["stream_span"], covers["task_ids"], covers["note_ids"]) == (
        ["a" * 64, "b" * 64], 2, [7, 9], ["x"], [])


# --- waiting drafts -------------------------------------------------------------------------------

def test_helper_drafts_awaiting_my_decision_are_counted_with_their_rooms_and_dates(tmp_path):
    from ouroboros.memory_waiting import measure, text as waiting_text
    from ouroboros.tools.chronicle import page_covers

    _install(tmp_path)
    room, said, _delivered = _thread()
    rows = [said(20 + n, f"word {n}") for n in range(3)]
    _append(tmp_path, *rows)
    store = ChronicleStore(tmp_path)
    drafts = []
    for row in rows:
        coverage = page_covers(tmp_path, room, from_addr=_addr(row), to_addr=_addr(row))["covers"]
        drafts.append(store.publish_page(room_id=room, text="A helper's telling.", covers=coverage, author=HELPER))
    assert all(d.ok for d in drafts) and len(store.waiting_drafts()) == 3
    assert store.decide(drafts[0].record["id"], True, shared.MIND, "Right.").ok
    part = store.publish_part(room_id=room, text="Both.", member_ids=[d.record["id"] for d in drafts[1:]], author=HELPER,
                              expected_sequence=store.room_head(room))
    assert part.ok
    waiting = store.waiting_drafts()
    assert len(waiting) == 3 and all(r == room and ts for r, ts in waiting)  # the part has its room and date too
    facts = measure([], {}, [], (), waiting)["drafts"]
    assert facts["count"] == 3 and facts["in_rooms"] == 1 and facts["oldest"] != "unknown" and facts["rooms"] == [[room, 3]]
    assert "members=true" in waiting_text({"drafts": facts})
    assert "drafts" not in measure([], {}, [], ())
