"""``memory_read(members=true)``: descent through my own tellings by the relations the chronicle records.

A room lists its acting tellings in publication order, then its untold rows as the maximal runs of the store's one
open-row set (``memory_inventory.open_room_rows``) in the room's reading order; a page seals rows and notes, a part
folds members, an account cites source versions, a selection replaces records, a legacy section retells a range of
its room. Each line carries the lead or the labelled opening, the author kind as stored, period, counts and readers.
The read never activates the chronicle: before activation a room names "chronicle not activated". No address is an
argument error, the answer pages by the tool's own bound, and deeper stored levels join an answer only whole.
No test calls a model or the network.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import re

import pytest

from ouroboros import chat_chain, dialogue_store, memory_inventory, tool_call_log as calls
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory_members import NO_LEAD, NOT_ACTIVATED
from ouroboros.tools import chronicle
from ouroboros.tools.chronicle import _chronicle_write, _memory_read, page_covers
from ouroboros.tools.registry import ToolContext
from tests import _memory_inventory_shared as shared

LIGHT = {"kind": "helper", "writer": "fallback_page", "route": {"model": "configured-light"},
         "attribution": "helper draft, not lived"}


def ctx_for(root: pathlib.Path, task_id: str = "root0001") -> ToolContext:
    return ToolContext(repo_dir=root, drive_root=root, task_id=task_id, current_chat_id=1)


def ts(n: int) -> str:
    return f"2026-10-01T00:{n // 60 % 60:02d}:{n % 60:02d}+00:00"


def addr(row) -> str:
    return chat_chain.format_address(chat_chain.row_address(row))


def write(ctx, **args) -> dict:
    reply = json.loads(_chronicle_write(ctx, **args))
    assert reply["ok"], reply
    return reply


def snapshot(root: pathlib.Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def chat(root: pathlib.Path) -> list:
    """Five rows of Main: my human asks, I answer, twice, and my human thanks."""
    said = [("in", "Please look at the build."), ("out", "The build is red. The test is flaky."),
            ("in", "Fix it then."), ("out", "Fixed."), ("in", "Thanks.")]
    rows = [{"chat_id": 1, "direction": d, "ts": ts(i), "text": t, **({"task_id": "taskA001"} if d == "out" else {})}
            for i, (d, t) in enumerate(said)]
    for row in rows:
        dialogue_store.append(root, "chat", row)
    return rows


def listed(text: str) -> list:
    """The answer's listed lines, without its header, its window line or any context level."""
    body = text.split("\n")[2:]
    level = next((i for i, line in enumerate(body) if line.startswith("context, level")), len(body))
    return body[:level]


def members(ctx, **args) -> str:
    return _memory_read(ctx, members=True, **args)


def install(root: pathlib.Path):
    """A page with its lead over rows 1-2 (the lead rule recorded), then a note written without it."""
    rows, ctx = chat(root), ctx_for(root)
    with calls.invocation_scope({"chronicle_lead_rule": 1}):
        page = write(ctx, kind="page", text="I found the build red. Then I fixed it.",
                     covers={"from": addr(rows[1]), "to": addr(rows[2])})
    note = write(ctx, kind="note", text="Remember the flaky test. Its name is in the log.")
    return ctx, rows, page["node_id"], note["node_id"]


# --- the room ---------------------------------------------------------------------------------------

def test_a_room_lists_its_tellings_in_publication_order_then_its_untold_rows_as_runs(tmp_path):
    ctx, rows, page, note = install(tmp_path)
    text = members(ctx, room_id="1")
    head = text.split("\n")[0]
    assert head.startswith("members of room:1: its acting tellings in publication order, then its untold rows")
    assert "untold rows: 3 in 2 runs" in head
    lines = listed(text)
    assert [line.split(" ", 3)[1:3] for line in lines] == [["page", page], ["note", note], ["untold", "rows"],
                                                          ["untold", "rows"]]
    # The page's delivered lead stands without a label; the note's first sentence is labelled as an opening.
    assert "«I found the build red.»;" in lines[0] and NO_LEAD not in lines[0]
    assert f"«Remember the flaky test.»{NO_LEAD};" in lines[1]
    assert "by mind (root, task root0001)" in lines[0] and "seals 2 rows" in lines[0]
    assert f"memory_read(node_id='{page}'); under it: memory_read(members=true, node_id='{page}')" in lines[0]
    assert "under it" not in lines[1]  # a note folds nothing and seals nothing
    # Row 0 alone, then rows 3-4: the page's sealed rows 1-2 break the run in the room's reading order.
    assert "1 row, " in lines[2] and f"from {addr(rows[0])} to {addr(rows[0])} in reading order" in lines[2]
    assert "2 rows, " in lines[3] and f"from {addr(rows[3])} to {addr(rows[4])} in reading order" in lines[3]
    assert f"memory_read(room_id='1', rows=true, from='{addr(rows[3])}', to='{addr(rows[4])}')" in lines[3]
    assert "speakers: " in lines[3] and "2026-10-01 00:00 → 2026-10-01 00:00" in lines[3]


def test_untold_rows_are_the_one_open_row_computation(tmp_path, monkeypatch):
    """The runs are collected by ``memory_inventory.open_room_rows`` from its own one read, never computed a second
    way: what it collects is what the room lists, so a narrower collection narrows the runs."""
    ctx, rows, _page, _note = install(tmp_path)
    assert "untold rows: 3 in 2 runs" in members(ctx, room_id="1").split("\n")[0]
    runs: list = []
    assert [entry[0] for run in (memory_inventory.open_room_rows(tmp_path, "1", runs=runs),) for entry in run] == [
        entry[0] for run in runs for entry in run]  # the open set and its runs: the same rows, one read
    assert [len(run) for run in runs] == [1, 2]
    real = memory_inventory.open_room_rows

    def narrow(root, room, gaps=None, *, runs=None, every=None):
        found: list = []
        kept = real(root, room, gaps, runs=found, every=every)
        if runs is not None:
            runs.append(found[0][:1])
        return kept[:1]

    monkeypatch.setattr(memory_inventory, "open_room_rows", narrow)
    text = members(ctx, room_id="1")
    assert "untold rows: 1 in 1 run" in text.split("\n")[0]
    assert [line for line in listed(text) if line.startswith("- untold")] == [
        line for line in listed(text) if f"from {addr(rows[0])} to {addr(rows[0])}" in line]


def test_before_activation_a_room_names_the_gap_and_the_read_activates_nothing(tmp_path, monkeypatch):
    """The path is passive: it never imports the legacy memory, while every other memory_read does first."""
    from ouroboros import chronicle_import

    rows, ctx = chat(tmp_path), ctx_for(tmp_path)
    before = snapshot(tmp_path)

    def refuse(*_args, **_kwargs):
        raise AssertionError("the chronicle was activated")

    monkeypatch.setattr(chronicle_import, "ensure_activated", refuse)
    text = members(ctx, room_id="1")
    assert text.split("\n")[0] == (f"members of room:1; {NOT_ACTIVATED}: the one-time legacy import has not run, so this "
                                   "room's tellings and untold rows are unknown here, not none; a memory_read of a room or a "
                                   "record, or any chronicle_write, activates it")
    assert "untold rows: none" not in text and listed(text) == [""]
    assert NOT_ACTIVATED in members(ctx, node_id="anything").split("\n")[0]
    assert snapshot(tmp_path) == before and not (tmp_path / "memory" / "chronicle").exists()
    with pytest.raises(AssertionError, match="activated"):  # the room's records read activates first
        _memory_read(ctx, room_id="1")
    monkeypatch.undo()
    write(ctx, kind="note", text="Now active.")
    text = members(ctx, room_id="1")
    assert NOT_ACTIVATED not in text and "untold rows: 5 in 1 run" in text.split("\n")[0]
    assert f"from {addr(rows[0])} to {addr(rows[4])}" in listed(text)[-1]


def test_the_chronicle_is_read_once_for_the_whole_answer(tmp_path, monkeypatch):
    """Chronicle state is read in batch: the index opens as often for a room of six tellings as for one of two."""
    ctx, rows, _page, _note = install(tmp_path)
    opened = []
    original = ChronicleStore._index

    def counted(store):
        opened.append(1)
        return original(store)

    monkeypatch.setattr(ChronicleStore, "_index", counted)
    members(ctx, room_id="1")
    two = len(opened)
    monkeypatch.setattr(ChronicleStore, "_index", original)
    for n in range(4):
        write(ctx, kind="note", text=f"Note {n}.")
    monkeypatch.setattr(ChronicleStore, "_index", counted)
    opened.clear()
    assert len(listed(members(ctx, room_id="1"))) == 8
    assert len(opened) == two


# --- each relation by its own name ------------------------------------------------------------------

def test_a_page_seals_rows_and_notes_and_a_part_folds_members(tmp_path):
    ctx, rows, page, note = install(tmp_path)
    sealer = write(ctx, kind="page", text="The note is kept here.", covers={"note_ids": [note]})["node_id"]
    head = ChronicleStore(tmp_path).room_head("1")
    part = write(ctx, kind="part", text="The build arc settled.", member_ids=[page], expected_sequence=head)["node_id"]
    text = members(ctx, node_id=page)
    assert text.split("\n")[0] == (f"members of page {page} (room:1): it seals rows (room:1 2 recorded, 2 found, 0 "
                                   "missing) and 0 notes: the rows as runs in each sealed room's reading order, then the notes")
    assert listed(text)[0].startswith("- sealed rows of room:1: 2 rows, ") and \
        f"from {addr(rows[1])} to {addr(rows[2])}" in listed(text)[0]
    notes = members(ctx, node_id=sealer)
    assert "it seals rows (none recorded) and 1 note" in notes.split("\n")[0]
    assert listed(notes)[0].startswith(f"- note {note} [")
    folded = members(ctx, node_id=part)
    assert folded.split("\n")[0] == f"members of part {part} (room:1): it folds 1 record, in its recorded order"
    assert listed(folded)[0].startswith(f"- page {page} [") and f"folded into part {part}" in listed(folded)[0]
    room = listed(members(ctx, room_id="1"))
    assert any(line.startswith(f"- part {part} [") and "folds 1 record" in line for line in room)
    # A note folds and seals nothing: its rung says so instead of inventing a member.
    assert members(ctx, node_id=note).split("\n")[0].startswith(
        f"members of note {note} (room:1): nothing stands under a note in the chronicle's relations")


def test_an_account_cites_versions_and_a_selection_replaces(tmp_path):
    ctx, _rows, page, note = install(tmp_path)
    account = write(ctx, kind="account", text="Across the build work I learned to rerun flaky tests.",
                    sources=[page, note])["node_id"]
    fix = write(ctx, kind="correction", target_id=page, text="The red was a flaky test, not my code.")["node_id"]
    chosen = write(ctx, kind="selection", target_id=account, replaces=[page], reason="told through it")["node_id"]
    cited = members(ctx, node_id=account)
    assert cited.split("\n")[0] == (f"members of account {account} (room:1): it cites 2 source versions, in its "
                                    "recorded order; nothing sealed or folded")
    first = listed(cited)[0]
    assert first.startswith(f"- page {page} [") and f"cites revision {page}" in first
    assert f"later corrections not in this account: {fix}" in first
    assert f"memory_read(node_id='{page}', revision='{page}')" in first
    replaced = members(ctx, node_id=chosen)
    assert "it 1 record" not in replaced and f"it replaces 1 record: it shows account {account}" in replaced
    assert f"told in my story by account {account}" in listed(replaced)[0]
    room = "\n".join(listed(members(ctx, room_id="1")))
    assert f"shown by selection {chosen} (memory_read(members=true, node_id='{chosen}'))" in room
    assert f"1 correction under it (revision {fix})" in room


def test_drafts_and_withdrawals_are_shown_as_what_they_are(tmp_path):
    ctx, rows, page, _note = install(tmp_path)
    store = ChronicleStore(tmp_path)
    covers = page_covers(tmp_path, "1", from_addr=addr(rows[3]), to_addr=addr(rows[4]))["covers"]
    draft = store.publish_page(room_id="1", text="A helper's retelling of the fix.", covers=covers, author=LIGHT)
    assert draft.ok
    line = next(line for line in listed(members(ctx, room_id="1")) if draft.record["id"] in line.split(" ")[2])
    assert "draft by Light, not yet accepted" in line and "by helper (" in line
    write(ctx, kind="decision", target_id=page, accepted=False, reason="I tell it anew")
    text = members(ctx, room_id="1")
    assert "1 withdrawn or candidate record of this room listed after the acting ones" in text.split("\n")[0]
    gone = [line for line in listed(text) if line.startswith(f"- page {page} ")]
    assert len(gone) == 1 and "; withdrawn;" in gone[0]  # paged after the acting records, named as what it is
    assert listed(text).index(gone[0]) > listed(text).index(line)
    assert members(ctx, node_id=page).split("\n")[0].startswith(f"members of page {page} (room:1), withdrawn: it "
                                                                "formerly sealed rows (room:1 2 recorded, 2 found")
    assert "untold rows: 3 in 1 run" in text.split("\n")[0]  # its rows reopen: 0-2 now one run before the draft's


def test_a_legacy_section_retells_a_positional_range_of_its_room(tmp_path):
    shared.world(tmp_path)
    ctx = ctx_for(tmp_path)
    room = listed(members(ctx, room_id="1"))
    legacy = [line for line in room if line.startswith("- legacy ")]
    assert legacy and all("by legacy_helper (" in line and NO_LEAD in line for line in legacy)
    assert "retells positions 0–6" in legacy[0]
    ident = legacy[0].split(" ")[2]
    text = members(ctx, node_id=ident)
    assert text.split("\n")[0] == f"members of legacy {ident} (room:1): it retells positions 0–6 of the chat stream"
    assert re.fullmatch(r"- range: positions 0–6; 4 rows of room:1 in it, 4 not sealed by a page; "
                        r"memory_read\(room_id='1', rows=true, from='row:1@\S+', to='row:\S+'\)", listed(text)[0])
    assert members(ctx, room_id="legacy").split("\n")[0].endswith("the legacy room names no chat, so no row stands in it")


# --- arguments, paging and context levels -----------------------------------------------------------

def test_no_address_and_other_modes_are_argument_errors(tmp_path):
    ctx, _rows, page, _note = install(tmp_path)
    for args in ({}, {"room_id": "1", "node_id": page}, {"room_id": "*"}, {"room_id": "Main"},
                 {"room_id": "1", "rows": True}, {"node_id": page, "revision": page}, {"room_id": "1", "after_seq": 2},
                 {"room_id": "1", "story": True}, {"room_id": "1", "from": "row:1@x#abc"}):
        assert "TOOL_ARG_ERROR" in members(ctx, **args), args
    assert "TOOL_ARG_ERROR" in _memory_read(ctx, room_id="1", words=["build"])  # words filter a members listing
    for args in ({"room_id": "1"}, {"node_id": page}, {"room_id": "1", "start": 3}):
        assert "TOOL_ARG_ERROR" not in members(ctx, **args), args


def test_the_answer_pages_by_the_tools_own_bound(tmp_path, monkeypatch):
    ctx, _rows, _page, _note = install(tmp_path)
    for n in range(12):
        write(ctx, kind="note", text=f"Note number {n} keeps one fact. More.")
    whole = members(ctx, room_id="1")
    document = "\n".join(whole.split("\n")[2:])
    monkeypatch.setattr(chronicle, "tool_result_limit", lambda _name: 900)
    seen, start = "", 0
    while True:
        page = members(ctx, room_id="1", start=start)
        assert len(page) <= 900 and page.split("\n")[0] == whole.split("\n")[0]
        window = page.split("\n", 2)
        seen += window[2]
        found = re.search(r"next_start=(\d+) \(listed order\); memory_read\(members=true, room_id='1', start=(\d+)\)",
                          window[1])
        if found is None:
            assert window[1].endswith("complete")
            break
        assert found.group(1) == found.group(2)
        start = int(found.group(1))
    # Paged, the level-1 list does not fit one page, so no context level joins it: the same document page by page.
    assert seen == document.split("\ncontext, level")[0]


def test_deeper_levels_join_only_whole_and_labelled_as_context(tmp_path, monkeypatch):
    ctx, _rows, page, _note = install(tmp_path)
    head = ChronicleStore(tmp_path).room_head("1")
    part = write(ctx, kind="part", text="The build arc settled.", member_ids=[page], expected_sequence=head)["node_id"]
    text = members(ctx, node_id=part)
    assert "\ncontext, level 2: what stands under the entries above, added whole because the page bound fits it" in text
    assert f"\nunder page {page}:\n  - sealed rows of room:1: 2 rows" in text
    level_one = text.split("\ncontext, level")[0]
    monkeypatch.setattr(chronicle, "tool_result_limit", lambda _name: len(level_one) + 120)
    narrow = members(ctx, node_id=part)
    assert "context, level" not in narrow and narrow.split("\n")[1].endswith("complete")
    assert listed(narrow) == listed(text)


def test_the_schema_carries_members_and_words_and_its_lead_names_them():
    """Where field prose does not arrive (the custom route) the lead alone says how to descend; Nano's standing set
    stays inside its budget with both (tests/test_nano_tool_policy.py)."""
    schema = next(entry.schema for entry in chronicle.chronicle_tools() if entry.name == "memory_read")
    properties = schema["parameters"]["properties"]
    assert properties["members"] == {"type": "boolean", "description": "true: list what stands under room_id or node_id."}
    assert properties["words"]["type"] == "array" and properties["words"]["items"] == {"type": "string"}
    assert properties["words"]["description"] == ("members: keep records whose own text or one correction matches an "
                                                  "item (word prefixes, any order; quotes exact).")
    assert ("members=true with room_id lists the room's tellings in publication order, then its untold rows as runs; "
            "with node_id, what that record seals, folds, cites, replaces or retells. Each line has its opening and "
            "readers. words keeps records whose own text or one correction matches; runs of rows are not filtered."
            ) in schema["description"]
    assert "members=true" in _memory_read(ctx_for(pathlib.Path("/nonexistent")), node_id="x", rows=True)


def test_words_drop_non_matching_listed_lines_over_the_whole_list_before_paging(tmp_path, monkeypatch):
    """The shared predicate (``dialogue_words``) keeps the matching lines of the listed set; the filter runs over every
    immediate entry first, so a smaller page never changes which entries match, and no context level joins."""
    ctx, _rows, page, note = install(tmp_path)
    for n in range(10):
        write(ctx, kind="note", text=f"Flaky runner number {n} failed again. More.")
    whole = members(ctx, room_id="1")
    assert len(listed(whole)) == 14  # page, 11 notes, two untold runs
    text = members(ctx, room_id="1", words=["flak"])
    assert ("; words: 11 of 12 listed records match in their own text or one correction (applied before paging); "
            "runs of rows are not filtered") in text.split("\n")[0] and "flak" not in text.split("\n")[0]
    lines = listed(text)
    assert [line.split(" ")[1] for line in lines[:-1]] == ["note"] * 11 and "context, level" not in text
    # The runs' rows were not read, so they are not called non-matching: one line names their word search.
    assert lines[-1] == ("- untold rows of room:1: 3 rows in 2 runs, not filtered by words here (this listing reads no "
                         "row's words); a word search of room:1 over their period, which also returns told rows of "
                         "that period: chat_history(words=['flak'], room_id='1', date_from='2026-10-01T00:00:00+00:00', "
                         "date_to='2026-10-01T00:00:04+00:00'); the unfiltered listing prints each run's exact readers")
    both = members(ctx, room_id="1", words=["build red", "flaky test"])
    assert [line.split(" ", 3)[2] for line in listed(both)[:2]] == [page, note]
    assert [line for line in listed(members(ctx, room_id="1", words=["нет такого"])) if "untold" not in line] == []
    monkeypatch.setattr(chronicle, "tool_result_limit", lambda _name: 700)
    seen, start = "", 0
    while True:
        window = members(ctx, room_id="1", words=["flak"], start=start).split("\n", 2)
        assert window[0] == text.split("\n")[0]
        seen += window[2]
        found = re.search(r"next_start=(\d+) \(listed order\); memory_read\(members=true, room_id='1', "
                          r"words=\['flak'\], start=(\d+)\)", window[1])
        if found is None:
            break
        start = int(found.group(1))
    assert seen == "\n".join(text.split("\n")[2:])
    for bad in (["   "], [], "flaky", [3]):
        assert "TOOL_ARG_ERROR" in members(ctx, room_id="1", words=bad), bad


def test_words_judge_what_a_record_says_never_its_id_facts_or_readers(tmp_path):
    """A word found only in reader syntax, an id or a byline matches nothing; a word in the body or in a correction
    standing under the record, though not in its printed opening, keeps the line with an excerpt showing why."""
    ctx, _rows, page, note = install(tmp_path)
    for words in (["members"], ["memory_read"], [page[:8]], ["mind"], ["seals"], ["node_id"]):
        text = members(ctx, room_id="1", words=words)
        assert ": 0 of 2 listed records match" in text.split("\n")[0], words
        assert not any(line.startswith(("- page", "- note")) for line in listed(text)), words
    body = listed(members(ctx, room_id="1", words=["fixed"]))
    assert body[0].startswith(f"- page {page} [") and body[0].endswith("; excerpt: «I found the build red. Then I fixed it.»")
    assert len([line for line in body if line.startswith(("- page", "- note"))]) == 1
    led = listed(members(ctx, room_id="1", words=["found build"]))[0]  # in the printed opening: no excerpt
    assert led.startswith(f"- page {page} [") and "excerpt" not in led
    write(ctx, kind="correction", target_id=note, text="The runner was the cause, not the test.")
    fixed = listed(members(ctx, room_id="1", words=["runner cause"]))
    assert fixed[0].startswith(f"- note {note} [") and "runner was the cause" in fixed[0]
    assert re.search(r"; excerpt \(correction [0-9a-f]+\): «", fixed[0])  # the excerpt names its correction
    sealed = listed(members(ctx, node_id=page, words=["build"]))
    assert len(sealed) == 1 and sealed[0].startswith("- sealed rows of room:1: 2 rows in 1 run, not filtered by words")
    assert "date_from='2026-10-01T00:00:01+00:00', date_to='2026-10-01T00:00:02+00:00'" in sealed[0]


def test_a_page_recorded_in_another_room_stands_on_the_rung_of_each_room_it_seals(tmp_path):
    """A conversation page is recorded in one room and seals rows of several: each sealed room lists it, saying where
    it is recorded; a room it seals nothing of does not."""
    from tests.test_chronicle_tools import _github_row

    shared.world(tmp_path)
    ctx = ctx_for(tmp_path)
    rows = [_github_row("repo.1", "i1", 1, "first issue"), _github_row("repo.1", "i2", 2, "second issue"),
            _github_row("repo.2", "i9", 3, "another repository")]
    for row in rows:
        dialogue_store.append(tmp_path, "chat", row)
    i1, i2, other = (str(row["chat_id"]) for row in rows)
    page = write(ctx, kind="page", room_id=i2, text="Two issues read; the repository is quiet.",
                 covers={"to": addr(rows[1]), "conversation": True})["node_id"]
    there = [line for line in listed(members(ctx, room_id=i1)) if line.startswith(f"- page {page} [")]
    assert len(there) == 1 and f"[recorded in room:{i2}, seals rows of room:{i1};" in there[0]
    own = [line for line in listed(members(ctx, room_id=i2)) if line.startswith(f"- page {page} [")]
    assert len(own) == 1 and f"[room:{i2};" in own[0]
    assert not any(page in line for line in listed(members(ctx, room_id=other)))
    under = members(ctx, node_id=page)
    assert f"room:{i1} 1 recorded, 1 found, 0 missing" in under.split("\n")[0]
    assert f"room:{i2} 1 recorded, 1 found, 0 missing" in under.split("\n")[0]


# --- review round 1: exact readers, honest counts, acting seals, fields -----------------------------

def _place_row(chat: int, minute: int, text: str, imported: bool = False) -> dict:
    """A place room's arrival; an imported one is an older message stored later, so the room reads by send time."""
    return {"ts": f"2026-10-02T10:{minute:02d}:00+00:00", "direction": "in", "chat_id": chat, "type": "place_arrival",
            "text": text, "source": "presence:chat", "client_message_id": f"m{minute}", "place_id": "place",
            **({"historical_import": {"version": 1, "authorship": "participant",
                                      "received_at": "2026-10-26T00:00:00+00:00"}} if imported else {})}


def _follow(ctx, line: str) -> list:
    """The row addresses every rows=true reader printed on ``line`` returns, in order."""
    found = []
    for call in re.findall(r"memory_read\(room_id='([^']+)', rows=true, from='(\S+)', to='(\S+)'\)", line):
        text = _memory_read(ctx, room_id=call[0], rows=True, **{"from": call[1], "to": call[2]})
        found += re.findall(r"; (row:\S+#[0-9a-f]{12})\]", text)
    return found


def test_a_runs_readers_return_exactly_its_rows_in_a_room_read_by_send_time(tmp_path):
    """Runs follow the room's reading order; rows=true reads stored order. Each run's readers are its stored slices,
    so following them returns exactly the run's rows, never zero rows nor a skipped member."""
    rows, ctx = chat(tmp_path), ctx_for(tmp_path)
    write(ctx, kind="note", text="Active.")
    chat_id = (1 << 40) + 77
    live_a, live_c = _place_row(chat_id, 10, "live a"), _place_row(chat_id, 50, "live c")
    old_b, old_x = _place_row(chat_id, 5, "imported b", True), _place_row(chat_id, 30, "imported x", True)
    for row in (live_a, live_c, old_b, old_x):  # stored a, c, b, x; read by send time b, a, x, c
        dialogue_store.append(tmp_path, "chat", row)
    room = str(chat_id)
    write(ctx, kind="page", room_id=room, text="x sealed.", covers={"from": addr(old_x), "to": addr(old_x)})
    runs = [line for line in listed(members(ctx, room_id=room)) if line.startswith("- untold")]
    assert [_follow(ctx, line) for line in runs] == [[addr(live_a), addr(old_b)], [addr(live_c)]]
    assert f"from {addr(old_b)} to {addr(live_a)} in reading order" in runs[0]  # b reads first, a is stored first
    assert runs[0].count("rows=true") == 2  # a and b are not one stored slice: c stands between them
    assert rows  # Main stays as it was


def test_a_page_counts_its_recorded_found_and_missing_rows(tmp_path):
    """An edited row changes its content hash: the page still names the old one, and says so, never fewer rows."""
    ctx, rows, page, _note = install(tmp_path)
    seq = next(entry.seq for entry in dialogue_store.iter_rows(tmp_path, "chat")
               if entry.row.get("text") == rows[1]["text"])
    dialogue_store.revise_text(tmp_path, "chat", seq, "The build is red. Edited.")
    text = members(ctx, node_id=page)
    assert "it seals rows (room:1 2 recorded, 1 found, 1 missing)" in text.split("\n")[0]
    gone = [line for line in listed(text) if "not found" in line]
    assert gone == [f"- sealed rows of room:1 not found: 1 recorded row no longer resolve (an edited row changes its "
                    f"hash): {chat_chain.source_row_id(rows[1])[:12]}"]
    from ouroboros.chronicle_store import capture
    from ouroboros.memory_members import _Lister

    with capture(tmp_path, tail="none"):
        named = _Lister(tmp_path, ChronicleStore(tmp_path))._named([page, "not-a-record"])
    assert named[0].line.startswith(f"- page {page} [")
    assert named[1].line == "- record not-a-record: NOT in this chronicle; its words cannot be read here"


def test_a_withdrawn_page_recorded_elsewhere_leaves_the_rooms_it_sealed(tmp_path):
    """Only acting seals make a cross-room page a telling here: withdraw it, reseal one of its rows with another page,
    and the room lists the new page, not the withdrawn one."""
    from tests.test_chronicle_tools import _github_row

    shared.world(tmp_path)
    ctx = ctx_for(tmp_path)
    rows = [_github_row("repo.1", "i1", 1, "first issue"), _github_row("repo.1", "i2", 2, "second issue")]
    for row in rows:
        dialogue_store.append(tmp_path, "chat", row)
    i1, i2 = (str(row["chat_id"]) for row in rows)
    page = write(ctx, kind="page", room_id=i2, text="Two issues read.", covers={"to": addr(rows[1]), "conversation": True})
    assert any(line.startswith(f"- page {page['node_id']} [") for line in listed(members(ctx, room_id=i1)))
    write(ctx, kind="decision", target_id=page["node_id"], accepted=False, reason="told anew")
    again = write(ctx, kind="page", room_id=i1, text="The first issue alone.",
                  covers={"from": addr(rows[0]), "to": addr(rows[0])})["node_id"]
    listing = listed(members(ctx, room_id=i1))
    assert not any(page["node_id"] in line for line in listing)
    assert any(line.startswith(f"- page {again} [") for line in listing)
    assert members(ctx, node_id=page["node_id"]).split("\n")[0].startswith(
        f"members of page {page['node_id']} (room:{i2}), withdrawn: it formerly sealed rows")


def test_words_judge_each_field_alone_and_a_cited_source_at_its_cited_revision(tmp_path):
    """An alternative is never completed across the text and a correction; an account's source is judged as the
    revision it cites (what its printed reader opens); a note matched in its opening shows no excerpt."""
    ctx, _rows, page, note = install(tmp_path)
    zebra = write(ctx, kind="correction", target_id=page, text="Zebra stripes were the cause.")["node_id"]
    for words, kept in ((["found zebra"], 0), (["found build"], 1), (["zebra"], 1)):
        text = members(ctx, room_id="1", words=words)
        assert f": {kept} of 2 listed records match" in text.split("\n")[0], words
    account = write(ctx, kind="account", text="Across the build work.", sources=[page, note])["node_id"]
    write(ctx, kind="correction", target_id=page, text="Quagga markings too.", expected_revision=zebra)
    cited = members(ctx, node_id=account, words=["quagga"])
    assert ": 0 of 2 listed records match" in cited.split("\n")[0]  # added after the cited revision
    assert ": 1 of 2 listed records match" in members(ctx, node_id=account, words=["zebra"]).split("\n")[0]
    opening = [line for line in listed(members(ctx, room_id="1", words=["flaky"])) if line.startswith("- note")]
    assert len(opening) == 1 and "excerpt" not in opening[0]


def test_a_cited_revision_missing_from_this_chronicle_has_no_searchable_words(tmp_path, monkeypatch):
    """A partial journal keeps the source but not the revision an account cites: its words are not judged by today's
    text (an excerpt its reader could not open); the line names the missing revision and matches nothing."""
    ctx, _rows, page, note = install(tmp_path)
    zebra = write(ctx, kind="correction", target_id=page, text="Zebra stripes were the cause.")["node_id"]
    account = write(ctx, kind="account", text="Across the build work.", sources=[page, note])["node_id"]
    assert ": 1 of 2 listed records match" in members(ctx, node_id=account, words=["zebra"]).split("\n")[0]
    original = ChronicleStore._revisions
    monkeypatch.setattr(ChronicleStore, "_revisions",
                        lambda store, db, ident: [rev for rev in original(store, db, ident) if rev != zebra])
    for words in (["zebra"], ["found build"]):
        text = members(ctx, node_id=account, words=words)
        assert ": 0 of 2 listed records match" in text.split("\n")[0], words
    line = next(line for line in listed(members(ctx, node_id=account)) if line.startswith(f"- page {page} ["))
    assert f"cites revision {zebra}" in line and "that revision is not in this chronicle; its words cannot be judged" in line
    assert f"revision='{zebra}'" not in line and f"memory_read(node_id='{page}')" in line


def test_a_room_with_incomplete_appearances_observes_no_untold_rows_without_claiming_none(tmp_path, monkeypatch):
    ctx, _rows, _page, _note = install(tmp_path)
    assert members(ctx, room_id="1").split("\n")[0].endswith("untold rows: 3 in 2 runs")

    def none_observed(root, room, gaps=None, *, runs=None, every=None):
        gaps.add("appearances_deriving")
        return []

    monkeypatch.setattr(memory_inventory, "open_room_rows", none_observed)
    head = members(ctx, room_id="1").split("\n")[0]
    assert head.endswith("; untold rows: none observed") and "incomplete: " in head


def test_an_excerpt_comes_from_the_shared_locator_on_the_original_text(tmp_path):
    """A word far into a long record, and a word inside an identifier (the shared tokenizer splits ``_``), are found
    on the original text; a cut names the field's length."""
    ctx, _rows, _page, _note = install(tmp_path)
    long = write(ctx, kind="note", text="Opening sentence. " + "filler " * 80 + "the chat_history reader answered.")["node_id"]
    line = next(line for line in listed(members(ctx, room_id="1", words=["history"])) if long in line)
    assert "; excerpt: «…" in line and "chat_history reader answered." in line
    assert f"[{len('Opening sentence. ' + 'filler ' * 80 + 'the chat_history reader answered.')} chars in full]" in line
