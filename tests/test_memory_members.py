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
    assert "1 row, " in lines[2] and f"from {addr(rows[0])} to {addr(rows[0])}" in lines[2]
    assert "2 rows, " in lines[3] and f"from {addr(rows[3])} to {addr(rows[4])}" in lines[3]
    assert f"memory_read(room_id='1', rows=true, from='{addr(rows[3])}', to='{addr(rows[4])}')" in lines[3]
    assert "speakers: " in lines[3] and "2026-10-01 00:00 → 2026-10-01 00:00" in lines[3]


def test_untold_rows_are_the_one_open_row_computation(tmp_path, monkeypatch):
    """The runs are taken from ``memory_inventory.open_room_rows``, never computed a second way: its answer is what
    the room lists, so a narrower answer from it narrows the runs."""
    ctx, rows, _page, _note = install(tmp_path)
    assert "untold rows: 3 in 2 runs" in members(ctx, room_id="1").split("\n")[0]
    one = memory_inventory.open_room_rows(tmp_path, "1")[:1]
    monkeypatch.setattr(memory_inventory, "open_room_rows", lambda root, room, gaps=None: list(one))
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
                                   "room's tellings and untold rows are unknown here, not none; any other memory_read or "
                                   "chronicle_write call activates it")
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
    assert text.split("\n")[0] == (f"members of page {page} (room:1): it seals 2 rows and 0 notes: the rows as runs in "
                                   "each sealed room's reading order (room:1), then the notes")
    assert listed(text)[0].startswith("- sealed rows of room:1: 2 rows, ") and \
        f"from {addr(rows[1])} to {addr(rows[2])}" in listed(text)[0]
    notes = members(ctx, node_id=sealer)
    assert "it seals 0 rows and 1 note" in notes.split("\n")[0]
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
    assert f"not acting, not listed: withdrawn page {page}" in text.split("\n")[0]
    assert not any(line.startswith(f"- page {page} ") for line in listed(text))
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
    assert properties["members"] == {"type": "boolean", "description": "true: what stands under room_id or node_id."}
    assert properties["words"]["type"] == "array" and properties["words"]["items"] == {"type": "string"}
    assert "members=true with room_id or node_id lists what stands under it by recorded relation" in schema["description"]
