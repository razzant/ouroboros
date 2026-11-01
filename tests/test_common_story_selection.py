"""Choosing what an account tells (#2522): the proposal read, kept quotes, my changes of what it tells, the room's line.

A selection takes the replaced records' words out of my story and their room's page. Before the choice,
``memory_read(proposed_selection=…)`` shows what it would change, rendered by the view's own capture and read
without a single write; the story and page it shows are the ones the real selection then makes. One captured
state answers the whole read, every record it would tell or stop telling stands whole whatever the comparison matched, and a
later page continues only the same document. The write itself answers with the teller relations counted after it,
never a claim about display. An account keeps the verified quotes I choose, as a page does. A signed correction or rejection of a record it tells stays beside it
unless a version its sources cite holds that record (then it is inside that version or reported as a later change).
The room whose records it tells names the account in one line. Every rule runs on the shared fixture through the
real writer, store and view; no model, network or record of the owner's chronicle enters here.
"""
from __future__ import annotations

import json
import re

import pytest

from ouroboros import chat_chain, memory_floor as mf, memory_view as mv
from ouroboros.chronicle_store import ChronicleStore, body_sha256
from ouroboros.memory_reader import read_memory
from ouroboros.tools import chronicle as tools
from ouroboros.tools.chronicle import _memory_read
from ouroboros.utils import sha256_text
from tests import _memory_inventory_shared as shared
from tests.test_chronicle_tools import snapshot as tree
from tests.test_common_story import G_TEXT, NANNY, _block, _ctx, _journal, _page, _part, _rooms, _snapshot, _story, _write

PAGE = mv.ViewSpec("integrator", room_id="1", room_lanes=False, origin_words=False, live_rooms="none", marks="none",
                   knowledge=False)  # the room page the proposal read compares


def _addr(root, pos):
    return chat_chain.format_address({p: a for a, _r, p in chat_chain.iter_rows(root)}[pos])


TELLS = "## What the account would tell, each record whole as it acts now\n"


def _continuation(page):
    """``(start, fingerprint)`` a page's continuation line names."""
    cont = page.split("\n", 2)[1]
    return int(cont.split("next_start=")[1].split(" ")[0]), re.search(r"fingerprint='([0-9a-f]{16})'\)$", cont).group(1)


def _header_fingerprint(page):
    """The fingerprint a page's first line names, whether or not the page continues."""
    return re.search(r"; fingerprint='([0-9a-f]{16})'\]$", page.split("\n", 1)[0]).group(1)


def _proposal(root, room="1", **proposal):
    """The whole proposal read, page by page: ``(first page, document after its two head lines)``."""
    pages, bound = [], {}
    while True:
        text = _memory_read(_ctx(root), proposed_selection=proposal, room_id=room, **bound)
        assert text.startswith(f"[proposed selection of account {proposal['target_id']}: "), text[:300]
        pages.append(text)
        if text.split("\n", 2)[1].endswith("complete"):
            return pages, "".join(page.split("\n", 2)[2] for page in pages)
        bound = dict(zip(("start", "fingerprint"), _continuation(text)))


def _page_view(root, spec=PAGE):
    return mv.render_room(mv.capture_memory_view(root, {}, spec))


def _main_part(root, b1):
    """A Main part over B1 and a second Main page, the second corrected before any account."""
    p12 = _page(root, "1", 12, 12, text="Main heard a long reply.")
    assert ChronicleStore(root).correct(p12, "MEMBER FIX BEFORE ANY ACCOUNT", shared.MIND).ok
    return p12, _part(root, "1", [b1, p12], text="Main began and went on.")


# --- the proposal read --------------------------------------------------------------------------------

def test_the_proposal_shows_the_story_and_page_the_real_selection_then_makes_and_writes_nothing(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    p12, part = _main_part(tmp_path, b1)
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, part])["node_id"]
    story_now, page_now = _story(tmp_path), _page_view(tmp_path)
    files = tree(tmp_path)
    pages, read = _proposal(tmp_path, target_id=g, replaces=[a1, part])
    assert tree(tmp_path) == files  # no record, no index byte, no file: a read
    head = pages[0].split("\n", 3)
    assert head[0] == (f"[proposed selection of account {g}: replaces 2, shown=True; a read, nothing written; "
                       f"fingerprint='{sha256_text(read)[:16]}']")  # the digest of the document, not of its head
    assert read.startswith(f"the proposal: replaces 2: {a1}, {part}; shown=True. Nothing is written;")
    assert "acting selection now: none, so the account is not in my story" in read
    assert ("the account itself enters my story; other entries leaving it: 2; returning to it: 0; the account would "
            "tell 4 records in all (an account's own "
            f"and a part's folded members included); their rooms: 1 (3), {alpha} (1)") in read
    # The real selection then makes exactly the compared lines, story and room page alike.
    chosen = _write(tmp_path, kind="selection", target_id=g, replaces=[a1, part], reason="told through it")["node_id"]
    story_then, page_then = _story(tmp_path).replace(chosen, "proposed"), _page_view(tmp_path)
    assert story_then != story_now and page_then != page_now
    assert "\n".join(tools._diff(story_now, story_then)) in read
    assert "\n".join(tools._diff(page_now, page_then)) in read
    assert f"my story, full view before any physical floor: {len(story_now):,} -> {len(story_then):,} chars" in read
    # What it would tell: every record whole, those whose words leave my story marked, the folded members too.
    composition = read.split(TELLS, 1)[1]
    for record, words in ((a1, "Alpha asked again and I counted."), (part, "Main began and went on.")):
        assert f"[{'page' if record == a1 else 'part'} {record}; " in composition and f"\n{words}\n" in composition
    assert composition.count("(its words leave my story with this selection)") == 2
    assert "\n  Main started and the owner asked me to go on." not in composition  # not indented: memory_read's form
    assert "Main started and the owner asked me to go on." in composition and "MEMBER FIX BEFORE ANY ACCOUNT" in composition
    assert f"; folded into {part}; " in composition and f"memory_read(node_id='{p12}')" in composition
    assert f"rows: memory_read(room_id='1', rows=true, from='{_addr(tmp_path, 12)}', to='{_addr(tmp_path, 12)}')" in composition
    # Read again now that it acts: nothing would change.
    _pages, again = _proposal(tmp_path, target_id=g, replaces=[a1, part])
    assert f"acting selection now: {chosen} (shown=True, replaces 2)" in again
    assert "itself stays in my story; other entries leaving it: 0; returning to it: 0" in again
    story_diff = again.split("('-' leaves, '+' arrives)\n", 1)[1].split("\n\n## ", 1)[0].split("\n")
    assert [line[0] for line in story_diff[1:] if line[:1] in "-+"] == ["-", "+"]  # a new selection is a new record
    assert story_diff[2].endswith(f"shown by selection {chosen}") and story_diff[3].endswith("shown by selection proposed")
    assert again.count("(unchanged)") == 1  # the room page


def test_a_proposal_to_supersede_or_withdraw_shows_what_returns_and_what_lies_beneath(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    g1 = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, b1])["node_id"]
    _write(tmp_path, kind="selection", target_id=g1, replaces=[a1, b1], reason="first telling")
    g2 = _write(tmp_path, kind="account", text="A SHORTER SECOND TELLING", sources=[g1])["node_id"]
    _pages, read = _proposal(tmp_path, target_id=g2, replaces=[g1])
    # The older account leaves; beneath it, the originals it told stand whole, already out of my story.
    assert f"-  {G_TEXT}" in read and "+  A SHORTER SECOND TELLING" in read
    composition = read.split(TELLS, 1)[1]
    assert composition.count("(its words leave my story") == 1  # g1's block; a1 and b1 left with the first telling
    assert "Alpha asked again and I counted." in composition and "Main started and the owner asked me to go on." in composition
    assert f"; told in my story by account {g1}; " in composition
    # Withdrawing the first telling instead: its words leave, what it told returns, and it would tell nothing.
    _pages, back = _proposal(tmp_path, target_id=g1, replaces=[a1, b1], shown=False)
    assert "itself leaves my story; other entries leaving it: 0; returning to it: 2;" in back
    assert "+  Alpha asked again and I counted." in back
    assert "nothing: shown with no replaces, or withdrawn" in back


def test_a_proposal_meets_the_writes_own_refusals_and_bad_arguments_write_nothing(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]
    files, ctx = tree(tmp_path), _ctx(tmp_path)
    for proposal, reason in (({"target_id": "nope"}, "target_missing"), ({"target_id": a1}, "invalid"),
                             ({"target_id": g, "replaces": [g]}, "invalid"),
                             ({"target_id": g, "replaces": [a1, a1]}, "invalid"),
                             ({"target_id": g, "replaces": ["nope"]}, "target_missing")):
        out = json.loads(_memory_read(ctx, proposed_selection=proposal))
        assert out["ok"] is False and out["reason"] == reason, (proposal, out)
        assert out["reason"] == _write(tmp_path, kind="selection", reason="r", **proposal)["reason"]  # the write's own
    for bad in ({"target": g}, {"target_id": g, "shown": "yes"}, {"target_id": g, "replaces": a1}, g, {"target_id": " "}):
        assert _memory_read(ctx, proposed_selection=bad).startswith("⚠️ TOOL_ARG_ERROR: proposed_selection is")
    assert "choose one mode" in _memory_read(ctx, proposed_selection={"target_id": g}, node_id=g)
    assert "is not a room address" in _memory_read(ctx, proposed_selection={"target_id": g}, room_id="Main")
    assert tree(tmp_path) == files


def test_a_tight_tool_result_pages_the_proposal_and_its_pages_join_into_the_whole(tmp_path, monkeypatch):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    p12, part = _main_part(tmp_path, b1)
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, part])["node_id"]
    _pages, whole = _proposal(tmp_path, target_id=g, replaces=[a1, part])
    monkeypatch.setattr(tools, "tool_result_limit", lambda _name: 1_200)
    pages, joined = _proposal(tmp_path, target_id=g, replaces=[a1, part])
    assert len(pages) > 3 and all(len(page) <= 1_200 for page in pages) and joined == whole
    start, fingerprint = _continuation(pages[1])
    assert pages[1].split("\n")[1].endswith(f"memory_read(proposed_selection=<same>, room_id='1', start={start}, "
                                            f"fingerprint='{fingerprint}')")
    assert fingerprint == _continuation(pages[0])[1]  # one document, named on every page
    assert {_header_fingerprint(page) for page in pages} == {fingerprint}  # and by every page's first line


TWELVE = "\n".join(f"Line {n} of what Main heard, kept word for word." for n in range(1, 13))


def test_a_record_whose_lines_my_account_repeats_still_reads_whole_in_the_proposal(tmp_path):
    shared.world(tmp_path, legacy=False)  # no legacy story between them: the account enters where the page leaves
    long = _page(tmp_path, "1", 12, 12, text=TWELVE)
    g = _write(tmp_path, kind="account", text=TWELVE + "\nAnd one line of my own.", sources=[long])["node_id"]
    _pages, read = _proposal(tmp_path, target_id=g, replaces=[long])
    comparison, composition = read.split(TELLS, 1)
    # The comparison matches the repeated lines as unchanged, so its interior lines are not in it at all...
    assert not any(line in comparison for line in TWELVE.split("\n")[1:-1])
    # ...and the page still reads whole, once, under its own header.
    assert f"[page {long}; room 1; " in composition
    assert f"\n{TWELVE}\n(its words leave my story with this selection)\nmemory_read(node_id='{long}')" in composition
    assert composition.count("Line 7 of what Main heard") == 1


BACK = "## What it tells now and would stop telling, each record whole as it acts now\n"


def test_a_withdrawal_or_narrowing_reads_each_returning_record_whole_though_its_lines_match(tmp_path):
    shared.world(tmp_path, legacy=False)
    long = _page(tmp_path, "1", 12, 12, text=TWELVE)
    other = _page(tmp_path, "1", 19, 19, text="Main heard one more thing.")
    g = _write(tmp_path, kind="account", text=TWELVE + "\nAnd one line of my own.", sources=[long, other])["node_id"]
    _write(tmp_path, kind="selection", target_id=g, replaces=[long, other], reason="told through it")
    _pages, read = _proposal(tmp_path, target_id=g, replaces=[long, other], shown=False)  # withdrawn: both return
    comparison, rest = read.split(TELLS, 1)
    tells, returning = rest.split(BACK, 1)
    assert not any(line in comparison for line in TWELVE.split("\n")[1:-1])  # the comparison cannot show them...
    assert "nothing: shown with no replaces, or withdrawn" in tells and "[page " not in tells
    assert "it would stop telling 2 it tells now (rooms: 1 (2))" in comparison
    for record in (long, other):  # ...so each reads whole, once, under its own header
        assert f"[page {record}; room 1; " in returning
    assert f"\n{TWELVE}\n(its words return to my story with this selection)\nmemory_read(node_id='{long}')" in returning
    assert returning.count("Line 7 of what Main heard") == 1 and "Main heard one more thing." in returning
    _pages, read = _proposal(tmp_path, target_id=g, replaces=[other])  # narrowed: only the long page returns
    tells, returning = read.split(TELLS, 1)[1].split(BACK, 1)
    assert f"[page {other}; " in tells and f"[page {other}; " not in returning
    assert f"\n{TWELVE}\n(its words return to my story with this selection)\n" in returning
    assert returning.count("Line 7 of what Main heard") == 1


def test_a_correction_published_while_the_proposal_reads_is_not_mixed_into_it(tmp_path, monkeypatch):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, b1])["node_id"]
    _pages, before = _proposal(tmp_path, target_id=g, replaces=[a1, b1])
    real, fixes = mv.capture_memory_view, []

    def racing(*args, **kwargs):  # another focus corrects A1 between the two views this read renders
        view = real(*args, **kwargs)
        if not fixes:
            fixes.append(ChronicleStore(tmp_path).correct(a1, "A CORRECTION PUBLISHED MID-READ", shared.MIND).record["id"])
        return view

    monkeypatch.setattr(mv, "capture_memory_view", racing)
    _pages, raced = _proposal(tmp_path, target_id=g, replaces=[a1, b1])
    assert fixes and raced == before  # one state: header, words and comparison all as before the correction
    monkeypatch.setattr(mv, "capture_memory_view", real)
    _pages, after = _proposal(tmp_path, target_id=g, replaces=[a1, b1])
    told = after.split(TELLS, 1)[1]
    header = next(line for line in told.split("\n") if line.startswith(f"[page {a1};"))
    assert f"revision {fixes[0]} (corrected by mind" in header and "A CORRECTION PUBLISHED MID-READ" in told


def test_a_later_page_of_a_proposal_whose_state_changed_is_refused_never_joined(tmp_path, monkeypatch):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, b1])["node_id"]
    proposal, ctx = {"target_id": g, "replaces": [a1, b1]}, _ctx(tmp_path)
    monkeypatch.setattr(tools, "tool_result_limit", lambda _name: 1_200)
    first = _memory_read(ctx, proposed_selection=proposal, room_id="1")
    start, fingerprint = _continuation(first)
    same = _memory_read(ctx, proposed_selection=proposal, room_id="1", start=start, fingerprint=fingerprint)
    assert same.split("\n", 2)[1].startswith(f"chars {start}–")  # the same document continues
    unbound = json.loads(_memory_read(ctx, proposed_selection=proposal, room_id="1", start=start))
    assert unbound["reason"] == "position_mismatch"  # a later page names its document
    chosen = _write(tmp_path, kind="selection", reason="chosen between pages", **proposal)
    files, current = tree(tmp_path), _continuation(_memory_read(ctx, proposed_selection=proposal, room_id="1"))[1]
    assert current != fingerprint
    for bound in ({"fingerprint": fingerprint}, {"start": start, "fingerprint": fingerprint},
                  {"start": start, "fingerprint": "0" * 16}):
        out = json.loads(_memory_read(ctx, proposed_selection=proposal, room_id="1", **bound))
        assert out["ok"] is False and out["reason"] == "position_mismatch", (bound, out)
        # It names no fingerprint to continue with: the new document is read from its start, never joined.
        assert out["detail"].endswith("Read it again from start=0") and current not in out["detail"]
    assert tree(tmp_path) == files
    # From its start, the read is of the state the selection made: nothing more would change.
    _pages, again = _proposal(tmp_path, **proposal)
    assert f"acting selection now: {chosen['node_id']} (shown=True, replaces 2)" in again
    assert _memory_read(ctx, node_id=g, fingerprint=fingerprint).startswith(
        "⚠️ TOOL_ARG_ERROR: fingerprint continues a proposed_selection read")


def test_a_complete_first_proposal_still_names_the_fingerprint_an_explicit_later_range_needs(tmp_path, monkeypatch):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, b1])["node_id"]
    proposal, ctx = {"target_id": g, "replaces": [a1, b1]}, _ctx(tmp_path)
    first = _memory_read(ctx, proposed_selection=proposal, room_id="1")
    head, cont, document = first.split("\n", 2)
    assert cont == f"chars 0–{len(document)} of {len(document)}; complete"  # the window names no recipe: nothing continues
    fingerprint = _header_fingerprint(first)  # the first line does, as the document's own digest
    assert fingerprint == sha256_text(document)[:16]
    assert head == (f"[proposed selection of account {g}: replaces 2, shown=True; a read, nothing written; "
                    f"fingerprint='{fingerprint}']")
    start = len(document) // 2  # an explicit range, asked for without any continuation
    later = _memory_read(ctx, proposed_selection=proposal, room_id="1", start=start, fingerprint=fingerprint)
    assert later.split("\n", 2)[1:] == [f"chars {start}–{len(document)} of {len(document)}; complete", document[start:]]
    assert _header_fingerprint(later) == fingerprint
    monkeypatch.setattr(tools, "tool_result_limit", lambda _name: 1_200)  # a tight reader pages the same document
    assert _continuation(_memory_read(ctx, proposed_selection=proposal, room_id="1"))[1] == fingerprint


def test_a_fingerprint_a_complete_proposal_named_is_refused_once_the_state_changed(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, b1])["node_id"]
    proposal, ctx = {"target_id": g, "replaces": [a1, b1]}, _ctx(tmp_path)
    first = _memory_read(ctx, proposed_selection=proposal, room_id="1")
    assert first.split("\n", 2)[1].endswith("complete")
    stale = _header_fingerprint(first)
    _write(tmp_path, kind="selection", reason="chosen after a complete read", **proposal)
    files, current = tree(tmp_path), _memory_read(ctx, proposed_selection=proposal, room_id="1")
    assert _header_fingerprint(current) != stale  # the document it describes is now another one
    for bound in ({"fingerprint": stale}, {"start": 40, "fingerprint": stale}):
        out = json.loads(_memory_read(ctx, proposed_selection=proposal, room_id="1", **bound))
        assert out["ok"] is False and out["reason"] == "position_mismatch", (bound, out)
        assert out["detail"].endswith("Read it again from start=0") and _header_fingerprint(current) not in out["detail"]
    assert tree(tmp_path) == files
    fresh = _memory_read(ctx, proposed_selection=proposal, room_id="1", start=40, fingerprint=_header_fingerprint(current))
    assert fresh.split("\n", 2)[1].startswith("chars 40–")  # the current fingerprint reads on


# --- the selection's receipt -----------------------------------------------------------------------------

def test_the_selection_answers_with_a_counted_receipt_of_what_it_tells_never_their_words(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    p12, part = _main_part(tmp_path, b1)
    assert ChronicleStore(tmp_path).correct(a1, "ALPHA FIX NO SOURCE CITES", shared.MIND).ok
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[part])["node_id"]  # A1 told, never cited

    def rooms_of(**counts):
        return ", ".join(f"{room} ({n})" for room, n in sorted(counts.items()))

    readers = ", ".join(f"memory_read(room_id='{room}')" for room in sorted({"1", alpha}))
    chosen = _write(tmp_path, kind="selection", target_id=g, replaces=[a1, part], reason="told through it")
    assert chosen["effect"].startswith("the account now tells 2 records: ")  # the measured effect, as before
    assert chosen["receipt"] == (
        "counted from the chronicle as read after this write: teller relations only, never display or what this write "
        "displaced (a later choice by any account changes them; what my story and a room's page show is read there, "
        "memory_read(story=true) and memory_read(room_id=…); memory_read(proposed_selection=…) compares the views "
        "before a choice); the account tells 4 records, 2 named by this selection and 2 through accounts it tells and "
        f"parts' members (rooms: {rooms_of(**{'1': 3, alpha: 1})}); 3 at a version its sources cite and 1 at none, "
        f"with my 1 signed change to those beside it. Each record with its words and teller: {readers}; composition, "
        f"source versions and selections: memory_read(node_id='{g}')")
    assert "1 signed change of mine to records it tells whose version none of its sources cites" in _story(tmp_path)
    # A later account telling the first: nesting is counted, as what it tells now.
    g2 = _write(tmp_path, kind="account", text="A SHORTER SECOND TELLING", sources=[g])["node_id"]
    second = _write(tmp_path, kind="selection", target_id=g2, replaces=[g], reason="told once more")["receipt"]
    assert "the account tells 5 records, 1 named by this selection and 4 through accounts it tells" in second
    # Withdrawn, it tells nothing; the receipt counts that and claims no record leaves or returns.
    back = _write(tmp_path, kind="selection", target_id=g2, replaces=[g], shown=False, reason="back")["receipt"]
    assert "the account tells 0 records, 0 named by this selection and 0 through accounts it tells and parts' members (rooms: none)" in back
    for receipt in (chosen["receipt"], second, back):  # counts and readers: the records' words and ids stay there
        assert not [word for word in (G_TEXT, "Alpha asked again and I counted.", "ALPHA FIX NO SOURCE CITES", a1, p12)
                    if word in receipt]
        assert not re.search(r"\b(leave|leaves|return|returns)\b", receipt)


def test_post_write_receipt_failure_keeps_the_successful_selection(tmp_path, monkeypatch):
    _rooms_map, _alpha, _beta, a1, _b1, _h1 = _rooms(tmp_path)
    account = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]

    def unavailable(*_args, **_kwargs):
        raise OSError("fixture read failure after publication")

    monkeypatch.setattr(tools, "selection_facts", unavailable)
    result = _write(tmp_path, kind="selection", target_id=account, replaces=[a1], reason="chosen")
    assert result["ok"] is True and "Selection saved" in result["receipt"]
    assert "Do not repeat the write" in result["receipt"]
    store = ChronicleStore(tmp_path)
    assert store.interpret([store.get(account)])[0]["selection"]["id"] == result["node_id"]


def test_receipt_does_not_attribute_a_later_selection_to_this_write(tmp_path, monkeypatch):
    _rooms_map, _alpha, _beta, a1, _b1, _h1 = _rooms(tmp_path)
    account = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]
    original = tools._selection_receipt
    later = []

    def racing(root, store, selection):
        later.append(store.select_account(account, replaces=[], author=shared.MIND, reason="later choice").record["id"])
        return original(root, store, selection)

    monkeypatch.setattr(tools, "_selection_receipt", racing)
    result = _write(tmp_path, kind="selection", target_id=account, replaces=[a1], reason="first choice")
    assert result["ok"] is True and later[0] != result["node_id"]
    assert "another selection is now acting" in result["receipt"]
    assert "leave my story" not in result["receipt"]


def test_receipt_reads_teller_relations_and_claims_no_displacement_another_accounts_choice_made(tmp_path, monkeypatch):
    _rooms_map, _alpha, _beta, a1, _b1, _h1 = _rooms(tmp_path)
    first = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]
    _write(tmp_path, kind="selection", target_id=first, replaces=[a1], reason="B tells it first")
    mine = _write(tmp_path, kind="account", text="MY OWN TELLING", sources=[a1])["node_id"]
    words = "Alpha asked again and I counted."
    assert words not in _story(tmp_path)  # hidden before this write
    original, withdrawn = tools._selection_receipt, []

    def racing(root, store, selection):  # B withdraws between A's write and A's receipt
        withdrawn.append(store.select_account(first, replaces=[], author=shared.MIND, reason="B withdraws", shown=False).ok)
        return original(root, store, selection)

    monkeypatch.setattr(tools, "_selection_receipt", racing)
    receipt = _write(tmp_path, kind="selection", target_id=mine, replaces=[a1], reason="A tells it")["receipt"]
    assert withdrawn == [True] and words not in _story(tmp_path)  # hidden throughout: nothing left, nothing returned
    assert "the account tells 1 record, 1 named by this selection and 0 through accounts" in receipt
    assert "teller relations only, never display or what this write displaced" in receipt
    assert not re.search(r"\b(leave|leaves|return|returns)\b", receipt)


def test_a_selection_replacing_a_note_the_room_still_shows_claims_no_record_leaving(tmp_path):
    _rooms_map, _alpha, _beta, a1, _b1, _h1 = _rooms(tmp_path)
    note = _write(tmp_path, kind="note", text="A NOTE OF MINE, STILL OPEN")["node_id"]
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]
    page = _page_view(tmp_path)
    assert "A NOTE OF MINE, STILL OPEN" in page
    receipt = _write(tmp_path, kind="selection", target_id=g, replaces=[note], reason="told through it")["receipt"]
    assert "A NOTE OF MINE, STILL OPEN" in _page_view(tmp_path)  # the note stays on the room's page
    assert "the account tells 1 record, 1 named by this selection" in receipt
    assert not re.search(r"\b(leave|leaves|return|returns)\b", receipt)
    assert "is read there" in receipt and "memory_read(story=true)" in receipt  # what shows is read where it shows


def test_withdrawing_one_teller_does_not_claim_a_page_hidden_by_another_returns(tmp_path):
    _rooms_map, _alpha, _beta, a1, _b1, _h1 = _rooms(tmp_path)
    earlier = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]
    _write(tmp_path, kind="selection", target_id=earlier, replaces=[a1], reason="earlier telling")
    later = _write(tmp_path, kind="account", text="A LATER TELLING", sources=[a1])["node_id"]
    _write(tmp_path, kind="selection", target_id=later, replaces=[a1], reason="later telling")
    assert "Alpha asked again and I counted." not in _story(tmp_path)
    result = _write(tmp_path, kind="selection", target_id=later, replaces=[a1], shown=False, reason="withdraw later")
    assert result["shown"] is False
    assert "Alpha asked again and I counted." not in _story(tmp_path)
    assert result["effect"] == (
        "withdrawal saved: other selections and folds still determine what appears. This "
        "task's view is not refreshed; the next start re-measures its Max room.")
    assert "the account tells 0 records" in result["receipt"]
    assert "the records it told return there" not in str(result)


def test_capture_materializes_json_after_releasing_the_index(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from ouroboros import chronicle_store as cs

    _rooms(tmp_path)
    original_index, original_snapshot = cs.ChronicleStore._index, cs._Snapshot
    held = []

    @contextmanager
    def observing_index(self):
        with original_index(self) as db:
            held.append(True)
            try:
                yield db
            finally:
                held.pop()

    def observing_snapshot(*args, **kwargs):
        assert not held, "snapshot decoding must happen after the index transaction closes"
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(cs.ChronicleStore, "_index", observing_index)
    monkeypatch.setattr(cs, "_Snapshot", observing_snapshot)
    with cs.capture(tmp_path):
        assert cs.ChronicleStore(tmp_path).records()


# --- quotes an account keeps -----------------------------------------------------------------------------

def test_an_account_keeps_the_verified_quotes_i_choose_and_a_false_one_writes_nothing(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    owner = {"address": _addr(tmp_path, 11), "text": "next please", "speaker": "human"}
    there = {"address": _addr(tmp_path, 13), "text": "alpha again", "speaker": "human"}
    before = _journal(tmp_path)
    for false in ({**owner, "text": "next, please"}, {**owner, "speaker": "ouroboros"},
                  {**owner, "address": _addr(tmp_path, 11)[:-12] + "0" * 12}, {**owner, "speaker": "owner"}):
        refused = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, b1], quotes=[there, false])
        assert refused["ok"] is False and refused["reason"] == "quote_mismatch", refused
    store = ChronicleStore(tmp_path)
    frozen = {"id": a1, "kind": "page", "room_id": alpha, "revision": a1, "sha256": body_sha256(store.get(a1)),
              "status": "final"}  # a hand-made, otherwise valid source edge: only the quote is unverified
    unverified = store.publish([{"kind": "account", "room_id": "1", "text": G_TEXT, "author": shared.MIND,
                                 "sources": [frozen], "quotes": [owner]}])
    assert unverified.reason == "quote_mismatch" and _journal(tmp_path) == before
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, b1], quotes=[owner, there])["node_id"]
    assert store.get(g)["quotes"] == [owner, there]
    _write(tmp_path, kind="selection", target_id=g, replaces=[a1, b1], reason="told through it")
    block = _block(_story(tmp_path), g)
    assert (f"  {G_TEXT}\n- quote (human, {owner['address']}): next please\n"
            f"- quote (human, {there['address']}): alpha again\n- written ") in block
    exact = _memory_read(_ctx(tmp_path), node_id=g)
    assert f"quote (human, {owner['address']}): next please" in exact and f"quote (human, {there['address']}): alpha again" in exact
    entry = next(e for e in read_memory(tmp_path, "catalog", {"area": "story", "limit": 100})["entries"] if e["id"] == g)
    assert entry["quotes_count"] == 2 and entry["told_fixes_count"] == 0
    # The account, its quotes included, stays whole under the physical floor: no rung addresses it.
    assert g not in {ident for _step, ident, _whole, _short in mf.floor_elements(_snapshot(tmp_path))}


# --- my changes of what an account tells ------------------------------------------------------------------

def test_my_changes_of_a_told_record_no_source_cites_stay_beside_the_account_and_a_cited_ones_do_not_repeat(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    assert store.correct(b1, "B1 FIX BEFORE THE ACCOUNT", shared.MIND).ok
    assert store.correct(a1, "A1 FIX INSIDE THE CITED VERSION", shared.MIND).ok
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]  # cites A1 as corrected, not B1 or H1
    _write(tmp_path, kind="selection", target_id=g, replaces=[a1, b1, h1], reason="told through it")
    story = _story(tmp_path)
    block = _block(story, g)
    assert "A1 FIX INSIDE THE CITED VERSION" not in story  # what a cited version holds is my account's to say
    assert ("- 1 signed change of mine to records it tells whose version none of its sources cites, oldest first:\n"
            f"- correction by me of page {b1} (room 1, ") in block and "  B1 FIX BEFORE THE ACCOUNT" in block
    # Later: a cited source's correction is its reported change; an uncited one's and a rejection join the list.
    assert store.correct(a1, "A1 FIX AFTER THE ACCOUNT", shared.MIND, expected_revision=store.interpret([store.get(a1)])[0]["revision"]).ok
    assert store.correct(b1, "B1 FIX AFTER THE ACCOUNT", shared.MIND, expected_revision=store.interpret([store.get(b1)])[0]["revision"]).ok
    assert store.decide(h1, False, shared.MIND, "H1 WAS WRONG").ok
    story = _story(tmp_path)
    block = _block(story, g)
    assert f"- later correction of its source {a1} (not in this account):\n  A1 FIX AFTER THE ACCOUNT" in block
    assert "- 3 signed changes of mine" in block and f"- my rejection of the draft page {h1} (room {beta}, " in block
    assert block.index("B1 FIX BEFORE") < block.index("B1 FIX AFTER") < block.index("H1 WAS WRONG")
    for words in ("A1 FIX AFTER", "B1 FIX BEFORE", "B1 FIX AFTER", "H1 WAS WRONG"):
        assert story.count(words) == 1, words
    entry = next(e for e in read_memory(tmp_path, "catalog", {"area": "story", "limit": 100})["entries"] if e["id"] == g)
    assert entry["told_fixes_count"] == 3
    # Reselected without B1: B1 is whole again, its corrections inside its own text, no longer beside the account.
    _write(tmp_path, kind="selection", target_id=g, replaces=[a1, h1], reason="B1 speaks for itself")
    story = _story(tmp_path)
    assert "- 1 signed change of mine" in _block(story, g) and "B1 FIX" not in _block(story, g)
    assert story.count("B1 FIX BEFORE THE ACCOUNT") == 1 and f"page {b1}\n  Main started" in story
    # Withdrawn: nothing stands beside an account that is not shown.
    _write(tmp_path, kind="selection", target_id=g, replaces=[], shown=False, reason="back")
    assert "signed change" not in _story(tmp_path)


def test_through_a_part_or_an_older_account_the_cited_edge_decides_and_never_the_replaced_id(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    p12, part = _main_part(tmp_path, b1)
    # A part replaced but not cited: its folded member's correction stays beside the account.
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]
    _write(tmp_path, kind="selection", target_id=g, replaces=[part], reason="told through it")
    assert f"- correction by me of page {p12} (room 1, " in _block(_story(tmp_path), g)
    _write(tmp_path, kind="selection", target_id=g, replaces=[], shown=False, reason="back")
    # Cited: the member's correction is the part's reported change, once, and not beside the account again.
    cited = _write(tmp_path, kind="account", text="THE PART CITED", sources=[part])["node_id"]
    _write(tmp_path, kind="selection", target_id=cited, replaces=[part], reason="told through it")
    story = _story(tmp_path)
    assert "signed change" not in story and story.count("MEMBER FIX BEFORE ANY ACCOUNT") == 1
    assert f"correction of nested source page {p12} (room 1) via {part}" in story
    _write(tmp_path, kind="selection", target_id=cited, replaces=[], shown=False, reason="back")
    # An older account told by a newer one: the newer's own sources decide, through their frozen edges.
    assert store.correct(a1, "A1 FIX BEFORE THE FIRST TELLING", shared.MIND).ok
    first = _write(tmp_path, kind="account", text="FIRST TELLING", sources=[a1])["node_id"]
    _write(tmp_path, kind="selection", target_id=first, replaces=[a1], reason="first")
    assert "A1 FIX BEFORE" not in _story(tmp_path)  # inside the version the first telling cites
    beside = _write(tmp_path, kind="account", text="SECOND TELLING, FIRST NOT READ", sources=[b1])["node_id"]
    _write(tmp_path, kind="selection", target_id=beside, replaces=[first], reason="second")
    block = _block(_story(tmp_path), beside)
    assert f"- correction by me of page {a1} (room {alpha}, " in block and "A1 FIX BEFORE THE FIRST TELLING" in block
    _write(tmp_path, kind="selection", target_id=beside, replaces=[], shown=False, reason="back")
    reading = _write(tmp_path, kind="account", text="SECOND TELLING OVER THE FIRST", sources=[first])["node_id"]
    _write(tmp_path, kind="selection", target_id=reading, replaces=[first], reason="second")
    assert "signed change" not in _story(tmp_path)  # the first telling's cited edge holds A1's correction
    assert store.correct(a1, "A1 FIX AFTER BOTH", shared.MIND, expected_revision=store.interpret([store.get(a1)])[0]["revision"]).ok
    story = _story(tmp_path)
    assert story.count("A1 FIX AFTER BOTH") == 1 and f"correction of nested source page {a1} (room {alpha}) via {first}" in story
    assert "signed change" not in story


# --- the room whose records an account tells -----------------------------------------------------------

def test_the_room_names_the_account_that_tells_its_records_through_nesting_and_not_after_withdrawal(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    alpha_task = {"id": "rootA001", "chat_id": rooms["alpha"]}
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1])["node_id"]
    _write(tmp_path, kind="selection", target_id=g, replaces=[a1], reason="told through it")
    line = (f"### Records of this room my accounts tell\n- 1 record, 2026-09-03 00:03 → 2026-09-03 00:03: told in my "
            f"story by my account {g}; memory_read(room_id='{alpha}') lists each with its words and teller")
    snapshot = _snapshot(tmp_path, alpha_task)
    room = mv.render_room(snapshot)
    assert line in room and "Alpha asked again" not in room
    listed = _memory_read(_ctx(tmp_path), room_id=alpha)
    assert f"[page {a1}; room {alpha}; " in listed and f"; told in my story by account {g}; " in listed
    # A tight window leaves the line as it is: it is no rung of the floor, already one address.
    level = mv.FloorLevel(addressed=tuple((step, tuple(ident for s, ident, _w, _s in mf.floor_elements(snapshot) if s == step))
                                          for step in mf.LADDER))
    assert line in mv.render_room(snapshot, level)
    # Told on by a newer account: the line names the one standing in my story, through the older.
    g2 = _write(tmp_path, kind="account", text="A SECOND TELLING", sources=[g])["node_id"]
    _write(tmp_path, kind="selection", target_id=g2, replaces=[g], reason="second")
    room = mv.render_room(_snapshot(tmp_path, alpha_task))
    assert f"told in my story by my account {g2}; memory_read(room_id='{alpha}')" in room
    assert (f"told in my story by my account {g2}; memory_read(room_id='1') lists each with its words and teller"
            in mv.render_room(_snapshot(tmp_path)))  # the older account is Main's record
    _write(tmp_path, kind="selection", target_id=g2, replaces=[], shown=False, reason="back")
    assert f"by my account {g}; " in mv.render_room(_snapshot(tmp_path, alpha_task))
    _write(tmp_path, kind="selection", target_id=g, replaces=[], shown=False, reason="back")
    assert "my accounts tell" not in mv.render_room(_snapshot(tmp_path, alpha_task))
    # A nanny carries no story: no account stands in for the room's records there.
    _write(tmp_path, kind="selection", target_id=g, replaces=[a1], reason="again")
    assert "my accounts tell" not in mv.render_room(_snapshot(tmp_path, {**NANNY, "chat_id": rooms["alpha"]}))
    # Two accounts telling one room's records: still one line, naming both.
    p12 = _page(tmp_path, "1", 12, 12, text="Main heard a long reply.")
    one = _write(tmp_path, kind="account", text="ONE", sources=[b1])["node_id"]
    other = _write(tmp_path, kind="account", text="OTHER", sources=[p12])["node_id"]
    _write(tmp_path, kind="selection", target_id=one, replaces=[b1], reason="one")
    _write(tmp_path, kind="selection", target_id=other, replaces=[p12], reason="other")
    main = mv.render_room(_snapshot(tmp_path))
    assert main.count("my accounts tell\n- ") == 1 and f"- 2 records, 2026-09-03 00:00 → 2026-09-03 00:02: told in my story by my accounts {one}, {other}; " in main


def _folded_page_told(root, *, fold_first):
    """Main's page P12 under a part, told by an account of mine; the fold and the selection in either order."""
    rooms, alpha, beta, a1, b1, h1 = _rooms(root)
    p12 = _page(root, "1", 12, 12, text="Main heard a long reply.")
    g = _write(root, kind="account", text=G_TEXT, sources=[p12])["node_id"]
    if fold_first:
        part = _part(root, "1", [b1, p12], text="Main began and went on.")
    _write(root, kind="selection", target_id=g, replaces=[p12], reason="told through it")
    if not fold_first:
        part = _part(root, "1", [b1, p12], text="Main began and went on.")
    assert p12 not in {e["id"] for e in _snapshot(root).story}  # only the part stands in the story, never the member
    return p12, part, g


def _assert_folded_page_pointer(root, p12, part, g):
    line = (f"### Records of this room my accounts tell\n- 1 record, 2026-09-03 00:02 → 2026-09-03 00:02: told in my story by "
            f"my account {g}; memory_read(room_id='1') lists each with its words and teller")
    room = mv.render_room(_snapshot(root))
    assert line in room and f"node_id='{p12}'" not in room  # told: it leaves the page under its part, the line stands
    assert f"told in my story by account {g}; " in _memory_read(_ctx(root), room_id="1")  # the pointer's reader names it
    # Told on by a newer account: the line names the one standing in my story, through the older.
    g2 = _write(root, kind="account", text="A SECOND TELLING", sources=[g])["node_id"]
    _write(root, kind="selection", target_id=g2, replaces=[g], reason="second")
    assert f"told in my story by my account {g2}; memory_read(room_id='1')" in mv.render_room(_snapshot(root))
    # A nanny carries no story: the page stays whole under its part, no account stands in for it.
    nanny = mv.render_room(_snapshot(root, {**NANNY, "chat_id": 1}))
    assert "my accounts tell" not in nanny and "Main heard a long reply." in nanny
    # Withdrawn: nothing tells the page, so it is under its part again and no line remains.
    _write(root, kind="selection", target_id=g2, replaces=[], shown=False, reason="back")
    _write(root, kind="selection", target_id=g, replaces=[], shown=False, reason="back")
    room = mv.render_room(_snapshot(root))
    assert "my accounts tell" not in room and f"under part {part} — " in room and f"memory_read(node_id='{p12}')" in room


def test_the_room_names_the_account_that_tells_a_page_folded_before_the_selection(tmp_path):
    _assert_folded_page_pointer(tmp_path, *_folded_page_told(tmp_path, fold_first=True))


def test_the_room_names_the_account_that_tells_a_page_folded_after_the_selection(tmp_path):
    _assert_folded_page_pointer(tmp_path, *_folded_page_told(tmp_path, fold_first=False))


@pytest.mark.parametrize("source_room", ["1", "legacy"])
def test_the_room_keeps_the_teller_of_an_individually_folded_legacy_record(tmp_path, source_room):
    for fold_first in (True, False):
        root = tmp_path / str(fold_first)
        shared.world(root, flat="Earlier life before rooms, retained exactly.")
        legacy = next(r["id"] for r in ChronicleStore(root).room_records(source_room) if r["kind"] == "legacy")
        g = _write(root, kind="account", text=G_TEXT, sources=[legacy])["node_id"]
        if fold_first:
            _part(root, source_room, [legacy], text="An earlier part.")
        _write(root, kind="selection", target_id=g, replaces=[legacy], reason="told here")
        if not fold_first:
            assert f"by my account {g}; memory_read(room_id='{source_room}')" in mv.render_room(_snapshot(root))
            _part(root, source_room, [legacy], text="An earlier part.")
        view = _snapshot(root)
        assert legacy not in {entry["id"] for entry in view.story}
        assert f"by my account {g}; memory_read(room_id='{source_room}')" in mv.render_room(view)
        assert f"told in my story by account {g};" in _memory_read(_ctx(root), room_id=source_room)
        assert "my accounts tell" not in mv.render_room(_snapshot(root, NANNY))
        assert mv.render_room(view) == mv.render_room(mv.snapshot_from_json(mv.snapshot_json(view)))


def test_an_edge_whose_cited_revision_the_journal_lost_cites_nothing_so_later_corrections_stay_beside(tmp_path):
    """The cited revision decides, not the id: once the journal lost the correction an edge names (its transaction
    became a gap), that edge holds no version of the record, so my later correction stands beside the account."""
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    lost = store.correct(a1, "A1 FIX THE JOURNAL LOSES", shared.MIND).record["id"]
    g = _write(tmp_path, kind="account", text=G_TEXT, sources=[{"id": a1, "revision": lost}])["node_id"]
    assert store.correct(a1, "A1 FIX AFTER THE LOSS", shared.MIND, expected_revision=lost).ok
    journal = tmp_path / "memory" / "chronicle" / "records.jsonl"
    lines = journal.read_bytes().split(b"\n")
    at = next(n for n, line in enumerate(lines) if line.strip()
              and any(record["id"] == lost for record in json.loads(line).get("records") or ()))
    lines[at] = b"x" * len(lines[at])  # one torn transaction, the same size: every later byte stays where it was
    journal.write_bytes(b"\n".join(lines))
    (tmp_path / "memory" / "chronicle" / "index.sqlite3").unlink()
    _write(tmp_path, kind="selection", target_id=g, replaces=[a1], reason="told through it")
    block = _block(_story(tmp_path), g)
    assert f"that revision of {a1} is not in this chronicle" in block  # the source line names the loss, not the words
    assert f"- correction by me of page {a1} (room {alpha}, " in block and "  A1 FIX AFTER THE LOSS" in block
